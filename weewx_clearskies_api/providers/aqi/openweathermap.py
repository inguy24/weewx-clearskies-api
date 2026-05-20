"""OpenWeatherMap Air Pollution AQI provider module (ADR-013, ADR-038).

Five responsibilities per ADR-038 §2:
  1. Outbound API call — single GET per cache miss:
       GET https://api.openweathermap.org/data/2.5/air_pollution
           ?lat={lat}&lon={lon}&appid={appid}
     No units= or lang= params (response is always µg/m³; OWM 1–5 ordinal scale).
     FREE tier endpoint — no subscription gate, no basic-tier-401 graceful-empty
     pattern (distinct from forecast/openweathermap.py which uses a paid endpoint).
  2. Response parsing — wire-shape Pydantic models with extra="ignore" (LC5):
       _OWMAirPollutionComponents — co/no/no2/o3/so2/pm2_5/pm10/nh3 (all float|None)
       _OWMAirPollutionMain — aqi: int|None (OWM 1–5; declared for wire validity,
         NOT read in translation per LC4 — canonical aqi is derived from concentrations)
       _OWMAirPollutionEntry — dt, main, components
       _OWMAirPollutionResponse — list: list[_OWMAirPollutionEntry] (shadows Python
         builtin; uses Field(default_factory=list) per LC11)
  3. Translation to canonical AQIReading (_wire_to_canonical):
       - aqi = main.aqi (OWM's native 1–5 ordinal; served as-is)
       - aqiScale = "owm" (1–5 ordinal scale, not EPA 0–500)
       - aqiCategory = None (dashboard-computed from aqi+aqiScale)
       - aqiMainPollutant = None (OWM Air Pollution does not supply dominant pollutant)
       - aqiLocation = None (PARTIAL-DOMAIN per LC12 — no location label at any tier)
       - pollutantPM25/PM10/O3/NO2/SO2/CO: raw µg/m³ from components (no conversion)
       - observedAt: epoch_to_utc_iso8601(entry.dt) — shared helper (DRY per LC17)
       - source = "openweathermap"
       - NH3 and NO dropped unconditionally (no EPA AQI band; not on canonical — LC16)
  4. Capability declaration — CAPABILITY symbol consumed at startup.
     Full max-surface MINUS aqiLocation (only PARTIAL-DOMAIN for OWM at all tiers).
     FREE-tier endpoint — no tier-conditional fields (distinct from Aeris which has
     paid-tier fields; L1 rule does not produce tier-conditional CAPABILITY here).
  5. Error handling — ProviderHTTPClient.get() raises canonical taxonomy with all
     attributes set (L2 carry-forward, 3b-4 audit F1). NO re-construction of
     canonical exceptions.  The ONLY narrow wrap in this module is:
       (ValidationError, ValueError) → ProviderProtocolError at wire-validation boundary
     This IS adding context the inner layer didn't have (wire-shape validation is a
     higher-level error class — ProviderHTTPClient only raises on HTTP-level errors).
     OWM Air Pollution uses HTTP status codes (401/429/5xx) — NOT a 200-success-false
     envelope (distinct from Aeris). No LC27 envelope mapping needed.

OWM AQI is a keyed provider:
  appid passed as query param per OWM auth convention.
  Credentials NOT in the cache key (LC7 — privacy/leakage concern).
  Provider-scoped per 3b-5 Q2 user decision — same env var as forecast/alerts OWM:
    WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID on settings.forecast.openweathermap_appid.
  Wired at startup via wire_aqi_settings() in endpoints/aqi.py.

Cache layer (ADR-017 / LC3 / LC6 / LC7):
  TTL: 900s (15 min) per ADR-017 AQI domain (same as openmeteo + aeris).
  Key: SHA-256 of (provider_id="openweathermap", endpoint="aqi_current", {lat4, lon4}).
  appid NOT in key (LC7 — privacy/leakage concern; same as aeris.py).
  Value: AQIReading.model_dump() dict (JSON-serializable for Redis backend).
  Sentinel: {"_no_reading": True} when wire response has empty list[] or all-null
    components (cached so re-polls within TTL skip the provider).
  Reconstruction on hit: AQIReading.model_validate(cached_dict).

OWM main.aqi (1–5) field:
  Served as-is as the canonical aqi value with aqiScale="owm".
  The dashboard converts to the operator's preferred display scale.

NH3 / NO handling (LC16):
  Both present on wire (nh3, no in components).  Neither has an EPA AQI band.
  Neither appears on canonical AQIReading.  Silently dropped during translation.
  Mirrors aeris.py dropping pm1.

Rate limiter (LC8):
  max_calls=5, window_seconds=1 (be-polite guard; 15-min TTL → ~96 calls/day,
  well below OWM free-tier 60 calls/min cap).

ruff: noqa: N815  (wire field names pm2_5, no2 etc. don't need camelCase)
"""

# ruff: noqa: N815

from __future__ import annotations

import hashlib
import json
import logging
from typing import List

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from weewx_clearskies_api.models.responses import AQIReading
from weewx_clearskies_api.providers._common.cache import get_cache
from weewx_clearskies_api.providers._common.capability import ProviderCapability
from weewx_clearskies_api.providers._common.datetime_utils import epoch_to_utc_iso8601
from weewx_clearskies_api.providers._common.errors import (
    KeyInvalid,
    ProviderProtocolError,
)
from weewx_clearskies_api.providers._common.http import ProviderHTTPClient
from weewx_clearskies_api.providers._common.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PROVIDER_ID = "openweathermap"
DOMAIN = "aqi"
DEFAULT_AQI_TTL_SECONDS = 900  # 15 min per ADR-017 / LC3
_API_VERSION = "0.1.0"
OWM_AIRPOL_BASE_URL = "https://api.openweathermap.org"
OWM_AIRPOL_PATH = "/data/2.5/air_pollution"


# ---------------------------------------------------------------------------
# Capability declaration (ADR-038 §4)
# ---------------------------------------------------------------------------

CAPABILITY = ProviderCapability(
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    supplied_canonical_fields=(
        # Full max-surface MINUS aqiLocation (PARTIAL-DOMAIN — no location field at ANY tier).
        # FREE-tier endpoint: no tier-conditional fields.  All listed fields are
        # delivered on the free tier; no paid-tier-only splits here.
        "aqi", "aqiCategory", "aqiMainPollutant",
        "pollutantPM25", "pollutantPM10",
        "pollutantO3", "pollutantNO2", "pollutantSO2", "pollutantCO",
        "observedAt", "source",
        # aqiLocation EXCLUDED — PARTIAL-DOMAIN (no location field on wire at any tier).
    ),
    geographic_coverage="global",
    auth_required=("appid",),
    default_poll_interval_seconds=DEFAULT_AQI_TTL_SECONDS,
    operator_notes=(
        "OpenWeatherMap Air Pollution API (/data/2.5/air_pollution). FREE-tier endpoint "
        "— a basic OWM appid works without a One Call subscription. "
        "Keyed (query-param appid); reuses provider-scoped credential from "
        "forecast/alerts OWM — same WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID env var "
        "per 3b-5 Q2 user decision. "
        "OWM main.aqi (1–5 ordinal) is served as-is with aqiScale='owm'. "
        "The dashboard converts to the operator's preferred display scale. "
        "aqiCategory=None (dashboard-computed). aqiMainPollutant=None "
        "(OWM Air Pollution does not supply a dominant pollutant field). "
        "aqiLocation is PARTIAL-DOMAIN (no location label on wire at any tier). "
        "NH3 and NO are dropped (no EPA AQI band; not on canonical AQIReading). "
        "Gas pollutants (O3, NO2, SO2, CO) passed through as µg/m³ (raw provider values)."
    ),
)

# ---------------------------------------------------------------------------
# Wire-shape Pydantic models (LC5 — extra="ignore"; required fields enumerated)
# Source: docs/reference/api-docs/openweathermap.md §Air Pollution
# ---------------------------------------------------------------------------


class _OWMAirPollutionComponents(BaseModel):
    """components object inside each list[] entry (LC5).

    All concentrations in µg/m³ — including gas pollutants (distinct from Aeris
    which returns gases in PPB).  nh3 and no are dropped during translation (LC16).
    """

    model_config = ConfigDict(extra="ignore")

    co: float | None = None      # Carbon monoxide µg/m³
    no: float | None = None      # Nitric oxide µg/m³ (dropped — no EPA AQI band)
    no2: float | None = None     # Nitrogen dioxide µg/m³
    o3: float | None = None      # Ozone µg/m³
    so2: float | None = None     # Sulphur dioxide µg/m³
    pm2_5: float | None = None   # Fine particulate matter µg/m³
    pm10: float | None = None    # Coarse particulate matter µg/m³
    nh3: float | None = None     # Ammonia µg/m³ (dropped — no EPA AQI band)


class _OWMAirPollutionMain(BaseModel):
    """main object inside each list[] entry (LC5).

    aqi is the OWM 1–5 ordinal scale (1=Good, 5=Very Poor).
    Served as-is as the canonical aqi value with aqiScale="owm".
    """

    model_config = ConfigDict(extra="ignore")

    aqi: int | None = None  # OWM 1–5 ordinal; served as-is with aqiScale="owm"


class _OWMAirPollutionEntry(BaseModel):
    """One entry in the list[] array (LC5).

    For the current endpoint, list[] contains a single entry.
    For forecast/history endpoints (not in scope), list[] has multiple entries.
    """

    model_config = ConfigDict(extra="ignore")

    dt: int                    # Unix UTC seconds (→ observedAt via epoch_to_utc_iso8601)
    main: _OWMAirPollutionMain
    components: _OWMAirPollutionComponents


class _OWMAirPollutionResponse(BaseModel):
    """Top-level Air Pollution API response (LC5).

    The field named 'list' shadows the Python builtin list.  Use
    Field(default_factory=list) for the default value (LC11 — verified pattern;
    no # noqa needed because the field name is on a Pydantic model, not a
    function signature where shadowing causes real confusion).

    coord is an [lat, lon] array (wire quirk — NOT an object like other OWM
    endpoints). Declared with extra="ignore" so it's silently dropped; we
    already have lat/lon from StationInfo (LC5).
    """

    model_config = ConfigDict(extra="ignore")

    list: List[_OWMAirPollutionEntry] = Field(default_factory=list)  # noqa: UP006 — List required; 'list[...]' annotation shadows the builtin when field is named 'list' (LC11)


# ---------------------------------------------------------------------------
# Rate limiter (LC8 — "be polite" guard; 5 req/s max)
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="openweathermap-aqi",
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    max_calls=5,
    window_seconds=1,
)

# ---------------------------------------------------------------------------
# HTTP client (module-level singleton — one per module, not per request)
# ---------------------------------------------------------------------------

_http_client: ProviderHTTPClient | None = None


def _client_for() -> ProviderHTTPClient:
    """Return the module-level HTTP client, constructing on first call."""
    global _http_client  # noqa: PLW0603
    if _http_client is None:
        _http_client = ProviderHTTPClient(
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
            user_agent=f"weewx-clearskies-api/{_API_VERSION}",
        )
    return _http_client


# ---------------------------------------------------------------------------
# Cache key construction (ADR-017 §Cache key / LC7)
# ---------------------------------------------------------------------------


def _build_cache_key(lat: float, lon: float) -> str:
    """Build a deterministic SHA-256 cache key for (provider_id, endpoint, {lat4, lon4}).

    appid NOT in key per LC7 — privacy/leakage concern; cache scope is
    per-location-per-provider, not per-tenant.

    Lat/lon rounded to 4 decimal places per ADR-017 §Cache key.
    Endpoint key "aqi_current" distinct from any other module's endpoint key.
    """
    payload = json.dumps(
        {
            "provider_id": PROVIDER_ID,
            "endpoint": "aqi_current",
            "params": {
                "lat4": round(lat, 4),
                "lon4": round(lon, 4),
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------


def _wire_to_canonical(entry: _OWMAirPollutionEntry) -> AQIReading | None:
    """Translate OWM list[0] entry to canonical AQIReading.

    Returns None if main.aqi AND all component values are null (no useful reading).

    OWM main.aqi (1–5 ordinal) is used directly as the canonical aqi value with
    aqiScale="owm".  The dashboard applies any display-scale conversion.

    Per LC16: NH3 (nh3) and NO (no) are silently dropped — they have no EPA
    AQI band and no slot on canonical AQIReading.
    """
    components = entry.components

    # aqi: OWM's native 1–5 ordinal (main.aqi).
    aqi_val: int | None = entry.main.aqi

    # Empty-result guard: return None if aqi AND all canonical pollutant values are null.
    has_data = aqi_val is not None or any(
        v is not None
        for v in (
            components.pm2_5, components.pm10,
            components.o3, components.no2, components.so2, components.co,
        )
    )
    if not has_data:
        return None

    # observedAt: Unix UTC epoch → ISO-8601 Z via shared helper (LC17 / DRY).
    observed_at = epoch_to_utc_iso8601(
        entry.dt,
        provider_id=PROVIDER_ID,
        domain=DOMAIN,
    )

    return AQIReading(
        aqi=aqi_val,
        aqiScale="owm",
        aqiCategory=None,
        aqiMainPollutant=None,          # OWM Air Pollution does not supply dominant pollutant
        aqiLocation=None,               # PARTIAL-DOMAIN — no location label at any tier (LC12)
        pollutantPM25=components.pm2_5,  # µg/m³ passthrough (group_concentration)
        pollutantPM10=components.pm10,   # µg/m³ passthrough (group_concentration)
        pollutantO3=components.o3,       # µg/m³ raw provider value
        pollutantNO2=components.no2,     # µg/m³ raw provider value
        pollutantSO2=components.so2,     # µg/m³ raw provider value
        pollutantCO=components.co,       # µg/m³ raw provider value
        observedAt=observed_at,
        source=PROVIDER_ID,
    )


# ---------------------------------------------------------------------------
# Public fetch entrypoint (ADR-038 §2)
# ---------------------------------------------------------------------------


def fetch(
    *,
    lat: float,
    lon: float,
    appid: str,
    http_client: ProviderHTTPClient | None = None,
) -> AQIReading | None:
    """GET /data/2.5/air_pollution and return canonical AQIReading or None.

    Cache-first: checks the cache before making an outbound HTTP call.
    Cache stores post-normalization AQIReading as a model_dump() dict (JSON-
    serializable for Redis per ADR-017); reconstructed via model_validate() on hit.

    None return: provider responded but no useful reading available (empty
    list[], or all component values null).

    appid validation: empty/None appid raises KeyInvalid BEFORE the outbound call
    (explicit fail-fast — same pattern as forecast/openweathermap.py).

    L2 carry-forward (3b-4 audit F1): ProviderHTTPClient.get() raises canonical
    taxonomy exceptions (KeyInvalid, QuotaExhausted, TransientNetworkError,
    ProviderProtocolError) with all structured attributes set (status_code,
    retry_after_seconds).  These propagate bare — do NOT re-construct.

    The only narrow wrap in this module is:
      (ValidationError, ValueError) → ProviderProtocolError at wire-validation boundary
    This IS adding context the inner layer didn't have — wire-shape validation is a
    higher-level error class (ProviderHTTPClient only sees HTTP-level errors).

    No LC20 graceful-empty-bundle pattern: OWM Air Pollution is FREE tier.
    A 401 from this endpoint means the appid is genuinely invalid (operator
    misconfiguration); KeyInvalid propagates from ProviderHTTPClient.get() bare.

    Args:
        lat: Station latitude (from services/station.py StationInfo).
        lon: Station longitude (from services/station.py StationInfo).
        appid: OWM API key (from settings.forecast.openweathermap_appid /
            WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID env var per LC18).
        http_client: Optional ProviderHTTPClient override for testing.
            When None, the module-level singleton is used.

    Returns:
        Canonical AQIReading or None (no useful reading at this location).

    Raises:
        KeyInvalid: appid is empty/None (pre-call guard), or provider returned 401/403.
        QuotaExhausted: Provider returned 429 (rate limit exceeded).
        TransientNetworkError: Network/DNS failure or 5xx after retries.
        ProviderProtocolError: Response JSON validation failed.
    """
    # Explicit fail-fast guard (LC20 / LC22) — empty appid means no credential at all.
    # Raise KeyInvalid before hitting the network rather than letting the provider
    # return a cryptic 401 with no context about where the missing key came from.
    if not appid:
        raise KeyInvalid(
            "OpenWeatherMap appid is empty or None — set "
            "WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID env var",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    cache_key = _build_cache_key(lat, lon)
    cached = get_cache().get(cache_key)
    if cached is not None:
        logger.debug(
            "Cache hit for OpenWeatherMap AQI",
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        if cached == {"_no_reading": True}:
            return None
        return AQIReading.model_validate(cached)

    logger.debug(
        "Cache miss for OpenWeatherMap AQI; calling %s",
        OWM_AIRPOL_BASE_URL + OWM_AIRPOL_PATH,
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )

    # appid in params dict (not URL path) — avoids logging credentials at INFO
    # level if the URL is logged (LC22 / security baseline).
    params = {
        "lat": str(round(lat, 6)),
        "lon": str(round(lon, 6)),
        "appid": appid,
    }

    client = http_client or _client_for()
    _rate_limiter.acquire()

    # L2 carry-forward: client.get() raises canonical taxonomy with all
    # attributes set.  Do NOT catch and re-raise as a new canonical exception
    # (would silently drop retry_after_seconds per 3b-4 audit F1 rule).
    response = client.get(OWM_AIRPOL_BASE_URL + OWM_AIRPOL_PATH, params=params)

    # Wire-shape validation: intentional (ValidationError, ValueError) → ProviderProtocolError
    # wrap.  This adds context the inner layer didn't have — wire-shape validation is a
    # higher-level error class.  Documented in commit body per non-obvious-provenance rule.
    try:
        wire = _OWMAirPollutionResponse.model_validate(response.json())
    except (ValidationError, ValueError) as exc:
        logger.error(
            "OpenWeatherMap AQI response validation failed: %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        raise ProviderProtocolError(
            f"OpenWeatherMap AQI response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    # Empty list[] guard — no reading at this location (LC23 / LC2).
    if not wire.list:
        logger.info(
            "OpenWeatherMap AQI: empty list[] for lat=%s lon=%s",
            round(lat, 4),
            round(lon, 4),
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        get_cache().set(
            cache_key,
            {"_no_reading": True},
            ttl_seconds=DEFAULT_AQI_TTL_SECONDS,
        )
        return None

    record = _wire_to_canonical(wire.list[0])

    if record is None:
        # All-null components — cache sentinel so re-polls within TTL skip the provider.
        logger.info(
            "OpenWeatherMap AQI: all-null reading for lat=%s lon=%s",
            round(lat, 4),
            round(lon, 4),
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        get_cache().set(
            cache_key,
            {"_no_reading": True},
            ttl_seconds=DEFAULT_AQI_TTL_SECONDS,
        )
        return None

    get_cache().set(cache_key, record.model_dump(), ttl_seconds=DEFAULT_AQI_TTL_SECONDS)

    logger.info(
        "OpenWeatherMap AQI fetched: aqi=%s mainPollutant=%s for lat=%s lon=%s",
        record.aqi,
        record.aqiMainPollutant,
        round(lat, 4),
        round(lon, 4),
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )
    return record


# ---------------------------------------------------------------------------
# Test reset helpers
# ---------------------------------------------------------------------------


def _reset_http_client_for_tests() -> None:
    """Reset module-level HTTP client singleton. Used in tests only."""
    global _http_client  # noqa: PLW0603
    _http_client = None
