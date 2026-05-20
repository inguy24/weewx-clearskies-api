"""Open-Meteo air-quality provider module (ADR-013, ADR-038).

Five responsibilities per ADR-038 §2:
  1. Outbound API call — single GET per cache miss:
       GET https://air-quality-api.open-meteo.com/v1/air-quality
     with current= projection listing all 13 variables (us_aqi + 6 sub-AQIs
     + 6 per-pollutant concentrations) and timezone=GMT (LC4).
  2. Response parsing — wire-shape Pydantic models _OpenMeteoCurrentBlock
     and _OpenMeteoAQResponse with extra="ignore" (LC5).
  3. Translation to canonical AQIReading (_wire_to_canonical):
       - aqi from current.us_aqi (rounded to int, capped at 500)
       - aqiScale = "epa" (us_aqi is EPA 0–500 native from provider)
       - aqiCategory = None (dashboard-computed from aqi+aqiScale)
       - aqiMainPollutant via argmax of 6 sub-AQI sub-fields (LC14)
       - aqiLocation always None (PARTIAL-DOMAIN per LC12 + L1 rule)
       - pollutantPM25/PM10/O3/NO2/SO2/CO pass through as µg/m³ (raw provider values)
       - observedAt = current.time + "Z" (LC4 — timezone=GMT, no double-shift)
       - source = "openmeteo"
  4. Capability declaration — CAPABILITY symbol consumed at startup.
  5. Error handling — ProviderHTTPClient.get() raises canonical taxonomy with
     all attributes set (L2 carry-forward rule, 3b-4 F1).  No re-construction.
     Only except clause is (ValidationError, ValueError) → ProviderProtocolError
     at the wire-validation boundary.

Open-Meteo is keyless (ADR-006 / LC11):
  No API key required.  auth_required=() (empty tuple).  No operator-managed
  secrets.  No env-var wiring needed for 3b-9.

Base URL distinct from forecast module (LC-endpoint):
  air-quality-api.open-meteo.com (NOT api.open-meteo.com used by forecast).

Cache layer (ADR-017 / LC3 / LC6 / LC7):
  TTL: 900s (15 min) per ADR-017 AQI domain.
  Key: SHA-256 of (provider_id, endpoint="aqi_current", {lat4, lon4}).
  Value: model_dump() dict (JSON-serializable for Redis backend).
  Sentinel: {"_no_reading": True} cached when provider returns all-null reading.
  Reconstruction on hit: AQIReading.model_validate(cached_dict).

Time conversion (ADR-020 / LC4):
  Open-Meteo returns current.time as "YYYY-MM-DDTHH:mm" — local-naive in the
  timezone requested.  We request timezone=GMT, so the wire IS GMT/UTC.
  Appending "Z" directly is correct.  Do NOT read utc_offset_seconds from the
  response — that would double-shift (we already asked for GMT).

aqiLocation PARTIAL-DOMAIN (L1 rule extension, 3b-7+):
  Open-Meteo air-quality API has no location-label field at any tier.
  aqiLocation is always None on the canonical AQIReading.
  Not in CAPABILITY.supplied_canonical_fields.

Rate limiter (LC8):
  max_calls=5, window_seconds=1 (courtesy guard; 15-min TTL → ~96 calls/day).

ruff: noqa: N815  (wire field names like us_aqi_pm2_5 don't need camelCase)
"""

# ruff: noqa: N815

from __future__ import annotations

import hashlib
import json
import logging

from pydantic import BaseModel, ConfigDict, ValidationError

from weewx_clearskies_api.models.responses import AQIReading
from weewx_clearskies_api.providers._common.cache import get_cache
from weewx_clearskies_api.providers._common.capability import ProviderCapability
from weewx_clearskies_api.providers._common.errors import ProviderProtocolError
from weewx_clearskies_api.providers._common.http import ProviderHTTPClient
from weewx_clearskies_api.providers._common.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PROVIDER_ID = "openmeteo"
DOMAIN = "aqi"
DEFAULT_AQI_TTL_SECONDS = 900  # 15 min per ADR-017 / LC3
_API_VERSION = "0.1.0"
OPENMETEO_AQ_BASE_URL = "https://air-quality-api.open-meteo.com"
OPENMETEO_AQ_PATH = "/v1/air-quality"

# Fixed current= CSV — all 13 variables in one request.
# us_aqi + 6 sub-AQIs (for aqiMainPollutant argmax) + 6 concentrations.
_REQUESTED_CURRENT_VARS = (
    "us_aqi,"
    "us_aqi_pm2_5,us_aqi_pm10,us_aqi_nitrogen_dioxide,"
    "us_aqi_ozone,us_aqi_sulphur_dioxide,us_aqi_carbon_monoxide,"
    "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
)

# Sub-AQI field order for argmax + canonical pollutant id mapping (LC14).
# Ties broken by table order (PM2.5 wins a tie with PM10 — deterministic).
_SUB_AQI_TO_POLLUTANT: list[tuple[str, str]] = [
    ("us_aqi_pm2_5",             "PM2.5"),
    ("us_aqi_pm10",              "PM10"),
    ("us_aqi_nitrogen_dioxide",  "NO2"),
    ("us_aqi_ozone",             "O3"),
    ("us_aqi_sulphur_dioxide",   "SO2"),
    ("us_aqi_carbon_monoxide",   "CO"),
]

# ---------------------------------------------------------------------------
# Capability declaration (ADR-038 §4)
# ---------------------------------------------------------------------------

CAPABILITY = ProviderCapability(
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    supplied_canonical_fields=(
        "aqi", "aqiCategory", "aqiMainPollutant",
        "pollutantPM25", "pollutantPM10",
        "pollutantO3", "pollutantNO2", "pollutantSO2", "pollutantCO",
        "observedAt", "source",
        # aqiLocation is PARTIAL-DOMAIN — Open-Meteo has no location field.
        # Not in this tuple per L1 rule extension (3b-7+).
    ),
    geographic_coverage="global",
    auth_required=(),  # keyless (LC11)
    default_poll_interval_seconds=DEFAULT_AQI_TTL_SECONDS,
    operator_notes=(
        "Open-Meteo air-quality endpoint. Keyless, no quota gate. "
        "Source data is CAMS European Air Quality Forecast (Europe) + CAMS "
        "Global Atmospheric Composition (rest of world). "
        "aqiLocation is not supplied by this provider (PARTIAL-DOMAIN per "
        "canonical §4.2 openmeteo column); always null on canonical bundle. "
        "aqiScale='epa' (us_aqi is EPA 0–500 native). aqiCategory=None — "
        "dashboard-computed from aqi+aqiScale. aqiMainPollutant derived "
        "client-side from per-pollutant sub-AQIs (provider does not supply "
        "either field directly). Per-gas concentrations passed through as "
        "µg/m³ (raw provider values; no conversion at ingest)."
    ),
)

# ---------------------------------------------------------------------------
# Wire-shape Pydantic models (LC5 — extra="ignore"; required fields enumerated)
# Source: docs/reference/api-docs/openmeteo.md (Air Quality subsection)
# ---------------------------------------------------------------------------


class _OpenMeteoCurrentBlock(BaseModel):
    """current= block of the air-quality response (LC5)."""

    model_config = ConfigDict(extra="ignore")

    # time is the only required field; all pollutant values are optional.
    # Open-Meteo may omit individual variables (regional model coverage gaps).
    time: str  # local-naive ISO ("YYYY-MM-DDTHH:mm") — wire is GMT per LC4

    # Overall US AQI
    us_aqi: float | None = None

    # Per-pollutant sub-AQIs (for aqiMainPollutant argmax per LC14)
    us_aqi_pm2_5: float | None = None
    us_aqi_pm10: float | None = None
    us_aqi_nitrogen_dioxide: float | None = None
    us_aqi_ozone: float | None = None
    us_aqi_sulphur_dioxide: float | None = None
    us_aqi_carbon_monoxide: float | None = None

    # Per-pollutant concentrations (µg/m³ — passed through as-is)
    pm2_5: float | None = None
    pm10: float | None = None
    ozone: float | None = None
    nitrogen_dioxide: float | None = None
    sulphur_dioxide: float | None = None
    carbon_monoxide: float | None = None


class _OpenMeteoAQResponse(BaseModel):
    """Top-level air-quality response with current= projection (LC5).

    Other top-level fields (elevation, generationtime_ms, utc_offset_seconds,
    timezone, timezone_abbreviation, current_units) are ignored via
    extra="ignore".  We do NOT use utc_offset_seconds — we requested
    timezone=GMT so the wire IS UTC (double-shift hazard per LC4 anti-pattern).
    """

    model_config = ConfigDict(extra="ignore")

    latitude: float
    longitude: float
    current: _OpenMeteoCurrentBlock


# ---------------------------------------------------------------------------
# Rate limiter (LC8 — "be polite" guard; 5 req/s max)
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="openmeteo-aqi",
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

    No target_unit dimension — AQI has no unit conversion at request time.
    Lat/lon rounded to 4 decimal places per ADR-017 §Cache key.
    Logical endpoint key "aqi_current" distinct from any other module's key.
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


def _main_pollutant_from_sub_aqis(current: _OpenMeteoCurrentBlock) -> str | None:
    """Return canonical pollutant id for the sub-AQI with the highest value.

    Per LC14: argmax over the six us_aqi_* sub-fields; None values excluded
    from the comparison.  Ties broken by table order (_SUB_AQI_TO_POLLUTANT)
    — PM2.5 wins a tie with PM10 (deterministic).

    Returns:
        Canonical pollutant id (e.g. "PM2.5", "O3", "CO") or None if all six
        sub-AQI values are None.
    """
    best_val: float | None = None
    best_pollutant: str | None = None

    for field_name, canonical_id in _SUB_AQI_TO_POLLUTANT:
        val = getattr(current, field_name, None)
        if val is None:
            continue
        # Strict > so the FIRST maximum wins (table-order tie-breaking).
        if best_val is None or val > best_val:
            best_val = val
            best_pollutant = canonical_id

    return best_pollutant


def _wire_to_canonical(wire: _OpenMeteoAQResponse) -> AQIReading | None:
    """Translate Open-Meteo wire response to canonical AQIReading.

    Returns None if no AQI value AND no per-pollutant concentration or
    sub-AQI value is populated — indicating no useful reading at this location.

    Otherwise constructs the canonical record per canonical §4.2:
      - aqi:               current.us_aqi (rounded to int if non-None; capped at 500)
      - aqiScale:          "epa" (Open-Meteo us_aqi is EPA 0–500 native)
      - aqiCategory:       None (dashboard-computed from aqi+aqiScale)
      - aqiMainPollutant:  argmax of sub-AQIs → canonical pollutant id (LC14)
      - aqiLocation:       None (PARTIAL-DOMAIN per LC12 / L1 rule)
      - pollutantPM25:     current.pm2_5 (µg/m³ — passthrough, group_concentration)
      - pollutantPM10:     current.pm10  (µg/m³ — passthrough, group_concentration)
      - pollutantO3:       current.ozone          (µg/m³ — raw provider value)
      - pollutantNO2:      current.nitrogen_dioxide (µg/m³ — raw provider value)
      - pollutantSO2:      current.sulphur_dioxide  (µg/m³ — raw provider value)
      - pollutantCO:       current.carbon_monoxide  (µg/m³ — raw provider value)
      - observedAt:        current.time + "Z" → UTC ISO-8601 (LC4)
      - source:            "openmeteo" (LC16)
    """
    current = wire.current

    # Check whether this response has any useful data at all.
    # If us_aqi AND all concentration fields are None, return None.
    has_data = current.us_aqi is not None or any(
        getattr(current, f) is not None
        for f in (
            "pm2_5", "pm10", "ozone",
            "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide",
            "us_aqi_pm2_5", "us_aqi_pm10", "us_aqi_nitrogen_dioxide",
            "us_aqi_ozone", "us_aqi_sulphur_dioxide", "us_aqi_carbon_monoxide",
        )
    )
    if not has_data:
        return None

    # AQI: round to int if non-None; cap at 500 (defensive per brief §per-module).
    aqi_raw = current.us_aqi
    aqi_int: float | None = None
    if aqi_raw is not None:
        aqi_int = min(round(aqi_raw), 500)

    # aqiMainPollutant: argmax of sub-AQIs (None if all sub-AQIs are null)
    main_pollutant = _main_pollutant_from_sub_aqis(current)

    # observedAt: append "Z" to the local-naive GMT timestamp (LC4).
    # current.time arrives as "YYYY-MM-DDTHH:mm" — Open-Meteo hourly grid.
    # Appending ":00Z" gives canonical UTC ISO-8601 with seconds.
    # Do NOT add utc_offset_seconds — would double-shift (we asked for GMT).
    observed_at = current.time + ":00Z"

    return AQIReading(
        aqi=aqi_int,
        aqiScale="epa",
        aqiCategory=None,
        aqiMainPollutant=main_pollutant,
        aqiLocation=None,          # PARTIAL-DOMAIN — Open-Meteo has no location field
        pollutantPM25=current.pm2_5,
        pollutantPM10=current.pm10,
        pollutantO3=current.ozone,
        pollutantNO2=current.nitrogen_dioxide,
        pollutantSO2=current.sulphur_dioxide,
        pollutantCO=current.carbon_monoxide,
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
    http_client: ProviderHTTPClient | None = None,
) -> AQIReading | None:
    """GET /v1/air-quality and return canonical AQIReading or None.

    None return: provider responded but us_aqi + all per-pollutant values
    were null (no useful reading for this location).

    Otherwise: canonical AQIReading with whatever fields the provider populated.

    L2 carry-forward (3b-4 F1): ProviderHTTPClient.get() raises members of the
    canonical taxonomy (KeyInvalid, QuotaExhausted, TransientNetworkError,
    ProviderProtocolError) with all attributes set (status_code,
    retry_after_seconds).  These propagate bare — do NOT re-construct.
    The only except clause here catches (ValidationError, ValueError) at the
    wire-validation boundary → ProviderProtocolError (this IS adding context
    the inner layer didn't have — wire-shape validation is a higher-level error).

    Args:
        lat: Station latitude from services/station.py StationInfo.
        lon: Station longitude from services/station.py StationInfo.
        http_client: Optional ProviderHTTPClient override for testing.
            When None, the module-level singleton is used.

    Returns:
        Canonical AQIReading or None (no useful reading at this location).

    Raises:
        QuotaExhausted: Provider returned 429 (rate limit exceeded).
        TransientNetworkError: Network/DNS failure or 5xx after retries.
        ProviderProtocolError: Response JSON validation failed.
        KeyInvalid: Provider returned 401/403 (unexpected for keyless, but handled).
    """
    cache_key = _build_cache_key(lat, lon)
    cached = get_cache().get(cache_key)
    if cached is not None:
        logger.debug(
            "Cache hit for Open-Meteo AQI",
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        # Sentinel for "no reading available at this location"
        if cached == {"_no_reading": True}:
            return None
        return AQIReading.model_validate(cached)

    logger.debug(
        "Cache miss for Open-Meteo AQI; calling %s%s",
        OPENMETEO_AQ_BASE_URL, OPENMETEO_AQ_PATH,
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )

    params = {
        "latitude": str(round(lat, 6)),
        "longitude": str(round(lon, 6)),
        "current": _REQUESTED_CURRENT_VARS,
        "timezone": "GMT",
    }

    client = http_client or _client_for()
    _rate_limiter.acquire()

    # L2 carry-forward: client.get() raises canonical taxonomy with all
    # attributes set.  Do NOT catch and re-raise as a new canonical exception
    # (would silently drop retry_after_seconds per 3b-4 F1 rule).
    response = client.get(OPENMETEO_AQ_BASE_URL + OPENMETEO_AQ_PATH, params=params)

    try:
        wire = _OpenMeteoAQResponse.model_validate(response.json())
    except (ValidationError, ValueError) as exc:
        logger.error(
            "Open-Meteo AQI response validation failed: %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        raise ProviderProtocolError(
            f"Open-Meteo AQI response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    record = _wire_to_canonical(wire)

    if record is None:
        # No AQI value present at all — cache the sentinel so the next dashboard
        # poll within TTL doesn't re-hit the provider unnecessarily.
        logger.info(
            "Open-Meteo AQI: no reading available for lat=%s lon=%s",
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
        "Open-Meteo AQI fetched: aqi=%s mainPollutant=%s for lat=%s lon=%s",
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
