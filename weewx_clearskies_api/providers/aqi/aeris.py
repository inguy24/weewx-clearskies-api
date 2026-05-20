"""Aeris (AerisWeather/Xweather) AQI provider module (ADR-013, ADR-038).

Five responsibilities per ADR-038 §2:
  1. Outbound API call — single GET per cache miss:
       GET https://data.api.xweather.com/airquality/{lat},{lon}
     with filter=airnow (lock US EPA methodology per ADR-013 / LC21) and
     keyed query-param credentials (client_id + client_secret per LC11).
  2. Response parsing — wire-shape Pydantic models for the Aeris
     success/error/response[]/periods[]/pollutants[] envelope (LC5).
     extra="ignore" on all models (Aeris carries many fields canonical
     AQIReading does not consume: color, method, health.*, per-pollutant
     aqi/category/color/method/name, profile, loc, id).
  3. Translation to canonical AQIReading (_wire_to_canonical):
       - aqi from periods[0].aqi (rounded to int, capped at 500)
       - aqiScale = "epa" (Aeris /airquality?filter=airnow is EPA 0–500 native)
       - aqiCategory = None (dashboard-computed from aqi+aqiScale)
       - aqiMainPollutant normalized from periods[0].dominant lowercase id
         to canonical id via _DOMINANT_TO_CANONICAL lookup (LC14)
       - aqiLocation from place.name (NOT PARTIAL-DOMAIN — Aeris supplies
         this; distinct from Open-Meteo which omits it)
       - pollutantPM25/PM10: periods[0].pollutants[] filtered by type,
         valueUGM3 passthrough in µg/m³ (LC15 + §4.2 aeris column)
       - pollutantO3/NO2/SO2/CO: valuePPB converted via ppb_to_ugm3()
         (formula: µg/m³ = ppb × MW / 24.45; Aeris returns valuePPB
         directly for gases)
       - observedAt: periods[0].dateTimeISO parsed as explicit-offset ISO
         string → UTC Z form via to_utc_iso8601_from_offset() (LC4)
       - source = "aeris"
  4. Capability declaration — CAPABILITY symbol consumed at startup.
     Full paid-tier max surface (12 fields) per L1 rule — Aeris supplies
     every canonical AQI field including aqiLocation.
  5. Error handling — ProviderHTTPClient.get() raises canonical taxonomy
     with all attributes set (L2 carry-forward, 3b-4 audit F1). No
     re-construction of canonical exceptions from HTTP-level errors.
     LC27 envelope mapping (intentional wire-level wrap — adds context
     the inner layer didn't have): success=false + error.code dispatch
     to KeyInvalid / QuotaExhausted / ProviderProtocolError.

Aeris is a keyed provider (ADR-006):
  client_id + client_secret passed as query params (LC11, LC22).
  Credentials NOT in the cache key (LC7 — privacy/leakage concern).
  Provider-scoped per 3b-4 Q1 user decision — same [aeris] section as
  forecast/alerts Aeris. Credentials live at settings.aeris.client_id +
  settings.aeris.client_secret; wired at startup via wire_aqi_settings().

Cache layer (ADR-017 / LC3 / LC6 / LC7):
  TTL: 900s (15 min) per ADR-017 AQI domain.
  Key: SHA-256 of (provider_id="aeris", endpoint="aqi_current", {lat4, lon4}).
  Credentials NOT in key (LC7 — privacy/leakage concern).
  Value: model_dump() dict (JSON-serializable for Redis backend).
  Sentinel: {"_no_reading": True} when provider returns empty response or
    all-null reading (cached so re-polls within TTL don't hit the provider).
  Reconstruction on hit: AQIReading.model_validate(cached_dict).

PM1 handling (LC26):
  Aeris pollutants[] may include {"type": "pm1", ...}.  Canonical AQIReading
  has no pollutantPM1 field — pm1 is silently dropped during translation.
  _DOMINANT_TO_CANONICAL does NOT include "pm1"; if dominant == "pm1",
  aqiMainPollutant = None with a logger.info notice.

LC27 envelope mapping (200-success-false):
  Aeris returns success=false on a 200 HTTP response for tier/auth/query
  errors. This IS adding context the inner layer didn't have (ProviderHTTPClient
  only sees HTTP-level status; the 200-with-error envelope is wire-level).
  error.code dispatch:
    "invalid_client" | "insufficient_scope" | "unauthorized" |
    "forbidden_access"              → KeyInvalid
    "maxhits_min"                   → QuotaExhausted (retry_after_seconds=None;
                                      not a 429 so no Retry-After from Aeris)
    anything else where success=False → ProviderProtocolError

Rate limiter (LC8):
  max_calls=5, window_seconds=1 (be-polite guard; 15-min TTL → ~96 calls/day).

Time conversion (ADR-020 / LC4):
  Aeris returns periods[0].dateTimeISO with an explicit UTC offset
  (e.g. "2026-04-30T10:00:00-07:00").  to_utc_iso8601_from_offset() from
  _common/datetime_utils.py handles explicit-offset → UTC Z form.  This is
  the DRY reuse of the shared helper (already used by alerts/aeris.py and
  forecast modules).  NOT epoch_to_utc_iso8601() — dateTimeISO is the right
  field per LC4.

ruff: noqa: N815  (wire field names include camelCase: dateTimeISO, valuePPB, valueUGM3)
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
from weewx_clearskies_api.providers._common.datetime_utils import to_utc_iso8601_from_offset
from weewx_clearskies_api.providers._common.errors import (
    KeyInvalid,
    ProviderProtocolError,
    QuotaExhausted,
)
from weewx_clearskies_api.providers._common.http import ProviderHTTPClient
from weewx_clearskies_api.providers._common.rate_limiter import RateLimiter
from weewx_clearskies_api.providers.aqi._units import ppb_to_ugm3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PROVIDER_ID = "aeris"
DOMAIN = "aqi"
DEFAULT_AQI_TTL_SECONDS = 900  # 15 min per ADR-017 / LC3
_API_VERSION = "0.1.0"
AERIS_AQ_BASE_URL = "https://data.api.xweather.com"
AERIS_AQ_PATH_TMPL = "/airquality/{lat},{lon}"  # location in path (LC22)

# Aeris dominant pollutant id (lowercase) → canonical pollutant id (LC14).
# pm1 intentionally omitted — canonical AQIReading has no pollutantPM1 field.
# If Aeris reports pm1 as dominant, aqiMainPollutant = None (LC26).
_DOMINANT_TO_CANONICAL: dict[str, str] = {
    "pm2.5": "PM2.5",
    "pm10":  "PM10",
    "o3":    "O3",
    "no2":   "NO2",
    "so2":   "SO2",
    "co":    "CO",
    # "pm1" intentionally omitted — canonical has no pollutantPM1 field (LC26)
}

# Aeris pollutants[].type (lowercase) → canonical AQIReading field name (LC15).
# pm1 intentionally omitted — dropped during translation (LC26).
_TYPE_TO_CANONICAL_FIELD: dict[str, str] = {
    "pm2.5": "pollutantPM25",
    "pm10":  "pollutantPM10",
    "o3":    "pollutantO3",
    "no2":   "pollutantNO2",
    "so2":   "pollutantSO2",
    "co":    "pollutantCO",
    # "pm1" intentionally omitted (LC26)
}

# Canonical fields that come from valuePPB (gases) need ppb_to_ugm3() conversion.
# Particulate fields (pollutantPM25 / pollutantPM10) read valueUGM3 directly
# via the unconditional else: branch in _wire_to_canonical's pollutant loop.
_GAS_FIELDS: frozenset[str] = frozenset({
    "pollutantO3", "pollutantNO2", "pollutantSO2", "pollutantCO",
})

# Maps canonical gas field name to the canonical pollutant id used by ppb_to_ugm3.
_GAS_FIELD_TO_POLLUTANT: dict[str, str] = {
    "pollutantO3":  "O3",
    "pollutantNO2": "NO2",
    "pollutantSO2": "SO2",
    "pollutantCO":  "CO",
}

# Aeris success=false error codes that indicate auth/credential failure (LC27).
# Maps to KeyInvalid (permanent until operator updates config).
_KEY_INVALID_CODES: frozenset[str] = frozenset({
    "invalid_client",
    "insufficient_scope",
    "unauthorized",
    "forbidden_access",
})

# Aeris success=false error codes that indicate rate limiting (LC27).
# Maps to QuotaExhausted (transient, retry after backoff).
_QUOTA_EXHAUSTED_CODES: frozenset[str] = frozenset({
    "maxhits_min",
})

# ---------------------------------------------------------------------------
# Capability declaration (ADR-038 §4)
# ---------------------------------------------------------------------------

CAPABILITY = ProviderCapability(
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    supplied_canonical_fields=(
        # Full paid-tier max surface per L1 rule — Aeris supplies all 12 fields.
        # aqiLocation is NOT partial-domain for Aeris (place.name is present);
        # distinct from Open-Meteo which omits aqiLocation.
        "aqi", "aqiCategory", "aqiMainPollutant", "aqiLocation",
        "pollutantPM25", "pollutantPM10",
        "pollutantO3", "pollutantNO2", "pollutantSO2", "pollutantCO",
        "observedAt", "source",
    ),
    geographic_coverage="global",
    auth_required=("client_id", "client_secret"),  # LC11
    default_poll_interval_seconds=DEFAULT_AQI_TTL_SECONDS,
    operator_notes=(
        "Aeris (Xweather) /airquality endpoint with filter=airnow (US EPA AQI). "
        "Keyed (query-param client_id + client_secret; reuses provider-scoped "
        "credentials from forecast/alerts Aeris — same [aeris] config section). "
        "Gas concentrations converted PPB→µg/m³ via providers/aqi/_units.ppb_to_ugm3 "
        "(formula: µg/m³ = ppb × MW / 24.45; Aeris returns valuePPB directly for O3/NO2/SO2/CO). "
        "aqiScale='epa'; aqiCategory=None (dashboard-computed from aqi+aqiScale). "
        "aqiMainPollutant normalized from lowercase periods[].dominant to canonical id. "
        "pm1 dropped during translation (no pollutantPM1 field on canonical AQIReading). "
        "aqiLocation supplied via place.name (NOT PARTIAL-DOMAIN for Aeris)."
    ),
)

# ---------------------------------------------------------------------------
# Wire-shape Pydantic models (LC5 — extra="ignore"; required fields enumerated)
# Source: docs/reference/api-docs/aeris.md §Air Quality
# ---------------------------------------------------------------------------


class _AerisPollutant(BaseModel):
    """One pollutant entry in periods[0].pollutants[] (LC5).

    Aeris returns an array of typed objects; filter by `type` to extract each
    canonical pollutant.  Gas pollutants (o3, no2, so2, co) populate valuePPB;
    particulates (pm2.5, pm10) populate valueUGM3 (valuePPB is null).
    """

    model_config = ConfigDict(extra="ignore")

    type: str  # "pm2.5", "pm10", "o3", "no2", "so2", "co", "pm1"
    valuePPB: float | None = None    # parts per billion (gases; null for particulates)
    valueUGM3: float | None = None   # µg/m³ (particulates; also present for gases but unused)
    # name, aqi, category, color, method are present on the wire but not consumed
    # by the canonical AQIReading model.


class _AerisPlace(BaseModel):
    """response[0].place (LC5) — source of aqiLocation."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None      # → aqiLocation (LC12 / canonical §4.2 aeris column)
    state: str | None = None
    country: str | None = None


class _AerisPeriod(BaseModel):
    """response[0].periods[0] (LC5) — the current observation period."""

    model_config = ConfigDict(extra="ignore")

    dateTimeISO: str              # explicit-offset ISO e.g. "2026-04-30T10:00:00-07:00"
    aqi: float | None = None     # overall US EPA AQI value
    dominant: str | None = None  # lowercase pollutant id e.g. "pm2.5"
    pollutants: list[_AerisPollutant] = []
    # category, color, method, health, timestamp are present but not consumed.


class _AerisLocation(BaseModel):
    """response[0] — one location object from the response array (LC5)."""

    model_config = ConfigDict(extra="ignore")

    place: _AerisPlace | None = None
    periods: list[_AerisPeriod] = []
    # id, loc, profile are present on the wire but not consumed.


class _AerisError(BaseModel):
    """Aeris error sub-object (present on success=false or success=true with warning)."""

    model_config = ConfigDict(extra="ignore")

    code: str
    description: str | None = None


class _AerisAQResponse(BaseModel):
    """Top-level airquality response envelope (LC5).

    Aeris's `:id` action on /airquality returns response as an ARRAY containing
    one location object (per api-docs wire-shape notes — the `:id` action is
    documented as single-object for most endpoints, but airquality specifically
    returns response as a list).  The module reads response[0].

    success=false with error.code → _raise_for_envelope_error() maps to
    the canonical taxonomy per LC27.
    """

    model_config = ConfigDict(extra="ignore")

    success: bool
    error: _AerisError | None = None
    response: list[_AerisLocation] = []


# ---------------------------------------------------------------------------
# Rate limiter (LC8 — "be polite" guard; 5 req/s max)
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="aeris-aqi",
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

    Credentials NOT in the key per LC7 — privacy/leakage concern; cache scope is
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
# LC27 envelope error dispatch
# ---------------------------------------------------------------------------


def _raise_for_envelope_error(wire: _AerisAQResponse) -> None:
    """Map Aeris 200-success-false envelope to the canonical taxonomy (LC27).

    This IS an intentional wrap — we're adding context the inner layer (
    ProviderHTTPClient) didn't have.  ProviderHTTPClient sees only HTTP status
    codes; the 200-with-error envelope is a wire-level Aeris protocol detail.
    Documented in commit body per non-obvious-provenance rule.

    LC27 dispatch table:
      error.code in _KEY_INVALID_CODES  → KeyInvalid
      error.code in _QUOTA_EXHAUSTED_CODES → QuotaExhausted (retry_after_seconds=None)
      anything else                     → ProviderProtocolError

    Raises:
        KeyInvalid: auth/credential failure codes.
        QuotaExhausted: rate-limit codes (no retry_after on 200-not-429).
        ProviderProtocolError: all other success=false error codes.
    """
    error_code = ""
    error_desc = "unknown error"
    if wire.error:
        error_code = wire.error.code
        error_desc = wire.error.description or "unknown error"

    logger.error(
        "Aeris AQI returned success=false: code=%r description=%r",
        error_code,
        error_desc,
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )

    if error_code in _KEY_INVALID_CODES:
        raise KeyInvalid(
            f"Aeris AQI auth failure: code={error_code!r} — {error_desc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )
    if error_code in _QUOTA_EXHAUSTED_CODES:
        # 200-not-429, so no Retry-After from Aeris; retry_after_seconds=None.
        raise QuotaExhausted(
            f"Aeris AQI rate limit: code={error_code!r} — {error_desc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
            retry_after_seconds=None,
        )
    raise ProviderProtocolError(
        f"Aeris AQI returned success=false: code={error_code!r} — {error_desc}",
        provider_id=PROVIDER_ID,
        domain=DOMAIN,
    )


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------


def _wire_to_canonical(location: _AerisLocation) -> AQIReading | None:
    """Translate Aeris response[0] to canonical AQIReading.

    Caller guarantees location.periods is non-empty (fetch() checks before call).

    Returns:
        Canonical AQIReading or None if aqi + all pollutant values are null
        (no useful reading at this location).
    """
    period = location.periods[0]

    # aqi: round to int and cap at 500 (defensive; EPA scale is 0-500;
    # provider-side bugs producing 501+ shouldn't crash us).
    aqi_int: int | None = None
    if period.aqi is not None:
        aqi_int = min(round(period.aqi), 500)

    # aqiMainPollutant: normalize Aeris lowercase dominant id to canonical (LC14).
    dominant_raw = period.dominant or ""
    main_pollutant = _DOMINANT_TO_CANONICAL.get(dominant_raw)
    if not main_pollutant and dominant_raw:
        # Unmappable dominant (e.g. "pm1" — no pollutantPM1 on canonical).
        # Log + return None for this field (LC26).
        logger.info(
            "Aeris AQI dominant pollutant %r not in canonical id table; "
            "aqiMainPollutant=None",
            dominant_raw,
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )

    # aqiLocation: place.name (NOT PARTIAL-DOMAIN for Aeris — supplied per LC12).
    aqi_location: str | None = None
    if location.place is not None:
        aqi_location = location.place.name

    # Pollutant values: filter pollutants[] by type, extract the right value
    # field, and convert PPB → µg/m³ for gases (formula: µg/m³ = ppb × MW / 24.45).
    # pm1 and any unknown type are silently skipped (LC26 / _TYPE_TO_CANONICAL_FIELD).
    pollutant_values: dict[str, float | None] = {}
    for entry in period.pollutants:
        canonical_field = _TYPE_TO_CANONICAL_FIELD.get(entry.type.lower())
        if canonical_field is None:
            # pm1 or unknown type — skip silently (LC26).
            continue
        if canonical_field in _GAS_FIELDS:
            # Gas: convert PPB → µg/m³ via ppb_to_ugm3 (MW-based formula).
            # Aeris returns valuePPB directly for gases; valueUGM3 is also present
            # for gases but using it would skip the ppb→µg/m³ round-trip here.
            pollutant_id = _GAS_FIELD_TO_POLLUTANT[canonical_field]
            pollutant_values[canonical_field] = (
                ppb_to_ugm3(entry.valuePPB, pollutant=pollutant_id)
                if entry.valuePPB is not None else None
            )
        else:
            # Particulate: valueUGM3 passthrough in µg/m³.
            pollutant_values[canonical_field] = entry.valueUGM3

    # observedAt: parse explicit-offset ISO → UTC Z form (LC4 / ADR-020).
    # Using the shared to_utc_iso8601_from_offset() helper (DRY — already
    # used by alerts/aeris.py and forecast modules; same wire-field shape).
    # NOT epoch_to_utc_iso8601() — dateTimeISO carries the explicit offset
    # and is the right field per LC4.
    observed_at = to_utc_iso8601_from_offset(
        period.dateTimeISO,
        provider_id=PROVIDER_ID,
        domain=DOMAIN,
    )

    # Empty-result guard: if aqi AND all pollutant values are null, return None.
    has_data = aqi_int is not None or any(
        v is not None for v in pollutant_values.values()
    )
    if not has_data:
        return None

    return AQIReading(
        aqi=aqi_int,
        aqiScale="epa",
        aqiCategory=None,
        aqiMainPollutant=main_pollutant,
        aqiLocation=aqi_location,
        pollutantPM25=pollutant_values.get("pollutantPM25"),
        pollutantPM10=pollutant_values.get("pollutantPM10"),
        pollutantO3=pollutant_values.get("pollutantO3"),
        pollutantNO2=pollutant_values.get("pollutantNO2"),
        pollutantSO2=pollutant_values.get("pollutantSO2"),
        pollutantCO=pollutant_values.get("pollutantCO"),
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
    client_id: str,
    client_secret: str,
    http_client: ProviderHTTPClient | None = None,
) -> AQIReading | None:
    """GET /airquality/{lat},{lon}?filter=airnow and return canonical AQIReading or None.

    Cache-first: checks the cache before making an outbound HTTP call.
    Cache stores post-normalization AQIReading as a model_dump() dict (JSON-
    serializable for Redis per ADR-017); reconstructed via model_validate() on hit.

    None return: provider responded but no useful reading available (empty
    response array, empty periods, or all fields null).

    L2 carry-forward (3b-4 audit F1): ProviderHTTPClient.get() raises canonical
    taxonomy exceptions (KeyInvalid, QuotaExhausted, TransientNetworkError,
    ProviderProtocolError) with all structured attributes set (status_code,
    retry_after_seconds).  These propagate bare — do NOT re-construct.

    The LC27 envelope-mapping (_raise_for_envelope_error) IS an intentional
    wrap: we're adding context (wire-level 200-success-false error code → canonical
    taxonomy member) that ProviderHTTPClient couldn't add (it only sees HTTP status).

    Args:
        lat: Station latitude (from services/station.py StationInfo).
        lon: Station longitude (from services/station.py StationInfo).
        client_id: Aeris client_id (from settings.aeris.client_id).
        client_secret: Aeris client_secret (from settings.aeris.client_secret).
        http_client: Optional ProviderHTTPClient override for testing.
            When None, the module-level singleton is used.

    Returns:
        Canonical AQIReading or None (no useful reading at this location).

    Raises:
        KeyInvalid: 401/403 from provider OR success=false with auth-failure code.
        QuotaExhausted: 429 from provider OR success=false with rate-limit code.
        TransientNetworkError: Network/DNS failure or 5xx after retries.
        ProviderProtocolError: Response JSON validation failed OR success=false
            with other error code.
    """
    cache_key = _build_cache_key(lat, lon)
    cached = get_cache().get(cache_key)
    if cached is not None:
        logger.debug(
            "Cache hit for Aeris AQI",
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        if cached == {"_no_reading": True}:
            return None
        return AQIReading.model_validate(cached)

    logger.debug(
        "Cache miss for Aeris AQI; calling %s",
        AERIS_AQ_BASE_URL + AERIS_AQ_PATH_TMPL.format(lat=round(lat, 6), lon=round(lon, 6)),
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )

    url = AERIS_AQ_BASE_URL + AERIS_AQ_PATH_TMPL.format(
        lat=round(lat, 6),
        lon=round(lon, 6),
    )
    # Credentials in query params (not URL path — avoids logging creds at INFO
    # level if the URL is logged; they stay in the params dict per LC11/LC22).
    # filter=airnow locks US EPA AQI methodology per ADR-013 / LC21.
    params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "filter": "airnow",
    }

    client = http_client or _client_for()
    _rate_limiter.acquire()

    # L2 carry-forward: client.get() raises canonical taxonomy with all
    # attributes set.  Do NOT catch and re-raise as a new canonical exception
    # (silently drops retry_after_seconds per 3b-4 audit F1 rule).
    response = client.get(url, params=params)

    try:
        wire = _AerisAQResponse.model_validate(response.json())
    except (ValidationError, ValueError) as exc:
        logger.error(
            "Aeris AQI response validation failed: %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        raise ProviderProtocolError(
            f"Aeris AQI response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    # LC27 envelope check — success=false means a wire-level error.
    # _raise_for_envelope_error dispatches error.code to the canonical taxonomy.
    # This is an intentional wrap: adds context the inner layer didn't have.
    if not wire.success:
        _raise_for_envelope_error(wire)

    # Empty response or empty periods — no reading at this location.
    if not wire.response or not wire.response[0].periods:
        logger.info(
            "Aeris AQI: no reading available for lat=%s lon=%s "
            "(empty response or empty periods)",
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

    record = _wire_to_canonical(wire.response[0])

    if record is None:
        # All-null reading — cache sentinel so re-polls within TTL skip the provider.
        logger.info(
            "Aeris AQI: all-null reading for lat=%s lon=%s",
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
        "Aeris AQI fetched: aqi=%s mainPollutant=%s aqiLocation=%r for lat=%s lon=%s",
        record.aqi,
        record.aqiMainPollutant,
        record.aqiLocation,
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
