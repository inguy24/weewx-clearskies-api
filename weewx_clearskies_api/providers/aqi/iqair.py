"""IQAir AirVisual AQI provider module (ADR-013, ADR-038).

Fourth + final AQI provider on the project (3b-12). Closes the AQI domain.

Five responsibilities per ADR-038 §2:
  1. Outbound API call — single GET per cache miss:
       GET https://api.airvisual.com/v2/nearest_city?lat={lat}&lon={lon}&key={key}
     Query-param key= is the API credential (NOT X-Key header — verified against
     pyairvisual source: kwargs["params"]["key"] = self._api_key).
     Lat/lon rounded to 6 decimal places per OWM/Aeris precedent.
  2. Response parsing — wire-shape Pydantic models with extra="ignore":
     _IQAirPollution / _IQAirWeather / _IQAirCurrent / _IQAirData / _IQAirResponse.
     No concentration fields declared (paid-tier only; unverified wire path).
  3. Translation to canonical AQIReading (_wire_to_canonical):
       - aqi from data.current.pollution.aqius (US EPA 0-500; NO conversion needed;
         distinct from OWM 1-5 ordinal + Open-Meteo sub-AQI computation paths)
       - aqiScale = "epa" (aqius is EPA 0–500 native from provider)
       - aqiCategory = None (dashboard-computed from aqi+aqiScale)
       - aqiMainPollutant normalized from mainus code to canonical id via
         _MAINUS_TO_CANONICAL lookup (LC2; mirrors Aeris _DOMINANT_TO_CANONICAL)
       - aqiLocation = f"{data.city}, {data.state}" (LC4 / Q3 user decision
         2026-05-11; None if either field missing)
       - pollutantPM25/PM10/O3/NO2/SO2/CO = None (LC5 — PARTIAL-DOMAIN on
         free Community tier; categorical absence, not tier-conditional)
       - observedAt: pollution.ts parsed via to_utc_iso8601_from_offset() (LC6;
         Py 3.11+ fromisoformat accepts Z suffix; DRY reuse, no new helper)
       - source = "iqair"
  4. Capability declaration — CAPABILITY symbol consumed at startup.
     Conservative scope per Q2 user decision 2026-05-11: 6 verified free-tier
     fields only; pollutant concentration fields stay PARTIAL-DOMAIN.
  5. Error handling:
       - ProviderHTTPClient.get() raises canonical taxonomy with all attributes
         set (L2 carry-forward, 3b-4 audit F1). Do NOT re-construct.
       - LC12/LC27 envelope mapping (intentional wire-level wrap — status:"fail"
         on a 200 response; adds context ProviderHTTPClient didn't have):
         dispatch on data.message string → KeyInvalid / QuotaExhausted /
         ProviderProtocolError. Documented here per non-obvious-provenance rule.
       - Wire-shape validation: (ValidationError, ValueError) → ProviderProtocolError
         (intentional wrap — adds wire-context the inner layer didn't have;
         per OWM/Aeris precedent). Documented per non-obvious-provenance rule.
       - Pre-call empty/None key check → KeyInvalid (LC13 fail-fast; mirrors
         OWM openweathermap.py:493-499 appid guard).

IQAir is AQI-only (not in forecast/alerts). Credential lives on AQISettings.iqair_key
per Q1 user decision 2026-05-11 (Option A: domain-scoped). Distinct from Aeris/OWM
which are provider-scoped (multi-domain).

Cache layer (ADR-017 / LC9):
  TTL: 900s (15 min) per ADR-017 AQI domain.
  Key: SHA-256 of (provider_id="iqair", endpoint="aqi_current", {lat4, lon4}).
  Credential NOT in key (LC9 — privacy/leakage concern; same as Aeris/OWM).
  Sentinel: {"_no_reading": True} for empty/null pollution block.
  Reconstruction on hit: AQIReading.model_validate(cached_dict).

Rate limiter (LC10):
  max_calls=5, window_seconds=60 — honors IQAir Community per-minute cap directly.
  STRICTER than OWM/Aeris (per-second) because IQAir's per-minute cap is the most
  restrictive of its three limits (5/min, 500/day, 10000/month).
  With 15-min TTL → ~96 calls/day, well within all three limits.

Envelope mapping (LC12/LC27):
  IQAir uses 200-success-false envelope (same shape as Aeris):
    status:"fail" + data.message → dispatch on message string:
      "incorrect_api_key" / "api_key_expired" / "payment required" /
      "permission_denied" / "forbidden" / "feature_not_available" → KeyInvalid
      "call_limit_reached" / "too_many_requests"                   → QuotaExhausted
      everything else (city_not_found, no_nearest_station, etc.)  → ProviderProtocolError
  This IS an intentional wrap — ProviderHTTPClient sees only HTTP status; the
  200-with-error envelope is wire-level IQAir protocol detail.

Pollutant code lookup:
  p1/p2/n2 confirmed via published examples; o3/s2/co inferred from naming
  convention. Real-capture should verify; unmappable codes → None + logger.info
  (LC3; mirrors Aeris pm1 handling).
"""

# ruff: noqa: N815  (wire field names include camelCase where needed)

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PROVIDER_ID = "iqair"
DOMAIN = "aqi"
DEFAULT_AQI_TTL_SECONDS = 900  # 15 min per ADR-017 / LC9
_API_VERSION = "0.1.0"
IQAIR_BASE_URL = "https://api.airvisual.com"
IQAIR_NEAREST_CITY_PATH = "/v2/nearest_city"

# IQAir mainus/maincn pollutant code → canonical pollutant id (LC2).
# p1/p2/n2 confirmed via published examples + third-party docs.
# o3/s2/co inferred from naming convention (IQAir AirVisual Pro data export
# uses p1_sum/p2_sum pattern consistently).  Real-capture should verify
# o3/s2/co; if a different code appears, amend this table + log entry.
# Unmappable codes return None for aqiMainPollutant with logger.info (LC3).
_MAINUS_TO_CANONICAL: dict[str, str] = {
    "p1": "PM10",
    "p2": "PM2.5",
    "n2": "NO2",
    "o3": "O3",
    "s2": "SO2",
    "co": "CO",
}

# IQAir error message strings that indicate auth/credential failure (LC12/LC27).
# From pyairvisual cloud_api.py ERROR_CODES dispatch table.
# Maps to KeyInvalid (permanent until operator updates config).
_KEY_INVALID_MESSAGES: frozenset[str] = frozenset({
    "incorrect_api_key",
    "api_key_expired",
    "payment required",
    "permission_denied",
    "forbidden",
    "feature_not_available",
})

# IQAir error message strings that indicate rate limiting (LC12/LC27).
# Maps to QuotaExhausted (transient, retry after backoff).
_QUOTA_EXHAUSTED_MESSAGES: frozenset[str] = frozenset({
    "call_limit_reached",
    "too_many_requests",
})

# ---------------------------------------------------------------------------
# Capability declaration (ADR-038 §4)
# ---------------------------------------------------------------------------

CAPABILITY = ProviderCapability(
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    supplied_canonical_fields=(
        # Conservative free-tier scope per Q2 user decision 2026-05-11.
        # Only the 6 fields verified on the Community (free) tier wire.
        # Pollutant concentrations (pollutantPM25/PM10/O3/NO2/SO2/CO) are
        # PARTIAL-DOMAIN on free Community tier — categorical absence (the data
        # isn't there at any time on this tier), not tier-conditional null.
        # A future round can lift them after paid-tier real-capture confirms
        # the wire field names and units (likely Startup+ plan).
        "aqi", "aqiCategory", "aqiMainPollutant", "aqiLocation",
        "observedAt", "source",
    ),
    geographic_coverage="global",
    auth_required=("key",),  # LC8 — single query-param credential; distinct from Aeris pair
    default_poll_interval_seconds=DEFAULT_AQI_TTL_SECONDS,
    operator_notes=(
        "IQAir AirVisual /v2/nearest_city endpoint (Community / free plan). "
        "Auth: query-param key= (NOT X-Key header — verified against pyairvisual source). "
        "Env var: WEEWX_CLEARSKIES_IQAIR_KEY (provider-scoped per LC11). "
        "Credential lives on AQISettings.iqair_key (domain-scoped per Q1 user decision "
        "2026-05-11 — IQAir is AQI-only, distinct from multi-domain Aeris/OWM). "
        "Rate limiter: 5/min (window_seconds=60) — IQAir Community per-minute cap; "
        "≈12× tighter per minute than OWM/Aeris's 5-req/sec limiters (which allow ~300/min). "
        "With 15-min TTL → ~96 calls/day, well within 500/day + 10000/month caps. "
        "aqius (US EPA AQI 0-500) is published directly; no breakpoint computation needed "
        "(distinct from OWM 1-5 ordinal and Open-Meteo sub-AQI computation paths). "
        "aqiScale='epa'; aqiCategory=None (dashboard-computed from aqi+aqiScale). "
        "aqiMainPollutant from mainus code via _MAINUS_TO_CANONICAL lookup "
        "(p1=PM10, p2=PM2.5, n2=NO2, o3=O3, s2=SO2, co=CO; "
        "p1/p2/n2 confirmed, o3/s2/co inferred — real-capture should verify). "
        "Unmappable mainus codes → None + logger.info notice (LC3). "
        "aqiLocation = f'{city}, {state}' (comma+space per Q3 user decision 2026-05-11). "
        "pollutantPM25/PM10/O3/NO2/SO2/CO = None (PARTIAL-DOMAIN on free Community tier; "
        "wire path for paid Startup+ tier unverified at 3b-12 brief time). "
        "Envelope: 200-success-false (status:'fail' + data.message dispatch, LC12/LC27). "
        "Known error message strings: incorrect_api_key, api_key_expired, payment required, "
        "permission_denied, forbidden, feature_not_available, call_limit_reached, "
        "too_many_requests, city_not_found, no_nearest_station, node not found."
    ),
)

# ---------------------------------------------------------------------------
# Wire-shape Pydantic models (extra="ignore"; required fields enumerated)
# Source: docs/reference/api-docs/iqair.md
# Free Community tier: pollution block = {ts, aqius, mainus, aqicn, maincn} only.
# Paid Startup+ tier adds per-pollutant concentrations (wire field names unverified).
# ---------------------------------------------------------------------------


class _IQAirPollution(BaseModel):
    """data.current.pollution — the air quality observation block.

    Free Community tier: ts, aqius, mainus, aqicn, maincn only.
    No concentration fields declared (paid-tier wire path unverified).
    extra="ignore" drops any paid-tier fields that might appear on a higher plan.
    """

    model_config = ConfigDict(extra="ignore")

    ts: str  # ISO-8601 UTC with ms + Z suffix e.g. "2019-04-08T18:00:00.000Z"
    aqius: int | None = None      # US EPA AQI 0-500 (direct; no conversion needed)
    mainus: str | None = None     # dominant pollutant code in US AQI e.g. "p2"
    aqicn: int | None = None      # China AQI (not consumed by canonical)
    maincn: str | None = None     # dominant pollutant code in China AQI (not consumed)


class _IQAirWeather(BaseModel):
    """data.current.weather — current weather snapshot (not consumed by AQI canonical).

    Declared so the _IQAirCurrent envelope validates cleanly.
    All fields extra="ignore" drops; only ts declared to anchor the model.
    """

    model_config = ConfigDict(extra="ignore")

    ts: str | None = None  # ISO-8601 UTC; may differ from pollution.ts by an hour


class _IQAirCurrent(BaseModel):
    """data.current — weather + pollution sub-object."""

    model_config = ConfigDict(extra="ignore")

    weather: _IQAirWeather | None = None
    pollution: _IQAirPollution  # required; if absent the response is malformed


class _IQAirData(BaseModel):
    """data — the main location + readings object."""

    model_config = ConfigDict(extra="ignore")

    city: str | None = None      # e.g. "Nashville" → part of aqiLocation (LC4)
    state: str | None = None     # e.g. "Tennessee" → part of aqiLocation (LC4)
    country: str | None = None   # e.g. "USA" (not used in aqiLocation per Q3)
    current: _IQAirCurrent       # required; if absent the response is malformed


class _IQAirResponse(BaseModel):
    """Top-level envelope.

    success: IQAir uses status:"success" / status:"fail" (string, not bool).
    On status:"fail", data.message carries the error string (LC12/LC27).
    Extra fields ignored — IQAir may add top-level metadata in future versions.
    """

    model_config = ConfigDict(extra="ignore")

    status: str               # "success" or "fail"
    data: _IQAirData | None = None  # present on success; on fail may be a nested message dict


# ---------------------------------------------------------------------------
# Rate limiter (LC10 — per-minute cap is most restrictive IQAir Community limit)
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="iqair-aqi",
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    max_calls=5,
    window_seconds=60,  # per-minute (stricter than OWM/Aeris per-second limiters)
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
# Cache key construction (ADR-017 / LC9)
# ---------------------------------------------------------------------------


def _build_cache_key(lat: float, lon: float) -> str:
    """Build a deterministic SHA-256 cache key for (provider_id, endpoint, {lat4, lon4}).

    Credentials NOT in the key per LC9 — privacy/leakage concern; cache scope is
    per-location-per-provider, not per-tenant.

    Lat/lon rounded to 4 decimal places per ADR-017 §Cache key.
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
# LC12/LC27 envelope error dispatch (200-success-false mapping)
# ---------------------------------------------------------------------------


def _raise_for_envelope_error(message: str) -> None:
    """Map IQAir status:"fail" envelope message to the canonical taxonomy (LC12/LC27).

    This IS an intentional wrap — we're adding context the inner layer
    (ProviderHTTPClient) didn't have.  ProviderHTTPClient sees only HTTP status
    codes; the 200-with-error envelope is a wire-level IQAir protocol detail.
    Documented in commit body per non-obvious-provenance rule.

    Dispatch table (from pyairvisual cloud_api.py ERROR_CODES):
      _KEY_INVALID_MESSAGES   → KeyInvalid (permanent; operator must reconfigure)
      _QUOTA_EXHAUSTED_MESSAGES → QuotaExhausted (retry_after_seconds=None;
                                  not a 429 so no Retry-After from IQAir)
      everything else         → ProviderProtocolError

    Args:
        message: The error message string from data.message (lowercased by caller).

    Raises:
        KeyInvalid: auth/credential failure message strings.
        QuotaExhausted: rate-limit message strings (no retry_after on 200-not-429).
        ProviderProtocolError: all other status:fail message strings.
    """
    logger.error(
        "IQAir AQI returned status=fail: message=%r",
        message,
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )

    if message in _KEY_INVALID_MESSAGES:
        raise KeyInvalid(
            f"IQAir AQI auth failure: message={message!r}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )
    if message in _QUOTA_EXHAUSTED_MESSAGES:
        # 200-not-429, so no Retry-After from IQAir; retry_after_seconds=None.
        raise QuotaExhausted(
            f"IQAir AQI rate limit: message={message!r}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
            retry_after_seconds=None,
        )
    raise ProviderProtocolError(
        f"IQAir AQI returned status=fail: message={message!r}",
        provider_id=PROVIDER_ID,
        domain=DOMAIN,
    )


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------


def _wire_to_canonical(data: _IQAirData) -> AQIReading | None:
    """Translate IQAir _IQAirData to canonical AQIReading.

    Returns:
        Canonical AQIReading or None if aqius is null (no useful reading).
    """
    pollution = data.current.pollution

    # aqi: IQAir publishes US EPA AQI directly (0-500 int).
    # No conversion, no breakpoint computation — distinct from OWM 1-5 ordinal
    # and Open-Meteo sub-AQI computation paths.
    aqi_val: int | None = pollution.aqius

    # Empty-result guard: if aqius is null, no useful reading.
    # (All pollutant* fields are None on free tier regardless.)
    if aqi_val is None:
        return None

    # aqiMainPollutant: normalize mainus code to canonical id (LC2).
    # Unknown codes → None + logger.info notice (LC3; mirrors Aeris pm1 handling).
    main_pollutant: str | None = None
    if pollution.mainus:
        mainus_lower = pollution.mainus.lower()
        main_pollutant = _MAINUS_TO_CANONICAL.get(mainus_lower)
        if main_pollutant is None:
            logger.info(
                "IQAir AQI mainus code %r not in _MAINUS_TO_CANONICAL; "
                "aqiMainPollutant=None. Add to lookup table if confirmed by real-capture.",
                pollution.mainus,
                extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
            )

    # aqiLocation: f"{city}, {state}" (LC4 / Q3 user decision 2026-05-11).
    # None if either field missing — don't emit partial location strings.
    aqi_location: str | None = None
    if data.city and data.state:
        aqi_location = f"{data.city}, {data.state}"

    # pollutantPM25/PM10/O3/NO2/SO2/CO: all None on free Community tier (LC5).
    # PARTIAL-DOMAIN — categorical absence (not tier-conditional null).
    # A future round can add these after paid-tier real-capture confirms wire shape.

    # observedAt: parse pollution.ts via shared helper (LC6 / ADR-020).
    # Py 3.11+ datetime.fromisoformat accepts Z suffix ("2019-04-08T18:00:00.000Z").
    # DRY reuse of to_utc_iso8601_from_offset — no new helper needed.
    # pollution.ts is the authoritative timestamp (not weather.ts, which may
    # differ by an hour or more from a different upstream source).
    observed_at = to_utc_iso8601_from_offset(
        pollution.ts,
        provider_id=PROVIDER_ID,
        domain=DOMAIN,
    )

    return AQIReading(
        aqi=aqi_val,
        aqiScale="epa",
        aqiCategory=None,
        aqiMainPollutant=main_pollutant,
        aqiLocation=aqi_location,
        pollutantPM25=None,   # LC5 — PARTIAL-DOMAIN free Community tier
        pollutantPM10=None,   # LC5 — PARTIAL-DOMAIN free Community tier
        pollutantO3=None,     # LC5 — PARTIAL-DOMAIN free Community tier
        pollutantNO2=None,    # LC5 — PARTIAL-DOMAIN free Community tier
        pollutantSO2=None,    # LC5 — PARTIAL-DOMAIN free Community tier
        pollutantCO=None,     # LC5 — PARTIAL-DOMAIN free Community tier
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
    key: str | None,
    http_client: ProviderHTTPClient | None = None,
) -> AQIReading | None:
    """GET /v2/nearest_city?lat={lat}&lon={lon}&key={key} and return canonical AQIReading.

    Cache-first: checks the cache before making an outbound HTTP call.
    Cache stores post-normalization AQIReading as model_dump() dict (JSON-
    serializable for Redis per ADR-017); reconstructed via model_validate() on hit.

    None return: provider responded but no useful reading available (aqius null).

    L2 carry-forward (3b-4 audit F1): ProviderHTTPClient.get() raises canonical
    taxonomy exceptions (KeyInvalid, QuotaExhausted, TransientNetworkError,
    ProviderProtocolError) with all structured attributes set (status_code,
    retry_after_seconds).  These propagate bare — do NOT re-construct.

    The LC12/LC27 envelope-mapping (_raise_for_envelope_error) IS an intentional
    wrap: we're adding context (wire-level status:fail message → canonical taxonomy
    member) that ProviderHTTPClient couldn't add (it only sees HTTP status).

    Wire-shape validation wrap (ValidationError, ValueError) → ProviderProtocolError
    IS an intentional wrap: adds wire-context the inner layer didn't have.
    Per OWM/Aeris precedent. Documented in commit body per non-obvious-provenance rule.

    Args:
        lat: Station latitude (from services/station.py StationInfo).
        lon: Station longitude (from services/station.py StationInfo).
        key: IQAir API key (from settings.aqi.iqair_key via wire_aqi_settings).
        http_client: Optional ProviderHTTPClient override for testing.
            When None, the module-level singleton is used.

    Returns:
        Canonical AQIReading or None (aqius null — no useful reading at this location).

    Raises:
        KeyInvalid: key is empty/None (pre-call guard), OR status:fail with auth message,
            OR provider returned 401/403.
        QuotaExhausted: status:fail with rate-limit message, OR provider returned 429.
        TransientNetworkError: Network/DNS failure or 5xx after retries.
        ProviderProtocolError: Response JSON validation failed, OR status:fail
            with other message (city_not_found, no_nearest_station, etc.).
    """
    # LC13 fail-fast guard — empty key means no credential at all.
    # Raise KeyInvalid before hitting the network rather than letting IQAir
    # return a cryptic error response with no context about where the key came from.
    # Mirrors OWM openweathermap.py:493-499 appid guard.
    if not key:
        raise KeyInvalid(
            "IQAir key is empty or None — set WEEWX_CLEARSKIES_IQAIR_KEY env var",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    cache_key = _build_cache_key(lat, lon)
    cached = get_cache().get(cache_key)
    if cached is not None:
        logger.debug(
            "Cache hit for IQAir AQI",
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        if cached == {"_no_reading": True}:
            return None
        return AQIReading.model_validate(cached)

    logger.debug(
        "Cache miss for IQAir AQI; calling %s",
        IQAIR_BASE_URL + IQAIR_NEAREST_CITY_PATH,
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )

    # key in params dict (not URL path) — avoids logging credentials at INFO
    # level if the URL is logged (security baseline §3.4 / LC14 redaction).
    # Lat/lon rounded to 6 decimal places per OWM/Aeris precedent.
    params = {
        "lat": str(round(lat, 6)),
        "lon": str(round(lon, 6)),
        "key": key,
    }

    client = http_client or _client_for()
    _rate_limiter.acquire()

    # L2 carry-forward: client.get() raises canonical taxonomy with all
    # attributes set.  Do NOT catch and re-raise as a new canonical exception
    # (would silently drop retry_after_seconds per 3b-4 audit F1 rule).
    response = client.get(IQAIR_BASE_URL + IQAIR_NEAREST_CITY_PATH, params=params)

    # Parse raw JSON once — used for both status check and Pydantic validation.
    # Parsing once avoids redundant decoding and makes the error path clear.
    try:
        raw_json = response.json()
    except ValueError as exc:
        logger.error(
            "IQAir AQI response is not valid JSON: %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        raise ProviderProtocolError(
            f"IQAir AQI response is not valid JSON: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    # LC12/LC27 envelope check FIRST — status:"fail" means a wire-level error.
    # MUST happen before Pydantic validation because IQAir's error envelope shape
    # {"status": "fail", "data": {"message": "..."}} has data.message but NOT
    # data.current, so _IQAirData Pydantic validation would fail on the error
    # path if we validated before checking status.  Check status from raw JSON
    # and dispatch before attempting to parse data as _IQAirData.
    # This is an intentional wrap: adds context the inner layer didn't have.
    raw_status = raw_json.get("status", "") if isinstance(raw_json, dict) else ""
    if raw_status != "success":
        raw_data = raw_json.get("data", {}) if isinstance(raw_json, dict) else {}
        if isinstance(raw_data, dict):
            message = str(raw_data.get("message", "unknown")).lower()
        else:
            message = "unknown"
        _raise_for_envelope_error(message)

    # Wire-shape validation (success path only): intentional
    # (ValidationError, ValueError) → ProviderProtocolError wrap.
    # Adds wire-context the inner layer didn't have.  Per OWM/Aeris precedent.
    # Documented in commit body per non-obvious-provenance rule.
    try:
        wire = _IQAirResponse.model_validate(raw_json)
    except (ValidationError, ValueError) as exc:
        logger.error(
            "IQAir AQI response validation failed: %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        raise ProviderProtocolError(
            f"IQAir AQI response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    # data should be present on success; guard against unexpected null.
    if wire.data is None:
        logger.info(
            "IQAir AQI: status=success but data is null for lat=%s lon=%s",
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

    record = _wire_to_canonical(wire.data)

    if record is None:
        # aqius was null — cache sentinel so re-polls within TTL skip the provider.
        logger.info(
            "IQAir AQI: null aqius for lat=%s lon=%s",
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
        "IQAir AQI fetched: aqi=%s mainPollutant=%s aqiLocation=%r for lat=%s lon=%s",
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
