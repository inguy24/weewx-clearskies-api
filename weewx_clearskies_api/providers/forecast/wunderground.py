"""Weather Underground (TWC/api.weather.com) forecast provider module (ADR-007, ADR-038).

Five responsibilities per ADR-038 §2:
  1. Outbound API calls — ONE per cache miss:
       GET /v3/wx/forecast/daily/5day?geocode={lat},{lon}&format=json&units={e|m|s}
           &language=en-US&apiKey={key}
       → returns column-oriented arrays (5 elements, one per day) at top level;
         daypart[0] arrays of 10 elements (5 days × 2 dayparts: D/N).
  2. Response parsing — wire-shape Pydantic models (_WUDaypart, _WU5DayResponse)
  3. Translation to canonical ForecastBundle (daily only; hourly=[] always;
     discussion=None always — PARTIAL-DOMAIN provider)
  4. Capability declaration — CAPABILITY symbol consumed at startup
  5. Error handling — provider errors translated to canonical taxonomy

PARTIAL-DOMAIN note (L1 rule extension, 3b-6):
  Wunderground PWS API has NO hourly forecast on any plan tier (api-docs §Known
  issues; canonical §4.1.2 Wunderground column = all "—").  Bundle ships
  hourly=[] unconditionally.  Wunderground has no discussion product (canonical
  §4.1.4 column = all "—").  Bundle ships discussion=None unconditionally.
  CAPABILITY enumerates ONLY DailyForecastPoint fields; hourly + discussion
  fields are categorically excluded (not tier-conditional).

Auth: TWO credentials (brief lead-call 14):
  - apiKey: query param on every request (env var WEEWX_CLEARSKIES_WUNDERGROUND_API_KEY).
  - pws_station_id: config-time gate per ADR-007 line 79 (env var
    WEEWX_CLEARSKIES_WUNDERGROUND_PWS_STATION_ID).  The forecast URL uses
    geocode=<lat>,<lon>, NOT the PWS station ID directly.  The PWS station ID
    is required as a defense-in-depth gate: Wunderground apiKeys are issued only
    to active PWS contributors; requiring both env vars ensures the operator's
    mental model matches the gating reality.  An apiKey without an active PWS
    eventually 401s anyway.
  - ADR-007 line 79 "config time" gate is operationalized as fetch-time KeyInvalid
    (same precedent as Aeris in 3b-4 and OWM in 3b-5; no "refuse to start the
    service" behavior — loud failure at first use is the loud posture).
  - Long-form provider-scoped env var naming per 3b-4/3b-5 precedent.
    Deviates from ADR-027 §3 literal schema (no domain prefix); no ADR amendment.

Cache layer (ADR-017):
  Caches the post-normalization ForecastBundle, not raw JSON.
  Key: SHA-256 of (provider_id="wunderground", endpoint="forecast_bundle",
    {lat4, lon4, target_unit}).  PWS station ID is NOT in the cache key —
    it is a config-time gate, not a per-request input.
  TTL: 1800s (30 min per ADR-017 defaults table for forecast).
  Cache stores model_dump(mode="json"); reconstructed via model_validate().

Unit handling (ADR-019, brief lead-call 15):
  Wunderground units= query param maps target_unit directly:
    US       → units=e (English/imperial): °F, mph, in
    METRIC   → units=m (Metric SI variant): °C, km/h, mm
    METRICWX → units=s (Pure SI): °C, m/s, mm
  No post-conversion required for any target_unit — Wunderground's native
  units match canonical units for each weewx target_unit exactly.
  (Contrast with OWM where units=metric returns m/s but METRIC needs km/h.)

Daypart index alignment (brief lead-call 16, 17, 21, 23, 27):
  Top-level slot i maps to dayparts [2*i, 2*i+1] (Day, Night).
  Daily canonical fields use daypart[0][2*i] (Day period) for daytime values.
  Past-period slots may be null; canonical-nullable applies per brief.

precipType derivation (brief lead-call 17):
  Wunderground returns "rain"/"snow"/"precip"/"ice" (or null) per daypart slot.
  "precip" → "rain" (mixed/general; logged DEBUG once per encounter).
  "ice"    → "freezing-rain" (ice category per canonical §3.3; LOG DEBUG once).
  Other/null → None (log DEBUG once on first encounter of unknown string).

sunrise/sunset (brief lead-call 25):
  Top-level sunriseTimeUtc[i] / sunsetTimeUtc[i] are epoch UTC seconds.
  Convert via epoch_to_utc_iso8601() from _common/datetime_utils.py.
  If the fixture or real response only carries sunriseTimeLocal/sunsetTimeLocal,
  the Utc field will be None (both are Pydantic Optional; per canonical §3.4
  sunrise/sunset are nullable).  Future-affordance: when Utc form is absent,
  fall back to parsing Local form with fromisoformat → astimezone(UTC) → Z form.
  This is NOT implemented in v0.1 (no verified case of absent Utc form from real
  PWS-tier response; defer until a real capture proves it necessary).

Rate limiter (ADR-038 §3, brief lead-call 30):
  RateLimiter("wunderground-forecast", max_calls=5, window_seconds=1).
  "Be polite" guard.  Wunderground quota: 1500/day, 30/min.  With 30-min TTL,
  operator hits Wunderground ~48 calls/day — 30× under quota.  Per-call acquire
  before the single outbound call per cache miss (3b-3 F4 lesson).

L2 carry-forward (3b-4): NO narrow-wrap for Wunderground.
  Unlike OWM (Q1 user decision for One-Call-401 graceful-empty), Wunderground
  has no such narrow wrap.  A 401 from Wunderground means apiKey invalid OR PWS
  no longer active; recovery action is the same either way (verify PWS at
  wunderground.com/member/api-keys).  Surface as standard KeyInvalid 502.
  All canonical exceptions propagate bare from client.get().

ruff: noqa: N815  (field names match TWC camelCase wire shape)
"""

# ruff: noqa: N815

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from weewx_clearskies_api.models.responses import (
    DailyForecastPoint,
    ForecastBundle,
    utc_isoformat,
)
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

PROVIDER_ID = "wunderground"
DOMAIN = "forecast"
WUNDERGROUND_BASE_URL = "https://api.weather.com"
WUNDERGROUND_FORECAST_PATH = "/v3/wx/forecast/daily/5day"
DEFAULT_FORECAST_TTL_SECONDS = 1800   # 30 min per ADR-017

_API_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Capability declaration (ADR-038 §4, brief lead-call 18)
# L1 PARTIAL-DOMAIN extension (3b-6 NEW): CAPABILITY enumerates ONLY
# DailyForecastPoint fields Wunderground supplies.  Hourly fields are
# categorically excluded (not tier-conditional).  Discussion fields excluded.
# ---------------------------------------------------------------------------

CAPABILITY = ProviderCapability(
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    supplied_canonical_fields=(
        # DailyForecastPoint fields (canonical §4.1.3 Wunderground column)
        "validDate",
        "tempMax",
        "tempMin",
        "precipAmount",
        "precipProbabilityMax",
        "windSpeedMax",
        "sunrise",
        "sunset",
        "uvIndexMax",
        "weatherCode",
        "weatherText",
        "narrative",
        # NB: HourlyForecastPoint fields NOT supplied — Wunderground PWS API
        # has no hourly forecast on any tier (canonical §4.1.2 column = all "—";
        # api-docs §Known issues confirms PWS tier has no hourly endpoint).
        # PARTIAL-DOMAIN exclusion, not tier-conditional.
        # NB: ForecastDiscussion fields NOT supplied — Wunderground PWS API has
        # no discussion product (canonical §4.1.4 column = all "—").
        # NB: windGustMax NOT supplied — canonical §4.1.3 Wunderground column
        # = "—" for windGustMax.
    ),
    geographic_coverage="global",   # Trust Wunderground's authoritative answer (brief lead-call 29)
    auth_required=("apiKey", "pws_station_id"),
    default_poll_interval_seconds=DEFAULT_FORECAST_TTL_SECONDS,
    operator_notes=(
        "Weather Underground PWS API (Personal Weather Station contributor tier).  "
        "apiKey gated to active PWS owners — see api-docs/wunderground.md §Authentication.  "
        "Forecast: daily-only (no hourly, no discussion).  5-day forecast horizon.  "
        "apiKey OR PWS-no-longer-active returns 401 → bundle.daily=[] via standard "
        "KeyInvalid 502 propagation."
    ),
)

# ---------------------------------------------------------------------------
# precipType lookup table (brief lead-call 17)
# Canonical §3.3 enum: "rain" / "snow" / "sleet" / "freezing-rain" / "hail" / None
# ---------------------------------------------------------------------------

_WU_PRECIP_TYPE_MAP: dict[str, str | None] = {
    "rain": "rain",
    "snow": "snow",
    "precip": "rain",         # mixed/general; logged DEBUG once per encounter
    "ice": "freezing-rain",   # ice on ground; freezing-rain is broader ice category
}

# Track which values have been logged to avoid log spam
_logged_unknown_precip: set[str] = set()
_logged_mixed_precip: set[str] = set()

# ---------------------------------------------------------------------------
# Wire-shape Pydantic models (security-baseline §3.5)
# Source: docs/reference/api-docs/wunderground.md §Daily forecast L138-189
# extras="ignore" so TWC additions don't break us; missing required fields
# raise ValidationError → translated to ProviderProtocolError.
# ---------------------------------------------------------------------------


class _WUDaypart(BaseModel):
    """Wire shape of the daypart[0] object in a 5-day response.

    Each field is a 10-element list (5 days × 2 dayparts: D/N).
    Slots for past periods may be null; all fields are Optional[list].
    Top-level slot i maps to daypart indices [2*i, 2*i+1] (Day, Night).
    """

    model_config = ConfigDict(extra="ignore")

    cloudCover: list[int | None] | None = None
    dayOrNight: list[str | None] | None = None
    daypartName: list[str | None] | None = None
    iconCode: list[int | None] | None = None
    iconCodeExtend: list[int | None] | None = None
    narrative: list[str | None] | None = None
    precipChance: list[int | None] | None = None       # percent (0-100)
    precipType: list[str | None] | None = None         # "rain"/"snow"/"precip"/"ice"/null
    qpf: list[float | None] | None = None
    qpfSnow: list[float | None] | None = None
    qualifierCode: list[str | None] | None = None
    qualifierPhrase: list[str | None] | None = None
    relativeHumidity: list[int | None] | None = None
    snowRange: list[str | None] | None = None
    temperature: list[int | None] | None = None
    uvIndex: list[int | None] | None = None            # UV index (daytime slot)
    windDirection: list[int | None] | None = None
    windDirectionCardinal: list[str | None] | None = None
    windPhrase: list[str | None] | None = None
    windSpeed: list[int | None] | None = None          # mph/km/h/m/s per units= param
    wxPhraseLong: list[str | None] | None = None
    wxPhraseShort: list[str | None] | None = None      # used for weatherText


class _WU5DayResponse(BaseModel):
    """Top-level wire shape for /v3/wx/forecast/daily/5day response.

    All top-level fields are 5-element lists (one per day).
    daypart is a 1-element list containing a _WUDaypart object (the API wraps
    the daypart object in an array).
    Past-period top-level slots carry valid values; daypart slots may be null.
    """

    model_config = ConfigDict(extra="ignore")

    calendarDayTemperatureMax: list[int | None] | None = None
    calendarDayTemperatureMin: list[int | None] | None = None
    dayOfWeek: list[str | None] | None = None
    expirationTimeUtc: list[int | None] | None = None
    moonPhase: list[str | None] | None = None
    moonPhaseCode: list[str | None] | None = None
    moonPhaseDay: list[int | None] | None = None
    moonriseTimeLocal: list[str | None] | None = None
    moonsetTimeLocal: list[str | None] | None = None
    narrative: list[str | None] | None = None          # top-level daily narrative
    qpf: list[float | None] | None = None              # precip in target_unit's unit
    qpfSnow: list[float | None] | None = None
    sunriseTimeLocal: list[str | None] | None = None
    sunriseTimeUtc: list[int | None] | None = None     # epoch seconds UTC
    sunsetTimeLocal: list[str | None] | None = None
    sunsetTimeUtc: list[int | None] | None = None      # epoch seconds UTC
    temperatureMax: list[int | None] | None = None
    temperatureMin: list[int | None] | None = None
    validTimeLocal: list[str | None] | None = None     # ISO-8601 with local offset
    validTimeUtc: list[int | None] | None = None       # epoch seconds UTC
    daypart: list[Any] | None = None                   # list of 1 _WUDaypart object


# ---------------------------------------------------------------------------
# Rate limiter (ADR-038 §3, brief lead-call 30)
# 5 req/s "be polite" guard.  Per-call acquire before the outbound call.
# Wunderground quota: 1500/day, 30/min.  With 30-min TTL: ~48 calls/day.
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="wunderground-forecast",
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    max_calls=5,
    window_seconds=1,
)

# ---------------------------------------------------------------------------
# HTTP client (module-level singleton)
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
# Cache key (ADR-017 §Cache key, brief lead-call 31)
# PWS station ID NOT in cache key — it is a config-time gate, not a
# per-request input.  One cache entry per (station lat/lon, target_unit).
# ---------------------------------------------------------------------------


def _build_cache_key(lat: float, lon: float, target_unit: str) -> str:
    """Build a deterministic cache key for (provider_id, endpoint, {lat, lon, unit}).

    endpoint="forecast_bundle" mirrors the Aeris/OWM logical-key convention.
    Lat/lon rounded to 4 decimal places per ADR-017.  target_unit included so
    US / METRIC / METRICWX get separate cache entries.
    """
    payload = json.dumps(
        {
            "provider_id": PROVIDER_ID,
            "endpoint": "forecast_bundle",
            "params": {
                "lat4": round(lat, 4),
                "lon4": round(lon, 4),
                "target_unit": target_unit,
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Helpers — precipType derivation (brief lead-call 17)
# ---------------------------------------------------------------------------


def _wu_precip_type_to_canonical(value: str | None) -> str | None:
    """Map Wunderground daypart precipType value to canonical §3.3 enum.

    Mapping:
      "rain"   → "rain"
      "snow"   → "snow"
      "precip" → "rain"  (mixed/general; log DEBUG once per encounter)
      "ice"    → "freezing-rain"  (broader ice category per canonical §3.3)
      other/null → None  (log DEBUG once on first encounter of unknown string)

    Per brief lead-call 17 + canonical §3.3 enum values used literally.
    """
    if value is None:
        return None

    result = _WU_PRECIP_TYPE_MAP.get(value)

    if result is not None:
        # Log mixed/ambiguous values once per process for observability
        if value == "precip" and value not in _logged_mixed_precip:
            _logged_mixed_precip.add(value)
            logger.debug(
                "Wunderground precipType 'precip' (mixed/general precipitation) "
                "mapped to 'rain' (canonical §3.3 has no mixed-precip enum; "
                "track for future canonical amendment)",
            )
        elif value == "ice" and value not in _logged_mixed_precip:
            _logged_mixed_precip.add(value)
            logger.debug(
                "Wunderground precipType 'ice' mapped to 'freezing-rain' "
                "(ice-on-ground category; canonical §3.3 freezing-rain is the "
                "broader ice class; sleet is more specific to pellets)",
            )
        return result

    # Unknown value — log once, return None
    if value not in _logged_unknown_precip:
        _logged_unknown_precip.add(value)
        logger.debug(
            "Wunderground unknown precipType %r → None "
            "(update _WU_PRECIP_TYPE_MAP if this is a known precip type)",
            value,
        )
    return None


# ---------------------------------------------------------------------------
# Helper — validDate extraction (brief lead-call 28)
# ---------------------------------------------------------------------------


def _wu_validdate_from_local(s: str) -> str:
    """Extract YYYY-MM-DD date from Wunderground validTimeLocal string.

    Wunderground's validTimeLocal carries the station-local time with offset
    (e.g. "2026-04-30T07:00:00-0700").  The date portion is already the
    station-local date — no timezone conversion needed.

    Per canonical §3.4 (validDate = station-local YYYY-MM-DD) and brief
    lead-call 28.

    Args:
        s: validTimeLocal string, expected to contain "T" separator.

    Returns:
        YYYY-MM-DD date string.

    Raises:
        ProviderProtocolError: s lacks "T" separator (provider schema change).
    """
    if "T" not in s:
        raise ProviderProtocolError(
            f"Wunderground validTimeLocal {s!r} lacks 'T' separator — "
            "unexpected wire format; provider schema may have changed",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )
    return s.split("T")[0]


# ---------------------------------------------------------------------------
# Helper — safe list element extraction
# ---------------------------------------------------------------------------


def _safe_get(lst: list[Any] | None, idx: int) -> Any:
    """Return lst[idx] or None if lst is None, empty, or idx out of range."""
    if lst is None:
        return None
    if idx >= len(lst):
        return None
    return lst[idx]


# ---------------------------------------------------------------------------
# Period → canonical translation
# ---------------------------------------------------------------------------


def _wu_to_daily_point(
    wire: _WU5DayResponse,
    day_idx: int,
    daypart: _WUDaypart | None,
) -> DailyForecastPoint:
    """Translate one Wunderground daily slot to canonical DailyForecastPoint.

    Args:
        wire: Full _WU5DayResponse for cross-array field access.
        day_idx: 0-based day index (0 = today, ..., 4 = +4 days).
        daypart: Parsed _WUDaypart object (may be None if daypart array absent).

    Daypart indexing (brief lead-call 16):
      Top-level slot i → daypart[2*i] (Day period) / daypart[2*i+1] (Night).
      Daytime canonical fields use daypart[2*i] (Day period).
      Past-period slots may be null; canonical-nullable applies.

    Fields mapped:
      validDate:           validTimeLocal[i].split("T")[0]
      tempMax:             temperatureMax[i]
      tempMin:             temperatureMin[i]
      precipAmount:        qpf[i]  (already in target_unit's precip unit)
      precipProbabilityMax: daypart.precipChance[2*i]  (already percent 0-100)
      windSpeedMax:        daypart.windSpeed[2*i]
      sunrise:             epoch_to_utc_iso8601(sunriseTimeUtc[i])
      sunset:              epoch_to_utc_iso8601(sunsetTimeUtc[i])
      uvIndexMax:          daypart.uvIndex[2*i]
      weatherCode:         str(daypart.iconCode[2*i])
      weatherText:         daypart.wxPhraseShort[2*i]
      narrative:           narrative[i]  (top-level, NOT daypart narrative)
      windGustMax:         None (canonical §4.1.3 Wunderground column = "—")
    """
    dp_idx = 2 * day_idx   # Day period index within the 10-element daypart arrays

    # --- validDate ---
    valid_time_local = _safe_get(wire.validTimeLocal, day_idx)
    valid_date: str | None = None
    if valid_time_local is not None:
        valid_date = _wu_validdate_from_local(valid_time_local)

    # --- tempMax / tempMin ---
    temp_max: int | None = _safe_get(wire.temperatureMax, day_idx)
    temp_min: int | None = _safe_get(wire.temperatureMin, day_idx)

    # --- precipAmount (top-level qpf[i]; already in target_unit's unit) ---
    precip_amount: float | None = _safe_get(wire.qpf, day_idx)

    # --- sunrise / sunset (epoch UTC seconds → ISO-8601 Z) ---
    sunrise_epoch = _safe_get(wire.sunriseTimeUtc, day_idx)
    sunrise_utc: str | None = None
    if sunrise_epoch is not None:
        sunrise_utc = epoch_to_utc_iso8601(sunrise_epoch, provider_id=PROVIDER_ID, domain=DOMAIN)

    sunset_epoch = _safe_get(wire.sunsetTimeUtc, day_idx)
    sunset_utc: str | None = None
    if sunset_epoch is not None:
        sunset_utc = epoch_to_utc_iso8601(sunset_epoch, provider_id=PROVIDER_ID, domain=DOMAIN)

    # --- narrative (top-level narrative[i], NOT daypart.narrative) ---
    narrative: str | None = _safe_get(wire.narrative, day_idx)

    # --- Daypart-derived fields (all from Day period = 2*i index) ---
    precip_prob: int | None = None
    wind_speed_max: int | None = None
    uv_index_max: int | None = None
    weather_code: str | None = None
    weather_text: str | None = None

    if daypart is not None:
        precip_prob = _safe_get(daypart.precipChance, dp_idx)
        wind_speed_max = _safe_get(daypart.windSpeed, dp_idx)
        uv_index_max = _safe_get(daypart.uvIndex, dp_idx)

        icon_code = _safe_get(daypart.iconCode, dp_idx)
        if icon_code is not None:
            weather_code = str(icon_code)

        weather_text = _safe_get(daypart.wxPhraseShort, dp_idx)
        # precipType: DailyForecastPoint has no precipType field (HourlyForecastPoint-
        # only per canonical §3.3/§3.4); _wu_precip_type_to_canonical() helper is
        # defined for this module per brief lead-call 17 but not applied to daily points.

    return DailyForecastPoint(
        validDate=valid_date,
        tempMax=temp_max,
        tempMin=temp_min,
        precipAmount=precip_amount,
        precipProbabilityMax=precip_prob,
        windSpeedMax=wind_speed_max,
        windGustMax=None,   # canonical §4.1.3 Wunderground column = "—"; always None
        sunrise=sunrise_utc,
        sunset=sunset_utc,
        uvIndexMax=uv_index_max,
        weatherCode=weather_code,
        weatherText=weather_text,
        narrative=narrative,
        source=PROVIDER_ID,
    )


# ---------------------------------------------------------------------------
# Wire → canonical normalization
# ---------------------------------------------------------------------------


def _wu_to_canonical_bundle(
    wire: _WU5DayResponse,
) -> ForecastBundle:
    """Translate Wunderground 5-day wire response to canonical ForecastBundle.

    PARTIAL-DOMAIN: hourly=[] unconditionally (no hourly on any PWS tier).
    discussion=None unconditionally (no discussion product on any tier).
    source: PROVIDER_ID ("wunderground").
    generatedAt: current UTC timestamp.

    Daypart object is wrapped in a 1-element list in the wire response
    (api-docs example shows daypart: [{...}]); we extract daypart[0].
    """
    # Extract the single daypart object from the wrapping list
    daypart: _WUDaypart | None = None
    if wire.daypart and len(wire.daypart) > 0:
        raw_dp = wire.daypart[0]
        if isinstance(raw_dp, dict):
            try:
                daypart = _WUDaypart.model_validate(raw_dp)
            except ValidationError as exc:
                logger.warning(
                    "Wunderground daypart validation warning: %s — "
                    "daypart-derived fields will be None for all days",
                    exc,
                )
                daypart = None
        elif isinstance(raw_dp, _WUDaypart):
            daypart = raw_dp

    # Determine how many days are in the response (should be 5 for /5day)
    n_days = 0
    if wire.validTimeLocal:
        n_days = len(wire.validTimeLocal)
    elif wire.temperatureMax:
        n_days = len(wire.temperatureMax)

    daily_points = [
        _wu_to_daily_point(wire, i, daypart)
        for i in range(n_days)
    ]

    return ForecastBundle(
        hourly=[],        # PARTIAL-DOMAIN: no hourly on any PWS tier
        daily=daily_points,
        discussion=None,  # PARTIAL-DOMAIN: no discussion product
        source=PROVIDER_ID,
        generatedAt=utc_isoformat(datetime.now(tz=UTC)),
    )


# ---------------------------------------------------------------------------
# Public fetch entrypoint (ADR-038 §2)
# ---------------------------------------------------------------------------


def fetch(
    *,
    lat: float,
    lon: float,
    target_unit: str,
    api_key: str | None,
    pws_station_id: str | None,
    http_client: ProviderHTTPClient | None = None,
) -> ForecastBundle:
    """Call Wunderground /v3/wx/forecast/daily/5day and return canonical ForecastBundle.

    One outbound call per cache miss.  Cache stores the post-normalization
    ForecastBundle as model_dump(mode="json"); reconstructed via
    ForecastBundle.model_validate() on cache hit.

    PARTIAL-DOMAIN: hourly=[] always.  discussion=None always.

    Auth requirement (brief lead-call 14):
      Both api_key AND pws_station_id must be non-empty.  If either is missing,
      raises KeyInvalid immediately (loud failure beats silent disable).
      The forecast URL itself uses geocode=lat,lon — pws_station_id is NOT in
      the URL.  It is a config-time gate per ADR-007 line 79: apiKeys are
      issued only to active PWS contributors; requiring both env vars ensures
      the operator's mental model matches the gating reality.

    ADR-007 line 79 "config time" interpretation:
      ADR-007 says "config time" loud failure; 3b-4/3b-5 precedent operationalizes
      this as "loud failure at first use" (fetch-time KeyInvalid) rather than
      "refuse to start the service."  This matches Aeris (brief 3b-4 lead-call 12)
      and OWM (brief 3b-5 lead-call 13).  Documented here; no ADR amendment.

    L2 carry-forward: NO narrow-wrap for Wunderground.
      All canonical exceptions from client.get() propagate bare.
      A Wunderground 401 → KeyInvalid → 502 ProviderProblem (standard path).
      No Q1-style graceful-empty-bundle path (unlike OWM 3b-5).

    Unit handling (ADR-019, brief lead-call 15):
      US       → units=e (English/imperial)
      METRIC   → units=m (Metric SI variant; km/h native; no post-conversion)
      METRICWX → units=s (Pure SI; m/s native; no post-conversion)

    Args:
        lat: Station latitude.
        lon: Station longitude.
        target_unit: Weewx unit system ("US" | "METRIC" | "METRICWX").
        api_key: Wunderground apiKey from env WEEWX_CLEARSKIES_WUNDERGROUND_API_KEY.
        pws_station_id: Wunderground PWS station ID from env
            WEEWX_CLEARSKIES_WUNDERGROUND_PWS_STATION_ID.  Config-time gate;
            not sent in the forecast URL.
        http_client: Optional ProviderHTTPClient override for testing.

    Returns:
        ForecastBundle — hourly=[] always, discussion=None always.
        daily contains 0-5 DailyForecastPoint entries.

    Raises:
        KeyInvalid: api_key or pws_station_id is missing/empty, or Wunderground
            returned 401 (apiKey invalid or PWS no longer active).
        QuotaExhausted: Wunderground returned 429.
        ProviderProtocolError: target_unit unknown, response validation failed,
            or validTimeLocal lacks T separator.
        TransientNetworkError: Network/DNS failure or 5xx after retries.
    """
    # --- Validate credentials (both required) ---
    if not api_key or not pws_station_id:
        missing = []
        if not api_key:
            missing.append("WEEWX_CLEARSKIES_WUNDERGROUND_API_KEY")
        if not pws_station_id:
            missing.append("WEEWX_CLEARSKIES_WUNDERGROUND_PWS_STATION_ID")
        raise KeyInvalid(
            f"Wunderground credentials missing — set {' and '.join(missing)}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    if target_unit not in {"US", "METRIC", "METRICWX"}:
        # Defensive: services/units.py validates at startup; should not fire.
        raise ProviderProtocolError(
            f"Unknown target_unit {target_unit!r}; expected US, METRIC, or METRICWX",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    cache_key = _build_cache_key(lat, lon, target_unit)
    cached = get_cache().get(cache_key)
    if cached is not None:
        logger.debug(
            "Cache hit for Wunderground forecast",
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        return ForecastBundle.model_validate(cached)

    logger.debug(
        "Cache miss for Wunderground forecast; calling /v3/wx/forecast/daily/5day",
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )

    # Map target_unit → Wunderground units param (brief lead-call 15)
    # US → "e" (English/imperial), METRIC → "m" (Metric SI), METRICWX → "s" (Pure SI)
    wu_units_map = {"US": "e", "METRIC": "m", "METRICWX": "s"}
    wu_units = wu_units_map[target_unit]

    params: dict[str, str] = {
        "geocode": f"{round(lat, 6)},{round(lon, 6)}",
        "format": "json",
        "units": wu_units,
        "language": "en-US",
        "apiKey": api_key,
    }

    client = http_client or _client_for()

    _rate_limiter.acquire()

    # Bare client.get() — all canonical exceptions propagate (L2 carry-forward).
    # No narrow-wrap for Wunderground (unlike OWM Q1 basic-tier-401 path).
    # A 401 from Wunderground = apiKey invalid OR PWS no longer active;
    # surface as standard KeyInvalid 502 in both cases.
    response = client.get(WUNDERGROUND_BASE_URL + WUNDERGROUND_FORECAST_PATH, params=params)

    # Parse and validate wire shape
    try:
        wire = _WU5DayResponse.model_validate(response.json())
    except (ValidationError, ValueError) as exc:
        logger.error(
            "Wunderground 5day response validation failed: %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
        )
        raise ProviderProtocolError(
            f"Wunderground 5day response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    bundle = _wu_to_canonical_bundle(wire)

    get_cache().set(
        cache_key,
        bundle.model_dump(mode="json"),
        ttl_seconds=DEFAULT_FORECAST_TTL_SECONDS,
    )

    logger.info(
        "Wunderground forecast fetched: %d daily point(s) (hourly=[] always, PARTIAL-DOMAIN)",
        len(bundle.daily),
        extra={
            "provider_id": PROVIDER_ID,
            "domain": DOMAIN,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "target_unit": target_unit,
        },
    )
    return bundle


def _reset_http_client_for_tests() -> None:
    """Reset module-level HTTP client singleton.  Used in tests only."""
    global _http_client  # noqa: PLW0603
    _http_client = None
