"""OpenWeatherMap One Call 3.0 forecast provider module (ADR-007, ADR-038).

Five responsibilities per ADR-038 §2:
  1. Outbound API calls — ONE per cache miss:
       GET /data/3.0/onecall?lat=&lon=&appid=&units=&exclude=current,minutely,alerts
       → returns hourly[] (48 entries) + daily[] (8 entries) in one payload.
  2. Response parsing — wire-shape Pydantic models (_OWMOneCallResponse et al.)
  3. Translation to canonical ForecastBundle (HourlyForecastPoint + DailyForecastPoint)
  4. Capability declaration — CAPABILITY symbol consumed at startup
  5. Error handling — provider errors translated to canonical taxonomy

OWM is the second keyed provider on this project (ADR-006):
  Single `appid` query param on every request.
  Sourced from env var WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID at startup
  (ADR-027 §3).  Long-form provider-scoped naming per brief Q2 user decision
  2026-05-08 (matches module filename + dispatch key "openweathermap").  Same
  naming-deviation class as Aeris (3b-4); no ADR amendment.

Cache layer (ADR-017):
  Caches the post-normalization ForecastBundle, not raw JSON.
  Key: SHA-256 of (provider_id="openweathermap", endpoint="forecast_bundle",
    {lat4, lon4, target_unit}) — "forecast_bundle" logical key mirrors Aeris
    (single endpoint per cache miss; brief lead-call 31).
  TTL: 1800s (30 min per ADR-017 defaults table for forecast).
  Cache stores model_dump(mode="json"); reconstructed via model_validate().

Q1 user decision (2026-05-08) — One-Call-3.0 basic-tier 401 → graceful empty bundle:
  The module wraps the ONE outbound call in a narrow try/except KeyInvalid block.
  When client.get(/data/3.0/onecall) raises KeyInvalid AND exc.status_code == 401:
    catch, log WARN once per process, return ForecastBundle(hourly=[], daily=[],
    discussion=None, source="openweathermap", generatedAt=<now>).
  Dispatch is on attribute (exc.status_code == 401), NOT message string
  (rules/coding.md §3 — per brief lead-call 9 + 18).
  This is NOT an L2 re-construct: the exception is swallowed (not re-raised as a
  new KeyInvalid); it's a deliberate dispatch-on-attribute swallow at one specific
  call site (brief lead-call 18 + Q1 user decision).  All other call sites are
  bare client.get() calls that let canonical exceptions propagate.
  Audit risk accepted per Q1: basic-tier 401 and entirely-invalid-key 401 are
  indistinguishable from the OWM response body alone; operator-side recovery
  action is the same (verify key at OWM dashboard).

Unit handling (ADR-019, brief lead-call 15):
  OWM units param: US → "imperial", METRIC/METRICWX → "metric".
  Pressure is ALWAYS hPa; precip is ALWAYS mm (OWM ignores units= for these
  fields per openweathermap.md gotchas §"Pressure and precipitation units do not
  change with units"). Post-convert at ingest:
    US:       hPa → inHg (× 0.02953), mm → in (÷ 25.4), wind mph (no convert)
    METRIC:   hPa → mb (= hPa, no convert), mm (no convert), m/s → km/h (× 3.6)
    METRICWX: hPa → mb (= hPa, no convert), mm (no convert), m/s (no convert)
  Conversion factors sourced from openweathermap.md §"Response format conventions"
  + standard unit conversion tables.  1 hPa = 0.02953 inHg; 1 in = 25.4 mm.

Precip type (brief lead-call 17):
  Range-based lookup from OWM weather code ID.  Canonical §3.3 enum values used
  literally ("rain"/"snow"/"sleet"/"freezing-rain"/"hail"/None).
  OWM code 906 (hail, rare) → "hail" for completeness.

L1 paid-tier-max-surface rule (3b-4 carry-forward, brief lead-call 19):
  CAPABILITY enumerates the One Call 3.0 max-surface; runtime population is
  conditional on the operator's tier.  Basic-tier → empty bundle (Q1).
  Paid-tier → full hourly+daily populated.

Rate limiter (ADR-038 §3, brief lead-call 30):
  RateLimiter("openweathermap-forecast", max_calls=5, window_seconds=1).
  "Be polite" guard covering the per-second cap; well below 1000/day One Call
  quota with 30-min cache TTL (≈48 real calls/day).
  Per-call acquire before the single outbound call per cache miss (3b-3 F4
  lesson: acquire must happen inside the fetch, not outside).

ruff: noqa: N815  (field names match OWM snake_case: wind_speed, wind_deg, etc.)
"""

# ruff: noqa: N815

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from weewx_clearskies_api.models.responses import (
    DailyForecastPoint,
    ForecastBundle,
    HourlyForecastPoint,
    ProviderConditions,
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

PROVIDER_ID = "openweathermap"
DOMAIN = "forecast"
OWM_BASE_URL = "https://api.openweathermap.org"
OWM_ONECALL_PATH = "/data/3.0/onecall"
DEFAULT_FORECAST_TTL_SECONDS = 1800   # 30 min per ADR-017
DEFAULT_CONDITIONS_TTL_SECONDS = 300  # 5 min per brief

_API_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Capability declaration (ADR-038 §4, brief lead-call 19)
# L1 rule: CAPABILITY enumerates One Call 3.0 max-surface; runtime population
# is conditional on the operator's tier (basic-tier → empty bundle per Q1).
# ---------------------------------------------------------------------------

CAPABILITY = ProviderCapability(
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    supplied_canonical_fields=(
        # HourlyForecastPoint fields (canonical §4.1.2 OWM column)
        "validTime",
        "outTemp",
        "outHumidity",
        "windSpeed",
        "windDir",
        "windGust",
        "precipProbability",
        "precipAmount",
        "precipType",
        "cloudCover",
        "weatherCode",
        "weatherText",
        # DailyForecastPoint fields (canonical §4.1.3 OWM column)
        "validDate",
        "tempMax",
        "tempMin",
        "precipAmount",
        "precipProbabilityMax",
        "windSpeedMax",
        "windGustMax",
        "sunrise",
        "sunset",
        "uvIndexMax",
        "weatherCode",
        "weatherText",
        "narrative",
        # ForecastDiscussion fields NOT supplied — canonical §4.1.4 OWM column = all "—".
        # Bundle ships discussion=None unconditionally (brief lead-call 33).
    ),
    geographic_coverage="global",   # Trust OWM's authoritative answer (brief lead-call 29)
    auth_required=("appid",),
    default_poll_interval_seconds=DEFAULT_FORECAST_TTL_SECONDS,
    operator_notes=(
        "OpenWeatherMap One Call 3.0 (paid 'One Call by Call' subscription "
        "required for /data/3.0/onecall). Basic-tier appid returns empty "
        "forecast bundle — bundle.hourly=[], bundle.daily=[] (Q1 user "
        "decision 2026-05-08; module dispatches on /data/3.0/onecall 401 "
        "to graceful empty rather than KeyInvalid 502). Coverage global "
        "per ADR-007 §Per-module behavior."
    ),
)

# ---------------------------------------------------------------------------
# OWM weather code → canonical precipType (brief lead-call 17)
# Canonical §3.3 enum: "rain" / "snow" / "sleet" / "freezing-rain" / "hail" / None
# Range-based mapping; unknown codes → None (log DEBUG once on first encounter).
# ---------------------------------------------------------------------------

_OWM_CODE_TO_PRECIP_TYPE: dict[int, str] = {
    # 2xx Thunderstorm — thunder accompanies rain (consistent with NWS tsra → rain,
    # Aeris T → rain from 3b-3 and 3b-4)
    200: "rain", 201: "rain", 202: "rain", 210: "rain", 211: "rain",
    212: "rain", 221: "rain", 230: "rain", 231: "rain", 232: "rain",
    # 3xx Drizzle — drizzle is rain class in canonical §3.3
    300: "rain", 301: "rain", 302: "rain", 310: "rain", 311: "rain",
    312: "rain", 313: "rain", 314: "rain", 321: "rain",
    # 5xx Rain
    500: "rain", 501: "rain", 502: "rain", 503: "rain", 504: "rain",
    520: "rain", 521: "rain", 522: "rain", 531: "rain",
    # 511 Freezing rain — only OWM freezing variant
    511: "freezing-rain",
    # 6xx Snow
    600: "snow", 601: "snow", 602: "snow", 620: "snow", 621: "snow", 622: "snow",
    # 611 Sleet + 612/613 light/heavy sleet
    611: "sleet", 612: "sleet", 613: "sleet",
    # 615/616 rain-snow mix → "rain" (canonical has no mixed-precip enum;
    # log DEBUG once so future canonical amendment is informed; brief lead-call 17)
    615: "rain", 616: "rain",
    # 906 Hail — rare but documented; map for completeness
    906: "hail",
    # 7xx Atmosphere (fog/haze/dust/etc) → None (not in table = None)
    # 800 Clear → None
    # 8xx Clouds → None
}

# Mixed-precip codes that warrant a DEBUG log on first encounter
_MIXED_PRECIP_CODES: frozenset[int] = frozenset({615, 616})

# Track which codes have been logged to avoid log spam
_logged_unknown_codes: set[int] = set()
_logged_mixed_precip_codes: set[int] = set()

# ---------------------------------------------------------------------------
# Wire-shape Pydantic models (security-baseline §3.5)
# Source: docs/reference/api-docs/openweathermap.md + brief §module spec
# extras="ignore" so OWM additions don't break us; missing required fields
# raise ValidationError → translated to ProviderProtocolError.
# ---------------------------------------------------------------------------


class _OWMWeatherEntry(BaseModel):
    """One entry in weather[] array (both hourly and daily)."""

    model_config = ConfigDict(extra="ignore")

    id: int
    main: str | None = None
    description: str | None = None
    icon: str | None = None


class _OWMHourlyPeriod(BaseModel):
    """One hourly period from One Call 3.0 hourly[] array."""

    model_config = ConfigDict(extra="ignore")

    dt: int                           # Unix UTC seconds
    temp: float | None = None         # °F (imperial) or °C (metric)
    humidity: float | None = None     # 0-100 percent (always percent, no unit change)
    wind_speed: float | None = None   # mph (imperial) or m/s (metric)
    wind_deg: float | None = None     # degrees (always degrees)
    wind_gust: float | None = None    # mph (imperial) or m/s (metric)
    pressure: float | None = None     # ALWAYS hPa regardless of units= param (gotcha!)
    clouds: float | None = None       # 0-100 percent cloud cover
    uvi: float | None = None          # UV index
    pop: float | None = None          # 0-1 precipitation probability (NOT percent; gotcha!)
    visibility: float | None = None   # meters
    weather: list[_OWMWeatherEntry] = Field(default_factory=list)
    # Precipitation: may be absent when no precipitation (gotcha!)
    rain: dict[str, float] | None = None   # {"1h": <mm>}
    snow: dict[str, float] | None = None   # {"1h": <mm>}
    # dew_point, feels_like present but not mapped to canonical (extras={} per lead-call 32)
    dew_point: float | None = None    # ignored; kept for extras={} future


class _OWMDailyTemp(BaseModel):
    """Nested temp object on daily[] periods."""

    model_config = ConfigDict(extra="ignore")

    min: float | None = None
    max: float | None = None
    morn: float | None = None
    day: float | None = None
    eve: float | None = None
    night: float | None = None


class _OWMDailyPeriod(BaseModel):
    """One daily period from One Call 3.0 daily[] array."""

    model_config = ConfigDict(extra="ignore")

    dt: int                           # Unix UTC seconds (used for validDate derivation)
    sunrise: int | None = None        # Unix UTC seconds
    sunset: int | None = None         # Unix UTC seconds
    temp: _OWMDailyTemp | None = None
    humidity: float | None = None     # 0-100 percent
    wind_speed: float | None = None   # mph (imperial) or m/s (metric)
    wind_deg: float | None = None     # degrees
    wind_gust: float | None = None    # mph (imperial) or m/s (metric)
    pressure: float | None = None     # ALWAYS hPa (gotcha!)
    clouds: float | None = None       # 0-100 percent
    uvi: float | None = None          # UV index
    pop: float | None = None          # 0-1 precipitation probability (NOT percent; gotcha!)
    summary: str | None = None        # human-readable summary (used for weatherText + narrative)
    weather: list[_OWMWeatherEntry] = Field(default_factory=list)
    # Daily rain/snow are scalar mm totals (NOT a {1h} sub-object — gotcha vs hourly!)
    rain: float | None = None
    snow: float | None = None
    # moon-related fields: moonrise/moonset/moon_phase present but not in canonical (extras={})
    moonrise: int | None = None
    moonset: int | None = None
    moon_phase: float | None = None


class _OWMCurrentBlock(BaseModel):
    """Current conditions block from One Call 3.0 current object.

    OWM returns a single current object (not an array) when current is not
    excluded.  clouds is the raw percent integer; cast to float at translation.
    weather[] reuses the existing _OWMWeatherEntry model.
    """

    model_config = ConfigDict(extra="ignore")

    temp: float | None = None
    feels_like: float | None = None
    humidity: float | None = None
    wind_speed: float | None = None
    wind_deg: float | None = None
    wind_gust: float | None = None
    weather: list[_OWMWeatherEntry] = Field(default_factory=list)
    clouds: float | None = None       # 0-100 percent cloud cover
    visibility: float | None = None
    uvi: float | None = None


class _OWMOneCallResponse(BaseModel):
    """Top-level One Call 3.0 response — wire shape."""

    model_config = ConfigDict(extra="ignore")

    lat: float
    lon: float
    timezone: str | None = None
    timezone_offset: int = 0          # UTC offset in seconds for station-local date derivation
    current: _OWMCurrentBlock | None = None  # present when not excluded
    hourly: list[_OWMHourlyPeriod] = Field(default_factory=list)
    daily: list[_OWMDailyPeriod] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Rate limiter (ADR-038 §3, brief lead-call 30)
# 5 req/s "be polite" guard — per-call acquire before the single outbound
# call per cache miss. Well within 1000/day One Call quota with 30-min TTL.
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="openweathermap-forecast",
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
# Cache key construction (ADR-017 §Cache key, brief lead-call 31)
# ---------------------------------------------------------------------------


def _build_cache_key(lat: float, lon: float, target_unit: str) -> str:
    """Build a deterministic cache key for (provider_id, endpoint, {lat, lon, unit}).

    endpoint="forecast_bundle" mirrors Aeris's logical-key convention (one endpoint
    per cache miss; brief lead-call 31).  Lat/lon rounded to 4 decimal places per
    ADR-017.  target_unit included so US and METRIC/METRICWX get separate cache
    entries (unit conversions happen at ingest time).
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
# Helpers — validDate derivation (brief lead-call 25)
# ---------------------------------------------------------------------------


def _owm_validdate(epoch_utc: int, tz_offset_seconds: int) -> str:
    """Derive station-local YYYY-MM-DD from OWM epoch UTC + timezone_offset.

    OWM provides timezone_offset (seconds) at the response root.  Adding the
    offset to the epoch before constructing the datetime shifts the wall-clock
    to station-local time; we then format the date part only.  The station's
    actual IANA tz isn't needed — OWM gives us the offset directly.

    Per canonical §3.4 (validDate = station-local YYYY-MM-DD) and brief
    lead-call 25.
    """
    shifted = datetime.fromtimestamp(epoch_utc + tz_offset_seconds, tz=UTC)
    return shifted.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Helpers — precipitation amount extraction (brief lead-calls 23, 24)
# ---------------------------------------------------------------------------


def _owm_hourly_precip_mm(period: _OWMHourlyPeriod) -> float:
    """Extract total hourly precip in mm from rain.1h + snow.1h.

    Per openweathermap.md gotchas §"Rain/snow keys may be absent":
    treat absence of rain or snow keys as 0 mm.  Also defensively handles
    the case where rain/snow are present but None (API inconsistency guard).
    """
    rain_mm = 0.0
    if period.rain is not None:
        rain_mm = period.rain.get("1h", 0) or 0

    snow_mm = 0.0
    if period.snow is not None:
        snow_mm = period.snow.get("1h", 0) or 0

    return rain_mm + snow_mm


def _owm_daily_precip_mm(day: _OWMDailyPeriod) -> float:
    """Extract total daily precip in mm from daily.rain + daily.snow.

    On daily[] these are scalar mm totals (NOT a {1h} sub-object).  Per
    api-docs example L194-201.  Treat absence as 0.
    """
    rain_mm = day.rain or 0.0
    snow_mm = day.snow or 0.0
    return rain_mm + snow_mm


# ---------------------------------------------------------------------------
# Helpers — unit conversions (ADR-019, brief lead-call 15)
# ---------------------------------------------------------------------------


def _convert_owm_units(
    value: float | None,
    *,
    field_kind: str,
    target_unit: str,
) -> float | None:
    """Apply target_unit post-conversion for OWM field kinds.

    OWM pressure and precipitation do NOT change with the units= query param
    (documented gotcha in openweathermap.md §"Pressure and precipitation units
    do not change with units").  This helper normalises them:

    field_kind values:
      "pressure"    : always hPa → inHg (US: × 0.02953) or mb (METRIC/METRICWX: pass-through)
      "precip_mm"   : always mm → in (US: ÷ 25.4) or mm (METRIC/METRICWX: pass-through)
      "wind_speed"  : mph (imperial) or m/s (metric) → no convert for US; METRIC: × 3.6 for km/h;
                      METRICWX: m/s (no convert, OWM metric is already m/s)
      "wind_gust"   : same conversions as wind_speed

    Conversion factors:
      1 hPa = 0.02953 inHg  (source: openweathermap.md gotchas + standard tables)
      1 in  = 25.4 mm       (source: standard definition; exact)
      1 m/s = 3.6 km/h      (source: standard definition; exact)

    Args:
        value: The raw value to convert, or None.
        field_kind: One of "pressure", "precip_mm", "wind_speed", "wind_gust".
        target_unit: Weewx unit system ("US" | "METRIC" | "METRICWX").

    Returns:
        Converted float value or None if value is None.
    """
    if value is None:
        return None

    if field_kind in ("wind_speed", "wind_gust"):
        # US: OWM imperial returns mph → no conversion needed
        # METRIC: OWM metric returns m/s → multiply by 3.6 for km/h
        # METRICWX: OWM metric returns m/s → already correct
        if target_unit == "US":
            return value        # mph, no convert
        elif target_unit == "METRIC":
            return value * 3.6  # m/s → km/h
        else:  # METRICWX
            return value        # m/s, no convert

    elif field_kind == "pressure":
        # OWM pressure is ALWAYS hPa regardless of units=
        # US: convert hPa → inHg (canonical unit for barometer in US system)
        # METRIC / METRICWX: hPa = mb, pass through
        if target_unit == "US":
            return value * 0.02953   # hPa → inHg; 1 hPa = 0.02953 inHg exactly
        else:
            return value             # hPa = mb, no convert

    elif field_kind == "precip_mm":
        # OWM precipitation is ALWAYS mm regardless of units=
        # US: convert mm → in
        # METRIC / METRICWX: mm, pass through
        if target_unit == "US":
            return value / 25.4     # mm → in; 1 in = 25.4 mm exactly
        else:
            return value            # mm, no convert

    # Unknown field_kind — defensive path; should not happen with typed callers
    logger.warning(
        "OWM unit conversion: unknown field_kind %r (value=%r, target_unit=%r)",
        field_kind, value, target_unit,
    )
    return value


# ---------------------------------------------------------------------------
# Helpers — precipType derivation (brief lead-call 17)
# ---------------------------------------------------------------------------


def _owm_weather_code_to_precip_type(code: int) -> str | None:
    """Derive canonical precipType from OWM weather code ID.

    Range-based lookup using _OWM_CODE_TO_PRECIP_TYPE table.  Unknown codes
    → None (log DEBUG once on first encounter so future table updates are
    informed by real-data prevalence).  Mixed-precip codes (615, 616) are
    mapped to "rain" and logged DEBUG once per process.

    Per brief lead-call 17 + canonical §3.3 enum values.
    """
    result = _OWM_CODE_TO_PRECIP_TYPE.get(code)

    if result is not None:
        # Log mixed-precip codes once (615/616 → "rain"; canonical has no mixed enum)
        if code in _MIXED_PRECIP_CODES and code not in _logged_mixed_precip_codes:
            _logged_mixed_precip_codes.add(code)
            logger.debug(
                "OWM weather code %d (rain-snow mix) mapped to 'rain' "
                "(canonical §3.3 has no mixed-precip enum; track for future amendment)",
                code,
            )
        return result

    # Unknown code — log once, return None
    if code not in _logged_unknown_codes:
        _logged_unknown_codes.add(code)
        logger.debug(
            "OWM unknown weather code %d → precipType=None "
            "(update _OWM_CODE_TO_PRECIP_TYPE if this is a known precip code)",
            code,
        )
    return None


# ---------------------------------------------------------------------------
# Helpers — weatherText extraction (brief lead-call 20)
# ---------------------------------------------------------------------------


def _safe_weather_text_daily(period: _OWMDailyPeriod) -> str | None:
    """Extract daily weatherText: prefer summary, fall back to weather[0].description.

    Canonical §4.1.3 OWM column: daily[].summary (preferred) or
    weather[0].description.  Operationalization per brief lead-call 20:
    prefer summary; if summary is None or whitespace-only, use description.
    """
    if period.summary and period.summary.strip():
        return period.summary.strip()
    if period.weather:
        return period.weather[0].description or None
    return None


# ---------------------------------------------------------------------------
# Period → canonical translation helpers
# ---------------------------------------------------------------------------


def _owm_to_hourly_point(
    period: _OWMHourlyPeriod,
    *,
    target_unit: str,
) -> HourlyForecastPoint:
    """Translate one OWM hourly period to canonical HourlyForecastPoint.

    Unit handling (ADR-019, brief lead-call 15):
      US       → temp °F (imperial), wind mph (no convert), pressure hPa→inHg, precip mm→in
      METRIC   → temp °C (metric), wind m/s→km/h, pressure hPa=mb (no convert), precip mm
      METRICWX → temp °C (metric), wind m/s (no convert), pressure hPa=mb, precip mm

    validTime: epoch_to_utc_iso8601(period.dt) → UTC ISO-8601 Z (ADR-020, brief call 27).
    precipProbability: period.pop × 100 (OWM pop is 0-1; canonical is 0-100 percent;
      openweathermap.md gotchas §"pop is 0–1, not percent"; brief lead-call 22).
    weatherCode: str(weather[0].id) — opaque pass-through per brief lead-call 16.
    weatherText: weather[0].description per canonical §4.1.2 OWM column.
    extras: {} per brief lead-call 32 (dew_point, feels_like, visibility not in canonical).
    """
    valid_time = epoch_to_utc_iso8601(
        period.dt, provider_id=PROVIDER_ID, domain=DOMAIN
    )

    weather_code: str | None = None
    weather_text: str | None = None
    precip_type: str | None = None
    if period.weather:
        entry = period.weather[0]
        weather_code = str(entry.id)
        weather_text = entry.description or None
        precip_type = _owm_weather_code_to_precip_type(entry.id)

    # Precipitation amount: raw mm → convert per target_unit
    raw_precip_mm = _owm_hourly_precip_mm(period)
    precip_amount = _convert_owm_units(raw_precip_mm, field_kind="precip_mm", target_unit=target_unit)

    # Wind speed/gust: OWM returns mph (imperial) or m/s (metric)
    wind_speed = _convert_owm_units(period.wind_speed, field_kind="wind_speed", target_unit=target_unit)
    wind_gust = _convert_owm_units(period.wind_gust, field_kind="wind_gust", target_unit=target_unit)

    # pop: multiply by 100 → canonical precipProbability (0-100 percent)
    precip_prob: float | None = None
    if period.pop is not None:
        precip_prob = period.pop * 100.0

    return HourlyForecastPoint(
        validTime=valid_time,
        outTemp=period.temp,
        outHumidity=period.humidity,
        windSpeed=wind_speed,
        windDir=period.wind_deg,
        windGust=wind_gust,
        precipProbability=precip_prob,
        precipAmount=precip_amount,
        precipType=precip_type,
        cloudCover=period.clouds,
        weatherCode=weather_code,
        weatherText=weather_text,
        source=PROVIDER_ID,
    )


def _owm_to_daily_point(
    day: _OWMDailyPeriod,
    *,
    target_unit: str,
    tz_offset_seconds: int,
) -> DailyForecastPoint:
    """Translate one OWM daily period to canonical DailyForecastPoint.

    Unit handling (ADR-019, brief lead-call 15):
      US       → temp °F, wind mph→no convert, pressure hPa→inHg, precip mm→in
      METRIC   → temp °C, wind m/s→km/h, pressure hPa=mb, precip mm
      METRICWX → temp °C, wind m/s, pressure hPa=mb, precip mm

    validDate: station-local YYYY-MM-DD via _owm_validdate(dt + tz_offset) (lead-call 25).
    sunrise/sunset: epoch_to_utc_iso8601 → UTC ISO-8601 Z (lead-calls 25, 26).
    precipProbabilityMax: pop × 100 (same 0-1 → 0-100 rule; brief lead-call 22).
    narrative: daily[].summary per canonical §4.1.3 (lead-call 21).
    weatherText: _safe_weather_text_daily (summary preferred, fallback description).
    weatherCode: str(weather[0].id) opaque pass-through (lead-call 16).
    extras: {} per brief lead-call 32 (moon fields, feels_like, etc. not in canonical).
    """
    valid_date = _owm_validdate(day.dt, tz_offset_seconds)

    # Temp max / min
    temp_max: float | None = None
    temp_min: float | None = None
    if day.temp is not None:
        temp_max = day.temp.max
        temp_min = day.temp.min

    # Precipitation amount: raw mm → convert per target_unit
    raw_precip_mm = _owm_daily_precip_mm(day)
    precip_amount = _convert_owm_units(raw_precip_mm, field_kind="precip_mm", target_unit=target_unit)

    # Wind speed/gust: OWM returns mph (imperial) or m/s (metric)
    wind_speed_max = _convert_owm_units(day.wind_speed, field_kind="wind_speed", target_unit=target_unit)
    wind_gust_max = _convert_owm_units(day.wind_gust, field_kind="wind_gust", target_unit=target_unit)

    # pop: multiply by 100 → canonical precipProbabilityMax (0-100 percent)
    precip_prob_max: float | None = None
    if day.pop is not None:
        precip_prob_max = day.pop * 100.0

    # Sunrise/sunset: epoch UTC → UTC ISO-8601 Z
    sunrise_utc: str | None = None
    if day.sunrise is not None:
        sunrise_utc = epoch_to_utc_iso8601(
            day.sunrise, provider_id=PROVIDER_ID, domain=DOMAIN
        )
    sunset_utc: str | None = None
    if day.sunset is not None:
        sunset_utc = epoch_to_utc_iso8601(
            day.sunset, provider_id=PROVIDER_ID, domain=DOMAIN
        )

    # Weather code + text + precipType
    weather_code: str | None = None
    weather_text: str | None = _safe_weather_text_daily(day)
    precip_type: str | None = None
    if day.weather:
        entry = day.weather[0]
        weather_code = str(entry.id)
        precip_type = _owm_weather_code_to_precip_type(entry.id)

    # narrative: daily[].summary per canonical §4.1.3 (lead-call 21).
    # Strip leading/trailing whitespace to match _safe_weather_text_daily's
    # treatment of weatherText — both fields must agree when summary is supplied
    # (brief lead-call 21).  3b-5 audit F3 remediation 2026-05-09.
    narrative: str | None = day.summary.strip() if day.summary and day.summary.strip() else None

    return DailyForecastPoint(
        validDate=valid_date,
        tempMax=temp_max,
        tempMin=temp_min,
        precipAmount=precip_amount,
        precipProbabilityMax=precip_prob_max,
        windSpeedMax=wind_speed_max,
        windGustMax=wind_gust_max,
        sunrise=sunrise_utc,
        sunset=sunset_utc,
        uvIndexMax=day.uvi,
        weatherCode=weather_code,
        weatherText=weather_text,
        narrative=narrative,
        source=PROVIDER_ID,
    )


# ---------------------------------------------------------------------------
# Wire → canonical normalization
# ---------------------------------------------------------------------------


def _owm_to_canonical_bundle(
    wire: _OWMOneCallResponse,
    *,
    target_unit: str,
) -> ForecastBundle:
    """Translate OWM One Call 3.0 wire response to canonical ForecastBundle.

    hourly: translated from wire.hourly (up to 48 entries).
    daily: translated from wire.daily (up to 8 entries).
    discussion: None unconditionally — OWM One Call 3.0 has no forecast-discussion
      product (canonical §4.1.4 OWM column = all "—"; lead-call 33).
    source: PROVIDER_ID ("openweathermap").
    generatedAt: current UTC timestamp.
    """
    tz_offset = wire.timezone_offset

    hourly_points = [
        _owm_to_hourly_point(p, target_unit=target_unit)
        for p in wire.hourly
    ]

    daily_points = [
        _owm_to_daily_point(d, target_unit=target_unit, tz_offset_seconds=tz_offset)
        for d in wire.daily
    ]

    return ForecastBundle(
        hourly=hourly_points,
        daily=daily_points,
        discussion=None,   # OWM has no forecast discussion product (lead-call 33)
        source=PROVIDER_ID,
        generatedAt=utc_isoformat(datetime.now(tz=UTC)),
    )


# ---------------------------------------------------------------------------
# Module-level state for log-once-per-process basic-tier warning
# Must be declared BEFORE fetch() which references it.
# ---------------------------------------------------------------------------

_owm_basic_tier_warned: bool = False


def _owm_basic_tier_warned_set() -> None:
    """Mark that the basic-tier warning has been emitted (module-level state)."""
    global _owm_basic_tier_warned  # noqa: PLW0603
    _owm_basic_tier_warned = True


# ---------------------------------------------------------------------------
# Public fetch entrypoint (ADR-038 §2)
# ---------------------------------------------------------------------------


def fetch(
    *,
    lat: float,
    lon: float,
    target_unit: str,
    appid: str | None,
    http_client: ProviderHTTPClient | None = None,
) -> ForecastBundle:
    """Call OWM /data/3.0/onecall and return canonical ForecastBundle.

    One outbound call per cache miss.  Cache stores the post-normalization
    ForecastBundle as model_dump(mode="json"); reconstructed via
    ForecastBundle.model_validate() on cache hit.

    Q1 user decision (2026-05-08) — narrow try/except KeyInvalid:
      This function wraps the One Call outbound call in a narrow
      try/except KeyInvalid block.  When the call raises KeyInvalid AND
      exc.status_code == 401 (basic-tier key hitting /data/3.0/onecall),
      the exception is intentionally swallowed and an empty ForecastBundle
      is returned.  This is NOT an L2 re-construct (we do not raise a new
      KeyInvalid); it's a deliberate dispatch-on-attribute swallow at one
      specific call site.  All other canonical exceptions propagate bare.
      Dispatch is on attribute (exc.status_code), NOT message string
      (rules/coding.md §3 — per lead-call 9).
      See brief lead-call 18 + Q1 for full audit trail.

    Args:
        lat: Station latitude from services/station.py StationInfo.
        lon: Station longitude from services/station.py StationInfo.
        target_unit: Weewx unit system ("US" | "METRIC" | "METRICWX") from
            services/units.py get_target_unit().
        appid: OWM API key from env var WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID.
            None if operator hasn't configured it.
        http_client: Optional ProviderHTTPClient override for testing.
            When None, the module-level singleton is used.

    Returns:
        ForecastBundle — single canonical Pydantic model.
        discussion is always None (OWM has no forecast discussion product).
        hourly=[], daily=[] when basic-tier key returns 401 (Q1 user decision).

    Raises:
        KeyInvalid: appid is None/empty, or OWM returned 401 with status_code != 401
            (defensive; should not happen with valid key lacking One Call sub).
        QuotaExhausted: OWM returned 429 (rate limit exceeded).
        ProviderProtocolError: target_unit unknown, or response validation failed.
        TransientNetworkError: Network/DNS failure or 5xx after retries.
    """
    # Validate credentials before making any outbound call.
    # Loud failure beats silent disable — operator intent is unambiguous.
    if not appid:
        raise KeyInvalid(
            "OpenWeatherMap appid missing — set WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID",
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
            "Cache hit for OWM forecast",
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        return ForecastBundle.model_validate(cached)

    logger.debug(
        "Cache miss for OWM forecast; calling /data/3.0/onecall",
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )

    # Map target_unit → OWM units param
    # US → "imperial", METRIC/METRICWX → "metric"
    owm_units = "imperial" if target_unit == "US" else "metric"

    client = http_client or _client_for()

    params: dict[str, str] = {
        "lat": str(round(lat, 6)),
        "lon": str(round(lon, 6)),
        "appid": appid,
        "units": owm_units,
        "exclude": "minutely,alerts",
    }

    _rate_limiter.acquire()

    # Q1 user decision (2026-05-08): narrow try/except KeyInvalid for the
    # One-Call-401 graceful-empty-bundle path.  Dispatch on attribute
    # (exc.status_code), NOT message string (rules/coding.md §3, lead-call 9).
    # This is intentional: basic-tier key hitting /data/3.0/onecall returns 401;
    # we catch it and return empty bundle rather than propagating as 502.
    # See module docstring + brief Q1 for full audit trail.
    # ALL OTHER canonical exceptions propagate bare (L2 carry-forward rule).
    try:
        response = client.get(OWM_BASE_URL + OWM_ONECALL_PATH, params=params)
    except KeyInvalid as exc:
        if exc.status_code == 401:
            # Basic-tier key lacks One Call 3.0 subscription (Q1 user decision).
            # Log WARN once per process; return empty bundle.
            # Using a module-level set to track whether we've warned already.
            if not _owm_basic_tier_warned:
                _owm_basic_tier_warned_set()
                logger.warning(
                    "OpenWeatherMap appid lacks One Call 3.0 subscription — "
                    "returning empty forecast bundle. "
                    "Upgrade to 'One Call by Call' at openweathermap.org/price. "
                    "(Q1 user decision 2026-05-08; brief lead-call 18)",
                    extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
                )
            empty_bundle = ForecastBundle(
                hourly=[],
                daily=[],
                discussion=None,
                source=PROVIDER_ID,
                generatedAt=utc_isoformat(datetime.now(tz=UTC)),
            )
            # Cache the empty bundle for the same TTL as the success path
            # (ADR-017).  Without this cache.set, basic-tier-misconfigured
            # deployments hit /data/3.0/onecall 401 on every dashboard poll —
            # capped only by the rate limiter (5 req/s = 432K/day) vs the
            # success path's 48 calls/day.  Brief Q1 covered the taxonomy
            # decision (graceful empty bundle vs 502); cache parity with the
            # success path is the operator-friendly operationalization.
            # 3b-5 audit F2 remediation 2026-05-09.
            get_cache().set(
                cache_key,
                empty_bundle.model_dump(mode="json"),
                ttl_seconds=DEFAULT_FORECAST_TTL_SECONDS,
            )
            return empty_bundle
        # status_code != 401 — defensive: re-raise as KeyInvalid (let canonical
        # taxonomy handle; 502 ProviderProblem KeyInvalid).
        raise

    # Parse and validate wire shape
    try:
        wire = _OWMOneCallResponse.model_validate(response.json())
    except (ValidationError, ValueError) as exc:
        logger.error(
            "OWM onecall response validation failed: %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
        )
        raise ProviderProtocolError(
            f"OWM onecall response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    bundle = _owm_to_canonical_bundle(wire, target_unit=target_unit)

    get_cache().set(
        cache_key,
        bundle.model_dump(mode="json"),
        ttl_seconds=DEFAULT_FORECAST_TTL_SECONDS,
    )

    logger.info(
        "OWM forecast fetched: %d hourly, %d daily point(s)",
        len(bundle.hourly),
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


def fetch_current_conditions(
    *,
    lat: float,
    lon: float,
    target_unit: str,
    appid: str | None,
    http_client: ProviderHTTPClient | None = None,
) -> ProviderConditions | None:
    """Extract current conditions from the OWM One Call 3.0 response.

    Reads from the same cache as fetch() because fetch() now includes the
    current block in the One Call response (exclude no longer omits "current").
    On a cache hit, the cached bundle contains current data embedded; this
    function fetches the raw wire response separately with its own 300 s TTL
    cache key so conditions can be fresher than the 1800 s forecast bundle.

    weatherText  = current.weather[0].description (if weather list non-empty).
    weatherCode  = str(current.weather[0].id).
    cloudCover   = current.clouds (0-100 percent).
    Unit conversions mirror fetch() hourly-period logic (_convert_owm_units).

    The same Q1 basic-tier-401 graceful path applies: a 401 from the One Call
    endpoint returns None rather than raising KeyInvalid, matching fetch()'s
    behavior.

    Args:
        lat: Station latitude.
        lon: Station longitude.
        target_unit: Weewx unit system ("US" | "METRIC" | "METRICWX").
        appid: OWM API key from env var WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID.
        http_client: Optional ProviderHTTPClient override for testing.

    Returns:
        ProviderConditions on success; None on basic-tier 401 or absent current block.

    Raises:
        KeyInvalid: appid is None/empty.
        QuotaExhausted: OWM returned 429.
        ProviderProtocolError: target_unit unknown or response validation failed.
        TransientNetworkError: Network/DNS failure or 5xx after retries.
    """
    if not appid:
        raise KeyInvalid(
            "OpenWeatherMap appid missing — set WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    if target_unit not in {"US", "METRIC", "METRICWX"}:
        raise ProviderProtocolError(
            f"Unknown target_unit {target_unit!r}; expected US, METRIC, or METRICWX",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    # Separate conditions cache key so TTL is independent of the forecast bundle.
    conditions_cache_key = hashlib.sha256(
        json.dumps(
            {
                "provider_id": PROVIDER_ID,
                "endpoint": "current_conditions",
                "params": {
                    "lat4": round(lat, 4),
                    "lon4": round(lon, 4),
                    "target_unit": target_unit,
                },
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    cached = get_cache().get(conditions_cache_key)
    if cached is not None:
        logger.debug(
            "Cache hit for OWM current conditions",
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        return ProviderConditions.model_validate(cached)

    logger.debug(
        "Cache miss for OWM current conditions; calling /data/3.0/onecall",
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )

    owm_units = "imperial" if target_unit == "US" else "metric"
    client = http_client or _client_for()

    params: dict[str, str] = {
        "lat": str(round(lat, 6)),
        "lon": str(round(lon, 6)),
        "appid": appid,
        "units": owm_units,
        "exclude": "minutely,alerts",
    }

    _rate_limiter.acquire()

    # Same Q1 narrow try/except as fetch(): basic-tier 401 returns None.
    try:
        response = client.get(OWM_BASE_URL + OWM_ONECALL_PATH, params=params)
    except KeyInvalid as exc:
        if exc.status_code == 401:
            logger.warning(
                "OpenWeatherMap appid lacks One Call 3.0 subscription — "
                "returning None for current conditions (Q1 user decision 2026-05-08)",
                extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
            )
            return None
        raise

    try:
        wire = _OWMOneCallResponse.model_validate(response.json())
    except (ValidationError, ValueError) as exc:
        logger.error(
            "OWM current conditions response validation failed: %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
        )
        raise ProviderProtocolError(
            f"OWM current conditions response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    if wire.current is None:
        logger.warning(
            "OWM response missing current block; returning None",
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        return None

    cur = wire.current

    weather_code: str | None = None
    weather_text: str | None = None
    precip_type: str | None = None
    if cur.weather:
        entry = cur.weather[0]
        weather_code = str(entry.id)
        weather_text = entry.description or None
        precip_type = _owm_weather_code_to_precip_type(entry.id)

    wind_speed = _convert_owm_units(cur.wind_speed, field_kind="wind_speed", target_unit=target_unit)
    wind_gust = _convert_owm_units(cur.wind_gust, field_kind="wind_gust", target_unit=target_unit)

    conditions = ProviderConditions(
        weatherText=weather_text,
        weatherCode=weather_code,
        precipType=precip_type,
        cloudCover=cur.clouds,
        isDay=None,   # OWM current block has no is_day field
        temperature=cur.temp,
        humidity=cur.humidity,
        windSpeed=wind_speed,
        windDir=cur.wind_deg,
        source=PROVIDER_ID,
    )

    get_cache().set(
        conditions_cache_key,
        conditions.model_dump(mode="json"),
        ttl_seconds=DEFAULT_CONDITIONS_TTL_SECONDS,
    )

    logger.info(
        "OWM current conditions fetched",
        extra={
            "provider_id": PROVIDER_ID,
            "domain": DOMAIN,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "target_unit": target_unit,
        },
    )
    return conditions


def _reset_http_client_for_tests() -> None:
    """Reset module-level HTTP client singleton.  Used in tests only."""
    global _http_client  # noqa: PLW0603
    _http_client = None


def _reset_basic_tier_warned_for_tests() -> None:
    """Reset module-level basic-tier warning flag.  Used in tests only."""
    global _owm_basic_tier_warned  # noqa: PLW0603
    _owm_basic_tier_warned = False
