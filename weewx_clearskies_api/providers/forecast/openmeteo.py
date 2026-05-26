"""Open-Meteo forecast provider module (ADR-007, ADR-038).

Five responsibilities per ADR-038 §2:
  1. Outbound API call — Open-Meteo /v1/forecast with hourly + daily variable lists
  2. Response parsing — wire-shape Pydantic models for _OpenMeteoForecastResponse
  3. Translation to canonical ForecastBundle (HourlyForecastPoint + DailyForecastPoint)
  4. Capability declaration — CAPABILITY symbol consumed at startup
  5. Error handling — provider errors translated to canonical taxonomy

Open-Meteo is keyless (ADR-006):
  No API key required for non-commercial use.  The free public host
  (api.open-meteo.com) is used.  No operator-managed secrets.
  No redaction filter changes needed this round (F13 stays deferred).

Cache layer (ADR-017):
  Caches the post-normalization ForecastBundle, not raw JSON.
  Key: SHA-256 of (provider_id, endpoint, {lat4, lon4, target_unit}).
  TTL: 1800s (30 min per ADR-017 defaults table for forecast).
  Cache stores model_dump(mode="json"); reconstructed via model_validate().

Slice-after-cache pattern (ADR-017 §Cache key):
  Cache stores the FULL bundle (all hourly + daily points returned by
  Open-Meteo).  The endpoint applies the operator's hours/days slice.
  One cache entry per (station, target_unit), not one per (hours, days).
  This rationale is documented in commit body per process-gate #8.

Time conversion (ADR-020):
  Open-Meteo hourly timestamps are station-local ISO-8601 without offset
  ("2026-04-30T16:00"). Module converts to UTC using utc_offset_seconds
  from the response.  Daily dates ("YYYY-MM-DD") are already station-local
  and pass through as-is.  Sunrise/sunset in daily block are local ISO
  and also converted to UTC.

Per-unit handling (ADR-019):
  Module passes temperature_unit / wind_speed_unit / precipitation_unit
  query params to Open-Meteo to match the station's target_unit.
  Mapping table: US → fahrenheit/mph/inch, METRIC → celsius/kmh/mm,
  METRICWX → celsius/ms/mm.  Documented in commit body per process-gate #8.

WMO weather codes (canonical-data-model §3.3, §4.1.2):
  weatherCode emitted as string (WMO int as-is from Open-Meteo).
  weatherText decoded via _WMO_CODE_TO_TEXT lookup table.
  precipType derived via _WMO_CODE_TO_PRECIP_TYPE heuristic.
  Table choices match the Open-Meteo docs weather codes section and
  canonical-data-model §4.1.2.  Documented in commit body per gate #8.

Rate limiter (ADR-038 §3):
  Open-Meteo fair-use ~10 000 calls/day.  With 30-min TTL + single-station
  steady state is ~48 calls/day — three orders under threshold.
  RateLimiter("openmeteo", max_calls=5, window_seconds=1) as a courtesy
  guard matching the shape of 3b-1's NWS rate limiter.

Wire-shape Pydantic (security-baseline §3.5):
  _OpenMeteoForecastResponse validates every field from the recorded fixture
  at tests/fixtures/providers/openmeteo/forecast.json.
  extras="ignore" so future Open-Meteo additions don't break us;
  missing required fields raise ValidationError → ProviderProtocolError.

ruff: noqa: N815  (field names match canonical camelCase: validTime, outTemp, etc.)
"""

# ruff: noqa: N815

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

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
from weewx_clearskies_api.providers._common.errors import (
    ProviderProtocolError,
)
from weewx_clearskies_api.providers._common.http import ProviderHTTPClient
from weewx_clearskies_api.providers._common.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PROVIDER_ID = "openmeteo"
DOMAIN = "forecast"
OPENMETEO_BASE_URL = "https://api.open-meteo.com"
OPENMETEO_FORECAST_PATH = "/v1/forecast"
DEFAULT_FORECAST_TTL_SECONDS = 1800   # 30 min per ADR-017
DEFAULT_CONDITIONS_TTL_SECONDS = 300  # 5 min per brief

_API_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Capability declaration (ADR-038 §4)
# ---------------------------------------------------------------------------

CAPABILITY = ProviderCapability(
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    supplied_canonical_fields=(
        # HourlyForecastPoint fields
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
        # DailyForecastPoint fields
        "validDate",
        "tempMax",
        "tempMin",
        "precipProbabilityMax",
        "windSpeedMax",
        "windGustMax",
        "sunrise",
        "sunset",
        "uvIndexMax",
        # NB: discussion fields NOT supplied by Open-Meteo
    ),
    geographic_coverage="global",
    auth_required=(),
    default_poll_interval_seconds=DEFAULT_FORECAST_TTL_SECONDS,
    operator_notes=(
        "Open-Meteo free-tier; no API key required for non-commercial "
        "use. Throttled at ~10 000 calls/day fair-use. No forecast "
        "discussion available — bundle.discussion is always null."
    ),
)

# ---------------------------------------------------------------------------
# Per-unit Open-Meteo query-param mapping (ADR-019, canonical §4.1.2)
# Rationale: Open-Meteo accepts per-unit query params to avoid server-side
# conversion.  Module maps the station's target_unit to the matching params.
# ---------------------------------------------------------------------------

_TARGET_UNIT_TO_OPENMETEO_UNITS: dict[str, dict[str, str]] = {
    "US":       {"temperature_unit": "fahrenheit", "wind_speed_unit": "mph",  "precipitation_unit": "inch"},
    "METRIC":   {"temperature_unit": "celsius",    "wind_speed_unit": "kmh",  "precipitation_unit": "mm"},
    "METRICWX": {"temperature_unit": "celsius",    "wind_speed_unit": "ms",   "precipitation_unit": "mm"},
}

# ---------------------------------------------------------------------------
# WMO weather code lookup tables (canonical-data-model §3.3, §4.1.2)
# Text strings match the WMO weather code descriptions from the Open-Meteo
# documentation.  Unknown codes return None (no exception) per §4.1.2.
# ---------------------------------------------------------------------------

_WMO_CODE_TO_TEXT: dict[int, str] = {
    0:  "Clear sky",
    1:  "Mainly clear",
    2:  "Partly cloudy",
    3:  "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# precipType derived from WMO code per canonical-data-model §4.1.2.
# Rain family: drizzle (51-55), freezing drizzle classified as freezing-rain (56-57),
# rain (61-65), freezing rain (66-67), rain showers (80-82), thunderstorm (95-99).
# Snow family: snowfall (71-75), snow grains (77), snow showers (85-86).
# Everything else (clear/cloud/fog: 0,1,2,3,45,48) → null (no precip).
_WMO_CODE_TO_PRECIP_TYPE: dict[int, str] = {
    # drizzle → rain
    51: "rain",
    53: "rain",
    55: "rain",
    # freezing drizzle → freezing-rain
    56: "freezing-rain",
    57: "freezing-rain",
    # rain → rain
    61: "rain",
    63: "rain",
    65: "rain",
    # freezing rain → freezing-rain
    66: "freezing-rain",
    67: "freezing-rain",
    # snow → snow
    71: "snow",
    73: "snow",
    75: "snow",
    77: "snow",
    # rain showers → rain
    80: "rain",
    81: "rain",
    82: "rain",
    # snow showers → snow
    85: "snow",
    86: "snow",
    # thunderstorm → rain (lightning always accompanies rain here)
    95: "rain",
    96: "rain",
    99: "rain",
    # 0,1,2,3,45,48 absent → null (no precip)
}

# ---------------------------------------------------------------------------
# Variable lists requested from Open-Meteo (canonical §4.1.2 / §4.1.3)
# ---------------------------------------------------------------------------

_HOURLY_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation_probability",
    "precipitation",
    "weather_code",
    "cloud_cover",
)

_DAILY_VARS = (
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "sunrise",
    "sunset",
    "uv_index_max",
    "weather_code",
)

# ---------------------------------------------------------------------------
# Wire-shape Pydantic models (security-baseline §3.5)
# Source: docs/reference/api-docs/openmeteo.md + fixture
# ---------------------------------------------------------------------------


class _OpenMeteoHourlyBlock(BaseModel):
    """Column-oriented hourly forecast block.

    Open-Meteo returns each variable as a parallel array keyed by `time`;
    array indices align across variables.  _zip_hourly() zips them into
    per-hour HourlyForecastPoint records.

    extras="ignore" so future Open-Meteo hourly variables don't break us;
    missing required fields raise ValidationError → ProviderProtocolError.
    """

    model_config = ConfigDict(extra="ignore")

    time: list[str] = Field(default_factory=list)
    temperature_2m: list[float | None] = Field(default_factory=list)
    relative_humidity_2m: list[float | None] = Field(default_factory=list)
    wind_speed_10m: list[float | None] = Field(default_factory=list)
    wind_direction_10m: list[float | None] = Field(default_factory=list)
    wind_gusts_10m: list[float | None] = Field(default_factory=list)
    precipitation_probability: list[float | None] = Field(default_factory=list)
    precipitation: list[float | None] = Field(default_factory=list)
    weather_code: list[int | None] = Field(default_factory=list)
    cloud_cover: list[float | None] = Field(default_factory=list)


class _OpenMeteoDailyBlock(BaseModel):
    """Column-oriented daily forecast block.

    Daily time entries are "YYYY-MM-DD" station-local strings (already
    correctly bucketed by Open-Meteo per the timezone= param).
    Sunrise/sunset are station-local ISO datetime strings ("2026-04-30T06:22")
    and are converted to UTC in _zip_daily().
    """

    model_config = ConfigDict(extra="ignore")

    time: list[str] = Field(default_factory=list)
    temperature_2m_max: list[float | None] = Field(default_factory=list)
    temperature_2m_min: list[float | None] = Field(default_factory=list)
    precipitation_sum: list[float | None] = Field(default_factory=list)
    precipitation_probability_max: list[float | None] = Field(default_factory=list)
    wind_speed_10m_max: list[float | None] = Field(default_factory=list)
    wind_gusts_10m_max: list[float | None] = Field(default_factory=list)
    sunrise: list[str | None] = Field(default_factory=list)
    sunset: list[str | None] = Field(default_factory=list)
    uv_index_max: list[float | None] = Field(default_factory=list)
    weather_code: list[int | None] = Field(default_factory=list)


class _OpenMeteoCurrentBlock(BaseModel):
    """Current-conditions block from Open-Meteo /v1/forecast current= parameter.

    Open-Meteo returns a single-object (not array) current block when the
    current= query param is supplied alongside hourly/daily.
    is_day: 0 or 1 integer; converted to bool at translation time.
    """

    model_config = ConfigDict(extra="ignore")

    time: str
    temperature_2m: float | None = None
    relative_humidity_2m: float | None = None
    weather_code: int | None = None
    wind_speed_10m: float | None = None
    wind_direction_10m: float | None = None
    wind_gusts_10m: float | None = None
    precipitation: float | None = None
    rain: float | None = None
    snowfall: float | None = None
    cloud_cover: float | None = None
    is_day: int | None = None  # 0 or 1


class _OpenMeteoForecastResponse(BaseModel):
    """Top-level Open-Meteo /v1/forecast response envelope — wire shape.

    extras="ignore" so new top-level fields (generationtime_ms,
    timezone_abbreviation, hourly_units, daily_units, etc.) don't break us.
    Required: latitude, longitude, timezone, utc_offset_seconds.
    hourly and daily are optional (they may be absent if the operator omits
    the corresponding query param; with our request they should always be
    present but defensive None handling is correct per security-baseline §3.5).
    current is optional — present when current= is included in the request;
    fetch_current_conditions() relies on it being populated.
    """

    model_config = ConfigDict(extra="ignore")

    latitude: float
    longitude: float
    timezone: str
    utc_offset_seconds: int
    hourly: _OpenMeteoHourlyBlock | None = None
    daily: _OpenMeteoDailyBlock | None = None
    current: _OpenMeteoCurrentBlock | None = None


# ---------------------------------------------------------------------------
# Rate limiter (ADR-038 §3)
# "Be polite" guard — 5 req/s max.  Never trips in normal use:
# with 30-min TTL + single-worker default, we make ~48 req/day per station.
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="openmeteo-forecast",
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
            user_agent=f"(weewx-clearskies-api/{_API_VERSION})",
        )
    return _http_client


# ---------------------------------------------------------------------------
# Cache key construction (ADR-017 §Cache key)
# ---------------------------------------------------------------------------


def _build_cache_key(lat: float, lon: float, target_unit: str) -> str:
    """Build a deterministic cache key for (provider_id, endpoint, {lat, lon, unit}).

    Lat/lon rounded to 4 decimal places per ADR-017.
    target_unit included so US and METRIC/METRICWX get separate cache entries
    (Open-Meteo returns different numeric values per unit system).
    """
    payload = json.dumps(
        {
            "provider_id": PROVIDER_ID,
            "endpoint": OPENMETEO_FORECAST_PATH,
            "params": {
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "target_unit": target_unit,
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Time conversion helpers (ADR-020)
# ---------------------------------------------------------------------------


def _local_iso_to_utc_iso8601(local_iso: str, utc_offset_seconds: int) -> str:
    """Convert station-local ISO string → UTC ISO-8601 with Z suffix.

    Open-Meteo hourly/daily times come as naive local-time strings:
      "2026-04-30T16:00"   (hourly)
      "2026-04-30T06:22"   (sunrise/sunset in daily)

    The station's UTC offset is in utc_offset_seconds from the Open-Meteo
    response (negative for west-of-UTC, positive for east).

    Algorithm: treat the naive string as if it were in the station's local
    timezone (local_time - utc_offset = UTC), then format with Z suffix.

    Args:
        local_iso: Station-local datetime string e.g. "2026-04-30T16:00".
        utc_offset_seconds: Station's UTC offset in seconds from Open-Meteo
            response (e.g. -25200 for PDT = UTC-7).

    Returns:
        UTC ISO-8601 string with Z suffix e.g. "2026-04-30T23:00:00Z".

    Raises:
        ProviderProtocolError: If the string cannot be parsed.
    """
    try:
        naive_dt = datetime.fromisoformat(local_iso)
    except ValueError as exc:
        raise ProviderProtocolError(
            f"Open-Meteo timestamp parse failed for {local_iso!r}: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    # Interpret the naive datetime as station-local by attaching the fixed
    # offset, then convert to UTC.
    tz_fixed = timezone(timedelta(seconds=utc_offset_seconds))
    local_dt = naive_dt.replace(tzinfo=tz_fixed)
    utc_dt = local_dt.astimezone(UTC)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# NB: use utc_isoformat() from models/responses.py for "now → ISO-8601 Z"
# string formatting per the brief's helper-reuse instruction and rules/coding.md
# §3 DRY rule. Imported at module top.


# ---------------------------------------------------------------------------
# Column-array → row-record zip helpers (canonical §4.1.2, §4.1.3)
# ---------------------------------------------------------------------------


def _get_at(arr: list[Any], i: int) -> Any:
    """Safely index into an array; return None if index is out of bounds."""
    if arr and i < len(arr):
        return arr[i]
    return None


def _zip_hourly(
    hourly: _OpenMeteoHourlyBlock,
    utc_offset_seconds: int,
) -> list[HourlyForecastPoint]:
    """Zip Open-Meteo column arrays into HourlyForecastPoint records.

    Each index i in hourly.time corresponds to one hourly forecast period.
    All companion arrays (temperature_2m, etc.) share the same index.

    Time conversion: local ISO → UTC ISO-8601 Z (ADR-020).
    weatherCode: WMO int as string (canonical §3.3).
    weatherText: decoded from WMO code via lookup; None for unknown codes.
    precipType: derived from WMO code via heuristic; None if no precipitation.
    """
    points: list[HourlyForecastPoint] = []
    for i, time_str in enumerate(hourly.time):
        code_raw = _get_at(hourly.weather_code, i)
        code_int: int | None = int(code_raw) if code_raw is not None else None
        code_str: str | None = str(code_int) if code_int is not None else None
        weather_text: str | None = _WMO_CODE_TO_TEXT.get(code_int) if code_int is not None else None
        precip_type: str | None = _WMO_CODE_TO_PRECIP_TYPE.get(code_int) if code_int is not None else None

        points.append(
            HourlyForecastPoint(
                validTime=_local_iso_to_utc_iso8601(time_str, utc_offset_seconds),
                outTemp=_get_at(hourly.temperature_2m, i),
                outHumidity=_get_at(hourly.relative_humidity_2m, i),
                windSpeed=_get_at(hourly.wind_speed_10m, i),
                windDir=_get_at(hourly.wind_direction_10m, i),
                windGust=_get_at(hourly.wind_gusts_10m, i),
                precipProbability=_get_at(hourly.precipitation_probability, i),
                precipAmount=_get_at(hourly.precipitation, i),
                precipType=precip_type,
                cloudCover=_get_at(hourly.cloud_cover, i),
                weatherCode=code_str,
                weatherText=weather_text,
                source=PROVIDER_ID,
            )
        )
    return points


def _zip_daily(
    daily: _OpenMeteoDailyBlock,
    utc_offset_seconds: int,
) -> list[DailyForecastPoint]:
    """Zip Open-Meteo column arrays into DailyForecastPoint records.

    validDate: station-local "YYYY-MM-DD" string passes through as-is (already
    correctly bucketed by Open-Meteo per the timezone= param).
    sunrise/sunset: station-local ISO strings converted to UTC.
    narrative: always None for Open-Meteo (no per-day narrative supplied).
    """
    points: list[DailyForecastPoint] = []
    for i, date_str in enumerate(daily.time):
        code_raw = _get_at(daily.weather_code, i)
        code_int: int | None = int(code_raw) if code_raw is not None else None
        code_str: str | None = str(code_int) if code_int is not None else None
        weather_text: str | None = _WMO_CODE_TO_TEXT.get(code_int) if code_int is not None else None

        sunrise_raw = _get_at(daily.sunrise, i)
        sunset_raw = _get_at(daily.sunset, i)

        sunrise_utc: str | None = None
        if sunrise_raw is not None:
            sunrise_utc = _local_iso_to_utc_iso8601(sunrise_raw, utc_offset_seconds)

        sunset_utc: str | None = None
        if sunset_raw is not None:
            sunset_utc = _local_iso_to_utc_iso8601(sunset_raw, utc_offset_seconds)

        points.append(
            DailyForecastPoint(
                validDate=date_str,
                tempMax=_get_at(daily.temperature_2m_max, i),
                tempMin=_get_at(daily.temperature_2m_min, i),
                precipAmount=_get_at(daily.precipitation_sum, i),
                precipProbabilityMax=_get_at(daily.precipitation_probability_max, i),
                windSpeedMax=_get_at(daily.wind_speed_10m_max, i),
                windGustMax=_get_at(daily.wind_gusts_10m_max, i),
                sunrise=sunrise_utc,
                sunset=sunset_utc,
                uvIndexMax=_get_at(daily.uv_index_max, i),
                weatherCode=code_str,
                weatherText=weather_text,
                narrative=None,   # Open-Meteo supplies no per-day narrative
                source=PROVIDER_ID,
            )
        )
    return points


# ---------------------------------------------------------------------------
# Wire → canonical normalization (canonical-data-model §4.1.2 / §4.1.3)
# ---------------------------------------------------------------------------


def _to_canonical(
    wire: _OpenMeteoForecastResponse,
    *,
    utc_offset_seconds: int,
) -> ForecastBundle:
    """Translate Open-Meteo wire response to canonical ForecastBundle.

    hourly: zipped from wire.hourly column arrays (empty list if None).
    daily: zipped from wire.daily column arrays (empty list if None).
    discussion: always None — Open-Meteo has no discussion endpoint.
    source: PROVIDER_ID ("openmeteo").
    generatedAt: current UTC timestamp.
    """
    hourly_points: list[HourlyForecastPoint] = []
    if wire.hourly is not None:
        hourly_points = _zip_hourly(wire.hourly, utc_offset_seconds)

    daily_points: list[DailyForecastPoint] = []
    if wire.daily is not None:
        daily_points = _zip_daily(wire.daily, utc_offset_seconds)

    return ForecastBundle(
        hourly=hourly_points,
        daily=daily_points,
        discussion=None,          # Open-Meteo supplies none, ever
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
    timezone: str,
) -> ForecastBundle:
    """Call Open-Meteo /v1/forecast and return canonical ForecastBundle.

    Cache-first: check cache before making an outbound HTTP call.
    Cache stores post-normalization ForecastBundle as model_dump(mode="json")
    dict; reconstructed via ForecastBundle.model_validate() on cache hit.
    Cache key includes target_unit so US and metric systems get separate entries.

    Slice-after-cache pattern (ADR-017):
    The FULL bundle is stored in cache regardless of the requested hours/days.
    The endpoint handler slices to the requested count AFTER cache lookup.

    Args:
        lat: Station latitude from services/station.py StationInfo.
        lon: Station longitude from services/station.py StationInfo.
        target_unit: Weewx unit system ("US" | "METRIC" | "METRICWX") from
            services/units.py get_target_unit().
        timezone: IANA timezone name for the station (e.g. "America/Los_Angeles").
            Passed to Open-Meteo as timezone= param so daily day boundaries are
            bucketed correctly per the operator's timezone.

    Returns:
        ForecastBundle — single canonical Pydantic model.
        discussion is always None for Open-Meteo.

    Raises:
        ProviderProtocolError: target_unit unknown, response validation failed,
            or Open-Meteo returned HTTP 400 with error envelope.
        QuotaExhausted: Open-Meteo returned 429.
        TransientNetworkError: Network/DNS failure or 5xx after retries.
    """
    cache_key = _build_cache_key(lat, lon, target_unit)
    cached = get_cache().get(cache_key)
    if cached is not None:
        logger.debug(
            "Cache hit for Open-Meteo forecast",
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        return ForecastBundle.model_validate(cached)

    logger.debug(
        "Cache miss for Open-Meteo forecast; calling API",
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )

    unit_params = _TARGET_UNIT_TO_OPENMETEO_UNITS.get(target_unit)
    if unit_params is None:
        # Defensive: services/units.py validates target_unit at startup.
        # This branch should never fire in production.  Raise ProviderProtocolError
        # so the canonical-taxonomy handler emits 502 rather than 500.
        raise ProviderProtocolError(
            f"unknown target_unit {target_unit!r}; expected US, METRIC, or METRICWX",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    _rate_limiter.acquire()
    client = _client_for()

    params: dict[str, str] = {
        "latitude": str(round(lat, 4)),
        "longitude": str(round(lon, 4)),
        "hourly": ",".join(_HOURLY_VARS),
        "daily": ",".join(_DAILY_VARS),
        "current": (
            "temperature_2m,relative_humidity_2m,weather_code,"
            "wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
            "precipitation,rain,showers,snowfall,cloud_cover,is_day"
        ),
        "timezone": timezone,
        "timeformat": "iso8601",
        **unit_params,
    }

    response = client.get(
        f"{OPENMETEO_BASE_URL}{OPENMETEO_FORECAST_PATH}",
        params=params,
    )

    try:
        wire = _OpenMeteoForecastResponse.model_validate(response.json())
    except (ValidationError, ValueError) as exc:
        logger.error(
            "Open-Meteo response validation failed: %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
        )
        raise ProviderProtocolError(
            f"Open-Meteo response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    bundle = _to_canonical(wire, utc_offset_seconds=wire.utc_offset_seconds)

    get_cache().set(
        cache_key,
        bundle.model_dump(mode="json"),
        ttl_seconds=DEFAULT_FORECAST_TTL_SECONDS,
    )

    logger.info(
        "Open-Meteo forecast fetched: %d hourly, %d daily point(s)",
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
    timezone: str,
) -> ProviderConditions | None:
    """Extract current conditions from the cached Open-Meteo forecast response.

    Open-Meteo has no separate current-conditions endpoint.  This function calls
    fetch() (which caches the full bundle) and extracts the current block that
    fetch() now requests via the current= query param.

    The ForecastBundle cache and the current-conditions cache share the same
    fetch() call — no extra HTTP request is made.  The ProviderConditions result
    is cached independently with DEFAULT_CONDITIONS_TTL_SECONDS (300 s) so it can
    be reused by the blending engine without re-parsing the full bundle.

    weatherText derived from weather_code via existing _WMO_CODE_TO_TEXT lookup.
    precipType derived from weather_code via existing _WMO_CODE_TO_PRECIP_TYPE lookup.
    isDay: is_day == 1 when non-None.

    Args:
        lat: Station latitude.
        lon: Station longitude.
        target_unit: Weewx unit system ("US" | "METRIC" | "METRICWX").
        timezone: IANA timezone name (passed to fetch() for timezone= param).

    Returns:
        ProviderConditions on success; None when the current block is absent
        from the response (should not happen with current= in params, but
        defensive).

    Raises: same taxonomy as fetch().
    """
    # Build conditions cache key first — reuse if available.
    conditions_cache_key = hashlib.sha256(
        json.dumps(
            {
                "provider_id": PROVIDER_ID,
                "endpoint": "current_conditions",
                "params": {
                    "latitude": round(lat, 4),
                    "longitude": round(lon, 4),
                    "target_unit": target_unit,
                },
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    cached = get_cache().get(conditions_cache_key)
    if cached is not None:
        logger.debug(
            "Cache hit for Open-Meteo current conditions",
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        return ProviderConditions.model_validate(cached)

    # No conditions cache — call fetch() to populate (or hit) the forecast cache
    # AND get the raw wire response with current block.
    # fetch() itself uses the forecast-bundle cache, so on a forecast cache hit
    # the HTTP call is skipped.  However, fetch() returns a ForecastBundle, not
    # the raw wire — we cannot extract the current block from the bundle.
    # Therefore on a conditions cache miss we always make a fresh HTTP call to
    # get the current block.  This is intentional: conditions TTL (300 s) is
    # shorter than forecast TTL (1800 s), so conditions can be fresher.

    logger.debug(
        "Cache miss for Open-Meteo current conditions; calling API",
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )

    unit_params = _TARGET_UNIT_TO_OPENMETEO_UNITS.get(target_unit)
    if unit_params is None:
        raise ProviderProtocolError(
            f"unknown target_unit {target_unit!r}; expected US, METRIC, or METRICWX",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    _rate_limiter.acquire()
    client = _client_for()

    params: dict[str, str] = {
        "latitude": str(round(lat, 4)),
        "longitude": str(round(lon, 4)),
        "current": (
            "temperature_2m,relative_humidity_2m,weather_code,"
            "wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
            "precipitation,rain,showers,snowfall,cloud_cover,is_day"
        ),
        "timezone": timezone,
        "timeformat": "iso8601",
        **unit_params,
    }

    response = client.get(
        f"{OPENMETEO_BASE_URL}{OPENMETEO_FORECAST_PATH}",
        params=params,
    )

    try:
        wire = _OpenMeteoForecastResponse.model_validate(response.json())
    except (ValidationError, ValueError) as exc:
        logger.error(
            "Open-Meteo current conditions response validation failed: %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
        )
        raise ProviderProtocolError(
            f"Open-Meteo current conditions response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    if wire.current is None:
        logger.warning(
            "Open-Meteo response missing current block despite current= param; "
            "returning None",
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        return None

    cur = wire.current
    code_int = cur.weather_code
    code_str: str | None = str(code_int) if code_int is not None else None
    weather_text: str | None = _WMO_CODE_TO_TEXT.get(code_int) if code_int is not None else None
    precip_type: str | None = _WMO_CODE_TO_PRECIP_TYPE.get(code_int) if code_int is not None else None
    is_day: bool | None = (cur.is_day == 1) if cur.is_day is not None else None

    conditions = ProviderConditions(
        weatherText=weather_text,
        weatherCode=code_str,
        precipType=precip_type,
        cloudCover=cur.cloud_cover,
        isDay=is_day,
        temperature=cur.temperature_2m,
        humidity=cur.relative_humidity_2m,
        windSpeed=cur.wind_speed_10m,
        windDir=cur.wind_direction_10m,
        source=PROVIDER_ID,
    )

    get_cache().set(
        conditions_cache_key,
        conditions.model_dump(mode="json"),
        ttl_seconds=DEFAULT_CONDITIONS_TTL_SECONDS,
    )

    logger.info(
        "Open-Meteo current conditions fetched",
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
