"""Aeris (AerisWeather/Xweather) forecast provider module (ADR-007, ADR-038).

Five responsibilities per ADR-038 §2:
  1. Outbound API calls — two per cache miss:
       GET /forecasts/{lat},{lon}?filter=1hr  → hourly periods
       GET /forecasts/{lat},{lon}?filter=daynight → paired day/night periods
  2. Response parsing — wire-shape Pydantic models for each response
  3. Translation to canonical ForecastBundle (HourlyForecastPoint + DailyForecastPoint)
  4. Capability declaration — CAPABILITY symbol consumed at startup
  5. Error handling — provider errors translated to canonical taxonomy

Aeris is the first keyed provider on this project (ADR-006):
  client_id + client_secret passed as query params on every request.
  Sourced from env vars WEEWX_CLEARSKIES_AERIS_CLIENT_ID +
  WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET at startup (ADR-027 §3).
  Naming deviation: provider-scoped (not domain-scoped) per brief Q1 user
  decision 2026-05-08. Operator pastes once; works for forecast AND future
  alerts/observation modules. Deviation from ADR-027 §3 literal schema
  documented here and in settings.py docstring; no ADR amendment.

Cache layer (ADR-017):
  Caches the post-normalization ForecastBundle, not raw JSON.
  Key: SHA-256 of (provider_id, endpoint="forecast_bundle", {lat4, lon4, target_unit}).
  TTL: 1800s (30 min per ADR-017 defaults table for forecast).
  Single cache entry covers BOTH upstream calls (hourly + daynight).
  Cache stores model_dump(mode="json"); reconstructed via model_validate().

Slice-after-cache pattern (ADR-017 §Cache key):
  Full bundle stored in cache; endpoint applies hours/days slice after lookup.
  One cache entry per (station, target_unit) — limit=240/14 captures all
  periods Aeris will return for typical entry-paid plans.

Time conversion (ADR-020):
  Aeris period timestamps include UTC offset ("2026-04-30T10:00:00-07:00").
  to_utc_iso8601_from_offset() from _common/datetime_utils.py normalises to
  UTC Z form. validDate extracted from dateTimeISO BEFORE conversion — the
  offset IS the station-local one Aeris applies via profile.tz lookup.

Per-unit handling (ADR-019):
  Aeris returns BOTH metric + imperial fields in same payload; no units= param.
  Module picks the right field names based on target_unit.
  US → *F / *MPH / *IN, METRIC → *C / *KPH / *MM, METRICWX → *C / *MPS / *MM.
  windSpeedMaxMPS / windGustMaxMPS: if absent from daynight payload but KPH
  variants present, post-convert at canonical-translation time (brief lead-call 13).

Aeris weather code pass-through (canonical-data-model §4.1.2, §4.1.3):
  weatherCode = weatherPrimaryCoded string passed through opaque.
  weatherText = weather string (e.g. "Partly Cloudy") passed through.
  precipType derived via _aeris_descriptor_to_precip_type() from the
  third colon-segment of weatherPrimaryCoded per brief lead-call 16.

ForecastDiscussion (brief Q2 runtime detection, user decision 2026-05-08):
  Module attempts runtime detection of paid-tier summary field at:
    response[0].summary  (response-level)
    response[0].periods[0].summary  (period-level)
  When present and non-empty: ForecastDiscussion(headline=weatherPrimary,
    body=<summary>, source="aeris", issuedAt=<UTC-converted dateTimeISO>).
  When absent/empty/whitespace-only: discussion=None.
  CAPABILITY.supplied_canonical_fields declares headline + body as max-surface;
  runtime population is conditional (user-accepted capability-vs-runtime drift).

Rate limiter (ADR-038 §3):
  RateLimiter("aeris-forecast", max_calls=5, window_seconds=1) as "be polite"
  guard. Per-call acquire before each of the two outbound calls per cache miss.

ruff: noqa: N815  (field names match Aeris camelCase: dateTimeISO, maxTempF, etc.)
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
    ForecastDiscussion,
    HourlyForecastPoint,
    utc_isoformat,
)
from weewx_clearskies_api.providers._common.cache import get_cache
from weewx_clearskies_api.providers._common.capability import ProviderCapability
from weewx_clearskies_api.providers._common.datetime_utils import to_utc_iso8601_from_offset
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

PROVIDER_ID = "aeris"
DOMAIN = "forecast"
AERIS_BASE_URL = "https://data.api.xweather.com"
AERIS_FORECASTS_PATH = "/forecasts"
DEFAULT_FORECAST_TTL_SECONDS = 1800   # 30 min per ADR-017
HOURLY_LIMIT = 240                     # 10 days × 24h, well above 384h ForecastQueryParams cap
DAYNIGHT_LIMIT = 14                    # 7 days × 2 (paired day/night)

_API_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Capability declaration (ADR-038 §4)
# ---------------------------------------------------------------------------

CAPABILITY = ProviderCapability(
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    supplied_canonical_fields=(
        # HourlyForecastPoint
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
        # DailyForecastPoint
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
        # ForecastDiscussion — max-surface; populated only on paid-tier responses
        # where summary field is detected at runtime (Q2 user decision 2026-05-08).
        # Free-tier returns bundle.discussion=null.  Auditor note: this is a
        # capability-vs-runtime-fidelity trade-off accepted by the user at brief Q2.
        "headline",
        "body",
        # NB: narrative NOT supplied by Aeris v0.1 (brief lead-call 20)
    ),
    geographic_coverage="global",   # Trust Aeris's authoritative answer per lead-call 17
    auth_required=("client_id", "client_secret"),
    default_poll_interval_seconds=DEFAULT_FORECAST_TTL_SECONDS,
    operator_notes=(
        "Aeris (AerisWeather/Xweather) free-tier and entry-paid plans. "
        "Requires client_id + client_secret bound to a registered domain "
        "or bundle id (see docs/reference/api-docs/aeris.md §Authentication). "
        "Forecast discussion populated when paid-tier summary field is present; "
        "free-tier returns bundle.discussion=null. Coverage per Aeris's "
        "authoritative answer; warn_location responses return empty bundle."
    ),
)

# ---------------------------------------------------------------------------
# Aeris weather descriptor → canonical precipType (brief lead-call 16)
# Third colon-segment of weatherPrimaryCoded (e.g. "::OVC" → descriptor="OVC").
# Unknown descriptors → None (log DEBUG once on first encounter).
# ---------------------------------------------------------------------------

_AERIS_DESCRIPTOR_TO_PRECIP_TYPE: dict[str, str] = {
    # rain family
    "R": "rain",
    "RW": "rain",
    "L": "rain",       # drizzle → rain
    # snow family
    "S": "snow",
    "SW": "snow",
    # freezing
    "ZR": "freezing-rain",
    "ZL": "freezing-rain",   # freezing drizzle
    # ice/sleet
    "IP": "sleet",
    # hail
    "A": "hail",
    # thunder accompanies rain in canonical framing (consistent with NWS tsra → rain, 3b-3)
    "T": "rain",
    # mixed precip — canonical has no mixed-precip enum; log DEBUG on encounter
    "RS": "rain",    # rain/snow mix
    "WM": "rain",    # wintry mix
    "SI": "rain",    # snow/sleet
}

# Track which unknown descriptors have been logged to avoid log spam.
_logged_unknown_descriptors: set[str] = set()
# Track which mixed-precip descriptors have been logged for future canonical amendment.
_logged_mixed_precip: set[str] = set()
_MIXED_PRECIP_DESCRIPTORS = frozenset({"RS", "WM", "SI"})

# ---------------------------------------------------------------------------
# Wire-shape Pydantic models (security-baseline §3.5)
# Source: docs/reference/api-docs/aeris.md + brief §per-module spec
# extras="ignore" so Aeris additions don't break us; missing required fields
# raise ValidationError → translated to ProviderProtocolError.
# ---------------------------------------------------------------------------


class _AerisLoc(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lat: float
    long: float


class _AerisProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tz: str | None = None
    elevFT: float | None = None
    elevM: float | None = None


class _AerisHourlyPeriod(BaseModel):
    """One hourly period from /forecasts?filter=1hr."""

    model_config = ConfigDict(extra="ignore")

    timestamp: int
    dateTimeISO: str
    # Temperature — both unit systems present in payload; module picks per target_unit
    tempC: float | None = None
    tempF: float | None = None
    # Humidity
    humidity: float | None = None
    # Wind speed — both unit systems
    windSpeedKPH: float | None = None
    windSpeedMPH: float | None = None
    windSpeedMPS: float | None = None
    windDirDEG: float | None = None
    # Wind gust
    windGustKPH: float | None = None
    windGustMPH: float | None = None
    windGustMPS: float | None = None
    # Precipitation
    precipMM: float | None = None
    precipIN: float | None = None
    pop: float | None = None    # probability of precipitation (0-100)
    # Dewpoint
    dewpointC: float | None = None
    dewpointF: float | None = None
    # Pressure
    pressureMB: float | None = None
    pressureIN: float | None = None
    # Sky/cloud
    sky: float | None = None   # 0-100 cloud cover percent
    # Weather codes
    weatherPrimaryCoded: str | None = None
    weather: str | None = None
    weatherPrimary: str | None = None
    uvi: float | None = None
    isDay: bool | None = None


class _AerisDayNightPeriod(BaseModel):
    """One day or night period from /forecasts?filter=daynight."""

    model_config = ConfigDict(extra="ignore")

    timestamp: int
    dateTimeISO: str
    # Temperature extremes — both unit systems
    maxTempC: float | None = None
    maxTempF: float | None = None
    minTempC: float | None = None
    minTempF: float | None = None
    # Wind speed max — both unit systems
    windSpeedKPH: float | None = None
    windSpeedMPH: float | None = None
    windSpeedMPS: float | None = None
    windSpeedMaxKPH: float | None = None
    windSpeedMaxMPH: float | None = None
    windSpeedMaxMPS: float | None = None   # may be absent; fall back to KPH÷3.6 per lead-call 13
    # Wind gust max — both unit systems
    windGustKPH: float | None = None
    windGustMPH: float | None = None
    windGustMPS: float | None = None
    windGustMaxKPH: float | None = None
    windGustMaxMPH: float | None = None
    windGustMaxMPS: float | None = None   # may be absent; fall back to KPH÷3.6 per lead-call 13
    # Precipitation
    precipMM: float | None = None
    precipIN: float | None = None
    pop: float | None = None
    # Sunrise/sunset
    sunriseISO: str | None = None
    sunsetISO: str | None = None
    # UV index
    uvi: float | None = None
    # Weather codes
    weatherPrimaryCoded: str | None = None
    weather: str | None = None
    weatherPrimary: str | None = None
    isDay: bool | None = None


class _AerisHourlyResponse(BaseModel):
    """Top-level /forecasts?filter=1hr response — wire shape."""

    model_config = ConfigDict(extra="ignore")

    loc: _AerisLoc | None = None
    profile: _AerisProfile | None = None
    periods: list[_AerisHourlyPeriod] = Field(default_factory=list)


class _AerisDayNightResponse(BaseModel):
    """Top-level /forecasts?filter=daynight response — wire shape."""

    model_config = ConfigDict(extra="ignore")

    loc: _AerisLoc | None = None
    profile: _AerisProfile | None = None
    periods: list[_AerisDayNightPeriod] = Field(default_factory=list)


class _AerisEnvelope(BaseModel):
    """Aeris response envelope — success/error wrapper."""

    model_config = ConfigDict(extra="ignore")

    success: bool
    error: dict[str, Any] | None = None
    # response is a list of location objects; we always use [0]
    response: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Rate limiter (ADR-038 §3, brief lead-call 18)
# 5 req/s "be polite" guard — per-call acquire before each of two outbound
# calls per cache miss. Covers lowest documented Aeris paid-tier (10/s entry).
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="aeris-forecast",
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
# Cache key construction (ADR-017 §Cache key)
# ---------------------------------------------------------------------------


def _build_cache_key(lat: float, lon: float, target_unit: str) -> str:
    """Build a deterministic cache key for (provider_id, endpoint, {lat, lon, unit}).

    endpoint="forecast_bundle" covers the two upstream calls (hourly + daynight)
    per brief §cache integration. Lat/lon rounded to 4 decimal places per ADR-017.
    target_unit included so US and METRIC/METRICWX get separate cache entries
    (module picks different field names per unit system at ingest time).
    """
    payload = json.dumps(
        {
            "provider_id": PROVIDER_ID,
            "endpoint": "forecast_bundle",
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
# Helpers — precipType derivation (brief lead-call 16)
# ---------------------------------------------------------------------------


def _aeris_descriptor_to_precip_type(coded: str | None) -> str | None:
    """Derive canonical precipType from Aeris weatherPrimaryCoded string.

    Aeris weatherPrimaryCoded is colon-delimited:
      <coverage>:<intensity>:<descriptor>
    e.g. ":HV:RW" (heavy rain shower), "::OVC" (overcast, no precip).
    The third segment (index 2) is the weather descriptor that drives
    precipType classification.

    Args:
        coded: Aeris weatherPrimaryCoded string or None.

    Returns:
        Canonical precipType string ("rain", "snow", "freezing-rain",
        "sleet", "hail") or None if no precipitation descriptor found.
    """
    if not coded:
        return None

    parts = coded.split(":")
    descriptor = parts[2] if len(parts) >= 3 else ""
    if not descriptor:
        return None

    result = _AERIS_DESCRIPTOR_TO_PRECIP_TYPE.get(descriptor)

    if result is not None:
        # Log mixed-precip descriptors once so future canonical amendment
        # is informed by real-data prevalence (brief lead-call 16).
        if descriptor in _MIXED_PRECIP_DESCRIPTORS and descriptor not in _logged_mixed_precip:
            _logged_mixed_precip.add(descriptor)
            logger.debug(
                "Aeris mixed-precip descriptor %r mapped to 'rain' "
                "(canonical has no mixed-precip enum; track for future amendment)",
                descriptor,
            )
        return result

    # Unknown descriptor — log once, return None
    if descriptor not in _logged_unknown_descriptors:
        _logged_unknown_descriptors.add(descriptor)
        logger.debug(
            "Aeris unknown weather descriptor %r → precipType=None "
            "(update _AERIS_DESCRIPTOR_TO_PRECIP_TYPE if this is a known type)",
            descriptor,
        )
    return None


# ---------------------------------------------------------------------------
# Helpers — unit-field selection (ADR-019, brief lead-call 13)
# ---------------------------------------------------------------------------


def _wind_speed_max_mps(period: _AerisDayNightPeriod) -> float | None:
    """Return windSpeedMax in m/s for METRICWX, with KPH fallback.

    Aeris doesn't document windSpeedMaxMPS explicitly; if absent, post-convert
    from windSpeedMaxKPH ÷ 3.6 per brief lead-call 13.
    """
    if period.windSpeedMaxMPS is not None:
        return period.windSpeedMaxMPS
    if period.windSpeedMaxKPH is not None:
        return period.windSpeedMaxKPH / 3.6
    return None


def _wind_gust_max_mps(period: _AerisDayNightPeriod) -> float | None:
    """Return windGustMax in m/s for METRICWX, with KPH fallback.

    Same post-convert pattern as _wind_speed_max_mps.
    """
    if period.windGustMaxMPS is not None:
        return period.windGustMaxMPS
    if period.windGustMaxKPH is not None:
        return period.windGustMaxKPH / 3.6
    return None


# ---------------------------------------------------------------------------
# Hourly period → canonical HourlyForecastPoint (canonical-data-model §4.1.2)
# ---------------------------------------------------------------------------


def _hourly_period_to_point(period: _AerisHourlyPeriod, target_unit: str) -> HourlyForecastPoint:
    """Translate one Aeris hourly period to canonical HourlyForecastPoint.

    Unit-field selection per ADR-019 + brief lead-call 13:
      US       → *F, *MPH, *IN
      METRIC   → *C, *KPH, *MM
      METRICWX → *C, *MPS, *MM
    Aeris returns both systems; module picks the matching field name.
    """
    # validTime: UTC ISO-8601 Z from offset-aware dateTimeISO
    valid_time = to_utc_iso8601_from_offset(
        period.dateTimeISO, provider_id=PROVIDER_ID, domain=DOMAIN
    )

    # Temperature
    if target_unit == "US":
        temp = period.tempF
        wind_speed = period.windSpeedMPH
        wind_gust = period.windGustMPH
        precip_amount = period.precipIN
    elif target_unit == "METRICWX":
        temp = period.tempC
        wind_speed = period.windSpeedMPS
        wind_gust = period.windGustMPS
        precip_amount = period.precipMM
    else:  # METRIC
        temp = period.tempC
        wind_speed = period.windSpeedKPH
        wind_gust = period.windGustKPH
        precip_amount = period.precipMM

    return HourlyForecastPoint(
        validTime=valid_time,
        outTemp=temp,
        outHumidity=period.humidity,
        windSpeed=wind_speed,
        windDir=period.windDirDEG,
        windGust=wind_gust,
        precipProbability=period.pop,
        precipAmount=precip_amount,
        precipType=_aeris_descriptor_to_precip_type(period.weatherPrimaryCoded),
        cloudCover=period.sky,
        weatherCode=period.weatherPrimaryCoded,
        weatherText=period.weather,
        source=PROVIDER_ID,
    )


# ---------------------------------------------------------------------------
# Day/night period → canonical DailyForecastPoint (canonical-data-model §4.1.3)
# Aeris filter=daynight returns alternating day and night periods.
# Module pairs consecutive periods: day period + immediately-following night period.
# Only day-period values are used for the canonical DailyForecastPoint;
# validDate is derived from the day period's dateTimeISO (local date, before UTC conv).
# ---------------------------------------------------------------------------


def _daynight_periods_to_daily(
    periods: list[_AerisDayNightPeriod],
    target_unit: str,
) -> list[DailyForecastPoint]:
    """Translate Aeris day/night period pairs to canonical DailyForecastPoint list.

    Aeris filter=daynight returns alternating day/night pairs in chronological
    order: [day0, night0, day1, night1, ...]. We iterate by stepping 2 and
    using the day period for canonical daily values.

    validDate: date portion of dateTimeISO BEFORE UTC conversion — the offset
    IS the station-local one Aeris applies via profile.tz lookup (brief call 22).
    sunrise/sunset: converted to UTC ISO-8601 Z via to_utc_iso8601_from_offset.
    """
    points: list[DailyForecastPoint] = []
    # Step through day periods (even indices — isDay=True)
    i = 0
    while i < len(periods):
        day_period = periods[i]
        # Skip night-only periods if they appear at the start (defensive)
        if day_period.isDay is False:
            i += 1
            continue

        # validDate: station-local date extracted from dateTimeISO before any conversion
        valid_date = day_period.dateTimeISO[:10]   # "YYYY-MM-DD"

        # Unit-field selection per ADR-019 + brief lead-call 13
        if target_unit == "US":
            temp_max = day_period.maxTempF
            temp_min = day_period.minTempF
            wind_speed_max = day_period.windSpeedMaxMPH
            wind_gust_max = day_period.windGustMaxMPH
            precip_amount = day_period.precipIN
        elif target_unit == "METRICWX":
            temp_max = day_period.maxTempC
            temp_min = day_period.minTempC
            wind_speed_max = _wind_speed_max_mps(day_period)
            wind_gust_max = _wind_gust_max_mps(day_period)
            precip_amount = day_period.precipMM
        else:  # METRIC
            temp_max = day_period.maxTempC
            temp_min = day_period.minTempC
            wind_speed_max = day_period.windSpeedMaxKPH
            wind_gust_max = day_period.windGustMaxKPH
            precip_amount = day_period.precipMM

        # Sunrise/sunset: convert from local ISO-with-offset to UTC Z
        sunrise_utc: str | None = None
        if day_period.sunriseISO:
            sunrise_utc = to_utc_iso8601_from_offset(
                day_period.sunriseISO, provider_id=PROVIDER_ID, domain=DOMAIN
            )

        sunset_utc: str | None = None
        if day_period.sunsetISO:
            sunset_utc = to_utc_iso8601_from_offset(
                day_period.sunsetISO, provider_id=PROVIDER_ID, domain=DOMAIN
            )

        points.append(
            DailyForecastPoint(
                validDate=valid_date,
                tempMax=temp_max,
                tempMin=temp_min,
                precipAmount=precip_amount,
                precipProbabilityMax=day_period.pop,
                windSpeedMax=wind_speed_max,
                windGustMax=wind_gust_max,
                sunrise=sunrise_utc,
                sunset=sunset_utc,
                uvIndexMax=day_period.uvi,
                weatherCode=day_period.weatherPrimaryCoded,
                weatherText=day_period.weather,
                narrative=None,   # Aeris paid-tier `text` field deferred to future round (call 20)
                source=PROVIDER_ID,
            )
        )
        # Skip both day and night periods (or just this day if no night follows)
        i += 2

    return points


# ---------------------------------------------------------------------------
# ForecastDiscussion runtime detection (brief Q2, user decision 2026-05-08)
# ---------------------------------------------------------------------------


def _extract_aeris_discussion(
    daynight_raw: dict[str, Any],
    first_period_raw: dict[str, Any] | None,
    *,
    provider_id: str,
    domain: str,
) -> ForecastDiscussion | None:
    """Attempt runtime detection of paid-tier summary field for ForecastDiscussion.

    Checks two candidate locations per brief lead-call 14:
      - daynight_raw["summary"]  (response-level summary)
      - first_period_raw["summary"]  (per-period summary)

    When a non-empty string is found, constructs ForecastDiscussion with:
      headline = weatherPrimary of first period
      body = detected summary string
      source = "aeris"
      issuedAt = UTC-converted dateTimeISO of first period
      validFrom = None (Aeris doesn't expose a forecast-valid-from timestamp)
      validUntil = None

    When absent/empty/whitespace-only → returns None (free-tier default).

    Args:
        daynight_raw: Raw dict for response[0] from daynight Pydantic model.
        first_period_raw: Raw dict for response[0].periods[0] or None.
        provider_id: For ProviderProtocolError context.
        domain: For ProviderProtocolError context.
    """
    summary_text: str | None = None

    # Check response-level summary first
    candidate = daynight_raw.get("summary")
    if isinstance(candidate, str) and candidate.strip():
        summary_text = candidate.strip()
        logger.debug("Aeris: detected response-level summary field (paid-tier)")

    # Fall back to period-level summary
    if summary_text is None and first_period_raw is not None:
        candidate = first_period_raw.get("summary")
        if isinstance(candidate, str) and candidate.strip():
            summary_text = candidate.strip()
            logger.debug("Aeris: detected period-level summary field (paid-tier)")

    if summary_text is None:
        return None

    # Build ForecastDiscussion from first period data
    headline: str | None = None
    issued_at: str | None = None

    if first_period_raw is not None:
        headline = first_period_raw.get("weatherPrimary") or None
        raw_dt = first_period_raw.get("dateTimeISO")
        if isinstance(raw_dt, str):
            try:
                issued_at = to_utc_iso8601_from_offset(
                    raw_dt, provider_id=provider_id, domain=domain
                )
            except ProviderProtocolError:
                # to_utc_iso8601_from_offset raises ProviderProtocolError on
                # malformed input. Discussion issuedAt is best-effort; absent
                # is acceptable per canonical §3.5 (issuedAt nullable).
                logger.debug("Aeris: could not parse dateTimeISO for discussion issuedAt")
                issued_at = None

    return ForecastDiscussion(
        headline=headline,
        body=summary_text,
        source=PROVIDER_ID,
        issuedAt=issued_at,
        validFrom=None,
        validUntil=None,
        senderName=None,
    )


# ---------------------------------------------------------------------------
# Wire → canonical normalization (canonical-data-model §4.1.2 / §4.1.3)
# ---------------------------------------------------------------------------


def _to_canonical(
    hourly_wire: _AerisHourlyResponse,
    daynight_wire: _AerisDayNightResponse,
    *,
    target_unit: str,
    daynight_raw: dict[str, Any],
) -> ForecastBundle:
    """Translate Aeris wire responses to canonical ForecastBundle.

    hourly: translated from hourly_wire.periods.
    daily: paired from daynight_wire.periods (day periods only).
    discussion: runtime-detected from daynight_raw (paid-tier only, None for free-tier).
    source: PROVIDER_ID ("aeris").
    generatedAt: current UTC timestamp.
    """
    hourly_points = [
        _hourly_period_to_point(p, target_unit) for p in hourly_wire.periods
    ]

    daily_points = _daynight_periods_to_daily(daynight_wire.periods, target_unit)

    # Runtime detection of paid-tier summary (Q2)
    first_period_raw: dict[str, Any] | None = None
    if daynight_raw.get("periods") and isinstance(daynight_raw["periods"], list):
        raw_periods = daynight_raw["periods"]
        first_period_raw = raw_periods[0] if raw_periods else None

    discussion = _extract_aeris_discussion(
        daynight_raw=daynight_raw,
        first_period_raw=first_period_raw,
        provider_id=PROVIDER_ID,
        domain=DOMAIN,
    )

    return ForecastBundle(
        hourly=hourly_points,
        daily=daily_points,
        discussion=discussion,
        source=PROVIDER_ID,
        generatedAt=utc_isoformat(datetime.now(tz=UTC)),
    )


# ---------------------------------------------------------------------------
# Internal fetch helpers — one per outbound call
# ---------------------------------------------------------------------------


def _fetch_hourly(
    client: ProviderHTTPClient,
    lat: float,
    lon: float,
    client_id: str,
    client_secret: str,
) -> _AerisHourlyResponse:
    """GET /forecasts/{lat},{lon}?filter=1hr and validate wire shape.

    Raises:
        KeyInvalid: HTTP 401 (invalid credentials).
        QuotaExhausted: HTTP 429 (rate limit exceeded).
        ProviderProtocolError: HTTP 200 with success=false, or validation failure.
        TransientNetworkError: Network failure / 5xx after retries (from ProviderHTTPClient).
    """
    location = f"{round(lat, 4)},{round(lon, 4)}"
    url = f"{AERIS_BASE_URL}{AERIS_FORECASTS_PATH}/{location}"
    params = {
        "filter": "1hr",
        "limit": str(HOURLY_LIMIT),
        "client_id": client_id,
        "client_secret": client_secret,
    }

    _rate_limiter.acquire()
    # ProviderHTTPClient.get raises canonical taxonomy exceptions (KeyInvalid,
    # QuotaExhausted, TransientNetworkError, ProviderProtocolError) with all
    # structured attributes set (status_code, retry_after_seconds). Let them
    # propagate; do NOT re-wrap (3b-4 audit F1/F2: re-construction dropped
    # retry_after_seconds from QuotaExhausted, and `except Exception` violates
    # rules/coding.md §3).
    response = client.get(url, params=params)

    return _parse_aeris_envelope(response, model_class=_AerisHourlyResponse, call_label="hourly")


def _fetch_daynight(
    client: ProviderHTTPClient,
    lat: float,
    lon: float,
    client_id: str,
    client_secret: str,
) -> tuple[_AerisDayNightResponse, dict[str, Any]]:
    """GET /forecasts/{lat},{lon}?filter=daynight and validate wire shape.

    Returns:
        Tuple of (validated _AerisDayNightResponse, raw response[0] dict).
        The raw dict is used for paid-tier summary detection (brief Q2).

    Raises: same taxonomy as _fetch_hourly.
    """
    location = f"{round(lat, 4)},{round(lon, 4)}"
    url = f"{AERIS_BASE_URL}{AERIS_FORECASTS_PATH}/{location}"
    params = {
        "filter": "daynight",
        "limit": str(DAYNIGHT_LIMIT),
        "client_id": client_id,
        "client_secret": client_secret,
    }

    _rate_limiter.acquire()
    # ProviderHTTPClient.get raises canonical taxonomy exceptions; let them
    # propagate (3b-4 audit F1/F2 — see _fetch_hourly).
    response = client.get(url, params=params)

    raw_response_list = _parse_aeris_envelope_raw(response, call_label="daynight")
    raw_first = raw_response_list[0] if raw_response_list else {}

    try:
        validated = _AerisDayNightResponse.model_validate(raw_first)
    except ValidationError as exc:
        logger.error(
            "Aeris daynight response[0] validation failed: %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
        )
        raise ProviderProtocolError(
            f"Aeris daynight response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    return validated, raw_first


# ---------------------------------------------------------------------------
# Envelope parsing helpers
# ---------------------------------------------------------------------------


def _parse_aeris_envelope_raw(response: Any, *, call_label: str) -> list[dict[str, Any]]:
    """Parse the Aeris success/error envelope and return the raw response list.

    On success=false: raises ProviderProtocolError.
    On success=true with warn_location: logs WARNING and returns empty list
      (caller returns empty bundle per brief lead-call 17).
    On success=true with response=[]: returns empty list.

    Args:
        response: httpx.Response from ProviderHTTPClient.get().
        call_label: "hourly" or "daynight" for error context.

    Raises:
        ProviderProtocolError: success=false or envelope parse failure.
    """
    try:
        envelope = _AerisEnvelope.model_validate(response.json())
    except (ValidationError, ValueError) as exc:
        logger.error(
            "Aeris %s envelope parse failed: %s. Body (first 2000 chars): %.2000s",
            call_label, exc, response.text,
        )
        raise ProviderProtocolError(
            f"Aeris {call_label} envelope parse failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    if not envelope.success:
        error_code = ""
        if envelope.error:
            error_code = envelope.error.get("code", "")
            error_desc = envelope.error.get("description", "")
        else:
            error_desc = "unknown error"
        raise ProviderProtocolError(
            f"Aeris {call_label} returned success=false: code={error_code!r} {error_desc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    # success=true with a warning (e.g. warn_location) — log and continue
    if envelope.error:
        warn_code = envelope.error.get("code", "")
        warn_desc = envelope.error.get("description", "")
        logger.warning(
            "Aeris %s returned success=true with warning: code=%r %s",
            call_label, warn_code, warn_desc,
        )

    return envelope.response


def _parse_aeris_envelope(
    response: Any,
    *,
    model_class: type,
    call_label: str,
) -> Any:
    """Parse envelope and validate response[0] against model_class.

    Used for the hourly call where we don't need the raw dict.
    Returns a validated Pydantic model instance.
    """
    raw_list = _parse_aeris_envelope_raw(response, call_label=call_label)
    raw_first = raw_list[0] if raw_list else {}

    try:
        return model_class.model_validate(raw_first)
    except ValidationError as exc:
        logger.error(
            "Aeris %s response[0] validation failed: %s. "
            "Response body (first 2000 chars): %.2000s",
            call_label, exc, response.text,
        )
        raise ProviderProtocolError(
            f"Aeris {call_label} response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc


# ---------------------------------------------------------------------------
# Public fetch entrypoint (ADR-038 §2)
# ---------------------------------------------------------------------------


def fetch(
    *,
    lat: float,
    lon: float,
    target_unit: str,
    client_id: str | None,
    client_secret: str | None,
) -> ForecastBundle:
    """Call Aeris /forecasts (1hr + daynight) and return canonical ForecastBundle.

    Two outbound calls per cache miss: filter=1hr for hourly, filter=daynight
    for paired day/night periods. Both results are normalised and cached as a
    single ForecastBundle.

    Cache-first: check cache before making outbound HTTP calls.
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
        client_id: Aeris client_id from env var WEEWX_CLEARSKIES_AERIS_CLIENT_ID.
            None if operator hasn't configured it.
        client_secret: Aeris client_secret from env var
            WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET. None if not configured.

    Returns:
        ForecastBundle — single canonical Pydantic model.
        discussion is None for free-tier; populated for paid-tier when
        summary field is detected (brief Q2).

    Raises:
        KeyInvalid: Credentials missing (both args None), or Aeris returned 401.
        QuotaExhausted: Aeris returned 429.
        ProviderProtocolError: target_unit unknown, response validation failed,
            or Aeris returned success=false envelope.
        TransientNetworkError: Network/DNS failure or 5xx after retries.
    """
    # Validate credentials before making any outbound call (brief lead-call 12).
    # Loud failure beats silent disable — operator intent is unambiguous.
    if not client_id or not client_secret:
        raise KeyInvalid(
            "Aeris credentials missing — set WEEWX_CLEARSKIES_AERIS_CLIENT_ID "
            "and WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET env vars",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    cache_key = _build_cache_key(lat, lon, target_unit)
    cached = get_cache().get(cache_key)
    if cached is not None:
        logger.debug(
            "Cache hit for Aeris forecast",
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        return ForecastBundle.model_validate(cached)

    logger.debug(
        "Cache miss for Aeris forecast; calling API (two calls: 1hr + daynight)",
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )

    if target_unit not in {"US", "METRIC", "METRICWX"}:
        # Defensive: services/units.py validates at startup; should not fire.
        raise ProviderProtocolError(
            f"Unknown target_unit {target_unit!r}; expected US, METRIC, or METRICWX",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    client = _client_for()

    # Call 1: hourly periods
    hourly_wire = _fetch_hourly(client, lat, lon, client_id, client_secret)

    # Call 2: daynight periods + raw dict for discussion detection
    daynight_wire, daynight_raw = _fetch_daynight(client, lat, lon, client_id, client_secret)

    bundle = _to_canonical(
        hourly_wire,
        daynight_wire,
        target_unit=target_unit,
        daynight_raw=daynight_raw,
    )

    get_cache().set(
        cache_key,
        bundle.model_dump(mode="json"),
        ttl_seconds=DEFAULT_FORECAST_TTL_SECONDS,
    )

    logger.info(
        "Aeris forecast fetched: %d hourly, %d daily point(s)",
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


def _reset_http_client_for_tests() -> None:
    """Reset module-level HTTP client singleton.  Used in tests only."""
    global _http_client  # noqa: PLW0603
    _http_client = None
