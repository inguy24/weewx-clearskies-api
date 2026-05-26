"""NWS forecast provider module (ADR-007, ADR-038).

Five responsibilities per ADR-038 §2:
  1. Outbound API calls — NWS two-step /points → /forecast/hourly + /forecast,
     plus /products?type=AFD + /products/{id} for the Area Forecast Discussion.
  2. Response parsing — wire-shape Pydantic models for five NWS response types.
  3. Translation to canonical ForecastBundle (HourlyForecastPoint +
     DailyForecastPoint + ForecastDiscussion).
  4. Capability declaration — CAPABILITY symbol consumed at startup.
  5. Error handling — provider errors translated to canonical taxonomy.

Five outbound calls per cache miss (call 10 from brief):
  1. GET /points/{lat},{lon}  → cwa, gridId, gridX, gridY, timeZone
  2. GET /gridpoints/{cwa}/{gridX},{gridY}/forecast/hourly → ~156 hourly periods
  3. GET /gridpoints/{cwa}/{gridX},{gridY}/forecast → ~14 day/night periods
  4. GET /products?type=AFD&location={cwa} → list of recent AFD products
  5. GET /products/{id} → AFD body (productText, issuanceTime, etc.)

NWS User-Agent (ADR-006, brief call 12):
  Operators put their own contact email/URL in api.conf:
    [forecast] nws_user_agent_contact = me@example.com
  Module composes UA as "(weewx-clearskies-api/<version>, <contact>)" when set;
  "(weewx-clearskies-api/<version>)" + one-time WARN when unset.
  NO project-level hardcoded fallback — that would put the project on the hook
  for any individual operator's traffic patterns per ADR-006.

Non-US location handling (brief call 13, ADR-007 §Per-module behavior):
  No client-side bounding-box check.  NWS /points returns 404 for non-US
  lat/lon.  ProviderHTTPClient translates 404 → ProviderProtocolError;
  this module intercepts and re-raises as GeographicallyUnsupported → 503.
  ADR-007: "USA-only check at config time" honored at runtime via this 503.

Unit handling (ADR-019, brief call 11):
  NWS forecast endpoints accept units=us|si query param.
  Mapping: US → us, METRIC → si, METRICWX → si + post-convert wind km/h → m/s.
  METRICWX wind post-conversion at the canonical-translation step (÷ 3.6).
  Documented here because it's a non-obvious choice: NWS offers no m/s option;
  the convert-at-ingest pattern (ADR-019) means we get km/h from NWS and
  convert before storing in the canonical model.

AFD soft-failure (brief call 14):
  AFD (calls 4-5) is supplementary — the forecast points are load-bearing.
  Any AFD failure (empty list, transient error, parse error) logs WARN and
  returns bundle with discussion=None.  Hourly/daily failures still raise.

Cache layer (ADR-017):
  Caches the post-normalization ForecastBundle for 30 min.
  Key: SHA-256 of (provider_id, endpoint="forecast_bundle", {lat4, lon4,
  target_unit}).  "forecast_bundle" is a deliberate logical key covering all
  five upstream calls — not a URL path.  Single cache entry per (station,
  target_unit); /points result is internal scaffolding, not a separate entry.

Day/night period pairing (brief call 18):
  NWS /forecast returns alternating isDaytime=True/False periods.  Module pairs
  each day-period with the immediately-following night-period to form one
  canonical DailyForecastPoint.  If the first period is a night period (e.g.,
  forecast generated late evening), skip to the first day-period before pairing.

windSpeed string parsing (brief call 19):
  /forecast returns ranges like "5 to 10 mph"; /forecast/hourly returns single
  values like "7 mph".  Upper bound taken in both cases per canonical §4.1.3.

weatherCode extraction (brief call 15):
  NWS icon URL segment → shortName extracted pre-comma (intensity strip).
  Tolerate unknown shortNames — pass through as-is; log DEBUG on first encounter.

precipType derivation (brief call 21, ADR-010 §3.3):
  NWS icon shortName → canonical enum: rain/snow/freezing-rain/sleet/null.
  Mixed-precip shortNames (mix, rain_snow, rain_showers_snow) → "rain" with
  DEBUG log; no canonical mixed-precip value exists (forward compatibility).

Rate limiter (brief call 25, ADR-038 §3):
  Separate per-module limiter (not shared with alerts/nws.py) to avoid coupling
  the two domain quotas.  Same parameters: max_calls=5, window_seconds=1.

Wire-shape Pydantic (security-baseline §3.5, ADR-038):
  Five wire-shape models validated against real recorded fixtures.
  extra="ignore" so NWS schema additions don't break us; missing required
  fields raise ValidationError → ProviderProtocolError.

ruff: noqa: N815  (field names match NWS wire camelCase: isDaytime, gridX, etc.)
"""

# ruff: noqa: N815

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.parse
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from weewx_clearskies_api.models.responses import (
    DailyForecastPoint,
    ForecastBundle,
    ForecastDiscussion,
    HourlyForecastPoint,
    ProviderConditions,
    utc_isoformat,
)
from weewx_clearskies_api.providers._common.cache import get_cache
from weewx_clearskies_api.providers._common.capability import ProviderCapability
from weewx_clearskies_api.providers._common.datetime_utils import to_utc_iso8601_from_offset
from weewx_clearskies_api.providers._common.errors import (
    GeographicallyUnsupported,
    ProviderProtocolError,
    QuotaExhausted,
    TransientNetworkError,
)
from weewx_clearskies_api.providers._common.http import ProviderHTTPClient
from weewx_clearskies_api.providers._common.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PROVIDER_ID = "nws"
DOMAIN = "forecast"
NWS_BASE_URL = "https://api.weather.gov"
NWS_POINTS_PATH = "/points"
NWS_PRODUCTS_PATH = "/products"
DEFAULT_FORECAST_TTL_SECONDS = 1800   # 30 min per ADR-017
DEFAULT_CONDITIONS_TTL_SECONDS = 300  # 5 min per brief
DEFAULT_STATION_LIST_TTL_SECONDS = 3600  # 1 h — station list rarely changes

_API_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Capability declaration (ADR-038 §4)
# ---------------------------------------------------------------------------

CAPABILITY = ProviderCapability(
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    supplied_canonical_fields=(
        # HourlyForecastPoint — what NWS /forecast/hourly actually supplies
        "validTime",
        "outTemp",
        "windSpeed",
        "windDir",
        "precipProbability",
        "precipType",
        "weatherCode",
        "weatherText",
        # DailyForecastPoint — paired day/night periods
        "validDate",
        "tempMax",
        "tempMin",
        "precipProbabilityMax",
        "windSpeedMax",
        "weatherCode",
        "weatherText",
        "narrative",
        # ForecastDiscussion — first end-to-end consumer
        "headline",
        "body",
        "issuedAt",
        "senderName",
        # NB: outHumidity, windGust, precipAmount, cloudCover (hourly),
        # precipAmount/windGustMax/sunrise/sunset/uvIndexMax (daily) require
        # the raw /gridpoints endpoint — out of scope this round.
    ),
    geographic_coverage="us",  # USA + territories
    auth_required=(),  # no key; UA contact recommended via [forecast] section
    default_poll_interval_seconds=DEFAULT_FORECAST_TTL_SECONDS,
    operator_notes=(
        "NWS forecast: USA-only coverage. Set [forecast] "
        "nws_user_agent_contact in api.conf for best results "
        "(reduces block risk during NWS security events). "
        "Hourly outHumidity / windGust / precipAmount / cloudCover and "
        "daily windGustMax / sunrise / sunset / uvIndexMax are not "
        "supplied via the standard forecast endpoints; see "
        "https://weather.gov for raw gridpoint data."
    ),
)

# ---------------------------------------------------------------------------
# Per-unit NWS query-param mapping (ADR-019, brief call 11)
# NWS accepts units=us|si; METRICWX → si + post-convert wind km/h → m/s.
# ---------------------------------------------------------------------------

_TARGET_UNIT_TO_NWS_UNITS: dict[str, str] = {
    "US": "us",
    "METRIC": "si",
    "METRICWX": "si",  # post-convert wind km/h → m/s in _zip_hourly/_zip_daily
}

# ---------------------------------------------------------------------------
# Compass abbreviation → degrees lookup (brief call 20)
# Standard 16-point compass table.
# ---------------------------------------------------------------------------

_COMPASS_TO_DEGREES: dict[str, float] = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}

# ---------------------------------------------------------------------------
# Icon shortName → precipType lookup (brief call 21, canonical §3.3)
# Canonical enum: "rain" | "snow" | "freezing-rain" | "sleet"
# ---------------------------------------------------------------------------

_ICON_TO_PRECIP_TYPE: dict[str, str] = {
    # Rain family
    "rain": "rain",
    "rain_showers": "rain",
    "rain_showers_hi": "rain",
    # Snow family
    "snow": "snow",
    "snow_showers": "snow",
    "blizzard": "snow",
    # Freezing precipitation
    "fzra": "freezing-rain",
    "rain_fzra": "freezing-rain",
    "snow_fzra": "freezing-rain",
    # Sleet
    "sleet": "sleet",
    "rain_sleet": "sleet",
    "snow_sleet": "sleet",
    # Thunderstorms → rain (canonical has no "thunderstorm" enum value)
    "tsra": "rain",
    "tsra_sct": "rain",
    "tsra_hi": "rain",
    # Mixed precip → rain (canonical has no mixed-precip enum; DEBUG logged)
    "mix": "rain",
    "rain_snow": "rain",
    "rain_showers_snow": "rain",
}

# Track icon shortNames we've already logged about at DEBUG (avoid log spam).
_unknown_icon_shortnames_logged: set[str] = set()
_mixed_precip_shortnames_logged: set[str] = set()

# ---------------------------------------------------------------------------
# Wire-shape Pydantic models (security-baseline §3.5)
# Source: docs/reference/api-docs/nws.md + recorded fixtures
# ---------------------------------------------------------------------------


class _NwsPointProperties(BaseModel):
    """NWS /points/{lat,lon} feature properties — wire shape."""

    model_config = ConfigDict(extra="ignore")

    cwa: str
    gridId: str
    gridX: int
    gridY: int
    forecast: str                         # URL for 12-hour day/night periods
    forecastHourly: str                   # URL for hourly periods
    timeZone: str                         # IANA TZ for station (e.g. "America/Los_Angeles")
    observationStations: str | None = None  # URL for nearest METAR station list
    radarStation: str | None = None         # not load-bearing


class _NwsPointResponse(BaseModel):
    """NWS /points/{lat,lon} GeoJSON Feature envelope."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["Feature"]
    properties: _NwsPointProperties


class _NwsForecastPeriod(BaseModel):
    """One period from /forecast (12-hour) or /forecast/hourly.

    /forecast (12-hour): windSpeed is "5 to 10 mph" range string.
    /forecast/hourly: windSpeed is "7 mph" single-value string.
    detailedForecast is only populated by /forecast, not /forecast/hourly.
    """

    model_config = ConfigDict(extra="ignore")

    number: int
    name: str | None = None
    startTime: str
    endTime: str
    isDaytime: bool
    temperature: float | None = None
    temperatureUnit: str  # "F" or "C"
    temperatureTrend: str | None = None
    probabilityOfPrecipitation: dict[str, Any] | None = None  # {unitCode, value}
    windSpeed: str | None = None    # range string (/forecast) or single (/hourly)
    windDirection: str | None = None  # compass abbreviation e.g. "NW"
    icon: str | None = None           # URL — extract shortName segment
    shortForecast: str | None = None
    detailedForecast: str | None = None


class _NwsForecastProperties(BaseModel):
    """Properties block of NWS GeoJSON forecast feature."""

    model_config = ConfigDict(extra="ignore")

    updated: str | None = None
    units: str | None = None           # "us" or "si"
    forecastGenerator: str | None = None
    generatedAt: str | None = None
    updateTime: str | None = None
    validTimes: str | None = None
    periods: list[_NwsForecastPeriod] = Field(default_factory=list)


class _NwsForecastResponse(BaseModel):
    """NWS /gridpoints/{cwa}/{x},{y}/forecast[/hourly] GeoJSON Feature envelope."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["Feature"]
    properties: _NwsForecastProperties


class _NwsAfdProductSummary(BaseModel):
    """One entry from /products?type=AFD&location={cwa} @graph list."""

    model_config = ConfigDict(extra="ignore")

    id: str
    wmoCollectiveId: str | None = None
    issuingOffice: str | None = None
    issuanceTime: str | None = None
    productCode: str | None = None
    productName: str | None = None


class _NwsAfdListResponse(BaseModel):
    """NWS /products?type=AFD&location={cwa} response envelope.

    NWS returns @graph for product list endpoints.
    """

    model_config = ConfigDict(extra="ignore")

    # @graph is the field name in the JSON; alias maps it.
    graph: list[_NwsAfdProductSummary] = Field(
        default_factory=list, alias="@graph"
    )


class _NwsAfdProductBody(BaseModel):
    """Body of /products/{id} — full AFD content.

    productText: plain ASCII AFD body (no DWML/CAP parsing needed).
    issuanceTime: ISO-8601 with offset → converted to UTC.
    wmoCollectiveId: used as headline when populated (e.g. "FXUS66").
    issuingOffice: sender name (e.g. "KSEW").
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    wmoCollectiveId: str | None = None
    issuingOffice: str | None = None
    issuanceTime: str
    productCode: str | None = None
    productName: str | None = None
    productText: str


class _NwsObsValue(BaseModel):
    """One quantitative value from an NWS observation (SI units, may be null)."""

    model_config = ConfigDict(extra="ignore")

    value: float | None = None
    unitCode: str | None = None


class _NwsObservationProperties(BaseModel):
    """Properties of a single NWS METAR observation."""

    model_config = ConfigDict(extra="ignore")

    textDescription: str | None = None
    temperature: _NwsObsValue | None = None
    windSpeed: _NwsObsValue | None = None
    windDirection: _NwsObsValue | None = None
    windGust: _NwsObsValue | None = None
    barometricPressure: _NwsObsValue | None = None
    relativeHumidity: _NwsObsValue | None = None
    icon: str | None = None


class _NwsObservationResponse(BaseModel):
    """NWS /stations/{stationId}/observations/latest GeoJSON Feature envelope."""

    model_config = ConfigDict(extra="ignore")

    properties: _NwsObservationProperties


# ---------------------------------------------------------------------------
# Rate limiter (ADR-038 §3, brief call 25)
# Separate from alerts/nws.py rate limiter — shared limiter would couple
# alerts and forecast quotas, penalizing alerts when forecast bursts.
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="nws-forecast",
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    max_calls=5,
    window_seconds=1,
)

# ---------------------------------------------------------------------------
# HTTP client (module-level singleton — one per module, not per request)
# Constructed lazily; keyed by user_agent string so UA change gets fresh client.
# ---------------------------------------------------------------------------

_http_client: ProviderHTTPClient | None = None
_http_client_ua: str = ""


def _get_http_client(user_agent: str) -> ProviderHTTPClient:
    """Return the module-level HTTP client, (re-)constructing if UA changed."""
    global _http_client, _http_client_ua  # noqa: PLW0603
    if _http_client is None or _http_client_ua != user_agent:
        _http_client = ProviderHTTPClient(
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
            user_agent=user_agent,
        )
        _http_client_ua = user_agent
    return _http_client


# ---------------------------------------------------------------------------
# User-Agent construction (ADR-006, brief call 12)
# ---------------------------------------------------------------------------

_warned_missing_contact = False


def _build_user_agent(contact: str | None) -> str:
    """Build the NWS User-Agent string per ADR-006.

    Contact should be an operator email or URL from api.conf
    [forecast] nws_user_agent_contact.  When unset, a WARN is logged once
    and the contact is omitted.

    NO project-level hardcoded fallback — would put the project on the hook
    for operator traffic patterns (ADR-006).
    """
    base = f"weewx-clearskies-api/{_API_VERSION}"
    if contact and contact.strip():
        return f"({base}, {contact.strip()})"
    return f"({base})"


def _warn_once_missing_contact() -> None:
    """Emit a one-time WARN when nws_user_agent_contact is not configured."""
    global _warned_missing_contact  # noqa: PLW0603
    if not _warned_missing_contact:
        logger.warning(
            "NWS forecast User-Agent contact is not set. "
            "Set [forecast] nws_user_agent_contact = <email-or-url> in api.conf "
            "to reduce the risk of being blocked during NWS security events. "
            "See ADR-006 for the operator-managed compliance model."
        )
        _warned_missing_contact = True


# ---------------------------------------------------------------------------
# Cache key construction (ADR-017 §Cache key, brief call 24)
# ---------------------------------------------------------------------------


def _build_cache_key(lat: float, lon: float, target_unit: str) -> str:
    """Build a deterministic cache key for NWS forecast bundle.

    "forecast_bundle" is a deliberate logical key covering all five upstream
    calls — it's not a URL path.  The cache stores the post-normalization
    ForecastBundle, so the key reflects the bundle's identity, not any single
    underlying URL.  This mirrors openmeteo's per-endpoint key pattern but
    uses a logical name since NWS has multiple endpoints per bundle.
    """
    payload = json.dumps(
        {
            "provider_id": PROVIDER_ID,
            "endpoint": "forecast_bundle",
            "params": {
                "lat4": str(round(lat, 4)),
                "lon4": str(round(lon, 4)),
                "target_unit": target_unit,
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Helper functions — pure compute, no I/O
# ---------------------------------------------------------------------------


def _extract_icon_shortname(icon_url: str | None) -> str | None:
    """Extract the NWS icon shortName from an icon URL.

    NWS icon URLs are of the form:
      /icons/land/day/sct,30?size=medium
      /icons/land/night/rain?size=medium

    Algorithm per brief call 15:
      1. Strip the query string.
      2. Take the basename (last path segment after the final '/').
      3. Split on comma and take [0] (strips intensity suffix like ",30").
      4. Return the result as the shortName.

    Returns None for None input, empty strings, or unparseable URLs
    (e.g., no path segments, malformed URL).

    Pure function; tested in isolation.
    """
    if not icon_url:
        return None
    try:
        # Strip query string via urlparse.
        parsed = urllib.parse.urlparse(icon_url)
        path = parsed.path  # e.g. "/icons/land/day/sct,30"
        if not path:
            return None
        basename = path.rstrip("/").rsplit("/", 1)[-1]
        if not basename:
            return None
        # Strip intensity comma-suffix: "sct,30" → "sct".
        shortname = basename.split(",", 1)[0]
        return shortname if shortname else None
    except (AttributeError, TypeError, ValueError):
        # AttributeError: non-string passed despite type hint
        # TypeError: urllib internals on unexpected types
        # ValueError: urlparse on truly degenerate inputs
        return None


def _compass_to_degrees(s: str | None) -> float | None:
    """Convert a 16-point compass abbreviation to degrees (0–360).

    Standard table per brief call 20.  Unknown/empty/None → None with DEBUG log.
    NWS sometimes emits null directly (low-wind conditions); caller passes None
    and this function returns None without parsing.

    Pure function.
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    degrees = _COMPASS_TO_DEGREES.get(s)
    if degrees is None:
        logger.debug(
            "Unknown NWS wind direction abbreviation %r; returning None",
            s,
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
    return degrees


def _parse_wind_speed(s: str | None) -> float | None:
    """Parse an NWS windSpeed string to a float (upper bound of range).

    NWS /forecast returns ranges like "5 to 10 mph".
    NWS /forecast/hourly returns single values like "7 mph".

    Algorithm per brief call 19:
      1. Strip the unit suffix (" mph" or " km/h").
      2. Split on " to "; take the last element (upper bound for ranges,
         only element for single values).
      3. Parse to float.
      4. Empty / None / unparseable → None (log at DEBUG with input).

    Upper bound is taken per canonical §4.1.3 note.

    Pure function.
    """
    if not s:
        return None
    try:
        # Strip unit suffix.
        value_str = s.strip()
        for unit in (" mph", " km/h"):
            if value_str.endswith(unit):
                value_str = value_str[: -len(unit)].strip()
                break
        # Split on " to " and take the last element.
        parts = value_str.split(" to ")
        upper = parts[-1].strip()
        if not upper:
            raise ValueError("empty after split")
        return float(upper)
    except (ValueError, IndexError):
        logger.debug(
            "Could not parse NWS windSpeed %r; returning None",
            s,
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        return None


def _get_precip_probability(period: _NwsForecastPeriod) -> float | None:
    """Extract precipProbability from the period's probabilityOfPrecipitation dict.

    NWS returns {unitCode: "...", value: <float|null>}.  Treat null as 0 per
    brief call 18 (precipProbabilityMax = max across day/night, null as 0).
    """
    if period.probabilityOfPrecipitation is None:
        return None
    val = period.probabilityOfPrecipitation.get("value")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _get_precip_type_from_icon(icon_url: str | None) -> str | None:
    """Derive precipType from NWS icon shortName per brief call 21.

    Logs DEBUG on first encounter of mixed-precip shortNames (for future
    canonical model amendment tracking) and unknown shortNames.
    """
    shortname = _extract_icon_shortname(icon_url)
    if shortname is None:
        return None
    precip_type = _ICON_TO_PRECIP_TYPE.get(shortname)
    if precip_type is None:
        # Not a precip-producing shortName (e.g., "sct", "bkn", "fog").
        # Log unknown shortNames once for awareness.
        non_precip = {
            "few", "sct", "bkn", "ovc", "skc", "wind", "fog",
            "hot", "dust", "smoke", "tornado", "hurricane", "tropical_storm",
        }
        if shortname not in non_precip and shortname not in _unknown_icon_shortnames_logged:
            logger.debug(
                "Unknown NWS icon shortName %r; precipType → None. "
                "If this is a new NWS shortName, consider extending _ICON_TO_PRECIP_TYPE.",
                shortname,
                extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
            )
            _unknown_icon_shortnames_logged.add(shortname)
    elif shortname in ("mix", "rain_snow", "rain_showers_snow"):
        # Mixed precip → "rain" (no canonical mixed-precip enum value).
        # Log DEBUG once so prevalence can inform future canonical model changes.
        if shortname not in _mixed_precip_shortnames_logged:
            logger.debug(
                "NWS icon shortName %r is mixed-precip; canonical has no "
                "mixed-precip enum — mapping to 'rain'. "
                "If this appears frequently, consider a future canonical model amendment.",
                shortname,
                extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
            )
            _mixed_precip_shortnames_logged.add(shortname)
    return precip_type


# ---------------------------------------------------------------------------
# AFD body parsing (canonical §4.1.4 NWS column)
# ---------------------------------------------------------------------------

# WMO header lines to skip when extracting the AFD body's first line.
# Real AFD productText starts:
#   000\n                          ← WMO sentinel
#   FXUS66 KSEW 080340\n           ← WMO ID + station + DDDDZZ datetime
#   AFDSEW\n                       ← product code
#   \n
#   Area Forecast Discussion\n     ← FIRST CONTENT LINE — the AFD title
#   National Weather Service Seattle WA\n
#   840 PM PDT Thu May 7 2026\n
#   ...
_AFD_WMO_SENTINEL_RE = re.compile(r"^\d{3}$")
_AFD_WMO_ID_RE = re.compile(r"^[A-Z]{3,4}\d{2}\s+[A-Z]{4}")  # "FXUS66 KSEW ..."
_AFD_PRODUCT_CODE_RE = re.compile(r"^AFD[A-Z]{3,4}$")        # "AFDSEW"
# The AFD body header line that names the sender — line is typically
# "National Weather Service Seattle WA" or "National Weather Service Boston/Norton MA".
_AFD_SENDER_PREFIX = "National Weather Service "


def _extract_afd_headline_and_sender(
    product_text: str,
) -> tuple[str | None, str | None]:
    """Extract canonical headline + senderName from AFD productText.

    Per canonical-data-model §4.1.4 NWS column:
      headline    = productText first line  (after WMO wire header)
      senderName  = wmoCollectiveId + issuingOffice composite,
                    e.g. "NWS Seattle WA" (from the AFD body's
                    "National Weather Service [Location]" line abbreviated).

    AFD wire format starts with:
      "000\\n"                       (WMO sentinel)
      "FXUS66 KSEW DDDDZZ ...\\n"    (WMO collective ID + originating station + time)
      "AFDSEW\\n"                    (product code: AFD + 3-4 letter office suffix)
      "\\n"                          (blank)
      "Area Forecast Discussion\\n"  ← first real content line
      "National Weather Service Seattle WA\\n"
      "[Issuance time line]\\n"
      ".SYNOPSIS...\\n"
      ...

    This helper:
      - Skips leading blanks, the WMO sentinel, the WMO ID line, and the
        AFD product code line.
      - Returns the first remaining non-blank line as headline.
      - Scans the body for a line starting with "National Weather Service "
        and abbreviates it to "NWS [Location]" for senderName.

    Both fields are None if extraction fails (malformed AFD body).
    Caller falls back to wmoCollectiveId/issuingOffice in that case.

    Pure function; tested in isolation.
    """
    if not product_text:
        return None, None

    headline: str | None = None
    sender: str | None = None

    for raw_line in product_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if headline is None:
            # Still skipping wire-format header.  Skip sentinel, WMO ID, product code.
            if _AFD_WMO_SENTINEL_RE.match(line):
                continue
            if _AFD_WMO_ID_RE.match(line):
                continue
            if _AFD_PRODUCT_CODE_RE.match(line):
                continue
            # First non-header content line.
            headline = line
        if sender is None and line.startswith(_AFD_SENDER_PREFIX):
            location = line[len(_AFD_SENDER_PREFIX):].strip()
            sender = f"NWS {location}" if location else "NWS"
        if headline is not None and sender is not None:
            break

    return headline, sender


# ---------------------------------------------------------------------------
# Hourly zip helper (canonical §4.1.2)
# ---------------------------------------------------------------------------


def _zip_hourly(
    periods: list[_NwsForecastPeriod],
    *,
    target_unit: str,
) -> list[HourlyForecastPoint]:
    """Convert NWS hourly periods to canonical HourlyForecastPoint records.

    validTime: ISO-8601 with offset → UTC via shared datetime_utils helper.
    weatherCode: icon shortName extracted via _extract_icon_shortname.
    weatherText: period.shortForecast directly.
    precipType: icon shortName → canonical enum via _get_precip_type_from_icon.
    windSpeed: parsed from string via _parse_wind_speed.
    windDir: compass abbrev → degrees via _compass_to_degrees.
    outTemp: period.temperature directly (NWS returns numeric for /hourly).
    precipProbability: from probabilityOfPrecipitation.value (0–100).

    METRICWX wind post-convert: NWS returns km/h for units=si; convert to m/s
    by dividing by 3.6 (ADR-019).  NWS offers no native m/s option.
    """
    points: list[HourlyForecastPoint] = []
    for period in periods:
        valid_time = to_utc_iso8601_from_offset(
            period.startTime, provider_id=PROVIDER_ID, domain=DOMAIN
        )
        shortname = _extract_icon_shortname(period.icon)
        weather_code = shortname  # pass through as-is per brief call 15

        wind_speed = _parse_wind_speed(period.windSpeed)
        if wind_speed is not None and target_unit == "METRICWX":
            # NWS returns km/h for units=si; convert to m/s (ADR-019).
            wind_speed = round(wind_speed / 3.6, 4)

        points.append(
            HourlyForecastPoint(
                validTime=valid_time,
                outTemp=period.temperature,
                windSpeed=wind_speed,
                windDir=_compass_to_degrees(period.windDirection),
                precipProbability=_get_precip_probability(period),
                precipType=_get_precip_type_from_icon(period.icon),
                weatherCode=weather_code,
                weatherText=period.shortForecast,
                source=PROVIDER_ID,
            )
        )
    return points


# ---------------------------------------------------------------------------
# Day/night pairing helpers (canonical §4.1.3, brief call 18)
# ---------------------------------------------------------------------------


def _pair_day_night(
    periods: list[_NwsForecastPeriod],
) -> list[tuple[_NwsForecastPeriod, _NwsForecastPeriod | None]]:
    """Pair NWS /forecast periods into (day, night) tuples.

    NWS /forecast returns alternating isDaytime=True / isDaytime=False periods.
    This function pairs each day-period with its immediately-following night-period.

    Edge case (per brief call 18): if the first period is a night period
    (forecast generated late evening), skip to the first day-period.
    A trailing day-period without a subsequent night gets paired with None
    (canonical tempMin will be None for that day).

    Returns:
        List of (day_period, night_period | None) tuples.
    """
    pairs: list[tuple[_NwsForecastPeriod, _NwsForecastPeriod | None]] = []
    i = 0
    n = len(periods)

    # Skip leading night periods.
    while i < n and not periods[i].isDaytime:
        i += 1

    while i < n:
        day = periods[i]
        if not day.isDaytime:
            # We're in sync — this shouldn't happen after the initial skip,
            # but skip any unexpected night periods to stay paired.
            i += 1
            continue

        # Look ahead for the next night period.
        night: _NwsForecastPeriod | None = None
        if i + 1 < n and not periods[i + 1].isDaytime:
            night = periods[i + 1]
            i += 2
        else:
            # Trailing day without a night.
            i += 1

        pairs.append((day, night))

    return pairs


def _zip_daily(
    pairs: list[tuple[_NwsForecastPeriod, _NwsForecastPeriod | None]],
    *,
    target_unit: str,
) -> list[DailyForecastPoint]:
    """Convert NWS day/night period pairs to canonical DailyForecastPoint records.

    Per brief call 18 mapping:
      validDate: day-period's startTime date part (YYYY-MM-DD, station-local).
      tempMax: day-period's temperature.
      tempMin: night-period's temperature (None if no night period).
      precipProbabilityMax: max across day + night (treat null as 0).
      windSpeedMax: upper bound of windSpeed string, max across day + night.
      weatherCode / weatherText / narrative: day-period's values.
    """
    points: list[DailyForecastPoint] = []
    for day, night in pairs:
        # validDate: station-local YYYY-MM-DD from day startTime (ADR-020).
        # startTime is ISO-8601 with offset; take the date part directly.
        valid_date = day.startTime[:10]  # "2026-04-30T..."[:10] → "2026-04-30"

        temp_min: float | None = night.temperature if night is not None else None

        # precipProbabilityMax: max across day + night, treat null as 0
        # (brief call 18).  Real-zero stays as 0 — canonical §4.1.3 has no
        # 0-collapses-to-null rule, and a clear-day "0%" is meaningfully
        # distinct from "data unavailable" for the dashboard consumer.
        day_prob = _get_precip_probability(day) or 0.0
        night_prob = (_get_precip_probability(night) or 0.0) if night is not None else 0.0
        precip_prob_max_val: float = max(day_prob, night_prob)

        # windSpeedMax: upper bound across day + night.
        day_wind = _parse_wind_speed(day.windSpeed)
        night_wind = _parse_wind_speed(night.windSpeed) if night is not None else None
        if day_wind is not None and night_wind is not None:
            wind_speed_max: float | None = max(day_wind, night_wind)
        elif day_wind is not None:
            wind_speed_max = day_wind
        elif night_wind is not None:
            wind_speed_max = night_wind
        else:
            wind_speed_max = None

        if wind_speed_max is not None and target_unit == "METRICWX":
            # NWS returns km/h for units=si; convert to m/s (ADR-019).
            wind_speed_max = round(wind_speed_max / 3.6, 4)

        shortname = _extract_icon_shortname(day.icon)

        points.append(
            DailyForecastPoint(
                validDate=valid_date,
                tempMax=day.temperature,
                tempMin=temp_min,
                precipProbabilityMax=precip_prob_max_val,
                windSpeedMax=wind_speed_max,
                weatherCode=shortname,
                weatherText=day.shortForecast,
                narrative=day.detailedForecast,
                source=PROVIDER_ID,
            )
        )
    return points


# ---------------------------------------------------------------------------
# Wire → canonical normalization (canonical §4.1.2–§4.1.4)
# ---------------------------------------------------------------------------


def _to_canonical(
    hourly_wire: _NwsForecastResponse,
    daily_wire: _NwsForecastResponse,
    discussion: ForecastDiscussion | None,
    *,
    target_unit: str,
) -> ForecastBundle:
    """Translate NWS wire responses to canonical ForecastBundle.

    hourly: zipped from hourly_wire.properties.periods.
    daily: paired day/night from daily_wire.properties.periods.
    discussion: ForecastDiscussion or None (soft-failure on AFD).
    source: PROVIDER_ID ("nws").
    generatedAt: current UTC timestamp.
    """
    hourly_points = _zip_hourly(
        hourly_wire.properties.periods,
        target_unit=target_unit,
    )

    pairs = _pair_day_night(daily_wire.properties.periods)
    daily_points = _zip_daily(pairs, target_unit=target_unit)

    return ForecastBundle(
        hourly=hourly_points,
        daily=daily_points,
        discussion=discussion,
        source=PROVIDER_ID,
        generatedAt=utc_isoformat(datetime.now(tz=UTC)),
    )


# ---------------------------------------------------------------------------
# Public fetch entrypoint (ADR-038 §2, brief §fetch spec)
# ---------------------------------------------------------------------------


def fetch(
    *,
    lat: float,
    lon: float,
    target_unit: str,
    user_agent_contact: str | None,
) -> ForecastBundle:
    """Call NWS forecast endpoints and return a canonical ForecastBundle.

    Five outbound calls per cache miss:
      1. /points/{lat},{lon}                             → cwa, gridX, gridY
      2. /gridpoints/{cwa}/{gridX},{gridY}/forecast/hourly → hourly periods
      3. /gridpoints/{cwa}/{gridX},{gridY}/forecast       → 12-hr day/night
      4. /products?type=AFD&location={cwa}                → AFD list
      5. /products/{id}                                   → AFD body

    Cache stores the post-normalization ForecastBundle for 30 min (ADR-017).
    Cache key: SHA-256 of (provider_id, "forecast_bundle", lat4, lon4, unit).
    "forecast_bundle" is a logical key covering all five calls (brief call 24).

    Non-US location handling (brief call 13, ADR-007):
      NWS /points returns 404 for non-US lat/lon.  ProviderHTTPClient
      translates 404 → ProviderProtocolError.  This module catches that
      exception when making the /points call and re-raises as
      GeographicallyUnsupported.  No client-side bounding box.

    METRICWX wind (ADR-019, brief call 11):
      NWS offers no native m/s — request units=si (km/h) and post-convert
      in _zip_hourly/_zip_daily by dividing by 3.6.

    AFD soft-failure (brief call 14):
      AFD calls (4-5) soft-fail on any error → discussion=None.
      Hourly/daily failures propagate through canonical taxonomy.

    Returns:
        ForecastBundle — single canonical Pydantic model.

    Raises:
        GeographicallyUnsupported: /points returned 404 (non-US lat/lon).
        QuotaExhausted: NWS returned 429.
        TransientNetworkError: Network/DNS failure or 5xx after retries.
        ProviderProtocolError: Wire-shape validation failed; or required
            forecast call (hourly/daily) returned malformed response.
    """
    if not user_agent_contact:
        _warn_once_missing_contact()

    # --- Cache lookup ---
    cache_key = _build_cache_key(lat, lon, target_unit)
    cached = get_cache().get(cache_key)
    if cached is not None:
        logger.debug(
            "Cache hit for NWS forecast bundle",
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        return ForecastBundle.model_validate(cached)

    logger.debug(
        "Cache miss for NWS forecast bundle; calling API",
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )

    # --- Validate target_unit and map to NWS units param ---
    nws_units = _TARGET_UNIT_TO_NWS_UNITS.get(target_unit)
    if nws_units is None:
        raise ProviderProtocolError(
            f"unknown target_unit {target_unit!r}; expected US, METRIC, or METRICWX",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    user_agent = _build_user_agent(user_agent_contact)
    client = _get_http_client(user_agent)

    # Rate-limiter: single acquire() per fetch() despite the 5 outbound HTTP
    # calls that follow on a cache miss.  Accepted deviation from
    # RateLimiter's "acquire before each outbound call" docstring contract:
    # NWS forecast TTL = 30 min (ADR-017), so the real call rate per station
    # is ~1 cache-miss-burst per 30 min, far below the polite-guard ceiling
    # of 5 calls/sec.  Per-call acquires would inflate the deque without
    # changing the practical guarantee.  NWS doesn't publish a real per-second
    # quota; the limiter is here as a polite floor, not a hard quota gate.
    # If a future provider with a published per-second quota lands (e.g.
    # Aeris paid plans), it should call acquire() before each outbound call
    # literally — see RateLimiter.acquire() docstring in providers/_common/
    # rate_limiter.py.
    _rate_limiter.acquire()

    # --- Step 1: /points/{lat},{lon} ---
    lat4 = round(lat, 4)
    lon4 = round(lon, 4)
    points_url = f"{NWS_BASE_URL}{NWS_POINTS_PATH}/{lat4},{lon4}"

    try:
        points_response = client.get(
            points_url,
            headers={"Accept": "application/geo+json"},
        )
    except ProviderProtocolError as exc:
        # ProviderHTTPClient raises ProviderProtocolError for all unexpected
        # 4xx responses including 404.  NWS /points 404 = non-US lat/lon
        # (brief call 13, ADR-007 §Per-module behavior NWS row).
        # Re-raise as GeographicallyUnsupported → 503 per canonical taxonomy.
        # Match against exc.status_code (structured attribute set by
        # ProviderHTTPClient) — NOT the message string, which is brittle
        # under wrapper-message refactors.  Pattern documented in
        # providers/_common/errors.py ProviderError.__init__ docstring.
        if exc.status_code == 404:
            raise GeographicallyUnsupported(
                f"NWS /points returned 404 for lat={lat4},lon={lon4} — "
                "location is outside NWS coverage (USA + territories only). "
                "Configure operator's station lat/lon to a US location. "
                "(ADR-007 §Per-module behavior, brief call 13.)",
                provider_id=PROVIDER_ID,
                domain=DOMAIN,
            ) from exc
        raise

    try:
        points_wire = _NwsPointResponse.model_validate(points_response.json())
    except (ValidationError, ValueError) as exc:
        logger.error(
            "NWS /points response validation failed: %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            points_response.text,
        )
        raise ProviderProtocolError(
            f"NWS /points response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    props = points_wire.properties
    cwa = props.cwa
    grid_x = props.gridX
    grid_y = props.gridY

    # --- Step 2: /gridpoints/{cwa}/{gridX},{gridY}/forecast/hourly ---
    hourly_url = f"{NWS_BASE_URL}/gridpoints/{cwa}/{grid_x},{grid_y}/forecast/hourly"
    hourly_response = client.get(
        hourly_url,
        params={"units": nws_units},
        headers={"Accept": "application/geo+json"},
    )
    try:
        hourly_wire = _NwsForecastResponse.model_validate(hourly_response.json())
    except (ValidationError, ValueError) as exc:
        logger.error(
            "NWS /forecast/hourly response validation failed: %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            hourly_response.text,
        )
        raise ProviderProtocolError(
            f"NWS /forecast/hourly response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    # --- Step 3: /gridpoints/{cwa}/{gridX},{gridY}/forecast ---
    daily_url = f"{NWS_BASE_URL}/gridpoints/{cwa}/{grid_x},{grid_y}/forecast"
    daily_response = client.get(
        daily_url,
        params={"units": nws_units},
        headers={"Accept": "application/geo+json"},
    )
    try:
        daily_wire = _NwsForecastResponse.model_validate(daily_response.json())
    except (ValidationError, ValueError) as exc:
        logger.error(
            "NWS /forecast response validation failed: %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            daily_response.text,
        )
        raise ProviderProtocolError(
            f"NWS /forecast response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    # --- Steps 4-5: AFD discussion (soft-failure per brief call 14) ---
    discussion: ForecastDiscussion | None = None
    try:
        afd_list_url = f"{NWS_BASE_URL}{NWS_PRODUCTS_PATH}"
        afd_list_response = client.get(
            afd_list_url,
            params={"type": "AFD", "location": cwa},
            headers={"Accept": "application/ld+json"},
        )
        try:
            afd_list_wire = _NwsAfdListResponse.model_validate(afd_list_response.json())
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "NWS AFD list response validation failed for cwa=%s: %s. "
                "Returning discussion=None (soft-failure per brief call 14).",
                cwa,
                exc,
                extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
            )
            afd_list_wire = None  # type: ignore[assignment]

        if afd_list_wire is not None and afd_list_wire.graph:
            # Take the first (most recent) AFD from the list.
            latest_afd = afd_list_wire.graph[0]
            product_id = latest_afd.id

            afd_body_url = f"{NWS_BASE_URL}{NWS_PRODUCTS_PATH}/{product_id}"
            afd_body_response = client.get(
                afd_body_url,
                headers={"Accept": "application/ld+json"},
            )
            try:
                afd_body_wire = _NwsAfdProductBody.model_validate(
                    afd_body_response.json()
                )
                # Build canonical ForecastDiscussion.
                issued_at = to_utc_iso8601_from_offset(
                    afd_body_wire.issuanceTime,
                    provider_id=PROVIDER_ID,
                    domain=DOMAIN,
                )
                # Per canonical §4.1.4 NWS column:
                #   headline    = productText first line (after WMO wire header)
                #   senderName  = "NWS [Location]" composite from body header
                # Fall back to wmoCollectiveId / issuingOffice when AFD body
                # parsing returns None (malformed or non-standard format).
                parsed_headline, parsed_sender = _extract_afd_headline_and_sender(
                    afd_body_wire.productText
                )
                headline = parsed_headline or afd_body_wire.wmoCollectiveId
                sender_name = parsed_sender or afd_body_wire.issuingOffice
                discussion = ForecastDiscussion(
                    headline=headline,
                    body=afd_body_wire.productText,
                    issuedAt=issued_at,
                    senderName=sender_name,
                    source=PROVIDER_ID,
                )
            except (ValidationError, ValueError, ProviderProtocolError) as exc:
                logger.warning(
                    "NWS AFD body parse/validation failed for product %s: %s. "
                    "Returning discussion=None (soft-failure per brief call 14).",
                    product_id,
                    exc,
                    extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
                )
        elif afd_list_wire is not None:
            # Empty graph — every CWA issues AFDs regularly so this is unusual.
            logger.warning(
                "NWS AFD list returned empty @graph for cwa=%s. "
                "Returning discussion=None (soft-failure per brief call 14).",
                cwa,
                extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
            )
    except (TransientNetworkError, QuotaExhausted, ProviderProtocolError) as exc:
        # Any provider-error on the AFD calls → soft-failure.
        # Hourly/daily are the load-bearing deliverable.
        logger.warning(
            "NWS AFD fetch failed for cwa=%s: %s. "
            "Returning discussion=None (soft-failure per brief call 14). "
            "Hourly/daily forecast data is unaffected.",
            cwa,
            exc,
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )

    # --- Build canonical bundle ---
    bundle = _to_canonical(
        hourly_wire,
        daily_wire,
        discussion,
        target_unit=target_unit,
    )

    # --- Cache the full bundle (slice happens in endpoints/forecast.py) ---
    get_cache().set(
        cache_key,
        bundle.model_dump(mode="json"),
        ttl_seconds=DEFAULT_FORECAST_TTL_SECONDS,
    )

    logger.info(
        "NWS forecast fetched: %d hourly, %d daily point(s), discussion=%s",
        len(bundle.hourly),
        len(bundle.daily),
        "present" if bundle.discussion is not None else "None",
        extra={
            "provider_id": PROVIDER_ID,
            "domain": DOMAIN,
            "lat": lat4,
            "lon": lon4,
            "target_unit": target_unit,
            "cwa": cwa,
        },
    )
    return bundle


def _build_current_conditions_cache_key(lat: float, lon: float, target_unit: str) -> str:
    """Build a deterministic cache key for NWS current-conditions data.

    Separate from the forecast bundle key so TTL and invalidation are independent.
    """
    payload = json.dumps(
        {
            "provider_id": PROVIDER_ID,
            "endpoint": "current_conditions",
            "params": {
                "lat4": str(round(lat, 4)),
                "lon4": str(round(lon, 4)),
                "target_unit": target_unit,
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _build_station_list_cache_key(lat: float, lon: float) -> str:
    """Build a deterministic cache key for the NWS observation-stations list.

    The station list is geometry-based (nearest stations to lat/lon), so target_unit
    is irrelevant.  TTL = 3600 s (1 h) — station proximity rarely changes.
    """
    payload = json.dumps(
        {
            "provider_id": PROVIDER_ID,
            "endpoint": "observation_stations",
            "params": {
                "lat4": str(round(lat, 4)),
                "lon4": str(round(lon, 4)),
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def fetch_current_conditions(
    *,
    lat: float,
    lon: float,
    target_unit: str,
    user_agent_contact: str | None,
) -> ProviderConditions | None:
    """Call NWS observation station endpoints and return ProviderConditions.

    Three outbound calls (on full cache miss):
      1. Reuse cached /points result to get observationStations URL
         (or re-fetch /points if not cached — the points call is shared
          with fetch() so its cache entry may already exist).
      2. GET <observationStations URL> — list of nearest METAR stations;
         cached separately with 3600 s TTL.
      3. GET /stations/{stationId}/observations/latest — current observation;
         cached with 300 s TTL.

    NWS observations are always SI on the wire:
      temperature in °C, windSpeed in km/h (wmoUnit:km_h-1 or variant),
      relativeHumidity in percent.

    Conversion per target_unit:
      US       → °C → °F (× 9/5 + 32), km/h → mph (÷ 1.60934)
      METRIC   → °C (no convert), km/h (no convert)
      METRICWX → °C (no convert), km/h → m/s (÷ 3.6)

    weatherText = properties.textDescription
    weatherCode = icon shortName extracted via existing _extract_icon_shortname().
    precipType  = icon shortName → canonical enum via existing
                  _get_precip_type_from_icon().

    Args:
        lat: Station latitude.
        lon: Station longitude.
        target_unit: Weewx unit system ("US" | "METRIC" | "METRICWX").
        user_agent_contact: Operator contact string for NWS User-Agent.

    Returns:
        ProviderConditions on success; None when:
          - /points lacks observationStations URL
          - station list is empty
          - latest observation has no usable data

    Raises:
        GeographicallyUnsupported: /points returned 404 (non-US lat/lon).
        QuotaExhausted: NWS returned 429.
        TransientNetworkError: Network/DNS failure or 5xx after retries.
        ProviderProtocolError: Wire-shape validation failed.
    """
    if not user_agent_contact:
        _warn_once_missing_contact()

    if target_unit not in {"US", "METRIC", "METRICWX"}:
        raise ProviderProtocolError(
            f"unknown target_unit {target_unit!r}; expected US, METRIC, or METRICWX",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    conditions_cache_key = _build_current_conditions_cache_key(lat, lon, target_unit)
    cached = get_cache().get(conditions_cache_key)
    if cached is not None:
        logger.debug(
            "Cache hit for NWS current conditions",
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        return ProviderConditions.model_validate(cached)

    logger.debug(
        "Cache miss for NWS current conditions; calling observation API",
        extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
    )

    user_agent = _build_user_agent(user_agent_contact)
    client = _get_http_client(user_agent)

    lat4 = round(lat, 4)
    lon4 = round(lon, 4)

    # --- Step 1: get observationStations URL from /points ---
    # Acquire rate limiter once for this call sequence.
    _rate_limiter.acquire()

    points_url = f"{NWS_BASE_URL}{NWS_POINTS_PATH}/{lat4},{lon4}"
    try:
        points_response = client.get(
            points_url,
            headers={"Accept": "application/geo+json"},
        )
    except ProviderProtocolError as exc:
        if exc.status_code == 404:
            raise GeographicallyUnsupported(
                f"NWS /points returned 404 for lat={lat4},lon={lon4} — "
                "location is outside NWS coverage (USA + territories only).",
                provider_id=PROVIDER_ID,
                domain=DOMAIN,
            ) from exc
        raise

    try:
        points_wire = _NwsPointResponse.model_validate(points_response.json())
    except (ValidationError, ValueError) as exc:
        logger.error(
            "NWS /points response validation failed (current conditions): %s. "
            "Response body (first 2000 chars): %.2000s",
            exc,
            points_response.text,
        )
        raise ProviderProtocolError(
            f"NWS /points response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    obs_stations_url = points_wire.properties.observationStations
    if not obs_stations_url:
        logger.warning(
            "NWS /points response missing observationStations URL for lat=%s,lon=%s; "
            "returning None for current conditions",
            lat4,
            lon4,
            extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
        )
        return None

    # --- Step 2: get observation station list (separately cached) ---
    station_list_cache_key = _build_station_list_cache_key(lat, lon)
    station_id: str | None = get_cache().get(station_list_cache_key)

    if station_id is None:
        _rate_limiter.acquire()
        stations_response = client.get(
            obs_stations_url,
            headers={"Accept": "application/geo+json"},
        )
        try:
            stations_data = stations_response.json()
            features = stations_data.get("features", [])
            if not features:
                logger.warning(
                    "NWS observation stations list empty for lat=%s,lon=%s; "
                    "returning None for current conditions",
                    lat4,
                    lon4,
                    extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
                )
                return None
            # First station is nearest; extract stationIdentifier from properties.
            first_props = features[0].get("properties", {})
            station_id = first_props.get("stationIdentifier")
            if not station_id:
                logger.warning(
                    "NWS first observation station has no stationIdentifier; "
                    "returning None for current conditions",
                    extra={"provider_id": PROVIDER_ID, "domain": DOMAIN},
                )
                return None
        except (ValueError, KeyError, TypeError) as exc:
            logger.error(
                "NWS observation stations response parse failed: %s. "
                "Response body (first 2000 chars): %.2000s",
                exc,
                stations_response.text,
            )
            raise ProviderProtocolError(
                f"NWS observation stations response parse failed: {exc}",
                provider_id=PROVIDER_ID,
                domain=DOMAIN,
            ) from exc

        get_cache().set(
            station_list_cache_key,
            station_id,
            ttl_seconds=DEFAULT_STATION_LIST_TTL_SECONDS,
        )

    # --- Step 3: GET /stations/{stationId}/observations/latest ---
    _rate_limiter.acquire()
    obs_url = f"{NWS_BASE_URL}/stations/{station_id}/observations/latest"
    obs_response = client.get(
        obs_url,
        headers={"Accept": "application/geo+json"},
    )

    try:
        obs_wire = _NwsObservationResponse.model_validate(obs_response.json())
    except (ValidationError, ValueError) as exc:
        logger.error(
            "NWS latest observation response validation failed for station %s: %s. "
            "Response body (first 2000 chars): %.2000s",
            station_id,
            exc,
            obs_response.text,
        )
        raise ProviderProtocolError(
            f"NWS latest observation response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    props = obs_wire.properties

    # NWS observations are always SI on the wire.
    # temperature in °C, windSpeed in km/h (wmoUnit:km_h-1), humidity in percent.
    temp_c: float | None = props.temperature.value if props.temperature else None
    wind_kph: float | None = props.windSpeed.value if props.windSpeed else None
    humidity: float | None = props.relativeHumidity.value if props.relativeHumidity else None
    wind_dir: float | None = props.windDirection.value if props.windDirection else None

    # Unit conversion per target_unit (ADR-019).
    if target_unit == "US":
        temperature = (temp_c * 9.0 / 5.0 + 32.0) if temp_c is not None else None
        wind_speed = (wind_kph / 1.60934) if wind_kph is not None else None
    elif target_unit == "METRICWX":
        temperature = temp_c
        wind_speed = (wind_kph / 3.6) if wind_kph is not None else None
    else:  # METRIC
        temperature = temp_c
        wind_speed = wind_kph

    conditions = ProviderConditions(
        weatherText=props.textDescription,
        weatherCode=_extract_icon_shortname(props.icon),
        precipType=_get_precip_type_from_icon(props.icon),
        cloudCover=None,   # NWS latest-observation does not supply cloud cover percent
        isDay=None,        # NWS latest-observation does not supply day/night flag
        temperature=temperature,
        humidity=humidity,
        windSpeed=wind_speed,
        windDir=wind_dir,
        source=PROVIDER_ID,
    )

    get_cache().set(
        conditions_cache_key,
        conditions.model_dump(mode="json"),
        ttl_seconds=DEFAULT_CONDITIONS_TTL_SECONDS,
    )

    logger.info(
        "NWS current conditions fetched from station %s",
        station_id,
        extra={
            "provider_id": PROVIDER_ID,
            "domain": DOMAIN,
            "lat": lat4,
            "lon": lon4,
            "target_unit": target_unit,
            "station_id": station_id,
        },
    )
    return conditions


def _reset_http_client_for_tests() -> None:
    """Reset module-level HTTP client singleton and warned-flag.  Used in tests only."""
    global _http_client, _http_client_ua, _warned_missing_contact  # noqa: PLW0603
    _http_client = None
    _http_client_ua = ""
    _warned_missing_contact = False
