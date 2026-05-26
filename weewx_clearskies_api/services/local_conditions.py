"""Local conditions blending engine (Phase 0B).

Derives a human-readable current conditions string by blending local sensor
data from the weewx archive with optional provider conditions text.

Public entry point:
    derive_conditions_text(observation, max_solar_rad, provider_conditions,
                           sun_altitude, target_unit) -> str | None

Returns a composed string like:
    "Partly Cloudy, Light Rain, Fresh Breeze and Humid"
or None when nothing can be determined.

The critical bug this module fixes: when the local rain gauge reads zero,
no precipitation text is emitted even if the forecast provider says "Patchy
Drizzle."  Precipitation text is only sourced from the local rain gauge
(or snow/sleet from the provider when wet-bulb is cold enough).

Unit system keys match weewx [StdConvert] target_unit values:
    "US"       — outTemp in °F, windSpeed in mph, rainRate in in/hr
    "METRIC"   — outTemp in °C, windSpeed in km/h, rainRate in cm/hr
    "METRICWX" — outTemp in °C, windSpeed in m/s, rainRate in mm/hr
"""

from __future__ import annotations

import logging
import math

from weewx_clearskies_api.models.responses import Observation, ProviderConditions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sky keyword extraction table (longest match checked first)
# ---------------------------------------------------------------------------

# Each tuple is (substring_to_match_in_provider_text, canonical_sky_label).
# Ordered longest-first so "Mostly Cloudy" matches before "Cloudy".
_SKY_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("Mostly Sunny", "Mostly Sunny"),
    ("Mostly Clear", "Mostly Clear"),
    ("Partly Cloudy", "Partly Cloudy"),
    ("Partly Sunny", "Partly Cloudy"),
    ("Mostly Cloudy", "Mostly Cloudy"),
    ("Overcast", "Overcast"),
    ("Cloudy", "Cloudy"),
    ("Sunny", "Clear"),
    ("Clear", "Clear"),
    ("Fair", "Clear"),
)


# ---------------------------------------------------------------------------
# Unit conversion helpers (private to this module)
# ---------------------------------------------------------------------------


def _to_fahrenheit(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return temp_c * 9.0 / 5.0 + 32.0


def _to_mph(speed: float, target_unit: str) -> float:
    """Convert windSpeed from the native unit system to mph.

    US       — already mph (no conversion needed)
    METRIC   — km/h → mph
    METRICWX — m/s  → mph
    """
    if target_unit == "US":
        return speed
    if target_unit == "METRIC":
        return speed / 1.60934
    # METRICWX
    return speed * 2.23694


def _rain_rate_to_in_per_hr(rate: float, target_unit: str) -> float:
    """Convert rainRate from the native unit system to in/hr.

    US       — already in/hr (no conversion needed)
    METRIC   — cm/hr  → in/hr  (1 in = 2.54 cm)
    METRICWX — mm/hr  → in/hr  (1 in = 25.4 mm)
    """
    if target_unit == "US":
        return rate
    if target_unit == "METRIC":
        # METRIC rainRate unit is cm/hr per units.py _SYSTEM_PRESETS
        return rate / 2.54
    # METRICWX rainRate unit is mm/hr
    return rate / 25.4


def _wet_bulb_f(temp_f: float, rh_pct: float) -> float:
    """Stull 2011 wet-bulb temperature approximation.

    Valid range: -20°C to 50°C, 5% to 99% RH.  Results outside this range
    may be imprecise but will not raise.

    Reference: Stull R.B. (2011) "Wet-Bulb Temperature from Relative Humidity
    and Air Temperature", Journal of Applied Meteorology and Climatology, 50(11).
    """
    tc = (temp_f - 32.0) / 1.8
    rh = rh_pct  # already in percent (e.g. 75 for 75%)
    tw = (
        tc * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(tc + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh**1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )
    return tw * 1.8 + 32.0


# ---------------------------------------------------------------------------
# Cloud cover → sky text
# ---------------------------------------------------------------------------


def _cloud_cover_to_sky(pct: float) -> str:
    """Map cloud cover percentage (0–100) to a sky condition label."""
    if pct <= 10:
        return "Clear"
    if pct <= 30:
        return "Mostly Clear"
    if pct <= 60:
        return "Partly Cloudy"
    if pct <= 85:
        return "Mostly Cloudy"
    return "Overcast"


# ---------------------------------------------------------------------------
# Internal derivation functions
# ---------------------------------------------------------------------------


def _derive_sky(
    observation: Observation,
    max_solar_rad: float | None,
    provider_conditions: ProviderConditions | None,
    sun_altitude: float | None,
) -> str | None:
    """Derive current sky condition text.

    Daytime (sun_altitude > 10°): uses the clearness index (Kt) computed
    from the local solar radiation sensor vs the theoretical maximum for
    this moment and location.  Falls back to provider when sensor data is
    unavailable.

    Nighttime / dawn / dusk (sun_altitude is None or <= 10°): station solar
    sensor is unreliable, so the provider is authoritative.  Extracts sky
    keywords from provider.weatherText or maps provider.cloudCover.
    """
    daytime = sun_altitude is not None and sun_altitude > 10.0

    if not daytime:
        # Use provider for night / dawn / dusk.
        return _sky_from_provider(provider_conditions)

    # Daytime: try Kt from local radiation sensor.
    obs_rad = observation.radiation
    if obs_rad is not None and max_solar_rad is not None and max_solar_rad > 0:
        kt = obs_rad / max_solar_rad
        if kt > 0.80:
            return "Clear"
        if kt > 0.65:
            return "Mostly Sunny"
        if kt > 0.45:
            return "Partly Cloudy"
        if kt > 0.30:
            return "Mostly Cloudy"
        return "Overcast"

    # Sensor data unavailable — fall back to provider.
    return _sky_from_provider(provider_conditions)


def _sky_from_provider(provider_conditions: ProviderConditions | None) -> str | None:
    """Extract a sky condition label from provider conditions.

    Tries weatherText keyword matching first (longest match), then cloudCover.
    Returns None when the provider is unavailable or carries no useful sky data.
    """
    if provider_conditions is None:
        return None

    # Try keyword extraction from provider weatherText.
    if provider_conditions.weatherText:
        text = provider_conditions.weatherText
        for keyword, label in _SKY_KEYWORDS:
            if keyword.lower() in text.lower():
                return label

    # Try cloud cover percentage.
    if provider_conditions.cloudCover is not None:
        return _cloud_cover_to_sky(provider_conditions.cloudCover)

    return None


def _derive_precipitation(
    observation: Observation,
    provider_conditions: ProviderConditions | None,
    target_unit: str,
) -> str | None:
    """Derive current precipitation text.

    Critical fix: precipitation text is driven by the LOCAL RAIN GAUGE only
    (observation.rainRate).  The provider is consulted ONLY for frozen
    precipitation (snow/sleet/freezing rain) when the rain gauge reads zero —
    and only when wet-bulb temperature confirms conditions are cold enough.

    This prevents the bug where provider forecast text ("Patchy Drizzle") is
    shown even when the local rain gauge reads zero.
    """
    rain_rate = observation.rainRate

    if rain_rate is not None and rain_rate > 0:
        # Convert to in/hr for threshold comparisons.
        rate_in_hr = _rain_rate_to_in_per_hr(rain_rate, target_unit)

        if rate_in_hr < 0.01:
            return "Drizzle"
        if rate_in_hr < 0.10:
            return "Light Rain"
        if rate_in_hr < 0.30:
            return "Moderate Rain"
        return "Heavy Rain"

    # Rain gauge reads zero (or is unavailable).
    # Check for frozen precipitation from the provider, gated by wet-bulb temp.
    if provider_conditions is None:
        return None

    precip_type = provider_conditions.precipType
    if precip_type not in ("snow", "freezing-rain", "sleet"):
        return None

    # Compute wet-bulb temperature to confirm conditions are cold enough.
    out_temp = observation.outTemp
    out_humidity = observation.outHumidity

    if out_temp is None or out_humidity is None:
        # Cannot confirm frozen precip without temperature + humidity.
        return None

    # Convert temperature to °F for wet-bulb calculation.
    if target_unit == "US":
        temp_f = out_temp
    else:
        # Both METRIC and METRICWX store outTemp in °C.
        temp_f = _to_fahrenheit(out_temp)

    wet_bulb = _wet_bulb_f(temp_f, out_humidity)

    # Only report frozen precip when wet-bulb is at or below 35°F (1.7°C).
    # Above 35°F, frozen precip is unlikely to be reaching the ground.
    if wet_bulb > 35.0:
        return None

    if precip_type == "snow":
        return "Snow"
    if precip_type == "freezing-rain":
        return "Freezing Rain"
    if precip_type == "sleet":
        return "Sleet"

    # Unreachable given the guard above, but satisfies the type checker.
    return None  # pragma: no cover


def _derive_wind(observation: Observation, target_unit: str) -> str | None:
    """Derive Beaufort wind descriptor from the local wind sensor.

    Uses the Beaufort scale thresholds that match the dashboard's
    beaufortLabel() function exactly.  "Calm" (< 1 mph) returns None so it
    is silently omitted from the composite string.

    Appends " and Gusty" when the gust exceeds sustained by >= 12 mph AND
    the gust is >= 18 mph.
    """
    wind_speed = observation.windSpeed
    if wind_speed is None:
        return None

    speed_mph = _to_mph(wind_speed, target_unit)

    # Beaufort thresholds (upper boundary, inclusive).
    if speed_mph < 1:
        return None  # Calm — omit from composite
    if speed_mph <= 3:
        label = "Light Air"
    elif speed_mph <= 7:
        label = "Light Breeze"
    elif speed_mph <= 12:
        label = "Gentle Breeze"
    elif speed_mph <= 18:
        label = "Moderate Breeze"
    elif speed_mph <= 24:
        label = "Fresh Breeze"
    elif speed_mph <= 31:
        label = "Strong Breeze"
    elif speed_mph <= 38:
        label = "Near Gale"
    elif speed_mph <= 46:
        label = "Gale"
    elif speed_mph <= 54:
        label = "Strong Gale"
    elif speed_mph <= 63:
        label = "Storm"
    elif speed_mph <= 72:
        label = "Violent Storm"
    else:
        label = "Hurricane"

    # Gusty qualifier: gust >= sustained + 12 mph AND gust >= 18 mph.
    wind_gust = observation.windGust
    if wind_gust is not None:
        gust_mph = _to_mph(wind_gust, target_unit)
        if gust_mph >= speed_mph + 12 and gust_mph >= 18:
            label = f"{label} and Gusty"

    return label


def _derive_comfort(observation: Observation, target_unit: str) -> str | None:
    """Derive humidity comfort descriptor from the local dewpoint sensor.

    Thresholds are in °F; values below 65°F return None (comfortable —
    omit from composite).

    Dewpoint >= 65°F → "Humid"
    Dewpoint >= 70°F → "Oppressive"
    Dewpoint >= 75°F → "Miserable"
    """
    dewpoint = observation.dewpoint
    if dewpoint is None:
        return None

    # Convert to °F for threshold comparison.
    if target_unit == "US":
        dp_f = dewpoint
    else:
        dp_f = _to_fahrenheit(dewpoint)

    if dp_f >= 75:
        return "Miserable"
    if dp_f >= 70:
        return "Oppressive"
    if dp_f >= 65:
        return "Humid"
    return None  # Comfortable — omit


def _compose(
    sky: str | None,
    precip: str | None,
    wind: str | None,
    comfort: str | None,
) -> str | None:
    """Compose the four components into a single conditions string.

    Filters out None values.  Returns None when nothing remains.
    Joins with ", " except uses " and " before the last element.

    Examples:
        ("Clear", None, None, None)           → "Clear"
        ("Partly Cloudy", None, "Fresh Breeze", None)
                                              → "Partly Cloudy and Fresh Breeze"
        ("Overcast", "Light Rain", "Fresh Breeze", "Humid")
                                              → "Overcast, Light Rain, Fresh Breeze and Humid"
    """
    parts = [p for p in (sky, precip, wind, comfort) if p is not None]

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"

    # Three or more: join all but the last with ", "; use " and " before last.
    return ", ".join(parts[:-1]) + " and " + parts[-1]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def derive_conditions_text(
    observation: Observation,
    max_solar_rad: float | None,
    provider_conditions: ProviderConditions | None,
    sun_altitude: float | None,
    target_unit: str,
) -> str | None:
    """Derive current conditions text by blending local sensor data and provider.

    Args:
        observation: The most-recent weewx archive observation.
        max_solar_rad: Theoretical maximum solar radiation for this moment and
            location (W/m²).  Used for the daytime Kt (clearness index)
            computation.  None when not available (falls back to provider sky).
        provider_conditions: Current conditions from a forecast provider.
            None when no provider is configured or the fetch has not yet run.
        sun_altitude: Current sun altitude in degrees above/below the horizon.
            Positive = daytime.  None when the ephemeris is unavailable.
        target_unit: weewx unit system identifier — "US" | "METRIC" | "METRICWX".

    Returns:
        Human-readable conditions string, e.g.
        "Partly Cloudy, Light Rain, Fresh Breeze and Humid", or None when
        nothing can be determined.
    """
    sky = _derive_sky(observation, max_solar_rad, provider_conditions, sun_altitude)
    precip = _derive_precipitation(observation, provider_conditions, target_unit)
    wind = _derive_wind(observation, target_unit)
    comfort = _derive_comfort(observation, target_unit)

    result = _compose(sky, precip, wind, comfort)

    logger.debug(
        "derive_conditions_text: sky=%r precip=%r wind=%r comfort=%r → %r",
        sky,
        precip,
        wind,
        comfort,
        result,
    )

    return result
