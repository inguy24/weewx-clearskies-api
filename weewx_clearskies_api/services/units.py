"""Units-block resolution per ADR-019 + canonical-data-model §2.

Loaded at startup (never per-request).  Reads weewx.conf once via ConfigObj,
extracts:
  1. [StdConvert] target_unit  → one of "US" | "METRIC" | "METRICWX"
  2. Per-skin [StdReport] [[<skin>]] [[[Units]]] [[[[Groups]]]] overrides

Builds and caches an immutable dict[canonical_field, unit_string] for use
by every response's `units` block.

Failure modes (per brief):
  - weewx.conf not at the configured path → WeewxConfNotFoundError raised
    at startup; __main__.py catches it and exits non-zero.
  - weewx.conf exists but [StdConvert] is absent → US defaults + WARN.
  - Override references an unknown unit string → WARN + fall back to system
    default for that group.

Notes (per ADR-019):
  - The cached dict is what every response reflects.
  - Per-row usUnits differences get a one-line WARN per request (caller's
    responsibility to log; this module only supplies the dict).
  - Reload-on-change is out of scope at v0.1; restart-to-pick-up-config is
    acceptable and documented.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import configobj

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WeewxConfNotFoundError(FileNotFoundError):
    """Raised when weewx.conf is not found at the configured path.

    __main__.py treats this as fatal and exits non-zero (same pattern as the
    task-2 write-probe).
    """


# ---------------------------------------------------------------------------
# Internal type aliases
# ---------------------------------------------------------------------------

# Field name → display-friendly unit string.
UnitsBlock = dict[str, str]

# weewx group name → display-friendly unit string for a given system.
_GroupUnitMap = dict[str, str]

# ---------------------------------------------------------------------------
# weewx unit-group → member canonical fields
# Hand-translated from canonical-data-model.md §2.1.
# Groups that have operator-defined members only are not pre-populated here;
# they can't be resolved without the operator mapping (ADR-035, Phase 4).
# ---------------------------------------------------------------------------

# ruff: noqa: N815  (canonical fields use weewx camelCase per ADR-010)
_GROUP_MEMBERS: Final[dict[str, list[str]]] = {
    "group_temperature": [
        "outTemp", "dewpoint", "windchill", "heatindex", "inTemp",
        "appTemp", "extraTemp1", "extraTemp2", "extraTemp3",
        "soilTemp1", "soilTemp2", "soilTemp3", "soilTemp4",
        "leafTemp1", "leafTemp2",
        "humidex", "THSW",
        # forecast-only canonical fields that share this group
        "tempMax", "tempMin",
    ],
    "group_speed": [
        "windSpeed", "windGust",
        # forecast-only
        "windSpeedMax", "windGustMax",
    ],
    "group_speed2": ["rms", "vecavg"],
    "group_direction": ["windDir", "windGustDir", "gustdir", "vecdir"],
    "group_pressure": ["barometer", "altimeter", "pressure"],
    "group_pressurerate": ["barometerRate", "altimeterRate", "pressureRate"],
    "group_rain": ["rain", "ET", "hail", "snow", "snowDepth", "precipAmount"],
    "group_rainrate": ["rainRate", "hailRate", "snowRate"],
    "group_radiation": ["radiation", "maxSolarRad"],
    "group_uv": ["UV", "uvIndexMax"],
    "group_percent": [
        "outHumidity", "inHumidity",
        "extraHumid1", "extraHumid2",
        "cloudcover", "cloudCover", "pop",
        "precipProbability", "precipProbabilityMax",
        "rxCheckPercent", "snowMoisture",
    ],
    "group_moisture": ["soilMoist1", "soilMoist2", "soilMoist3", "soilMoist4"],
    "group_count": [
        "leafWet1", "leafWet2",
        "lightning_strike_count", "lightning_disturber_count",
        "lightning_noise_count",
        "felt",
    ],
    "group_distance": ["windrun", "lightning_distance"],
    "group_altitude": ["altitude", "cloudbase"],
    "group_volt": [
        "consBatteryVoltage", "heatingVoltage",
        "referenceVoltage", "supplyVoltage",
    ],
    "group_db": ["noise"],
    "group_deltatime": ["rainDur", "sunshineDur", "daySunshineDur", "sunshineDurDoc"],
    "group_degree_day": ["cooldeg", "heatdeg", "growdeg"],
    "group_concentration": [
        "pollutantPM25", "pollutantPM10",
        "pm1_0", "pm2_5", "pm10_0", "no2",
    ],
    "group_fraction": [
        "pollutantO3", "pollutantSO2", "pollutantCO", "pollutantNO2",
        "co", "co2", "nh3", "o3", "pb", "so2",
    ],
    "group_illuminance": ["illuminance"],
    "group_interval": ["interval"],
}

# ---------------------------------------------------------------------------
# System presets — group → unit string per target_unit system.
# Source: canonical-data-model.md §2.1 verbatim.
# ---------------------------------------------------------------------------

_SYSTEM_PRESETS: Final[dict[str, _GroupUnitMap]] = {
    "US": {
        "group_temperature": "°F",
        "group_speed": "mph",
        "group_speed2": "mph",
        "group_direction": "°",
        "group_pressure": "inHg",
        "group_pressurerate": "inHg/h",
        "group_rain": "in",
        "group_rainrate": "in/h",
        "group_radiation": "W/m²",
        "group_uv": "uv_index",
        "group_percent": "%",
        "group_moisture": "cb",
        "group_count": "count",
        "group_distance": "mile",
        "group_altitude": "foot",
        "group_volt": "V",
        "group_amp": "amp",
        "group_power": "W",
        "group_energy": "Wh",
        "group_energy2": "Ws",
        "group_data": "byte",
        "group_db": "dB",
        "group_deltatime": "s",
        "group_degree_day": "°F·day",
        "group_concentration": "µg/m³",
        "group_fraction": "ppm",
        "group_frequency": "Hz",
        "group_illuminance": "lx",
        "group_interval": "minute",
        "group_length": "inch",
        "group_volume": "gallon",
    },
    "METRIC": {
        "group_temperature": "°C",
        "group_speed": "km/h",
        "group_speed2": "km/h",
        "group_direction": "°",
        "group_pressure": "mbar",
        "group_pressurerate": "mbar/h",
        "group_rain": "cm",
        "group_rainrate": "cm/h",
        "group_radiation": "W/m²",
        "group_uv": "uv_index",
        "group_percent": "%",
        "group_moisture": "cb",
        "group_count": "count",
        "group_distance": "km",
        "group_altitude": "meter",
        "group_volt": "V",
        "group_amp": "amp",
        "group_power": "W",
        "group_energy": "Wh",
        "group_energy2": "Ws",
        "group_data": "byte",
        "group_db": "dB",
        "group_deltatime": "s",
        "group_degree_day": "°C·day",
        "group_concentration": "µg/m³",
        "group_fraction": "ppm",
        "group_frequency": "Hz",
        "group_illuminance": "lx",
        "group_interval": "minute",
        "group_length": "cm",
        "group_volume": "liter",
    },
    "METRICWX": {
        "group_temperature": "°C",
        "group_speed": "m/s",
        "group_speed2": "m/s",
        "group_direction": "°",
        "group_pressure": "mbar",
        "group_pressurerate": "mbar/h",
        "group_rain": "mm",
        "group_rainrate": "mm/h",
        "group_radiation": "W/m²",
        "group_uv": "uv_index",
        "group_percent": "%",
        "group_moisture": "cb",
        "group_count": "count",
        "group_distance": "km",
        "group_altitude": "meter",
        "group_volt": "V",
        "group_amp": "amp",
        "group_power": "W",
        "group_energy": "Wh",
        "group_energy2": "Ws",
        "group_data": "byte",
        "group_db": "dB",
        "group_deltatime": "s",
        "group_degree_day": "°C·day",
        "group_concentration": "µg/m³",
        "group_fraction": "ppm",
        "group_frequency": "Hz",
        "group_illuminance": "lx",
        "group_interval": "minute",
        "group_length": "cm",
        "group_volume": "liter",
    },
}

# weewx internal unit names → display-friendly strings.
# Used when applying operator [StdReport] group overrides.
_WEEWX_UNIT_TO_DISPLAY: Final[dict[str, str]] = {
    # Temperature
    "degree_F": "°F",
    "degree_C": "°C",
    # Speed
    "mile_per_hour": "mph",
    "km_per_hour": "km/h",
    "meter_per_second": "m/s",
    "knot": "knot",
    # Pressure
    "inHg": "inHg",
    "mbar": "mbar",
    "hPa": "hPa",
    "kPa": "kPa",
    # Rain / depth
    "inch": "in",
    "cm": "cm",
    "mm": "mm",
    # Rain rate
    "inch_per_hour": "in/h",
    "cm_per_hour": "cm/h",
    "mm_per_hour": "mm/h",
    # Radiation
    "watt_per_meter_squared": "W/m²",
    # UV
    "uv_index": "uv_index",
    # Percent
    "percent": "%",
    # Moisture
    "centibar": "cb",
    # Count
    "count": "count",
    # Distance
    "mile": "mile",
    "km": "km",
    # Altitude
    "foot": "foot",
    "meter": "meter",
    # Volt
    "volt": "V",
    # Electrical
    "amp": "amp",
    "watt": "W",
    "watt_hour": "Wh",
    "watt_second": "Ws",
    # Data
    "byte": "byte",
    # dB
    "dB": "dB",
    # Time
    "second": "s",
    "minute": "minute",
    # Degree-day
    "degree_F_day": "°F·day",
    "degree_C_day": "°C·day",
    # Concentration
    "microgram_per_meter_cubed": "µg/m³",
    # Fraction
    "ppm": "ppm",
    "ppb": "ppb",
    # Frequency
    "hertz": "Hz",
    # Illuminance
    "lux": "lx",
    # Length
    "inch": "in",
    # Volume
    "gallon": "gallon",
    "liter": "liter",
}

# ---------------------------------------------------------------------------
# Module-level cache — populated at startup by load_units_block().
# ---------------------------------------------------------------------------

_cached_units_block: UnitsBlock | None = None
_cached_target_unit: str | None = None


def _build_units_block_from_group_map(group_map: _GroupUnitMap) -> UnitsBlock:
    """Expand a per-group unit map into a per-field unit map.

    For each group in group_map, emit one entry per member field in
    _GROUP_MEMBERS.  Fields whose group is not in group_map are omitted
    (they stay absent from the units block per canonical-data-model §2.3).
    """
    block: UnitsBlock = {}
    for group, unit_str in group_map.items():
        for field in _GROUP_MEMBERS.get(group, []):
            block[field] = unit_str
    return block


def _apply_stdreport_overrides(
    base_block: UnitsBlock,
    cfg: configobj.ConfigObj,
    system_group_map: _GroupUnitMap,
) -> UnitsBlock:
    """Apply per-skin [StdReport][[[Units]]][[[[Groups]]]] overrides.

    weewx allows operators to override the unit for a group inside each skin's
    report config block.  We walk all skins and collect group overrides.  The
    last skin that specifies a group wins (arbitrary but deterministic).

    Any unknown unit string → WARN + fall back to the system default.
    """
    overridden_groups: dict[str, str] = {}

    std_report = cfg.get("StdReport")
    if not isinstance(std_report, dict):
        return base_block

    for _skin_name, skin_section in std_report.items():
        if not isinstance(skin_section, dict):
            continue
        units_section = skin_section.get("Units")
        if not isinstance(units_section, dict):
            continue
        groups_section = units_section.get("Groups")
        if not isinstance(groups_section, dict):
            continue
        for group_name, unit_val in groups_section.items():
            if not isinstance(group_name, str) or not isinstance(unit_val, str):
                continue
            overridden_groups[group_name] = unit_val

    if not overridden_groups:
        return base_block

    # Build effective group map: start with system defaults, apply overrides.
    effective_group_map: _GroupUnitMap = dict(system_group_map)
    for group_name, unit_val in overridden_groups.items():
        display = _WEEWX_UNIT_TO_DISPLAY.get(unit_val)
        if display is None:
            # Also accept values that are already in display form (hPa, knot, etc.)
            if unit_val in _WEEWX_UNIT_TO_DISPLAY.values():
                display = unit_val
        if display is None:
            # Unknown unit — fall back to system default for that group.
            default_for_group = system_group_map.get(group_name)
            logger.warning(
                "Unknown unit override %r for group %r in weewx.conf "
                "[StdReport]; falling back to system default %r.",
                unit_val,
                group_name,
                default_for_group,
                extra={
                    "group": group_name,
                    "unit_val": unit_val,
                    "fallback": default_for_group,
                },
            )
            # Keep system default; don't update effective_group_map.
        else:
            effective_group_map[group_name] = display
            logger.debug(
                "Applying unit override: group %r → %r",
                group_name,
                display,
            )

    return _build_units_block_from_group_map(effective_group_map)


def load_units_block(weewx_conf_path: str | Path) -> tuple[UnitsBlock, str]:
    """Load and cache the units block from weewx.conf.

    Called once at startup.  Subsequent calls return the cached value.

    Args:
        weewx_conf_path: Path to weewx.conf.

    Returns:
        (units_block, target_unit) — the field→unit dict and the unit system
        label ("US" | "METRIC" | "METRICWX").

    Raises:
        WeewxConfNotFoundError: weewx.conf not found at the given path.
    """
    global _cached_units_block, _cached_target_unit  # noqa: PLW0603

    if _cached_units_block is not None and _cached_target_unit is not None:
        return _cached_units_block, _cached_target_unit

    path = Path(weewx_conf_path)
    if not path.exists():
        raise WeewxConfNotFoundError(
            f"FATAL: weewx.conf not found at {path}. "
            "Set [weewx] config_path in api.conf to the correct path, "
            "or ensure weewx.conf exists at the default location /etc/weewx/weewx.conf."
        )

    try:
        cfg = configobj.ConfigObj(str(path), interpolation=False)
    except configobj.ConfigObjError as exc:
        raise WeewxConfNotFoundError(
            f"FATAL: weewx.conf at {path} could not be parsed: {exc}. "
            "Check the file is valid INI/configobj format."
        ) from exc
    except OSError as exc:
        raise WeewxConfNotFoundError(
            f"FATAL: Cannot read weewx.conf at {path}: {exc}. "
            "Check file permissions (readable by the clearskies-api process)."
        ) from exc

    # Read target_unit from [StdConvert].
    std_convert = cfg.get("StdConvert")
    if not isinstance(std_convert, dict):
        logger.warning(
            "weewx.conf at %s has no [StdConvert] section; "
            "defaulting to US unit system.",
            path,
        )
        target_unit = "US"
    else:
        raw = str(std_convert.get("target_unit", "US")).strip().upper()
        if raw not in _SYSTEM_PRESETS:
            logger.warning(
                "Unrecognised weewx target_unit %r in [StdConvert]; "
                "defaulting to US.",
                raw,
            )
            target_unit = "US"
        else:
            target_unit = raw

    system_group_map = _SYSTEM_PRESETS[target_unit]
    base_block = _build_units_block_from_group_map(system_group_map)
    units_block = _apply_stdreport_overrides(base_block, cfg, system_group_map)

    logger.info(
        "Units block loaded from weewx.conf",
        extra={
            "target_unit": target_unit,
            "weewx_conf_path": str(path),
            "field_count": len(units_block),
        },
    )

    _cached_units_block = units_block
    _cached_target_unit = target_unit
    return units_block, target_unit


def get_units_block() -> UnitsBlock:
    """Return the cached units block.

    Raises RuntimeError if load_units_block() has not been called yet
    (should never happen in production; startup sequence loads it first).
    """
    if _cached_units_block is None:
        raise RuntimeError(
            "Units block not loaded. Call load_units_block() at startup before "
            "serving requests."
        )
    return _cached_units_block


def get_target_unit() -> str:
    """Return the cached target unit system string ("US" | "METRIC" | "METRICWX").

    Raises RuntimeError if load_units_block() has not been called yet.
    """
    if _cached_target_unit is None:
        raise RuntimeError(
            "Units block not loaded. Call load_units_block() at startup before "
            "serving requests."
        )
    return _cached_target_unit


def reset_cache() -> None:
    """Reset the module-level cache.  Used in tests only."""
    global _cached_units_block, _cached_target_unit  # noqa: PLW0603
    _cached_units_block = None
    _cached_target_unit = None
