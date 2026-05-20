"""Station metadata service (3a-2).

Loads station identity from weewx.conf [Station] at startup and caches it.
The cached object is a partially-populated StationMetadata; firstRecord and
lastRecord are filled per-request from a DB query.

Station-id default: slug of weewx.conf [Station] location (per resolved
call #3 in the brief).  Operator override via api.conf [station] station_id.

Timezone source priority (ADR-020):
  1. api.conf [station] timezone
  2. weewx.conf [Station] timezone (if present)
  3. OS timezone via time.tzname + zoneinfo lookup
  4. UTC + WARN

NO timezonefinder — that is Phase 4 setup-wizard scope per ADR-020 §Consequences.

configobj comma-normalisation (F1 remediation):
  Real weewx.conf files write fields like `location = Belchertown, MA` unquoted.
  configobj parses any unquoted comma-containing value as a Python list.
  _get_str_field() handles both the string and list forms so that location,
  station_type, and timezone all survive the list-parse path.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import configobj

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StationConfigError(ValueError):
    """Raised when required weewx.conf [Station] fields are missing.

    __main__.py treats this as fatal and exits non-zero.
    """


# ---------------------------------------------------------------------------
# Station metadata data class
# ---------------------------------------------------------------------------


class StationInfo:
    """Partially-populated station metadata (firstRecord / lastRecord filled per-request)."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    altitude: float
    timezone: str
    timezone_offset_minutes: int
    unit_system: str
    hardware: str | None

    def __init__(
        self,
        station_id: str,
        name: str,
        latitude: float,
        longitude: float,
        altitude: float,
        timezone: str,
        timezone_offset_minutes: int,
        unit_system: str,
        hardware: str | None,
    ) -> None:
        self.station_id = station_id
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone = timezone
        self.timezone_offset_minutes = timezone_offset_minutes
        self.unit_system = unit_system
        self.hardware = hardware


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_cached_station: StationInfo | None = None


# ---------------------------------------------------------------------------
# Slug helper
# ---------------------------------------------------------------------------


def _get_str_field(section: dict, key: str, default: str = "") -> str:  # type: ignore[type-arg]
    """Read a scalar string field from a configobj section.

    configobj parses unquoted comma-containing values (e.g.
    ``location = Belchertown, MA``) as Python lists.  This helper
    normalises both the list and string forms back to a plain string.

    Args:
        section: The configobj section dict.
        key: The key to look up.
        default: Returned when the key is absent.

    Returns:
        Stripped string value.
    """
    raw = section.get(key, default)
    if isinstance(raw, list):
        raw = ", ".join(str(item).strip() for item in raw)
    return str(raw).strip()


def _slugify(value: str) -> str:
    """Convert a location string to a URL slug.

    "Belchertown, MA" → "belchertown-ma"
    Lowercases, strips leading/trailing whitespace, replaces non-alphanumeric
    runs with a single hyphen.
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value


# ---------------------------------------------------------------------------
# Timezone resolution
# ---------------------------------------------------------------------------


def _resolve_timezone(api_tz: str | None, weewx_tz: str | None) -> str:
    """Resolve the station timezone using the priority order from ADR-020.

    Priority:
      1. api.conf [station] timezone (api_tz)
      2. weewx.conf [Station] timezone (weewx_tz)
      3. OS timezone via time.tzname
      4. UTC + WARN

    Returns an IANA TZ identifier string.
    """
    for candidate, source in [
        (api_tz, "api.conf [station] timezone"),
        (weewx_tz, "weewx.conf [Station] timezone"),
    ]:
        if candidate and candidate.strip():
            tz_str = candidate.strip()
            try:
                ZoneInfo(tz_str)
                logger.debug("Timezone resolved from %s: %s", source, tz_str)
                return tz_str
            except ZoneInfoNotFoundError:
                logger.warning(
                    "Timezone %r from %s is not a valid IANA identifier; "
                    "trying next source.",
                    tz_str,
                    source,
                )

    # Try OS timezone via time.tzname.
    os_tz_name = time.tzname[0] if time.tzname else None
    if os_tz_name:
        try:
            ZoneInfo(os_tz_name)
            logger.debug("Timezone resolved from OS tzname: %s", os_tz_name)
            return os_tz_name
        except ZoneInfoNotFoundError:
            pass

    # Final fallback.
    logger.warning(
        "Could not resolve station timezone from api.conf, weewx.conf, or OS; "
        "defaulting to UTC. Set [station] timezone = <IANA-TZ> in api.conf."
    )
    return "UTC"


def _tz_offset_minutes(iana_tz: str) -> int:
    """Return current UTC offset in minutes for the given IANA TZ."""
    try:
        zi = ZoneInfo(iana_tz)
        now_utc = datetime.now(tz=UTC)
        offset = zi.utcoffset(now_utc)
        if offset is None:
            return 0
        total_seconds = int(offset.total_seconds())
        return total_seconds // 60
    except ZoneInfoNotFoundError:
        return 0


# ---------------------------------------------------------------------------
# Altitude parse helper
# ---------------------------------------------------------------------------


def _parse_altitude(raw: str) -> float:
    """Parse weewx altitude string 'value, unit' and return the numeric value.

    weewx.conf [Station] altitude stores e.g. "700, foot" or "200, meter".
    We pass through the numeric value unchanged (ADR-019: no server-side
    conversion; the units block carries the unit string).

    Args:
        raw: The raw string from weewx.conf.

    Returns:
        Numeric altitude value as float.

    Raises:
        StationConfigError: String cannot be parsed.
    """
    parts = raw.split(",", 1)
    try:
        return float(parts[0].strip())
    except (ValueError, IndexError) as exc:
        raise StationConfigError(
            f"Cannot parse altitude from weewx.conf [Station] altitude = {raw!r}. "
            "Expected format: 'value, unit' (e.g. '700, foot')."
        ) from exc


# ---------------------------------------------------------------------------
# Startup loader
# ---------------------------------------------------------------------------


def load_station_metadata(
    cfg: configobj.ConfigObj,
    api_station_id: str | None,
    api_timezone: str | None,
    unit_system: str,
) -> StationInfo:
    """Load and cache station metadata from the parsed weewx.conf ConfigObj.

    Called once at startup (from __main__.py after load_weewx_conf and
    load_units_block have both succeeded).

    Args:
        cfg: Parsed weewx.conf ConfigObj (from services.weewx_conf).
        api_station_id: Optional operator override from api.conf [station] station_id.
        api_timezone: Optional operator override from api.conf [station] timezone.
        unit_system: The resolved target_unit string from services.units.

    Returns:
        StationInfo with all fields populated except firstRecord / lastRecord
        (those are filled per-request from a DB query).

    Raises:
        StationConfigError: Required fields missing in weewx.conf [Station].
    """
    global _cached_station  # noqa: PLW0603

    station_section = cfg.get("Station")
    if not isinstance(station_section, dict):
        raise StationConfigError(
            "FATAL: weewx.conf is missing the [Station] section. "
            "This is required for clearskies-api. "
            "Check your weewx.conf and ensure weewx has been configured."
        )

    # --- Required: location (maps to name) ---
    # Use _get_str_field so "location = Belchertown, MA" (unquoted in real
    # weewx.conf) is normalised from configobj's list form back to a string.
    raw_location = _get_str_field(station_section, "location")
    if not raw_location:
        raise StationConfigError(
            "FATAL: weewx.conf [Station] location is missing or empty. "
            "Set [Station] location = <station name> in weewx.conf."
        )

    # --- Required: latitude ---
    raw_lat = station_section.get("latitude", "").strip()
    if not raw_lat:
        raise StationConfigError(
            "FATAL: weewx.conf [Station] latitude is missing or empty. "
            "Set [Station] latitude = <decimal degrees> in weewx.conf."
        )
    try:
        latitude = float(raw_lat)
    except ValueError as exc:
        raise StationConfigError(
            f"FATAL: weewx.conf [Station] latitude = {raw_lat!r} is not a number."
        ) from exc

    # --- Required: longitude ---
    raw_lon = station_section.get("longitude", "").strip()
    if not raw_lon:
        raise StationConfigError(
            "FATAL: weewx.conf [Station] longitude is missing or empty. "
            "Set [Station] longitude = <decimal degrees> in weewx.conf."
        )
    try:
        longitude = float(raw_lon)
    except ValueError as exc:
        raise StationConfigError(
            f"FATAL: weewx.conf [Station] longitude = {raw_lon!r} is not a number."
        ) from exc

    # --- Altitude (required by OpenAPI; use 0 with WARN if absent) ---
    # configobj parses "altitude = 700, foot" as a list ['700', 'foot'] because
    # the comma is the INI list separator.  Handle both the string and list forms.
    raw_altitude_val = station_section.get("altitude", "")
    if isinstance(raw_altitude_val, list):
        # List form: ['700', 'foot'] or ['700'] — join back for _parse_altitude.
        raw_altitude = ", ".join(str(x) for x in raw_altitude_val)
    else:
        raw_altitude = str(raw_altitude_val).strip()

    if not raw_altitude:
        logger.warning(
            "weewx.conf [Station] altitude is missing; defaulting to 0. "
            "Set altitude = <value, unit> in weewx.conf."
        )
        altitude = 0.0
    else:
        altitude = _parse_altitude(raw_altitude)

    # --- Station ID: api.conf override → slug of location ---
    if api_station_id and api_station_id.strip():
        station_id = api_station_id.strip()
    else:
        station_id = _slugify(raw_location)

    # --- Timezone ---
    # Use _get_str_field — an IANA timezone with a comma is not a real case, but
    # consistent use of the helper costs nothing and avoids a future surprise.
    weewx_tz = _get_str_field(station_section, "timezone") or None
    timezone = _resolve_timezone(api_timezone, weewx_tz)
    tz_offset = _tz_offset_minutes(timezone)

    # --- Optional: hardware (station_type) ---
    # Use _get_str_field — "station_type = Davis Vantage Pro2, USA" (unquoted)
    # would produce a list without this.
    raw_hardware = _get_str_field(station_section, "station_type")
    hardware: str | None = raw_hardware if raw_hardware else None

    info = StationInfo(
        station_id=station_id,
        name=raw_location,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        timezone=timezone,
        timezone_offset_minutes=tz_offset,
        unit_system=unit_system,
        hardware=hardware,
    )

    logger.info(
        "Station metadata loaded",
        extra={
            "station_id": station_id,
            "station_name": raw_location,
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "unit_system": unit_system,
        },
    )

    _cached_station = info
    return info


def get_station_info() -> StationInfo:
    """Return the cached StationInfo.

    Raises:
        RuntimeError: load_station_metadata() has not been called yet.
    """
    if _cached_station is None:
        raise RuntimeError(
            "Station metadata not loaded. "
            "Call load_station_metadata() at startup before serving requests."
        )
    return _cached_station


def reset_cache() -> None:
    """Reset the module-level cache.  Used in tests only."""
    global _cached_station  # noqa: PLW0603
    _cached_station = None
