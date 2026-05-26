"""Almanac service — Skyfield-based ephemeris computation (ADR-014, 3a-2).

Wire at startup:
    wire_ephemeris_directory(path) — loads de421.bsp once; caches ts + eph.

Request-time helpers:
    compute_almanac(date, lat, lon, alt) → AlmanacDay
    compute_sun_times_year(year, lat, lon, alt) → list[SunDay]
    compute_moon_phases(year, month?, lat, lon) → list[MoonDay]

Polar-edge handling: when the sun or moon does not rise/set on a given day,
Skyfield's find_discrete returns no events. In those cases rise/set/transit
are returned as None.  daylightMinutes = 0 on polar night, 1440 on polar day.

Moon phase name mapping (8-bin by ecliptic longitude of phase angle):
  The phase angle is the angle from new moon (0°) through full (180°) back to
  new (360°).  We bin it uniformly:

  0° to <22.5°   → new
  22.5° to <67.5°  → waxing-crescent
  67.5° to <112.5° → first-quarter
  112.5° to <157.5°→ waxing-gibbous
  157.5° to <202.5°→ full
  202.5° to <247.5°→ waning-gibbous
  247.5° to <292.5°→ last-quarter
  292.5° to <337.5°→ waning-crescent
  337.5° to 360°   → new  (wraps back to new)

Each bin is centred on its canonical angle (new=0°, waxing-crescent=45°, etc.)
with ±22.5° width.
"""

from __future__ import annotations

import logging
import os
import stat
import urllib.error
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from skyfield import almanac
from skyfield.api import Loader, wgs84
from skyfield.framelib import ecliptic_frame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache — populated by wire_ephemeris_directory().
# ---------------------------------------------------------------------------

_ts: object | None = None  # skyfield Timescale
_eph: object | None = None  # skyfield EphemerisFile


# ---------------------------------------------------------------------------
# Phase-name 8-bin mapping
# ---------------------------------------------------------------------------

_PHASE_BINS: tuple[tuple[float, str], ...] = (
    # (upper_bound_exclusive, name)  — angle is 0..360
    (22.5, "new"),
    (67.5, "waxing-crescent"),
    (112.5, "first-quarter"),
    (157.5, "waxing-gibbous"),
    (202.5, "full"),
    (247.5, "waning-gibbous"),
    (292.5, "last-quarter"),
    (337.5, "waning-crescent"),
    # 337.5..360 wraps to "new"
)


def _phase_name_from_angle(angle_deg: float) -> str:
    """Map a moon phase angle (0–360°) to one of the 8 canonical phase names.

    The angle is the ecliptic longitude difference (Moon − Sun), increasing
    as the moon waxes.  0° = new moon, 180° = full moon.
    """
    # Normalise to [0, 360).
    a = angle_deg % 360.0
    for upper, name in _PHASE_BINS:
        if a < upper:
            return name
    # 337.5 ≤ a < 360: wraps to new.
    return "new"


# ---------------------------------------------------------------------------
# Startup wiring
# ---------------------------------------------------------------------------


def wire_ephemeris_directory(directory: str) -> None:
    """Load the DE421 ephemeris and cache ts + eph at module level.

    Called once from __main__.py between load_units_block and
    wire_reports_directory (per the brief's startup sequence).

    Failure modes (fail-closed per the brief):
      - Cache directory not writable AND ephemeris not present → CRITICAL + exit.
      - Cache writable but no internet (first run download fails) → CRITICAL + exit.
      - Ephemeris present on disk → load and continue (no network needed).

    Raises:
        SystemExit: On fatal failure; caller is __main__.py which will not recover.
    """
    global _ts, _eph  # noqa: PLW0603

    import sys

    path = Path(directory)

    # Create the cache directory if it doesn't exist (mode 0755).
    if not path.exists():
        try:
            path.mkdir(parents=True, mode=stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            logger.info("Created ephemeris cache directory: %s", path)
        except OSError as exc:
            logger.critical(
                "FATAL: Cannot create ephemeris cache directory %s: %s. "
                "Create it manually or choose a writable path via "
                "[almanac] ephemeris_directory in api.conf.",
                path,
                exc,
            )
            sys.exit(1)

    # Check writability + ephemeris presence before invoking Loader.
    # If the dir is not writable and de421.bsp is not already there,
    # the Loader will attempt a download that will fail with a permissions error.
    ephemeris_file = path / "de421.bsp"
    dir_writable = os.access(str(path), os.W_OK)

    if not ephemeris_file.exists() and not dir_writable:
        logger.critical(
            "FATAL: Ephemeris file de421.bsp not found at %s and the directory "
            "is not writable (cannot download). "
            "Either pre-place de421.bsp in %s (offline install) or make the "
            "directory writable so clearskies-api can download it on first run. "
            "See the installation guide for offline-install instructions.",
            ephemeris_file,
            path,
        )
        sys.exit(1)

    import time as _time

    t_start = _time.monotonic()

    try:
        loader = Loader(str(path))
        eph = loader("de421.bsp")
        ts = loader.timescale()
    except (OSError, urllib.error.URLError, ValueError) as exc:
        # OSError/IOError: directory not writable, file missing, permission denied.
        # URLError: network failure on first-run download.
        # ValueError: ephemeris file corrupted or out-of-range for DE421.
        logger.critical(
            "FATAL: Failed to load ephemeris de421.bsp from %s: %s. "
            "On first run, clearskies-api downloads DE421 (~17 MB) from JPL. "
            "For offline installs, pre-place de421.bsp in %s.",
            path,
            exc,
            path,
        )
        sys.exit(1)
    except Exception:
        # Unknown exception type — re-raise with CRITICAL so it surfaces
        # at startup rather than being silently swallowed.
        logger.critical(
            "FATAL: Unexpected error loading ephemeris from %s. "
            "See traceback above.",
            path,
            exc_info=True,
        )
        raise

    elapsed = _time.monotonic() - t_start
    size_mb = ephemeris_file.stat().st_size / (1024 * 1024) if ephemeris_file.exists() else 0.0

    logger.info(
        "Ephemeris loaded",
        extra={
            "path": str(ephemeris_file),
            "size_mb": round(size_mb, 1),
            "load_time_s": round(elapsed, 3),
        },
    )

    _ts = ts
    _eph = eph


def get_ts_eph() -> tuple:  # type: ignore[return]
    """Return the cached (timescale, ephemeris) tuple.

    If not yet wired (e.g., in unit tests that call compute functions directly),
    attempts a lazy load from the default ephemeris directory.  In production,
    wire_ephemeris_directory() is always called at startup before any requests.

    Raises:
        RuntimeError: Ephemeris not loaded and lazy load also failed.
    """
    global _ts, _eph  # noqa: PLW0603
    if _ts is not None and _eph is not None:
        return _ts, _eph

    # Lazy load — used by tests that call compute_* directly.
    # Tries the default cache directory and falls through on failure.
    import os as _os
    default_dir = _os.environ.get(
        "CLEARSKIES_EPHEMERIS_DIR",
        "/var/cache/weewx-clearskies/skyfield/",
    )
    try:
        loader = Loader(default_dir)
        eph = loader("de421.bsp")
        ts = loader.timescale()
        _ts = ts
        _eph = eph
        return _ts, _eph
    except (OSError, urllib.error.URLError, ValueError) as exc:
        # Known failure modes: missing file, network error, corrupted ephemeris.
        raise RuntimeError(
            "Ephemeris not loaded. In production, call wire_ephemeris_directory() "
            "at startup. In tests, set CLEARSKIES_EPHEMERIS_DIR to a directory "
            "containing de421.bsp or call wire_ephemeris_directory() in a fixture."
            f" Original error: {exc}"
        ) from exc


def reset_cache() -> None:
    """Reset module-level cache.  Used in tests only."""
    global _ts, _eph  # noqa: PLW0603
    _ts = None
    _eph = None


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SunInfo:
    """Sun data for one date."""

    rise: str | None = None
    set: str | None = None
    transit: str | None = None
    civil_twilight_dawn: str | None = None
    civil_twilight_dusk: str | None = None
    azimuth: float | None = None
    altitude: float | None = None
    right_ascension: float | None = None
    declination: float | None = None
    daylight_minutes: int = 0
    daylight_delta_vs_yesterday_minutes: int | None = None
    next_equinox: str | None = None
    next_solstice: str | None = None


@dataclass
class MoonInfo:
    """Moon data for one date."""

    rise: str | None = None
    set: str | None = None
    transit: str | None = None
    azimuth: float | None = None
    altitude: float | None = None
    right_ascension: float | None = None
    declination: float | None = None
    phase_name: str | None = None
    illumination_percent: float | None = None
    next_full_moon: str | None = None
    next_new_moon: str | None = None


@dataclass
class AlmanacDay:
    """Full almanac snapshot for one date."""

    date_str: str
    sun: SunInfo
    moon: MoonInfo


@dataclass
class SunDay:
    """Sunrise / sunset / daylight for one date (used by /almanac/sun-times)."""

    date_str: str
    sunrise: str | None
    sunset: str | None
    daylight_minutes: int | None


@dataclass
class MoonDay:
    """Moon phase for one date (used by /almanac/moon-phases)."""

    date_str: str
    phase_name: str
    illumination_percent: float


# ---------------------------------------------------------------------------
# UTC ISO-8601 formatter
# ---------------------------------------------------------------------------


def _to_utc_z(t: object) -> str:
    """Convert a Skyfield Time object to UTC ISO-8601 with Z suffix."""

    dt = t.utc_datetime()  # type: ignore[attr-defined]
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Station-local day-window helper (F2 fix)
# ---------------------------------------------------------------------------


def _station_local_window(
    ts: object,
    d: date,
    station_tz: str,
) -> tuple[object, object]:
    """Return a Skyfield (t0, t1) window spanning the station-local calendar day.

    The window is [d 00:00 station-local, d+1 00:00 station-local) converted
    to UTC and then to Skyfield Time objects.

    Using UTC midnight-to-midnight was the round-1 bug (F2): for an EDT station
    (UTC-4) in summer, sunset falls after 00:00Z the *next* UTC day, so the UTC
    window for Jun 21 misses it and returns the previous evening's sunset instead.

    Args:
        ts: Skyfield Timescale.
        d: Station-local calendar date.
        station_tz: IANA timezone identifier for the station.

    Returns:
        (t0, t1) Skyfield Time objects bounding the station-local day.
    """
    try:
        zi = ZoneInfo(station_tz)
    except ZoneInfoNotFoundError:
        zi = ZoneInfo("UTC")

    # Build station-local midnight and the following midnight.
    local_midnight = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=zi)
    next_day = d + timedelta(days=1)
    local_next_midnight = datetime(
        next_day.year, next_day.month, next_day.day, 0, 0, 0, tzinfo=zi
    )

    # Convert to UTC and unpack for Skyfield.
    utc_start = local_midnight.astimezone(UTC)
    utc_end = local_next_midnight.astimezone(UTC)

    t0 = ts.utc(  # type: ignore[call-arg]
        utc_start.year, utc_start.month, utc_start.day,
        utc_start.hour, utc_start.minute, utc_start.second,
    )
    t1 = ts.utc(  # type: ignore[call-arg]
        utc_end.year, utc_end.month, utc_end.day,
        utc_end.hour, utc_end.minute, utc_end.second,
    )
    return t0, t1


# ---------------------------------------------------------------------------
# Core compute helpers
# ---------------------------------------------------------------------------


def _compute_sun_for_date(
    ts: object,
    eph: object,
    d: date,
    location: object,
    station_tz: str,
    include_delta: bool = False,
    yesterday_daylight: int | None = None,
) -> SunInfo:
    """Compute sun info for a single date at a given location.

    Args:
        ts: Skyfield Timescale.
        eph: Loaded ephemeris.
        d: The station-local calendar date to compute for.
        location: Skyfield WGS84 geographic position.
        station_tz: IANA timezone identifier — used to build a station-local
            midnight-to-midnight window (F2 fix: was UTC midnight before).
        include_delta: Whether to compute daylightDeltaVsYesterdayMinutes.
        yesterday_daylight: Pre-computed yesterday daylight minutes (for delta).
    """
    # Build a Time interval spanning the station-local calendar day.
    # Prior to the F2 fix this used UTC midnight-to-midnight, which caused
    # daylightMinutes=0 on the summer solstice for western-hemisphere stations:
    # EDT sunset (~00:29Z) falls outside the UTC Jun 21 window.
    t0, t1 = _station_local_window(ts, d, station_tz)

    sun = eph["sun"]  # type: ignore[index]
    earth = eph["earth"]  # type: ignore[index]

    observer = earth + location  # type: ignore[operator]

    # --- Sunrise / sunset ---
    # Polar-edge handling: Skyfield's find_discrete returns an empty events array
    # when the sun does not rise or set — NOT an exception.  The try/except that
    # lived here was catching real bugs (AttributeError, TypeError) and demoting
    # them to DEBUG, producing silent null/zero output.  Removed per F3 fix.
    rise_time: str | None = None
    set_time: str | None = None
    daylight_mins = 0

    f_rise = almanac.risings_and_settings(eph, sun, location)  # type: ignore[arg-type]
    times, events = almanac.find_discrete(t0, t1, f_rise)  # type: ignore[arg-type]

    rise_ts = None
    set_ts = None
    for t, e in zip(times, events, strict=False):
        if e == 1 and rise_ts is None:
            rise_ts = t
        elif e == 0 and set_ts is None:
            set_ts = t

    if rise_ts is not None:
        rise_time = _to_utc_z(rise_ts)
    if set_ts is not None:
        set_time = _to_utc_z(set_ts)

    # Daylight minutes from rise to set.
    if rise_ts is not None and set_ts is not None:
        rise_epoch = rise_ts.tt  # type: ignore[attr-defined]
        set_epoch = set_ts.tt  # type: ignore[attr-defined]
        delta_days = set_epoch - rise_epoch
        daylight_mins = max(0, int(round(delta_days * 1440)))
    elif rise_ts is None and set_ts is None:
        # No rise and no set: determine polar day vs polar night.
        # Skyfield returns empty events — not an exception — for polar regions.
        # Check altitude at station-local noon.
        t_noon = ts.utc(d.year, d.month, d.day, 12, 0, 0)  # type: ignore[call-arg]
        astrometric = observer.at(t_noon).observe(sun)  # type: ignore[attr-defined]
        apparent = astrometric.apparent()  # type: ignore[attr-defined]
        alt, _az, _dist = apparent.altaz()  # type: ignore[attr-defined]
        if alt.degrees > 0:  # type: ignore[attr-defined]
            daylight_mins = 1440  # polar day
        else:
            daylight_mins = 0  # polar night

    # --- Civil twilight ---
    dawn_time: str | None = None
    dusk_time: str | None = None

    f_twilight = almanac.dark_twilight_day(eph, location)  # type: ignore[arg-type]
    times_tw, events_tw = almanac.find_discrete(t0, t1, f_twilight)  # type: ignore[arg-type]
    # Event values: 0=dark night, 1=astronomical twilight, 2=nautical,
    # 3=civil, 4=day.  Civil dawn = transition 2→3 or 3→4; dusk = 4→3 or 3→2.
    # We want the civil twilight boundaries: sun at -6°.
    # Dawn: first event where state transitions TO ≥ 3.
    # Dusk: last event where state transitions FROM ≥ 3.
    for t, e in zip(times_tw, events_tw, strict=False):
        if e >= 3 and dawn_time is None:
            dawn_time = _to_utc_z(t)
    for t, e in zip(reversed(times_tw), reversed(events_tw), strict=False):  # type: ignore[call-overload]
        if e >= 3 and dusk_time is None:
            dusk_time = _to_utc_z(t)

    # --- Solar noon position (azimuth / altitude / RA / Dec) ---
    azimuth: float | None = None
    altitude_deg: float | None = None
    ra_hours: float | None = None
    dec_deg: float | None = None
    transit_time: str | None = None

    t_noon = ts.utc(d.year, d.month, d.day, 12, 0, 0)  # type: ignore[call-arg]
    astrometric = observer.at(t_noon).observe(sun)  # type: ignore[attr-defined]
    apparent = astrometric.apparent()  # type: ignore[attr-defined]
    alt_obj, az_obj, _dist = apparent.altaz()  # type: ignore[attr-defined]
    azimuth = round(float(az_obj.degrees), 2)  # type: ignore[attr-defined]
    altitude_deg = round(float(alt_obj.degrees), 2)  # type: ignore[attr-defined]
    ra_obj, dec_obj, _dist2 = apparent.radec()  # type: ignore[attr-defined]
    ra_hours = round(float(ra_obj.hours), 4)  # type: ignore[attr-defined]
    dec_deg = round(float(dec_obj.degrees), 4)  # type: ignore[attr-defined]

    # Transit: find culmination (highest altitude) between t0 and t1.
    f_transit = almanac.meridian_transits(eph, sun, location)  # type: ignore[arg-type]
    t_transits, _ = almanac.find_discrete(t0, t1, f_transit)  # type: ignore[arg-type]
    if len(t_transits) > 0:
        transit_time = _to_utc_z(t_transits[0])

    # --- Next equinox / solstice ---
    next_equinox: str | None = None
    next_solstice: str | None = None

    t_start = ts.utc(d.year, d.month, d.day)  # type: ignore[call-arg]
    # Search up to 2 years out to be safe.
    t_end_yr = ts.utc(d.year + 2, d.month, d.day)  # type: ignore[call-arg]
    f_seasons = almanac.seasons(eph)  # type: ignore[arg-type]
    t_seasons, events_seasons = almanac.find_discrete(t_start, t_end_yr, f_seasons)  # type: ignore[arg-type]
    # Events: 0=vernal equinox, 1=summer solstice, 2=autumnal equinox, 3=winter solstice.
    for t, e in zip(t_seasons, events_seasons, strict=False):
        if e in (0, 2) and next_equinox is None:
            next_equinox = _to_utc_z(t)
        if e in (1, 3) and next_solstice is None:
            next_solstice = _to_utc_z(t)
        if next_equinox and next_solstice:
            break

    # --- Delta vs yesterday ---
    delta: int | None = None
    if include_delta and yesterday_daylight is not None:
        delta = daylight_mins - yesterday_daylight

    return SunInfo(
        rise=rise_time,
        set=set_time,
        transit=transit_time,
        civil_twilight_dawn=dawn_time,
        civil_twilight_dusk=dusk_time,
        azimuth=azimuth,
        altitude=altitude_deg,
        right_ascension=ra_hours,
        declination=dec_deg,
        daylight_minutes=daylight_mins,
        daylight_delta_vs_yesterday_minutes=delta,
        next_equinox=next_equinox,
        next_solstice=next_solstice,
    )


def _compute_moon_for_date(
    ts: object,
    eph: object,
    d: date,
    location: object,
    station_tz: str,
    include_next_phases: bool = True,
) -> MoonInfo:
    """Compute moon info for a single date.

    Args:
        ts: Skyfield Timescale.
        eph: Loaded ephemeris.
        d: The station-local calendar date.
        location: Skyfield WGS84 geographic position.
        station_tz: IANA timezone identifier — used to build a station-local
            midnight-to-midnight window (F2 fix).
        include_next_phases: Whether to search for next full/new moon.
    """
    # Station-local day window (F2 fix — was UTC midnight before).
    t0, t1 = _station_local_window(ts, d, station_tz)

    moon = eph["moon"]  # type: ignore[index]
    sun = eph["sun"]  # type: ignore[index]
    earth = eph["earth"]  # type: ignore[index]

    observer = earth + location  # type: ignore[operator]

    # --- Moon rise/set ---
    # Skyfield returns empty events for days when the moon does not rise/set;
    # no exception is raised.  The try/except removed here was masking real bugs.
    rise_time: str | None = None
    set_time: str | None = None
    transit_time: str | None = None

    f_moon = almanac.risings_and_settings(eph, moon, location)  # type: ignore[arg-type]
    times_m, events_m = almanac.find_discrete(t0, t1, f_moon)  # type: ignore[arg-type]
    for t, e in zip(times_m, events_m, strict=False):
        if e == 1 and rise_time is None:
            rise_time = _to_utc_z(t)
        elif e == 0 and set_time is None:
            set_time = _to_utc_z(t)

    f_moon_transit = almanac.meridian_transits(eph, moon, location)  # type: ignore[arg-type]
    t_transits_m, _ = almanac.find_discrete(t0, t1, f_moon_transit)  # type: ignore[arg-type]
    if len(t_transits_m) > 0:
        transit_time = _to_utc_z(t_transits_m[0])

    # --- Moon position at local noon ---
    import math

    t_noon = ts.utc(d.year, d.month, d.day, 12, 0, 0)  # type: ignore[call-arg]
    astrometric = observer.at(t_noon).observe(moon)  # type: ignore[attr-defined]
    apparent = astrometric.apparent()  # type: ignore[attr-defined]
    alt_obj, az_obj, _dist = apparent.altaz()  # type: ignore[attr-defined]
    azimuth = round(float(az_obj.degrees), 2)  # type: ignore[attr-defined]
    altitude_deg = round(float(alt_obj.degrees), 2)  # type: ignore[attr-defined]
    ra_obj, dec_obj, _dist2 = apparent.radec()  # type: ignore[attr-defined]
    ra_hours = round(float(ra_obj.hours), 4)  # type: ignore[attr-defined]
    dec_deg = round(float(dec_obj.degrees), 4)  # type: ignore[attr-defined]

    # Phase angle via ecliptic frame.
    sun_ecl = earth.at(t_noon).observe(sun).apparent().frame_latlon(ecliptic_frame)  # type: ignore[attr-defined]
    moon_ecl = earth.at(t_noon).observe(moon).apparent().frame_latlon(ecliptic_frame)  # type: ignore[attr-defined]
    sun_lon = float(sun_ecl[1].degrees)  # type: ignore[index]
    moon_lon = float(moon_ecl[1].degrees)  # type: ignore[index]
    phase_angle = (moon_lon - sun_lon) % 360.0
    phase_name = _phase_name_from_angle(phase_angle)

    # Illumination: cos²((phase_angle - 180°) / 2) × 100
    # For phase_angle: 0° = new (0%), 180° = full (100%).
    illum = math.cos(math.radians((phase_angle - 180.0) / 2.0)) ** 2 * 100.0
    illumination_percent = round(max(0.0, min(100.0, illum)), 1)

    # --- Next full and new moon ---
    next_full_moon: str | None = None
    next_new_moon: str | None = None
    if include_next_phases:
        t_search_start = ts.utc(d.year, d.month, d.day)  # type: ignore[call-arg]
        t_search_end = ts.utc(d.year + 1, d.month, d.day)  # type: ignore[call-arg]
        f_phases = almanac.moon_phases(eph)  # type: ignore[arg-type]
        t_phases, events_phases = almanac.find_discrete(t_search_start, t_search_end, f_phases)  # type: ignore[arg-type]
        # Events: 0=new moon, 1=first quarter, 2=full moon, 3=last quarter.
        for t, e in zip(t_phases, events_phases, strict=False):
            if e == 2 and next_full_moon is None:
                next_full_moon = _to_utc_z(t)
            if e == 0 and next_new_moon is None:
                next_new_moon = _to_utc_z(t)
            if next_full_moon and next_new_moon:
                break

    return MoonInfo(
        rise=rise_time,
        set=set_time,
        transit=transit_time,
        azimuth=azimuth,
        altitude=altitude_deg,
        right_ascension=ra_hours,
        declination=dec_deg,
        phase_name=phase_name,
        illumination_percent=illumination_percent,
        next_full_moon=next_full_moon,
        next_new_moon=next_new_moon,
    )


# ---------------------------------------------------------------------------
# Public compute functions
# ---------------------------------------------------------------------------


def compute_almanac(
    d: date,
    lat: float,
    lon: float,
    alt_m: float,
    station_tz: str = "UTC",
) -> AlmanacDay:
    """Compute full almanac snapshot for a single date.

    Args:
        d: The station-local calendar date.
        lat: Station latitude (decimal degrees, signed).
        lon: Station longitude (decimal degrees, signed).
        alt_m: Station altitude in metres above sea level.
        station_tz: IANA timezone identifier (ADR-020).  The day window is
            built as [d 00:00 station-local, d+1 00:00 station-local] converted
            to UTC before passing to Skyfield.  Defaulting to UTC is safe for
            the polar cases and for stations in UTC but wrong for EDT/PDT etc.
            Always pass the real station TZ in production (F2 fix).

    Returns:
        AlmanacDay with sun and moon info.
    """
    ts, eph = get_ts_eph()
    location = wgs84.latlon(lat, lon, elevation_m=alt_m)  # type: ignore[call-arg]

    # Compute yesterday's daylight for the daylightDeltaVsYesterdayMinutes field.
    # "Yesterday" is the station-local calendar day before d — not UTC yesterday.
    yesterday = d - timedelta(days=1)
    yesterday_sun = _compute_sun_for_date(
        ts, eph, yesterday, location, station_tz=station_tz
    )

    sun_info = _compute_sun_for_date(
        ts,
        eph,
        d,
        location,
        station_tz=station_tz,
        include_delta=True,
        yesterday_daylight=yesterday_sun.daylight_minutes,
    )
    moon_info = _compute_moon_for_date(
        ts, eph, d, location, station_tz=station_tz
    )

    return AlmanacDay(
        date_str=d.isoformat(),
        sun=sun_info,
        moon=moon_info,
    )


def compute_current_sun_altitude(lat: float, lon: float, alt_m: float) -> float | None:
    """Return the sun's current altitude in degrees above/below the horizon.

    Positive = above the horizon (daytime), negative = below (night).
    Returns None only when the ephemeris is not loaded.

    Uses the same ephemeris loading pattern as compute_almanac() — no
    duplication of de421.bsp loading logic.
    """
    try:
        ts, eph = get_ts_eph()
    except RuntimeError:
        return None

    location = wgs84.latlon(lat, lon, elevation_m=alt_m)  # type: ignore[call-arg]
    earth = eph["earth"]  # type: ignore[index]
    sun = eph["sun"]  # type: ignore[index]
    observer = earth + location  # type: ignore[operator]
    t_now = ts.now()  # type: ignore[attr-defined]
    astrometric = observer.at(t_now).observe(sun)  # type: ignore[attr-defined]
    apparent = astrometric.apparent()  # type: ignore[attr-defined]
    alt_obj, _az, _dist = apparent.altaz()  # type: ignore[attr-defined]
    return round(float(alt_obj.degrees), 4)  # type: ignore[attr-defined]


def compute_sun_times_year(
    year: int,
    lat: float,
    lon: float,
    alt_m: float,
    station_tz: str = "UTC",
) -> list[SunDay]:
    """Compute sunrise / sunset / daylight for every day of a year.

    Args:
        year: The station-local calendar year.
        lat, lon, alt_m: Station location.
        station_tz: IANA timezone identifier.  The year loop iterates over
            station-local calendar days; the day window for each day is
            station-local midnight-to-midnight (F2 fix).

    Returns:
        List of SunDay, one per station-local calendar day (365 or 366 entries).
    """
    ts, eph = get_ts_eph()
    location = wgs84.latlon(lat, lon, elevation_m=alt_m)  # type: ignore[call-arg]

    results: list[SunDay] = []
    d = date(year, 1, 1)
    while d.year == year:
        sun_info = _compute_sun_for_date(
            ts, eph, d, location, station_tz=station_tz
        )
        results.append(
            SunDay(
                date_str=d.isoformat(),
                sunrise=sun_info.rise,
                sunset=sun_info.set,
                daylight_minutes=sun_info.daylight_minutes,
            )
        )
        d += timedelta(days=1)

    return results


def compute_moon_phases(
    year: int,
    lat: float,
    lon: float,
    month: int | None = None,
    station_tz: str = "UTC",
) -> list[MoonDay]:
    """Compute moon phase name + illumination for each day of a month or year.

    Args:
        year: The station-local calendar year.
        lat, lon: Station location (altitude doesn't affect phase angle).
        month: If provided, only that station-local month.  None = full year.
        station_tz: IANA timezone identifier.  The day window for each day
            is station-local midnight-to-midnight (F2 fix).

    Returns:
        List of MoonDay, one per station-local calendar day.
    """
    ts, eph = get_ts_eph()
    # Phase angle is geocentric — use a reference earth position.
    # For illumination percent the location doesn't matter; we still pass
    # a location for rise/set (not computed here) consistency.
    location = wgs84.latlon(lat, lon)  # type: ignore[call-arg]

    results: list[MoonDay] = []

    if month is not None:
        start = date(year, month, 1)
        # Compute end of month.
        import calendar
        _, last_day = calendar.monthrange(year, month)
        end = date(year, month, last_day)
    else:
        start = date(year, 1, 1)
        end = date(year, 12, 31)

    d = start
    while d <= end:
        moon_info = _compute_moon_for_date(
            ts, eph, d, location, station_tz=station_tz, include_next_phases=False
        )
        results.append(
            MoonDay(
                date_str=d.isoformat(),
                phase_name=moon_info.phase_name or "new",
                illumination_percent=moon_info.illumination_percent or 0.0,
            )
        )
        d += timedelta(days=1)

    return results
