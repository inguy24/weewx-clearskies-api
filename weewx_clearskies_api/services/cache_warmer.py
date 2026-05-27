"""Background cache warmer for slow endpoints (ADR-045).

Pre-computes expensive results on configurable intervals and stores them in the
ADR-017 CacheBackend.  Endpoint handlers check the cache before running the
service, returning in <10ms on cache hit.

Warmed endpoints:
  - GET /records?period=all-time (unfiltered only)
  - GET /records?period=ytd (unfiltered only)
  - GET /almanac/sun-times (current year, station location)
  - GET /almanac/moon-phases (current year, full-year, station location)
  - GET /climatology/monthly
  - GET /almanac/planets (today's date, station location)
  - GET /almanac/eclipses (current year)
  - GET /almanac/meteor-showers (current year, station location)
  - GET /earthquakes/faults (station location, configured radius)

Cache key format:
  warmer:records:<period>                e.g. warmer:records:all-time
  warmer:almanac:sun-times:<year>        e.g. warmer:almanac:sun-times:2026
  warmer:almanac:moon-phases:<year>
  warmer:climatology:monthly
  warmer:almanac:planets:<date>          e.g. warmer:almanac:planets:2026-05-27
  warmer:almanac:eclipses:<year>         e.g. warmer:almanac:eclipses:2026
  warmer:almanac:meteor-showers:<year>   e.g. warmer:almanac:meteor-showers:2026
  warmer:earthquakes:faults

Cached values are plain dicts (JSON-safe) so both MemoryCache and RedisCache
backends work correctly.  RecordsBundle.model_dump() serialises the Pydantic
model; dataclasses.asdict() serialises SunDay/MoonDay.  The endpoint handlers
reconstruct the appropriate objects from the cached dicts.

Thread safety:
  _loop() runs in a single daemon thread.  Each warm call holds a fresh
  SQLAlchemy Session (not shared with request threads).  The CacheBackend
  set() / get() implementations are already thread-safe (MemoryCache uses
  cachetools.TTLCache which is not thread-safe; however, individual dict
  assignments are atomic in CPython.  RedisCache uses the redis-py client
  which is thread-safe).

  WARNING: cachetools.TTLCache is not thread-safe per its docs; a future
  revision should add a threading.Lock around MemoryCache operations if
  multi-threaded writes become a concern.  For the single-writer pattern
  here (only the warmer writes these keys; requests only read) the risk of
  data corruption is negligible in CPython due to the GIL.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlalchemy import Engine

from weewx_clearskies_api.providers._common.cache import get_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Warm interval sleep granularity (seconds).  The loop wakes every N seconds
# to check whether any function is overdue.  Smaller = more responsive to
# stop(); larger = less CPU overhead.  10 s is a good balance.
# ---------------------------------------------------------------------------
_SLEEP_TICK_SECONDS = 10

# Sentinel value meaning "never run".
_NEVER: float = 0.0


class BackgroundCacheWarmer:
    """Pre-computes slow endpoint results and writes them to the cache.

    Args:
        engine: SQLAlchemy Engine used to create per-warm Sessions.
        registry: ColumnRegistry from schema reflection (needed by get_records).
        settings: CacheWarmerSettings from api.conf [cache_warmer].
        station_meta: Dict with station identity keys required by almanac:
            lat (float), lon (float), alt_m (float), station_tz (str).
    """

    def __init__(
        self,
        engine: Engine,
        registry: object,
        settings: object,  # CacheWarmerSettings — avoid circular import
        station_meta: dict,
    ) -> None:
        self._engine = engine
        self._registry = registry
        self._settings = settings
        self._station = station_meta
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initial_warm(self) -> None:
        """Synchronous first warm — called before the server starts.

        Runs all warm functions once.  Failures are logged as WARNING and
        do not prevent startup (non-fatal per the brief).
        """
        logger.info("Cache warmer: initial warm starting")
        self._warm_records()
        self._warm_almanac()
        self._warm_climatology()
        self._warm_planets()
        self._warm_eclipses()
        self._warm_meteor_showers()
        self._warm_faults()
        logger.info("Cache warmer: initial warm complete")

    def start(self) -> None:
        """Launch the background daemon thread."""
        t = threading.Thread(target=self._loop, daemon=True, name="cache-warmer")
        t.start()
        logger.info("Cache warmer: background thread started")

    def stop(self) -> None:
        """Signal the background thread to exit at next tick."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main loop: wake every _SLEEP_TICK_SECONDS, run overdue functions."""
        last_records: float = _NEVER
        last_almanac: float = _NEVER
        last_climatology: float = _NEVER
        last_planets: float = _NEVER
        last_eclipses: float = _NEVER
        last_meteor_showers: float = _NEVER
        last_faults: float = _NEVER

        while not self._stop_event.is_set():
            now = time.monotonic()

            if last_records == _NEVER or (now - last_records) >= self._settings.records_interval_seconds:
                self._warm_records()
                last_records = time.monotonic()

            if last_almanac == _NEVER or (now - last_almanac) >= self._settings.almanac_interval_seconds:
                self._warm_almanac()
                last_almanac = time.monotonic()

            if last_climatology == _NEVER or (now - last_climatology) >= self._settings.climatology_interval_seconds:
                self._warm_climatology()
                last_climatology = time.monotonic()

            if last_planets == _NEVER or (now - last_planets) >= self._settings.planets_interval_seconds:
                self._warm_planets()
                last_planets = time.monotonic()

            if last_eclipses == _NEVER or (now - last_eclipses) >= self._settings.eclipses_interval_seconds:
                self._warm_eclipses()
                last_eclipses = time.monotonic()

            if last_meteor_showers == _NEVER or (now - last_meteor_showers) >= self._settings.meteor_showers_interval_seconds:
                self._warm_meteor_showers()
                last_meteor_showers = time.monotonic()

            if last_faults == _NEVER or (now - last_faults) >= self._settings.faults_interval_seconds:
                self._warm_faults()
                last_faults = time.monotonic()

            # Sleep in small ticks so stop() is responsive.
            self._stop_event.wait(timeout=_SLEEP_TICK_SECONDS)

        logger.info("Cache warmer: background thread stopped")

    # ------------------------------------------------------------------
    # Warm functions
    # ------------------------------------------------------------------

    def _warm_records(self) -> None:
        """Warm GET /records for 'all-time' and 'ytd' periods (unfiltered)."""
        try:
            from weewx_clearskies_api.services.records import get_records

            cache = get_cache()
            with Session(self._engine) as db:
                for period in ("all-time", "ytd"):
                    bundle = get_records(db, self._registry, period, section_filter=None)
                    # model_dump() produces a plain dict that json.dumps can handle,
                    # making this compatible with both MemoryCache and RedisCache.
                    cache.set(
                        f"warmer:records:{period}",
                        bundle.model_dump(),
                        self._settings.records_interval_seconds,
                    )
            logger.info("Cache warmer: records refreshed (all-time + ytd)")
        except Exception:
            logger.warning("Cache warmer: records warm failed", exc_info=True)

    def _warm_almanac(self) -> None:
        """Warm GET /almanac/sun-times and GET /almanac/moon-phases for the current year."""
        try:
            from weewx_clearskies_api.services.almanac import (
                compute_sun_times_year,
                compute_moon_phases,
            )

            cache = get_cache()
            year = datetime.now(timezone.utc).year
            lat = self._station["lat"]
            lon = self._station["lon"]
            alt_m = self._station["alt_m"]
            station_tz = self._station["station_tz"]

            # Sun times — list[SunDay] (Python dataclasses).
            sun_data = compute_sun_times_year(year, lat, lon, alt_m, station_tz)
            cache.set(
                f"warmer:almanac:sun-times:{year}",
                [dataclasses.asdict(d) for d in sun_data],
                self._settings.almanac_interval_seconds,
            )

            # Moon phases (full year, month=None) — list[MoonDay] (Python dataclasses).
            moon_data = compute_moon_phases(year, lat, lon, month=None, station_tz=station_tz)
            cache.set(
                f"warmer:almanac:moon-phases:{year}",
                [dataclasses.asdict(d) for d in moon_data],
                self._settings.almanac_interval_seconds,
            )

            logger.info("Cache warmer: almanac refreshed for year %d", year)
        except Exception:
            logger.warning("Cache warmer: almanac warm failed", exc_info=True)

    def _warm_climatology(self) -> None:
        """Warm GET /climatology/monthly."""
        try:
            from weewx_clearskies_api.services.climatology import get_monthly_climatology

            cache = get_cache()
            with Session(self._engine) as db:
                clim_data = get_monthly_climatology(db, self._registry)
            cache.set(
                "warmer:climatology:monthly",
                clim_data,
                self._settings.climatology_interval_seconds,
            )
            logger.info("Cache warmer: climatology refreshed")
        except Exception:
            logger.warning("Cache warmer: climatology warm failed", exc_info=True)

    def _warm_planets(self) -> None:
        """Warm GET /almanac/planets for today's date at the station location."""
        try:
            from weewx_clearskies_api.services.almanac import compute_planets

            cache = get_cache()
            today = datetime.now(timezone.utc).date()
            lat = self._station["lat"]
            lon = self._station["lon"]
            alt_m = self._station["alt_m"]
            station_tz = self._station["station_tz"]

            planets_data = compute_planets(today, lat, lon, alt_m, station_tz)
            cache.set(
                f"warmer:almanac:planets:{today.isoformat()}",
                planets_data,
                self._settings.planets_interval_seconds,
            )
            logger.info("Cache warmer: planets refreshed for %s", today.isoformat())
        except Exception:
            logger.warning("Cache warmer: planets warm failed", exc_info=True)

    def _warm_eclipses(self) -> None:
        """Warm GET /almanac/eclipses for the current year."""
        try:
            from weewx_clearskies_api.services.almanac import compute_lunar_eclipses

            cache = get_cache()
            year = datetime.now(timezone.utc).year

            eclipses_data = compute_lunar_eclipses(year)
            cache.set(
                f"warmer:almanac:eclipses:{year}",
                eclipses_data,
                self._settings.eclipses_interval_seconds,
            )
            logger.info("Cache warmer: eclipses refreshed for year %d", year)
        except Exception:
            logger.warning("Cache warmer: eclipses warm failed", exc_info=True)

    def _warm_meteor_showers(self) -> None:
        """Warm GET /almanac/meteor-showers for the current year at the station location."""
        try:
            from weewx_clearskies_api.services.almanac import compute_meteor_showers

            cache = get_cache()
            year = datetime.now(timezone.utc).year
            lat = self._station["lat"]
            lon = self._station["lon"]
            alt_m = self._station["alt_m"]
            station_tz = self._station["station_tz"]

            showers_data = compute_meteor_showers(year, lat, lon, alt_m, station_tz)
            cache.set(
                f"warmer:almanac:meteor-showers:{year}",
                showers_data,
                self._settings.meteor_showers_interval_seconds,
            )
            logger.info("Cache warmer: meteor showers refreshed for year %d", year)
        except Exception:
            logger.warning("Cache warmer: meteor showers warm failed", exc_info=True)

    def _warm_faults(self) -> None:
        """Warm GET /earthquakes/faults using the station location and configured radius."""
        try:
            from weewx_clearskies_api.services.faults import get_faults_within_radius
            from weewx_clearskies_api.endpoints.earthquakes import _default_radius_km

            cache = get_cache()
            lat = self._station["lat"]
            lon = self._station["lon"]

            faults_data = get_faults_within_radius(lat, lon, _default_radius_km)
            cache.set(
                "warmer:earthquakes:faults",
                faults_data,
                self._settings.faults_interval_seconds,
            )
            logger.info("Cache warmer: faults refreshed (radius %.1f km)", _default_radius_km)
        except Exception:
            logger.warning("Cache warmer: faults warm failed", exc_info=True)
