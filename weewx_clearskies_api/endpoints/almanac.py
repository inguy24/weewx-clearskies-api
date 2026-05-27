"""Almanac endpoints (3a-2).

GET /almanac              — sun + moon snapshot for one date
GET /almanac/sun-times    — year-long sunrise/sunset/daylight series
GET /almanac/moon-phases  — per-day moon-phase grid (month or full year)

All three are pure-compute: no DB hit, no provider dependency.
Params validated via Depends(_get_*_params) pattern per security-baseline §3.5.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from weewx_clearskies_api.models.params import (
    AlmanacQueryParams,
    MoonPhasesQueryParams,
    SunTimesQueryParams,
    PlanetsQueryParams,
    EclipsesQueryParams,
    MeteorShowersQueryParams,
    MoonNamesQueryParams,
)
from weewx_clearskies_api.models.responses import (
    AlmanacResponse,
    AlmanacSnapshot,
    EclipseResponse,
    LunarEclipseEntry,
    LunarEclipseList,
    MeteorShowerEntry,
    MeteorShowerList,
    MeteorShowerResponse,
    MoonNamesCalendar,
    MoonNamesResponse,
    MoonPhaseCalendar,
    MoonPhaseDay,
    MoonPhaseResponse,
    MoonSnapshot,
    PlanetEntry,
    PlanetResponse,
    PlanetVisibility,
    SpecialMoonEntry,
    SunSnapshot,
    SunTimesDay,
    SunTimesResponse,
    SunTimesSeries,
    utc_isoformat,
)
from weewx_clearskies_api.providers._common.cache import get_cache
from weewx_clearskies_api.services import almanac as almanac_svc
from weewx_clearskies_api.services.station import get_station_info

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Param wrapper functions (Pydantic + Depends pattern per security-baseline §3.5)
# ---------------------------------------------------------------------------


def _get_almanac_params(request: Request) -> AlmanacQueryParams:
    """Validate GET /almanac query parameters.  Rejects unknown keys."""
    try:
        return AlmanacQueryParams.model_validate(dict(request.query_params))
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def _get_sun_times_params(request: Request) -> SunTimesQueryParams:
    """Validate GET /almanac/sun-times query parameters."""
    try:
        return SunTimesQueryParams.model_validate(dict(request.query_params))
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def _get_moon_phases_params(request: Request) -> MoonPhasesQueryParams:
    """Validate GET /almanac/moon-phases query parameters."""
    try:
        return MoonPhasesQueryParams.model_validate(dict(request.query_params))
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def _get_planets_params(request: Request) -> PlanetsQueryParams:
    """Validate GET /almanac/planets query parameters."""
    try:
        return PlanetsQueryParams.model_validate(dict(request.query_params))
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def _get_eclipses_params(request: Request) -> EclipsesQueryParams:
    """Validate GET /almanac/eclipses query parameters."""
    try:
        return EclipsesQueryParams.model_validate(dict(request.query_params))
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def _get_meteor_showers_params(request: Request) -> MeteorShowersQueryParams:
    """Validate GET /almanac/meteor-showers query parameters."""
    try:
        return MeteorShowersQueryParams.model_validate(dict(request.query_params))
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def _get_moon_names_params(request: Request) -> MoonNamesQueryParams:
    """Validate GET /almanac/moon-names query parameters."""
    try:
        return MoonNamesQueryParams.model_validate(dict(request.query_params))
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _station_location() -> tuple[float, float, float]:
    """Return (lat, lon, altitude) from the cached station metadata.

    Altitude is in whatever unit weewx configured (feet or metres). Skyfield's
    wgs84.latlon elevation_m parameter expects metres.  We pass the value
    through as-is per ADR-019 (no server-side conversion).  The altitude is
    used only for the observer's horizon calculation — a few hundred feet vs
    metres makes negligible difference for rise/set times.
    """
    info = get_station_info()
    return info.latitude, info.longitude, info.altitude


def _current_year_in_station_tz() -> int:
    """Return the current calendar year in station-local time."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    info = get_station_info()
    try:
        zi = ZoneInfo(info.timezone)
        now = datetime.now(tz=zi)
    except ZoneInfoNotFoundError:
        now = datetime.now(tz=UTC)
    return now.year


def _today_in_station_tz() -> date:
    """Return today's date in station-local time."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    info = get_station_info()
    try:
        zi = ZoneInfo(info.timezone)
        now = datetime.now(tz=zi)
    except ZoneInfoNotFoundError:
        now = datetime.now(tz=UTC)
    return now.date()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/almanac", summary="Sun and moon snapshot", tags=["Almanac"])
def get_almanac(
    params: Annotated[AlmanacQueryParams, Depends(_get_almanac_params)],
) -> AlmanacResponse:
    """Sun and moon snapshot for a given date (Skyfield-computed, no DB hit)."""
    target_date = params.date if params.date is not None else _today_in_station_tz()
    lat, lon, alt = _station_location()
    station_tz = get_station_info().timezone

    day = almanac_svc.compute_almanac(target_date, lat, lon, alt, station_tz=station_tz)

    sun = SunSnapshot(
        rise=day.sun.rise,
        set=day.sun.set,
        transit=day.sun.transit,
        civilTwilightDawn=day.sun.civil_twilight_dawn,
        civilTwilightDusk=day.sun.civil_twilight_dusk,
        azimuth=day.sun.azimuth,
        altitude=day.sun.altitude,
        rightAscension=day.sun.right_ascension,
        declination=day.sun.declination,
        daylightMinutes=day.sun.daylight_minutes,
        daylightDeltaVsYesterdayMinutes=day.sun.daylight_delta_vs_yesterday_minutes,
        nextEquinox=day.sun.next_equinox,
        nextSolstice=day.sun.next_solstice,
    )
    moon = MoonSnapshot(
        rise=day.moon.rise,
        set=day.moon.set,
        transit=day.moon.transit,
        azimuth=day.moon.azimuth,
        altitude=day.moon.altitude,
        rightAscension=day.moon.right_ascension,
        declination=day.moon.declination,
        phaseName=day.moon.phase_name,
        illuminationPercent=day.moon.illumination_percent,
        nextFullMoon=day.moon.next_full_moon,
        nextNewMoon=day.moon.next_new_moon,
    )

    return AlmanacResponse(
        data=AlmanacSnapshot(date=day.date_str, sun=sun, moon=moon),
        generatedAt=utc_isoformat(datetime.now(tz=UTC)),
    )


@router.get("/almanac/sun-times", summary="Year-long sunrise/sunset series", tags=["Almanac"])
def get_sun_times(
    params: Annotated[SunTimesQueryParams, Depends(_get_sun_times_params)],
) -> SunTimesResponse:
    """Year-long sunrise / sunset / daylight series (no DB hit)."""
    year = params.year if params.year is not None else _current_year_in_station_tz()
    lat, lon, alt = _station_location()
    station_tz = get_station_info().timezone

    # Cache-check-first guard (ADR-045).  The warmer pre-computes the current
    # year for the station location; use it when the request matches.
    try:
        cached = get_cache().get(f"warmer:almanac:sun-times:{year}")
        if cached is not None:
            logger.debug("sun-times cache hit: year=%d", year)
            days = [
                SunTimesDay(
                    date=d["date_str"],
                    sunrise=d["sunrise"],
                    sunset=d["sunset"],
                    daylightMinutes=d["daylight_minutes"],
                )
                for d in cached
            ]
            return SunTimesResponse(
                data=SunTimesSeries(year=year, days=days),
                generatedAt=utc_isoformat(datetime.now(tz=UTC)),
            )
    except Exception:
        logger.debug("sun-times cache miss or error: year=%d", year, exc_info=True)

    days_raw = almanac_svc.compute_sun_times_year(year, lat, lon, alt, station_tz=station_tz)
    days = [
        SunTimesDay(
            date=d.date_str,
            sunrise=d.sunrise,
            sunset=d.sunset,
            daylightMinutes=d.daylight_minutes,
        )
        for d in days_raw
    ]

    return SunTimesResponse(
        data=SunTimesSeries(year=year, days=days),
        generatedAt=utc_isoformat(datetime.now(tz=UTC)),
    )


@router.get("/almanac/moon-phases", summary="Moon-phase calendar", tags=["Almanac"])
def get_moon_phases(
    params: Annotated[MoonPhasesQueryParams, Depends(_get_moon_phases_params)],
) -> MoonPhaseResponse:
    """Per-day moon-phase calendar for a month or full year (no DB hit)."""
    year = params.year if params.year is not None else _current_year_in_station_tz()
    month = params.month  # None = full year
    lat, lon, _alt = _station_location()
    station_tz = get_station_info().timezone

    # Cache-check-first guard (ADR-045).  The warmer pre-computes the full year
    # (month=None) only; per-month requests bypass the cache.
    if month is None:
        try:
            cached = get_cache().get(f"warmer:almanac:moon-phases:{year}")
            if cached is not None:
                logger.debug("moon-phases cache hit: year=%d", year)
                days = [
                    MoonPhaseDay(
                        date=d["date_str"],
                        phaseName=d["phase_name"],
                        illuminationPercent=d["illumination_percent"],
                    )
                    for d in cached
                ]
                return MoonPhaseResponse(
                    data=MoonPhaseCalendar(year=year, month=month, days=days),
                    generatedAt=utc_isoformat(datetime.now(tz=UTC)),
                )
        except Exception:
            logger.debug("moon-phases cache miss or error: year=%d", year, exc_info=True)

    days_raw = almanac_svc.compute_moon_phases(year, lat, lon, month, station_tz=station_tz)
    days = [
        MoonPhaseDay(
            date=d.date_str,
            phaseName=d.phase_name,
            illuminationPercent=d.illumination_percent,
        )
        for d in days_raw
    ]

    return MoonPhaseResponse(
        data=MoonPhaseCalendar(year=year, month=month, days=days),
        generatedAt=utc_isoformat(datetime.now(tz=UTC)),
    )


@router.get("/almanac/moon-names", summary="Special full moon names", tags=["Almanac"])
def get_moon_names(
    params: Annotated[MoonNamesQueryParams, Depends(_get_moon_names_params)],
) -> MoonNamesResponse:
    """Full moons for a year with traditional and special name annotations.

    Returns one entry per full moon: traditional name (Wolf, Snow, etc.),
    Harvest Moon, Blue Moon, Hunter's Moon, and Supermoon flags.
    """
    year = params.year if params.year is not None else _current_year_in_station_tz()

    moons_raw = almanac_svc.compute_special_moon_names(year)
    moons = [
        SpecialMoonEntry(
            date=m["date"],
            traditionalName=m["traditionalName"],
            isHarvestMoon=m["isHarvestMoon"],
            isBlueMoon=m["isBlueMoon"],
            isHuntersMoon=m["isHuntersMoon"],
            isSupermoon=m["isSupermoon"],
        )
        for m in moons_raw
    ]

    return MoonNamesResponse(
        data=MoonNamesCalendar(year=year, moons=moons),
        generatedAt=utc_isoformat(datetime.now(tz=UTC)),
    )


@router.get("/almanac/planets", summary="Planet visibility", tags=["Almanac"])
def get_planets(
    params: Annotated[PlanetsQueryParams, Depends(_get_planets_params)],
) -> PlanetResponse:
    """Evening/morning/all-night planet visibility for a given date.

    Returns Mercury through Saturn classified by visibility period.
    Only planets with apparent magnitude < 6.0 are included.
    """
    target_date = params.date if params.date is not None else _today_in_station_tz()
    lat, lon, alt = _station_location()
    station_tz = get_station_info().timezone

    # Cache-check-first guard (ADR-045).  The warmer pre-computes today's date
    # at the station location; use the cached result when the request matches.
    try:
        cache_key = f"warmer:almanac:planets:{target_date.isoformat()}"
        cached = get_cache().get(cache_key)
        if cached is not None:
            logger.debug("planets cache hit: date=%s", target_date.isoformat())
            visibility_raw = cached

            def _to_entries_cached(raw_list: list[dict]) -> list[PlanetEntry]:
                return [
                    PlanetEntry(
                        name=p["name"],
                        magnitude=p["magnitude"],
                        rise=p["rise"],
                        set=p["set"],
                        constellation=p["constellation"],
                    )
                    for p in raw_list
                ]

            return PlanetResponse(
                data=PlanetVisibility(
                    evening=_to_entries_cached(visibility_raw["evening"]),
                    morning=_to_entries_cached(visibility_raw["morning"]),
                    allNight=_to_entries_cached(visibility_raw["allNight"]),
                ),
                generatedAt=utc_isoformat(datetime.now(tz=UTC)),
            )
    except Exception:
        logger.debug("planets cache miss or error: date=%s", target_date.isoformat(), exc_info=True)

    visibility_raw = almanac_svc.compute_planets(
        target_date, lat, lon, alt, station_tz=station_tz
    )

    def _to_entries(raw_list: list[dict]) -> list[PlanetEntry]:
        return [
            PlanetEntry(
                name=p["name"],
                magnitude=p["magnitude"],
                rise=p["rise"],
                set=p["set"],
                constellation=p["constellation"],
            )
            for p in raw_list
        ]

    return PlanetResponse(
        data=PlanetVisibility(
            evening=_to_entries(visibility_raw["evening"]),
            morning=_to_entries(visibility_raw["morning"]),
            allNight=_to_entries(visibility_raw["allNight"]),
        ),
        generatedAt=utc_isoformat(datetime.now(tz=UTC)),
    )


@router.get("/almanac/eclipses", summary="Lunar eclipses", tags=["Almanac"])
def get_eclipses(
    params: Annotated[EclipsesQueryParams, Depends(_get_eclipses_params)],
) -> EclipseResponse:
    """Lunar eclipse dates and types for a given year.

    Uses skyfield.eclipselib; returns an empty list if unavailable.
    Types: penumbral, partial, total.
    """
    year = params.year if params.year is not None else _current_year_in_station_tz()

    # Cache-check-first guard (ADR-045).  The warmer pre-computes the current
    # year; use the cached result when the request matches.
    try:
        cache_key = f"warmer:almanac:eclipses:{year}"
        cached = get_cache().get(cache_key)
        if cached is not None:
            logger.debug("eclipses cache hit: year=%d", year)
            eclipses = [
                LunarEclipseEntry(date=e["date"], type=e["type"])
                for e in cached
            ]
            return EclipseResponse(
                data=LunarEclipseList(year=year, eclipses=eclipses),
                generatedAt=utc_isoformat(datetime.now(tz=UTC)),
            )
    except Exception:
        logger.debug("eclipses cache miss or error: year=%d", year, exc_info=True)

    eclipses_raw = almanac_svc.compute_lunar_eclipses(year)
    eclipses = [
        LunarEclipseEntry(date=e["date"], type=e["type"])
        for e in eclipses_raw
    ]

    return EclipseResponse(
        data=LunarEclipseList(year=year, eclipses=eclipses),
        generatedAt=utc_isoformat(datetime.now(tz=UTC)),
    )


@router.get("/almanac/meteor-showers", summary="Meteor shower viewing conditions", tags=["Almanac"])
def get_meteor_showers(
    params: Annotated[MeteorShowersQueryParams, Depends(_get_meteor_showers_params)],
) -> MeteorShowerResponse:
    """Meteor shower viewing conditions for a given year.

    Returns 12 major annual showers with radiant altitude, moon illumination,
    and a viewing conditions rating (excellent / good / fair / poor).
    """
    year = params.year if params.year is not None else _current_year_in_station_tz()
    lat, lon, alt = _station_location()
    station_tz = get_station_info().timezone

    # Cache-check-first guard (ADR-045).  The warmer pre-computes the current
    # year at the station location; use the cached result when available.
    try:
        cache_key = f"warmer:almanac:meteor-showers:{year}"
        cached = get_cache().get(cache_key)
        if cached is not None:
            logger.debug("meteor-showers cache hit: year=%d", year)
            showers = [
                MeteorShowerEntry(
                    name=s["name"],
                    peakDate=s["peakDate"],
                    zhr=s["zhr"],
                    radiantAltitudeDeg=s["radiantAltitudeDeg"],
                    moonIlluminationPercent=s["moonIlluminationPercent"],
                    viewingConditions=s["viewingConditions"],
                    parentBody=s["parentBody"],
                )
                for s in cached
            ]
            return MeteorShowerResponse(
                data=MeteorShowerList(year=year, showers=showers),
                generatedAt=utc_isoformat(datetime.now(tz=UTC)),
            )
    except Exception:
        logger.debug("meteor-showers cache miss or error: year=%d", year, exc_info=True)

    showers_raw = almanac_svc.compute_meteor_showers(
        year, lat, lon, alt, station_tz=station_tz
    )
    showers = [
        MeteorShowerEntry(
            name=s["name"],
            peakDate=s["peakDate"],
            zhr=s["zhr"],
            radiantAltitudeDeg=s["radiantAltitudeDeg"],
            moonIlluminationPercent=s["moonIlluminationPercent"],
            viewingConditions=s["viewingConditions"],
            parentBody=s["parentBody"],
        )
        for s in showers_raw
    ]

    return MeteorShowerResponse(
        data=MeteorShowerList(year=year, showers=showers),
        generatedAt=utc_isoformat(datetime.now(tz=UTC)),
    )
