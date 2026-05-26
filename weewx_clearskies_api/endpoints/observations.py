"""Observation endpoints: GET /current and GET /archive.

Per ADR-018: URL-path versioned under /api/v1/.
Per ADR-012: read-only, per-request sessions via get_db_session().
Per ADR-019: units block embedded in every response, no server-side conversion.
Per ADR-020: UTC ISO-8601 with Z on the wire.
Per security-baseline §3.5: query params validated via Pydantic with
  ConfigDict(extra="forbid"), enforced through Depends() + model_validate()
  on the raw query-param dict so FastAPI's routing layer does not silently
  discard unknown keys before the model sees them.

Conditions blending engine (Phase 0B):
  wire_conditions_settings() is called from __main__.py after settings load.
  It stores the engine mode ("auto" | "provider" | "off") and the station
  coordinates at module level so the GET /current handler can call
  derive_conditions_text() without a DB hit per request.

ruff: noqa: N815  (canonical field names are weewx camelCase per ADR-010)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError
from sqlalchemy.orm import Session

from weewx_clearskies_api.db.registry import get_registry
from weewx_clearskies_api.db.session import get_db_session
from weewx_clearskies_api.models.params import ArchiveQueryParams

# Alias kept for backwards-compatibility with tests that import ArchiveParams
# from this module (test_archive_params.py).  The class is defined in
# models/params.py and re-exported here under the original name.
ArchiveParams = ArchiveQueryParams
from weewx_clearskies_api.models.responses import (
    ArchiveResponse,
    Observation,
    ObservationResponse,
    ProviderConditions,
)
from weewx_clearskies_api.services.archive import (
    decode_cursor,
    get_archive,
    get_current,
)
from weewx_clearskies_api.services.units import get_units_block

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level conditions engine settings (populated at startup)
# ---------------------------------------------------------------------------

_conditions_engine: str = "off"   # "auto" | "provider" | "off"
_conditions_lat: float | None = None
_conditions_lon: float | None = None
_conditions_alt_m: float | None = None
_conditions_target_unit: str = "US"
_conditions_timezone: str = "UTC"


def wire_conditions_settings(settings: object) -> None:
    """Wire conditions engine mode and station coordinates at startup.

    Called from __main__.py after load_station_metadata() and load_units_block()
    have both completed (so station lat/lon and target_unit are available).

    Reads:
      settings.conditions.engine   — "auto" | "provider" | "off"

    Station coordinates, timezone, and target_unit are read from
    services/station.py and services/units.py (same sources as
    endpoints/forecast.py).
    """
    global _conditions_engine, _conditions_lat, _conditions_lon  # noqa: PLW0603
    global _conditions_alt_m, _conditions_target_unit  # noqa: PLW0603
    global _conditions_timezone  # noqa: PLW0603

    conditions_settings = getattr(settings, "conditions", None)
    engine = getattr(conditions_settings, "engine", "off")
    _conditions_engine = engine if engine else "off"

    # Station metadata — may not be wired in tests; handled gracefully at
    # request time when _conditions_lat is None.
    try:
        from weewx_clearskies_api.services.station import get_station_info  # noqa: PLC0415
        station = get_station_info()
        _conditions_lat = station.latitude
        _conditions_lon = station.longitude
        _conditions_alt_m = station.altitude
        _conditions_timezone = station.timezone
    except RuntimeError:
        # Station not yet loaded — conditions engine will return None per-request.
        _conditions_lat = None
        _conditions_lon = None
        _conditions_alt_m = None
        _conditions_timezone = "UTC"

    # target_unit from the units service.
    try:
        from weewx_clearskies_api.services.units import get_target_unit  # noqa: PLC0415
        _conditions_target_unit = get_target_unit()
    except RuntimeError:
        _conditions_target_unit = "US"


# ---------------------------------------------------------------------------
# Provider conditions dispatch
# ---------------------------------------------------------------------------


def _fetch_provider_conditions() -> ProviderConditions | None:
    """Fetch current conditions from the configured forecast provider.

    Mirrors the provider dispatch pattern in endpoints/forecast.py:
      - Reads the capability registry for the "forecast" domain.
      - Imports the matching provider module and calls fetch_current_conditions().
      - Credentials are read from forecast.py's module-level variables (single
        source of truth — no duplicate wiring needed in observations.py).

    Returns None (never raises) — a conditions failure must never prevent
    GET /current from returning a valid observation.  ProviderError subclasses
    are caught and logged at WARNING level.  All other exceptions are caught
    and logged at ERROR level.

    Called only when _conditions_engine != "off" and station coords are wired.
    """
    from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
        get_provider_registry,
    )

    provider_registry = get_provider_registry()
    forecast_providers = [p for p in provider_registry if p.domain == "forecast"]

    if not forecast_providers:
        return None

    # Single source per deploy per ADR-007; take the first (and only) entry.
    provider_id = forecast_providers[0].provider_id

    try:
        if provider_id == "openmeteo":
            from weewx_clearskies_api.providers.forecast import openmeteo  # noqa: PLC0415
            return openmeteo.fetch_current_conditions(
                lat=_conditions_lat,
                lon=_conditions_lon,
                target_unit=_conditions_target_unit,
                timezone=_conditions_timezone,
            )
        elif provider_id == "nws":
            from weewx_clearskies_api.providers.forecast import nws as forecast_nws  # noqa: PLC0415
            from weewx_clearskies_api.endpoints.forecast import (  # noqa: PLC0415
                _nws_user_agent_contact,
            )
            return forecast_nws.fetch_current_conditions(
                lat=_conditions_lat,
                lon=_conditions_lon,
                target_unit=_conditions_target_unit,
                user_agent_contact=_nws_user_agent_contact,
            )
        elif provider_id == "aeris":
            from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415
            from weewx_clearskies_api.endpoints.forecast import (  # noqa: PLC0415
                _aeris_client_id,
                _aeris_client_secret,
            )
            return aeris.fetch_current_conditions(
                lat=_conditions_lat,
                lon=_conditions_lon,
                target_unit=_conditions_target_unit,
                client_id=_aeris_client_id,
                client_secret=_aeris_client_secret,
            )
        elif provider_id == "openweathermap":
            from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
            from weewx_clearskies_api.endpoints.forecast import (  # noqa: PLC0415
                _openweathermap_appid,
            )
            return openweathermap.fetch_current_conditions(
                lat=_conditions_lat,
                lon=_conditions_lon,
                target_unit=_conditions_target_unit,
                appid=_openweathermap_appid,
            )
        elif provider_id == "wunderground":
            from weewx_clearskies_api.providers.forecast import wunderground  # noqa: PLC0415
            from weewx_clearskies_api.endpoints.forecast import (  # noqa: PLC0415
                _wunderground_api_key,
                _wunderground_pws_station_id,
            )
            return wunderground.fetch_current_conditions(
                lat=_conditions_lat,
                lon=_conditions_lon,
                target_unit=_conditions_target_unit,
                api_key=_wunderground_api_key,
                pws_station_id=_wunderground_pws_station_id,
            )
        else:
            logger.warning(
                "Unknown forecast provider %r in capability registry; "
                "skipping conditions fetch.",
                provider_id,
            )
            return None

    except Exception:
        # ProviderError subclasses (QuotaExhausted, KeyInvalid, etc.) and any
        # other unexpected error are caught here.  Conditions failure must never
        # break GET /current — log and return None.
        logger.warning(
            "Provider conditions fetch failed for provider %r; "
            "continuing without provider conditions.",
            provider_id,
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc_z() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Dependency: parse + validate /archive query params
#
# FastAPI's routing layer extracts only the declared param names before
# populating a Depends() model, so extra="forbid" in a plain Depends(Model)
# call never fires for unknown HTTP keys.  The fix is to validate the full
# raw query-param dict via model_validate() inside a dependency function —
# that way Pydantic sees every key the client sent and raises ValidationError
# on any unknown one.  The ValidationError is caught here and surfaced as 400
# problem+json via FastAPI's RequestValidationError handler.
# ---------------------------------------------------------------------------


def _get_archive_params(request: Request) -> ArchiveQueryParams:
    """Dependency: parse and validate /archive query params from the raw dict.

    Calling model_validate(dict(request.query_params)) passes every HTTP key
    the client sent to Pydantic.  ConfigDict(extra="forbid") on ArchiveQueryParams
    then rejects any key not in the model's field set (security-baseline §3.5).
    """
    try:
        return ArchiveQueryParams.model_validate(dict(request.query_params))
    except ValidationError as exc:
        # Re-raise as RequestValidationError so the existing RFC 9457 error
        # handler shapes it as 400/422 problem+json.
        from fastapi.exceptions import RequestValidationError
        raise RequestValidationError(exc.errors()) from exc


# ---------------------------------------------------------------------------
# GET /current
# ---------------------------------------------------------------------------


@router.get(
    "/current",
    summary="Most recent observation",
    tags=["Observations"],
    response_model=ObservationResponse,
)
def get_current_endpoint(
    db: Annotated[Session, Depends(get_db_session)],
) -> ObservationResponse:
    """Return the most-recent archive row.

    Returns data: null when the archive is empty — not 404 (brief §1).

    When the conditions engine is enabled (engine != "off"), derives a
    human-readable weatherText string from the local sensor data.  See
    services/local_conditions.py for the blending logic.
    """
    registry = get_registry()
    units = get_units_block()

    observation = get_current(db, registry)

    # --- Conditions blending engine (Phase 0B) ---
    if observation is not None and _conditions_engine != "off":
        _apply_conditions_text(observation)

    return ObservationResponse(
        data=observation,
        units=units,
        source="weewx",
        generatedAt=_now_utc_z(),
    )


def _apply_conditions_text(observation: Observation) -> None:
    """Derive and attach weatherText to the observation (in-place).

    Separated from the route handler to keep the handler body readable.
    Errors are caught and logged — a failed conditions derivation must never
    cause GET /current to return a 500.

    Called only when _conditions_engine != "off" and observation is not None.
    """
    # Guard: station coordinates must have been wired at startup.
    if _conditions_lat is None or _conditions_lon is None or _conditions_alt_m is None:
        logger.debug(
            "Conditions engine enabled but station coordinates not wired; "
            "skipping weatherText derivation."
        )
        return

    try:
        # --- Sun altitude ---
        from weewx_clearskies_api.services.almanac import compute_current_sun_altitude  # noqa: PLC0415
        sun_alt = compute_current_sun_altitude(
            _conditions_lat, _conditions_lon, _conditions_alt_m
        )

        # Fetch current conditions from the configured forecast provider.
        provider_conditions: ProviderConditions | None = _fetch_provider_conditions()

        if _conditions_engine == "provider" and provider_conditions is not None:
            # "provider" mode: use provider weatherText verbatim (no local blending).
            if provider_conditions.weatherText:
                observation.weatherText = provider_conditions.weatherText
        else:
            # "auto" mode (default): blend local sensor data with provider.
            from weewx_clearskies_api.services.local_conditions import derive_conditions_text  # noqa: PLC0415
            result = derive_conditions_text(
                observation=observation,
                max_solar_rad=observation.maxSolarRad,
                provider_conditions=provider_conditions,
                sun_altitude=sun_alt,
                target_unit=_conditions_target_unit,
            )
            if result is not None:
                observation.weatherText = result

    except Exception:
        # Never surface a conditions-engine failure as a 500.
        logger.exception(
            "Conditions engine failed during weatherText derivation; "
            "returning observation without weatherText."
        )


# ---------------------------------------------------------------------------
# GET /archive
# ---------------------------------------------------------------------------


@router.get(
    "/archive",
    summary="Historical archive records",
    tags=["Observations"],
    response_model=ArchiveResponse,
)
def get_archive_endpoint(
    db: Annotated[Session, Depends(get_db_session)],
    params: Annotated[ArchiveQueryParams, Depends(_get_archive_params)],
) -> ArchiveResponse:
    """Return archive records within a time window.

    Supports raw / hour / day interval aggregation and cursor + page pagination.
    Unknown query parameters are rejected with 400 per security-baseline §3.5.
    """
    registry = get_registry()

    # Validate cursor if provided.
    if params.cursor is not None:
        try:
            decode_cursor(params.cursor)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid cursor: {exc}") from exc

    # Validate fields if provided.
    parsed_field_names: list[str] | None = None
    if params.fields is not None:
        names = [f.strip() for f in params.fields.split(",") if f.strip()]
        mapped_names = {
            info.canonical_name
            for info in registry.stock.values()
            if info.canonical_name is not None
        }
        unknown = [n for n in names if n not in mapped_names]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown field(s): {', '.join(unknown)}",
            )
        parsed_field_names = names

    units = get_units_block()

    try:
        records, page_info = get_archive(
            db=db,
            registry=registry,
            from_dt=params.from_,
            to_dt=params.to,
            interval=params.interval,
            fields=parsed_field_names,
            limit=params.limit,
            cursor=params.cursor,
            page=params.page,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ArchiveResponse(
        data=records,
        units=units,
        source="weewx",
        generatedAt=_now_utc_z(),
        page=page_info,
    )
