"""GET /aqi/current and GET /aqi/history (ADR-013).

Behavior decision tree for /aqi/current per brief §per-endpoint spec:

  1. No AQI provider in capability registry → 200, data=null, source="none"
  2. Provider configured, fetch succeeds with valid AQI → 200 + AQIReading
  3. Provider configured, fetch succeeds but all values null → 200, data=null
  4. Network failure / 5xx after retries → 502 (TransientNetworkError → RFC 9457)
  5. Provider returns 429 → 503 + Retry-After (QuotaExhausted → RFC 9457)
  6. Provider returns 401/403 → 502 (KeyInvalid)
  7. Pydantic validation failure on wire model → 502 (ProviderProtocolError → RFC 9457)

/aqi/history reads from the weewx archive (ADR-013 corrected, P4-T3):
  Path A: AQI columns present in archive → returns historical AQIReading list.
  Path B: no [aqi.history] columns configured → returns empty list (not error).
  DB session injected via Depends(get_db_session).  AQIHistorySettings wired
  at startup via wire_aqi_settings().

No DB hit for /aqi/current.  AQI comes from the configured provider (Path B).

Operator lat/lon: from get_station_info() (services/station.py) per ADR-011
  (single-station scope).  No ?station= param.

Pydantic + Depends pattern (coding.md §1, security-baseline §3.5):
  Unknown query keys rejected with 422 via extra="forbid" + Depends wrapper.

Provider discovery: endpoint reads the capability registry at request time.
  _wire_providers_from_config() at startup registers the configured provider's
  CAPABILITY; this endpoint checks the registry for an "aqi" domain entry.
  Tests that need the openmeteo path call wire_providers([openmeteo.CAPABILITY]).

wire_aqi_settings (LC18/LC19):
  No-op for Open-Meteo (keyless, no credentials to wire).
  For Aeris: extracts client_id + client_secret from settings.aeris and stores
  in module-level _AERIS_CLIENT_ID + _AERIS_CLIENT_SECRET for dispatch (3b-10).
  Future-proof for keyed providers (3b-11 OWM, 3b-12 IQAir).

Units block (LC lead-call in brief §per-endpoint spec):
  Populated via get_units_block() / get_target_unit() from services/units.py.
  Same wiring as /forecast and /alerts.  No _default_units_block() helper
  exists as a shared utility — inline construction mirrors forecast.py.
  Flagged for follow-up DRY-extraction (brief §per-endpoint spec).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

import pydantic
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from sqlalchemy.orm import Session

from weewx_clearskies_api.config.settings import AQIHistorySettings
from weewx_clearskies_api.db.session import get_db_session
from weewx_clearskies_api.models.params import AQIHistoryQueryParams, AQIQueryParams
from weewx_clearskies_api.models.responses import (
    AQIHistoryResponse,
    AQIReading,
    AQIResponse,
    utc_isoformat,
)
from weewx_clearskies_api.providers._common.capability import get_provider_registry
from weewx_clearskies_api.services.aqi_history import get_aqi_history
from weewx_clearskies_api.services.station import get_station_info
from weewx_clearskies_api.services.units import get_units_block

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level credential wiring (populated at startup by wire_aqi_settings).
# Aeris: client_id + client_secret (3b-10).
# OWM: appid (3b-11 — provider-scoped per 3b-5 Q2; same env var as forecast/alerts OWM).
# IQAir: key (3b-12 — domain-scoped per Q1 user decision 2026-05-11; AQI-only provider).
# AQI history: column mapping from [aqi.history] section (P4-T3, ADR-013 corrected).
# ---------------------------------------------------------------------------

_AERIS_CLIENT_ID: str | None = None
_AERIS_CLIENT_SECRET: str | None = None
_OWM_APPID: str | None = None
_IQAIR_KEY: str | None = None
# AQI history settings — defaults to all-empty (Path B) until wired at startup.
_AQI_HISTORY_SETTINGS: AQIHistorySettings = AQIHistorySettings({})

# ---------------------------------------------------------------------------
# Depends wrappers — Pydantic + Depends pattern (coding.md §1)
# ---------------------------------------------------------------------------


def _get_aqi_params(request: Request) -> AQIQueryParams:
    """Extract and validate /aqi/current query parameters via Pydantic.

    Using Depends(model_validate(dict(request.query_params))) pattern so
    extra="forbid" actually fires for unknown query keys (coding.md §1,
    security-baseline §3.5).
    """
    try:
        return AQIQueryParams.model_validate(dict(request.query_params))
    except pydantic.ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def _get_aqi_history_params(request: Request) -> AQIHistoryQueryParams:
    """Extract and validate /aqi/history query parameters via Pydantic.

    Params are validated (invalid → 422) before the weewx archive query
    executes.  This ensures a coherent response: unknown keys get 422,
    valid params proceed to the archive query (Path A) or return an empty
    result (Path B — no AQI columns configured).
    """
    try:
        return AQIHistoryQueryParams.model_validate(dict(request.query_params))
    except pydantic.ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


# ---------------------------------------------------------------------------
# Credential / settings wiring (LC18)
# ---------------------------------------------------------------------------


def wire_aqi_settings(settings: object) -> None:
    """Wire AQI-related settings from the Settings object.

    For Open-Meteo (keyless): no-op for credentials — but AQI history settings
      are always wired from settings.aqi_history (P4-T3).
    For Aeris (3b-10): extracts client_id + client_secret from settings.forecast
      (provider-scoped per 3b-4 Q1 user decision; same [aeris] section as
      forecast/alerts Aeris) and stores in module-level _AERIS_CLIENT_ID +
      _AERIS_CLIENT_SECRET for the dispatch to pass to aeris.fetch().
    For OWM (3b-11): extracts openweathermap_appid from settings.forecast
      (provider-scoped per 3b-5 Q2 user decision; same env var as forecast/alerts OWM:
      WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID) and stores in _OWM_APPID.
    For IQAir (3b-12): extracts iqair_key from settings.aqi (domain-scoped per
      Q1 user decision 2026-05-11; IQAir is AQI-only, not multi-domain like Aeris/OWM)
      and stores in _IQAIR_KEY.
    AQI history (P4-T3): stores settings.aqi_history in _AQI_HISTORY_SETTINGS so
      get_aqi_history() can read archive column mappings at request time.
    """
    global _AERIS_CLIENT_ID, _AERIS_CLIENT_SECRET, _OWM_APPID, _IQAIR_KEY  # noqa: PLW0603
    global _AQI_HISTORY_SETTINGS  # noqa: PLW0603

    # Wire AQI history column settings (Path A operators configure these;
    # Path B operators get all-empty defaults — no error).
    aqi_history_section = getattr(settings, "aqi_history", None)
    if aqi_history_section is not None:
        _AQI_HISTORY_SETTINGS = aqi_history_section

    aqi_section = getattr(settings, "aqi", None)
    if aqi_section is None:
        return

    provider = getattr(aqi_section, "provider", None)

    if provider == "aeris":
        # Provider-scoped credentials per 3b-4 Q1 decision — same env vars as
        # forecast/alerts Aeris (WEEWX_CLEARSKIES_AERIS_CLIENT_ID +
        # WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET).  Settings class stores these on
        # the forecast + alerts section objects (no standalone [aeris] section in
        # Settings); access via settings.forecast which was first to wire these.
        # Brief LC18/LC19 said "settings.aeris.client_id" but Settings has no
        # aeris attribute — brief provenance note: brief was written assuming a
        # standalone section that doesn't exist in the Settings class.  Using
        # settings.forecast.aeris_client_id is the correct code path.
        forecast_section = getattr(settings, "forecast", None)
        if forecast_section is None:
            logger.error(
                "[aqi] provider = aeris but [forecast] settings section missing; "
                "credentials cannot be wired"
            )
            return

        _AERIS_CLIENT_ID = getattr(forecast_section, "aeris_client_id", None)
        _AERIS_CLIENT_SECRET = getattr(forecast_section, "aeris_client_secret", None)

        if not _AERIS_CLIENT_ID or not _AERIS_CLIENT_SECRET:
            logger.error(
                "[aqi] provider = aeris but WEEWX_CLEARSKIES_AERIS_CLIENT_ID/"
                "WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET env vars missing; "
                "capability still registered but /aqi/current will return 502 until wired"
            )

    elif provider == "openweathermap":
        # Provider-scoped credential per 3b-5 Q2 user decision — same env var as
        # forecast/alerts OWM (WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID).
        # Settings class stores this on ForecastSettings.openweathermap_appid;
        # no standalone [openweathermap] section in Settings.
        forecast_section = getattr(settings, "forecast", None)
        if forecast_section is None:
            logger.error(
                "[aqi] provider = openweathermap but [forecast] settings section missing; "
                "credentials cannot be wired"
            )
            return

        _OWM_APPID = getattr(forecast_section, "openweathermap_appid", None)

        if not _OWM_APPID:
            logger.error(
                "[aqi] provider = openweathermap but "
                "WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID env var missing; "
                "capability still registered but /aqi/current will return 502 until wired"
            )

    elif provider == "iqair":
        # Domain-scoped credential per Q1 user decision 2026-05-11 — IQAir is
        # AQI-only (not multi-domain like Aeris/OWM), so the credential lives
        # directly on AQISettings.iqair_key (populated from
        # WEEWX_CLEARSKIES_IQAIR_KEY env var at AQISettings.__init__ time).
        _IQAIR_KEY = getattr(aqi_section, "iqair_key", None)

        if not _IQAIR_KEY:
            logger.error(
                "[aqi] provider = iqair but WEEWX_CLEARSKIES_IQAIR_KEY env var missing; "
                "capability still registered but /aqi/current will return 502 until wired"
            )

    # Open-Meteo is keyless (auth_required=()); nothing to wire.


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/aqi/current",
    summary="Current AQI reading",
    tags=["AQI"],
    response_model=AQIResponse,
)
def get_aqi_current(
    params: Annotated[AQIQueryParams, Depends(_get_aqi_params)],
) -> AQIResponse:
    """Return the current AQI reading from the configured provider.

    Reads the capability registry for the aqi domain at request time.
    Returns AQIResponse(data=None, source="none") when no provider is registered.
    Cache integration happens transparently in the provider module's fetch().
    """
    now_str = utc_isoformat(datetime.now(tz=UTC))

    # --- Assemble units block (inline; no shared helper — flag for DRY-extraction) ---
    try:
        units = get_units_block()
    except RuntimeError as exc:
        # Defense-in-depth: units should always be wired before uvicorn starts.
        logger.error(
            "Units block not available at /aqi/current — "
            "this should not happen after successful startup"
        )
        raise HTTPException(status_code=503, detail="Service starting") from exc

    # --- Find the configured AQI provider in the capability registry ---
    provider_registry = get_provider_registry()
    aqi_providers = [p for p in provider_registry if p.domain == "aqi"]

    # --- Decision tree branch 1: no provider configured ---
    if not aqi_providers:
        logger.debug("No AQI provider in registry; returning null data")
        return AQIResponse(
            data=None,
            units=units,
            source="none",
            generatedAt=now_str,
        )

    # Single source per deploy per ADR-013; take the first (and only) entry.
    provider_id = aqi_providers[0].provider_id

    # --- Obtain station lat/lon (ADR-011: single-station, no ?station= param) ---
    try:
        station = get_station_info()
    except RuntimeError as exc:
        logger.error(
            "Station metadata not available at /aqi/current — "
            "this should not happen after successful startup"
        )
        raise HTTPException(status_code=503, detail="Service starting") from exc

    # --- Dispatch to provider module ---
    if provider_id == "openmeteo":
        from weewx_clearskies_api.providers.aqi import openmeteo  # noqa: PLC0415

        record: AQIReading | None = openmeteo.fetch(
            lat=station.latitude,
            lon=station.longitude,
        )
    elif provider_id == "aeris":
        from weewx_clearskies_api.providers.aqi import aeris  # noqa: PLC0415

        if not _AERIS_CLIENT_ID or not _AERIS_CLIENT_SECRET:
            logger.error(
                "Aeris AQI provider configured but credentials not wired at request time"
            )
            raise HTTPException(status_code=502, detail="Aeris credentials missing")

        record = aeris.fetch(
            lat=station.latitude,
            lon=station.longitude,
            client_id=_AERIS_CLIENT_ID,
            client_secret=_AERIS_CLIENT_SECRET,
        )
    elif provider_id == "openweathermap":
        from weewx_clearskies_api.providers.aqi import openweathermap  # noqa: PLC0415

        if not _OWM_APPID:
            logger.error(
                "OpenWeatherMap AQI provider configured but appid not wired at request time"
            )
            raise HTTPException(status_code=502, detail="OpenWeatherMap appid missing")

        record = openweathermap.fetch(
            lat=station.latitude,
            lon=station.longitude,
            appid=_OWM_APPID,
        )
    elif provider_id == "iqair":
        from weewx_clearskies_api.providers.aqi import iqair  # noqa: PLC0415

        if not _IQAIR_KEY:
            logger.error(
                "IQAir AQI provider configured but key not wired at request time"
            )
            raise HTTPException(status_code=502, detail="IQAir key missing")

        record = iqair.fetch(
            lat=station.latitude,
            lon=station.longitude,
            key=_IQAIR_KEY,
        )
    else:
        # Unknown provider should have been caught at startup by _wire_providers_from_config.
        logger.error("Unknown AQI provider at request time: %r", provider_id)
        raise HTTPException(status_code=502, detail=f"Unknown AQI provider: {provider_id!r}")

    return AQIResponse(
        data=record,
        units=units,
        source=provider_id,
        generatedAt=now_str,
    )


@router.get(
    "/aqi/history",
    summary="Historical AQI readings",
    tags=["AQI"],
    response_model=AQIHistoryResponse,
)
def get_aqi_history_endpoint(
    db: Annotated[Session, Depends(get_db_session)],
    params: Annotated[AQIHistoryQueryParams, Depends(_get_aqi_history_params)],
) -> AQIHistoryResponse:
    """Return historical AQI readings from the weewx archive (ADR-013 corrected, P4-T3).

    Path A (archive columns configured): queries the weewx archive for AQI data
      and returns paginated AQIReading objects.
    Path B (no archive columns): returns an empty data list and total=0.
      This is the expected state for operators who only use the /aqi/current
      provider-based path and have no AQI columns in their weewx archive.

    Supports cursor-based and page-number pagination (mutually exclusive).
    Units block populated from the cached units service (same as /aqi/current).
    """
    now_str = utc_isoformat(datetime.now(tz=UTC))

    # Assemble units block (AQI fields are unit-system-invariant;
    # block populated by the units service from weewx.conf at startup).
    try:
        units = get_units_block()
    except RuntimeError as exc:
        logger.error(
            "Units block not available at /aqi/history — "
            "this should not happen after successful startup"
        )
        raise HTTPException(status_code=503, detail="Service starting") from exc

    try:
        readings, page_info = get_aqi_history(
            db=db,
            hist=_AQI_HISTORY_SETTINGS,
            from_dt=params.from_,
            to_dt=params.to,
            limit=params.limit,
            cursor=params.cursor,
            page=params.page,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AQIHistoryResponse(
        data=readings,
        units=units,
        source="weewx",
        generatedAt=now_str,
        page=page_info,
    )
