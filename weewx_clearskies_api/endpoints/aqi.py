"""GET /aqi/current and GET /aqi/history (ADR-013).

Behavior decision tree for /aqi/current per brief §per-endpoint spec:

  1. No AQI provider in capability registry → 200, data=null, source="none"
  2. Provider configured, fetch succeeds with valid AQI → 200 + AQIReading
  3. Provider configured, fetch succeeds but all values null → 200, data=null
  4. Network failure / 5xx after retries → 502 (TransientNetworkError → RFC 9457)
  5. Provider returns 429 → 503 + Retry-After (QuotaExhausted → RFC 9457)
  6. Provider returns 401/403 → 502 (KeyInvalid — not expected for keyless Open-Meteo)
  7. Pydantic validation failure on wire model → 502 (ProviderProtocolError → RFC 9457)

/aqi/history always returns 501 Not Implemented (LC21):
  AQI history persistence is deferred to a future round per ADR-013 §Out of scope.
  RFC 9457 problem+json body regardless of provider config or query params.
  Params are validated first (invalid params → 422; valid params → 501).

No DB hit.  AQI comes from the configured provider (Path B), not weewx archive.

Operator lat/lon: from get_station_info() (services/station.py) per ADR-011
  (single-station scope).  No ?station= param.

Pydantic + Depends pattern (coding.md §1, security-baseline §3.5):
  Unknown query keys rejected with 422 via extra="forbid" + Depends wrapper.

Provider discovery: endpoint reads the capability registry at request time.
  _wire_providers_from_config() at startup registers the configured provider's
  CAPABILITY; this endpoint checks the registry for an "aqi" domain entry.
  Tests that need the openmeteo path call wire_providers([openmeteo.CAPABILITY]).

wire_aqi_settings (LC18):
  No-op for Open-Meteo (keyless, no credentials to wire).
  Future-proof for keyed providers (3b-10 Aeris, 3b-11 OWM, 3b-12 IQAir).

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
from fastapi.responses import JSONResponse

from weewx_clearskies_api.models.params import AQIHistoryQueryParams, AQIQueryParams
from weewx_clearskies_api.models.responses import AQIReading, AQIResponse, utc_isoformat
from weewx_clearskies_api.providers._common.capability import get_provider_registry
from weewx_clearskies_api.services.station import get_station_info
from weewx_clearskies_api.services.units import get_units_block

logger = logging.getLogger(__name__)

router = APIRouter()

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

    Params are validated (invalid → 422) before the 501 handler fires
    (valid params → 501).  This ensures a coherent response: unknown keys
    get 422, valid params get 501.
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

    For Open-Meteo (keyless): no-op — no credentials to extract.
    Future-proof for keyed providers (3b-10 Aeris, 3b-11 OWM, 3b-12 IQAir):
    extract provider credentials and call the appropriate wire_*_credentials()
    functions here when those providers land.
    """
    # Open-Meteo is keyless (auth_required=()); nothing to wire in 3b-9.
    # When 3b-10 lands Aeris AQI: extract aeris_client_id + aeris_client_secret.
    # When 3b-11 lands OWM AQI: extract openweathermap_appid.
    # When 3b-12 lands IQAir: extract iqair_api_key.
    pass


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
    except RuntimeError:
        # Defense-in-depth: units should always be wired before uvicorn starts.
        logger.error(
            "Units block not available at /aqi/current — "
            "this should not happen after successful startup"
        )
        raise HTTPException(status_code=503, detail="Service starting")

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
    except RuntimeError:
        logger.error(
            "Station metadata not available at /aqi/current — "
            "this should not happen after successful startup"
        )
        raise HTTPException(status_code=503, detail="Service starting")

    # --- Dispatch to provider module ---
    if provider_id == "openmeteo":
        from weewx_clearskies_api.providers.aqi import openmeteo  # noqa: PLC0415

        record: AQIReading | None = openmeteo.fetch(
            lat=station.latitude,
            lon=station.longitude,
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
)
def get_aqi_history(
    params: Annotated[AQIHistoryQueryParams, Depends(_get_aqi_history_params)],
) -> JSONResponse:
    """Return 501 Not Implemented — AQI history persistence is deferred (LC21).

    AQI history requires a writeable datastore separate from the read-only
    weewx archive (ADR-013 §Out of scope).  Deferred to a future round.
    This stub always returns 501 regardless of provider config or query params.
    """
    return JSONResponse(
        status_code=501,
        content={
            "type": "https://example.com/probs/not-implemented",
            "title": "Not Implemented",
            "status": 501,
            "detail": (
                "AQI history persistence is not yet implemented. "
                "Tracked for a future release."
            ),
            "instance": "/aqi/history",
        },
        media_type="application/problem+json",
    )
