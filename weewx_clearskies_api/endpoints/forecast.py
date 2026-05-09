"""GET /forecast — forecast bundle (hourly + daily + discussion) (ADR-007).

Behavior decision tree per brief §per-endpoint spec:

  1. No forecast provider in capability registry  → 200, hourly=[], daily=[],
     discussion=null, source="none". No upstream call. No error.
  2. Provider configured, Open-Meteo returns 200 → normalize hourly + daily
     per canonical-data-model §4.1.2 / §4.1.3; slice to hours/days; return 200.
  3. Provider configured, NWS returns 200 → normalize hourly + daily + discussion
     per canonical-data-model §4.1.2 / §4.1.3 / §4.1.4; return 200.
  4. Network failure / 5xx after retries → 502 ProviderProblem (TransientNetworkError)
  5. Provider returns 429 → 503 ProviderProblem (QuotaExhausted) + Retry-After
  6. Provider returns 400/error envelope → 502 ProviderProblem (ProviderProtocolError)
  7. Pydantic validation failure on wire model → 502 ProviderProblem (ProviderProtocolError)
  8. NWS /points 404 → 503 ProviderProblem (GeographicallyUnsupported)
  9. Aeris credentials unset → 502 ProviderProblem (KeyInvalid)
 10. Aeris returns 401 → 502 ProviderProblem (KeyInvalid)
 11. Aeris returns 429 → 503 ProviderProblem (QuotaExhausted) + Retry-After
 12. OWM appid unset → 502 ProviderProblem (KeyInvalid)
 13. OWM returns 401 (basic-tier key) → 200 ForecastResponse with hourly=[], daily=[]
     (Q1 user decision 2026-05-08; graceful empty bundle, not an error)
 14. OWM returns 429 → 503 ProviderProblem (QuotaExhausted) + Retry-After
 15. Wunderground api_key or pws_station_id unset → 502 ProviderProblem (KeyInvalid)
 16. Wunderground returns 401 (apiKey invalid or PWS no longer active) →
     502 ProviderProblem (KeyInvalid); standard KeyInvalid 502 propagation.
 17. Wunderground returns 429 → 503 ProviderProblem (QuotaExhausted) + Retry-After
 18. Wunderground returns 200 → normalize daily (5 entries) per canonical §4.1.3;
     hourly=[] always (PARTIAL-DOMAIN); discussion=null always; return 200.

Slice-after-cache pattern (ADR-017 §Cache key):
  Cache stores the FULL bundle (every hourly + daily point returned by provider).
  Endpoint applies the operator's hours / days slice on the cached canonical bundle.
  One cache entry per (station, target_unit), not one per (hours, days) tuple.

No DB hit. Forecast comes from the provider, not weewx archive.

Operator lat/lon / target_unit / timezone source (ADR-011 single-station):
  Read from services/station.py StationInfo (lat, lon, timezone) and
  services/units.py get_target_unit() (target_unit).  No ?station= param.

Pydantic + Depends pattern (coding.md §1, security-baseline §3.5):
  Unknown query keys rejected with 422/400 via extra="forbid" + Depends wrapper.

Provider discovery: endpoint reads the capability registry at request time.
  _wire_providers_from_config() at startup registers the configured provider's
  CAPABILITY; this endpoint checks the registry for a "forecast" domain entry.
  Tests that need the openmeteo path call wire_providers([openmeteo.CAPABILITY]).
  Tests that need the nws path call wire_providers([nws_forecast.CAPABILITY]).

NWS user-agent contact: wired separately via wire_forecast_settings() in
  __main__.py after settings load.  Tests without a full startup use None (no
  contact), which triggers a one-time WARNING log but works correctly.
  Mirrors the pattern in endpoints/alerts.py (wire_alerts_settings).

Aeris credentials: wired via wire_aeris_credentials() (called from
  wire_forecast_settings()). Module-level _aeris_client_id and
  _aeris_client_secret are passed to aeris.fetch() from the dispatch branch.
  Missing credentials → KeyInvalid at fetch time (brief lead-call 12).

OWM appid: wired via wire_openweathermap_credentials() (called from
  wire_forecast_settings()). Module-level _openweathermap_appid is passed to
  openweathermap.fetch() from the dispatch branch.
  Missing appid → KeyInvalid at fetch time (same loud-failure posture as Aeris).

Wunderground credentials: wired via wire_wunderground_credentials() (called from
  wire_forecast_settings()). Module-level _wunderground_api_key and
  _wunderground_pws_station_id passed to wunderground.fetch() from the dispatch branch.
  Missing either credential → KeyInvalid at fetch time (same loud-failure posture).
  PARTIAL-DOMAIN: Wunderground bundle ships hourly=[] and discussion=None always.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

import pydantic
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError

from weewx_clearskies_api.models.params import ForecastQueryParams
from weewx_clearskies_api.models.responses import (
    ForecastBundle,
    ForecastResponse,
    utc_isoformat,
)
from weewx_clearskies_api.providers._common.capability import get_provider_registry
from weewx_clearskies_api.services.station import get_station_info
from weewx_clearskies_api.services.units import get_target_unit, get_units_block

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level NWS UA contact wiring (populated at startup)
# Mirrors the pattern from endpoints/alerts.py (wire_nws_user_agent_contact).
# ---------------------------------------------------------------------------

_nws_user_agent_contact: str | None = None


def wire_nws_user_agent_contact(contact: str | None) -> None:
    """Store the NWS User-Agent contact string for use by the forecast endpoint.

    Called from wire_forecast_settings() which is called in __main__.py after
    settings load.  Tests that don't care about the UA leave this as None.
    """
    global _nws_user_agent_contact  # noqa: PLW0603
    _nws_user_agent_contact = contact


# ---------------------------------------------------------------------------
# Module-level Aeris credentials wiring (populated at startup)
# Mirrors wire_nws_user_agent_contact pattern. Both credentials are set
# together so the provider module sees a consistent pair at fetch time.
# ---------------------------------------------------------------------------

_aeris_client_id: str | None = None
_aeris_client_secret: str | None = None


def wire_aeris_credentials(client_id: str | None, client_secret: str | None) -> None:
    """Store Aeris credentials read from env vars at startup.

    Per ADR-027 §3, secrets come from env vars (loaded by systemd
    EnvironmentFile / docker-compose env_file).  Tests that don't care
    about Aeris leave both as None; if [forecast] provider = aeris and
    credentials are unset, the module raises KeyInvalid at fetch time
    per brief lead-call 12 (loud failure beats silent disable).
    """
    global _aeris_client_id, _aeris_client_secret  # noqa: PLW0603
    _aeris_client_id = client_id
    _aeris_client_secret = client_secret


# ---------------------------------------------------------------------------
# Module-level OWM appid wiring (populated at startup)
# Mirrors wire_aeris_credentials pattern. Single credential (appid only).
# ---------------------------------------------------------------------------

_openweathermap_appid: str | None = None


def wire_openweathermap_credentials(appid: str | None) -> None:
    """Store OWM appid read from env var at startup.

    Per ADR-027 §3, secrets come from env vars (loaded by systemd
    EnvironmentFile / docker-compose env_file).  Long-form provider-scoped
    env var name WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID per brief Q2 user
    decision 2026-05-08.  Tests that don't care about OWM leave this as None;
    if [forecast] provider = openweathermap and appid is unset, the module
    raises KeyInvalid at fetch time (loud failure beats silent disable).
    """
    global _openweathermap_appid  # noqa: PLW0603
    _openweathermap_appid = appid


# ---------------------------------------------------------------------------
# Module-level Wunderground credentials wiring (populated at startup)
# Mirrors wire_aeris_credentials pattern but takes two args (api_key + pws_station_id).
# Both are required per ADR-007 line 79 defense-in-depth gate (brief lead-call 14).
# ---------------------------------------------------------------------------

_wunderground_api_key: str | None = None
_wunderground_pws_station_id: str | None = None


def wire_wunderground_credentials(api_key: str | None, pws_station_id: str | None) -> None:
    """Store Wunderground credentials read from env vars at startup.

    Per ADR-027 §3, secrets come from env vars (loaded by systemd
    EnvironmentFile / docker-compose env_file).  Long-form provider-scoped
    env var names (WEEWX_CLEARSKIES_WUNDERGROUND_API_KEY and
    WEEWX_CLEARSKIES_WUNDERGROUND_PWS_STATION_ID) per 3b-4/3b-5 precedent.

    Both credentials are required for Wunderground per ADR-007 line 79:
      - api_key: sent as query param on every forecast request.
      - pws_station_id: config-time gate (NOT in the URL); defense-in-depth
        to ensure operator has an active PWS (apiKeys are only issued to
        active PWS contributors).

    Tests that don't care about Wunderground leave both as None; if
    [forecast] provider = wunderground and either is unset, the module raises
    KeyInvalid at fetch time per brief lead-call 14 (loud failure beats silent
    disable).
    """
    global _wunderground_api_key, _wunderground_pws_station_id  # noqa: PLW0603
    _wunderground_api_key = api_key
    _wunderground_pws_station_id = pws_station_id


def wire_forecast_settings(settings: object) -> None:
    """Wire forecast-related settings from the Settings object.

    Convenience wrapper for __main__.py — extracts nws_user_agent_contact,
    Aeris credentials, OWM appid, and Wunderground credentials from
    settings.forecast and calls the per-provider wire_*() helpers.
    Mirrors wire_alerts_settings() in endpoints/alerts.py.
    """
    forecast_settings = getattr(settings, "forecast", None)
    contact = getattr(forecast_settings, "nws_user_agent_contact", None)
    wire_nws_user_agent_contact(contact)

    aeris_id = getattr(forecast_settings, "aeris_client_id", None)
    aeris_secret = getattr(forecast_settings, "aeris_client_secret", None)
    wire_aeris_credentials(aeris_id, aeris_secret)

    owm_appid = getattr(forecast_settings, "openweathermap_appid", None)
    wire_openweathermap_credentials(owm_appid)

    wu_api_key = getattr(forecast_settings, "wunderground_api_key", None)
    wu_pws_station_id = getattr(forecast_settings, "wunderground_pws_station_id", None)
    wire_wunderground_credentials(wu_api_key, wu_pws_station_id)


# ---------------------------------------------------------------------------
# Depends wrapper — Pydantic + Depends pattern (coding.md §1)
# ---------------------------------------------------------------------------


def _get_forecast_params(request: Request) -> ForecastQueryParams:
    """Extract and validate /forecast query parameters via Pydantic.

    Using Depends(model_validate(dict(request.query_params))) pattern so
    extra="forbid" actually fires for unknown query keys (coding.md §1,
    security-baseline §3.5).  Individual FastAPI Query() declarations
    silently ignore unknown keys — not acceptable.
    """
    try:
        return ForecastQueryParams.model_validate(dict(request.query_params))
    except pydantic.ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get(
    "/forecast",
    summary="Forecast bundle (hourly + daily + discussion)",
    tags=["Forecast"],
    response_model=ForecastResponse,
)
def get_forecast(
    params: Annotated[ForecastQueryParams, Depends(_get_forecast_params)],
) -> ForecastResponse:
    """Return forecast bundle from the configured provider.

    Reads the capability registry for the forecast domain at request time.
    Returns ForecastBundle(hourly=[], daily=[], discussion=None, source="none")
    when no provider is registered.
    Cache integration and hours/days slice happen transparently in this handler.
    """
    now_str = utc_isoformat(datetime.now(tz=UTC))

    # --- Assemble units block (same wiring as observations + records) ---
    try:
        units = get_units_block()
        target_unit = get_target_unit()
    except RuntimeError:
        # Defense-in-depth: units should always be wired before uvicorn starts.
        # This branch is theoretically unreachable if startup order is correct.
        logger.error(
            "Units block not available at forecast endpoint — "
            "this should not happen after successful startup"
        )
        raise HTTPException(status_code=503, detail="Service starting")

    # --- Find the configured forecast provider in the capability registry ---
    provider_registry = get_provider_registry()
    forecast_providers = [p for p in provider_registry if p.domain == "forecast"]

    # --- Decision tree branch 1: no provider configured ---
    if not forecast_providers:
        logger.debug("No forecast provider in registry; returning empty bundle")
        return ForecastResponse(
            data=ForecastBundle(
                hourly=[],
                daily=[],
                discussion=None,
                source="none",
                generatedAt=now_str,
            ),
            units=units,
            source="none",
            generatedAt=now_str,
        )

    # Single source per deploy per ADR-007; take the first (and only) entry.
    provider_id = forecast_providers[0].provider_id

    # --- Obtain station lat/lon / timezone (ADR-011: single-station, no ?station= param) ---
    try:
        station = get_station_info()
    except RuntimeError:
        # Defense-in-depth: station should always be wired before uvicorn starts.
        logger.error(
            "Station metadata not available at forecast endpoint — "
            "this should not happen after successful startup"
        )
        raise HTTPException(status_code=503, detail="Service starting")

    # --- Dispatch to provider module ---
    if provider_id == "openmeteo":
        from weewx_clearskies_api.providers.forecast import openmeteo  # noqa: PLC0415

        # fetch() returns the FULL canonical bundle (all hours/days from Open-Meteo).
        # Cache stores the full bundle; slice is applied below after cache lookup.
        bundle = openmeteo.fetch(
            lat=station.latitude,
            lon=station.longitude,
            target_unit=target_unit,
            timezone=station.timezone,
        )
    elif provider_id == "nws":
        from weewx_clearskies_api.providers.forecast import nws as forecast_nws  # noqa: PLC0415

        # fetch() returns the FULL canonical bundle (all NWS hourly + daily + discussion).
        # Cache stores the full bundle; slice is applied below after cache lookup.
        # _nws_user_agent_contact is set at startup via wire_forecast_settings().
        bundle = forecast_nws.fetch(
            lat=station.latitude,
            lon=station.longitude,
            target_unit=target_unit,
            user_agent_contact=_nws_user_agent_contact,
        )
    elif provider_id == "aeris":
        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415

        # fetch() returns the FULL canonical bundle (hourly + daily from two upstream
        # calls: filter=1hr and filter=daynight). Cache stores the full bundle;
        # slice is applied below after cache lookup.
        # _aeris_client_id + _aeris_client_secret set at startup via wire_forecast_settings().
        bundle = aeris.fetch(
            lat=station.latitude,
            lon=station.longitude,
            target_unit=target_unit,
            client_id=_aeris_client_id,
            client_secret=_aeris_client_secret,
        )
    elif provider_id == "openweathermap":
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415

        # fetch() returns the FULL canonical bundle (one upstream call:
        # /data/3.0/onecall with exclude=current,minutely,alerts). Cache stores
        # the full bundle; slice is applied below after cache lookup.
        # _openweathermap_appid set at startup via wire_forecast_settings().
        # Basic-tier 401 → graceful empty bundle per Q1 user decision 2026-05-08.
        bundle = openweathermap.fetch(
            lat=station.latitude,
            lon=station.longitude,
            target_unit=target_unit,
            appid=_openweathermap_appid,
        )
    elif provider_id == "wunderground":
        from weewx_clearskies_api.providers.forecast import wunderground  # noqa: PLC0415

        # fetch() returns the FULL canonical bundle (one upstream call:
        # GET /v3/wx/forecast/daily/5day with geocode=<lat>,<lon>).
        # PARTIAL-DOMAIN: bundle.hourly=[] always; bundle.discussion=None always.
        # Cache stores the full bundle; slice is applied below after cache lookup.
        # _wunderground_api_key + _wunderground_pws_station_id set at startup
        # via wire_forecast_settings().
        # Missing either credential → KeyInvalid at fetch time (loud failure).
        # A 401 from Wunderground → KeyInvalid 502 (apiKey invalid OR PWS no
        # longer active; bare canonical exception propagation per L2 rule).
        bundle = wunderground.fetch(
            lat=station.latitude,
            lon=station.longitude,
            target_unit=target_unit,
            api_key=_wunderground_api_key,
            pws_station_id=_wunderground_pws_station_id,
        )
    else:
        # Unknown provider should have been caught at startup by _wire_providers_from_config.
        # If we reach here, it means a bug in the startup sequence — treat as 502.
        logger.error("Unknown forecast provider at request time: %r", provider_id)
        raise HTTPException(
            status_code=502,
            detail=f"Unknown forecast provider: {provider_id!r}",
        )

    # --- Apply hours / days slice AFTER cache lookup (ADR-017, slice-after-cache) ---
    # Truncate from the head (first N points); Open-Meteo returns chronological order.
    # If the requested count exceeds what the provider returned, use all available.
    sliced_hourly = bundle.hourly[: params.hours]
    sliced_daily = bundle.daily[: params.days]

    sliced_bundle = ForecastBundle(
        hourly=sliced_hourly,
        daily=sliced_daily,
        discussion=bundle.discussion,
        source=provider_id,
        generatedAt=bundle.generatedAt,
    )

    return ForecastResponse(
        data=sliced_bundle,
        units=units,
        source=provider_id,
        generatedAt=now_str,
    )
