"""Radar endpoints: frames + tile proxy (ADR-015, 3b-14 / 3b-15).

Endpoints:
  GET /radar/providers/{provider_id}/frames        — radar frame index (3b-14)
  GET /radar/providers/{provider_id}/tiles/{z}/{x}/{y}  — binary tile proxy (3b-15)

Behavior decision tree for /frames per brief §per-endpoint spec:

  1. provider_id not in radar dispatch table → 404 Problem.
  2. provider_id IS in dispatch but NOT in capability registry → 404 Problem
     (operator configured a different provider). Same HTTP status as #1;
     detail text distinguishes them.
  3. Provider configured + registered, fetch succeeds → 200 RadarFramesResponse.
  4. Frame-index fetch returns network failure / 5xx after retries → 502 ProviderProblem
     (TransientNetworkError).
  5. Frame-index fetch returns 429 → 503 ProviderProblem (QuotaExhausted) + Retry-After.
  6. Frame-index parse failure (JSON malformed / XML missing TIME dimension) → 502
     ProviderProblem (ProviderProtocolError).

Behavior decision tree for /tiles/{z}/{x}/{y} per brief §per-endpoint spec:

  1. provider_id not in _KEYED_RADAR_PROVIDERS frozenset → 404 Problem.
     (Distinct from /frames: /tiles is keyed-only; keyless providers are not
     supported through this proxy — the browser fetches them directly per ADR-037.)
  2. provider_id IS in _KEYED_RADAR_PROVIDERS but NOT in capability registry
     (operator configured a different radar provider) → 404 Problem, distinguishing detail.
  3. Credentials missing (env vars unset) → 502 Problem.
  4. z/x/y out of valid range → FastAPI auto-422 from Path constraints. No special handling.
  5. Cache hit → return cached bytes + content_type (no upstream call).
  6. Cache miss → call provider module's get_tile() → cache → return 200 binary response.
  7. Upstream 404 (tile out-of-domain) → ProviderProtocolError(status_code=404)
     → endpoint catches + maps to HTTP 404.  (LC-H: no new canonical class needed.)
  8. Upstream network failure / 5xx → 502 (TransientNetworkError).
  9. Upstream 429 → 503 + Retry-After (QuotaExhausted).
  10. Upstream 401/403 → 502 (KeyInvalid — operator's credentials are wrong/revoked).

wire_radar_settings() (3b-15):
  Mirrors wire_aqi_settings() in endpoints/aqi.py.
  Extracts credentials from settings.forecast (provider-scoped per 3b-5 Q2
  user decision; same env vars as forecast/alerts Aeris + OWM).
  Called from __main__.py at startup step 6n, after wire_providers (6i).

Dispatch table:
  /frames: _KNOWN_RADAR_PROVIDERS — all 7 providers (5 keyless + 2 keyed).
  /tiles:  _KEYED_RADAR_PROVIDERS — keyed-only (aeris, openweathermap).

No DB hit — radar frames / tiles come from the provider, not weewx archive.

Binary response note (/tiles):
  This is one of two non-JSON endpoints in the codebase (the other is the
  deferred /aqi/history 501 stub).  The handler returns
  fastapi.Response(content=bytes, media_type=ct) — no Pydantic response model.
  Future readers should not look for a *Response Pydantic class for this route.

Cache for /tiles:
  Handled in the provider module's get_tile().  Cache hit path returns without
  an upstream call; the module handles the base64 envelope (LC-A, 3b-15).
  TTL: 300s (LC-B).  Key: SHA-256 of (provider_id, endpoint, z, x, y, t_normalized).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Path, Response

from weewx_clearskies_api.models.responses import RadarFramesResponse, utc_isoformat
from weewx_clearskies_api.providers._common.capability import get_provider_registry
from weewx_clearskies_api.providers._common.dispatch import get_provider_module
from weewx_clearskies_api.providers._common.errors import (
    KeyInvalid,
    ProviderProtocolError,
    QuotaExhausted,
    TransientNetworkError,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level credential wiring (populated at startup by wire_radar_settings).
# Aeris: client_id + client_secret.
# OWM: appid.
# Keyless providers: nothing to wire.
# ---------------------------------------------------------------------------

_RADAR_AERIS_CLIENT_ID: str | None = None
_RADAR_AERIS_CLIENT_SECRET: str | None = None
_RADAR_OWM_APPID: str | None = None

# Known radar provider ids (dispatch table keys for /frames).
# Includes all 7 providers: 5 keyless (3b-14) + 2 keyed (3b-15).
# mapbox_jma deferred per ADR-015 2026-05-11 amendment.
_KNOWN_RADAR_PROVIDERS = frozenset(
    {
        "rainviewer",
        "iem_nexrad",
        "noaa_mrms",
        "msc_geomet",
        "dwd_radolan",
        "aeris",
        "openweathermap",
    }
)

# Keyed-only radar providers (dispatch table keys for /tiles proxy).
# /tiles is for keyed providers only — keyless providers are fetched directly
# by the browser per ADR-037 (keys never reach the browser).
_KEYED_RADAR_PROVIDERS = frozenset({"aeris", "openweathermap"})


# ---------------------------------------------------------------------------
# Credential / settings wiring (3b-15)
# ---------------------------------------------------------------------------


def wire_radar_settings(settings: object) -> None:
    """Wire radar-related credentials from the Settings object.

    For keyless providers (rainviewer, iem_nexrad, noaa_mrms, msc_geomet,
    dwd_radolan): no-op — no credentials to extract.
    For Aeris: extracts client_id + client_secret from settings.forecast
      (provider-scoped per 3b-4 Q1 + 3b-5 Q2 user decisions; same env vars
      as forecast/alerts/AQI Aeris: WEEWX_CLEARSKIES_AERIS_CLIENT_ID +
      WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET).
    For OpenWeatherMap: extracts openweathermap_appid from settings.forecast
      (provider-scoped per 3b-5 Q2 user decision; same env var as
      forecast/alerts/AQI OWM: WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID).

    Brief LC-C note: credentials live on settings.forecast, NOT on a standalone
    settings.aeris or settings.openweathermap attribute — those sections don't
    exist in Settings.  Same pattern as wire_aqi_settings() (aqi.py line 144-146).
    """
    global _RADAR_AERIS_CLIENT_ID, _RADAR_AERIS_CLIENT_SECRET, _RADAR_OWM_APPID  # noqa: PLW0603

    radar_section = getattr(settings, "radar", None)
    if radar_section is None:
        return

    provider = getattr(radar_section, "provider", None)

    if provider == "aeris":
        forecast_section = getattr(settings, "forecast", None)
        if forecast_section is None:
            logger.error(
                "[radar] provider=aeris but [forecast] settings section missing; "
                "credentials cannot be wired — /radar/.../tiles will 502"
            )
            return

        _RADAR_AERIS_CLIENT_ID = getattr(forecast_section, "aeris_client_id", None)
        _RADAR_AERIS_CLIENT_SECRET = getattr(forecast_section, "aeris_client_secret", None)

        if not _RADAR_AERIS_CLIENT_ID or not _RADAR_AERIS_CLIENT_SECRET:
            logger.error(
                "[radar] provider=aeris but WEEWX_CLEARSKIES_AERIS_CLIENT_ID/"
                "WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET env vars missing; "
                "capability still registered but /radar/.../tiles will 502 until wired"
            )

    elif provider == "openweathermap":
        forecast_section = getattr(settings, "forecast", None)
        if forecast_section is None:
            logger.error(
                "[radar] provider=openweathermap but [forecast] settings section missing; "
                "credentials cannot be wired — /radar/.../tiles will 502"
            )
            return

        _RADAR_OWM_APPID = getattr(forecast_section, "openweathermap_appid", None)

        if not _RADAR_OWM_APPID:
            logger.error(
                "[radar] provider=openweathermap but "
                "WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID env var missing; "
                "capability still registered but /radar/.../tiles will 502 until wired"
            )

    # Keyless providers: nothing to wire.


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/radar/providers/{provider_id}/frames",
    summary="Available radar frames (timestamps)",
    tags=["Radar"],
    response_model=RadarFramesResponse,
)
def get_radar_frames(provider_id: str) -> RadarFramesResponse:
    """Return the available radar frame timestamps for the given provider.

    Reads the capability registry at request time to confirm the operator has
    configured this radar provider.  Decision tree:

      1. provider_id not in known radar dispatch table → 404.
      2. provider_id in dispatch but not in registry → 404 (different detail).
      3. Provider registered → call get_frames(), return 200 RadarFramesResponse.
      4. get_frames() raises TransientNetworkError → FastAPI error handler → 502.
      5. get_frames() raises QuotaExhausted → FastAPI error handler → 503 + Retry-After.
      6. get_frames() raises ProviderProtocolError → FastAPI error handler → 502.
    """
    now_str = utc_isoformat(datetime.now(tz=UTC))

    # --- Decision tree branch 1: unknown provider_id (not in dispatch table) ---
    if provider_id not in _KNOWN_RADAR_PROVIDERS:
        logger.debug("Radar provider_id %r not in dispatch table", provider_id)
        raise HTTPException(
            status_code=404,
            detail=(
                f"Radar provider {provider_id!r} is not supported. "
                f"Known providers: {sorted(_KNOWN_RADAR_PROVIDERS)}"
            ),
        )

    # --- Decision tree branch 2: in dispatch but not registered ---
    provider_registry = get_provider_registry()
    radar_providers = {p.provider_id for p in provider_registry if p.domain == "radar"}

    if provider_id not in radar_providers:
        logger.debug(
            "Radar provider %r is in dispatch table but not in capability registry "
            "(operator configured a different provider)",
            provider_id,
        )
        raise HTTPException(
            status_code=404,
            detail=(
                f"Radar provider {provider_id!r} is not configured for this deployment. "
                "Check the [radar] section in api.conf."
            ),
        )

    # --- Decision tree branch 3: dispatch + fetch ---
    # KeyError would only fire here if _KNOWN_RADAR_PROVIDERS contains a key
    # missing from PROVIDER_MODULES — a programming error caught at startup.
    module = get_provider_module(domain="radar", provider_id=provider_id)

    # Keyed providers need credentials forwarded to get_frames().
    # Keyless providers expose get_frames() with no arguments.
    if provider_id == "aeris":
        frames_list = module.get_frames(  # type: ignore[attr-defined]
            client_id=_RADAR_AERIS_CLIENT_ID,
            client_secret=_RADAR_AERIS_CLIENT_SECRET,
        )
    elif provider_id == "openweathermap":
        frames_list = module.get_frames(  # type: ignore[attr-defined]
            appid=_RADAR_OWM_APPID,
        )
    else:
        # Keyless providers: get_frames() takes no arguments.
        frames_list = module.get_frames()  # type: ignore[attr-defined]

    return RadarFramesResponse(
        data=frames_list,
        generatedAt=now_str,
    )


@router.get(
    "/radar/providers/{provider_id}/tiles/{z}/{x}/{y}",
    summary="Radar map-tile proxy (for keyed providers)",
    tags=["Radar"],
    # No response_model: binary response, not JSON (non-JSON endpoint #1).
    # See module docstring for context.
)
def get_radar_tile(
    provider_id: str,
    z: int = Path(..., ge=0, le=22, description="Slippy-map zoom level"),
    x: int = Path(..., ge=0, description="Slippy-map tile X"),
    y: int = Path(..., ge=0, description="Slippy-map tile Y"),
    t: str | None = None,
) -> Response:
    """Server-side proxy for keyed radar tile providers (ADR-015 + ADR-037).

    Returns raw tile bytes (PNG) from the configured keyed radar provider.
    Keys never reach the browser — the api holds credentials and proxies
    the tile request server-side per ADR-037.

    This is one of two non-JSON endpoints in the codebase.  The handler
    returns fastapi.Response(content=bytes, media_type=ct) — no Pydantic
    response model exists for this route.

    Decision tree (see module docstring for full spec):
      1. provider_id not in _KEYED_RADAR_PROVIDERS → 404.
      2. provider_id in _KEYED_RADAR_PROVIDERS but not in registry → 404.
      3. Credentials missing → 502.
      4. z/x/y invalid range → FastAPI auto-422 (Path constraints).
      5. Cache hit → 200 binary response (no upstream call).
      6. Cache miss → provider.get_tile() → cache → 200 binary response.
      7. Upstream 404 (tile out-of-domain) → HTTP 404 (LC-H).
      8. Upstream 5xx / network → 502.
      9. Upstream 429 → 503 + Retry-After.
      10. Upstream 401/403 → 502.
    """
    # LC-F: ?t accepted but logged + ignored at v0.1.
    if t is not None:
        logger.debug(
            "[radar tile proxy] provider=%r z=%d x=%d y=%d t=%r (ignored at v0.1)",
            provider_id,
            z,
            x,
            y,
            t,
        )

    # --- Branch 1: /tiles is keyed-only ---
    if provider_id not in _KEYED_RADAR_PROVIDERS:
        logger.debug(
            "Radar tile proxy: provider_id %r not in keyed provider set "
            "(keyless providers are fetched directly by the browser per ADR-037)",
            provider_id,
        )
        raise HTTPException(
            status_code=404,
            detail=(
                f"Radar tile proxy is only available for keyed providers "
                f"({sorted(_KEYED_RADAR_PROVIDERS)}). "
                f"{provider_id!r} is not a keyed provider or is not supported."
            ),
        )

    # --- Branch 2: in keyed set but not registered (different radar provider configured) ---
    provider_registry = get_provider_registry()
    radar_providers = {p.provider_id for p in provider_registry if p.domain == "radar"}

    if provider_id not in radar_providers:
        logger.debug(
            "Radar tile proxy: provider %r in keyed set but not in capability registry "
            "(operator configured a different radar provider)",
            provider_id,
        )
        raise HTTPException(
            status_code=404,
            detail=(
                f"Radar provider {provider_id!r} is not configured for this deployment. "
                "Check the [radar] section in api.conf."
            ),
        )

    # --- Branch 3: credentials missing ---
    if provider_id == "aeris":
        if not _RADAR_AERIS_CLIENT_ID or not _RADAR_AERIS_CLIENT_SECRET:
            logger.error(
                "Aeris radar provider configured but credentials not wired at request time"
            )
            raise HTTPException(status_code=502, detail="Aeris credentials missing")

    elif provider_id == "openweathermap":
        if not _RADAR_OWM_APPID:
            logger.error(
                "OpenWeatherMap radar provider configured but appid not wired at request time"
            )
            raise HTTPException(status_code=502, detail="OpenWeatherMap appid missing")

    # --- Branches 5-10: delegate to provider module ---
    module = get_provider_module(domain="radar", provider_id=provider_id)

    try:
        if provider_id == "aeris":
            tile_bytes, content_type = module.get_tile(  # type: ignore[attr-defined]
                z,
                x,
                y,
                t=t,
                client_id=_RADAR_AERIS_CLIENT_ID,
                client_secret=_RADAR_AERIS_CLIENT_SECRET,
            )
        elif provider_id == "openweathermap":
            tile_bytes, content_type = module.get_tile(  # type: ignore[attr-defined]
                z,
                x,
                y,
                t=t,
                appid=_RADAR_OWM_APPID,
            )
        else:
            # Should not reach here — guarded by _KEYED_RADAR_PROVIDERS check above.
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected provider_id {provider_id!r} in tile proxy",
            )

    except KeyInvalid as exc:
        # Operator's credentials are wrong or revoked.
        logger.error(
            "Radar tile proxy KeyInvalid for provider=%r: %s",
            provider_id,
            exc,
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    except QuotaExhausted as exc:
        # Upstream 429 — propagate Retry-After.
        logger.warning(
            "Radar tile proxy QuotaExhausted for provider=%r retry_after=%s",
            provider_id,
            exc.retry_after_seconds,
        )
        headers: dict[str, str] = {}
        if exc.retry_after_seconds is not None:
            headers["Retry-After"] = str(exc.retry_after_seconds)
        raise HTTPException(status_code=503, detail=str(exc), headers=headers) from exc

    except ProviderProtocolError as exc:
        # LC-H: upstream 404 (tile out-of-domain) or other unexpected 4xx.
        # Dispatch on status_code attribute — no message-string matching.
        if exc.status_code == 404:
            logger.debug(
                "Radar tile proxy: upstream 404 from provider=%r z=%d x=%d y=%d "
                "(tile out of provider domain)",
                provider_id,
                z,
                x,
                y,
            )
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Radar tile not available from {provider_id!r} at "
                    f"z={z} x={x} y={y} (out of provider domain or zoom range)"
                ),
            ) from exc
        # Other 4xx → 502.
        logger.error(
            "Radar tile proxy ProviderProtocolError for provider=%r: %s",
            provider_id,
            exc,
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    except TransientNetworkError as exc:
        # Network failure or 5xx after retries.
        logger.error(
            "Radar tile proxy TransientNetworkError for provider=%r: %s",
            provider_id,
            exc,
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return Response(content=tile_bytes, media_type=content_type)
