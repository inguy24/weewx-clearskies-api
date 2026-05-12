"""OpenWeatherMap radar provider module — Weather Maps 1.0 (ADR-015, ADR-038, 3b-15).

Five responsibilities per ADR-038 §2:
  1. Outbound API call — Weather Maps 1.0 XYZ tile fetch:
       GET https://tile.openweathermap.org/map/precipitation_new/{z}/{x}/{y}.png
           ?appid={appid}
     Layer hardcoded to `precipitation_new` per ADR-015 (model precipitation,
     NOT radar reflectivity).  Frame index is synthesized — no upstream call.
  2. Response parsing — tile response is binary PNG; no Pydantic wire model.
     Frame index synthesized in Python (no upstream call needed).
  3. Translation — radar has no canonical-entity mapping (canonical §4.5).
     get_frames() builds RadarFrame(time=<now-iso>, kind="current").
     get_tile() returns (response.content, response.headers.get("Content-Type",
     "image/png")).
  4. Capability declaration — CAPABILITY symbol consumed at startup.
     tile_url_template does NOT include the appid (LC-D — template is
     public-shape; credentials inject at api-proxy time).
  5. Error handling — canonical taxonomy via ProviderHTTPClient.get()
     (KeyInvalid, QuotaExhausted, TransientNetworkError, ProviderProtocolError).
     No re-construction.  Empty-appid guard raises KeyInvalid before any network
     call (LC-I).

Keyed provider:
  appid passed as query param per OWM auth convention.
  Reuses existing WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID env var (LC-C) →
  settings.forecast.openweathermap_appid.  Wired at startup via
  wire_radar_settings() in endpoints/radar.py.
  Credentials NOT in the cache key (privacy/leakage — same as AQI keyed modules).

Cache layer (ADR-017, LC-A, LC-B):
  Frame index: TTL 60s (parity with 3b-14 keyless precedent; conscious ADR-017
    deviation noted in 3b-14 parking lot).  Synthesized result still cached to
    keep the pattern uniform and avoid per-request datetime drift.
  Tile bytes:  TTL 300s (ADR-017 default for "tile bytes (proxied keyed
    providers)").  Cache value is a base64 envelope (LC-A) because existing
    RedisCache.set() is json.dumps-based and cannot store raw bytes:
      {"_tile_b64": "<base64>", "content_type": "image/png"}
    On hit: decode _tile_b64 → bytes; return with cached content_type.

v0.1 scope:
  get_frames() → single kind="current" frame synthesized at request time (LC-G).
  ?t query param accepted at the endpoint but IGNORED here (LC-F); current-only.
  Weather Maps 1.0 has no documented frame index / time-step API.

Rate limiter (LC-J): max_calls=5, window_seconds=1 (be-polite guard).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from datetime import UTC, datetime

from weewx_clearskies_api.models.responses import RadarFrame, RadarFrameList, utc_isoformat
from weewx_clearskies_api.providers._common.cache import get_cache
from weewx_clearskies_api.providers._common.capability import ProviderCapability
from weewx_clearskies_api.providers._common.errors import KeyInvalid
from weewx_clearskies_api.providers._common.http import ProviderHTTPClient
from weewx_clearskies_api.providers._common.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PROVIDER_ID = "openweathermap"
DOMAIN = "radar"
_TILE_BASE_URL = "https://tile.openweathermap.org/map/precipitation_new"
# TTL deviation: ADR-017 default for radar frame metadata is 5 min; 60s used
# for parity with 3b-14 keyless precedent (same conscious deviation).
_FRAMES_TTL = 60   # 60 s — frame index (synthesized)
_TILE_TTL = 300    # 300 s — tile bytes per ADR-017 tile default
_API_VERSION = "0.1.0"

ATTRIBUTION = "OpenWeatherMap (https://openweathermap.org/)"

# ---------------------------------------------------------------------------
# Capability declaration (ADR-038 §4)
# ---------------------------------------------------------------------------

CAPABILITY = ProviderCapability(
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    supplied_canonical_fields=(),  # radar has no canonical-entity mapping (§4.5)
    geographic_coverage="global",
    auth_required=("appid",),
    default_poll_interval_seconds=_FRAMES_TTL,
    operator_notes=(
        "OpenWeatherMap Weather Maps 1.0 — precipitation_new layer (NWP model "
        "precipitation, NOT radar reflectivity). UI label: 'Model precipitation' "
        "per ADR-015. Keyed (query-param appid); reuses provider-scoped credential "
        "from forecast/alerts/AQI OWM (WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID env "
        "var). Current-only at v0.1; ?t query param ignored. Tile bytes cached "
        "300s; frame index synthesized + cached 60s."
    ),
    # LC-D: tile_url_template is public-shape — no credentials embedded.
    tile_url_template="https://tile.openweathermap.org/map/precipitation_new/{z}/{x}/{y}.png",
    wms_endpoint_url=None,
    wms_layer_name=None,
    tile_content_type="image/png",
)

# ---------------------------------------------------------------------------
# Rate limiter (LC-J) — polite-use guard
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="openweathermap-radar",
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    max_calls=5,
    window_seconds=1,
)

# ---------------------------------------------------------------------------
# HTTP client (module-level singleton)
# ---------------------------------------------------------------------------

_http_client: ProviderHTTPClient | None = None


def _get_http_client() -> ProviderHTTPClient:
    global _http_client  # noqa: PLW0603
    if _http_client is None:
        _http_client = ProviderHTTPClient(
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
            user_agent=f"weewx-clearskies-api/{_API_VERSION}",
        )
    return _http_client


# ---------------------------------------------------------------------------
# Cache key helpers (ADR-017)
# ---------------------------------------------------------------------------


def _frames_cache_key() -> str:
    """Cache key for the frame index (no credentials — LC-C)."""
    payload = json.dumps({"provider_id": PROVIDER_ID, "kind": "frames"}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _build_tile_cache_key(z: int, x: int, y: int, t: str | None) -> str:
    """Cache key for a tile byte response.

    Includes (provider_id, endpoint, z, x, y, t_normalized).
    Credentials NOT in key (privacy/leakage per LC-C; same as AQI keyed modules).
    t_normalized is None at v0.1 since ?t is always ignored (LC-F).
    """
    payload = json.dumps(
        {
            "provider_id": PROVIDER_ID,
            "endpoint": "tile",
            "z": z,
            "x": x,
            "y": y,
            "t": t,  # None at v0.1
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Cache serialisation helpers (LC-A — base64 envelope for tile bytes)
# ---------------------------------------------------------------------------


def _tile_to_cacheable(tile_bytes: bytes, content_type: str) -> dict:  # type: ignore[type-arg]
    """Wrap raw tile bytes into a JSON-safe dict for the cache backend.

    RedisCache.set() calls json.dumps() — raw bytes are not JSON-encodable.
    base64 envelope keeps the existing cache abstraction unchanged (LC-A).
    ~33% storage overhead per tile is acceptable at v0.1.
    """
    return {
        "_tile_b64": base64.b64encode(tile_bytes).decode("ascii"),
        "content_type": content_type,
    }


def _tile_from_cached(cached: dict) -> tuple[bytes, str]:  # type: ignore[type-arg]
    """Reconstruct (bytes, content_type) from a cached base64 envelope (LC-A)."""
    return base64.b64decode(cached["_tile_b64"]), cached["content_type"]


# ---------------------------------------------------------------------------
# Cache serialisation helpers (frame index)
# ---------------------------------------------------------------------------


def _frames_to_cacheable(frames_list: RadarFrameList) -> dict:  # type: ignore[type-arg]
    """Serialise RadarFrameList to a JSON-safe dict for cache storage."""
    return frames_list.model_dump(mode="json")


def _frames_from_cached(cached: dict) -> RadarFrameList:  # type: ignore[type-arg]
    """Reconstruct RadarFrameList from a cached dict."""
    return RadarFrameList.model_validate(cached)


# ---------------------------------------------------------------------------
# Public frame-index entrypoint (ADR-038 §2)
# ---------------------------------------------------------------------------


def get_frames(*, appid: str | None = None) -> RadarFrameList:
    """Return a single synthesized 'current' frame for OWM radar (LC-G).

    Weather Maps 1.0 has no documented frame index or time-step API.
    Frame is synthesized at request time and cached 60s to avoid per-request
    datetime drift within the cache window.

    Args:
        appid: OWM API key.  LC-I: raises KeyInvalid if empty/None BEFORE
            any network call.  For frame synthesis no network call is made,
            but the guard is applied for consistency — a frame from an
            unconfigured provider should not appear as valid.

    Returns:
        RadarFrameList with a single kind="current" frame.

    Raises:
        KeyInvalid: appid is None or empty.
    """
    # LC-I: empty-credential guard at provider entrypoint.
    if not appid:
        raise KeyInvalid(
            "OpenWeatherMap radar: appid is missing. "
            "Set WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID env var.",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    cache = get_cache()
    key = _frames_cache_key()
    hit = cache.get(key)
    if hit is not None:
        return _frames_from_cached(hit)

    # Synthesize — no upstream call needed (LC-G).
    now_str = utc_isoformat(datetime.now(tz=UTC))
    result = RadarFrameList(
        providerId=PROVIDER_ID,
        frames=[RadarFrame(time=now_str, kind="current")],
        attribution=ATTRIBUTION,
        tileHost=None,  # XYZ template-based; no per-fetch host
    )

    cache.set(key, _frames_to_cacheable(result), ttl_seconds=_FRAMES_TTL)

    logger.debug(
        "[%s] radar frames synthesized (single current frame; no upstream call)",
        PROVIDER_ID,
    )
    return result


# ---------------------------------------------------------------------------
# Public tile entrypoint (ADR-038 §2) — NOVEL: binary response
# ---------------------------------------------------------------------------


def get_tile(
    z: int,
    x: int,
    y: int,
    *,
    t: str | None = None,
    appid: str | None = None,
) -> tuple[bytes, str]:
    """Fetch a single tile from Weather Maps 1.0 and return (bytes, content_type).

    This is one of two non-JSON entrypoints in the codebase.  The response
    is raw binary PNG bytes — no Pydantic response model, no JSON body.
    The endpoint handler (endpoints/radar.py get_radar_tile) wraps these
    bytes in fastapi.Response(content=bytes, media_type=ct).

    Cache uses a base64 envelope (LC-A):
      {"_tile_b64": "<base64>", "content_type": "image/png"}
    TTL: 300s (LC-B, ADR-017 tile-bytes default).

    Args:
        z: Slippy-map zoom level (0-22).
        x: Tile X coordinate.
        y: Tile Y coordinate.
        t: Optional frame timestamp (ISO-8601 UTC).  IGNORED at v0.1 (LC-F);
            OWM Weather Maps 1.0 is current-only.  Logged at DEBUG.
        appid: OWM API key.  LC-I: raises KeyInvalid if empty/None.

    Returns:
        (tile_bytes, content_type) — caller wraps in fastapi.Response.

    Raises:
        KeyInvalid: appid missing.
        QuotaExhausted: upstream 429.
        TransientNetworkError: network/5xx after retries.
        ProviderProtocolError: unexpected 4xx (including 404 for out-of-domain
            tiles — status_code=404 on the exception; endpoint maps to HTTP 404).
    """
    # LC-I: empty-credential guard at provider entrypoint.
    if not appid:
        raise KeyInvalid(
            "OpenWeatherMap radar: appid is missing. "
            "Set WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID env var.",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        )

    # LC-F: ?t accepted but ignored at v0.1.
    if t is not None:
        logger.debug(
            "[%s] tile ?t=%r received but ignored at v0.1 (current-only)",
            PROVIDER_ID,
            t,
        )
    # t_normalized is always None at v0.1 for the cache key.
    t_normalized: str | None = None

    # Cache check.
    cache = get_cache()
    key = _build_tile_cache_key(z, x, y, t_normalized)
    hit = cache.get(key)
    if hit is not None:
        return _tile_from_cached(hit)

    # Rate limiter (LC-J).
    _rate_limiter.acquire()

    # Build tile URL: https://tile.openweathermap.org/map/precipitation_new/{z}/{x}/{y}.png
    tile_url = f"{_TILE_BASE_URL}/{z}/{x}/{y}.png"

    response = _get_http_client().get(tile_url, params={"appid": appid})

    content_type = response.headers.get("Content-Type", "image/png")
    tile_bytes = response.content

    # Cache with base64 envelope (LC-A).
    cache.set(key, _tile_to_cacheable(tile_bytes, content_type), ttl_seconds=_TILE_TTL)

    logger.debug(
        "[%s] tile fetched: z=%d x=%d y=%d content_type=%r size=%d bytes",
        PROVIDER_ID,
        z,
        x,
        y,
        content_type,
        len(tile_bytes),
    )
    return tile_bytes, content_type


def _reset_http_client_for_tests() -> None:
    """Reset module-level HTTP client singleton.  Used in tests only."""
    global _http_client  # noqa: PLW0603
    _http_client = None
