"""AerisWeather / Xweather radar provider module — Raster Maps (ADR-015, ADR-038, 3b-15).

Five responsibilities per ADR-038 §2:
  1. Outbound API call — Xweather Raster Maps XYZ tile fetch:
       GET https://maps.api.xweather.com/{client_id}_{client_secret}/radar/{z}/{x}/{y}/current.png
     Layer hardcoded to `radar` (global radar mosaic) per ADR-015.
     The `current` offset is hardcoded at v0.1 — time-step animation deferred (LC-F).
     Frame index is synthesized — no upstream call.

     SECURITY: credentials are embedded in the URL PATH.  ProviderHTTPClient.get()
     logs the full URL at INFO by default.  This module uses the `log_url` parameter
     (LC-E extension to http.py, 3b-15) to pass a redacted URL for all logging,
     and _redact_url() for any local log calls.  See LC-E.

  2. Response parsing — tile bytes; no Pydantic wire model.
     Frame index synthesized in Python (no upstream call).
  3. Translation — radar has no canonical-entity mapping (canonical §4.5).
     get_frames() builds RadarFrame(time=<now-iso>, kind="current").
     get_tile() returns (response.content, content_type).
  4. Capability declaration — CAPABILITY symbol consumed at startup.
     tile_url_template uses {auth} placeholder for the credential segment so
     the template stays public-shape (LC-D).
  5. Error handling — canonical taxonomy via ProviderHTTPClient.get()
     (KeyInvalid, QuotaExhausted, TransientNetworkError, ProviderProtocolError).
     No re-construction.  Empty-credential guard raises KeyInvalid before any
     network call (LC-I).

Keyed provider:
  client_id + client_secret embedded in URL path (per Xweather Raster Maps API).
  Reuses existing WEEWX_CLEARSKIES_AERIS_CLIENT_ID + _AERIS_CLIENT_SECRET env
  vars (LC-C) → settings.forecast.aeris_client_id + aeris_client_secret.
  Wired at startup via wire_radar_settings() in endpoints/radar.py.
  Credentials NOT in the cache key (privacy/leakage — same as AQI keyed modules).

Cache layer (ADR-017, LC-A, LC-B):
  Frame index: TTL 60s (parity with 3b-14 keyless precedent).  Synthesized
    result still cached to keep the pattern uniform.
  Tile bytes:  TTL 300s (ADR-017 default for "tile bytes (proxied keyed
    providers)").  Cache value is a base64 envelope (LC-A):
      {"_tile_b64": "<base64>", "content_type": "image/png"}
    On hit: decode _tile_b64 → bytes; return with cached content_type.

URL credential redaction (LC-E — SECURITY BASELINE):
  Aeris is the only path-credential provider in the codebase.  The existing
  logging.Filter (ADR-029) redacts header/body credentials but NOT path-embedded
  ones.  Strategy:
    1. _redact_url(url) replaces the "{id}_{secret}" path segment with
       "<redacted>" for any local logger.* calls.
    2. _get_http_client().get(real_url, log_url=_redact_url(real_url)) passes
       a safe URL to the HTTP client's INFO log line (LC-E extension to http.py).
  Both together ensure no log message ever contains live credentials.

v0.1 scope:
  get_frames() → single kind="current" frame synthesized at request time (LC-G).
  ?t query param accepted at the endpoint but IGNORED here (LC-F); current-only.
  Future: wire Aeris /info or /maps/img endpoints for past-frame timestamps.

Rate limiter (LC-J): max_calls=5, window_seconds=1 (be-polite guard).

Aeris free path (per ADR-015): PWSWeather Contributor Plan typically bundles
  Maps API access.  Operator should confirm access at fixture capture; surface
  any access-restriction changes via STOP per brief process gate.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
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

PROVIDER_ID = "aeris"
DOMAIN = "radar"
_TILE_BASE = "https://maps.api.xweather.com"
# TTL deviation: ADR-017 default for radar frame metadata is 5 min; 60s used
# for parity with 3b-14 keyless precedent (same conscious deviation).
_FRAMES_TTL = 60   # 60 s — frame index (synthesized)
_TILE_TTL = 300    # 300 s — tile bytes per ADR-017 tile default
_API_VERSION = "0.1.0"

ATTRIBUTION = "AerisWeather / Xweather (https://www.xweather.com/)"

# ---------------------------------------------------------------------------
# Capability declaration (ADR-038 §4)
# ---------------------------------------------------------------------------

CAPABILITY = ProviderCapability(
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    supplied_canonical_fields=(),  # radar has no canonical-entity mapping (§4.5)
    geographic_coverage="global",
    auth_required=("client_id", "client_secret"),
    default_poll_interval_seconds=_FRAMES_TTL,
    operator_notes=(
        "AerisWeather/Xweather Raster Maps — radar layer (global radar mosaic). "
        "Keyed (path-embedded client_id_client_secret); reuses provider-scoped "
        "credentials from forecast/alerts/AQI Aeris (WEEWX_CLEARSKIES_AERIS_CLIENT_ID "
        "+ _AERIS_CLIENT_SECRET env vars). Free path via PWSWeather Contributor Plan "
        "(per ADR-015) — confirm at fixture capture; flag if access has tightened. "
        "Current-only at v0.1; ?t query param ignored. Time-step animation deferred. "
        "Tile bytes cached 300s; frame index synthesized + cached 60s. "
        "URL-credential redaction: log_url helper sanitizes path before any logging "
        "(security baseline — Aeris is the only path-credential provider in the codebase)."
    ),
    # LC-D: tile_url_template is public-shape — {auth} placeholder, not real credentials.
    tile_url_template="https://maps.api.xweather.com/{auth}/radar/{z}/{x}/{y}/current.png",
    wms_endpoint_url=None,
    wms_layer_name=None,
    tile_content_type="image/png",
)

# ---------------------------------------------------------------------------
# Rate limiter (LC-J) — polite-use guard
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="aeris-radar",
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
# URL credential redaction helper (LC-E — SECURITY BASELINE)
# ---------------------------------------------------------------------------

# Matches the "{client_id}_{client_secret}" path segment in the Aeris tile URL.
# The segment is the first path component after the host, e.g.:
#   https://maps.api.xweather.com/myid_mysecret/radar/4/4/6/current.png
#                                  ^^^^^^^^^^^^^^
# Regex: after the host and a slash, capture word characters (id),
# an underscore, word characters (secret), followed by a slash.
_AERIS_CRED_SEGMENT_RE = re.compile(
    r"(https://maps\.api\.xweather\.com/)[^/]+(/)"
)


def _redact_url(
    url: str,
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> str:
    """Replace the Aeris path-credential segment with '<redacted>'.

    Input:
        https://maps.api.xweather.com/myid_mysecret/radar/4/4/6/current.png
    Output:
        https://maps.api.xweather.com/<redacted>/radar/4/4/6/current.png

    When client_id and client_secret are provided, replaces the literal
    "{client_id}_{client_secret}" substring directly (more precise than regex
    for edge-case credential strings).  Falls back to regex when credentials
    are not provided (e.g., for use in tests with raw URLs only).

    Used BEFORE any logger.* call in this module AND passed to
    ProviderHTTPClient.get(log_url=...) to prevent key leakage in INFO logs (LC-E).
    """
    if client_id and client_secret:
        return url.replace(f"{client_id}_{client_secret}", "<redacted>", 1)
    return _AERIS_CRED_SEGMENT_RE.sub(r"\1<redacted>\2", url)


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
    Credentials NOT in key (privacy/leakage; same as AQI keyed modules).
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


def get_frames(*, client_id: str | None = None, client_secret: str | None = None) -> RadarFrameList:
    """Return a single synthesized 'current' frame for Aeris radar (LC-G).

    Aeris Raster Maps has no frame-index API at v0.1 (time-step animation
    deferred per LC-F).  Frame is synthesized at request time and cached 60s.

    Args:
        client_id: Aeris client ID.  LC-I: raises KeyInvalid if empty/None.
        client_secret: Aeris client secret.  LC-I: raises KeyInvalid if empty/None.

    Returns:
        RadarFrameList with a single kind="current" frame.

    Raises:
        KeyInvalid: credentials are None or empty.
    """
    # LC-I: empty-credential guard at provider entrypoint.
    if not client_id or not client_secret:
        raise KeyInvalid(
            "Aeris radar: client_id and/or client_secret missing. "
            "Set WEEWX_CLEARSKIES_AERIS_CLIENT_ID + WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET.",
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
    client_id: str | None = None,
    client_secret: str | None = None,
) -> tuple[bytes, str]:
    """Fetch a single tile from Xweather Raster Maps and return (bytes, content_type).

    This is one of two non-JSON entrypoints in the codebase.  The response
    is raw binary PNG bytes — no Pydantic response model, no JSON body.
    The endpoint handler (endpoints/radar.py get_radar_tile) wraps these bytes
    in fastapi.Response(content=bytes, media_type=ct).

    SECURITY: client_id + client_secret are embedded in the URL path.
    _redact_url() is called BEFORE any log message and passed as log_url to
    ProviderHTTPClient.get() to prevent credential leakage in INFO logs (LC-E).

    Cache uses a base64 envelope (LC-A):
      {"_tile_b64": "<base64>", "content_type": "image/png"}
    TTL: 300s (LC-B, ADR-017 tile-bytes default).

    Args:
        z: Slippy-map zoom level (0-22).
        x: Tile X coordinate.
        y: Tile Y coordinate.
        t: Optional frame timestamp (ISO-8601 UTC).  IGNORED at v0.1 (LC-F);
            Aeris Raster Maps time-step is deferred.  Logged at DEBUG.
        client_id: Aeris client ID.  LC-I: raises KeyInvalid if empty/None.
        client_secret: Aeris client secret.  LC-I: raises KeyInvalid if empty/None.

    Returns:
        (tile_bytes, content_type) — caller wraps in fastapi.Response.

    Raises:
        KeyInvalid: credentials missing.
        QuotaExhausted: upstream 429.
        TransientNetworkError: network/5xx after retries.
        ProviderProtocolError: unexpected 4xx (including 404 for out-of-domain
            tiles — status_code=404 on the exception; endpoint maps to HTTP 404).
    """
    # LC-I: empty-credential guard at provider entrypoint.
    if not client_id or not client_secret:
        raise KeyInvalid(
            "Aeris radar: client_id and/or client_secret missing. "
            "Set WEEWX_CLEARSKIES_AERIS_CLIENT_ID + WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET.",
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

    # Build tile URL — credentials in path (LC-E).
    auth_segment = f"{client_id}_{client_secret}"
    tile_url = f"{_TILE_BASE}/{auth_segment}/radar/{z}/{x}/{y}/current.png"

    # LC-E: compute redacted URL BEFORE any log call.
    # Pass client_id/client_secret for precise string replacement (more reliable
    # than the regex approach for edge-case credential strings).
    redacted_url = _redact_url(tile_url, client_id=client_id, client_secret=client_secret)
    logger.debug(
        "[%s] tile fetch: %s",
        PROVIDER_ID,
        redacted_url,
    )

    # Pass log_url to suppress credential exposure in http.py INFO log (LC-E).
    response = _get_http_client().get(tile_url, log_url=redacted_url)

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
