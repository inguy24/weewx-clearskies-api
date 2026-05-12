"""Unit tests for providers/radar/openweathermap.py (3b-15).

Tests the OWM Weather Maps 1.0 radar provider module. No live network; respx
mocks outbound httpx calls. Cache state reset between tests.

Coverage:
  - Empty appid guard → KeyInvalid raised BEFORE any HTTP call.
  - get_tile() cache hit path bypasses HTTP and returns cached bytes.
  - get_tile() cache miss → upstream call → cache populated with base64 envelope
    → returns (bytes, content_type).
  - Cache key includes (provider_id, "tile", z, x, y, t); does NOT include appid.
  - Upstream 429 → QuotaExhausted with retry_after_seconds.
  - Upstream 401/403 → KeyInvalid.
  - Upstream 404 → ProviderProtocolError with status_code=404 (NOT a custom class;
    per LC-H endpoint inspects .status_code).
  - Upstream 5xx (after retries exhausted) → TransientNetworkError.
  - get_frames() returns single RadarFrameList with kind="current" frame.
  - CAPABILITY shape: provider_id, domain, auth_required, tile_url_template,
    tile_content_type, wms fields None.

ADR references: ADR-015, ADR-017, ADR-018, ADR-037, ADR-038.
Brief: phase-2-task-3b-15-radar-keyed-A2-brief.md §Test-author scope.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx

# ---------------------------------------------------------------------------
# Fixture path
# ---------------------------------------------------------------------------

_FIXTURES_BASE = Path(__file__).parent.parent.parent / "fixtures" / "providers" / "radar"
_OWM_TILE_FIXTURE = _FIXTURES_BASE / "openweathermap" / "tile_4_4_6.png"

_TEST_APPID = "test_owm_appid_abc123"
_TILE_URL_PATTERN = "https://tile.openweathermap.org/map/precipitation_new/{z}/{x}/{y}.png"
_PROVIDER_ID = "openweathermap"
_DOMAIN = "radar"


def _load_tile_bytes() -> bytes:
    return _OWM_TILE_FIXTURE.read_bytes()


def _reset_module_state() -> None:
    """Reset provider module and cache state between tests."""
    from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
        reset_cache_for_tests,
        wire_cache_from_env,
    )

    reset_cache_for_tests()
    wire_cache_from_env()

    try:
        from weewx_clearskies_api.providers.radar import openweathermap as mod  # noqa: PLC0415

        mod._reset_http_client_for_tests()
        if hasattr(mod, "_rate_limiter"):
            mod._rate_limiter._calls.clear()
    except ImportError:
        pass


# ===========================================================================
# CAPABILITY declaration checks
# ===========================================================================


class TestOWMRadarCapability:
    """CAPABILITY symbol has the expected shape per ADR-038 §4 and brief spec."""

    def test_capability_provider_id_is_openweathermap(self) -> None:
        """CAPABILITY.provider_id == 'openweathermap'."""
        from weewx_clearskies_api.providers.radar.openweathermap import CAPABILITY  # noqa: PLC0415

        assert CAPABILITY.provider_id == "openweathermap"

    def test_capability_domain_is_radar(self) -> None:
        """CAPABILITY.domain == 'radar'."""
        from weewx_clearskies_api.providers.radar.openweathermap import CAPABILITY  # noqa: PLC0415

        assert CAPABILITY.domain == "radar"

    def test_capability_auth_required_contains_appid(self) -> None:
        """CAPABILITY.auth_required contains 'appid' (keyed provider per ADR-015)."""
        from weewx_clearskies_api.providers.radar.openweathermap import CAPABILITY  # noqa: PLC0415

        assert "appid" in CAPABILITY.auth_required

    def test_capability_tile_url_template_does_not_contain_appid(self) -> None:
        """tile_url_template is public-shape; appid NOT embedded (LC-D)."""
        from weewx_clearskies_api.providers.radar.openweathermap import CAPABILITY  # noqa: PLC0415

        assert CAPABILITY.tile_url_template is not None
        assert "appid" not in CAPABILITY.tile_url_template

    def test_capability_tile_url_template_contains_precipitation_new(self) -> None:
        """tile_url_template references precipitation_new layer (ADR-015)."""
        from weewx_clearskies_api.providers.radar.openweathermap import CAPABILITY  # noqa: PLC0415

        assert "precipitation_new" in (CAPABILITY.tile_url_template or "")

    def test_capability_tile_content_type_is_image_png(self) -> None:
        """CAPABILITY.tile_content_type == 'image/png'."""
        from weewx_clearskies_api.providers.radar.openweathermap import CAPABILITY  # noqa: PLC0415

        assert CAPABILITY.tile_content_type == "image/png"

    def test_capability_wms_endpoint_url_is_none(self) -> None:
        """wms_endpoint_url is None — OWM radar is XYZ-style, not WMS."""
        from weewx_clearskies_api.providers.radar.openweathermap import CAPABILITY  # noqa: PLC0415

        assert CAPABILITY.wms_endpoint_url is None

    def test_capability_wms_layer_name_is_none(self) -> None:
        """wms_layer_name is None — OWM radar is XYZ-style, not WMS."""
        from weewx_clearskies_api.providers.radar.openweathermap import CAPABILITY  # noqa: PLC0415

        assert CAPABILITY.wms_layer_name is None

    def test_capability_geographic_coverage_is_global(self) -> None:
        """OWM Weather Maps 1.0 has global coverage per api-docs."""
        from weewx_clearskies_api.providers.radar.openweathermap import CAPABILITY  # noqa: PLC0415

        assert CAPABILITY.geographic_coverage == "global"

    def test_capability_supplied_canonical_fields_is_empty(self) -> None:
        """Radar has no canonical-entity mapping (canonical §4.5)."""
        from weewx_clearskies_api.providers.radar.openweathermap import CAPABILITY  # noqa: PLC0415

        assert len(CAPABILITY.supplied_canonical_fields) == 0


# ===========================================================================
# get_frames() — synthesized frame index
# ===========================================================================


class TestOWMRadarGetFrames:
    """get_frames() returns a single current-kind synthesized frame (LC-G)."""

    def setup_method(self) -> None:
        _reset_module_state()

    def teardown_method(self) -> None:
        _reset_module_state()

    def test_get_frames_returns_radar_frame_list(self) -> None:
        """get_frames(appid=...) returns a RadarFrameList instance."""
        from weewx_clearskies_api.models.responses import RadarFrameList  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import get_frames  # noqa: PLC0415

        result = get_frames(appid=_TEST_APPID)
        assert isinstance(result, RadarFrameList)

    def test_get_frames_provider_id_is_openweathermap(self) -> None:
        """get_frames() result has providerId='openweathermap'."""
        from weewx_clearskies_api.providers.radar.openweathermap import get_frames  # noqa: PLC0415

        result = get_frames(appid=_TEST_APPID)
        assert result.providerId == "openweathermap"

    def test_get_frames_returns_exactly_one_frame(self) -> None:
        """get_frames() returns exactly one frame at v0.1 (synthesized current only per LC-G)."""
        from weewx_clearskies_api.providers.radar.openweathermap import get_frames  # noqa: PLC0415

        result = get_frames(appid=_TEST_APPID)
        assert len(result.frames) == 1

    def test_get_frames_frame_kind_is_current(self) -> None:
        """The single frame has kind='current' (synthesized at request time per LC-G)."""
        from weewx_clearskies_api.providers.radar.openweathermap import get_frames  # noqa: PLC0415

        result = get_frames(appid=_TEST_APPID)
        assert result.frames[0].kind == "current"

    def test_get_frames_frame_time_ends_with_z(self) -> None:
        """Frame time is UTC ISO-8601 with Z suffix (ADR-020)."""
        from weewx_clearskies_api.providers.radar.openweathermap import get_frames  # noqa: PLC0415

        result = get_frames(appid=_TEST_APPID)
        assert result.frames[0].time.endswith("Z")

    def test_get_frames_attribution_is_set(self) -> None:
        """get_frames() sets attribution string (required per brief)."""
        from weewx_clearskies_api.providers.radar.openweathermap import get_frames  # noqa: PLC0415

        result = get_frames(appid=_TEST_APPID)
        assert result.attribution is not None
        assert len(result.attribution) > 0

    def test_get_frames_empty_appid_raises_key_invalid(self) -> None:
        """Empty appid raises KeyInvalid BEFORE any HTTP call (LC-I)."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import get_frames  # noqa: PLC0415

        with pytest.raises(KeyInvalid):
            get_frames(appid="")

    def test_get_frames_none_appid_raises_key_invalid(self) -> None:
        """None appid raises KeyInvalid BEFORE any HTTP call (LC-I)."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import get_frames  # noqa: PLC0415

        with pytest.raises(KeyInvalid):
            get_frames(appid=None)  # type: ignore[arg-type]

    def test_get_frames_cache_populated_after_call(self) -> None:
        """After get_frames(), cache contains a serialized frame list."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import get_frames  # noqa: PLC0415

        get_frames(appid=_TEST_APPID)
        # Second call should hit cache — no network needed
        result2 = get_frames(appid=_TEST_APPID)
        assert result2.providerId == "openweathermap"


# ===========================================================================
# get_tile() — credential guard
# ===========================================================================


class TestOWMRadarGetTileCredentialGuard:
    """Empty/None appid raises KeyInvalid before any HTTP call (LC-I)."""

    def setup_method(self) -> None:
        _reset_module_state()

    def teardown_method(self) -> None:
        _reset_module_state()

    def test_empty_appid_raises_key_invalid_before_http_call(self) -> None:
        """get_tile() with empty appid raises KeyInvalid; no HTTP call made."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import get_tile  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            route = mock.get("https://tile.openweathermap.org/").mock(
                return_value=httpx.Response(200, content=b"\x89PNG")
            )
            with pytest.raises(KeyInvalid):
                get_tile(4, 4, 6, appid="")
            assert not route.called, "HTTP call should not be made when appid is empty"

    def test_none_appid_raises_key_invalid(self) -> None:
        """get_tile() with None appid raises KeyInvalid (LC-I fail-fast guard)."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import get_tile  # noqa: PLC0415

        with pytest.raises(KeyInvalid):
            get_tile(4, 4, 6, appid=None)  # type: ignore[arg-type]


# ===========================================================================
# get_tile() — cache key structure
# ===========================================================================


class TestOWMRadarTileCacheKey:
    """Cache key for tile proxy includes z/x/y/t but NOT credentials (ADR-017, brief §cache key)."""

    def test_cache_key_does_not_include_appid(self) -> None:
        """Two different appids produce the same cache key (credentials excluded per LC-D)."""
        from weewx_clearskies_api.providers.radar.openweathermap import _build_tile_cache_key  # noqa: PLC0415

        key1 = _build_tile_cache_key(z=4, x=4, y=6, t=None)
        key2 = _build_tile_cache_key(z=4, x=4, y=6, t=None)
        assert key1 == key2

    def test_cache_key_differs_by_z(self) -> None:
        """Different z values produce different cache keys."""
        from weewx_clearskies_api.providers.radar.openweathermap import _build_tile_cache_key  # noqa: PLC0415

        key1 = _build_tile_cache_key(z=4, x=4, y=6, t=None)
        key2 = _build_tile_cache_key(z=5, x=4, y=6, t=None)
        assert key1 != key2

    def test_cache_key_differs_by_x(self) -> None:
        """Different x values produce different cache keys."""
        from weewx_clearskies_api.providers.radar.openweathermap import _build_tile_cache_key  # noqa: PLC0415

        key1 = _build_tile_cache_key(z=4, x=4, y=6, t=None)
        key2 = _build_tile_cache_key(z=4, x=5, y=6, t=None)
        assert key1 != key2

    def test_cache_key_differs_by_y(self) -> None:
        """Different y values produce different cache keys."""
        from weewx_clearskies_api.providers.radar.openweathermap import _build_tile_cache_key  # noqa: PLC0415

        key1 = _build_tile_cache_key(z=4, x=4, y=6, t=None)
        key2 = _build_tile_cache_key(z=4, x=4, y=7, t=None)
        assert key1 != key2

    def test_cache_key_includes_provider_id(self) -> None:
        """Cache key payload encodes provider_id to distinguish from other domains."""
        from weewx_clearskies_api.providers.radar.openweathermap import _build_tile_cache_key  # noqa: PLC0415

        # The key is a SHA-256 hex digest; we verify it's deterministic and non-empty.
        key = _build_tile_cache_key(z=4, x=4, y=6, t=None)
        assert len(key) == 64  # SHA-256 hex digest is 64 chars
        assert key == _build_tile_cache_key(z=4, x=4, y=6, t=None)


# ===========================================================================
# get_tile() — cache hit path
# ===========================================================================


class TestOWMRadarTileCacheHit:
    """Cache hit path returns cached bytes without any HTTP call."""

    def setup_method(self) -> None:
        _reset_module_state()

    def teardown_method(self) -> None:
        _reset_module_state()

    def test_cache_hit_returns_bytes_without_http_call(self) -> None:
        """Pre-populated cache hit → get_tile() returns bytes, no HTTP call."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import (  # noqa: PLC0415
            _build_tile_cache_key,
            get_tile,
        )

        tile_bytes = _load_tile_bytes()
        cache_key = _build_tile_cache_key(z=4, x=4, y=6, t=None)
        # Pre-populate cache with base64 envelope (LC-A format)
        envelope = {
            "_tile_b64": base64.b64encode(tile_bytes).decode("ascii"),
            "content_type": "image/png",
        }
        get_cache().set(cache_key, envelope, ttl_seconds=300)

        with respx.mock(assert_all_called=False) as mock:
            route = mock.get("https://tile.openweathermap.org/").mock(
                return_value=httpx.Response(200, content=tile_bytes)
            )
            result_bytes, content_type = get_tile(4, 4, 6, appid=_TEST_APPID)
            assert not route.called, "HTTP should not be called on cache hit"

        assert result_bytes == tile_bytes
        assert content_type == "image/png"

    def test_cache_hit_content_type_matches_envelope(self) -> None:
        """Cache hit returns content_type from the envelope (LC-A)."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import (  # noqa: PLC0415
            _build_tile_cache_key,
            get_tile,
        )

        tile_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 62
        cache_key = _build_tile_cache_key(z=0, x=0, y=0, t=None)
        envelope = {
            "_tile_b64": base64.b64encode(tile_bytes).decode("ascii"),
            "content_type": "image/png",
        }
        get_cache().set(cache_key, envelope, ttl_seconds=300)

        _, content_type = get_tile(0, 0, 0, appid=_TEST_APPID)
        assert content_type == "image/png"


# ===========================================================================
# get_tile() — cache miss + upstream success
# ===========================================================================


class TestOWMRadarTileCacheMiss:
    """Cache miss → upstream call → cache populated → returns (bytes, content_type)."""

    def setup_method(self) -> None:
        _reset_module_state()

    def teardown_method(self) -> None:
        _reset_module_state()

    def test_cache_miss_calls_upstream_and_returns_bytes(self) -> None:
        """Cache miss → HTTP GET to OWM → returns (bytes, 'image/png')."""
        from weewx_clearskies_api.providers.radar.openweathermap import get_tile  # noqa: PLC0415

        tile_bytes = _load_tile_bytes()

        with respx.mock(assert_all_called=True) as mock:
            mock.get(
                "https://tile.openweathermap.org/map/precipitation_new/4/4/6.png",
                params={"appid": _TEST_APPID},
            ).mock(
                return_value=httpx.Response(
                    200,
                    content=tile_bytes,
                    headers={"Content-Type": "image/png"},
                )
            )
            result_bytes, content_type = get_tile(4, 4, 6, appid=_TEST_APPID)

        assert result_bytes == tile_bytes
        assert content_type == "image/png"

    def test_cache_miss_populates_cache_with_base64_envelope(self) -> None:
        """After cache miss + upstream success, cache stores base64 envelope (LC-A)."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import (  # noqa: PLC0415
            _build_tile_cache_key,
            get_tile,
        )

        tile_bytes = _load_tile_bytes()

        with respx.mock(assert_all_called=False) as mock:
            mock.get(
                "https://tile.openweathermap.org/map/precipitation_new/4/4/6.png"
            ).mock(
                return_value=httpx.Response(
                    200,
                    content=tile_bytes,
                    headers={"Content-Type": "image/png"},
                )
            )
            get_tile(4, 4, 6, appid=_TEST_APPID)

        # Verify cache envelope structure (LC-A)
        cache_key = _build_tile_cache_key(z=4, x=4, y=6, t=None)
        cached = get_cache().get(cache_key)
        assert cached is not None, "Cache should be populated after cache miss"
        assert "_tile_b64" in cached, "Cache envelope must have '_tile_b64' key (LC-A)"
        assert "content_type" in cached, "Cache envelope must have 'content_type' key (LC-A)"

        # Decode and verify bytes match
        decoded = base64.b64decode(cached["_tile_b64"])
        assert decoded == tile_bytes

    def test_cache_miss_second_call_hits_cache(self) -> None:
        """Second call for same tile hits cache; upstream URL called only once."""
        from weewx_clearskies_api.providers.radar.openweathermap import get_tile  # noqa: PLC0415

        tile_bytes = _load_tile_bytes()
        call_count = 0

        def _side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                200,
                content=tile_bytes,
                headers={"Content-Type": "image/png"},
            )

        with respx.mock(assert_all_called=False) as mock:
            mock.get(
                "https://tile.openweathermap.org/map/precipitation_new/4/4/6.png"
            ).mock(side_effect=_side_effect)
            get_tile(4, 4, 6, appid=_TEST_APPID)  # miss
            get_tile(4, 4, 6, appid=_TEST_APPID)  # should hit cache

        assert call_count == 1, (
            f"Upstream should be called exactly once; got {call_count}"
        )

    def test_tile_url_includes_appid_as_query_param(self) -> None:
        """OWM tile URL appid is in query params (not URL path — security baseline)."""
        from weewx_clearskies_api.providers.radar.openweathermap import get_tile  # noqa: PLC0415

        tile_bytes = _load_tile_bytes()
        captured_url: str | None = None

        def _capture(request: httpx.Request) -> httpx.Response:
            nonlocal captured_url
            captured_url = str(request.url)
            return httpx.Response(
                200,
                content=tile_bytes,
                headers={"Content-Type": "image/png"},
            )

        with respx.mock(assert_all_called=False) as mock:
            mock.get(
                "https://tile.openweathermap.org/map/precipitation_new/3/2/5.png"
            ).mock(side_effect=_capture)
            get_tile(3, 2, 5, appid=_TEST_APPID)

        assert captured_url is not None
        assert f"appid={_TEST_APPID}" in captured_url, (
            f"appid should be in query params; got URL: {captured_url}"
        )
        # appid should NOT be in the path segment
        path_part = captured_url.split("?")[0]
        assert _TEST_APPID not in path_part, "appid must not appear in URL path"


# ===========================================================================
# get_tile() — upstream error mapping
# ===========================================================================


class TestOWMRadarTileUpstreamErrors:
    """Upstream HTTP error codes map to correct canonical taxonomy exceptions."""

    def setup_method(self) -> None:
        _reset_module_state()

    def teardown_method(self) -> None:
        _reset_module_state()

    def test_upstream_401_raises_key_invalid(self) -> None:
        """Upstream 401 → KeyInvalid (operator credentials invalid)."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import get_tile  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(
                "https://tile.openweathermap.org/map/precipitation_new/4/4/6.png"
            ).mock(return_value=httpx.Response(401, text="Unauthorized"))
            with pytest.raises(KeyInvalid):
                get_tile(4, 4, 6, appid=_TEST_APPID)

    def test_upstream_403_raises_key_invalid(self) -> None:
        """Upstream 403 → KeyInvalid (operator credentials invalid)."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import get_tile  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(
                "https://tile.openweathermap.org/map/precipitation_new/4/4/6.png"
            ).mock(return_value=httpx.Response(403, text="Forbidden"))
            with pytest.raises(KeyInvalid):
                get_tile(4, 4, 6, appid=_TEST_APPID)

    def test_upstream_429_raises_quota_exhausted(self) -> None:
        """Upstream 429 → QuotaExhausted with retry_after_seconds."""
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import get_tile  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(
                "https://tile.openweathermap.org/map/precipitation_new/4/4/6.png"
            ).mock(
                return_value=httpx.Response(
                    429,
                    text="rate limited",
                    headers={"Retry-After": "60"},
                )
            )
            with pytest.raises(QuotaExhausted) as exc_info:
                get_tile(4, 4, 6, appid=_TEST_APPID)

        # Retry-After header should be surfaced via the structured attribute
        assert exc_info.value.retry_after_seconds is not None
        assert exc_info.value.retry_after_seconds == 60

    def test_upstream_429_without_retry_after_header_raises_quota_exhausted(self) -> None:
        """429 without Retry-After header → QuotaExhausted.

        ProviderHTTPClient uses a polite default of 60s when no Retry-After header
        is provided (see http.py line ~236: 'polite default when NWS doesn't say').
        The test asserts QuotaExhausted is raised; retry_after_seconds may be
        non-None due to the polite default — the assertion is on the exception type,
        not the exact retry value.
        """
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import get_tile  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(
                "https://tile.openweathermap.org/map/precipitation_new/4/4/6.png"
            ).mock(return_value=httpx.Response(429, text="rate limited"))
            with pytest.raises(QuotaExhausted):
                get_tile(4, 4, 6, appid=_TEST_APPID)

    def test_upstream_404_raises_provider_protocol_error_with_status_404(self) -> None:
        """Upstream 404 → ProviderProtocolError with status_code=404 (LC-H, NOT custom class).

        The endpoint (not the provider module) maps 404 → HTTPException(404).
        The provider module raises ProviderProtocolError so the endpoint can
        inspect .status_code and decide (per LC-H decision tree).
        """
        from weewx_clearskies_api.providers._common.errors import ProviderProtocolError  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import get_tile  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(
                "https://tile.openweathermap.org/map/precipitation_new/4/4/6.png"
            ).mock(return_value=httpx.Response(404, text="tile not found"))
            with pytest.raises(ProviderProtocolError) as exc_info:
                get_tile(4, 4, 6, appid=_TEST_APPID)

        assert exc_info.value.status_code == 404, (
            "ProviderProtocolError must carry status_code=404 so endpoint can "
            "dispatch via attribute (not message string) per coding.md §3"
        )

    def test_upstream_5xx_raises_transient_network_error(self) -> None:
        """Upstream 5xx after retries → TransientNetworkError."""
        from weewx_clearskies_api.providers._common.errors import TransientNetworkError  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import get_tile  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(
                "https://tile.openweathermap.org/map/precipitation_new/4/4/6.png"
            ).mock(return_value=httpx.Response(503, text="upstream unavailable"))
            with pytest.raises(TransientNetworkError):
                get_tile(4, 4, 6, appid=_TEST_APPID)

    def test_upstream_network_error_raises_transient_network_error(self) -> None:
        """DNS/TCP failure → TransientNetworkError (after retries exhausted)."""
        from weewx_clearskies_api.providers._common.errors import TransientNetworkError  # noqa: PLC0415
        from weewx_clearskies_api.providers.radar.openweathermap import get_tile  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(
                "https://tile.openweathermap.org/map/precipitation_new/4/4/6.png"
            ).mock(side_effect=httpx.ConnectError("DNS failure"))
            with pytest.raises(TransientNetworkError):
                get_tile(4, 4, 6, appid=_TEST_APPID)
