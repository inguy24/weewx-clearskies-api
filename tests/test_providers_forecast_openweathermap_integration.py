"""Integration tests for the OpenWeatherMap forecast provider (3b round 5).

All tests carry @pytest.mark.integration and run against the docker-compose
dev/test stack (MariaDB or SQLite backend per BACKEND env var).

Same integration suite runs twice per ADR-012 (once MariaDB, once SQLite)
to catch dialect drift. The forecast endpoint itself has no DB dependency,
but running in the real stack confirms the endpoint is DB-stack-agnostic
and wires correctly alongside the DB-backed endpoints.

Redis-backed integration tests carry both @pytest.mark.integration and
@pytest.mark.redis and are skipped unless `pytest -m "integration and redis"`
is used. Per brief §Process gates: the Redis tier MUST PASS, not skip.
If Redis is not reachable on weather-dev, this is a brief-gate failure
that must be surfaced to the lead via SendMessage BEFORE closeout.

Cache integration covered:
  - Memory cache: miss → fetch (1 OWM call) → bundle returned.
  - Memory cache: second fetch hits cache → 0 OWM outbound calls.
  - Redis cache: miss → fetch (1 OWM call) → bundle stored in Redis.
  - Redis cache: second fetch hits Redis cache → 0 OWM outbound calls.

Dispatch table:
  - ('forecast', 'openweathermap') is in PROVIDER_MODULES dispatch table.

Startup wiring:
  - ForecastSettings with provider='openweathermap' passes validate().
  - wire_openweathermap_credentials() picks up env var.

Q1 path:
  - Basic-tier 401 → empty bundle via /api/v1/forecast → 200 with empty arrays.

ADR references: ADR-006, ADR-007, ADR-012, ADR-017, ADR-019, ADR-027, ADR-038.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Generator

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

_BACKEND = os.environ.get("BACKEND", "mariadb").lower()
_MARIADB_HOST_PORT = os.environ.get("MARIADB_HOST_PORT", "3307")
_MARIADB_DB = os.environ.get("MARIADB_DATABASE", "weewx")
_MARIADB_RO_PASSWORD = os.environ.get("MARIADB_RO_PASSWORD", "clearskies_ro_test")
_SQLITE_SDB_PATH = os.environ.get(
    "SQLITE_SDB_PATH",
    os.path.join(os.environ.get("SQLITE_DATA_PATH", "/tmp"), "weewx.sdb"),
)
_REDIS_URL = os.environ.get("CLEARSKIES_CACHE_URL", "redis://localhost:6379/0")

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "providers" / "openweathermap"

_LAT = 47.6062
_LON = -122.3321
_OWM_ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"
_TEST_APPID = "INTEGRATION_TEST_APPID_12345"


def _load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture from tests/fixtures/providers/openweathermap/."""
    path = _FIXTURES_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return json.loads(fh.read())


def _require_mariadb_password() -> None:
    if not _MARIADB_RO_PASSWORD:
        pytest.skip("MARIADB_RO_PASSWORD not set; start dev stack and set env var")


def _require_sqlite_file() -> None:
    try:
        exists = Path(_SQLITE_SDB_PATH).exists()
    except PermissionError:
        pytest.skip(f"Cannot access {_SQLITE_SDB_PATH} (PermissionError)")
    if not exists:
        pytest.skip(f"SQLite file not found: {_SQLITE_SDB_PATH}")


def _require_redis() -> None:
    """Skip if Redis is not reachable.

    Per brief §Brief-gate honesty: if Redis is not reachable, this function
    skips the individual test. The test-author must SendMessage the lead if
    the Redis tier is not passing before submitting the closeout.
    """
    try:
        import redis as redis_lib  # noqa: PLC0415
        r = redis_lib.Redis.from_url(_REDIS_URL)
        r.ping()
    except (ImportError, ConnectionError, OSError) as exc:
        pytest.skip(
            f"Redis not reachable at {_REDIS_URL} ({type(exc).__name__}); "
            "start redis compose profile"
        )
    except Exception as exc:  # noqa: BLE001 — narrow to redis-py errors below
        import redis as _redis_lib  # noqa: PLC0415
        if isinstance(exc, _redis_lib.exceptions.RedisError):
            pytest.skip(
                f"Redis not reachable at {_REDIS_URL} ({type(exc).__name__}); "
                "start redis compose profile"
            )
        raise


# ---------------------------------------------------------------------------
# Engine fixtures (module-scoped — same DB for all integration tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mariadb_ro_engine() -> Generator[Engine, None, None]:
    """Module-scoped read-only MariaDB engine (clearskies_ro user)."""
    _require_mariadb_password()
    engine = create_engine(
        f"mysql+pymysql://clearskies_ro:{_MARIADB_RO_PASSWORD}"
        f"@127.0.0.1:{_MARIADB_HOST_PORT}/{_MARIADB_DB}?charset=utf8mb4",
        future=True,
        pool_pre_ping=True,
    )
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def sqlite_ro_engine() -> Generator[Engine, None, None]:
    """Module-scoped read-only SQLite engine."""
    _require_sqlite_file()
    from sqlalchemy.pool import NullPool  # noqa: PLC0415

    engine = create_engine(
        f"sqlite+pysqlite:///file:///{_SQLITE_SDB_PATH}?mode=ro&uri=true",
        poolclass=NullPool,
        future=True,
    )
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def db_engine(
    mariadb_ro_engine: Engine, sqlite_ro_engine: Engine
) -> Generator[Engine, None, None]:
    """Yield the appropriate engine based on BACKEND env var."""
    if _BACKEND == "sqlite":
        yield sqlite_ro_engine
    else:
        yield mariadb_ro_engine


# ---------------------------------------------------------------------------
# Shared wiring helper
# ---------------------------------------------------------------------------


def _wire_integration_stack(
    engine: Engine,
    forecast_provider: str | None = None,
    appid: str | None = None,
) -> tuple[Any, Any]:
    """Wire the full integration stack for a test app.

    Returns (settings, app) with DB, station, units, cache, providers all wired.
    Handles both MariaDB and SQLite backends identically.
    """
    from weewx_clearskies_api.app import create_app  # noqa: PLC0415
    from weewx_clearskies_api.config.settings import (  # noqa: PLC0415
        AlertsSettings,
        ApiSettings,
        DatabaseSettings,
        ForecastSettings,
        HealthSettings,
        LoggingSettings,
        RateLimitSettings,
        Settings,
    )
    from weewx_clearskies_api.db.reflection import STOCK_COLUMN_MAP, ColumnInfo, ColumnRegistry  # noqa: PLC0415
    from weewx_clearskies_api.db.registry import wire_registry  # noqa: PLC0415
    from weewx_clearskies_api.db.session import wire_engine  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
        reset_cache_for_tests,
        wire_cache_from_env,
    )
    from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
        ProviderCapability,
        reset_provider_registry_for_tests,
        wire_providers,
    )
    from weewx_clearskies_api.services import station as station_mod  # noqa: PLC0415
    from weewx_clearskies_api.services import units as units_mod  # noqa: PLC0415
    from weewx_clearskies_api.services.station import StationInfo, reset_cache  # noqa: PLC0415
    from weewx_clearskies_api.services.units import (  # noqa: PLC0415
        _GROUP_MEMBERS,
        _SYSTEM_PRESETS,
        reset_cache as reset_units_cache,
    )
    from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
        _reset_basic_tier_warned_for_tests,
        _reset_http_client_for_tests,
    )
    import weewx_clearskies_api.providers.forecast.openweathermap as _owm  # noqa: PLC0415
    from weewx_clearskies_api.endpoints import forecast as forecast_endpoint  # noqa: PLC0415

    # Reset state
    reset_cache_for_tests()
    reset_provider_registry_for_tests()
    _reset_http_client_for_tests()
    _reset_basic_tier_warned_for_tests()
    _owm._rate_limiter._calls.clear()

    # Wire OWM credentials for the integration stack
    forecast_endpoint.wire_openweathermap_credentials(appid or _TEST_APPID)

    # Wire DB
    wire_engine(engine)
    registry = ColumnRegistry()
    registry.stock = {
        col: ColumnInfo(db_name=col, canonical_name=canon, is_stock=True)
        for col, canon in STOCK_COLUMN_MAP.items()
    }
    wire_registry(registry)

    # Wire station — Seattle coordinates match OWM fixtures
    reset_cache()
    station_mod._cached_station = StationInfo(
        station_id="integration-test-station",
        name="Integration Test Station",
        latitude=_LAT,
        longitude=_LON,
        altitude=100.0,
        timezone="America/Los_Angeles",
        timezone_offset_minutes=-420,
        unit_system="US",
        hardware=None,
    )

    # Wire units
    reset_units_cache()
    system_map = _SYSTEM_PRESETS["US"]
    block: dict[str, str] = {}
    for group, unit in system_map.items():
        for field in _GROUP_MEMBERS.get(group, []):
            block[field] = unit
    units_mod._cached_units_block = block
    units_mod._cached_target_unit = "US"

    # Wire cache
    wire_cache_from_env()

    # Build capability list
    capabilities: list[ProviderCapability] = []
    if forecast_provider == "openweathermap":
        from weewx_clearskies_api.providers.forecast import openweathermap as forecast_owm  # noqa: PLC0415
        capabilities.append(forecast_owm.CAPABILITY)

    wire_providers(capabilities)

    settings = Settings(
        api=ApiSettings({}),
        health=HealthSettings({}),
        logging_settings=LoggingSettings({}),
        ratelimit=RateLimitSettings({}),
        database=DatabaseSettings({}),
        alerts=AlertsSettings({}),
        forecast=ForecastSettings({"provider": forecast_provider} if forecast_provider else {}),
    )
    app = create_app(settings)
    return settings, app


def _reset_owm_state() -> None:
    """Reset OWM provider state between tests."""
    from weewx_clearskies_api.providers._common.cache import reset_cache_for_tests  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.capability import reset_provider_registry_for_tests  # noqa: PLC0415
    from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
        _reset_basic_tier_warned_for_tests,
        _reset_http_client_for_tests,
    )
    import weewx_clearskies_api.providers.forecast.openweathermap as _owm  # noqa: PLC0415

    reset_cache_for_tests()
    reset_provider_registry_for_tests()
    _reset_http_client_for_tests()
    _reset_basic_tier_warned_for_tests()
    _owm._rate_limiter._calls.clear()


# ---------------------------------------------------------------------------
# Integration app fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def integration_app_owm(db_engine: Engine) -> Generator[FastAPI, None, None]:
    """Integration app with OWM forecast provider configured."""
    _, app = _wire_integration_stack(db_engine, forecast_provider="openweathermap")
    yield app
    _reset_owm_state()


@pytest.fixture
def integration_client_owm(integration_app_owm: FastAPI) -> TestClient:
    """TestClient for the integration app with OWM configured."""
    return TestClient(integration_app_owm, raise_server_exceptions=False)


# ===========================================================================
# Integration test: dispatch table — openweathermap is in PROVIDER_MODULES
# ===========================================================================


class TestIntegrationDispatchTableHasOpenWeatherMap:
    """('forecast', 'openweathermap') is in the dispatch table as of 3b-5."""

    def test_openweathermap_is_in_forecast_dispatch_table(self) -> None:
        """get_provider_module(domain='forecast', provider_id='openweathermap') returns module."""
        from weewx_clearskies_api.providers._common.dispatch import get_provider_module  # noqa: PLC0415
        module = get_provider_module(domain="forecast", provider_id="openweathermap")
        assert module is not None
        assert hasattr(module, "CAPABILITY")
        assert hasattr(module, "fetch")
        assert module.CAPABILITY.provider_id == "openweathermap"

    def test_openweathermap_module_has_correct_domain(self) -> None:
        """OWM module from dispatch table has domain='forecast'."""
        from weewx_clearskies_api.providers._common.dispatch import get_provider_module  # noqa: PLC0415
        module = get_provider_module(domain="forecast", provider_id="openweathermap")
        assert module.CAPABILITY.domain == "forecast"


# ===========================================================================
# Integration test: /forecast endpoint with OWM + respx-mocked
# ===========================================================================


class TestIntegrationForecastOpenWeatherMap:
    """/forecast with OWM + respx-mocked → 200 source='openweathermap'."""

    def test_owm_returns_200_with_bundle(
        self, integration_client_owm: TestClient
    ) -> None:
        """/forecast with OWM configured → 200 with source='openweathermap'."""
        fixture = _load_fixture("onecall.json")
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(200, json=fixture)
            )
            response = integration_client_owm.get("/api/v1/forecast")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["source"] == "openweathermap"

    def test_owm_returns_48_hourly_and_8_daily(
        self, integration_client_owm: TestClient
    ) -> None:
        """/forecast with OWM → 48 hourly + 8 daily in bundle.

        Uses ?days=8 to request all 8 daily entries from OWM; the endpoint's
        default days=7 (ForecastQueryParams default) would otherwise slice the
        bundle to 7.  Hourly uses the default hours=48 which matches OWM's
        natural supply.
        """
        fixture = _load_fixture("onecall.json")
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(200, json=fixture)
            )
            response = integration_client_owm.get("/api/v1/forecast?days=8")
        body = response.json()
        assert len(body["data"]["hourly"]) == 48
        assert len(body["data"]["daily"]) == 8

    def test_owm_discussion_is_null(
        self, integration_client_owm: TestClient
    ) -> None:
        """discussion is null — OWM has no forecast discussion product (lead-call 33)."""
        fixture = _load_fixture("onecall.json")
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(200, json=fixture)
            )
            response = integration_client_owm.get("/api/v1/forecast")
        body = response.json()
        assert body["data"]["discussion"] is None

    def test_owm_generated_at_ends_with_z(
        self, integration_client_owm: TestClient
    ) -> None:
        """generatedAt ends with Z (UTC per ADR-020)."""
        fixture = _load_fixture("onecall.json")
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(200, json=fixture)
            )
            response = integration_client_owm.get("/api/v1/forecast")
        body = response.json()
        assert body["data"]["generatedAt"].endswith("Z")
        assert body["generatedAt"].endswith("Z")

    def test_owm_slice_params_respected(
        self, integration_client_owm: TestClient
    ) -> None:
        """hours=12&days=3 slice params applied after cache lookup."""
        fixture = _load_fixture("onecall.json")
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(200, json=fixture)
            )
            response = integration_client_owm.get(
                "/api/v1/forecast", params={"hours": 12, "days": 3}
            )
        body = response.json()
        assert len(body["data"]["hourly"]) == 12
        assert len(body["data"]["daily"]) == 3

    def test_owm_units_block_present(
        self, integration_client_owm: TestClient
    ) -> None:
        """units block is present in the response."""
        fixture = _load_fixture("onecall.json")
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(200, json=fixture)
            )
            response = integration_client_owm.get("/api/v1/forecast")
        body = response.json()
        assert "units" in body
        assert isinstance(body["units"], dict)


# ===========================================================================
# Integration test: Q1 path — basic-tier 401 → empty bundle, 200 response
# ===========================================================================


class TestIntegrationQ1BasicTierEmptyBundle:
    """Basic-tier 401 → empty bundle, 200 response (Q1 user decision 2026-05-08)."""

    def test_basic_tier_401_returns_200_with_empty_bundle(
        self, db_engine: Engine
    ) -> None:
        """OWM 401 (basic-tier) → 200 ForecastResponse with hourly=[], daily=[]."""
        _, app = _wire_integration_stack(db_engine, forecast_provider="openweathermap")
        client = TestClient(app, raise_server_exceptions=False)

        error_fixture = _load_fixture("error_401_basic_tier.json")
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(401, json=error_fixture)
            )
            response = client.get("/api/v1/forecast")

        assert response.status_code == 200, (
            f"Expected 200 for basic-tier 401 (Q1 decision), got {response.status_code}"
        )
        body = response.json()
        assert body["data"]["hourly"] == [], "Basic-tier 401 should return empty hourly"
        assert body["data"]["daily"] == [], "Basic-tier 401 should return empty daily"
        assert body["data"]["discussion"] is None
        assert body["data"]["source"] == "openweathermap"
        _reset_owm_state()

    def test_any_other_4xx_non_401_returns_502(
        self, db_engine: Engine
    ) -> None:
        """Any other 4xx (non-401, e.g. 403) → 502 ProviderProblem KeyInvalid."""
        _, app = _wire_integration_stack(db_engine, forecast_provider="openweathermap")
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(403, json={"cod": 403, "message": "Forbidden"})
            )
            response = client.get("/api/v1/forecast")

        assert response.status_code == 502, (
            f"Expected 502 for non-401 4xx, got {response.status_code}"
        )
        _reset_owm_state()

    def test_endpoint_missing_appid_returns_502_key_invalid(
        self, db_engine: Engine
    ) -> None:
        """OWM provider configured but appid unset → /api/v1/forecast returns 502 ProviderProblem KeyInvalid.

        Brief §Per-endpoint spec decision-tree branch 12: when [forecast]
        provider=openweathermap is set but WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID
        is unset, the endpoint translates the module's KeyInvalid to a
        502 ProviderProblem with errorCode="KeyInvalid" (RFC 9457).

        3b-5 audit F1 remediation 2026-05-09: prior coverage exercised the
        module-level KeyInvalid only; the endpoint-level translation was
        asserted nowhere.
        """
        from weewx_clearskies_api.endpoints import forecast as forecast_endpoint  # noqa: PLC0415

        _, app = _wire_integration_stack(db_engine, forecast_provider="openweathermap")
        client = TestClient(app, raise_server_exceptions=False)

        # Clear appid AFTER stack wiring — _wire_integration_stack calls
        # create_app() which wires the appid from env via
        # wire_forecast_settings(). The pre-clear must come after to override.
        forecast_endpoint.wire_openweathermap_credentials(None)

        # No respx mock — the request never reaches OWM (KeyInvalid raised
        # before any outbound call).
        response = client.get("/api/v1/forecast")

        assert response.status_code == 502, (
            f"Expected 502 for missing appid, got {response.status_code}"
        )
        body = response.json()
        assert body.get("errorCode") == "KeyInvalid", (
            f"Expected errorCode='KeyInvalid', got {body.get('errorCode')!r}"
        )
        _reset_owm_state()


# ===========================================================================
# Integration test: memory cache — miss → fetch → hit (both backends)
# ===========================================================================


class TestIntegrationOwmMemoryCacheMissAndHit:
    """OWM forecast provider: memory cache miss → fetch → cache hit flow."""

    def test_cache_miss_fetches_from_owm_and_stores_bundle(
        self, db_engine: Engine
    ) -> None:
        """Memory cache miss → one OWM HTTP call → bundle stored."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            reset_cache_for_tests,
            wire_cache_from_env,
            get_cache,
        )
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _reset_basic_tier_warned_for_tests,
            _reset_http_client_for_tests,
        )

        reset_cache_for_tests()
        _reset_http_client_for_tests()
        _reset_basic_tier_warned_for_tests()
        openweathermap._rate_limiter._calls.clear()
        wire_cache_from_env()

        fixture = _load_fixture("onecall.json")
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(200, json=fixture)
            )
            bundle = openweathermap.fetch(
                lat=_LAT,
                lon=_LON,
                target_unit="US",
                appid=_TEST_APPID,
            )
            call_count = mock.calls.call_count

        assert call_count == 1, f"Expected 1 OWM call on cache miss, got {call_count}"
        assert bundle.source == "openweathermap"
        assert len(bundle.hourly) == 48
        assert len(bundle.daily) == 8

        # Cache populated
        cache_key = openweathermap._build_cache_key(_LAT, _LON, "US")
        cached = get_cache().get(cache_key)
        assert cached is not None

        reset_cache_for_tests()
        _reset_http_client_for_tests()

    def test_cache_hit_skips_owm_calls_and_returns_same_bundle(
        self, db_engine: Engine
    ) -> None:
        """Memory cache hit → zero OWM HTTP calls; bundle matches cached."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            reset_cache_for_tests,
            wire_cache_from_env,
        )
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _reset_basic_tier_warned_for_tests,
            _reset_http_client_for_tests,
        )

        reset_cache_for_tests()
        _reset_http_client_for_tests()
        _reset_basic_tier_warned_for_tests()
        openweathermap._rate_limiter._calls.clear()
        wire_cache_from_env()

        fixture = _load_fixture("onecall.json")

        # First fetch — fills memory cache
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(200, json=fixture)
            )
            bundle1 = openweathermap.fetch(
                lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
            )

        # Second fetch — should come from cache (zero calls)
        with respx.mock(assert_all_called=False) as mock2:
            bundle2 = openweathermap.fetch(
                lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
            )
            cache_hit_calls = mock2.calls.call_count

        assert cache_hit_calls == 0, (
            f"Expected 0 OWM calls on cache hit, got {cache_hit_calls}"
        )
        assert bundle2.source == "openweathermap"
        assert len(bundle2.hourly) == len(bundle1.hourly)
        assert len(bundle2.daily) == len(bundle1.daily)

        reset_cache_for_tests()
        _reset_http_client_for_tests()


# ===========================================================================
# Integration test: Redis cache (optional, redis mark — MUST PASS per brief)
# ===========================================================================


@pytest.mark.redis
class TestIntegrationOwmRedisBackend:
    """Real Redis from the docker-compose redis profile.

    Per brief §Process gates: Redis tier MUST PASS, not skip.
    If Redis is not reachable on weather-dev, this is a brief-gate failure
    that must be surfaced to the lead via SendMessage BEFORE closeout.
    """

    def test_owm_forecast_with_real_redis_cache_miss_makes_one_call(
        self, db_engine: Engine
    ) -> None:
        """Redis cache miss → one OWM HTTP call → bundle stored in Redis."""
        _require_redis()

        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            RedisCache,
            reset_cache_for_tests,
        )
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _reset_basic_tier_warned_for_tests,
            _reset_http_client_for_tests,
        )

        reset_cache_for_tests()
        _reset_http_client_for_tests()
        _reset_basic_tier_warned_for_tests()
        openweathermap._rate_limiter._calls.clear()

        redis_cache = RedisCache(url=_REDIS_URL)
        redis_cache._client.flushdb()  # type: ignore[attr-defined]
        import weewx_clearskies_api.providers._common.cache as cache_mod  # noqa: PLC0415
        cache_mod._cache = redis_cache

        fixture = _load_fixture("onecall.json")

        try:
            with respx.mock(assert_all_called=False) as mock:
                mock.get(_OWM_ONECALL_URL).mock(
                    return_value=httpx.Response(200, json=fixture)
                )
                bundle = openweathermap.fetch(
                    lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
                )
                call_count = mock.calls.call_count

            assert call_count == 1, (
                f"Expected 1 OWM call on Redis cache miss, got {call_count}"
            )
            assert bundle.source == "openweathermap"
            assert len(bundle.hourly) == 48
            assert len(bundle.daily) == 8

            # Verify bundle is in Redis
            cache_key = openweathermap._build_cache_key(_LAT, _LON, "US")
            import weewx_clearskies_api.providers._common.cache as cache_mod2  # noqa: PLC0415
            cached = cache_mod2._cache.get(cache_key)
            assert cached is not None, "Bundle should be stored in Redis after cache miss"

        finally:
            redis_cache._client.flushdb()  # type: ignore[attr-defined]
            reset_cache_for_tests()
            _reset_http_client_for_tests()

    def test_owm_forecast_with_real_redis_cache_hit_skips_owm_calls(
        self, db_engine: Engine
    ) -> None:
        """Redis cache hit → zero OWM HTTP calls; bundle returned from Redis."""
        _require_redis()

        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            RedisCache,
            reset_cache_for_tests,
        )
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _reset_basic_tier_warned_for_tests,
            _reset_http_client_for_tests,
        )

        reset_cache_for_tests()
        _reset_http_client_for_tests()
        _reset_basic_tier_warned_for_tests()
        openweathermap._rate_limiter._calls.clear()

        redis_cache = RedisCache(url=_REDIS_URL)
        redis_cache._client.flushdb()  # type: ignore[attr-defined]
        import weewx_clearskies_api.providers._common.cache as cache_mod  # noqa: PLC0415
        cache_mod._cache = redis_cache

        fixture = _load_fixture("onecall.json")

        try:
            # First fetch — fills Redis cache
            with respx.mock(assert_all_called=False) as mock:
                mock.get(_OWM_ONECALL_URL).mock(
                    return_value=httpx.Response(200, json=fixture)
                )
                bundle1 = openweathermap.fetch(
                    lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
                )
                first_call_count = mock.calls.call_count

            assert first_call_count > 0, "First request should have called OWM API"
            assert bundle1.source == "openweathermap"

            # Second fetch — should hit Redis cache; zero calls
            with respx.mock(assert_all_called=False) as mock2:
                bundle2 = openweathermap.fetch(
                    lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
                )
                second_call_count = mock2.calls.call_count

            assert second_call_count == 0, (
                f"Expected 0 OWM calls on Redis cache hit, got {second_call_count}"
            )
            assert bundle2.source == "openweathermap"
            assert len(bundle2.hourly) == len(bundle1.hourly)
            assert len(bundle2.daily) == len(bundle1.daily)

        finally:
            redis_cache._client.flushdb()  # type: ignore[attr-defined]
            reset_cache_for_tests()
            _reset_http_client_for_tests()

    def test_owm_forecast_with_redis_discussion_none_round_trips(
        self, db_engine: Engine
    ) -> None:
        """Bundle discussion=None round-trips correctly through Redis."""
        _require_redis()

        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            RedisCache,
            reset_cache_for_tests,
        )
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _reset_basic_tier_warned_for_tests,
            _reset_http_client_for_tests,
        )

        reset_cache_for_tests()
        _reset_http_client_for_tests()
        _reset_basic_tier_warned_for_tests()
        openweathermap._rate_limiter._calls.clear()

        redis_cache = RedisCache(url=_REDIS_URL)
        redis_cache._client.flushdb()  # type: ignore[attr-defined]
        import weewx_clearskies_api.providers._common.cache as cache_mod  # noqa: PLC0415
        cache_mod._cache = redis_cache

        fixture = _load_fixture("onecall.json")

        try:
            # First fetch → stores in Redis
            with respx.mock(assert_all_called=False) as mock:
                mock.get(_OWM_ONECALL_URL).mock(
                    return_value=httpx.Response(200, json=fixture)
                )
                openweathermap.fetch(
                    lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
                )

            # Second fetch → from Redis; discussion should be None
            with respx.mock(assert_all_called=False):
                bundle2 = openweathermap.fetch(
                    lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
                )

            assert bundle2.discussion is None, (
                "discussion=None should survive Redis serialization round-trip"
            )

        finally:
            redis_cache._client.flushdb()  # type: ignore[attr-defined]
            reset_cache_for_tests()
            _reset_http_client_for_tests()


# ===========================================================================
# Integration test: startup wiring + validation
# ===========================================================================


class TestIntegrationOwmStartupWiring:
    """ForecastSettings with openweathermap provider wires and validates correctly."""

    def test_forecast_settings_with_openweathermap_passes_validate(self) -> None:
        """ForecastSettings({'provider': 'openweathermap'}).validate() does not raise."""
        from weewx_clearskies_api.config.settings import ForecastSettings  # noqa: PLC0415
        settings = ForecastSettings({"provider": "openweathermap"})
        settings.validate()  # Should not raise

    def test_openweathermap_capability_wires_into_provider_registry(
        self, db_engine: Engine
    ) -> None:
        """wire_providers([owm.CAPABILITY]) → get_provider_registry() has owm entry."""
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            get_provider_registry,
            reset_provider_registry_for_tests,
            wire_providers,
        )
        from weewx_clearskies_api.providers.forecast.openweathermap import CAPABILITY  # noqa: PLC0415

        reset_provider_registry_for_tests()
        wire_providers([CAPABILITY])

        registry = get_provider_registry()
        owm_entries = [p for p in registry if p.provider_id == "openweathermap"]
        assert len(owm_entries) == 1
        assert owm_entries[0].domain == "forecast"

        reset_provider_registry_for_tests()

    def test_openweathermap_missing_appid_raises_key_invalid_at_fetch_time(
        self, db_engine: Engine
    ) -> None:
        """OWM configured but appid missing → KeyInvalid at fetch (not startup)."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            reset_cache_for_tests,
            wire_cache_from_env,
        )
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _reset_basic_tier_warned_for_tests,
            _reset_http_client_for_tests,
        )

        reset_cache_for_tests()
        _reset_http_client_for_tests()
        _reset_basic_tier_warned_for_tests()
        openweathermap._rate_limiter._calls.clear()
        wire_cache_from_env()

        # Missing appid — should raise KeyInvalid at call time
        with pytest.raises(KeyInvalid):
            openweathermap.fetch(
                lat=_LAT, lon=_LON, target_unit="US", appid=None
            )

        reset_cache_for_tests()
        _reset_http_client_for_tests()

    def test_wire_openweathermap_credentials_sets_module_appid(self) -> None:
        """wire_openweathermap_credentials() wires the appid into the endpoint module."""
        from weewx_clearskies_api.endpoints import forecast as forecast_endpoint  # noqa: PLC0415

        forecast_endpoint.wire_openweathermap_credentials("MYAPPID_TEST")
        assert forecast_endpoint._openweathermap_appid == "MYAPPID_TEST"
        # Cleanup
        forecast_endpoint.wire_openweathermap_credentials(None)
