"""Integration tests for the alerts provider domain (3b round 1).

All tests carry @pytest.mark.integration and run against the docker-compose
dev/test stack (MariaDB or SQLite backend per BACKEND env var).

Same integration suite runs twice per ADR-012 (once MariaDB, once SQLite)
to catch dialect drift. The alerts endpoint itself has no DB dependency, but
running in the real stack confirms the endpoint is DB-stack-agnostic and
wires correctly alongside the DB-backed endpoints.

Redis-backed integration tests carry both @pytest.mark.integration and
@pytest.mark.redis and are skipped unless `pytest -m "integration and redis"`
is used.

Endpoints covered:
  - GET /api/v1/alerts  no-provider → 200 source="none"
  - GET /api/v1/alerts  NWS configured + respx-mocked NWS → 200 source="nws"
  - GET /api/v1/capabilities  NWS configured → providers list non-empty
  - Startup failure: unknown_provider → KeyError (unit-style test)
  - Startup failure: unreachable Redis → exception on ping
  - Redis integration (optional, redis mark): real Redis from compose redis profile

ADR references: ADR-012, ADR-016, ADR-017, ADR-018, ADR-038.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Backend configuration (mirrors test_3a2_endpoints_integration.py pattern)
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

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "providers" / "nws"


def _load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture from tests/fixtures/providers/nws/."""
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
    """Skip if REDIS_URL is not reachable."""
    try:
        import redis as redis_lib  # noqa: PLC0415
        r = redis_lib.Redis.from_url(_REDIS_URL)
        r.ping()
    except Exception:  # noqa: BLE001
        pytest.skip(f"Redis not reachable at {_REDIS_URL}; start redis compose profile")


# ---------------------------------------------------------------------------
# Engine fixtures (module-scoped — same DB for all integration tests in module)
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
# App + client fixtures
# ---------------------------------------------------------------------------


def _make_integration_settings(provider: str | None = None) -> Any:
    """Build minimal Settings for integration test app creation."""
    from weewx_clearskies_api.config.settings import (  # noqa: PLC0415
        AlertsSettings,
        ApiSettings,
        DatabaseSettings,
        HealthSettings,
        LoggingSettings,
        RateLimitSettings,
        Settings,
    )

    return Settings(
        api=ApiSettings({}),
        health=HealthSettings({}),
        logging_settings=LoggingSettings({}),
        ratelimit=RateLimitSettings({}),
        database=DatabaseSettings({}),
        alerts=AlertsSettings({"provider": provider} if provider else {}),
    )


@pytest.fixture(scope="module")
def integration_app_no_provider(db_engine: Engine) -> Generator[FastAPI, None, None]:
    """Integration app with no provider configured."""
    from weewx_clearskies_api.app import create_app  # noqa: PLC0415
    from weewx_clearskies_api.db.reflection import STOCK_COLUMN_MAP  # noqa: PLC0415
    from weewx_clearskies_api.db.reflection import ColumnInfo, ColumnRegistry  # noqa: PLC0415
    from weewx_clearskies_api.db.registry import wire_registry  # noqa: PLC0415
    from weewx_clearskies_api.db.session import wire_engine  # noqa: PLC0415
    from weewx_clearskies_api.endpoints.alerts import wire_alerts_settings  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.cache import wire_cache_from_env  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.capability import wire_providers  # noqa: PLC0415
    from weewx_clearskies_api.services import station as station_mod  # noqa: PLC0415
    from weewx_clearskies_api.services.station import StationInfo, reset_cache  # noqa: PLC0415
    from weewx_clearskies_api.services import units as units_mod  # noqa: PLC0415
    from weewx_clearskies_api.services.units import (  # noqa: PLC0415
        _GROUP_MEMBERS,
        _SYSTEM_PRESETS,
        reset_cache as reset_units_cache,
    )

    # Wire DB
    wire_engine(db_engine)
    registry = ColumnRegistry()
    registry.stock = {
        col: ColumnInfo(db_name=col, canonical_name=canon, is_stock=True)
        for col, canon in STOCK_COLUMN_MAP.items()
    }
    wire_registry(registry)

    # Wire station
    reset_cache()
    station_mod._cached_station = StationInfo(
        station_id="integration-test-station",
        name="Integration Test Station",
        latitude=42.375,
        longitude=-72.519,
        altitude=100.0,
        timezone="America/New_York",
        timezone_offset_minutes=-240,
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

    # Wire cache (no-provider path; memory default)
    wire_cache_from_env()
    wire_providers([])

    settings = _make_integration_settings(provider=None)
    wire_alerts_settings(settings)
    app = create_app(settings)
    yield app


@pytest.fixture(scope="module")
def integration_client_no_provider(
    integration_app_no_provider: FastAPI,
) -> TestClient:
    return TestClient(integration_app_no_provider, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def integration_app_nws(db_engine: Engine) -> Generator[FastAPI, None, None]:
    """Integration app with NWS provider configured."""
    from weewx_clearskies_api.app import create_app  # noqa: PLC0415
    from weewx_clearskies_api.db.reflection import STOCK_COLUMN_MAP  # noqa: PLC0415
    from weewx_clearskies_api.db.reflection import ColumnInfo, ColumnRegistry  # noqa: PLC0415
    from weewx_clearskies_api.db.registry import wire_registry  # noqa: PLC0415
    from weewx_clearskies_api.db.session import wire_engine  # noqa: PLC0415
    from weewx_clearskies_api.endpoints.alerts import wire_alerts_settings  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.cache import wire_cache_from_env  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.capability import wire_providers  # noqa: PLC0415
    from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
    from weewx_clearskies_api.services import station as station_mod  # noqa: PLC0415
    from weewx_clearskies_api.services.station import StationInfo, reset_cache  # noqa: PLC0415
    from weewx_clearskies_api.services import units as units_mod  # noqa: PLC0415
    from weewx_clearskies_api.services.units import (  # noqa: PLC0415
        _GROUP_MEMBERS,
        _SYSTEM_PRESETS,
        reset_cache as reset_units_cache,
    )
    from weewx_clearskies_api.providers._common.cache import reset_cache_for_tests  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
        reset_provider_registry_for_tests,
    )

    # Reset state
    reset_cache_for_tests()
    reset_provider_registry_for_tests()

    # Wire DB
    wire_engine(db_engine)
    registry = ColumnRegistry()
    registry.stock = {
        col: ColumnInfo(db_name=col, canonical_name=canon, is_stock=True)
        for col, canon in STOCK_COLUMN_MAP.items()
    }
    wire_registry(registry)

    # Wire station (42.375, -72.519 — must match _build_cache_key calls)
    reset_cache()
    station_mod._cached_station = StationInfo(
        station_id="integration-test-station",
        name="Integration Test Station",
        latitude=42.375,
        longitude=-72.519,
        altitude=100.0,
        timezone="America/New_York",
        timezone_offset_minutes=-240,
        unit_system="US",
        hardware=None,
    )

    # Wire units
    reset_units_cache()
    system_map = _SYSTEM_PRESETS["US"]
    block_: dict[str, str] = {}
    for group, unit in system_map.items():
        for field in _GROUP_MEMBERS.get(group, []):
            block_[field] = unit
    units_mod._cached_units_block = block_
    units_mod._cached_target_unit = "US"

    # Wire cache and providers
    wire_cache_from_env()
    wire_providers([nws.CAPABILITY])

    settings = _make_integration_settings(provider="nws")
    wire_alerts_settings(settings)
    app = create_app(settings)
    yield app

    # Teardown: reset provider state
    reset_cache_for_tests()
    reset_provider_registry_for_tests()


@pytest.fixture(scope="module")
def integration_client_nws(integration_app_nws: FastAPI) -> TestClient:
    return TestClient(integration_app_nws, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Per-test cache reset (avoid TTL bleed between tests)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cache_between_tests() -> Generator[None, None, None]:
    """Reset just the cache between tests to avoid inter-test bleed.

    Does NOT reset the provider registry or settings — those are module-scoped
    per the integration_app_* fixtures.
    """
    yield
    from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
        reset_cache_for_tests,
        wire_cache_from_env,
    )
    reset_cache_for_tests()
    wire_cache_from_env()


# ===========================================================================
# 1. /alerts no-provider → 200 source="none"
# ===========================================================================


class TestAlertsNoProviderIntegration:
    """/alerts returns 200 source='none' when no provider is configured."""

    def test_no_provider_returns_200_with_source_none(
        self, integration_client_no_provider: TestClient
    ) -> None:
        """GET /api/v1/alerts with no provider → 200 AlertListResponse source='none'."""
        response = integration_client_no_provider.get("/api/v1/alerts")
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data["data"]["source"] == "none", (
            f"Expected source='none', got {data['data']['source']!r}"
        )
        assert data["data"]["alerts"] == [], (
            "alerts must be [] when no provider configured"
        )
        assert "retrievedAt" in data["data"], "retrievedAt must be present"
        assert data["source"] == "none", "envelope source must be 'none'"
        assert "generatedAt" in data

    def test_no_provider_returns_valid_alert_list_response_shape(
        self, integration_client_no_provider: TestClient
    ) -> None:
        """Response shape matches AlertListResponse OpenAPI contract."""
        response = integration_client_no_provider.get("/api/v1/alerts")
        assert response.status_code == 200
        data = response.json()
        # AlertListResponse: data, source, generatedAt
        assert "data" in data
        assert "source" in data
        assert "generatedAt" in data
        # AlertList inner: alerts, retrievedAt, source
        inner = data["data"]
        assert "alerts" in inner
        assert "retrievedAt" in inner
        assert "source" in inner

    def test_no_provider_runs_against_active_db_backend(
        self, integration_client_no_provider: TestClient
    ) -> None:
        """Confirms endpoint is DB-stack-agnostic; no DB hit for alerts."""
        # This test passes on both MariaDB and SQLite backends because
        # the alerts endpoint does not touch the DB.
        response = integration_client_no_provider.get("/api/v1/alerts")
        assert response.status_code == 200, (
            f"DB backend {_BACKEND!r}: expected 200, got {response.status_code}"
        )


# ===========================================================================
# 2. /alerts NWS configured + respx-mocked NWS → happy path
# ===========================================================================


class TestAlertsNWSIntegration:
    """/alerts with NWS configured and mocked NWS → 200 source='nws'."""

    def test_nws_configured_happy_path_returns_alerts(
        self, integration_client_nws: TestClient
    ) -> None:
        """NWS configured + respx mock → 200 with alerts from fixture."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415

        fixture = _load_fixture("alerts_active.json")
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(200, json=fixture)
            )
            response = integration_client_nws.get("/api/v1/alerts")

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data["data"]["source"] == "nws", (
            f"Expected source='nws', got {data['data']['source']!r}"
        )
        assert len(data["data"]["alerts"]) == 2, (
            f"Expected 2 alerts from fixture, got {len(data['data']['alerts'])}"
        )

    def test_nws_alerts_response_shape_matches_openapi_contract(
        self, integration_client_nws: TestClient
    ) -> None:
        """Each AlertRecord in the response has all required fields per OpenAPI schema."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415

        fixture = _load_fixture("alerts_active.json")
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(200, json=fixture)
            )
            response = integration_client_nws.get("/api/v1/alerts")

        assert response.status_code == 200
        data = response.json()
        # AlertRecord required fields per OpenAPI: id, headline, severity, event, effective, source
        for alert in data["data"]["alerts"]:
            assert "id" in alert, f"AlertRecord missing 'id': {alert}"
            assert "headline" in alert, f"AlertRecord missing 'headline': {alert}"
            assert "severity" in alert, f"AlertRecord missing 'severity': {alert}"
            assert alert["severity"] in ("advisory", "watch", "warning"), (
                f"severity must be canonical value, got {alert['severity']!r}"
            )
            assert "event" in alert, f"AlertRecord missing 'event': {alert}"
            assert "effective" in alert, f"AlertRecord missing 'effective': {alert}"
            assert "source" in alert, f"AlertRecord missing 'source': {alert}"
            assert alert["source"] == "nws", f"source must be 'nws', got {alert['source']!r}"

    def test_nws_effective_field_has_utc_z_suffix(
        self, integration_client_nws: TestClient
    ) -> None:
        """effective field in AlertRecord response is UTC ISO-8601 with Z suffix (ADR-020)."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415

        fixture = _load_fixture("alerts_active.json")
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(200, json=fixture)
            )
            response = integration_client_nws.get("/api/v1/alerts")

        assert response.status_code == 200
        data = response.json()
        for alert in data["data"]["alerts"]:
            effective = alert["effective"]
            assert effective.endswith("Z"), (
                f"effective must end with 'Z' suffix (ADR-020), got {effective!r}"
            )

    def test_nws_integration_with_empty_fixture_returns_empty_alerts(
        self, integration_client_nws: TestClient
    ) -> None:
        """NWS returning empty features → /alerts returns 200 alerts=[] source='nws'."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415

        fixture = _load_fixture("alerts_active_empty.json")
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(200, json=fixture)
            )
            response = integration_client_nws.get("/api/v1/alerts")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["alerts"] == []
        assert data["data"]["source"] == "nws", (
            "source must still be 'nws' even when no alerts active"
        )


# ===========================================================================
# 3. /capabilities NWS configured → providers list populated
# ===========================================================================


class TestCapabilitiesWithNWSIntegration:
    """/capabilities response includes NWS provider entry when wired."""

    def test_capabilities_with_nws_has_one_provider_entry(
        self, integration_client_nws: TestClient
    ) -> None:
        """GET /capabilities with NWS wired → providers has one nws entry."""
        response = integration_client_nws.get("/api/v1/capabilities")
        assert response.status_code == 200
        data = response.json()
        providers = data["data"]["providers"]
        assert len(providers) == 1, f"Expected 1 provider, got {len(providers)}"
        assert providers[0]["providerId"] == "nws"
        assert providers[0]["domain"] == "alerts"

    def test_capabilities_canonical_fields_available_union_includes_nws_fields(
        self, integration_client_nws: TestClient
    ) -> None:
        """canonicalFieldsAvailable is union of weewx stock + NWS supplied fields."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415

        response = integration_client_nws.get("/api/v1/capabilities")
        assert response.status_code == 200
        data = response.json()
        available = set(data["data"]["canonicalFieldsAvailable"])
        nws_fields = set(nws.CAPABILITY.supplied_canonical_fields)
        missing = nws_fields - available
        assert not missing, (
            f"canonicalFieldsAvailable must contain NWS-supplied fields; "
            f"missing: {missing}"
        )

    def test_capabilities_without_provider_returns_empty_providers(
        self, integration_client_no_provider: TestClient
    ) -> None:
        """With no provider wired, providers is []."""
        response = integration_client_no_provider.get("/api/v1/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["providers"] == []


# ===========================================================================
# 4. Startup failure paths (integration-context, unit-style calls)
# ===========================================================================


class TestStartupFailurePathsIntegration:
    """Startup failure tests confirming the service refuses to start in bad states."""

    def test_unknown_provider_id_raises_key_error_from_dispatch(self) -> None:
        """get_provider_module with unknown_provider raises KeyError.

        Tests the dispatch.get_provider_module() code path that __main__.py
        calls at startup — an unknown provider id must fail closed.
        """
        from weewx_clearskies_api.providers._common.dispatch import get_provider_module  # noqa: PLC0415
        with pytest.raises(KeyError):
            get_provider_module(domain="alerts", provider_id="unknown_provider")

    def test_redis_unreachable_raises_on_cache_construction(self) -> None:
        """RedisCache construction fails when Redis is not reachable."""
        from weewx_clearskies_api.providers._common.cache import RedisCache  # noqa: PLC0415
        with pytest.raises((RuntimeError, Exception)):
            RedisCache(url="redis://127.0.0.1:16379/0")

    def test_bogus_cache_scheme_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """wire_cache_from_env raises ConfigError for unrecognised URL scheme."""
        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            ConfigError,
            reset_cache_for_tests,
            wire_cache_from_env,
        )
        monkeypatch.setenv("CLEARSKIES_CACHE_URL", "memcached://localhost:11211")
        reset_cache_for_tests()
        with pytest.raises(ConfigError):
            wire_cache_from_env()
        monkeypatch.delenv("CLEARSKIES_CACHE_URL")
        reset_cache_for_tests()
        wire_cache_from_env()  # re-wire with memory backend


# ===========================================================================
# 5. Redis-backed integration (optional — requires redis compose profile)
# ===========================================================================


@pytest.mark.redis
class TestAlertsRedisBackendIntegration:
    """/alerts end-to-end against a real Redis from the compose redis profile.

    Run with: pytest -m "integration and redis"
    Skipped automatically when Redis is not reachable.
    """

    @pytest.fixture(autouse=True)
    def _require_redis_fixture(self) -> None:
        """Skip this test class when Redis is not reachable."""
        _require_redis()

    @pytest.fixture(autouse=True)
    def _wire_redis_cache(self, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
        """Wire the Redis cache backend for this test class."""
        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            reset_cache_for_tests,
            wire_cache_from_env,
        )
        monkeypatch.setenv("CLEARSKIES_CACHE_URL", _REDIS_URL)
        reset_cache_for_tests()
        wire_cache_from_env()
        yield
        # Cleanup: flush the Redis test keys
        import redis as redis_lib  # noqa: PLC0415
        try:
            r = redis_lib.Redis.from_url(_REDIS_URL)
            r.flushdb()
        except Exception:  # noqa: BLE001
            pass
        reset_cache_for_tests()
        monkeypatch.delenv("CLEARSKIES_CACHE_URL", raising=False)
        wire_cache_from_env()

    def test_alerts_no_provider_returns_200_with_redis_cache(
        self, integration_client_no_provider: TestClient
    ) -> None:
        """With Redis cache active, no-provider path still returns 200 source='none'."""
        response = integration_client_no_provider.get("/api/v1/alerts")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["source"] == "none"

    def test_nws_alerts_cache_hit_via_real_redis(
        self, integration_client_nws: TestClient
    ) -> None:
        """With real Redis, second /alerts call returns cached data (one NWS call total)."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415

        fixture = _load_fixture("alerts_active.json")
        nws_call_count = 0

        def _mock_nws(request: httpx.Request) -> httpx.Response:
            nonlocal nws_call_count
            nws_call_count += 1
            return httpx.Response(200, json=fixture)

        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(side_effect=_mock_nws)
            # First call: cache miss → NWS called
            response1 = integration_client_nws.get("/api/v1/alerts")
            assert response1.status_code == 200
            # Second call: cache hit → NWS NOT called
            response2 = integration_client_nws.get("/api/v1/alerts")
            assert response2.status_code == 200

        assert nws_call_count == 1, (
            f"Expected 1 NWS call (second should be Redis cache hit), got {nws_call_count}"
        )
