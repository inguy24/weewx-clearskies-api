"""Integration tests for the Aeris AQI provider (3b-10).

All tests carry @pytest.mark.integration and run against the docker-compose
dev/test stack (MariaDB or SQLite backend per BACKEND env var).

Same integration suite runs twice per ADR-012 (once MariaDB, once SQLite)
to catch dialect drift. The AQI endpoint itself has no DB dependency, but
running in the real stack confirms the endpoint is DB-stack-agnostic and
wires correctly alongside the DB-backed endpoints.

Redis-backed integration tests carry both @pytest.mark.integration and
@pytest.mark.redis and are skipped unless `pytest -m "integration and redis"`.
Per brief §Process gates: the Redis tier MUST PASS, not skip. If Redis is
not reachable on weather-dev, this is a brief-gate failure that must be
surfaced to the lead before closeout.

End-to-end paths covered:
  - Full startup with [aqi] provider = aeris + [aeris] credentials in config.
  - GET /api/v1/aqi/current aeris configured + respx-mocked → 200 source="aeris".
  - Canonical AQIReading: aqi, aqiCategory, aqiMainPollutant, aqiLocation, observedAt, source.
  - observedAt is UTC ISO-8601 Z format (LC4 + ADR-020).
  - aqiLocation = "seattle" (Aeris supplies place.name — NOT PARTIAL-DOMAIN).
  - GET /api/v1/aqi/current credentials missing → 502 error.
  - Provider 5xx → 502 RFC 9457 problem+json.
  - Provider 429 → 503 RFC 9457 + Retry-After.
  - AQIResponse envelope validates against OpenAPI AQIResponse schema shape
    (source="aeris" present; data shape matches AQIReading).
  - Memory cache: miss → fetch → hit (both DB backends).
  - Redis cache: miss → fetch → hit (redis mark; must pass on weather-dev).
  - wire_providers([aeris.CAPABILITY]) registers in capability registry.

ADR references: ADR-012, ADR-013, ADR-017, ADR-018, ADR-020, ADR-038.
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

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "providers" / "aqi"

# Station coordinates matching fixture
_LAT = 47.6062
_LON = -122.3321
_LAT4 = round(_LAT, 4)
_LON4 = round(_LON, 4)
_AERIS_AQ_BASE_URL = "https://data.api.xweather.com"
# Full URL as constructed by fetch(): lat/lon rounded to 6dp per aeris.py fetch()
_AERIS_AQ_URL = f"{_AERIS_AQ_BASE_URL}/airquality/{round(_LAT, 6)},{round(_LON, 6)}"

_TEST_CLIENT_ID = "INTEGRATION_TEST_CLIENT_ID"
_TEST_CLIENT_SECRET = "INTEGRATION_TEST_CLIENT_SECRET"


def _load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture from tests/fixtures/providers/aqi/."""
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
        pytest.skip(f"SQLite file {_SQLITE_SDB_PATH!r} not found; seed the dev stack")


def _require_redis() -> None:
    """Skip if Redis is not reachable (brief gate: must pass, not skip on weather-dev)."""
    import redis as redis_lib  # noqa: PLC0415
    try:
        r = redis_lib.from_url(_REDIS_URL)
        r.ping()
    except Exception:
        pytest.skip(f"Redis not reachable at {_REDIS_URL}; start compose redis profile")


# ---------------------------------------------------------------------------
# Engine fixture
# ---------------------------------------------------------------------------


def _make_engine() -> Engine:
    """Build the DB engine for the current backend (MariaDB or SQLite)."""
    if _BACKEND == "mariadb":
        _require_mariadb_password()
        url = (
            f"mysql+pymysql://clearskies_ro:{_MARIADB_RO_PASSWORD}"
            f"@127.0.0.1:{_MARIADB_HOST_PORT}/{_MARIADB_DB}"
        )
        return create_engine(url, pool_pre_ping=True)
    else:
        _require_sqlite_file()
        return create_engine(
            f"sqlite+pysqlite:///file:///{_SQLITE_SDB_PATH}?mode=ro&uri=true",
            connect_args={"check_same_thread": False},
        )


@pytest.fixture(scope="class")
def db_engine() -> Generator[Engine, None, None]:
    """Class-scoped engine fixture. Disposes after each test class."""
    engine = _make_engine()
    yield engine
    engine.dispose()


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------


def _wire_db(engine: Engine) -> None:
    """Wire the real DB engine into weewx_clearskies_api's session layer."""
    from weewx_clearskies_api.db.reflection import reflect_schema  # noqa: PLC0415
    from weewx_clearskies_api.db.registry import wire_registry  # noqa: PLC0415
    from weewx_clearskies_api.db.session import wire_engine  # noqa: PLC0415

    wire_engine(engine)
    registry = reflect_schema(engine)
    wire_registry(registry)


def _wire_test_station() -> None:
    """Wire station at Seattle coordinates matching the AQI fixture."""
    from weewx_clearskies_api.services import station as station_mod  # noqa: PLC0415
    from weewx_clearskies_api.services.station import StationInfo, reset_cache  # noqa: PLC0415

    reset_cache()
    station_mod._cached_station = StationInfo(
        station_id="integration-test-aeris-aqi",
        name="Integration Test Station (Aeris AQI)",
        latitude=_LAT,
        longitude=_LON,
        altitude=59.0,
        timezone="America/Los_Angeles",
        timezone_offset_minutes=-420,
        unit_system="US",
        hardware=None,
    )


def _wire_test_units() -> None:
    """Wire US unit block for integration tests."""
    from weewx_clearskies_api.services import units as _units_mod  # noqa: PLC0415
    from weewx_clearskies_api.services.units import (  # noqa: PLC0415
        _GROUP_MEMBERS,
        _SYSTEM_PRESETS,
        reset_cache,
    )

    reset_cache()
    system_map = _SYSTEM_PRESETS["US"]
    block: dict[str, str] = {}
    for group, unit in system_map.items():
        for field in _GROUP_MEMBERS.get(group, []):
            block[field] = unit
    _units_mod._cached_units_block = block
    _units_mod._cached_target_unit = "US"


def _reset_aeris_provider_state() -> None:
    """Reset provider registry, cache, aeris http client + rate limiter."""
    from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
        reset_cache_for_tests,
        wire_cache_from_env,
    )
    from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
        reset_provider_registry_for_tests,
    )
    from weewx_clearskies_api.providers.aqi.aeris import _reset_http_client_for_tests  # noqa: PLC0415
    import weewx_clearskies_api.providers.aqi.aeris as _aeris  # noqa: PLC0415

    reset_cache_for_tests()
    reset_provider_registry_for_tests()
    _reset_http_client_for_tests()
    _aeris._rate_limiter._calls.clear()
    wire_cache_from_env()


def _make_integration_app(
    engine: Engine,
    wire_credentials: bool = True,
) -> FastAPI:
    """Build a full integration FastAPI app with aeris AQI registered.

    wire_credentials: if True, sets _AERIS_CLIENT_ID + _AERIS_CLIENT_SECRET.
    """
    from weewx_clearskies_api.app import create_app  # noqa: PLC0415
    from weewx_clearskies_api.config.settings import (  # noqa: PLC0415
        ApiSettings,
        DatabaseSettings,
        HealthSettings,
        LoggingSettings,
        RateLimitSettings,
        Settings,
    )
    from weewx_clearskies_api.providers._common.capability import wire_providers  # noqa: PLC0415
    import weewx_clearskies_api.endpoints.aqi as _aqi_endpoint  # noqa: PLC0415

    _reset_aeris_provider_state()
    _wire_db(engine)
    _wire_test_station()
    _wire_test_units()

    from weewx_clearskies_api.providers.aqi.aeris import CAPABILITY  # noqa: PLC0415
    wire_providers([CAPABILITY])

    if wire_credentials:
        _aqi_endpoint._AERIS_CLIENT_ID = _TEST_CLIENT_ID
        _aqi_endpoint._AERIS_CLIENT_SECRET = _TEST_CLIENT_SECRET
    else:
        _aqi_endpoint._AERIS_CLIENT_ID = None
        _aqi_endpoint._AERIS_CLIENT_SECRET = None

    settings = Settings(
        api=ApiSettings({}),
        health=HealthSettings({}),
        logging_settings=LoggingSettings({}),
        ratelimit=RateLimitSettings({}),
        database=DatabaseSettings({}),
    )
    return create_app(settings)


# ---------------------------------------------------------------------------
# Client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def integration_client(db_engine: Engine) -> TestClient:
    """TestClient with aeris registered + credentials wired."""
    app = _make_integration_app(db_engine, wire_credentials=True)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def integration_client_no_credentials(db_engine: Engine) -> TestClient:
    """TestClient with aeris registered but credentials NOT wired."""
    app = _make_integration_app(db_engine, wire_credentials=False)
    return TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# 1. End-to-end happy path — aeris registered + credentials wired
# ===========================================================================


class TestIntegrationAerisAqiHappyPath:
    """Full stack GET /aqi/current with aeris configured → 200 AQIReading."""

    def test_aeris_aqi_returns_200(self, integration_client: TestClient) -> None:
        """aeris registered + credentials wired + respx-mocked → 200."""
        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            response = integration_client.get("/api/v1/aqi/current")

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text[:300]}"
        )

    def test_aeris_aqi_response_source_is_aeris(self, integration_client: TestClient) -> None:
        """source = 'aeris' in AQIResponse envelope."""
        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            response = integration_client.get("/api/v1/aqi/current")

        body = response.json()
        assert body["source"] == "aeris", (
            f"Expected source='aeris', got {body.get('source')!r}"
        )

    def test_aeris_aqi_data_is_aqi_reading_with_aqi_33(self, integration_client: TestClient) -> None:
        """data.aqi = 33 (real fixture value, Good band)."""
        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            response = integration_client.get("/api/v1/aqi/current")

        body = response.json()
        assert body["data"] is not None
        assert body["data"]["aqi"] == 33, (
            f"Expected aqi=33, got {body['data'].get('aqi')!r}"
        )

    def test_aeris_aqi_data_aqi_category_is_good(self, integration_client: TestClient) -> None:
        """data.aqiCategory = 'Good' (AQI 33 → EPA 0–50 band)."""
        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            response = integration_client.get("/api/v1/aqi/current")

        body = response.json()
        assert body["data"]["aqiCategory"] == "Good", (
            f"Expected aqiCategory='Good', got {body['data'].get('aqiCategory')!r}"
        )

    def test_aeris_aqi_data_aqi_location_is_seattle(self, integration_client: TestClient) -> None:
        """data.aqiLocation = 'seattle' (Aeris supplies place.name — NOT PARTIAL-DOMAIN)."""
        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            response = integration_client.get("/api/v1/aqi/current")

        body = response.json()
        assert body["data"]["aqiLocation"] == "seattle", (
            f"Expected aqiLocation='seattle', got {body['data'].get('aqiLocation')!r}"
        )

    def test_aeris_aqi_data_observed_at_is_utc_z(self, integration_client: TestClient) -> None:
        """data.observedAt ends with Z (UTC ISO-8601, ADR-020)."""
        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            response = integration_client.get("/api/v1/aqi/current")

        body = response.json()
        observed_at = body["data"]["observedAt"]
        assert observed_at.endswith("Z"), (
            f"observedAt must end with Z, got {observed_at!r}"
        )

    def test_aeris_aqi_data_source_is_aeris(self, integration_client: TestClient) -> None:
        """data.source = 'aeris' (AQIReading.source = 'aeris')."""
        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            response = integration_client.get("/api/v1/aqi/current")

        body = response.json()
        assert body["data"]["source"] == "aeris"

    def test_aeris_aqi_envelope_has_all_required_fields(self, integration_client: TestClient) -> None:
        """AQIResponse envelope has all OpenAPI-mandated fields: data, units, source, generatedAt."""
        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            response = integration_client.get("/api/v1/aqi/current")

        body = response.json()
        for field in ("data", "units", "source", "generatedAt"):
            assert field in body, f"AQIResponse envelope missing required field {field!r}"

    def test_aeris_aqi_generated_at_is_utc_z(self, integration_client: TestClient) -> None:
        """generatedAt ends with Z (ADR-020 UTC at API boundary)."""
        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            response = integration_client.get("/api/v1/aqi/current")

        body = response.json()
        assert body["generatedAt"].endswith("Z"), (
            f"generatedAt must end with Z, got {body['generatedAt']!r}"
        )


# ===========================================================================
# 2. Credentials missing path
# ===========================================================================


class TestIntegrationAerisAqiCredentialsMissing:
    """aeris registered but credentials NOT wired → 502 error."""

    def test_credentials_missing_returns_502(
        self, integration_client_no_credentials: TestClient
    ) -> None:
        """aeris registered, credentials=None → 502 (checked before provider call)."""
        with respx.mock(assert_all_called=False):
            response = integration_client_no_credentials.get("/api/v1/aqi/current")

        assert response.status_code == 502, (
            f"Expected 502 for missing credentials, got {response.status_code}: {response.text[:300]}"
        )


# ===========================================================================
# 3. Error paths
# ===========================================================================


class TestIntegrationAerisAqiErrorPaths:
    """Provider error handling: 5xx → 502, 429 → 503 + Retry-After."""

    def test_provider_5xx_returns_502_rfc9457(self, integration_client: TestClient) -> None:
        """Provider 5xx → 502 application/problem+json."""
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(500, json={"reason": "server error"})
            )
            response = integration_client.get("/api/v1/aqi/current")

        assert response.status_code == 502, (
            f"Provider 5xx must map to 502, got {response.status_code}: {response.text[:300]}"
        )
        assert "application/problem+json" in response.headers.get("content-type", ""), (
            "502 must return application/problem+json (RFC 9457)"
        )

    def test_provider_429_returns_503_rfc9457(self, integration_client: TestClient) -> None:
        """Provider 429 → 503 application/problem+json."""
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={"reason": "too many requests"},
                    headers={"Retry-After": "60"},
                )
            )
            response = integration_client.get("/api/v1/aqi/current")

        assert response.status_code == 503, (
            f"Provider 429 must map to 503, got {response.status_code}: {response.text[:300]}"
        )
        assert "application/problem+json" in response.headers.get("content-type", ""), (
            "503 must return application/problem+json (RFC 9457)"
        )

    def test_provider_429_includes_retry_after_header(self, integration_client: TestClient) -> None:
        """Provider 429 → 503 response includes Retry-After header (ADR-018)."""
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={"reason": "rate limit"},
                    headers={"Retry-After": "120"},
                )
            )
            response = integration_client.get("/api/v1/aqi/current")

        assert "Retry-After" in response.headers, (
            "503 from QuotaExhausted must include Retry-After header"
        )


# ===========================================================================
# 4. OpenAPI schema validation
# ===========================================================================


class TestIntegrationAerisAqiOpenApiSchema:
    """AQIResponse shape matches OpenAPI AQIResponse contract."""

    def test_aqi_reading_fields_match_openapi_aqi_reading_schema(
        self, integration_client: TestClient
    ) -> None:
        """data fields match OpenAPI AQIReading schema (aqi, aqiCategory, etc.)."""
        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            response = integration_client.get("/api/v1/aqi/current")

        body = response.json()
        reading = body["data"]
        assert reading is not None

        # These fields must be present in every AQIReading per OpenAPI contract
        expected_keys = {
            "aqi", "aqiCategory", "aqiMainPollutant", "aqiLocation",
            "pollutantPM25", "pollutantPM10",
            "pollutantO3", "pollutantNO2", "pollutantSO2", "pollutantCO",
            "observedAt", "source",
        }
        for key in expected_keys:
            assert key in reading, (
                f"AQIReading missing OpenAPI-required field {key!r}"
            )

    def test_aqi_response_source_is_aeris_not_none(
        self, integration_client: TestClient
    ) -> None:
        """AQIResponse.source = 'aeris' (not 'none', not null) when aeris is configured."""
        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            response = integration_client.get("/api/v1/aqi/current")

        body = response.json()
        assert body["source"] not in (None, "none"), (
            f"Expected source='aeris', got {body.get('source')!r}"
        )
        assert body["source"] == "aeris"


# ===========================================================================
# 5. Capability registry wiring
# ===========================================================================


class TestIntegrationAerisCapabilityRegistry:
    """wire_providers([aeris.CAPABILITY]) registers correctly in capability registry."""

    def test_aeris_capability_registered_in_registry(self, db_engine: Engine) -> None:
        """wire_providers([aeris.CAPABILITY]) → ('aqi', 'aeris') in registry."""
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            get_provider_registry,
            reset_provider_registry_for_tests,
            wire_providers,
        )
        from weewx_clearskies_api.providers.aqi.aeris import CAPABILITY  # noqa: PLC0415

        reset_provider_registry_for_tests()
        wire_providers([CAPABILITY])
        registry = get_provider_registry()
        assert any(p.provider_id == "aeris" and p.domain == "aqi" for p in registry), (
            "wire_providers must register aeris aqi in registry"
        )
        reset_provider_registry_for_tests()


# ===========================================================================
# 6. Memory cache: miss → fetch → hit
# ===========================================================================


class TestIntegrationAerisAqiMemoryCache:
    """Aeris AQI provider: memory cache miss → fetch → hit (both DB backends)."""

    def test_cache_miss_fetches_from_provider_and_caches_result(
        self, db_engine: Engine
    ) -> None:
        """Memory cache miss → one HTTP call; result cached for next poll."""
        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            get_cache,
            reset_cache_for_tests,
            wire_cache_from_env,
        )
        from weewx_clearskies_api.providers.aqi import aeris  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import (  # noqa: PLC0415
            _build_cache_key,
            _reset_http_client_for_tests,
        )

        reset_cache_for_tests()
        _reset_http_client_for_tests()
        aeris._rate_limiter._calls.clear()
        wire_cache_from_env()

        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(return_value=httpx.Response(200, json=data))
            reading = aeris.fetch(
                lat=_LAT,
                lon=_LON,
                client_id=_TEST_CLIENT_ID,
                client_secret=_TEST_CLIENT_SECRET,
            )
            call_count = mock.calls.call_count

        assert call_count == 1, f"Expected 1 HTTP call on cache miss, got {call_count}"
        assert reading is not None
        assert reading.source == "aeris"
        assert reading.aqi == 33

        # Cache was populated
        cached = get_cache().get(_build_cache_key(_LAT, _LON))
        assert cached is not None, "Reading must be cached after cache miss"

        reset_cache_for_tests()
        _reset_http_client_for_tests()

    def test_cache_hit_skips_provider_call(self, db_engine: Engine) -> None:
        """Memory cache hit → zero HTTP calls; cached reading returned."""
        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            reset_cache_for_tests,
            wire_cache_from_env,
        )
        from weewx_clearskies_api.providers.aqi import aeris  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import _reset_http_client_for_tests  # noqa: PLC0415

        reset_cache_for_tests()
        _reset_http_client_for_tests()
        aeris._rate_limiter._calls.clear()
        wire_cache_from_env()

        data = _load_fixture("aeris_current.json")

        # First fetch — fills memory cache
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(return_value=httpx.Response(200, json=data))
            reading1 = aeris.fetch(
                lat=_LAT,
                lon=_LON,
                client_id=_TEST_CLIENT_ID,
                client_secret=_TEST_CLIENT_SECRET,
            )

        # Second fetch — should come from cache (zero calls)
        with respx.mock(assert_all_called=False) as mock2:
            reading2 = aeris.fetch(
                lat=_LAT,
                lon=_LON,
                client_id=_TEST_CLIENT_ID,
                client_secret=_TEST_CLIENT_SECRET,
            )
            cache_hit_calls = mock2.calls.call_count

        assert cache_hit_calls == 0, (
            f"Expected 0 HTTP calls on cache hit, got {cache_hit_calls}"
        )
        assert reading1 is not None and reading2 is not None
        assert reading1.aqi == reading2.aqi
        assert reading1.source == reading2.source

        reset_cache_for_tests()
        _reset_http_client_for_tests()


# ===========================================================================
# 7. Redis cache (redis mark — MUST PASS per brief)
# ===========================================================================


@pytest.mark.redis
class TestIntegrationAerisAqiRedisCache:
    """Real Redis from the docker-compose redis profile.

    Per brief §Process gates: Redis tier MUST PASS, not skip.
    If Redis is not reachable on weather-dev, this is a brief-gate failure
    that must be surfaced to the lead via SendMessage BEFORE closeout.
    """

    def test_aeris_aqi_redis_cache_miss_stores_reading(self, db_engine: Engine) -> None:
        """Redis cache miss → one HTTP call → reading stored in Redis."""
        _require_redis()

        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            RedisCache,
            reset_cache_for_tests,
        )
        from weewx_clearskies_api.providers.aqi import aeris  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import (  # noqa: PLC0415
            _build_cache_key,
            _reset_http_client_for_tests,
        )

        reset_cache_for_tests()
        _reset_http_client_for_tests()
        aeris._rate_limiter._calls.clear()

        redis_cache = RedisCache(url=_REDIS_URL)
        redis_cache._client.flushdb()  # type: ignore[attr-defined]
        import weewx_clearskies_api.providers._common.cache as cache_mod  # noqa: PLC0415
        cache_mod._cache = redis_cache

        data = _load_fixture("aeris_current.json")

        try:
            with respx.mock(assert_all_called=False) as mock:
                mock.get(_AERIS_AQ_URL).mock(
                    return_value=httpx.Response(200, json=data)
                )
                reading = aeris.fetch(
                    lat=_LAT,
                    lon=_LON,
                    client_id=_TEST_CLIENT_ID,
                    client_secret=_TEST_CLIENT_SECRET,
                )
                call_count = mock.calls.call_count

            assert call_count == 1, (
                f"Expected 1 HTTP call on Redis cache miss, got {call_count}"
            )
            assert reading is not None
            assert reading.source == "aeris"

            # Verify reading is in Redis
            cache_key = _build_cache_key(_LAT, _LON)
            cached = cache_mod._cache.get(cache_key)
            assert cached is not None, "Reading must be stored in Redis after cache miss"

        finally:
            redis_cache._client.flushdb()  # type: ignore[attr-defined]
            reset_cache_for_tests()
            _reset_http_client_for_tests()

    def test_aeris_aqi_redis_cache_hit_skips_provider_call(
        self, db_engine: Engine
    ) -> None:
        """Redis cache hit → zero HTTP calls; reading returned from Redis."""
        _require_redis()

        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            RedisCache,
            reset_cache_for_tests,
        )
        from weewx_clearskies_api.providers.aqi import aeris  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import _reset_http_client_for_tests  # noqa: PLC0415

        reset_cache_for_tests()
        _reset_http_client_for_tests()
        aeris._rate_limiter._calls.clear()

        redis_cache = RedisCache(url=_REDIS_URL)
        redis_cache._client.flushdb()  # type: ignore[attr-defined]
        import weewx_clearskies_api.providers._common.cache as cache_mod  # noqa: PLC0415
        cache_mod._cache = redis_cache

        data = _load_fixture("aeris_current.json")

        try:
            # First fetch — fills Redis cache
            with respx.mock(assert_all_called=False) as mock:
                mock.get(_AERIS_AQ_URL).mock(
                    return_value=httpx.Response(200, json=data)
                )
                reading1 = aeris.fetch(
                    lat=_LAT,
                    lon=_LON,
                    client_id=_TEST_CLIENT_ID,
                    client_secret=_TEST_CLIENT_SECRET,
                )
                assert mock.calls.call_count > 0

            # Second fetch — should hit Redis; zero calls
            with respx.mock(assert_all_called=False) as mock2:
                reading2 = aeris.fetch(
                    lat=_LAT,
                    lon=_LON,
                    client_id=_TEST_CLIENT_ID,
                    client_secret=_TEST_CLIENT_SECRET,
                )
                second_call_count = mock2.calls.call_count

            assert second_call_count == 0, (
                f"Expected 0 HTTP calls on Redis cache hit, got {second_call_count}"
            )
            assert reading1 is not None and reading2 is not None
            assert reading1.aqi == reading2.aqi
            assert reading1.source == reading2.source

        finally:
            redis_cache._client.flushdb()  # type: ignore[attr-defined]
            reset_cache_for_tests()
            _reset_http_client_for_tests()
