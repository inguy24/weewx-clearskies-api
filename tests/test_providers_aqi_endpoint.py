"""Endpoint tests for GET /aqi/current + GET /aqi/history (3b-9, extended 3b-10).

Covers per the task-3b-9 brief §Test-author parallel scope (test_aqi.py):

  /aqi/current — no provider configured:
  - 200 + data: null + source: "none" when no AQI provider in registry.

  /aqi/current — openmeteo configured + respx-mocked:
  - 200 + canonical AQIReading in data + source="openmeteo".
  - AQIResponse envelope shape: data, units, source, generatedAt all present.
  - generatedAt is UTC ISO-8601 Z format.

  /aqi/current — provider error paths:
  - respx 5xx from provider → 502 RFC 9457 problem+json.
  - respx 429 from provider → 503 RFC 9457 + Retry-After header.

  /aqi/current — unknown query key:
  - ?unknown_key=bad → 422 (extra="forbid" via Depends pattern).

  /aqi/history (P4-T3, ADR-013 corrected — reads from weewx archive):
  - → 200 AQIHistoryResponse with data, units, source, generatedAt, page fields.
  - No AQI columns configured (Path B default) → 200 + empty data list.
  - source = "weewx" (always; reads from archive, not external provider).
  - page field present with limit and totalRecords=0 for unconfigured Path B.
  - Valid from/to params → 200 (not 501; endpoint is now implemented).
  - With unknown query key → 422 (extra="forbid" fires).

3b-10 extension — /aqi/current — aeris provider:
  - aeris registered + credentials wired + respx mock → 200 + canonical AQIReading.
  - aeris registered but credentials NOT wired → 502 "Aeris credentials missing".
  - aeris + respx 401 → 502 RFC 9457 (KeyInvalid → 502).
  - aeris + respx 429 → 503 RFC 9457 + Retry-After.

Pydantic + Depends pattern (coding.md §1, security-baseline §3.5):
  Unknown query keys rejected with 422 via extra="forbid" + Depends wrapper.

ADR references: ADR-013, ADR-017, ADR-018, ADR-038.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixture directory
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "providers" / "aqi"
_OPENMETEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Station coordinates matching fixture
_LAT = 47.6062
_LON = -122.3321


def _load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture file from tests/fixtures/providers/aqi/."""
    path = _FIXTURES_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return json.loads(fh.read())


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------


def _reset_provider_state() -> None:
    """Reset cache, registry, and module-level http client."""
    import weewx_clearskies_api.providers.aqi.openmeteo as _om_aqi  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
        reset_cache_for_tests,
        wire_cache_from_env,
    )
    from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
        reset_provider_registry_for_tests,
    )
    from weewx_clearskies_api.providers.aqi.openmeteo import (  # noqa: PLC0415
        _reset_http_client_for_tests,
    )

    reset_cache_for_tests()
    reset_provider_registry_for_tests()
    _reset_http_client_for_tests()
    _om_aqi._rate_limiter._calls.clear()
    wire_cache_from_env()


def _wire_test_station_at_seattle() -> None:
    """Wire station at Seattle coordinates (matching the AQI fixture)."""
    from weewx_clearskies_api.services import station as station_mod  # noqa: PLC0415
    from weewx_clearskies_api.services.station import StationInfo, reset_cache  # noqa: PLC0415
    reset_cache()
    station_mod._cached_station = StationInfo(
        station_id="test-aqi-station",
        name="Test AQI Station",
        latitude=_LAT,
        longitude=_LON,
        altitude=59.0,
        timezone="America/Los_Angeles",
        timezone_offset_minutes=-420,
        unit_system="US",
        hardware=None,
    )


def _make_aqi_app(provider: str | None = None) -> FastAPI:
    """Build a test FastAPI app with the AQI endpoint registered.

    provider: "openmeteo" to register the capability; None for no-provider path.
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

    _reset_provider_state()
    _wire_test_station_at_seattle()

    # Register the AQI capability if provider specified
    if provider == "openmeteo":
        from weewx_clearskies_api.providers.aqi.openmeteo import CAPABILITY  # noqa: PLC0415
        wire_providers([CAPABILITY])

    settings = Settings(
        api=ApiSettings({}),
        health=HealthSettings({}),
        logging_settings=LoggingSettings({}),
        ratelimit=RateLimitSettings({}),
        database=DatabaseSettings({}),
    )
    return create_app(settings)


# ===========================================================================
# 1. /aqi/current — no provider configured
# ===========================================================================


class TestAqiCurrentNoProvider:
    """/aqi/current → 200 data:null source:'none' when no AQI provider in registry."""

    def test_no_provider_returns_200(self) -> None:
        """No AQI provider configured → 200 (not 404 or 503; per LC19 decision tree)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/current")
        assert response.status_code == 200, (
            f"Expected 200 (no provider), got {response.status_code}: {response.text[:300]}"
        )

    def test_no_provider_data_is_null(self) -> None:
        """No AQI provider → data field is null (AQIResponse.data: null)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/current")
        body = response.json()
        assert body["data"] is None, (
            f"Expected data=null with no provider, got {body.get('data')!r}"
        )

    def test_no_provider_source_is_none_string(self) -> None:
        """No AQI provider → source = 'none' (per LC19 decision tree)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/current")
        body = response.json()
        assert body["source"] == "none", (
            f"Expected source='none' with no provider, got {body.get('source')!r}"
        )

    def test_no_provider_response_has_generated_at(self) -> None:
        """No AQI provider → generatedAt field present in response."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/current")
        body = response.json()
        assert "generatedAt" in body, "generatedAt must be present in AQIResponse"
        assert body["generatedAt"].endswith("Z"), (
            f"generatedAt must be UTC Z format, got {body['generatedAt']!r}"
        )

    def test_no_provider_response_has_units_block(self) -> None:
        """No AQI provider → units block present in response envelope."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/current")
        body = response.json()
        assert "units" in body, "AQIResponse envelope must have 'units' field"
        assert isinstance(body["units"], dict), (
            f"units must be a dict, got {type(body['units']).__name__!r}"
        )


# ===========================================================================
# 2. /aqi/current — openmeteo provider configured + respx-mocked
# ===========================================================================


class TestAqiCurrentOpenMeteoRegistered:
    """/aqi/current with openmeteo CAPABILITY registered + respx mock."""

    def _get_response_with_fixture(self, fixture_name: str = "openmeteo_current.json") -> Any:
        """Build app, make request with respx-mocked upstream, return response."""
        app = _make_aqi_app(provider="openmeteo")
        client = TestClient(app, raise_server_exceptions=False)
        data = _load_fixture(fixture_name)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            return client.get("/api/v1/aqi/current"), mock

    def test_openmeteo_registered_returns_200(self) -> None:
        """openmeteo registered + valid response → 200."""
        response, _ = self._get_response_with_fixture()
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text[:300]}"
        )

    def test_openmeteo_registered_source_is_openmeteo(self) -> None:
        """source = 'openmeteo' in AQIResponse envelope."""
        response, _ = self._get_response_with_fixture()
        body = response.json()
        assert body["source"] == "openmeteo", (
            f"Expected source='openmeteo', got {body.get('source')!r}"
        )

    def test_openmeteo_registered_data_is_aqi_reading(self) -> None:
        """data field is AQIReading (not null) when provider returns a reading."""
        response, _ = self._get_response_with_fixture()
        body = response.json()
        data = body["data"]
        assert data is not None, "Expected AQIReading in data, got null"
        assert isinstance(data, dict), (
            f"data must be a dict (AQIReading), got {type(data).__name__!r}"
        )

    def test_openmeteo_registered_aqi_value_is_73(self) -> None:
        """data.aqi = 73 (from fixture; rounded to int per spec)."""
        response, _ = self._get_response_with_fixture()
        body = response.json()
        assert body["data"]["aqi"] == 73, (
            f"Expected aqi=73, got {body['data'].get('aqi')!r}"
        )

    def test_openmeteo_registered_aqi_scale_is_epa(self) -> None:
        """data.aqiScale = 'epa' (Open-Meteo us_aqi is EPA 0–500 native)."""
        response, _ = self._get_response_with_fixture()
        body = response.json()
        assert body["data"]["aqiScale"] == "epa", (
            f"Expected aqiScale='epa', got {body['data'].get('aqiScale')!r}"
        )

    def test_openmeteo_registered_aqi_category_is_null(self) -> None:
        """data.aqiCategory = null (dashboard-computed; parsers set None)."""
        response, _ = self._get_response_with_fixture()
        body = response.json()
        assert body["data"]["aqiCategory"] is None, (
            f"Expected aqiCategory=null (dashboard-computed), got {body['data'].get('aqiCategory')!r}"
        )

    def test_openmeteo_registered_aqi_main_pollutant_is_pm25(self) -> None:
        """data.aqiMainPollutant = 'PM2.5' (argmax of sub-AQIs from fixture)."""
        response, _ = self._get_response_with_fixture()
        body = response.json()
        assert body["data"]["aqiMainPollutant"] == "PM2.5", (
            f"Expected aqiMainPollutant='PM2.5', got {body['data'].get('aqiMainPollutant')!r}"
        )

    def test_openmeteo_registered_aqi_location_is_null(self) -> None:
        """data.aqiLocation = null (PARTIAL-DOMAIN — Open-Meteo has no location field)."""
        response, _ = self._get_response_with_fixture()
        body = response.json()
        assert body["data"].get("aqiLocation") is None, (
            f"Expected aqiLocation=null (PARTIAL-DOMAIN), "
            f"got {body['data'].get('aqiLocation')!r}"
        )

    def test_openmeteo_registered_data_source_is_openmeteo(self) -> None:
        """data.source = 'openmeteo' (provider_id literal on AQIReading)."""
        response, _ = self._get_response_with_fixture()
        body = response.json()
        assert body["data"]["source"] == "openmeteo"

    def test_openmeteo_registered_observed_at_is_utc_z(self) -> None:
        """data.observedAt ends with Z (UTC ISO-8601, LC4 + ADR-020)."""
        response, _ = self._get_response_with_fixture()
        body = response.json()
        observed_at = body["data"]["observedAt"]
        assert observed_at.endswith("Z"), (
            f"observedAt must end with Z, got {observed_at!r}"
        )

    def test_openmeteo_registered_envelope_has_required_fields(self) -> None:
        """AQIResponse envelope has all four required fields: data, units, source, generatedAt."""
        response, _ = self._get_response_with_fixture()
        body = response.json()
        for field in ("data", "units", "source", "generatedAt"):
            assert field in body, f"AQIResponse envelope missing required field {field!r}"

    def test_openmeteo_registered_generated_at_is_utc_z(self) -> None:
        """generatedAt ends with Z (UTC ISO-8601, ADR-020)."""
        response, _ = self._get_response_with_fixture()
        body = response.json()
        assert body["generatedAt"].endswith("Z"), (
            f"generatedAt must end with Z, got {body['generatedAt']!r}"
        )


# ===========================================================================
# 3. /aqi/current — provider error paths
# ===========================================================================


class TestAqiCurrentProviderErrors:
    """/aqi/current error handling: 5xx → 502, 429 → 503 + Retry-After."""

    def test_provider_5xx_returns_502_rfc9457(self) -> None:
        """respx 5xx from Open-Meteo → 502 application/problem+json."""
        app = _make_aqi_app(provider="openmeteo")
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(500, json={"reason": "server error"})
            )
            response = client.get("/api/v1/aqi/current")

        assert response.status_code == 502, (
            f"Provider 5xx must map to 502, got {response.status_code}: {response.text[:300]}"
        )
        assert "application/problem+json" in response.headers.get("content-type", ""), (
            "502 must return application/problem+json (RFC 9457)"
        )

    def test_provider_5xx_response_body_has_type_and_status(self) -> None:
        """Provider 5xx → 502 body has 'type' and 'status' fields (RFC 9457 shape)."""
        app = _make_aqi_app(provider="openmeteo")
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(500, json={"reason": "error"})
            )
            response = client.get("/api/v1/aqi/current")

        body = response.json()
        assert "type" in body, "RFC 9457 error must have 'type' field"
        assert "status" in body, "RFC 9457 error must have 'status' field"
        assert body["status"] == 502

    def test_provider_429_returns_503_rfc9457(self) -> None:
        """respx 429 from Open-Meteo → 503 application/problem+json."""
        app = _make_aqi_app(provider="openmeteo")
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={"reason": "too many requests"},
                    headers={"Retry-After": "60"},
                )
            )
            response = client.get("/api/v1/aqi/current")

        assert response.status_code == 503, (
            f"Provider 429 must map to 503, got {response.status_code}: {response.text[:300]}"
        )
        assert "application/problem+json" in response.headers.get("content-type", ""), (
            "503 must return application/problem+json (RFC 9457)"
        )

    def test_provider_429_response_has_retry_after_header(self) -> None:
        """Provider 429 → 503 response includes Retry-After header (ADR-018, LC20)."""
        app = _make_aqi_app(provider="openmeteo")
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={"reason": "rate limit"},
                    headers={"Retry-After": "90"},
                )
            )
            response = client.get("/api/v1/aqi/current")

        assert "Retry-After" in response.headers, (
            "503 from QuotaExhausted must include Retry-After header"
        )


# ===========================================================================
# 4. /aqi/current — unknown query key rejection
# ===========================================================================


class TestAqiCurrentUnknownQueryKey:
    """Unknown query keys → 422 (extra='forbid' via Depends pattern, coding.md §1)."""

    def test_unknown_query_key_returns_422(self) -> None:
        """?unknown_key=bad → 422 (not 200; extra='forbid' fires via Depends wrapper)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/current?unknown_key=bad")
        assert response.status_code == 422, (
            f"Expected 422 for unknown query key, got {response.status_code}"
        )

    def test_no_params_returns_200_not_422(self) -> None:
        """No params (empty query string) → 200 (AQIQueryParams accepts empty)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/current")
        assert response.status_code == 200


# ===========================================================================
# 5. /aqi/history — reads from weewx archive (P4-T3, ADR-013 corrected)
# ===========================================================================


class TestAqiHistory:
    """/aqi/history → 200 AQIHistoryResponse (P4-T3; reads from weewx archive).

    Default state: no AQI columns configured in [aqi.history] (Path B).
    Endpoint returns 200 + empty data list + total=0 — not an error.
    source is always "weewx" (archive-backed, not provider-backed).
    """

    def test_aqi_history_returns_200(self) -> None:
        """/aqi/history → 200 OK (implemented; reads from weewx archive)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history")
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text[:300]}"
        )

    def test_aqi_history_content_type_is_json(self) -> None:
        """/aqi/history content-type is application/json."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history")
        assert "application/json" in response.headers.get("content-type", ""), (
            f"Expected application/json, got {response.headers.get('content-type')!r}"
        )

    def test_aqi_history_envelope_has_required_fields(self) -> None:
        """/aqi/history response has data, units, source, generatedAt, page."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history")
        body = response.json()
        for field in ("data", "units", "source", "generatedAt", "page"):
            assert field in body, f"AQIHistoryResponse missing required field {field!r}"

    def test_aqi_history_no_columns_returns_empty_data(self) -> None:
        """No [aqi.history] columns configured (Path B) → data is an empty list."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history")
        body = response.json()
        assert body["data"] == [], (
            f"Expected empty data list (Path B), got {body.get('data')!r}"
        )

    def test_aqi_history_source_is_weewx(self) -> None:
        """source = 'weewx' (reads from archive, not from an external provider)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history")
        body = response.json()
        assert body["source"] == "weewx", (
            f"Expected source='weewx', got {body.get('source')!r}"
        )

    def test_aqi_history_generated_at_is_utc_z(self) -> None:
        """generatedAt ends with Z (UTC ISO-8601, ADR-020)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history")
        body = response.json()
        assert body["generatedAt"].endswith("Z"), (
            f"generatedAt must end with Z, got {body['generatedAt']!r}"
        )

    def test_aqi_history_page_has_limit_field(self) -> None:
        """page block present and contains a limit field."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history")
        body = response.json()
        page = body.get("page", {})
        assert "limit" in page, f"page block must have 'limit', got {page!r}"

    def test_aqi_history_with_openmeteo_provider_returns_200(self) -> None:
        """/aqi/history returns 200 when openmeteo is configured (archive-based; provider-independent)."""
        app = _make_aqi_app(provider="openmeteo")
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history")
        assert response.status_code == 200, (
            f"Expected 200 with openmeteo configured, got {response.status_code}"
        )

    def test_aqi_history_with_valid_from_to_params_returns_200(self) -> None:
        """Valid from/to params → 200 (endpoint is implemented; params validated OK)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get(
            "/api/v1/aqi/history?from=2026-05-01T00:00:00Z&to=2026-05-10T00:00:00Z"
        )
        assert response.status_code == 200, (
            f"Expected 200 for valid from/to params, got {response.status_code}"
        )

    def test_aqi_history_with_unknown_query_key_returns_422(self) -> None:
        """Unknown query param on /aqi/history → 422 (extra='forbid' via Depends wrapper)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history?totally_unknown=oops")
        assert response.status_code == 422, (
            f"Expected 422 for unknown query param on /aqi/history, got {response.status_code}"
        )


# ===========================================================================
# 6. /aqi/current — aeris provider (3b-10 extension)
# ===========================================================================

# Aeris AQI base URL for respx mocking
_AERIS_AQ_BASE_URL = "https://data.api.xweather.com"
# Full URL as constructed by fetch(): lat/lon rounded to 6dp per aeris.py fetch()
_AERIS_AQ_URL = f"{_AERIS_AQ_BASE_URL}/airquality/{round(_LAT, 6)},{round(_LON, 6)}"
_TEST_CLIENT_ID = "TEST_AERIS_ID"
_TEST_CLIENT_SECRET = "TEST_AERIS_SECRET"


def _reset_aeris_provider_state() -> None:
    """Reset provider registry, cache, aeris http client + rate limiter."""
    import weewx_clearskies_api.providers.aqi.aeris as _aeris  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
        reset_cache_for_tests,
        wire_cache_from_env,
    )
    from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
        reset_provider_registry_for_tests,
    )
    from weewx_clearskies_api.providers.aqi.aeris import (
        _reset_http_client_for_tests,  # noqa: PLC0415
    )

    reset_cache_for_tests()
    reset_provider_registry_for_tests()
    _reset_http_client_for_tests()
    _aeris._rate_limiter._calls.clear()
    wire_cache_from_env()


def _make_aeris_aqi_app(wire_credentials: bool = True) -> FastAPI:
    """Build test FastAPI app with Aeris AQI registered.

    wire_credentials: if True, sets _AERIS_CLIENT_ID + _AERIS_CLIENT_SECRET on the
    endpoint module (simulating wire_aqi_settings being called at startup).
    If False, leaves them None to exercise the missing-credentials 502 path.
    """
    import weewx_clearskies_api.endpoints.aqi as _aqi_endpoint  # noqa: PLC0415
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

    _reset_aeris_provider_state()
    _wire_test_station_at_seattle()

    # Register Aeris AQI CAPABILITY
    from weewx_clearskies_api.providers.aqi.aeris import CAPABILITY  # noqa: PLC0415
    wire_providers([CAPABILITY])

    # Wire credentials into the endpoint module (simulating wire_aqi_settings)
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


class TestAqiCurrentAerisRegistered:
    """/aqi/current with aeris CAPABILITY registered + respx mock (3b-10)."""

    def _get_aeris_response(self, wire_credentials: bool = True) -> Any:
        """Build app with aeris, request /aqi/current with respx-mocked upstream."""
        app = _make_aeris_aqi_app(wire_credentials=wire_credentials)
        client = TestClient(app, raise_server_exceptions=False)
        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            return client.get("/api/v1/aqi/current"), mock

    def test_aeris_registered_returns_200(self) -> None:
        """aeris registered + valid response + credentials wired → 200."""
        response, _ = self._get_aeris_response()
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text[:300]}"
        )

    def test_aeris_registered_source_is_aeris(self) -> None:
        """source = 'aeris' in AQIResponse envelope."""
        response, _ = self._get_aeris_response()
        body = response.json()
        assert body["source"] == "aeris", (
            f"Expected source='aeris', got {body.get('source')!r}"
        )

    def test_aeris_registered_data_is_aqi_reading(self) -> None:
        """data field is AQIReading (not null) when aeris returns a reading."""
        response, _ = self._get_aeris_response()
        body = response.json()
        data = body["data"]
        assert data is not None, "Expected AQIReading in data, got null"
        assert isinstance(data, dict), (
            f"data must be a dict (AQIReading), got {type(data).__name__!r}"
        )

    def test_aeris_registered_aqi_value_is_33(self) -> None:
        """data.aqi = 33 (from real fixture; AQI rounded to int)."""
        response, _ = self._get_aeris_response()
        body = response.json()
        assert body["data"]["aqi"] == 33, (
            f"Expected aqi=33, got {body['data'].get('aqi')!r}"
        )

    def test_aeris_registered_aqi_scale_is_epa(self) -> None:
        """data.aqiScale = 'epa' (Aeris airnow filter is EPA native)."""
        response, _ = self._get_aeris_response()
        body = response.json()
        assert body["data"]["aqiScale"] == "epa", (
            f"Expected aqiScale='epa', got {body['data'].get('aqiScale')!r}"
        )

    def test_aeris_registered_aqi_category_is_null(self) -> None:
        """data.aqiCategory = null (dashboard-computed; parsers set None)."""
        response, _ = self._get_aeris_response()
        body = response.json()
        assert body["data"]["aqiCategory"] is None, (
            f"Expected aqiCategory=null (dashboard-computed), got {body['data'].get('aqiCategory')!r}"
        )

    def test_aeris_registered_aqi_location_is_seattle(self) -> None:
        """data.aqiLocation = 'seattle' (Aeris supplies place.name — NOT PARTIAL-DOMAIN)."""
        response, _ = self._get_aeris_response()
        body = response.json()
        assert body["data"].get("aqiLocation") == "seattle", (
            f"Expected aqiLocation='seattle', got {body['data'].get('aqiLocation')!r}"
        )

    def test_aeris_registered_data_source_is_aeris(self) -> None:
        """data.source = 'aeris' (provider_id literal on AQIReading)."""
        response, _ = self._get_aeris_response()
        body = response.json()
        assert body["data"]["source"] == "aeris"

    def test_aeris_registered_observed_at_is_utc_z(self) -> None:
        """data.observedAt ends with Z (UTC ISO-8601, LC4 + ADR-020)."""
        response, _ = self._get_aeris_response()
        body = response.json()
        observed_at = body["data"]["observedAt"]
        assert observed_at.endswith("Z"), (
            f"observedAt must end with Z, got {observed_at!r}"
        )

    def test_aeris_registered_envelope_has_required_fields(self) -> None:
        """AQIResponse envelope has all four required fields: data, units, source, generatedAt."""
        response, _ = self._get_aeris_response()
        body = response.json()
        for field in ("data", "units", "source", "generatedAt"):
            assert field in body, f"AQIResponse envelope missing required field {field!r}"

    def test_aeris_credentials_missing_returns_502(self) -> None:
        """aeris registered but credentials NOT wired → 502 'Aeris credentials missing'."""
        app = _make_aeris_aqi_app(wire_credentials=False)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False):
            response = client.get("/api/v1/aqi/current")

        assert response.status_code == 502, (
            f"Expected 502 for missing credentials, got {response.status_code}: {response.text[:300]}"
        )

    def test_aeris_credentials_missing_502_rfc9457_body(self) -> None:
        """Missing credentials 502 returns application/problem+json RFC 9457 body."""
        app = _make_aeris_aqi_app(wire_credentials=False)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False):
            response = client.get("/api/v1/aqi/current")

        # Verify RFC 9457 shape (error handler wraps HTTPException detail; check shape not text)
        assert "application/problem+json" in response.headers.get("content-type", ""), (
            "502 must return application/problem+json (RFC 9457)"
        )
        body = response.json()
        assert "status" in body, "RFC 9457 body must have 'status' field"
        assert body["status"] == 502, f"Expected status=502, got {body.get('status')!r}"


class TestAqiCurrentAerisErrorPaths:
    """/aqi/current aeris provider error handling (3b-10)."""

    def test_aeris_provider_401_returns_502_rfc9457(self) -> None:
        """respx 401 from Aeris → 502 application/problem+json (KeyInvalid → 502)."""
        app = _make_aeris_aqi_app(wire_credentials=True)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(401, json={"error": "unauthorized"})
            )
            response = client.get("/api/v1/aqi/current")

        assert response.status_code == 502, (
            f"Provider 401 must map to 502, got {response.status_code}: {response.text[:300]}"
        )
        assert "application/problem+json" in response.headers.get("content-type", ""), (
            "502 must return application/problem+json (RFC 9457)"
        )

    def test_aeris_provider_401_response_has_rfc9457_shape(self) -> None:
        """Provider 401 → 502 body has 'type' and 'status' fields (RFC 9457 shape)."""
        app = _make_aeris_aqi_app(wire_credentials=True)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(401, json={"error": "unauthorized"})
            )
            response = client.get("/api/v1/aqi/current")

        body = response.json()
        assert "type" in body, "RFC 9457 error must have 'type' field"
        assert "status" in body, "RFC 9457 error must have 'status' field"
        assert body["status"] == 502

    def test_aeris_provider_429_returns_503_rfc9457(self) -> None:
        """respx 429 from Aeris → 503 application/problem+json (QuotaExhausted → 503)."""
        app = _make_aeris_aqi_app(wire_credentials=True)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={"reason": "too many requests"},
                    headers={"Retry-After": "60"},
                )
            )
            response = client.get("/api/v1/aqi/current")

        assert response.status_code == 503, (
            f"Provider 429 must map to 503, got {response.status_code}: {response.text[:300]}"
        )
        assert "application/problem+json" in response.headers.get("content-type", ""), (
            "503 must return application/problem+json (RFC 9457)"
        )

    def test_aeris_provider_429_includes_retry_after_header(self) -> None:
        """Provider 429 → 503 response includes Retry-After header (ADR-018)."""
        app = _make_aeris_aqi_app(wire_credentials=True)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={"reason": "rate limit"},
                    headers={"Retry-After": "90"},
                )
            )
            response = client.get("/api/v1/aqi/current")

        assert "Retry-After" in response.headers, (
            "503 from QuotaExhausted must include Retry-After header"
        )


# ===========================================================================
# 7. /aqi/current — openweathermap provider (3b-11 extension)
# ===========================================================================

# OWM Air Pollution URL for respx mocking
_OWM_AIRPOL_BASE_URL = "https://api.openweathermap.org"
_OWM_AIRPOL_URL = _OWM_AIRPOL_BASE_URL + "/data/2.5/air_pollution"
_TEST_OWM_APPID = "TEST_OWM_APPID_ENDPOINT"


def _reset_owm_aqi_provider_state() -> None:
    """Reset provider registry, cache, OWM http client + rate limiter."""
    import weewx_clearskies_api.providers.aqi.openweathermap as _owm_aqi  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
        reset_cache_for_tests,
        wire_cache_from_env,
    )
    from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
        reset_provider_registry_for_tests,
    )
    from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
        _reset_http_client_for_tests,
    )

    reset_cache_for_tests()
    reset_provider_registry_for_tests()
    _reset_http_client_for_tests()
    _owm_aqi._rate_limiter._calls.clear()
    wire_cache_from_env()


def _make_owm_aqi_app(wire_appid: bool = True) -> FastAPI:
    """Build test FastAPI app with OWM AQI registered.

    wire_appid: if True, sets _OWM_APPID on the endpoint module.
    If False, leaves it None to exercise the missing-credentials 502 path.
    """
    import weewx_clearskies_api.endpoints.aqi as _aqi_endpoint  # noqa: PLC0415
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

    _reset_owm_aqi_provider_state()
    _wire_test_station_at_seattle()

    # Register OWM AQI CAPABILITY
    from weewx_clearskies_api.providers.aqi.openweathermap import CAPABILITY  # noqa: PLC0415
    wire_providers([CAPABILITY])

    # Wire appid into the endpoint module (simulating wire_aqi_settings)
    if wire_appid:
        _aqi_endpoint._OWM_APPID = _TEST_OWM_APPID
    else:
        _aqi_endpoint._OWM_APPID = None

    settings = Settings(
        api=ApiSettings({}),
        health=HealthSettings({}),
        logging_settings=LoggingSettings({}),
        ratelimit=RateLimitSettings({}),
        database=DatabaseSettings({}),
    )
    return create_app(settings)


class TestAqiCurrentOpenWeatherMapRegistered:
    """/aqi/current with openweathermap CAPABILITY registered + respx mock (3b-11)."""

    def _get_owm_response(self, wire_appid: bool = True) -> Any:
        """Build app with OWM AQI, request /aqi/current with respx-mocked upstream."""
        app = _make_owm_aqi_app(wire_appid=wire_appid)
        client = TestClient(app, raise_server_exceptions=False)
        data = _load_fixture("openweathermap_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            return client.get("/api/v1/aqi/current"), mock

    def test_owm_registered_returns_200(self) -> None:
        """openweathermap registered + valid response + appid wired → 200."""
        response, _ = self._get_owm_response()
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text[:300]}"
        )

    def test_owm_registered_source_is_openweathermap(self) -> None:
        """source = 'openweathermap' in AQIResponse envelope."""
        response, _ = self._get_owm_response()
        body = response.json()
        assert body["source"] == "openweathermap", (
            f"Expected source='openweathermap', got {body.get('source')!r}"
        )

    def test_owm_registered_data_is_aqi_reading(self) -> None:
        """data field is AQIReading (not null) when OWM returns a reading."""
        response, _ = self._get_owm_response()
        body = response.json()
        data = body["data"]
        assert data is not None, "Expected AQIReading in data, got null"
        assert isinstance(data, dict), (
            f"data must be a dict (AQIReading), got {type(data).__name__!r}"
        )

    def test_owm_registered_aqi_value_is_owm_ordinal_2(self) -> None:
        """data.aqi = 2 (OWM main.aqi ordinal from fixture, served as-is)."""
        response, _ = self._get_owm_response()
        body = response.json()
        assert body["data"]["aqi"] == 2, (
            f"Expected aqi=2 (OWM main.aqi ordinal from fixture), got {body['data'].get('aqi')!r}"
        )

    def test_owm_registered_aqi_scale_is_owm(self) -> None:
        """data.aqiScale = 'owm' (OWM 1–5 ordinal scale)."""
        response, _ = self._get_owm_response()
        body = response.json()
        assert body["data"]["aqiScale"] == "owm", (
            f"Expected aqiScale='owm', got {body['data'].get('aqiScale')!r}"
        )

    def test_owm_registered_aqi_category_is_null(self) -> None:
        """data.aqiCategory = null (dashboard-computed; parsers set None)."""
        response, _ = self._get_owm_response()
        body = response.json()
        assert body["data"]["aqiCategory"] is None, (
            f"Expected aqiCategory=null (dashboard-computed), got {body['data'].get('aqiCategory')!r}"
        )

    def test_owm_registered_aqi_main_pollutant_is_null(self) -> None:
        """data.aqiMainPollutant = null (OWM Air Pollution does not supply dominant pollutant)."""
        response, _ = self._get_owm_response()
        body = response.json()
        assert body["data"]["aqiMainPollutant"] is None, (
            f"Expected aqiMainPollutant=null (not supplied by OWM), got {body['data'].get('aqiMainPollutant')!r}"
        )

    def test_owm_registered_aqi_location_is_null(self) -> None:
        """data.aqiLocation = null (PARTIAL-DOMAIN — OWM Air Pollution has no location field)."""
        response, _ = self._get_owm_response()
        body = response.json()
        assert body["data"].get("aqiLocation") is None, (
            f"Expected aqiLocation=null (PARTIAL-DOMAIN), "
            f"got {body['data'].get('aqiLocation')!r}"
        )

    def test_owm_registered_data_source_is_openweathermap(self) -> None:
        """data.source = 'openweathermap' (provider_id literal on AQIReading)."""
        response, _ = self._get_owm_response()
        body = response.json()
        assert body["data"]["source"] == "openweathermap"

    def test_owm_registered_observed_at_is_utc_z(self) -> None:
        """data.observedAt ends with Z (UTC ISO-8601, LC17 + ADR-020)."""
        response, _ = self._get_owm_response()
        body = response.json()
        observed_at = body["data"]["observedAt"]
        assert observed_at.endswith("Z"), (
            f"observedAt must end with Z, got {observed_at!r}"
        )

    def test_owm_registered_envelope_has_required_fields(self) -> None:
        """AQIResponse envelope has all four required fields: data, units, source, generatedAt."""
        response, _ = self._get_owm_response()
        body = response.json()
        for field in ("data", "units", "source", "generatedAt"):
            assert field in body, f"AQIResponse envelope missing required field {field!r}"

    def test_owm_appid_missing_returns_502(self) -> None:
        """openweathermap registered but appid NOT wired → 502 'OpenWeatherMap appid missing'."""
        app = _make_owm_aqi_app(wire_appid=False)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False):
            response = client.get("/api/v1/aqi/current")

        assert response.status_code == 502, (
            f"Expected 502 for missing appid, got {response.status_code}: {response.text[:300]}"
        )

    def test_owm_appid_missing_502_rfc9457_body(self) -> None:
        """Missing appid 502 returns application/problem+json RFC 9457 body."""
        app = _make_owm_aqi_app(wire_appid=False)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False):
            response = client.get("/api/v1/aqi/current")

        assert "application/problem+json" in response.headers.get("content-type", ""), (
            "502 must return application/problem+json (RFC 9457)"
        )
        body = response.json()
        assert "status" in body, "RFC 9457 body must have 'status' field"
        assert body["status"] == 502


class TestAqiCurrentOpenWeatherMapErrorPaths:
    """/aqi/current OWM provider error handling (3b-11)."""

    def test_owm_provider_401_returns_502_rfc9457(self) -> None:
        """respx 401 from OWM → 502 application/problem+json (KeyInvalid → 502)."""
        app = _make_owm_aqi_app(wire_appid=True)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(401, json={"cod": 401, "message": "Invalid API key"})
            )
            response = client.get("/api/v1/aqi/current")

        assert response.status_code == 502, (
            f"Provider 401 must map to 502, got {response.status_code}: {response.text[:300]}"
        )
        assert "application/problem+json" in response.headers.get("content-type", ""), (
            "502 must return application/problem+json (RFC 9457)"
        )

    def test_owm_provider_401_response_has_rfc9457_shape(self) -> None:
        """Provider 401 → 502 body has 'type' and 'status' fields (RFC 9457 shape)."""
        app = _make_owm_aqi_app(wire_appid=True)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(401, json={"cod": 401, "message": "Invalid API key"})
            )
            response = client.get("/api/v1/aqi/current")

        body = response.json()
        assert "type" in body, "RFC 9457 error must have 'type' field"
        assert "status" in body, "RFC 9457 error must have 'status' field"
        assert body["status"] == 502

    def test_owm_provider_429_returns_503_rfc9457(self) -> None:
        """respx 429 from OWM → 503 application/problem+json (QuotaExhausted → 503)."""
        app = _make_owm_aqi_app(wire_appid=True)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={"cod": 429, "message": "too many requests"},
                    headers={"Retry-After": "60"},
                )
            )
            response = client.get("/api/v1/aqi/current")

        assert response.status_code == 503, (
            f"Provider 429 must map to 503, got {response.status_code}: {response.text[:300]}"
        )
        assert "application/problem+json" in response.headers.get("content-type", ""), (
            "503 must return application/problem+json (RFC 9457)"
        )

    def test_owm_provider_429_includes_retry_after_header(self) -> None:
        """Provider 429 → 503 response includes Retry-After header (ADR-018)."""
        app = _make_owm_aqi_app(wire_appid=True)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={"message": "rate limit"},
                    headers={"Retry-After": "90"},
                )
            )
            response = client.get("/api/v1/aqi/current")

        assert "Retry-After" in response.headers, (
            "503 from QuotaExhausted must include Retry-After header"
        )

    def test_owm_provider_5xx_returns_502_rfc9457(self) -> None:
        """respx 5xx from OWM → 502 application/problem+json (TransientNetworkError → 502)."""
        app = _make_owm_aqi_app(wire_appid=True)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(500, json={"error": "server error"})
            )
            response = client.get("/api/v1/aqi/current")

        assert response.status_code == 502, (
            f"Provider 5xx must map to 502, got {response.status_code}: {response.text[:300]}"
        )
        assert "application/problem+json" in response.headers.get("content-type", ""), (
            "502 must return application/problem+json (RFC 9457)"
        )


# ===========================================================================
# 8. /aqi/history — Path A (archive columns configured + rows present)
# ===========================================================================


class TestAqiHistoryPathA:
    """Path A: AQI columns configured in AQIHistorySettings + row in archive.

    Tests the service function directly (get_aqi_history) rather than going
    through HTTP so we can control the DB schema freely without touching the
    shared in-memory SQLite fixture.  This exercises _build_column_map →
    parameterized SQL query → row-to-AQIReading mapping (source="weewx").
    """

    def test_path_a_returns_one_reading_with_aqi_populated(self) -> None:
        """Path A: one archive row with aqi column → one AQIReading, aqi field set."""
        from datetime import UTC, datetime, timedelta  # noqa: PLC0415

        from sqlalchemy import Column, Float, Integer, MetaData, Table, create_engine  # noqa: PLC0415
        from sqlalchemy.orm import Session  # noqa: PLC0415
        from sqlalchemy.pool import StaticPool  # noqa: PLC0415

        from weewx_clearskies_api.config.settings import AQIHistorySettings  # noqa: PLC0415
        from weewx_clearskies_api.services.aqi_history import get_aqi_history  # noqa: PLC0415

        # Build a private in-memory SQLite engine with an "aqi" column present.
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        meta = MetaData()
        Table(
            "archive",
            meta,
            Column("dateTime", Integer, primary_key=True),
            Column("usUnits", Integer, nullable=False),
            Column("interval", Integer, nullable=False),
            Column("aqi", Float, nullable=True),
        )
        meta.create_all(engine)

        # Insert one row with a known aqi value.
        now_epoch = int(datetime.now(tz=UTC).timestamp())
        one_hour_ago = now_epoch - 3600
        with engine.begin() as conn:
            conn.execute(
                meta.tables["archive"].insert(),
                {"dateTime": one_hour_ago, "usUnits": 1, "interval": 5, "aqi": 42.0},
            )

        # Wire AQIHistorySettings with column_aqi pointing to the "aqi" column.
        hist = AQIHistorySettings({})
        hist.column_aqi = "aqi"

        # Use a time window that encompasses the inserted row.
        from_dt = datetime.now(tz=UTC) - timedelta(hours=48)
        to_dt = datetime.now(tz=UTC) + timedelta(hours=1)

        with Session(engine) as db:
            readings, page_info = get_aqi_history(
                db=db,
                hist=hist,
                from_dt=from_dt,
                to_dt=to_dt,
                limit=50,
                cursor=None,
                page=None,
            )

        assert len(readings) == 1, (
            f"Expected 1 AQIReading (Path A), got {len(readings)}"
        )
        reading = readings[0]
        assert reading.aqi == 42.0, (
            f"Expected aqi=42.0, got {reading.aqi!r}"
        )
        assert reading.source == "weewx", (
            f"Expected source='weewx', got {reading.source!r}"
        )
        assert reading.observedAt.endswith("Z"), (
            f"observedAt must be UTC Z format, got {reading.observedAt!r}"
        )

    def test_path_a_empty_time_window_returns_empty_list(self) -> None:
        """Path A: AQI column configured but no rows in range → empty list (not error)."""
        from datetime import UTC, datetime, timedelta  # noqa: PLC0415

        from sqlalchemy import Column, Float, Integer, MetaData, Table, create_engine  # noqa: PLC0415
        from sqlalchemy.orm import Session  # noqa: PLC0415
        from sqlalchemy.pool import StaticPool  # noqa: PLC0415

        from weewx_clearskies_api.config.settings import AQIHistorySettings  # noqa: PLC0415
        from weewx_clearskies_api.services.aqi_history import get_aqi_history  # noqa: PLC0415

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        meta = MetaData()
        Table(
            "archive",
            meta,
            Column("dateTime", Integer, primary_key=True),
            Column("usUnits", Integer, nullable=False),
            Column("interval", Integer, nullable=False),
            Column("aqi", Float, nullable=True),
        )
        meta.create_all(engine)
        # No rows inserted — table is empty.

        hist = AQIHistorySettings({})
        hist.column_aqi = "aqi"

        # Time window in the distant future — no rows can match.
        from_dt = datetime.now(tz=UTC) + timedelta(days=365)
        to_dt = datetime.now(tz=UTC) + timedelta(days=366)

        with Session(engine) as db:
            readings, page_info = get_aqi_history(
                db=db,
                hist=hist,
                from_dt=from_dt,
                to_dt=to_dt,
                limit=50,
                cursor=None,
                page=None,
            )

        assert readings == [], (
            f"Expected empty list for out-of-range window (Path A), got {readings!r}"
        )
        assert page_info.totalRecords is None  # cursor mode, not page mode

    def test_path_a_malformed_cursor_raises_400_at_endpoint(self) -> None:
        """Malformed cursor on /aqi/history with Path A configured → 400.

        Path B (no columns) returns 200 immediately without touching the cursor.
        Path A reaches decode_cursor and raises ValueError, which the endpoint
        catches and returns as 400.
        """
        import weewx_clearskies_api.endpoints.aqi as _aqi_endpoint  # noqa: PLC0415

        from sqlalchemy import Column, Float, Integer, MetaData, Table, create_engine  # noqa: PLC0415
        from sqlalchemy.pool import StaticPool  # noqa: PLC0415

        from weewx_clearskies_api.config.settings import AQIHistorySettings  # noqa: PLC0415
        from weewx_clearskies_api.db.session import wire_engine  # noqa: PLC0415

        # Build a private SQLite engine with the "aqi" column.
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        meta = MetaData()
        Table(
            "archive",
            meta,
            Column("dateTime", Integer, primary_key=True),
            Column("usUnits", Integer, nullable=False),
            Column("interval", Integer, nullable=False),
            Column("aqi", Float, nullable=True),
        )
        meta.create_all(engine)
        wire_engine(engine)

        # Wire Path A settings into the endpoint module.
        hist = AQIHistorySettings({})
        hist.column_aqi = "aqi"
        _aqi_endpoint._AQI_HISTORY_SETTINGS = hist

        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history?cursor=!!!notbase64!!!")

        # Reset to Path B defaults after the test.
        _aqi_endpoint._AQI_HISTORY_SETTINGS = AQIHistorySettings({})

        assert response.status_code == 400, (
            f"Malformed cursor (Path A) must return 400, got {response.status_code}: {response.text[:300]}"
        )
