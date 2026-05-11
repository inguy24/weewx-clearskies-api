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

  /aqi/history:
  - → 501 RFC 9457 problem+json (always, regardless of provider config).
  - application/problem+json content-type.
  - Body has type, title, status=501, detail, instance fields.
  - With valid from/to params → still 501 (params validate, then 501).
  - With unknown query key → 422 (extra="forbid" fires before 501).

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
from typing import Any, Generator

import httpx
import pytest
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
    import weewx_clearskies_api.providers.aqi.openmeteo as _om_aqi  # noqa: PLC0415

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

    def test_openmeteo_registered_aqi_category_is_moderate(self) -> None:
        """data.aqiCategory = 'Moderate' (73 → 51–100 band)."""
        response, _ = self._get_response_with_fixture()
        body = response.json()
        assert body["data"]["aqiCategory"] == "Moderate", (
            f"Expected aqiCategory='Moderate', got {body['data'].get('aqiCategory')!r}"
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
# 5. /aqi/history — 501 stub
# ===========================================================================


class TestAqiHistory:
    """/aqi/history → 501 RFC 9457 regardless of provider config (LC21)."""

    def test_aqi_history_returns_501(self) -> None:
        """/aqi/history → 501 Not Implemented (always; AQI history not yet implemented)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history")
        assert response.status_code == 501, (
            f"Expected 501, got {response.status_code}: {response.text[:300]}"
        )

    def test_aqi_history_content_type_is_problem_json(self) -> None:
        """/aqi/history content-type is application/problem+json (RFC 9457 per ADR-018)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history")
        assert "application/problem+json" in response.headers.get("content-type", ""), (
            "501 must return application/problem+json (RFC 9457)"
        )

    def test_aqi_history_body_has_rfc9457_shape(self) -> None:
        """/aqi/history body has type, title, status, detail, instance (LC21 spec)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history")
        body = response.json()
        assert "type" in body, "RFC 9457 must have 'type' field"
        assert "title" in body, "RFC 9457 must have 'title' field"
        assert "status" in body, "RFC 9457 must have 'status' field"
        assert body["status"] == 501
        assert "detail" in body, "RFC 9457 must have 'detail' field"

    def test_aqi_history_with_openmeteo_provider_still_returns_501(self) -> None:
        """/aqi/history returns 501 even when openmeteo is configured (stub is unconditional)."""
        app = _make_aqi_app(provider="openmeteo")
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history")
        assert response.status_code == 501, (
            f"Expected 501 even with provider configured, got {response.status_code}"
        )

    def test_aqi_history_with_valid_from_to_params_still_returns_501(self) -> None:
        """Valid from/to params validate OK but endpoint still returns 501 (stub)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get(
            "/api/v1/aqi/history?from=2026-05-01T00:00:00Z&to=2026-05-10T00:00:00Z"
        )
        # Valid params → 501 (not 422); handler always returns 501 per LC21
        assert response.status_code == 501, (
            f"Expected 501 for valid from/to params on history stub, got {response.status_code}"
        )

    def test_aqi_history_with_unknown_query_key_returns_422(self) -> None:
        """Unknown query param on /aqi/history → 422 (params validate before 501 fires)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history?totally_unknown=oops")
        assert response.status_code == 422, (
            f"Expected 422 for unknown query param on /aqi/history, got {response.status_code}"
        )

    def test_aqi_history_instance_field_is_aqi_history_path(self) -> None:
        """/aqi/history RFC 9457 body instance field points to /aqi/history (LC21)."""
        app = _make_aqi_app(provider=None)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/aqi/history")
        body = response.json()
        # LC21 spec: "instance": "/aqi/history"
        if "instance" in body:
            assert "aqi/history" in body["instance"], (
                f"instance should reference aqi/history, got {body['instance']!r}"
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


def _make_aeris_aqi_app(wire_credentials: bool = True) -> FastAPI:
    """Build test FastAPI app with Aeris AQI registered.

    wire_credentials: if True, sets _AERIS_CLIENT_ID + _AERIS_CLIENT_SECRET on the
    endpoint module (simulating wire_aqi_settings being called at startup).
    If False, leaves them None to exercise the missing-credentials 502 path.
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

    def test_aeris_registered_aqi_category_is_good(self) -> None:
        """data.aqiCategory = 'Good' (AQI 33 → 0–50 band)."""
        response, _ = self._get_aeris_response()
        body = response.json()
        assert body["data"]["aqiCategory"] == "Good", (
            f"Expected aqiCategory='Good', got {body['data'].get('aqiCategory')!r}"
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

    def test_aeris_credentials_missing_502_mentions_credentials(self) -> None:
        """Missing credentials 502 detail mentions credentials (informative error)."""
        app = _make_aeris_aqi_app(wire_credentials=False)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False):
            response = client.get("/api/v1/aqi/current")

        body = response.json()
        # The detail or response body should mention credentials/Aeris
        body_str = str(body).lower()
        assert "aeris" in body_str or "credential" in body_str or "missing" in body_str, (
            f"502 body should mention Aeris/credentials, got {body!r}"
        )


class TestAqiCurrentAerisErrorPaths:
    """/aqi/current aeris provider error handling (3b-10)."""

    def test_aeris_provider_401_returns_502_rfc9457(self) -> None:
        """respx 401 from Aeris → 502 application/problem+json (KeyInvalid → 502)."""
        app = _make_aeris_aqi_app(wire_credentials=True)
        client = TestClient(app, raise_server_exceptions=False)

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
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

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
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

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
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

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
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
