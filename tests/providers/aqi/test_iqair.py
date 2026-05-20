"""Unit tests for the IQAir AirVisual AQI provider module (3b-12).

Covers per the task-3b-12 brief §Coverage shape (test_iqair.py):

  Wire-shape Pydantic validation:
  - Nashville published-example fixture loads cleanly (status="success").
  - Extra fields (location.*, weather.* sub-fields) ignored (extra="ignore").
  - Required fields enforced (status required on envelope).
  - envelope status="success" accepted without error.

  _wire_to_canonical happy path:
  - Nashville example: aqius=10, mainus="p2" → aqi=10, aqiCategory="Good",
    aqiMainPollutant="PM2.5", aqiLocation="Nashville, Tennessee",
    observedAt="2019-04-08T18:00:00Z", source="iqair".
  - All pollutant* concentration fields are None (PARTIAL-DOMAIN on free tier).

  _wire_to_canonical edge cases:
  - All-null pollution block (aqius=None) → returns None.
  - Missing city → aqiLocation = None.
  - Missing state → aqiLocation = None.
  - Each mainus code in lookup table → correct canonical id.
  - Unknown mainus code → aqiMainPollutant = None (+ logger.info).

  Envelope error mapping (LC27 / LC12 — mirrors Aeris):
  - status="fail" + message="incorrect_api_key" → KeyInvalid.
  - status="fail" + message="api_key_expired" → KeyInvalid.
  - status="fail" + message="payment required" → KeyInvalid.
  - status="fail" + message="permission_denied" → KeyInvalid.
  - status="fail" + message="forbidden" → KeyInvalid.
  - status="fail" + message="feature_not_available" → KeyInvalid.
  - status="fail" + message="call_limit_reached" → QuotaExhausted(retry_after_seconds=None).
  - status="fail" + message="too_many_requests" → QuotaExhausted(retry_after_seconds=None).
  - status="fail" + message="city_not_found" → ProviderProtocolError.
  - status="fail" + message="no_nearest_station" → ProviderProtocolError.
  - status="fail" + message="node not found" → ProviderProtocolError.

  Pre-call key validation (LC13):
  - Empty key "" → KeyInvalid BEFORE HTTP call.
  - None key → KeyInvalid BEFORE HTTP call.

  Cache 3-way path:
  - Cache hit → canonical reconstruction from cached dict; no HTTP call.
  - Cache hit with _no_reading sentinel → None returned; no HTTP call.
  - Cache miss → HTTP call → reading returned and cached.

  Cache key (LC9):
  - Credentials (key=) NOT in cache key.
  - lat/lon rounded to 4 decimal places (consistent with other AQI providers).
  - Same lat/lon → same key (deterministic).
  - Different lat/lon → different key.
  - Key is 64-char hex string (SHA-256).

  Rate limiter (LC10):
  - RateLimiter configured max_calls=5, window_seconds=60.

  Capability declaration:
  - CAPABILITY.provider_id = "iqair", domain = "aqi".
  - CAPABILITY.auth_required = ("key",).
  - CAPABILITY.supplied_canonical_fields has exactly the 6 free-tier fields.
  - CAPABILITY.supplied_canonical_fields excludes all pollutant concentrations.
  - CAPABILITY.geographic_coverage = "global".
  - CAPABILITY.default_poll_interval_seconds = 900.
  - wire_providers([iqair.CAPABILITY]) → registry has iqair aqi entry.

No DB, no live network. respx mocks outbound httpx calls.
Wire-shape rule: fixtures loaded from tests/fixtures/providers/aqi/.
ADR references: ADR-013, ADR-017, ADR-020, ADR-038.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "providers" / "aqi"

# Coordinates for tests — Nashville is our synthetic fixture
_LAT_NASHVILLE = 36.1767
_LON_NASHVILLE = -86.7386

# Belchertown station coords (used for cache-key tests)
_LAT = 42.2993
_LON = -72.4509
_LAT4 = round(_LAT, 4)
_LON4 = round(_LON, 4)

_IQAIR_BASE_URL = "https://api.airvisual.com"
_IQAIR_NEAREST_CITY_PATH = "/v2/nearest_city"
_IQAIR_NEAREST_CITY_URL = _IQAIR_BASE_URL + _IQAIR_NEAREST_CITY_PATH

_TEST_KEY = "TEST_IQAIR_KEY_ABC123"


def _load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture from tests/fixtures/providers/aqi/."""
    path = _FIXTURES_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return json.loads(fh.read())


# ---------------------------------------------------------------------------
# State-reset helpers
# ---------------------------------------------------------------------------


def _reset_provider_state() -> None:
    """Reset provider registry, cache, rate limiter, and re-wire memory cache.

    Also flushes Redis if CLEARSKIES_CACHE_URL is set — prevents cached readings
    from one test contaminating the next (3b-11 isolation pattern applied to unit tests).
    """
    import os  # noqa: PLC0415

    import weewx_clearskies_api.providers.aqi.iqair as _iqair  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
        reset_cache_for_tests,
        wire_cache_from_env,
    )
    from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
        reset_provider_registry_for_tests,
    )
    from weewx_clearskies_api.providers.aqi.iqair import (
        _reset_http_client_for_tests,  # noqa: PLC0415
    )

    # Flush Redis if configured (3b-11 isolation pattern) — prevents cache hits from
    # earlier tests masking misses/errors in subsequent tests at the same coordinates.
    cache_url = os.environ.get("CLEARSKIES_CACHE_URL")
    if cache_url:
        try:
            import redis as redis_lib  # noqa: PLC0415
            r = redis_lib.from_url(cache_url)
            r.flushdb()
        except Exception:  # noqa: BLE001
            pass  # Redis not reachable — skip flush; tests may fail if cache is dirty

    reset_cache_for_tests()
    reset_provider_registry_for_tests()
    _reset_http_client_for_tests()
    _iqair._rate_limiter._calls.clear()
    wire_cache_from_env()


# ===========================================================================
# 1. Wire-shape Pydantic validation
# ===========================================================================


class TestIQAirWireShapeValidation:
    """Wire-shape models validate correctly against the fixture and edge-case shapes."""

    def test_nashville_fixture_loads_cleanly_via_response_model(self) -> None:
        """iqair_nearest_city_nashville.json loads via _IQAirResponse without error."""
        from weewx_clearskies_api.providers.aqi.iqair import _IQAirResponse  # noqa: PLC0415
        raw = _load_fixture("iqair_nearest_city_nashville.json")
        response = _IQAirResponse.model_validate(raw)
        assert response.status == "success"
        assert response.data is not None

    def test_nashville_fixture_status_is_success(self) -> None:
        """Fixture status="success" parses correctly."""
        from weewx_clearskies_api.providers.aqi.iqair import _IQAirResponse  # noqa: PLC0415
        raw = _load_fixture("iqair_nearest_city_nashville.json")
        response = _IQAirResponse.model_validate(raw)
        assert response.status == "success", (
            f"Expected status='success', got {response.status!r}"
        )

    def test_nashville_fixture_extra_fields_are_ignored(self) -> None:
        """Extra wire fields (location, weather sub-fields) ignored (extra='ignore')."""
        from weewx_clearskies_api.providers.aqi.iqair import _IQAirResponse  # noqa: PLC0415
        raw = _load_fixture("iqair_nearest_city_nashville.json")
        # Inject extra fields the model does not declare
        raw["unexpected_future_field"] = "should be silently dropped"
        raw["data"]["future_field"] = "also dropped"
        response = _IQAirResponse.model_validate(raw)
        assert response is not None, "Extra fields must not cause ValidationError"

    def test_nashville_fixture_city_and_state_parsed(self) -> None:
        """data.city='Nashville' and data.state='Tennessee' parsed correctly."""
        from weewx_clearskies_api.providers.aqi.iqair import _IQAirResponse  # noqa: PLC0415
        raw = _load_fixture("iqair_nearest_city_nashville.json")
        response = _IQAirResponse.model_validate(raw)
        assert response.data is not None
        assert response.data.city == "Nashville", (
            f"Expected city='Nashville', got {response.data.city!r}"
        )
        assert response.data.state == "Tennessee", (
            f"Expected state='Tennessee', got {response.data.state!r}"
        )

    def test_nashville_fixture_pollution_aqius_is_10(self) -> None:
        """data.current.pollution.aqius = 10 (EPA AQI value, direct pass-through)."""
        from weewx_clearskies_api.providers.aqi.iqair import _IQAirResponse  # noqa: PLC0415
        raw = _load_fixture("iqair_nearest_city_nashville.json")
        response = _IQAirResponse.model_validate(raw)
        assert response.data is not None
        assert response.data.current.pollution.aqius == 10, (
            f"Expected aqius=10, got {response.data.current.pollution.aqius!r}"
        )

    def test_nashville_fixture_pollution_mainus_is_p2(self) -> None:
        """data.current.pollution.mainus = 'p2' (dominant pollutant code for PM2.5)."""
        from weewx_clearskies_api.providers.aqi.iqair import _IQAirResponse  # noqa: PLC0415
        raw = _load_fixture("iqair_nearest_city_nashville.json")
        response = _IQAirResponse.model_validate(raw)
        assert response.data is not None
        assert response.data.current.pollution.mainus == "p2", (
            f"Expected mainus='p2', got {response.data.current.pollution.mainus!r}"
        )

    def test_nashville_fixture_pollution_ts_is_iso_z_string(self) -> None:
        """data.current.pollution.ts = '2019-04-08T18:00:00.000Z' (ISO-8601 Z string)."""
        from weewx_clearskies_api.providers.aqi.iqair import _IQAirResponse  # noqa: PLC0415
        raw = _load_fixture("iqair_nearest_city_nashville.json")
        response = _IQAirResponse.model_validate(raw)
        assert response.data is not None
        ts = response.data.current.pollution.ts
        assert ts == "2019-04-08T18:00:00.000Z", (
            f"Expected pollution ts '2019-04-08T18:00:00.000Z', got {ts!r}"
        )

    def test_pollution_model_has_no_concentration_fields(self) -> None:
        """_IQAirPollution model has no concentration fields (pm25/pm10/o3/no2/so2/co).

        Free tier does not supply concentrations; model must not declare them
        (would silently None on real data that lacks them — invisible mismatch).
        Extra fields on the wire are dropped by extra='ignore'.
        """
        from weewx_clearskies_api.providers.aqi.iqair import _IQAirPollution  # noqa: PLC0415
        # The declared fields must be only the 5 free-tier pollution fields
        declared = set(_IQAirPollution.model_fields.keys())
        concentration_field_names = {"pm25", "pm10", "o3", "no2", "so2", "co"}
        overlap = declared & concentration_field_names
        assert not overlap, (
            f"_IQAirPollution must not declare concentration fields "
            f"(paid-tier only, wire shape unverified); found: {overlap!r}"
        )

    def test_status_fail_with_no_data_parses_cleanly(self) -> None:
        """Error envelope status='fail' with no data field validates via _IQAirResponse.

        The IQAir error envelope shape is {"status": "fail", "data": {"message": "..."}}.
        The data sub-dict has no 'current' field, so it cannot be parsed as _IQAirData.
        _IQAirResponse.data is Optional[_IQAirData], so when data fails to parse as
        _IQAirData, the implementation uses raw JSON parsing (response.json()) to extract
        data.message — not the Pydantic model. This test verifies that the error-free
        form (data=None at top level) parses cleanly as an _IQAirResponse sentinel.
        """
        from weewx_clearskies_api.providers.aqi.iqair import _IQAirResponse  # noqa: PLC0415
        # Top-level data=None is the cleanest error signal the model can represent
        error_payload_no_data = {"status": "fail"}
        response = _IQAirResponse.model_validate(error_payload_no_data)
        assert response.status == "fail"
        assert response.data is None

    def test_all_optional_pollution_fields_accept_none(self) -> None:
        """Optional pollution fields (aqius, mainus, aqicn, maincn) accept None."""
        from weewx_clearskies_api.providers.aqi.iqair import _IQAirPollution  # noqa: PLC0415
        p = _IQAirPollution.model_validate({
            "ts": "2019-04-08T18:00:00.000Z",
            "aqius": None,
            "mainus": None,
            "aqicn": None,
            "maincn": None,
        })
        assert p.aqius is None
        assert p.mainus is None


# ===========================================================================
# 2. _wire_to_canonical — Nashville happy path
# ===========================================================================


class TestWireToCanonicalNashvilleHappyPath:
    """_wire_to_canonical translates the Nashville fixture to correct canonical AQIReading."""

    def _get_data_from_fixture(self) -> Any:
        """Load Nashville fixture and return the parsed _IQAirData object."""
        from weewx_clearskies_api.providers.aqi.iqair import _IQAirResponse  # noqa: PLC0415
        raw = _load_fixture("iqair_nearest_city_nashville.json")
        response = _IQAirResponse.model_validate(raw)
        return response.data

    def test_nashville_fixture_produces_non_none_aqi_reading(self) -> None:
        """_wire_to_canonical returns AQIReading (not None) for the Nashville fixture."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None, (
            "_wire_to_canonical must return AQIReading for valid Nashville fixture"
        )

    def test_nashville_aqi_is_10(self) -> None:
        """aqi = 10 (aqius from pollution block — direct EPA AQI, no conversion needed)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.aqi == 10, f"Expected aqi=10, got {result.aqi!r}"

    def test_nashville_aqi_scale_is_epa(self) -> None:
        """aqiScale = 'epa' (aqius is EPA 0–500 native from IQAir)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.aqiScale == "epa", (
            f"Expected aqiScale='epa', got {result.aqiScale!r}"
        )

    def test_nashville_aqi_category_is_none(self) -> None:
        """aqiCategory = None (dashboard-computed; parsers set None)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.aqiCategory is None, (
            f"Expected aqiCategory=None (dashboard-computed), got {result.aqiCategory!r}"
        )

    def test_nashville_aqi_main_pollutant_is_pm25(self) -> None:
        """aqiMainPollutant = 'PM2.5' (mainus='p2' mapped via _MAINUS_TO_CANONICAL)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.aqiMainPollutant == "PM2.5", (
            f"Expected aqiMainPollutant='PM2.5' (p2→PM2.5), got {result.aqiMainPollutant!r}"
        )

    def test_nashville_aqi_location_is_city_comma_state(self) -> None:
        """aqiLocation = 'Nashville, Tennessee' (city + ', ' + state per LC4)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.aqiLocation == "Nashville, Tennessee", (
            f"Expected aqiLocation='Nashville, Tennessee', got {result.aqiLocation!r}"
        )

    def test_nashville_observed_at_is_utc_z_format(self) -> None:
        """observedAt ends with 'Z' (UTC ISO-8601, LC6 + ADR-020)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.observedAt is not None
        assert result.observedAt.endswith("Z"), (
            f"observedAt must end with Z, got {result.observedAt!r}"
        )

    def test_nashville_observed_at_drops_milliseconds(self) -> None:
        """observedAt = '2019-04-08T18:00:00Z' (millis dropped per ADR-020).

        IQAir pollution.ts = '2019-04-08T18:00:00.000Z'.
        After to_utc_iso8601_from_offset(): millis stripped → '2019-04-08T18:00:00Z'.
        """
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.observedAt == "2019-04-08T18:00:00Z", (
            f"Expected '2019-04-08T18:00:00Z', got {result.observedAt!r}"
        )

    def test_nashville_source_is_iqair(self) -> None:
        """source = 'iqair' (provider_id literal on AQIReading)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.source == "iqair", f"Expected source='iqair', got {result.source!r}"

    def test_nashville_pollutant_pm25_is_none_partial_domain(self) -> None:
        """pollutantPM25 = None (PARTIAL-DOMAIN — free tier has no concentrations)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.pollutantPM25 is None, (
            f"Expected pollutantPM25=None (PARTIAL-DOMAIN), got {result.pollutantPM25!r}"
        )

    def test_nashville_pollutant_pm10_is_none_partial_domain(self) -> None:
        """pollutantPM10 = None (PARTIAL-DOMAIN — free tier has no concentrations)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.pollutantPM10 is None

    def test_nashville_pollutant_o3_is_none_partial_domain(self) -> None:
        """pollutantO3 = None (PARTIAL-DOMAIN — free tier has no concentrations)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.pollutantO3 is None

    def test_nashville_pollutant_no2_is_none_partial_domain(self) -> None:
        """pollutantNO2 = None (PARTIAL-DOMAIN — free tier has no concentrations)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.pollutantNO2 is None

    def test_nashville_pollutant_so2_is_none_partial_domain(self) -> None:
        """pollutantSO2 = None (PARTIAL-DOMAIN — free tier has no concentrations)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.pollutantSO2 is None

    def test_nashville_pollutant_co_is_none_partial_domain(self) -> None:
        """pollutantCO = None (PARTIAL-DOMAIN — free tier has no concentrations)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._get_data_from_fixture()
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.pollutantCO is None


# ===========================================================================
# 3. _wire_to_canonical — edge cases
# ===========================================================================


class TestWireToCanonicalEdgeCases:
    """_wire_to_canonical edge cases: nulls, missing city/state, unknown codes."""

    def _make_data(
        self,
        aqius: int | None = 10,
        mainus: str | None = "p2",
        city: str | None = "Nashville",
        state: str | None = "Tennessee",
        ts: str = "2019-04-08T18:00:00.000Z",
    ) -> Any:
        """Build a minimal _IQAirData object from arguments."""
        from weewx_clearskies_api.providers.aqi.iqair import (  # noqa: PLC0415
            _IQAirCurrent,
            _IQAirData,
            _IQAirPollution,
            _IQAirWeather,
        )
        pollution = _IQAirPollution(ts=ts, aqius=aqius, mainus=mainus)
        weather = _IQAirWeather(ts=ts)
        current = _IQAirCurrent(weather=weather, pollution=pollution)
        return _IQAirData(city=city, state=state, country="USA", current=current)

    def test_all_null_pollution_aqius_returns_none(self) -> None:
        """_wire_to_canonical returns None when aqius=None (no AQI value to map)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._make_data(aqius=None, mainus=None)
        result = _wire_to_canonical(data)
        assert result is None, (
            "Expected None when aqius=None (no AQI value to map)"
        )

    def test_missing_city_returns_none_for_aqi_location(self) -> None:
        """aqiLocation = None when city is None (both city+state required per LC4)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._make_data(city=None, state="Tennessee")
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.aqiLocation is None, (
            f"Expected aqiLocation=None when city=None, got {result.aqiLocation!r}"
        )

    def test_missing_state_returns_none_for_aqi_location(self) -> None:
        """aqiLocation = None when state is None (both city+state required per LC4)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._make_data(city="Nashville", state=None)
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.aqiLocation is None, (
            f"Expected aqiLocation=None when state=None, got {result.aqiLocation!r}"
        )

    def test_mainus_none_returns_none_for_aqi_main_pollutant(self) -> None:
        """aqiMainPollutant = None when mainus is None."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._make_data(mainus=None)
        result = _wire_to_canonical(data)
        assert result is not None
        assert result.aqiMainPollutant is None, (
            f"Expected aqiMainPollutant=None when mainus=None, got {result.aqiMainPollutant!r}"
        )

    def test_unknown_mainus_code_returns_none_and_logs_info(self) -> None:
        """Unknown mainus code → aqiMainPollutant=None + logger.info notice (LC3)."""
        from weewx_clearskies_api.providers.aqi.iqair import _wire_to_canonical  # noqa: PLC0415
        data = self._make_data(mainus="unknown_code_xyz")
        with patch("weewx_clearskies_api.providers.aqi.iqair.logger") as mock_logger:
            result = _wire_to_canonical(data)
            # Must call logger.info with the unknown code in the message
            assert mock_logger.info.called, (
                "logger.info must be called for unknown mainus code"
            )
        assert result is not None
        assert result.aqiMainPollutant is None, (
            "Expected aqiMainPollutant=None for unknown mainus code"
        )


# ===========================================================================
# 4. Pollutant code lookup table — all six codes
# ===========================================================================


class TestMainusToCanonicalLookupTable:
    """_MAINUS_TO_CANONICAL maps all six known codes to correct canonical ids."""

    @pytest.mark.parametrize("mainus_code,expected_canonical", [
        ("p1", "PM10"),
        ("p2", "PM2.5"),
        ("n2", "NO2"),
        ("o3", "O3"),
        ("s2", "SO2"),
        ("co", "CO"),
    ])
    def test_mainus_code_maps_to_correct_canonical_id(
        self, mainus_code: str, expected_canonical: str
    ) -> None:
        """_MAINUS_TO_CANONICAL[code] == expected_canonical_id."""
        from weewx_clearskies_api.providers.aqi.iqair import _MAINUS_TO_CANONICAL  # noqa: PLC0415
        result = _MAINUS_TO_CANONICAL.get(mainus_code)
        assert result == expected_canonical, (
            f"Expected _MAINUS_TO_CANONICAL[{mainus_code!r}] = {expected_canonical!r}, "
            f"got {result!r}"
        )

    @pytest.mark.parametrize("mainus_code,expected_canonical", [
        ("p1", "PM10"),
        ("p2", "PM2.5"),
        ("n2", "NO2"),
        ("o3", "O3"),
        ("s2", "SO2"),
        ("co", "CO"),
    ])
    def test_wire_to_canonical_produces_correct_pollutant_for_each_mainus(
        self, mainus_code: str, expected_canonical: str
    ) -> None:
        """_wire_to_canonical produces correct aqiMainPollutant for each mainus code."""
        from weewx_clearskies_api.providers.aqi.iqair import (  # noqa: PLC0415
            _IQAirCurrent,
            _IQAirData,
            _IQAirPollution,
            _IQAirWeather,
            _wire_to_canonical,
        )
        pollution = _IQAirPollution(
            ts="2019-04-08T18:00:00.000Z",
            aqius=50,
            mainus=mainus_code,
        )
        weather = _IQAirWeather(ts="2019-04-08T19:00:00.000Z")
        current = _IQAirCurrent(weather=weather, pollution=pollution)
        data = _IQAirData(city="TestCity", state="TestState", country="USA", current=current)

        result = _wire_to_canonical(data)
        assert result is not None
        assert result.aqiMainPollutant == expected_canonical, (
            f"mainus={mainus_code!r} → expected aqiMainPollutant={expected_canonical!r}, "
            f"got {result.aqiMainPollutant!r}"
        )

    def test_mainus_table_has_exactly_six_entries(self) -> None:
        """_MAINUS_TO_CANONICAL has exactly 6 entries (p1,p2,n2,o3,s2,co)."""
        from weewx_clearskies_api.providers.aqi.iqair import _MAINUS_TO_CANONICAL  # noqa: PLC0415
        assert len(_MAINUS_TO_CANONICAL) == 6, (
            f"Expected 6 entries in _MAINUS_TO_CANONICAL, got {len(_MAINUS_TO_CANONICAL)}"
        )


# ===========================================================================
# 5. Envelope error mapping (LC27 / LC12)
# ===========================================================================


class TestEnvelopeErrorMapping:
    """status='fail' message strings dispatch to correct canonical exception classes.

    Tests target _raise_for_envelope_error() directly rather than going through
    fetch(), which avoids cache-state dependencies from earlier tests in the run.
    The fetch()-level integration of the envelope error dispatch is covered by the
    integration test suite.
    """

    @pytest.mark.parametrize("error_message", [
        "incorrect_api_key",
        "api_key_expired",
        "payment required",
        "permission_denied",
        "forbidden",
        "feature_not_available",
    ])
    def test_key_invalid_messages_raise_key_invalid(self, error_message: str) -> None:
        """status='fail' + auth/expired/permission message → KeyInvalid."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.iqair import (
            _raise_for_envelope_error,  # noqa: PLC0415
        )
        with pytest.raises(KeyInvalid):
            _raise_for_envelope_error(error_message)

    @pytest.mark.parametrize("error_message", [
        "call_limit_reached",
        "too_many_requests",
    ])
    def test_quota_messages_raise_quota_exhausted(self, error_message: str) -> None:
        """status='fail' + rate-limit message → QuotaExhausted(retry_after_seconds=None)."""
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.iqair import (
            _raise_for_envelope_error,  # noqa: PLC0415
        )
        with pytest.raises(QuotaExhausted) as exc_info:
            _raise_for_envelope_error(error_message)
        assert exc_info.value.retry_after_seconds is None, (
            "IQAir envelope errors don't include Retry-After; must be None"
        )

    @pytest.mark.parametrize("error_message", [
        "city_not_found",
        "no_nearest_station",
        "node not found",
    ])
    def test_geographic_error_messages_raise_provider_protocol_error(
        self, error_message: str
    ) -> None:
        """status='fail' + geographic/not-found message → ProviderProtocolError."""
        from weewx_clearskies_api.providers._common.errors import (
            ProviderProtocolError,  # noqa: PLC0415
        )
        from weewx_clearskies_api.providers.aqi.iqair import (
            _raise_for_envelope_error,  # noqa: PLC0415
        )
        with pytest.raises(ProviderProtocolError):
            _raise_for_envelope_error(error_message)

    def test_unknown_fail_message_raises_provider_protocol_error(self) -> None:
        """status='fail' + unknown message → ProviderProtocolError (defensive default)."""
        from weewx_clearskies_api.providers._common.errors import (
            ProviderProtocolError,  # noqa: PLC0415
        )
        from weewx_clearskies_api.providers.aqi.iqair import (
            _raise_for_envelope_error,  # noqa: PLC0415
        )
        with pytest.raises(ProviderProtocolError):
            _raise_for_envelope_error("some_unknown_error_not_in_table")

    def test_envelope_error_dispatch_via_fetch_incorrect_api_key_raises_key_invalid(
        self,
    ) -> None:
        """200+fail envelope with `incorrect_api_key` → KeyInvalid (LC12/LC27 end-to-end).

        Locks in the b02c6ce parse-order fix: fetch() must check `status` from raw JSON
        BEFORE Pydantic validation, then dispatch on `data.message` to the canonical
        taxonomy. If a future refactor reintroduces Pydantic-first ordering this test
        will fail (ProviderProtocolError leaks instead of KeyInvalid).
        """
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.iqair import fetch  # noqa: PLC0415

        lat, lon = 0.0001, 0.0001
        _reset_provider_state()
        payload = {"status": "fail", "data": {"message": "incorrect_api_key"}}
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_IQAIR_NEAREST_CITY_URL).mock(
                return_value=httpx.Response(200, json=payload)
            )
            with pytest.raises(KeyInvalid):
                fetch(lat=lat, lon=lon, key=_TEST_KEY)
        _reset_provider_state()

    def test_envelope_error_dispatch_via_fetch_call_limit_raises_quota_exhausted(
        self,
    ) -> None:
        """200+fail envelope with `call_limit_reached` → QuotaExhausted (LC12/LC27 end-to-end).

        Parallel to the KeyInvalid dispatch test above; locks in the b02c6ce fix for
        the rate-limit branch of `_raise_for_envelope_error`. retry_after_seconds=None
        because IQAir's 200-not-429 envelope path doesn't carry a Retry-After header.
        """
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.iqair import fetch  # noqa: PLC0415

        lat, lon = 0.0002, 0.0002
        _reset_provider_state()
        payload = {"status": "fail", "data": {"message": "call_limit_reached"}}
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_IQAIR_NEAREST_CITY_URL).mock(
                return_value=httpx.Response(200, json=payload)
            )
            with pytest.raises(QuotaExhausted) as exc_info:
                fetch(lat=lat, lon=lon, key=_TEST_KEY)
            assert exc_info.value.retry_after_seconds is None
        _reset_provider_state()


# ===========================================================================
# 6. Pre-call key validation (LC13)
# ===========================================================================


class TestPreCallKeyValidation:
    """Empty/None key raises KeyInvalid BEFORE any HTTP call is made."""

    def test_empty_string_key_raises_key_invalid_before_http(self) -> None:
        """fetch(key='') → KeyInvalid without making any HTTP call."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.iqair import fetch  # noqa: PLC0415
        _reset_provider_state()

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_IQAIR_NEAREST_CITY_URL).mock(
                return_value=httpx.Response(200, json={})
            )
            with pytest.raises(KeyInvalid):
                fetch(lat=_LAT_NASHVILLE, lon=_LON_NASHVILLE, key="")
            assert mock.calls.call_count == 0, (
                "No HTTP call should be made when key is empty"
            )

    def test_none_key_raises_key_invalid_before_http(self) -> None:
        """fetch(key=None) → KeyInvalid without making any HTTP call."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.iqair import fetch  # noqa: PLC0415
        _reset_provider_state()

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_IQAIR_NEAREST_CITY_URL).mock(
                return_value=httpx.Response(200, json={})
            )
            with pytest.raises(KeyInvalid):
                fetch(lat=_LAT_NASHVILLE, lon=_LON_NASHVILLE, key=None)  # type: ignore[arg-type]
            assert mock.calls.call_count == 0, (
                "No HTTP call should be made when key is None"
            )

    def test_key_invalid_message_names_env_var(self) -> None:
        """KeyInvalid raised on empty key names WEEWX_CLEARSKIES_IQAIR_KEY env var.

        Operators reading the error must learn the env var name without
        cracking open the source. (Merged from F2-deleted FLAT test file 2026-05-11.)
        """
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.iqair import fetch  # noqa: PLC0415
        _reset_provider_state()
        with pytest.raises(KeyInvalid) as exc_info:
            fetch(lat=_LAT_NASHVILLE, lon=_LON_NASHVILLE, key="")
        assert "WEEWX_CLEARSKIES_IQAIR_KEY" in str(exc_info.value), (
            "KeyInvalid message must name the env var so operator knows how to fix it"
        )


# ===========================================================================
# 6.5. EPA category band parametrized coverage (merged from F2-deleted FLAT file 2026-05-11)
# ===========================================================================


class TestCategoryBandsParametrized:
    """epa_category(aqius) bands beyond the Nashville aqi=10 'Good' fixture.

    Locks in LC1 derivation (`aqiCategory` from `aqius` via EPA bands) for
    Moderate / Unhealthy-Sensitive / Hazardous bands. Below 'Good' is covered
    by the Nashville happy-path fixture; above 'Hazardous' (>500) is defensive
    cap behavior tested directly in `tests/providers/aqi/test_units.py`.
    """

    @pytest.mark.parametrize(
        ("aqius", "expected_category"),
        [
            (75, "Moderate"),
            (125, "Unhealthy for Sensitive Groups"),
            (400, "Hazardous"),
        ],
    )
    def test_category_band_for_aqius_value(
        self,
        aqius: int,
        expected_category: str,
    ) -> None:
        from weewx_clearskies_api.providers.aqi.iqair import (  # noqa: PLC0415
            _IQAirData,
            _wire_to_canonical,
        )
        raw = {
            "city": "TestCity",
            "state": "TestState",
            "country": "USA",
            "current": {
                "weather": {"ts": "2019-04-08T19:00:00.000Z"},
                "pollution": {
                    "ts": "2019-04-08T18:00:00.000Z",
                    "aqius": aqius,
                    "mainus": "p2",
                },
            },
        }
        data = _IQAirData.model_validate(raw)
        reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.aqiCategory == expected_category


# ===========================================================================
# 7. Cache 3-way path: miss → hit, sentinel
# ===========================================================================


class TestCacheThreeWayPath:
    """Cache miss → HTTP call → cached; cache hit → no HTTP call; sentinel → None."""

    def test_cache_miss_makes_http_call_and_returns_reading(self) -> None:
        """Cache miss → 1 HTTP call → canonical AQIReading returned and cached."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.iqair import (  # noqa: PLC0415
            _build_cache_key,
            fetch,
        )
        _reset_provider_state()

        data = _load_fixture("iqair_nearest_city_nashville.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_IQAIR_NEAREST_CITY_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            reading = fetch(
                lat=_LAT_NASHVILLE,
                lon=_LON_NASHVILLE,
                key=_TEST_KEY,
            )
            call_count = mock.calls.call_count

        assert call_count == 1, f"Expected 1 HTTP call on cache miss, got {call_count}"
        assert reading is not None
        assert reading.source == "iqair"
        assert reading.aqi == 10

        # Verify reading is cached
        cache_key = _build_cache_key(_LAT_NASHVILLE, _LON_NASHVILLE)
        cached = get_cache().get(cache_key)
        assert cached is not None, "Reading must be cached after cache miss"
        _reset_provider_state()

    def test_cache_hit_skips_http_call(self) -> None:
        """Cache hit → 0 HTTP calls; cached AQIReading returned."""
        from weewx_clearskies_api.providers.aqi.iqair import fetch  # noqa: PLC0415
        _reset_provider_state()

        data = _load_fixture("iqair_nearest_city_nashville.json")

        # First fetch — fills cache
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_IQAIR_NEAREST_CITY_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            reading1 = fetch(lat=_LAT_NASHVILLE, lon=_LON_NASHVILLE, key=_TEST_KEY)

        # Second fetch — must come from cache
        with respx.mock(assert_all_called=False) as mock2:
            reading2 = fetch(lat=_LAT_NASHVILLE, lon=_LON_NASHVILLE, key=_TEST_KEY)
            cache_hit_calls = mock2.calls.call_count

        assert cache_hit_calls == 0, (
            f"Expected 0 HTTP calls on cache hit, got {cache_hit_calls}"
        )
        assert reading1 is not None and reading2 is not None
        assert reading1.aqi == reading2.aqi
        assert reading1.source == reading2.source
        _reset_provider_state()

    def test_sentinel_in_cache_returns_none_without_http_call(self) -> None:
        """Cache hit with _no_reading sentinel → None returned; 0 HTTP calls."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.iqair import (  # noqa: PLC0415
            _build_cache_key,
            fetch,
        )
        _reset_provider_state()

        # Manually inject sentinel (kwarg name is ttl_seconds, not ttl)
        cache_key = _build_cache_key(_LAT_NASHVILLE, _LON_NASHVILLE)
        get_cache().set(cache_key, {"_no_reading": True}, ttl_seconds=900)

        with respx.mock(assert_all_called=False) as mock:
            result = fetch(lat=_LAT_NASHVILLE, lon=_LON_NASHVILLE, key=_TEST_KEY)
            assert mock.calls.call_count == 0, (
                "No HTTP call should be made when sentinel is cached"
            )

        assert result is None, f"Expected None from sentinel cache, got {result!r}"
        _reset_provider_state()

    def test_wire_validation_failure_raises_provider_protocol_error(self) -> None:
        """Cache miss + malformed JSON (missing required status) → ProviderProtocolError."""
        from weewx_clearskies_api.providers._common.errors import (
            ProviderProtocolError,  # noqa: PLC0415
        )
        from weewx_clearskies_api.providers.aqi.iqair import fetch  # noqa: PLC0415
        _reset_provider_state()

        # Response with missing required 'status' field
        malformed = {"data": {"city": "Nashville"}}  # missing status

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_IQAIR_NEAREST_CITY_URL).mock(
                return_value=httpx.Response(200, json=malformed)
            )
            with pytest.raises(ProviderProtocolError):
                fetch(lat=_LAT_NASHVILLE, lon=_LON_NASHVILLE, key=_TEST_KEY)
        _reset_provider_state()


# ===========================================================================
# 8. Cache key properties (LC9)
# ===========================================================================


class TestCacheKeyProperties:
    """_build_cache_key produces correct, privacy-safe keys."""

    def test_same_lat_lon_produces_same_key(self) -> None:
        """Same lat/lon → identical cache key (deterministic)."""
        from weewx_clearskies_api.providers.aqi.iqair import _build_cache_key  # noqa: PLC0415
        key1 = _build_cache_key(_LAT, _LON)
        key2 = _build_cache_key(_LAT, _LON)
        assert key1 == key2, "Cache key must be deterministic for same lat/lon"

    def test_different_lat_lon_produces_different_key(self) -> None:
        """Different lat/lon → different cache key."""
        from weewx_clearskies_api.providers.aqi.iqair import _build_cache_key  # noqa: PLC0415
        key1 = _build_cache_key(_LAT, _LON)
        key2 = _build_cache_key(_LAT + 1.0, _LON + 1.0)
        assert key1 != key2, "Different lat/lon must produce different cache keys"

    def test_cache_key_is_64_char_hex_sha256(self) -> None:
        """Cache key is a 64-character hexadecimal SHA-256 string."""
        from weewx_clearskies_api.providers.aqi.iqair import _build_cache_key  # noqa: PLC0415
        key = _build_cache_key(_LAT, _LON)
        assert len(key) == 64, f"SHA-256 hex key must be 64 chars, got {len(key)}"
        assert all(c in "0123456789abcdef" for c in key), (
            f"Cache key must be lowercase hex, got {key!r}"
        )

    def test_cache_key_uses_4dp_lat_lon(self) -> None:
        """Lat/lon rounded to 4dp — nearby coords within 4dp share cache key."""
        from weewx_clearskies_api.providers.aqi.iqair import _build_cache_key  # noqa: PLC0415
        # These two differ only at 5th decimal place — must hash to same key
        key1 = _build_cache_key(42.29930, -72.45090)
        key2 = _build_cache_key(42.29934, -72.45094)
        assert key1 == key2, (
            "Coords differing only at 5th decimal place should share cache key (4dp rounding)"
        )

    def test_cache_key_does_not_contain_api_key(self) -> None:
        """API key value must NOT appear in the cache key (privacy/leakage per LC9)."""
        # _build_cache_key takes only lat/lon — verifying signature does not accept key
        import inspect  # noqa: PLC0415

        from weewx_clearskies_api.providers.aqi.iqair import _build_cache_key  # noqa: PLC0415
        sig = inspect.signature(_build_cache_key)
        param_names = list(sig.parameters.keys())
        assert "key" not in param_names, (
            "_build_cache_key must not accept API key param (privacy/leakage per LC9)"
        )

    def test_iqair_aqi_cache_key_distinct_from_aeris_aqi_key_at_same_coords(self) -> None:
        """IQAir cache key differs from Aeris cache key at same lat/lon (different provider_id)."""
        from weewx_clearskies_api.providers.aqi.aeris import (
            _build_cache_key as aeris_key,  # noqa: PLC0415
        )
        from weewx_clearskies_api.providers.aqi.iqair import (
            _build_cache_key as iqair_key,  # noqa: PLC0415
        )
        k1 = iqair_key(_LAT, _LON)
        k2 = aeris_key(_LAT, _LON)
        assert k1 != k2, (
            "IQAir and Aeris must produce different cache keys at same coordinates"
        )


# ===========================================================================
# 9. Rate limiter configuration (LC10)
# ===========================================================================


class TestRateLimiterConfiguration:
    """RateLimiter configured with correct name, provider_id, max_calls, window."""

    def test_rate_limiter_max_calls_is_5(self) -> None:
        """Rate limiter _max_calls=5 (IQAir Community per-minute cap)."""
        import weewx_clearskies_api.providers.aqi.iqair as _iqair  # noqa: PLC0415
        assert _iqair._rate_limiter._max_calls == 5, (
            f"Expected _max_calls=5 (IQAir per-minute cap), got {_iqair._rate_limiter._max_calls!r}"
        )

    def test_rate_limiter_window_seconds_is_60(self) -> None:
        """Rate limiter _window_seconds=60 (per-minute, not per-second like OWM/Aeris)."""
        import weewx_clearskies_api.providers.aqi.iqair as _iqair  # noqa: PLC0415
        assert _iqair._rate_limiter._window_seconds == 60, (
            f"Expected _window_seconds=60, got {_iqair._rate_limiter._window_seconds!r}"
        )

    def test_rate_limiter_name_contains_iqair(self) -> None:
        """Rate limiter _name contains 'iqair' for correct attribution in logs."""
        import weewx_clearskies_api.providers.aqi.iqair as _iqair  # noqa: PLC0415
        assert "iqair" in _iqair._rate_limiter._name, (
            f"Rate limiter _name must contain 'iqair', got {_iqair._rate_limiter._name!r}"
        )


# ===========================================================================
# 10. Capability declaration
# ===========================================================================


class TestCapabilityDeclaration:
    """CAPABILITY symbol correct shape, domain, auth, fields."""

    def test_capability_provider_id_is_iqair(self) -> None:
        """CAPABILITY.provider_id = 'iqair'."""
        from weewx_clearskies_api.providers.aqi.iqair import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.provider_id == "iqair"

    def test_capability_domain_is_aqi(self) -> None:
        """CAPABILITY.domain = 'aqi'."""
        from weewx_clearskies_api.providers.aqi.iqair import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.domain == "aqi"

    def test_capability_auth_required_is_key_tuple(self) -> None:
        """CAPABILITY.auth_required = ('key',) (single query-param credential, LC8)."""
        from weewx_clearskies_api.providers.aqi.iqair import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.auth_required == ("key",), (
            f"Expected auth_required=('key',), got {CAPABILITY.auth_required!r}"
        )

    def test_capability_geographic_coverage_is_global(self) -> None:
        """CAPABILITY.geographic_coverage = 'global'."""
        from weewx_clearskies_api.providers.aqi.iqair import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.geographic_coverage == "global"

    def test_capability_default_poll_interval_is_900s(self) -> None:
        """CAPABILITY.default_poll_interval_seconds = 900 (15 min per ADR-017)."""
        from weewx_clearskies_api.providers.aqi.iqair import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.default_poll_interval_seconds == 900, (
            f"Expected 900s poll interval, got {CAPABILITY.default_poll_interval_seconds!r}"
        )

    def test_capability_supplied_fields_includes_six_free_tier_fields(self) -> None:
        """CAPABILITY.supplied_canonical_fields has all 6 verified free-tier fields (LC7)."""
        from weewx_clearskies_api.providers.aqi.iqair import CAPABILITY  # noqa: PLC0415
        expected_fields = {
            "aqi", "aqiCategory", "aqiMainPollutant", "aqiLocation", "observedAt", "source"
        }
        supplied = set(CAPABILITY.supplied_canonical_fields)
        missing = expected_fields - supplied
        assert not missing, (
            f"CAPABILITY missing expected free-tier fields: {missing!r}"
        )

    def test_capability_supplied_fields_excludes_pollutant_concentrations(self) -> None:
        """CAPABILITY.supplied_canonical_fields excludes all concentration fields (PARTIAL-DOMAIN)."""
        from weewx_clearskies_api.providers.aqi.iqair import CAPABILITY  # noqa: PLC0415
        concentration_fields = {
            "pollutantPM25", "pollutantPM10",
            "pollutantO3", "pollutantNO2", "pollutantSO2", "pollutantCO",
        }
        supplied = set(CAPABILITY.supplied_canonical_fields)
        overlap = concentration_fields & supplied
        assert not overlap, (
            f"CAPABILITY must NOT include concentration fields (PARTIAL-DOMAIN free tier): {overlap!r}"
        )

    def test_capability_supplied_fields_has_exactly_six_entries(self) -> None:
        """CAPABILITY.supplied_canonical_fields has exactly 6 entries (conservative per Q2 decision)."""
        from weewx_clearskies_api.providers.aqi.iqair import CAPABILITY  # noqa: PLC0415
        count = len(CAPABILITY.supplied_canonical_fields)
        assert count == 6, (
            f"Expected exactly 6 CAPABILITY fields (free-tier conservative), got {count}"
        )

    def test_wire_providers_registers_iqair_aqi_in_registry(self) -> None:
        """wire_providers([iqair.CAPABILITY]) → ('aqi', 'iqair') in registry."""
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            get_provider_registry,
            reset_provider_registry_for_tests,
            wire_providers,
        )
        from weewx_clearskies_api.providers.aqi.iqair import CAPABILITY  # noqa: PLC0415

        reset_provider_registry_for_tests()
        wire_providers([CAPABILITY])
        registry = get_provider_registry()
        assert any(
            p.provider_id == "iqair" and p.domain == "aqi" for p in registry
        ), "wire_providers must register iqair aqi in registry"
        reset_provider_registry_for_tests()


# ===========================================================================
# 11. HTTP error propagation (L2 carry-forward — bare canonical taxonomy)
# ===========================================================================


class TestHttpErrorPropagation:
    """HTTP-level errors propagate as canonical exceptions (L2 carry-forward)."""

    def test_http_401_raises_key_invalid(self) -> None:
        """Provider HTTP 401 → KeyInvalid (L2 bare propagation from ProviderHTTPClient)."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.iqair import fetch  # noqa: PLC0415
        _reset_provider_state()

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_IQAIR_NEAREST_CITY_URL).mock(
                return_value=httpx.Response(401, json={"message": "Unauthorized"})
            )
            with pytest.raises(KeyInvalid):
                fetch(lat=_LAT_NASHVILLE, lon=_LON_NASHVILLE, key=_TEST_KEY)
        _reset_provider_state()

    def test_http_403_raises_key_invalid(self) -> None:
        """Provider HTTP 403 → KeyInvalid (L2 bare propagation)."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.iqair import fetch  # noqa: PLC0415
        _reset_provider_state()

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_IQAIR_NEAREST_CITY_URL).mock(
                return_value=httpx.Response(403, json={"message": "Forbidden"})
            )
            with pytest.raises(KeyInvalid):
                fetch(lat=_LAT_NASHVILLE, lon=_LON_NASHVILLE, key=_TEST_KEY)
        _reset_provider_state()

    def test_http_429_raises_quota_exhausted(self) -> None:
        """Provider HTTP 429 → QuotaExhausted (L2 bare propagation)."""
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.iqair import fetch  # noqa: PLC0415
        _reset_provider_state()

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_IQAIR_NEAREST_CITY_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={"message": "Too Many Requests"},
                    headers={"Retry-After": "60"},
                )
            )
            with pytest.raises(QuotaExhausted):
                fetch(lat=_LAT_NASHVILLE, lon=_LON_NASHVILLE, key=_TEST_KEY)
        _reset_provider_state()

    def test_http_5xx_raises_transient_network_error(self) -> None:
        """Provider HTTP 5xx → TransientNetworkError (L2 bare propagation)."""
        from weewx_clearskies_api.providers._common.errors import (
            TransientNetworkError,  # noqa: PLC0415
        )
        from weewx_clearskies_api.providers.aqi.iqair import fetch  # noqa: PLC0415
        _reset_provider_state()

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_IQAIR_NEAREST_CITY_URL).mock(
                return_value=httpx.Response(500, json={"reason": "server error"})
            )
            with pytest.raises(TransientNetworkError):
                fetch(lat=_LAT_NASHVILLE, lon=_LON_NASHVILLE, key=_TEST_KEY)
        _reset_provider_state()
