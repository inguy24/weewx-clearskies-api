"""Unit tests for the IQAir AirVisual AQI provider module (3b-12).

Tests are fully isolated — no network calls, no Redis, no MariaDB.
Cache backend uses memory:// (wire_cache_from_env default when no env var is set).

Coverage:
  - Wire-shape validation: extra="ignore" drops unknown keys; required fields enforced.
  - _wire_to_canonical translation:
      Nashville fixture (aqius=10, mainus=p2) → full AQIReading.
      All-null pollution block (aqius=None) → None return.
      Missing city OR state → aqiLocation = None.
      Each mainus code in _MAINUS_TO_CANONICAL → correct canonical id.
      Unknown mainus code → aqiMainPollutant = None + logger.info notice.
  - LC12/LC27 envelope error mapping:
      status=fail + incorrect_api_key → KeyInvalid.
      status=fail + call_limit_reached → QuotaExhausted(retry_after_seconds=None).
      status=fail + city_not_found → ProviderProtocolError.
      Each known error message string mapped correctly.
  - LC13 pre-call key guard: empty/None key → KeyInvalid before HTTP.
  - Cache hit/miss/sentinel 3-way path.
  - Cache key: credentials NOT in key; lat/lon rounded to 4 decimals; deterministic.
  - Rate limiter: configured with max_calls=5, window_seconds=60 (per-minute cap).
"""

from __future__ import annotations

import hashlib
import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from weewx_clearskies_api.providers._common.errors import (
    KeyInvalid,
    ProviderProtocolError,
    QuotaExhausted,
)
from weewx_clearskies_api.providers.aqi import iqair
from weewx_clearskies_api.providers.aqi.iqair import (
    CAPABILITY,
    PROVIDER_ID,
    _IQAirCurrent,
    _IQAirData,
    _IQAirPollution,
    _IQAirResponse,
    _IQAirWeather,
    _MAINUS_TO_CANONICAL,
    _build_cache_key,
    _raise_for_envelope_error,
    _wire_to_canonical,
    fetch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pollution(**overrides: object) -> _IQAirPollution:
    """Build a valid _IQAirPollution with Nashville defaults, applying overrides."""
    defaults = {
        "ts": "2019-04-08T18:00:00.000Z",
        "aqius": 10,
        "mainus": "p2",
        "aqicn": 3,
        "maincn": "p2",
    }
    defaults.update(overrides)  # type: ignore[arg-type]
    return _IQAirPollution.model_validate(defaults)


def _make_data(**overrides: object) -> _IQAirData:
    """Build a valid _IQAirData with Nashville defaults, applying overrides."""
    defaults = {
        "city": "Nashville",
        "state": "Tennessee",
        "country": "USA",
        "current": {
            "weather": {"ts": "2019-04-08T19:00:00.000Z"},
            "pollution": {
                "ts": "2019-04-08T18:00:00.000Z",
                "aqius": 10,
                "mainus": "p2",
                "aqicn": 3,
                "maincn": "p2",
            },
        },
    }
    defaults.update(overrides)  # type: ignore[arg-type]
    return _IQAirData.model_validate(defaults)


# ---------------------------------------------------------------------------
# 1. Wire-shape model tests
# ---------------------------------------------------------------------------


class TestIQAirPydanticModels:
    """Wire-shape validation — extra fields ignored, required fields enforced."""

    def test_pollution_extra_fields_ignored(self) -> None:
        """extra='ignore' on _IQAirPollution — paid-tier concentration fields dropped."""
        raw = {
            "ts": "2019-04-08T18:00:00.000Z",
            "aqius": 10,
            "mainus": "p2",
            "aqicn": 3,
            "maincn": "p2",
            # Hypothetical paid-tier fields that should be dropped
            "pm25": 5.2,
            "pm10": 12.1,
            "o3": 0.031,
        }
        p = _IQAirPollution.model_validate(raw)
        assert p.aqius == 10
        assert p.mainus == "p2"
        # No pm25/pm10/o3 attrs should be present (extra="ignore")
        assert not hasattr(p, "pm25")
        assert not hasattr(p, "pm10")

    def test_pollution_ts_is_required(self) -> None:
        """_IQAirPollution requires ts field (ValidationError if missing)."""
        from pydantic import ValidationError  # noqa: PLC0415
        with pytest.raises(ValidationError):
            _IQAirPollution.model_validate({"aqius": 10, "mainus": "p2"})

    def test_pollution_optional_fields_default_to_none(self) -> None:
        """aqius, mainus, aqicn, maincn are all optional → None when absent."""
        p = _IQAirPollution.model_validate({"ts": "2019-04-08T18:00:00.000Z"})
        assert p.aqius is None
        assert p.mainus is None
        assert p.aqicn is None
        assert p.maincn is None

    def test_weather_extra_fields_ignored(self) -> None:
        """extra='ignore' on _IQAirWeather — all weather fields dropped except ts."""
        raw = {
            "ts": "2019-04-08T19:00:00.000Z",
            "tp": 18,
            "hu": 88,
            "pr": 1012,
            "wd": 90,
            "ws": 3.1,
            "ic": "04d",
        }
        w = _IQAirWeather.model_validate(raw)
        assert w.ts == "2019-04-08T19:00:00.000Z"
        assert not hasattr(w, "tp")

    def test_data_extra_fields_ignored(self) -> None:
        """extra='ignore' on _IQAirData — location/extra fields dropped."""
        raw = {
            "city": "Nashville",
            "state": "Tennessee",
            "country": "USA",
            "location": {"type": "Point", "coordinates": [-86.7386, 36.1767]},
            "current": {
                "weather": {"ts": "2019-04-08T19:00:00.000Z"},
                "pollution": {"ts": "2019-04-08T18:00:00.000Z", "aqius": 10, "mainus": "p2"},
            },
        }
        d = _IQAirData.model_validate(raw)
        assert d.city == "Nashville"
        assert d.state == "Tennessee"
        assert not hasattr(d, "location")

    def test_response_envelope_success_parses(self) -> None:
        """_IQAirResponse parses a full success envelope from Nashville fixture."""
        raw = {
            "status": "success",
            "data": {
                "city": "Nashville",
                "state": "Tennessee",
                "country": "USA",
                "current": {
                    "weather": {"ts": "2019-04-08T19:00:00.000Z"},
                    "pollution": {
                        "ts": "2019-04-08T18:00:00.000Z",
                        "aqius": 10,
                        "mainus": "p2",
                        "aqicn": 3,
                        "maincn": "p2",
                    },
                },
            },
        }
        r = _IQAirResponse.model_validate(raw)
        assert r.status == "success"
        assert r.data is not None
        assert r.data.city == "Nashville"
        assert r.data.current.pollution.aqius == 10


# ---------------------------------------------------------------------------
# 2. Translation tests (_wire_to_canonical)
# ---------------------------------------------------------------------------


class TestWireToCanonical:
    """_wire_to_canonical translation contract."""

    def test_nashville_fixture_aqi_value(self) -> None:
        """Nashville fixture: aqi = 10 (aqius passthrough, no conversion)."""
        data = _make_data()
        reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.aqi == 10

    def test_nashville_fixture_category_is_good(self) -> None:
        """Nashville fixture: aqiCategory = 'Good' (AQI 10 → EPA 0–50 band)."""
        data = _make_data()
        reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.aqiCategory == "Good"

    def test_nashville_fixture_main_pollutant_is_pm25(self) -> None:
        """Nashville fixture: aqiMainPollutant = 'PM2.5' (mainus='p2' lookup)."""
        data = _make_data()
        reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.aqiMainPollutant == "PM2.5"

    def test_nashville_fixture_location_is_nashville_tennessee(self) -> None:
        """Nashville fixture: aqiLocation = 'Nashville, Tennessee' (LC4 comma+space)."""
        data = _make_data()
        reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.aqiLocation == "Nashville, Tennessee"

    def test_nashville_fixture_observed_at_utc_z(self) -> None:
        """Nashville fixture: observedAt = '2019-04-08T18:00:00Z' (millis dropped, Z suffix)."""
        data = _make_data()
        reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.observedAt == "2019-04-08T18:00:00Z"

    def test_nashville_fixture_source_is_iqair(self) -> None:
        """Nashville fixture: source = 'iqair'."""
        data = _make_data()
        reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.source == "iqair"

    def test_nashville_all_pollutant_fields_are_none(self) -> None:
        """All pollutant* fields = None (PARTIAL-DOMAIN on free Community tier, LC5)."""
        data = _make_data()
        reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.pollutantPM25 is None
        assert reading.pollutantPM10 is None
        assert reading.pollutantO3 is None
        assert reading.pollutantNO2 is None
        assert reading.pollutantSO2 is None
        assert reading.pollutantCO is None

    def test_null_aqius_returns_none(self) -> None:
        """All-null pollution block (aqius=None) → _wire_to_canonical returns None."""
        raw = {
            "city": "Nashville",
            "state": "Tennessee",
            "country": "USA",
            "current": {
                "weather": {"ts": "2019-04-08T19:00:00.000Z"},
                "pollution": {
                    "ts": "2019-04-08T18:00:00.000Z",
                    # aqius absent → defaults to None
                },
            },
        }
        data = _IQAirData.model_validate(raw)
        reading = _wire_to_canonical(data)
        assert reading is None, "aqius=None must produce None return from _wire_to_canonical"

    def test_missing_city_produces_none_location(self) -> None:
        """Missing city → aqiLocation = None (don't emit partial location strings)."""
        raw = {
            "city": None,
            "state": "Tennessee",
            "country": "USA",
            "current": {
                "weather": {"ts": "2019-04-08T19:00:00.000Z"},
                "pollution": {
                    "ts": "2019-04-08T18:00:00.000Z",
                    "aqius": 10,
                    "mainus": "p2",
                },
            },
        }
        data = _IQAirData.model_validate(raw)
        reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.aqiLocation is None, "Missing city must produce aqiLocation=None"

    def test_missing_state_produces_none_location(self) -> None:
        """Missing state → aqiLocation = None (don't emit partial location strings)."""
        raw = {
            "city": "Nashville",
            "state": None,
            "country": "USA",
            "current": {
                "weather": {"ts": "2019-04-08T19:00:00.000Z"},
                "pollution": {
                    "ts": "2019-04-08T18:00:00.000Z",
                    "aqius": 10,
                    "mainus": "p2",
                },
            },
        }
        data = _IQAirData.model_validate(raw)
        reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.aqiLocation is None, "Missing state must produce aqiLocation=None"

    @pytest.mark.parametrize("code,expected", [
        ("p1", "PM10"),
        ("p2", "PM2.5"),
        ("n2", "NO2"),
        ("o3", "O3"),
        ("s2", "SO2"),
        ("co", "CO"),
    ])
    def test_mainus_code_lookup(self, code: str, expected: str) -> None:
        """Each mainus code in _MAINUS_TO_CANONICAL → correct canonical pollutant id."""
        raw = {
            "city": "TestCity",
            "state": "TestState",
            "country": "USA",
            "current": {
                "weather": {"ts": "2019-04-08T19:00:00.000Z"},
                "pollution": {
                    "ts": "2019-04-08T18:00:00.000Z",
                    "aqius": 50,
                    "mainus": code,
                },
            },
        }
        data = _IQAirData.model_validate(raw)
        reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.aqiMainPollutant == expected, (
            f"mainus={code!r} must map to {expected!r}, got {reading.aqiMainPollutant!r}"
        )

    def test_unknown_mainus_code_returns_none_and_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown mainus code → aqiMainPollutant = None + logger.info notice (LC3)."""
        raw = {
            "city": "TestCity",
            "state": "TestState",
            "country": "USA",
            "current": {
                "weather": {"ts": "2019-04-08T19:00:00.000Z"},
                "pollution": {
                    "ts": "2019-04-08T18:00:00.000Z",
                    "aqius": 50,
                    "mainus": "p4_unknown",  # not in _MAINUS_TO_CANONICAL
                },
            },
        }
        data = _IQAirData.model_validate(raw)
        with caplog.at_level(logging.INFO, logger="weewx_clearskies_api.providers.aqi.iqair"):
            reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.aqiMainPollutant is None, (
            "Unknown mainus code must produce aqiMainPollutant=None"
        )
        assert any("p4_unknown" in r.message for r in caplog.records), (
            "Unknown mainus code must log an INFO notice with the code value"
        )

    def test_category_bands_moderate(self) -> None:
        """AQI 75 → aqiCategory = 'Moderate' (51–100 EPA band)."""
        raw = {
            "city": "TestCity",
            "state": "TestState",
            "country": "USA",
            "current": {
                "weather": {"ts": "2019-04-08T19:00:00.000Z"},
                "pollution": {
                    "ts": "2019-04-08T18:00:00.000Z",
                    "aqius": 75,
                    "mainus": "p2",
                },
            },
        }
        data = _IQAirData.model_validate(raw)
        reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.aqiCategory == "Moderate"

    def test_category_bands_unhealthy_sensitive(self) -> None:
        """AQI 125 → aqiCategory = 'Unhealthy for Sensitive Groups' (101–150 EPA band)."""
        raw = {
            "city": "TestCity",
            "state": "TestState",
            "country": "USA",
            "current": {
                "weather": {"ts": "2019-04-08T19:00:00.000Z"},
                "pollution": {
                    "ts": "2019-04-08T18:00:00.000Z",
                    "aqius": 125,
                    "mainus": "o3",
                },
            },
        }
        data = _IQAirData.model_validate(raw)
        reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.aqiCategory == "Unhealthy for Sensitive Groups"

    def test_category_bands_hazardous(self) -> None:
        """AQI 400 → aqiCategory = 'Hazardous' (301–500 EPA band)."""
        raw = {
            "city": "TestCity",
            "state": "TestState",
            "country": "USA",
            "current": {
                "weather": {"ts": "2019-04-08T19:00:00.000Z"},
                "pollution": {
                    "ts": "2019-04-08T18:00:00.000Z",
                    "aqius": 400,
                    "mainus": "p2",
                },
            },
        }
        data = _IQAirData.model_validate(raw)
        reading = _wire_to_canonical(data)
        assert reading is not None
        assert reading.aqiCategory == "Hazardous"


# ---------------------------------------------------------------------------
# 3. Envelope error mapping (LC12/LC27)
# ---------------------------------------------------------------------------


class TestEnvelopeErrorMapping:
    """LC12/LC27: status='fail' + data.message dispatch to canonical taxonomy."""

    @pytest.mark.parametrize("msg", [
        "incorrect_api_key",
        "api_key_expired",
        "payment required",
        "permission_denied",
        "forbidden",
        "feature_not_available",
    ])
    def test_key_invalid_messages(self, msg: str) -> None:
        """Auth failure message strings → KeyInvalid (permanent)."""
        with pytest.raises(KeyInvalid):
            _raise_for_envelope_error(msg)

    @pytest.mark.parametrize("msg", [
        "call_limit_reached",
        "too_many_requests",
    ])
    def test_quota_exhausted_messages(self, msg: str) -> None:
        """Rate-limit message strings → QuotaExhausted with retry_after_seconds=None."""
        with pytest.raises(QuotaExhausted) as exc_info:
            _raise_for_envelope_error(msg)
        assert exc_info.value.retry_after_seconds is None, (
            "200-not-429 envelope error must have retry_after_seconds=None "
            "(no Retry-After header available from IQAir on this path)"
        )

    @pytest.mark.parametrize("msg", [
        "city_not_found",
        "no_nearest_station",
        "node not found",
        "unknown_error_code",
    ])
    def test_other_messages_raise_provider_protocol_error(self, msg: str) -> None:
        """Non-auth non-quota message strings → ProviderProtocolError."""
        with pytest.raises(ProviderProtocolError):
            _raise_for_envelope_error(msg)

    def test_key_invalid_has_provider_context(self) -> None:
        """KeyInvalid from envelope includes provider_id='iqair'."""
        with pytest.raises(KeyInvalid) as exc_info:
            _raise_for_envelope_error("incorrect_api_key")
        assert exc_info.value.provider_id == PROVIDER_ID

    def test_quota_exhausted_has_provider_context(self) -> None:
        """QuotaExhausted from envelope includes provider_id='iqair'."""
        with pytest.raises(QuotaExhausted) as exc_info:
            _raise_for_envelope_error("call_limit_reached")
        assert exc_info.value.provider_id == PROVIDER_ID

    def test_protocol_error_has_provider_context(self) -> None:
        """ProviderProtocolError from envelope includes provider_id='iqair'."""
        with pytest.raises(ProviderProtocolError) as exc_info:
            _raise_for_envelope_error("city_not_found")
        assert exc_info.value.provider_id == PROVIDER_ID


# ---------------------------------------------------------------------------
# 4. LC13 pre-call key guard (fetch() with empty/None key)
# ---------------------------------------------------------------------------


class TestPreCallKeyGuard:
    """LC13: empty/None key → KeyInvalid before any HTTP call."""

    def test_empty_key_raises_key_invalid(self) -> None:
        """fetch() with key='' raises KeyInvalid before hitting the network."""
        with pytest.raises(KeyInvalid) as exc_info:
            fetch(lat=36.1767, lon=-86.7386, key="")
        assert exc_info.value.provider_id == PROVIDER_ID
        # Confirm the error message names the env var so operator knows what to fix
        assert "WEEWX_CLEARSKIES_IQAIR_KEY" in str(exc_info.value)

    def test_none_key_raises_key_invalid(self) -> None:
        """fetch() with key=None raises KeyInvalid before hitting the network."""
        with pytest.raises(KeyInvalid):
            fetch(lat=36.1767, lon=-86.7386, key=None)  # type: ignore[arg-type]

    def test_whitespace_only_key_raises_key_invalid(self) -> None:
        """fetch() with key='   ' (whitespace only) raises KeyInvalid.

        Wire-level validation should treat whitespace-only keys like empty keys.
        Note: AQISettings strips whitespace before storing, so this tests the
        module's own guard when called directly with a bad value.
        """
        with pytest.raises(KeyInvalid):
            fetch(lat=36.1767, lon=-86.7386, key="   ")


# ---------------------------------------------------------------------------
# 5. Cache key construction
# ---------------------------------------------------------------------------


class TestCacheKeyConstruction:
    """Cache key: credentials NOT in key; lat/lon rounded to 4 decimals; deterministic."""

    def test_cache_key_is_deterministic(self) -> None:
        """Same lat/lon → same SHA-256 key every time."""
        k1 = _build_cache_key(36.1767, -86.7386)
        k2 = _build_cache_key(36.1767, -86.7386)
        assert k1 == k2

    def test_cache_key_uses_4_decimal_rounding(self) -> None:
        """lat/lon rounded to 4 decimals — lat 36.17671 and 36.17674 produce same key."""
        k1 = _build_cache_key(36.17671, -86.7386)
        k2 = _build_cache_key(36.17674, -86.7386)
        assert k1 == k2, "lat/lon rounded to 4dp → same cache key"

    def test_cache_key_changes_with_different_location(self) -> None:
        """Different coordinates produce different cache keys."""
        k1 = _build_cache_key(36.1767, -86.7386)
        k2 = _build_cache_key(42.1234, -72.5678)
        assert k1 != k2

    def test_cache_key_does_not_contain_provider_id_iqair(self) -> None:
        """Cache key is a SHA-256 hex string; provider_id is in the payload not plaintext.

        This verifies the credential is NOT in the key and the key is opaque.
        (The provider_id IS embedded in the SHA-256 payload but not as plaintext
        in the hex digest — which is the correct privacy pattern.)
        """
        k = _build_cache_key(36.1767, -86.7386)
        # Key must be a hex digest (64 chars for SHA-256)
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)

    def test_cache_key_matches_expected_sha256(self) -> None:
        """Cache key SHA-256 matches manually-computed expected value."""
        payload = json.dumps(
            {
                "provider_id": "iqair",
                "endpoint": "aqi_current",
                "params": {"lat4": 36.1767, "lon4": -86.7386},
            },
            sort_keys=True,
        )
        expected = hashlib.sha256(payload.encode()).hexdigest()
        assert _build_cache_key(36.1767, -86.7386) == expected


# ---------------------------------------------------------------------------
# 6. Cache hit / miss / sentinel (fetch() with mocked HTTP client + memory cache)
# ---------------------------------------------------------------------------


class TestCacheHitMissSentinel:
    """3-way cache path coverage: miss, hit, sentinel."""

    def setup_method(self) -> None:
        """Reset cache and HTTP client before each test."""
        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            reset_cache_for_tests,
            wire_cache_from_env,
        )
        from weewx_clearskies_api.providers.aqi.iqair import _reset_http_client_for_tests  # noqa: PLC0415

        reset_cache_for_tests()
        _reset_http_client_for_tests()
        iqair._rate_limiter._calls.clear()
        wire_cache_from_env()

    def teardown_method(self) -> None:
        """Clean up cache and HTTP client after each test."""
        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            reset_cache_for_tests,
        )
        from weewx_clearskies_api.providers.aqi.iqair import _reset_http_client_for_tests  # noqa: PLC0415

        reset_cache_for_tests()
        _reset_http_client_for_tests()
        iqair._rate_limiter._calls.clear()

    def _make_mock_response(self, json_data: object, status_code: int = 200) -> MagicMock:
        """Build a mock httpx.Response for the HTTP client."""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data
        mock_resp.text = str(json_data)
        return mock_resp

    def _nashville_fixture(self) -> dict:
        return {
            "status": "success",
            "data": {
                "city": "Nashville",
                "state": "Tennessee",
                "country": "USA",
                "current": {
                    "weather": {"ts": "2019-04-08T19:00:00.000Z"},
                    "pollution": {
                        "ts": "2019-04-08T18:00:00.000Z",
                        "aqius": 10,
                        "mainus": "p2",
                        "aqicn": 3,
                        "maincn": "p2",
                    },
                },
            },
        }

    def test_cache_miss_calls_http_and_returns_reading(self) -> None:
        """Cache miss → HTTP call made → AQIReading returned and cached."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415

        mock_client = MagicMock()
        mock_client.get.return_value = self._make_mock_response(self._nashville_fixture())

        reading = fetch(lat=36.1767, lon=-86.7386, key="TEST_KEY", http_client=mock_client)

        assert mock_client.get.call_count == 1, "Cache miss must make exactly 1 HTTP call"
        assert reading is not None
        assert reading.aqi == 10
        assert reading.source == "iqair"

        # Verify cached
        cached = get_cache().get(_build_cache_key(36.1767, -86.7386))
        assert cached is not None, "Result must be cached after cache miss"

    def test_cache_hit_skips_http(self) -> None:
        """Cache hit → no HTTP call; cached reading returned."""
        mock_client = MagicMock()
        mock_client.get.return_value = self._make_mock_response(self._nashville_fixture())

        # First call — fills cache
        reading1 = fetch(lat=36.1767, lon=-86.7386, key="TEST_KEY", http_client=mock_client)
        call_count_after_miss = mock_client.get.call_count

        # Second call — should hit cache
        reading2 = fetch(lat=36.1767, lon=-86.7386, key="TEST_KEY", http_client=mock_client)

        assert mock_client.get.call_count == call_count_after_miss, (
            "Cache hit must not make additional HTTP calls"
        )
        assert reading1 is not None
        assert reading2 is not None
        assert reading1.aqi == reading2.aqi

    def test_sentinel_cached_on_null_aqius(self) -> None:
        """Null aqius → _wire_to_canonical returns None → sentinel cached → None returned."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415

        null_fixture = {
            "status": "success",
            "data": {
                "city": "Empty",
                "state": "TestState",
                "country": "USA",
                "current": {
                    "weather": {"ts": "2019-04-08T19:00:00.000Z"},
                    "pollution": {
                        "ts": "2019-04-08T18:00:00.000Z",
                        # aqius absent — null reading
                    },
                },
            },
        }
        mock_client = MagicMock()
        mock_client.get.return_value = self._make_mock_response(null_fixture)

        reading = fetch(lat=36.1767, lon=-86.7386, key="TEST_KEY", http_client=mock_client)

        assert reading is None, "Null aqius must produce None return from fetch()"

        # Sentinel must be cached
        cached = get_cache().get(_build_cache_key(36.1767, -86.7386))
        assert cached == {"_no_reading": True}, (
            "Sentinel {'_no_reading': True} must be cached for null reading"
        )

    def test_sentinel_hit_returns_none_without_http(self) -> None:
        """After sentinel is cached, subsequent call returns None without HTTP."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415

        # Manually store the sentinel
        get_cache().set(
            _build_cache_key(36.1767, -86.7386),
            {"_no_reading": True},
            ttl_seconds=900,
        )

        mock_client = MagicMock()
        reading = fetch(lat=36.1767, lon=-86.7386, key="TEST_KEY", http_client=mock_client)

        assert reading is None
        assert mock_client.get.call_count == 0, (
            "Sentinel cache hit must not make HTTP calls"
        )


# ---------------------------------------------------------------------------
# 7. Capability declaration
# ---------------------------------------------------------------------------


class TestCapabilityDeclaration:
    """CAPABILITY symbol validation."""

    def test_capability_provider_id_is_iqair(self) -> None:
        """CAPABILITY.provider_id = 'iqair'."""
        assert CAPABILITY.provider_id == "iqair"

    def test_capability_domain_is_aqi(self) -> None:
        """CAPABILITY.domain = 'aqi'."""
        assert CAPABILITY.domain == "aqi"

    def test_capability_supplied_fields_has_six_fields(self) -> None:
        """CAPABILITY.supplied_canonical_fields has exactly 6 free-tier fields."""
        supplied = set(CAPABILITY.supplied_canonical_fields)
        expected = {
            "aqi", "aqiCategory", "aqiMainPollutant", "aqiLocation",
            "observedAt", "source",
        }
        assert supplied == expected, (
            f"Expected 6 free-tier fields {expected!r}, got {supplied!r}"
        )

    def test_capability_auth_required_is_key(self) -> None:
        """CAPABILITY.auth_required = ('key',) — single query-param credential."""
        assert CAPABILITY.auth_required == ("key",)

    def test_capability_has_global_coverage(self) -> None:
        """CAPABILITY.geographic_coverage = 'global'."""
        assert CAPABILITY.geographic_coverage == "global"

    def test_capability_poll_interval_is_900s(self) -> None:
        """CAPABILITY.default_poll_interval_seconds = 900 (15 min per ADR-017)."""
        assert CAPABILITY.default_poll_interval_seconds == 900

    def test_no_pollutant_concentration_fields_in_capability(self) -> None:
        """CAPABILITY does NOT include pollutant concentration fields (free-tier PARTIAL-DOMAIN)."""
        supplied = set(CAPABILITY.supplied_canonical_fields)
        concentration_fields = {
            "pollutantPM25", "pollutantPM10",
            "pollutantO3", "pollutantNO2", "pollutantSO2", "pollutantCO",
        }
        overlap = supplied & concentration_fields
        assert not overlap, (
            f"Free-tier CAPABILITY must not include concentration fields; found {overlap!r}"
        )


# ---------------------------------------------------------------------------
# 8. Rate limiter configuration
# ---------------------------------------------------------------------------


class TestRateLimiterConfig:
    """Rate limiter configured for IQAir Community per-minute cap (LC10)."""

    def test_rate_limiter_max_calls_is_5(self) -> None:
        """Rate limiter max_calls = 5 (IQAir Community tier per-minute cap)."""
        assert iqair._rate_limiter.max_calls == 5

    def test_rate_limiter_window_seconds_is_60(self) -> None:
        """Rate limiter window_seconds = 60 (per-minute, stricter than OWM/Aeris per-second)."""
        assert iqair._rate_limiter.window_seconds == 60

    def test_rate_limiter_provider_id_is_iqair(self) -> None:
        """Rate limiter provider_id = 'iqair'."""
        assert iqair._rate_limiter.provider_id == "iqair"


# ---------------------------------------------------------------------------
# 9. Pollutant code lookup table completeness
# ---------------------------------------------------------------------------


class TestMainusToCanonical:
    """_MAINUS_TO_CANONICAL table shape and completeness."""

    def test_all_six_canonical_pollutants_present(self) -> None:
        """All 6 canonical pollutants are in the lookup table."""
        expected_values = {"PM10", "PM2.5", "NO2", "O3", "SO2", "CO"}
        actual_values = set(_MAINUS_TO_CANONICAL.values())
        assert actual_values == expected_values, (
            f"Expected canonical values {expected_values!r}, got {actual_values!r}"
        )

    def test_all_six_mainus_codes_present(self) -> None:
        """All 6 mainus codes are in the lookup table."""
        expected_keys = {"p1", "p2", "n2", "o3", "s2", "co"}
        actual_keys = set(_MAINUS_TO_CANONICAL.keys())
        assert actual_keys == expected_keys, (
            f"Expected mainus codes {expected_keys!r}, got {actual_keys!r}"
        )

    def test_p1_maps_to_pm10(self) -> None:
        assert _MAINUS_TO_CANONICAL["p1"] == "PM10"

    def test_p2_maps_to_pm25(self) -> None:
        assert _MAINUS_TO_CANONICAL["p2"] == "PM2.5"

    def test_n2_maps_to_no2(self) -> None:
        assert _MAINUS_TO_CANONICAL["n2"] == "NO2"

    def test_o3_maps_to_o3(self) -> None:
        assert _MAINUS_TO_CANONICAL["o3"] == "O3"

    def test_s2_maps_to_so2(self) -> None:
        assert _MAINUS_TO_CANONICAL["s2"] == "SO2"

    def test_co_maps_to_co(self) -> None:
        assert _MAINUS_TO_CANONICAL["co"] == "CO"
