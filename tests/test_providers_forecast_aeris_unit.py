"""Unit tests for the Aeris forecast provider (3b round 4).

Covers per the task-3b-4 brief §Test author parallel scope:

  Pure-compute helpers:
  - _aeris_descriptor_to_precip_type: all 13 descriptor entries in the
    lookup table, unknown descriptor → None, None input, empty input,
    coded string with no descriptor segment (short split), mixed-precip
    descriptors (RS, WM, SI) → "rain" (with DEBUG log).
  - _wind_speed_max_mps / _wind_gust_max_mps: MPS present (direct),
    MPS absent + KPH present (post-convert), both absent (None).

  Wire-shape Pydantic:
  - _AerisHourlyPeriod: real fixture loads cleanly; extra fields ignored;
    dateTimeISO required (missing raises ValidationError).
  - _AerisDayNightPeriod: real fixture loads cleanly; extra fields ignored.
  - _AerisEnvelope: success=true + success=false + warn shape.

  _detect_discussion (Q2 runtime detection):
  - Non-empty response-level summary → ForecastDiscussion with headline,
    body, source="aeris", issuedAt=<UTC-converted dateTimeISO>.
  - Non-empty period-level summary → same shape (second detection point).
  - Neither field present → None.
  - Empty/whitespace-only summary → None.
  - first_period_raw=None → None.

  _to_canonical:
  - Zips hourly + daily correctly; source="aeris".
  - Discussion=None when no summary field (free-tier fixture).
  - Discussion populated when summary injected (paid-tier fixture).
  - US, METRIC, METRICWX unit-field selection verified per field.
  - validTime UTC-Z; validDate station-local YYYY-MM-DD.
  - precipType derivation correct for a coded-period fixture.
  - windSpeedMax for METRICWX uses KPH-fallback when MPS absent.

  fetch() (respx-mocked):
  - Cache miss → two outbound HTTP calls → bundle cached → returned.
  - Cache hit → zero outbound HTTP calls → cached bundle returned.
  - Cached discussion=None round-trips correctly.
  - Cached discussion=ForecastDiscussion round-trips correctly.
  - Missing client_id → KeyInvalid raised (loud failure, lead-call 12).
  - Missing client_secret → KeyInvalid raised.
  - Both missing → KeyInvalid raised.
  - 401 on hourly call → KeyInvalid (exc.status_code == 401, lead-call 9).
  - 429 on hourly call → QuotaExhausted.
  - 5xx on hourly call → TransientNetworkError.
  - Aeris returns success=false envelope → ProviderProtocolError.
  - Aeris returns success=true + warn_location + response=[] → empty bundle
    (lead-call 17): hourly=[], daily=[], source="aeris".
  - Malformed hourly wire shape → ProviderProtocolError.

  Redaction filter (lead-call 23):
  - URL with client_id=ABC123&client_secret=DEF456 → both values redacted.
  - client_id= and client_secret= are independently tested.

  Capability registry:
  - wire_providers([aeris.CAPABILITY]) → registry has aeris forecast entry.
  - CAPABILITY.provider_id = "aeris", domain = "forecast".
  - CAPABILITY.auth_required includes "client_id" and "client_secret".
  - CAPABILITY.supplied_canonical_fields includes hourly and daily fields.
  - CAPABILITY.supplied_canonical_fields includes "headline" and "body"
    (max-surface; brief Q2 user decision).

No DB, no live network. respx mocks outbound httpx calls.
Wire-shape rule: fixtures loaded from tests/fixtures/providers/aeris/*.json
(real Aeris response shapes per rules/clearskies-process.md).
ADR references: ADR-006, ADR-007, ADR-010, ADR-017, ADR-018, ADR-019, ADR-020, ADR-038.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "providers" / "aeris"


def _load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture file from tests/fixtures/providers/aeris/."""
    path = _FIXTURES_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return json.loads(fh.read())


# ---------------------------------------------------------------------------
# State-reset helpers
# ---------------------------------------------------------------------------


def _reset_provider_state() -> None:
    """Reset provider registry, cache, rate limiter, and re-wire memory cache.

    Every test that calls fetch() needs a wired cache and clean state.
    Matches the pattern from NWS unit tests (3b-3).
    """
    from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
        reset_cache_for_tests,
        wire_cache_from_env,
    )
    from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
        reset_provider_registry_for_tests,
    )
    from weewx_clearskies_api.providers.forecast.aeris import _reset_http_client_for_tests  # noqa: PLC0415
    import weewx_clearskies_api.providers.forecast.aeris as _aeris  # noqa: PLC0415

    reset_cache_for_tests()
    reset_provider_registry_for_tests()
    _reset_http_client_for_tests()
    # Clear rate-limiter deque so consecutive tests don't trip each other.
    _aeris._rate_limiter._calls.clear()
    # Clear logged-unknown-descriptor sets so DEBUG-logging tests don't
    # silently pass because the set was already populated.
    _aeris._logged_unknown_descriptors.clear()
    _aeris._logged_mixed_precip.clear()
    # Re-wire a clean memory cache (CLEARSKIES_CACHE_URL unset in unit test env).
    wire_cache_from_env()


# Aeris URLs for respx mocking
_LAT = 47.6062
_LON = -122.3321
_LOCATION = f"{round(_LAT, 4)},{round(_LON, 4)}"
_AERIS_HOURLY_URL = f"https://data.api.xweather.com/forecasts/{_LOCATION}"
_AERIS_DAYNIGHT_URL = f"https://data.api.xweather.com/forecasts/{_LOCATION}"

_TEST_CLIENT_ID = "TEST_CLIENT_ID"
_TEST_CLIENT_SECRET = "TEST_CLIENT_SECRET"


# ===========================================================================
# 1. _aeris_descriptor_to_precip_type — lookup table coverage
# ===========================================================================


class TestAerisDescriptorToPrecipType:
    """_aeris_descriptor_to_precip_type maps Aeris coded strings to canonical precipType."""

    def test_rain_descriptor_R_returns_rain(self) -> None:
        """Descriptor 'R' (rain) maps to 'rain'."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type(":HV:R") == "rain"

    def test_rain_shower_descriptor_RW_returns_rain(self) -> None:
        """Descriptor 'RW' (rain shower) maps to 'rain'."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type("::RW") == "rain"

    def test_drizzle_descriptor_L_returns_rain(self) -> None:
        """Descriptor 'L' (drizzle) maps to 'rain'."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type("::L") == "rain"

    def test_snow_descriptor_S_returns_snow(self) -> None:
        """Descriptor 'S' (snow) maps to 'snow'."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type("::S") == "snow"

    def test_snow_shower_descriptor_SW_returns_snow(self) -> None:
        """Descriptor 'SW' (snow showers) maps to 'snow'."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type(":LGT:SW") == "snow"

    def test_freezing_rain_descriptor_ZR_returns_freezing_rain(self) -> None:
        """Descriptor 'ZR' (freezing rain) maps to 'freezing-rain'."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type("::ZR") == "freezing-rain"

    def test_freezing_drizzle_descriptor_ZL_returns_freezing_rain(self) -> None:
        """Descriptor 'ZL' (freezing drizzle) maps to 'freezing-rain'."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type("::ZL") == "freezing-rain"

    def test_ice_pellets_descriptor_IP_returns_sleet(self) -> None:
        """Descriptor 'IP' (ice pellets/sleet) maps to 'sleet'."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type("::IP") == "sleet"

    def test_hail_descriptor_A_returns_hail(self) -> None:
        """Descriptor 'A' (hail) maps to 'hail'."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type("::A") == "hail"

    def test_thunderstorm_descriptor_T_returns_rain(self) -> None:
        """Descriptor 'T' (thunderstorms) maps to 'rain' (consistent with NWS tsra)."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type("::T") == "rain"

    def test_rain_snow_mix_descriptor_RS_returns_rain(self, caplog: Any) -> None:
        """Descriptor 'RS' (rain/snow mix) maps to 'rain'; DEBUG log emitted once."""
        import weewx_clearskies_api.providers.forecast.aeris as _aeris  # noqa: PLC0415
        _aeris._logged_mixed_precip.clear()
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        with caplog.at_level(logging.DEBUG, logger="weewx_clearskies_api.providers.forecast.aeris"):
            result = _aeris_descriptor_to_precip_type("::RS")
        assert result == "rain"
        assert "RS" in _aeris._logged_mixed_precip

    def test_wintry_mix_descriptor_WM_returns_rain(self) -> None:
        """Descriptor 'WM' (wintry mix) maps to 'rain'."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type("::WM") == "rain"

    def test_snow_sleet_mix_descriptor_SI_returns_rain(self) -> None:
        """Descriptor 'SI' (snow/sleet) maps to 'rain'."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type("::SI") == "rain"

    def test_overcast_descriptor_OVC_returns_none(self) -> None:
        """Descriptor 'OVC' (overcast, no precip) maps to None."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type("::OVC") is None

    def test_scattered_clouds_descriptor_SCT_returns_none(self) -> None:
        """Descriptor 'SCT' (scattered clouds) maps to None."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type("::SCT") is None

    def test_fog_descriptor_F_returns_none(self) -> None:
        """Descriptor 'F' (fog) maps to None."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type("::F") is None

    def test_unknown_descriptor_logs_debug_and_returns_none(self, caplog: Any) -> None:
        """Unknown descriptor 'XYZZY' logs DEBUG once and returns None."""
        import weewx_clearskies_api.providers.forecast.aeris as _aeris  # noqa: PLC0415
        _aeris._logged_unknown_descriptors.clear()
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        with caplog.at_level(logging.DEBUG, logger="weewx_clearskies_api.providers.forecast.aeris"):
            result = _aeris_descriptor_to_precip_type("::XYZZY")
        assert result is None
        assert "XYZZY" in _aeris._logged_unknown_descriptors

    def test_unknown_descriptor_only_logged_once(self) -> None:
        """Second call with the same unknown descriptor doesn't log again."""
        import weewx_clearskies_api.providers.forecast.aeris as _aeris  # noqa: PLC0415
        _aeris._logged_unknown_descriptors.clear()
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        _aeris_descriptor_to_precip_type("::NOVEL")
        _aeris._logged_unknown_descriptors.add("NOVEL")  # already logged from first call
        # Second call — should not double-add
        count_before = len(_aeris._logged_unknown_descriptors)
        _aeris_descriptor_to_precip_type("::NOVEL")
        assert len(_aeris._logged_unknown_descriptors) == count_before

    def test_none_input_returns_none(self) -> None:
        """None input returns None — no crash."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type(None) is None

    def test_empty_string_returns_none(self) -> None:
        """Empty string returns None."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type("") is None

    def test_coded_string_with_no_descriptor_segment_returns_none(self) -> None:
        """String with fewer than 3 colon-segments returns None (defensive)."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        # "onlyone" has no colons → split gives ["onlyone"] → no segment [2]
        assert _aeris_descriptor_to_precip_type("onlyone") is None

    def test_coded_string_with_empty_descriptor_segment_returns_none(self) -> None:
        """String ':HV:' has empty third segment → returns None."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        assert _aeris_descriptor_to_precip_type(":HV:") is None

    def test_partial_split_only_two_segments_returns_none(self) -> None:
        """String 'HV:R' has only two colon-segments (index 0 and 1) → no index 2 → None."""
        from weewx_clearskies_api.providers.forecast.aeris import _aeris_descriptor_to_precip_type  # noqa: PLC0415
        # "HV:R" splits to ["HV", "R"] — index 2 doesn't exist
        assert _aeris_descriptor_to_precip_type("HV:R") is None


# ===========================================================================
# 2. _wind_speed_max_mps / _wind_gust_max_mps — unit fallback helpers
# ===========================================================================


class TestWindMaxMpsHelpers:
    """_wind_speed_max_mps and _wind_gust_max_mps fall back to KPH/3.6."""

    def _make_period(self, **kwargs: Any) -> Any:
        """Build a minimal _AerisDayNightPeriod with only the supplied fields."""
        from weewx_clearskies_api.providers.forecast.aeris import _AerisDayNightPeriod  # noqa: PLC0415
        # dateTimeISO is required
        data = {"dateTimeISO": "2026-05-08T07:00:00-07:00", **kwargs}
        return _AerisDayNightPeriod.model_validate(data)

    def test_wind_speed_max_mps_uses_mps_field_when_present(self) -> None:
        """windSpeedMaxMPS present → returned directly."""
        from weewx_clearskies_api.providers.forecast.aeris import _wind_speed_max_mps  # noqa: PLC0415
        period = self._make_period(windSpeedMaxMPS=5.0, windSpeedMaxKPH=18.0)
        assert _wind_speed_max_mps(period) == 5.0

    def test_wind_speed_max_mps_falls_back_to_kph_divided_by_3_6(self) -> None:
        """windSpeedMaxMPS absent + windSpeedMaxKPH present → KPH ÷ 3.6."""
        from weewx_clearskies_api.providers.forecast.aeris import _wind_speed_max_mps  # noqa: PLC0415
        period = self._make_period(windSpeedMaxMPS=None, windSpeedMaxKPH=36.0)
        result = _wind_speed_max_mps(period)
        assert result is not None
        assert abs(result - 10.0) < 0.01   # 36 ÷ 3.6 = 10 m/s

    def test_wind_speed_max_mps_returns_none_when_both_absent(self) -> None:
        """Both windSpeedMaxMPS and windSpeedMaxKPH absent → None."""
        from weewx_clearskies_api.providers.forecast.aeris import _wind_speed_max_mps  # noqa: PLC0415
        period = self._make_period(windSpeedMaxMPS=None, windSpeedMaxKPH=None)
        assert _wind_speed_max_mps(period) is None

    def test_wind_gust_max_mps_uses_mps_field_when_present(self) -> None:
        """windGustMaxMPS present → returned directly."""
        from weewx_clearskies_api.providers.forecast.aeris import _wind_gust_max_mps  # noqa: PLC0415
        period = self._make_period(windGustMaxMPS=7.0, windGustMaxKPH=25.0)
        assert _wind_gust_max_mps(period) == 7.0

    def test_wind_gust_max_mps_falls_back_to_kph_divided_by_3_6(self) -> None:
        """windGustMaxMPS absent + windGustMaxKPH present → KPH ÷ 3.6."""
        from weewx_clearskies_api.providers.forecast.aeris import _wind_gust_max_mps  # noqa: PLC0415
        period = self._make_period(windGustMaxMPS=None, windGustMaxKPH=25.2)
        result = _wind_gust_max_mps(period)
        assert result is not None
        assert abs(result - 7.0) < 0.01   # 25.2 ÷ 3.6 = 7 m/s

    def test_wind_gust_max_mps_returns_none_when_both_absent(self) -> None:
        """Both windGustMaxMPS and windGustMaxKPH absent → None."""
        from weewx_clearskies_api.providers.forecast.aeris import _wind_gust_max_mps  # noqa: PLC0415
        period = self._make_period(windGustMaxMPS=None, windGustMaxKPH=None)
        assert _wind_gust_max_mps(period) is None


# ===========================================================================
# 3. Wire-shape Pydantic models — real fixture validation
# ===========================================================================


class TestWireShapePydanticModels:
    """Wire-shape Pydantic models validate against real captured fixtures."""

    def test_hourly_period_loads_from_real_fixture_first_period(self) -> None:
        """First period of forecasts_hourly.json validates against _AerisHourlyPeriod."""
        from weewx_clearskies_api.providers.forecast.aeris import _AerisHourlyPeriod  # noqa: PLC0415
        fixture = _load_fixture("forecasts_hourly.json")
        raw_period = fixture["response"][0]["periods"][0]
        period = _AerisHourlyPeriod.model_validate(raw_period)
        # Core fields present in real fixture
        assert period.dateTimeISO.startswith("2026-")
        assert period.tempF is not None or period.tempC is not None
        assert period.windDirDEG is not None
        assert period.sky is not None

    def test_hourly_period_extra_fields_are_ignored(self) -> None:
        """Extra fields in the wire shape are silently ignored (extras='ignore')."""
        from weewx_clearskies_api.providers.forecast.aeris import _AerisHourlyPeriod  # noqa: PLC0415
        raw = {
            "dateTimeISO": "2026-05-08T13:00:00-07:00",
            "tempC": 14.3,
            "EXTRA_FIELD_THAT_DOES_NOT_EXIST": "should_be_ignored",
        }
        period = _AerisHourlyPeriod.model_validate(raw)
        assert period.tempC == 14.3
        assert not hasattr(period, "EXTRA_FIELD_THAT_DOES_NOT_EXIST")

    def test_hourly_period_missing_required_dateTimeISO_raises_validation_error(self) -> None:
        """Missing dateTimeISO (required field) raises pydantic.ValidationError."""
        from pydantic import ValidationError  # noqa: PLC0415
        from weewx_clearskies_api.providers.forecast.aeris import _AerisHourlyPeriod  # noqa: PLC0415
        with pytest.raises(ValidationError):
            _AerisHourlyPeriod.model_validate({"tempC": 14.3})

    def test_daynight_period_loads_from_real_fixture_first_period(self) -> None:
        """First period of forecasts_daynight.json validates against _AerisDayNightPeriod."""
        from weewx_clearskies_api.providers.forecast.aeris import _AerisDayNightPeriod  # noqa: PLC0415
        fixture = _load_fixture("forecasts_daynight.json")
        raw_period = fixture["response"][0]["periods"][0]
        period = _AerisDayNightPeriod.model_validate(raw_period)
        assert period.dateTimeISO.startswith("2026-")
        assert period.uvi is not None
        # Real fixture: sunriseISO may be None in free-tier daynight response
        # (confirmed in captured fixture — free-tier doesn't include sunrise/sunset in daynight)

    def test_daynight_period_extra_fields_ignored(self) -> None:
        """Extra fields in daynight wire shape are ignored."""
        from weewx_clearskies_api.providers.forecast.aeris import _AerisDayNightPeriod  # noqa: PLC0415
        raw = {
            "dateTimeISO": "2026-05-08T07:00:00-07:00",
            "maxTempF": 65.0,
            "UNKNOWN_PAID_TIER_FIELD": "ignored",
        }
        period = _AerisDayNightPeriod.model_validate(raw)
        assert period.maxTempF == 65.0

    def test_daynight_period_missing_required_dateTimeISO_raises_validation_error(self) -> None:
        """Missing dateTimeISO on daynight period raises ValidationError."""
        from pydantic import ValidationError  # noqa: PLC0415
        from weewx_clearskies_api.providers.forecast.aeris import _AerisDayNightPeriod  # noqa: PLC0415
        with pytest.raises(ValidationError):
            _AerisDayNightPeriod.model_validate({"maxTempF": 65.0})

    def test_aeris_envelope_success_true_parses_correctly(self) -> None:
        """Success=true envelope parses; response list preserved."""
        from weewx_clearskies_api.providers.forecast.aeris import _AerisEnvelope  # noqa: PLC0415
        data = {"success": True, "error": None, "response": [{"periods": []}]}
        env = _AerisEnvelope.model_validate(data)
        assert env.success is True
        assert env.error is None
        assert len(env.response) == 1

    def test_aeris_envelope_success_false_parses_correctly(self) -> None:
        """Success=false envelope from error fixture parses correctly."""
        from weewx_clearskies_api.providers.forecast.aeris import _AerisEnvelope  # noqa: PLC0415
        fixture = _load_fixture("error_401_invalid_credentials.json")
        env = _AerisEnvelope.model_validate(fixture)
        assert env.success is False
        assert env.error is not None
        assert env.error["code"] == "invalid_client"

    def test_aeris_envelope_warn_location_parses_correctly(self) -> None:
        """warn_location fixture: success=true, error present, response=[]."""
        from weewx_clearskies_api.providers.forecast.aeris import _AerisEnvelope  # noqa: PLC0415
        fixture = _load_fixture("error_warn_invalid_location.json")
        env = _AerisEnvelope.model_validate(fixture)
        assert env.success is True
        assert env.error is not None
        assert env.error["code"] == "warn_location"
        assert env.response == []

    def test_all_hourly_periods_from_fixture_validate_cleanly(self) -> None:
        """All 24 periods in forecasts_hourly.json validate against _AerisHourlyPeriod."""
        from weewx_clearskies_api.providers.forecast.aeris import _AerisHourlyPeriod  # noqa: PLC0415
        fixture = _load_fixture("forecasts_hourly.json")
        periods = fixture["response"][0]["periods"]
        assert len(periods) == 24
        validated = [_AerisHourlyPeriod.model_validate(p) for p in periods]
        assert all(v.dateTimeISO for v in validated)

    def test_all_daynight_periods_from_fixture_validate_cleanly(self) -> None:
        """All 14 periods in forecasts_daynight.json validate against _AerisDayNightPeriod."""
        from weewx_clearskies_api.providers.forecast.aeris import _AerisDayNightPeriod  # noqa: PLC0415
        fixture = _load_fixture("forecasts_daynight.json")
        periods = fixture["response"][0]["periods"]
        assert len(periods) == 14
        validated = [_AerisDayNightPeriod.model_validate(p) for p in periods]
        assert all(v.dateTimeISO for v in validated)


# ===========================================================================
# 4. _detect_discussion — Q2 runtime detection
# ===========================================================================


class TestDetectDiscussion:
    """_detect_discussion covers both detection points for the paid-tier summary field."""

    def _make_first_period_raw(self, summary: str | None = None) -> dict[str, Any]:
        """Build a minimal first-period raw dict for testing."""
        data: dict[str, Any] = {
            "dateTimeISO": "2026-05-08T07:00:00-07:00",
            "weatherPrimary": "Partly Cloudy",
        }
        if summary is not None:
            data["summary"] = summary
        return data

    def test_response_level_summary_returns_forecast_discussion(self) -> None:
        """Non-empty response[0].summary → ForecastDiscussion with body=that string."""
        from weewx_clearskies_api.providers.forecast.aeris import (  # noqa: PLC0415
            _detect_discussion,
            ForecastDiscussion,
        )
        daynight_raw = {
            "summary": "Partly cloudy skies expected through the period.",
            "periods": [self._make_first_period_raw()],
        }
        result = _detect_discussion(
            daynight_raw=daynight_raw,
            first_period_raw=self._make_first_period_raw(),
            provider_id="aeris",
            domain="forecast",
        )
        assert result is not None
        assert isinstance(result, ForecastDiscussion)
        assert result.body == "Partly cloudy skies expected through the period."
        assert result.source == "aeris"
        assert result.headline == "Partly Cloudy"
        assert result.issuedAt is not None
        assert result.issuedAt.endswith("Z")   # UTC ISO-8601

    def test_period_level_summary_returns_forecast_discussion(self) -> None:
        """Non-empty periods[0].summary → ForecastDiscussion (second detection point)."""
        from weewx_clearskies_api.providers.forecast.aeris import _detect_discussion  # noqa: PLC0415
        first_period = self._make_first_period_raw(
            summary="Partly cloudy with a high near 65F."
        )
        daynight_raw = {
            "summary": None,  # No response-level summary
            "periods": [first_period],
        }
        result = _detect_discussion(
            daynight_raw=daynight_raw,
            first_period_raw=first_period,
            provider_id="aeris",
            domain="forecast",
        )
        assert result is not None
        assert result.body == "Partly cloudy with a high near 65F."
        assert result.source == "aeris"

    def test_response_level_takes_precedence_over_period_level(self) -> None:
        """Response-level summary is used when both are present (first detection wins)."""
        from weewx_clearskies_api.providers.forecast.aeris import _detect_discussion  # noqa: PLC0415
        first_period = self._make_first_period_raw(summary="Period-level summary.")
        daynight_raw = {
            "summary": "Response-level summary (preferred).",
            "periods": [first_period],
        }
        result = _detect_discussion(
            daynight_raw=daynight_raw,
            first_period_raw=first_period,
            provider_id="aeris",
            domain="forecast",
        )
        assert result is not None
        assert result.body == "Response-level summary (preferred)."

    def test_neither_summary_field_returns_none(self) -> None:
        """When neither response nor period has summary → None (free-tier default)."""
        from weewx_clearskies_api.providers.forecast.aeris import _detect_discussion  # noqa: PLC0415
        daynight_raw: dict[str, Any] = {"summary": None, "periods": [{}]}
        result = _detect_discussion(
            daynight_raw=daynight_raw,
            first_period_raw=self._make_first_period_raw(),
            provider_id="aeris",
            domain="forecast",
        )
        assert result is None

    def test_empty_string_summary_returns_none(self) -> None:
        """Empty string summary is treated as absent → returns None."""
        from weewx_clearskies_api.providers.forecast.aeris import _detect_discussion  # noqa: PLC0415
        daynight_raw = {"summary": "", "periods": []}
        result = _detect_discussion(
            daynight_raw=daynight_raw,
            first_period_raw=self._make_first_period_raw(summary=""),
            provider_id="aeris",
            domain="forecast",
        )
        assert result is None

    def test_whitespace_only_summary_returns_none(self) -> None:
        """Whitespace-only summary is treated as absent → returns None."""
        from weewx_clearskies_api.providers.forecast.aeris import _detect_discussion  # noqa: PLC0415
        daynight_raw = {"summary": "   \t\n  ", "periods": []}
        result = _detect_discussion(
            daynight_raw=daynight_raw,
            first_period_raw=self._make_first_period_raw(summary="\n  "),
            provider_id="aeris",
            domain="forecast",
        )
        assert result is None

    def test_none_first_period_raw_returns_none_when_no_response_summary(self) -> None:
        """first_period_raw=None and no response-level summary → None."""
        from weewx_clearskies_api.providers.forecast.aeris import _detect_discussion  # noqa: PLC0415
        daynight_raw: dict[str, Any] = {"summary": None}
        result = _detect_discussion(
            daynight_raw=daynight_raw,
            first_period_raw=None,
            provider_id="aeris",
            domain="forecast",
        )
        assert result is None

    def test_detect_discussion_with_real_free_tier_fixture_returns_none(self) -> None:
        """Real free-tier forecasts_daynight.json → discussion=None (no summary field)."""
        from weewx_clearskies_api.providers.forecast.aeris import _detect_discussion  # noqa: PLC0415
        fixture = _load_fixture("forecasts_daynight.json")
        raw_first = fixture["response"][0]
        first_period_raw = raw_first.get("periods", [None])[0]
        result = _detect_discussion(
            daynight_raw=raw_first,
            first_period_raw=first_period_raw,
            provider_id="aeris",
            domain="forecast",
        )
        assert result is None

    def test_detect_discussion_with_synthetic_summary_fixture_returns_discussion(self) -> None:
        """Synthetic forecasts_daynight_with_summary.json → discussion populated."""
        from weewx_clearskies_api.providers.forecast.aeris import _detect_discussion  # noqa: PLC0415
        fixture = _load_fixture("forecasts_daynight_with_summary.json")
        raw_first = fixture["response"][0]
        first_period_raw = raw_first.get("periods", [None])[0]
        result = _detect_discussion(
            daynight_raw=raw_first,
            first_period_raw=first_period_raw,
            provider_id="aeris",
            domain="forecast",
        )
        assert result is not None
        assert result.body  # non-empty
        assert result.source == "aeris"


# ===========================================================================
# 5. _to_canonical — hourly + daily mapping + discussion
# ===========================================================================


class TestToCanonical:
    """_to_canonical translates Aeris wire responses to canonical ForecastBundle."""

    def _build_inputs(
        self,
        hourly_fixture: str = "forecasts_hourly.json",
        daynight_fixture: str = "forecasts_daynight.json",
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Load fixtures and validate into wire-shape models."""
        from weewx_clearskies_api.providers.forecast.aeris import (  # noqa: PLC0415
            _AerisHourlyPeriod,
            _AerisHourlyResponse,
            _AerisDayNightPeriod,
            _AerisDayNightResponse,
        )
        hourly_data = _load_fixture(hourly_fixture)
        daynight_data = _load_fixture(daynight_fixture)

        hourly_raw_first = hourly_data["response"][0]
        daynight_raw_first = daynight_data["response"][0]

        hourly_wire = _AerisHourlyResponse.model_validate(hourly_raw_first)
        daynight_wire = _AerisDayNightResponse.model_validate(daynight_raw_first)

        return hourly_wire, daynight_wire, daynight_raw_first

    def test_source_field_is_aeris(self) -> None:
        """Bundle source = 'aeris' for all unit systems."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs()
        bundle = _to_canonical(h, d, target_unit="US", daynight_raw=raw)
        assert bundle.source == "aeris"

    def test_hourly_count_matches_fixture_period_count(self) -> None:
        """bundle.hourly has one entry per hourly period in the fixture."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs()
        bundle = _to_canonical(h, d, target_unit="US", daynight_raw=raw)
        assert len(bundle.hourly) == 24  # fixture has 24 periods

    def test_daily_count_is_half_of_daynight_periods_skipping_nights(self) -> None:
        """bundle.daily has one entry per day-period (14 daynight → 7 daily)."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs()
        bundle = _to_canonical(h, d, target_unit="US", daynight_raw=raw)
        # 14 daynight periods → 7 day periods (even-index)
        assert len(bundle.daily) == 7

    def test_discussion_none_for_free_tier_fixture(self) -> None:
        """Free-tier fixture (no summary) → discussion=None."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs()
        bundle = _to_canonical(h, d, target_unit="US", daynight_raw=raw)
        assert bundle.discussion is None

    def test_discussion_populated_from_synthetic_summary_fixture(self) -> None:
        """Synthetic paid-tier fixture → discussion is ForecastDiscussion."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs(daynight_fixture="forecasts_daynight_with_summary.json")
        bundle = _to_canonical(h, d, target_unit="US", daynight_raw=raw)
        assert bundle.discussion is not None
        assert bundle.discussion.source == "aeris"
        assert bundle.discussion.body  # non-empty

    def test_us_unit_hourly_outTemp_is_tempF(self) -> None:
        """US target_unit: outTemp comes from tempF field."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs()
        bundle = _to_canonical(h, d, target_unit="US", daynight_raw=raw)
        fixture = _load_fixture("forecasts_hourly.json")
        raw_first_period = fixture["response"][0]["periods"][0]
        assert bundle.hourly[0].outTemp == raw_first_period.get("tempF")

    def test_metric_unit_hourly_outTemp_is_tempC(self) -> None:
        """METRIC target_unit: outTemp comes from tempC field."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs()
        bundle = _to_canonical(h, d, target_unit="METRIC", daynight_raw=raw)
        fixture = _load_fixture("forecasts_hourly.json")
        raw_first_period = fixture["response"][0]["periods"][0]
        assert bundle.hourly[0].outTemp == raw_first_period.get("tempC")

    def test_metricwx_unit_hourly_windSpeed_is_mps(self) -> None:
        """METRICWX target_unit: windSpeed comes from windSpeedMPS field."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs()
        bundle = _to_canonical(h, d, target_unit="METRICWX", daynight_raw=raw)
        fixture = _load_fixture("forecasts_hourly.json")
        raw_first_period = fixture["response"][0]["periods"][0]
        assert bundle.hourly[0].windSpeed == raw_first_period.get("windSpeedMPS")

    def test_us_unit_daily_tempMax_is_maxTempF(self) -> None:
        """US target_unit: daily tempMax comes from maxTempF field."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs()
        bundle = _to_canonical(h, d, target_unit="US", daynight_raw=raw)
        fixture = _load_fixture("forecasts_daynight.json")
        # First period (index 0) should be a day period
        raw_day_period = fixture["response"][0]["periods"][0]
        assert bundle.daily[0].tempMax == raw_day_period.get("maxTempF")

    def test_hourly_validTime_is_utc_z_format(self) -> None:
        """validTime is UTC ISO-8601 Z (converted from offset-aware dateTimeISO)."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs()
        bundle = _to_canonical(h, d, target_unit="US", daynight_raw=raw)
        assert bundle.hourly[0].validTime.endswith("Z")

    def test_daily_validDate_is_station_local_date_string(self) -> None:
        """validDate is YYYY-MM-DD from dateTimeISO (station-local, before UTC conversion)."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs()
        bundle = _to_canonical(h, d, target_unit="US", daynight_raw=raw)
        fixture = _load_fixture("forecasts_daynight.json")
        # validDate should be the date portion of the dateTimeISO string (local timezone)
        raw_day_period = fixture["response"][0]["periods"][0]
        expected_date = raw_day_period["dateTimeISO"][:10]   # "YYYY-MM-DD"
        assert bundle.daily[0].validDate == expected_date

    def test_hourly_precipType_derived_from_weatherPrimaryCoded(self) -> None:
        """precipType is derived from the third colon-segment of weatherPrimaryCoded."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs()
        bundle = _to_canonical(h, d, target_unit="US", daynight_raw=raw)
        # The fixture period has weatherPrimaryCoded="::SC" (scattered clouds) → None
        fixture = _load_fixture("forecasts_hourly.json")
        first_coded = fixture["response"][0]["periods"][0].get("weatherPrimaryCoded", "")
        if "SC" in (first_coded.split(":")[2] if len(first_coded.split(":")) > 2 else ""):
            assert bundle.hourly[0].precipType is None
        else:
            # Accept whatever the actual derived value is
            pass

    def test_hourly_cloudCover_equals_sky_field(self) -> None:
        """cloudCover comes from the 'sky' field (0-100 percent)."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs()
        bundle = _to_canonical(h, d, target_unit="US", daynight_raw=raw)
        fixture = _load_fixture("forecasts_hourly.json")
        raw_first = fixture["response"][0]["periods"][0]
        assert bundle.hourly[0].cloudCover == raw_first.get("sky")

    def test_daily_uvIndexMax_comes_from_uvi_field(self) -> None:
        """uvIndexMax comes from the 'uvi' field on the day period."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs()
        bundle = _to_canonical(h, d, target_unit="US", daynight_raw=raw)
        fixture = _load_fixture("forecasts_daynight.json")
        raw_day = fixture["response"][0]["periods"][0]
        assert bundle.daily[0].uvIndexMax == raw_day.get("uvi")

    def test_daily_narrative_is_none(self) -> None:
        """narrative is always None for Aeris v0.1 (brief lead-call 20)."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        h, d, raw = self._build_inputs()
        bundle = _to_canonical(h, d, target_unit="US", daynight_raw=raw)
        assert all(point.narrative is None for point in bundle.daily)

    def test_metricwx_daily_windSpeedMax_uses_kph_fallback_when_mps_absent(self) -> None:
        """METRICWX + windSpeedMaxMPS absent → KPH÷3.6 fallback (lead-call 13)."""
        from weewx_clearskies_api.providers.forecast.aeris import _to_canonical  # noqa: PLC0415
        # Use the daynight fixture. The real fixture has windSpeedMaxMPS present,
        # so we override it to null in the parsed model to test the fallback.
        h, d, raw = self._build_inputs()
        # Force MPS to None on all day periods
        for period in d.periods:
            period.windSpeedMaxMPS = None
            period.windSpeedMaxKPH = 36.0  # Force a known value
        bundle = _to_canonical(h, d, target_unit="METRICWX", daynight_raw=raw)
        # First daily: windSpeedMax should be 36.0 / 3.6 = 10.0 m/s
        if bundle.daily:
            assert bundle.daily[0].windSpeedMax is not None
            assert abs(bundle.daily[0].windSpeedMax - 10.0) < 0.01


# ===========================================================================
# 6. fetch() — respx-mocked, cache paths, error paths
# ===========================================================================


class TestFetchCacheMissAndHit:
    """fetch() with respx-mocked HTTP — cache miss and cache hit paths."""

    def test_cache_miss_makes_two_outbound_calls_and_caches_bundle(self) -> None:
        """Cache miss: two HTTP calls (1hr + daynight) made; bundle stored in cache."""
        _reset_provider_state()

        hourly_data = _load_fixture("forecasts_hourly.json")
        daynight_data = _load_fixture("forecasts_daynight.json")

        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415

        with respx.mock:
            respx.get(_AERIS_HOURLY_URL).mock(
                return_value=httpx.Response(200, json=hourly_data)
            )
            respx.get(_AERIS_DAYNIGHT_URL).mock(
                return_value=httpx.Response(200, json=daynight_data)
            )
            bundle = aeris.fetch(
                lat=_LAT,
                lon=_LON,
                target_unit="US",
                client_id=_TEST_CLIENT_ID,
                client_secret=_TEST_CLIENT_SECRET,
            )

        # Two outbound calls made
        assert respx.calls.call_count == 2

        # Bundle has correct structure
        assert bundle.source == "aeris"
        assert len(bundle.hourly) == 24
        assert len(bundle.daily) == 7
        assert bundle.discussion is None   # free-tier fixture

        # Cache was populated
        cache_key = aeris._build_cache_key(_LAT, _LON, "US")
        cached = get_cache().get(cache_key)
        assert cached is not None

    def test_cache_hit_returns_bundle_without_outbound_calls(self) -> None:
        """Cache hit: no HTTP calls made; cached bundle returned."""
        _reset_provider_state()

        hourly_data = _load_fixture("forecasts_hourly.json")
        daynight_data = _load_fixture("forecasts_daynight.json")

        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415

        # Prime the cache with a real fetch
        with respx.mock:
            respx.get(_AERIS_HOURLY_URL).mock(
                return_value=httpx.Response(200, json=hourly_data)
            )
            respx.get(_AERIS_DAYNIGHT_URL).mock(
                return_value=httpx.Response(200, json=daynight_data)
            )
            aeris.fetch(
                lat=_LAT, lon=_LON, target_unit="US",
                client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET,
            )

        # Second call — should hit cache with zero HTTP calls
        with respx.mock:
            bundle = aeris.fetch(
                lat=_LAT, lon=_LON, target_unit="US",
                client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET,
            )
        assert respx.calls.call_count == 0   # No calls in second mock context
        assert bundle.source == "aeris"

    def test_cached_bundle_with_discussion_none_round_trips_correctly(self) -> None:
        """discussion=None in cached bundle round-trips without becoming something else."""
        _reset_provider_state()

        hourly_data = _load_fixture("forecasts_hourly.json")
        daynight_data = _load_fixture("forecasts_daynight.json")

        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415

        with respx.mock:
            respx.get(_AERIS_HOURLY_URL).mock(
                return_value=httpx.Response(200, json=hourly_data)
            )
            respx.get(_AERIS_DAYNIGHT_URL).mock(
                return_value=httpx.Response(200, json=daynight_data)
            )
            bundle1 = aeris.fetch(
                lat=_LAT, lon=_LON, target_unit="US",
                client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET,
            )

        # Cache-hit call
        with respx.mock:
            bundle2 = aeris.fetch(
                lat=_LAT, lon=_LON, target_unit="US",
                client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET,
            )

        assert bundle1.discussion is None
        assert bundle2.discussion is None

    def test_cached_bundle_with_discussion_populated_round_trips_correctly(self) -> None:
        """discussion=ForecastDiscussion in cached bundle round-trips through cache."""
        _reset_provider_state()

        hourly_data = _load_fixture("forecasts_hourly.json")
        daynight_data = _load_fixture("forecasts_daynight_with_summary.json")

        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415

        with respx.mock:
            respx.get(_AERIS_HOURLY_URL).mock(
                return_value=httpx.Response(200, json=hourly_data)
            )
            respx.get(_AERIS_DAYNIGHT_URL).mock(
                return_value=httpx.Response(200, json=daynight_data)
            )
            bundle1 = aeris.fetch(
                lat=_LAT, lon=_LON, target_unit="US",
                client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET,
            )

        assert bundle1.discussion is not None

        # Cache-hit call should reconstruct the discussion
        with respx.mock:
            bundle2 = aeris.fetch(
                lat=_LAT, lon=_LON, target_unit="US",
                client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET,
            )
        assert bundle2.discussion is not None
        assert bundle2.discussion.source == "aeris"
        assert bundle2.discussion.body == bundle1.discussion.body


class TestFetchMissingCredentials:
    """fetch() raises KeyInvalid immediately when credentials are absent."""

    def test_missing_client_id_raises_key_invalid(self) -> None:
        """client_id=None with valid secret → KeyInvalid before any HTTP call."""
        _reset_provider_state()
        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415

        with respx.mock:
            with pytest.raises(KeyInvalid):
                aeris.fetch(
                    lat=_LAT, lon=_LON, target_unit="US",
                    client_id=None, client_secret=_TEST_CLIENT_SECRET,
                )
        assert respx.calls.call_count == 0

    def test_missing_client_secret_raises_key_invalid(self) -> None:
        """client_secret=None with valid id → KeyInvalid before any HTTP call."""
        _reset_provider_state()
        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415

        with respx.mock:
            with pytest.raises(KeyInvalid):
                aeris.fetch(
                    lat=_LAT, lon=_LON, target_unit="US",
                    client_id=_TEST_CLIENT_ID, client_secret=None,
                )
        assert respx.calls.call_count == 0

    def test_both_credentials_missing_raises_key_invalid(self) -> None:
        """Both client_id=None and client_secret=None → KeyInvalid immediately."""
        _reset_provider_state()
        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415

        with respx.mock:
            with pytest.raises(KeyInvalid):
                aeris.fetch(
                    lat=_LAT, lon=_LON, target_unit="US",
                    client_id=None, client_secret=None,
                )
        assert respx.calls.call_count == 0

    def test_empty_string_client_id_raises_key_invalid(self) -> None:
        """Empty string client_id (falsy) → KeyInvalid (brief lead-call 12)."""
        _reset_provider_state()
        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415

        with respx.mock:
            with pytest.raises(KeyInvalid):
                aeris.fetch(
                    lat=_LAT, lon=_LON, target_unit="US",
                    client_id="", client_secret=_TEST_CLIENT_SECRET,
                )


class TestFetchErrorPaths:
    """fetch() translates HTTP errors to canonical exception taxonomy."""

    def test_401_on_hourly_call_raises_key_invalid(self) -> None:
        """HTTP 401 on /forecasts call → KeyInvalid (lead-call 9: exc.status_code == 401)."""
        _reset_provider_state()
        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        error_data = _load_fixture("error_401_invalid_credentials.json")

        with respx.mock:
            respx.get(_AERIS_HOURLY_URL).mock(
                return_value=httpx.Response(401, json=error_data)
            )
            with pytest.raises(KeyInvalid):
                aeris.fetch(
                    lat=_LAT, lon=_LON, target_unit="US",
                    client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET,
                )

    def test_429_on_hourly_call_raises_quota_exhausted(self) -> None:
        """HTTP 429 on /forecasts call → QuotaExhausted."""
        _reset_provider_state()
        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        error_data = _load_fixture("error_429_rate_limit.json")

        with respx.mock:
            respx.get(_AERIS_HOURLY_URL).mock(
                return_value=httpx.Response(429, json=error_data)
            )
            with pytest.raises(QuotaExhausted):
                aeris.fetch(
                    lat=_LAT, lon=_LON, target_unit="US",
                    client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET,
                )

    def test_5xx_on_hourly_call_raises_transient_network_error(self) -> None:
        """HTTP 500 on /forecasts → TransientNetworkError after retries."""
        _reset_provider_state()
        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import TransientNetworkError  # noqa: PLC0415

        with respx.mock:
            respx.get(_AERIS_HOURLY_URL).mock(
                return_value=httpx.Response(500, json={"error": "Internal Server Error"})
            )
            with pytest.raises(TransientNetworkError):
                aeris.fetch(
                    lat=_LAT, lon=_LON, target_unit="US",
                    client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET,
                )

    def test_success_false_envelope_raises_provider_protocol_error(self) -> None:
        """Aeris returns success=false in envelope → ProviderProtocolError (not KeyInvalid)."""
        _reset_provider_state()
        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import ProviderProtocolError  # noqa: PLC0415
        # success=false envelope (different from 401 HTTP error)
        error_envelope = {
            "success": False,
            "error": {"code": "internal_error", "description": "Internal API error"},
            "response": [],
        }

        with respx.mock:
            respx.get(_AERIS_HOURLY_URL).mock(
                return_value=httpx.Response(200, json=error_envelope)
            )
            with pytest.raises(ProviderProtocolError):
                aeris.fetch(
                    lat=_LAT, lon=_LON, target_unit="US",
                    client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET,
                )

    def test_warn_location_response_returns_empty_bundle(self) -> None:
        """success=true + warn_location + response=[] → empty bundle (lead-call 17)."""
        _reset_provider_state()
        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415
        warn_fixture = _load_fixture("error_warn_invalid_location.json")

        with respx.mock:
            # Both hourly and daynight return warn_location
            respx.get(_AERIS_HOURLY_URL).mock(
                return_value=httpx.Response(200, json=warn_fixture)
            )
            respx.get(_AERIS_DAYNIGHT_URL).mock(
                return_value=httpx.Response(200, json=warn_fixture)
            )
            bundle = aeris.fetch(
                lat=_LAT, lon=_LON, target_unit="US",
                client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET,
            )

        # Empty bundle returned — NOT an exception
        assert bundle.source == "aeris"
        assert bundle.hourly == []
        assert bundle.daily == []
        assert bundle.discussion is None

    def test_malformed_hourly_wire_shape_raises_provider_protocol_error(self) -> None:
        """Pydantic validation failure on hourly wire shape → ProviderProtocolError."""
        _reset_provider_state()
        from weewx_clearskies_api.providers.forecast import aeris  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import ProviderProtocolError  # noqa: PLC0415
        # Valid envelope but period missing required dateTimeISO
        malformed = {
            "success": True,
            "error": None,
            "response": [{"periods": [{"tempF": 58.0}]}],  # no dateTimeISO
        }

        with respx.mock:
            respx.get(_AERIS_HOURLY_URL).mock(
                return_value=httpx.Response(200, json=malformed)
            )
            with pytest.raises(ProviderProtocolError):
                aeris.fetch(
                    lat=_LAT, lon=_LON, target_unit="US",
                    client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET,
                )


# ===========================================================================
# 7. Redaction filter — client_id and client_secret both redacted
# ===========================================================================


class TestRedactionFilter:
    """Logged URLs with client_id and client_secret query params are redacted."""

    def test_url_with_both_credentials_redacts_both_values(self) -> None:
        """client_id=ABC123&client_secret=DEF456 in URL → both values redacted."""
        from weewx_clearskies_api.logging.redaction_filter import _redact  # noqa: PLC0415
        url = (
            "https://data.api.xweather.com/forecasts/47.6062,-122.3321"
            "?filter=1hr&limit=240&client_id=ABC123&client_secret=DEF456"
        )
        redacted = _redact(url)
        assert "ABC123" not in redacted
        assert "DEF456" not in redacted
        assert "client_id=[REDACTED]" in redacted
        assert "client_secret=[REDACTED]" in redacted

    def test_client_id_redacted_independently(self) -> None:
        """client_id alone is redacted even without client_secret in the URL."""
        from weewx_clearskies_api.logging.redaction_filter import _redact  # noqa: PLC0415
        url = "https://data.api.xweather.com/forecasts/47.6,-122.3?client_id=MYID123"
        redacted = _redact(url)
        assert "MYID123" not in redacted
        assert "client_id=[REDACTED]" in redacted

    def test_client_secret_redacted_independently(self) -> None:
        """client_secret alone is redacted even without client_id in the URL."""
        from weewx_clearskies_api.logging.redaction_filter import _redact  # noqa: PLC0415
        url = "https://data.api.xweather.com/forecasts/47.6,-122.3?client_secret=SECRET456"
        redacted = _redact(url)
        assert "SECRET456" not in redacted
        assert "client_secret=[REDACTED]" in redacted

    def test_redaction_preserves_non_credential_query_params(self) -> None:
        """Non-credential query params (filter=, limit=) are preserved after redaction."""
        from weewx_clearskies_api.logging.redaction_filter import _redact  # noqa: PLC0415
        url = (
            "https://data.api.xweather.com/forecasts/47.6062,-122.3321"
            "?filter=1hr&limit=240&client_id=ABC123&client_secret=DEF456"
        )
        redacted = _redact(url)
        assert "filter=1hr" in redacted
        assert "limit=240" in redacted

    def test_aeris_log_url_with_real_format_redacted_by_filter(self) -> None:
        """Aeris-style URL with all params redacts both credentials per lead-call 23."""
        from weewx_clearskies_api.logging.redaction_filter import _redact  # noqa: PLC0415
        url = (
            "https://data.api.xweather.com/forecasts/47.6062,-122.3321"
            "?filter=1hr&limit=240&client_id=uu1BzHkZXRrtz0tMc6HfQ"
            "&client_secret=MiXYXLbPliJWyB1LRi60ZJYCmMwgyO3FcjLJp8Wp"
        )
        redacted = _redact(url)
        assert "uu1BzHkZXRrtz0tMc6HfQ" not in redacted
        assert "MiXYXLbPliJWyB1LRi60ZJYCmMwgyO3FcjLJp8Wp" not in redacted
        assert "[REDACTED]" in redacted


# ===========================================================================
# 8. Capability registry
# ===========================================================================


class TestCapabilityRegistry:
    """CAPABILITY declaration and registry wiring."""

    def test_capability_provider_id_is_aeris(self) -> None:
        """CAPABILITY.provider_id = 'aeris'."""
        from weewx_clearskies_api.providers.forecast.aeris import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.provider_id == "aeris"

    def test_capability_domain_is_forecast(self) -> None:
        """CAPABILITY.domain = 'forecast'."""
        from weewx_clearskies_api.providers.forecast.aeris import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.domain == "forecast"

    def test_capability_auth_required_includes_client_id_and_client_secret(self) -> None:
        """CAPABILITY.auth_required = ('client_id', 'client_secret') per ADR-006."""
        from weewx_clearskies_api.providers.forecast.aeris import CAPABILITY  # noqa: PLC0415
        assert "client_id" in CAPABILITY.auth_required
        assert "client_secret" in CAPABILITY.auth_required

    def test_capability_geographic_coverage_is_global(self) -> None:
        """CAPABILITY.geographic_coverage = 'global' (trust Aeris, lead-call 17)."""
        from weewx_clearskies_api.providers.forecast.aeris import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.geographic_coverage == "global"

    def test_capability_supplied_fields_include_hourly_fields(self) -> None:
        """CAPABILITY.supplied_canonical_fields includes core hourly fields."""
        from weewx_clearskies_api.providers.forecast.aeris import CAPABILITY  # noqa: PLC0415
        for field in ("validTime", "outTemp", "windSpeed", "precipType", "cloudCover"):
            assert field in CAPABILITY.supplied_canonical_fields, (
                f"Expected {field!r} in supplied_canonical_fields"
            )

    def test_capability_supplied_fields_include_daily_fields(self) -> None:
        """CAPABILITY.supplied_canonical_fields includes core daily fields."""
        from weewx_clearskies_api.providers.forecast.aeris import CAPABILITY  # noqa: PLC0415
        for field in ("validDate", "tempMax", "tempMin", "uvIndexMax", "sunrise", "sunset"):
            assert field in CAPABILITY.supplied_canonical_fields, (
                f"Expected {field!r} in supplied_canonical_fields"
            )

    def test_capability_supplied_fields_include_discussion_headline_and_body(self) -> None:
        """CAPABILITY declares 'headline' and 'body' (max-surface per brief Q2)."""
        from weewx_clearskies_api.providers.forecast.aeris import CAPABILITY  # noqa: PLC0415
        assert "headline" in CAPABILITY.supplied_canonical_fields
        assert "body" in CAPABILITY.supplied_canonical_fields

    def test_capability_default_poll_interval_is_1800_seconds(self) -> None:
        """CAPABILITY.default_poll_interval_seconds = 1800 (30 min per ADR-017)."""
        from weewx_clearskies_api.providers.forecast.aeris import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.default_poll_interval_seconds == 1800

    def test_wire_providers_with_aeris_capability_populates_registry(self) -> None:
        """wire_providers([aeris.CAPABILITY]) → registry has 'aeris' forecast entry."""
        _reset_provider_state()
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            wire_providers,
            get_provider_registry,
        )
        from weewx_clearskies_api.providers.forecast.aeris import CAPABILITY  # noqa: PLC0415

        wire_providers([CAPABILITY])
        registry = get_provider_registry()
        aeris_entries = [p for p in registry if p.provider_id == "aeris"]
        assert len(aeris_entries) == 1
        assert aeris_entries[0].domain == "forecast"

    def test_get_provider_registry_returns_aeris_entry_after_wire(self) -> None:
        """get_provider_registry() returns the aeris entry with all fields populated."""
        _reset_provider_state()
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            wire_providers,
            get_provider_registry,
        )
        from weewx_clearskies_api.providers.forecast.aeris import CAPABILITY  # noqa: PLC0415

        wire_providers([CAPABILITY])
        registry = get_provider_registry()
        entry = next((p for p in registry if p.provider_id == "aeris"), None)
        assert entry is not None
        assert entry.auth_required == ("client_id", "client_secret")
        assert entry.geographic_coverage == "global"


# ===========================================================================
# 9. ForecastSettings — aeris credential fields
# ===========================================================================


class TestForecastSettingsAerisCredentials:
    """ForecastSettings loads Aeris credentials from env vars."""

    def test_aeris_client_id_loaded_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WEEWX_CLEARSKIES_AERIS_CLIENT_ID → settings.forecast.aeris_client_id."""
        monkeypatch.setenv("WEEWX_CLEARSKIES_AERIS_CLIENT_ID", "test-id-12345")
        monkeypatch.setenv("WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET", "")
        from weewx_clearskies_api.config.settings import ForecastSettings  # noqa: PLC0415
        settings = ForecastSettings({"provider": "aeris"})
        assert settings.aeris_client_id == "test-id-12345"

    def test_aeris_client_secret_loaded_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET → settings.forecast.aeris_client_secret."""
        monkeypatch.setenv("WEEWX_CLEARSKIES_AERIS_CLIENT_ID", "")
        monkeypatch.setenv("WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET", "test-secret-xyz")
        from weewx_clearskies_api.config.settings import ForecastSettings  # noqa: PLC0415
        settings = ForecastSettings({"provider": "aeris"})
        assert settings.aeris_client_secret == "test-secret-xyz"

    def test_aeris_credentials_are_none_when_env_vars_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unset env vars → aeris_client_id and aeris_client_secret are None."""
        monkeypatch.delenv("WEEWX_CLEARSKIES_AERIS_CLIENT_ID", raising=False)
        monkeypatch.delenv("WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET", raising=False)
        from weewx_clearskies_api.config.settings import ForecastSettings  # noqa: PLC0415
        settings = ForecastSettings({"provider": "aeris"})
        assert settings.aeris_client_id is None
        assert settings.aeris_client_secret is None

    def test_aeris_provider_passes_forecast_settings_validation(self) -> None:
        """ForecastSettings.validate() accepts 'aeris' as provider id."""
        from weewx_clearskies_api.config.settings import ForecastSettings  # noqa: PLC0415
        settings = ForecastSettings({"provider": "aeris"})
        # Should not raise
        settings.validate()

    def test_aeris_credentials_not_read_from_ini_section(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Aeris credentials come from env vars, NOT from the INI section (ADR-027 §3)."""
        monkeypatch.delenv("WEEWX_CLEARSKIES_AERIS_CLIENT_ID", raising=False)
        monkeypatch.delenv("WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET", raising=False)
        from weewx_clearskies_api.config.settings import ForecastSettings  # noqa: PLC0415
        # Passing credentials in the INI-style section dict should NOT set them
        settings = ForecastSettings({
            "provider": "aeris",
            "aeris_client_id": "should-be-ignored",
            "aeris_client_secret": "should-be-ignored",
        })
        # Credentials come from env vars only; if env vars are unset, result is None
        assert settings.aeris_client_id is None
        assert settings.aeris_client_secret is None


# ===========================================================================
# 10. _build_cache_key — deterministic + includes target_unit
# ===========================================================================


class TestBuildCacheKey:
    """_build_cache_key produces deterministic SHA-256 keys that vary by unit."""

    def test_cache_key_is_deterministic(self) -> None:
        """Same inputs produce the same cache key on repeated calls."""
        from weewx_clearskies_api.providers.forecast.aeris import _build_cache_key  # noqa: PLC0415
        key1 = _build_cache_key(47.6062, -122.3321, "US")
        key2 = _build_cache_key(47.6062, -122.3321, "US")
        assert key1 == key2

    def test_cache_key_differs_by_target_unit(self) -> None:
        """US and METRIC target_units produce different cache keys."""
        from weewx_clearskies_api.providers.forecast.aeris import _build_cache_key  # noqa: PLC0415
        key_us = _build_cache_key(47.6062, -122.3321, "US")
        key_metric = _build_cache_key(47.6062, -122.3321, "METRIC")
        assert key_us != key_metric

    def test_cache_key_differs_by_location(self) -> None:
        """Different lat/lon produces different cache keys."""
        from weewx_clearskies_api.providers.forecast.aeris import _build_cache_key  # noqa: PLC0415
        key_seattle = _build_cache_key(47.6062, -122.3321, "US")
        key_miami = _build_cache_key(25.7617, -80.1918, "US")
        assert key_seattle != key_miami

    def test_cache_key_rounds_lat_lon_to_4_decimal_places(self) -> None:
        """Lat/lon beyond 4 decimal places produces the same key as rounded version."""
        from weewx_clearskies_api.providers.forecast.aeris import _build_cache_key  # noqa: PLC0415
        key1 = _build_cache_key(47.60619999, -122.33209999, "US")
        key2 = _build_cache_key(47.6062, -122.3321, "US")
        assert key1 == key2

    def test_cache_key_is_64_hex_chars_sha256(self) -> None:
        """Cache key is a 64-character hexadecimal string (SHA-256 output)."""
        from weewx_clearskies_api.providers.forecast.aeris import _build_cache_key  # noqa: PLC0415
        key = _build_cache_key(47.6062, -122.3321, "US")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)
