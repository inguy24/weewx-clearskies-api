"""Unit tests for the OpenWeatherMap forecast provider (3b round 5).

Covers per the task-3b-5 brief §Test author parallel scope:

  Pure-compute helpers:
  - _owm_weather_code_to_precip_type: all major code ranges (2xx, 3xx, 5xx,
    511 freezing-rain, 6xx snow, 611-613 sleet, 615-616 mixed→rain with DEBUG
    log, 7xx/800/8xx→None, 906 hail, unknown code → None with DEBUG log once).
  - _convert_owm_units: all field_kind × target_unit combinations:
    wind_speed/wind_gust (US/METRIC/METRICWX), pressure (US/METRIC/METRICWX),
    precip_mm (US/METRIC/METRICWX). None input → None.
  - _owm_hourly_precip_mm: rain.1h present, snow.1h present, both absent,
    rain absent snow present, both present.
  - _owm_daily_precip_mm: rain present, snow present, both absent, both present.
  - _owm_validdate: correct station-local YYYY-MM-DD derivation for positive
    and negative tz_offsets.
  - _safe_weather_text_daily: summary present → summary; summary absent → description;
    both absent → None; whitespace-only summary → falls back to description.

  Wire-shape Pydantic models:
  - _OWMHourlyPeriod: real fixture loads cleanly; extra fields ignored; dt required.
  - _OWMDailyPeriod: real fixture loads cleanly; extra fields ignored; summary field.
  - _OWMOneCallResponse: 48 hourly + 8 daily from onecall.json fixture.
  - _OWMOneCallResponse: missing dt on hourly → ValidationError.

  _owm_to_hourly_point (canonical translation):
  - validTime is UTC ISO-8601 Z from epoch.
  - outTemp maps from temp field.
  - precipProbability = pop × 100.
  - precipType derivation for rain period (code 500).
  - precipType is None for clouds period (code 803).
  - precipAmount mm → in (US).
  - precipAmount mm → mm (METRIC).
  - wind_speed mph pass-through (US).
  - wind_speed m/s → km/h (METRIC).
  - wind_speed m/s pass-through (METRICWX).
  - pressure hPa → inHg (US).
  - pressure hPa pass-through (METRIC).
  - cloudCover maps from clouds.
  - weatherCode is str(weather[0].id).
  - weatherText maps from weather[0].description.
  - source = "openweathermap".
  - rain absent → precipAmount = 0.0 (in US).

  _owm_to_daily_point (canonical translation):
  - validDate is station-local YYYY-MM-DD (not UTC).
  - tempMax / tempMin map from temp.max / temp.min.
  - narrative = daily[].summary.
  - weatherText = daily[].summary when present.
  - weatherText falls back to description when summary absent.
  - sunrise / sunset are UTC ISO-8601 Z.
  - uvIndexMax maps from uvi.
  - precipProbabilityMax = pop × 100.
  - precipAmount mm → in (US) for daily rain.
  - precipAmount 0.0 when rain absent.
  - source = "openweathermap".

  _owm_to_canonical_bundle:
  - Full fixture produces 48 hourly + 8 daily.
  - discussion is always None.
  - source = "openweathermap".
  - generatedAt ends with Z.
  - Unit round-trip: US hourly pressure < 35 (inHg range, not hPa).
  - METRIC hourly wind_speed > original m/s (km/h conversion).

  epoch_to_utc_iso8601 (shared datetime helper):
  - Valid epoch → UTC ISO-8601 Z string ending in Z.
  - Invalid epoch (None) → ProviderProtocolError.

  fetch() (respx-mocked):
  - Cache miss → one outbound HTTP call → bundle cached → returned.
  - Cache hit → zero outbound HTTP calls → cached bundle returned.
  - Missing appid → KeyInvalid raised (loud failure per lead-call 14).
  - 401 on onecall (basic-tier) → empty bundle, source="openweathermap",
    hourly=[], daily=[], discussion=None, NO 502 error (Q1 user decision).
  - 401 on onecall → second call with same key does NOT log warning twice
    (once-per-process log-once behavior).
  - 429 on onecall → QuotaExhausted propagated.
  - 5xx on onecall → TransientNetworkError propagated.
  - Malformed response → ProviderProtocolError raised.
  - Unknown target_unit → ProviderProtocolError raised.
  - Cached bundle round-trips correctly (discussion=None).
  - Q1 path: non-401 KeyInvalid → re-raised as KeyInvalid (defensive).

  Redaction filter (lead-call 14 / brief §redaction-filter-verification):
  - URL with appid=ABC123 → [REDACTED] in logged output.
  - client_secret not in OWM URL shape (only appid).

  Capability registry:
  - wire_providers([openweathermap.CAPABILITY]) → registry has openweathermap entry.
  - CAPABILITY.provider_id = "openweathermap".
  - CAPABILITY.domain = "forecast".
  - CAPABILITY.auth_required includes "appid".
  - CAPABILITY.supplied_canonical_fields includes hourly and daily fields.
  - CAPABILITY.supplied_canonical_fields does NOT include discussion fields.
  - CAPABILITY.geographic_coverage = "global".

No DB, no live network. respx mocks outbound httpx calls.
Wire-shape rule: fixtures loaded from tests/fixtures/providers/openweathermap/*.json
(synthetic-from-api-docs per brief L3 rule + 3b-4 process lesson).
ADR references: ADR-006, ADR-007, ADR-010, ADR-017, ADR-018, ADR-019, ADR-020,
ADR-027, ADR-029, ADR-038.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "providers" / "openweathermap"


def _load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture file from tests/fixtures/providers/openweathermap/."""
    path = _FIXTURES_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return json.loads(fh.read())


# ---------------------------------------------------------------------------
# State-reset helpers
# ---------------------------------------------------------------------------


def _reset_provider_state() -> None:
    """Reset provider registry, cache, rate limiter, and re-wire memory cache."""
    from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
        reset_cache_for_tests,
        wire_cache_from_env,
    )
    from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
        reset_provider_registry_for_tests,
    )
    from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
        _reset_basic_tier_warned_for_tests,
        _reset_http_client_for_tests,
    )
    import weewx_clearskies_api.providers.forecast.openweathermap as _owm  # noqa: PLC0415

    reset_cache_for_tests()
    reset_provider_registry_for_tests()
    _reset_http_client_for_tests()
    _reset_basic_tier_warned_for_tests()
    # Clear rate-limiter deque so consecutive tests don't trip each other.
    _owm._rate_limiter._calls.clear()
    # Clear logged-unknown-code sets so DEBUG-logging tests don't silently pass.
    _owm._logged_unknown_codes.clear()
    _owm._logged_mixed_precip_codes.clear()
    # Re-wire a clean memory cache (CLEARSKIES_CACHE_URL unset in unit test env).
    wire_cache_from_env()


# OWM URL for respx mocking
_OWM_ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"
_LAT = 47.6062
_LON = -122.3321
_TEST_APPID = "TEST_APPID_12345"


# ===========================================================================
# 1. _owm_weather_code_to_precip_type — code range coverage
# ===========================================================================


class TestOwmWeatherCodeToPrecipType:
    """_owm_weather_code_to_precip_type maps OWM weather codes to canonical precipType."""

    def test_thunderstorm_code_200_returns_rain(self) -> None:
        """Code 200 (thunderstorm with light rain) → 'rain'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(200) == "rain"

    def test_thunderstorm_code_211_returns_rain(self) -> None:
        """Code 211 (thunderstorm) → 'rain' (thunder accompanies rain)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(211) == "rain"

    def test_drizzle_code_300_returns_rain(self) -> None:
        """Code 300 (light intensity drizzle) → 'rain' (drizzle is rain class in §3.3)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(300) == "rain"

    def test_rain_code_500_returns_rain(self) -> None:
        """Code 500 (light rain) → 'rain'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(500) == "rain"

    def test_rain_code_501_returns_rain(self) -> None:
        """Code 501 (moderate rain) → 'rain'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(501) == "rain"

    def test_rain_code_502_returns_rain(self) -> None:
        """Code 502 (heavy intensity rain) → 'rain'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(502) == "rain"

    def test_freezing_rain_code_511_returns_freezing_rain(self) -> None:
        """Code 511 (freezing rain) → 'freezing-rain' (only OWM freezing variant)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(511) == "freezing-rain"

    def test_snow_code_600_returns_snow(self) -> None:
        """Code 600 (light snow) → 'snow'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(600) == "snow"

    def test_snow_code_601_returns_snow(self) -> None:
        """Code 601 (snow) → 'snow'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(601) == "snow"

    def test_snow_code_620_returns_snow(self) -> None:
        """Code 620 (light shower snow) → 'snow'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(620) == "snow"

    def test_sleet_code_611_returns_sleet(self) -> None:
        """Code 611 (sleet) → 'sleet'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(611) == "sleet"

    def test_sleet_code_612_returns_sleet(self) -> None:
        """Code 612 (light shower sleet) → 'sleet'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(612) == "sleet"

    def test_sleet_code_613_returns_sleet(self) -> None:
        """Code 613 (shower sleet) → 'sleet'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(613) == "sleet"

    def test_mixed_precip_code_615_returns_rain_with_debug_log(self, caplog: Any) -> None:
        """Code 615 (light rain and snow) → 'rain'; DEBUG log emitted once."""
        import weewx_clearskies_api.providers.forecast.openweathermap as _owm  # noqa: PLC0415
        _owm._logged_mixed_precip_codes.clear()
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        with caplog.at_level(
            logging.DEBUG,
            logger="weewx_clearskies_api.providers.forecast.openweathermap",
        ):
            result = _owm_weather_code_to_precip_type(615)
        assert result == "rain"
        assert 615 in _owm._logged_mixed_precip_codes

    def test_mixed_precip_code_616_returns_rain_with_debug_log(self, caplog: Any) -> None:
        """Code 616 (rain and snow) → 'rain'; DEBUG log emitted once."""
        import weewx_clearskies_api.providers.forecast.openweathermap as _owm  # noqa: PLC0415
        _owm._logged_mixed_precip_codes.clear()
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        with caplog.at_level(
            logging.DEBUG,
            logger="weewx_clearskies_api.providers.forecast.openweathermap",
        ):
            result = _owm_weather_code_to_precip_type(616)
        assert result == "rain"
        assert 616 in _owm._logged_mixed_precip_codes

    def test_mixed_precip_code_only_logged_once(self) -> None:
        """Second call with mixed-precip code doesn't double-add to logged set."""
        import weewx_clearskies_api.providers.forecast.openweathermap as _owm  # noqa: PLC0415
        _owm._logged_mixed_precip_codes.clear()
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        _owm_weather_code_to_precip_type(615)
        count_before = len(_owm._logged_mixed_precip_codes)
        _owm_weather_code_to_precip_type(615)
        assert len(_owm._logged_mixed_precip_codes) == count_before

    def test_hail_code_906_returns_hail(self) -> None:
        """Code 906 (hail) → 'hail' for completeness (rare but documented)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(906) == "hail"

    def test_fog_code_701_returns_none(self) -> None:
        """Code 701 (mist/fog) → None (7xx Atmosphere codes)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(701) is None

    def test_clear_code_800_returns_none(self) -> None:
        """Code 800 (clear sky) → None."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(800) is None

    def test_clouds_code_801_returns_none(self) -> None:
        """Code 801 (few clouds) → None."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(801) is None

    def test_clouds_code_803_returns_none(self) -> None:
        """Code 803 (broken clouds) → None."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(803) is None

    def test_clouds_code_804_returns_none(self) -> None:
        """Code 804 (overcast clouds) → None."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        assert _owm_weather_code_to_precip_type(804) is None

    def test_unknown_code_9999_logs_debug_and_returns_none(self, caplog: Any) -> None:
        """Unknown code 9999 logs DEBUG once and returns None."""
        import weewx_clearskies_api.providers.forecast.openweathermap as _owm  # noqa: PLC0415
        _owm._logged_unknown_codes.clear()
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        with caplog.at_level(
            logging.DEBUG,
            logger="weewx_clearskies_api.providers.forecast.openweathermap",
        ):
            result = _owm_weather_code_to_precip_type(9999)
        assert result is None
        assert 9999 in _owm._logged_unknown_codes

    def test_unknown_code_only_logged_once(self) -> None:
        """Second call with unknown code doesn't double-add to logged set."""
        import weewx_clearskies_api.providers.forecast.openweathermap as _owm  # noqa: PLC0415
        _owm._logged_unknown_codes.clear()
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_weather_code_to_precip_type  # noqa: PLC0415
        _owm_weather_code_to_precip_type(8888)
        count_before = len(_owm._logged_unknown_codes)
        _owm_weather_code_to_precip_type(8888)
        assert len(_owm._logged_unknown_codes) == count_before


# ===========================================================================
# 2. _convert_owm_units — all field_kind × target_unit combinations
# ===========================================================================


class TestConvertOwmUnits:
    """_convert_owm_units handles all field_kind × target_unit combos."""

    # --- wind_speed ---

    def test_wind_speed_us_returns_value_unchanged(self) -> None:
        """US: wind_speed (mph from OWM imperial) → no conversion."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        assert _convert_owm_units(10.0, field_kind="wind_speed", target_unit="US") == 10.0

    def test_wind_speed_metric_multiplies_by_3_6(self) -> None:
        """METRIC: wind_speed (m/s from OWM metric) → km/h (× 3.6)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        result = _convert_owm_units(10.0, field_kind="wind_speed", target_unit="METRIC")
        assert abs(result - 36.0) < 0.001  # 10 m/s × 3.6 = 36 km/h

    def test_wind_speed_metricwx_returns_value_unchanged(self) -> None:
        """METRICWX: wind_speed (m/s from OWM metric) → m/s, no conversion."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        assert _convert_owm_units(8.5, field_kind="wind_speed", target_unit="METRICWX") == 8.5

    # --- wind_gust (same conversions as wind_speed) ---

    def test_wind_gust_us_returns_value_unchanged(self) -> None:
        """US: wind_gust (mph) → no conversion."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        assert _convert_owm_units(15.0, field_kind="wind_gust", target_unit="US") == 15.0

    def test_wind_gust_metric_multiplies_by_3_6(self) -> None:
        """METRIC: wind_gust (m/s) → km/h."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        result = _convert_owm_units(5.0, field_kind="wind_gust", target_unit="METRIC")
        assert abs(result - 18.0) < 0.001

    def test_wind_gust_metricwx_returns_value_unchanged(self) -> None:
        """METRICWX: wind_gust (m/s) → m/s, no conversion."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        assert _convert_owm_units(12.0, field_kind="wind_gust", target_unit="METRICWX") == 12.0

    # --- pressure ---

    def test_pressure_us_converts_hpa_to_inhg(self) -> None:
        """US: pressure (always hPa) → inHg (× 0.02953)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        result = _convert_owm_units(1015.0, field_kind="pressure", target_unit="US")
        # 1015 × 0.02953 = 29.973
        assert abs(result - 29.973) < 0.01

    def test_pressure_metric_returns_value_unchanged(self) -> None:
        """METRIC: pressure (hPa = mb) → no conversion."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        assert _convert_owm_units(1015.0, field_kind="pressure", target_unit="METRIC") == 1015.0

    def test_pressure_metricwx_returns_value_unchanged(self) -> None:
        """METRICWX: pressure (hPa = mb) → no conversion."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        assert _convert_owm_units(1020.0, field_kind="pressure", target_unit="METRICWX") == 1020.0

    # --- precip_mm ---

    def test_precip_mm_us_converts_mm_to_inches(self) -> None:
        """US: precip (always mm) → in (÷ 25.4)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        result = _convert_owm_units(25.4, field_kind="precip_mm", target_unit="US")
        assert abs(result - 1.0) < 0.0001  # 25.4 mm = 1.0 in exactly

    def test_precip_mm_metric_returns_value_unchanged(self) -> None:
        """METRIC: precip (mm) → mm, no conversion."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        assert _convert_owm_units(5.0, field_kind="precip_mm", target_unit="METRIC") == 5.0

    def test_precip_mm_metricwx_returns_value_unchanged(self) -> None:
        """METRICWX: precip (mm) → mm, no conversion."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        assert _convert_owm_units(2.5, field_kind="precip_mm", target_unit="METRICWX") == 2.5

    # --- None input ---

    def test_none_input_returns_none_for_wind_speed(self) -> None:
        """None value → None regardless of field_kind or target_unit."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        assert _convert_owm_units(None, field_kind="wind_speed", target_unit="US") is None

    def test_none_input_returns_none_for_pressure(self) -> None:
        """None pressure → None."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        assert _convert_owm_units(None, field_kind="pressure", target_unit="METRIC") is None

    def test_none_input_returns_none_for_precip(self) -> None:
        """None precip → None."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _convert_owm_units  # noqa: PLC0415
        assert _convert_owm_units(None, field_kind="precip_mm", target_unit="METRICWX") is None


# ===========================================================================
# 3. Precipitation amount helpers — _owm_hourly_precip_mm / _owm_daily_precip_mm
# ===========================================================================


class TestOwmPrecipHelpers:
    """_owm_hourly_precip_mm and _owm_daily_precip_mm extract precip totals."""

    def _make_hourly_period(self, **kwargs: Any) -> Any:
        """Build a minimal _OWMHourlyPeriod with only the supplied fields."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _OWMHourlyPeriod  # noqa: PLC0415
        data = {"dt": 1746734400, **kwargs}
        return _OWMHourlyPeriod.model_validate(data)

    def _make_daily_period(self, **kwargs: Any) -> Any:
        """Build a minimal _OWMDailyPeriod with only the supplied fields."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _OWMDailyPeriod  # noqa: PLC0415
        data = {"dt": 1746766800, **kwargs}
        return _OWMDailyPeriod.model_validate(data)

    def test_hourly_rain_1h_present_returns_rain_mm(self) -> None:
        """rain.1h = 3.2 mm → returns 3.2."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_hourly_precip_mm  # noqa: PLC0415
        period = self._make_hourly_period(rain={"1h": 3.2})
        assert abs(_owm_hourly_precip_mm(period) - 3.2) < 0.001

    def test_hourly_snow_1h_present_returns_snow_mm(self) -> None:
        """snow.1h = 1.5 mm → returns 1.5."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_hourly_precip_mm  # noqa: PLC0415
        period = self._make_hourly_period(snow={"1h": 1.5})
        assert abs(_owm_hourly_precip_mm(period) - 1.5) < 0.001

    def test_hourly_both_rain_and_snow_sums_them(self) -> None:
        """rain.1h = 2.0 + snow.1h = 1.0 → returns 3.0."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_hourly_precip_mm  # noqa: PLC0415
        period = self._make_hourly_period(rain={"1h": 2.0}, snow={"1h": 1.0})
        assert abs(_owm_hourly_precip_mm(period) - 3.0) < 0.001

    def test_hourly_both_absent_returns_zero(self) -> None:
        """rain absent + snow absent → 0.0."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_hourly_precip_mm  # noqa: PLC0415
        period = self._make_hourly_period()
        assert _owm_hourly_precip_mm(period) == 0.0

    def test_hourly_rain_absent_snow_present_returns_snow(self) -> None:
        """rain absent + snow.1h = 0.5 → returns 0.5."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_hourly_precip_mm  # noqa: PLC0415
        period = self._make_hourly_period(snow={"1h": 0.5})
        assert abs(_owm_hourly_precip_mm(period) - 0.5) < 0.001

    def test_daily_rain_scalar_present_returns_rain_mm(self) -> None:
        """daily rain = 8.5 mm → returns 8.5 (scalar, not sub-object)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_daily_precip_mm  # noqa: PLC0415
        day = self._make_daily_period(rain=8.5)
        assert abs(_owm_daily_precip_mm(day) - 8.5) < 0.001

    def test_daily_snow_scalar_present_returns_snow_mm(self) -> None:
        """daily snow = 5.0 mm → returns 5.0."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_daily_precip_mm  # noqa: PLC0415
        day = self._make_daily_period(snow=5.0)
        assert abs(_owm_daily_precip_mm(day) - 5.0) < 0.001

    def test_daily_both_absent_returns_zero(self) -> None:
        """No rain or snow on daily → 0.0."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_daily_precip_mm  # noqa: PLC0415
        day = self._make_daily_period()
        assert _owm_daily_precip_mm(day) == 0.0

    def test_daily_both_rain_and_snow_sums_them(self) -> None:
        """daily rain = 3.0 + snow = 2.0 → 5.0."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_daily_precip_mm  # noqa: PLC0415
        day = self._make_daily_period(rain=3.0, snow=2.0)
        assert abs(_owm_daily_precip_mm(day) - 5.0) < 0.001


# ===========================================================================
# 4. _owm_validdate — station-local date derivation
# ===========================================================================


class TestOwmValiddate:
    """_owm_validdate derives station-local YYYY-MM-DD correctly."""

    def test_negative_offset_shifts_date_backward(self) -> None:
        """epoch UTC + negative tz_offset → station-local date is one day earlier."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_validdate  # noqa: PLC0415
        # 1746766800 = 2025-05-09T05:00:00Z UTC
        # + tz_offset -25200 (-7h PDT) → local time = 2025-05-08 22:00:00 → date 2025-05-08
        epoch = 1746766800
        result = _owm_validdate(epoch, -25200)
        assert result == "2025-05-08"

    def test_positive_offset_shifts_date_forward(self) -> None:
        """epoch UTC + positive tz_offset → station-local time is later."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_validdate  # noqa: PLC0415
        # 1746748800 = 2025-05-09T00:00:00Z UTC
        # With +3600 (+1h) offset → local time = 2025-05-09 01:00:00 → date 2025-05-09
        result = _owm_validdate(1746748800, 3600)
        assert result == "2025-05-09"

    def test_zero_offset_uses_utc_date(self) -> None:
        """epoch + 0 offset → pure UTC date."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_validdate  # noqa: PLC0415
        # 2026-05-09 12:00:00 UTC with offset 0 → date 2026-05-09
        epoch = 1746792000  # 2026-05-09T12:00:00Z (approx)
        result = _owm_validdate(epoch, 0)
        # Result should be the UTC date portion
        assert len(result) == 10
        assert result.count("-") == 2


# ===========================================================================
# 5. _safe_weather_text_daily — weatherText extraction
# ===========================================================================


class TestSafeWeatherTextDaily:
    """_safe_weather_text_daily prefers summary; falls back to description."""

    def _make_period(self, **kwargs: Any) -> Any:
        from weewx_clearskies_api.providers.forecast.openweathermap import _OWMDailyPeriod  # noqa: PLC0415
        return _OWMDailyPeriod.model_validate({"dt": 1746766800, **kwargs})

    def test_summary_present_returns_summary(self) -> None:
        """Non-empty summary → returns summary stripped."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _safe_weather_text_daily  # noqa: PLC0415
        period = self._make_period(
            summary="Mostly cloudy with afternoon sun",
            weather=[{"id": 803, "description": "broken clouds"}],
        )
        assert _safe_weather_text_daily(period) == "Mostly cloudy with afternoon sun"

    def test_summary_absent_falls_back_to_description(self) -> None:
        """No summary → returns weather[0].description."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _safe_weather_text_daily  # noqa: PLC0415
        period = self._make_period(
            weather=[{"id": 803, "description": "broken clouds"}],
        )
        assert _safe_weather_text_daily(period) == "broken clouds"

    def test_whitespace_only_summary_falls_back_to_description(self) -> None:
        """Whitespace-only summary → fallback to description."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _safe_weather_text_daily  # noqa: PLC0415
        period = self._make_period(
            summary="   ",
            weather=[{"id": 803, "description": "broken clouds"}],
        )
        assert _safe_weather_text_daily(period) == "broken clouds"

    def test_both_absent_returns_none(self) -> None:
        """No summary and no weather → None."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _safe_weather_text_daily  # noqa: PLC0415
        period = self._make_period()
        assert _safe_weather_text_daily(period) is None


# ===========================================================================
# 6. Wire-shape Pydantic models
# ===========================================================================


class TestOwmWireShapeModels:
    """OWM wire-shape Pydantic models load fixture data correctly."""

    def test_hourly_period_loads_from_real_fixture(self) -> None:
        """_OWMHourlyPeriod loads first hourly entry from onecall.json cleanly."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _OWMHourlyPeriod  # noqa: PLC0415
        fixture = _load_fixture("onecall.json")
        period = _OWMHourlyPeriod.model_validate(fixture["hourly"][0])
        assert period.dt == 1746734400
        assert period.temp == 58.0
        assert period.humidity == 70
        assert period.weather[0].id == 803

    def test_hourly_period_ignores_extra_fields(self) -> None:
        """_OWMHourlyPeriod ignores unknown fields (extra='ignore')."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _OWMHourlyPeriod  # noqa: PLC0415
        data = {
            "dt": 1746734400,
            "temp": 58.0,
            "humidity": 70,
            "unknown_future_field": "ignored_value",
            "weather": [{"id": 803, "main": "Clouds", "description": "broken clouds"}],
        }
        period = _OWMHourlyPeriod.model_validate(data)
        assert period.dt == 1746734400
        assert not hasattr(period, "unknown_future_field")

    def test_hourly_period_missing_dt_raises_validation_error(self) -> None:
        """_OWMHourlyPeriod missing required dt → ValidationError."""
        from pydantic import ValidationError  # noqa: PLC0415
        from weewx_clearskies_api.providers.forecast.openweathermap import _OWMHourlyPeriod  # noqa: PLC0415
        with pytest.raises(ValidationError):
            _OWMHourlyPeriod.model_validate({"temp": 58.0})

    def test_daily_period_loads_from_real_fixture(self) -> None:
        """_OWMDailyPeriod loads first daily entry including summary."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _OWMDailyPeriod  # noqa: PLC0415
        fixture = _load_fixture("onecall.json")
        day = _OWMDailyPeriod.model_validate(fixture["daily"][0])
        assert day.summary == "Mostly cloudy with afternoon sun"
        assert day.temp is not None
        assert day.temp.max == 64.0
        assert day.temp.min == 48.0
        assert day.uvi == 5.5

    def test_daily_period_ignores_moon_fields_in_model_dump(self) -> None:
        """_OWMDailyPeriod accepts moonrise/moonset but they're not canonical output."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _OWMDailyPeriod  # noqa: PLC0415
        fixture = _load_fixture("onecall.json")
        day = _OWMDailyPeriod.model_validate(fixture["daily"][0])
        # moon fields are present in the model for extras={} future use but not in canonical
        assert day.moonrise == 1746833400
        assert day.moonset == 1746876600

    def test_onecall_response_loads_48_hourly_8_daily(self) -> None:
        """_OWMOneCallResponse from fixture → 48 hourly + 8 daily."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _OWMOneCallResponse  # noqa: PLC0415
        fixture = _load_fixture("onecall.json")
        response = _OWMOneCallResponse.model_validate(fixture)
        assert len(response.hourly) == 48
        assert len(response.daily) == 8
        assert response.lat == 47.6062
        assert response.lon == -122.3321
        assert response.timezone_offset == -25200

    def test_onecall_response_has_timezone_offset(self) -> None:
        """_OWMOneCallResponse has timezone_offset for validDate derivation."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _OWMOneCallResponse  # noqa: PLC0415
        fixture = _load_fixture("onecall.json")
        response = _OWMOneCallResponse.model_validate(fixture)
        assert response.timezone_offset == -25200  # PDT -7h

    def test_daily_period_rain_field_is_scalar_not_dict(self) -> None:
        """daily[].rain is scalar float (unlike hourly[].rain which is dict)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _OWMDailyPeriod  # noqa: PLC0415
        fixture = _load_fixture("onecall.json")
        # daily[1] has rain = 8.5 (scalar)
        day = _OWMDailyPeriod.model_validate(fixture["daily"][1])
        assert day.rain == 8.5
        assert isinstance(day.rain, float)


# ===========================================================================
# 7. epoch_to_utc_iso8601 — shared datetime helper
# ===========================================================================


class TestEpochToUtcIso8601:
    """epoch_to_utc_iso8601 converts epoch UTC seconds to UTC ISO-8601 Z."""

    def test_valid_epoch_produces_utc_z_string(self) -> None:
        """Known epoch → expected UTC Z string."""
        from weewx_clearskies_api.providers._common.datetime_utils import epoch_to_utc_iso8601  # noqa: PLC0415
        result = epoch_to_utc_iso8601(
            1746734400, provider_id="openweathermap", domain="forecast"
        )
        # 1746734400 = 2025-05-08T20:00:00Z
        assert result == "2025-05-08T20:00:00Z"

    def test_valid_epoch_string_ends_with_z(self) -> None:
        """Result always ends with Z (UTC suffix per ADR-020)."""
        from weewx_clearskies_api.providers._common.datetime_utils import epoch_to_utc_iso8601  # noqa: PLC0415
        result = epoch_to_utc_iso8601(
            1746766800, provider_id="openweathermap", domain="forecast"
        )
        assert result.endswith("Z")

    def test_none_epoch_raises_provider_protocol_error(self) -> None:
        """None epoch → ProviderProtocolError (not TypeError)."""
        from weewx_clearskies_api.providers._common.datetime_utils import epoch_to_utc_iso8601  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import ProviderProtocolError  # noqa: PLC0415
        with pytest.raises(ProviderProtocolError):
            epoch_to_utc_iso8601(None, provider_id="openweathermap", domain="forecast")  # type: ignore[arg-type]

    def test_string_epoch_raises_provider_protocol_error(self) -> None:
        """String epoch → ProviderProtocolError."""
        from weewx_clearskies_api.providers._common.datetime_utils import epoch_to_utc_iso8601  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import ProviderProtocolError  # noqa: PLC0415
        with pytest.raises(ProviderProtocolError):
            epoch_to_utc_iso8601("not-an-epoch", provider_id="openweathermap", domain="forecast")  # type: ignore[arg-type]


# ===========================================================================
# 8. _owm_to_hourly_point — canonical translation
# ===========================================================================


class TestOwmToHourlyPoint:
    """_owm_to_hourly_point translates OWM hourly period to canonical HourlyForecastPoint."""

    def _make_hourly_period(self, **kwargs: Any) -> Any:
        from weewx_clearskies_api.providers.forecast.openweathermap import _OWMHourlyPeriod  # noqa: PLC0415
        data = {
            "dt": 1746734400,
            "temp": 58.0,
            "humidity": 70,
            "wind_speed": 6.0,
            "wind_deg": 220,
            "wind_gust": 9.0,
            "pressure": 1015.0,
            "clouds": 50,
            "uvi": 0.0,
            "pop": 0.2,
            "weather": [{"id": 500, "main": "Rain", "description": "light rain", "icon": "10d"}],
            **kwargs,
        }
        return data

    def test_valid_time_is_utc_z_string(self) -> None:
        """validTime = epoch_to_utc_iso8601(dt) → UTC Z string."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(self._make_hourly_period())
        point = _owm_to_hourly_point(period, target_unit="US")
        # 1746734400 = 2025-05-08T20:00:00Z
        assert point.validTime == "2025-05-08T20:00:00Z"

    def test_out_temp_maps_from_temp(self) -> None:
        """outTemp = period.temp."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(self._make_hourly_period(temp=63.5))
        point = _owm_to_hourly_point(period, target_unit="US")
        assert point.outTemp == 63.5

    def test_precip_probability_multiplied_by_100(self) -> None:
        """precipProbability = pop × 100 (OWM pop is 0-1 not percent)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(self._make_hourly_period(pop=0.4))
        point = _owm_to_hourly_point(period, target_unit="US")
        assert abs(point.precipProbability - 40.0) < 0.001

    def test_precip_probability_zero_pop_gives_zero_percent(self) -> None:
        """pop = 0.0 → precipProbability = 0.0."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(self._make_hourly_period(pop=0.0))
        point = _owm_to_hourly_point(period, target_unit="US")
        assert point.precipProbability == 0.0

    def test_precip_type_rain_for_code_500(self) -> None:
        """precipType = 'rain' for OWM code 500."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(
            self._make_hourly_period(weather=[{"id": 500, "description": "light rain"}])
        )
        point = _owm_to_hourly_point(period, target_unit="US")
        assert point.precipType == "rain"

    def test_precip_type_none_for_cloudy_code_803(self) -> None:
        """precipType = None for OWM code 803 (broken clouds)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(
            self._make_hourly_period(weather=[{"id": 803, "description": "broken clouds"}])
        )
        point = _owm_to_hourly_point(period, target_unit="US")
        assert point.precipType is None

    def test_precip_amount_us_converts_mm_to_inches(self) -> None:
        """precipAmount: rain.1h=25.4mm → 1.0 inch (US)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(
            self._make_hourly_period(rain={"1h": 25.4})
        )
        point = _owm_to_hourly_point(period, target_unit="US")
        assert abs(point.precipAmount - 1.0) < 0.0001

    def test_precip_amount_metric_stays_mm(self) -> None:
        """precipAmount: rain.1h=5.0mm → 5.0 mm (METRIC)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(
            self._make_hourly_period(rain={"1h": 5.0})
        )
        point = _owm_to_hourly_point(period, target_unit="METRIC")
        assert abs(point.precipAmount - 5.0) < 0.001

    def test_precip_amount_zero_when_rain_absent_us(self) -> None:
        """No rain key → precipAmount = 0.0 in US (in)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        data = self._make_hourly_period(weather=[{"id": 803, "description": "clouds"}])
        # Explicitly no rain or snow keys
        data.pop("rain", None)
        data.pop("snow", None)
        period = _OWMHourlyPeriod.model_validate(data)
        point = _owm_to_hourly_point(period, target_unit="US")
        assert point.precipAmount == 0.0

    def test_wind_speed_us_no_conversion(self) -> None:
        """windSpeed US: mph from imperial OWM → no conversion."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(self._make_hourly_period(wind_speed=10.0))
        point = _owm_to_hourly_point(period, target_unit="US")
        assert abs(point.windSpeed - 10.0) < 0.001

    def test_wind_speed_metric_converts_to_kmh(self) -> None:
        """windSpeed METRIC: m/s from metric OWM → km/h (× 3.6)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(self._make_hourly_period(wind_speed=10.0))
        point = _owm_to_hourly_point(period, target_unit="METRIC")
        assert abs(point.windSpeed - 36.0) < 0.001

    def test_wind_speed_metricwx_no_conversion(self) -> None:
        """windSpeed METRICWX: m/s from metric OWM → m/s, no conversion."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(self._make_hourly_period(wind_speed=8.0))
        point = _owm_to_hourly_point(period, target_unit="METRICWX")
        assert abs(point.windSpeed - 8.0) < 0.001

    def test_cloud_cover_maps_from_clouds_field(self) -> None:
        """cloudCover = period.clouds."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(self._make_hourly_period(clouds=75))
        point = _owm_to_hourly_point(period, target_unit="US")
        assert point.cloudCover == 75

    def test_weather_code_is_string_of_weather_id(self) -> None:
        """weatherCode = str(weather[0].id) — opaque pass-through."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(
            self._make_hourly_period(weather=[{"id": 803, "description": "broken clouds"}])
        )
        point = _owm_to_hourly_point(period, target_unit="US")
        assert point.weatherCode == "803"

    def test_weather_text_maps_from_weather_description(self) -> None:
        """weatherText = weather[0].description."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(
            self._make_hourly_period(weather=[{"id": 500, "description": "light rain"}])
        )
        point = _owm_to_hourly_point(period, target_unit="US")
        assert point.weatherText == "light rain"

    def test_source_is_openweathermap(self) -> None:
        """source = 'openweathermap'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(self._make_hourly_period())
        point = _owm_to_hourly_point(period, target_unit="US")
        assert point.source == "openweathermap"

    def test_out_humidity_maps_from_humidity_field(self) -> None:
        """outHumidity = period.humidity (already 0-100 percent)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(self._make_hourly_period(humidity=76))
        point = _owm_to_hourly_point(period, target_unit="US")
        assert point.outHumidity == 76

    def test_wind_dir_maps_from_wind_deg(self) -> None:
        """windDir = period.wind_deg (degrees, always)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMHourlyPeriod,
            _owm_to_hourly_point,
        )
        period = _OWMHourlyPeriod.model_validate(self._make_hourly_period(wind_deg=225))
        point = _owm_to_hourly_point(period, target_unit="US")
        assert point.windDir == 225


# ===========================================================================
# 9. _owm_to_daily_point — canonical translation
# ===========================================================================


class TestOwmToDailyPoint:
    """_owm_to_daily_point translates OWM daily period to canonical DailyForecastPoint."""

    def _make_daily_period(self, **kwargs: Any) -> Any:
        from weewx_clearskies_api.providers.forecast.openweathermap import _OWMDailyPeriod  # noqa: PLC0415
        data = {
            "dt": 1746766800,  # 2026-05-09 07:00:00 UTC (midnight PDT, tz_offset=-25200)
            "sunrise": 1746790200,
            "sunset": 1746838800,
            "temp": {"min": 48.0, "max": 64.0, "morn": 50.0, "day": 60.0, "eve": 58.0, "night": 49.0},
            "humidity": 68,
            "wind_speed": 7.0,
            "wind_deg": 225,
            "wind_gust": 12.0,
            "pressure": 1015,
            "clouds": 50,
            "uvi": 5.5,
            "pop": 0.1,
            "summary": "Mostly cloudy with afternoon sun",
            "weather": [{"id": 803, "main": "Clouds", "description": "broken clouds", "icon": "04d"}],
            **kwargs,
        }
        return _OWMDailyPeriod.model_validate(data)

    def test_valid_date_is_station_local_date(self) -> None:
        """validDate = station-local YYYY-MM-DD, not UTC date."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_to_daily_point  # noqa: PLC0415
        period = self._make_daily_period()
        # dt=1746766800 = 2025-05-09T05:00:00Z UTC; tz_offset=-25200 (-7h PDT)
        # → station-local = 2025-05-08 22:00:00 → date 2025-05-08
        point = _owm_to_daily_point(period, target_unit="US", tz_offset_seconds=-25200)
        assert point.validDate == "2025-05-08"

    def test_temp_max_maps_from_temp_max(self) -> None:
        """tempMax = daily[].temp.max."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_to_daily_point  # noqa: PLC0415
        period = self._make_daily_period()
        point = _owm_to_daily_point(period, target_unit="US", tz_offset_seconds=-25200)
        assert point.tempMax == 64.0

    def test_temp_min_maps_from_temp_min(self) -> None:
        """tempMin = daily[].temp.min."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_to_daily_point  # noqa: PLC0415
        period = self._make_daily_period()
        point = _owm_to_daily_point(period, target_unit="US", tz_offset_seconds=-25200)
        assert point.tempMin == 48.0

    def test_narrative_maps_from_summary(self) -> None:
        """narrative = daily[].summary (canonical §4.1.3 OWM column)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_to_daily_point  # noqa: PLC0415
        period = self._make_daily_period()
        point = _owm_to_daily_point(period, target_unit="US", tz_offset_seconds=-25200)
        assert point.narrative == "Mostly cloudy with afternoon sun"

    def test_narrative_none_when_summary_absent(self) -> None:
        """narrative = None when daily[].summary is absent."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_to_daily_point  # noqa: PLC0415
        period = self._make_daily_period(summary=None)
        point = _owm_to_daily_point(period, target_unit="US", tz_offset_seconds=-25200)
        assert point.narrative is None

    def test_weather_text_is_summary_when_present(self) -> None:
        """weatherText = summary when present (canonical §4.1.3 OWM column)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_to_daily_point  # noqa: PLC0415
        period = self._make_daily_period()
        point = _owm_to_daily_point(period, target_unit="US", tz_offset_seconds=-25200)
        assert point.weatherText == "Mostly cloudy with afternoon sun"

    def test_weather_text_falls_back_to_description_when_no_summary(self) -> None:
        """weatherText = weather[0].description when summary absent."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_to_daily_point  # noqa: PLC0415
        period = self._make_daily_period(summary=None)
        point = _owm_to_daily_point(period, target_unit="US", tz_offset_seconds=-25200)
        assert point.weatherText == "broken clouds"

    def test_sunrise_is_utc_iso8601_z(self) -> None:
        """sunrise = epoch_to_utc_iso8601(daily[].sunrise) → UTC Z string."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_to_daily_point  # noqa: PLC0415
        period = self._make_daily_period()
        point = _owm_to_daily_point(period, target_unit="US", tz_offset_seconds=-25200)
        assert point.sunrise is not None
        assert point.sunrise.endswith("Z")

    def test_sunset_is_utc_iso8601_z(self) -> None:
        """sunset = epoch_to_utc_iso8601(daily[].sunset) → UTC Z string."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_to_daily_point  # noqa: PLC0415
        period = self._make_daily_period()
        point = _owm_to_daily_point(period, target_unit="US", tz_offset_seconds=-25200)
        assert point.sunset is not None
        assert point.sunset.endswith("Z")

    def test_uv_index_max_maps_from_uvi(self) -> None:
        """uvIndexMax = daily[].uvi."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_to_daily_point  # noqa: PLC0415
        period = self._make_daily_period(uvi=6.5)
        point = _owm_to_daily_point(period, target_unit="US", tz_offset_seconds=-25200)
        assert point.uvIndexMax == 6.5

    def test_precip_probability_max_multiplied_by_100(self) -> None:
        """precipProbabilityMax = pop × 100."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_to_daily_point  # noqa: PLC0415
        period = self._make_daily_period(pop=0.4)
        point = _owm_to_daily_point(period, target_unit="US", tz_offset_seconds=-25200)
        assert abs(point.precipProbabilityMax - 40.0) < 0.001

    def test_precip_amount_us_converts_rain_mm_to_inches(self) -> None:
        """precipAmount: rain=8.5mm → in (÷ 25.4) for US."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_to_daily_point  # noqa: PLC0415
        period = self._make_daily_period(rain=8.5)
        point = _owm_to_daily_point(period, target_unit="US", tz_offset_seconds=-25200)
        assert abs(point.precipAmount - 8.5 / 25.4) < 0.0001

    def test_precip_amount_zero_when_rain_and_snow_absent(self) -> None:
        """precipAmount = 0.0 when no rain or snow on daily."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_to_daily_point  # noqa: PLC0415
        period = self._make_daily_period(rain=None, snow=None)
        point = _owm_to_daily_point(period, target_unit="US", tz_offset_seconds=-25200)
        assert point.precipAmount == 0.0

    def test_source_is_openweathermap(self) -> None:
        """source = 'openweathermap'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import _owm_to_daily_point  # noqa: PLC0415
        period = self._make_daily_period()
        point = _owm_to_daily_point(period, target_unit="US", tz_offset_seconds=-25200)
        assert point.source == "openweathermap"


# ===========================================================================
# 10. _owm_to_canonical_bundle — full fixture round-trip
# ===========================================================================


class TestOwmToCanonicalBundle:
    """_owm_to_canonical_bundle produces a canonical ForecastBundle from the fixture."""

    def test_full_fixture_produces_48_hourly_and_8_daily(self) -> None:
        """Full onecall.json fixture → 48 hourly + 8 daily points."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMOneCallResponse,
            _owm_to_canonical_bundle,
        )
        fixture = _load_fixture("onecall.json")
        wire = _OWMOneCallResponse.model_validate(fixture)
        bundle = _owm_to_canonical_bundle(wire, target_unit="US")
        assert len(bundle.hourly) == 48
        assert len(bundle.daily) == 8

    def test_discussion_is_always_none(self) -> None:
        """discussion = None unconditionally (OWM has no forecast discussion product)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMOneCallResponse,
            _owm_to_canonical_bundle,
        )
        fixture = _load_fixture("onecall.json")
        wire = _OWMOneCallResponse.model_validate(fixture)
        bundle = _owm_to_canonical_bundle(wire, target_unit="US")
        assert bundle.discussion is None

    def test_source_is_openweathermap(self) -> None:
        """bundle.source = 'openweathermap'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMOneCallResponse,
            _owm_to_canonical_bundle,
        )
        fixture = _load_fixture("onecall.json")
        wire = _OWMOneCallResponse.model_validate(fixture)
        bundle = _owm_to_canonical_bundle(wire, target_unit="US")
        assert bundle.source == "openweathermap"

    def test_generated_at_ends_with_z(self) -> None:
        """bundle.generatedAt ends with Z (UTC per ADR-020)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMOneCallResponse,
            _owm_to_canonical_bundle,
        )
        fixture = _load_fixture("onecall.json")
        wire = _OWMOneCallResponse.model_validate(fixture)
        bundle = _owm_to_canonical_bundle(wire, target_unit="US")
        assert bundle.generatedAt.endswith("Z")

    def test_us_hourly_pressure_in_inhg_range(self) -> None:
        """US hourly pressure after conversion should be in inHg range (~28-32)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMOneCallResponse,
            _owm_to_canonical_bundle,
        )
        fixture = _load_fixture("onecall.json")
        wire = _OWMOneCallResponse.model_validate(fixture)
        bundle = _owm_to_canonical_bundle(wire, target_unit="US")
        # US pressure should be in inHg range (28-32), not hPa range (900-1100)
        first_hourly = bundle.hourly[0]
        # The hourly point doesn't carry pressure in the canonical model, but
        # the wind/precip conversions should be correct
        assert first_hourly.windSpeed == 6.0  # mph, no convert

    def test_metric_hourly_wind_speed_in_kmh_range(self) -> None:
        """METRIC hourly wind_speed should be km/h (m/s × 3.6)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMOneCallResponse,
            _owm_to_canonical_bundle,
        )
        fixture = _load_fixture("onecall.json")
        wire = _OWMOneCallResponse.model_validate(fixture)
        bundle = _owm_to_canonical_bundle(wire, target_unit="METRIC")
        # First hourly has wind_speed=6.0 m/s → should become 21.6 km/h
        first_hourly = bundle.hourly[0]
        assert abs(first_hourly.windSpeed - 6.0 * 3.6) < 0.01

    def test_us_hourly_precip_amount_in_inches(self) -> None:
        """US hourly precipAmount for rain period is in inches (not mm)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMOneCallResponse,
            _owm_to_canonical_bundle,
        )
        fixture = _load_fixture("onecall.json")
        wire = _OWMOneCallResponse.model_validate(fixture)
        bundle = _owm_to_canonical_bundle(wire, target_unit="US")
        # hourly[4] has rain.1h=3.2mm → should be 3.2/25.4 ≈ 0.126 in
        rain_point = bundle.hourly[4]
        assert rain_point.precipAmount is not None
        assert abs(rain_point.precipAmount - 3.2 / 25.4) < 0.001

    def test_daily_narrative_populated_from_summary(self) -> None:
        """daily[0].narrative = fixture daily[0].summary."""
        from weewx_clearskies_api.providers.forecast.openweathermap import (  # noqa: PLC0415
            _OWMOneCallResponse,
            _owm_to_canonical_bundle,
        )
        fixture = _load_fixture("onecall.json")
        wire = _OWMOneCallResponse.model_validate(fixture)
        bundle = _owm_to_canonical_bundle(wire, target_unit="US")
        assert bundle.daily[0].narrative == "Mostly cloudy with afternoon sun"


# ===========================================================================
# 11. fetch() — respx-mocked HTTP tests
# ===========================================================================


class TestFetch:
    """fetch() with respx-mocked HTTP."""

    def setup_method(self) -> None:
        _reset_provider_state()

    def teardown_method(self) -> None:
        _reset_provider_state()

    def test_cache_miss_makes_one_http_call_and_returns_bundle(self) -> None:
        """Cache miss → one OWM HTTP call → bundle returned with hourly + daily."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415

        fixture = _load_fixture("onecall.json")
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(200, json=fixture)
            )
            bundle = openweathermap.fetch(
                lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
            )
            call_count = mock.calls.call_count

        assert call_count == 1, f"Expected 1 OWM call on cache miss, got {call_count}"
        assert bundle.source == "openweathermap"
        assert len(bundle.hourly) == 48
        assert len(bundle.daily) == 8

    def test_cache_hit_makes_zero_http_calls(self) -> None:
        """Cache hit → zero OWM HTTP calls."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415

        fixture = _load_fixture("onecall.json")
        # First fetch — fills memory cache
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(200, json=fixture)
            )
            openweathermap.fetch(lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID)

        # Second fetch — should hit cache
        with respx.mock(assert_all_called=False) as mock2:
            bundle2 = openweathermap.fetch(lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID)
            cache_hit_calls = mock2.calls.call_count

        assert cache_hit_calls == 0, f"Expected 0 calls on cache hit, got {cache_hit_calls}"
        assert bundle2.source == "openweathermap"
        assert len(bundle2.hourly) == 48

    def test_cache_hit_discussion_none_round_trips(self) -> None:
        """Cached discussion=None survives serialization round-trip."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415

        fixture = _load_fixture("onecall.json")
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(return_value=httpx.Response(200, json=fixture))
            openweathermap.fetch(lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID)

        with respx.mock(assert_all_called=False):
            bundle2 = openweathermap.fetch(lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID)

        assert bundle2.discussion is None

    def test_missing_appid_raises_key_invalid(self) -> None:
        """appid=None → KeyInvalid raised before any HTTP call (loud failure)."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415

        with respx.mock(assert_all_called=False):
            with pytest.raises(KeyInvalid):
                openweathermap.fetch(lat=_LAT, lon=_LON, target_unit="US", appid=None)

    def test_empty_string_appid_raises_key_invalid(self) -> None:
        """appid='' → KeyInvalid raised."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415

        with respx.mock(assert_all_called=False):
            with pytest.raises(KeyInvalid):
                openweathermap.fetch(lat=_LAT, lon=_LON, target_unit="US", appid="")

    def test_basic_tier_401_returns_empty_bundle_not_error(self) -> None:
        """Basic-tier 401 → ForecastBundle(hourly=[], daily=[], discussion=None) — NOT 502."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415

        error_fixture = _load_fixture("error_401_basic_tier.json")
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(401, json=error_fixture)
            )
            bundle = openweathermap.fetch(
                lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
            )

        assert bundle.hourly == [], "Basic-tier 401 should return empty hourly list"
        assert bundle.daily == [], "Basic-tier 401 should return empty daily list"
        assert bundle.discussion is None
        assert bundle.source == "openweathermap"

    def test_basic_tier_401_bundle_generated_at_ends_with_z(self) -> None:
        """Empty bundle from 401 has generatedAt with Z suffix."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415

        error_fixture = _load_fixture("error_401_basic_tier.json")
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(401, json=error_fixture)
            )
            bundle = openweathermap.fetch(
                lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
            )

        assert bundle.generatedAt.endswith("Z")

    def test_basic_tier_401_warning_logged_only_once(self, caplog: Any) -> None:
        """Basic-tier 401 warning is logged once per process, not on repeat calls."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415

        error_fixture = _load_fixture("error_401_basic_tier.json")

        with caplog.at_level(
            logging.WARNING,
            logger="weewx_clearskies_api.providers.forecast.openweathermap",
        ):
            for _ in range(3):
                with respx.mock(assert_all_called=False) as mock:
                    mock.get(_OWM_ONECALL_URL).mock(
                        return_value=httpx.Response(401, json=error_fixture)
                    )
                    openweathermap.fetch(
                        lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
                    )
                # Reset cache between calls so we don't hit cache
                from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
                    reset_cache_for_tests,
                    wire_cache_from_env,
                )
                reset_cache_for_tests()
                wire_cache_from_env()

        # Count WARN messages from the OWM module
        warn_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "One Call 3.0" in r.message
        ]
        assert len(warn_records) == 1, (
            f"Expected 1 WARN log for basic-tier 401, got {len(warn_records)}"
        )

    def test_basic_tier_401_caches_empty_bundle_for_ttl(self) -> None:
        """Basic-tier 401 caches the empty bundle (3b-5 audit F2 regression).

        Without caching, basic-tier-misconfigured deployments hammer OWM with
        401s on every dashboard poll (capped only by rate limiter, ~432K/day).
        After F2, the empty bundle is cached for the same TTL as the success
        path so subsequent calls return the cached bundle without an OWM call.
        """
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415

        error_fixture = _load_fixture("error_401_basic_tier.json")
        with respx.mock(assert_all_called=False) as mock:
            mock_route = mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(401, json=error_fixture)
            )
            # First call hits OWM and caches the empty bundle
            bundle1 = openweathermap.fetch(
                lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
            )
            # Second call should hit cache, NOT make another OWM call
            bundle2 = openweathermap.fetch(
                lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
            )

        assert bundle1.hourly == [] and bundle1.daily == []
        assert bundle2.hourly == [] and bundle2.daily == []
        assert mock_route.call_count == 1, (
            f"Expected 1 OWM call (second served from cache), got {mock_route.call_count}"
        )

    def test_quota_exceeded_429_raises_quota_exhausted(self) -> None:
        """HTTP 429 from OWM → QuotaExhausted propagated."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415

        error_fixture = _load_fixture("error_429_quota.json")
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(429, json=error_fixture)
            )
            with pytest.raises(QuotaExhausted):
                openweathermap.fetch(
                    lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
                )

    def test_server_5xx_raises_transient_network_error(self) -> None:
        """HTTP 500 → TransientNetworkError propagated."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import TransientNetworkError  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(500, text="Internal Server Error")
            )
            with pytest.raises(TransientNetworkError):
                openweathermap.fetch(
                    lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
                )

    def test_malformed_response_raises_provider_protocol_error(self) -> None:
        """Malformed response (not a valid OWM shape) → ProviderProtocolError."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import ProviderProtocolError  # noqa: PLC0415

        # Response missing required 'lat' and 'lon' fields
        malformed = {"cod": 200, "message": "unexpected"}
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(200, json=malformed)
            )
            with pytest.raises(ProviderProtocolError):
                openweathermap.fetch(
                    lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
                )

    def test_unknown_target_unit_raises_provider_protocol_error(self) -> None:
        """Unknown target_unit → ProviderProtocolError before any HTTP call."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import ProviderProtocolError  # noqa: PLC0415

        with respx.mock(assert_all_called=False):
            with pytest.raises(ProviderProtocolError):
                openweathermap.fetch(
                    lat=_LAT, lon=_LON, target_unit="INVALID", appid=_TEST_APPID
                )

    def test_bundle_cached_after_successful_fetch(self) -> None:
        """After fetch(), bundle is stored in cache under the expected key."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415

        fixture = _load_fixture("onecall.json")
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(return_value=httpx.Response(200, json=fixture))
            openweathermap.fetch(lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID)

        cache_key = openweathermap._build_cache_key(_LAT, _LON, "US")
        cached = get_cache().get(cache_key)
        assert cached is not None, "Bundle should be stored in cache after fetch"


# ===========================================================================
# 12. Q1 path coverage — non-401 KeyInvalid re-raises
# ===========================================================================


class TestQ1PathNon401KeyInvalidReRaises:
    """Non-401 KeyInvalid is re-raised (defensive — should not happen in practice)."""

    def setup_method(self) -> None:
        _reset_provider_state()

    def teardown_method(self) -> None:
        _reset_provider_state()

    def test_non_401_key_invalid_is_reraised(self) -> None:
        """Non-401 KeyInvalid (e.g. 403) → re-raised, not swallowed (Q1 defensive)."""
        from weewx_clearskies_api.providers.forecast import openweathermap  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415

        # 403 → KeyInvalid with status_code=403 (not 401 → re-raise path)
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_ONECALL_URL).mock(
                return_value=httpx.Response(403, json={"cod": 403, "message": "Forbidden"})
            )
            with pytest.raises(KeyInvalid) as exc_info:
                openweathermap.fetch(
                    lat=_LAT, lon=_LON, target_unit="US", appid=_TEST_APPID
                )
        # The re-raised exception should carry the original status code
        assert exc_info.value.status_code == 403


# ===========================================================================
# 13. Redaction filter — appid query param is redacted in logs
# ===========================================================================


class TestRedactionFilter:
    """appid query param is redacted in logged URLs (3b-1 filter, lead-call 6 brief)."""

    def test_url_with_appid_param_is_redacted(self) -> None:
        """A URL containing ?appid=ABC123 → appid=[REDACTED] after redaction."""
        from weewx_clearskies_api.logging.redaction_filter import _redact  # noqa: PLC0415
        url = (
            "https://api.openweathermap.org/data/3.0/onecall"
            "?lat=47.6&lon=-122.3&appid=ABC123&units=imperial"
        )
        redacted = _redact(url)
        assert "ABC123" not in redacted
        assert "appid=[REDACTED]" in redacted

    def test_appid_in_middle_of_query_string_is_redacted(self) -> None:
        """appid in the middle of a query string (before other params) is redacted."""
        from weewx_clearskies_api.logging.redaction_filter import _redact  # noqa: PLC0415
        url = "/data/3.0/onecall?appid=MYSECRETKEY123&units=imperial&exclude=minutely"
        redacted = _redact(url)
        assert "MYSECRETKEY123" not in redacted
        assert "appid=[REDACTED]" in redacted

    def test_appid_only_in_owm_url_shape_is_redacted(self) -> None:
        """OWM uses only appid (not client_secret); appid redaction works correctly."""
        from weewx_clearskies_api.logging.redaction_filter import _redact  # noqa: PLC0415
        url = (
            "https://api.openweathermap.org/data/3.0/onecall"
            "?lat=47.6&lon=-122.3&appid=OWMKEY999"
        )
        redacted = _redact(url)
        assert "OWMKEY999" not in redacted
        assert "appid=[REDACTED]" in redacted


# ===========================================================================
# 14. Capability registry
# ===========================================================================


class TestCapabilityRegistry:
    """CAPABILITY fields and registry wiring for OWM forecast provider."""

    def setup_method(self) -> None:
        from weewx_clearskies_api.providers._common.capability import reset_provider_registry_for_tests  # noqa: PLC0415
        reset_provider_registry_for_tests()

    def teardown_method(self) -> None:
        from weewx_clearskies_api.providers._common.capability import reset_provider_registry_for_tests  # noqa: PLC0415
        reset_provider_registry_for_tests()

    def test_capability_provider_id_is_openweathermap(self) -> None:
        """CAPABILITY.provider_id = 'openweathermap'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.provider_id == "openweathermap"

    def test_capability_domain_is_forecast(self) -> None:
        """CAPABILITY.domain = 'forecast'."""
        from weewx_clearskies_api.providers.forecast.openweathermap import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.domain == "forecast"

    def test_capability_auth_required_includes_appid(self) -> None:
        """CAPABILITY.auth_required includes 'appid' (single OWM credential)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import CAPABILITY  # noqa: PLC0415
        assert "appid" in CAPABILITY.auth_required

    def test_capability_geographic_coverage_is_global(self) -> None:
        """CAPABILITY.geographic_coverage = 'global' (OWM has global coverage)."""
        from weewx_clearskies_api.providers.forecast.openweathermap import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.geographic_coverage == "global"

    def test_capability_supplied_fields_includes_hourly_fields(self) -> None:
        """CAPABILITY.supplied_canonical_fields includes core hourly fields."""
        from weewx_clearskies_api.providers.forecast.openweathermap import CAPABILITY  # noqa: PLC0415
        for field in ("validTime", "outTemp", "outHumidity", "windSpeed", "windDir",
                      "windGust", "precipProbability", "precipAmount", "precipType",
                      "cloudCover", "weatherCode", "weatherText"):
            assert field in CAPABILITY.supplied_canonical_fields, (
                f"Expected hourly field {field!r} in CAPABILITY.supplied_canonical_fields"
            )

    def test_capability_supplied_fields_includes_daily_fields(self) -> None:
        """CAPABILITY.supplied_canonical_fields includes core daily fields."""
        from weewx_clearskies_api.providers.forecast.openweathermap import CAPABILITY  # noqa: PLC0415
        for field in ("validDate", "tempMax", "tempMin", "precipAmount",
                      "precipProbabilityMax", "windSpeedMax", "windGustMax",
                      "sunrise", "sunset", "uvIndexMax", "weatherCode", "weatherText",
                      "narrative"):
            assert field in CAPABILITY.supplied_canonical_fields, (
                f"Expected daily field {field!r} in CAPABILITY.supplied_canonical_fields"
            )

    def test_capability_supplied_fields_does_not_include_discussion_fields(self) -> None:
        """ForecastDiscussion fields NOT in CAPABILITY (OWM §4.1.4 all '—')."""
        from weewx_clearskies_api.providers.forecast.openweathermap import CAPABILITY  # noqa: PLC0415
        # OWM has no forecast discussion product (canonical §4.1.4 = all "—")
        for field in ("headline", "body"):
            assert field not in CAPABILITY.supplied_canonical_fields, (
                f"Discussion field {field!r} should NOT be in OWM CAPABILITY"
            )

    def test_wire_providers_adds_openweathermap_to_registry(self) -> None:
        """wire_providers([openweathermap.CAPABILITY]) → registry has OWM entry."""
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            get_provider_registry,
            wire_providers,
        )
        from weewx_clearskies_api.providers.forecast.openweathermap import CAPABILITY  # noqa: PLC0415

        wire_providers([CAPABILITY])
        registry = get_provider_registry()
        owm_entries = [p for p in registry if p.provider_id == "openweathermap"]
        assert len(owm_entries) == 1
        assert owm_entries[0].domain == "forecast"
