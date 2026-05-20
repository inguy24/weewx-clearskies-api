"""Unit tests for the OpenWeatherMap AQI provider module (3b-11).

Covers per the task-3b-11 brief §Test coverage shape (test_openweathermap.py):

  Wire-shape Pydantic validation:
  - Real captured fixture (openweathermap_current.json) loads cleanly.
  - Extra fields (coord, no, nh3) ignored (extra="ignore").
  - list[] field shadows Python builtin -- Field(default_factory=list) pattern.
  - Missing required 'dt' → ValidationError.
  - Empty list[] validates cleanly (empty-result sentinel path).

  _wire_to_canonical happy path:
  - Real fixture → canonical AQIReading with all fields populated correctly.
  - aqi = 2 (OWM main.aqi ordinal served as-is).
  - aqiScale = "owm" (1–5 ordinal scale).
  - aqiCategory = None (dashboard-computed).
  - aqiMainPollutant = None (OWM does not supply dominant pollutant).
  - aqiLocation = None (PARTIAL-DOMAIN — no location field on OWM Air Pollution wire).
  - observedAt = "2026-05-11T03:56:58Z" (epoch 1778471818 → UTC Z, LC17).
  - source = "openweathermap".
  - All gases (O3, NO2, SO2, CO) and PM2.5/PM10 passed through as µg/m³.
  - NH3 and NO silently dropped (no EPA AQI band; LC16).

  _wire_to_canonical edge cases:
  - main.aqi=None AND all-null components → returns None + would cache sentinel.
  - Empty list[] → no entry to process → None returned.
  - main.aqi ordinal served as-is for all valid OWM values (1–5).

  _build_cache_key:
  - Same lat/lon → same key (deterministic).
  - Different lat/lon → different key.
  - Key is 64-char hex string (SHA-256).
  - Lat/lon rounded to 4 decimal places (LC7).
  - Key does NOT encode credentials (appid not in key — privacy/leakage LC7).
  - OWM AQI key distinct from openmeteo AQI key at same coordinates.
  - OWM AQI key distinct from OWM forecast key at same coordinates.

  fetch():
  - Cache hit → canonical reconstruction from cached dict; no HTTP call.
  - Cache hit with _no_reading sentinel → None returned.
  - Cache miss happy path via respx mock + real fixture → canonical AQIReading.
  - Cache miss + wire-validation failure → ProviderProtocolError.
  - Cache miss + missing appid (empty string) → KeyInvalid BEFORE outbound call.
  - Cache miss + provider HTTP 401 → KeyInvalid (L2 carry-forward; bare propagation).
  - Cache miss + provider HTTP 429 → QuotaExhausted (L2 carry-forward).
  - Cache miss + provider HTTP 5xx → TransientNetworkError (L2 carry-forward).
  - Cache miss + empty list[] → None + sentinel cached.
  - Cache miss + all-null components → None + sentinel cached.
  - retry_after_seconds propagated on QuotaExhausted (3b-4 F1 carry-forward).

  Capability declaration:
  - CAPABILITY.provider_id = "openweathermap", domain = "aqi".
  - CAPABILITY.auth_required = ("appid",).
  - CAPABILITY.geographic_coverage = "global".
  - CAPABILITY.supplied_canonical_fields includes the 11 OWM-supplied fields.
  - CAPABILITY.supplied_canonical_fields excludes aqiLocation (PARTIAL-DOMAIN).
  - CAPABILITY.default_poll_interval_seconds = 900 (15 min per ADR-017 / LC3).
  - wire_providers([CAPABILITY]) → registry has openweathermap aqi entry.

No DB, no live network. respx mocks outbound httpx calls.
Wire-shape rule: fixtures loaded from tests/fixtures/providers/aqi/.
ADR references: ADR-013, ADR-017, ADR-020, ADR-038.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "providers" / "aqi"
_OWM_AIRPOL_BASE_URL = "https://api.openweathermap.org"
_OWM_AIRPOL_PATH = "/data/2.5/air_pollution"
_OWM_AIRPOL_URL = _OWM_AIRPOL_BASE_URL + _OWM_AIRPOL_PATH

# Coordinates matching fixture — match 6dp precision used in URL construction
_LAT = 47.6062
_LON = -122.3321
_LAT4 = round(_LAT, 4)
_LON4 = round(_LON, 4)
_LAT6 = round(_LAT, 6)
_LON6 = round(_LON, 6)

_TEST_APPID = "TEST_OWM_APPID_12345"


def _load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture file from tests/fixtures/providers/aqi/."""
    path = _FIXTURES_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return json.loads(fh.read())


# ---------------------------------------------------------------------------
# State-reset helpers
# ---------------------------------------------------------------------------


def _reset_provider_state() -> None:
    """Reset provider registry, cache, rate limiter, and re-wire memory cache."""
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


# ===========================================================================
# 1. Wire-shape Pydantic validation
# ===========================================================================


class TestWireShapePydanticValidation:
    """Wire-shape models validate correctly against the fixture and edge-case shapes."""

    def test_real_fixture_loads_cleanly_via_response_model(self) -> None:
        """openweathermap_current.json loads via _OWMAirPollutionResponse without error."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _OWMAirPollutionResponse,
        )
        raw = _load_fixture("openweathermap_current.json")
        response = _OWMAirPollutionResponse.model_validate(raw)
        assert len(response.list) == 1, (
            f"Expected 1 entry in list[], got {len(response.list)}"
        )

    def test_real_fixture_extra_fields_are_ignored(self) -> None:
        """coord field (unread) is ignored by extra='ignore' on the response model."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _OWMAirPollutionResponse,
        )
        raw = _load_fixture("openweathermap_current.json")
        assert "coord" in raw, "Fixture must have coord field to test extra='ignore'"
        # Should not raise even though coord is not declared in the model
        response = _OWMAirPollutionResponse.model_validate(raw)
        assert response is not None

    def test_future_extra_field_is_ignored(self) -> None:
        """Unknown extra fields silently ignored (extra='ignore' — forward-compat)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _OWMAirPollutionResponse,
        )
        raw = _load_fixture("openweathermap_current.json")
        raw["future_field_not_in_spec"] = "should be ignored"
        # Should not raise
        response = _OWMAirPollutionResponse.model_validate(raw)
        assert response is not None

    def test_list_field_shadows_builtin_but_validates_correctly(self) -> None:
        """_OWMAirPollutionResponse.list is a list (Field shadows Python builtin — LC11)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _OWMAirPollutionResponse,
        )
        raw = _load_fixture("openweathermap_current.json")
        response = _OWMAirPollutionResponse.model_validate(raw)
        assert isinstance(response.list, list), (
            f"response.list must be a Python list, got {type(response.list)!r}"
        )

    def test_empty_list_validates_cleanly(self) -> None:
        """Empty list[] validates cleanly — OWM can return empty for no data at location."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _OWMAirPollutionResponse,
        )
        empty_response = {"coord": {"lon": _LON, "lat": _LAT}, "list": []}
        response = _OWMAirPollutionResponse.model_validate(empty_response)
        assert response.list == [], "Empty list[] must parse to empty Python list"

    def test_all_component_fields_optional_allows_none(self) -> None:
        """Components with all None values validates cleanly (all fields optional)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _OWMAirPollutionResponse,
        )
        null_components = {
            "coord": {"lon": _LON, "lat": _LAT},
            "list": [{
                "dt": 1778471818,
                "main": {"aqi": None},
                "components": {
                    "co": None, "no": None, "no2": None, "o3": None,
                    "so2": None, "pm2_5": None, "pm10": None, "nh3": None,
                },
            }],
        }
        response = _OWMAirPollutionResponse.model_validate(null_components)
        assert response.list[0].components.pm2_5 is None
        assert response.list[0].components.co is None

    def test_fixture_dt_field_is_present_and_integer(self) -> None:
        """Fixture list[0].dt = 1778471818 (Unix UTC seconds — integer)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _OWMAirPollutionResponse,
        )
        raw = _load_fixture("openweathermap_current.json")
        response = _OWMAirPollutionResponse.model_validate(raw)
        assert response.list[0].dt == 1778471818, (
            f"Expected dt=1778471818, got {response.list[0].dt!r}"
        )

    def test_fixture_main_aqi_field_is_parsed(self) -> None:
        """list[0].main.aqi = 2 (OWM 1–5 ordinal — parsed and served as canonical aqi)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _OWMAirPollutionResponse,
        )
        raw = _load_fixture("openweathermap_current.json")
        response = _OWMAirPollutionResponse.model_validate(raw)
        assert response.list[0].main.aqi == 2, (
            f"Expected main.aqi=2 (OWM 1-5 ordinal, served as-is), got {response.list[0].main.aqi!r}"
        )

    def test_fixture_components_pm25_and_pm10_parsed(self) -> None:
        """Fixture components.pm2_5 = 0.5 µg/m³ and pm10 = 0.81 µg/m³ parsed correctly."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _OWMAirPollutionResponse,
        )
        raw = _load_fixture("openweathermap_current.json")
        response = _OWMAirPollutionResponse.model_validate(raw)
        components = response.list[0].components
        assert components.pm2_5 == pytest.approx(0.5, rel=1e-6)
        assert components.pm10 == pytest.approx(0.81, rel=1e-6)

    def test_fixture_gas_concentrations_parsed(self) -> None:
        """Fixture components gas values (co, no2, o3, so2) parsed correctly."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _OWMAirPollutionResponse,
        )
        raw = _load_fixture("openweathermap_current.json")
        response = _OWMAirPollutionResponse.model_validate(raw)
        c = response.list[0].components
        assert c.co == pytest.approx(139.79, rel=1e-5)
        assert c.no2 == pytest.approx(2.05, rel=1e-5)
        assert c.o3 == pytest.approx(66.23, rel=1e-5)
        assert c.so2 == pytest.approx(0.34, rel=1e-5)


# ===========================================================================
# 2. _wire_to_canonical — happy path from real fixture
# ===========================================================================


class TestWireToCanonicalHappyPath:
    """_wire_to_canonical translates the real fixture to a correct AQIReading."""

    def _load_entry(self, filename: str = "openweathermap_current.json") -> Any:
        """Load fixture and extract the first list entry."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _OWMAirPollutionResponse,
        )
        raw = _load_fixture(filename)
        response = _OWMAirPollutionResponse.model_validate(raw)
        return response.list[0]

    def test_fixture_produces_non_none_aqi_reading(self) -> None:
        """_wire_to_canonical returns AQIReading (not None) for the real fixture."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None, "_wire_to_canonical must return AQIReading for valid fixture"

    def test_fixture_aqi_is_owm_ordinal_2(self) -> None:
        """aqi = 2 (OWM main.aqi ordinal served as-is with aqiScale='owm').

        Fixture main.aqi=2 (OWM 1–5 ordinal, 2=Fair).
        The API no longer derives EPA AQI from concentrations — raw ordinal is served.
        """
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        assert result.aqi == 2, (
            f"Expected aqi=2 (OWM ordinal from main.aqi), got {result.aqi!r}"
        )

    def test_fixture_aqi_scale_is_owm(self) -> None:
        """aqiScale = 'owm' (OWM 1–5 ordinal scale)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        assert result.aqiScale == "owm", (
            f"Expected aqiScale='owm', got {result.aqiScale!r}"
        )

    def test_fixture_aqi_category_is_none(self) -> None:
        """aqiCategory = None (dashboard-computed; parsers set None)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        assert result.aqiCategory is None, (
            f"Expected aqiCategory=None (dashboard-computed), got {result.aqiCategory!r}"
        )

    def test_fixture_aqi_main_pollutant_is_none(self) -> None:
        """aqiMainPollutant = None (OWM Air Pollution does not supply dominant pollutant)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        assert result.aqiMainPollutant is None, (
            f"Expected aqiMainPollutant=None (not supplied by OWM Air Pollution), "
            f"got {result.aqiMainPollutant!r}"
        )

    def test_fixture_aqi_location_is_none_partial_domain(self) -> None:
        """aqiLocation = None (PARTIAL-DOMAIN — OWM Air Pollution has no location label)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        assert result.aqiLocation is None, (
            f"Expected aqiLocation=None (PARTIAL-DOMAIN), got {result.aqiLocation!r}"
        )

    def test_fixture_source_is_openweathermap(self) -> None:
        """source = 'openweathermap' (provider_id literal on AQIReading)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        assert result.source == "openweathermap", (
            f"Expected source='openweathermap', got {result.source!r}"
        )

    def test_fixture_observed_at_is_utc_z_format(self) -> None:
        """observedAt ends with Z (UTC ISO-8601 per LC17 + ADR-020)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        assert result.observedAt is not None
        assert result.observedAt.endswith("Z"), (
            f"observedAt must end with Z, got {result.observedAt!r}"
        )

    def test_fixture_observed_at_matches_epoch_1778471818(self) -> None:
        """observedAt = '2026-05-11T03:56:58Z' (epoch 1778471818 → UTC via epoch_to_utc_iso8601)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        assert result.observedAt == "2026-05-11T03:56:58Z", (
            f"Expected '2026-05-11T03:56:58Z' (epoch 1778471818), got {result.observedAt!r}"
        )

    def test_fixture_pm25_passes_through_in_ugm3(self) -> None:
        """pollutantPM25 = 0.5 µg/m³ (fixture value; passthrough, no conversion)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        assert result.pollutantPM25 == pytest.approx(0.5, rel=1e-6), (
            f"pollutantPM25 should be 0.5 µg/m³ (passthrough), got {result.pollutantPM25!r}"
        )

    def test_fixture_pm10_passes_through_in_ugm3(self) -> None:
        """pollutantPM10 = 0.81 µg/m³ (fixture value; passthrough, no conversion)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        assert result.pollutantPM10 == pytest.approx(0.81, rel=1e-6), (
            f"pollutantPM10 should be 0.81 µg/m³ (passthrough), got {result.pollutantPM10!r}"
        )

    def test_fixture_o3_passes_through_in_ugm3(self) -> None:
        """pollutantO3 = 66.23 µg/m³ (fixture value; raw passthrough, no conversion)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        assert result.pollutantO3 == pytest.approx(66.23, rel=1e-6), (
            f"O3: expected 66.23 µg/m³ (passthrough), got {result.pollutantO3!r}"
        )

    def test_fixture_no2_passes_through_in_ugm3(self) -> None:
        """pollutantNO2 = 2.05 µg/m³ (fixture value; raw passthrough, no conversion)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        assert result.pollutantNO2 == pytest.approx(2.05, rel=1e-6), (
            f"NO2: expected 2.05 µg/m³ (passthrough), got {result.pollutantNO2!r}"
        )

    def test_fixture_so2_passes_through_in_ugm3(self) -> None:
        """pollutantSO2 = 0.34 µg/m³ (fixture value; raw passthrough, no conversion)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        assert result.pollutantSO2 == pytest.approx(0.34, rel=1e-6), (
            f"SO2: expected 0.34 µg/m³ (passthrough), got {result.pollutantSO2!r}"
        )

    def test_fixture_co_passes_through_in_ugm3(self) -> None:
        """pollutantCO = 139.79 µg/m³ (fixture value; raw passthrough, no conversion)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        assert result.pollutantCO == pytest.approx(139.79, rel=1e-6), (
            f"CO: expected 139.79 µg/m³ (passthrough), got {result.pollutantCO!r}"
        )

    def test_canonical_output_has_no_pollutant_nh3_or_no_field(self) -> None:
        """NH3 and NO fields are dropped — not present on canonical AQIReading (LC16)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._load_entry()
        result = _wire_to_canonical(entry)
        assert result is not None
        result_dict = result.model_dump()
        # NH3 and NO have no canonical fields; should not appear
        for absent_key in ("pollutantNH3", "pollutantNO", "nh3", "no"):
            assert absent_key not in result_dict, (
                f"'{absent_key}' must NOT appear in canonical output (dropped per LC16)"
            )


# ===========================================================================
# 3. _wire_to_canonical edge cases
# ===========================================================================


class TestWireToCanonicalEdgeCases:
    """Edge cases: all-null, empty list, single-pollutant reads."""

    def _make_entry(
        self,
        components: dict[str, Any],
        dt: int = 1778471818,
        main_aqi: int | None = 1,
    ) -> Any:
        """Build and validate a minimal OWM list entry."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _OWMAirPollutionResponse,
        )
        data = {
            "coord": {"lon": _LON, "lat": _LAT},
            "list": [{
                "dt": dt,
                "main": {"aqi": main_aqi},
                "components": components,
            }],
        }
        response = _OWMAirPollutionResponse.model_validate(data)
        return response.list[0]

    def test_all_null_components_and_null_aqi_returns_none(self) -> None:
        """All component values None AND main.aqi=None → _wire_to_canonical returns None."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._make_entry(
            {
                "co": None, "no": None, "no2": None, "o3": None,
                "so2": None, "pm2_5": None, "pm10": None, "nh3": None,
            },
            main_aqi=None,
        )
        result = _wire_to_canonical(entry)
        assert result is None, (
            "_wire_to_canonical must return None when aqi AND all components are null"
        )

    def test_null_aqi_but_some_components_returns_reading(self) -> None:
        """main.aqi=None but pm2_5 non-null → AQIReading returned (has_data via concentrations)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        entry = self._make_entry(
            {
                "co": None, "no": None, "no2": None, "o3": None,
                "so2": None, "pm2_5": 5.0, "pm10": None, "nh3": None,
            },
            main_aqi=None,
        )
        result = _wire_to_canonical(entry)
        assert result is not None, "pm2_5=5.0 with aqi=None should yield a non-None reading"
        assert result.aqi is None
        assert result.aqiScale == "owm"
        assert result.aqiMainPollutant is None, (
            "OWM does not supply aqiMainPollutant — must be None"
        )

    def test_aqi_ordinal_served_as_is(self) -> None:
        """main.aqi is served directly as canonical aqi with aqiScale='owm'."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _wire_to_canonical,
        )
        for owm_val in (1, 2, 3, 4, 5):
            entry = self._make_entry(
                {"co": None, "no": None, "no2": None, "o3": None,
                 "so2": None, "pm2_5": 5.0, "pm10": None, "nh3": None},
                main_aqi=owm_val,
            )
            result = _wire_to_canonical(entry)
            assert result is not None
            assert result.aqi == owm_val, (
                f"main.aqi={owm_val} must be served as canonical aqi, got {result.aqi!r}"
            )
            assert result.aqiScale == "owm"
            assert result.aqiMainPollutant is None


# ===========================================================================
# 4. _build_cache_key — determinism and privacy
# ===========================================================================


class TestBuildCacheKey:
    """_build_cache_key is deterministic, rounds lat/lon, and excludes credentials."""

    def test_same_lat_lon_produces_same_key(self) -> None:
        """Same lat/lon → same cache key (deterministic)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (
            _build_cache_key,  # noqa: PLC0415
        )
        key1 = _build_cache_key(_LAT, _LON)
        key2 = _build_cache_key(_LAT, _LON)
        assert key1 == key2, "Same coordinates must produce the same cache key"

    def test_different_lat_lon_produces_different_key(self) -> None:
        """Different lat/lon → different cache key."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (
            _build_cache_key,  # noqa: PLC0415
        )
        key1 = _build_cache_key(_LAT, _LON)
        key2 = _build_cache_key(40.7128, -74.0060)
        assert key1 != key2, "Different coordinates must produce different cache keys"

    def test_key_is_64_char_lowercase_hex(self) -> None:
        """Cache key is a 64-character lowercase hexadecimal string (SHA-256)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (
            _build_cache_key,  # noqa: PLC0415
        )
        key = _build_cache_key(_LAT, _LON)
        assert len(key) == 64, f"Expected 64-char key, got {len(key)}"
        assert all(c in "0123456789abcdef" for c in key), (
            "Cache key must be lowercase hex"
        )

    def test_lat_lon_rounded_to_4_decimal_places_for_key(self) -> None:
        """High-precision lat/lon rounds to 4dp — equivalent coordinates produce same key."""
        from weewx_clearskies_api.providers.aqi.openweathermap import (
            _build_cache_key,  # noqa: PLC0415
        )
        # These differ only beyond 4dp — must produce the same key
        key1 = _build_cache_key(47.60620001, -122.33210001)
        key2 = _build_cache_key(47.60620009, -122.33210009)
        assert key1 == key2, (
            "Coordinates identical at 4dp must produce the same cache key"
        )

    def test_owm_aqi_key_distinct_from_openmeteo_aqi_key(self) -> None:
        """OWM AQI key differs from Open-Meteo AQI key at same coordinates (provider_id differs)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (  # noqa: PLC0415
            _build_cache_key as om_key,
        )
        from weewx_clearskies_api.providers.aqi.openweathermap import (
            _build_cache_key,  # noqa: PLC0415
        )
        owm_key = _build_cache_key(_LAT, _LON)
        openmeteo_key = om_key(_LAT, _LON)
        assert owm_key != openmeteo_key, (
            "OWM AQI and Open-Meteo AQI must have distinct cache keys at same coordinates"
        )

    def test_owm_aqi_key_distinct_from_aeris_aqi_key(self) -> None:
        """OWM AQI key differs from Aeris AQI key at same coordinates."""
        from weewx_clearskies_api.providers.aqi.aeris import (  # noqa: PLC0415
            _build_cache_key as aeris_key,
        )
        from weewx_clearskies_api.providers.aqi.openweathermap import (
            _build_cache_key,  # noqa: PLC0415
        )
        owm_key = _build_cache_key(_LAT, _LON)
        aeris_cache_key = aeris_key(_LAT, _LON)
        assert owm_key != aeris_cache_key, (
            "OWM AQI and Aeris AQI must have distinct cache keys at same coordinates"
        )

    def test_appid_not_accepted_as_parameter_in_cache_key(self) -> None:
        """_build_cache_key signature does NOT accept appid (credentials not in key — LC7)."""
        import inspect  # noqa: PLC0415

        from weewx_clearskies_api.providers.aqi.openweathermap import (
            _build_cache_key,  # noqa: PLC0415
        )
        sig = inspect.signature(_build_cache_key)
        param_names = list(sig.parameters.keys())
        assert "appid" not in param_names, (
            "_build_cache_key must not accept appid (privacy/leakage — LC7)"
        )


# ===========================================================================
# 6. fetch() — cache paths
# ===========================================================================


class TestFetchCachePaths:
    """fetch() cache hit / miss / sentinel reconstruction."""

    def setup_method(self) -> None:
        """Reset all provider state before each test."""
        _reset_provider_state()

    def test_cache_hit_returns_reading_without_http_call(self) -> None:
        """Cache hit → AQIReading reconstructed from dict; NO outbound HTTP call made."""
        from weewx_clearskies_api.models.responses import AQIReading  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            DEFAULT_AQI_TTL_SECONDS,
            _build_cache_key,
            fetch,
        )

        # Populate cache manually
        cache = get_cache()
        key = _build_cache_key(_LAT, _LON)
        cached_reading = AQIReading(
            aqi=42,
            aqiCategory="Good",
            aqiMainPollutant="PM2.5",
            aqiLocation=None,
            pollutantPM25=5.0,
            pollutantPM10=None,
            pollutantO3=None,
            pollutantNO2=None,
            pollutantSO2=None,
            pollutantCO=None,
            observedAt="2026-05-11T03:56:58Z",
            source="openweathermap",
        )
        cache.set(key, cached_reading.model_dump(), ttl_seconds=DEFAULT_AQI_TTL_SECONDS)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(200, json={})
            )
            result = fetch(lat=_LAT, lon=_LON, appid=_TEST_APPID)
            # respx would capture any call — if any were made, the mock would record them
            assert not mock.calls, "Cache hit must NOT make an outbound HTTP call"

        assert result is not None
        assert result.aqi == 42, f"Expected cached aqi=42, got {result.aqi!r}"
        assert result.source == "openweathermap"

    def test_cache_hit_with_sentinel_returns_none(self) -> None:
        """Cache hit with _no_reading sentinel → None returned without HTTP call."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            DEFAULT_AQI_TTL_SECONDS,  # noqa: PLC0415
            _build_cache_key,
            fetch,
        )
        cache = get_cache()
        key = _build_cache_key(_LAT, _LON)
        cache.set(key, {"_no_reading": True}, ttl_seconds=DEFAULT_AQI_TTL_SECONDS)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(200, json={})
            )
            result = fetch(lat=_LAT, lon=_LON, appid=_TEST_APPID)
            assert not mock.calls, "Sentinel hit must NOT make outbound HTTP call"

        assert result is None, "Sentinel cache hit must return None"

    def test_cache_miss_happy_path_returns_aqi_reading(self) -> None:
        """Cache miss + valid OWM response → canonical AQIReading returned + cached."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _build_cache_key,
            fetch,
        )

        fixture = _load_fixture("openweathermap_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(200, json=fixture)
            )
            result = fetch(lat=_LAT, lon=_LON, appid=_TEST_APPID)

        assert result is not None, "Cache miss + valid response must return AQIReading"
        assert result.source == "openweathermap"
        assert result.aqi == 2  # OWM main.aqi ordinal from fixture
        assert result.aqiScale == "owm"
        assert result.aqiCategory is None
        assert result.aqiMainPollutant is None

        # Verify cache was populated
        cache = get_cache()
        key = _build_cache_key(_LAT, _LON)
        cached = cache.get(key)
        assert cached is not None, "Cache must be populated after cache-miss fetch"
        assert cached.get("source") == "openweathermap"

    def test_cache_miss_empty_list_returns_none_and_caches_sentinel(self) -> None:
        """Cache miss + empty list[] → None returned + sentinel cached."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _build_cache_key,
            fetch,
        )

        empty_response = {"coord": {"lon": _LON, "lat": _LAT}, "list": []}

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(200, json=empty_response)
            )
            result = fetch(lat=_LAT, lon=_LON, appid=_TEST_APPID)

        assert result is None, "Empty list[] → must return None"

        cache = get_cache()
        key = _build_cache_key(_LAT, _LON)
        cached = cache.get(key)
        assert cached is not None, "Sentinel must be cached after empty response"
        assert cached.get("_no_reading") is True, (
            f"Cached value must be sentinel, got {cached!r}"
        )

    def test_cache_miss_all_null_aqi_and_components_returns_none_and_caches_sentinel(self) -> None:
        """Cache miss + main.aqi=None AND all-null components → None returned + sentinel cached."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.openweathermap import (  # noqa: PLC0415
            _build_cache_key,
            fetch,
        )

        null_response = {
            "coord": {"lon": _LON, "lat": _LAT},
            "list": [{
                "dt": 1778471818,
                "main": {"aqi": None},
                "components": {
                    "co": None, "no": None, "no2": None, "o3": None,
                    "so2": None, "pm2_5": None, "pm10": None, "nh3": None,
                },
            }],
        }

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(200, json=null_response)
            )
            result = fetch(lat=_LAT, lon=_LON, appid=_TEST_APPID)

        assert result is None, "All-null aqi + components → must return None"

        cache = get_cache()
        key = _build_cache_key(_LAT, _LON)
        cached = cache.get(key)
        assert cached is not None
        assert cached.get("_no_reading") is True


# ===========================================================================
# 7. fetch() — error paths
# ===========================================================================


class TestFetchErrorPaths:
    """fetch() canonical exception propagation (L2 carry-forward, 3b-4 F1)."""

    def setup_method(self) -> None:
        """Reset all provider state before each test."""
        _reset_provider_state()

    def test_missing_appid_raises_key_invalid_before_http_call(self) -> None:
        """appid='' → KeyInvalid raised BEFORE outbound call (explicit-fail-fast per LC20)."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.openweathermap import fetch  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(200, json={})
            )
            with pytest.raises(KeyInvalid) as exc_info:
                fetch(lat=_LAT, lon=_LON, appid="")
            assert not mock.calls, "Empty appid must raise KeyInvalid BEFORE HTTP call"

        assert exc_info.value.provider_id == "openweathermap", (
            f"KeyInvalid.provider_id must be 'openweathermap', got {exc_info.value.provider_id!r}"
        )

    def test_provider_401_raises_key_invalid(self) -> None:
        """Provider HTTP 401 → KeyInvalid (invalid appid; bare propagation per L2)."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.openweathermap import fetch  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(401, json={"cod": 401, "message": "Invalid API key"})
            )
            with pytest.raises(KeyInvalid):
                fetch(lat=_LAT, lon=_LON, appid=_TEST_APPID)

    def test_provider_429_raises_quota_exhausted(self) -> None:
        """Provider HTTP 429 → QuotaExhausted (bare propagation per L2)."""
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.openweathermap import fetch  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={"cod": 429, "message": "too many requests"},
                    headers={"Retry-After": "60"},
                )
            )
            with pytest.raises(QuotaExhausted):
                fetch(lat=_LAT, lon=_LON, appid=_TEST_APPID)

    def test_provider_429_retry_after_seconds_propagated(self) -> None:
        """Provider 429 with Retry-After: 90 → QuotaExhausted.retry_after_seconds = 90."""
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.openweathermap import fetch  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={"message": "rate limited"},
                    headers={"Retry-After": "90"},
                )
            )
            with pytest.raises(QuotaExhausted) as exc_info:
                fetch(lat=_LAT, lon=_LON, appid=_TEST_APPID)

        assert exc_info.value.retry_after_seconds == 90, (
            f"Expected retry_after_seconds=90, got {exc_info.value.retry_after_seconds!r}"
        )

    def test_provider_5xx_raises_transient_network_error(self) -> None:
        """Provider HTTP 5xx → TransientNetworkError (bare propagation per L2)."""
        from weewx_clearskies_api.providers._common.errors import (
            TransientNetworkError,  # noqa: PLC0415
        )
        from weewx_clearskies_api.providers.aqi.openweathermap import fetch  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(500, json={"error": "internal server error"})
            )
            with pytest.raises(TransientNetworkError):
                fetch(lat=_LAT, lon=_LON, appid=_TEST_APPID)

    def test_wire_validation_failure_raises_provider_protocol_error(self) -> None:
        """Malformed list entry → ProviderProtocolError (LC10 intentional narrow wrap).

        _OWMAirPollutionResponse.list accepts empty lists (all fields optional with
        defaults), but a list entry with dt='not-an-int' fails Pydantic validation
        because _OWMAirPollutionEntry.dt is required as an int.
        """
        from weewx_clearskies_api.providers._common.errors import (
            ProviderProtocolError,  # noqa: PLC0415
        )
        from weewx_clearskies_api.providers.aqi.openweathermap import fetch  # noqa: PLC0415

        # dt must be int; "not-a-timestamp" string fails Pydantic validation
        malformed = {
            "coord": {"lon": _LON, "lat": _LAT},
            "list": [{"dt": "not-a-timestamp", "main": {"aqi": 1}, "components": {}}],
        }

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(200, json=malformed)
            )
            with pytest.raises(ProviderProtocolError):
                fetch(lat=_LAT, lon=_LON, appid=_TEST_APPID)

    def test_no_envelope_error_dispatch_needed_for_owm(self) -> None:
        """OWM Air Pollution uses HTTP status codes (not 200-success-false envelope).

        Verifies that a 200 response with valid-looking JSON is treated as
        a successful response (no Aeris-style success=false envelope dispatch).
        """
        from weewx_clearskies_api.providers.aqi.openweathermap import fetch  # noqa: PLC0415

        fixture = _load_fixture("openweathermap_current.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OWM_AIRPOL_URL).mock(
                return_value=httpx.Response(200, json=fixture)
            )
            # Should NOT raise — OWM 200 responses don't have a success=false pattern
            result = fetch(lat=_LAT, lon=_LON, appid=_TEST_APPID)

        assert result is not None, "Valid 200 response must return AQIReading"


# ===========================================================================
# 8. Capability declaration
# ===========================================================================


class TestCapabilityDeclaration:
    """CAPABILITY symbol validates against the brief spec (ADR-038 §4, LC12)."""

    def test_capability_provider_id_is_openweathermap(self) -> None:
        """CAPABILITY.provider_id = 'openweathermap'."""
        from weewx_clearskies_api.providers.aqi.openweathermap import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.provider_id == "openweathermap", (
            f"Expected provider_id='openweathermap', got {CAPABILITY.provider_id!r}"
        )

    def test_capability_domain_is_aqi(self) -> None:
        """CAPABILITY.domain = 'aqi'."""
        from weewx_clearskies_api.providers.aqi.openweathermap import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.domain == "aqi", (
            f"Expected domain='aqi', got {CAPABILITY.domain!r}"
        )

    def test_capability_auth_required_contains_appid(self) -> None:
        """CAPABILITY.auth_required = ('appid',) — OWM uses appid query param."""
        from weewx_clearskies_api.providers.aqi.openweathermap import CAPABILITY  # noqa: PLC0415
        assert "appid" in CAPABILITY.auth_required, (
            f"Expected 'appid' in auth_required, got {CAPABILITY.auth_required!r}"
        )

    def test_capability_geographic_coverage_is_global(self) -> None:
        """CAPABILITY.geographic_coverage = 'global' (OWM covers worldwide)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.geographic_coverage == "global", (
            f"Expected geographic_coverage='global', got {CAPABILITY.geographic_coverage!r}"
        )

    def test_capability_poll_interval_is_900_seconds(self) -> None:
        """CAPABILITY.default_poll_interval_seconds = 900 (15 min per ADR-017 / LC3)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.default_poll_interval_seconds == 900, (
            f"Expected 900s poll interval, got {CAPABILITY.default_poll_interval_seconds!r}"
        )

    def test_capability_supplied_fields_includes_aqi_and_category(self) -> None:
        """CAPABILITY includes 'aqi' and 'aqiCategory' (derived client-side)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import CAPABILITY  # noqa: PLC0415
        for field in ("aqi", "aqiCategory"):
            assert field in CAPABILITY.supplied_canonical_fields, (
                f"CAPABILITY must include '{field}'"
            )

    def test_capability_supplied_fields_includes_all_pollutants(self) -> None:
        """CAPABILITY includes all 6 canonical pollutant fields."""
        from weewx_clearskies_api.providers.aqi.openweathermap import CAPABILITY  # noqa: PLC0415
        expected_pollutants = (
            "pollutantPM25", "pollutantPM10",
            "pollutantO3", "pollutantNO2", "pollutantSO2", "pollutantCO",
        )
        for field in expected_pollutants:
            assert field in CAPABILITY.supplied_canonical_fields, (
                f"CAPABILITY must include '{field}'"
            )

    def test_capability_excludes_aqi_location_partial_domain(self) -> None:
        """CAPABILITY excludes 'aqiLocation' (PARTIAL-DOMAIN — no location on wire per LC12)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import CAPABILITY  # noqa: PLC0415
        assert "aqiLocation" not in CAPABILITY.supplied_canonical_fields, (
            "aqiLocation must NOT be in supplied_canonical_fields (PARTIAL-DOMAIN)"
        )

    def test_capability_includes_observed_at_and_source(self) -> None:
        """CAPABILITY includes 'observedAt' and 'source'."""
        from weewx_clearskies_api.providers.aqi.openweathermap import CAPABILITY  # noqa: PLC0415
        for field in ("observedAt", "source"):
            assert field in CAPABILITY.supplied_canonical_fields, (
                f"CAPABILITY must include '{field}'"
            )

    def test_capability_includes_aqi_main_pollutant(self) -> None:
        """CAPABILITY includes 'aqiMainPollutant' (derived client-side via argmax)."""
        from weewx_clearskies_api.providers.aqi.openweathermap import CAPABILITY  # noqa: PLC0415
        assert "aqiMainPollutant" in CAPABILITY.supplied_canonical_fields, (
            "CAPABILITY must include 'aqiMainPollutant'"
        )

    def test_wire_providers_registers_openweathermap_aqi_entry(self) -> None:
        """wire_providers([CAPABILITY]) registers ('aqi', 'openweathermap') in registry."""
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            get_provider_registry,
            reset_provider_registry_for_tests,
            wire_providers,
        )
        from weewx_clearskies_api.providers.aqi.openweathermap import CAPABILITY  # noqa: PLC0415

        reset_provider_registry_for_tests()
        wire_providers([CAPABILITY])
        registry = get_provider_registry()
        assert any(
            p.provider_id == "openweathermap" and p.domain == "aqi"
            for p in registry
        ), "wire_providers([CAPABILITY]) must register openweathermap aqi in registry"
        reset_provider_registry_for_tests()
