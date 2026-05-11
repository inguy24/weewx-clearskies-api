"""Unit tests for the Aeris AQI provider module (3b-10).

Covers per the task-3b-10 brief §Test-author parallel scope (test_aeris.py):

  Wire-shape Pydantic validation:
  - Real captured fixture (aeris_current.json) loads cleanly.
  - Extra fields (health, color, method, profile, loc, id) ignored (extra="ignore").

  _wire_to_canonical happy path:
  - Real fixture → canonical AQIReading with all fields populated correctly.
  - aqi = 33 → int; aqiCategory = "Good" (EPA band); aqiMainPollutant = "O3" (o3→O3).
  - observedAt = "2026-05-10T23:00:00Z" (LC4: explicit-offset -07:00 → UTC Z).
  - source = "aeris" (provider_id literal).
  - aqiLocation = "seattle" (place.name from Aeris — NOT PARTIAL-DOMAIN).
  - Gases (O3, NO2, SO2, CO) in ppm (ppb/1000); particulates (PM2.5, PM10) in µg/m³.

  _wire_to_canonical edge cases:
  - Empty pollutants list: aqi/category populate from period.aqi, per-pollutant fields None.
  - dominant = "pm1": aqiMainPollutant = None (drop; no PM1 canonical field).
  - dominant = None / missing: aqiMainPollutant = None.
  - All-null aqi + all-null pollutant values → _wire_to_canonical returns None.
  - pm1 entry in pollutants[] array does not appear in canonical output.

  _iso_offset_to_utc_z:
  - Explicit negative offset (-07:00) → UTC Z (converts -07:00 to +07:00 hours).
  - Explicit zero offset (+00:00) → UTC Z (replaces +00:00 suffix with Z).
  - Explicit positive offset (+05:30) → UTC Z.
  - Already UTC but explicit (+00:00) → Z suffix.

  _build_cache_key:
  - Same lat/lon → same key (deterministic).
  - Different lat/lon → different key.
  - Key is 64-char hex string (SHA-256).
  - Lat/lon rounded to 4 decimal places (LC7).
  - Key does NOT encode credentials (privacy/leakage — LC7).

  fetch():
  - Cache hit → canonical reconstruction from cached dict; no HTTP call.
  - Cache hit with _no_reading sentinel → None returned.
  - Cache miss happy path via respx mock + real fixture → canonical AQIReading.
  - Cache miss + wire-validation failure → ProviderProtocolError.
  - Cache miss + provider HTTP 401 → KeyInvalid (L2 carry-forward).
  - Cache miss + provider HTTP 403 → KeyInvalid (L2 carry-forward).
  - Cache miss + provider HTTP 429 → QuotaExhausted (L2 carry-forward).
  - Cache miss + provider HTTP 5xx → TransientNetworkError (L2 carry-forward).
  - Cache miss + success=false + error.code="invalid_client" → KeyInvalid (LC27).
  - Cache miss + success=false + error.code="invalid_query" → ProviderProtocolError (LC27).
  - Cache miss + empty response[] → None + sentinel cached.
  - Cache miss + non-empty response but empty periods[] → None + sentinel cached.

  Capability declaration:
  - CAPABILITY.provider_id = "aeris", domain = "aqi".
  - CAPABILITY.auth_required = ("client_id", "client_secret").
  - CAPABILITY.supplied_canonical_fields includes all 12 canonical AQI fields.
  - CAPABILITY.geographic_coverage = "global".
  - wire_providers([aeris.CAPABILITY]) → registry entry for ("aqi", "aeris").

No DB, no live network. respx mocks outbound httpx calls.
ADR references: ADR-013, ADR-017, ADR-018, ADR-020, ADR-038.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "providers" / "aqi"
_AERIS_AQ_BASE_URL = "https://data.api.xweather.com"

# Coordinates matching fixture — round to 6dp to match fetch() URL construction
_LAT = 47.6062
_LON = -122.3321
_LAT4 = round(_LAT, 4)
_LON4 = round(_LON, 4)
_LAT6 = round(_LAT, 6)
_LON6 = round(_LON, 6)
# Full URL as constructed by fetch(): lat/lon rounded to 6dp per aeris.py fetch()
_AERIS_AQ_URL = f"{_AERIS_AQ_BASE_URL}/airquality/{_LAT6},{_LON6}"

_TEST_CLIENT_ID = "TEST_CLIENT_ID"
_TEST_CLIENT_SECRET = "TEST_CLIENT_SECRET"


def _load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture from tests/fixtures/providers/aqi/."""
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
    from weewx_clearskies_api.providers.aqi.aeris import _reset_http_client_for_tests  # noqa: PLC0415
    import weewx_clearskies_api.providers.aqi.aeris as _aeris  # noqa: PLC0415

    reset_cache_for_tests()
    reset_provider_registry_for_tests()
    _reset_http_client_for_tests()
    _aeris._rate_limiter._calls.clear()
    wire_cache_from_env()


# ===========================================================================
# 1. Fixture loading — Pydantic wire-shape validation
# ===========================================================================


class TestAerisFixtureLoadsCleanly:
    """Real captured fixture validates against the Aeris wire-shape Pydantic models."""

    def test_fixture_loads_as_valid_json(self) -> None:
        """aeris_current.json is parseable JSON."""
        data = _load_fixture("aeris_current.json")
        assert isinstance(data, dict), "Fixture must be a dict at top level"

    def test_fixture_success_is_true(self) -> None:
        """Fixture has success=true (real capture, not an error response)."""
        data = _load_fixture("aeris_current.json")
        assert data["success"] is True

    def test_fixture_response_is_non_empty_list(self) -> None:
        """Fixture has non-empty response[] array."""
        data = _load_fixture("aeris_current.json")
        assert isinstance(data["response"], list)
        assert len(data["response"]) > 0

    def test_fixture_periods_is_non_empty_list(self) -> None:
        """Fixture has non-empty periods[] on response[0]."""
        data = _load_fixture("aeris_current.json")
        periods = data["response"][0]["periods"]
        assert isinstance(periods, list)
        assert len(periods) > 0

    def test_fixture_validates_against_wire_model(self) -> None:
        """aeris_current.json validates cleanly against _AerisAQResponse Pydantic model."""
        from weewx_clearskies_api.providers.aqi.aeris import _AerisAQResponse  # noqa: PLC0415
        data = _load_fixture("aeris_current.json")
        response = _AerisAQResponse.model_validate(data)
        assert response.success is True
        assert len(response.response) == 1

    def test_fixture_extra_fields_ignored_by_wire_model(self) -> None:
        """Extra wire fields (health, color, method, profile) ignored (extra='ignore')."""
        from weewx_clearskies_api.providers.aqi.aeris import _AerisAQResponse  # noqa: PLC0415
        data = _load_fixture("aeris_current.json")
        # Should not raise ValidationError even though fixture has many extra fields
        response = _AerisAQResponse.model_validate(data)
        assert response is not None

    def test_fixture_six_pollutants_in_periods(self) -> None:
        """Fixture has exactly six pollutants in periods[0].pollutants[]."""
        from weewx_clearskies_api.providers.aqi.aeris import _AerisAQResponse  # noqa: PLC0415
        data = _load_fixture("aeris_current.json")
        response = _AerisAQResponse.model_validate(data)
        pollutants = response.response[0].periods[0].pollutants
        assert len(pollutants) == 6


# ===========================================================================
# 2. _wire_to_canonical — happy path from fixture
# ===========================================================================


class TestWireToCanonicalHappyPath:
    """_wire_to_canonical translates the real fixture to correct canonical AQIReading."""

    def _get_location_from_fixture(self) -> Any:
        """Load fixture and parse into _AerisLocation model."""
        from weewx_clearskies_api.providers.aqi.aeris import _AerisAQResponse  # noqa: PLC0415
        data = _load_fixture("aeris_current.json")
        response = _AerisAQResponse.model_validate(data)
        return response.response[0]

    def test_fixture_produces_canonical_aqi_reading(self) -> None:
        """_wire_to_canonical returns AQIReading (not None) for the real fixture."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._get_location_from_fixture()
        result = _wire_to_canonical(location)
        assert result is not None, "_wire_to_canonical must return AQIReading for valid fixture"

    def test_fixture_aqi_is_33(self) -> None:
        """data.aqi = 33 (from fixture; capped/rounded to int per spec)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._get_location_from_fixture()
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqi == 33, f"Expected aqi=33, got {result.aqi!r}"

    def test_fixture_aqi_category_is_good(self) -> None:
        """data.aqiCategory = 'Good' (AQI 33 → 0–50 band)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._get_location_from_fixture()
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqiCategory == "Good", (
            f"Expected aqiCategory='Good' (AQI 33), got {result.aqiCategory!r}"
        )

    def test_fixture_aqi_main_pollutant_is_O3(self) -> None:
        """data.aqiMainPollutant = 'O3' (fixture dominant='o3' normalized to canonical id)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._get_location_from_fixture()
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqiMainPollutant == "O3", (
            f"Expected aqiMainPollutant='O3', got {result.aqiMainPollutant!r}"
        )

    def test_fixture_aqi_location_is_seattle(self) -> None:
        """data.aqiLocation = 'seattle' (from place.name — NOT PARTIAL-DOMAIN for Aeris)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._get_location_from_fixture()
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqiLocation == "seattle", (
            f"Expected aqiLocation='seattle', got {result.aqiLocation!r}"
        )

    def test_fixture_source_is_aeris(self) -> None:
        """data.source = 'aeris' (provider_id literal on AQIReading)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._get_location_from_fixture()
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.source == "aeris", f"Expected source='aeris', got {result.source!r}"

    def test_fixture_observed_at_is_utc_z(self) -> None:
        """observedAt ends with Z (UTC ISO-8601, LC4 + ADR-020)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._get_location_from_fixture()
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.observedAt is not None
        assert result.observedAt.endswith("Z"), (
            f"observedAt must end with Z, got {result.observedAt!r}"
        )

    def test_fixture_observed_at_converts_minus_07_offset_to_utc(self) -> None:
        """dateTimeISO '2026-05-10T17:00:00-07:00' → '2026-05-11T00:00:00Z' (UTC)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._get_location_from_fixture()
        result = _wire_to_canonical(location)
        assert result is not None
        # 17:00 - 7 hours = 24:00 = 00:00 next day
        assert result.observedAt == "2026-05-11T00:00:00Z", (
            f"Expected '2026-05-11T00:00:00Z', got {result.observedAt!r}"
        )

    def test_fixture_o3_is_ppm_from_ppb(self) -> None:
        """pollutantO3 is in ppm (valuePPB=36 → 0.036 ppm) not µg/m³."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._get_location_from_fixture()
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.pollutantO3 is not None
        # 36 ppb / 1000 = 0.036 ppm
        assert abs(result.pollutantO3 - 0.036) < 1e-9, (
            f"O3 36 ppb → expected 0.036 ppm, got {result.pollutantO3!r}"
        )

    def test_fixture_co_is_ppm_from_ppb(self) -> None:
        """pollutantCO is in ppm (valuePPB=143 → 0.143 ppm)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._get_location_from_fixture()
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.pollutantCO is not None
        assert abs(result.pollutantCO - 0.143) < 1e-9, (
            f"CO 143 ppb → expected 0.143 ppm, got {result.pollutantCO!r}"
        )

    def test_fixture_no2_is_ppm_from_ppb(self) -> None:
        """pollutantNO2 is in ppm (valuePPB=3 → 0.003 ppm)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._get_location_from_fixture()
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.pollutantNO2 is not None
        assert abs(result.pollutantNO2 - 0.003) < 1e-9, (
            f"NO2 3 ppb → expected 0.003 ppm, got {result.pollutantNO2!r}"
        )

    def test_fixture_so2_is_ppm_from_ppb_zero(self) -> None:
        """pollutantSO2 is in ppm (valuePPB=0 → 0.0 ppm)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._get_location_from_fixture()
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.pollutantSO2 is not None
        assert result.pollutantSO2 == 0.0, (
            f"SO2 0 ppb → expected 0.0 ppm, got {result.pollutantSO2!r}"
        )

    def test_fixture_pm25_is_ugm3_passthrough(self) -> None:
        """pollutantPM25 is in µg/m³ (valueUGM3=5.8 passthrough — no conversion)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._get_location_from_fixture()
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.pollutantPM25 == 5.8, (
            f"PM2.5 valueUGM3=5.8 → expected 5.8 µg/m³, got {result.pollutantPM25!r}"
        )

    def test_fixture_pm10_is_ugm3_passthrough(self) -> None:
        """pollutantPM10 is in µg/m³ (valueUGM3=8.0 passthrough — no conversion)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._get_location_from_fixture()
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.pollutantPM10 == 8.0, (
            f"PM10 valueUGM3=8.0 → expected 8.0 µg/m³, got {result.pollutantPM10!r}"
        )


# ===========================================================================
# 3. _wire_to_canonical — edge cases
# ===========================================================================


class TestWireToCanonicalEdgeCases:
    """Edge cases for _wire_to_canonical."""

    def _make_minimal_location(
        self,
        aqi: float | None = 42.0,
        dominant: str | None = "pm2.5",
        pollutants: list[dict[str, Any]] | None = None,
        place_name: str | None = "testcity",
        date_time_iso: str = "2026-04-30T10:00:00-07:00",
    ) -> Any:
        """Build a minimal _AerisLocation for edge-case tests."""
        from weewx_clearskies_api.providers.aqi.aeris import _AerisAQResponse  # noqa: PLC0415
        if pollutants is None:
            pollutants = [
                {"type": "pm2.5", "valuePPB": None, "valueUGM3": 8.5},
            ]
        data = {
            "success": True,
            "error": None,
            "response": [{
                "place": {"name": place_name},
                "periods": [{
                    "dateTimeISO": date_time_iso,
                    "aqi": aqi,
                    "dominant": dominant,
                    "pollutants": pollutants,
                }],
            }],
        }
        resp = _AerisAQResponse.model_validate(data)
        return resp.response[0]

    def test_empty_pollutants_list_yields_none_per_pollutant_fields(self) -> None:
        """Empty pollutants[] → all per-pollutant canonical fields are None."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(pollutants=[])
        result = _wire_to_canonical(location)
        assert result is not None, "aqi=42 still provides has_data=True"
        assert result.pollutantPM25 is None
        assert result.pollutantPM10 is None
        assert result.pollutantO3 is None
        assert result.pollutantNO2 is None
        assert result.pollutantSO2 is None
        assert result.pollutantCO is None

    def test_empty_pollutants_list_aqi_and_category_still_populate(self) -> None:
        """Empty pollutants[] → aqi and aqiCategory still populate from period.aqi."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(aqi=75.0, pollutants=[])
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqi == 75, f"Expected aqi=75, got {result.aqi!r}"
        assert result.aqiCategory == "Moderate", (
            f"AQI 75 → expected 'Moderate', got {result.aqiCategory!r}"
        )

    def test_dominant_pm1_yields_none_main_pollutant(self) -> None:
        """dominant='pm1' → aqiMainPollutant=None (pm1 dropped — no canonical PM1 field)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(dominant="pm1")
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqiMainPollutant is None, (
            f"Expected aqiMainPollutant=None for pm1 dominant, got {result.aqiMainPollutant!r}"
        )

    def test_dominant_none_yields_none_main_pollutant(self) -> None:
        """dominant=None → aqiMainPollutant=None (missing dominant)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(dominant=None)
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqiMainPollutant is None, (
            f"Expected aqiMainPollutant=None for None dominant, got {result.aqiMainPollutant!r}"
        )

    def test_all_null_aqi_and_pollutants_returns_none(self) -> None:
        """aqi=None + all pollutant values null → _wire_to_canonical returns None."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(
            aqi=None,
            pollutants=[
                {"type": "pm2.5", "valuePPB": None, "valueUGM3": None},
                {"type": "pm10", "valuePPB": None, "valueUGM3": None},
                {"type": "o3", "valuePPB": None, "valueUGM3": None},
            ],
        )
        result = _wire_to_canonical(location)
        assert result is None, (
            "_wire_to_canonical must return None when aqi=None and all pollutants null"
        )

    def test_pm1_pollutant_entry_does_not_appear_in_canonical_output(self) -> None:
        """pm1 entry in pollutants[] is silently dropped (no pollutantPM1 on canonical)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(
            pollutants=[
                {"type": "pm1", "valuePPB": None, "valueUGM3": 3.2},
                {"type": "pm2.5", "valuePPB": None, "valueUGM3": 5.0},
            ],
            dominant="pm2.5",
        )
        result = _wire_to_canonical(location)
        assert result is not None
        # pm1 must not appear anywhere on the canonical record
        result_dict = result.model_dump()
        assert "pollutantPM1" not in result_dict, (
            "pollutantPM1 must NOT appear in canonical output"
        )

    def test_pm25_dominant_normalizes_to_canonical_PM25(self) -> None:
        """dominant='pm2.5' → aqiMainPollutant='PM2.5' (with dot, canonical spelling)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(dominant="pm2.5")
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqiMainPollutant == "PM2.5"

    def test_pm10_dominant_normalizes_to_canonical_PM10(self) -> None:
        """dominant='pm10' → aqiMainPollutant='PM10'."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(dominant="pm10")
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqiMainPollutant == "PM10"

    def test_co_dominant_normalizes_to_canonical_CO(self) -> None:
        """dominant='co' → aqiMainPollutant='CO'."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(dominant="co")
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqiMainPollutant == "CO"

    def test_no2_dominant_normalizes_to_canonical_NO2(self) -> None:
        """dominant='no2' → aqiMainPollutant='NO2'."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(dominant="no2")
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqiMainPollutant == "NO2"

    def test_so2_dominant_normalizes_to_canonical_SO2(self) -> None:
        """dominant='so2' → aqiMainPollutant='SO2'."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(dominant="so2")
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqiMainPollutant == "SO2"

    def test_o3_dominant_normalizes_to_canonical_O3(self) -> None:
        """dominant='o3' → aqiMainPollutant='O3' (from real fixture dominant)."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(dominant="o3")
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqiMainPollutant == "O3"

    def test_aqi_capped_at_500(self) -> None:
        """aqi > 500 is capped to 500 (defensive; min(round(aqi), 500))."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(aqi=600.0)
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqi == 500, f"aqi > 500 must be capped at 500, got {result.aqi!r}"

    def test_aqi_float_rounds_to_int(self) -> None:
        """aqi=42.7 → 43 (rounds to int via round())."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(aqi=42.7)
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqi == 43, f"42.7 → expected 43 (rounded), got {result.aqi!r}"

    def test_none_place_name_yields_none_aqi_location(self) -> None:
        """place.name=None → aqiLocation=None."""
        from weewx_clearskies_api.providers.aqi.aeris import _wire_to_canonical  # noqa: PLC0415
        location = self._make_minimal_location(place_name=None)
        result = _wire_to_canonical(location)
        assert result is not None
        assert result.aqiLocation is None


# ===========================================================================
# 4. Explicit-offset → UTC Z conversion (via shared to_utc_iso8601_from_offset)
# ===========================================================================


class TestIsoOffsetToUtcZ:
    """Aeris observedAt uses to_utc_iso8601_from_offset (shared helper, ADR-020).

    The impl uses providers._common.datetime_utils.to_utc_iso8601_from_offset
    rather than a module-local _iso_offset_to_utc_z helper (DRY rule applied
    at impl time; same helper used by alerts/forecast aeris modules).

    Tests verify:
    - The shared helper converts explicit-offset ISO-8601 → UTC Z correctly.
    - The conversion is exercised end-to-end via _wire_to_canonical (observedAt
      field on the canonical record reflects the UTC conversion).
    """

    def test_minus_07_offset_converts_to_utc_z(self) -> None:
        """-07:00 offset converts correctly to UTC Z (shared helper)."""
        from weewx_clearskies_api.providers._common.datetime_utils import (  # noqa: PLC0415
            to_utc_iso8601_from_offset,
        )
        result = to_utc_iso8601_from_offset(
            "2026-04-30T10:00:00-07:00", provider_id="aeris", domain="aqi"
        )
        # 10:00 + 7 = 17:00 UTC
        assert result == "2026-04-30T17:00:00Z", (
            f"Expected '2026-04-30T17:00:00Z', got {result!r}"
        )

    def test_zero_offset_converts_to_utc_z(self) -> None:
        """+00:00 offset → UTC Z (replaces +00:00 with Z, shared helper)."""
        from weewx_clearskies_api.providers._common.datetime_utils import (  # noqa: PLC0415
            to_utc_iso8601_from_offset,
        )
        result = to_utc_iso8601_from_offset(
            "2026-04-30T17:00:00+00:00", provider_id="aeris", domain="aqi"
        )
        assert result == "2026-04-30T17:00:00Z", (
            f"Expected '2026-04-30T17:00:00Z' for +00:00 input, got {result!r}"
        )

    def test_positive_offset_converts_to_utc_z(self) -> None:
        """+05:30 offset (IST) converts to UTC Z (shared helper)."""
        from weewx_clearskies_api.providers._common.datetime_utils import (  # noqa: PLC0415
            to_utc_iso8601_from_offset,
        )
        result = to_utc_iso8601_from_offset(
            "2026-04-30T10:30:00+05:30", provider_id="aeris", domain="aqi"
        )
        # 10:30 - 5:30 = 5:00 UTC
        assert result == "2026-04-30T05:00:00Z", (
            f"Expected '2026-04-30T05:00:00Z', got {result!r}"
        )

    def test_result_ends_with_z(self) -> None:
        """Result always ends with Z (ADR-020 UTC at API boundary)."""
        from weewx_clearskies_api.providers._common.datetime_utils import (  # noqa: PLC0415
            to_utc_iso8601_from_offset,
        )
        result = to_utc_iso8601_from_offset(
            "2026-04-30T10:00:00-07:00", provider_id="aeris", domain="aqi"
        )
        assert result.endswith("Z"), f"Result must end with Z, got {result!r}"

    def test_fixture_timestamp_converts_correctly_via_wire_to_canonical(self) -> None:
        """Real fixture dateTimeISO '2026-05-10T17:00:00-07:00' → '2026-05-11T00:00:00Z'.

        Tests the conversion end-to-end via _wire_to_canonical (observedAt field).
        """
        from weewx_clearskies_api.providers.aqi.aeris import (  # noqa: PLC0415
            _AerisAQResponse,
            _wire_to_canonical,
        )
        data = {
            "success": True,
            "error": None,
            "response": [{
                "place": {"name": "testcity"},
                "periods": [{
                    "dateTimeISO": "2026-05-10T17:00:00-07:00",
                    "aqi": 33.0,
                    "dominant": "o3",
                    "pollutants": [{"type": "pm2.5", "valuePPB": None, "valueUGM3": 5.0}],
                }],
            }],
        }
        resp = _AerisAQResponse.model_validate(data)
        result = _wire_to_canonical(resp.response[0])
        assert result is not None
        assert result.observedAt == "2026-05-11T00:00:00Z", (
            f"Expected '2026-05-11T00:00:00Z', got {result.observedAt!r}"
        )

    def test_midnight_offset_wraps_to_next_day_correctly(self) -> None:
        """Midnight local with -07:00 offset → 07:00 UTC (shared helper)."""
        from weewx_clearskies_api.providers._common.datetime_utils import (  # noqa: PLC0415
            to_utc_iso8601_from_offset,
        )
        result = to_utc_iso8601_from_offset(
            "2026-04-30T00:00:00-07:00", provider_id="aeris", domain="aqi"
        )
        # midnight local = 07:00 UTC
        assert result == "2026-04-30T07:00:00Z", (
            f"Expected '2026-04-30T07:00:00Z', got {result!r}"
        )


# ===========================================================================
# 5. _build_cache_key — determinism and privacy
# ===========================================================================


class TestBuildCacheKey:
    """_build_cache_key is deterministic, rounds lat/lon, and excludes credentials."""

    def test_same_lat_lon_produces_same_key(self) -> None:
        """Same lat/lon → same cache key (deterministic)."""
        from weewx_clearskies_api.providers.aqi.aeris import _build_cache_key  # noqa: PLC0415
        key1 = _build_cache_key(47.6062, -122.3321)
        key2 = _build_cache_key(47.6062, -122.3321)
        assert key1 == key2

    def test_different_lat_lon_produces_different_key(self) -> None:
        """Different lat/lon → different cache key."""
        from weewx_clearskies_api.providers.aqi.aeris import _build_cache_key  # noqa: PLC0415
        key1 = _build_cache_key(47.6062, -122.3321)
        key2 = _build_cache_key(40.7128, -74.0060)
        assert key1 != key2

    def test_key_is_64_char_hex_string(self) -> None:
        """Cache key is a 64-character hexadecimal string (SHA-256)."""
        from weewx_clearskies_api.providers.aqi.aeris import _build_cache_key  # noqa: PLC0415
        key = _build_cache_key(47.6062, -122.3321)
        assert len(key) == 64, f"Expected 64-char key, got {len(key)!r}"
        assert all(c in "0123456789abcdef" for c in key), (
            "Cache key must be lowercase hex"
        )

    def test_lat_lon_rounded_to_4_decimal_places(self) -> None:
        """High-precision lat/lon rounds to 4dp — keys match for equivalent coordinates."""
        from weewx_clearskies_api.providers.aqi.aeris import _build_cache_key  # noqa: PLC0415
        # These differ only beyond 4dp — should produce the same key
        key1 = _build_cache_key(47.60620001, -122.33210001)
        key2 = _build_cache_key(47.60620009, -122.33210009)
        assert key1 == key2, (
            "Coordinates identical at 4dp must produce the same cache key"
        )

    def test_aeris_aqi_key_distinct_from_openmeteo_key(self) -> None:
        """Aeris AQI key differs from Open-Meteo key at same coordinates (provider_id differs)."""
        from weewx_clearskies_api.providers.aqi.aeris import _build_cache_key  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.openmeteo import _build_cache_key as om_key  # noqa: PLC0415
        aeris_key = _build_cache_key(47.6062, -122.3321)
        openmeteo_key = om_key(47.6062, -122.3321)
        assert aeris_key != openmeteo_key, (
            "Aeris and Open-Meteo must have distinct cache keys at same coordinates"
        )

    def test_credentials_not_in_cache_key(self) -> None:
        """Cache key signature takes only lat/lon — credentials cannot be embedded."""
        from weewx_clearskies_api.providers.aqi.aeris import _build_cache_key  # noqa: PLC0415
        import inspect  # noqa: PLC0415
        sig = inspect.signature(_build_cache_key)
        param_names = list(sig.parameters.keys())
        # Must not accept client_id or client_secret
        assert "client_id" not in param_names, (
            "_build_cache_key must not accept client_id (credentials not in key)"
        )
        assert "client_secret" not in param_names, (
            "_build_cache_key must not accept client_secret (credentials not in key)"
        )


# ===========================================================================
# 6. fetch() — cache hit paths
# ===========================================================================


class TestFetchCacheHit:
    """fetch() returns cached canonical record without making HTTP calls."""

    def setup_method(self) -> None:
        _reset_provider_state()

    def test_cache_hit_returns_canonical_reading_without_http_call(self) -> None:
        """Cache hit → canonical AQIReading returned; no outbound HTTP call made."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import _build_cache_key, fetch  # noqa: PLC0415
        from weewx_clearskies_api.models.responses import AQIReading  # noqa: PLC0415

        # Pre-populate cache with a known record
        reading = AQIReading(
            aqi=33,
            aqiCategory="Good",
            aqiMainPollutant="O3",
            aqiLocation="seattle",
            pollutantPM25=5.8,
            pollutantO3=0.036,
            source="aeris",
            observedAt="2026-05-11T00:00:00Z",
        )
        cache_key = _build_cache_key(_LAT, _LON)
        get_cache().set(cache_key, reading.model_dump(), ttl_seconds=900)

        with respx.mock(assert_all_called=False) as mock:
            # No routes added — any HTTP call would raise
            result = fetch(
                lat=_LAT,
                lon=_LON,
                client_id=_TEST_CLIENT_ID,
                client_secret=_TEST_CLIENT_SECRET,
            )
            assert len(mock.calls) == 0, "No HTTP calls expected on cache hit"

        assert result is not None
        assert result.aqi == 33
        assert result.aqiLocation == "seattle"
        assert result.source == "aeris"

    def test_cache_hit_sentinel_returns_none_without_http_call(self) -> None:
        """Cache hit with _no_reading sentinel → None; no outbound HTTP call."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import _build_cache_key, fetch  # noqa: PLC0415

        cache_key = _build_cache_key(_LAT, _LON)
        get_cache().set(cache_key, {"_no_reading": True}, ttl_seconds=900)

        with respx.mock(assert_all_called=False) as mock:
            result = fetch(
                lat=_LAT,
                lon=_LON,
                client_id=_TEST_CLIENT_ID,
                client_secret=_TEST_CLIENT_SECRET,
            )
            assert len(mock.calls) == 0

        assert result is None


# ===========================================================================
# 7. fetch() — cache miss + HTTP paths
# ===========================================================================


class TestFetchCacheMiss:
    """fetch() cache miss paths: happy path, errors, sentinel."""

    def setup_method(self) -> None:
        _reset_provider_state()

    def test_cache_miss_happy_path_returns_canonical_reading(self) -> None:
        """Cache miss + valid fixture from respx → canonical AQIReading returned."""
        from weewx_clearskies_api.providers.aqi.aeris import fetch  # noqa: PLC0415
        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=True) as mock:
            mock.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            result = fetch(
                lat=_LAT,
                lon=_LON,
                client_id=_TEST_CLIENT_ID,
                client_secret=_TEST_CLIENT_SECRET,
            )

        assert result is not None
        assert result.aqi == 33
        assert result.aqiLocation == "seattle"
        assert result.source == "aeris"

    def test_cache_miss_happy_path_populates_cache(self) -> None:
        """Cache miss happy path → result cached for subsequent reads."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import _build_cache_key, fetch  # noqa: PLC0415
        data = _load_fixture("aeris_current.json")

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            fetch(lat=_LAT, lon=_LON, client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET)

        cache_key = _build_cache_key(_LAT, _LON)
        cached = get_cache().get(cache_key)
        assert cached is not None, "Result must be cached after cache miss"

    def test_wire_validation_failure_raises_provider_protocol_error(self) -> None:
        """Malformed JSON response → ProviderProtocolError (wire-validation failure)."""
        from weewx_clearskies_api.providers._common.errors import ProviderProtocolError  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import fetch  # noqa: PLC0415

        malformed = {"not_success": "bad", "totally": "wrong"}

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=malformed)
            )
            with pytest.raises(ProviderProtocolError):
                fetch(lat=_LAT, lon=_LON, client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET)

    def test_http_401_raises_key_invalid(self) -> None:
        """HTTP 401 from Aeris → KeyInvalid (L2 carry-forward from ProviderHTTPClient)."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import fetch  # noqa: PLC0415

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(401, json={"error": "unauthorized"})
            )
            with pytest.raises(KeyInvalid):
                fetch(lat=_LAT, lon=_LON, client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET)

    def test_http_403_raises_key_invalid(self) -> None:
        """HTTP 403 from Aeris → KeyInvalid (L2 carry-forward from ProviderHTTPClient)."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import fetch  # noqa: PLC0415

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(403, json={"error": "forbidden"})
            )
            with pytest.raises(KeyInvalid):
                fetch(lat=_LAT, lon=_LON, client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET)

    def test_http_429_raises_quota_exhausted(self) -> None:
        """HTTP 429 from Aeris → QuotaExhausted (L2 carry-forward from ProviderHTTPClient)."""
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import fetch  # noqa: PLC0415

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={"reason": "rate limit"},
                    headers={"Retry-After": "60"},
                )
            )
            with pytest.raises(QuotaExhausted):
                fetch(lat=_LAT, lon=_LON, client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET)

    def test_http_429_quota_exhausted_preserves_retry_after(self) -> None:
        """HTTP 429 QuotaExhausted includes retry_after_seconds from Retry-After header."""
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import fetch  # noqa: PLC0415

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={"reason": "rate limit"},
                    headers={"Retry-After": "90"},
                )
            )
            with pytest.raises(QuotaExhausted) as exc_info:
                fetch(lat=_LAT, lon=_LON, client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET)
        assert exc_info.value.retry_after_seconds == 90

    def test_http_500_raises_transient_network_error(self) -> None:
        """HTTP 5xx from Aeris → TransientNetworkError (L2 carry-forward)."""
        from weewx_clearskies_api.providers._common.errors import TransientNetworkError  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import fetch  # noqa: PLC0415

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(500, json={"reason": "server error"})
            )
            with pytest.raises(TransientNetworkError):
                fetch(lat=_LAT, lon=_LON, client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET)

    def test_success_false_invalid_client_raises_key_invalid(self) -> None:
        """success=false + error.code='invalid_client' → KeyInvalid (LC27 envelope mapping)."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import fetch  # noqa: PLC0415

        envelope_error = {
            "success": False,
            "error": {"code": "invalid_client", "description": "Invalid client credentials"},
            "response": [],
        }

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=envelope_error)
            )
            with pytest.raises(KeyInvalid):
                fetch(lat=_LAT, lon=_LON, client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET)

    def test_success_false_insufficient_scope_raises_key_invalid(self) -> None:
        """success=false + error.code='insufficient_scope' → KeyInvalid (LC27)."""
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import fetch  # noqa: PLC0415

        envelope_error = {
            "success": False,
            "error": {"code": "insufficient_scope", "description": "Plan does not include airquality"},
            "response": [],
        }

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=envelope_error)
            )
            with pytest.raises(KeyInvalid):
                fetch(lat=_LAT, lon=_LON, client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET)

    def test_success_false_invalid_query_raises_provider_protocol_error(self) -> None:
        """success=false + error.code='invalid_query' → ProviderProtocolError (LC27)."""
        from weewx_clearskies_api.providers._common.errors import ProviderProtocolError  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import fetch  # noqa: PLC0415

        envelope_error = {
            "success": False,
            "error": {"code": "invalid_query", "description": "The location query is invalid"},
            "response": [],
        }

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=envelope_error)
            )
            with pytest.raises(ProviderProtocolError):
                fetch(lat=_LAT, lon=_LON, client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET)

    def test_empty_response_array_returns_none_and_caches_sentinel(self) -> None:
        """success=true + empty response[] → None returned + sentinel cached."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import _build_cache_key, fetch  # noqa: PLC0415

        empty_response = {"success": True, "error": None, "response": []}

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=empty_response)
            )
            result = fetch(lat=_LAT, lon=_LON, client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET)

        assert result is None, "Empty response[] must return None"
        cache_key = _build_cache_key(_LAT, _LON)
        cached = get_cache().get(cache_key)
        assert cached == {"_no_reading": True}, (
            "Empty response[] must cache _no_reading sentinel"
        )

    def test_empty_periods_returns_none_and_caches_sentinel(self) -> None:
        """success=true + response[0] with empty periods[] → None + sentinel cached."""
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.aeris import _build_cache_key, fetch  # noqa: PLC0415

        empty_periods = {
            "success": True,
            "error": None,
            "response": [{
                "place": {"name": "seattle"},
                "periods": [],
            }],
        }

        with respx.mock(assert_all_called=False):
            respx.get(_AERIS_AQ_URL).mock(
                return_value=httpx.Response(200, json=empty_periods)
            )
            result = fetch(lat=_LAT, lon=_LON, client_id=_TEST_CLIENT_ID, client_secret=_TEST_CLIENT_SECRET)

        assert result is None, "Empty periods[] must return None"
        cache_key = _build_cache_key(_LAT, _LON)
        cached = get_cache().get(cache_key)
        assert cached == {"_no_reading": True}, (
            "Empty periods[] must cache _no_reading sentinel"
        )


# ===========================================================================
# 8. Capability declaration
# ===========================================================================


class TestCapabilityDeclaration:
    """CAPABILITY symbol declares the correct provider metadata."""

    def test_capability_provider_id_is_aeris(self) -> None:
        """CAPABILITY.provider_id = 'aeris'."""
        from weewx_clearskies_api.providers.aqi.aeris import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.provider_id == "aeris"

    def test_capability_domain_is_aqi(self) -> None:
        """CAPABILITY.domain = 'aqi'."""
        from weewx_clearskies_api.providers.aqi.aeris import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.domain == "aqi"

    def test_capability_auth_required_includes_client_id_and_secret(self) -> None:
        """CAPABILITY.auth_required contains 'client_id' and 'client_secret'."""
        from weewx_clearskies_api.providers.aqi.aeris import CAPABILITY  # noqa: PLC0415
        assert "client_id" in CAPABILITY.auth_required
        assert "client_secret" in CAPABILITY.auth_required

    def test_capability_geographic_coverage_is_global(self) -> None:
        """CAPABILITY.geographic_coverage = 'global' (Aeris /airquality is global)."""
        from weewx_clearskies_api.providers.aqi.aeris import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.geographic_coverage == "global"

    def test_capability_supplied_fields_includes_all_12_canonical_aqi_fields(self) -> None:
        """CAPABILITY.supplied_canonical_fields includes all 12 canonical AQI fields."""
        from weewx_clearskies_api.providers.aqi.aeris import CAPABILITY  # noqa: PLC0415
        required_fields = {
            "aqi", "aqiCategory", "aqiMainPollutant", "aqiLocation",
            "pollutantPM25", "pollutantPM10",
            "pollutantO3", "pollutantNO2", "pollutantSO2", "pollutantCO",
            "observedAt", "source",
        }
        supplied = set(CAPABILITY.supplied_canonical_fields)
        missing = required_fields - supplied
        assert not missing, (
            f"CAPABILITY.supplied_canonical_fields missing: {missing!r}"
        )

    def test_wire_providers_registers_aeris_aqi_capability(self) -> None:
        """wire_providers([aeris.CAPABILITY]) → registry entry for ('aqi', 'aeris')."""
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            get_provider_registry,
            reset_provider_registry_for_tests,
            wire_providers,
        )
        from weewx_clearskies_api.providers.aqi.aeris import CAPABILITY  # noqa: PLC0415

        reset_provider_registry_for_tests()
        wire_providers([CAPABILITY])
        registry = get_provider_registry()
        # Registry is a list of ProviderCapability objects
        assert any(p.provider_id == "aeris" and p.domain == "aqi" for p in registry), (
            "wire_providers must register aeris aqi capability in registry"
        )

    def test_capability_default_poll_interval_is_900_seconds(self) -> None:
        """CAPABILITY.default_poll_interval_seconds = 900 (ADR-017 AQI TTL)."""
        from weewx_clearskies_api.providers.aqi.aeris import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.default_poll_interval_seconds == 900, (
            f"Expected 900s TTL, got {CAPABILITY.default_poll_interval_seconds!r}"
        )
