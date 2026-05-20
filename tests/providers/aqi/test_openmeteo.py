"""Unit tests for the Open-Meteo AQI provider module (3b-9).

Covers per the task-3b-9 brief §Test-author parallel scope (test_openmeteo.py):

  Wire-shape Pydantic validation:
  - Real captured fixture loads cleanly against _OpenMeteoAQResponse.
  - Extra fields (elevation, generationtime_ms, current_units) ignored (extra="ignore").
  - Missing required 'time' field in current block → ValidationError → ProviderProtocolError.
  - Missing 'current' block entirely → ValidationError → ProviderProtocolError.

  _wire_to_canonical happy path:
  - Real fixture → canonical AQIReading with all fields populated correctly.
  - aqi = 73 → int round; aqiCategory = "Moderate"; aqiMainPollutant = "PM2.5".
  - observedAt = "2026-05-10T22:00:00Z" (LC4: append Z to GMT-naive local time).
  - source = "openmeteo" (provider_id literal).
  - aqiLocation = None (PARTIAL-DOMAIN — Open-Meteo has no location field).
  - Per-pollutant conversions correct (O3/NO2/SO2/CO µg/m³→ppm; PM2.5/PM10 passthrough).

  _wire_to_canonical edge cases:
  - All-null fixture → returns None (no useful reading; sentinel path).
  - us_aqi_only fixture → aqi populates, aqiMainPollutant=None, per-pollutant=None.
  - Partial nulls: some sub-AQIs null, main AQI populated → argmax over non-null only.

  _main_pollutant_from_sub_aqis:
  - Argmax: highest sub-AQI wins.
  - Tie-break: PM2.5 beats PM10 when tied (table order per LC14).
  - Tie-break: PM10 beats NO2 when tied (table order).
  - All None → None (no main pollutant derivable).
  - Single non-None value wins regardless of value.
  - Mapping table: each sub-AQI key maps to correct canonical pollutant id.

  _build_cache_key:
  - Same lat/lon → same key (deterministic).
  - Different lat/lon → different key.
  - Key is 64-char hex string (SHA-256).
  - Lat/lon rounded to 4 decimal places (LC7).
  - AQI key distinct from forecast key at same coordinates.

  fetch() cache paths:
  - Cache hit → canonical reconstruction from cached dict; no HTTP call.
  - Cache hit with _no_reading sentinel → returns None; no HTTP call.
  - Cache miss + valid wire → canonical record returned + cached.
  - Cache miss + all-null response → None returned + sentinel cached.

  fetch() error paths:
  - Cache miss + wire-validation failure → ProviderProtocolError.
  - Cache miss + provider 5xx → TransientNetworkError (L2 bare propagation).
  - Cache miss + provider 429 → QuotaExhausted (L2 bare propagation).
  - Cache miss + provider 422 → ProviderProtocolError or TransientNetworkError (L2 propagation).
  - retry_after_seconds propagated on QuotaExhausted (3b-4 F1 carry-forward).

  Capability:
  - CAPABILITY.provider_id = "openmeteo", domain = "aqi".
  - CAPABILITY.auth_required = () (keyless).
  - CAPABILITY.geographic_coverage = "global".
  - CAPABILITY.supplied_canonical_fields includes 11 expected fields.
  - CAPABILITY.supplied_canonical_fields excludes aqiLocation (PARTIAL-DOMAIN).
  - CAPABILITY.default_poll_interval_seconds = 900 (15 min per ADR-017 / LC3).
  - wire_providers([CAPABILITY]) → registry has openmeteo aqi entry.

No DB, no live network. respx mocks outbound httpx calls.
Wire-shape rule: fixtures loaded from tests/fixtures/providers/aqi/.
ADR references: ADR-013, ADR-017, ADR-038.
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

_LAT = 47.6062
_LON = -122.3321
_OPENMETEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


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
    # Clear rate-limiter deque so consecutive tests don't trip each other.
    _om_aqi._rate_limiter._calls.clear()
    # Re-wire a clean memory cache (CLEARSKIES_CACHE_URL unset in unit test env).
    wire_cache_from_env()


# ===========================================================================
# 1. Wire-shape Pydantic validation
# ===========================================================================


class TestWireShapePydanticValidation:
    """Wire-shape models validate correctly against the fixture shapes."""

    def test_real_fixture_loads_cleanly_via_response_model(self) -> None:
        """openmeteo_current.json loads via _OpenMeteoAQResponse without error."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (  # noqa: PLC0415
            _OpenMeteoAQResponse,
        )
        raw = _load_fixture("openmeteo_current.json")
        response = _OpenMeteoAQResponse.model_validate(raw)
        assert response.current is not None
        assert response.current.us_aqi == 73

    def test_real_fixture_extra_top_level_fields_are_ignored(self) -> None:
        """Top-level extra fields (elevation, generationtime_ms) ignored (extra='ignore')."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (  # noqa: PLC0415
            _OpenMeteoAQResponse,
        )
        raw = _load_fixture("openmeteo_current.json")
        # Confirm the fixture has extras that would be rejected by extra="forbid"
        assert "elevation" in raw
        assert "generationtime_ms" in raw
        # Should not raise
        response = _OpenMeteoAQResponse.model_validate(raw)
        assert response.latitude == pytest.approx(47.600006, rel=1e-5)

    def test_current_block_time_field_is_required(self) -> None:
        """Missing 'time' in current block → ValidationError (time is required)."""
        from pydantic import ValidationError  # noqa: PLC0415

        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _OpenMeteoAQResponse,  # noqa: PLC0415
        )
        broken = _load_fixture("openmeteo_current.json")
        del broken["current"]["time"]
        with pytest.raises(ValidationError):
            _OpenMeteoAQResponse.model_validate(broken)

    def test_missing_current_block_raises_validation_error(self) -> None:
        """Missing top-level 'current' key → ValidationError (current is required)."""
        from pydantic import ValidationError  # noqa: PLC0415

        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _OpenMeteoAQResponse,  # noqa: PLC0415
        )
        broken = _load_fixture("openmeteo_current.json")
        del broken["current"]
        with pytest.raises(ValidationError):
            _OpenMeteoAQResponse.model_validate(broken)

    def test_all_null_fixture_loads_cleanly(self) -> None:
        """openmeteo_current_all_null.json loads cleanly (all fields optional except time)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _OpenMeteoAQResponse,  # noqa: PLC0415
        )
        raw = _load_fixture("openmeteo_current_all_null.json")
        response = _OpenMeteoAQResponse.model_validate(raw)
        assert response.current.us_aqi is None
        assert response.current.pm2_5 is None

    def test_us_aqi_only_fixture_loads_cleanly(self) -> None:
        """openmeteo_current_us_aqi_only.json loads cleanly."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _OpenMeteoAQResponse,  # noqa: PLC0415
        )
        raw = _load_fixture("openmeteo_current_us_aqi_only.json")
        response = _OpenMeteoAQResponse.model_validate(raw)
        assert response.current.us_aqi == 73
        assert response.current.us_aqi_pm2_5 is None

    def test_current_extra_fields_ignored(self) -> None:
        """Extra fields in current block are silently ignored (extra='ignore')."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _OpenMeteoAQResponse,  # noqa: PLC0415
        )
        raw = _load_fixture("openmeteo_current.json")
        raw["current"]["FUTURE_FIELD_NOT_IN_SPEC"] = "should be ignored"
        # Should not raise
        response = _OpenMeteoAQResponse.model_validate(raw)
        assert response.current.time == "2026-05-10T22:00"

    def test_latitude_and_longitude_are_required(self) -> None:
        """Missing latitude → ValidationError (latitude is a required field)."""
        from pydantic import ValidationError  # noqa: PLC0415

        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _OpenMeteoAQResponse,  # noqa: PLC0415
        )
        raw = _load_fixture("openmeteo_current.json")
        del raw["latitude"]
        with pytest.raises(ValidationError):
            _OpenMeteoAQResponse.model_validate(raw)


# ===========================================================================
# 2. _wire_to_canonical — happy path
# ===========================================================================


class TestWireToCanonicalHappyPath:
    """_wire_to_canonical translates the real fixture to a correct AQIReading."""

    def _load_wire(self, filename: str = "openmeteo_current.json") -> Any:
        """Load and parse a fixture into the wire model."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _OpenMeteoAQResponse,  # noqa: PLC0415
        )
        return _OpenMeteoAQResponse.model_validate(_load_fixture(filename))

    def test_aqi_field_is_integer_73(self) -> None:
        """Real fixture us_aqi=73 → canonical aqi=73 (rounded to int)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.aqi == 73, f"Expected aqi=73, got {reading.aqi!r}"

    def test_aqi_scale_is_epa(self) -> None:
        """aqiScale = 'epa' (Open-Meteo us_aqi is EPA 0–500 native)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.aqiScale == "epa", (
            f"Expected aqiScale='epa', got {reading.aqiScale!r}"
        )

    def test_aqi_category_is_none(self) -> None:
        """aqiCategory = None (dashboard-computed; parsers set None)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.aqiCategory is None, (
            f"Expected aqiCategory=None (dashboard-computed), got {reading.aqiCategory!r}"
        )

    def test_aqi_main_pollutant_is_pm25_for_highest_sub_aqi(self) -> None:
        """us_aqi_pm2_5=73 (highest sub-AQI) → aqiMainPollutant='PM2.5' (LC14 argmax)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.aqiMainPollutant == "PM2.5", (
            f"Argmax of sub-AQIs → expected 'PM2.5', got {reading.aqiMainPollutant!r}"
        )

    def test_aqi_location_is_none_partial_domain(self) -> None:
        """aqiLocation = None (PARTIAL-DOMAIN — Open-Meteo has no location label field)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.aqiLocation is None, (
            f"Expected aqiLocation=None (PARTIAL-DOMAIN), got {reading.aqiLocation!r}"
        )

    def test_source_is_openmeteo_literal(self) -> None:
        """source = 'openmeteo' (provider_id literal per LC16)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.source == "openmeteo", (
            f"Expected source='openmeteo', got {reading.source!r}"
        )

    def test_observed_at_is_utc_z_format(self) -> None:
        """observedAt ends with Z (UTC ISO-8601 per LC4 + ADR-020)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.observedAt.endswith("Z"), (
            f"observedAt must end with Z, got {reading.observedAt!r}"
        )

    def test_observed_at_matches_fixture_time_with_z_suffix(self) -> None:
        """observedAt = '2026-05-10T22:00:00Z' (fixture time='2026-05-10T22:00' + Z suffix)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert reading is not None
        # LC4: append Z and add :00 seconds for canonical ISO8601
        assert reading.observedAt == "2026-05-10T22:00:00Z", (
            f"Expected '2026-05-10T22:00:00Z', got {reading.observedAt!r}"
        )

    def test_pollutant_pm25_passes_through_in_ugm3(self) -> None:
        """pollutantPM25 = 3.1 µg/m³ (fixture pm2_5=3.1; passthrough, no conversion)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.pollutantPM25 == pytest.approx(3.1, rel=1e-6), (
            f"pollutantPM25 should be 3.1 µg/m³ (passthrough), got {reading.pollutantPM25!r}"
        )

    def test_pollutant_pm10_passes_through_in_ugm3(self) -> None:
        """pollutantPM10 = 4.5 µg/m³ (fixture pm10=4.5; passthrough, no conversion)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.pollutantPM10 == pytest.approx(4.5, rel=1e-6), (
            f"pollutantPM10 should be 4.5 µg/m³ (passthrough), got {reading.pollutantPM10!r}"
        )

    def test_pollutant_o3_passes_through_in_ugm3(self) -> None:
        """pollutantO3 = 87.0 µg/m³ (fixture ozone=87.0; raw passthrough, no conversion)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.pollutantO3 is not None
        assert math.isclose(reading.pollutantO3, 87.0, rel_tol=1e-6), (
            f"pollutantO3: expected 87.0 µg/m³ (passthrough), got {reading.pollutantO3!r}"
        )

    def test_pollutant_no2_passes_through_in_ugm3(self) -> None:
        """pollutantNO2 = 0.2 µg/m³ (fixture nitrogen_dioxide=0.2; raw passthrough)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.pollutantNO2 is not None
        assert math.isclose(reading.pollutantNO2, 0.2, rel_tol=1e-6), (
            f"pollutantNO2: expected 0.2 µg/m³ (passthrough), got {reading.pollutantNO2!r}"
        )

    def test_pollutant_so2_passes_through_in_ugm3(self) -> None:
        """pollutantSO2 = 0.1 µg/m³ (fixture sulphur_dioxide=0.1; raw passthrough)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.pollutantSO2 is not None
        assert math.isclose(reading.pollutantSO2, 0.1, rel_tol=1e-6), (
            f"pollutantSO2: expected 0.1 µg/m³ (passthrough), got {reading.pollutantSO2!r}"
        )

    def test_pollutant_co_passes_through_in_ugm3(self) -> None:
        """pollutantCO = 155.0 µg/m³ (fixture carbon_monoxide=155.0; raw passthrough)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.pollutantCO is not None
        assert math.isclose(reading.pollutantCO, 155.0, rel_tol=1e-6), (
            f"pollutantCO: expected 155.0 µg/m³ (passthrough), got {reading.pollutantCO!r}"
        )

    def test_reading_is_aqi_reading_instance(self) -> None:
        """Return type is AQIReading (not dict, not None) for valid wire input."""
        from weewx_clearskies_api.models.responses import AQIReading  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.openmeteo import _wire_to_canonical  # noqa: PLC0415
        wire = self._load_wire()
        reading = _wire_to_canonical(wire)
        assert isinstance(reading, AQIReading), (
            f"Expected AQIReading instance, got {type(reading).__name__!r}"
        )


# ===========================================================================
# 3. _wire_to_canonical — edge cases
# ===========================================================================


class TestWireToCanonicalEdgeCases:
    """_wire_to_canonical handles null/partial wire values correctly."""

    def test_all_null_fixture_returns_none(self) -> None:
        """All-null current block (no us_aqi AND no per-pollutant) → returns None."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (  # noqa: PLC0415
            _OpenMeteoAQResponse,
            _wire_to_canonical,
        )
        raw = _load_fixture("openmeteo_current_all_null.json")
        wire = _OpenMeteoAQResponse.model_validate(raw)
        result = _wire_to_canonical(wire)
        assert result is None, (
            f"All-null wire response must return None (no useful reading), got {result!r}"
        )

    def test_us_aqi_only_fixture_populates_aqi_and_scale(self) -> None:
        """us_aqi=73 with all sub-AQIs null → aqi=73, aqiScale='epa', aqiCategory=None."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (  # noqa: PLC0415
            _OpenMeteoAQResponse,
            _wire_to_canonical,
        )
        raw = _load_fixture("openmeteo_current_us_aqi_only.json")
        wire = _OpenMeteoAQResponse.model_validate(raw)
        reading = _wire_to_canonical(wire)
        assert reading is not None, "us_aqi=73 with null sub-AQIs must return AQIReading"
        assert reading.aqi == 73
        assert reading.aqiScale == "epa"
        assert reading.aqiCategory is None

    def test_us_aqi_only_fixture_main_pollutant_is_none(self) -> None:
        """All six us_aqi_* sub-AQIs null → aqiMainPollutant=None (no argmax possible)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (  # noqa: PLC0415
            _OpenMeteoAQResponse,
            _wire_to_canonical,
        )
        raw = _load_fixture("openmeteo_current_us_aqi_only.json")
        wire = _OpenMeteoAQResponse.model_validate(raw)
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.aqiMainPollutant is None, (
            f"All sub-AQIs null → aqiMainPollutant=None, got {reading.aqiMainPollutant!r}"
        )

    def test_us_aqi_only_fixture_per_pollutant_concentrations_are_none(self) -> None:
        """All concentration fields null → pollutantPM25/PM10/O3/NO2/SO2/CO all None."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (  # noqa: PLC0415
            _OpenMeteoAQResponse,
            _wire_to_canonical,
        )
        raw = _load_fixture("openmeteo_current_us_aqi_only.json")
        wire = _OpenMeteoAQResponse.model_validate(raw)
        reading = _wire_to_canonical(wire)
        assert reading is not None
        for field in ("pollutantPM25", "pollutantPM10", "pollutantO3",
                      "pollutantNO2", "pollutantSO2", "pollutantCO"):
            val = getattr(reading, field)
            assert val is None, (
                f"{field} must be None when wire concentration is null, got {val!r}"
            )

    def test_partial_null_sub_aqis_excluded_from_argmax(self) -> None:
        """Some sub-AQIs null → argmax only over non-null values; correct pollutant identified."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (  # noqa: PLC0415
            _OpenMeteoAQResponse,
            _wire_to_canonical,
        )
        # pm10 sub-AQI is 80 (highest non-null); pm2_5 is null; others 0 or null.
        raw = _load_fixture("openmeteo_current.json")
        raw["current"]["us_aqi_pm2_5"] = None
        raw["current"]["us_aqi_pm10"] = 80
        raw["current"]["us_aqi_nitrogen_dioxide"] = 10
        raw["current"]["us_aqi_ozone"] = 15
        raw["current"]["us_aqi_sulphur_dioxide"] = None
        raw["current"]["us_aqi_carbon_monoxide"] = 5
        wire = _OpenMeteoAQResponse.model_validate(raw)
        reading = _wire_to_canonical(wire)
        assert reading is not None
        assert reading.aqiMainPollutant == "PM10", (
            f"With pm10 sub-AQI=80 highest and pm2_5 null, expected 'PM10', "
            f"got {reading.aqiMainPollutant!r}"
        )

    def test_aqi_none_yields_none_aqi_and_none_category(self) -> None:
        """us_aqi=None → aqi=None, aqiScale='epa', aqiCategory=None."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (  # noqa: PLC0415
            _OpenMeteoAQResponse,
            _wire_to_canonical,
        )
        # null us_aqi but some pollutant concentration present → not None (not all-null)
        raw = _load_fixture("openmeteo_current.json")
        raw["current"]["us_aqi"] = None
        wire = _OpenMeteoAQResponse.model_validate(raw)
        reading = _wire_to_canonical(wire)
        # Reading is not None (concentrations are present), but aqi is None
        assert reading is not None
        assert reading.aqi is None, f"Expected aqi=None, got {reading.aqi!r}"
        assert reading.aqiScale == "epa", f"Expected aqiScale='epa', got {reading.aqiScale!r}"
        assert reading.aqiCategory is None, (
            f"Expected aqiCategory=None (always None per refactor), got {reading.aqiCategory!r}"
        )


# ===========================================================================
# 4. _main_pollutant_from_sub_aqis — argmax, tie-breaking, all-None
# ===========================================================================


class TestMainPollutantFromSubAqis:
    """_main_pollutant_from_sub_aqis: argmax + tie-break + all-None per LC14."""

    def _make_current_block(self, **sub_aqi_overrides: float | None) -> Any:
        """Build a _OpenMeteoCurrentBlock with specified sub-AQI values."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _OpenMeteoCurrentBlock,  # noqa: PLC0415
        )
        data: dict[str, Any] = {
            "time": "2026-05-10T22:00",
            "us_aqi": 50,
            "us_aqi_pm2_5": None,
            "us_aqi_pm10": None,
            "us_aqi_nitrogen_dioxide": None,
            "us_aqi_ozone": None,
            "us_aqi_sulphur_dioxide": None,
            "us_aqi_carbon_monoxide": None,
        }
        data.update(sub_aqi_overrides)
        return _OpenMeteoCurrentBlock.model_validate(data)

    def test_argmax_returns_pollutant_with_highest_sub_aqi(self) -> None:
        """Highest sub-AQI wins: us_aqi_ozone=100 > pm2_5=50 → 'O3'."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _main_pollutant_from_sub_aqis,  # noqa: PLC0415
        )
        current = self._make_current_block(
            us_aqi_pm2_5=50, us_aqi_pm10=30, us_aqi_ozone=100,
        )
        result = _main_pollutant_from_sub_aqis(current)
        assert result == "O3", f"Expected 'O3' (highest), got {result!r}"

    def test_tie_break_pm25_beats_pm10(self) -> None:
        """Tie: pm2_5=80 == pm10=80 → 'PM2.5' wins (PM2.5 before PM10 in table, LC14)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _main_pollutant_from_sub_aqis,  # noqa: PLC0415
        )
        current = self._make_current_block(
            us_aqi_pm2_5=80, us_aqi_pm10=80,
        )
        result = _main_pollutant_from_sub_aqis(current)
        assert result == "PM2.5", (
            f"Tie between PM2.5 and PM10: expected 'PM2.5' (table order), got {result!r}"
        )

    def test_tie_break_pm10_beats_no2(self) -> None:
        """Tie: pm10=60 == no2=60 → 'PM10' wins (PM10 before NO2 in table, LC14)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _main_pollutant_from_sub_aqis,  # noqa: PLC0415
        )
        current = self._make_current_block(
            us_aqi_pm10=60, us_aqi_nitrogen_dioxide=60,
        )
        result = _main_pollutant_from_sub_aqis(current)
        assert result == "PM10", (
            f"Tie between PM10 and NO2: expected 'PM10' (table order), got {result!r}"
        )

    def test_all_none_returns_none(self) -> None:
        """All six sub-AQI fields None → None (cannot derive main pollutant)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _main_pollutant_from_sub_aqis,  # noqa: PLC0415
        )
        current = self._make_current_block()  # all defaults are None
        result = _main_pollutant_from_sub_aqis(current)
        assert result is None, (
            f"All-None sub-AQIs must return None, got {result!r}"
        )

    def test_single_non_none_value_wins_regardless(self) -> None:
        """Single non-None value (even if small) → that pollutant wins."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _main_pollutant_from_sub_aqis,  # noqa: PLC0415
        )
        current = self._make_current_block(us_aqi_sulphur_dioxide=1)
        result = _main_pollutant_from_sub_aqis(current)
        assert result == "SO2", (
            f"Single non-None SO2=1 → expected 'SO2', got {result!r}"
        )

    def test_pm25_sub_aqi_maps_to_pm25_canonical(self) -> None:
        """us_aqi_pm2_5 highest → canonical pollutant id 'PM2.5' (LC14 mapping table)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _main_pollutant_from_sub_aqis,  # noqa: PLC0415
        )
        current = self._make_current_block(us_aqi_pm2_5=73)
        assert _main_pollutant_from_sub_aqis(current) == "PM2.5"

    def test_pm10_sub_aqi_maps_to_pm10_canonical(self) -> None:
        """us_aqi_pm10 highest → canonical pollutant id 'PM10' (LC14 mapping table)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _main_pollutant_from_sub_aqis,  # noqa: PLC0415
        )
        current = self._make_current_block(us_aqi_pm10=80)
        assert _main_pollutant_from_sub_aqis(current) == "PM10"

    def test_nitrogen_dioxide_sub_aqi_maps_to_no2_canonical(self) -> None:
        """us_aqi_nitrogen_dioxide highest → canonical 'NO2' (LC14 mapping table)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _main_pollutant_from_sub_aqis,  # noqa: PLC0415
        )
        current = self._make_current_block(us_aqi_nitrogen_dioxide=90)
        assert _main_pollutant_from_sub_aqis(current) == "NO2"

    def test_ozone_sub_aqi_maps_to_o3_canonical(self) -> None:
        """us_aqi_ozone highest → canonical 'O3' (LC14 mapping table)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _main_pollutant_from_sub_aqis,  # noqa: PLC0415
        )
        current = self._make_current_block(us_aqi_ozone=70)
        assert _main_pollutant_from_sub_aqis(current) == "O3"

    def test_sulphur_dioxide_sub_aqi_maps_to_so2_canonical(self) -> None:
        """us_aqi_sulphur_dioxide highest → canonical 'SO2' (LC14 mapping table)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _main_pollutant_from_sub_aqis,  # noqa: PLC0415
        )
        current = self._make_current_block(us_aqi_sulphur_dioxide=55)
        assert _main_pollutant_from_sub_aqis(current) == "SO2"

    def test_carbon_monoxide_sub_aqi_maps_to_co_canonical(self) -> None:
        """us_aqi_carbon_monoxide highest → canonical 'CO' (LC14 mapping table)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _main_pollutant_from_sub_aqis,  # noqa: PLC0415
        )
        current = self._make_current_block(us_aqi_carbon_monoxide=30)
        assert _main_pollutant_from_sub_aqis(current) == "CO"


# ===========================================================================
# 5. _build_cache_key — determinism + lat/lon rounding
# ===========================================================================


class TestBuildCacheKey:
    """_build_cache_key is deterministic and uses 4-decimal-place rounding."""

    def test_same_coordinates_produce_same_key(self) -> None:
        """Same lat/lon always produces the same cache key (deterministic)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _build_cache_key  # noqa: PLC0415
        key1 = _build_cache_key(_LAT, _LON)
        key2 = _build_cache_key(_LAT, _LON)
        assert key1 == key2

    def test_different_coordinates_produce_different_keys(self) -> None:
        """Different lat/lon produces different cache keys."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _build_cache_key  # noqa: PLC0415
        key1 = _build_cache_key(_LAT, _LON)
        key2 = _build_cache_key(41.6022, -98.9178)
        assert key1 != key2

    def test_key_is_64_char_hex_string(self) -> None:
        """Cache key is SHA-256 hex digest (64 characters, all hex digits)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _build_cache_key  # noqa: PLC0415
        key = _build_cache_key(_LAT, _LON)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_lat_lon_rounding_to_4_decimal_places(self) -> None:
        """Lat/lon differing only in 5th+ decimal → same cache key (rounded to 4dp)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import _build_cache_key  # noqa: PLC0415
        # 47.60621 and 47.60624 both round to 47.6062 at 4dp
        key1 = _build_cache_key(47.60621, -122.3321)
        key2 = _build_cache_key(47.60624, -122.3321)
        assert key1 == key2, (
            "Coordinates differing only in 5th decimal should produce same key"
        )

    def test_aqi_cache_key_distinct_from_forecast_cache_key(self) -> None:
        """AQI cache key differs from forecast openmeteo cache key at same coordinates.

        Logical endpoint key 'aqi_current' (LC7) distinct from 'forecast_bundle'
        ensures separate cache entries for AQI and forecast even at same station.
        """
        from weewx_clearskies_api.providers.aqi.openmeteo import (
            _build_cache_key as _aqi_key,  # noqa: PLC0415
        )
        from weewx_clearskies_api.providers.forecast.openmeteo import (
            _build_cache_key as _forecast_key,  # noqa: PLC0415
        )
        aqi_cache_key = _aqi_key(_LAT, _LON)
        forecast_cache_key = _forecast_key(_LAT, _LON, "US")
        assert aqi_cache_key != forecast_cache_key, (
            "AQI cache key must differ from forecast cache key (separate logical endpoints)"
        )


# ===========================================================================
# 6. fetch() — cache hit and miss paths
# ===========================================================================


class TestFetchCachePaths:
    """fetch() — cache hit/miss/sentinel paths with respx-mocked HTTP."""

    def _make_valid_wire_response(self) -> dict[str, Any]:
        """Return a valid Open-Meteo AQI wire response based on the real fixture."""
        return _load_fixture("openmeteo_current.json")

    def test_cache_miss_makes_outbound_call_and_returns_reading(self) -> None:
        """Cache miss: HTTP call made; AQIReading returned."""
        _reset_provider_state()
        from weewx_clearskies_api.models.responses import AQIReading  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi import openmeteo  # noqa: PLC0415
        data = self._make_valid_wire_response()

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            reading = openmeteo.fetch(lat=_LAT, lon=_LON)
            assert mock.calls.call_count == 1

        assert isinstance(reading, AQIReading), (
            f"Expected AQIReading, got {type(reading).__name__!r}"
        )
        assert reading.aqi == 73
        assert reading.source == "openmeteo"

    def test_cache_hit_returns_reading_without_outbound_call(self) -> None:
        """Cache hit: no HTTP call made; cached reading returned."""
        _reset_provider_state()
        from weewx_clearskies_api.providers.aqi import openmeteo  # noqa: PLC0415
        data = self._make_valid_wire_response()

        # Prime cache with first call
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            openmeteo.fetch(lat=_LAT, lon=_LON)
            assert mock.calls.call_count == 1

        # Second call should hit cache — zero HTTP calls
        with respx.mock(assert_all_called=False) as mock2:
            reading2 = openmeteo.fetch(lat=_LAT, lon=_LON)
            assert mock2.calls.call_count == 0, (
                f"Expected 0 HTTP calls on cache hit, got {mock2.calls.call_count}"
            )

        assert reading2 is not None
        assert reading2.aqi == 73

    def test_cache_hit_no_reading_sentinel_returns_none(self) -> None:
        """Cache hit with _no_reading sentinel → fetch() returns None; no HTTP call."""
        _reset_provider_state()
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi import openmeteo  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.openmeteo import (  # noqa: PLC0415
            DEFAULT_AQI_TTL_SECONDS,
            _build_cache_key,
        )

        # Manually inject the sentinel into the cache
        cache_key = _build_cache_key(_LAT, _LON)
        get_cache().set(cache_key, {"_no_reading": True}, ttl_seconds=DEFAULT_AQI_TTL_SECONDS)

        with respx.mock(assert_all_called=False) as mock:
            result = openmeteo.fetch(lat=_LAT, lon=_LON)
            assert mock.calls.call_count == 0, (
                "Sentinel cache hit must not make HTTP call"
            )

        assert result is None, (
            f"Sentinel cache hit must return None, got {result!r}"
        )

    def test_cache_miss_all_null_response_returns_none_and_caches_sentinel(self) -> None:
        """Cache miss + all-null wire response → None returned + sentinel cached."""
        _reset_provider_state()
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi import openmeteo  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi.openmeteo import _build_cache_key  # noqa: PLC0415
        data = _load_fixture("openmeteo_current_all_null.json")

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            result = openmeteo.fetch(lat=_LAT, lon=_LON)
            assert mock.calls.call_count == 1

        assert result is None, (
            f"All-null wire response must return None, got {result!r}"
        )
        # Sentinel should be cached
        cached = get_cache().get(_build_cache_key(_LAT, _LON))
        assert cached == {"_no_reading": True}, (
            f"Expected sentinel {{_no_reading: True}} in cache, got {cached!r}"
        )

    def test_cached_reading_round_trips_through_model_dump_validate(self) -> None:
        """Records cached as dict and reconstructed via AQIReading.model_validate on hit."""
        _reset_provider_state()
        from weewx_clearskies_api.providers.aqi import openmeteo  # noqa: PLC0415
        data = self._make_valid_wire_response()

        # First fetch — populates cache
        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(200, json=data)
            )
            reading1 = openmeteo.fetch(lat=_LAT, lon=_LON)

        # Second fetch — from cache
        with respx.mock(assert_all_called=False):
            reading2 = openmeteo.fetch(lat=_LAT, lon=_LON)

        assert reading1 is not None and reading2 is not None
        assert reading1.aqi == reading2.aqi
        assert reading1.aqiScale == reading2.aqiScale
        assert reading1.source == reading2.source
        assert reading1.observedAt == reading2.observedAt
        assert reading1.aqiCategory == reading2.aqiCategory
        assert reading1.aqiMainPollutant == reading2.aqiMainPollutant


# ===========================================================================
# 7. fetch() — error paths (L2 canonical taxonomy propagation)
# ===========================================================================


class TestFetchErrorPaths:
    """fetch() propagates canonical exceptions bare (L2 carry-forward, 3b-4 F1)."""

    def test_wire_validation_failure_raises_provider_protocol_error(self) -> None:
        """Cache miss + missing 'current' block → ProviderProtocolError (ValidationError → wrap)."""
        _reset_provider_state()
        from weewx_clearskies_api.providers._common.errors import (
            ProviderProtocolError,  # noqa: PLC0415
        )
        from weewx_clearskies_api.providers.aqi import openmeteo  # noqa: PLC0415
        broken = _load_fixture("openmeteo_current.json")
        del broken["current"]  # Remove required field → ValidationError

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(200, json=broken)
            )
            with pytest.raises(ProviderProtocolError):
                openmeteo.fetch(lat=_LAT, lon=_LON)

    def test_provider_502_raises_transient_network_error(self) -> None:
        """Provider 5xx → TransientNetworkError (L2 bare propagation)."""
        _reset_provider_state()
        from weewx_clearskies_api.providers._common.errors import (
            TransientNetworkError,  # noqa: PLC0415
        )
        from weewx_clearskies_api.providers.aqi import openmeteo  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(502, json={"reason": "Bad Gateway"})
            )
            with pytest.raises(TransientNetworkError):
                openmeteo.fetch(lat=_LAT, lon=_LON)

    def test_provider_500_raises_transient_network_error(self) -> None:
        """Provider 500 → TransientNetworkError (L2 bare propagation)."""
        _reset_provider_state()
        from weewx_clearskies_api.providers._common.errors import (
            TransientNetworkError,  # noqa: PLC0415
        )
        from weewx_clearskies_api.providers.aqi import openmeteo  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(500, json={"reason": "Internal Server Error"})
            )
            with pytest.raises(TransientNetworkError):
                openmeteo.fetch(lat=_LAT, lon=_LON)

    def test_provider_429_raises_quota_exhausted(self) -> None:
        """Provider 429 → QuotaExhausted (L2 bare propagation)."""
        _reset_provider_state()
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi import openmeteo  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(429, json={"reason": "Too Many Requests"})
            )
            with pytest.raises(QuotaExhausted):
                openmeteo.fetch(lat=_LAT, lon=_LON)

    def test_provider_429_retry_after_seconds_propagated(self) -> None:
        """429 with Retry-After header → QuotaExhausted.retry_after_seconds not None (3b-4 F1)."""
        _reset_provider_state()
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        from weewx_clearskies_api.providers.aqi import openmeteo  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={"reason": "Too Many Requests"},
                    headers={"Retry-After": "120"},
                )
            )
            with pytest.raises(QuotaExhausted) as exc_info:
                openmeteo.fetch(lat=_LAT, lon=_LON)
        assert exc_info.value.retry_after_seconds is not None, (
            "QuotaExhausted.retry_after_seconds must be set when Retry-After header present "
            "(3b-4 F1 carry-forward: re-construction of exception drops retry_after)"
        )

    def test_provider_422_raises_canonical_exception(self) -> None:
        """Provider 422 → canonical exception (ProviderProtocolError or TransientNetworkError)."""
        _reset_provider_state()
        from weewx_clearskies_api.providers._common.errors import (  # noqa: PLC0415
            ProviderError,
        )
        from weewx_clearskies_api.providers.aqi import openmeteo  # noqa: PLC0415

        with respx.mock(assert_all_called=False) as mock:
            mock.get(_OPENMETEO_AQ_URL).mock(
                return_value=httpx.Response(422, json={"reason": "Unprocessable"})
            )
            with pytest.raises(ProviderError):
                openmeteo.fetch(lat=_LAT, lon=_LON)


# ===========================================================================
# 8. Capability declaration
# ===========================================================================


class TestCapabilityDeclaration:
    """CAPABILITY symbol declares correct provider metadata (ADR-038 §4)."""

    def test_capability_provider_id_is_openmeteo(self) -> None:
        """CAPABILITY.provider_id = 'openmeteo'."""
        from weewx_clearskies_api.providers.aqi.openmeteo import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.provider_id == "openmeteo"

    def test_capability_domain_is_aqi(self) -> None:
        """CAPABILITY.domain = 'aqi' (LC9)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.domain == "aqi"

    def test_capability_auth_required_is_empty(self) -> None:
        """CAPABILITY.auth_required = () — keyless provider (LC11)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.auth_required == (), (
            f"Expected empty auth_required for keyless Open-Meteo, got {CAPABILITY.auth_required!r}"
        )

    def test_capability_geographic_coverage_is_global(self) -> None:
        """CAPABILITY.geographic_coverage = 'global' (LC10: CAMS data worldwide)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.geographic_coverage == "global"

    def test_capability_default_poll_interval_is_900_seconds(self) -> None:
        """CAPABILITY.default_poll_interval_seconds = 900 (15 min, LC3 / ADR-017)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import CAPABILITY  # noqa: PLC0415
        assert CAPABILITY.default_poll_interval_seconds == 900

    def test_capability_supplied_fields_includes_eleven_canonical_fields(self) -> None:
        """CAPABILITY.supplied_canonical_fields includes all 11 Open-Meteo-supplied fields (LC12)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import CAPABILITY  # noqa: PLC0415
        expected_fields = (
            "aqi", "aqiCategory", "aqiMainPollutant",
            "pollutantPM25", "pollutantPM10",
            "pollutantO3", "pollutantNO2", "pollutantSO2", "pollutantCO",
            "observedAt", "source",
        )
        for field in expected_fields:
            assert field in CAPABILITY.supplied_canonical_fields, (
                f"CAPABILITY.supplied_canonical_fields missing {field!r}"
            )

    def test_capability_excludes_aqi_location_partial_domain(self) -> None:
        """CAPABILITY.supplied_canonical_fields excludes aqiLocation (PARTIAL-DOMAIN, LC12)."""
        from weewx_clearskies_api.providers.aqi.openmeteo import CAPABILITY  # noqa: PLC0415
        assert "aqiLocation" not in CAPABILITY.supplied_canonical_fields, (
            "PARTIAL-DOMAIN: aqiLocation must not be in CAPABILITY "
            "(Open-Meteo has no location label field)"
        )

    def test_wire_providers_registers_openmeteo_aqi_capability(self) -> None:
        """wire_providers([CAPABILITY]) → registry has openmeteo aqi entry."""
        _reset_provider_state()
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            get_provider_registry,
            wire_providers,
        )
        from weewx_clearskies_api.providers.aqi.openmeteo import CAPABILITY  # noqa: PLC0415
        wire_providers([CAPABILITY])
        registry = get_provider_registry()
        aqi_entries = [
            p for p in registry
            if p.provider_id == "openmeteo" and p.domain == "aqi"
        ]
        assert len(aqi_entries) == 1, (
            f"Expected 1 openmeteo aqi entry in registry, found {len(aqi_entries)}"
        )
        # Teardown: reset provider registry so downstream tests (e.g.
        # test_3a2_endpoints_integration.py TestCapabilitiesIntegration which
        # asserts providers==[] ) don't see a polluted global registry.
        # This test runs before the 3a2 integration module's capabilities tests
        # in the full pytest run; without cleanup the registry leak causes F.
        _reset_provider_state()
