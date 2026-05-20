"""Pure-compute unit tests for providers/aqi/_units.py (3b-9, extended 3b-10).

Covers per the task-3b-9 brief §Test-author parallel scope (test_units.py):

  ugm3_to_ppm round-trips:
  - Known values for each supported gas (O3, NO2, SO2, CO).
  - None propagates → returns None.
  - Unknown pollutant raises KeyError.

  epa_category boundary tests:
  - Every breakpoint boundary value (0, 50, 51, 100, 101, 150, 151, 200, 201, 300, 301, 500).
  - Values above 500 → "Hazardous" (defensive cap).
  - None → None.
  - Floating-point AQI (e.g. 50.0, 50.5, 100.9) handled correctly.

Formula reference (canonical-data-model §4.2 footnote):
  ppm = µg/m³ × 24.45 / molecular_weight
  where 24.45 L/mol is the molar volume at 25°C / 1 atm.

Molecular weights:
  O3:  48.00 g/mol
  NO2: 46.01 g/mol
  SO2: 64.07 g/mol
  CO:  28.01 g/mol

No DB, no HTTP, no external state.
ADR references: ADR-013, ADR-038.
"""

from __future__ import annotations

import math

import pytest

# ===========================================================================
# 1. ugm3_to_ppm — conversion round-trips and edge cases
# ===========================================================================


class TestUgm3ToPpm:
    """ugm3_to_ppm converts µg/m³ to ppm for the four supported gases.

    Formula (corrected 2026-05-11): ppm = µg/m³ × 24.45 / (MW × 1000),
    equivalent to ppm = µg/m³ × 0.02445 / MW. The /1000 factor was missing
    in the pre-fix version, which made the function return ppb-as-ppm.
    Particulates (PM2.5, PM10) are NOT in the table and raise KeyError.
    None input → None output (pass-through for missing wire values).
    """

    def test_ozone_100_ugm3_converts_to_expected_ppm(self) -> None:
        """O3 100 µg/m³ → 100 × 24.45 / (48.00 × 1000) ≈ 0.0509375 ppm."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        result = ugm3_to_ppm(100.0, pollutant="O3")
        assert result is not None
        expected = 100.0 * 24.45 / (48.00 * 1000)
        assert math.isclose(result, expected, rel_tol=1e-9), (
            f"O3 100 µg/m³ → expected {expected:.6f} ppm, got {result:.6f}"
        )

    def test_ozone_zero_ugm3_returns_zero(self) -> None:
        """O3 0.0 µg/m³ → 0.0 ppm (zero input stays zero)."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        result = ugm3_to_ppm(0.0, pollutant="O3")
        assert result == 0.0

    def test_no2_100_ugm3_converts_to_expected_ppm(self) -> None:
        """NO2 100 µg/m³ → 100 × 24.45 / (46.01 × 1000) ≈ 0.053140 ppm."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        result = ugm3_to_ppm(100.0, pollutant="NO2")
        assert result is not None
        expected = 100.0 * 24.45 / (46.01 * 1000)
        assert math.isclose(result, expected, rel_tol=1e-9), (
            f"NO2 100 µg/m³ → expected {expected:.6f} ppm, got {result:.6f}"
        )

    def test_so2_100_ugm3_converts_to_expected_ppm(self) -> None:
        """SO2 100 µg/m³ → 100 × 24.45 / (64.07 × 1000) ≈ 0.038162 ppm."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        result = ugm3_to_ppm(100.0, pollutant="SO2")
        assert result is not None
        expected = 100.0 * 24.45 / (64.07 * 1000)
        assert math.isclose(result, expected, rel_tol=1e-9), (
            f"SO2 100 µg/m³ → expected {expected:.6f} ppm, got {result:.6f}"
        )

    def test_co_100_ugm3_converts_to_expected_ppm(self) -> None:
        """CO 100 µg/m³ → 100 × 24.45 / (28.01 × 1000) ≈ 0.087254 ppm."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        result = ugm3_to_ppm(100.0, pollutant="CO")
        assert result is not None
        expected = 100.0 * 24.45 / (28.01 * 1000)
        assert math.isclose(result, expected, rel_tol=1e-9), (
            f"CO 100 µg/m³ → expected {expected:.6f} ppm, got {result:.6f}"
        )

    def test_co_fixture_value_155_ugm3_converts_correctly(self) -> None:
        """CO 155.0 µg/m³ (fixture value) → 155 × 24.45 / (28.01 × 1000) ≈ 0.135306 ppm."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        result = ugm3_to_ppm(155.0, pollutant="CO")
        assert result is not None
        expected = 155.0 * 24.45 / (28.01 * 1000)
        assert math.isclose(result, expected, rel_tol=1e-9), (
            f"CO 155.0 µg/m³ → expected {expected:.6f} ppm, got {result:.6f}"
        )

    def test_ozone_fixture_value_87_ugm3_converts_correctly(self) -> None:
        """O3 87.0 µg/m³ (fixture value) → 87 × 24.45 / (48.00 × 1000) ≈ 0.044316 ppm."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        result = ugm3_to_ppm(87.0, pollutant="O3")
        assert result is not None
        expected = 87.0 * 24.45 / (48.00 * 1000)
        assert math.isclose(result, expected, rel_tol=1e-9), (
            f"O3 87.0 µg/m³ → expected {expected:.6f} ppm, got {result:.6f}"
        )

    def test_none_input_returns_none_for_o3(self) -> None:
        """ugm3_to_ppm(None, pollutant='O3') → None (None propagates, ADR-010 null passthrough)."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        result = ugm3_to_ppm(None, pollutant="O3")
        assert result is None, (
            f"Expected None propagation for None input, got {result!r}"
        )

    def test_none_input_returns_none_for_no2(self) -> None:
        """ugm3_to_ppm(None, pollutant='NO2') → None."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        assert ugm3_to_ppm(None, pollutant="NO2") is None

    def test_none_input_returns_none_for_so2(self) -> None:
        """ugm3_to_ppm(None, pollutant='SO2') → None."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        assert ugm3_to_ppm(None, pollutant="SO2") is None

    def test_none_input_returns_none_for_co(self) -> None:
        """ugm3_to_ppm(None, pollutant='CO') → None."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        assert ugm3_to_ppm(None, pollutant="CO") is None

    def test_unknown_pollutant_raises_key_error(self) -> None:
        """ugm3_to_ppm(100, pollutant='UNKNOWN') → KeyError (not in MW table)."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        with pytest.raises(KeyError):
            ugm3_to_ppm(100.0, pollutant="UNKNOWN")

    def test_pm25_not_in_conversion_table_raises_key_error(self) -> None:
        """PM2.5 raises KeyError — particulates stay in µg/m³, no conversion (canonical §3.8)."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        with pytest.raises(KeyError):
            ugm3_to_ppm(3.1, pollutant="PM2.5")

    def test_pm10_not_in_conversion_table_raises_key_error(self) -> None:
        """PM10 raises KeyError — particulates stay in µg/m³, no conversion (canonical §3.8)."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        with pytest.raises(KeyError):
            ugm3_to_ppm(4.5, pollutant="PM10")

    def test_result_is_float_not_none_for_valid_input(self) -> None:
        """Result is a float (not None) for a valid non-None input."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        result = ugm3_to_ppm(50.0, pollutant="O3")
        assert isinstance(result, float), (
            f"Expected float result, got {type(result).__name__!r}"
        )

    def test_molar_volume_and_mw_are_reflected_in_result(self) -> None:
        """Doubling the input doubles the output (linear formula sanity check)."""
        from weewx_clearskies_api.providers.aqi._units import ugm3_to_ppm  # noqa: PLC0415
        result1 = ugm3_to_ppm(50.0, pollutant="CO")
        result2 = ugm3_to_ppm(100.0, pollutant="CO")
        assert result1 is not None and result2 is not None
        assert math.isclose(result2, result1 * 2, rel_tol=1e-9), (
            "Linear formula: doubling µg/m³ must double ppm"
        )


# ===========================================================================
# 2. epa_category — boundary tests
# ===========================================================================


class TestEpaCategory:
    """epa_category maps EPA AQI values to canonical category names.

    Boundary tests for every breakpoint per brief §test_units.py spec.
    Canonical spelling per canonical-data-model §3.8 (LC13):
      0–50:    "Good"
      51–100:  "Moderate"
      101–150: "Unhealthy for Sensitive Groups"
      151–200: "Unhealthy"
      201–300: "Very Unhealthy"
      301–500: "Hazardous"
    Values >500 → "Hazardous" (defensive cap; provider-side bugs shouldn't crash).
    None → None.
    """

    def test_aqi_zero_is_good(self) -> None:
        """AQI 0 → 'Good' (lower edge of lowest band)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(0) == "Good", "AQI 0 must be 'Good'"

    def test_aqi_50_is_good(self) -> None:
        """AQI 50 → 'Good' (upper bound of Good band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(50) == "Good", "AQI 50 must be 'Good'"

    def test_aqi_51_is_moderate(self) -> None:
        """AQI 51 → 'Moderate' (first value in Moderate band)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(51) == "Moderate", "AQI 51 must be 'Moderate'"

    def test_aqi_100_is_moderate(self) -> None:
        """AQI 100 → 'Moderate' (upper bound of Moderate band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(100) == "Moderate", "AQI 100 must be 'Moderate'"

    def test_aqi_101_is_unhealthy_for_sensitive_groups(self) -> None:
        """AQI 101 → 'Unhealthy for Sensitive Groups' (first value in USG band)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(101) == "Unhealthy for Sensitive Groups", (
            "AQI 101 must be 'Unhealthy for Sensitive Groups'"
        )

    def test_aqi_150_is_unhealthy_for_sensitive_groups(self) -> None:
        """AQI 150 → 'Unhealthy for Sensitive Groups' (upper bound inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(150) == "Unhealthy for Sensitive Groups", (
            "AQI 150 must be 'Unhealthy for Sensitive Groups'"
        )

    def test_aqi_151_is_unhealthy(self) -> None:
        """AQI 151 → 'Unhealthy' (first value in Unhealthy band)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(151) == "Unhealthy", "AQI 151 must be 'Unhealthy'"

    def test_aqi_200_is_unhealthy(self) -> None:
        """AQI 200 → 'Unhealthy' (upper bound of Unhealthy band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(200) == "Unhealthy", "AQI 200 must be 'Unhealthy'"

    def test_aqi_201_is_very_unhealthy(self) -> None:
        """AQI 201 → 'Very Unhealthy' (first value in Very Unhealthy band)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(201) == "Very Unhealthy", "AQI 201 must be 'Very Unhealthy'"

    def test_aqi_300_is_very_unhealthy(self) -> None:
        """AQI 300 → 'Very Unhealthy' (upper bound inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(300) == "Very Unhealthy", "AQI 300 must be 'Very Unhealthy'"

    def test_aqi_301_is_hazardous(self) -> None:
        """AQI 301 → 'Hazardous' (first value in Hazardous band)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(301) == "Hazardous", "AQI 301 must be 'Hazardous'"

    def test_aqi_500_is_hazardous(self) -> None:
        """AQI 500 → 'Hazardous' (top of the defined EPA range, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(500) == "Hazardous", "AQI 500 must be 'Hazardous'"

    def test_aqi_501_is_hazardous_defensive_cap(self) -> None:
        """AQI 501 → 'Hazardous' (above spec range; defensive cap, not an error).

        Provider-side bugs can emit values >500. Per brief §module-2 spec:
        cap at 'Hazardous' rather than raising — dashboards shouldn't crash
        on sensor/provider bugs.
        """
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(501) == "Hazardous", (
            "AQI > 500 must cap at 'Hazardous' (defensive cap)"
        )

    def test_aqi_600_is_hazardous_defensive_cap(self) -> None:
        """AQI 600 → 'Hazardous' (well above spec range; defensive cap still applies)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(600) == "Hazardous", (
            "AQI 600 must cap at 'Hazardous' (defensive cap)"
        )

    def test_none_aqi_returns_none(self) -> None:
        """epa_category(None) → None (None propagates; no provider reading → no category)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        result = epa_category(None)
        assert result is None, f"Expected None for None AQI input, got {result!r}"

    def test_float_aqi_50_point_0_is_good(self) -> None:
        """AQI 50.0 (float) → 'Good' (float comparison works with <=)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(50.0) == "Good"

    def test_float_aqi_50_point_5_is_moderate(self) -> None:
        """AQI 50.5 (float) → 'Moderate' (above Good upper bound 50)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(50.5) == "Moderate"

    def test_float_aqi_100_point_9_is_moderate(self) -> None:
        """AQI 100.9 (float) → 'Moderate' (below Moderate upper bound 100? No — 100.9 > 100).

        Expect 'Unhealthy for Sensitive Groups': 100.9 > 100 → falls into USG band.
        """
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        # 100.9 > 100.0 → not in Moderate (upper=100); falls into USG band (101–150)
        assert epa_category(100.9) == "Unhealthy for Sensitive Groups", (
            "AQI 100.9 > 100 → 'Unhealthy for Sensitive Groups' (float boundary check)"
        )

    def test_fixture_aqi_73_is_moderate(self) -> None:
        """AQI 73 (from real fixture) → 'Moderate' (51–100 band)."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        assert epa_category(73) == "Moderate", (
            "Fixture AQI 73 must be 'Moderate' (51–100 band)"
        )

    def test_result_is_str_for_valid_non_none_input(self) -> None:
        """Return type is str (not None) for any valid numeric AQI."""
        from weewx_clearskies_api.providers.aqi._units import epa_category  # noqa: PLC0415
        result = epa_category(125)
        assert isinstance(result, str), (
            f"Expected str result, got {type(result).__name__!r}"
        )


# ===========================================================================
# 3. ppb_to_ppm — conversion round-trips and edge cases (3b-10 extension)
# ===========================================================================


class TestPpbToPpm:
    """ppb_to_ppm converts ppb (parts per billion) to ppm (parts per million).

    Formula: ppm = ppb / 1000.0 (no molar-weight involved — same divisor for
    all gases; Aeris provides valuePPB directly).

    Brief reference: LC16 in phase-2-task-3b-10 brief.
    No pollutant arg needed — the conversion is gas-agnostic.
    None input → None output (pass-through for missing wire values).
    """

    def test_o3_32_point_1_ppb_converts_to_0_point_0321_ppm(self) -> None:
        """O3 32.1 ppb → 0.0321 ppm (exact division by 1000)."""
        from weewx_clearskies_api.providers.aqi._units import ppb_to_ppm  # noqa: PLC0415
        result = ppb_to_ppm(32.1)
        assert result is not None
        assert abs(result - 0.0321) < 1e-9, (
            f"32.1 ppb → expected 0.0321 ppm, got {result!r}"
        )

    def test_co_143_ppb_converts_to_0_point_143_ppm(self) -> None:
        """CO 143 ppb (fixture value) → 0.143 ppm."""
        from weewx_clearskies_api.providers.aqi._units import ppb_to_ppm  # noqa: PLC0415
        result = ppb_to_ppm(143.0)
        assert result is not None
        assert abs(result - 0.143) < 1e-9, (
            f"143.0 ppb → expected 0.143 ppm, got {result!r}"
        )

    def test_no2_3_ppb_converts_to_0_point_003_ppm(self) -> None:
        """NO2 3.0 ppb (fixture value) → 0.003 ppm."""
        from weewx_clearskies_api.providers.aqi._units import ppb_to_ppm  # noqa: PLC0415
        result = ppb_to_ppm(3.0)
        assert result is not None
        assert abs(result - 0.003) < 1e-9, (
            f"3.0 ppb → expected 0.003 ppm, got {result!r}"
        )

    def test_so2_zero_ppb_converts_to_zero_ppm(self) -> None:
        """SO2 0 ppb (fixture value) → 0.0 ppm (zero input stays zero)."""
        from weewx_clearskies_api.providers.aqi._units import ppb_to_ppm  # noqa: PLC0415
        result = ppb_to_ppm(0.0)
        assert result == 0.0, f"0.0 ppb → expected 0.0 ppm, got {result!r}"

    def test_1000_ppb_converts_to_1_ppm(self) -> None:
        """1000 ppb → 1.0 ppm (round-number boundary check)."""
        from weewx_clearskies_api.providers.aqi._units import ppb_to_ppm  # noqa: PLC0415
        result = ppb_to_ppm(1000.0)
        assert result is not None
        assert abs(result - 1.0) < 1e-9, (
            f"1000.0 ppb → expected 1.0 ppm, got {result!r}"
        )

    def test_none_input_returns_none(self) -> None:
        """ppb_to_ppm(None) → None (None propagates; ADR-010 null passthrough)."""
        from weewx_clearskies_api.providers.aqi._units import ppb_to_ppm  # noqa: PLC0415
        result = ppb_to_ppm(None)
        assert result is None, f"Expected None for None input, got {result!r}"

    def test_result_is_float_for_valid_input(self) -> None:
        """Return type is float (not None) for a valid non-None input."""
        from weewx_clearskies_api.providers.aqi._units import ppb_to_ppm  # noqa: PLC0415
        result = ppb_to_ppm(50.0)
        assert isinstance(result, float), (
            f"Expected float result, got {type(result).__name__!r}"
        )

    def test_doubling_ppb_doubles_ppm(self) -> None:
        """Linear formula: doubling ppb must double ppm."""
        from weewx_clearskies_api.providers.aqi._units import ppb_to_ppm  # noqa: PLC0415
        result1 = ppb_to_ppm(100.0)
        result2 = ppb_to_ppm(200.0)
        assert result1 is not None and result2 is not None
        assert abs(result2 - result1 * 2) < 1e-9, (
            "Doubling ppb must double ppm (linear formula sanity check)"
        )

    def test_fixture_o3_36_ppb_converts_correctly(self) -> None:
        """O3 36 ppb (real fixture value) → 0.036 ppm."""
        from weewx_clearskies_api.providers.aqi._units import ppb_to_ppm  # noqa: PLC0415
        result = ppb_to_ppm(36.0)
        assert result is not None
        assert abs(result - 0.036) < 1e-9, (
            f"36.0 ppb → expected 0.036 ppm, got {result!r}"
        )

    def test_ppb_to_ppm_does_not_require_pollutant_arg(self) -> None:
        """ppb_to_ppm takes only ppb — no pollutant kwarg needed (gas-agnostic)."""
        from weewx_clearskies_api.providers.aqi._units import ppb_to_ppm  # noqa: PLC0415
        # Call with positional arg only — must not raise TypeError
        result = ppb_to_ppm(25.0)
        assert result is not None


# ===========================================================================
# 3b. ppb_to_ugm3 — ppb → µg/m³ conversion (MW-based formula)
# ===========================================================================


class TestPpbToUgm3:
    """ppb_to_ugm3 converts ppb to µg/m³ using formula: µg/m³ = ppb × MW / 24.45.

    Inverse of ugm3_to_ppm × 1000 (since ppm = ppb / 1000).
    Requires pollutant keyword arg — uses _MOLAR_WEIGHTS table.
    Unknown pollutant → returns None (does not raise).
    None input not accepted (ppb is float, not float|None — caller guards).
    """

    def test_o3_1000_ppb_converts_to_expected_ugm3(self) -> None:
        """O3 1000 ppb → 1000 × 48.00 / 24.45 ≈ 1963.2 µg/m³."""
        import math  # noqa: PLC0415

        from weewx_clearskies_api.providers.aqi._units import ppb_to_ugm3  # noqa: PLC0415
        result = ppb_to_ugm3(1000.0, pollutant="O3")
        assert result is not None
        expected = 1000.0 * 48.00 / 24.45
        assert math.isclose(result, expected, rel_tol=1e-9), (
            f"O3 1000 ppb → expected {expected:.4f} µg/m³, got {result!r}"
        )

    def test_no2_36_ppb_converts_to_expected_ugm3(self) -> None:
        """NO2 36 ppb → 36 × 46.01 / 24.45 ≈ 67.74 µg/m³."""
        import math  # noqa: PLC0415

        from weewx_clearskies_api.providers.aqi._units import ppb_to_ugm3  # noqa: PLC0415
        result = ppb_to_ugm3(36.0, pollutant="NO2")
        assert result is not None
        expected = 36.0 * 46.01 / 24.45
        assert math.isclose(result, expected, rel_tol=1e-9), (
            f"NO2 36 ppb → expected {expected:.4f} µg/m³, got {result!r}"
        )

    def test_so2_0_ppb_returns_zero(self) -> None:
        """SO2 0 ppb → 0.0 µg/m³ (zero input stays zero)."""
        from weewx_clearskies_api.providers.aqi._units import ppb_to_ugm3  # noqa: PLC0415
        result = ppb_to_ugm3(0.0, pollutant="SO2")
        assert result == 0.0, f"SO2 0 ppb → expected 0.0 µg/m³, got {result!r}"

    def test_co_143_ppb_converts_to_expected_ugm3(self) -> None:
        """CO 143 ppb → 143 × 28.01 / 24.45 ≈ 163.87 µg/m³."""
        import math  # noqa: PLC0415

        from weewx_clearskies_api.providers.aqi._units import ppb_to_ugm3  # noqa: PLC0415
        result = ppb_to_ugm3(143.0, pollutant="CO")
        assert result is not None
        expected = 143.0 * 28.01 / 24.45
        assert math.isclose(result, expected, rel_tol=1e-9), (
            f"CO 143 ppb → expected {expected:.4f} µg/m³, got {result!r}"
        )

    def test_unknown_pollutant_returns_none(self) -> None:
        """ppb_to_ugm3(100, pollutant='UNKNOWN') → None (not in MW table)."""
        from weewx_clearskies_api.providers.aqi._units import ppb_to_ugm3  # noqa: PLC0415
        result = ppb_to_ugm3(100.0, pollutant="UNKNOWN")
        assert result is None, (
            f"Unknown pollutant must return None, got {result!r}"
        )

    def test_pm25_not_in_table_returns_none(self) -> None:
        """ppb_to_ugm3(1.0, pollutant='PM2.5') → None (particulates not in MW table)."""
        from weewx_clearskies_api.providers.aqi._units import ppb_to_ugm3  # noqa: PLC0415
        result = ppb_to_ugm3(1.0, pollutant="PM2.5")
        assert result is None

    def test_result_is_float_for_valid_input(self) -> None:
        """Return type is float (not None) for a valid gas pollutant."""
        from weewx_clearskies_api.providers.aqi._units import ppb_to_ugm3  # noqa: PLC0415
        result = ppb_to_ugm3(50.0, pollutant="O3")
        assert isinstance(result, float)

    def test_doubling_ppb_doubles_ugm3(self) -> None:
        """Linear formula: doubling ppb must double µg/m³."""
        import math  # noqa: PLC0415

        from weewx_clearskies_api.providers.aqi._units import ppb_to_ugm3  # noqa: PLC0415
        r1 = ppb_to_ugm3(100.0, pollutant="NO2")
        r2 = ppb_to_ugm3(200.0, pollutant="NO2")
        assert r1 is not None and r2 is not None
        assert math.isclose(r2, r1 * 2, rel_tol=1e-9)

    def test_ppb_to_ugm3_is_inverse_of_ugm3_to_ppm_times_1000(self) -> None:
        """Round-trip: ppb_to_ugm3(ugm3_to_ppm(c) × 1000) ≈ c for all supported gases."""
        import math  # noqa: PLC0415

        from weewx_clearskies_api.providers.aqi._units import ppb_to_ugm3, ugm3_to_ppm  # noqa: PLC0415
        for pollutant, c_ugm3 in [("O3", 87.0), ("NO2", 100.0), ("SO2", 50.0), ("CO", 155.0)]:
            ppm = ugm3_to_ppm(c_ugm3, pollutant=pollutant)
            assert ppm is not None
            ppb = ppm * 1000.0
            result = ppb_to_ugm3(ppb, pollutant=pollutant)
            assert result is not None
            assert math.isclose(result, c_ugm3, rel_tol=1e-9), (
                f"{pollutant}: round-trip failed — expected {c_ugm3}, got {result}"
            )


# ===========================================================================
# 4. concentration_to_sub_aqi — EPA breakpoint per-pollutant tables (3b-11)
# ===========================================================================
#
# Source for breakpoints:
#   EPA Technical Assistance Document for the Reporting of Daily Air Quality
#   https://document.airnow.gov/technical-assistance-document-for-the-reporting-of-daily-air-quailty.pdf
#   (PM2.5 reflects the 2024-09-18 NAAQS revision: 24-hr standard lowered to 9.0 µg/m³)
#
# Averaging-period choice per Q1 user decision 2026-05-10 (Option A):
#   O3 uses 8-hr table only; cap at sub-AQI 300 above 0.200 ppm.
#   SO2 uses 1-hr table only; cap at sub-AQI 200 above 0.304 ppm.
#
# Band boundary semantics: C_low is inclusive, C_high is inclusive.
# The formula round(((I_high-I_low)/(C_high-C_low))*(C-C_low)+I_low) handles boundaries.
# ===========================================================================


class TestConcentrationToSubAqiPM25:
    """concentration_to_sub_aqi for PM2.5 (µg/m³ — 24-hr avg, 2024 revised breakpoints).

    Bands (C_low, C_high, I_low, I_high):
      (0.0,   9.0,   0,   50)
      (9.1,  35.4,  51,  100)
      (35.5,  55.4, 101,  150)
      (55.5, 125.4, 151,  200)
      (125.5, 225.4, 201, 300)
      (225.5, 325.4, 301, 500)
    """

    def test_pm25_zero_is_sub_aqi_0(self) -> None:
        """PM2.5 = 0.0 µg/m³ → sub-AQI 0 (bottom of first band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.0, pollutant="PM2.5")
        assert result == 0, f"PM2.5=0.0 → expected sub-AQI 0, got {result!r}"

    def test_pm25_9_0_is_sub_aqi_50_boundary_exact(self) -> None:
        """PM2.5 = 9.0 µg/m³ exactly → sub-AQI 50 (top of first band).

        EPA boundary test: 9.0 must map to 50, NOT 51.
        The rounding formula produces exactly 50 at C_high = C = 9.0.
        """
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(9.0, pollutant="PM2.5")
        assert result == 50, (
            f"PM2.5=9.0 exactly → expected sub-AQI 50 (NOT 51), got {result!r}"
        )

    def test_pm25_4_5_midpoint_of_first_band(self) -> None:
        """PM2.5 = 4.5 µg/m³ (midpoint of first band 0–9) → sub-AQI 25."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(4.5, pollutant="PM2.5")
        assert result == 25, f"PM2.5=4.5 midpoint → expected 25, got {result!r}"

    def test_pm25_9_1_is_sub_aqi_51_bottom_of_second_band(self) -> None:
        """PM2.5 = 9.1 µg/m³ → sub-AQI 51 (bottom of Moderate band)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(9.1, pollutant="PM2.5")
        assert result == 51, f"PM2.5=9.1 → expected 51, got {result!r}"

    def test_pm25_35_4_is_sub_aqi_100_top_of_moderate(self) -> None:
        """PM2.5 = 35.4 µg/m³ → sub-AQI 100 (top of Moderate band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(35.4, pollutant="PM2.5")
        assert result == 100, f"PM2.5=35.4 → expected 100, got {result!r}"

    def test_pm25_22_25_midpoint_of_moderate_band(self) -> None:
        """PM2.5 = 22.25 µg/m³ (approx midpoint of Moderate band 9.1–35.4) → sub-AQI ~75."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(22.25, pollutant="PM2.5")
        # sub = (100-51)/(35.4-9.1) * (22.25-9.1) + 51 = (49/26.3)*13.15+51 ≈ 24.5+51 ≈ 75/76
        assert result in (75, 76), f"PM2.5=22.25 midpoint of moderate → expected ~75-76, got {result!r}"

    def test_pm25_35_5_is_sub_aqi_101_bottom_of_usg(self) -> None:
        """PM2.5 = 35.5 µg/m³ → sub-AQI 101 (bottom of Unhealthy for Sensitive Groups)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(35.5, pollutant="PM2.5")
        assert result == 101, f"PM2.5=35.5 → expected 101, got {result!r}"

    def test_pm25_55_4_is_sub_aqi_150_top_of_usg(self) -> None:
        """PM2.5 = 55.4 µg/m³ → sub-AQI 150 (top of USG band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(55.4, pollutant="PM2.5")
        assert result == 150, f"PM2.5=55.4 → expected 150, got {result!r}"

    def test_pm25_above_325_4_caps_at_500(self) -> None:
        """PM2.5 > 325.4 µg/m³ (above table top) → cap at 500."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(400.0, pollutant="PM2.5")
        assert result == 500, f"PM2.5 above table top → expected 500 cap, got {result!r}"

    def test_pm25_negative_returns_zero(self) -> None:
        """PM2.5 < 0 (defensive) → sub-AQI 0 (below-table floor)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(-1.0, pollutant="PM2.5")
        assert result == 0, f"PM2.5=-1.0 below table → expected 0, got {result!r}"

    def test_pm25_none_returns_none(self) -> None:
        """PM2.5 = None → sub-AQI None (None propagates per ADR-010)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(None, pollutant="PM2.5")
        assert result is None, f"None PM2.5 → expected None, got {result!r}"

    def test_pm25_fixture_value_0_5_is_sub_aqi_3(self) -> None:
        """PM2.5 = 0.5 µg/m³ (OWM fixture value) → sub-AQI 3."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        # sub = (50-0)/(9.0-0.0)*(0.5-0)+0 = 50/9*0.5 = 2.78 → round → 3
        result = concentration_to_sub_aqi(0.5, pollutant="PM2.5")
        assert result == 3, f"PM2.5=0.5 (fixture) → expected 3, got {result!r}"


class TestConcentrationToSubAqiPM10:
    """concentration_to_sub_aqi for PM10 (µg/m³ — 24-hr avg).

    Bands (C_low, C_high, I_low, I_high):
      (0,   54,    0,   50)
      (55,  154,  51,  100)
      (155, 254, 101,  150)
      (255, 354, 151,  200)
      (355, 424, 201,  300)
      (425, 604, 301,  500)
    """

    def test_pm10_zero_is_sub_aqi_0(self) -> None:
        """PM10 = 0 µg/m³ → sub-AQI 0."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.0, pollutant="PM10")
        assert result == 0, f"PM10=0.0 → expected 0, got {result!r}"

    def test_pm10_54_is_sub_aqi_50_top_of_good(self) -> None:
        """PM10 = 54 µg/m³ → sub-AQI 50 (top of Good band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(54.0, pollutant="PM10")
        assert result == 50, f"PM10=54 → expected 50, got {result!r}"

    def test_pm10_27_midpoint_of_good_band(self) -> None:
        """PM10 = 27 µg/m³ (midpoint of Good band 0–54) → sub-AQI 25."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(27.0, pollutant="PM10")
        assert result == 25, f"PM10=27 midpoint → expected 25, got {result!r}"

    def test_pm10_55_is_sub_aqi_51_bottom_of_moderate(self) -> None:
        """PM10 = 55 µg/m³ → sub-AQI 51 (bottom of Moderate band)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(55.0, pollutant="PM10")
        assert result == 51, f"PM10=55 → expected 51, got {result!r}"

    def test_pm10_154_is_sub_aqi_100_top_of_moderate(self) -> None:
        """PM10 = 154 µg/m³ → sub-AQI 100 (top of Moderate band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(154.0, pollutant="PM10")
        assert result == 100, f"PM10=154 → expected 100, got {result!r}"

    def test_pm10_above_604_caps_at_500(self) -> None:
        """PM10 > 604 µg/m³ → cap at 500."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(700.0, pollutant="PM10")
        assert result == 500, f"PM10 above table top → expected 500 cap, got {result!r}"

    def test_pm10_none_returns_none(self) -> None:
        """PM10 = None → sub-AQI None."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        assert concentration_to_sub_aqi(None, pollutant="PM10") is None

    def test_pm10_fixture_value_0_81_is_sub_aqi_1(self) -> None:
        """PM10 = 0.81 µg/m³ (OWM fixture value) → sub-AQI 1."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        # sub = (50-0)/(54-0)*(0.81-0)+0 = 50/54*0.81 = 0.75 → round → 1
        result = concentration_to_sub_aqi(0.81, pollutant="PM10")
        assert result == 1, f"PM10=0.81 (fixture) → expected 1, got {result!r}"


class TestConcentrationToSubAqiO3:
    """concentration_to_sub_aqi for O3 (ppm — 8-hr avg, Q1 Option A, cap at 300).

    Bands (C_low, C_high, I_low, I_high):
      (0.000, 0.054,   0,  50)
      (0.055, 0.070,  51, 100)
      (0.071, 0.085, 101, 150)
      (0.086, 0.105, 151, 200)
      (0.106, 0.200, 201, 300)
    Cap: > 0.200 ppm → sub-AQI 300 (Q1 Option A: 8-hr table only).
    """

    def test_o3_zero_is_sub_aqi_0(self) -> None:
        """O3 = 0.0 ppm → sub-AQI 0."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.0, pollutant="O3")
        assert result == 0, f"O3=0.0 → expected 0, got {result!r}"

    def test_o3_0_054_is_sub_aqi_50_top_of_good(self) -> None:
        """O3 = 0.054 ppm → sub-AQI 50 (top of Good band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.054, pollutant="O3")
        assert result == 50, f"O3=0.054 → expected 50, got {result!r}"

    def test_o3_0_027_midpoint_of_good_band(self) -> None:
        """O3 = 0.027 ppm (midpoint of Good band) → sub-AQI 25."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.027, pollutant="O3")
        assert result == 25, f"O3=0.027 midpoint → expected 25, got {result!r}"

    def test_o3_0_055_is_sub_aqi_51_bottom_of_moderate(self) -> None:
        """O3 = 0.055 ppm → sub-AQI 51 (bottom of Moderate band)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.055, pollutant="O3")
        assert result == 51, f"O3=0.055 → expected 51, got {result!r}"

    def test_o3_0_070_is_sub_aqi_100_top_of_moderate(self) -> None:
        """O3 = 0.070 ppm → sub-AQI 100 (top of Moderate band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.070, pollutant="O3")
        assert result == 100, f"O3=0.070 → expected 100, got {result!r}"

    def test_o3_0_200_is_sub_aqi_300_top_of_table(self) -> None:
        """O3 = 0.200 ppm → sub-AQI 300 (top of 8-hr table, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.200, pollutant="O3")
        assert result == 300, f"O3=0.200 → expected 300 (table top), got {result!r}"

    def test_o3_above_0_200_caps_at_300_not_500(self) -> None:
        """O3 > 0.200 ppm → cap at sub-AQI 300 (Q1 Option A: 8-hr table only).

        This is distinct from PM2.5/PM10/CO/NO2 which cap at 500.
        The 8-hr O3 table has no 301–500 band; we cap at the table top (300).
        """
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.5, pollutant="O3")
        assert result == 300, (
            f"O3 > 0.200 ppm must cap at 300 (Q1 Option A, 8-hr table only), got {result!r}"
        )

    def test_o3_none_returns_none(self) -> None:
        """O3 = None → sub-AQI None."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        assert concentration_to_sub_aqi(None, pollutant="O3") is None

    def test_o3_0_153_midpoint_of_very_unhealthy_band(self) -> None:
        """O3 = 0.153 ppm (midpoint of 201–300 band 0.106–0.200) → sub-AQI 250."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        # sub = (300-201)/(0.200-0.106)*(0.153-0.106)+201 = (99/0.094)*0.047+201 ≈ 49.5+201 = 250/251
        result = concentration_to_sub_aqi(0.153, pollutant="O3")
        assert result in (250, 251), f"O3=0.153 midpoint VU band → expected ~250-251, got {result!r}"


class TestConcentrationToSubAqiCO:
    """concentration_to_sub_aqi for CO (ppm — 8-hr avg).

    Bands (C_low, C_high, I_low, I_high):
      (0.0,   4.4,   0,   50)
      (4.5,   9.4,  51,  100)
      (9.5,  12.4, 101,  150)
      (12.5,  15.4, 151, 200)
      (15.5,  30.4, 201, 300)
      (30.5,  50.4, 301, 500)
    """

    def test_co_zero_is_sub_aqi_0(self) -> None:
        """CO = 0.0 ppm → sub-AQI 0."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.0, pollutant="CO")
        assert result == 0, f"CO=0.0 → expected 0, got {result!r}"

    def test_co_4_4_is_sub_aqi_50_top_of_good(self) -> None:
        """CO = 4.4 ppm → sub-AQI 50 (top of Good band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(4.4, pollutant="CO")
        assert result == 50, f"CO=4.4 → expected 50, got {result!r}"

    def test_co_2_2_midpoint_of_good_band(self) -> None:
        """CO = 2.2 ppm (midpoint of Good band 0–4.4) → sub-AQI 25."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(2.2, pollutant="CO")
        assert result == 25, f"CO=2.2 midpoint → expected 25, got {result!r}"

    def test_co_4_5_is_sub_aqi_51_bottom_of_moderate(self) -> None:
        """CO = 4.5 ppm → sub-AQI 51 (bottom of Moderate band)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(4.5, pollutant="CO")
        assert result == 51, f"CO=4.5 → expected 51, got {result!r}"

    def test_co_9_4_is_sub_aqi_100_top_of_moderate(self) -> None:
        """CO = 9.4 ppm → sub-AQI 100 (top of Moderate band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(9.4, pollutant="CO")
        assert result == 100, f"CO=9.4 → expected 100, got {result!r}"

    def test_co_above_50_4_caps_at_500(self) -> None:
        """CO > 50.4 ppm → cap at 500 (full table range, no early cap for CO)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(100.0, pollutant="CO")
        assert result == 500, f"CO > 50.4 ppm → expected 500 cap, got {result!r}"

    def test_co_none_returns_none(self) -> None:
        """CO = None → sub-AQI None."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        assert concentration_to_sub_aqi(None, pollutant="CO") is None


class TestConcentrationToSubAqiSO2:
    """concentration_to_sub_aqi for SO2 (ppm — 1-hr avg, Q1 Option A, cap at 200).

    Bands (C_low, C_high, I_low, I_high):
      (0.000, 0.035,   0,  50)
      (0.036, 0.075,  51, 100)
      (0.076, 0.185, 101, 150)
      (0.186, 0.304, 151, 200)
    Cap: > 0.304 ppm → sub-AQI 200 (Q1 Option A: 1-hr table only, no 201–300 band).
    """

    def test_so2_zero_is_sub_aqi_0(self) -> None:
        """SO2 = 0.0 ppm → sub-AQI 0."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.0, pollutant="SO2")
        assert result == 0, f"SO2=0.0 → expected 0, got {result!r}"

    def test_so2_0_035_is_sub_aqi_50_top_of_good(self) -> None:
        """SO2 = 0.035 ppm → sub-AQI 50 (top of Good band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.035, pollutant="SO2")
        assert result == 50, f"SO2=0.035 → expected 50, got {result!r}"

    def test_so2_0_036_is_sub_aqi_51_bottom_of_moderate(self) -> None:
        """SO2 = 0.036 ppm → sub-AQI 51 (bottom of Moderate band)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.036, pollutant="SO2")
        assert result == 51, f"SO2=0.036 → expected 51, got {result!r}"

    def test_so2_0_075_is_sub_aqi_100_top_of_moderate(self) -> None:
        """SO2 = 0.075 ppm → sub-AQI 100 (top of Moderate band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.075, pollutant="SO2")
        assert result == 100, f"SO2=0.075 → expected 100, got {result!r}"

    def test_so2_0_304_is_sub_aqi_200_top_of_table(self) -> None:
        """SO2 = 0.304 ppm → sub-AQI 200 (top of 1-hr table, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.304, pollutant="SO2")
        assert result == 200, f"SO2=0.304 → expected 200 (table top), got {result!r}"

    def test_so2_above_0_304_caps_at_200_not_500(self) -> None:
        """SO2 > 0.304 ppm → cap at sub-AQI 200 (Q1 Option A: 1-hr table only).

        Distinct from PM2.5/PM10/CO/NO2 which cap at 500.
        The 1-hr SO2 table has no 201–300 or 301–500 bands.
        """
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.5, pollutant="SO2")
        assert result == 200, (
            f"SO2 > 0.304 ppm must cap at 200 (Q1 Option A, 1-hr table only), got {result!r}"
        )

    def test_so2_none_returns_none(self) -> None:
        """SO2 = None → sub-AQI None."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        assert concentration_to_sub_aqi(None, pollutant="SO2") is None

    def test_so2_0_055_midpoint_of_moderate_band(self) -> None:
        """SO2 = 0.055 ppm (approx midpoint of Moderate 0.036–0.075) → sub-AQI 75."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        # sub = (100-51)/(0.075-0.036)*(0.055-0.036)+51 = (49/0.039)*0.019+51 ≈ 23.8+51 ≈ 75
        result = concentration_to_sub_aqi(0.055, pollutant="SO2")
        assert result in (75, 76), f"SO2=0.055 midpoint → expected ~75-76, got {result!r}"


class TestConcentrationToSubAqiNO2:
    """concentration_to_sub_aqi for NO2 (ppm — 1-hr avg).

    Bands (C_low, C_high, I_low, I_high):
      (0.000, 0.053,   0,   50)
      (0.054, 0.100,  51,  100)
      (0.101, 0.360, 101,  150)
      (0.361, 0.649, 151,  200)
      (0.650, 1.249, 201,  300)
      (1.250, 2.049, 301,  500)
    """

    def test_no2_zero_is_sub_aqi_0(self) -> None:
        """NO2 = 0.0 ppm → sub-AQI 0."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.0, pollutant="NO2")
        assert result == 0, f"NO2=0.0 → expected 0, got {result!r}"

    def test_no2_0_053_is_sub_aqi_50_top_of_good(self) -> None:
        """NO2 = 0.053 ppm → sub-AQI 50 (top of Good band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.053, pollutant="NO2")
        assert result == 50, f"NO2=0.053 → expected 50, got {result!r}"

    def test_no2_0_027_midpoint_of_good_band(self) -> None:
        """NO2 = 0.027 ppm (approx midpoint of Good band 0–0.053) → sub-AQI 25."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.027, pollutant="NO2")
        # sub = (50-0)/(0.053-0.000)*(0.027-0)+0 ≈ 943.4*0.027 ≈ 25.5 → 25 or 26
        assert result in (25, 26), f"NO2=0.027 midpoint → expected ~25-26, got {result!r}"

    def test_no2_0_054_is_sub_aqi_51_bottom_of_moderate(self) -> None:
        """NO2 = 0.054 ppm → sub-AQI 51 (bottom of Moderate band)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.054, pollutant="NO2")
        assert result == 51, f"NO2=0.054 → expected 51, got {result!r}"

    def test_no2_0_100_is_sub_aqi_100_top_of_moderate(self) -> None:
        """NO2 = 0.100 ppm → sub-AQI 100 (top of Moderate band, inclusive)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.100, pollutant="NO2")
        assert result == 100, f"NO2=0.100 → expected 100, got {result!r}"

    def test_no2_above_2_049_caps_at_500(self) -> None:
        """NO2 > 2.049 ppm → cap at 500."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(3.0, pollutant="NO2")
        assert result == 500, f"NO2 above table top → expected 500 cap, got {result!r}"

    def test_no2_none_returns_none(self) -> None:
        """NO2 = None → sub-AQI None."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        assert concentration_to_sub_aqi(None, pollutant="NO2") is None

    def test_no2_1_089_ppm_synthetic_high_is_in_very_unhealthy_band(self) -> None:
        """NO2 = 1.089 ppm (synthetic high concentration) → sub-AQI ~274.

        Synthetic, not fixture-derived. Real Seattle NO2 from the OWM fixture is
        2.05 µg/m³ = 0.001089 ppm (per chemistry fix 2026-05-11), which lands in
        the (0, 0.053, 0, 50) Good band — see test_no2_fixture_value_0_001089_ppm_is_good.
        This test exercises the upper band math by injecting 1000× the fixture value.

        Band [0.650, 1.249, 201, 300]: sub = (300-201)/(1.249-0.650)*(1.089-0.650)+201.
        Approximate: (99/0.599)*0.439+201 ≈ 72.6+201 = 273.6 → 274
        """
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(1.089, pollutant="NO2")
        assert result is not None
        assert 270 <= result <= 278, (
            f"NO2=1.089 ppm (synthetic) → expected ~274 (Very Unhealthy band), got {result!r}"
        )

    def test_no2_fixture_value_0_001089_ppm_is_good(self) -> None:
        """NO2 = 0.001089 ppm (real OWM fixture value, corrected chemistry) → sub-AQI ~1.

        Fixture wire: 2.05 µg/m³ × 24.45 / (46.01 × 1000) ≈ 0.001089 ppm.
        Band (0.000, 0.053, 0, 50): sub = (50/0.053) × 0.001089 ≈ 1.03 → 1.
        Verifies the fixture-real-value mapping against EPA after the chemistry fix.
        """
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        result = concentration_to_sub_aqi(0.001089, pollutant="NO2")
        assert result is not None
        assert result <= 5, (
            f"NO2=0.001089 ppm (fixture) → expected ≤5 (Good band), got {result!r}"
        )


class TestConcentrationToSubAqiUnknownPollutant:
    """Unknown pollutant id raises KeyError (not silently ignored)."""

    def test_unknown_pollutant_raises_key_error(self) -> None:
        """concentration_to_sub_aqi('NH3') → KeyError (NH3 has no EPA AQI band)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        with pytest.raises(KeyError):
            concentration_to_sub_aqi(10.0, pollutant="NH3")

    def test_totally_unknown_pollutant_raises_key_error(self) -> None:
        """concentration_to_sub_aqi('UNKNOWN') → KeyError (canonical id required)."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        with pytest.raises(KeyError):
            concentration_to_sub_aqi(10.0, pollutant="UNKNOWN")

    def test_lowercase_pollutant_id_raises_key_error(self) -> None:
        """concentration_to_sub_aqi('pm2.5') → KeyError (must use canonical 'PM2.5')."""
        from weewx_clearskies_api.providers.aqi._units import (
            concentration_to_sub_aqi,  # noqa: PLC0415
        )
        with pytest.raises(KeyError):
            concentration_to_sub_aqi(5.0, pollutant="pm2.5")
