"""µg/m³ → ppm gas conversion, PPB → ppm direct conversion, EPA AQI category band table,
EPA per-pollutant breakpoint table for sub-AQI computation.

Conversion formula (corrected 2026-05-11 — 3b-11 round-close chemistry fix):
    ppm = µg/m³ × 24.45 / (molecular_weight × 1000)
        = µg/m³ × 0.02445 / molecular_weight
where 24.45 L/mol is the molar volume of an ideal gas at 25°C and 1 atm.

Derivation: 1 m³ of gas at STP contains 1/0.02445 ≈ 40.9 mol. A pollutant
concentration of C µg/m³ contributes (C × 10⁻⁶)/MW mol/m³, giving a mole
fraction of (C × 10⁻⁶)/(MW × 40.9) = C × 24.45 × 10⁻⁹/MW. Multiplying by
10⁶ to express as ppm (parts per million by volume) yields C × 24.45/(MW × 10³).

The pre-fix formula (`µg/m³ × 24.45 / MW` with no /1000 factor) produced PPB,
not PPM — values 1000× the canonical-data-model §3.8 group_fraction (ppm) spec.
Surfaced by 3b-11 OWM real-capture fixture (Seattle CO=139.79 µg/m³ computed
sub-AQI 500 / "Hazardous" before fix; ~1 / "Good" after). See round-close
decision log for the bug-propagation analysis.

PPB → ppm direct conversion (for providers that supply valuePPB directly, e.g. Aeris):
    ppm = ppb / 1000

EPA AQI category breakpoints per canonical-data-model §3.8 (canonical):
    0-50      → Good
    51-100    → Moderate
    101-150   → Unhealthy for Sensitive Groups
    151-200   → Unhealthy
    201-300   → Very Unhealthy
    301-500   → Hazardous

EPA per-pollutant sub-AQI breakpoints (_EPA_BREAKPOINTS + concentration_to_sub_aqi):
    Piecewise-linear interpolation per EPA Technical Assistance Document (TAD).
    Source: https://document.airnow.gov/technical-assistance-document-for-the-reporting-of-daily-air-quailty.pdf
    (2024-09-18 PM2.5 revision applies for PM2.5 breakpoints.)
    Concentrations in the same units canonical AQIReading uses for each field:
      PM2.5, PM10 — µg/m³ (group_concentration)
      O3, CO, SO2, NO2 — ppm (group_fraction, after ugm3_to_ppm conversion)
    Averaging-period choice per Q1 user decision 2026-05-10 (Option A):
      O3 uses the 8-hr table only; cap at sub-AQI 300 above 0.200 ppm.
      SO2 uses the 1-hr table only; cap at sub-AQI 200 above 0.304 ppm.
      Rationale: honest about the cap; matches the conservative posture other
      AQI services take for unspecified-averaging-period inputs (OWM returns
      a snapshot that doesn't distinguish averaging periods).

Tables are static; molecular weights and EPA bands are constants of nature
+ EPA regulation respectively (not provider-specific).
"""

from __future__ import annotations

# Molar volume at 25°C / 1 atm.  Used by µg/m³ ↔ ppm conversions.
_MOLAR_VOLUME = 24.45  # L/mol

# Molecular weights for the four canonical gas pollutants.
# Particulates (PM2.5, PM10) stay in µg/m³ — no molar conversion.
_MOLECULAR_WEIGHTS_G_PER_MOL: dict[str, float] = {
    "O3":  48.00,
    "NO2": 46.01,
    "SO2": 64.07,
    "CO":  28.01,
}

# Alias used by ppb_to_ugm3 — same table, shorter name.
_MOLAR_WEIGHTS = _MOLECULAR_WEIGHTS_G_PER_MOL


def ugm3_to_ppm(ugm3: float | None, *, pollutant: str) -> float | None:
    """Convert µg/m³ concentration to ppm for the given gas.

    Formula: ppm = µg/m³ × 24.45 / (MW × 1000) = µg/m³ × 0.02445 / MW
    (at 25°C, 1 atm; molar volume 24.45 L/mol).

    Args:
        ugm3: concentration in µg/m³ (or None).
        pollutant: canonical pollutant id ("O3", "NO2", "SO2", "CO").

    Returns:
        ppm value (None propagates).

    Raises:
        KeyError: if pollutant is not in the conversion table.
    """
    if ugm3 is None:
        return None
    mw = _MOLECULAR_WEIGHTS_G_PER_MOL[pollutant]
    # µg/m³ × 24.45 / MW = ppb. Divide by 1000 to get ppm.
    return ugm3 * _MOLAR_VOLUME / (mw * 1000.0)


# EPA AQI category breakpoints (upper bounds, inclusive).
# Bisect-by-upper-bound dispatch: aqi value <= upper → that category.
# Order matters — list MUST be sorted by upper bound ascending.
_EPA_CATEGORY_BANDS: list[tuple[int, str]] = [
    (50,  "Good"),
    (100, "Moderate"),
    (150, "Unhealthy for Sensitive Groups"),
    (200, "Unhealthy"),
    (300, "Very Unhealthy"),
    (500, "Hazardous"),
]


def ppb_to_ppm(ppb: float | None) -> float | None:
    """Convert ppb (parts per billion) to ppm (parts per million).

    Used by the Aeris AQI provider which returns gas concentrations in valuePPB
    directly (O3, NO2, SO2, CO).  Distinct from ugm3_to_ppm — no molar volume
    or molecular weight needed; the conversion is purely a 1000x scale factor.

    Args:
        ppb: concentration in ppb (or None).

    Returns:
        ppm value (None propagates).  ppm = ppb / 1000.
    """
    if ppb is None:
        return None
    return ppb / 1000.0


def ppb_to_ugm3(ppb: float, *, pollutant: str) -> float | None:
    """Convert ppb (parts per billion) to µg/m³ for the given gas.

    Formula: µg/m³ = ppb × MW / 24.45
    (at 25°C, 1 atm; molar volume 24.45 L/mol).
    Inverse of ugm3_to_ppm × 1000 (since ppm = ppb / 1000).

    Args:
        ppb: concentration in ppb.
        pollutant: canonical pollutant id ("O3", "NO2", "SO2", "CO").

    Returns:
        µg/m³ value, or None if pollutant is not in the conversion table.
    """
    mw = _MOLAR_WEIGHTS.get(pollutant.upper())
    if mw is None:
        return None
    return ppb * mw / 24.45


def epa_category(aqi: int | float | None) -> str | None:
    """Map a 0–500 EPA AQI value to its category name.

    Args:
        aqi: AQI value (or None).

    Returns:
        EPA category name (canonical spelling per canonical §3.8) or None.
        Values > 500 fall into "Hazardous" (max band) for safety.
    """
    if aqi is None:
        return None
    for upper, name in _EPA_CATEGORY_BANDS:
        if aqi <= upper:
            return name
    # Above 500 — cap at "Hazardous" (top band) rather than raising. Spec is
    # 0-500 but provider-side bugs producing 501+ shouldn't crash us.
    return "Hazardous"


# ---------------------------------------------------------------------------
# EPA per-pollutant sub-AQI breakpoint tables (added in 3b-11 for OWM AQI)
# ---------------------------------------------------------------------------
#
# Source: EPA Technical Assistance Document for the Reporting of Daily Air Quality
#   https://document.airnow.gov/technical-assistance-document-for-the-reporting-of-daily-air-quailty.pdf
#   PM2.5 breakpoints reflect the 2024-09-18 revision (24-hr NAAQS lowered from
#   12.0/35.4 to 9.0/35.4 µg/m³ for the first two breakpoints).
#
# Structure: pollutant_id → list of (C_low, C_high, I_low, I_high) tuples
#   where C_low/C_high are concentration bounds and I_low/I_high are AQI index bounds.
#   Tuples are in ascending AQI order (lowest band first).
#   Concentrations in the units canonical AQIReading uses for that field.
#
# Averaging-period choice per Q1 user decision 2026-05-10 (Option A):
#   O3: 8-hr table only; cap at I_high=300 above 0.200 ppm.
#   SO2: 1-hr table only; cap at I_high=200 above 0.304 ppm.
#
# PM2.5 — µg/m³ — 24-hr avg (2024 revised breakpoints):
#   Bands:  0–50 / 51–100 / 101–150 / 151–200 / 201–300 / 301–500
#   C_low:  0.0 / 9.1 / 35.5 / 55.5 / 125.5 / 225.5
#   C_high: 9.0 / 35.4 / 55.4 / 125.4 / 225.4 / 325.4
#
# PM10 — µg/m³ — 24-hr avg:
#   Bands:  0–50 / 51–100 / 101–150 / 151–200 / 201–300 / 301–500
#   C_low:  0 / 55 / 155 / 255 / 355 / 425
#   C_high: 54 / 154 / 254 / 354 / 424 / 604
#
# O3 — ppm — 8-hr avg (Q1 Option A: cap at sub-AQI 300 above 0.200 ppm):
#   Bands:  0–50 / 51–100 / 101–150 / 151–200 / 201–300
#   C_low:  0.000 / 0.055 / 0.071 / 0.086 / 0.106
#   C_high: 0.054 / 0.070 / 0.085 / 0.105 / 0.200
#   (No 301–500 band in the 8-hr O3 table.)
#
# CO — ppm — 8-hr avg:
#   Bands:  0–50 / 51–100 / 101–150 / 151–200 / 201–300 / 301–500
#   C_low:  0.0 / 4.5 / 9.5 / 12.5 / 15.5 / 30.5
#   C_high: 4.4 / 9.4 / 12.4 / 15.4 / 30.4 / 50.4
#
# SO2 — ppm — 1-hr avg (Q1 Option A: cap at sub-AQI 200 above 0.304 ppm):
#   Bands:  0–50 / 51–100 / 101–150 / 151–200
#   C_low:  0.000 / 0.036 / 0.076 / 0.186
#   C_high: 0.035 / 0.075 / 0.185 / 0.304
#   (No 201–300 / 301–500 bands in the 1-hr SO2 table.)
#
# NO2 — ppm — 1-hr avg:
#   Bands:  0–50 / 51–100 / 101–150 / 151–200 / 201–300 / 301–500
#   C_low:  0.000 / 0.054 / 0.101 / 0.361 / 0.650 / 1.250
#   C_high: 0.053 / 0.100 / 0.360 / 0.649 / 1.249 / 2.049

# Each tuple: (C_low, C_high, I_low, I_high)
_EPA_BREAKPOINTS: dict[str, list[tuple[float, float, int, int]]] = {
    "PM2.5": [
        (0.0,   9.0,   0,   50),
        (9.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 125.4, 151,  200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
    ],
    "PM10": [
        (0,   54,    0,   50),
        (55,  154,  51,  100),
        (155, 254, 101,  150),
        (255, 354, 151,  200),
        (355, 424, 201,  300),
        (425, 604, 301,  500),
    ],
    "O3": [
        # 8-hr table only (Q1 Option A). Cap at I_high=300 above 0.200 ppm.
        (0.000, 0.054,   0,  50),
        (0.055, 0.070,  51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ],
    "CO": [
        (0.0,   4.4,   0,   50),
        (4.5,   9.4,  51,  100),
        (9.5,  12.4, 101,  150),
        (12.5,  15.4, 151, 200),
        (15.5,  30.4, 201, 300),
        (30.5,  50.4, 301, 500),
    ],
    "SO2": [
        # 1-hr table only (Q1 Option A). Cap at I_high=200 above 0.304 ppm.
        (0.000, 0.035,   0,  50),
        (0.036, 0.075,  51, 100),
        (0.076, 0.185, 101, 150),
        (0.186, 0.304, 151, 200),
    ],
    "NO2": [
        (0.000, 0.053,   0,   50),
        (0.054, 0.100,  51,  100),
        (0.101, 0.360, 101,  150),
        (0.361, 0.649, 151,  200),
        (0.650, 1.249, 201,  300),
        (1.250, 2.049, 301,  500),
    ],
}


def concentration_to_sub_aqi(
    concentration: float | None,
    *,
    pollutant: str,
) -> int | None:
    """Compute EPA sub-AQI from a pollutant concentration via piecewise-linear interpolation.

    Uses the EPA breakpoint table per the Technical Assistance Document:
        https://document.airnow.gov/technical-assistance-document-for-the-reporting-of-daily-air-quailty.pdf

    Formula (EPA TAD §4.4.1):
        sub_aqi = round( ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low )

    Cap behavior (Q1 user decision 2026-05-10, Option A):
        Values above the table's top C_high return the table-top I_high:
          O3: cap at sub-AQI 300 (8-hr table; 0.200 ppm)
          SO2: cap at sub-AQI 200 (1-hr table; 0.304 ppm)
          Others (PM2.5, PM10, CO, NO2): cap at 500.
        This is the conservative honest answer for OWM's instantaneous snapshots
        which don't carry averaging-period information.

    Args:
        concentration: pollutant concentration in canonical units for that field
            (µg/m³ for PM2.5/PM10; ppm for O3/CO/SO2/NO2).
        pollutant: canonical pollutant id — one of "PM2.5", "PM10", "O3",
            "CO", "SO2", "NO2".

    Returns:
        Integer 0–500 sub-AQI (or table-top cap for O3/SO2), or None when
        concentration is None.
        Values below the table's bottom C_low (typically 0.0) return 0.

    Raises:
        KeyError: pollutant not in _EPA_BREAKPOINTS (canonical id required).
    """
    if concentration is None:
        return None

    bands = _EPA_BREAKPOINTS[pollutant]  # KeyError propagates for unknown pollutant

    # Below the table's minimum — return 0 (defensive; concentrations should not be negative)
    if concentration < bands[0][0]:
        return 0

    # Walk bands to find the matching interval
    for c_low, c_high, i_low, i_high in bands:
        if concentration <= c_high:
            # Piecewise-linear interpolation per EPA TAD §4.4.1
            sub = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(sub)

    # Above the table's top breakpoint — return table-top I_high (cap behavior).
    # For O3 this is 300; for SO2 this is 200; for others it is 500.
    return bands[-1][3]
