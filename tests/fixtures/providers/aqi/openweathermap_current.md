# openweathermap_current.json — fixture sidecar

## Capture metadata

- **Provider:** OpenWeatherMap Air Pollution API (free tier)
- **Endpoint:** `GET https://api.openweathermap.org/data/2.5/air_pollution`
- **Capture date (UTC):** 2026-05-11T03:57:08Z
- **Coordinates:** `lat=47.6062 lon=-122.3321` (Seattle, WA — same as 3b-9/3b-10 AQI fixtures)
- **Curl invocation (appid redacted):**
  ```
  curl "https://api.openweathermap.org/data/2.5/air_pollution?lat=47.6062&lon=-122.3321&appid=REDACTED"
  ```
- **sha256 of raw JSON body:**
  `b25bccd70804302974e365f33122f486c5294a267a84f5b06180147979fc101d`
- **Fixture origin:** Real capture (free-tier; no L3 synthetic fallback needed)

## Wire values

| Field | Value | Notes |
|---|---|---|
| `list[0].main.aqi` | 2 | OWM 1–5 ordinal (2 = Fair). IGNORED per LC4 — canonical aqi derived from concentrations. |
| `list[0].components.co` | 139.79 µg/m³ | Converted to ppm for canonical + sub-AQI computation |
| `list[0].components.no` | 0 µg/m³ | Dropped (no EPA AQI band for NO) |
| `list[0].components.no2` | 2.05 µg/m³ | Converted to ppm |
| `list[0].components.o3` | 66.23 µg/m³ | Converted to ppm |
| `list[0].components.so2` | 0.34 µg/m³ | Converted to ppm |
| `list[0].components.pm2_5` | 0.5 µg/m³ | Passthrough (group_concentration) |
| `list[0].components.pm10` | 0.81 µg/m³ | Passthrough (group_concentration) |
| `list[0].components.nh3` | 0.37 µg/m³ | Dropped (no EPA AQI band for NH3) |
| `list[0].dt` | 1778471818 | Unix UTC seconds → `2026-05-11T03:56:58Z` |

## Expected canonical output

With the **corrected** ugm3_to_ppm formula (`ppm = µg/m³ × 24.45 / (MW × 1000)`, chemistry fix 2026-05-11) and EPA breakpoint tables:

| Pollutant | µg/m³ | ppm (formula) | EPA band | sub-AQI |
|---|---|---|---|---|
| O3 | 66.23 | 0.033736 | (0.000, 0.054, 0, 50) | round(50 × 0.033736 / 0.054) = 31 |
| NO2 | 2.05 | 0.001089 | (0.000, 0.053, 0, 50) | round(50 × 0.001089 / 0.053) = 1 |
| SO2 | 0.34 | 0.0001297 | (0.000, 0.035, 0, 50) | round(50 × 0.0001297 / 0.035) = 0 |
| CO | 139.79 | 0.12202 | (0.0, 4.4, 0, 50) | round(50 × 0.12202 / 4.4) = 1 |
| PM2.5 | 0.5 | — | (0.0, 9.0, 0, 50) | round(50 × 0.5 / 9.0) = 3 |
| PM10 | 0.81 | — | (0, 54, 0, 50) | round(50 × 0.81 / 54) = 1 |

- **aqi** = max(31, 1, 0, 1, 3, 1) = 31
- **aqiCategory** = "Good" (AQI 0–50 band)
- **aqiMainPollutant** = "O3" (argmax; sub-AQI 31 is unambiguous winner)
- **aqiLocation** = null (PARTIAL-DOMAIN — no location field on OWM Air Pollution wire)
- **observedAt** = "2026-05-11T03:56:58Z" (epoch 1778471818 → UTC ISO-8601)
- **source** = "openweathermap"

### Pre-fix expected (encoded the chemistry bug — left here for round-close audit trail)

Before the 2026-05-11 chemistry fix, the `ugm3_to_ppm` formula omitted the `/1000` factor and produced ppb-as-ppm. Computed values **with the bug** were: O3=33.736 / NO2=1.089 / SO2=0.1297 / CO=122.02 (all 1000× the correct ppm). EPA breakpoint comparison against these inflated values yielded sub-AQIs O3=300(cap) / NO2=274 / SO2=125 / CO=500(cap) → final aqi=500 / aqiMainPollutant=CO / aqiCategory=Hazardous — wildly wrong for a normal-air Seattle reading. Bug surfaced at 3b-11 round close; fix landed before 3b-11 close commit.
