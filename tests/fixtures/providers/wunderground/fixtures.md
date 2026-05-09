# Weather Underground Forecast Provider Fixtures

Sidecar documentation per the synthetic-from-api-docs fixture pattern (L3 carry-forward from
3b-4; codified in `.claude/agents/clearskies-test-author.md`).

## Fixture origin

**ALL fixtures in this directory are SYNTHETIC — constructed from
`docs/reference/api-docs/wunderground.md` L138-189 example response. Fields are mirrored
from the api-docs example; NO fixtures were captured live against a real Wunderground PWS API
endpoint.**

Reason: Wunderground PWS API gating is the strictest of the three keyed providers (active PWS
contributor required; key not issued to non-uploading accounts). Neither test-author nor lead
has an active PWS at fixture-capture time (2026-05-09). Synthetic-from-api-docs pattern applied
per brief L3 rule, carry-forward from 3b-4 process lesson.

When a PWS account becomes available in a future round, real-capture fixtures can be swapped in;
the test code should not need to change.

Provenance confirmed to lead (Opus) via SendMessage at fixture-capture time per the L3 rule.

---

## forecast_daily_5day.json

- **Type:** Synthetic — constructed from `docs/reference/api-docs/wunderground.md` L138-189
  example response. Fields mirrored from the example; NOT captured live.
- **Created:** 2026-05-09
- **Lat/Lon:** 47.6062 N, 122.3321 W (Seattle, WA — same coordinates as NWS/Aeris/OWM fixtures)
- **Endpoint:** `/v3/wx/forecast/daily/5day?geocode=47.6062,-122.3321&format=json&units=e&language=en-US`
- **Units simulated:** `e` (English/imperial — °F, mph, in)
- **Days:** 5 (all daypart[0] slots populated, 0–9 for 5 days × Day/Night)

### Fields mirrored from api-docs L138-189

Top-level array fields (5 elements, one per day):
- `calendarDayTemperatureMax`, `calendarDayTemperatureMin`
- `dayOfWeek`
- `expirationTimeUtc`
- `moonPhase`, `moonPhaseCode`, `moonPhaseDay`
- `moonriseTimeLocal`, `moonsetTimeLocal`
- `narrative`
- `qpf`, `qpfSnow`
- `temperatureMax`, `temperatureMin`
- `validTimeLocal`, `validTimeUtc`

### Fields added beyond api-docs L138-189 for canonical-mapping coverage

These fields are referenced in `canonical-data-model.md` §4.1.3 Wunderground column but NOT
shown in the truncated api-docs example (which shows only `sunriseTimeLocal`/`sunsetTimeLocal`):

- **`sunriseTimeUtc`** — 5-element epoch array. Injected as realistic epoch values for
  2026-04-30 through 2026-05-04 (~06:15 PDT = ~13:15 UTC ≈ epoch 1746017700+).
  Canonical field `sunrise` maps from this via `epoch_to_utc_iso8601()`. Per brief lead-call 25:
  "community references confirm both sunriseTimeUtc/sunsetTimeUtc and Local forms exist in real
  PWS-tier responses."
- **`sunsetTimeUtc`** — 5-element epoch array. Injected similarly (~19:05 PDT ≈ epoch 1746063900+).

`daypart[0]` arrays (10 elements = 5 days × D/N slots):
- All daypart fields from api-docs L157-185 are present and fully populated for all 10 slots.

### precipType coverage in this fixture

- Slot 0 (Today/D): `"rain"` → canonical `"rain"`
- Slot 1 (Tonight/N): `"rain"` → canonical `"rain"`
- Slot 2 (Friday/D): `"rain"` → canonical `"rain"`
- Slot 3 (Friday Night/N): `"rain"` → canonical `"rain"`
- Slot 4 (Saturday/D): `null` → canonical `None`
- Slot 5 (Saturday Night/N): `"rain"` → canonical `"rain"`
- Slot 6 (Sunday/D): `"rain"` → canonical `"rain"`
- Slot 7 (Sunday Night/N): `"rain"` → canonical `"rain"`
- Slot 8 (Monday/D): `null` → canonical `None`
- Slot 9 (Monday Night/N): `null` → canonical `None`

Note: `"snow"`, `"precip"`, and `"ice"` precipType values are NOT in this fixture.
Those code paths are tested with synthetic per-day dicts in the unit suite.

---

## forecast_daily_5day_passed_today.json

- **Type:** Synthetic variant of `forecast_daily_5day.json`.
- **Created:** 2026-05-09
- **Difference from base:** `daypart[0]` slot 0 (Today/Day period) is null across all
  daypart-derived fields. This simulates a late-afternoon request where the day period has
  already passed per api-docs §Notes L191: "Past-period slots may be null."
- **Top-level fields stay populated:** `temperatureMax[0]`, `temperatureMin[0]`,
  `validTimeLocal[0]`, `narrative[0]`, `qpf[0]`, `sunriseTimeUtc[0]`, `sunsetTimeUtc[0]`
  are NOT null (these are top-level, not daypart-derived).
- **Canonical impact:** `precipProbabilityMax`, `windSpeedMax`, `uvIndexMax`, `weatherCode`,
  `weatherText` for day 0 must emit as `None` (daypart-derived fields, slot is null).

---

## error_401_invalid_key.json

- **Type:** Synthetic — based on standard HTTP 401 Unauthorized error envelope shape for
  api.weather.com. Matches the documented behavior per
  `docs/reference/api-docs/wunderground.md` §Authentication: "apiKey invalid OR PWS no
  longer active."
- **Created:** 2026-05-09
- **HTTP status:** 401
- **Used to test:** `KeyInvalid` exception propagation → 502 ProviderProblem KeyInvalid.

---

## error_429_quota.json

- **Type:** Synthetic — based on standard HTTP 429 Too Many Requests error envelope shape for
  api.weather.com. Matches the documented rate-limit behavior per
  `docs/reference/api-docs/wunderground.md` §"Rate limits": 1500 calls/day, 30 calls/minute.
- **Created:** 2026-05-09
- **HTTP status:** 429
- **Used to test:** `QuotaExhausted` exception propagation → 503 ProviderProblem
  QuotaExhausted with `retry_after_seconds` attribute.
