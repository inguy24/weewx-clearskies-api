# Fixture: alerts_active.json

**Source location:** Seattle, WA — `?point=47.6062,-122.3321` (NWS example from `docs/reference/api-docs/nws.md`)

**Provenance:** Hand-crafted 2026-05-07 from the exact wire shape documented in
`docs/reference/api-docs/nws.md` §Active alerts (lines 215–249). The field set
matches the NWS `/alerts/active` GeoJSON FeatureCollection envelope verbatim
(type, features[].id, features[].type, features[].geometry, features[].properties.*,
title, updated).

**Alert count:** 2 alerts
  - Alert 1: Wind Advisory — severity=Moderate → canonical `advisory`
  - Alert 2: Small Craft Advisory — severity=Minor → canonical `advisory`

**Why not captured live:** Live NWS capture runs on `weather-dev` (per
`rules/clearskies-process.md`), not on the Windows DILBERT workstation.
This hand-crafted fixture uses the real NWS response shape from the api-docs
reference rather than a synthetic minimum-viable fixture — every field that
`_NwsAlertProperties` validates is present, including all optional ones that
may surface protocol-evolution bugs. The `instruction` field is populated on
alert 1 and null on alert 2, exercising both paths in `_to_canonical`.

**Re-capture instructions (for weather-dev):**
```bash
curl -H "User-Agent: (clearskies-test-fixture-capture, contact@example.com)" \
     -H "Accept: application/geo+json" \
     "https://api.weather.gov/alerts/active?point=47.6062,-122.3321" \
     | python -m json.tool > tests/fixtures/providers/nws/alerts_active.json
# Then update this .md with the capture date, location, and alert count.
```

**Schema shape rule:** This fixture is the source of truth for `_NwsAlertProperties`
wire-shape Pydantic model field coverage. Do not reduce to a synthetic `{"id", "headline"}`
subset — that hides protocol-evolution bugs per `rules/clearskies-process.md`
"Real schemas in unit tests where the schema shape matters."
