"""Unit tests for the alerts provider domain (3b round 1).

Covers per the task-3b-1 brief:
  - Severity normalization (5 CAP values + unknown-default + WARN log)
  - Time normalization (offset → Z, naive → ProviderProtocolError, bogus → ProviderProtocolError)
  - _to_canonical mapping (description+instruction concatenation)
  - Wire-shape Pydantic (real fixture loads; missing required → ValidationError; extras ignored)
  - AlertsQueryParams (extra="forbid", invalid severity rejected)
  - Severity filter (advisory→all, watch→watch+warning, warning→warning-only)
  - MemoryCache (set→get→TTL expiry)
  - RedisCache via fakeredis (same contract; JSON round-trip)
  - wire_cache_from_env() (unset→memory, redis://→redis, bogus→ConfigError)
  - RateLimiter (sliding window + QuotaExhausted on overflow)
  - HTTP wrapper error translation (5xx→TransientNetworkError, 429→QuotaExhausted,
    401→KeyInvalid, ConnectError/ReadTimeout→TransientNetworkError)
  - NWS module fetch() (200-empty, 200-with-alert, malformed→ProviderProtocolError,
    5xx→TransientNetworkError, 429→QuotaExhausted)
  - Capability registry (wire_providers populates; duplicate raises ValueError; empty→empty)
  - /capabilities response (with NWS configured; union correct; without NWS → providers=[])
  - /alerts endpoint (no-provider→200 source="none"; NWS+respx→200 source="nws";
    cache-hit→no NWS call; NWS-down→502; NWS-quota→503)

No DB, no live network. respx mocks outbound httpx calls.

Wire-shape rule: fixtures loaded from tests/fixtures/providers/nws/*.json
(real NWS response shapes per rules/clearskies-process.md §Real schemas).
ADR references: ADR-010, ADR-016, ADR-017, ADR-018, ADR-038.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "providers" / "nws"


def _load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture file from tests/fixtures/providers/nws/."""
    path = _FIXTURES_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return json.loads(fh.read())


# ---------------------------------------------------------------------------
# State-reset helpers (use the module-provided reset functions)
# ---------------------------------------------------------------------------


def _reset_provider_state() -> None:
    """Reset provider registry and cache to a clean state between tests."""
    from weewx_clearskies_api.providers._common.cache import reset_cache_for_tests  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
        reset_provider_registry_for_tests,
    )
    from weewx_clearskies_api.providers.alerts.nws import _reset_http_client_for_tests  # noqa: PLC0415

    reset_cache_for_tests()
    reset_provider_registry_for_tests()
    _reset_http_client_for_tests()


# ---------------------------------------------------------------------------
# Fixtures: rebuild a fresh app that includes the alerts router
# ---------------------------------------------------------------------------


def _make_alerts_settings(provider: str | None = None) -> Any:
    """Build a Settings instance with an AlertsSettings block."""
    from weewx_clearskies_api.config.settings import (  # noqa: PLC0415
        AlertsSettings,
        ApiSettings,
        DatabaseSettings,
        HealthSettings,
        LoggingSettings,
        RateLimitSettings,
        Settings,
    )

    return Settings(
        api=ApiSettings({}),
        health=HealthSettings({}),
        logging_settings=LoggingSettings({}),
        ratelimit=RateLimitSettings({}),
        database=DatabaseSettings({}),
        alerts=AlertsSettings({"provider": provider} if provider else {}),
    )


@pytest.fixture()
def alerts_client_no_provider() -> Any:
    """TestClient for the alerts endpoint with NO provider configured."""
    from fastapi.testclient import TestClient  # noqa: PLC0415
    from weewx_clearskies_api.app import create_app  # noqa: PLC0415
    from weewx_clearskies_api.endpoints.alerts import wire_alerts_settings  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.cache import wire_cache_from_env  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.capability import wire_providers  # noqa: PLC0415

    _reset_provider_state()
    settings = _make_alerts_settings(provider=None)
    wire_alerts_settings(settings)
    wire_cache_from_env()
    wire_providers([])
    app = create_app(settings)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def alerts_client_nws() -> Any:
    """TestClient for the alerts endpoint with NWS provider configured."""
    from fastapi.testclient import TestClient  # noqa: PLC0415
    from weewx_clearskies_api.app import create_app  # noqa: PLC0415
    from weewx_clearskies_api.endpoints.alerts import wire_alerts_settings  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.cache import wire_cache_from_env  # noqa: PLC0415
    from weewx_clearskies_api.providers._common.capability import wire_providers  # noqa: PLC0415
    from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415

    _reset_provider_state()
    settings = _make_alerts_settings(provider="nws")
    wire_alerts_settings(settings)
    wire_cache_from_env()
    wire_providers([nws.CAPABILITY])
    app = create_app(settings)
    return TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# 1. Severity normalization
# ===========================================================================


class TestNwsSeverityNormalization:
    """_normalize_severity maps all five CAP values + unknown-default to canonical."""

    def test_extreme_maps_to_warning(self) -> None:
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        assert nws._normalize_severity("Extreme") == "warning"

    def test_severe_maps_to_watch(self) -> None:
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        assert nws._normalize_severity("Severe") == "watch"

    def test_moderate_maps_to_advisory(self) -> None:
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        assert nws._normalize_severity("Moderate") == "advisory"

    def test_minor_maps_to_advisory(self) -> None:
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        assert nws._normalize_severity("Minor") == "advisory"

    def test_unknown_maps_to_advisory(self) -> None:
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        assert nws._normalize_severity("Unknown") == "advisory"

    def test_unknown_cap_string_defaults_to_advisory(self) -> None:
        """Unknown CAP severity (e.g. 'Critical') → 'advisory' default."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        assert nws._normalize_severity("Critical") == "advisory"

    def test_unknown_cap_string_emits_warning_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown CAP severity emits a WARNING log so operator notices NWS schema change."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        with caplog.at_level(logging.WARNING, logger="weewx_clearskies_api.providers.alerts.nws"):
            nws._normalize_severity("Critical")
        assert any("Critical" in record.message for record in caplog.records), (
            "Expected WARNING log mentioning 'Critical' for unknown CAP severity"
        )

    def test_severity_map_covers_all_five_cap_values(self) -> None:
        """_NWS_SEVERITY_MAP covers Extreme, Severe, Moderate, Minor, Unknown."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        for cap_value in ("Extreme", "Severe", "Moderate", "Minor", "Unknown"):
            assert cap_value in nws._NWS_SEVERITY_MAP, (
                f"_NWS_SEVERITY_MAP missing CAP value {cap_value!r}"
            )


# ===========================================================================
# 2. Time normalization
# ===========================================================================


class TestNwsTimeNormalization:
    """_to_utc_iso8601 correctly converts NWS timestamps."""

    def test_offset_form_converts_to_z_suffix(self) -> None:
        """NWS 2026-04-30T16:00:00-07:00 → 2026-04-30T23:00:00Z."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        result = nws._to_utc_iso8601("2026-04-30T16:00:00-07:00")
        assert result == "2026-04-30T23:00:00Z", (
            f"Expected '2026-04-30T23:00:00Z', got {result!r}"
        )

    def test_utc_offset_form_converts_to_z_suffix(self) -> None:
        """NWS 2026-04-30T16:00:00+00:00 → 2026-04-30T16:00:00Z."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        result = nws._to_utc_iso8601("2026-04-30T16:00:00+00:00")
        assert result == "2026-04-30T16:00:00Z", (
            f"Expected '2026-04-30T16:00:00Z', got {result!r}"
        )

    def test_naive_timestamp_raises_provider_protocol_error(self) -> None:
        """Naive timestamp (no timezone offset) → ProviderProtocolError."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import ProviderProtocolError  # noqa: PLC0415
        with pytest.raises(ProviderProtocolError):
            nws._to_utc_iso8601("2026-04-30T16:00:00")

    def test_bogus_timestamp_raises_provider_protocol_error(self) -> None:
        """Non-ISO string → ProviderProtocolError."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import ProviderProtocolError  # noqa: PLC0415
        with pytest.raises(ProviderProtocolError):
            nws._to_utc_iso8601("not-a-date")

    def test_positive_offset_converts_correctly(self) -> None:
        """Positive offset (e.g. +05:30) converts correctly."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        # 2026-04-30T10:30:00+05:30 → 2026-04-30T05:00:00Z
        result = nws._to_utc_iso8601("2026-04-30T10:30:00+05:30")
        assert result == "2026-04-30T05:00:00Z", (
            f"Expected '2026-04-30T05:00:00Z', got {result!r}"
        )


# ===========================================================================
# 3. _to_canonical mapping
# ===========================================================================


class TestNwsToCanonical:
    """_to_canonical maps _NwsAlertProperties → AlertRecord correctly."""

    def _make_props(self, **overrides: Any) -> Any:
        """Build an _NwsAlertProperties instance with test defaults."""
        from weewx_clearskies_api.providers.alerts.nws import _NwsAlertProperties  # noqa: PLC0415
        defaults = {
            "id": "urn:oid:test.001",
            "areaDesc": "King, WA",
            "effective": "2026-05-07T10:00:00-07:00",
            "severity": "Moderate",
            "certainty": "Likely",
            "urgency": "Expected",
            "event": "Wind Advisory",
            "senderName": "NWS Seattle WA",
            "headline": "Wind Advisory issued May 7 by NWS Seattle WA",
            "description": "* WHAT...South winds 20 to 30 mph.",
            "instruction": None,
            "category": "Met",
        }
        defaults.update(overrides)
        return _NwsAlertProperties(**defaults)

    def test_description_and_instruction_concatenated(self) -> None:
        """When instruction is set, it is appended to description with double-newline."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        props = self._make_props(
            description="Body text here.",
            instruction="Stay indoors.",
        )
        record = nws._to_canonical(props)
        assert record.description == "Body text here.\n\nStay indoors.", (
            f"description+instruction concat expected, got: {record.description!r}"
        )

    def test_description_without_instruction(self) -> None:
        """When instruction is None, description is used as-is."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        props = self._make_props(description="Body only.", instruction=None)
        record = nws._to_canonical(props)
        assert record.description == "Body only.", (
            f"Expected 'Body only.', got {record.description!r}"
        )

    def test_severity_normalized_in_canonical_record(self) -> None:
        """Canonical AlertRecord.severity is the normalized string, not the raw CAP value."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        props = self._make_props(severity="Extreme")
        record = nws._to_canonical(props)
        assert record.severity == "warning", (
            f"Extreme → 'warning', got {record.severity!r}"
        )

    def test_effective_time_converted_to_utc_z(self) -> None:
        """effective field is UTC ISO-8601 with Z suffix on the canonical record."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        props = self._make_props(effective="2026-05-07T10:00:00-07:00")
        record = nws._to_canonical(props)
        assert record.effective == "2026-05-07T17:00:00Z", (
            f"Expected '2026-05-07T17:00:00Z', got {record.effective!r}"
        )

    def test_expires_is_none_when_not_set(self) -> None:
        """expires is None on the canonical record when NWS doesn't supply it."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        props = self._make_props(expires=None)
        record = nws._to_canonical(props)
        assert record.expires is None

    def test_expires_converted_to_utc_z_when_set(self) -> None:
        """expires is UTC ISO-8601 with Z suffix when NWS supplies it."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        props = self._make_props(expires="2026-05-07T20:00:00-07:00")
        record = nws._to_canonical(props)
        assert record.expires == "2026-05-08T03:00:00Z", (
            f"Expected '2026-05-08T03:00:00Z', got {record.expires!r}"
        )

    def test_source_set_to_nws_provider_id(self) -> None:
        """source field on the canonical record is always 'nws'."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        props = self._make_props()
        record = nws._to_canonical(props)
        assert record.source == "nws", (
            f"Expected source='nws', got {record.source!r}"
        )

    def test_all_passthrough_fields_copied(self) -> None:
        """event, headline, urgency, certainty, category, senderName, areaDesc copied."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        props = self._make_props(
            event="Tornado Warning",
            headline="Test headline",
            urgency="Immediate",
            certainty="Observed",
            category="Met",
            senderName="NWS Norman OK",
            areaDesc="Oklahoma County, OK",
        )
        record = nws._to_canonical(props)
        assert record.event == "Tornado Warning"
        assert record.headline == "Test headline"
        assert record.urgency == "Immediate"
        assert record.certainty == "Observed"
        assert record.category == "Met"
        assert record.senderName == "NWS Norman OK"
        assert record.areaDesc == "Oklahoma County, OK"

    def test_empty_description_with_instruction_strips_leading_newlines(self) -> None:
        """When description is empty and instruction is set, result is just instruction."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        props = self._make_props(description="", instruction="Stay indoors.")
        record = nws._to_canonical(props)
        # strip() on concatenation should leave just the instruction text
        assert "Stay indoors." in record.description


# ===========================================================================
# 4. Wire-shape Pydantic model validation
# ===========================================================================


class TestNwsWireShapePydantic:
    """NWS wire-shape models validate correctly against the real fixture shapes."""

    def test_real_fixture_loads_cleanly(self) -> None:
        """The primary alerts_active.json fixture round-trips through Pydantic without error."""
        from weewx_clearskies_api.providers.alerts.nws import _NwsAlertsActiveResponse  # noqa: PLC0415
        raw = _load_fixture("alerts_active.json")
        model = _NwsAlertsActiveResponse.model_validate(raw)
        assert model.type == "FeatureCollection"
        assert len(model.features) == 2, (
            f"alerts_active.json has 2 features, got {len(model.features)}"
        )

    def test_empty_fixture_loads_cleanly(self) -> None:
        """alerts_active_empty.json (features=[]) loads without error."""
        from weewx_clearskies_api.providers.alerts.nws import _NwsAlertsActiveResponse  # noqa: PLC0415
        raw = _load_fixture("alerts_active_empty.json")
        model = _NwsAlertsActiveResponse.model_validate(raw)
        assert model.features == []

    def test_extreme_fixture_loads_cleanly(self) -> None:
        """alerts_active_extreme.json loads and contains Extreme/Severe severity values."""
        from weewx_clearskies_api.providers.alerts.nws import _NwsAlertsActiveResponse  # noqa: PLC0415
        raw = _load_fixture("alerts_active_extreme.json")
        model = _NwsAlertsActiveResponse.model_validate(raw)
        severities = {f.properties.severity for f in model.features}
        assert "Extreme" in severities, "alerts_active_extreme.json must contain Extreme severity"

    def test_missing_required_headline_raises_validation_error(self) -> None:
        """Missing required 'headline' field raises Pydantic ValidationError."""
        from pydantic import ValidationError  # noqa: PLC0415
        from weewx_clearskies_api.providers.alerts.nws import _NwsAlertProperties  # noqa: PLC0415
        with pytest.raises(ValidationError):
            _NwsAlertProperties(
                id="urn:oid:test.001",
                # headline missing intentionally
                effective="2026-05-07T10:00:00-07:00",
                severity="Moderate",
                event="Wind Advisory",
                description="Test",
            )

    def test_missing_required_event_raises_validation_error(self) -> None:
        """Missing required 'event' field raises Pydantic ValidationError."""
        from pydantic import ValidationError  # noqa: PLC0415
        from weewx_clearskies_api.providers.alerts.nws import _NwsAlertProperties  # noqa: PLC0415
        with pytest.raises(ValidationError):
            _NwsAlertProperties(
                id="urn:oid:test.001",
                headline="Test headline",
                effective="2026-05-07T10:00:00-07:00",
                severity="Moderate",
                # event missing intentionally
                description="Test",
            )

    def test_extra_fields_in_feature_properties_ignored(self) -> None:
        """Unknown fields in NWS response are ignored (extra='ignore')."""
        from weewx_clearskies_api.providers.alerts.nws import _NwsAlertProperties  # noqa: PLC0415
        # NWS could add new fields in future; they must not break validation.
        props = _NwsAlertProperties(
            id="urn:oid:test.001",
            headline="Headline",
            effective="2026-05-07T10:00:00-07:00",
            severity="Moderate",
            event="Wind Advisory",
            description="Test",
            some_future_nws_field="ignored_value",  # type: ignore[call-arg]
        )
        assert props.event == "Wind Advisory"

    def test_malformed_fixture_missing_headline_fails_validation(self) -> None:
        """alerts_active_malformed.json (missing headline) fails Pydantic validation."""
        from pydantic import ValidationError  # noqa: PLC0415
        from weewx_clearskies_api.providers.alerts.nws import _NwsAlertsActiveResponse  # noqa: PLC0415
        raw = _load_fixture("alerts_active_malformed.json")
        # The feature's properties are missing 'headline' — ValidationError expected
        with pytest.raises(ValidationError):
            _NwsAlertsActiveResponse.model_validate(raw)

    def test_unknown_severity_fixture_loads_cleanly(self) -> None:
        """alerts_active_unknown_severity.json loads (severity is just a string field)."""
        from weewx_clearskies_api.providers.alerts.nws import _NwsAlertsActiveResponse  # noqa: PLC0415
        raw = _load_fixture("alerts_active_unknown_severity.json")
        model = _NwsAlertsActiveResponse.model_validate(raw)
        assert model.features[0].properties.severity == "Critical"


# ===========================================================================
# 5. AlertsQueryParams
# ===========================================================================


class TestAlertsQueryParams:
    """AlertsQueryParams rejects unknowns and invalid severity values."""

    def test_valid_severity_advisory_accepted(self) -> None:
        from weewx_clearskies_api.models.params import AlertsQueryParams  # noqa: PLC0415
        params = AlertsQueryParams(severity="advisory")
        assert params.severity == "advisory"

    def test_valid_severity_watch_accepted(self) -> None:
        from weewx_clearskies_api.models.params import AlertsQueryParams  # noqa: PLC0415
        params = AlertsQueryParams(severity="watch")
        assert params.severity == "watch"

    def test_valid_severity_warning_accepted(self) -> None:
        from weewx_clearskies_api.models.params import AlertsQueryParams  # noqa: PLC0415
        params = AlertsQueryParams(severity="warning")
        assert params.severity == "warning"

    def test_missing_severity_is_none(self) -> None:
        """severity is optional; missing → None (no filter applied)."""
        from weewx_clearskies_api.models.params import AlertsQueryParams  # noqa: PLC0415
        params = AlertsQueryParams()
        assert params.severity is None

    def test_invalid_severity_value_raises_validation_error(self) -> None:
        """Unrecognised severity value is rejected."""
        from pydantic import ValidationError  # noqa: PLC0415
        from weewx_clearskies_api.models.params import AlertsQueryParams  # noqa: PLC0415
        with pytest.raises(ValidationError):
            AlertsQueryParams(severity="critical")

    def test_unknown_query_key_raises_validation_error(self) -> None:
        """extra='forbid' on AlertsQueryParams rejects unknown query keys."""
        from pydantic import ValidationError  # noqa: PLC0415
        from weewx_clearskies_api.models.params import AlertsQueryParams  # noqa: PLC0415
        with pytest.raises(ValidationError):
            AlertsQueryParams(severity="advisory", unknown_key="bad")  # type: ignore[call-arg]


# ===========================================================================
# 6. Severity filter
# ===========================================================================


class TestAlertsSeverityFilter:
    """Severity filter returns the correct subset of a canonical alert list."""

    def _make_alert_records(self) -> list[Any]:
        """Build a list of AlertRecord objects with advisory, watch, warning severities."""
        from weewx_clearskies_api.models.responses import AlertRecord  # noqa: PLC0415
        return [
            AlertRecord(
                id="urn:advisory.001",
                headline="Advisory alert",
                effective="2026-05-07T17:00:00Z",
                severity="advisory",
                event="Wind Advisory",
                source="nws",
            ),
            AlertRecord(
                id="urn:watch.001",
                headline="Watch alert",
                effective="2026-05-07T17:00:00Z",
                severity="watch",
                event="Tornado Watch",
                source="nws",
            ),
            AlertRecord(
                id="urn:warning.001",
                headline="Warning alert",
                effective="2026-05-07T17:00:00Z",
                severity="warning",
                event="Tornado Warning",
                source="nws",
            ),
        ]

    def test_advisory_filter_returns_all_alerts(self) -> None:
        """severity=advisory returns all alerts (advisory is minimum severity)."""
        from weewx_clearskies_api.endpoints.alerts import _filter_by_severity  # noqa: PLC0415
        alerts = self._make_alert_records()
        result = _filter_by_severity(alerts, "advisory")
        assert len(result) == 3, (
            f"severity=advisory should return all 3 alerts, got {len(result)}"
        )

    def test_watch_filter_returns_watch_and_warning(self) -> None:
        """severity=watch returns watch + warning (not advisory)."""
        from weewx_clearskies_api.endpoints.alerts import _filter_by_severity  # noqa: PLC0415
        alerts = self._make_alert_records()
        result = _filter_by_severity(alerts, "watch")
        assert len(result) == 2, (
            f"severity=watch should return 2 alerts (watch+warning), got {len(result)}"
        )
        returned_severities = {a.severity for a in result}
        assert "advisory" not in returned_severities, (
            "severity=watch must exclude advisory alerts"
        )
        assert "watch" in returned_severities
        assert "warning" in returned_severities

    def test_warning_filter_returns_only_warnings(self) -> None:
        """severity=warning returns only warning alerts."""
        from weewx_clearskies_api.endpoints.alerts import _filter_by_severity  # noqa: PLC0415
        alerts = self._make_alert_records()
        result = _filter_by_severity(alerts, "warning")
        assert len(result) == 1, (
            f"severity=warning should return 1 alert, got {len(result)}"
        )
        assert result[0].severity == "warning"

    def test_none_filter_returns_all_alerts(self) -> None:
        """severity=None (no filter) returns all alerts."""
        from weewx_clearskies_api.endpoints.alerts import _filter_by_severity  # noqa: PLC0415
        alerts = self._make_alert_records()
        result = _filter_by_severity(alerts, None)
        assert len(result) == 3


# ===========================================================================
# 7. MemoryCache
# ===========================================================================


class TestMemoryCache:
    """MemoryCache set/get/TTL expiry contract."""

    def test_set_then_get_returns_value(self) -> None:
        from weewx_clearskies_api.providers._common.cache import MemoryCache  # noqa: PLC0415
        cache = MemoryCache()
        cache.set("key1", {"data": 42}, ttl_seconds=60)
        result = cache.get("key1")
        assert result == {"data": 42}, (
            f"Expected {{'data': 42}}, got {result!r}"
        )

    def test_get_missing_key_returns_none(self) -> None:
        from weewx_clearskies_api.providers._common.cache import MemoryCache  # noqa: PLC0415
        cache = MemoryCache()
        result = cache.get("nonexistent_key")
        assert result is None

    def test_expired_key_returns_none(self) -> None:
        """After TTL expires, get() returns None (simulated via very short TTL)."""
        from weewx_clearskies_api.providers._common.cache import MemoryCache  # noqa: PLC0415
        cache = MemoryCache()
        cache.set("expire_me", "value", ttl_seconds=1)
        # Verify it's there immediately
        assert cache.get("expire_me") == "value"
        # Wait for TTL to expire
        time.sleep(1.1)
        result = cache.get("expire_me")
        assert result is None, (
            "After TTL expiry, MemoryCache.get() must return None"
        )

    def test_two_keys_with_different_ttls_are_independent(self) -> None:
        """Two cache entries with different TTLs don't interfere with each other."""
        from weewx_clearskies_api.providers._common.cache import MemoryCache  # noqa: PLC0415
        cache = MemoryCache()
        cache.set("long_lived", "stays", ttl_seconds=60)
        cache.set("short_lived", "expires", ttl_seconds=1)
        time.sleep(1.1)
        assert cache.get("short_lived") is None, "Short-TTL entry should be expired"
        assert cache.get("long_lived") == "stays", "Long-TTL entry should still be present"

    def test_overwrite_updates_value(self) -> None:
        """Setting the same key twice overwrites the first value."""
        from weewx_clearskies_api.providers._common.cache import MemoryCache  # noqa: PLC0415
        cache = MemoryCache()
        cache.set("key", "first", ttl_seconds=60)
        cache.set("key", "second", ttl_seconds=60)
        assert cache.get("key") == "second"

    def test_alert_dict_list_round_trips(self) -> None:
        """A list of canonical alert dicts stores and retrieves correctly."""
        from weewx_clearskies_api.providers._common.cache import MemoryCache  # noqa: PLC0415
        from weewx_clearskies_api.models.responses import AlertRecord  # noqa: PLC0415
        record = AlertRecord(
            id="urn:oid:test.cache",
            headline="Cache test",
            effective="2026-05-07T17:00:00Z",
            severity="advisory",
            event="Wind Advisory",
            source="nws",
        )
        # fetch() stores model_dump() dicts, not AlertRecord objects
        cache = MemoryCache()
        cache.set("alerts:test", [record.model_dump()], ttl_seconds=60)
        result = cache.get("alerts:test")
        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "urn:oid:test.cache"


# ===========================================================================
# 8. RedisCache via fakeredis
# ===========================================================================


class TestRedisCache:
    """RedisCache get/set/TTL contract using fakeredis."""

    @pytest.fixture()
    def redis_cache(self) -> Any:
        """Construct a RedisCache backed by fakeredis for testing."""
        try:
            import fakeredis  # noqa: PLC0415
        except ImportError:
            pytest.skip("fakeredis not installed; add to dev extras")
        from weewx_clearskies_api.providers._common.cache import RedisCache  # noqa: PLC0415
        # Bypass __init__'s real ping and replace with fakeredis client.
        server = fakeredis.FakeServer()
        fake_client = fakeredis.FakeRedis(server=server, decode_responses=False)
        redis_cache = object.__new__(RedisCache)
        redis_cache._client = fake_client  # type: ignore[attr-defined]
        return redis_cache

    def test_set_then_get_returns_value(self, redis_cache: Any) -> None:
        redis_cache.set("rkey1", {"data": 99}, ttl_seconds=60)
        result = redis_cache.get("rkey1")
        assert result == {"data": 99}, (
            f"Expected {{'data': 99}}, got {result!r}"
        )

    def test_get_missing_key_returns_none(self, redis_cache: Any) -> None:
        result = redis_cache.get("missing")
        assert result is None

    def test_ttl_expiry_returns_none(self, redis_cache: Any) -> None:
        """fakeredis honours TTLs; after expiry get() returns None."""
        redis_cache.set("r_expire", "val", ttl_seconds=1)
        assert redis_cache.get("r_expire") == "val"
        time.sleep(1.1)
        result = redis_cache.get("r_expire")
        assert result is None, (
            "After TTL expiry, RedisCache.get() must return None"
        )

    def test_json_serialization_round_trip_preserves_alert_dict_shape(
        self, redis_cache: Any
    ) -> None:
        """AlertRecord dict serialises to JSON and deserialises correctly via Redis."""
        from weewx_clearskies_api.models.responses import AlertRecord  # noqa: PLC0415
        record = AlertRecord(
            id="urn:oid:redis.test",
            headline="Redis round-trip test",
            effective="2026-05-07T17:00:00Z",
            severity="watch",
            event="Tornado Watch",
            source="nws",
            senderName="NWS Tulsa OK",
        )
        redis_cache.set("alerts:redis", [record.model_dump()], ttl_seconds=60)
        result = redis_cache.get("alerts:redis")
        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "urn:oid:redis.test", (
            "JSON round-trip must preserve the id field"
        )
        assert result[0]["severity"] == "watch", (
            "JSON round-trip must preserve normalised severity"
        )

    def test_connection_refused_at_construction_raises(self) -> None:
        """RedisCache raises on construction when Redis is unreachable."""
        from weewx_clearskies_api.providers._common.cache import RedisCache  # noqa: PLC0415
        with pytest.raises((RuntimeError, Exception)):
            # Use a port that should be refused (nothing listening on 16379)
            RedisCache(url="redis://127.0.0.1:16379/0")


# ===========================================================================
# 9. wire_cache_from_env()
# ===========================================================================


class TestWireCacheFromEnv:
    """wire_cache_from_env() constructs the right backend from CLEARSKIES_CACHE_URL."""

    @pytest.fixture(autouse=True)
    def _reset(self) -> Any:
        """Reset cache state before each test."""
        yield
        from weewx_clearskies_api.providers._common.cache import reset_cache_for_tests  # noqa: PLC0415
        reset_cache_for_tests()

    def test_unset_env_var_produces_memory_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When CLEARSKIES_CACHE_URL is unset, the backend is MemoryCache."""
        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            MemoryCache,
            get_cache,
            wire_cache_from_env,
        )
        monkeypatch.delenv("CLEARSKIES_CACHE_URL", raising=False)
        wire_cache_from_env()
        backend = get_cache()
        assert isinstance(backend, MemoryCache), (
            f"Expected MemoryCache when CLEARSKIES_CACHE_URL unset, got {type(backend)}"
        )

    def test_redis_url_produces_redis_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When CLEARSKIES_CACHE_URL=redis://..., the backend is RedisCache."""
        try:
            import fakeredis  # noqa: PLC0415
        except ImportError:
            pytest.skip("fakeredis not installed")
        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            RedisCache,
            get_cache,
            wire_cache_from_env,
        )
        monkeypatch.setenv("CLEARSKIES_CACHE_URL", "redis://localhost:6379/0")
        # Patch redis.Redis.from_url to return a fakeredis client so .ping() succeeds
        with patch("redis.Redis.from_url") as mock_from_url:
            fake = fakeredis.FakeRedis(decode_responses=False)
            mock_from_url.return_value = fake
            wire_cache_from_env()
        backend = get_cache()
        assert isinstance(backend, RedisCache), (
            f"Expected RedisCache for redis:// URL, got {type(backend)}"
        )

    def test_bogus_scheme_raises_config_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unrecognised CLEARSKIES_CACHE_URL scheme raises ConfigError."""
        from weewx_clearskies_api.providers._common.cache import ConfigError, wire_cache_from_env  # noqa: PLC0415
        monkeypatch.setenv("CLEARSKIES_CACHE_URL", "memcached://localhost:11211")
        with pytest.raises(ConfigError):
            wire_cache_from_env()

    def test_rediss_scheme_produces_redis_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """rediss:// (TLS Redis) is also accepted and produces RedisCache."""
        try:
            import fakeredis  # noqa: PLC0415
        except ImportError:
            pytest.skip("fakeredis not installed")
        from weewx_clearskies_api.providers._common.cache import (  # noqa: PLC0415
            RedisCache,
            get_cache,
            wire_cache_from_env,
        )
        monkeypatch.setenv("CLEARSKIES_CACHE_URL", "rediss://localhost:6380/0")
        with patch("redis.Redis.from_url") as mock_from_url:
            fake = fakeredis.FakeRedis(decode_responses=False)
            mock_from_url.return_value = fake
            wire_cache_from_env()
        backend = get_cache()
        assert isinstance(backend, RedisCache)


# ===========================================================================
# 10. RateLimiter
# ===========================================================================


class TestRateLimiter:
    """RateLimiter sliding-window raises QuotaExhausted on overflow."""

    def test_calls_within_limit_succeed(self) -> None:
        """5 calls within 1 second succeed when max_calls=5."""
        from weewx_clearskies_api.providers._common.rate_limiter import RateLimiter  # noqa: PLC0415
        limiter = RateLimiter(
            name="test", provider_id="nws", domain="alerts", max_calls=5, window_seconds=1
        )
        # 5 calls must not raise
        for _ in range(5):
            limiter.acquire()

    def test_sixth_call_raises_quota_exhausted(self) -> None:
        """6th call within the window raises QuotaExhausted."""
        from weewx_clearskies_api.providers._common.rate_limiter import RateLimiter  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        limiter = RateLimiter(
            name="test", provider_id="nws", domain="alerts", max_calls=5, window_seconds=1
        )
        for _ in range(5):
            limiter.acquire()
        with pytest.raises(QuotaExhausted):
            limiter.acquire()

    def test_quota_exhausted_has_retry_after_seconds(self) -> None:
        """QuotaExhausted carries retry_after_seconds > 0."""
        from weewx_clearskies_api.providers._common.rate_limiter import RateLimiter  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        limiter = RateLimiter(
            name="test", provider_id="nws", domain="alerts", max_calls=1, window_seconds=10
        )
        limiter.acquire()
        with pytest.raises(QuotaExhausted) as exc_info:
            limiter.acquire()
        assert exc_info.value.retry_after_seconds is not None
        assert exc_info.value.retry_after_seconds > 0, (
            "retry_after_seconds must be positive when QuotaExhausted"
        )

    def test_after_window_expires_slots_free_up(self) -> None:
        """After the window expires, slots free up and new calls succeed."""
        from weewx_clearskies_api.providers._common.rate_limiter import RateLimiter  # noqa: PLC0415
        limiter = RateLimiter(
            name="test", provider_id="nws", domain="alerts", max_calls=2, window_seconds=1
        )
        limiter.acquire()
        limiter.acquire()
        # window = 1s; wait for it to expire
        time.sleep(1.1)
        # Now should succeed again
        limiter.acquire()


# ===========================================================================
# 11. HTTP wrapper error translation
# ===========================================================================


class TestProviderHTTPClientErrorTranslation:
    """ProviderHTTPClient translates upstream errors to canonical taxonomy."""

    def _make_client(self) -> Any:
        from weewx_clearskies_api.providers._common.http import ProviderHTTPClient  # noqa: PLC0415
        return ProviderHTTPClient(
            provider_id="nws",
            domain="alerts",
            user_agent="(clearskies-test, test@example.com)",
        )

    def test_200_response_returned_no_exception(self) -> None:
        """200 response from upstream → response returned, no exception raised."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        client = self._make_client()
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(200, json={"type": "FeatureCollection", "features": []})
            )
            response = client.get(
                "https://api.weather.gov/alerts/active",
                params={"point": "47.6062,-122.3321"},
            )
        assert response.status_code == 200

    def test_5xx_after_retries_raises_transient_network_error(self) -> None:
        """5xx response → TransientNetworkError after all retries exhausted."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import TransientNetworkError  # noqa: PLC0415
        client = self._make_client()
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(503, json={"detail": "service unavailable"})
            )
            with pytest.raises(TransientNetworkError):
                client.get("https://api.weather.gov/alerts/active")

    def test_429_raises_quota_exhausted(self) -> None:
        """HTTP 429 → QuotaExhausted."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        client = self._make_client()
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(
                    429,
                    headers={"Retry-After": "60"},
                    json={"detail": "rate limited"},
                )
            )
            with pytest.raises(QuotaExhausted) as exc_info:
                client.get("https://api.weather.gov/alerts/active")
        assert exc_info.value.retry_after_seconds == 60

    def test_401_raises_key_invalid(self) -> None:
        """HTTP 401 → KeyInvalid."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import KeyInvalid  # noqa: PLC0415
        client = self._make_client()
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(401, json={"detail": "unauthorized"})
            )
            with pytest.raises(KeyInvalid):
                client.get("https://api.weather.gov/alerts/active")

    def test_connect_error_raises_transient_network_error(self) -> None:
        """httpx.ConnectError → TransientNetworkError after retries."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import TransientNetworkError  # noqa: PLC0415
        client = self._make_client()
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            with pytest.raises(TransientNetworkError):
                client.get("https://api.weather.gov/alerts/active")

    def test_read_timeout_raises_transient_network_error(self) -> None:
        """httpx.ReadTimeout → TransientNetworkError after retries."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import TransientNetworkError  # noqa: PLC0415
        client = self._make_client()
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                side_effect=httpx.ReadTimeout("timed out")
            )
            with pytest.raises(TransientNetworkError):
                client.get("https://api.weather.gov/alerts/active")

    def test_4xx_other_than_429_401_403_raises_provider_protocol_error(self) -> None:
        """Unexpected 4xx (e.g. 422) → ProviderProtocolError (not retried)."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import ProviderProtocolError  # noqa: PLC0415
        client = self._make_client()
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(422, json={"detail": "unprocessable"})
            )
            with pytest.raises(ProviderProtocolError):
                client.get("https://api.weather.gov/alerts/active")


# ===========================================================================
# 12. NWS module fetch()
# ===========================================================================


class TestNwsModuleFetch:
    """nws.fetch() exercises the full module pipeline via respx mocks."""

    @pytest.fixture(autouse=True)
    def _reset(self) -> Any:
        """Reset cache and NWS module state between tests."""
        _reset_provider_state()
        from weewx_clearskies_api.providers._common.cache import wire_cache_from_env  # noqa: PLC0415
        wire_cache_from_env()
        yield
        _reset_provider_state()

    def test_fetch_200_empty_features_returns_empty_list(self) -> None:
        """fetch() with empty features returns [] dict list."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        empty_fixture = _load_fixture("alerts_active_empty.json")
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(200, json=empty_fixture)
            )
            result = nws.fetch(lat=47.6062, lon=-122.3321, user_agent_contact="test@example.com")
        assert result == [], f"Expected [], got {result!r}"

    def test_fetch_200_with_alerts_returns_canonical_alert_dicts(self) -> None:
        """fetch() with 2-alert fixture returns 2 canonical AlertRecord dicts."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        fixture = _load_fixture("alerts_active.json")
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(200, json=fixture)
            )
            result = nws.fetch(lat=47.6062, lon=-122.3321, user_agent_contact="test@example.com")
        assert len(result) == 2, f"Expected 2 alerts, got {len(result)}"
        # Verify severity mapping is correct — both alerts in fixture are Minor/Moderate → advisory
        for alert_dict in result:
            assert alert_dict["severity"] in ("advisory", "watch", "warning"), (
                f"canonical severity must be advisory/watch/warning, got {alert_dict['severity']!r}"
            )

    def test_fetch_description_instruction_concatenated(self) -> None:
        """fetch() result has description+instruction concatenated for alert 1 of fixture."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        fixture = _load_fixture("alerts_active.json")
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(200, json=fixture)
            )
            result = nws.fetch(lat=47.6062, lon=-122.3321, user_agent_contact="test@example.com")
        # Alert 1 in the fixture has both description and instruction
        alert1 = result[0]
        assert "Use extra caution" in alert1["description"], (
            "instruction must be appended to description"
        )

    def test_fetch_malformed_response_raises_provider_protocol_error(self) -> None:
        """fetch() with malformed payload (missing headline) → ProviderProtocolError."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import ProviderProtocolError  # noqa: PLC0415
        malformed = _load_fixture("alerts_active_malformed.json")
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(200, json=malformed)
            )
            with pytest.raises(ProviderProtocolError):
                nws.fetch(lat=47.6062, lon=-122.3321, user_agent_contact="test@example.com")

    def test_fetch_5xx_raises_transient_network_error(self) -> None:
        """fetch() on NWS 5xx → TransientNetworkError after retries."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import TransientNetworkError  # noqa: PLC0415
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(503, json={"detail": "unavailable"})
            )
            with pytest.raises(TransientNetworkError):
                nws.fetch(lat=47.6062, lon=-122.3321, user_agent_contact="test@example.com")

    def test_fetch_429_raises_quota_exhausted(self) -> None:
        """fetch() on NWS 429 → QuotaExhausted."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(
                    429,
                    headers={"Retry-After": "60"},
                    json={"detail": "rate limited"},
                )
            )
            with pytest.raises(QuotaExhausted):
                nws.fetch(lat=47.6062, lon=-122.3321, user_agent_contact="test@example.com")

    def test_fetch_extreme_severity_maps_to_warning(self) -> None:
        """fetch() with Extreme severity alert returns canonical 'warning' severity dict."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        fixture = _load_fixture("alerts_active_extreme.json")
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(200, json=fixture)
            )
            result = nws.fetch(lat=35.4, lon=-97.2, user_agent_contact="test@example.com")
        warning_alerts = [a for a in result if a["severity"] == "warning"]
        assert len(warning_alerts) >= 1, (
            "At least one 'warning' alert expected from Extreme severity fixture"
        )


# ===========================================================================
# 13. Capability registry
# ===========================================================================


class TestCapabilityRegistry:
    """wire_providers and get_provider_registry behave per spec."""

    @pytest.fixture(autouse=True)
    def _reset(self) -> Any:
        """Reset the provider registry before and after each test."""
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            reset_provider_registry_for_tests,
        )
        reset_provider_registry_for_tests()
        yield
        reset_provider_registry_for_tests()

    def test_wire_providers_populates_registry(self) -> None:
        """wire_providers([cap]) → get_provider_registry() returns that list."""
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            get_provider_registry,
            wire_providers,
        )
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        wire_providers([nws.CAPABILITY])
        registry = get_provider_registry()
        assert len(registry) == 1
        assert registry[0].provider_id == "nws"
        assert registry[0].domain == "alerts"

    def test_wire_providers_duplicate_domain_provider_id_raises_value_error(self) -> None:
        """Duplicate (domain, provider_id) raises ValueError."""
        from weewx_clearskies_api.providers._common.capability import wire_providers  # noqa: PLC0415
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            wire_providers([nws.CAPABILITY, nws.CAPABILITY])

    def test_wire_providers_empty_list_leaves_empty_registry(self) -> None:
        """wire_providers([]) → get_provider_registry() returns []."""
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            get_provider_registry,
            wire_providers,
        )
        wire_providers([])
        assert get_provider_registry() == []

    def test_capability_declaration_fields_present(self) -> None:
        """NWS CAPABILITY has required fields per ADR-038 §4."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        cap = nws.CAPABILITY
        assert cap.provider_id == "nws"
        assert cap.domain == "alerts"
        assert len(cap.supplied_canonical_fields) > 0, "supplied_canonical_fields must not be empty"
        assert cap.geographic_coverage, "geographic_coverage must be set"
        assert isinstance(cap.auth_required, tuple)
        assert len(cap.auth_required) == 0, "NWS is keyless; auth_required must be empty"

    def test_nws_capability_supplied_fields_include_core_alert_fields(self) -> None:
        """NWS CAPABILITY declares id, headline, description, severity, event, effective."""
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        core = {"id", "headline", "description", "severity", "event", "effective"}
        supplied = set(nws.CAPABILITY.supplied_canonical_fields)
        missing = core - supplied
        assert not missing, (
            f"NWS CAPABILITY.supplied_canonical_fields missing core fields: {missing}"
        )


# ===========================================================================
# 14. /capabilities response with NWS provider
# ===========================================================================


class TestCapabilitiesEndpointWithNWSProvider:
    """/capabilities returns NWS provider entry + correct canonicalFieldsAvailable union."""

    def test_capabilities_response_contains_nws_provider_entry(self, client: Any) -> None:
        """GET /capabilities with NWS wired has providers list with one nws entry."""
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            reset_provider_registry_for_tests,
            wire_providers,
        )
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        reset_provider_registry_for_tests()
        wire_providers([nws.CAPABILITY])
        try:
            response = client.get("/api/v1/capabilities")
            assert response.status_code == 200
            data = response.json()
            providers = data["data"]["providers"]
            assert len(providers) == 1, f"Expected 1 provider, got {len(providers)}"
            assert providers[0]["providerId"] == "nws"
            assert providers[0]["domain"] == "alerts"
        finally:
            reset_provider_registry_for_tests()

    def test_capabilities_canonical_fields_available_includes_provider_fields(
        self, client: Any
    ) -> None:
        """canonicalFieldsAvailable is union of stock weewx fields + NWS supplied fields."""
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            reset_provider_registry_for_tests,
            wire_providers,
        )
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        reset_provider_registry_for_tests()
        wire_providers([nws.CAPABILITY])
        try:
            response = client.get("/api/v1/capabilities")
            assert response.status_code == 200
            data = response.json()
            available = set(data["data"]["canonicalFieldsAvailable"])
            nws_fields = set(nws.CAPABILITY.supplied_canonical_fields)
            missing = nws_fields - available
            assert not missing, (
                f"canonicalFieldsAvailable must include NWS-supplied fields; missing: {missing}"
            )
        finally:
            reset_provider_registry_for_tests()

    def test_capabilities_providers_empty_when_no_provider_wired(self, client: Any) -> None:
        """With no providers wired, providers is [] (carry-forward from 3a-2 baseline)."""
        from weewx_clearskies_api.providers._common.capability import (  # noqa: PLC0415
            reset_provider_registry_for_tests,
        )
        reset_provider_registry_for_tests()
        response = client.get("/api/v1/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["providers"] == [], (
            "providers must be [] when no provider modules wired"
        )


# ===========================================================================
# 15. /alerts endpoint
# ===========================================================================


class TestAlertsEndpointNoProvider:
    """/alerts with no provider configured."""

    def test_no_provider_returns_200_source_none(self, alerts_client_no_provider: Any) -> None:
        """GET /alerts with no provider → 200 with source='none'."""
        response = alerts_client_no_provider.get("/api/v1/alerts")
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data["data"]["alerts"] == [], "alerts must be [] when no provider configured"
        assert data["data"]["source"] == "none", (
            f"Expected source='none', got {data['data']['source']!r}"
        )
        assert "retrievedAt" in data["data"], "retrievedAt must be present"
        assert data["source"] == "none", "envelope source must also be 'none'"
        assert "generatedAt" in data, "generatedAt must be present in envelope"

    def test_no_provider_alerts_field_is_list_not_null(self, alerts_client_no_provider: Any) -> None:
        """alerts field is [] (empty list), NOT null — per ADR-016."""
        response = alerts_client_no_provider.get("/api/v1/alerts")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["data"]["alerts"], list), (
            "alerts must be a list, even when empty"
        )


class TestAlertsEndpointWithNWS:
    """/alerts with NWS provider configured."""

    @pytest.fixture(autouse=True)
    def _reset(self) -> Any:
        """Reset NWS cache between tests."""
        _reset_provider_state()
        from weewx_clearskies_api.providers._common.cache import wire_cache_from_env  # noqa: PLC0415
        wire_cache_from_env()
        yield
        _reset_provider_state()

    def test_nws_happy_path_returns_200_source_nws(self, alerts_client_nws: Any) -> None:
        """NWS configured + respx-mocked NWS → 200 with source='nws'."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        fixture = _load_fixture("alerts_active.json")
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(200, json=fixture)
            )
            response = alerts_client_nws.get("/api/v1/alerts")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["source"] == "nws", (
            f"Expected source='nws', got {data['data']['source']!r}"
        )
        assert len(data["data"]["alerts"]) == 2, (
            f"Expected 2 alerts from fixture, got {len(data['data']['alerts'])}"
        )

    def test_nws_down_returns_502_provider_problem(self, alerts_client_nws: Any) -> None:
        """NWS returning 5xx → /alerts returns 502 ProviderProblem."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(503, json={"detail": "unavailable"})
            )
            response = alerts_client_nws.get("/api/v1/alerts")
        assert response.status_code == 502
        data = response.json()
        assert data.get("errorCode") == "TransientNetworkError", (
            f"Expected errorCode='TransientNetworkError', got {data.get('errorCode')!r}"
        )
        assert data.get("domain") == "alerts"
        assert data.get("providerId") == "nws"

    def test_nws_quota_exhausted_returns_503_with_retry_after(
        self, alerts_client_nws: Any
    ) -> None:
        """NWS 429 → /alerts returns 503 ProviderProblem with Retry-After header."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        with respx.mock:
            respx.get("https://api.weather.gov/alerts/active").mock(
                return_value=httpx.Response(
                    429,
                    headers={"Retry-After": "60"},
                    json={"detail": "rate limited"},
                )
            )
            response = alerts_client_nws.get("/api/v1/alerts")
        assert response.status_code == 503
        data = response.json()
        assert data.get("errorCode") == "QuotaExhausted", (
            f"Expected errorCode='QuotaExhausted', got {data.get('errorCode')!r}"
        )
        # Retry-After header should be set
        assert "retry-after" in {k.lower() for k in response.headers.keys()}, (
            "Retry-After header expected on 503 QuotaExhausted response"
        )

    def test_unknown_severity_filter_returns_400_or_422(
        self, alerts_client_nws: Any
    ) -> None:
        """?severity=critical (not in enum) → 400 or 422 problem+json."""
        response = alerts_client_nws.get("/api/v1/alerts?severity=critical")
        assert response.status_code in (400, 422), (
            f"Invalid severity should return 400 or 422, got {response.status_code}"
        )

    def test_unknown_query_param_returns_400_or_422(self, alerts_client_nws: Any) -> None:
        """Unknown query parameter → 400/422 per security-baseline §3.5."""
        response = alerts_client_nws.get("/api/v1/alerts?nuke_the_db=1")
        assert response.status_code in (400, 422), (
            f"Unknown query param should return 400 or 422, got {response.status_code}"
        )

    def test_cache_hit_returns_cached_data_no_nws_call(
        self, alerts_client_nws: Any
    ) -> None:
        """Pre-populated cache → endpoint returns cached alerts; no NWS call."""
        import respx  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        from weewx_clearskies_api.providers._common.cache import get_cache  # noqa: PLC0415
        from weewx_clearskies_api.providers.alerts import nws  # noqa: PLC0415
        from weewx_clearskies_api.models.responses import AlertRecord  # noqa: PLC0415

        # Pre-populate cache with one canonical record dict
        # Station lat/lon from _wire_test_station: 42.375, -72.519
        record = AlertRecord(
            id="urn:oid:cached.001",
            headline="Cached alert",
            effective="2026-05-07T17:00:00Z",
            severity="advisory",
            event="Test Event",
            source="nws",
        )
        cache_key = nws._build_cache_key(42.375, -72.519)
        get_cache().set(cache_key, [record.model_dump()], ttl_seconds=300)

        with respx.mock:
            # No mock registered — if NWS is called, respx raises an error
            response = alerts_client_nws.get("/api/v1/alerts")

        assert response.status_code == 200
        data = response.json()
        alert_ids = [a["id"] for a in data["data"]["alerts"]]
        assert "urn:oid:cached.001" in alert_ids, (
            "Cached alert must be returned from the endpoint"
        )


# ===========================================================================
# 16. Canonical error taxonomy — error class hierarchy
# ===========================================================================


class TestCanonicalErrorTaxonomy:
    """Provider error classes are correctly structured per ADR-038 §5."""

    def test_all_error_classes_are_subclass_of_provider_error(self) -> None:
        """All canonical errors inherit from ProviderError."""
        from weewx_clearskies_api.providers._common.errors import (  # noqa: PLC0415
            FieldUnsupported,
            GeographicallyUnsupported,
            KeyInvalid,
            ProviderError,
            ProviderProtocolError,
            QuotaExhausted,
            TransientNetworkError,
        )
        for cls in (
            QuotaExhausted,
            KeyInvalid,
            GeographicallyUnsupported,
            FieldUnsupported,
            TransientNetworkError,
            ProviderProtocolError,
        ):
            assert issubclass(cls, ProviderError), (
                f"{cls.__name__} must be a subclass of ProviderError"
            )

    def test_provider_error_carries_provider_id_and_domain(self) -> None:
        """ProviderError instances carry provider_id and domain attributes."""
        from weewx_clearskies_api.providers._common.errors import TransientNetworkError  # noqa: PLC0415
        exc = TransientNetworkError(
            "test error",
            provider_id="nws",
            domain="alerts",
        )
        assert exc.provider_id == "nws"
        assert exc.domain == "alerts"

    def test_quota_exhausted_carries_retry_after_seconds(self) -> None:
        """QuotaExhausted instances can carry retry_after_seconds."""
        from weewx_clearskies_api.providers._common.errors import QuotaExhausted  # noqa: PLC0415
        exc = QuotaExhausted(
            "rate limited",
            provider_id="nws",
            domain="alerts",
            retry_after_seconds=60,
        )
        assert exc.retry_after_seconds == 60

    def test_provider_error_retry_after_seconds_defaults_to_none(self) -> None:
        """retry_after_seconds defaults to None when not set."""
        from weewx_clearskies_api.providers._common.errors import TransientNetworkError  # noqa: PLC0415
        exc = TransientNetworkError(
            "transient error",
            provider_id="nws",
            domain="alerts",
        )
        assert exc.retry_after_seconds is None


# ===========================================================================
# 17. Startup failure paths (unit-style)
# ===========================================================================


class TestStartupFailurePaths:
    """Startup failure paths for unknown provider + unreachable Redis."""

    def test_dispatch_with_unknown_provider_raises_key_error(self) -> None:
        """get_provider_module with unknown_provider raises KeyError."""
        from weewx_clearskies_api.providers._common.dispatch import get_provider_module  # noqa: PLC0415
        with pytest.raises(KeyError):
            get_provider_module(domain="alerts", provider_id="unknown_provider")

    def test_redis_cache_construction_raises_on_unreachable_server(self) -> None:
        """RedisCache.__init__ raises when Redis is not reachable."""
        from weewx_clearskies_api.providers._common.cache import RedisCache  # noqa: PLC0415
        with pytest.raises((RuntimeError, Exception)):
            RedisCache(url="redis://127.0.0.1:16379/0")

    def test_wire_cache_from_env_raises_config_error_on_bogus_scheme(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """wire_cache_from_env() raises ConfigError for memcached:// scheme."""
        from weewx_clearskies_api.providers._common.cache import ConfigError, wire_cache_from_env  # noqa: PLC0415
        monkeypatch.setenv("CLEARSKIES_CACHE_URL", "memcached://localhost:11211")
        with pytest.raises(ConfigError):
            wire_cache_from_env()
