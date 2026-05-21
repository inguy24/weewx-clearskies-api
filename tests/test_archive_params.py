"""Unit tests for Pydantic parameter models used by the 6 DB endpoints.

Covers:
  - extra="forbid" rejection of unknown query keys (security-baseline §3.5).
  - Range validation (limit 1..10000, page ≥ 1, year ≥ 1900, month 1..12).
  - Mutual exclusion of cursor + page on /archive.
  - Period parsing for /records: ytd, all-time, valid 4-digit year, reject
    two-digit year, reject non-numeric, reject 1899.
  - Interval enum on /archive: raw, hour, day; reject unknown.

Models live in the endpoint modules:
  ArchiveParams  → endpoints/observations.py
  RecordsParams  → endpoints/records.py
  Path params are validated inline in endpoints/reports.py (int with min).

ADR references: ADR-018 (RFC 9457), security-baseline §3.5.
"""

from __future__ import annotations

import pytest


class TestArchiveParamsModel:
    """/archive query params: ArchiveParams Pydantic model."""

    def test_default_values_accepted(self) -> None:
        """Empty params dict → model fills in defaults without error."""
        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        params = ArchiveParams()
        assert params.interval == "raw"
        assert params.limit == 1000
        assert params.cursor is None
        assert params.page is None
        assert params.fields is None

    def test_valid_raw_interval_accepted(self) -> None:
        """interval=raw is a valid enum value."""
        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams(interval="raw")
        assert p.interval == "raw"

    def test_valid_hour_interval_accepted(self) -> None:
        """interval=hour is a valid enum value."""
        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams(interval="hour")
        assert p.interval == "hour"

    def test_valid_day_interval_accepted(self) -> None:
        """interval=day is a valid enum value."""
        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams(interval="day")
        assert p.interval == "day"

    def test_unknown_interval_rejected(self) -> None:
        """interval=week (not in enum) → ValidationError."""
        from pydantic import ValidationError

        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        with pytest.raises(ValidationError):
            ArchiveParams(interval="week")

    def test_limit_minimum_boundary_accepted(self) -> None:
        """limit=1 is the minimum allowed value."""
        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams(limit=1)
        assert p.limit == 1

    def test_limit_maximum_boundary_accepted(self) -> None:
        """limit=10000 is the maximum allowed value."""
        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams(limit=10000)
        assert p.limit == 10000

    def test_limit_below_minimum_rejected(self) -> None:
        """limit=0 is below the allowed minimum → ValidationError."""
        from pydantic import ValidationError

        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        with pytest.raises(ValidationError):
            ArchiveParams(limit=0)

    def test_limit_above_maximum_rejected(self) -> None:
        """limit=10001 exceeds the allowed maximum → ValidationError."""
        from pydantic import ValidationError

        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        with pytest.raises(ValidationError):
            ArchiveParams(limit=10001)

    def test_cursor_and_page_mutually_exclusive(self) -> None:
        """cursor + page both supplied → ValidationError (mutually exclusive)."""
        from pydantic import ValidationError

        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        with pytest.raises(ValidationError):
            ArchiveParams(cursor="abc123", page=2)

    def test_cursor_alone_accepted(self) -> None:
        """cursor without page → valid."""
        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams(cursor="abc123")
        assert p.cursor == "abc123"
        assert p.page is None

    def test_page_alone_accepted(self) -> None:
        """page without cursor → valid."""
        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams(page=3)
        assert p.page == 3
        assert p.cursor is None

    def test_page_minimum_boundary_accepted(self) -> None:
        """page=1 is the minimum allowed (1-based per OpenAPI)."""
        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams(page=1)
        assert p.page == 1

    def test_page_below_minimum_rejected(self) -> None:
        """page=0 is below 1-based minimum → ValidationError."""
        from pydantic import ValidationError

        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        with pytest.raises(ValidationError):
            ArchiveParams(page=0)

    def test_unknown_query_key_rejected_by_extra_forbid(self) -> None:
        """Unknown keys rejected when model has extra='forbid' (security-baseline §3.5)."""
        from pydantic import ValidationError

        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        with pytest.raises(ValidationError):
            ArchiveParams(foo="bar")  # type: ignore[call-arg]

    def test_fields_comma_separated_accepted(self) -> None:
        """fields=outTemp,rain → accepted without validation error."""
        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams(fields="outTemp,rain")
        assert p.fields == "outTemp,rain"

    def test_model_has_extra_forbid_config(self) -> None:
        """ArchiveParams model_config has extra='forbid'."""
        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        config = ArchiveParams.model_config
        assert config.get("extra") == "forbid", (
            "ArchiveParams must have extra='forbid' per security-baseline §3.5"
        )

    def test_date_only_from_is_utc_midnight(self) -> None:
        """from=2026-05-01 (date-only, no tz) → UTC-aware midnight, not local-time midnight.

        weewx archive stores Unix epoch seconds (UTC).  A naive datetime produced
        by Pydantic from a date-only string would be treated as local time by
        datetime.timestamp(), shifting the query window by the host's UTC offset.
        The normalise_to_utc validator must attach UTC so the epoch is correct.
        """
        from datetime import UTC

        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams.model_validate({"from": "2026-05-01"})
        assert p.from_ is not None
        assert p.from_.tzinfo is not None, (
            "from_ parsed from a date-only string must be tz-aware (UTC)"
        )
        assert p.from_.tzinfo == UTC, (
            f"from_ must be UTC, got tzinfo={p.from_.tzinfo!r}"
        )
        # 2026-05-01T00:00:00Z as a Unix epoch
        assert int(p.from_.timestamp()) == 1777593600, (
            "date-only 'from' must produce the UTC-midnight epoch, not a local-time epoch"
        )

    def test_date_only_to_is_utc_midnight(self) -> None:
        """to=2026-05-21 (date-only, no tz) → UTC-aware midnight."""
        from datetime import UTC

        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams.model_validate({"to": "2026-05-21"})
        assert p.to is not None
        assert p.to.tzinfo == UTC
        # 2026-05-21T00:00:00Z
        assert int(p.to.timestamp()) == 1779321600

    def test_utc_z_from_is_accepted_unchanged(self) -> None:
        """from=2026-05-01T00:00:00Z → UTC-aware, timestamp unchanged."""
        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams.model_validate({"from": "2026-05-01T00:00:00Z"})
        assert p.from_ is not None
        assert int(p.from_.timestamp()) == 1777593600

    def test_non_utc_offset_from_is_normalised_to_utc(self) -> None:
        """from=2026-05-01T00:00:00-07:00 → converted to UTC (2026-05-01T07:00:00Z)."""
        from datetime import UTC

        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams.model_validate({"from": "2026-05-01T00:00:00-07:00"})
        assert p.from_ is not None
        assert p.from_.tzinfo == UTC
        # Midnight Pacific = 07:00 UTC
        assert int(p.from_.timestamp()) == 1777618800

    def test_none_from_stays_none(self) -> None:
        """Omitting from → from_=None (normaliser is a no-op for None)."""
        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams()
        assert p.from_ is None

    def test_none_to_stays_none(self) -> None:
        """Omitting to → to=None (normaliser is a no-op for None)."""
        from weewx_clearskies_api.endpoints.observations import ArchiveParams

        p = ArchiveParams()
        assert p.to is None


class TestRecordsParamsModel:
    """/records query params: RecordsParams Pydantic model."""

    def test_default_period_is_ytd(self) -> None:
        """Omitting period → defaults to 'ytd'."""
        from weewx_clearskies_api.endpoints.records import RecordsParams

        p = RecordsParams()
        assert p.period == "ytd"

    def test_ytd_period_accepted(self) -> None:
        """period='ytd' → valid."""
        from weewx_clearskies_api.endpoints.records import RecordsParams

        p = RecordsParams(period="ytd")
        assert p.period == "ytd"

    def test_all_time_period_accepted(self) -> None:
        """period='all-time' → valid."""
        from weewx_clearskies_api.endpoints.records import RecordsParams

        p = RecordsParams(period="all-time")
        assert p.period == "all-time"

    def test_four_digit_year_period_accepted(self) -> None:
        """period='2025' (4-digit year) → valid."""
        from weewx_clearskies_api.endpoints.records import RecordsParams

        p = RecordsParams(period="2025")
        assert p.period == "2025"

    def test_four_digit_year_1900_boundary_accepted(self) -> None:
        """period='1900' (boundary year from OpenAPI minimum) → valid."""
        from weewx_clearskies_api.endpoints.records import RecordsParams

        p = RecordsParams(period="1900")
        assert p.period == "1900"

    def test_two_digit_year_rejected(self) -> None:
        """period='25' (2-digit year) → ValidationError."""
        from pydantic import ValidationError

        from weewx_clearskies_api.endpoints.records import RecordsParams

        with pytest.raises(ValidationError):
            RecordsParams(period="25")

    def test_non_numeric_period_rejected(self) -> None:
        """period='abc' → ValidationError."""
        from pydantic import ValidationError

        from weewx_clearskies_api.endpoints.records import RecordsParams

        with pytest.raises(ValidationError):
            RecordsParams(period="abc")

    def test_year_below_1900_rejected(self) -> None:
        """period='1899' → ValidationError (below OpenAPI minimum year=1900)."""
        from pydantic import ValidationError

        from weewx_clearskies_api.endpoints.records import RecordsParams

        with pytest.raises(ValidationError):
            RecordsParams(period="1899")

    def test_valid_section_accepted(self) -> None:
        """section='temperature' → valid enum value."""
        from weewx_clearskies_api.endpoints.records import RecordsParams

        p = RecordsParams(section="temperature")
        assert p.section == "temperature"

    def test_all_section_enum_values_accepted(self) -> None:
        """All 9 section enum values from OpenAPI are accepted."""
        from weewx_clearskies_api.endpoints.records import RecordsParams

        valid_sections = [
            "temperature", "wind", "rain", "humidity",
            "barometer", "sun", "aqi", "inside-temp", "custom",
        ]
        for sec in valid_sections:
            p = RecordsParams(section=sec)
            assert p.section == sec, f"Section {sec!r} should be accepted"

    def test_unknown_section_rejected(self) -> None:
        """section='lightning' (not in OpenAPI enum) → ValidationError."""
        from pydantic import ValidationError

        from weewx_clearskies_api.endpoints.records import RecordsParams

        with pytest.raises(ValidationError):
            RecordsParams(section="lightning")

    def test_unknown_query_key_rejected_by_extra_forbid(self) -> None:
        """Unknown keys rejected per extra='forbid'."""
        from pydantic import ValidationError

        from weewx_clearskies_api.endpoints.records import RecordsParams

        with pytest.raises(ValidationError):
            RecordsParams(unknown_key="value")  # type: ignore[call-arg]

    def test_model_has_extra_forbid_config(self) -> None:
        """RecordsParams model_config has extra='forbid'."""
        from weewx_clearskies_api.endpoints.records import RecordsParams

        config = RecordsParams.model_config
        assert config.get("extra") == "forbid", (
            "RecordsParams must have extra='forbid' per security-baseline §3.5"
        )
