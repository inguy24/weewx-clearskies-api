"""Pydantic parameter models for the DB-backed and almanac endpoints.

All models use ConfigDict(extra="forbid") per security-baseline §3.5 —
unknown query keys are rejected with 422 (reshaped to 400 problem+json by
the error handler).

ruff: noqa: N815  (canonical field names use weewx camelCase per ADR-010)
"""

from __future__ import annotations

import re
import datetime as _datetime_mod
from datetime import UTC, date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# /archive query params
# ---------------------------------------------------------------------------

_ARCHIVE_INTERVAL_CHOICES = frozenset({"raw", "hour", "day"})


class ArchiveQueryParams(BaseModel):
    """Validated query parameters for GET /archive."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: datetime | None = Field(default=None, alias="from")
    to: datetime | None = None
    interval: str = "raw"
    fields: str | None = None
    limit: int = Field(default=1000, ge=1, le=10000)
    cursor: str | None = None
    page: int | None = Field(default=None, ge=1)

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: str) -> str:
        if v not in _ARCHIVE_INTERVAL_CHOICES:
            raise ValueError(
                f"interval must be one of {sorted(_ARCHIVE_INTERVAL_CHOICES)}"
            )
        return v

    @model_validator(mode="after")
    def check_cursor_page_exclusive(self) -> "ArchiveQueryParams":
        if self.cursor is not None and self.page is not None:
            raise ValueError("cursor and page are mutually exclusive")
        return self


# ---------------------------------------------------------------------------
# /records query params
# ---------------------------------------------------------------------------

_VALID_SECTIONS = frozenset({
    "temperature", "wind", "rain", "humidity", "barometer",
    "sun", "aqi", "inside-temp", "custom",
})

_YEAR_RE = re.compile(r"^\d{4}$")


class RecordsQueryParams(BaseModel):
    """Validated query parameters for GET /records."""

    model_config = ConfigDict(extra="forbid")

    period: str = "ytd"
    section: str | None = None

    @field_validator("period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        if v in ("ytd", "all-time"):
            return v
        if _YEAR_RE.match(v):
            year = int(v)
            if year < 1900:
                raise ValueError("Year must be >= 1900")
            # Optionally reject future years.
            current_year = datetime.now(tz=UTC).year
            if year > current_year:
                raise ValueError(f"Year {year} is in the future")
            return v
        raise ValueError(
            "period must be 'ytd', 'all-time', or a 4-digit year (e.g. '2025')"
        )

    @field_validator("section")
    @classmethod
    def validate_section(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_SECTIONS:
            raise ValueError(
                f"section must be one of: {', '.join(sorted(_VALID_SECTIONS))}"
            )
        return v


# ---------------------------------------------------------------------------
# /reports/{year}/{month} path params
# ---------------------------------------------------------------------------


class ReportMonthlyParams(BaseModel):
    """Validated path parameters for GET /reports/{year}/{month}."""

    model_config = ConfigDict(extra="forbid")

    year: int = Field(ge=1900)
    month: int = Field(ge=1, le=12)


# ---------------------------------------------------------------------------
# /reports/{year} path params
# ---------------------------------------------------------------------------


class ReportYearlyParams(BaseModel):
    """Validated path parameters for GET /reports/{year}."""

    model_config = ConfigDict(extra="forbid")

    year: int = Field(ge=1900)


# ---------------------------------------------------------------------------
# /almanac query params
# ---------------------------------------------------------------------------


class AlmanacQueryParams(BaseModel):
    """Validated query parameters for GET /almanac.

    The 'date' field uses `_datetime_mod.date` (fully qualified) as its type
    annotation to avoid a Pydantic 2 forward-reference resolution bug that
    fires when a field name ('date') shadows the imported stdlib type name
    ('date' from datetime) under 'from __future__ import annotations'.
    See: https://github.com/pydantic/pydantic/issues/8900
    """

    model_config = ConfigDict(extra="forbid")

    # Fully-qualified type reference avoids Pydantic forward-ref shadowing bug.
    date: _datetime_mod.date | None = None

    @field_validator("date", mode="before")
    @classmethod
    def validate_date_field(cls, v: object) -> object:
        if v is None:
            return v
        if isinstance(v, str):
            try:
                return _datetime_mod.date.fromisoformat(v)
            except ValueError as exc:
                raise ValueError(
                    f"date must be a valid ISO date (YYYY-MM-DD), got {v!r}"
                ) from exc
        return v


# ---------------------------------------------------------------------------
# /almanac/sun-times query params
# ---------------------------------------------------------------------------


class SunTimesQueryParams(BaseModel):
    """Validated query parameters for GET /almanac/sun-times."""

    model_config = ConfigDict(extra="forbid")

    year: int | None = Field(default=None, ge=1900)


# ---------------------------------------------------------------------------
# /almanac/moon-phases query params
# ---------------------------------------------------------------------------


class MoonPhasesQueryParams(BaseModel):
    """Validated query parameters for GET /almanac/moon-phases."""

    model_config = ConfigDict(extra="forbid")

    year: int | None = Field(default=None, ge=1900)
    month: int | None = Field(default=None, ge=1, le=12)


# ---------------------------------------------------------------------------
# /alerts query params
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# /forecast query params
# ---------------------------------------------------------------------------

# Open-Meteo supports forecast_days=0..16 (docs/reference/api-docs/openmeteo.md).
# Hourly max = 16 days × 24 h = 384 hours.
_FORECAST_MAX_HOURS = 384
_FORECAST_MAX_DAYS = 16


class ForecastQueryParams(BaseModel):
    """Validated query parameters for GET /forecast.

    hours: Number of hourly forecast points (default 48, max 384).
    days: Number of daily forecast points (default 7, max 16).

    extra="forbid" per security-baseline §3.5 — unknown query keys are
    rejected with 422 (reshaped to 400 problem+json by the error handler).

    Slice-after-cache pattern (ADR-017 §Cache key brief §lead-call 13):
    These params are applied at the endpoint layer after cache lookup.
    The module always asks Open-Meteo for the full default forecast window
    so the cache entry is operator-uniform.
    """

    model_config = ConfigDict(extra="forbid")

    hours: int = Field(default=48, ge=0, le=_FORECAST_MAX_HOURS)
    days: int = Field(default=7, ge=0, le=_FORECAST_MAX_DAYS)



_SEVERITY_CHOICES = frozenset({"advisory", "watch", "warning"})
# Severity order: higher index = more severe.
# Severity filter: advisory returns all; watch returns watch+warning; warning returns warning only.
# Exposed as a module-level constant so the /alerts endpoint can use it for filtering
# without re-defining the mapping in a second place (DRY per coding.md §3).
SEVERITY_ORDER = {"advisory": 0, "watch": 1, "warning": 2}


class AlertsQueryParams(BaseModel):
    """Validated query parameters for GET /alerts.

    severity: Optional minimum severity filter.
      advisory → return all alerts
      watch    → return watch + warning alerts
      warning  → return warning alerts only

    extra="forbid" per security-baseline §3.5 (blocks unknown query keys).
    """

    model_config = ConfigDict(extra="forbid")

    severity: str | None = None

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str | None) -> str | None:
        if v is not None and v not in _SEVERITY_CHOICES:
            raise ValueError(
                f"severity must be one of: {', '.join(sorted(_SEVERITY_CHOICES))}. "
                f"Got {v!r}."
            )
        return v

    def min_severity_level(self) -> int:
        """Return the numeric severity level for the filter (0 = advisory, 1 = watch, 2 = warning)."""
        if self.severity is None:
            return 0
        return SEVERITY_ORDER.get(self.severity, 0)


# ---------------------------------------------------------------------------
# /aqi/current query params
# ---------------------------------------------------------------------------


class AQIQueryParams(BaseModel):
    """Validated query parameters for GET /aqi/current.

    No query parameters accepted — extra="forbid" rejects unknown keys (422).
    Implements the security-baseline §3.5 Depends-wrapper pattern.
    """

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# /aqi/history query params
# ---------------------------------------------------------------------------


class AQIHistoryQueryParams(BaseModel):
    """Validated query parameters for GET /aqi/history.

    Per OpenAPI /aqi/history parameters: from, to, limit, cursor, page.
    extra="forbid" rejects unknown keys (422).
    Handler always returns 501 regardless of params; validation happens first
    so invalid params return 422, valid params return 501.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: datetime | None = Field(default=None, alias="from")
    to: datetime | None = None
    limit: int = Field(default=1000, ge=1, le=10000)
    cursor: str | None = None
    page: int | None = Field(default=None, ge=1)


# ---------------------------------------------------------------------------
# /earthquakes query params
# ---------------------------------------------------------------------------


class EarthquakesQueryParams(BaseModel):
    """Query params for /earthquakes (OpenAPI getEarthquakes operation).

    extra="forbid" so unknown query keys reject with 422 per security-baseline §3.5.
    The Depends-wrapper pattern (coding.md §1) ensures the full query string flows
    through Pydantic so extra="forbid" actually fires.

    from_ / to: ISO 8601 timestamps bounding the event time window.
    min_magnitude: filter to events at or above this magnitude.
    radius_km: override the operator-configured radius (km from station lat/lon).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: datetime | None = Field(default=None, alias="from")
    to: datetime | None = None
    min_magnitude: float | None = Field(None, ge=0)
    radius_km: float | None = Field(None, ge=0)
