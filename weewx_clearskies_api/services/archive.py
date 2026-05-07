"""Archive query service — current observation and historical records.

Supports three interval modes per the task brief:
  - raw:  plain archive row read.
  - day:  reads from weewx archive_day_<obs> summary tables.
  - hour: on-the-fly grouping from the archive table using a dialect helper.

Dialect-specific SQL is wrapped in _HourDialect to avoid branching inside
route handlers.

SQL note: column identifiers in query text come exclusively from
_STOCK_OBS_COLS and the archive table's known schema — trusted constants, not
user-supplied values.  All value bindings use named parameters (:name).

ruff: noqa: N815  (canonical fields use weewx camelCase per ADR-010)
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from weewx_clearskies_api.db.reflection import STOCK_COLUMN_MAP, ColumnRegistry
from weewx_clearskies_api.models.responses import (
    ArchiveRecord,
    Observation,
    PageInfo,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-field aggregator for interval=day (proposed default mapping).
# Lead reviews at closeout.
#
# Default logic:
#   Temperatures, humidity, barometer, pressure → "avg" (daily mean).
#   Rain / ET / hail / windrun → "sum" (accumulation).
#   Wind speed / gust / radiation / UV / solar → "max" (peak).
#   Rain rate, hail rate → "max" (peak rate is the meaningful stat).
#   Windchill → "min" (most extreme cold).
#   Heatindex, THSW, humidex → "max" (most extreme heat).
# ---------------------------------------------------------------------------

# archive_day_* aggregator column name for each category.
# weewx archive_day cols: min, mintime, max, maxtime, sum, count, wsum, sumtime, avg
_DAY_AGG_TO_COL: dict[str, str] = {
    "avg": "avg",
    "max": "max",
    "min": "min",
    "sum": "sum",
}

# Canonical field → which archive_day_* column to read.
DAY_AGGREGATOR: dict[str, str] = {
    "outTemp": "avg",
    "outHumidity": "avg",
    "windSpeed": "max",
    "windDir": "avg",
    "windGust": "max",
    "windGustDir": "avg",
    "barometer": "avg",
    "pressure": "avg",
    "altimeter": "avg",
    "dewpoint": "avg",
    "windchill": "min",
    "heatindex": "max",
    "rainRate": "max",
    "rain": "sum",
    "radiation": "max",
    "UV": "max",
    "inTemp": "avg",
    "inHumidity": "avg",
    "ET": "sum",
    "hail": "sum",
    "hailRate": "max",
    "appTemp": "avg",
    "cloudbase": "avg",
    "extraTemp1": "avg",
    "extraTemp2": "avg",
    "extraTemp3": "avg",
    "extraHumid1": "avg",
    "extraHumid2": "avg",
    "soilTemp1": "avg",
    "soilTemp2": "avg",
    "soilTemp3": "avg",
    "soilTemp4": "avg",
    "soilMoist1": "avg",
    "soilMoist2": "avg",
    "soilMoist3": "avg",
    "soilMoist4": "avg",
    "leafTemp1": "avg",
    "leafTemp2": "avg",
    "leafWet1": "max",
    "leafWet2": "max",
    "THSW": "max",
    "humidex": "max",
    "snow": "sum",
    "snowDepth": "max",
    "snowRate": "max",
    "maxSolarRad": "max",
    "sunshineDur": "sum",
    "rainDur": "sum",
    "windrun": "sum",
    "illuminance": "max",
    "rxCheckPercent": "avg",
    "consBatteryVoltage": "avg",
    "heatingVoltage": "avg",
    "referenceVoltage": "avg",
    "supplyVoltage": "avg",

    # Newly first-class fields (added per user directive 2026-05-06).
    # Judgment calls flagged in closeout report for lead review.
    "cloudcover": "avg",          # percent — average
    "cooldeg": "sum",             # degree-days — accumulation
    "daySunshineDur": "max",      # running cumulative since midnight; max = end-of-day total
    "gustdir": "avg",             # direction — avg (JUDGMENT: max could be argued)
    "heatdeg": "sum",             # degree-days — accumulation
    "lightning_distance": "max",  # peak distance to nearest strike
    "lightning_disturber_count": "sum",  # count — accumulation
    "lightning_noise_count": "sum",      # count — accumulation
    "lightning_strike_count": "sum",     # count — accumulation
    "noise": "avg",               # dB — average ambient noise level
    "pop": "avg",                 # percent — average probability
    "rms": "avg",                 # speed2 — statistical mean
    "vecavg": "avg",              # speed2 — vector-mean of averages
    "vecdir": "avg",              # direction — average
}

# Meta columns — present in the archive table but NOT observation fields.
_META_COLS: frozenset[str] = frozenset({"dateTime", "usUnits", "interval"})

# First-class observation canonical names.
# Derived from STOCK_COLUMN_MAP minus meta columns so this set stays in sync
# with the stock-column lookup automatically; no hand-maintained second list.
# Per the user directive 2026-05-06: every stock weewx column is first-class
# on Observation; extras is operator-custom-only.
_FIRST_CLASS_FIELDS: frozenset[str] = frozenset(
    canonical
    for db_col, canonical in STOCK_COLUMN_MAP.items()
    if db_col not in _META_COLS
)


# ---------------------------------------------------------------------------
# Cursor helpers
# ---------------------------------------------------------------------------


def encode_cursor(after_datetime: int) -> str:
    """Encode a cursor from a dateTime epoch value."""
    payload = json.dumps({"after_dateTime": after_datetime})
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(cursor: str) -> int:
    """Decode a cursor and return the after_dateTime epoch value.

    Raises:
        ValueError: If the cursor is malformed or missing the required key.
    """
    try:
        payload = base64.urlsafe_b64decode(cursor.encode()).decode()
        data = json.loads(payload)
        after = data["after_dateTime"]
        if not isinstance(after, int):
            raise ValueError("after_dateTime must be an integer")
        return after
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid cursor: {exc}") from exc


# ---------------------------------------------------------------------------
# Row → model helpers
# ---------------------------------------------------------------------------


def _epoch_to_utc_z(epoch: int | float) -> str:
    return datetime.fromtimestamp(float(epoch), tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row_to_observation(row: Any, registry: ColumnRegistry) -> Observation:
    """Convert a raw DB row (mapping) to an Observation model.

    Routing per user directive 2026-05-06 (two branches only, no middle bucket):
      - Stock column (in registry.stock) → first-class field on Observation.
        The Observation model now carries every canonical name in STOCK_COLUMN_MAP.
        Columns the operator's archive doesn't have are simply absent from the row
        dict; Observation's defaults fill them as None.
      - Non-stock column (in registry.unmapped) → operator-custom → extras.
        Stock weewx columns NEVER appear in extras.
    """
    row_dict: dict[str, Any] = dict(row._mapping)  # noqa: SLF001

    timestamp_epoch = row_dict.get("dateTime")
    timestamp_str = _epoch_to_utc_z(timestamp_epoch) if timestamp_epoch is not None else ""

    obs_fields: dict[str, Any] = {"timestamp": timestamp_str, "source": "weewx"}
    extras: dict[str, Any] = {}

    for db_col, val in row_dict.items():
        if db_col in _META_COLS:
            continue
        col_info = registry.stock.get(db_col)
        if col_info is not None:
            # Stock column — always first-class, regardless of which specific field.
            obs_fields[col_info.canonical_name] = val
        else:
            # Non-stock (operator-custom) column → extras.
            extras[db_col] = val

    obs_fields["extras"] = extras
    return Observation(**obs_fields)


def _row_to_archive_record(row: Any, registry: ColumnRegistry) -> ArchiveRecord:
    obs = _row_to_observation(row, registry)
    row_dict = dict(row._mapping)  # noqa: SLF001
    interval_val = row_dict.get("interval", 0)
    return ArchiveRecord(**obs.model_dump(), interval=int(interval_val or 0))


# ---------------------------------------------------------------------------
# Dialect helper for hourly aggregation
# ---------------------------------------------------------------------------


class _HourDialect:
    """Encapsulate dialect-specific SQL for hourly GROUP BY."""

    def __init__(self, dialect_name: str) -> None:
        self._name = dialect_name

    def hour_bucket_expr(self) -> str:
        """SQL expression (trusted constant) that truncates dateTime to the hour.

        Returns the raw SQL fragment to embed in a SQLAlchemy text() query.

        Note on % escaping: SQLAlchemy text() compiles for the target driver.
        pymysql (MariaDB) uses pyformat (%s / %(name)s) — SQLAlchemy escapes
        any literal % in text() SQL to %% before sending to pymysql.  The
        MariaDB variant therefore pre-doubles the % so the final wire SQL has
        the correct single-% format codes MySQL expects.

        SQLite uses qmark (?) params — no % escaping issue; single % is fine.
        """
        if self._name == "sqlite":
            return "strftime('%Y-%m-%d %H:00:00', datetime(dateTime, 'unixepoch'))"
        # Double the % so SQLAlchemy's pyformat escaping produces single % on the wire.
        return "FROM_UNIXTIME(dateTime, '%%Y-%%m-%%d %%H:00:00')"


# ---------------------------------------------------------------------------
# Primary service functions
# ---------------------------------------------------------------------------


def get_current(
    db: Session,
    registry: ColumnRegistry,
) -> Observation | None:
    """Return the most recent archive row as an Observation, or None.

    Returns None when the archive is empty (no rows).  Also returns None when
    the archive table does not exist yet (fresh weewx install before any data
    has been collected — "no such table: archive").  All other OperationalErrors
    (DB unreachable, permissions revoked, schema corruption) re-raise so the
    caller surfaces them as 500.
    """
    from sqlalchemy.exc import OperationalError

    sql = text("SELECT * FROM archive ORDER BY dateTime DESC LIMIT 1")
    try:
        row = db.execute(sql).fetchone()
    except OperationalError as exc:
        # Only swallow "no such table: archive" — that is a fresh-install / empty-test
        # condition, not a DB failure. All other OperationalErrors re-raise.
        orig_msg = str(exc.orig).lower() if exc.orig is not None else str(exc).lower()
        if "no such table" in orig_msg:
            return None
        raise
    if row is None:
        return None
    return _row_to_observation(row, registry)


def get_archive(
    db: Session,
    registry: ColumnRegistry,
    from_dt: datetime | None,
    to_dt: datetime | None,
    interval: str,
    fields: list[str] | None,
    limit: int,
    cursor: str | None,
    page: int | None,
) -> tuple[list[ArchiveRecord], PageInfo]:
    """Query the archive and return (records, page_info)."""
    now = datetime.now(tz=UTC)
    effective_to = to_dt if to_dt is not None else now
    effective_from = from_dt if from_dt is not None else (now - timedelta(hours=24))

    from_epoch = int(effective_from.timestamp())
    to_epoch = int(effective_to.timestamp())

    offset = 0
    if page is not None:
        offset = (page - 1) * limit

    if cursor is not None:
        after_dt_epoch = decode_cursor(cursor)
        from_epoch = after_dt_epoch + 1

    if interval == "raw":
        return _fetch_raw(db, registry, from_epoch, to_epoch, limit, offset, page, cursor, fields)
    if interval == "hour":
        return _fetch_hourly(db, registry, from_epoch, to_epoch, limit, offset, page, cursor, fields)
    if interval == "day":
        return _fetch_daily(db, registry, from_epoch, to_epoch, limit, offset, page, cursor, fields)
    raise ValueError(f"Unknown interval: {interval!r}")


# ---------------------------------------------------------------------------
# Raw mode
# ---------------------------------------------------------------------------


def _fetch_raw(
    db: Session,
    registry: ColumnRegistry,
    from_epoch: int,
    to_epoch: int,
    limit: int,
    offset: int,
    page: int | None,
    cursor: str | None,
    fields: list[str] | None,
) -> tuple[list[ArchiveRecord], PageInfo]:
    sql = text(
        "SELECT * FROM archive "
        "WHERE dateTime >= :from_ts AND dateTime < :to_ts "
        "ORDER BY dateTime ASC "
        "LIMIT :lim OFFSET :off"
    )
    rows = db.execute(
        sql, {"from_ts": from_epoch, "to_ts": to_epoch, "lim": limit + 1, "off": offset}
    ).fetchall()

    has_more = len(rows) > limit
    rows = rows[:limit]

    records = [_row_to_archive_record(r, registry) for r in rows]

    if fields is not None:
        records = _filter_record_fields(records, fields)

    next_cursor: str | None = None
    if has_more and rows:
        last_epoch = dict(rows[-1]._mapping).get("dateTime")  # noqa: SLF001
        if last_epoch is not None:
            next_cursor = encode_cursor(int(last_epoch))

    total_pages: int | None = None
    total_records: int | None = None

    if page is not None:
        count_sql = text(
            "SELECT COUNT(*) FROM archive "
            "WHERE dateTime >= :from_ts AND dateTime < :to_ts"
        )
        total_records = db.execute(
            count_sql, {"from_ts": from_epoch, "to_ts": to_epoch}
        ).scalar() or 0
        total_pages = max(1, (total_records + limit - 1) // limit)

    return records, PageInfo(
        cursor=next_cursor, limit=limit, page=page,
        totalPages=total_pages, totalRecords=total_records,
    )


# ---------------------------------------------------------------------------
# Hourly mode
# ---------------------------------------------------------------------------


def _stock_obs_columns(registry: ColumnRegistry) -> list[str]:
    """Stock archive columns that are observation values (not meta)."""
    return [col for col in registry.stock if col not in _META_COLS]


def _fetch_hourly(
    db: Session,
    registry: ColumnRegistry,
    from_epoch: int,
    to_epoch: int,
    limit: int,
    offset: int,
    page: int | None,
    cursor: str | None,
    fields: list[str] | None,
) -> tuple[list[ArchiveRecord], PageInfo]:
    dialect = _HourDialect(db.bind.dialect.name)  # type: ignore[union-attr]
    bucket_expr = dialect.hour_bucket_expr()  # trusted dialect constant

    # Build the SELECT list from trusted stock column names.
    # Only include stock columns that map to first-class numeric observation fields
    # to avoid AVG() on text columns.
    stock_cols = [
        col for col in _stock_obs_columns(registry)
        if registry.stock.get(col) is not None
        and registry.stock[col].canonical_name in _FIRST_CLASS_FIELDS
    ]
    if fields is not None:
        field_set = set(fields)
        stock_cols = [
            c for c in stock_cols
            if registry.stock.get(c) is not None
            and registry.stock[c].canonical_name in field_set
        ]

    # Trusted column identifiers — sourced from schema reflection, not user input.
    agg_parts = ", ".join(f"AVG({col}) AS {col}" for col in stock_cols)
    if agg_parts:
        agg_parts = ", " + agg_parts

    # `interval` is a reserved word in MariaDB.
    # Use a quoted alias — backtick quoting is supported by both SQLite and MariaDB.
    sql = text(
        f"SELECT {bucket_expr} AS hour_bucket, "
        f"MIN(dateTime) AS dateTime, "
        f"MAX(usUnits) AS usUnits, "
        f"60 AS `interval`"
        f"{agg_parts} "
        f"FROM archive "
        f"WHERE dateTime >= :from_ts AND dateTime < :to_ts "
        f"GROUP BY hour_bucket "
        f"ORDER BY hour_bucket ASC "
        f"LIMIT :lim OFFSET :off"
    )
    rows = db.execute(
        sql, {"from_ts": from_epoch, "to_ts": to_epoch, "lim": limit + 1, "off": offset}
    ).fetchall()

    has_more = len(rows) > limit
    rows = rows[:limit]

    records = [_row_to_archive_record(r, registry) for r in rows]

    next_cursor: str | None = None
    if has_more and rows:
        last_dt = dict(rows[-1]._mapping).get("dateTime")  # noqa: SLF001
        if last_dt is not None:
            next_cursor = encode_cursor(int(last_dt))

    total_pages: int | None = None
    total_records: int | None = None

    if page is not None:
        count_sql = text(
            f"SELECT COUNT(*) FROM ("
            f"  SELECT {bucket_expr} AS hour_bucket "
            f"  FROM archive "
            f"  WHERE dateTime >= :from_ts AND dateTime < :to_ts "
            f"  GROUP BY hour_bucket"
            f") sub"
        )
        total_records = db.execute(
            count_sql, {"from_ts": from_epoch, "to_ts": to_epoch}
        ).scalar() or 0
        total_pages = max(1, (total_records + limit - 1) // limit)

    return records, PageInfo(
        cursor=next_cursor, limit=limit, page=page,
        totalPages=total_pages, totalRecords=total_records,
    )


# ---------------------------------------------------------------------------
# Daily mode
# ---------------------------------------------------------------------------


def _fetch_daily(
    db: Session,
    registry: ColumnRegistry,
    from_epoch: int,
    to_epoch: int,
    limit: int,
    offset: int,
    page: int | None,
    cursor: str | None,
    fields: list[str] | None,
) -> tuple[list[ArchiveRecord], PageInfo]:
    """Read from archive_day_* summary tables.

    Each canonical field gets its value from the appropriate column in the
    field's own summary table.
    """
    all_stock_keys = [k for k in registry.stock if k not in _META_COLS]
    if fields is not None:
        target_fields = [f for f in fields if f in registry.stock]
    else:
        target_fields = all_stock_keys

    # Limit to fields that have a known day-aggregator and a day table.
    aggregable = [f for f in target_fields if f in DAY_AGGREGATOR]

    day_rows = _fetch_day_aggregates(
        db, aggregable, from_epoch, to_epoch, limit, offset
    )

    records: list[ArchiveRecord] = []
    for day in day_rows:
        dt_epoch = day.get("dateTime")
        ts_str = _epoch_to_utc_z(dt_epoch) if dt_epoch is not None else ""
        obs_kwargs: dict[str, Any] = {"timestamp": ts_str, "source": "weewx", "extras": {}}
        for field_name in aggregable:
            if field_name in day:
                obs_kwargs[field_name] = day[field_name]
        records.append(ArchiveRecord(**obs_kwargs, interval=1440))

    next_cursor: str | None = None
    total_pages: int | None = None
    total_records: int | None = None

    if day_rows and len(day_rows) == limit:
        last_day_epoch = day_rows[-1].get("dateTime")
        if last_day_epoch is not None:
            next_cursor = encode_cursor(int(last_day_epoch))

    if page is not None:
        # Best-effort count from a representative day table.
        for field_name in aggregable:
            table_name = f"archive_day_{field_name}"
            try:
                count_sql = text(
                    f"SELECT COUNT(*) FROM {table_name} "
                    f"WHERE dateTime >= :from_ts AND dateTime < :to_ts"
                )
                total_records = db.execute(
                    count_sql, {"from_ts": from_epoch, "to_ts": to_epoch}
                ).scalar() or 0
                total_pages = max(1, (total_records + limit - 1) // limit)
                break
            except (OperationalError, ProgrammingError):
                # Table doesn't exist for this field (archive_day_* tables are
                # per-observation; not all are guaranteed present). Try the next.
                continue

    return records, PageInfo(
        cursor=next_cursor, limit=limit, page=page,
        totalPages=total_pages, totalRecords=total_records,
    )


def _fetch_day_aggregates(
    db: Session,
    fields: list[str],
    from_epoch: int,
    to_epoch: int,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    """Fetch per-day aggregated rows from archive_day_* tables.

    Returns list of dicts keyed by canonical field name + "dateTime".
    Table names (archive_day_<field>) are constructed from trusted stock field
    names, not user input.
    """
    # Collect rows keyed by dateTime bucket.
    day_data: dict[int, dict[str, Any]] = {}
    # We need a primary table to establish the dateTime bucket list.
    primary_rows: list[Any] = []
    primary_table_found = False

    for field_name in fields:
        agg_col = DAY_AGGREGATOR.get(field_name)
        if agg_col is None:
            continue
        # Table name is archive_day_<stock_field_name> — trusted constant.
        table_name = f"archive_day_{field_name}"
        agg_col_name = _DAY_AGG_TO_COL.get(agg_col, "avg")

        sql = text(
            f"SELECT dateTime, {agg_col_name} AS val "
            f"FROM {table_name} "
            f"WHERE dateTime >= :from_ts AND dateTime < :to_ts "
            f"ORDER BY dateTime ASC "
            f"LIMIT :lim OFFSET :off"
        )
        try:
            rows = db.execute(
                sql,
                {"from_ts": from_epoch, "to_ts": to_epoch, "lim": limit, "off": offset},
            ).fetchall()
        except (OperationalError, ProgrammingError) as exc:
            # Table doesn't exist for this field; skip and try the next one.
            logger.debug("Day summary table %r not available: %s", table_name, exc)
            continue

        if not primary_table_found and rows:
            primary_rows = rows
            primary_table_found = True

        for row in rows:
            dt_key = int(row[0])
            if dt_key not in day_data:
                day_data[dt_key] = {"dateTime": dt_key}
            day_data[dt_key][field_name] = row[1]

    if not day_data:
        return []

    return [day_data[dt] for dt in sorted(day_data.keys())]


# ---------------------------------------------------------------------------
# Field filtering
# ---------------------------------------------------------------------------


def _filter_record_fields(
    records: list[ArchiveRecord], fields: list[str]
) -> list[ArchiveRecord]:
    """Set unrequested optional fields to None.  Always keeps timestamp/source/extras/interval."""
    keep = set(fields) | {"timestamp", "source", "extras", "interval"}
    filtered: list[ArchiveRecord] = []
    for rec in records:
        d = rec.model_dump()
        for k in list(d.keys()):
            if k not in keep and k not in ("extras", "timestamp", "source", "interval"):
                d[k] = None
        filtered.append(ArchiveRecord(**d))
    return filtered
