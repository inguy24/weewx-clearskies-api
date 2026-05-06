"""Unit tests for db/reflection.py (ADR-012, ADR-035).

Tests:
  - Stock columns are auto-mapped to canonical names.
  - Non-stock columns are surfaced as unmapped (canonical_name=None).
  - STOCK_COLUMN_MAP is the lookup source — no hardcoding in tests.
  - refresh() re-runs reflect and updates the registry.
  - Missing archive table raises RuntimeError.
  - dateTime maps to "timestamp" (the one non-identity mapping).

We use SQLite in-memory engines populated with representative column sets.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from weewx_clearskies_api.db.reflection import (
    STOCK_COLUMN_MAP,
    ColumnRegistry,
    SchemaReflector,
    _build_registry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine_with_columns(columns: list[str]):  # type: ignore[return]
    """Return in-memory SQLite engine with an archive table containing `columns`."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    col_defs = ", ".join(f'"{c}" REAL' for c in columns)
    with engine.connect() as conn:
        conn.execute(text(f"CREATE TABLE archive ({col_defs})"))  # noqa: S608
        conn.commit()
    return engine


# ---------------------------------------------------------------------------
# _build_registry unit tests (pure function, no DB required)
# ---------------------------------------------------------------------------


def test_stock_columns_auto_mapped() -> None:
    """Stock columns (in STOCK_COLUMN_MAP) get canonical names assigned."""
    stock_cols = ["outTemp", "barometer", "windSpeed"]
    registry = _build_registry(stock_cols)
    for col in stock_cols:
        assert col in registry.stock
        assert registry.stock[col].canonical_name == STOCK_COLUMN_MAP[col]
        assert registry.stock[col].is_stock is True


def test_non_stock_columns_are_unmapped() -> None:
    """Columns not in STOCK_COLUMN_MAP go to the unmapped bucket."""
    cols = ["outTemp", "aqi_value", "custom_sensor"]
    registry = _build_registry(cols)
    assert "outTemp" in registry.stock
    assert "aqi_value" in registry.unmapped
    assert "custom_sensor" in registry.unmapped
    assert registry.unmapped["aqi_value"].canonical_name is None
    assert registry.unmapped["aqi_value"].is_stock is False


def test_datetime_maps_to_timestamp() -> None:
    """dateTime → timestamp is the one non-identity mapping in the stock map."""
    registry = _build_registry(["dateTime"])
    assert registry.stock["dateTime"].canonical_name == "timestamp"


def test_empty_column_list() -> None:
    """Empty column list produces empty registry without error."""
    registry = _build_registry([])
    assert registry.stock == {}
    assert registry.unmapped == {}


def test_all_columns_returns_combined_list() -> None:
    """all_columns() returns stock + unmapped together."""
    cols = ["outTemp", "dateTime", "my_custom_col"]
    registry = _build_registry(cols)
    all_names = {c.db_name for c in registry.all_columns()}
    assert all_names == set(cols)


def test_get_canonical_stock() -> None:
    """get_canonical() returns canonical name for stock column."""
    registry = _build_registry(["outTemp"])
    assert registry.get_canonical("outTemp") == "outTemp"


def test_get_canonical_unmapped() -> None:
    """get_canonical() returns None for unmapped column."""
    registry = _build_registry(["outTemp", "my_aqi"])
    assert registry.get_canonical("my_aqi") is None


def test_get_canonical_missing() -> None:
    """get_canonical() returns None for a column that wasn't even in the table."""
    registry = _build_registry(["outTemp"])
    assert registry.get_canonical("nonexistent") is None


# ---------------------------------------------------------------------------
# SchemaReflector integration tests (SQLite in-memory)
# ---------------------------------------------------------------------------


def test_reflect_populates_stock_columns() -> None:
    """reflect() maps stock columns from the real table schema."""
    cols = ["dateTime", "outTemp", "outHumidity", "windSpeed"]
    engine = _engine_with_columns(cols)
    try:
        reflector = SchemaReflector(engine)
        registry = reflector.reflect()
        for col in cols:
            assert col in registry.stock, f"{col!r} should be stock"
    finally:
        engine.dispose()


def test_reflect_surfaces_non_stock_as_unmapped() -> None:
    """Non-stock columns discovered during reflect() go to unmapped."""
    cols = ["dateTime", "outTemp", "aqi_index", "pm2_5_raw"]
    engine = _engine_with_columns(cols)
    try:
        reflector = SchemaReflector(engine)
        registry = reflector.reflect()
        assert "aqi_index" in registry.unmapped
        assert "pm2_5_raw" in registry.unmapped
    finally:
        engine.dispose()


def test_reflect_registry_accessible_via_property() -> None:
    """reflector.registry returns the same registry as reflect()."""
    engine = _engine_with_columns(["dateTime", "outTemp"])
    try:
        reflector = SchemaReflector(engine)
        returned = reflector.reflect()
        assert reflector.registry is returned
    finally:
        engine.dispose()


def test_refresh_updates_registry() -> None:
    """refresh() re-runs reflection; adding a column is visible after refresh."""
    engine = _engine_with_columns(["dateTime", "outTemp"])
    try:
        reflector = SchemaReflector(engine)
        reflector.reflect()
        assert "barometer" not in reflector.registry.stock
        assert "barometer" not in reflector.registry.unmapped

        # Simulate operator adding a new column between reflect() calls.
        with engine.connect() as conn:
            conn.execute(text('ALTER TABLE archive ADD COLUMN "barometer" REAL'))
            conn.commit()

        reflector.refresh()
        assert "barometer" in reflector.registry.stock
    finally:
        engine.dispose()


def test_reflect_raises_when_archive_missing() -> None:
    """reflect() raises RuntimeError when the archive table does not exist."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    try:
        reflector = SchemaReflector(engine)
        with pytest.raises(RuntimeError, match="archive.*not found"):
            reflector.reflect()
    finally:
        engine.dispose()


def test_usunits_and_interval_are_stock() -> None:
    """usUnits and interval are in the stock map (meta-columns, not extras)."""
    registry = _build_registry(["dateTime", "usUnits", "interval"])
    assert "usUnits" in registry.stock
    assert "interval" in registry.stock
