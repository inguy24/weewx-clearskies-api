"""Unit tests for db/probe.py (ADR-012, security-baseline §3.3).

Tests:
  - Read-only user (INSERT raises OperationalError) → probe passes.
  - Writable user (INSERT succeeds) → sys.exit(1) is called.
  - SQLite engine URL missing mode=ro → sys.exit(1) is called.
  - DB unreachable (OperationalError on connect) → RuntimeError raised.

We use SQLite in-memory engines where possible, and mock engine.connect()
for cases that can't be expressed with SQLite's permissions model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

from weewx_clearskies_api.db.probe import run_write_probe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _writable_engine_with_archive():  # type: ignore[return]
    """Return a writable in-memory SQLite engine with an archive table.

    Writable == no mode=ro in URL → probe's URL check fires for SQLite dialect.
    But we also use this as the base for mock-patching the connect call.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE archive (dateTime INTEGER PRIMARY KEY)"))
        conn.commit()
    return engine


def _readonly_sqlite_engine(tmp_path: Path):  # type: ignore[return]
    """Return a read-only SQLite engine backed by a real file with mode=ro.

    We need a real file because SQLite mode=ro doesn't work with :memory:.
    """
    from sqlalchemy import create_engine as _ce

    db_file = tmp_path / "test_archive.sdb"
    # Create the archive table in the file first (writable connection).
    setup_engine = _ce(f"sqlite:///{db_file}")
    with setup_engine.connect() as conn:
        conn.execute(text("CREATE TABLE archive (dateTime INTEGER PRIMARY KEY)"))
        conn.commit()
    setup_engine.dispose()

    # Now open read-only.
    ro_engine = _ce(
        f"sqlite:////{db_file}?mode=ro&uri=true",
        connect_args={"check_same_thread": False},
    )
    return ro_engine


# ---------------------------------------------------------------------------
# Tests — SQLite URL missing mode=ro → sys.exit(1)
# (This fires before any connection attempt)
# ---------------------------------------------------------------------------


def test_sqlite_missing_mode_ro_causes_exit() -> None:
    """SQLite engine URL without ?mode=ro → probe calls sys.exit(1)."""
    engine = _writable_engine_with_archive()
    try:
        with pytest.raises(SystemExit) as exc_info:
            run_write_probe(engine)
        assert exc_info.value.code == 1
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Tests — writable user detected via mock → sys.exit(1)
# ---------------------------------------------------------------------------


def test_writable_user_causes_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """When INSERT succeeds, probe calls sys.exit(1).

    We mock a MySQL engine (so the SQLite URL check is skipped) and
    simulate a connection where the INSERT succeeds (write_succeeded=True path).
    """
    mock_engine = MagicMock()
    mock_engine.dialect.name = "mysql"
    mock_engine.url = MagicMock()
    mock_engine.url.__str__ = lambda _: "mysql+pymysql://user:pass@127.0.0.1:3306/weewx"

    # Build a mock connection that: context-manages successfully, has begin()
    # returning a transaction object, and execute() does nothing (INSERT succeeds).
    mock_trans = MagicMock()
    mock_trans.rollback = MagicMock()

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.begin = MagicMock(return_value=mock_trans)
    mock_conn.execute = MagicMock(return_value=None)  # INSERT succeeds silently

    mock_engine.connect = MagicMock(return_value=mock_conn)

    with pytest.raises(SystemExit) as exc_info:
        run_write_probe(mock_engine)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Tests — read-only user (INSERT raises) → probe passes
# ---------------------------------------------------------------------------


def test_readonly_user_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """When INSERT raises OperationalError, probe passes without calling exit."""
    mock_engine = MagicMock()
    mock_engine.dialect.name = "mysql"
    mock_engine.url = MagicMock()
    mock_engine.url.__str__ = lambda _: "mysql+pymysql://user:pass@127.0.0.1:3306/weewx"

    mock_trans = MagicMock()
    mock_trans.rollback = MagicMock()

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.begin = MagicMock(return_value=mock_trans)
    # Simulate INSERT denied — the read-only user case.
    mock_conn.execute = MagicMock(
        side_effect=OperationalError("INSERT command denied to user", None, None)
    )

    mock_engine.connect = MagicMock(return_value=mock_conn)

    # Should NOT raise SystemExit — INSERT was rejected (read-only user).
    run_write_probe(mock_engine)


def test_readonly_sqlite_file_passes(tmp_path: Path) -> None:
    """A real read-only SQLite URI passes the probe (INSERT returns error)."""
    engine = _readonly_sqlite_engine(tmp_path)
    try:
        # Should pass — mode=ro URL is present AND INSERT is blocked by SQLite.
        run_write_probe(engine)
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Tests — rollback is called even when INSERT succeeds (writable user)
# ---------------------------------------------------------------------------


def test_rollback_called_on_writable_user() -> None:
    """trans.rollback() is called before sys.exit(1) on a writable user."""
    mock_engine = MagicMock()
    mock_engine.dialect.name = "mysql"
    mock_engine.url = MagicMock()
    mock_engine.url.__str__ = lambda _: "mysql+pymysql://user:pass@127.0.0.1:3306/weewx"

    mock_trans = MagicMock()
    mock_trans.rollback = MagicMock()

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.begin = MagicMock(return_value=mock_trans)
    mock_conn.execute = MagicMock(return_value=None)  # INSERT succeeds

    mock_engine.connect = MagicMock(return_value=mock_conn)

    with pytest.raises(SystemExit):
        run_write_probe(mock_engine)

    # Rollback must be called before exit — the INSERT should not be committed.
    mock_trans.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — DB unreachable → RuntimeError raised
# ---------------------------------------------------------------------------


def test_unreachable_db_raises_runtime_error() -> None:
    """OperationalError on engine.connect() raises RuntimeError (not sys.exit)."""
    mock_engine = MagicMock()
    mock_engine.dialect.name = "mysql"
    mock_engine.url = MagicMock()
    mock_engine.url.__str__ = lambda _: "mysql+pymysql://user:pass@127.0.0.1:19999/weewx"

    mock_engine.connect.side_effect = OperationalError(
        "Can't connect to MySQL server on '127.0.0.1'", None, None
    )

    with pytest.raises(RuntimeError, match="Database unreachable during write-probe"):
        run_write_probe(mock_engine)
