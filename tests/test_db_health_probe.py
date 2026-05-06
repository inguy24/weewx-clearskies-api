"""Unit tests for db/health.py (ADR-030, ADR-012).

Tests:
  - db_probe() returns ProbeResult(name="database", status="ok") on success.
  - db_probe() returns ProbeResult(status="unhealthy") on OperationalError.
  - db_probe() returns ProbeResult(status="unhealthy") when engine not wired.
  - wire_db_health_probe() registers db_probe with the health registry.

We test db_probe() by wiring a real in-memory SQLite engine (success case)
or a mock engine (failure case).  We do not hit a real MariaDB.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool
from unittest.mock import MagicMock, patch

from weewx_clearskies_api.db.health import db_probe, wire_db_health_probe
from weewx_clearskies_api.health import ProbeResult, _readiness_probes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    """Reset module state (engine + probe registry) between tests."""
    import weewx_clearskies_api.db.session as session_mod
    original_engine = session_mod._engine
    _readiness_probes.clear()
    yield
    session_mod._engine = original_engine
    _readiness_probes.clear()


def _make_engine():  # type: ignore[return]
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


# ---------------------------------------------------------------------------
# Tests — db_probe()
# ---------------------------------------------------------------------------


def test_probe_ok_when_engine_healthy() -> None:
    """db_probe() returns status 'ok' when SELECT 1 succeeds."""
    from weewx_clearskies_api.db.session import wire_engine
    engine = _make_engine()
    wire_engine(engine)
    try:
        result = db_probe()
        assert result.name == "database"
        assert result.status == "ok"
        assert result.messages == []
    finally:
        engine.dispose()


def test_probe_unhealthy_on_operational_error() -> None:
    """db_probe() returns status 'unhealthy' when the DB is unreachable."""
    from weewx_clearskies_api.db.session import wire_engine

    # Create an engine then patch connect() to raise OperationalError.
    engine = _make_engine()

    # Replace the engine's connect with a mock that raises OperationalError.
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = OperationalError(
        "Connection refused", None, None
    )
    wire_engine(mock_engine)

    result = db_probe()
    assert result.status == "unhealthy"
    assert result.name == "database"
    assert len(result.messages) > 0
    assert "DB connection failed" in result.messages[0]


def test_probe_unhealthy_when_engine_not_wired() -> None:
    """db_probe() returns 'unhealthy' when engine has never been wired."""
    import weewx_clearskies_api.db.session as session_mod
    session_mod._engine = None

    result = db_probe()
    assert result.status == "unhealthy"
    assert result.name == "database"
    # Message should mention the startup sequence / engine not initialised.
    assert len(result.messages) > 0


# ---------------------------------------------------------------------------
# Tests — wire_db_health_probe()
# ---------------------------------------------------------------------------


def test_wire_db_health_probe_registers_probe() -> None:
    """wire_db_health_probe() adds db_probe to the health registry."""
    assert len(_readiness_probes) == 0
    wire_db_health_probe()
    assert len(_readiness_probes) == 1


def test_wire_db_health_probe_registers_correct_function() -> None:
    """The registered probe is db_probe itself."""
    wire_db_health_probe()
    assert _readiness_probes[0] is db_probe


def test_probe_result_is_probe_result_type() -> None:
    """db_probe() always returns a ProbeResult instance."""
    from weewx_clearskies_api.db.session import wire_engine
    engine = _make_engine()
    wire_engine(engine)
    try:
        result = db_probe()
        assert isinstance(result, ProbeResult)
    finally:
        engine.dispose()
