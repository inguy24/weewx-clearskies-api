"""Integration tests for the DB layer — ADR-012 security controls, ADR-030
health probe, and ADR-035 schema reflection registry.

Coverage map (security-baseline §3.3):
  - Read-only DB user at the database          → positive tests: RO user passes
  - Startup write-probe                         → negative tests: writable user exits non-zero
  - Parameterized queries everywhere            → covered by engine construction + reflection
    (no f-string SQL in db/ module — enforced by ruff S608 CI gate; not re-tested here)
  - Per-request session lifecycle              → session_lifecycle_* tests

Both backends run per ADR-012 CI matrix:
  BACKEND environment variable selects "mariadb" or "sqlite".
  CI matrix file (.github/workflows/test.yml) sets BACKEND for each job.
  Local dev: export BACKEND=mariadb (requires the dev/test stack up) or
             export BACKEND=sqlite  (requires seed-sqlite to have run).

Test data:
  Uses the real seed snapshot at repos/weewx-clearskies-stack/dev/snapshot/data/
  (real production columns including stock + non-stock extension columns).
  Asserts against the exact column set captured from the production weewx DB.

Markers:
  integration — all tests here are integration-level. Run with:
    uv run pytest -m integration -x
  Skip is NOT allowed for accessibility (see rules/coding.md §5) but DB
  integration tests are backend-specific; backends that aren't configured
  are skipped explicitly with a reason, not silently.

Fixtures:
  mariadb_rw_url  — writable seed user (for negative write-probe tests)
  mariadb_ro_url  — SELECT-only clearskies_ro user (for positive tests)
  sqlite_ro_url   — existing seeded SQLite file, mode=ro URI
  sqlite_rw_url   — same SQLite file without mode=ro (simulates writable)
  seeded_engine   — session-scoped Engine; both backends share the same fixture
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import DatabaseError, OperationalError

from weewx_clearskies_api.db.health import db_probe
from weewx_clearskies_api.db.probe import run_write_probe
from weewx_clearskies_api.db.reflection import SchemaReflector
from weewx_clearskies_api.db.session import wire_engine
from weewx_clearskies_api.health import ProbeResult, _readiness_probes, create_health_app

# ---------------------------------------------------------------------------
# Non-stock extension columns present in the production weewx schema.
# Source: DESCRIBE archive on the production weewx container (2026-05-06).
# These are the expected "unmapped" columns because STOCK_COLUMN_MAP does not
# include them — they were added by the AirVisual and OpenWeather extensions.
# ---------------------------------------------------------------------------
EXPECTED_NON_STOCK_COLUMNS: set[str] = {
    "aqi",
    "main_pollutant",
    "aqi_level",
    "ow_aqi",
    "ow_cloud_cover",
    "ow_co",
    "ow_nh3",
    "ow_no",
    "ow_no2",
    "ow_ozone",
    "ow_pm10",
    "ow_pm25",
    "ow_so2",
    "ow_visibility",
    "aqi_location",
}

# Stock columns that must be present in the seeded archive.
# Minimum required set for meaningful integration tests. The full set is in
# REFLECTION_STOCK_MAP; this is the "must-be-in-seed-data" subset.
EXPECTED_STOCK_COLUMNS_REQUIRED: set[str] = {
    "dateTime",
    "usUnits",
    "interval",
    "outTemp",
    "outHumidity",
    "barometer",
    "windSpeed",
    "windDir",
    "rain",
    "dewpoint",
    "radiation",
    "UV",
    "inTemp",
    "inHumidity",
}

# ---------------------------------------------------------------------------
# Environment-driven backend selection
# ---------------------------------------------------------------------------

_BACKEND = os.environ.get("BACKEND", "mariadb").lower()
_MARIADB_HOST_PORT = os.environ.get("MARIADB_HOST_PORT", "3307")
_MARIADB_DB = os.environ.get("MARIADB_DATABASE", "weewx")
_MARIADB_SEED_USER = os.environ.get("MARIADB_USER", "weewx")
_MARIADB_SEED_PASSWORD = os.environ.get("MARIADB_PASSWORD", "")
# clearskies_ro is the SELECT-only user created by mariadb-init/01-clearskies-ro.sql
_MARIADB_RO_PASSWORD = os.environ.get("MARIADB_RO_PASSWORD", "clearskies_ro_test")

# SQLite paths — these are populated by the seed-sqlite profile.
_SQLITE_DATA_VOLUME = os.environ.get("SQLITE_DATA_PATH", "/tmp/clearskies-test-sqlite")
_SQLITE_SDB_PATH = os.path.join(_SQLITE_DATA_VOLUME, "weewx.sdb")


def _skip_if_backend_not_configured(backend: str) -> None:
    """Skip the calling test if BACKEND env var does not match `backend`."""
    if _BACKEND != backend:
        pytest.skip(
            f"Skipping {backend} test: BACKEND={_BACKEND!r}. "
            f"Set BACKEND={backend} to run these tests."
        )


def _require_mariadb_password() -> None:
    if not _MARIADB_SEED_PASSWORD:
        pytest.skip(
            "Skipping MariaDB test: MARIADB_PASSWORD env var not set. "
            "Start the dev/test stack and set MARIADB_PASSWORD before running."
        )


def _require_sqlite_file() -> None:
    if not Path(_SQLITE_SDB_PATH).exists():
        pytest.skip(
            f"Skipping SQLite test: {_SQLITE_SDB_PATH} not found. "
            "Run 'docker compose --profile sqlite run --rm seed-sqlite' first."
        )


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def _mariadb_ro_url() -> str:
    return (
        f"mysql+pymysql://clearskies_ro:{_MARIADB_RO_PASSWORD}"
        f"@127.0.0.1:{_MARIADB_HOST_PORT}/{_MARIADB_DB}?charset=utf8mb4"
    )


def _mariadb_rw_url() -> str:
    return (
        f"mysql+pymysql://{_MARIADB_SEED_USER}:{_MARIADB_SEED_PASSWORD}"
        f"@127.0.0.1:{_MARIADB_HOST_PORT}/{_MARIADB_DB}?charset=utf8mb4"
    )


def _sqlite_ro_url() -> str:
    return f"sqlite:////{_SQLITE_SDB_PATH}?mode=ro&uri=true"


def _sqlite_rw_url() -> str:
    """A writable SQLite URL — same file without mode=ro."""
    return f"sqlite:////{_SQLITE_SDB_PATH}"


# ---------------------------------------------------------------------------
# Session-scoped engines (one per backend; shared across all tests in a session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mariadb_ro_engine() -> Generator[Engine, None, None]:
    """Session-scoped Engine connected as clearskies_ro (SELECT-only)."""
    _require_mariadb_password()
    engine = create_engine(_mariadb_ro_url(), future=True, pool_pre_ping=True)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def sqlite_ro_engine() -> Generator[Engine, None, None]:
    """Session-scoped Engine connected to the seeded SQLite file in read-only mode."""
    _require_sqlite_file()
    from sqlalchemy.pool import NullPool

    engine = create_engine(
        _sqlite_ro_url(),
        poolclass=NullPool,
        future=True,
    )
    yield engine
    engine.dispose()


# ---------------------------------------------------------------------------
# Probe registry isolation fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def _clear_probe_registry() -> Generator[None, None, None]:
    """Clear the health probe registry around each test that wires DB probes."""
    _readiness_probes.clear()
    yield
    _readiness_probes.clear()


# ---------------------------------------------------------------------------
# NEGATIVE TESTS: write-probe should refuse to start
# ---------------------------------------------------------------------------


class TestWriteProbeRefusesStartupOnWritableUser:
    """Verify that run_write_probe() calls sys.exit(1) when the user can write.

    BUG REPORT (route to api-dev): The probe uses
      INSERT INTO archive (dateTime) VALUES (:ts)
    which fails with IntegrityError (NOT NULL constraint on usUnits/interval) on
    the real production schema. Since the probe catches ALL DatabaseError subclasses
    (including IntegrityError) as "INSERT denied = good", it silently passes for a
    writable user against the real multi-column NOT NULL schema.

    The correct fix is for the probe to catch IntegrityError separately as
    "constraint violation = user DOES have write access" rather than treating it the
    same as "privilege denied = user lacks write access."

    In the meantime, the MariaDB negative test below uses an in-memory SQLite backend
    with a single-column archive table to exercise the write_succeeded=True path that
    the probe should reach for writable users.
    """

    def test_writable_user_probe_exits_nonzero_on_minimal_schema(self) -> None:
        """Writable user on a minimal archive table → run_write_probe raises SystemExit.

        Uses an in-memory SQLite with only `dateTime NOT NULL` in archive, so the
        probe's INSERT succeeds (no other NOT NULL columns to trip a constraint
        violation). Verifies the probe's write_succeeded path calls sys.exit(1).
        """
        from sqlalchemy.pool import NullPool

        # Create an in-memory writable SQLite DB with the minimal archive schema
        # the probe expects (only dateTime). This is the schema that lets the
        # probe's INSERT actually succeed, reaching write_succeeded=True.
        writable_engine = create_engine("sqlite:///:memory:", poolclass=NullPool, future=True)
        with writable_engine.connect() as conn:
            conn.execute(text("CREATE TABLE archive (dateTime INTEGER NOT NULL PRIMARY KEY)"))
            conn.commit()

        try:
            with pytest.raises(SystemExit) as exc_info:
                run_write_probe(writable_engine)
            assert exc_info.value.code == 1, (
                "run_write_probe must exit with code 1 when write access is detected"
            )
        finally:
            writable_engine.dispose()

    def test_sqlite_without_mode_ro_probe_exits_nonzero(self) -> None:
        """SQLite without mode=ro in the URL → run_write_probe raises SystemExit.

        The probe checks for 'mode=ro' in the URL before attempting the
        INSERT. A URL without it exits immediately (defense-in-depth).
        """
        _require_sqlite_file()
        from sqlalchemy.pool import NullPool

        engine = create_engine(
            _sqlite_rw_url(),
            poolclass=NullPool,
            future=True,
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                run_write_probe(engine)
            assert exc_info.value.code == 1, (
                "run_write_probe must exit with code 1 when SQLite URL lacks mode=ro"
            )
        finally:
            engine.dispose()

    @pytest.mark.xfail(
        reason=(
            "KNOWN PROBE BUG (route to api-dev): run_write_probe silently passes for "
            "a writable MariaDB user when the archive table has usUnits+interval as "
            "NOT NULL without defaults. The probe's INSERT INTO archive (dateTime) "
            "fails with IntegrityError (constraint violation) which is incorrectly "
            "treated as 'privilege denied'. Fix: catch IntegrityError separately as "
            "'user has write access'. Tracking: Phase 2 task 2 closeout."
        ),
        strict=True,
    )
    def test_mariadb_writable_seed_user_probe_exits_nonzero_known_bug(self) -> None:
        """Documents that the probe silently passes for a writable MariaDB user.

        This test is marked xfail because it FAILS (the probe does NOT exit
        when it should). The failure mode: probe's INSERT fails with IntegrityError
        (NOT NULL violation) rather than privilege-denied, so probe treats it as
        read-only and accepts startup. The test will turn green once api-dev fixes
        the probe to distinguish IntegrityError from OperationalError 1142.
        """
        _require_mariadb_password()
        engine = create_engine(_mariadb_rw_url(), future=True, pool_pre_ping=True)
        try:
            # This should raise SystemExit(1) but currently doesn't.
            with pytest.raises(SystemExit):
                run_write_probe(engine)
        finally:
            engine.dispose()


# ---------------------------------------------------------------------------
# POSITIVE TESTS: write-probe should accept startup for read-only users
# ---------------------------------------------------------------------------


class TestWriteProbeAcceptsReadOnlyUser:
    """Verify that run_write_probe() passes cleanly for a SELECT-only user."""

    def test_mariadb_clearskies_ro_user_probe_passes(
        self, mariadb_ro_engine: Engine
    ) -> None:
        """clearskies_ro user (GRANT SELECT only) → run_write_probe returns normally."""
        _skip_if_backend_not_configured("mariadb")
        # Must NOT raise SystemExit or any exception.
        run_write_probe(mariadb_ro_engine)

    def test_sqlite_mode_ro_uri_probe_passes(self, sqlite_ro_engine: Engine) -> None:
        """SQLite with mode=ro&uri=true → run_write_probe returns normally."""
        _skip_if_backend_not_configured("sqlite")
        # Must NOT raise SystemExit.  The INSERT attempt should fail with a
        # write-protection error from SQLite, which the probe interprets as
        # "user has no write access — correct."
        run_write_probe(sqlite_ro_engine)


# ---------------------------------------------------------------------------
# HEALTH PROBE DB CONNECTIVITY
# ---------------------------------------------------------------------------


class TestDbReadinessProbe:
    """Verify /health/ready reflects DB connectivity per ADR-030."""

    def test_mariadb_health_ready_returns_200_with_database_ok(
        self, mariadb_ro_engine: Engine, _clear_probe_registry: None
    ) -> None:
        """/health/ready returns 200 with database:ok when MariaDB is reachable."""
        _skip_if_backend_not_configured("mariadb")
        wire_engine(mariadb_ro_engine)
        from weewx_clearskies_api.db.health import wire_db_health_probe

        wire_db_health_probe()

        client = TestClient(create_health_app(), raise_server_exceptions=False)
        response = client.get("/health/ready")

        assert response.status_code == 200, (
            f"Expected 200 when MariaDB is healthy; got {response.status_code}"
        )
        body = response.json()
        assert "checks" in body
        assert "database" in body["checks"], (
            "/health/ready body must include a 'database' check"
        )
        assert body["checks"]["database"]["status"] == "ok", (
            f"Expected database:ok when MariaDB reachable; got {body['checks']['database']}"
        )

    def test_sqlite_health_ready_returns_200_with_database_ok(
        self, sqlite_ro_engine: Engine, _clear_probe_registry: None
    ) -> None:
        """/health/ready returns 200 with database:ok when SQLite file is readable."""
        _skip_if_backend_not_configured("sqlite")
        wire_engine(sqlite_ro_engine)
        from weewx_clearskies_api.db.health import wire_db_health_probe

        wire_db_health_probe()

        client = TestClient(create_health_app(), raise_server_exceptions=False)
        response = client.get("/health/ready")

        assert response.status_code == 200
        body = response.json()
        assert body["checks"]["database"]["status"] == "ok"

    def test_health_ready_returns_503_when_engine_unreachable(
        self, _clear_probe_registry: None
    ) -> None:
        """/health/ready returns 503 when the DB engine cannot connect.

        Simulates a downed DB by wiring an engine that points at a non-existent
        host. The probe's SELECT 1 will fail, flipping the check to 'unhealthy'.
        """
        from sqlalchemy.pool import NullPool

        # Port 19999 on loopback: unlikely to have anything listening.
        unreachable_engine = create_engine(
            "mysql+pymysql://nobody:nopass@127.0.0.1:19999/nodb"
            "?connect_timeout=1&charset=utf8mb4",
            poolclass=NullPool,
            future=True,
        )
        wire_engine(unreachable_engine)
        from weewx_clearskies_api.db.health import wire_db_health_probe

        wire_db_health_probe()

        client = TestClient(create_health_app(), raise_server_exceptions=False)
        response = client.get("/health/ready")

        assert response.status_code == 503, (
            f"Expected 503 when DB unreachable; got {response.status_code}"
        )
        body = response.json()
        assert body["checks"]["database"]["status"] == "unhealthy"

    def test_db_probe_function_returns_ok_when_engine_healthy(
        self, mariadb_ro_engine: Engine
    ) -> None:
        """db_probe() returns ProbeResult(status='ok') when DB is reachable."""
        _skip_if_backend_not_configured("mariadb")
        wire_engine(mariadb_ro_engine)
        result = db_probe()
        assert isinstance(result, ProbeResult)
        assert result.name == "database"
        assert result.status == "ok"

    def test_sqlite_db_probe_function_returns_ok_when_file_readable(
        self, sqlite_ro_engine: Engine
    ) -> None:
        """db_probe() returns ProbeResult(status='ok') for readable SQLite file."""
        _skip_if_backend_not_configured("sqlite")
        wire_engine(sqlite_ro_engine)
        result = db_probe()
        assert result.status == "ok"

    def test_db_probe_returns_unhealthy_when_engine_not_wired(
        self, _clear_probe_registry: None
    ) -> None:
        """db_probe() returns unhealthy when wire_engine() was never called."""
        import weewx_clearskies_api.db.session as session_module  # noqa: PLC0415

        original_engine = session_module._engine  # noqa: SLF001
        session_module._engine = None  # noqa: SLF001
        try:
            result = db_probe()
            assert result.status == "unhealthy", (
                "db_probe must return unhealthy when engine is not wired"
            )
        finally:
            session_module._engine = original_engine  # noqa: SLF001


# ---------------------------------------------------------------------------
# SCHEMA REFLECTION REGISTRY (ADR-035)
# ---------------------------------------------------------------------------


class TestSchemaReflectionRegistry:
    """Verify the column registry shape after reflecting the seeded archive."""

    def test_mariadb_registry_contains_expected_stock_columns(
        self, mariadb_ro_engine: Engine
    ) -> None:
        """MariaDB registry maps the required stock weewx columns to canonical names."""
        _skip_if_backend_not_configured("mariadb")
        reflector = SchemaReflector(mariadb_ro_engine)
        registry = reflector.reflect()

        for col in EXPECTED_STOCK_COLUMNS_REQUIRED:
            assert col in registry.stock, (
                f"Expected stock column {col!r} in registry.stock after reflection"
            )
            info = registry.stock[col]
            assert info.is_stock is True
            assert info.canonical_name is not None
            assert info.db_name == col

    def test_mariadb_registry_non_stock_columns_are_flagged_unmapped(
        self, mariadb_ro_engine: Engine
    ) -> None:
        """Non-stock extension columns (aqi, ow_*, etc.) appear in registry.unmapped."""
        _skip_if_backend_not_configured("mariadb")
        reflector = SchemaReflector(mariadb_ro_engine)
        registry = reflector.reflect()

        for col in EXPECTED_NON_STOCK_COLUMNS:
            assert col in registry.unmapped, (
                f"Expected non-stock column {col!r} in registry.unmapped — "
                "it should not be in STOCK_COLUMN_MAP for task 2"
            )
            info = registry.unmapped[col]
            assert info.is_stock is False
            assert info.canonical_name is None, (
                f"Non-stock column {col!r} must have canonical_name=None at task 2"
            )

    def test_sqlite_registry_contains_expected_stock_columns(
        self, sqlite_ro_engine: Engine
    ) -> None:
        """SQLite registry maps required stock columns (same dataset, different backend)."""
        _skip_if_backend_not_configured("sqlite")
        reflector = SchemaReflector(sqlite_ro_engine)
        registry = reflector.reflect()

        for col in EXPECTED_STOCK_COLUMNS_REQUIRED:
            assert col in registry.stock, (
                f"SQLite: expected stock column {col!r} in registry.stock"
            )

    def test_sqlite_registry_non_stock_columns_are_flagged_unmapped(
        self, sqlite_ro_engine: Engine
    ) -> None:
        """SQLite: non-stock extension columns appear in registry.unmapped."""
        _skip_if_backend_not_configured("sqlite")
        reflector = SchemaReflector(sqlite_ro_engine)
        registry = reflector.reflect()

        for col in EXPECTED_NON_STOCK_COLUMNS:
            assert col in registry.unmapped, (
                f"SQLite: expected non-stock column {col!r} in registry.unmapped"
            )

    def test_mariadb_all_columns_partitioned_exhaustively(
        self, mariadb_ro_engine: Engine
    ) -> None:
        """Every reflected column appears in exactly one of stock or unmapped."""
        _skip_if_backend_not_configured("mariadb")
        reflector = SchemaReflector(mariadb_ro_engine)
        registry = reflector.reflect()

        stock_names = set(registry.stock.keys())
        unmapped_names = set(registry.unmapped.keys())
        assert stock_names.isdisjoint(unmapped_names), (
            "A column must not appear in both registry.stock and registry.unmapped"
        )

        all_columns = {info.db_name for info in registry.all_columns()}
        assert all_columns == stock_names | unmapped_names, (
            "registry.all_columns() must equal stock ∪ unmapped"
        )

    def test_mariadb_datetime_maps_to_canonical_timestamp(
        self, mariadb_ro_engine: Engine
    ) -> None:
        """dateTime (epoch int) maps to canonical 'timestamp' per ADR-010."""
        _skip_if_backend_not_configured("mariadb")
        reflector = SchemaReflector(mariadb_ro_engine)
        registry = reflector.reflect()

        assert "dateTime" in registry.stock
        assert registry.stock["dateTime"].canonical_name == "timestamp", (
            "dateTime must map to canonical 'timestamp' per STOCK_COLUMN_MAP"
        )

    def test_mariadb_reflect_refresh_returns_same_registry(
        self, mariadb_ro_engine: Engine
    ) -> None:
        """refresh() re-runs reflection and returns a consistent ColumnRegistry."""
        _skip_if_backend_not_configured("mariadb")
        reflector = SchemaReflector(mariadb_ro_engine)
        registry1 = reflector.reflect()
        registry2 = reflector.refresh()

        assert set(registry2.stock.keys()) == set(registry1.stock.keys()), (
            "refresh() must return the same set of stock columns as initial reflect()"
        )
        assert set(registry2.unmapped.keys()) == set(registry1.unmapped.keys()), (
            "refresh() must return the same set of unmapped columns as initial reflect()"
        )

    def test_reflection_fails_gracefully_on_missing_archive_table(self) -> None:
        """SchemaReflector.reflect() raises RuntimeError when archive table missing."""
        # Use an in-memory SQLite DB (no tables).
        from sqlalchemy.pool import NullPool

        empty_engine = create_engine("sqlite:///:memory:", poolclass=NullPool, future=True)
        reflector = SchemaReflector(empty_engine)

        with pytest.raises(RuntimeError, match="archive"):
            reflector.reflect()

        empty_engine.dispose()


# ---------------------------------------------------------------------------
# PER-REQUEST SESSION LIFECYCLE (ADR-012)
# ---------------------------------------------------------------------------


class TestPerRequestSessionLifecycle:
    """Verify that sessions are closed after each request and don't leak.

    The session DI (get_db_session) yields one Session per request and closes
    it in the finally block.  We simulate multiple sequential request cycles
    and verify the pool returns to its idle state.
    """

    def test_mariadb_session_closed_after_each_use(
        self, mariadb_ro_engine: Engine
    ) -> None:
        """Each manually-managed session cycle leaves the pool fully idle."""
        _skip_if_backend_not_configured("mariadb")
        from weewx_clearskies_api.db.session import get_db_session

        wire_engine(mariadb_ro_engine)

        for request_cycle in range(3):
            # Simulate the FastAPI DI lifecycle: iterate the generator.
            gen = get_db_session()
            session = next(gen)
            try:
                # Use the session — a simple SELECT 1 is enough.
                session.execute(text("SELECT 1"))
            finally:
                # Simulate FastAPI's generator teardown (StopIteration path).
                try:
                    next(gen)
                except StopIteration:
                    pass

            # After the generator is exhausted the session should be closed.
            # A closed Session raises when you try to execute on it.
            # We use the pool's checked-out count (0 = no leaks) as the check.
            if hasattr(mariadb_ro_engine.pool, "checkedout"):
                checked_out = mariadb_ro_engine.pool.checkedout()  # type: ignore[attr-defined]
                assert checked_out == 0, (
                    f"Request cycle {request_cycle}: pool has {checked_out} checked-out "
                    "connections after session teardown — session leaked"
                )

    def test_sqlite_session_closed_after_each_use(
        self, sqlite_ro_engine: Engine
    ) -> None:
        """SQLite: each session cycle leaves no leaked connections."""
        _skip_if_backend_not_configured("sqlite")
        from weewx_clearskies_api.db.session import get_db_session

        wire_engine(sqlite_ro_engine)

        # SQLite NullPool doesn't pool — just verify no exceptions during the
        # full generator lifecycle.
        for _ in range(3):
            gen = get_db_session()
            session = next(gen)
            try:
                session.execute(text("SELECT 1"))
            finally:
                try:
                    next(gen)
                except StopIteration:
                    pass

    def test_session_is_closed_even_when_query_raises(
        self, mariadb_ro_engine: Engine
    ) -> None:
        """The finally block in get_db_session closes the session even on exception."""
        _skip_if_backend_not_configured("mariadb")
        from weewx_clearskies_api.db.session import get_db_session

        wire_engine(mariadb_ro_engine)

        gen = get_db_session()
        session = next(gen)

        # Force an error (query a non-existent table).
        with pytest.raises((OperationalError, DatabaseError)):
            session.execute(text("SELECT * FROM clearskies_nonexistent_table_xyz"))

        # Now exhaust the generator — the finally block must run.
        try:
            next(gen)
        except StopIteration:
            pass

        # Pool must be clean regardless of the query error.
        if hasattr(mariadb_ro_engine.pool, "checkedout"):
            checked_out = mariadb_ro_engine.pool.checkedout()  # type: ignore[attr-defined]
            assert checked_out == 0, (
                "Session must be closed even when a query raises — no pool leak"
            )


# ---------------------------------------------------------------------------
# ARCHIVE DATA SANITY: verify the seed loaded real rows
# ---------------------------------------------------------------------------


class TestSeededArchiveData:
    """Verify the seeded archive table has the expected real-data characteristics.

    These tests exercise the read path and ensure we're testing against actual
    realistic data, not an empty table.
    """

    def test_mariadb_archive_has_rows(self, mariadb_ro_engine: Engine) -> None:
        """Archive table contains at least 1 row after seeding."""
        _skip_if_backend_not_configured("mariadb")
        with mariadb_ro_engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM archive"))
            count = result.scalar()
        assert count is not None and count > 0, (
            "Seeded archive table must have at least 1 row"
        )

    def test_sqlite_archive_has_rows(self, sqlite_ro_engine: Engine) -> None:
        """SQLite: archive table contains at least 1 row after seeding."""
        _skip_if_backend_not_configured("sqlite")
        with sqlite_ro_engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM archive"))
            count = result.scalar()
        assert count is not None and count > 0

    def test_mariadb_archive_rows_have_realistic_temperature_range(
        self, mariadb_ro_engine: Engine
    ) -> None:
        """Archive outTemp values fall within a physically plausible range (°F).

        Production data from Huntington Beach, CA. Temperature should be
        between 0°F (-18°C) and 130°F (54°C) for any legitimate observation.
        """
        _skip_if_backend_not_configured("mariadb")
        with mariadb_ro_engine.connect() as conn:
            result = conn.execute(
                text("SELECT MIN(outTemp), MAX(outTemp) FROM archive WHERE outTemp IS NOT NULL")
            )
            row = result.fetchone()

        if row is None or row[0] is None:
            pytest.skip("No outTemp values in seeded archive — cannot check range")

        min_temp, max_temp = float(row[0]), float(row[1])
        assert min_temp >= 0.0, (
            f"Min outTemp {min_temp}°F is below plausible range — suspicious seed data"
        )
        assert max_temp <= 130.0, (
            f"Max outTemp {max_temp}°F is above plausible range — suspicious seed data"
        )

    def test_mariadb_archive_has_extension_column_values(
        self, mariadb_ro_engine: Engine
    ) -> None:
        """AirVisual extension column 'aqi' has at least one non-null value in seed."""
        _skip_if_backend_not_configured("mariadb")
        with mariadb_ro_engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM archive WHERE aqi IS NOT NULL")
            )
            aqi_count = result.scalar()

        assert aqi_count is not None and aqi_count > 0, (
            "Seeded archive must have at least one non-null aqi value — "
            "real production data from the AirVisual extension includes this"
        )

    def test_mariadb_archive_extension_string_columns_have_values(
        self, mariadb_ro_engine: Engine
    ) -> None:
        """String extension columns (main_pollutant, aqi_level) have non-null rows."""
        _skip_if_backend_not_configured("mariadb")
        with mariadb_ro_engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT COUNT(*) FROM archive "
                    "WHERE main_pollutant IS NOT NULL AND aqi_level IS NOT NULL"
                )
            )
            count = result.scalar()

        assert count is not None and count > 0, (
            "Seeded archive must have non-null main_pollutant and aqi_level values"
        )
