"""Startup write-probe per ADR-012 / security-baseline §3.3.

Purpose:
    Verify that the DB user connected to the archive table has NO write
    privileges. If a write succeeds, the service refuses to start (log
    critical + sys.exit(1)). This is defense-in-depth: the DB-level GRANT
    is the primary control; this probe is the second layer.

Approach — INSERT inside an explicit ROLLBACK transaction:
    We attempt an INSERT into the archive table inside a transaction, then
    unconditionally call conn.rollback(). This approach:

    1. Works for both SQLite and MariaDB — no dialect-specific introspection
       of INFORMATION_SCHEMA.PRIVILEGES (which varies across MariaDB versions
       and requires additional grants to query reliably).
    2. Does NOT leave a sentinel row behind — the transaction is always
       rolled back, whether the INSERT succeeded or failed.
    3. Detects write access accurately: a truly read-only user (no INSERT
       privilege) will raise an OperationalError or ProgrammingError; a
       writable user will successfully execute the INSERT before we roll back.

    Alternative considered: CREATE TABLE clearskies_probe_sentinel + DROP.
    Rejected because CREATE TABLE requires DDL privileges (GRANT CREATE), not
    DML INSERT, and a weewx DB user with only SELECT+INSERT would pass the
    INSERT test but fail on CREATE TABLE — that's the right semantic even if
    the alternative seems cleaner. The INSERT+ROLLBACK is the minimal test for
    what ADR-012 actually forbids: INSERT/UPDATE/DELETE on archive data.

SQLite special case (ADR-012):
    For SQLite, the URI must carry ?mode=ro. We also run the INSERT attempt —
    SQLite's mode=ro enforcement returns a write-protection error before any
    INSERT reaches the table. Both checks run.
"""

from __future__ import annotations

import logging
import sys

from sqlalchemy import Engine, text
from sqlalchemy.exc import DatabaseError, OperationalError, ProgrammingError

logger = logging.getLogger(__name__)

# A harmless sentinel row we attempt to INSERT.  We use a dummy dateTime that
# is far in the future (year 9999) so that if something goes catastrophically
# wrong and the ROLLBACK somehow fails, the row can be identified and purged.
# The archive table always has at minimum a dateTime column — no other columns
# are assumed, avoiding breakage on stripped-down schemas.
_PROBE_INSERT_SQL = text(
    "INSERT INTO archive (dateTime) VALUES (:ts)"
)
_PROBE_DATETIME_VALUE = 253402300800  # 9999-12-31 00:00:00 UTC (epoch seconds)


def run_write_probe(engine: Engine) -> None:
    """Attempt a write against the archive table and abort startup if it succeeds.

    Connects to the database, opens an explicit transaction, attempts an INSERT
    into the archive table, then ALWAYS rolls back. If the INSERT succeeded the
    connected user has write privileges — log critical and exit.

    For SQLite: also checks that the connection URL contains mode=ro.

    Args:
        engine: The SQLAlchemy Engine to probe.

    Side-effects:
        Calls sys.exit(1) if write access is detected. Logs critical message
        with explicit operator instructions before exiting.
    """
    # --- SQLite URI check ---------------------------------------------------
    # For SQLite, ADR-012 requires mode=ro in the URL. Extract the URL from
    # the engine and verify it before attempting the connection.
    db_url = str(engine.url)
    if engine.dialect.name == "sqlite":
        # The URL must contain mode=ro.  The engine factory enforces this at
        # build time, but we verify here as defense-in-depth.
        if "mode=ro" not in db_url:
            logger.critical(
                "FATAL: SQLite database URL does not contain '?mode=ro'. "
                "The engine was built without the read-only URI parameter "
                "required by ADR-012. "
                "Edit [database] path in api.conf to point at the .sdb file; "
                "the engine factory adds mode=ro automatically. "
                "See INSTALL.md for setup instructions. "
                "Service will not start."
            )
            sys.exit(1)

    # --- Write attempt ------------------------------------------------------
    write_succeeded = False
    error_detail: str = ""

    try:
        with engine.connect() as conn:
            # Use begin_nested() or explicit transaction + rollback pattern.
            # We call rollback() unconditionally in the finally block to ensure
            # nothing is committed regardless of outcome.
            trans = conn.begin()
            try:
                conn.execute(_PROBE_INSERT_SQL, {"ts": _PROBE_DATETIME_VALUE})
                # If we reach here, the INSERT was accepted by the DB engine.
                write_succeeded = True
            except (OperationalError, ProgrammingError, DatabaseError) as exc:
                # Expected for a properly read-only user.  The exception means
                # "INSERT was denied" — that's exactly what we want.
                error_detail = str(exc).split("\n")[0]  # first line only
            finally:
                # Unconditional rollback — never commit.  If the INSERT
                # succeeded on a writable user, this cleans it up before we
                # log and exit.  If the INSERT failed, this is a no-op.
                try:
                    trans.rollback()
                except (OperationalError, DatabaseError):
                    # Rollback can fail on a dead connection — acceptable here
                    # since we're already in the failure/exit path.
                    pass
    except (OperationalError, DatabaseError) as exc:
        # Connection itself failed — the DB is unreachable.  That's a
        # readiness issue, not a permissions issue.  Raise so the caller can
        # surface it as a startup error distinct from write-access detection.
        raise RuntimeError(
            f"Database unreachable during write-probe: {exc}"
        ) from exc

    if write_succeeded:
        # The INSERT succeeded, meaning the connected user has INSERT privilege.
        # ADR-012 / security-baseline §3.3: abort startup immediately.
        logger.critical(
            "FATAL: The database user has write access to the archive table. "
            "clearskies-api must connect with a SELECT-only user. "
            "The INSERT probe succeeded — the connected user has INSERT (and "
            "possibly UPDATE/DELETE/DROP) privileges on the archive table. "
            "Action required: "
            "(1) Create a read-only database user per INSTALL.md, section "
            "'Database — read-only user setup'. "
            "(2) Set WEEWX_CLEARSKIES_DB_USER and WEEWX_CLEARSKIES_DB_PASSWORD "
            "in /etc/weewx-clearskies/secrets.env (mode 0600) to the new user. "
            "(3) Restart the service. "
            "Service will not start until a read-only user is configured."
        )
        sys.exit(1)

    # Write was rejected — user is read-only.  Good.
    logger.info(
        "Write-probe passed: DB user has no INSERT privilege on archive table.",
        extra={"probe_rejection_detail": error_detail},
    )
