"""Database readiness probe (ADR-030, ADR-012).

Registers a probe via register_readiness_probe() that runs SELECT 1 against
the engine.  Connection success → "ok"; failure → "unhealthy".

Deliberately narrow scope at v0.1 (task 2 brief):
  - Connection works or it doesn't.  No latency / degraded logic.
  - Latency thresholds and degraded states are out of scope for v0.1.

The probe is registered by wire_db_health_probe() which is called from
__main__.py after the engine is built and the write-probe passes.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.exc import DatabaseError, OperationalError

from weewx_clearskies_api.db.session import get_engine
from weewx_clearskies_api.health import ProbeResult, register_readiness_probe

logger = logging.getLogger(__name__)


def db_probe() -> ProbeResult:
    """Run SELECT 1 against the engine and return a ProbeResult.

    Returns:
        ProbeResult with status "ok" on success, "unhealthy" on failure.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return ProbeResult(name="database", status="ok")
    except (OperationalError, DatabaseError) as exc:
        # First line of the exception string is enough context for the health
        # body; the full stack trace is in the operator log.
        detail = str(exc).split("\n")[0]
        logger.error("Database readiness probe failed: %s", detail)
        return ProbeResult(
            name="database",
            status="unhealthy",
            messages=[f"DB connection failed: {detail}"],
        )
    except RuntimeError as exc:
        # Engine not wired yet — startup sequence bug.
        logger.error("DB probe: engine not initialised — %s", exc)
        return ProbeResult(
            name="database",
            status="unhealthy",
            messages=[str(exc)],
        )


def wire_db_health_probe() -> None:
    """Register db_probe with the health subsystem.

    Called from __main__.py startup sequence after the engine is wired.
    """
    register_readiness_probe(db_probe)
    logger.debug("Database readiness probe registered.")
