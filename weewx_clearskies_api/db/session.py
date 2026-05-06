"""Per-request SQLAlchemy session via FastAPI dependency injection (ADR-012).

The session is a SQLAlchemy 2.x Session (not AsyncSession) — weewx archive
reads are synchronous, and mixing sync + async SQLAlchemy adds complexity for
no benefit at v0.1.

Usage in an endpoint:
    from weewx_clearskies_api.db.session import get_db_session
    from sqlalchemy.orm import Session

    @router.get("/some-path")
    def my_endpoint(db: Session = Depends(get_db_session)):
        ...

The dependency holds one session per HTTP request and closes it in the
finally block regardless of success or exception. No long-lived sessions
escape into endpoint code.

The engine is module-level state set by wire_engine() once at startup. This
avoids importing the engine at module load time (tests may need to wire a
mock engine before the real one exists) while keeping the DI function free of
settings imports.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# Set by wire_engine() during startup — never directly by endpoint code.
_engine: Engine | None = None
# Session factory — rebuilt whenever wire_engine() is called.
_SessionLocal: sessionmaker[Session] | None = None


def wire_engine(engine: Engine) -> None:
    """Register the engine for use by get_db_session.

    Builds a sessionmaker bound to the engine and stores it module-level.
    Called once from __main__.py after the engine is built and the write-probe
    passes.  Tests may call this with a mock / in-memory engine.
    """
    global _engine, _SessionLocal  # noqa: PLW0603 — intentional module-level registry
    _engine = engine
    # SQLAlchemy 2.x: pass engine as first positional arg (bind= removed in 2.0).
    _SessionLocal = sessionmaker(engine, autobegin=True, autoflush=False)


def get_engine() -> Engine:
    """Return the registered engine.

    Raises:
        RuntimeError: Engine has not been wired (startup sequence bug).
    """
    if _engine is None:
        raise RuntimeError(
            "Database engine is not initialised. "
            "wire_engine() must be called before the first request. "
            "This is a startup-sequence bug — check __main__.py."
        )
    return _engine


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency: yield one Session per request, close on teardown.

    Always yields a Session from the wired engine.  The finally block runs
    after the response is sent (standard FastAPI DI teardown), closing the
    session and returning the connection to the pool.
    """
    if _SessionLocal is None:
        raise RuntimeError(
            "Database session factory is not initialised. "
            "wire_engine() must be called before the first request. "
            "This is a startup-sequence bug — check __main__.py."
        )
    session: Session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()
