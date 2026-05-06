"""Database access layer (ADR-012, ADR-035).

Public API:
  build_engine(settings)   — db/engine.py: construct the SQLAlchemy Engine.
  wire_engine(engine)      — db/session.py: register engine for DI.
  get_db_session()         — db/session.py: FastAPI DI dependency.
  run_write_probe(engine)  — db/probe.py: abort startup if user can write.
  SchemaReflector          — db/reflection.py: MetaData.reflect + registry.
  ColumnRegistry           — db/reflection.py: result of reflection.
  db_probe()               — db/health.py: SELECT 1 readiness probe.
  wire_db_health_probe()   — db/health.py: register probe with health system.
"""

from weewx_clearskies_api.db.engine import build_engine
from weewx_clearskies_api.db.health import db_probe, wire_db_health_probe
from weewx_clearskies_api.db.probe import run_write_probe
from weewx_clearskies_api.db.reflection import ColumnRegistry, SchemaReflector
from weewx_clearskies_api.db.session import get_db_session, wire_engine

__all__ = [
    "build_engine",
    "db_probe",
    "get_db_session",
    "run_write_probe",
    "wire_db_health_probe",
    "wire_engine",
    "ColumnRegistry",
    "SchemaReflector",
]
