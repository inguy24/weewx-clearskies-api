"""SQLAlchemy 2.x engine factory (ADR-012).

One config knob selects the backend:
  kind = sqlite  → uri mode, mandatory ?mode=ro&uri=true (ADR-012)
  kind = mysql   → MariaDB/MySQL via pymysql driver

Credentials come from env vars only (ADR-027 §3):
  WEEWX_CLEARSKIES_DB_USER
  WEEWX_CLEARSKIES_DB_PASSWORD

Connection pool (ADR-012):
  Default pool_size=5, max_overflow=10. Configurable via [database] section.

IPv4/IPv6 dual-stack (coding.md §1):
  DB host validated with ipaddress.ip_address when it looks like a bare IP.
  Hostname strings pass directly to the driver, which resolves via getaddrinfo.
  We never call gethostbyname.

Why pymysql instead of mysqlclient (MySQL-Connector-Python)?
  pymysql is a pure-Python implementation with no native build step. It works
  out of the box on every platform without a C extension and a system MySQL
  client library. mysqlclient requires a system libmysqlclient; mysql-connector
  ships its own C extension and has had licensing friction on PyPI.  pymysql is
  the standard choice for SQLAlchemy + MariaDB in small-to-medium Python
  services with no special throughput requirements.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from urllib.parse import quote_plus

from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import NullPool, QueuePool

from weewx_clearskies_api.config.settings import DatabaseSettings

logger = logging.getLogger(__name__)

# Environment-variable names per ADR-027 §3 / etc/api.conf.example.
_ENV_DB_USER = "WEEWX_CLEARSKIES_DB_USER"
_ENV_DB_PASSWORD = "WEEWX_CLEARSKIES_DB_PASSWORD"


def _validate_db_host(host: str) -> None:
    """Validate the DB host when it looks like a bare IP address.

    Uses ipaddress.ip_address (coding.md §1) which accepts both IPv4 and
    IPv6 literals.  Hostname strings are intentionally skipped — the driver
    calls getaddrinfo at connect time, which handles both families.

    Raises:
        ValueError: When the host string looks like an IP but is malformed
                    (e.g. has a typo that makes it neither a valid IP nor
                    a plausible hostname).
    """
    # Heuristic: contains only digits and dots → treat as IPv4 literal to
    # validate (catches "999.0.0.1"); contains colons → treat as IPv6 literal.
    # Anything else is a hostname — let getaddrinfo deal with it.
    stripped = host.strip("[]")  # strip brackets in case caller passed [::1]
    if stripped.replace(".", "").replace(":", "").isdigit() or ":" in stripped:
        # Looks like an IP literal — validate it.
        ipaddress.ip_address(stripped)  # raises ValueError on invalid


def _build_sqlite_url(settings: DatabaseSettings) -> str:
    """Return a read-only SQLite URL per ADR-012.

    Mandatory: ?mode=ro&uri=true on the connection string.
    SQLAlchemy's sqlite+pysqlite dialect passes the uri flag through when
    the URL contains ?uri=true — this is the standard pattern.

    The path is taken verbatim from settings.path. The caller is
    responsible for the file existing; if it doesn't, SQLite will raise
    on connection, which the probe layer catches.
    """
    # Per ADR-012: SQLite uses the URI connection string with mode=ro.
    # SQLAlchemy wants: sqlite:///absolute/path/file.db?mode=ro&uri=true
    # Use NullPool for SQLite — no point pooling connections to a file.
    path = settings.path
    return f"sqlite:////{path}?mode=ro&uri=true"


def _build_mysql_url(settings: DatabaseSettings) -> str:
    """Return a pymysql connection URL from settings + env-var credentials.

    Host is validated with ipaddress.ip_address when it looks like a bare IP
    (coding.md §1). Hostname strings pass through to the driver, which uses
    getaddrinfo internally.

    IPv6 literal hosts are wrapped in brackets in the URL per RFC 3986
    (urllib.parse does this; we do it manually here since we build the URL
    ourselves to keep the password out of logs).

    Raises:
        ValueError: Missing credentials or invalid host.
    """
    user = os.environ.get(_ENV_DB_USER, "").strip()
    password = os.environ.get(_ENV_DB_PASSWORD, "").strip()
    if not user:
        raise ValueError(
            f"Database user not set. "
            f"Set {_ENV_DB_USER} in /etc/weewx-clearskies/secrets.env (mode 0600). "
            "See INSTALL.md for the SQL GRANT required for the read-only DB user."
        )
    if not password:
        raise ValueError(
            f"Database password not set. "
            f"Set {_ENV_DB_PASSWORD} in /etc/weewx-clearskies/secrets.env (mode 0600). "
            "See INSTALL.md for setup instructions."
        )

    host = settings.host
    _validate_db_host(host)

    # Wrap IPv6 literal in brackets for the URL (RFC 3986 §3.2.2).
    try:
        addr = ipaddress.ip_address(host.strip("[]"))
        if addr.version == 6:
            host_in_url = f"[{addr.compressed}]"
        else:
            host_in_url = addr.compressed
    except ValueError:
        # Hostname string — pass through as-is.
        host_in_url = host

    # URL-encode user/password; the password in particular may contain
    # special characters.  Never log the password.
    encoded_user = quote_plus(user)
    encoded_password = quote_plus(password)
    db_name = settings.name

    return (
        f"mysql+pymysql://{encoded_user}:{encoded_password}"
        f"@{host_in_url}:{settings.port}/{db_name}"
        "?charset=utf8mb4"
    )


def build_engine(settings: DatabaseSettings) -> Engine:
    """Build and return a SQLAlchemy 2.x Engine per ADR-012.

    Args:
        settings: DatabaseSettings from the parsed config file.

    Returns:
        Configured Engine. The engine is not connected on return; the first
        actual query triggers connection checkout from the pool.

    Raises:
        ValueError: Invalid settings or missing credentials.
        sqlalchemy.exc.OperationalError: On first connection attempt if the
            DB is unreachable (propagated by the caller — probe layer).
    """
    kind = settings.kind.lower()

    if kind == "sqlite":
        url = _build_sqlite_url(settings)
        # NullPool for SQLite: file-locking semantics make persistent pools
        # unreliable and SQLite has no real connection overhead.
        engine = create_engine(
            url,
            poolclass=NullPool,
            future=True,  # SQLAlchemy 2.x behaviour
            echo=False,
        )
        logger.info(
            "SQLite engine created (read-only URI mode)",
            extra={"db_path": settings.path},
        )
        return engine

    if kind == "mysql":
        url = _build_mysql_url(settings)
        pool_size = settings.pool_size
        max_overflow = settings.max_overflow
        engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # validate connections on checkout
            future=True,
            echo=False,
        )
        # Log host/db but not credentials.
        logger.info(
            "MySQL/MariaDB engine created",
            extra={
                "db_host": settings.host,
                "db_port": settings.port,
                "db_name": settings.name,
                "pool_size": pool_size,
                "max_overflow": max_overflow,
            },
        )
        return engine

    raise ValueError(
        f"Unsupported database kind: {kind!r}. "
        "Supported values: 'sqlite', 'mysql'. "
        "Check [database] kind in api.conf."
    )
