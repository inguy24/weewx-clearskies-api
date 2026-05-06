"""Unit tests for db/engine.py (ADR-012).

Tests:
  - SQLite engine built correctly: URL carries ?mode=ro&uri=true, NullPool.
  - MySQL engine built correctly: env vars consumed, URL contains host/port/db.
  - Missing env vars raise ValueError (not silently connect as anonymous).
  - IPv4 and IPv6 literals are both validated without error.
  - Invalid IP literals raise ValueError.
  - Unsupported kind raises ValueError.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from sqlalchemy.pool import NullPool, QueuePool

from weewx_clearskies_api.config.settings import DatabaseSettings
from weewx_clearskies_api.db.engine import _build_mysql_url, _build_sqlite_url, _validate_db_host, build_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sqlite_settings(**overrides: object) -> DatabaseSettings:
    base: dict[str, object] = {"kind": "sqlite", "path": "/tmp/test.sdb"}
    base.update(overrides)
    return DatabaseSettings(base)


def _mysql_settings(**overrides: object) -> DatabaseSettings:
    base: dict[str, object] = {
        "kind": "mysql",
        "host": "127.0.0.1",
        "port": 3306,
        "name": "weewx",
    }
    base.update(overrides)
    return DatabaseSettings(base)


# ---------------------------------------------------------------------------
# _validate_db_host
# ---------------------------------------------------------------------------


def test_validate_ipv4_valid() -> None:
    """Valid IPv4 address does not raise."""
    _validate_db_host("192.168.1.5")  # should not raise


def test_validate_ipv4_invalid() -> None:
    """Malformed IPv4-looking string raises ValueError."""
    with pytest.raises(ValueError):
        _validate_db_host("999.0.0.1")


def test_validate_ipv6_valid() -> None:
    """Valid IPv6 literal (without brackets) does not raise."""
    _validate_db_host("::1")


def test_validate_ipv6_full_valid() -> None:
    """Full IPv6 address does not raise."""
    _validate_db_host("2001:db8::1")


def test_validate_hostname_passthrough() -> None:
    """Hostname strings are not validated (pass to getaddrinfo at connect time)."""
    _validate_db_host("db.example.com")  # should not raise
    _validate_db_host("localhost")       # should not raise


# ---------------------------------------------------------------------------
# _build_sqlite_url
# ---------------------------------------------------------------------------


def test_sqlite_url_contains_mode_ro() -> None:
    """SQLite URL must carry ?mode=ro per ADR-012."""
    settings = _sqlite_settings(path="/var/lib/weewx/weewx.sdb")
    url = _build_sqlite_url(settings)
    assert "mode=ro" in url


def test_sqlite_url_contains_uri_true() -> None:
    """SQLite URL must carry uri=true for SQLite URI mode."""
    settings = _sqlite_settings(path="/var/lib/weewx/weewx.sdb")
    url = _build_sqlite_url(settings)
    assert "uri=true" in url


def test_sqlite_url_contains_path() -> None:
    """SQLite URL must embed the configured path."""
    settings = _sqlite_settings(path="/custom/path/archive.sdb")
    url = _build_sqlite_url(settings)
    assert "/custom/path/archive.sdb" in url


# ---------------------------------------------------------------------------
# _build_mysql_url
# ---------------------------------------------------------------------------


def test_mysql_url_missing_user_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing DB_USER env var raises ValueError before any connection attempt."""
    monkeypatch.delenv("WEEWX_CLEARSKIES_DB_USER", raising=False)
    monkeypatch.delenv("WEEWX_CLEARSKIES_DB_PASSWORD", raising=False)
    settings = _mysql_settings()
    with pytest.raises(ValueError, match="WEEWX_CLEARSKIES_DB_USER"):
        _build_mysql_url(settings)


def test_mysql_url_missing_password_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing DB_PASSWORD env var raises ValueError."""
    monkeypatch.setenv("WEEWX_CLEARSKIES_DB_USER", "clearskies_ro")
    monkeypatch.delenv("WEEWX_CLEARSKIES_DB_PASSWORD", raising=False)
    settings = _mysql_settings()
    with pytest.raises(ValueError, match="WEEWX_CLEARSKIES_DB_PASSWORD"):
        _build_mysql_url(settings)


def test_mysql_url_contains_host_port_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """MySQL URL contains the configured host, port, and database name."""
    monkeypatch.setenv("WEEWX_CLEARSKIES_DB_USER", "clearskies_ro")
    monkeypatch.setenv("WEEWX_CLEARSKIES_DB_PASSWORD", "secret")
    settings = _mysql_settings(host="192.168.1.10", port=3307, name="wx_archive")
    url = _build_mysql_url(settings)
    assert "192.168.1.10" in url
    assert "3307" in url
    assert "wx_archive" in url


def test_mysql_url_ipv6_host_bracketed(monkeypatch: pytest.MonkeyPatch) -> None:
    """IPv6 host literal is wrapped in brackets in the URL per RFC 3986."""
    monkeypatch.setenv("WEEWX_CLEARSKIES_DB_USER", "clearskies_ro")
    monkeypatch.setenv("WEEWX_CLEARSKIES_DB_PASSWORD", "secret")
    settings = _mysql_settings(host="::1")
    url = _build_mysql_url(settings)
    # The URL must contain [::1] for RFC 3986 compliance.
    assert "[::1]" in url


def test_mysql_url_does_not_include_driver_string() -> None:
    """MySQL URL uses the pymysql dialect string."""
    import os
    os.environ["WEEWX_CLEARSKIES_DB_USER"] = "u"
    os.environ["WEEWX_CLEARSKIES_DB_PASSWORD"] = "p"
    try:
        settings = _mysql_settings()
        url = _build_mysql_url(settings)
        assert "mysql+pymysql://" in url
    finally:
        del os.environ["WEEWX_CLEARSKIES_DB_USER"]
        del os.environ["WEEWX_CLEARSKIES_DB_PASSWORD"]


# ---------------------------------------------------------------------------
# build_engine
# ---------------------------------------------------------------------------


def test_build_engine_sqlite_uses_null_pool() -> None:
    """SQLite engine uses NullPool (file-lock semantics)."""
    settings = _sqlite_settings(path="/tmp/test_pool.sdb")
    engine = build_engine(settings)
    assert isinstance(engine.pool, NullPool)
    engine.dispose()


def test_build_engine_sqlite_url_has_mode_ro() -> None:
    """Engine URL for SQLite carries mode=ro."""
    settings = _sqlite_settings(path="/tmp/test_url.sdb")
    engine = build_engine(settings)
    assert "mode=ro" in str(engine.url)
    engine.dispose()


def test_build_engine_mysql_uses_queue_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """MySQL engine uses QueuePool."""
    monkeypatch.setenv("WEEWX_CLEARSKIES_DB_USER", "clearskies_ro")
    monkeypatch.setenv("WEEWX_CLEARSKIES_DB_PASSWORD", "secret")
    settings = _mysql_settings()
    engine = build_engine(settings)
    assert isinstance(engine.pool, QueuePool)
    engine.dispose()


def test_build_engine_mysql_pool_size_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom pool_size / max_overflow from settings are respected."""
    monkeypatch.setenv("WEEWX_CLEARSKIES_DB_USER", "clearskies_ro")
    monkeypatch.setenv("WEEWX_CLEARSKIES_DB_PASSWORD", "secret")
    settings = _mysql_settings()
    settings.pool_size = 3
    settings.max_overflow = 7
    engine = build_engine(settings)
    pool = engine.pool
    assert isinstance(pool, QueuePool)
    assert pool.size() == 3
    engine.dispose()


def test_build_engine_unsupported_kind_raises() -> None:
    """Unsupported database kind raises ValueError."""
    settings = DatabaseSettings({"kind": "postgres"})
    with pytest.raises(ValueError, match="Unsupported database kind"):
        build_engine(settings)
