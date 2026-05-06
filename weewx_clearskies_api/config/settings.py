"""Settings model and config loader.

Config loading order per ADR-027:
  1. CLEARSKIES_CONFIG env var (if set, points directly to the .conf file)
  2. /etc/weewx-clearskies/api.conf
  3. ~/.config/weewx-clearskies/api.conf  (XDG fallback)

Secrets come from environment variables only (loaded from
/etc/weewx-clearskies/secrets.env by the process manager before startup).
The operator is responsible for mode 0600 on secrets.env per ADR-027 §3.

Section mapping:
  [api]      → ApiSettings
  [health]   → HealthSettings
  [logging]  → LoggingSettings
  [ratelimit]→ RateLimitSettings
  [database] → DatabaseSettings  (stub — DB wired in Task 2)
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
from pathlib import Path
from typing import Any

import configobj

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel config paths (ADR-027 search order)
# ---------------------------------------------------------------------------
_CONFIG_SEARCH_PATH: list[Path] = [
    Path("/etc/weewx-clearskies/api.conf"),
    Path.home() / ".config" / "weewx-clearskies" / "api.conf",
]

# Pattern that flags a leaf key that looks like a secret pasted into
# the .conf file instead of secrets.env (ADR-027 §3 secret-leak guard).
# This is not exhaustive — it catches the common mistake only.
_SECRET_KEY_RE = re.compile(r"(?i)_(KEY|SECRET|TOKEN|PASSWORD)$")


# ---------------------------------------------------------------------------
# Settings dataclasses (hand-rolled — avoids pydantic-settings env-var
# coupling for the INI sections; env vars for *secrets* only)
# ---------------------------------------------------------------------------


class ApiSettings:
    """[api] section settings."""

    #: Bind host for the public API. Default loopback per ADR-037.
    bind_host: str
    #: Bind port for the public API.
    bind_port: int
    #: Maximum request body size in bytes (default 1 MiB, security baseline §3.1).
    max_request_bytes: int
    #: Extra CORS origins (comma-separated or INI list).
    cors_origins: list[str]

    def __init__(self, section: dict[str, Any]) -> None:
        self.bind_host = str(section.get("bind_host", "127.0.0.1"))
        self.bind_port = int(section.get("bind_port", 8765))
        self.max_request_bytes = int(section.get("max_request_bytes", 1 * 1024 * 1024))
        raw_origins = section.get("cors_origins", [])
        if isinstance(raw_origins, str):
            # Single-value INI line
            raw_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
        self.cors_origins = list(raw_origins)

    def validate(self) -> None:
        """Raise ValueError on bad values. Called at startup."""
        # Validate bind_host is a legal IP address or hostname.
        # ipaddress.ip_address accepts both IPv4 and IPv6 per coding.md §1.
        # Hostname strings are allowed too — they'll be resolved via getaddrinfo.
        if self.bind_host not in ("", "localhost"):
            try:
                ipaddress.ip_address(self.bind_host)
            except ValueError:
                # Not a bare IP — accept it as a hostname; resolution happens at bind time.
                pass
        if not (1 <= self.bind_port <= 65535):
            raise ValueError(f"[api] bind_port {self.bind_port!r} out of range 1–65535")
        if self.max_request_bytes < 1:
            raise ValueError("[api] max_request_bytes must be >= 1")


class HealthSettings:
    """[health] section settings."""

    #: Bind host for the health port. Default loopback per ADR-030.
    bind_host: str
    #: Bind port for /health/live and /health/ready (default 8081 per ADR-030).
    bind_port: int

    def __init__(self, section: dict[str, Any]) -> None:
        self.bind_host = str(section.get("bind_host", "127.0.0.1"))
        self.bind_port = int(section.get("bind_port", 8081))

    def validate(self) -> None:
        if not (1 <= self.bind_port <= 65535):
            raise ValueError(f"[health] bind_port {self.bind_port!r} out of range 1–65535")


class LoggingSettings:
    """[logging] section settings."""

    #: Log level. Overridden by CLEARSKIES_LOG_LEVEL env var at runtime.
    level: str

    def __init__(self, section: dict[str, Any]) -> None:
        env_level = os.environ.get("CLEARSKIES_LOG_LEVEL", "").upper()
        raw_level = env_level or str(section.get("level", "INFO")).upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if raw_level not in valid:
            raise ValueError(f"[logging] level {raw_level!r} not in {valid}")
        self.level = raw_level


class RateLimitSettings:
    """[ratelimit] section settings."""

    #: Requests per minute per IP for unauthenticated paths (security baseline §3.1).
    requests_per_minute: int
    #: Window size in seconds. Default 60 (1 minute).
    window_seconds: int

    def __init__(self, section: dict[str, Any]) -> None:
        self.requests_per_minute = int(section.get("requests_per_minute", 60))
        self.window_seconds = int(section.get("window_seconds", 60))

    def validate(self) -> None:
        if self.requests_per_minute < 1:
            raise ValueError("[ratelimit] requests_per_minute must be >= 1")
        if self.window_seconds < 1:
            raise ValueError("[ratelimit] window_seconds must be >= 1")


class DatabaseSettings:
    """[database] section settings (ADR-012, ADR-027).

    Non-secret fields come from the INI config file.
    Credentials (user/password) are read from env vars at engine-build time
    by db/engine.py — they never touch this object:
      WEEWX_CLEARSKIES_DB_USER
      WEEWX_CLEARSKIES_DB_PASSWORD

    Pool settings are configurable per ADR-012 (defaults: pool_size=5,
    max_overflow=10).  SQLite ignores pool settings (NullPool is used).
    """

    #: Database type: "sqlite" or "mysql".
    kind: str
    #: For sqlite: path to the .sdb file.
    path: str
    #: For mysql: host — IP or hostname.  IPv4/IPv6 both accepted (coding.md §1).
    host: str
    #: For mysql: port.
    port: int
    #: For mysql: database name.
    name: str
    #: Connection pool size (mysql only, ignored for sqlite). ADR-012 default: 5.
    pool_size: int
    #: Max pool overflow (mysql only, ignored for sqlite). ADR-012 default: 10.
    max_overflow: int

    def __init__(self, section: dict[str, Any]) -> None:
        self.kind = str(section.get("kind", "sqlite"))
        self.path = str(section.get("path", "/var/lib/weewx/weewx.sdb"))
        self.host = str(section.get("host", "127.0.0.1"))
        self.port = int(section.get("port", 3306))
        self.name = str(section.get("name", "weewx"))
        self.pool_size = int(section.get("pool_size", 5))
        self.max_overflow = int(section.get("max_overflow", 10))

    def validate(self) -> None:
        """Raise ValueError on bad values. Called at startup."""
        valid_kinds = {"sqlite", "mysql"}
        if self.kind.lower() not in valid_kinds:
            raise ValueError(
                f"[database] kind {self.kind!r} not in {valid_kinds}. "
                "Supported values: 'sqlite', 'mysql'."
            )
        if self.kind.lower() == "mysql":
            if not (1 <= self.port <= 65535):
                raise ValueError(
                    f"[database] port {self.port!r} out of range 1–65535"
                )
            if not self.name:
                raise ValueError("[database] name must not be empty for mysql kind")
            if not self.host:
                raise ValueError("[database] host must not be empty for mysql kind")
        if self.kind.lower() == "sqlite" and not self.path:
            raise ValueError("[database] path must not be empty for sqlite kind")
        if self.pool_size < 1:
            raise ValueError("[database] pool_size must be >= 1")
        if self.max_overflow < 0:
            raise ValueError("[database] max_overflow must be >= 0")


class Settings:
    """Top-level runtime settings, assembled from INI file + env vars."""

    api: ApiSettings
    health: HealthSettings
    logging: LoggingSettings
    ratelimit: RateLimitSettings
    database: DatabaseSettings

    def __init__(
        self,
        api: ApiSettings,
        health: HealthSettings,
        logging_settings: LoggingSettings,
        ratelimit: RateLimitSettings,
        database: DatabaseSettings,
    ) -> None:
        self.api = api
        self.health = health
        self.logging = logging_settings
        self.ratelimit = ratelimit
        self.database = database

    def validate(self) -> None:
        """Validate all sections. Raises ValueError on the first failure."""
        self.api.validate()
        self.health.validate()
        self.ratelimit.validate()
        self.database.validate()


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _find_config_file() -> Path | None:
    """Return the first config file that exists, following ADR-027 search order."""
    env_path = os.environ.get("CLEARSKIES_CONFIG", "").strip()
    if env_path:
        return Path(env_path)
    for candidate in _CONFIG_SEARCH_PATH:
        if candidate.exists():
            return candidate
    return None


def _check_for_secrets_in_conf(cfg: configobj.ConfigObj) -> None:
    """Raise RuntimeError if any leaf key looks like a secret pasted into .conf.

    ADR-027 §3: secrets belong in secrets.env (mode 0600), never in .conf.
    This guard catches the common mistake. It is not adversarially exhaustive.
    Fires when the key name (not the value) matches the secret-name pattern.
    """
    def _walk(obj: Any, path: str, key: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{path}.{k}", k)
        else:
            # Leaf node — check if the key name looks like a secret.
            if _SECRET_KEY_RE.search(key):
                raise RuntimeError(
                    f"FATAL: config key {path!r} looks like a secret. "
                    "Secrets belong in secrets.env (mode 0600), not in api.conf. "
                    "See ADR-027 for details."
                )

    for top_key, top_val in dict(cfg).items():
        _walk(top_val, top_key, top_key)


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate settings from the INI config file.

    Args:
        config_path: Override path for tests. When None, uses ADR-027 search order.

    Returns:
        Validated Settings instance.

    Raises:
        FileNotFoundError: No config file found at any path in the ADR-027 search order.
        RuntimeError: Secret detected in .conf file (ADR-027 leak guard).
        ValueError: A config value failed validation.
    """
    path = config_path or _find_config_file()

    if path is None:
        # No config found. Per ADR-027: service refuses to start. Config generation
        # belongs in weewx-clearskies-config (weewx-clearskies-stack repo) per ADR-027 §4.
        searched = ", ".join(str(p) for p in _CONFIG_SEARCH_PATH)
        raise FileNotFoundError(
            f"No configuration file found at {searched}. "
            "Generate one with the `weewx-clearskies-config` tool from the "
            "`weewx-clearskies-stack` repo, or copy `etc/api.conf.example` to "
            "`/etc/weewx-clearskies/api.conf` and edit it."
        )

    if not path.exists():
        # Explicit config_path was passed but doesn't exist. Don't silently accept
        # a typo'd path — configobj would create an empty config from a missing
        # file and the service would start with all defaults, which is a footgun.
        raise FileNotFoundError(f"Configuration file not found: {path}")

    cfg = configobj.ConfigObj(str(path), interpolation=False)
    _check_for_secrets_in_conf(cfg)

    api_cfg = ApiSettings(dict(cfg.get("api", {})))
    health_cfg = HealthSettings(dict(cfg.get("health", {})))
    log_cfg = LoggingSettings(dict(cfg.get("logging", {})))
    rl_cfg = RateLimitSettings(dict(cfg.get("ratelimit", {})))
    db_cfg = DatabaseSettings(dict(cfg.get("database", {})))

    settings = Settings(
        api=api_cfg,
        health=health_cfg,
        logging_settings=log_cfg,
        ratelimit=rl_cfg,
        database=db_cfg,
    )
    settings.validate()

    logger.debug("Configuration loaded from %s", path)
    return settings
