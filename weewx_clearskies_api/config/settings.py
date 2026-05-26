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

import contextlib
import ipaddress
import logging
import os
import re
from pathlib import Path
from typing import Any

import configobj

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised for operator configuration errors that prevent startup.

    Used by:
      - cache.py: unsupported CLEARSKIES_CACHE_URL scheme
      - __main__.py catches and exits non-zero (same pattern as write-probe)
    """


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
            # Not a bare IP — accept it as a hostname; resolution happens at bind time.
            with contextlib.suppress(ValueError):
                ipaddress.ip_address(self.bind_host)
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
    #: Expose Prometheus /metrics on the health port (ADR-031). Default False.
    #: Env var CLEARSKIES_METRICS_ENABLED wins over the INI value.
    metrics_enabled: bool

    def __init__(self, section: dict[str, Any]) -> None:
        self.bind_host = str(section.get("bind_host", "127.0.0.1"))
        self.bind_port = int(section.get("bind_port", 8081))
        # ADR-031: opt-in metrics endpoint. Env var wins over ini file.
        env_val = os.environ.get("CLEARSKIES_METRICS_ENABLED", "").strip().lower()
        if env_val:
            self.metrics_enabled = env_val in ("true", "1", "yes")
        else:
            ini_val = str(section.get("metrics_enabled", "false")).strip().lower()
            self.metrics_enabled = ini_val in ("true", "1", "yes")

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


class WeewxSettings:
    """[weewx] section settings.

    Holds the path to weewx.conf (read at startup by services/units.py) and
    the reports directory path (used by the /reports endpoints).

    Default weewx.conf path: /etc/weewx/weewx.conf (stock Debian deb install).
    Default reports directory: /var/www/html/weewx/NOAA (stock Debian deb install
    with SeasonsReport NOAA submodule).  Override to match your installation.
    """

    #: Path to weewx.conf.
    config_path: str
    #: Directory where weewx writes NOAA-*.txt report files.
    reports_directory: str

    def __init__(self, section: dict[str, Any]) -> None:
        self.config_path = str(section.get("config_path", "/etc/weewx/weewx.conf"))
        self.reports_directory = str(
            section.get("reports_directory", "/var/www/html/weewx/NOAA")
        )


class StationSettings:
    """[station] section settings (3a-2; default_locale added ADR-021).

    Optional overrides for station identity.  Absent → clearskies-api derives
    from weewx.conf [Station].
    """

    #: Supported locale codes for the dashboard (ADR-021).
    SUPPORTED_LOCALES: frozenset[str] = frozenset({
        "en", "de", "es", "fil", "fr", "it", "ja",
        "nl", "pt-PT", "pt-BR", "ru", "zh-CN", "zh-TW",
    })

    #: Optional station_id override.  Absent → slug of weewx.conf location.
    station_id: str | None
    #: Optional IANA TZ override (api.conf is highest priority per ADR-020).
    timezone: str | None
    #: Default locale for the dashboard (ADR-021). Env var wins over INI.
    default_locale: str
    #: Comma-separated slugs (or INI list) of built-in pages to hide.
    hidden_pages: list[str]

    def __init__(self, section: dict[str, Any]) -> None:
        raw_id = str(section.get("station_id", "")).strip()
        self.station_id = raw_id if raw_id else None

        raw_tz = str(section.get("timezone", "")).strip()
        self.timezone = raw_tz if raw_tz else None

        # ADR-021: env var wins over INI value; default "en".
        env_locale = os.environ.get("CLEARSKIES_DEFAULT_LOCALE", "").strip()
        if env_locale:
            self.default_locale = env_locale
        else:
            self.default_locale = str(section.get("default_locale", "en"))

        raw_hidden = section.get("hidden", [])
        if isinstance(raw_hidden, str):
            raw_hidden = [s.strip() for s in raw_hidden.split(",") if s.strip()]
        self.hidden_pages = list(raw_hidden)

    def validate(self) -> None:
        """Raise ValueError if default_locale is not in the supported set (ADR-021)."""
        if self.default_locale not in self.SUPPORTED_LOCALES:
            raise ValueError(
                f"[station] default_locale {self.default_locale!r} not in "
                f"supported set: {sorted(self.SUPPORTED_LOCALES)}"
            )


class AlmanacSettings:
    """[almanac] section settings (3a-2).

    Ephemeris cache directory.  Default /var/cache/weewx-clearskies/skyfield/.
    """

    #: Directory where de421.bsp is cached (or pre-placed for offline installs).
    ephemeris_directory: str

    def __init__(self, section: dict[str, Any]) -> None:
        self.ephemeris_directory = str(
            section.get(
                "ephemeris_directory",
                "/var/cache/weewx-clearskies/skyfield/",
            )
        )


class ContentSettings:
    """[content] section settings (3a-2).

    Directory containing about.md and legal.md.  Default /etc/weewx-clearskies/content/.
    """

    #: Directory containing operator-authored markdown files.
    directory: str

    def __init__(self, section: dict[str, Any]) -> None:
        self.directory = str(
            section.get("directory", "/etc/weewx-clearskies/content/")
        )


class PagesSettings:
    """[pages] section settings (3a-2).

    Per-page hide control.
    """

    #: Comma-separated slugs (or INI list) of built-in pages to hide.
    hidden: list[str]

    def __init__(self, section: dict[str, Any]) -> None:
        raw_hidden = section.get("hidden", [])
        if isinstance(raw_hidden, str):
            raw_hidden = [s.strip() for s in raw_hidden.split(",") if s.strip()]
        self.hidden = list(raw_hidden)


class AlertsSettings:
    """[alerts] section settings (3b-1, extended 3b-7 with Aeris credentials,
    extended 3b-8 with OWM appid).

    Provider id and NWS-specific knobs.  Aeris and OWM credentials are loaded
    from env vars at __init__ time per ADR-027 §3 (secrets never in INI; sourced
    from secrets.env loaded by the process manager).

    Naming deviation (brief Q1, user decision 2026-05-08):
      WEEWX_CLEARSKIES_AERIS_CLIENT_ID and WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET
      are provider-scoped (same env vars ForecastSettings reads).  Aeris
      credentials are provider-wide — one key works for /forecasts + /alerts.
      Domain-scoped names would force the operator to paste identical keys into
      two env vars.  Deviation documented here; no ADR amendment.

    OWM naming (3b-8, mirrors 3b-7 Aeris precedent):
      WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID is provider-scoped (same env var
      ForecastSettings reads).  One key works for forecast + alerts.
      Provider-scoped per 3b-5 brief Q2 user decision 2026-05-08.

    nws_user_agent_contact: operator's email or URL for NWS User-Agent.
    Per ADR-006, NO project-level default — operator responsibility.
    """

    #: Provider id: "nws", "aeris", "openweathermap", or absent.
    provider: str | None
    #: NWS User-Agent contact (email or URL).  Optional but recommended.
    nws_user_agent_contact: str | None
    #: Aeris client_id from env var WEEWX_CLEARSKIES_AERIS_CLIENT_ID (ADR-027 §3).
    #: Provider-scoped per 3b-4 brief Q1 user decision 2026-05-08.
    aeris_client_id: str | None
    #: Aeris client_secret from env var WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET.
    aeris_client_secret: str | None
    #: OWM appid from env var WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID (ADR-027 §3).
    #: Provider-scoped per 3b-5 brief Q2 user decision 2026-05-08; same key works
    #: for forecast + alerts (mirrors 3b-7 Aeris precedent).
    openweathermap_appid: str | None

    def __init__(self, section: dict[str, Any]) -> None:
        raw_provider = str(section.get("provider", "")).strip()
        self.provider = raw_provider if raw_provider else None

        raw_contact = str(section.get("nws_user_agent_contact", "")).strip()
        self.nws_user_agent_contact = raw_contact if raw_contact else None

        # Aeris credentials — env vars only, never from the [alerts] INI section.
        # Per ADR-027 §3: secrets come from the process manager's secrets.env file.
        # Same env vars as ForecastSettings (provider-scoped, not domain-scoped).
        raw_aeris_id = os.environ.get("WEEWX_CLEARSKIES_AERIS_CLIENT_ID", "").strip()
        self.aeris_client_id = raw_aeris_id if raw_aeris_id else None

        raw_aeris_secret = os.environ.get("WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET", "").strip()
        self.aeris_client_secret = raw_aeris_secret if raw_aeris_secret else None

        # OWM appid — env var only, never from INI. Long-form provider-scoped name
        # per 3b-5 brief Q2 user decision 2026-05-08 (matches module filename +
        # dispatch key). Same env var as ForecastSettings.openweathermap_appid.
        raw_owm_appid = os.environ.get("WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID", "").strip()
        self.openweathermap_appid = raw_owm_appid if raw_owm_appid else None

    def validate(self) -> None:
        """Raise ValueError on invalid provider id."""
        valid_providers = {"nws", "aeris", "openweathermap"}
        if self.provider is not None and self.provider not in valid_providers:
            raise ValueError(
                f"[alerts] provider {self.provider!r} not in {valid_providers}. "
                "Supported values: 'nws', 'aeris', 'openweathermap'."
            )


class AQISettings:
    """[aqi] section settings (3b-9, extended 3b-10 with Aeris, 3b-12 with IQAir).

    Provider id for the AQI data source.  Open-Meteo is keyless — no env vars
    needed.  Aeris (3b-10) is keyed — credentials come from the shared [aeris]
    section (provider-scoped per 3b-4 Q1 user decision; same env vars as
    forecast/alerts Aeris).  OWM (3b-11) is keyed — provider-scoped per 3b-5 Q2
    decision; same env var as forecast/alerts OWM.  IQAir (3b-12) is keyed —
    domain-scoped per Q1 user decision 2026-05-11 (IQAir is AQI-only; distinct
    from multi-domain Aeris/OWM).

    Per ADR-013: single AQI provider per deploy.  No multi-provider fallback.
    """

    #: Provider id: "openmeteo", "aeris", "openweathermap", "iqair".
    provider: str | None
    #: IQAir API key (domain-scoped per Q1 user decision 2026-05-11; AQI-only provider).
    iqair_key: str | None

    def __init__(self, section: dict[str, Any]) -> None:
        raw_provider = str(section.get("provider", "")).strip()
        self.provider = raw_provider if raw_provider else None

        # IQAir API key — env var only, never from INI.  Long-form provider-scoped
        # naming per LC11 / OWM precedent.  Domain-scoped because IQAir serves only
        # the AQI domain (not forecast/alerts — distinct from Aeris/OWM which are
        # provider-scoped across multiple domains).  Q1 user decision 2026-05-11.
        raw_iqair_key = os.environ.get("WEEWX_CLEARSKIES_IQAIR_KEY", "").strip()
        self.iqair_key = raw_iqair_key if raw_iqair_key else None

    def validate(self) -> None:
        """Raise ValueError on invalid provider id."""
        valid_providers = {"openmeteo", "aeris", "openweathermap", "iqair"}
        if self.provider is not None and self.provider not in valid_providers:
            raise ValueError(
                f"[aqi] provider {self.provider!r} not in {valid_providers}. "
                "Supported values: 'openmeteo', 'aeris', 'openweathermap', 'iqair'."
            )


class AQIHistorySettings:
    """[aqi.history] section settings (P4-T3, ADR-013 corrected).

    Maps canonical AQI field names to weewx archive column names.
    Empty string = field not available in this operator's archive.

    Path A operators: populate columns matching their archive schema.
    Path B operators: leave all fields empty (all-empty defaults); /aqi/history
      returns an empty data list (not an error).

    Column names come exclusively from this config object (trusted constants),
    never from user input.  Bound into the service via wire_aqi_settings().
    """

    #: Archive column name for the composite AQI value.  Empty = not present.
    column_aqi: str
    #: Archive column name for the AQI category label.  Empty = not present.
    column_aqi_category: str
    #: Archive column name for the main pollutant label.  Empty = not present.
    column_aqi_main_pollutant: str
    #: Archive column name for the AQI location label.  Empty = not present.
    column_aqi_location: str
    #: Archive column name for PM2.5 concentration (µg/m³).  Empty = not present.
    column_pm25: str
    #: Archive column name for PM10 concentration (µg/m³).  Empty = not present.
    column_pm10: str
    #: Archive column name for O3 concentration (ppm).  Empty = not present.
    column_o3: str
    #: Archive column name for NO2 concentration (ppm).  Empty = not present.
    column_no2: str
    #: Archive column name for SO2 concentration (ppm).  Empty = not present.
    column_so2: str
    #: Archive column name for CO concentration (ppm).  Empty = not present.
    column_co: str

    def __init__(self, section: dict[str, Any]) -> None:
        self.column_aqi = str(section.get("column_aqi", "")).strip()
        self.column_aqi_category = str(section.get("column_aqi_category", "")).strip()
        self.column_aqi_main_pollutant = str(section.get("column_aqi_main_pollutant", "")).strip()
        self.column_aqi_location = str(section.get("column_aqi_location", "")).strip()
        self.column_pm25 = str(section.get("column_pm25", "")).strip()
        self.column_pm10 = str(section.get("column_pm10", "")).strip()
        self.column_o3 = str(section.get("column_o3", "")).strip()
        self.column_no2 = str(section.get("column_no2", "")).strip()
        self.column_so2 = str(section.get("column_so2", "")).strip()
        self.column_co = str(section.get("column_co", "")).strip()


class EarthquakesSettings:
    """[earthquakes] section settings (3b-13).

    Provider id for the earthquake data source.  All four day-1 providers (usgs,
    geonet, emsc, renass) are keyless — no env vars needed.

    Per ADR-040: single earthquake provider per deploy.  No multi-provider
    fallback or aggregation.
    """

    #: Provider id: "usgs", "geonet", "emsc", "renass", or absent.
    provider: str | None
    #: Default radius in km from station lat/lon.  Override per-request via ?radius_km.
    default_radius_km: float

    def __init__(self, section: dict[str, Any]) -> None:
        raw_provider = str(section.get("provider", "")).strip()
        self.provider = raw_provider if raw_provider else None

        raw_radius = section.get("default_radius_km", 100)
        try:
            self.default_radius_km = float(raw_radius)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"[earthquakes] default_radius_km {raw_radius!r} must be a number."
            ) from exc
        if self.default_radius_km < 0:
            raise ValueError(
                f"[earthquakes] default_radius_km {self.default_radius_km!r} must be >= 0."
            )

    def validate(self) -> None:
        """Raise ValueError on invalid provider id."""
        valid_providers = {"usgs", "geonet", "emsc", "renass"}
        if self.provider is not None and self.provider not in valid_providers:
            raise ValueError(
                f"[earthquakes] provider {self.provider!r} not in {valid_providers}. "
                "Supported values: 'usgs', 'geonet', 'emsc', 'renass'."
            )


class RadarSettings:
    """[radar] section settings (3b-14; extended 3b-15 with 2 keyed providers;
    extended 3b-16 with iframe provider).

    Provider id for the radar data source.  Five keyless providers (rainviewer,
    iem_nexrad, noaa_mrms, msc_geomet, dwd_radolan), two keyed providers
    (aeris, openweathermap) per ADR-015, and one iframe provider (iframe).

    Note: mapbox_jma is NOT included — deferred per ADR-015 2026-05-11 amendment
    (Mapbox JMA tilesets are raster-array shape, GL-JS-only; incompatible with
    Leaflet).

    Per ADR-015: single radar provider per deploy (operator picks one per
    their lat/lon).  Per-region auto-pick is a setup-wizard concern (out of scope).

    Keyed provider credentials (aeris, openweathermap) are NOT stored here —
    they are wired at startup via wire_radar_settings() in endpoints/radar.py,
    which reads them from settings.forecast (provider-scoped per 3b-5 Q2).

    iframe provider: operator supplies iframe_url in [radar] section; the
    dashboard embeds the URL directly.  No frames index, no tile proxy.
    """

    #: Provider id: "rainviewer", "iem_nexrad", "noaa_mrms", "msc_geomet",
    #: "dwd_radolan", "aeris", "openweathermap", "iframe", or absent.
    provider: str | None
    #: iframe embed URL.  Required when provider == "iframe"; None otherwise.
    iframe_url: str | None

    def __init__(self, section: dict[str, Any]) -> None:
        raw_provider = str(section.get("provider", "")).strip()
        self.provider = raw_provider if raw_provider else None
        raw_iframe_url = str(section.get("iframe_url", "")).strip()
        self.iframe_url = raw_iframe_url if raw_iframe_url else None

    def validate(self) -> None:
        """Raise ValueError on invalid provider id."""
        valid_providers = {
            "rainviewer",
            "iem_nexrad",
            "noaa_mrms",
            "msc_geomet",
            "dwd_radolan",
            "aeris",       # keyed — added 3b-15; credentials in settings.forecast
            "openweathermap",  # keyed — added 3b-15; credentials in settings.forecast
            "iframe",      # iframe embed — added 3b-16; requires iframe_url in [radar]
        }
        # mapbox_jma is NOT valid — deferred per ADR-015 2026-05-11 amendment.
        if self.provider is not None and self.provider not in valid_providers:
            raise ValueError(
                f"[radar] provider {self.provider!r} not in {valid_providers}. "
                "Supported values: 'rainviewer', 'iem_nexrad', 'noaa_mrms', "
                "'msc_geomet', 'dwd_radolan' (keyless); "
                "'aeris', 'openweathermap' (keyed; credentials in [forecast] section); "
                "'iframe' (embed; requires iframe_url in [radar] section)."
            )
        if self.provider == "iframe" and not self.iframe_url:
            raise ValueError(
                "[radar] provider='iframe' requires iframe_url to be set in api.conf"
            )


class ForecastSettings:
    """[forecast] section settings (3b-2, extended 3b-3 with NWS UA contact,
    extended 3b-4 with Aeris credentials, extended 3b-5 with OWM appid).

    Provider id and NWS-specific knobs. Open-Meteo is keyless (no knobs).
    Aeris and OWM credentials are loaded from env vars at __init__ time per
    ADR-027 §3 (secrets never in INI; sourced from secrets.env loaded by the
    process manager).

    Naming deviation (brief Q1, user decision 2026-05-08):
      WEEWX_CLEARSKIES_AERIS_CLIENT_ID and WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET
      are provider-scoped (not domain-scoped as ADR-027 §3's literal schema
      prescribes).  Rationale: Aeris credentials are provider-wide — the same
      key works for /forecasts, /alerts, and /observations.  Domain-scoped names
      would force the operator to paste identical keys into two env vars.
      Deviation documented here and in providers/forecast/aeris.py; no ADR amendment.

    OWM naming (brief Q2, user decision 2026-05-08):
      WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID is provider-scoped, long-form.
      Matches the module filename (openweathermap.py) and dispatch key
      ("openweathermap").  Consistent with the 3b-4 Aeris precedent.
      No ADR amendment needed — same deviation class as Aeris.

    nws_user_agent_contact: operator's email or URL for NWS User-Agent.
    Per ADR-006, NO project-level default — operator responsibility.

    Accepts all five ADR-007 day-1 forecast providers even though only
    "openmeteo", "nws", "aeris", and "openweathermap" are in dispatch this
    round. Providers not yet in dispatch raise KeyError at startup
    (fail-closed, same pattern as AlertsSettings).
    """

    #: Provider id: "openmeteo", "nws", "aeris", "openweathermap", "wunderground", or absent.
    provider: str | None
    #: NWS User-Agent contact (email or URL).  Optional but recommended (ADR-006).
    nws_user_agent_contact: str | None
    #: Aeris client_id from env var WEEWX_CLEARSKIES_AERIS_CLIENT_ID (ADR-027 §3).
    aeris_client_id: str | None
    #: Aeris client_secret from env var WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET (ADR-027 §3).
    aeris_client_secret: str | None
    #: OWM appid from env var WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID (ADR-027 §3).
    #: Long-form provider-scoped naming per brief Q2 user decision 2026-05-08.
    openweathermap_appid: str | None
    #: Wunderground apiKey from env var WEEWX_CLEARSKIES_WUNDERGROUND_API_KEY.
    #: Long-form provider-scoped naming per 3b-4/3b-5 precedent (same deviation
    #: from ADR-027 §3 literal schema as Aeris + OWM; no ADR amendment).
    wunderground_api_key: str | None
    #: Wunderground PWS station ID from env var WEEWX_CLEARSKIES_WUNDERGROUND_PWS_STATION_ID.
    #: Required alongside the apiKey per ADR-007 line 79 defense-in-depth gate:
    #: apiKeys are issued only to active PWS contributors, so requiring both env
    #: vars ensures operator's mental model matches the gating reality.
    #: PWS station ID isn't strictly a secret but is co-located with the apiKey
    #: for operational simplicity (all Wunderground config in env vars together).
    wunderground_pws_station_id: str | None

    def __init__(self, section: dict[str, Any]) -> None:
        raw_provider = str(section.get("provider", "")).strip()
        self.provider = raw_provider if raw_provider else None

        raw_contact = str(section.get("nws_user_agent_contact", "")).strip()
        self.nws_user_agent_contact = raw_contact if raw_contact else None

        # Aeris credentials — env vars only, never from the [forecast] INI section.
        # Per ADR-027 §3: secrets come from the process manager's secrets.env file.
        raw_aeris_id = os.environ.get("WEEWX_CLEARSKIES_AERIS_CLIENT_ID", "").strip()
        self.aeris_client_id = raw_aeris_id if raw_aeris_id else None

        raw_aeris_secret = os.environ.get("WEEWX_CLEARSKIES_AERIS_CLIENT_SECRET", "").strip()
        self.aeris_client_secret = raw_aeris_secret if raw_aeris_secret else None

        # OWM appid — env var only, never from INI. Long-form provider-scoped name
        # per brief Q2 user decision 2026-05-08 (matches module filename + dispatch key).
        raw_owm_appid = os.environ.get("WEEWX_CLEARSKIES_OPENWEATHERMAP_APPID", "").strip()
        self.openweathermap_appid = raw_owm_appid if raw_owm_appid else None

        # Wunderground credentials — env vars only, never from INI.
        # Long-form provider-scoped naming per 3b-4/3b-5 precedent.
        # ADR-007 line 79 "config time" gate operationalized as fetch-time KeyInvalid
        # (same precedent as Aeris/OWM; documented in wunderground.py module docstring).
        raw_wu_key = os.environ.get("WEEWX_CLEARSKIES_WUNDERGROUND_API_KEY", "").strip()
        self.wunderground_api_key = raw_wu_key if raw_wu_key else None

        raw_wu_pws = os.environ.get("WEEWX_CLEARSKIES_WUNDERGROUND_PWS_STATION_ID", "").strip()
        self.wunderground_pws_station_id = raw_wu_pws if raw_wu_pws else None

    def validate(self) -> None:
        """Raise ValueError on invalid provider id."""
        valid_providers = {"openmeteo", "nws", "aeris", "openweathermap", "wunderground"}
        if self.provider is not None and self.provider not in valid_providers:
            raise ValueError(
                f"[forecast] provider {self.provider!r} not in {valid_providers}. "
                "Supported values: 'openmeteo', 'nws', 'aeris', 'openweathermap', 'wunderground'."
            )


class BrandingSettings:
    """[branding] section settings (ADR-022, Gap #10).

    Operator-controlled branding values served by GET /api/v1/branding.
    All fields are optional with safe defaults so the endpoint works even if
    no [branding] section is present in api.conf.

    v0.1 scope: read-only; logo upload pipeline is Phase 2.
    """

    #: Accent color name — one of the curated palette entries per ADR-022.
    accent: str
    #: Default theme mode for new visitors per ADR-023.
    default_theme_mode: str
    #: URL to operator's custom stylesheet. None = no custom CSS.
    custom_css_url: str | None

    #: Valid accent values from ADR-022 (curated palette, AA-tested).
    VALID_ACCENTS: frozenset[str] = frozenset({
        "blue", "teal", "indigo", "purple", "green", "amber"
    })
    #: Valid theme modes from ADR-023.
    VALID_THEME_MODES: frozenset[str] = frozenset({
        "light", "dark", "auto-os", "auto-sunrise-sunset"
    })

    def __init__(self, section: dict[str, Any]) -> None:
        self.accent = str(section.get("accent", "blue")).strip()
        self.default_theme_mode = str(
            section.get("default_theme_mode", "auto-os")
        ).strip()
        raw_css_url = str(section.get("custom_css_url", "")).strip()
        self.custom_css_url = raw_css_url if raw_css_url else None

    def validate(self) -> None:
        """Raise ValueError on invalid accent or theme mode."""
        if self.accent not in self.VALID_ACCENTS:
            raise ValueError(
                f"[branding] accent {self.accent!r} not in "
                f"{sorted(self.VALID_ACCENTS)}. "
                "Supported values per ADR-022: "
                "'blue', 'teal', 'indigo', 'purple', 'green', 'amber'."
            )
        if self.default_theme_mode not in self.VALID_THEME_MODES:
            raise ValueError(
                f"[branding] default_theme_mode {self.default_theme_mode!r} not in "
                f"{sorted(self.VALID_THEME_MODES)}. "
                "Supported values per ADR-023: "
                "'light', 'dark', 'auto-os', 'auto-sunrise-sunset'."
            )


class TlsSettings:
    """[tls] section settings (ADR-038).

    Paths to operator-supplied TLS cert and key PEM files.  Both must be set
    together or not at all (validated in validate()).  When absent, the API
    auto-generates a self-signed Ed25519 cert in config_dir at startup.

    Paths may also be overridden at startup via --tls-cert / --tls-key CLI flags;
    those override these settings values in main() before any file I/O happens.
    """

    #: Path to operator-supplied TLS certificate (PEM).  Empty = auto-generate.
    cert_path: str
    #: Path to operator-supplied TLS private key (PEM).  Empty = auto-generate.
    key_path: str

    def __init__(self, section: dict[str, Any]) -> None:
        self.cert_path = str(section.get("cert_path", "")).strip()
        self.key_path = str(section.get("key_path", "")).strip()

    def validate(self) -> None:
        """Raise ValueError if exactly one of cert_path / key_path is set."""
        has_cert = bool(self.cert_path)
        has_key = bool(self.key_path)
        if has_cert != has_key:
            raise ValueError(
                "[tls] cert_path and key_path must both be set or both be absent. "
                "Supply both paths for operator-managed TLS, or omit both to "
                "use auto-generated self-signed certificates."
            )


class ConditionsSettings:
    """[conditions] section settings (Phase 0B).

    Controls the current conditions text blending engine that populates
    the weatherText field on Observation responses.

    engine values:
      "auto"     — blends local sensor data with provider conditions (default).
      "provider" — uses provider conditions text verbatim (no local blending).
      "off"      — weatherText is always None on observations.
    """

    #: Blending engine mode: "auto" | "provider" | "off".
    engine: str

    _VALID_ENGINES: frozenset[str] = frozenset({"auto", "provider", "off"})

    def __init__(self, section: dict[str, Any]) -> None:
        self.engine = str(section.get("engine", "auto")).strip()

    def validate(self) -> None:
        """Raise ValueError on invalid engine value."""
        if self.engine not in self._VALID_ENGINES:
            raise ValueError(
                f"[conditions] engine {self.engine!r} not in "
                f"{sorted(self._VALID_ENGINES)}. "
                "Supported values: 'auto', 'provider', 'off'."
            )


class Settings:
    """Top-level runtime settings, assembled from INI file + env vars."""

    configured: bool
    api: ApiSettings
    health: HealthSettings
    logging: LoggingSettings
    ratelimit: RateLimitSettings
    database: DatabaseSettings
    weewx: WeewxSettings
    station: StationSettings
    almanac: AlmanacSettings
    content: ContentSettings
    pages: PagesSettings
    alerts: AlertsSettings
    aqi: AQISettings
    aqi_history: AQIHistorySettings
    earthquakes: EarthquakesSettings
    radar: RadarSettings
    forecast: ForecastSettings
    tls: TlsSettings
    branding: BrandingSettings
    conditions: ConditionsSettings

    def __init__(
        self,
        api: ApiSettings,
        health: HealthSettings,
        logging_settings: LoggingSettings,
        ratelimit: RateLimitSettings,
        database: DatabaseSettings,
        weewx: WeewxSettings | None = None,
        station: StationSettings | None = None,
        almanac: AlmanacSettings | None = None,
        content: ContentSettings | None = None,
        pages: PagesSettings | None = None,
        alerts: AlertsSettings | None = None,
        aqi: AQISettings | None = None,
        aqi_history: AQIHistorySettings | None = None,
        earthquakes: EarthquakesSettings | None = None,
        radar: RadarSettings | None = None,
        forecast: ForecastSettings | None = None,
        tls: TlsSettings | None = None,
        branding: BrandingSettings | None = None,
        conditions: ConditionsSettings | None = None,
        configured: bool = True,
    ) -> None:
        self.configured = configured
        self.api = api
        self.health = health
        self.logging = logging_settings
        self.ratelimit = ratelimit
        self.database = database
        self.weewx = weewx if weewx is not None else WeewxSettings({})
        self.station = station if station is not None else StationSettings({})
        self.almanac = almanac if almanac is not None else AlmanacSettings({})
        self.content = content if content is not None else ContentSettings({})
        self.pages = pages if pages is not None else PagesSettings({})
        self.alerts = alerts if alerts is not None else AlertsSettings({})
        self.aqi = aqi if aqi is not None else AQISettings({})
        self.aqi_history = aqi_history if aqi_history is not None else AQIHistorySettings({})
        self.earthquakes = earthquakes if earthquakes is not None else EarthquakesSettings({})
        self.radar = radar if radar is not None else RadarSettings({})
        self.forecast = forecast if forecast is not None else ForecastSettings({})
        self.tls = tls if tls is not None else TlsSettings({})
        self.branding = branding if branding is not None else BrandingSettings({})
        self.conditions = conditions if conditions is not None else ConditionsSettings({})

    def validate(self) -> None:
        """Validate all sections. Raises ValueError on the first failure."""
        self.api.validate()
        self.health.validate()
        self.ratelimit.validate()
        self.database.validate()
        self.station.validate()
        self.alerts.validate()
        self.aqi.validate()
        self.earthquakes.validate()
        self.radar.validate()
        self.forecast.validate()
        self.tls.validate()
        self.branding.validate()
        self.conditions.validate()


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def find_config_file() -> Path | None:
    """Return the first config file that exists, following ADR-027 search order."""
    env_path = os.environ.get("CLEARSKIES_CONFIG", "").strip()
    if env_path:
        return Path(env_path)
    for candidate in _CONFIG_SEARCH_PATH:
        if candidate.exists():
            return candidate
    return None


# Keep the private alias for backwards-compat with any internal callers.
_find_config_file = find_config_file


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
        Validated Settings instance. When no config file is found, returns a minimal
        Settings with configured=False so the API can start in setup mode.

    Raises:
        RuntimeError: Secret detected in .conf file (ADR-027 leak guard).
        ValueError: A config value failed validation.
    """
    path = config_path or _find_config_file()

    if path is None:
        # No config found — start in setup mode. The API binds on defaults and
        # serves only /setup/* and GET /api/v1/status so the wizard can connect.
        logger.info("No configuration file found — starting in setup mode")
        return Settings(
            api=ApiSettings({"bind_host": "0.0.0.0", "bind_port": 8765}),
            health=HealthSettings({"bind_host": "127.0.0.1", "bind_port": 8081}),
            logging_settings=LoggingSettings({}),
            ratelimit=RateLimitSettings({}),
            database=DatabaseSettings({}),
            tls=TlsSettings({}),
            configured=False,
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
    weewx_cfg = WeewxSettings(dict(cfg.get("weewx", {})))
    station_cfg = StationSettings(dict(cfg.get("station", {})))
    almanac_cfg = AlmanacSettings(dict(cfg.get("almanac", {})))
    content_cfg = ContentSettings(dict(cfg.get("content", {})))
    pages_cfg = PagesSettings(dict(cfg.get("pages", {})))
    alerts_cfg = AlertsSettings(dict(cfg.get("alerts", {})))
    aqi_cfg = AQISettings(dict(cfg.get("aqi", {})))
    aqi_history_cfg = AQIHistorySettings(dict(cfg.get("aqi.history", {})))
    earthquakes_cfg = EarthquakesSettings(dict(cfg.get("earthquakes", {})))
    radar_cfg = RadarSettings(dict(cfg.get("radar", {})))
    forecast_cfg = ForecastSettings(dict(cfg.get("forecast", {})))
    tls_cfg = TlsSettings(dict(cfg.get("tls", {})))
    branding_cfg = BrandingSettings(dict(cfg.get("branding", {})))
    conditions_cfg = ConditionsSettings(dict(cfg.get("conditions", {})))

    settings = Settings(
        api=api_cfg,
        health=health_cfg,
        logging_settings=log_cfg,
        ratelimit=rl_cfg,
        database=db_cfg,
        weewx=weewx_cfg,
        station=station_cfg,
        almanac=almanac_cfg,
        content=content_cfg,
        pages=pages_cfg,
        alerts=alerts_cfg,
        aqi=aqi_cfg,
        aqi_history=aqi_history_cfg,
        earthquakes=earthquakes_cfg,
        radar=radar_cfg,
        forecast=forecast_cfg,
        tls=tls_cfg,
        branding=branding_cfg,
        conditions=conditions_cfg,
    )
    settings.validate()

    logger.debug("Configuration loaded from %s", path)
    return settings
