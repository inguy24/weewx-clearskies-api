"""Entry point for weewx-clearskies-api.

Run with:
    python -m weewx_clearskies_api
    weewx-clearskies-api  (via pyproject.toml scripts entry point)

IPv4/IPv6 dual-stack listener (coding.md §1, ADR-037):
    Default bind = 127.0.0.1 per ADR-037 (loopback, behind the reverse proxy).
    When operator sets [api] bind_host to a non-loopback address, we resolve
    via socket.getaddrinfo to get the full (family, address) set and start
    one uvicorn Server per (family, addr) pair.

    We never use gethostbyname — it is IPv4-only and violates coding.md §1.
    We use ipaddress.ip_address to validate the bind_host only when it looks
    like a bare IP; hostnames are passed to getaddrinfo directly.

Startup warning for cross-host without proxy secret (ADR-008):
    When bind_host is non-loopback and WEEWX_CLEARSKIES_PROXY_SECRET is unset,
    emit a loud WARNING at startup (and schedule a repeat every 60 s).

Startup sequence (ADR-012):
    1. load settings          — parse api.conf, validate all sections.
    2. setup logging          — JSON formatter active before any DB work.
    3. build engine           — SQLAlchemy engine from [database] settings.
    4. run write-probe        — exits 1 if the DB user has write privileges.
    5. run schema reflection  — MetaData.reflect() on the archive table;
                                logs warnings on unmapped columns; does NOT exit.
    6. load weewx.conf        — shared ConfigObj parse for units + station.
    6b. load units block      — resolves per-field unit strings from weewx.conf.
    6c. load station metadata — reads [Station] from weewx.conf (fatal if missing).
    6d. wire ephemeris        — loads de421.bsp for almanac (fatal if not available).
    6e. wire reports dir      — non-fatal; empty /reports on missing dir.
    6f. wire content dir      — non-fatal; 404 /content/* on missing dir.
    6g. wire hidden pages     — pages excluded from /pages response.
    6h. wire cache            — construct MemoryCache or RedisCache (fail-closed).
    6i. wire providers        — register configured provider CAPABILITY declarations.
    6j. wire alerts settings  — pass settings to alerts endpoint.
    6k. wire aqi settings     — pass settings to aqi endpoint (no-op for Open-Meteo;
                                credentials wired for Aeris per 3b-10).
    6l. wire earthquakes settings — pass settings to earthquakes endpoint (default_radius_km).
    6m. wire forecast settings — pass settings to forecast endpoint (NWS UA).
    6n. wire conditions settings — pass engine mode + station coords to observations endpoint.
    6o. wire radar — register configured radar provider's CAPABILITY in registry.
    6p. wire radar settings — wire credentials for keyed radar providers (aeris, openweathermap).
    7. register DB probe      — health subsystem wired with SELECT 1 probe.
    8. start uvicorn          — public API + health app.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from weewx_clearskies_api.app import create_app
from weewx_clearskies_api.config.settings import Settings, find_config_file, load_settings
from weewx_clearskies_api.tls import compute_fingerprint, ensure_tls_cert
from weewx_clearskies_api.trust import TrustManager
from weewx_clearskies_api.db.engine import build_engine
from weewx_clearskies_api.db.health import wire_db_health_probe
from weewx_clearskies_api.db.probe import run_write_probe
from weewx_clearskies_api.db.reflection import SchemaReflector
from weewx_clearskies_api.db.registry import wire_registry
from weewx_clearskies_api.db.session import wire_engine
from weewx_clearskies_api.endpoints.alerts import wire_alerts_settings
from weewx_clearskies_api.endpoints.aqi import wire_aqi_settings
from weewx_clearskies_api.endpoints.branding import wire_branding_settings
from weewx_clearskies_api.endpoints.earthquakes import wire_earthquakes_settings
from weewx_clearskies_api.endpoints.forecast import wire_forecast_settings
from weewx_clearskies_api.endpoints.observations import wire_conditions_settings
from weewx_clearskies_api.endpoints.pages import wire_hidden_pages
from weewx_clearskies_api.endpoints.radar import wire_radar_settings
from weewx_clearskies_api.health import create_health_app
from weewx_clearskies_api.logging.setup import setup_logging
from weewx_clearskies_api.providers._common.cache import ConfigError as CacheConfigError
from weewx_clearskies_api.providers._common.cache import wire_cache_from_env
from weewx_clearskies_api.providers._common.capability import ProviderCapability, wire_providers
from weewx_clearskies_api.providers._common.dispatch import get_provider_module
from weewx_clearskies_api.services.almanac import wire_ephemeris_directory
from weewx_clearskies_api.services.content import wire_content_directory
from weewx_clearskies_api.services.reports import wire_reports_directory
from weewx_clearskies_api.services.station import StationConfigError, load_station_metadata
from weewx_clearskies_api.services.units import (
    WeewxConfNotFoundError,
    get_target_unit,
    load_units_block,
)
from weewx_clearskies_api.services.weewx_conf import WeewxConfLoadError, load_weewx_conf

logger = logging.getLogger(__name__)

_LOOPBACK_PREFIXES = ("127.", "::1", "localhost")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clear Skies Weather API")
    parser.add_argument("--tls-cert", type=Path, help="Path to TLS certificate (PEM)")
    parser.add_argument("--tls-key", type=Path, help="Path to TLS private key (PEM)")
    return parser.parse_args()


def _is_loopback(host: str) -> bool:
    """Return True if host is a loopback address (IPv4 or IPv6)."""
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_loopback
    except ValueError:
        # Hostname — check well-known loopback names.
        return host in ("localhost",)


def _warn_non_loopback_loop(host: str, interval: int = 60) -> None:
    """Log a loud warning every `interval` seconds when bound non-loopback
    without WEEWX_CLEARSKIES_PROXY_SECRET set (ADR-008).

    Runs in a daemon thread — stops automatically when the main process exits.
    """
    while True:
        time.sleep(interval)
        logger.warning(
            "clearskies-api is bound to a non-loopback address (%s) without "
            "WEEWX_CLEARSKIES_PROXY_SECRET set. Any host that can reach this address "
            "can read this service directly, bypassing your reverse proxy. "
            "See SECURITY.md for the recommended cross-host config.",
            host,
        )


def _resolve_bind_addresses(host: str, port: int) -> list[tuple[str, int]]:
    """Resolve host to all (address, port) pairs via getaddrinfo (coding.md §1).

    Returns a list of (ip_address_string, port) tuples — one per address
    family resolved. For "127.0.0.1" this returns [("127.0.0.1", port)].
    For "localhost" this typically returns both ("127.0.0.1", port) and
    ("::1", port) on dual-stack systems.
    """
    results: list[tuple[str, int]] = []
    try:
        for _family, _type, _proto, _cname, sockaddr in socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM
        ):
            ip_str = sockaddr[0]
            if (ip_str, port) not in results:
                results.append((ip_str, port))
    except socket.gaierror as exc:
        logger.error("Failed to resolve bind address %r: %s", host, exc)
        raise

    return results


def _run_server(
    settings: Settings,
    cert_path: Path,
    key_path: Path,
    app: FastAPI,
) -> None:
    """Start the public API and health servers.

    Public API: one uvicorn Server per resolved (family, addr) from [api] bind_host.
    Health API: one uvicorn Server per resolved (family, addr) from [health] bind_host.

    Both run concurrently via asyncio.gather in the main thread.
    TLS is applied to every uvicorn Config via ssl_certfile / ssl_keyfile (ADR-038).

    app is passed in (rather than created here) so main() can attach state to it
    before the server starts.
    """
    health_app = create_health_app(
        metrics_enabled=settings.health.metrics_enabled,
        configured=settings.configured,
    )

    api_addresses = _resolve_bind_addresses(settings.api.bind_host, settings.api.bind_port)
    health_addresses = _resolve_bind_addresses(
        settings.health.bind_host, settings.health.bind_port
    )

    # Cross-host without proxy secret warning (ADR-008).
    if not _is_loopback(settings.api.bind_host):
        proxy_secret = os.environ.get("WEEWX_CLEARSKIES_PROXY_SECRET", "").strip()
        if not proxy_secret:
            logger.warning(
                "clearskies-api is bound to a non-loopback address (%s) without "
                "WEEWX_CLEARSKIES_PROXY_SECRET set. Any host that can reach this address "
                "can read this service directly, bypassing your reverse proxy. "
                "See SECURITY.md for the recommended cross-host config.",
                settings.api.bind_host,
            )
            t = threading.Thread(
                target=_warn_non_loopback_loop,
                args=(settings.api.bind_host,),
                daemon=True,
            )
            t.start()

    log_level = settings.logging.level.lower()
    cert_str = str(cert_path)
    key_str = str(key_path)

    # Build uvicorn configs for each bind address — TLS on every listener (ADR-038).
    api_configs = [
        uvicorn.Config(
            app,
            host=addr,
            port=port,
            log_level=log_level,
            access_log=False,
            ssl_certfile=cert_str,
            ssl_keyfile=key_str,
        )
        for addr, port in api_addresses
    ]
    health_configs = [
        uvicorn.Config(
            health_app,
            host=addr,
            port=port,
            log_level=log_level,
            access_log=False,
            ssl_certfile=cert_str,
            ssl_keyfile=key_str,
        )
        for addr, port in health_addresses
    ]

    all_configs = api_configs + health_configs

    logger.info(
        "Starting weewx-clearskies-api",
        extra={
            "api_addresses": api_addresses,
            "health_addresses": health_addresses,
        },
    )

    async def _serve_all() -> None:
        servers = [uvicorn.Server(cfg) for cfg in all_configs]
        await asyncio.gather(*[server.serve() for server in servers])

    asyncio.run(_serve_all())


def _format_address_for_url(host: str, port: int) -> str:
    """Return ``https://<host>:<port>`` with IPv6 literals in brackets.

    When host is a wildcard (``::`` or ``0.0.0.0``), substitutes ``localhost``
    so the operator gets a usable address to paste into a browser.
    """
    if host in ("::", "0.0.0.0"):
        host = "localhost"
    # Wrap raw IPv6 literals in brackets per coding.md §1 / RFC 3986.
    try:
        addr = ipaddress.ip_address(host)
        if addr.version == 6:
            host = f"[{host}]"
    except ValueError:
        pass  # hostname — no brackets needed
    return f"https://{host}:{port}"


def _wire_providers_from_config(settings: Settings) -> None:
    """Build the provider declarations list from operator config and register.

    Single source per domain per ADR-016 / ADR-007 / ADR-013.  If [alerts],
    [aqi], or [forecast] provider is set, look up the module via dispatch and
    register its CAPABILITY.

    Future rounds extend this with earthquakes, radar.

    Failure modes:
      - [alerts] provider = <unknown-id> → KeyError → CRITICAL + exit 1.
      - [alerts] provider absent → empty contribution; /alerts returns
        source="none" per ADR-016 §Out-of-scope.
      - [aqi] provider = <unknown-id> → KeyError → CRITICAL + exit 1.
      - [aqi] provider absent → empty contribution; /aqi/current returns
        data=null, source="none" per ADR-013.
      - [forecast] provider = <unknown-id> → KeyError → CRITICAL + exit 1.
      - [forecast] provider = <ADR-007-listed-but-not-yet-wired> (e.g. "nws")
        → ForecastSettings.validate() accepts the id; dispatch KeyError fires
        at startup (fail-closed, same pattern as alerts case).
      - [forecast] provider absent → empty contribution; /forecast returns
        source="none" per ADR-007.
    """
    declarations: list[ProviderCapability] = []

    if settings.alerts.provider:
        provider_id = settings.alerts.provider
        try:
            module = get_provider_module(domain="alerts", provider_id=provider_id)
        except KeyError as exc:
            logger.critical(
                "FATAL: Unknown alerts provider %r in api.conf — clearskies-api cannot start. "
                "Cause: %s. "
                "Check [alerts] provider in api.conf. "
                "Supported values: nws, aeris, openweathermap.",
                provider_id,
                exc,
            )
            sys.exit(1)
        declarations.append(module.CAPABILITY)

    if settings.aqi.provider:
        provider_id = settings.aqi.provider
        try:
            module = get_provider_module(domain="aqi", provider_id=provider_id)
        except KeyError as exc:
            logger.critical(
                "FATAL: Unknown aqi provider %r in api.conf — clearskies-api cannot start. "
                "Cause: %s. "
                "Check [aqi] provider in api.conf. "
                "Supported values: openmeteo, aeris, openweathermap, iqair. ",
                provider_id,
                exc,
            )
            sys.exit(1)
        declarations.append(module.CAPABILITY)

    if settings.earthquakes.provider:
        provider_id = settings.earthquakes.provider
        try:
            module = get_provider_module(domain="earthquakes", provider_id=provider_id)
        except KeyError as exc:
            logger.critical(
                "FATAL: Unknown earthquakes provider %r in api.conf — clearskies-api cannot start. "
                "Cause: %s. "
                "Check [earthquakes] provider in api.conf. "
                "Supported values: usgs, geonet, emsc, renass (all keyless per ADR-040).",
                provider_id,
                exc,
            )
            sys.exit(1)
        declarations.append(module.CAPABILITY)

    if settings.forecast.provider:
        provider_id = settings.forecast.provider
        try:
            module = get_provider_module(domain="forecast", provider_id=provider_id)
        except KeyError as exc:
            logger.critical(
                "FATAL: Unknown forecast provider %r in api.conf — clearskies-api cannot start. "
                "Cause: %s. "
                "Check [forecast] provider in api.conf. "
                "Currently wired: openmeteo, nws. "
                "Accepted by config (ADR-007 day-1 set) but not yet wired: "
                "aeris, openweathermap, wunderground.",
                provider_id,
                exc,
            )
            sys.exit(1)
        declarations.append(module.CAPABILITY)

    # 3b-14: radar domain (keyless half — rainviewer, iem_nexrad, noaa_mrms,
    # msc_geomet, dwd_radolan).
    # 3b-15: keyed providers added (aeris, openweathermap).
    # 3b-16: iframe provider added; uses make_capability() factory (not CAPABILITY
    #   constant) so the iframe_url is embedded in the registered capability.
    # mapbox_jma deferred per ADR-015 2026-05-11 amendment.
    # Credentials for keyed providers wired separately via wire_radar_settings().
    if settings.radar.provider:
        provider_id = settings.radar.provider
        try:
            module = get_provider_module(domain="radar", provider_id=provider_id)
        except KeyError as exc:
            logger.critical(
                "FATAL: Unknown radar provider %r in api.conf — clearskies-api cannot start. "
                "Cause: %s. "
                "Check [radar] provider in api.conf. "
                "Supported values: "
                "rainviewer, iem_nexrad, noaa_mrms, msc_geomet, dwd_radolan (keyless); "
                "aeris, openweathermap (keyed); "
                "iframe (embed; requires iframe_url). "
                "mapbox_jma is not supported — deferred per ADR-015 2026-05-11 amendment.",
                provider_id,
                exc,
            )
            sys.exit(1)
        if provider_id == "iframe":
            declarations.append(module.make_capability(iframe_url=settings.radar.iframe_url))
        else:
            declarations.append(module.CAPABILITY)

    wire_providers(declarations)


def main() -> None:
    """Main entry point.

    Startup sequence (ADR-012, extended for ADR-038 TLS):
      0. Parse CLI args (--tls-cert, --tls-key).
      1. Bootstrap logging (INFO) so config-load errors are JSON.
      2. Load + validate settings from api.conf.
      2a. Apply CLI TLS overrides to settings.tls.
      3. Re-configure logging at the operator's log level.
      3a. Determine config_dir from the loaded config file path.
      3b. Ensure TLS cert exists (auto-generate if absent) — ADR-038.
      3c. Compute cert fingerprint for the operator banner.
      3d. Init TrustManager — generates/reads setup token from secrets.env.
      3e. Attach trust_manager to app.state for setup endpoints (Round 2).
      3f. Print operator startup banner.
      4. Build the SQLAlchemy engine.
      5. Run the write-probe — exits 1 if DB user has write privileges.
      6. Run schema reflection — logs unmapped columns; does NOT exit.
      6i. Wire provider registry.
      6j. Wire alerts settings.
      6k. Wire aqi settings.
      6l. Wire earthquakes settings.
      6m. Wire forecast settings.
      6n. Wire conditions settings (Phase 0B blending engine mode + station coords).
      6o. Wire radar settings (keyed provider credentials — 3b-15; no-op for keyless).
      7. Register DB health probe.
      8. Start uvicorn (TLS-enabled).
    """
    # Step 0: Parse CLI args before logging so --help works cleanly.
    args = _parse_args()

    # Step 1: Bootstrap logging before anything else so config errors appear
    # as JSON (ADR-029).
    setup_logging("INFO")

    # Step 2: Load and validate settings.
    settings = load_settings()

    # Step 2a: CLI flags override [tls] section values.
    if args.tls_cert is not None:
        settings.tls.cert_path = str(args.tls_cert)
    if args.tls_key is not None:
        settings.tls.key_path = str(args.tls_key)

    # Step 3: Reconfigure logging at the operator's level.
    setup_logging(settings.logging.level)

    # Step 3a: Determine config_dir from the resolved config file path.
    # find_config_file() follows the same ADR-027 search order as load_settings().
    # We call it here (after load_settings succeeded) so config_dir is always valid.
    _config_file = find_config_file()
    config_dir = _config_file.parent if _config_file is not None else Path("/etc/weewx-clearskies")

    # Step 3b: Ensure TLS cert (auto-generate if operator hasn't supplied one).
    cli_cert = Path(settings.tls.cert_path) if settings.tls.cert_path else None
    cli_key = Path(settings.tls.key_path) if settings.tls.key_path else None
    try:
        cert_path, key_path = ensure_tls_cert(config_dir, cli_cert, cli_key)
    except FileNotFoundError as exc:
        logger.critical(
            "FATAL: TLS cert/key not found — clearskies-api cannot start. Cause: %s. "
            "Check --tls-cert / --tls-key paths or [tls] cert_path / key_path in api.conf.",
            exc,
        )
        sys.exit(1)

    # Step 3c: Compute fingerprint for the operator banner.
    fingerprint = compute_fingerprint(cert_path)

    # Step 3d: Init TrustManager (reads/generates setup token from secrets.env).
    secrets_path = config_dir / "secrets.env"
    trust_manager = TrustManager(secrets_path=secrets_path)

    # Step 3e: Attach trust_manager, settings, and config_dir to the app for
    # setup endpoints (Round 2).
    app = create_app(settings)
    app.state.trust_manager = trust_manager
    app.state.settings = settings
    app.state.config_dir = config_dir

    # Step 3f: Print operator startup banner.
    address_url = _format_address_for_url(settings.api.bind_host, settings.api.bind_port)

    if not settings.configured:
        print(
            f"No configuration found — starting in setup mode.\n"
            f"  Address:     {address_url}\n"
            f"  Trust token: {trust_manager.token}\n"
            f"  Fingerprint: {fingerprint}\n"
            f"Visit the setup wizard to configure this installation."
        )
        # Skip DB, schema reflection, providers — none are available yet.
        _run_server(settings, cert_path=cert_path, key_path=key_path, app=app)
        return

    if trust_manager.setup_complete:
        print(f"API ready at {address_url}")
    else:
        print(
            f"API ready. To connect your config UI:\n"
            f"  Address:     {address_url}\n"
            f"  Trust token: {trust_manager.token}\n"
            f"  Fingerprint: {fingerprint}"
        )

    # Step 4: Build the SQLAlchemy engine.
    engine = build_engine(settings.database)
    wire_engine(engine)

    # Step 5: Write-probe — exits 1 if the connected user can write.
    # This must run BEFORE uvicorn starts and BEFORE schema reflection,
    # so the critical log appears before any other startup output.
    run_write_probe(engine)

    # Step 6: Schema reflection — build column registry.
    # Fatal on RuntimeError (missing archive table = service cannot serve data).
    # ADR-012: "refuse to start in known-bad states."  Operator ensures weewx
    # has run at least once to create the archive table, then restarts the api.
    # Individual unmapped-column warnings do NOT abort startup; only full
    # reflection failure (table missing, DB error) is fatal.
    reflector = SchemaReflector(engine)
    try:
        registry = reflector.reflect()
    except RuntimeError as exc:
        logger.critical(
            "FATAL: Schema reflection failed — clearskies-api cannot start. "
            "Cause: %s. "
            "Ensure weewx has run at least once so the archive table exists, "
            "then restart clearskies-api. "
            "Check [database] kind/path/host/name in api.conf and verify the "
            "DB user can SELECT from the archive table.",
            exc,
        )
        sys.exit(1)

    # Wire the column registry for DI use in endpoints.
    wire_registry(registry)

    # Step 6b: Load weewx.conf (shared ConfigObj cache for units + station).
    # Fatal if missing — required by both units and station metadata loaders.
    try:
        weewx_cfg = load_weewx_conf(settings.weewx.config_path)
    except WeewxConfLoadError as exc:
        logger.critical("%s", exc)
        sys.exit(1)

    # Step 6c: Load the units block from weewx.conf.  Fatal if missing.
    try:
        load_units_block(settings.weewx.config_path)
    except WeewxConfNotFoundError as exc:
        logger.critical("%s", exc)
        sys.exit(1)

    # Step 6d: Load station metadata from weewx.conf [Station].  Fatal if
    # required fields are missing (no location/latitude/longitude = misconfigured).
    try:
        load_station_metadata(
            cfg=weewx_cfg,
            api_station_id=settings.station.station_id,
            api_timezone=settings.station.timezone,
            unit_system=get_target_unit(),
            api_default_locale=settings.station.default_locale,
        )
    except StationConfigError as exc:
        logger.critical("%s", exc)
        sys.exit(1)

    # Step 6e: Wire ephemeris directory for almanac endpoints (ADR-014).
    # Fatal if directory not writable and de421.bsp not present, or if
    # download fails on first run. wire_ephemeris_directory calls sys.exit(1)
    # on fatal failure.
    wire_ephemeris_directory(settings.almanac.ephemeris_directory)

    # Step 6f: Wire reports directory.  Non-fatal — missing dir → empty
    # /reports response, not a startup abort.
    wire_reports_directory(settings.weewx.reports_directory)

    # Step 6g: Wire content directory.  Non-fatal — missing dir → 404 on
    # /content/* requests.
    wire_content_directory(settings.content.directory)

    # Step 6g: Wire hidden pages list.
    wire_hidden_pages(settings.pages.hidden)

    # Step 6h: Wire cache backend (ADR-017).
    # MemoryCache by default; RedisCache when CLEARSKIES_CACHE_URL is set.
    # Fail-closed: unreachable Redis → CRITICAL log + exit 1 (same as write-probe).
    try:
        wire_cache_from_env()
    except CacheConfigError as exc:
        logger.critical(
            "FATAL: Cache configuration error — clearskies-api cannot start. "
            "Cause: %s. "
            "Fix CLEARSKIES_CACHE_URL in your environment or secrets.env.",
            exc,
        )
        sys.exit(1)
    except RuntimeError as exc:
        logger.critical(
            "FATAL: Cache backend connection failed — clearskies-api cannot start. "
            "Cause: %s. "
            "Verify Redis is running and CLEARSKIES_CACHE_URL is correct.",
            exc,
        )
        sys.exit(1)

    # Step 6h½: Wire DB metrics (ADR-031).
    # SQLAlchemy event listeners for query timing. Metrics are always created
    # and always incremented; the /metrics endpoint exposure is controlled by
    # settings.health.metrics_enabled.
    from weewx_clearskies_api.metrics import wire_db_metrics  # noqa: PLC0415

    wire_db_metrics(engine)
    if settings.health.metrics_enabled:
        logger.info("Prometheus metrics enabled on health port (/metrics)")

    # Step 6i: Wire provider capability registry (ADR-038 §4).
    # Registers configured providers' CAPABILITY declarations.
    # Fail-closed: unknown provider id → CRITICAL log + exit 1.
    _wire_providers_from_config(settings)

    # Step 6j: Pass settings to alerts endpoint for provider dispatch.
    wire_alerts_settings(settings)

    # Step 6k: Pass settings to aqi endpoint.
    # Open-Meteo: no-op (keyless).
    # Aeris (3b-10): extracts client_id + client_secret from settings.aeris.
    # Future: OWM (3b-11), IQAir (3b-12).
    wire_aqi_settings(settings)

    # Step 6l: Pass settings to earthquakes endpoint (default_radius_km from api.conf).
    # All four providers are keyless — no credential wiring needed (ADR-040).
    wire_earthquakes_settings(settings)

    # Step 6m: Pass settings to forecast endpoint (NWS UA contact wiring).
    wire_forecast_settings(settings)

    # Step 6n: Wire conditions engine mode and station coordinates.
    # Must run after load_station_metadata() (step 6d) and load_units_block()
    # (step 6c) so that station lat/lon and target_unit are available.
    wire_conditions_settings(settings)

    # Step 6o: Pass settings to radar endpoint for keyed-provider credential wiring.
    # Keyless providers (rainviewer, iem_nexrad, noaa_mrms, msc_geomet, dwd_radolan):
    # no-op. Aeris + OWM: extracts credentials from settings.forecast per 3b-5 Q2
    # provider-scoped decision (same env vars as forecast/alerts/AQI).
    wire_radar_settings(settings)

    # Step 6p: Wire branding settings (ADR-022, Gap #10).
    wire_branding_settings(settings.branding)

    # Step 7: Register DB readiness probe.
    wire_db_health_probe()

    # Step 8: Start servers (TLS-enabled via cert_path / key_path from step 3b).
    _run_server(settings, cert_path=cert_path, key_path=key_path, app=app)


if __name__ == "__main__":
    main()
