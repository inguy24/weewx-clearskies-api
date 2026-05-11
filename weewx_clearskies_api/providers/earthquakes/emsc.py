"""EMSC SeismicPortal earthquake provider module (ADR-040, ADR-038).

Five responsibilities per ADR-038 §2:
  1. Outbound API call — EMSC FDSN-Event /query?format=json
  2. Response parsing — wire-shape Pydantic models for _EmscResponse
  3. Translation to canonical EarthquakeRecord (field mapping per §4.4)
  4. Capability declaration — CAPABILITY symbol consumed at startup
  5. Error handling — provider errors translated to canonical taxonomy

EMSC-specific notes:
  - `properties.time` is ISO 8601 UTC with Z (micro-precision varies). Use
    to_utc_iso8601_from_offset() to normalize.
  - `properties.depth` is positive km below surface.
    DO NOT use geometry.coordinates[2] — that's negative (GeoJSON Z-up).
  - `properties.magtype` is LOWERCASE (differs from USGS/ReNaSS camelCase magType).
  - `properties.flynn_region` maps to canonical `place`.
  - `id` = top-level Feature.id (same as properties.unid; either works).
  - No status field in JSON flavor (XML has it); route via extras if needed.
  - EMSC uses `minmag` (not USGS's `minmagnitude`).
  - url: not in response; construct from unid.

Cache layer (ADR-017):
  TTL: 60 s per user decision Q2 2026-05-11.

Wire-shape Pydantic (security-baseline §3.5):
  extra="ignore"; missing required fields raise ValidationError -> ProviderProtocolError.
  _to_canonical takes parsed model + raw dict (for extras population).

ruff: noqa: N815  (field names match EMSC wire shape: flynn_region, magtype, etc.)
"""

# ruff: noqa: N815

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from weewx_clearskies_api.models.responses import EarthquakeRecord
from weewx_clearskies_api.providers._common.cache import get_cache
from weewx_clearskies_api.providers._common.capability import ProviderCapability
from weewx_clearskies_api.providers._common.datetime_utils import to_utc_iso8601_from_offset
from weewx_clearskies_api.providers._common.errors import ProviderProtocolError
from weewx_clearskies_api.providers._common.http import ProviderHTTPClient
from weewx_clearskies_api.providers._common.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PROVIDER_ID = "emsc"
DOMAIN = "earthquakes"
BASE_URL = "https://www.seismicportal.eu"
PATH = "/fdsnws/event/1/query"
_EMSC_CACHE_TTL = 60  # 60 s per user decision Q2 2026-05-11
_API_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Capability declaration (ADR-038 §4)
# ---------------------------------------------------------------------------

CAPABILITY = ProviderCapability(
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    supplied_canonical_fields=(
        "id",
        "time",
        "latitude",
        "longitude",
        "magnitude",
        "magnitudeType",
        "depth",
        "place",
        "url",
        "source",
    ),
    geographic_coverage="global, primary in eu+mediterranean",  # user decision Q1 2026-05-11
    auth_required=(),
    default_poll_interval_seconds=_EMSC_CACHE_TTL,
    operator_notes=(
        "EMSC provides primary coverage for EU + Mediterranean; global supplementary. "
        "No API key required. CC BY 4.0 — attribution: EMSC "
        "(https://www.emsc-csem.org/), CC BY 4.0."
    ),
)

# ---------------------------------------------------------------------------
# Rate limiter (ADR-038 §3) — 5 req/s polite-use guard
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="emsc-earthquakes",
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    max_calls=5,
    window_seconds=1,
)

# ---------------------------------------------------------------------------
# Wire-shape Pydantic models (security-baseline §3.5)
# Source: docs/reference/api-docs/emsc.md + live capture 2026-05-11
# ---------------------------------------------------------------------------


class _EmscEventProperties(BaseModel):
    """EMSC FDSN-Event JSON feature properties — wire shape.

    extra="ignore" so EMSC schema additions don't break us.
    magtype is lowercase (EMSC convention, differs from USGS/ReNaSS magType camelCase).
    depth is positive km; geometry.coordinates[2] is negative (GeoJSON Z-up) — use depth.
    """

    model_config = ConfigDict(extra="ignore")

    time: str              # ISO 8601 UTC string (variable micro-precision)
    depth: float           # positive km below surface
    lat: float             # latitude (duplicated from geometry)
    lon: float             # longitude (duplicated from geometry)
    mag: float             # magnitude
    magtype: str | None = None     # lowercase e.g. "md", "ml", "mb", "mw"
    flynn_region: str | None = None   # canonical place
    unid: str | None = None           # EMSC unique ID (same as Feature.id)
    evtype: str | None = None         # extras: event type code e.g. "ke", "se"
    auth: str | None = None           # extras: publishing agency code
    source_id: str | None = None      # extras: upstream catalog ID
    source_catalog: str | None = None # extras: upstream catalog name
    lastupdate: str | None = None     # extras: last revision timestamp


class _EmscEventGeometry(BaseModel):
    """EMSC GeoJSON geometry — wire shape."""

    model_config = ConfigDict(extra="ignore")

    type: str
    coordinates: list[float]  # [lon, lat, -depth_km] (negative Z = below surface)


class _EmscEventFeature(BaseModel):
    """EMSC GeoJSON Feature wrapping one earthquake event."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["Feature"]
    id: str                            # top-level Feature ID (= properties.unid)
    properties: _EmscEventProperties
    geometry: _EmscEventGeometry


class _EmscResponse(BaseModel):
    """EMSC FDSN-Event JSON FeatureCollection response envelope."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["FeatureCollection"]
    features: list[_EmscEventFeature] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# HTTP client (module-level singleton)
# ---------------------------------------------------------------------------

_http_client: ProviderHTTPClient | None = None


def _get_http_client() -> ProviderHTTPClient:
    global _http_client  # noqa: PLW0603
    if _http_client is None:
        _http_client = ProviderHTTPClient(
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
            user_agent=f"weewx-clearskies-api/{_API_VERSION}",
        )
    return _http_client


# ---------------------------------------------------------------------------
# Cache key construction (ADR-017 §Cache key)
# ---------------------------------------------------------------------------


def _build_cache_key(
    lat: float,
    lon: float,
    radius_km: float,
    from_dt: str | None,
    to_dt: str | None,
) -> str:
    """Build a deterministic cache key."""
    payload = json.dumps(
        {
            "provider_id": PROVIDER_ID,
            "endpoint": PATH,
            "params": {
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "maxradiuskm": radius_km,
                "starttime": from_dt,
                "endtime": to_dt,
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Wire -> canonical translation (canonical-data-model §4.4)
# ---------------------------------------------------------------------------


def _to_canonical(
    feature: _EmscEventFeature,
    raw_feature: dict[str, Any],
) -> EarthquakeRecord:
    """Map EMSC feature to a canonical EarthquakeRecord.

    Args:
        feature: Parsed Pydantic feature model (typed access to canonical fields).
        raw_feature: Full GeoJSON Feature dict (must contain "properties" sub-key);
            used for extras population per lead-resolved call #1.
    """
    props = feature.properties

    # url: construct from unid (not in response per emsc.md).
    unid = props.unid or feature.id
    url = f"https://www.seismicportal.eu/eventdetails.html?unid={unid}"

    raw_props: dict[str, Any] = raw_feature["properties"]

    # Extras per §4.4: evtype, auth, source_id, source_catalog, lastupdate.
    extras: dict[str, Any] = {}
    for key in ("evtype", "auth", "source_id", "source_catalog", "lastupdate"):
        val = raw_props.get(key)
        if val is not None:
            extras[key] = val

    return EarthquakeRecord(
        id=feature.id,
        time=to_utc_iso8601_from_offset(props.time, provider_id=PROVIDER_ID, domain=DOMAIN),
        latitude=props.lat,
        longitude=props.lon,
        magnitude=props.mag,
        magnitudeType=props.magtype,   # lowercase from EMSC; canonical accepts as-is
        depth=props.depth,
        place=props.flynn_region,
        url=url,
        tsunami=None,    # EMSC JSON flavor does not supply tsunami flag
        felt=None,       # EMSC does not supply felt count
        mmi=None,        # EMSC does not supply MMI
        alert=None,      # EMSC does not supply PAGER alert
        status=None,     # EMSC JSON flavor drops status; XML has it (out of v0.1 scope)
        extras=extras,
        source=PROVIDER_ID,
    )


# ---------------------------------------------------------------------------
# Public fetch entrypoint (ADR-038 §2)
# ---------------------------------------------------------------------------


def fetch(
    *,
    lat: float,
    lon: float,
    radius_km: float,
    from_dt: datetime | None,
    to_dt: datetime | None,
) -> list[EarthquakeRecord]:
    """Call EMSC FDSN-Event API and return canonical EarthquakeRecord models.

    Cache stores post-normalization dicts (JSON-serialisable per ADR-017).

    Args:
        lat: Station latitude (WGS84).
        lon: Station longitude (WGS84).
        radius_km: Radius in km from station (server-side filter via maxradiuskm).
        from_dt: Start time (passed to EMSC starttime as ISO 8601). None = no lower bound.
        to_dt: End time (passed to EMSC endtime as ISO 8601). None = no upper bound.

    Returns:
        List of canonical EarthquakeRecord models, possibly empty.

    Raises:
        QuotaExhausted: EMSC returned 429.
        KeyInvalid: EMSC returned 401/403 (exotic; EMSC is keyless).
        TransientNetworkError: Network/DNS failure or 5xx after retries.
        ProviderProtocolError: Response validation failed (EMSC schema change).
    """
    from_iso = from_dt.isoformat() if from_dt is not None else None
    to_iso = to_dt.isoformat() if to_dt is not None else None

    cache_key = _build_cache_key(lat, lon, radius_km, from_iso, to_iso)
    cached_dicts = get_cache().get(cache_key)
    if cached_dicts is not None:
        return [EarthquakeRecord.model_validate(d) for d in cached_dicts]

    _rate_limiter.acquire()

    # EMSC uses `minmag` (not `minmagnitude` like USGS) and `maxradiuskm`.
    params: dict[str, Any] = {
        "format": "json",
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "maxradiuskm": radius_km,
        "orderby": "time",
    }
    if from_iso is not None:
        params["starttime"] = from_iso
    if to_iso is not None:
        params["endtime"] = to_iso

    response = _get_http_client().get(f"{BASE_URL}{PATH}", params=params)

    try:
        raw_json = response.json()
        wire = _EmscResponse.model_validate(raw_json)
    except (ValidationError, ValueError) as exc:
        logger.error(
            "EMSC response validation failed: %s. Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
        )
        raise ProviderProtocolError(
            f"EMSC response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    raw_features: list[dict[str, Any]] = raw_json.get("features", [])
    try:
        paired = list(zip(wire.features, raw_features, strict=True))
    except ValueError as exc:
        raise ProviderProtocolError(
            f"EMSC feature list length mismatch: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    canonical_records: list[EarthquakeRecord] = [
        _to_canonical(feature, raw_feature) for feature, raw_feature in paired
    ]

    get_cache().set(
        cache_key,
        [record.model_dump() for record in canonical_records],
        ttl_seconds=_EMSC_CACHE_TTL,
    )

    logger.info(
        "EMSC earthquakes fetched: %d event(s) for lat=%.4f lon=%.4f radius_km=%.1f",
        len(canonical_records),
        lat,
        lon,
        radius_km,
    )
    return canonical_records


def _reset_http_client_for_tests() -> None:
    """Reset module-level HTTP client singleton.  Used in tests only."""
    global _http_client  # noqa: PLW0603
    _http_client = None
