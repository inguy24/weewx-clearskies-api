"""USGS earthquake provider module (ADR-040, ADR-038).

Five responsibilities per ADR-038 §2:
  1. Outbound API call — USGS FDSN-Event /query?format=geojson
  2. Response parsing — wire-shape Pydantic models for _UsgsResponse
  3. Translation to canonical EarthquakeRecord (field mapping per §4.4)
  4. Capability declaration — CAPABILITY symbol consumed at startup
  5. Error handling — provider errors translated to canonical taxonomy

USGS-specific notes:
  - `properties.time` is epoch milliseconds (not ISO 8601). Use
    epoch_ms_to_utc_iso8601() — NOT epoch_to_utc_iso8601() which takes seconds.
    Numerical sanity: 1778492931604 ms / 1000 = 1778492931.604 s -> 2026-05-11.
  - `properties.tsunami` is 0/1 integer; cast to bool at canonical mapping.
  - `geometry.coordinates[2]` is positive km below surface (no sign flip needed).
  - USGS-specific param names: `minmagnitude` (not `minmag`), `maxradiuskm`.
  - `id` is top-level Feature.id (stable across re-publishes per usgs.md).

Cache layer (ADR-017):
  Caches post-normalization canonical list (JSON-serialisable for Redis).
  Key: SHA-256 of (provider_id, path, {latitude, longitude, maxradiuskm, starttime, endtime}).
  TTL: 60 s (earthquake feeds update every ~minute per user decision Q2 2026-05-11).

Wire-shape Pydantic (security-baseline §3.5):
  extra="ignore" so USGS schema additions don't break us; missing required
  fields raise ValidationError -> ProviderProtocolError.
  _to_canonical takes parsed model + raw dict (for extras population) per
  lead-resolved call #1 — matches providers/forecast/openweathermap.py precedent.

ruff: noqa: N815  (field names match wire camelCase: magType, etc.)
"""

# ruff: noqa: N815

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from weewx_clearskies_api.models.responses import EarthquakeRecord
from weewx_clearskies_api.providers._common.cache import get_cache
from weewx_clearskies_api.providers._common.capability import ProviderCapability
from weewx_clearskies_api.providers._common.datetime_utils import epoch_ms_to_utc_iso8601
from weewx_clearskies_api.providers._common.errors import ProviderProtocolError
from weewx_clearskies_api.providers._common.http import ProviderHTTPClient
from weewx_clearskies_api.providers._common.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PROVIDER_ID = "usgs"
DOMAIN = "earthquakes"
BASE_URL = "https://earthquake.usgs.gov"
PATH = "/fdsnws/event/1/query"
_USGS_CACHE_TTL = 60  # 60 s per user decision Q2 2026-05-11

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
        "tsunami",
        "felt",
        "mmi",
        "alert",
        "status",
        "source",
    ),
    geographic_coverage="global",  # user decision Q1 2026-05-11
    auth_required=(),
    default_poll_interval_seconds=_USGS_CACHE_TTL,
    operator_notes=(
        "USGS provides global M2.5+ coverage and is the recommended fallback "
        "for operators without a regional provider (ADR-040 §No uncovered-region case). "
        "No API key required. Polite-use: do not poll faster than the TTL."
    ),
)

# ---------------------------------------------------------------------------
# Rate limiter (ADR-038 §3) — 5 req/s polite-use guard
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="usgs-earthquakes",
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    max_calls=5,
    window_seconds=1,
)

# ---------------------------------------------------------------------------
# Wire-shape Pydantic models (security-baseline §3.5)
# Source: docs/reference/api-docs/usgs.md + live capture 2026-05-11
# ---------------------------------------------------------------------------


class _UsgsEventProperties(BaseModel):
    """USGS GeoJSON feature properties — wire shape.

    extra="ignore" so USGS schema additions don't break us; missing required
    fields raise ValidationError -> ProviderProtocolError.
    """

    model_config = ConfigDict(extra="ignore")

    mag: float
    place: str | None = None
    time: int  # epoch milliseconds
    updated: int | None = None  # epoch milliseconds
    url: str | None = None
    felt: int | None = None
    cdi: float | None = None   # Community Decimal Intensity; extras
    mmi: float | None = None
    alert: str | None = None
    status: str | None = None
    tsunami: int = 0            # 0 or 1 integer; cast to bool at canonical layer
    sig: int | None = None      # significance; extras
    net: str | None = None      # contributing network; extras
    code: str | None = None     # network event code; extras
    ids: str | None = None      # comma-separated event ids; extras
    sources: str | None = None  # comma-separated networks; extras
    types: str | None = None    # comma-separated product types; extras
    nst: int | None = None      # number of stations; extras
    dmin: float | None = None   # minimum distance; extras
    rms: float | None = None    # RMS travel time residual; extras
    gap: float | None = None    # azimuthal gap; extras
    magType: str | None = None
    type: str | None = None     # event type (earthquake/explosion/etc.); extras
    title: str | None = None    # extras


class _UsgsEventGeometry(BaseModel):
    """USGS GeoJSON geometry — wire shape."""

    model_config = ConfigDict(extra="ignore")

    type: str
    coordinates: list[float]  # [lon, lat, depth_km_positive]


class _UsgsEventFeature(BaseModel):
    """USGS GeoJSON Feature wrapping one earthquake event."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["Feature"]
    id: str                             # top-level Feature ID (stable)
    properties: _UsgsEventProperties
    geometry: _UsgsEventGeometry


class _UsgsResponse(BaseModel):
    """USGS FDSN-Event GeoJSON FeatureCollection response envelope."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["FeatureCollection"]
    features: list[_UsgsEventFeature] = Field(default_factory=list)
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
    """Build a deterministic cache key.

    Includes radius_km + from_dt + to_dt — different radii / time windows
    return different result sets (lead-resolved call #8 2026-05-11).
    """
    payload = json.dumps(
        {
            "provider_id": PROVIDER_ID,
            "endpoint": PATH,
            "params": {
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
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
    feature: _UsgsEventFeature,
    raw_feature: dict[str, Any],
) -> EarthquakeRecord:
    """Map USGS feature to a canonical EarthquakeRecord.

    Args:
        feature: Parsed Pydantic feature model (typed access to canonical fields).
        raw_feature: Raw feature dict (for extras population per lead-resolved call #1).
            May be the full GeoJSON Feature dict (with "properties" sub-key) OR just
            the properties dict. Extras extraction reads from raw_feature["properties"]
            when present, otherwise treats raw_feature as the properties dict directly.
    """
    props = feature.properties
    coords = feature.geometry.coordinates

    # Depth: coordinates[2] is positive km below surface for USGS (no sign flip).
    depth_km = coords[2] if len(coords) >= 3 else None

    # Extras per §4.4: route provider-specific fields not in canonical.
    # Normalize: handle both full feature dict ({"properties": {...}}) and bare props dict.
    raw_props: dict[str, Any] = raw_feature.get("properties", raw_feature)

    extras: dict[str, Any] = {}
    for key in ("cdi", "sig", "net", "code", "ids", "sources", "types", "nst", "dmin", "rms", "gap", "type", "title"):
        val = raw_props.get(key)
        if val is not None:
            extras[key] = val

    return EarthquakeRecord(
        id=feature.id,
        time=epoch_ms_to_utc_iso8601(props.time, provider_id=PROVIDER_ID, domain=DOMAIN),
        latitude=coords[1],
        longitude=coords[0],
        magnitude=props.mag,
        magnitudeType=props.magType,
        depth=depth_km,
        place=props.place,
        url=props.url,
        tsunami=bool(props.tsunami),
        felt=props.felt,
        mmi=props.mmi,
        alert=props.alert,
        status=props.status,
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
    from_dt: str | None,
    to_dt: str | None,
) -> list[EarthquakeRecord]:
    """Call USGS FDSN-Event API and return canonical EarthquakeRecord models.

    Cache stores post-normalization dicts (JSON-serialisable per ADR-017);
    on cache hit the dicts are reconstructed into EarthquakeRecord models.

    Args:
        lat: Station latitude (WGS84).
        lon: Station longitude (WGS84).
        radius_km: Radius in km from station to include events (server-side filter).
        from_dt: ISO 8601 start time (passed to USGS starttime param). None = no lower bound.
        to_dt: ISO 8601 end time (passed to USGS endtime param). None = no upper bound.

    Returns:
        List of canonical EarthquakeRecord models, possibly empty.

    Raises:
        QuotaExhausted: USGS returned 429.
        KeyInvalid: USGS returned 401/403 (exotic; USGS is keyless).
        TransientNetworkError: Network/DNS failure or 5xx after retries.
        ProviderProtocolError: Response validation failed (USGS schema change).
    """
    cache_key = _build_cache_key(lat, lon, radius_km, from_dt, to_dt)
    cached_dicts = get_cache().get(cache_key)
    if cached_dicts is not None:
        return [EarthquakeRecord.model_validate(d) for d in cached_dicts]

    _rate_limiter.acquire()

    params: dict[str, Any] = {
        "format": "geojson",
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "maxradiuskm": radius_km,
        "orderby": "time",
    }
    if from_dt is not None:
        params["starttime"] = from_dt
    if to_dt is not None:
        params["endtime"] = to_dt

    response = _get_http_client().get(f"{BASE_URL}{PATH}", params=params)

    try:
        raw_json = response.json()
        wire = _UsgsResponse.model_validate(raw_json)
    except (ValidationError, ValueError) as exc:
        logger.error(
            "USGS response validation failed: %s. Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
        )
        raise ProviderProtocolError(
            f"USGS response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    # raw_features: list of raw dicts for extras population (lead-resolved call #1).
    raw_features: list[dict[str, Any]] = raw_json.get("features", [])

    canonical_records: list[EarthquakeRecord] = []
    for idx, feature in enumerate(wire.features):
        # Pass the full raw feature dict (contains "properties" sub-key).
        # _to_canonical normalizes to extract properties-level extras.
        try:
            raw_feature = raw_features[idx]
        except (IndexError, TypeError):
            raw_feature = {}
        canonical_records.append(_to_canonical(feature, raw_feature))

    get_cache().set(
        cache_key,
        [record.model_dump() for record in canonical_records],
        ttl_seconds=_USGS_CACHE_TTL,
    )

    logger.info(
        "USGS earthquakes fetched: %d event(s) for lat=%.4f lon=%.4f radius_km=%.1f",
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


# ---------------------------------------------------------------------------
# Name aliases for test compatibility
# Test-author used all-caps provider prefix (_USGSResponse, _USGSFeature)
# rather than the brief-prescribed mixed-case (_UsgsResponse, _UsgsEventFeature).
# Private implementation names; aliases allow tests to import either form.
# ---------------------------------------------------------------------------
_USGSResponse = _UsgsResponse
_USGSFeature = _UsgsEventFeature
