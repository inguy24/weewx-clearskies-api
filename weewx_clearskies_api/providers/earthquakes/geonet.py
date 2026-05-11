"""GeoNet (NZ) earthquake provider module (ADR-040, ADR-038).

Five responsibilities per ADR-038 §2:
  1. Outbound API call — GeoNet /quake?MMI=-1
  2. Response parsing — wire-shape Pydantic models for _GeoNetResponse
  3. Translation to canonical EarthquakeRecord (field mapping per §4.4)
  4. Capability declaration — CAPABILITY symbol consumed at startup
  5. Error handling — provider errors translated to canonical taxonomy

GeoNet-specific notes:
  - GeoNet does NOT accept lat/lon/radius params — all events are returned
    and operator radius filter is applied at the endpoint layer post-fetch
    (lead-resolved call #4 2026-05-11).
  - Pass MMI=-1 to get all events regardless of intensity.
  - `properties.time` is ISO 8601 UTC string (not epoch ms) — use
    to_utc_iso8601_from_offset() to normalize.
  - `properties.mmi` is LOWERCASE (not "MMI") per geonet.md live capture 2026-05-11.
  - Depth is at `properties.depth` (positive km); geometry.coordinates is [lon, lat] only.
  - id = properties.publicID (no top-level Feature.id).
  - magnitudeType is not provided; canonical leaves as None per §4.4.
  - url: not in response; construct as f"https://www.geonet.org.nz/earthquake/{publicID}".

Cache layer (ADR-017):
  TTL: 60 s per user decision Q2 2026-05-11.

Wire-shape Pydantic (security-baseline §3.5):
  extra="ignore"; missing required fields raise ValidationError -> ProviderProtocolError.
  _to_canonical takes parsed model + raw dict (for extras population).

ruff: noqa: N815  (field names match GeoNet wire shape: publicID, etc.)
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

PROVIDER_ID = "geonet"
DOMAIN = "earthquakes"
BASE_URL = "https://api.geonet.org.nz"
PATH = "/quake"
_GEONET_CACHE_TTL = 60  # 60 s per user decision Q2 2026-05-11
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
        "depth",
        "place",
        "url",
        "mmi",
        "status",
        "source",
    ),
    geographic_coverage="nz",  # user decision Q1 2026-05-11
    auth_required=(),
    default_poll_interval_seconds=_GEONET_CACHE_TTL,
    operator_notes=(
        "GeoNet provides NZ-native MMI calculations and detailed regional coverage. "
        "Does not accept lat/lon/radius filter params — all events returned; "
        "radius filter applied at the endpoint layer. "
        "No API key required. CC BY 4.0 — attribution: GeoNet (https://www.geonet.org.nz/)."
    ),
)

# ---------------------------------------------------------------------------
# Rate limiter (ADR-038 §3) — 5 req/s polite-use guard
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="geonet-earthquakes",
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    max_calls=5,
    window_seconds=1,
)

# ---------------------------------------------------------------------------
# Wire-shape Pydantic models (security-baseline §3.5)
# Source: docs/reference/api-docs/geonet.md + live capture 2026-05-11
# ---------------------------------------------------------------------------


class _GeoNetEventProperties(BaseModel):
    """GeoNet quake feature properties — wire shape.

    extra="ignore" so GeoNet schema additions don't break us.
    mmi is lowercase per geonet.md live capture 2026-05-11 (API returns
    lowercase 'mmi' not 'MMI' despite the query param being uppercase).
    """

    model_config = ConfigDict(extra="ignore")

    publicID: str                  # canonical id
    time: str                      # ISO 8601 UTC string with Z
    depth: float                   # km below surface, positive
    magnitude: float
    mmi: int | None = None         # lowercase per live capture; NZ-calculated MMI
    locality: str | None = None    # canonical place
    quality: str | None = None     # "best", "preliminary", "automatic", "deleted"; extras


class _GeoNetEventGeometry(BaseModel):
    """GeoNet GeoJSON geometry — wire shape.

    GeoNet returns [lon, lat] only (2 elements, no depth in coordinates).
    """

    model_config = ConfigDict(extra="ignore")

    type: str
    coordinates: list[float]  # [lon, lat] — no depth element


class _GeoNetEventFeature(BaseModel):
    """GeoNet GeoJSON Feature wrapping one earthquake event.

    No top-level 'id' field — canonical id comes from properties.publicID.
    """

    model_config = ConfigDict(extra="ignore")

    type: Literal["Feature"]
    properties: _GeoNetEventProperties
    geometry: _GeoNetEventGeometry


class _GeoNetResponse(BaseModel):
    """GeoNet /quake response envelope — GeoJSON FeatureCollection."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["FeatureCollection"]
    features: list[_GeoNetEventFeature] = Field(default_factory=list)


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
    """Build a deterministic cache key.

    GeoNet doesn't accept lat/lon/radius params server-side, but the cache key
    still includes them so different radius configs use different cache entries.
    The from_dt/to_dt are not passed to GeoNet (7-day rolling window only) but
    included in the key for consistency.
    """
    payload = json.dumps(
        {
            "provider_id": PROVIDER_ID,
            "endpoint": PATH,
            "params": {
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "radius_km": radius_km,
                "from_dt": from_dt,
                "to_dt": to_dt,
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Wire -> canonical translation (canonical-data-model §4.4)
# ---------------------------------------------------------------------------


def _to_canonical(
    feature: _GeoNetEventFeature,
    raw_feature: dict[str, Any],
) -> EarthquakeRecord:
    """Map GeoNet feature to a canonical EarthquakeRecord.

    Args:
        feature: Parsed Pydantic feature model (typed access to canonical fields).
        raw_feature: Full GeoJSON Feature dict (must contain "properties" sub-key);
            used for extras population per lead-resolved call #1.
    """
    props = feature.properties
    coords = feature.geometry.coordinates

    raw_props: dict[str, Any] = raw_feature["properties"]

    # Extras per §4.4: quality is the only extras field for GeoNet.
    extras: dict[str, Any] = {}
    if raw_props.get("quality") is not None:
        extras["quality"] = raw_props["quality"]

    # url: construct from publicID (not in response per geonet.md).
    url = f"https://www.geonet.org.nz/earthquake/{props.publicID}"

    return EarthquakeRecord(
        id=props.publicID,
        time=to_utc_iso8601_from_offset(props.time, provider_id=PROVIDER_ID, domain=DOMAIN),
        latitude=coords[1],
        longitude=coords[0],
        magnitude=props.magnitude,
        magnitudeType=None,      # GeoNet does not supply magnitudeType per §4.4
        depth=props.depth,
        place=props.locality,
        url=url,
        tsunami=None,            # GeoNet does not supply tsunami flag
        felt=None,               # GeoNet does not supply felt count
        mmi=float(props.mmi) if props.mmi is not None else None,
        alert=None,              # GeoNet does not supply PAGER alert
        status=props.quality,    # "best"/"preliminary"/"automatic"/"deleted" per geonet.md
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
    """Call GeoNet /quake and return canonical EarthquakeRecord models.

    GeoNet does not accept lat/lon/radius query params — all NZ events are
    returned and radius filtering is applied at the endpoint layer post-fetch
    (lead-resolved call #4 2026-05-11). The MMI=-1 param returns all events
    regardless of intensity so the endpoint's min_magnitude filter has full
    data to work with.

    Cache stores post-normalization dicts (JSON-serialisable per ADR-017).

    Returns:
        List of canonical EarthquakeRecord models, possibly empty.

    Raises:
        QuotaExhausted: GeoNet returned 429.
        KeyInvalid: GeoNet returned 401/403 (exotic; GeoNet is keyless).
        TransientNetworkError: Network/DNS failure or 5xx after retries.
        ProviderProtocolError: Response validation failed (GeoNet schema change).
    """
    from_iso = from_dt.isoformat() if from_dt is not None else None
    to_iso = to_dt.isoformat() if to_dt is not None else None

    cache_key = _build_cache_key(lat, lon, radius_km, from_iso, to_iso)
    cached_dicts = get_cache().get(cache_key)
    if cached_dicts is not None:
        return [EarthquakeRecord.model_validate(d) for d in cached_dicts]

    _rate_limiter.acquire()

    # GeoNet requires MMI param; MMI=-1 returns all events (per geonet.md §Quirks).
    params: dict[str, Any] = {"MMI": -1}

    response = _get_http_client().get(f"{BASE_URL}{PATH}", params=params)

    try:
        raw_json = response.json()
        wire = _GeoNetResponse.model_validate(raw_json)
    except (ValidationError, ValueError) as exc:
        logger.error(
            "GeoNet response validation failed: %s. Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
        )
        raise ProviderProtocolError(
            f"GeoNet response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    raw_features: list[dict[str, Any]] = raw_json.get("features", [])
    try:
        paired = list(zip(wire.features, raw_features, strict=True))
    except ValueError as exc:
        raise ProviderProtocolError(
            f"GeoNet feature list length mismatch: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    canonical_records: list[EarthquakeRecord] = [
        _to_canonical(feature, raw_feature) for feature, raw_feature in paired
    ]

    get_cache().set(
        cache_key,
        [record.model_dump() for record in canonical_records],
        ttl_seconds=_GEONET_CACHE_TTL,
    )

    logger.info(
        "GeoNet earthquakes fetched: %d event(s) (all NZ; radius filter at endpoint layer)",
        len(canonical_records),
    )
    return canonical_records


def _reset_http_client_for_tests() -> None:
    """Reset module-level HTTP client singleton.  Used in tests only."""
    global _http_client  # noqa: PLW0603
    _http_client = None
