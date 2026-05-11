"""ReNaSS / EPOS-France earthquake provider module (ADR-040, ADR-038).

Five responsibilities per ADR-038 §2:
  1. Outbound API call — ReNaSS FDSN-Event /query?format=json
  2. Response parsing — wire-shape Pydantic models for _RenassResponse
  3. Translation to canonical EarthquakeRecord (field mapping per §4.4)
  4. Capability declaration — CAPABILITY symbol consumed at startup
  5. Error handling — provider errors translated to canonical taxonomy

ReNaSS-specific notes:
  - NEW endpoint host: api.franceseisme.fr (legacy renass.unistra.fr returns
    HTTP 404 since EPOS-France migration; verified 2026-05-11).
  - `properties.time` is ISO 8601 UTC with Z (microsecond precision). Use
    to_utc_iso8601_from_offset() to normalize.
  - `properties.depth` is positive km below surface.
    DO NOT use geometry.coordinates[2] — that's negative (GeoJSON Z-up).
  - `properties.description` is a bilingual dict {fr, en}. canonical `place`
    reads .en; .fr routes to extras["description_fr"] (flat key per LC#5).
  - `properties.url` is a bilingual dict {fr, en}. canonical `url` reads .en;
    .fr routes to extras["url_fr"] (flat key per LC#5).
  - `properties.automatic` boolean -> status: true->"automatic", false->"reviewed".
  - `properties.magType` is camelCase (like USGS; differs from EMSC's lowercase).
  - `id` is top-level Feature.id (no properties.publicID / unid).
  - Extras per §4.4: type, description_fr, url_fr.

Cache layer (ADR-017):
  TTL: 60 s per user decision Q2 2026-05-11.

Wire-shape Pydantic (security-baseline §3.5):
  extra="ignore"; missing required fields raise ValidationError -> ProviderProtocolError.
  description and url declared as dict[str, str] | None to handle bilingual objects.
  _to_canonical takes parsed model + raw dict (for extras population).

ruff: noqa: N815  (field names match ReNaSS wire shape: magType, etc.)
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

PROVIDER_ID = "renass"
DOMAIN = "earthquakes"
# NEW URL per ADR-040 References and renass.md (legacy renass.unistra.fr returns 404).
BASE_URL = "https://api.franceseisme.fr"
PATH = "/fdsnws/event/1/query"
_RENASS_CACHE_TTL = 60  # 60 s per user decision Q2 2026-05-11
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
        "status",
        "source",
    ),
    geographic_coverage="fr",  # user decision Q1 2026-05-11
    auth_required=(),
    default_poll_interval_seconds=_RENASS_CACHE_TTL,
    operator_notes=(
        "ReNaSS / EPOS-France provides regional coverage for mainland France and "
        "neighbouring countries. No API key required. "
        "CC BY 4.0 — attribution: BCSF-Rénass / EPOS-France "
        "(https://api.franceseisme.fr/), CC BY 4.0. "
        "Note: the feed includes quarry blasts and explosions (properties.type); "
        "filter at the dashboard layer if only seismic earthquakes are wanted."
    ),
)

# ---------------------------------------------------------------------------
# Rate limiter (ADR-038 §3) — 5 req/s polite-use guard
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter(
    name="renass-earthquakes",
    provider_id=PROVIDER_ID,
    domain=DOMAIN,
    max_calls=5,
    window_seconds=1,
)

# ---------------------------------------------------------------------------
# Wire-shape Pydantic models (security-baseline §3.5)
# Source: docs/reference/api-docs/renass.md + live capture 2026-05-11
# ---------------------------------------------------------------------------


class _RenassEventProperties(BaseModel):
    """ReNaSS FDSN-Event JSON feature properties — wire shape.

    extra="ignore" so ReNaSS schema additions don't break us.
    description and url are bilingual dicts {fr, en}; we read .en for canonical
    fields and route .fr to extras (lead-resolved call #5 2026-05-11).
    automatic boolean maps to canonical status.
    """

    model_config = ConfigDict(extra="ignore")

    time: str                               # ISO 8601 UTC string with Z
    depth: float                            # positive km below surface
    latitude: float                         # from properties (duplicated in geometry)
    longitude: float                        # from properties (duplicated in geometry)
    mag: float                              # magnitude
    magType: str | None = None              # camelCase e.g. "ML", "MLv"
    description: dict[str, str] | None = None  # bilingual {fr, en} place name
    url: dict[str, str] | None = None          # bilingual {fr, en} detail URL
    automatic: bool | None = None              # true=automatic, false=reviewed
    type: str | None = None                    # extras: event type (null/"earthquake"/"quarry blast")


class _RenassEventGeometry(BaseModel):
    """ReNaSS GeoJSON geometry — wire shape."""

    model_config = ConfigDict(extra="ignore")

    type: str
    coordinates: list[float]  # [lon, lat, -depth_km] (negative Z = below surface)


class _RenassEventFeature(BaseModel):
    """ReNaSS GeoJSON Feature wrapping one event."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["Feature"]
    id: str                              # top-level Feature ID (only id source)
    properties: _RenassEventProperties
    geometry: _RenassEventGeometry


class _RenassResponse(BaseModel):
    """ReNaSS FDSN-Event JSON FeatureCollection response envelope."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["FeatureCollection"]
    features: list[_RenassEventFeature] = Field(default_factory=list)


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
    feature: _RenassEventFeature,
    raw_feature: dict[str, Any],
) -> EarthquakeRecord:
    """Map ReNaSS feature to a canonical EarthquakeRecord.

    Args:
        feature: Parsed Pydantic feature model (typed access to canonical fields).
        raw_feature: Full GeoJSON Feature dict (must contain "properties" sub-key);
            used for extras population per lead-resolved call #1.
    """
    props = feature.properties

    # place: description.en (bilingual object); None if absent.
    place: str | None = None
    if props.description and isinstance(props.description, dict):
        place = props.description.get("en")

    # url: url.en (bilingual object); None if absent.
    url_str: str | None = None
    if props.url and isinstance(props.url, dict):
        url_str = props.url.get("en")

    # status: derived from automatic boolean per §4.4.
    status: str | None = None
    if props.automatic is True:
        status = "automatic"
    elif props.automatic is False:
        status = "reviewed"

    raw_props: dict[str, Any] = raw_feature["properties"]

    # Extras per §4.4 (LC#5): type, description_fr (flat key), url_fr (flat key).
    # Note: type is always included in extras even when null — the test verifies
    # extras["type"] is None for standard earthquake events and "quarry blast" for others.
    extras: dict[str, Any] = {}
    # "type" key exists in ReNaSS response even when null (None); always include.
    extras["type"] = raw_props.get("type")  # None or "earthquake"/"quarry blast"/"explosion"
    raw_desc = raw_props.get("description")
    if isinstance(raw_desc, dict) and "fr" in raw_desc:
        extras["description_fr"] = raw_desc["fr"]
    raw_url = raw_props.get("url")
    if isinstance(raw_url, dict) and "fr" in raw_url:
        extras["url_fr"] = raw_url["fr"]

    return EarthquakeRecord(
        id=feature.id,
        time=to_utc_iso8601_from_offset(props.time, provider_id=PROVIDER_ID, domain=DOMAIN),
        latitude=props.latitude,
        longitude=props.longitude,
        magnitude=props.mag,
        magnitudeType=props.magType,
        depth=props.depth,
        place=place,
        url=url_str,
        tsunami=None,   # ReNaSS does not supply tsunami flag
        felt=None,      # ReNaSS does not supply felt count
        mmi=None,       # ReNaSS does not supply MMI
        alert=None,     # ReNaSS does not supply PAGER alert
        status=status,
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
    """Call ReNaSS FDSN-Event API and return canonical EarthquakeRecord models.

    Uses the new api.franceseisme.fr endpoint (legacy renass.unistra.fr returns
    HTTP 404 since EPOS-France migration — per ADR-040 References and renass.md).

    Cache stores post-normalization dicts (JSON-serialisable per ADR-017).

    Args:
        lat: Station latitude (WGS84).
        lon: Station longitude (WGS84).
        radius_km: Radius in km from station (server-side filter via maxradiuskm).
        from_dt: Start time (passed to ReNaSS starttime as ISO 8601). None = no lower bound.
        to_dt: End time (passed to ReNaSS endtime as ISO 8601). None = no upper bound.

    Returns:
        List of canonical EarthquakeRecord models, possibly empty.

    Raises:
        QuotaExhausted: ReNaSS returned 429.
        KeyInvalid: ReNaSS returned 401/403 (exotic; ReNaSS is keyless).
        TransientNetworkError: Network/DNS failure or 5xx after retries.
        ProviderProtocolError: Response validation failed (ReNaSS schema change).
    """
    from_iso = from_dt.isoformat() if from_dt is not None else None
    to_iso = to_dt.isoformat() if to_dt is not None else None

    cache_key = _build_cache_key(lat, lon, radius_km, from_iso, to_iso)
    cached_dicts = get_cache().get(cache_key)
    if cached_dicts is not None:
        return [EarthquakeRecord.model_validate(d) for d in cached_dicts]

    _rate_limiter.acquire()

    # ReNaSS uses `minmag` (not `minmagnitude`) and `maxradiuskm`.
    # Param names mirror EMSC (same FDSN-Event implementation, different host).
    params: dict[str, Any] = {
        "format": "json",
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
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
        wire = _RenassResponse.model_validate(raw_json)
    except (ValidationError, ValueError) as exc:
        logger.error(
            "ReNaSS response validation failed: %s. Response body (first 2000 chars): %.2000s",
            exc,
            response.text,
        )
        raise ProviderProtocolError(
            f"ReNaSS response validation failed: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    raw_features: list[dict[str, Any]] = raw_json.get("features", [])
    try:
        paired = list(zip(wire.features, raw_features, strict=True))
    except ValueError as exc:
        raise ProviderProtocolError(
            f"ReNaSS feature list length mismatch: {exc}",
            provider_id=PROVIDER_ID,
            domain=DOMAIN,
        ) from exc

    canonical_records: list[EarthquakeRecord] = [
        _to_canonical(feature, raw_feature) for feature, raw_feature in paired
    ]

    get_cache().set(
        cache_key,
        [record.model_dump() for record in canonical_records],
        ttl_seconds=_RENASS_CACHE_TTL,
    )

    logger.info(
        "ReNaSS earthquakes fetched: %d event(s) for lat=%.4f lon=%.4f radius_km=%.1f",
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
