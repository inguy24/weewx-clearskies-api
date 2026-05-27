"""Climatology endpoint: GET /climatology/monthly.

Returns 12-month average values (high temp, low temp, dewpoint, rainfall)
computed from the full weewx archive.

Per ADR-018: URL-path versioned under /api/v1/.
Per ADR-019: units block is NOT included — climatology values are raw weewx
  archive units; the dashboard applies unit conversion for display.
Per ADR-020: generatedAt is UTC ISO-8601 with Z suffix.

Fields self-hide when their backing archive column is absent from the
ColumnRegistry (consistent with the records service self-hide rule).

ruff: noqa: N815  (canonical field names are weewx camelCase per ADR-010)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from weewx_clearskies_api.db.registry import get_registry
from weewx_clearskies_api.db.session import get_db_session
from weewx_clearskies_api.providers._common.cache import get_cache
from weewx_clearskies_api.services.climatology import get_monthly_climatology

logger = logging.getLogger(__name__)

router = APIRouter()


def _now_utc_z() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@router.get(
    "/climatology/monthly",
    summary="12-month climatology averages",
    tags=["Climatology"],
)
def get_monthly_climatology_endpoint(
    db: Annotated[Session, Depends(get_db_session)],
) -> dict:
    """Return 12-element arrays of monthly climatology averages.

    Computes averages across all years in the archive:
      - avgHighTemp: average of daily maximum outTemp per month
      - avgLowTemp:  average of daily minimum outTemp per month
      - avgDewpoint: straight average dewpoint per month
      - avgRainfall: average of monthly total rainfall per month

    Fields whose archive column is absent from the ColumnRegistry are
    omitted from the response (self-hide rule).
    """
    # Cache-check-first guard (ADR-045).  The warmer pre-computes the monthly
    # climatology on a 6-hour interval; use the cached result when available.
    try:
        cached = get_cache().get("warmer:climatology:monthly")
        if cached is not None:
            logger.debug("climatology cache hit")
            return {
                "data": cached,
                "source": "weewx",
                "generatedAt": _now_utc_z(),
            }
    except Exception:
        logger.debug("climatology cache miss or error", exc_info=True)

    registry = get_registry()

    clim_data = get_monthly_climatology(db=db, registry=registry)

    return {
        "data": clim_data,
        "source": "weewx",
        "generatedAt": _now_utc_z(),
    }
