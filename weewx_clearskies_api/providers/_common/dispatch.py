"""Provider module dispatch.

Maps (domain, provider_id) → the provider module's CAPABILITY + fetch entrypoint.
Phase-2 simple: explicit dict, NOT entry-points (per ADR-038 §Internal contract —
no runtime plugin loading; outside contributors PR into the bundled set).

Adding a new provider = importing the new module and adding one row here.
When 3b round 2 adds Aeris alerts: one new import + one new row.
When forecast domain lands: five new rows (one per provider per ADR-007 day-1 set).
  Wired so far: openmeteo (3b-2), nws (3b-3), aeris (3b-4), openweathermap (3b-5).
  Future rounds add wunderground.
  ForecastSettings.validate() accepts all five; dispatch KeyError at startup
  catches the "accepted but not yet wired" case (fail-closed per brief).
"""

from __future__ import annotations

from types import ModuleType

from weewx_clearskies_api.providers.alerts import nws as alerts_nws
from weewx_clearskies_api.providers.forecast import aeris as forecast_aeris
from weewx_clearskies_api.providers.forecast import nws as forecast_nws
from weewx_clearskies_api.providers.forecast import openmeteo as forecast_openmeteo
from weewx_clearskies_api.providers.forecast import openweathermap as forecast_openweathermap

PROVIDER_MODULES: dict[tuple[str, str], ModuleType] = {
    ("alerts", "nws"): alerts_nws,
    ("forecast", "openmeteo"): forecast_openmeteo,
    ("forecast", "nws"): forecast_nws,
    ("forecast", "aeris"): forecast_aeris,
    ("forecast", "openweathermap"): forecast_openweathermap,
}


def get_provider_module(*, domain: str, provider_id: str) -> ModuleType:
    """Return the provider module by (domain, provider_id).

    Args:
        domain: Provider domain e.g. "alerts", "forecast".
        provider_id: Provider id e.g. "nws", "aeris".

    Returns:
        The provider module (has CAPABILITY symbol and fetch() callable).

    Raises:
        KeyError: Unknown (domain, provider_id) pair.
    """
    key = (domain, provider_id)
    if key not in PROVIDER_MODULES:
        raise KeyError(
            f"Unknown provider: domain={domain!r}, provider_id={provider_id!r}. "
            f"Known providers: {sorted(PROVIDER_MODULES.keys())}"
        )
    return PROVIDER_MODULES[key]
