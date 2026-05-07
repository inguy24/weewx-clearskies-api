"""Unit tests for the aggregation service's per-canonical-field aggregator defaults.

Asserts that services/archive.py exposes a DAY_AGGREGATOR constant with the
expected defaults per the brief:
  outTemp  → avg (daily mean temperature)
  rain     → sum
  windGust → max

Also asserts that DAY_AGGREGATOR and _FIRST_CLASS_FIELDS stay in sync: every
_FIRST_CLASS_FIELDS entry has an aggregator, no orphan aggregator keys exist.

daySunshineDur note: weewx writes this as a running cumulative (resets at midnight).
The correct per-day aggregate is max (end-of-day value = daily total), NOT sum
(which would double-count). api-dev commit 8b7649d flipped sum→max; this file
asserts max.

ADR references: brief §2 interval=day spec, brief §test-author-parallel-scope.
"""

from __future__ import annotations

import pytest


class TestDayAggregatorConstant:
    """DAY_AGGREGATOR is published in services/archive.py and contains expected defaults."""

    def test_day_aggregator_is_importable(self) -> None:
        """DAY_AGGREGATOR can be imported from services/archive."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        assert DAY_AGGREGATOR is not None
        assert isinstance(DAY_AGGREGATOR, dict)

    def test_rain_aggregator_is_sum(self) -> None:
        """rain → sum (accumulated rainfall over the day)."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        assert "rain" in DAY_AGGREGATOR, (
            "DAY_AGGREGATOR must contain an entry for 'rain'"
        )
        assert DAY_AGGREGATOR["rain"] == "sum", (
            f"rain aggregator must be 'sum', got {DAY_AGGREGATOR['rain']!r}"
        )

    def test_wind_gust_aggregator_is_max(self) -> None:
        """windGust → max (peak gust for the day is the meaningful metric)."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        assert "windGust" in DAY_AGGREGATOR
        assert DAY_AGGREGATOR["windGust"] == "max", (
            f"windGust aggregator must be 'max', got {DAY_AGGREGATOR['windGust']!r}"
        )

    def test_out_temp_aggregator_is_avg(self) -> None:
        """outTemp → avg (daily mean temperature from archive_day_outTemp)."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        assert "outTemp" in DAY_AGGREGATOR
        assert DAY_AGGREGATOR["outTemp"] == "avg", (
            f"outTemp aggregator must be 'avg', got {DAY_AGGREGATOR['outTemp']!r}"
        )

    def test_rain_rate_aggregator_is_max(self) -> None:
        """rainRate → max (peak rate within the day)."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        assert "rainRate" in DAY_AGGREGATOR
        assert DAY_AGGREGATOR["rainRate"] == "max", (
            f"rainRate aggregator must be 'max', got {DAY_AGGREGATOR['rainRate']!r}"
        )

    def test_out_humidity_aggregator_is_avg(self) -> None:
        """outHumidity → avg (average humidity over the day)."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        assert "outHumidity" in DAY_AGGREGATOR
        assert DAY_AGGREGATOR["outHumidity"] == "avg", (
            f"outHumidity aggregator must be 'avg', got {DAY_AGGREGATOR['outHumidity']!r}"
        )

    def test_barometer_aggregator_is_avg(self) -> None:
        """barometer → avg (average pressure over the day)."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        assert "barometer" in DAY_AGGREGATOR
        assert DAY_AGGREGATOR["barometer"] == "avg", (
            f"barometer aggregator must be 'avg', got {DAY_AGGREGATOR['barometer']!r}"
        )

    def test_wind_speed_aggregator_is_max(self) -> None:
        """windSpeed → max (peak recorded wind speed; archive_day_windSpeed.max)."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        assert "windSpeed" in DAY_AGGREGATOR
        assert DAY_AGGREGATOR["windSpeed"] == "max"

    def test_radiation_aggregator_is_max(self) -> None:
        """radiation → max (peak solar irradiance in the day)."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        assert "radiation" in DAY_AGGREGATOR
        assert DAY_AGGREGATOR["radiation"] == "max"

    def test_uv_aggregator_is_max(self) -> None:
        """UV → max (peak UV index for the day)."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        assert "UV" in DAY_AGGREGATOR
        assert DAY_AGGREGATOR["UV"] == "max"

    def test_day_sunshine_dur_aggregator_is_max_not_sum(self) -> None:
        """daySunshineDur → max, NOT sum.

        daySunshineDur is a running cumulative that resets at midnight.
        The end-of-day value IS the daily total; summing across archive intervals
        double-counts. max is the correct semantic.

        This assertion pins the lead-confirmed flip in api-dev commit 8b7649d.
        """
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        assert "daySunshineDur" in DAY_AGGREGATOR
        assert DAY_AGGREGATOR["daySunshineDur"] == "max", (
            f"daySunshineDur must be 'max' (running cumulative — not 'sum'), "
            f"got {DAY_AGGREGATOR['daySunshineDur']!r}"
        )

    def test_sunshine_dur_aggregator_is_sum(self) -> None:
        """sunshineDur (per-interval sunshine) → sum (accumulation within the day)."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        assert "sunshineDur" in DAY_AGGREGATOR
        assert DAY_AGGREGATOR["sunshineDur"] == "sum", (
            f"sunshineDur (per-interval duration) must be 'sum', "
            f"got {DAY_AGGREGATOR['sunshineDur']!r}"
        )

    def test_et_aggregator_is_sum(self) -> None:
        """ET (evapotranspiration) → sum (accumulation over the day)."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        assert "ET" in DAY_AGGREGATOR
        assert DAY_AGGREGATOR["ET"] == "sum"

    def test_degree_day_fields_aggregator_is_sum(self) -> None:
        """heatdeg + cooldeg → sum (degree-days accumulate over the day)."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        assert "heatdeg" in DAY_AGGREGATOR
        assert DAY_AGGREGATOR["heatdeg"] == "sum", (
            f"heatdeg must be 'sum', got {DAY_AGGREGATOR['heatdeg']!r}"
        )
        assert "cooldeg" in DAY_AGGREGATOR
        assert DAY_AGGREGATOR["cooldeg"] == "sum", (
            f"cooldeg must be 'sum', got {DAY_AGGREGATOR['cooldeg']!r}"
        )

    def test_lightning_count_aggregator_is_sum(self) -> None:
        """lightning_strike_count + noise/disturber counts → sum (count accumulation)."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        for field in (
            "lightning_strike_count",
            "lightning_noise_count",
            "lightning_disturber_count",
        ):
            assert field in DAY_AGGREGATOR, f"{field!r} missing from DAY_AGGREGATOR"
            assert DAY_AGGREGATOR[field] == "sum", (
                f"{field} aggregator must be 'sum', got {DAY_AGGREGATOR[field]!r}"
            )

    def test_all_full_observation_fields_have_aggregator(self) -> None:
        """Every field in the full 69-field Observation surface has a DAY_AGGREGATOR entry.

        Derived from _FIRST_CLASS_FIELDS (which is auto-derived from STOCK_COLUMN_MAP
        minus meta) so this test stays in sync with schema changes automatically.
        """
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR, _FIRST_CLASS_FIELDS

        # Fields that are intentionally absent from DAY_AGGREGATOR (no archive_day_* table):
        # Direction fields (windDir, windGustDir, vecdir, gustdir) have archive_day tables
        # but may not be included if the api-dev omits them — the test is lenient here.
        # The following fields do NOT have archive_day_* tables in a stock weewx install
        # and are intentionally excluded from DAY_AGGREGATOR:
        explicitly_excluded: set[str] = set()
        # This exclusion set is a safety valve for fields without archive_day_* tables.
        # Currently empty: all first-class fields either have aggregators in DAY_AGGREGATOR
        # or the test expectation is that they should be added.

        missing = _FIRST_CLASS_FIELDS - set(DAY_AGGREGATOR.keys()) - explicitly_excluded
        assert not missing, (
            f"_FIRST_CLASS_FIELDS entries not in DAY_AGGREGATOR: {sorted(missing)}. "
            "Every first-class Observation field needs a defined per-day aggregator. "
            "Add missing entries to DAY_AGGREGATOR or add to the explicitly_excluded set "
            "with a documented reason."
        )

    def test_no_orphan_aggregator_entries_not_in_first_class_fields(self) -> None:
        """Every DAY_AGGREGATOR key is a member of _FIRST_CLASS_FIELDS.

        Catches entries that were added to DAY_AGGREGATOR but whose canonical field
        name was removed from or never added to _FIRST_CLASS_FIELDS (i.e., the name
        is wrong, or the field was deleted from the stock column map).
        """
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR, _FIRST_CLASS_FIELDS

        orphans = set(DAY_AGGREGATOR.keys()) - _FIRST_CLASS_FIELDS
        assert not orphans, (
            f"DAY_AGGREGATOR has entries not in _FIRST_CLASS_FIELDS: {sorted(orphans)}. "
            "These are orphan aggregator entries — their canonical names do not exist in "
            "the stock column map. Remove them or fix the canonical name."
        )

    def test_all_aggregator_values_are_recognized_strings(self) -> None:
        """Every value in DAY_AGGREGATOR is a recognized aggregator string."""
        from weewx_clearskies_api.services.archive import DAY_AGGREGATOR

        valid_values = {"avg", "max", "min", "sum"}
        for field, aggregator in DAY_AGGREGATOR.items():
            assert aggregator in valid_values, (
                f"DAY_AGGREGATOR[{field!r}] = {aggregator!r} is not a recognized "
                f"aggregator. Valid values: {valid_values}"
            )
