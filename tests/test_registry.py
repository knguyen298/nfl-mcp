"""Tests for the dataset registry — structure, completeness, and loader validity."""

import dataclasses

import nflreadpy
import pytest

from nfl_mcp.registry import (
    ALL_DATASETS,
    DEFAULT_DATASETS,
    KNOWN_SOURCES,
    PBP_DATASET_ID,
    PBP_LOADER_FN,
    PBP_TABLE_NAME,
    REGISTRY,
    SOURCE_ESPN,
    SOURCE_NFLREADPY,
    STORAGE_APPEND_BY_SEASON,
    STORAGE_REPLACE,
    STORAGE_STRATEGIES,
    DatasetDef,
    earliest_season,
)


# ── Structure ──────────────────────────────────────────────────────────────────

class TestRegistryStructure:
    def test_registry_is_not_empty(self):
        assert len(REGISTRY) > 0

    def test_all_datasets_matches_registry_keys(self):
        assert set(ALL_DATASETS) == set(REGISTRY.keys())

    def test_no_duplicate_dataset_ids(self):
        assert len(ALL_DATASETS) == len(set(ALL_DATASETS))

    def test_no_duplicate_table_names(self):
        table_names = [d.table_name for d in REGISTRY.values()]
        assert len(table_names) == len(set(table_names)), \
            "Multiple datasets share the same table_name"

    def test_default_datasets_is_subset_of_all(self):
        assert set(DEFAULT_DATASETS) <= set(ALL_DATASETS)

    def test_default_datasets_not_empty(self):
        assert len(DEFAULT_DATASETS) > 0

    def test_all_entries_are_datasetdef(self):
        for d in REGISTRY.values():
            assert isinstance(d, DatasetDef)

    def test_storage_values_are_valid(self):
        for d in REGISTRY.values():
            assert d.storage in STORAGE_STRATEGIES, \
                f"{d.dataset_id} has invalid storage={d.storage!r}"

    def test_storage_agrees_with_seasonal(self):
        """The strategy the registry names is the one the ingest performs.

        _ingest_generic_dataset branches on `seasonal` alone: a seasonal dataset is deleted
        and re-inserted one season at a time, a static one is rebuilt whole. storage reads
        off the same flag, so the two cannot disagree.
        """
        for d in REGISTRY.values():
            expected = STORAGE_APPEND_BY_SEASON if d.seasonal else STORAGE_REPLACE
            assert d.storage == expected, (
                f"{d.dataset_id} is seasonal={d.seasonal} but reports storage={d.storage!r}"
            )

    def test_storage_cannot_be_declared_and_thus_cannot_disagree(self):
        # storage is derived, so there is no keyword left that would state a strategy the
        # ingest does not perform. Passing one is a mistake caught at import time.
        assert "storage" not in {f.name for f in dataclasses.fields(DatasetDef)}
        with pytest.raises(TypeError):
            DatasetDef(dataset_id="x", loader_fn="load_x", table_name="x",
                       seasonal=False, default=False, storage="append_by_season")

    def test_derived_storage_matches_the_two_branches(self):
        static = DatasetDef(dataset_id="x", loader_fn="load_x", table_name="x",
                            seasonal=False, default=False)
        seasonal = DatasetDef(dataset_id="y", loader_fn="load_y", table_name="y",
                              seasonal=True, default=False)
        assert static.storage == STORAGE_REPLACE
        assert seasonal.storage == STORAGE_APPEND_BY_SEASON

    def test_every_dataset_declares_a_source_the_router_knows(self):
        """The router in nfl_mcp/ingest.py raises for a source outside KNOWN_SOURCES.

        A registered dataset with such a source would stop a whole run, so the registry must
        only ever name one that has an ingest helper.
        """
        unknown = [f"{d.dataset_id}: {d.source!r}" for d in REGISTRY.values()
                   if d.source not in KNOWN_SOURCES]
        assert unknown == []

    def test_every_known_source_is_used(self):
        # A source nobody uses is a constant the router still branches on for nothing.
        assert {d.source for d in REGISTRY.values()} == set(KNOWN_SOURCES)

    def test_the_pbp_path_names_the_same_dataset_the_registry_declares(self):
        # nfl_mcp/ingest.py picks its special path by PBP_DATASET_ID. Rename the entry and
        # that path stops matching, so play-by-play goes down the generic loader path
        # instead and lands with no enhanced_description column.
        assert REGISTRY[PBP_DATASET_ID].dataset_id == PBP_DATASET_ID
        assert REGISTRY[PBP_DATASET_ID].loader_fn == PBP_LOADER_FN
        assert REGISTRY[PBP_DATASET_ID].table_name == PBP_TABLE_NAME

    def test_descriptions_are_non_empty(self):
        for d in REGISTRY.values():
            assert d.description, f"{d.dataset_id} has empty description"


# ── Season coverage windows ────────────────────────────────────────────────────

class TestSeasonCoverageWindows:
    def test_min_max_ordering(self):
        for d in REGISTRY.values():
            if d.min_season and d.max_season:
                assert d.min_season <= d.max_season, \
                    f"{d.dataset_id}: min_season {d.min_season} > max_season {d.max_season}"

    def test_non_seasonal_datasets_have_no_coverage_window(self):
        for d in REGISTRY.values():
            if not d.seasonal:
                assert d.min_season is None, \
                    f"{d.dataset_id} is non-seasonal but has min_season"
                assert d.max_season is None, \
                    f"{d.dataset_id} is non-seasonal but has max_season"

    def test_season_values_are_plausible(self):
        for d in REGISTRY.values():
            if d.min_season:
                assert 1900 <= d.min_season <= 2030, \
                    f"{d.dataset_id} min_season={d.min_season} is implausible"
            if d.max_season:
                assert 1900 <= d.max_season <= 2030, \
                    f"{d.dataset_id} max_season={d.max_season} is implausible"

    @pytest.mark.parametrize("dataset_id,expected_min", [
        ("officials",             2015),
        ("participation",         2016),
        ("nextgen_stats_passing", 2016),
        ("pfr_advstats_pass",     2018),
        ("ftn_charting",          2022),
        ("snap_counts",           2013),
        ("injuries",              2009),
        ("schedules",             1999),
        ("pbp",                   1999),
        ("rosters",               1920),
    ])
    def test_known_min_seasons(self, dataset_id, expected_min):
        assert REGISTRY[dataset_id].min_season == expected_min

    def test_participation_is_not_capped_at_the_top(self):
        # Upstream publishes participation for every season from 2016, so an upper
        # bound here would silently stop ingesting new seasons.
        assert REGISTRY["participation"].max_season is None

    def test_every_seasonal_dataset_declares_a_min_season(self):
        # Depth comes from min_season: a seasonal dataset without one would fall back
        # to the global floor and burn network requests on seasons that never existed.
        missing = [d.dataset_id for d in REGISTRY.values()
                   if d.seasonal and d.min_season is None]
        assert missing == []

    def test_hard_floor_matches_the_oldest_dataset(self):
        # EARLIEST_SEASON is what --start may reach; it must track the oldest registered
        # dataset, otherwise deep seasons are silently unreachable.
        from nfl_mcp.seasons import EARLIEST_SEASON
        assert EARLIEST_SEASON == earliest_season()

    def test_default_start_is_reachable_but_conservative(self):
        # FIRST_SEASON is the default backfill start (1966, the merger agreement year). It
        # must sit inside the range --start accepts, and must not be deeper than the
        # sources allow.
        from nfl_mcp.seasons import EARLIEST_SEASON, FIRST_SEASON
        assert EARLIEST_SEASON <= FIRST_SEASON
        assert FIRST_SEASON == 1966

    def test_default_start_only_clips_the_oldest_dataset(self):
        # Only rosters predate the 1966 default, so the default costs exactly one
        # dataset's history. If this starts failing, a newly registered dataset is being
        # silently truncated by the default and its real floor should be documented.
        from nfl_mcp.seasons import FIRST_SEASON
        clipped = [d.dataset_id for d in REGISTRY.values()
                   if d.seasonal and d.min_season is not None and d.min_season < FIRST_SEASON]
        assert clipped == ["rosters"]

    def test_earliest_season_is_the_oldest_declared_floor(self):
        assert earliest_season() == min(
            d.min_season for d in REGISTRY.values()
            if d.seasonal and d.min_season is not None
        )


# ── Loader functions ───────────────────────────────────────────────────────────

class TestLoaderFunctions:
    def test_all_loader_fns_exist_in_nflreadpy(self):
        missing = [
            d.dataset_id
            for d in REGISTRY.values()
            if d.source == SOURCE_NFLREADPY and not hasattr(nflreadpy, d.loader_fn)
        ]
        assert not missing, f"loader_fn not found in nflreadpy: {missing}"

    def test_extra_params_are_dicts(self):
        for d in REGISTRY.values():
            assert isinstance(d.extra_params, dict), \
                f"{d.dataset_id} extra_params is not a dict"

    def test_pbp_loader_is_load_pbp(self):
        assert REGISTRY["pbp"].loader_fn == "load_pbp"
        assert REGISTRY["pbp"].table_name == "plays"

    def test_non_seasonal_datasets_have_no_seasons_param(self):
        # Spot-check: these are known static datasets
        for ds_id in ("teams", "players", "contracts", "trades", "combine"):
            assert not REGISTRY[ds_id].seasonal, \
                f"{ds_id} should be non-seasonal"


# ── Default dataset completeness ───────────────────────────────────────────────

class TestDefaultDatasets:
    def test_every_nflreadpy_dataset_is_default(self):
        """Full nflverse import is baked in; only the external pull is opt-in."""
        non_default = [
            d.dataset_id for d in REGISTRY.values()
            if not d.default and d.source == SOURCE_NFLREADPY
        ]
        assert non_default == [], \
            f"these nflreadpy datasets are not default but should be: {non_default}"

    def test_external_sources_are_opt_in(self):
        # A dataset that reaches another publisher at ingest time must never ride along on
        # the container's scheduled refresh, so it stays out of 'default'.
        for d in REGISTRY.values():
            if d.source != SOURCE_NFLREADPY:
                assert d.default is False, f"{d.dataset_id} is external but default"

    def test_only_the_espn_records_pull_is_opt_in(self):
        assert {d.dataset_id for d in REGISTRY.values() if not d.default} == {"espn_records"}

    def test_default_datasets_equals_all_except_opt_in(self):
        assert set(DEFAULT_DATASETS) == set(ALL_DATASETS) - {"espn_records"}


class TestExternalRecordsSource:
    """The pre-1999 records pull is the one dataset that is not nflreadpy."""

    def setup_method(self):
        self.defn = REGISTRY["espn_records"]

    def test_declared_as_external(self):
        assert self.defn.source == SOURCE_ESPN
        assert self.defn.table_name == "espn_team_records"
        assert self.defn.loader_fn not in dir(nflreadpy), "records loader is not an nflreadpy function"

    def test_window_matches_the_franchise_registry(self):
        from nfl_mcp.franchises import RECORDS_FIRST_SEASON, RECORDS_LAST_ESPN_SEASON
        assert self.defn.min_season == RECORDS_FIRST_SEASON
        assert self.defn.max_season == RECORDS_LAST_ESPN_SEASON

    def test_window_abuts_the_nflverse_schedule_floor(self):
        # The two sources must meet with no gap and no overlap: ESPN stops where nflverse starts.
        assert self.defn.max_season + 1 == REGISTRY["schedules"].min_season

    def test_depth_helpers_are_unaffected(self):
        assert earliest_season() == 1920  # still rosters, the oldest nflreadpy asset
