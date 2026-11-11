"""
Dataset registry — declarative map of every nflreadpy loader to its
DuckDB table, ingestion strategy, and metadata.

Usage:
    from nfl_mcp.registry import REGISTRY, DEFAULT_DATASETS, ALL_DATASETS
"""

from dataclasses import dataclass, field

# ── Names the ingest routes on ────────────────────────────────────────────────
#
# Written out once, here, so the registry and the router in nfl_mcp/ingest.py cannot
# disagree about what a source or a dataset is called. `source` picks the ingest helper;
# the router raises for anything outside KNOWN_SOURCES rather than sending it to nflreadpy.

SOURCE_NFLREADPY = "nflreadpy"
SOURCE_ESPN = "espn"
KNOWN_SOURCES: tuple[str, ...] = (SOURCE_NFLREADPY, SOURCE_ESPN)

# Play-by-play is the one dataset on its own ingest path (schema discovery,
# enhanced_description, null filtering). These spell its id, its target table and its
# loader for that path.
PBP_DATASET_ID = "pbp"
PBP_TABLE_NAME = "plays"
PBP_LOADER_FN = "load_pbp"

# How a dataset's rows land in its table. See DatasetDef.storage: the value is derived from
# `seasonal`, so it is spelled here but never declared per dataset.
STORAGE_APPEND_BY_SEASON = "append_by_season"
STORAGE_REPLACE = "replace"
STORAGE_STRATEGIES: tuple[str, ...] = (STORAGE_APPEND_BY_SEASON, STORAGE_REPLACE)


@dataclass(frozen=True)
class DatasetDef:
    dataset_id: str
    loader_fn: str          # nflreadpy function name
    table_name: str         # DuckDB target table
    seasonal: bool          # accepts seasons arg?
    default: bool           # included in default init (no --dataset flag needed)
    description: str = ""
    extra_params: dict = field(default_factory=dict, compare=False, hash=False)
    # Known coverage window — seasons outside this range are skipped without
    # making a network request.  None means no known bound.  min_season is the
    # authoritative "how far back does this stat go" answer: it is set to the first
    # season whose upstream file actually contains rows, verified against nflverse /
    # DynastyProcess / ffverse, and always matches nflreadpy's own season validator.
    min_season: int | None = None
    max_season: int | None = None
    # Where the rows come from. SOURCE_NFLREADPY is every nflverse/DynastyProcess/ffverse
    # dataset; SOURCE_ESPN is the external records fetch, which is not an nflreadpy loader
    # and is opt-in because it is rate-limited and belongs to another publisher.
    source: str = SOURCE_NFLREADPY

    @property
    def storage(self) -> str:
        """Write strategy: "replace" for a static dataset, "append_by_season" for a seasonal one.

        Derived from `seasonal` instead of declared next to it. `_ingest_generic_dataset`
        picks its write strategy from `seasonal` alone, so a declared storage could say one
        thing while the ingest did another. The two strategies are exactly the two branches
        that function already has: a static dataset is rebuilt whole, a seasonal one is
        deleted and re-inserted one season at a time.
        """
        return STORAGE_APPEND_BY_SEASON if self.seasonal else STORAGE_REPLACE


# ── Registry ───────────────────────────────────────────────────────────────────

_DEFS: list[DatasetDef] = [

    # ── Static / reference (no season filter) ──────────────────────────────
    DatasetDef(
        dataset_id="teams",
        loader_fn="load_teams",
        table_name="teams",
        seasonal=False, default=True,
        description="Team metadata — colors, logos, abbreviations",
    ),
    DatasetDef(
        dataset_id="players",
        loader_fn="load_players",
        table_name="players",
        seasonal=False, default=True,
        description="Player directory — names, draft info, cross-source ID mappings",
    ),
    DatasetDef(
        dataset_id="contracts",
        loader_fn="load_contracts",
        table_name="contracts",
        seasonal=False, default=True,
        description="Historical player contract data",
    ),
    DatasetDef(
        dataset_id="trades",
        loader_fn="load_trades",
        table_name="trades",
        seasonal=False, default=True,
        description="NFL trade history",
    ),
    DatasetDef(
        dataset_id="ff_playerids",
        loader_fn="load_ff_playerids",
        table_name="ff_playerids",
        seasonal=False, default=True,
        description="Fantasy football player ID crosswalk (DynastyProcess.com)",
    ),
    DatasetDef(
        dataset_id="ff_rankings_draft",
        loader_fn="load_ff_rankings",
        table_name="ff_rankings_draft",
        seasonal=False, default=True,
        description="Fantasy football draft rankings/projections",
        extra_params={"type": "draft"},
    ),
    DatasetDef(
        dataset_id="ff_rankings_week",
        loader_fn="load_ff_rankings",
        table_name="ff_rankings_week",
        seasonal=False, default=True,
        description="Fantasy football weekly rankings/projections",
        extra_params={"type": "week"},
    ),
    DatasetDef(
        dataset_id="combine",
        loader_fn="load_combine",
        table_name="combine",
        seasonal=False, default=True,
        description="NFL Combine measurables — all years",
    ),
    DatasetDef(
        dataset_id="draft_picks",
        loader_fn="load_draft_picks",
        table_name="draft_picks",
        seasonal=False, default=True,
        description="NFL draft pick data (1980–current)",
    ),

    # ── Core seasonal ───────────────────────────────────────────────
    DatasetDef(
        dataset_id=PBP_DATASET_ID,
        loader_fn=PBP_LOADER_FN,
        table_name=PBP_TABLE_NAME,
        seasonal=True, default=True,
        description="Play-by-play data — the primary fact table",
        min_season=1999,
    ),
    DatasetDef(
        dataset_id="schedules",
        loader_fn="load_schedules",
        table_name="schedules",
        seasonal=True, default=True,
        description="Game-level schedule and results",
        min_season=1999,   # schedules/games published seasons are 1999+ (verified)
    ),
    DatasetDef(
        dataset_id="rosters",
        loader_fn="load_rosters",
        table_name="rosters",
        seasonal=True, default=True,
        description="Season-level roster snapshots",
        min_season=1920,
    ),
    DatasetDef(
        dataset_id="player_stats",
        loader_fn="load_player_stats",
        table_name="player_stats",
        seasonal=True, default=True,
        description="Pre-aggregated player stats by week/season",
        extra_params={"summary_level": "week"},
        min_season=1999,
    ),
    DatasetDef(
        dataset_id="team_stats_raw",
        loader_fn="load_team_stats",
        table_name="team_stats_raw",
        seasonal=True, default=True,
        description="Pre-aggregated team stats by week/season",
        extra_params={"summary_level": "week"},
        min_season=1999,
    ),
    DatasetDef(
        dataset_id="injuries",
        loader_fn="load_injuries",
        table_name="injuries",
        seasonal=True, default=True,
        description="Weekly injury reports",
        min_season=2009,
    ),
    DatasetDef(
        dataset_id="snap_counts",
        loader_fn="load_snap_counts",
        table_name="snap_counts",
        seasonal=True, default=True,
        description="Player snap counts (Pro Football Reference)",
        min_season=2013,  # the 2012 file exists and downloads fine but holds 0 rows
    ),

    # ── Analytics enrichment ───────────────────────────────────────
    DatasetDef(
        dataset_id="rosters_weekly",
        loader_fn="load_rosters_weekly",
        table_name="rosters_weekly",
        seasonal=True, default=True,
        description="Weekly roster snapshots — large dataset",
        min_season=2002,
    ),
    DatasetDef(
        dataset_id="depth_charts",
        loader_fn="load_depth_charts",
        table_name="depth_charts",
        seasonal=True, default=True,
        description="Weekly depth charts",
        min_season=2001,
    ),
    DatasetDef(
        dataset_id="officials",
        loader_fn="load_officials",
        table_name="officials",
        seasonal=True, default=True,
        description="Game officials assignments",
        min_season=2015,
    ),
    DatasetDef(
        dataset_id="participation",
        loader_fn="load_participation",
        table_name="participation",
        seasonal=True, default=True,
        description="Player participation per play — very large dataset",
        min_season=2016,
        # No upper bound: upstream publishes every season from 2016 onward.
    ),
    DatasetDef(
        dataset_id="nextgen_stats_passing",
        loader_fn="load_nextgen_stats",
        table_name="nextgen_stats_passing",
        seasonal=True, default=True,
        description="Next Gen Stats — passing",
        extra_params={"stat_type": "passing"},
        min_season=2016,
    ),
    DatasetDef(
        dataset_id="nextgen_stats_receiving",
        loader_fn="load_nextgen_stats",
        table_name="nextgen_stats_receiving",
        seasonal=True, default=True,
        description="Next Gen Stats — receiving",
        extra_params={"stat_type": "receiving"},
        min_season=2016,
    ),
    DatasetDef(
        dataset_id="nextgen_stats_rushing",
        loader_fn="load_nextgen_stats",
        table_name="nextgen_stats_rushing",
        seasonal=True, default=True,
        description="Next Gen Stats — rushing",
        extra_params={"stat_type": "rushing"},
        min_season=2016,
    ),
    DatasetDef(
        dataset_id="pfr_advstats_pass",
        loader_fn="load_pfr_advstats",
        table_name="pfr_advstats_pass",
        seasonal=True, default=True,
        description="PFR advanced passing stats",
        extra_params={"stat_type": "pass"},
        min_season=2018,
    ),
    DatasetDef(
        dataset_id="pfr_advstats_rush",
        loader_fn="load_pfr_advstats",
        table_name="pfr_advstats_rush",
        seasonal=True, default=True,
        description="PFR advanced rushing stats",
        extra_params={"stat_type": "rush"},
        min_season=2018,
    ),
    DatasetDef(
        dataset_id="pfr_advstats_rec",
        loader_fn="load_pfr_advstats",
        table_name="pfr_advstats_rec",
        seasonal=True, default=True,
        description="PFR advanced receiving stats",
        extra_params={"stat_type": "rec"},
        min_season=2018,
    ),
    DatasetDef(
        dataset_id="pfr_advstats_def",
        loader_fn="load_pfr_advstats",
        table_name="pfr_advstats_def",
        seasonal=True, default=True,
        description="PFR advanced defensive stats",
        extra_params={"stat_type": "def"},
        min_season=2018,
    ),

    # ── External / fantasy / commercial ────────────────────────────
    DatasetDef(
        dataset_id="ftn_charting",
        loader_fn="load_ftn_charting",
        table_name="ftn_charting",
        seasonal=True, default=True,
        description="FTN advanced charting data",
        min_season=2022,
    ),
    DatasetDef(
        dataset_id="ff_opportunity",
        loader_fn="load_ff_opportunity",
        table_name="ff_opportunity",
        seasonal=True, default=True,
        description="Fantasy football opportunity model",
        extra_params={"stat_type": "weekly"},
        min_season=2006,
    ),

    # ── External publisher (opt-in) ─────────────────────────────────────────
    DatasetDef(
        dataset_id="espn_records",
        loader_fn="espn_records",
        table_name="espn_team_records",
        seasonal=True, default=False,
        description="Final season win-loss records 1970-1998, from ESPN (nflverse has no games before 1999)",
        min_season=1970,
        max_season=1998,
        source=SOURCE_ESPN,
    ),
]

# ── Public accessors ──────────────────────────────────────────────────────────

REGISTRY: dict[str, DatasetDef] = {d.dataset_id: d for d in _DEFS}

DEFAULT_DATASETS: list[str] = [d.dataset_id for d in _DEFS if d.default]

ALL_DATASETS: list[str] = [d.dataset_id for d in _DEFS]


def earliest_season() -> int:
    """Oldest season any registered seasonal dataset can supply.

    This is the floor the default backfill should start at. Keeping it derived from the
    registry means adding an older dataset automatically deepens the backfill; the test
    suite asserts it stays equal to seasons.FIRST_SEASON.
    """
    return min(
        d.min_season for d in _DEFS
        if d.seasonal and d.min_season is not None
    )
