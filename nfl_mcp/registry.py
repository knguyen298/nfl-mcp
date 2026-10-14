"""
Dataset registry — declarative map of every nflreadpy loader to its
DuckDB table, ingestion strategy, and metadata.

Usage:
    from nfl_mcp.registry import REGISTRY, DEFAULT_DATASETS, ALL_DATASETS
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetDef:
    dataset_id: str
    loader_fn: str          # nflreadpy function name
    table_name: str         # DuckDB target table
    seasonal: bool          # accepts seasons arg?
    default: bool           # included in default init (no --dataset flag needed)
    description: str = ""
    extra_params: dict = field(default_factory=dict, compare=False, hash=False)
    storage: str = "append_by_season"   # "replace" | "append_by_season"
    # Known coverage window — seasons outside this range are skipped without
    # making a network request.  None means no known bound.  min_season is the
    # authoritative "how far back does this stat go" answer: it is set to the first
    # season whose upstream file actually contains rows, verified against nflverse /
    # DynastyProcess / ffverse, and always matches nflreadpy's own season validator.
    min_season: int | None = None
    max_season: int | None = None
    # Where the rows come from. "nflreadpy" is every nflverse/DynastyProcess/ffverse
    # dataset; "espn" is the external records fetch, which is not an nflreadpy loader and
    # is opt-in because it is rate-limited and belongs to another publisher.
    source: str = "nflreadpy"


# ── Registry ───────────────────────────────────────────────────────────────────

_DEFS: list[DatasetDef] = [

    # ── Static / reference (no season filter) ──────────────────────────────
    DatasetDef(
        dataset_id="teams",
        loader_fn="load_teams",
        table_name="teams",
        seasonal=False, default=True, storage="replace",
        description="Team metadata — colors, logos, abbreviations",
    ),
    DatasetDef(
        dataset_id="players",
        loader_fn="load_players",
        table_name="players",
        seasonal=False, default=True, storage="replace",
        description="Player directory — names, draft info, cross-source ID mappings",
    ),
    DatasetDef(
        dataset_id="contracts",
        loader_fn="load_contracts",
        table_name="contracts",
        seasonal=False, default=True, storage="replace",
        description="Historical player contract data",
    ),
    DatasetDef(
        dataset_id="trades",
        loader_fn="load_trades",
        table_name="trades",
        seasonal=False, default=True, storage="replace",
        description="NFL trade history",
    ),
    DatasetDef(
        dataset_id="ff_playerids",
        loader_fn="load_ff_playerids",
        table_name="ff_playerids",
        seasonal=False, default=True, storage="replace",
        description="Fantasy football player ID crosswalk (DynastyProcess.com)",
    ),
    DatasetDef(
        dataset_id="ff_rankings_draft",
        loader_fn="load_ff_rankings",
        table_name="ff_rankings_draft",
        seasonal=False, default=True, storage="replace",
        description="Fantasy football draft rankings/projections",
        extra_params={"type": "draft"},
    ),
    DatasetDef(
        dataset_id="ff_rankings_week",
        loader_fn="load_ff_rankings",
        table_name="ff_rankings_week",
        seasonal=False, default=True, storage="replace",
        description="Fantasy football weekly rankings/projections",
        extra_params={"type": "week"},
    ),
    DatasetDef(
        dataset_id="combine",
        loader_fn="load_combine",
        table_name="combine",
        seasonal=False, default=True, storage="replace",
        description="NFL Combine measurables — all years",
    ),
    DatasetDef(
        dataset_id="draft_picks",
        loader_fn="load_draft_picks",
        table_name="draft_picks",
        seasonal=False, default=True, storage="replace",
        description="NFL draft pick data (1980–current)",
    ),

    # ── Core seasonal ───────────────────────────────────────────────
    DatasetDef(
        dataset_id="pbp",
        loader_fn="load_pbp",
        table_name="plays",
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
        source="espn",
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
