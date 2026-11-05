"""What each table holds and how tables join: the facts behind nfl_catalog, nfl_schema and the
nfl:// resources. Pure data plus two small helpers. A relationship is listed only after its
match rate was measured on a loaded database (see RELATIONSHIPS).
"""

from __future__ import annotations

from .registry import REGISTRY

# Tables rebuilt from loaded data at ingest time (see ingest.rebuild_derived_tables).
DERIVED_TABLES: tuple[str, ...] = (
    "team_offense_stats", "team_defense_stats", "situational_stats", "formation_effectiveness",
    "team_records", "player_td_luck", "player_role_trend", "player_separation_opportunity",
    "player_drop_rate", "player_contract_value", "injury_return_curve",
)

# table -> (grain, join keys). The keys are the columns worth joining on, not a primary key.
TABLE_INFO: dict[str, tuple[str, list[str]]] = {
    "plays": ("one row per play (nflverse play-by-play, 1999 on)", ["game_id", "play_id"]),
    "schedules": ("one row per game", ["game_id"]),
    "players": ("one row per player", ["gsis_id", "pfr_id"]),
    "teams": ("one row per team abbreviation", ["team_abbr"]),
    "rosters": ("one row per player per team per season", ["gsis_id", "pfr_id"]),
    "rosters_weekly": ("one row per player per team per week", ["gsis_id"]),
    "injuries": ("one row per player per team per week (the weekly injury report)", ["gsis_id"]),
    "snap_counts": ("one row per player per game (Pro Football Reference)", ["game_id", "pfr_player_id"]),
    "depth_charts": ("one row per depth chart entry: team, position slot and week or snapshot date",
                     ["gsis_id"]),
    "officials": ("one row per official per game", ["game_id"]),
    "participation": ("one row per play (who was on the field)", ["nflverse_game_id", "play_id"]),
    "player_stats": ("one row per player per week per season type", ["player_id"]),
    "team_stats_raw": ("one row per team per game", ["game_id", "team"]),
    "nextgen_stats_passing": ("one row per passer per week (week 0 is the season total)",
                              ["player_gsis_id"]),
    "nextgen_stats_receiving": ("one row per receiver per week (week 0 is the season total)",
                                ["player_gsis_id"]),
    "nextgen_stats_rushing": ("one row per rusher per week (week 0 is the season total)",
                              ["player_gsis_id"]),
    "pfr_advstats_pass": ("one row per passer per game", ["game_id", "pfr_player_id"]),
    "pfr_advstats_rush": ("one row per rusher per game", ["game_id", "pfr_player_id"]),
    "pfr_advstats_rec": ("one row per receiver per game", ["game_id", "pfr_player_id"]),
    "pfr_advstats_def": ("one row per defender per game", ["game_id", "pfr_player_id"]),
    "ftn_charting": ("one row per play (FTN manual charting, 2022 on)",
                     ["nflverse_game_id", "nflverse_play_id"]),
    "ff_opportunity": ("one row per player per game (fantasy opportunity model)",
                       ["game_id", "player_id"]),
    "ff_playerids": ("one row per player, with that player's id in each fantasy and stats source",
                     ["gsis_id", "pfr_id", "fantasypros_id", "mfl_id"]),
    "ff_rankings_draft": ("one row per player per ranking set, latest scrape only", ["id"]),
    "ff_rankings_week": ("one row per player per ranking page, latest scrape only", ["fantasypros_id"]),
    "contracts": ("one row per player contract (a player can have several)", ["gsis_id"]),
    "trades": ("one row per player or pick moved in a trade", ["trade_id", "pfr_id"]),
    "combine": ("one row per player per combine", ["pfr_id"]),
    "draft_picks": ("one row per draft pick", ["gsis_id", "pfr_player_id"]),
    "espn_team_records": ("one row per team per season, 1970-1998 (ESPN, opt-in)", ["team"]),
    "team_offense_stats": ("one row per team per season per season type (REG or POST)",
                           ["team", "season_year"]),
    "team_defense_stats": ("one row per team per season per season type (REG or POST)",
                           ["team", "season_year"]),
    "situational_stats": ("one row per team per season per season type per situation",
                          ["team", "season_year"]),
    "formation_effectiveness": ("one row per team per season per season type per formation and play type",
                                ["team", "season_year"]),
    "team_records": ("one row per franchise per season: the final regular-season record, 1970 on",
                     ["team", "season"]),
    "player_td_luck": ("one row per player per season per season type (REG or POST)",
                       ["player_id"]),
    "player_role_trend": ("one row per player per week", ["player_id"]),
    "player_separation_opportunity": ("one row per player per season", ["player_id"]),
    "player_drop_rate": ("one row per player per team per season per season type (REG or POST)",
                         ["player_id"]),
    "player_contract_value": ("one row per player per season", ["player_id"]),
    "injury_return_curve": ("one row per injury type, position and week after return",
                            ["injury_type", "position"]),
    "_schema_version": ("one row per schema version stamp (the database version is the highest)",
                        ["version"]),
}

_UNKNOWN_GRAIN = "not described yet; see nfl_schema(table=...) for the columns"


def table_info(table: str) -> tuple[str, list[str]]:
    """(grain, join_keys) for a table. A table this list does not know gets a plain fallback."""
    grain, keys = TABLE_INFO.get(table, (_UNKNOWN_GRAIN, []))
    return grain, list(keys)


# Each pair of key lists is positional. The match rate is the share of non-null left rows
# that found a right row, measured on a loaded 2024-2026 database.
RELATIONSHIPS: list[dict] = [
    {"left_table": "participation", "left_keys": ["nflverse_game_id", "play_id"],
     "right_table": "plays", "right_keys": ["game_id", "play_id"], "match": "100%"},
    {"left_table": "ftn_charting", "left_keys": ["nflverse_game_id", "nflverse_play_id"],
     "right_table": "plays", "right_keys": ["game_id", "play_id"], "match": "95%"},
    {"left_table": "plays", "left_keys": ["game_id"],
     "right_table": "schedules", "right_keys": ["game_id"], "match": "100%"},
    {"left_table": "snap_counts", "left_keys": ["game_id"],
     "right_table": "schedules", "right_keys": ["game_id"], "match": "100%"},
    {"left_table": "snap_counts", "left_keys": ["pfr_player_id"],
     "right_table": "players", "right_keys": ["pfr_id"], "match": "99.8%"},
    {"left_table": "rosters", "left_keys": ["gsis_id"],
     "right_table": "players", "right_keys": ["gsis_id"], "match": "94%"},
    {"left_table": "plays", "left_keys": ["passer_player_id"],
     "right_table": "players", "right_keys": ["gsis_id"], "match": "100%"},
    {"left_table": "plays", "left_keys": ["rusher_player_id"],
     "right_table": "players", "right_keys": ["gsis_id"], "match": "100%"},
    {"left_table": "plays", "left_keys": ["receiver_player_id"],
     "right_table": "players", "right_keys": ["gsis_id"], "match": "100%"},
    {"left_table": "injuries", "left_keys": ["gsis_id"],
     "right_table": "players", "right_keys": ["gsis_id"], "match": "100%"},
    {"left_table": "depth_charts", "left_keys": ["gsis_id"],
     "right_table": "players", "right_keys": ["gsis_id"], "match": "100%"},
    {"left_table": "ff_opportunity", "left_keys": ["player_id"],
     "right_table": "players", "right_keys": ["gsis_id"], "match": "100%"},
    {"left_table": "player_stats", "left_keys": ["player_id"],
     "right_table": "players", "right_keys": ["gsis_id"], "match": "100%"},
    {"left_table": "nextgen_stats_passing", "left_keys": ["player_gsis_id"],
     "right_table": "players", "right_keys": ["gsis_id"], "match": "100%"},
    {"left_table": "draft_picks", "left_keys": ["gsis_id"],
     "right_table": "players", "right_keys": ["gsis_id"], "match": "100%"},
    {"left_table": "pfr_advstats_rec", "left_keys": ["pfr_player_id"],
     "right_table": "players", "right_keys": ["pfr_id"], "match": "99.9%"},
    {"left_table": "player_td_luck", "left_keys": ["player_id"],
     "right_table": "players", "right_keys": ["gsis_id"], "match": "100%"},
    {"left_table": "player_role_trend", "left_keys": ["player_id"],
     "right_table": "players", "right_keys": ["gsis_id"], "match": "100%"},
    {"left_table": "team_records", "left_keys": ["team"],
     "right_table": "teams", "right_keys": ["team_abbr"], "match": "100%"},
]


def relationships_for(tables: set[str]) -> list[dict]:
    """The relationships whose two tables are both loaded."""
    return [dict(r) for r in RELATIONSHIPS if r["left_table"] in tables and r["right_table"] in tables]


def _start_seasons() -> str:
    """One sentence of dataset start seasons, taken from the registry so it cannot drift."""
    starts: dict[int, list[str]] = {}
    for d in REGISTRY.values():
        if d.seasonal and d.min_season is not None and d.source == "nflreadpy":
            starts.setdefault(d.min_season, []).append(d.table_name)
    return "; ".join(f"{year}: {', '.join(names)}" for year, names in sorted(starts.items()))


DATA_NOTES: list[str] = [
    ("FTN charting (ftn_charting, and the nfl_ftn_charting and nfl_drop_rate tools) is manual "
    "charting by FTN staff. It is not tracking data. It starts in 2022."),
    ("Next Gen Stats (nextgen_stats_passing, nextgen_stats_receiving, nextgen_stats_rushing) "
    "start in 2016. They are the NFL's tracking-data stats. The separation numbers in "
    "nfl_separation_opportunity come from them."),
    ("plays covers 1999 to the current season. Earlier seasons have no play-by-play. "
    "team_records goes back to 1970 through the optional 'nfl-mcp ingest --pre1999' pull."),
    ("First season of each dataset (nothing loads before it), by start year: "
    + _start_seasons() + "."),
    ("season_type is REG or POST in plays and in the derived team and player tables. "
    "Playoff weeks continue the regular-season numbering."),
    ("The derived tables (see derived_tables) are rebuilt from the loaded data at ingest time. "
    "team_offense_stats, team_defense_stats, situational_stats and formation_effectiveness "
    "have separate REG and POST rows, so filter season_type before you sum."),
    ("Two player ids appear. gsis_id is the NFL id and most tables use it (player_id in "
    "ff_opportunity and the derived player tables is a gsis_id). pfr_id is the Pro Football "
    "Reference id (snap_counts and the pfr_advstats tables). players maps one to the other."),
    ("Team abbreviations differ by table. team_records uses the franchise's current "
    "abbreviation (the Raiders are LV in every season). plays.posteam uses the abbreviation "
    "of that season (OAK before the move)."),
    "ff_rankings_draft and ff_rankings_week hold the latest scrape only. They are not historical.",
    ("Every tool response carries data_as_of: the newest load time (UTC) of the tables that "
    "tool read. nfl_status gives the newest season and week and the load time per dataset."),
]
