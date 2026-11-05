"""Every table and column the tools read, in one place.

`nfl-mcp validate` checks a freshly ingested slot against this list before the swap, and
an integration test checks it against the loaded database. Derived from the SQL in
tools.py: add a column here whenever a tool starts reading one.
"""

REQUIRED_COLUMNS: dict[str, list[str]] = {
    # nfl_status, nfl_search_plays, nfl_player_stats, nfl_compare, nfl_ftn_charting
    "plays": [
        "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam",
        "down", "ydstogo", "yardline_100", "qtr", "half_seconds_remaining", "play_type",
        "yards_gained", "epa", "desc", "touchdown", "interception", "fumble_lost",
        "passer_player_name", "rusher_player_name", "receiver_player_name",
        "complete_pass", "pass_touchdown", "rush_touchdown", "passing_yards",
        "rushing_yards", "receiving_yards", "yards_after_catch", "air_yards", "cpoe",
        # nfl_search_plays: filters, sort keys and the role id columns of player_role
        "shotgun", "no_huddle", "goal_to_go", "score_differential", "wp", "wpa",
        "passer_player_id", "rusher_player_id", "receiver_player_id", "kicker_player_id",
        "punter_player_id", "kickoff_returner_player_id", "punt_returner_player_id",
        "sack_player_id", "half_sack_1_player_id", "half_sack_2_player_id",
        "solo_tackle_1_player_id", "solo_tackle_2_player_id", "assist_tackle_1_player_id",
        "assist_tackle_2_player_id", "assist_tackle_3_player_id", "assist_tackle_4_player_id",
        "tackle_with_assist_1_player_id", "tackle_with_assist_2_player_id",
        "interception_player_id", "pass_defense_1_player_id", "pass_defense_2_player_id",
        "forced_fumble_player_1_player_id", "forced_fumble_player_2_player_id",
        "qb_hit_1_player_id", "qb_hit_2_player_id", "tackle_for_loss_1_player_id",
        "tackle_for_loss_2_player_id",
    ],
    # nfl_schedule
    "schedules": [
        "game_id", "season", "week", "game_type", "gameday", "gametime", "away_team",
        "home_team", "away_score", "home_score", "result", "overtime", "spread_line",
        "total_line", "div_game", "roof", "surface", "temp", "wind", "away_coach",
        "home_coach", "referee", "stadium",
    ],
    # nfl_roster
    "rosters": [
        "full_name", "position", "depth_chart_position", "team", "season",
        "jersey_number", "status", "years_exp", "college", "height", "weight",
    ],
    # nfl_injuries
    "injuries": [
        "full_name", "position", "team", "season", "week", "report_primary_injury",
        "report_secondary_injury", "report_status", "practice_primary_injury",
        "practice_secondary_injury", "practice_status",
    ],
    # nfl_snap_counts
    "snap_counts": [
        "player", "position", "team", "opponent", "season", "week", "game_type",
        "offense_snaps", "offense_pct", "defense_snaps", "defense_pct", "st_snaps",
        "st_pct", "pfr_player_id",
    ],
    # nfl_team_records
    "team_records": [
        "season", "team", "source_team", "games", "wins", "losses", "ties", "source",
    ],
    # nfl_team_stats, nfl_compare (team mode). The tools select * and filter on team,
    # season_year and season_type; the rest are the columns group F added or redefined.
    "team_offense_stats": [
        "team", "season_year", "season_type", "pass_plays", "pass_attempts",
    ],
    "team_defense_stats": [
        "team", "season_year", "season_type", "turnovers_forced", "def_touchdowns",
    ],
    "situational_stats": [
        "team", "season_year", "season_type", "situation", "conversion_rule",
    ],
    # nfl_fantasy_opportunity
    "ff_opportunity": [
        "season", "week", "posteam", "full_name", "position", "player_id", "rec_attempt",
        "rush_attempt", "rec_air_yards", "receptions", "receptions_exp",
        "rec_yards_gained", "rush_yards_gained", "rec_touchdown", "rush_touchdown",
        "total_fantasy_points", "total_fantasy_points_exp", "total_fantasy_points_diff",
        "rec_attempt_team", "rush_attempt_team",
    ],
    # nfl_fantasy_rankings (table picked by scope)
    "ff_rankings_draft": [
        "player", "pos", "team", "ecr", "sd", "best", "worst", "bye", "page_type",
        "ecr_type", "scrape_date",
    ],
    "ff_rankings_week": [
        "player_name", "pos", "team", "rank", "ecr", "sd", "best", "worst", "pos_rank",
        "player_opponent", "player_bye_week", "start_sit_grade", "note", "page",
        "scrape_date",
    ],
    # nfl_ftn_charting (joined to plays on game_id, play_id)
    "ftn_charting": [
        "nflverse_game_id", "nflverse_play_id", "is_play_action", "is_rpo",
        "is_screen_pass", "is_no_huddle", "is_motion", "is_trick_play", "n_defense_box",
        "n_pass_rushers", "n_blitzers",
    ],
    # Derived analytic tables
    "player_td_luck": [
        "player_id", "full_name", "position", "team", "season", "rec_td", "rec_td_exp",
        "rec_td_luck", "rush_td", "rush_td_exp", "rush_td_luck", "total_td_luck_score",
    ],
    "player_role_trend": [
        "player_id", "full_name", "position", "team", "season", "week", "snap_pct",
        "target_share_pct", "carry_share_pct", "air_yards_share_pct", "snap_pct_3wk",
        "target_share_pct_3wk", "carry_share_pct_3wk", "air_yards_share_pct_3wk",
        "snap_pct_delta", "target_share_pct_delta", "carry_share_pct_delta",
        "air_yards_share_pct_delta",
    ],
    "player_separation_opportunity": [
        "player_id", "full_name", "position", "team", "season", "games", "fp_diff",
        "fp_diff_per_game", "td_luck", "catch_luck", "target_share_pct",
        "air_yards_share_pct", "avg_separation", "avg_yac_above_expectation",
        "catch_percentage", "regression_candidate",
    ],
    "player_drop_rate": [
        "player_id", "player", "team", "season", "catchable_targets", "drops",
        "drop_rate_pct", "contested_targets", "created_receptions",
    ],
    "player_contract_value": [
        "player_id", "full_name", "position", "team", "season", "apy", "cap_pct",
        "total_fp", "total_fp_exp", "fp_per_million",
    ],
    "injury_return_curve": [
        "injury_type", "position", "week_post_return", "avg_snap_pct_recovery",
        "median_snap_pct_recovery", "sample_size",
    ],
    # nfl_status and nfl_catalog
    "_ingest_metadata": [
        "dataset_id", "table_name", "season", "row_count", "loaded_at",
    ],
    # Not read by a tool today: the player-identity and stat-definition work reads them.
    "players": ["gsis_id", "display_name", "position"],
    "player_stats": [
        "player_id", "player_display_name", "position", "team", "season", "week",
        "season_type",
    ],
}

# Full snapshots replaced on every run: a smaller one is normal (rankings change week to
# week), so validate requires them non-empty instead of holding them to the row-count floor.
SNAPSHOT_TABLES: frozenset[str] = frozenset({"ff_rankings_draft", "ff_rankings_week"})
