"""Synthetic data for the rebuilt team tables.

Ten plays, one team box row per team per season_type, and two schedule rows - small enough to add
up by hand, which is the point: every number asserted in TestTeamTables is written out here.

  row  team  type down dist  to-go  spot  play            yards  epa     flags
  1    KC    REG   1   10     19     run   12              0.1   carries, first down, explosive
  2    KC    REG   3    7     45     pass   8              0.2   complete, third_down_converted
  3    KC    REG   3    2     18     run    1              0.3   third_down_converted by a flag:
                                                                 1 yard on 3rd and 2, so a rate
                                                                 built from yards against ydstogo
                                                                 would call it a failure
  4    KC    REG   4    1     10     pass  10              1.1   fourth_down_converted, pass TD
  5    KC    REG   4    5     30     punt  40             -0.5   NOT a scrimmage play
  6    KC    REG   4    8     25     FG     0              0.9   NOT a scrimmage play
  7    KC    REG   3    9     15     pass -12             -2.0   interception returned for a TD:
                                                                 touchdown=1, pass_touchdown=0
  8    KC    REG  xx  xx      xx     pass   0             -0.947  two_point_attempt=1
  9    BAL   REG   1   10     60     pass   6              0.05
 10    BAL   POST  1   10     40     pass  12              0.5

KC's red zone plays (spot <= 20) are rows 1, 3, 4, 7 and 8, on two drives: d2 (row 1) and d1
(rows 3, 4, 7, 8). Two trips, one touchdown: the rate is 50.0. Counting row 7's pick-six as an
offensive touchdown would make it 100.0, and dividing by the five red-zone plays instead of the two
trips would make it 20.0.

KC scrimmage rows are 1, 2, 3, 4 and 7: 19 yards over 5 plays (3.8), EPA -0.3 over 5 (-0.06).
Rows 5, 6 are out (pass = 0 and rush = 0) and row 8 is out (two_point_attempt = 1), so a punt, a
field goal and a two-point try are nowhere in the per-play figures.

Two rows carry first_down_penalty = 1: row 5 (KC REG, defensive pass interference on the punt, the
down is replayed) and row 10 (BAL POST, defensive holding on a first down). They pick a punt and a
play whose box row already counts a passing first down, because the box charges a penalty first down
to neither of its two first-down columns: it belongs to no passer, rusher or receiver. So
first_downs is the box's two columns plus these rows - KC REG 2 + 1 + 1 = 4, BAL REG 1 + 0 + 0 = 1,
BAL POST 1 + 0 + 1 = 2.
"""

import polars as pl

TEAM_PLAYS = {
    "posteam": ["KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "BAL", "BAL"],
    "defteam": ["BAL", "BAL", "BAL", "BAL", "BAL", "BAL", "BAL", "BAL", "KC", "KC"],
    "season": [2024] * 10,
    "season_type": ["REG"] * 9 + ["POST"],
    "down": [1, 3, 3, 4, 4, 4, 3, None, 1, 1],
    "ydstogo": [10, 7, 2, 1, 5, 8, 9, 1, 10, 10],
    "yardline_100": [19, 45, 18, 10, 30, 25, 15, 1, 60, 40],
    "play_type": ["run", "pass", "run", "pass", "punt", "field_goal", "pass", "pass", "pass", "pass"],
    "pass": [0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    "rush": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "pass_attempt": [0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    "rush_attempt": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "yards_gained": [12, 8, 1, 10, 40, 0, -12, 0, 6, 12],
    "play_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "epa": [0.1, 0.2, 0.3, 1.1, -0.5, 0.9, -2.0, -0.947, 0.05, 0.5],
    "touchdown": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "pass_touchdown": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "rush_touchdown": [0] * 10,
    "interception": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "fumble_lost": [0] * 10,
    "sack": [0] * 10,
    "third_down_converted": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    "third_down_failed": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "fourth_down_converted": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "fourth_down_failed": [0] * 10,
    # Row 5 and row 10 drew a defensive penalty that is an automatic first down. The box credits
    # neither the passer nor the rusher with them, so plays is the only place they exist.
    "first_down_penalty": [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    "two_point_attempt": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "game_id": ["g1"] * 8 + ["g1", "g2"],
    "drive": ["d2", "d1", "d1", "d1", "d1", "d1", "d1", "d1", "d1", "d1"],
    "qtr": [1, 2, 3, 4, 4, 4, 3, 4, 1, 2],
    "time": ["10:00"] * 10,
    "shotgun": [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    "no_huddle": [0] * 10,
}

# The week-level team box. It is the authority for a counting stat, so a value here wins over what
# counting plays would say. A defence's sacks are its opponents' sacks_suffered, not its own
# def_sacks: KC's def_sacks below is 1, but BAL suffered 0 sacks, and 0 is what the table shows.
TEAM_BOX = {
    "team": ["KC", "BAL", "BAL"],
    "opponent_team": ["BAL", "KC", "KC"],
    "season": [2024, 2024, 2024],
    "season_type": ["REG", "REG", "POST"],
    "attempts": [3, 1, 1],
    "completions": [2, 1, 1],
    "passing_yards": [18, 6, 12],
    "sacks_suffered": [0, 0, 0],
    "sack_yards_lost": [0, 0, 0],
    "carries": [2, 0, 0],
    "rushing_yards": [14, 0, 0],
    "passing_tds": [1, 0, 0],
    "rushing_tds": [0, 0, 0],
    "special_teams_tds": [0, 0, 0],
    "passing_interceptions": [1, 0, 0],
    "fumbles_lost_total": [0, 0, 0],
    "passing_first_downs": [2, 1, 1],
    "rushing_first_downs": [1, 0, 0],
    # Set equal to passing_first_downs, the way the real box has them (544 of 544 Kansas City
    # 2024 rows), so a build that added this column again would overshoot instead of agreeing.
    "receiving_first_downs": [2, 1, 1],
    "penalties": [3, 1, 0],
    "penalty_yards": [20, 5, 0],
    "def_sacks": [1, 0, 0],
    "def_interceptions": [0, 1, 0],
}

TEAM_SCHEDULE = {
    "season": [2024, 2024],
    "game_id": ["g1", "g2"],
    "game_type": ["REG", "WC"],
    "home_team": ["KC", "BAL"],
    "away_team": ["BAL", "KC"],
    "home_score": [21, 10],
    "away_score": [14, 17],
}


def load_team_sources(conn, write_df, *, with_box=True, with_schedule=True):
    """Write the ten plays, and the box and the schedule unless asked to leave one out."""
    conn.execute("DROP TABLE IF EXISTS plays")
    write_df(conn, "plays", pl.DataFrame(TEAM_PLAYS), replace=True)
    for table, keep in (("team_stats_raw", with_box), ("schedules", with_schedule)):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        if keep:
            write_df(conn, table, pl.DataFrame(TEAM_BOX if table == "team_stats_raw"
                                               else TEAM_SCHEDULE), replace=True)


def rows(conn, sql, *params):
    cols = [d[0] for d in conn.execute(sql, params).description]
    return [dict(zip(cols, r)) for r in conn.execute(sql, params).fetchall()]
