"""Group H: team codes, season_type, argument hygiene.

Spec: nfl-mcp-notes/PLAN_FIX_GROUPS_MAIN_2026-11-04.md section H; evidence in
AUDIT_MAIN_2026-11-04.md items 16-18, 22-24, 27, 29-31.

Interface contract (beyond the shared brief)
- Team helper: every team / opponent parameter goes through canonical_team (strip, upper
  case). A filter on a canonical franchise expands to ALL codes of that franchise
  (key + aliases), so team="OAK" and team="LV" both match a 2019 posteam 'OAK' row and a
  2024 posteam 'LV' row. "JAC" becomes an alias of JAX (franchises.py lacks it today).
  ff_rankings_* store LAR / JAC, so the expansion is what makes team="LA" find LAR rows.
- Unknown code (including "" and a non-string such as 12): {"error": "Unknown team '<code>'"}
  where <code> is the value as typed. "" is an error, not "no filter".
- season_type on nfl_schedule and on nfl_snap_counts (new parameter, same name): REG, POST,
  WC, DIV, CON, SB, case-insensitive; POST = WC+DIV+CON+SB. Anything else:
  {"error": ...} whose message lists the allowed values.
- season_type on play-based tools (nfl_search_plays, nfl_player_stats, nfl_compare player
  mode, nfl_ftn_charting): REG or POST, case-insensitive. "SB" (and WC/DIV/CON) is an
  ERROR for these tools, not a silent map to POST, because plays only stores REG/POST.
- season=0 / week=0 are applied as filters (zero rows, optionally with "reason"); they are
  never treated as "no filter". min_yards=0 is applied as yards_gained >= 0.
- Out-of-range season or week: zero rows plus "reason" (str). A season reason names the
  loaded season range (here 2015 and 2024). A week reason names the loaded week range
  (here the last loaded week, 22).
- Boolean flags given as strings: "true"/"false" (any case) parse as booleans; any other
  non-bool value returns {"error": ...}.
- nfl_team_stats(side="bogus") and nfl_search_plays(situation="bogus") return {"error": ...}.
- nfl_compare team mode keys the result by canonical codes; two entities that resolve to
  the same franchise return {"error": ...}.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from nfl_mcp import tools

# ── Synthetic database ─────────────────────────────────────────────────────────

PLAY_COLS = {
    "game_id": "VARCHAR", "play_id": "DOUBLE", "season": "INTEGER", "week": "INTEGER",
    "season_type": "VARCHAR", "posteam": "VARCHAR", "defteam": "VARCHAR",
    "play_type": "VARCHAR", "down": "DOUBLE", "ydstogo": "DOUBLE", "yardline_100": "DOUBLE",
    "qtr": "DOUBLE", "half_seconds_remaining": "DOUBLE", "game_seconds_remaining": "DOUBLE",
    "goal_to_go": "INTEGER", "shotgun": "INTEGER", "no_huddle": "INTEGER",
    "score_differential": "DOUBLE", "wp": "DOUBLE", "desc": "VARCHAR", "epa": "DOUBLE",
    "yards_gained": "DOUBLE", "passer_player_name": "VARCHAR", "passer_player_id": "VARCHAR",
    "rusher_player_name": "VARCHAR", "rusher_player_id": "VARCHAR",
    "receiver_player_name": "VARCHAR", "receiver_player_id": "VARCHAR",
    "touchdown": "INTEGER", "pass_touchdown": "INTEGER", "rush_touchdown": "INTEGER",
    "interception": "INTEGER", "fumble_lost": "INTEGER", "complete_pass": "INTEGER",
    "passing_yards": "DOUBLE", "rushing_yards": "DOUBLE", "receiving_yards": "DOUBLE",
    "air_yards": "DOUBLE", "yards_after_catch": "DOUBLE", "cpoe": "DOUBLE", "sack": "INTEGER",
    "qb_dropback": "INTEGER", "pass_attempt": "INTEGER", "rush_attempt": "INTEGER",
    "qb_scramble": "INTEGER", "two_point_attempt": "INTEGER", "pass": "INTEGER",
    "rush": "INTEGER", "qb_kneel": "INTEGER", "qb_spike": "INTEGER",
}

QB, TE, WR, RB = ("P.Mahomes", "00-0000001"), ("T.Kelce", "00-0000002"), \
    ("R.Rice", "00-0000003"), ("I.Pacheco", "00-0000004")

# (n, season, week, season_type, posteam, defteam, play_type, yards, td, passer, receiver, rusher)
PLAYS = [
    (1, 2015, 1, "REG", "STL", "SEA", "run", 4, 0, None, None, ("X.Ram", "00-0000011")),
    (2, 2016, 1, "REG", "SD", "KC", "pass", 10, 0, ("X.Bolt", "00-0000012"), None, None),
    (3, 2019, 1, "REG", "OAK", "DEN", "pass", 8, 0, ("X.Raider", "00-0000013"), None, None),
    (4, 2019, 1, "REG", "JAC", "HOU", "run", 3, 0, None, None, ("X.Jag", "00-0000014")),
    (5, 2019, 1, "REG", "WAS", "PHI", "pass", 6, 0, ("X.Wash", "00-0000015"), None, None),
    (6, 2024, 1, "REG", "LV", "LAC", "pass", 12, 0, ("X.Raider", "00-0000013"), None, None),
    (7, 2024, 1, "REG", "LAC", "LV", "run", 5, 0, None, None, ("X.Bolt2", "00-0000016")),
    (8, 2024, 1, "REG", "LA", "DET", "pass", 7, 0, ("X.Ram2", "00-0000017"), None, None),
    (9, 2024, 1, "REG", "JAX", "MIA", "run", 2, 0, None, None, ("X.Jag2", "00-0000018")),
    (10, 2024, 1, "REG", "WAS", "TB", "pass", 9, 0, ("X.Wash2", "00-0000019"), None, None),
    (11, 2024, 1, "REG", "KC", "BAL", "pass", 15, 1, QB, TE, None),
    (12, 2024, 1, "REG", "KC", "BAL", "run", -3, 0, None, None, RB),
    (13, 2024, 2, "REG", "KC", "CIN", "pass", 20, 0, QB, TE, None),
    (14, 2024, 2, "REG", "KC", "CIN", "pass", 0, 0, QB, WR, None),
    (15, 2024, 22, "POST", "KC", "PHI", "pass", 11, 1, QB, TE, None),
    (16, 2024, 22, "POST", "PHI", "KC", "run", 6, 0, None, None, ("S.Barkley", "00-0000020")),
    (17, 2024, 2, "REG", "BUF", "NYJ", "pass", 7, 0, ("J.Allen", "00-0000021"), None, None),
]
N_PLAYS = 17          # rows above
N_PLAYS_NONNEG = 16   # all but play 12 (-3 yards)
N_TD_PLAYS = 2        # plays 11 and 15
KC_PLAYS, KC_REG_PLAYS, KC_POST_PLAYS = 5, 4, 1   # plays 11-15; 11-14 REG; 15 POST


def _play_row(p):
    n, season, week, st, pos, de, ptype, yds, td, passer, receiver, rusher = p
    is_pass = ptype == "pass"
    complete = int(is_pass and yds > 0)
    r = {c: None for c in PLAY_COLS}
    r.update(
        game_id=f"{season}_{week:02d}_{de}_{pos}", play_id=float(n), season=season, week=week,
        season_type=st, posteam=pos, defteam=de, play_type=ptype, down=1.0, ydstogo=10.0,
        yardline_100=50.0, qtr=1.0, half_seconds_remaining=900.0, game_seconds_remaining=2700.0,
        goal_to_go=0, shotgun=1, no_huddle=0, score_differential=0.0, wp=0.5,
        desc=f"synthetic play {n}", epa=0.1 * n, yards_gained=float(yds), touchdown=td,
        pass_touchdown=td if is_pass else 0, rush_touchdown=td if not is_pass else 0,
        interception=0, fumble_lost=0, complete_pass=complete,
        passing_yards=float(yds) if complete else None,
        receiving_yards=float(yds) if complete and receiver else None,
        rushing_yards=None if is_pass else float(yds), air_yards=5.0 if is_pass else None,
        yards_after_catch=2.0 if complete else None, cpoe=1.0 if is_pass else None, sack=0,
        qb_dropback=int(is_pass), pass_attempt=int(is_pass), rush_attempt=int(not is_pass),
        qb_scramble=0, two_point_attempt=0, **{"pass": int(is_pass), "rush": int(not is_pass)},
        qb_kneel=0, qb_spike=0,
    )
    for (name_col, id_col), who in (
        (("passer_player_name", "passer_player_id"), passer),
        (("receiver_player_name", "receiver_player_id"), receiver),
        (("rusher_player_name", "rusher_player_id"), rusher),
    ):
        if who:
            r[name_col], r[id_col] = who
    return tuple(r[c] for c in PLAY_COLS)


# Schedules: 11 games.  game_type REG 6, WC 2, DIV 1, CON 1, SB 1.
# 2024: REG 3 (all KC), WC 2 (HOU_KC, PIT_BUF), DIV 1, CON 1, SB 1 -> 2024 POST = 5.
# KC 2024: REG 3, POST 4 (one of each playoff round).  KC all seasons: 8 (adds 2016 SD_KC).
SCHEDULES = [
    ("2015_01_STL_SEA", 2015, "REG", 1, "STL", "SEA"),
    ("2016_01_SD_KC", 2016, "REG", 1, "SD", "KC"),
    ("2019_01_DEN_OAK", 2019, "REG", 1, "DEN", "OAK"),
    ("2024_01_BAL_KC", 2024, "REG", 1, "BAL", "KC"),
    ("2024_02_CIN_KC", 2024, "REG", 2, "CIN", "KC"),
    ("2024_03_KC_LV", 2024, "REG", 3, "KC", "LV"),
    ("2024_19_HOU_KC", 2024, "WC", 19, "HOU", "KC"),
    ("2024_19_PIT_BUF", 2024, "WC", 19, "PIT", "BUF"),
    ("2024_20_BUF_KC", 2024, "DIV", 20, "BUF", "KC"),
    ("2024_21_KC_BAL", 2024, "CON", 21, "KC", "BAL"),
    ("2024_22_KC_PHI", 2024, "SB", 22, "KC", "PHI"),
]

# Snap counts: 9 rows.  KC 2024: REG 2, WC 1, DIV 1, CON 1, SB 2 -> 7, POST 5.
SNAPS = [
    (2015, "REG", 1, "X.Ram", "STL", "SEA"),
    (2024, "REG", 1, "Patrick Mahomes", "KC", "BAL"),
    (2024, "REG", 1, "Travis Kelce", "KC", "BAL"),
    (2024, "WC", 19, "Patrick Mahomes", "KC", "HOU"),
    (2024, "DIV", 20, "Patrick Mahomes", "KC", "BUF"),
    (2024, "CON", 21, "Patrick Mahomes", "KC", "BAL"),
    (2024, "SB", 22, "Patrick Mahomes", "KC", "PHI"),
    (2024, "SB", 22, "Travis Kelce", "KC", "PHI"),
    (2024, "REG", 1, "X.Raider", "LV", "LAC"),
]

# rosters 6 (KC 3), injuries 4 (KC 2)
ROSTERS = [(2015, "STL", "QB", "X Ram", "00-0000011"), (2019, "OAK", "QB", "X Raider", "00-0000013"),
           (2024, "KC", "QB", "Patrick Mahomes", "00-0000001"), (2024, "KC", "TE", "Travis Kelce", "00-0000002"),
           (2024, "KC", "WR", "Rashee Rice", "00-0000003"), (2024, "LV", "QB", "X Raider", "00-0000013")]
INJURIES = [(2015, 1, "STL", "X Ram"), (2024, 1, "KC", "Rashee Rice"),
            (2024, 2, "KC", "Travis Kelce"), (2024, 1, "LV", "X Raider")]

# ff_rankings_draft: KC 2, LAR 2, JAC 1, LV 1 (FantasyPros codes, as in the real DB)
RANKINGS = [("Patrick Mahomes", "QB", "KC", 1.0), ("Travis Kelce", "TE", "KC", 2.0),
            ("Puka Nacua", "WR", "LAR", 3.0), ("Cooper Kupp", "WR", "LAR", 4.0),
            ("Travis Etienne", "RB", "JAC", 5.0), ("X Raider", "QB", "LV", 6.0)]

# player_* derived tables, ff_opportunity: KC 2, LV 1 each (season 2024)
PLAYER_ROWS = [("00-0000002", "Travis Kelce", "TE", "KC"), ("00-0000003", "Rashee Rice", "WR", "KC"),
               ("00-0000013", "X Raider", "QB", "LV")]

# team_records 6: KC 1; LV 2 (2019 source OAK, 2024); LAC 1 (2016 SD); LA 1 (2015 STL); BUF 1
RECORDS = [(2015, "LA", "STL"), (2016, "LAC", "SD"), (2019, "LV", "OAK"), (2024, "LV", "LV"),
           (2024, "KC", "KC"), (2024, "BUF", "BUF")]

# team stat tables: (team, season_year, season_type)
TEAM_SEASONS = [("STL", 2015, "REG"), ("OAK", 2019, "REG"), ("KC", 2024, "REG"), ("KC", 2024, "POST"),
                ("BUF", 2024, "REG"), ("BUF", 2024, "POST"), ("LV", 2024, "REG")]


def _build(path: Path) -> None:
    c = duckdb.connect(str(path))
    cols = ", ".join(f'"{k}" {v}' for k, v in PLAY_COLS.items())
    c.execute(f"CREATE TABLE plays ({cols})")
    c.executemany(f"INSERT INTO plays VALUES ({', '.join('?' * len(PLAY_COLS))})",
                  [_play_row(p) for p in PLAYS])

    c.execute("""CREATE TABLE schedules (game_id VARCHAR, season INTEGER, game_type VARCHAR,
        week INTEGER, gameday VARCHAR, gametime VARCHAR, away_team VARCHAR, home_team VARCHAR,
        away_score INTEGER, home_score INTEGER, result INTEGER, overtime INTEGER,
        spread_line DOUBLE, total_line DOUBLE, div_game INTEGER, roof VARCHAR, surface VARCHAR,
        temp INTEGER, wind INTEGER, away_coach VARCHAR, home_coach VARCHAR, referee VARCHAR,
        stadium VARCHAR)""")
    c.executemany("INSERT INTO schedules VALUES (?,?,?,?,?,'13:00',?,?,20,24,4,0,3.0,45.0,0,"
                  "'outdoors','grass',60,5,'A','B','R','S')",
                  [(g, s, t, w, f"{s}-09-{w:02d}", a, h) for g, s, t, w, a, h in SCHEDULES])

    c.execute("""CREATE TABLE snap_counts (game_id VARCHAR, pfr_game_id VARCHAR, season INTEGER,
        game_type VARCHAR, week INTEGER, player VARCHAR, pfr_player_id VARCHAR, position VARCHAR,
        team VARCHAR, opponent VARCHAR, offense_snaps DOUBLE, offense_pct DOUBLE,
        defense_snaps DOUBLE, defense_pct DOUBLE, st_snaps DOUBLE, st_pct DOUBLE)""")
    c.executemany("INSERT INTO snap_counts VALUES (?,?,?,?,?,?,?,?,?,?,60,0.9,0,0,2,0.1)",
                  [(f"{s}_{w}_{t}", f"p{s}{w}", s, gt, w, pl, f"pfr{pl[:3]}", "QB", t, o)
                   for s, gt, w, pl, t, o in SNAPS])

    c.execute("""CREATE TABLE rosters (season INTEGER, team VARCHAR, position VARCHAR,
        depth_chart_position VARCHAR, jersey_number INTEGER, status VARCHAR, full_name VARCHAR,
        gsis_id VARCHAR, years_exp INTEGER, college VARCHAR, height VARCHAR, weight INTEGER,
        week INTEGER, game_type VARCHAR)""")
    c.executemany("INSERT INTO rosters VALUES (?,?,?,?,15,'ACT',?,?,3,'U','6-2',225,1,'REG')",
                  [(s, t, p, p, n, g) for s, t, p, n, g in ROSTERS])

    c.execute("""CREATE TABLE injuries (season INTEGER, game_type VARCHAR, team VARCHAR,
        week INTEGER, gsis_id VARCHAR, position VARCHAR, full_name VARCHAR,
        report_primary_injury VARCHAR, report_secondary_injury VARCHAR, report_status VARCHAR,
        practice_primary_injury VARCHAR, practice_secondary_injury VARCHAR,
        practice_status VARCHAR, season_type VARCHAR)""")
    c.executemany("INSERT INTO injuries VALUES (?,'REG',?,?,'id','WR',?,'Knee',NULL,'Out',"
                  "'Knee',NULL,'DNP','REG')", [(s, t, w, n) for s, w, t, n in INJURIES])

    c.execute("""CREATE TABLE ff_rankings_draft (player VARCHAR, pos VARCHAR, team VARCHAR,
        ecr DOUBLE, sd DOUBLE, best BIGINT, worst BIGINT, bye BIGINT, page_type VARCHAR,
        ecr_type VARCHAR, scrape_date VARCHAR)""")
    c.executemany("INSERT INTO ff_rankings_draft VALUES (?,?,?,?,1.0,1,9,10,'redraft-overall',"
                  "'ro','2024-08-01')", RANKINGS)

    c.execute("""CREATE TABLE ftn_charting (nflverse_game_id VARCHAR, nflverse_play_id DOUBLE,
        season INTEGER, week INTEGER, n_defense_box INTEGER, is_no_huddle BOOLEAN,
        is_motion BOOLEAN, is_play_action BOOLEAN, is_screen_pass BOOLEAN, is_rpo BOOLEAN,
        is_trick_play BOOLEAN, is_catchable_ball BOOLEAN, is_contested_ball BOOLEAN,
        is_created_reception BOOLEAN, is_drop BOOLEAN, n_blitzers INTEGER, n_pass_rushers INTEGER)""")
    # Charting for plays 6 (LV) and 11-15 (KC: 4 REG + 1 POST).
    c.execute("INSERT INTO ftn_charting SELECT game_id, play_id, season, week, 7, false, true, "
              "false, false, false, false, true, false, false, false, 1, 4 FROM plays "
              "WHERE play_id IN (6, 11, 12, 13, 14, 15)")

    for table, extra in (("player_td_luck", "rec_td DOUBLE, rec_td_exp DOUBLE, rec_td_luck DOUBLE, "
                          "rush_td DOUBLE, rush_td_exp DOUBLE, rush_td_luck DOUBLE, total_td_luck_score DOUBLE"),
                         ("player_contract_value", "apy DOUBLE, cap_pct DOUBLE, total_fp DOUBLE, "
                          "total_fp_exp DOUBLE, fp_per_million DOUBLE")):
        c.execute(f"CREATE TABLE {table} (player_id VARCHAR, full_name VARCHAR, position VARCHAR, "
                  f"team VARCHAR, season INTEGER, {extra})")
        n_extra = extra.count("DOUBLE")
        c.executemany(f"INSERT INTO {table} VALUES (?,?,?,?,2024{',1.0' * n_extra})", PLAYER_ROWS)
    trend_cols = ["snap_pct", "target_share_pct", "carry_share_pct", "air_yards_share_pct"]
    trend_all = trend_cols + [f"{x}_3wk" for x in trend_cols] + [f"{x}_delta" for x in trend_cols]
    c.execute("CREATE TABLE player_role_trend (player_id VARCHAR, full_name VARCHAR, position VARCHAR, "
              "team VARCHAR, season INTEGER, week INTEGER, "
              + ", ".join(f"{x} DOUBLE" for x in trend_all) + ")")
    c.executemany(f"INSERT INTO player_role_trend VALUES (?,?,?,?,2024,2{',50.0' * len(trend_all)})",
                  PLAYER_ROWS)
    c.execute("""CREATE TABLE player_drop_rate (player_id VARCHAR, player VARCHAR, team VARCHAR,
        season INTEGER, catchable_targets INTEGER, drops INTEGER, drop_rate_pct DOUBLE,
        contested_targets INTEGER, created_receptions INTEGER)""")
    c.executemany("INSERT INTO player_drop_rate VALUES (?,?,?,2024,10,1,10.0,2,1)",
                  [(i, n, t) for i, n, _, t in PLAYER_ROWS])
    ffo = ["rec_attempt", "rush_attempt", "rec_air_yards", "receptions", "receptions_exp",
           "rec_yards_gained", "rush_yards_gained", "rec_touchdown", "rush_touchdown",
           "total_fantasy_points", "total_fantasy_points_exp", "total_fantasy_points_diff",
           "rec_attempt_team", "rush_attempt_team"]
    c.execute("CREATE TABLE ff_opportunity (season INTEGER, week INTEGER, posteam VARCHAR, "
              "full_name VARCHAR, position VARCHAR, player_id VARCHAR, "
              + ", ".join(f"{x} DOUBLE" for x in ffo) + ")")
    c.executemany(f"INSERT INTO ff_opportunity VALUES (2024,2,?,?,?,?{',1.0' * len(ffo)})",
                  [(t, n, p, i) for i, n, p, t in PLAYER_ROWS])

    c.execute("""CREATE TABLE team_records (season INTEGER, team VARCHAR, source_team VARCHAR,
        games INTEGER, wins INTEGER, losses INTEGER, ties INTEGER, source VARCHAR)""")
    c.executemany("INSERT INTO team_records VALUES (?,?,?,17,9,8,0,'nflverse')", RECORDS)

    c.execute("""CREATE TABLE team_offense_stats (team VARCHAR, season_year INTEGER,
        season_type VARCHAR, total_plays INTEGER, total_yards INTEGER, yards_per_play DOUBLE,
        rush_plays INTEGER, pass_plays INTEGER, touchdowns INTEGER, avg_epa DOUBLE)""")
    c.execute("""CREATE TABLE team_defense_stats (team VARCHAR, season_year INTEGER,
        season_type VARCHAR, plays_against INTEGER, yards_allowed INTEGER, sacks INTEGER,
        avg_epa_allowed DOUBLE)""")
    c.execute("""CREATE TABLE situational_stats (team VARCHAR, season_year INTEGER,
        season_type VARCHAR, situation VARCHAR, plays INTEGER, avg_yards DOUBLE,
        touchdowns INTEGER, conversion_pct DOUBLE, avg_epa DOUBLE)""")
    c.executemany("INSERT INTO team_offense_stats VALUES (?,?,?,100,500,5.0,50,50,3,0.1)", TEAM_SEASONS)
    c.executemany("INSERT INTO team_defense_stats VALUES (?,?,?,100,500,3,0.0)", TEAM_SEASONS)
    c.executemany("INSERT INTO situational_stats VALUES (?,?,?,'Red Zone',10,3.0,2,40.0,0.2)", TEAM_SEASONS)

    c.execute("""CREATE TABLE players (gsis_id VARCHAR, display_name VARCHAR, first_name VARCHAR,
        last_name VARCHAR, short_name VARCHAR, football_name VARCHAR, suffix VARCHAR,
        pfr_id VARCHAR, position VARCHAR, latest_team VARCHAR, rookie_season INTEGER,
        last_season INTEGER, status VARCHAR)""")
    c.executemany("INSERT INTO players VALUES (?,?,?,?,?,?,NULL,?,?,?,2017,2024,'ACT')", [
        ("00-0000001", "Patrick Mahomes", "Patrick", "Mahomes", "P.Mahomes", "Patrick", "pfrPat", "QB", "KC"),
        ("00-0000002", "Travis Kelce", "Travis", "Kelce", "T.Kelce", "Travis", "pfrTra", "TE", "KC"),
        ("00-0000003", "Rashee Rice", "Rashee", "Rice", "R.Rice", "Rashee", "pfrRas", "WR", "KC"),
        ("00-0000004", "Isiah Pacheco", "Isiah", "Pacheco", "I.Pacheco", "Isiah", "pfrIsi", "RB", "KC"),
    ])

    c.execute("""CREATE TABLE _ingest_metadata (dataset_id VARCHAR, table_name VARCHAR,
        season INTEGER, row_count BIGINT, loaded_at TIMESTAMP, loader_fn VARCHAR)""")
    for ds, tbl in (("pbp", "plays"), ("schedules", "schedules"), ("rosters", "rosters"),
                    ("injuries", "injuries"), ("snap_counts", "snap_counts")):
        for s in (2015, 2016, 2019, 2024):
            c.execute("INSERT INTO _ingest_metadata VALUES (?,?,?,1,TIMESTAMP '2024-09-01',?)",
                      [ds, tbl, s, f"load_{ds}"])
    c.close()


@pytest.fixture(scope="module")
def _synth_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("grp_h") / "synth.duckdb"
    _build(path)
    return path


@pytest.fixture
def db(_synth_path, monkeypatch):
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(_synth_path))
    return _synth_path


def _n(result: dict, key: str) -> int:
    """Rows returned, read from the tool's own key, falling back to the shared 'rows'."""
    assert isinstance(result, dict), result
    if key == "total_plays":
        return int(result.get("total_plays") or 0)
    rows = result.get(key)
    if rows is None:
        rows = result.get("rows", [])
    return len(rows)


# tool -> (call taking a team value, result key, hand count for "KC" or None)
TEAM_TOOLS = {
    "nfl_search_plays": (lambda t: tools.nfl_search_plays(team=t), "rows", KC_PLAYS),
    "nfl_team_stats": (lambda t: tools.nfl_team_stats(t, side="offense"), "offense", None),
    "nfl_roster": (lambda t: tools.nfl_roster(team=t), "players", 3),
    "nfl_injuries": (lambda t: tools.nfl_injuries(team=t), "injuries", 2),
    "nfl_schedule": (lambda t: tools.nfl_schedule(team=t), "games", 8),
    "nfl_snap_counts": (lambda t: tools.nfl_snap_counts(team=t), "snap_counts", 7),
    "nfl_fantasy_rankings": (lambda t: tools.nfl_fantasy_rankings(team=t), "fantasy_rankings", 2),
    "nfl_ftn_charting": (lambda t: tools.nfl_ftn_charting(team=t), "total_plays", 5),
    "nfl_td_luck": (lambda t: tools.nfl_td_luck(team=t), "td_luck", 2),
    "nfl_role_trend": (lambda t: tools.nfl_role_trend(team=t), "role_trend", 2),
    "nfl_drop_rate": (lambda t: tools.nfl_drop_rate(team=t), "drop_rate", 2),
    "nfl_contract_value": (lambda t: tools.nfl_contract_value(team=t), "contract_value", 2),
    "nfl_fantasy_opportunity": (lambda t: tools.nfl_fantasy_opportunity(team=t), "fantasy_opportunity", 2),
    "nfl_team_records": (lambda t: tools.nfl_team_records(team=t), "records", 1),
}

_XF = pytest.mark.xfail(strict=True, reason="group H: team helper (canonical_team, strip, upper)")


def _params(cases, xfail_ids, mark=_XF):
    return [pytest.param(*c, id="-".join(map(str, c)),
                         marks=[mark] if "-".join(map(str, c)) in xfail_ids else [])
            for c in cases]


# ── 1. Case-insensitive team on every team-taking tool ────────────────────────

_CASE_PASS = {f"{t}-kc" for t in TEAM_TOOLS if t != "nfl_search_plays"} | {"nfl_team_records- KC "}
_CASE = [(t, s) for t in TEAM_TOOLS for s in ("kc", " KC ")]


@pytest.mark.parametrize("tool,spelling", _params(
    _CASE, {"-".join(c) for c in _CASE} - _CASE_PASS))
def test_team_case_insensitive_matches_upper(db, tool, spelling):
    call, key, hand = TEAM_TOOLS[tool]
    expected = _n(call("KC"), key)
    if hand is not None:
        assert expected == hand
    assert expected > 0
    assert _n(call(spelling), key) == expected


# ── 2. Aliases expand to every code of the franchise ──────────────────────────

_ALIAS = [("OAK", 2), ("LV", 2), ("SD", 2), ("LAC", 2), ("STL", 2), ("LA", 2), ("LAR", 2),
          ("JAC", 2), ("JAX", 2), ("WSH", 2), ("WAS", 2)]


@pytest.mark.parametrize("code,expected", [
    pytest.param(c, e, id=c, marks=[] if c == "WAS" else
                 [pytest.mark.xfail(strict=True, reason="group H: alias expansion in plays filters")])
    for c, e in _ALIAS])
def test_search_plays_alias_returns_all_franchise_codes(db, code, expected):
    # Two rows per franchise: OAK 2019 + LV 2024, SD 2016 + LAC 2024, STL 2015 + LA 2024,
    # JAC 2019 + JAX 2024, WAS 2019 + WAS 2024.
    res = tools.nfl_search_plays(team=code)
    assert _n(res, "rows") == expected
    assert len({r["season"] for r in res["rows"]}) == 2


@pytest.mark.xfail(strict=True, reason="group H: alias expansion for opponent")
def test_search_plays_opponent_alias_oak_finds_lv_defense(db):
    # defteam LV appears once (play 7, 2024); no OAK defteam rows -> 1
    assert _n(tools.nfl_search_plays(opponent="oak"), "rows") == 1


@pytest.mark.parametrize("code,expected", [
    pytest.param("LAR", 2, id="LAR"), pytest.param("JAC", 1, id="JAC"),
    pytest.param("LA", 2, id="LA", marks=_XF), pytest.param("JAX", 1, id="JAX", marks=_XF),
    pytest.param("stl", 2, id="stl", marks=_XF)])
def test_fantasy_rankings_alias_finds_fantasypros_codes(db, code, expected):
    # ff_rankings_draft stores LAR (Nacua, Kupp) and JAC (Etienne)
    assert _n(tools.nfl_fantasy_rankings(team=code), "fantasy_rankings") == expected


# ── 3. Unknown team code ──────────────────────────────────────────────────────

_UNK = [(t, c) for t in TEAM_TOOLS for c in ("XX", "KCC", "", 12)]
# Pass today: nfl_team_records already uses canonical_team; nfl_search_plays(team=12) hits a
# DuckDB conversion error that is returned as {"error": ...}.
_UNK_PASS = {"nfl_team_records-XX", "nfl_team_records-KCC", "nfl_team_records-", "nfl_search_plays-12"}


@pytest.mark.parametrize("tool,code", _params(_UNK, {"-".join(map(str, c)) for c in _UNK} - _UNK_PASS))
def test_unknown_team_returns_error(db, tool, code):
    res = TEAM_TOOLS[tool][0](code)
    assert isinstance(res, dict) and "error" in res, res
    if isinstance(code, str) and code:
        assert "Unknown team" in res["error"] and code in res["error"]


# ── 4. season_type on nfl_schedule and nfl_snap_counts ───────────────────────

_XST = pytest.mark.xfail(strict=True, reason="group H: season_type helper (POST -> WC/DIV/CON/SB)")


@pytest.mark.parametrize("value,expected", [
    pytest.param("POST", 5, marks=_XST), pytest.param("post", 5, marks=_XST),
    ("WC", 2), ("DIV", 1), ("CON", 1), ("SB", 1), ("REG", 3), ("reg", 3), ("sb", 1)])
def test_schedule_season_type_2024(db, value, expected):
    # 2024: REG 3, WC 2, DIV 1, CON 1, SB 1 -> POST 5
    res = tools.nfl_schedule(season=2024, season_type=value)
    assert _n(res, "games") == expected


@pytest.mark.xfail(strict=True, reason="group H: season_type helper (POST -> WC/DIV/CON/SB)")
def test_schedule_post_kc_has_one_game_per_round(db):
    res = tools.nfl_schedule(team="KC", season=2024, season_type="POST")
    assert sorted(g["game_type"] for g in res["games"]) == ["CON", "DIV", "SB", "WC"]


def test_schedule_reg_excludes_playoffs(db):
    res = tools.nfl_schedule(team="KC", season=2024, season_type="REG")
    assert {g["game_type"] for g in res["games"]} == {"REG"} and res["count"] == 3


@pytest.mark.xfail(strict=True, reason="group H: season_type validation")
@pytest.mark.parametrize("value", ["XYZ", "playoffs"])
def test_schedule_unknown_season_type_lists_allowed(db, value):
    res = tools.nfl_schedule(season=2024, season_type=value)
    assert "error" in res
    for allowed in ("REG", "POST", "WC", "DIV", "CON", "SB"):
        assert allowed in res["error"]


@pytest.mark.xfail(strict=True, reason="group H: nfl_snap_counts gains season_type")
@pytest.mark.parametrize("value,expected", [("POST", 5), ("post", 5), ("SB", 2), ("WC", 1), ("REG", 2)])
def test_snap_counts_season_type(db, value, expected):
    # KC 2024 snaps: REG 2, WC 1, DIV 1, CON 1, SB 2 -> POST 5
    res = tools.nfl_snap_counts(team="KC", season=2024, **{"season_type": value})
    assert _n(res, "snap_counts") == expected


@pytest.mark.xfail(strict=True, reason="group H: nfl_snap_counts gains season_type")
def test_snap_counts_unknown_season_type_errors(db):
    res = tools.nfl_snap_counts(team="KC", **{"season_type": "XYZ"})
    assert "error" in res and "POST" in res["error"]


# ── 5. season_type on play-based tools ────────────────────────────────────────

_XPT = pytest.mark.xfail(strict=True, reason="group H: REG/POST upper-cased and validated for plays")


@pytest.mark.parametrize("value,expected", [
    ("REG", KC_REG_PLAYS), pytest.param("reg", KC_REG_PLAYS, marks=_XPT),
    ("POST", KC_POST_PLAYS), pytest.param("post", KC_POST_PLAYS, marks=_XPT)])
def test_search_plays_season_type_case(db, value, expected):
    assert _n(tools.nfl_search_plays(team="KC", season_type=value), "rows") == expected


@_XPT
def test_player_stats_season_type_lower_equals_upper(db):
    upper = tools.nfl_player_stats(player_name="Patrick Mahomes", season_type="REG")
    lower = tools.nfl_player_stats(player_name="Patrick Mahomes", season_type="reg")
    assert len(upper["seasons"]) == 1  # 2024 REG only
    assert lower["seasons"] == upper["seasons"]


@_XPT
def test_compare_player_season_type_lower_equals_upper(db):
    kw = dict(compare_type="player")
    upper = tools.nfl_compare("Patrick Mahomes", "Travis Kelce", season_type="REG", **kw)
    lower = tools.nfl_compare("Patrick Mahomes", "Travis Kelce", season_type="reg", **kw)
    assert "error" not in upper
    assert lower == upper


@pytest.mark.parametrize("value", ["REG", "reg"])
def test_ftn_charting_season_type_case(db, value):
    # KC charted REG plays: 11, 12, 13, 14 -> 4
    assert tools.nfl_ftn_charting(team="KC", season_type=value)["total_plays"] == KC_REG_PLAYS


@_XPT
@pytest.mark.parametrize("tool", ["search_plays", "player_stats", "ftn_charting", "compare"])
def test_plays_tools_reject_sb(db, tool):
    calls = {
        "search_plays": lambda: tools.nfl_search_plays(team="KC", season_type="SB"),
        "player_stats": lambda: tools.nfl_player_stats(player_name="Patrick Mahomes", season_type="SB"),
        "ftn_charting": lambda: tools.nfl_ftn_charting(team="KC", season_type="SB"),
        "compare": lambda: tools.nfl_compare("Patrick Mahomes", "Travis Kelce",
                                             compare_type="player", season_type="SB"),
    }
    res = calls[tool]()
    assert "error" in res and "REG" in res["error"] and "POST" in res["error"]


# ── 6. Zero is a filter, not "no filter" ──────────────────────────────────────

_XZ = pytest.mark.xfail(strict=True, reason="group H: explicit None checks (0 is a value)")
# tool -> (call(**kw), key, unfiltered hand count)
_ZERO = {
    "search_plays": (lambda **k: tools.nfl_search_plays(**k), "rows", N_PLAYS),
    "player_stats": (lambda **k: tools.nfl_player_stats(player_name="Patrick Mahomes", **k), "seasons", 2),
    "roster": (lambda **k: tools.nfl_roster(**k), "players", 6),
    "injuries": (lambda **k: tools.nfl_injuries(**k), "injuries", 4),
    "schedule": (lambda **k: tools.nfl_schedule(**k), "games", 11),
    "snap_counts": (lambda **k: tools.nfl_snap_counts(**k), "snap_counts", 9),
}


@_XZ
@pytest.mark.parametrize("tool,arg", [(t, "season") for t in _ZERO]
                         + [(t, "week") for t in ("search_plays", "injuries", "schedule", "snap_counts")])
def test_zero_season_or_week_is_applied(db, tool, arg):
    call, key, unfiltered = _ZERO[tool]
    assert _n(call(), key) == unfiltered
    res = call(**{arg: 0})
    assert _n(res, key) == 0 != unfiltered


@pytest.mark.parametrize("team,expected", [(None, N_PLAYS_NONNEG), ("KC", KC_PLAYS - 1)])
def test_min_yards_zero_is_applied(db, team, expected):
    # Only play 12 (KC, -3 yards) is below 0, so min_yards=0 drops exactly one row.
    assert _n(tools.nfl_search_plays(team=team, min_yards=0), "rows") == expected


# ── 7. Out-of-range season or week gives a reason ─────────────────────────────

_XR = pytest.mark.xfail(strict=True, reason="group H: out-of-range reason")
_RANGE = {
    "search_plays": (lambda: tools.nfl_search_plays(season=2030), "rows"),
    "player_stats": (lambda: tools.nfl_player_stats(player_name="Patrick Mahomes", season=2030), "seasons"),
    "schedule": (lambda: tools.nfl_schedule(season=2030), "games"),
    "team_stats": (lambda: tools.nfl_team_stats("KC", season=2030), "offense"),
    "roster": (lambda: tools.nfl_roster(team="KC", season=2030), "players"),
}


@_XR
@pytest.mark.parametrize("tool", list(_RANGE))
def test_season_out_of_range_reason_names_loaded_range(db, tool):
    call, key = _RANGE[tool]
    res = call()
    assert _n(res, key) == 0
    assert isinstance(res.get("reason"), str)
    assert "2015" in res["reason"] and "2024" in res["reason"]  # synthetic range 2015-2024


@_XR
@pytest.mark.parametrize("kw", [{"season": 2024, "week": 25}, {"week": 25}], ids=["2024", "any"])
def test_week_out_of_range_reason(db, kw):
    res = tools.nfl_search_plays(**kw)
    assert _n(res, "rows") == 0
    assert isinstance(res.get("reason"), str) and "22" in res["reason"]  # last loaded week 22


# ── 8. Empty / None / malformed arguments return an error ─────────────────────

_XE = pytest.mark.xfail(strict=True, reason="group H: argument hygiene (error, never raise)")
_BAD = {
    "team_stats_none": lambda: tools.nfl_team_stats(None),
    "team_stats_empty": lambda: tools.nfl_team_stats(""),
    "player_stats_empty": lambda: tools.nfl_player_stats(player_name=""),
    "player_stats_none": lambda: tools.nfl_player_stats(player_name=None),
    "compare_team_empty": lambda: tools.nfl_compare("", ""),
    "compare_player_empty": lambda: tools.nfl_compare("", "", compare_type="player"),
    "search_season_abc": lambda: tools.nfl_search_plays(season="abc"),
    "search_week_abc": lambda: tools.nfl_search_plays(week="abc"),
    "role_trend_min_snap_x": lambda: tools.nfl_role_trend(min_snap_pct="x"),
    "team_stats_side_bogus": lambda: tools.nfl_team_stats("KC", side="bogus"),
    "search_situation_bogus": lambda: tools.nfl_search_plays(situation="bogus"),
    "search_is_touchdown_maybe": lambda: tools.nfl_search_plays(is_touchdown="maybe"),
}


@_XE
@pytest.mark.parametrize("case", list(_BAD))
def test_bad_argument_returns_error(db, case):
    res = _BAD[case]()
    assert isinstance(res, dict) and "error" in res, res


@pytest.mark.parametrize("value,expected", [
    pytest.param("false", N_PLAYS, marks=_XE), pytest.param("FALSE", N_PLAYS, marks=_XE),
    ("true", N_TD_PLAYS), (True, N_TD_PLAYS), (False, N_PLAYS)])
def test_is_touchdown_string_booleans_parse(db, value, expected):
    # 17 plays, 2 touchdowns (plays 11, 15)
    assert _n(tools.nfl_search_plays(is_touchdown=value), "rows") == expected


# ── 9. nfl_compare team mode keys ─────────────────────────────────────────────

@pytest.mark.xfail(strict=True, reason="group H: compare keys by canonical code")
@pytest.mark.parametrize("a,b,ka,kb", [("kc", "buf", "KC", "BUF"), (" Kc", "oak", "KC", "LV")])
def test_compare_team_keys_are_canonical(db, a, b, ka, kb):
    res = tools.nfl_compare(a, b)
    assert ka in res and kb in res and a not in res and b not in res


@pytest.mark.xfail(strict=True, reason="group H: compare same franchise is an error")
@pytest.mark.parametrize("a,b", [("KC", "KC"), ("KC", "kc"), ("OAK", "LV")])
def test_compare_team_with_itself_errors(db, a, b):
    assert "error" in tools.nfl_compare(a, b)


# ── 10. nfl_team_records keeps canonical_team (regression, passes today) ─────

@pytest.mark.parametrize("alias,canon,seasons", [
    ("OAK", "LV", {2019, 2024}), ("SD", "LAC", {2016}), ("STL", "LA", {2015}), ("oak", "LV", {2019, 2024})])
def test_team_records_accepts_historical_alias(db, alias, canon, seasons):
    res = tools.nfl_team_records(team=alias)
    assert {r["team"] for r in res["records"]} == {canon}
    assert {r["season"] for r in res["records"]} == seasons


@pytest.mark.parametrize("code", ["XX", "KCC"])
def test_team_records_rejects_unknown_code(db, code):
    res = tools.nfl_team_records(team=code)
    assert "error" in res and code in res["error"] and "records" not in res


# ── Part 2: integration against the real DB ───────────────────────────────────

def _real_db() -> Path:
    from nfl_mcp.config import get_duckdb_path
    path = get_duckdb_path()
    if not path.exists():
        pytest.skip(f"No DuckDB database at {path}")
    with duckdb.connect(str(path), read_only=True) as c:
        if c.execute("SELECT COUNT(*) FROM plays WHERE season = 2024").fetchone()[0] == 0:
            pytest.skip("plays has no 2024 rows")
    return path


def _scalar(path: Path, sql: str, params=None):
    with duckdb.connect(str(path), read_only=True) as c:
        return c.execute(sql, params or []).fetchone()[0]


_KC_GAMES = ("SELECT COUNT(*) FROM schedules WHERE season = 2024 "
             "AND (home_team = 'KC' OR away_team = 'KC') AND game_type IN ")


@pytest.mark.integration
@pytest.mark.xfail(strict=True, reason="group H: season_type helper (POST -> WC/DIV/CON/SB)")
def test_integration_schedule_kc_2024_post():
    path = _real_db()
    expected = _scalar(path, _KC_GAMES + "('WC', 'DIV', 'CON', 'SB')")
    # Hand check: KC was the AFC #1 seed (WC bye) and played DIV, CON, SB -> 3.
    assert expected == 3
    res = tools.nfl_schedule(team="KC", season=2024, season_type="POST")
    assert res["count"] == expected


@pytest.mark.integration
@pytest.mark.parametrize("value", ["REG", pytest.param("reg")])
def test_integration_schedule_kc_2024_reg(value):
    path = _real_db()
    expected = _scalar(path, _KC_GAMES + "('REG')")
    assert expected == 17
    assert tools.nfl_schedule(team="KC", season=2024, season_type=value)["count"] == expected


@pytest.mark.integration
@pytest.mark.xfail(strict=True, reason="group H: team helper (canonical_team, strip, upper)")
def test_integration_search_plays_lowercase_team():
    _real_db()
    res = tools.nfl_search_plays(team="kc", season=2024, max_rows=5)
    assert len(res["rows"]) == 5 and {r["posteam"] for r in res["rows"]} == {"KC"}


@pytest.mark.integration
@pytest.mark.xfail(strict=True, reason="group H: alias expansion (LA -> LAR in ff_rankings)")
def test_integration_fantasy_rankings_la_finds_lar():
    path = _real_db()
    n_lar = _scalar(path, "SELECT COUNT(*) FROM ff_rankings_draft WHERE team = 'LAR'")
    assert n_lar > 0
    res = tools.nfl_fantasy_rankings(team="LA", limit=5)
    assert _n(res, "fantasy_rankings") == min(5, n_lar)


@pytest.mark.integration
@pytest.mark.xfail(strict=True, reason="group H: nfl_snap_counts gains season_type")
def test_integration_snap_counts_kc_2024_sb():
    path = _real_db()
    expected = _scalar(path, "SELECT COUNT(*) FROM snap_counts WHERE season = 2024 "
                             "AND team = 'KC' AND game_type = 'SB'")
    assert 0 < expected < 100
    res = tools.nfl_snap_counts(team="KC", season=2024, **{"season_type": "SB"})
    assert _n(res, "snap_counts") == expected
    assert {r["game_type"] for r in res["snap_counts"]} == {"SB"}
