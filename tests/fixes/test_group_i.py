"""Group I: pagination on every list tool and the nfl_search_plays rewrite.

Spec: PLAN_FIX_GROUPS_MAIN_2026-11-04.md section I; evidence AUDIT_MAIN items 6, 19-22.

Interface contract (beyond the shared brief)
- Every list tool (nfl_roster, nfl_injuries, nfl_schedule, nfl_snap_counts, nfl_fantasy_opportunity,
  nfl_fantasy_rankings, nfl_search_plays, nfl_td_luck, nfl_role_trend, nfl_separation_opportunity,
  nfl_drop_rate, nfl_contract_value, nfl_injury_return, nfl_team_records) takes limit (int, default
  100, 1..500) and offset (int >= 0, default 0). limit 0, -1, 501 or a non-int ("5") -> {"error": ...}.
  The response keeps its list key (players, injuries, games, snap_counts, fantasy_opportunity,
  fantasy_rankings, rows, td_luck, role_trend, separation_opportunity, drop_rate, contract_value,
  injury_return, records) and adds "count", "truncated" and, only when truncated, "hint" (a string
  that names "offset" and "nfl_query").
- Page order is each tool's existing ORDER BY (fixture sort keys are distinct, so A..G is the order);
  the fix must add a unique tiebreak so offset pages never overlap on real data with ties.
- nfl_snap_counts rows gain "pfr_player_id" (needed to prove page coverage on real data). A call
  without season_type covers every game_type (REG and playoffs).
- nfl_search_plays(..., player_id=None, player_role=None, down=None, qtr=None,
  min_yardline_100=None, max_yardline_100=None, min_score_differential=None,
  max_score_differential=None, shotgun=None, no_huddle=None, min_wp=None, max_wp=None,
  sort_by="abs_epa", columns=None, limit=100, offset=0, max_rows=<alias of limit>).
  Numeric bounds are inclusive. shotgun/no_huddle True -> column = 1, False -> column = 0.
- player_role with player_id: passer -> passer_player_id; rusher -> rusher_player_id; receiver ->
  receiver_player_id; any_offense -> any of those three; defender -> any of sack, half_sack_1/2,
  solo_tackle_1/2, assist_tackle_1..4, tackle_with_assist_1/2, interception, pass_defense_1/2,
  forced_fumble_player_1/2, qb_hit_1/2 (and tackle_for_loss_1/2) *_player_id; kicker ->
  kicker_player_id only; punter -> punter_player_id; returner -> kickoff_returner_player_id or
  punt_returner_player_id; any -> union of every role. Each play appears once. Unknown role -> error.
- play_type accepts the nine non-NULL values of SELECT DISTINCT play_type (the tenth DISTINCT value is
  NULL, which is not selectable); unknown -> error that lists the allowed values. server.py's enum
  equals those nine.
- situation: red_zone = yardline_100 <= 20 (any play type, unchanged); third_down/fourth_down = down 3/4;
  two_minute = half_seconds_remaining <= 120 in qtr 2, 4 or OT; two_minute_half = qtr 2 and
  half_seconds_remaining <= 120; two_minute_game = qtr >= 4 and half_seconds_remaining <= 120;
  goal_to_go = goal_to_go = 1. Unknown situation -> error.
- sort_by: abs_epa (default, ABS(epa) DESC), epa DESC, wpa DESC, yards_gained DESC, game_order
  (game_id ASC, play_id ASC). Unknown -> error.
- columns: list of extra plays columns, checked against the live schema; unknown -> {"error",
  "did_you_mean": [...]} and no "rows". Default row keys include game_id, play_id, passer/rusher/
  receiver_player_id, posteam, defteam, season, week, qtr, down, play_type, desc, epa, wpa.
"""

import json

import duckdb
import pytest

from nfl_mcp import tools

XF = "group I: "

# ── synthetic plays (60 rows, two games) ─────────────────────────────────────────

_ID_ROLES = (
    "td passer receiver rusher lateral_receiver lateral_rusher lateral_sack interception "
    "lateral_interception punt_returner lateral_punt_returner kickoff_returner "
    "lateral_kickoff_returner punter kicker own_kickoff_recovery blocked tackle_for_loss_1 "
    "tackle_for_loss_2 qb_hit_1 qb_hit_2 forced_fumble_player_1 forced_fumble_player_2 "
    "solo_tackle_1 solo_tackle_2 assist_tackle_1 assist_tackle_2 assist_tackle_3 assist_tackle_4 "
    "tackle_with_assist_1 tackle_with_assist_2 pass_defense_1 pass_defense_2 fumbled_1 fumbled_2 "
    "fumble_recovery_1 fumble_recovery_2 sack half_sack_1 half_sack_2 penalty safety fantasy"
).split()
_PLAY_DOUBLES = (
    "play_id qtr down ydstogo yardline_100 goal_to_go quarter_seconds_remaining "
    "half_seconds_remaining game_seconds_remaining score_differential posteam_score defteam_score "
    "shotgun no_huddle qb_dropback qb_scramble pass_attempt rush_attempt sack two_point_attempt "
    "yards_gained epa wpa wp def_wp vegas_wp ep qb_epa success first_down touchdown pass_touchdown "
    "rush_touchdown interception fumble fumble_lost complete_pass incomplete_pass air_yards "
    "yards_after_catch cpoe cp xpass pass_oe passing_yards rushing_yards receiving_yards "
    "kick_distance return_yards penalty drive order_sequence special_teams_play timeout qb_kneel "
    "qb_spike punt_attempt kickoff_attempt field_goal_attempt extra_point_attempt solo_tackle "
    "assist_tackle qb_hit"
).split()
_PLAY_TEXT = (
    "game_id season_type game_date home_team away_team posteam defteam posteam_type play_type "
    "play_type_nfl desc game_half field_goal_result extra_point_result pass_location run_location "
    "run_gap pass_length time yrdln side_of_field"
).split()
PLAY_COLS = {"season": "INTEGER", "week": "INTEGER"}
PLAY_COLS.update({c: "DOUBLE" for c in _PLAY_DOUBLES})
PLAY_COLS.update({c: "VARCHAR" for c in _PLAY_TEXT})
for _r in _ID_ROLES:
    PLAY_COLS[f"{_r}_player_id"] = "VARCHAR"
    PLAY_COLS[f"{_r}_player_name"] = "VARCHAR"

X1, DF1, K1, RT1, P1 = "00-0090001", "00-0090002", "00-0090003", "00-0090004", "00-0090005"
QB1, RB1, WR1, QB2, RB2, WR2 = (f"00-00900{n}" for n in (11, 12, 13, 21, 22, 23))
QB3, RB3, WR3, DF2, K2, K3, P2, RT2, RT3 = (f"00-00900{n}" for n in (31, 32, 33, 41, 42, 43, 44, 45, 46))
NAMES = {X1: "X.One", QB1: "Q.One", RB1: "R.One", WR1: "W.One", QB2: "Q.Two", RB2: "R.Two",
         WR2: "W.Two", QB3: "Q.Three", RB3: "R.Three", WR3: "W.Three"}
ROLE_COL = {"passer": "passer", "rusher": "rusher", "receiver": "receiver", "sack": "sack",
            "half_sack": "half_sack_1", "intc": "interception", "pdef": "pass_defense_1",
            "ff": "forced_fumble_player_1", "solo": "solo_tackle_1", "assist": "assist_tackle_1",
            "qbhit": "qb_hit_1", "kicker": "kicker", "punter": "punter", "kr": "kickoff_returner",
            "pr": "punt_returner"}
GAMES = {"A": ("2024_01_BAL_KC", 1, "KC", "BAL"), "B": ("2024_02_KC_CIN", 2, "CIN", "KC")}


def P(pid, game, qtr, qsr, ptype, down=None, yl=None, yds=0, pos="KC", sg=0, nh=0, gtg=0,
      sd=0, wp=0.5, **kw):
    gid, week, home, away = GAMES[game]
    hsr = qsr + 900 if qtr in (1, 3) else qsr
    gsr = {1: qsr + 2700, 2: qsr + 1800, 3: qsr + 900}.get(qtr, qsr)
    row = dict(game_id=gid, play_id=pid * 10.0, order_sequence=float(pid), season=2024, week=week,
               season_type="REG", home_team=home, away_team=away, posteam=pos,
               defteam=away if pos == home else home, qtr=qtr, down=down, ydstogo=10,
               yardline_100=yl, quarter_seconds_remaining=qsr, half_seconds_remaining=hsr,
               game_seconds_remaining=gsr, play_type=ptype, desc=f"#{pid} {ptype}",
               yards_gained=yds, shotgun=sg, no_huddle=nh, goal_to_go=gtg, score_differential=sd,
               wp=wp, epa=round(((pid * 37) % 61 - 30) / 100, 3),
               wpa=round(((pid * 13) % 61 - 30) / 1000, 4))
    for key, val in kw.items():
        if key in ROLE_COL:
            row[f"{ROLE_COL[key]}_player_id"] = val
            if key in ("passer", "rusher", "receiver"):
                row[f"{ROLE_COL[key]}_player_name"] = NAMES.get(val)
        else:
            row[key] = val
    return row


ROWS = [
    # game A, Q1
    P(1, "A", 1, 900, "kickoff", yl=35, kicker=K1, kr=RT1, pos="BAL"),
    P(2, "A", 1, 880, "run", 1, 75, 4, rusher=RB1, solo=DF2),
    P(3, "A", 1, 850, "pass", 2, 71, 12, sg=1, passer=QB1, receiver=WR1, solo=DF2),
    P(4, "A", 1, 820, "pass", 1, 59, 9, sg=1, nh=1, passer=QB1, receiver=X1),
    P(5, "A", 1, 790, "run", 2, 50, 3, rusher=X1),
    P(6, "A", 1, 760, "pass", 1, 47, -7, sg=1, passer=QB1, sack=DF1),
    P(7, "A", 1, 730, "pass", 2, 54, 0, sg=1, passer=QB1, receiver=WR1, pdef=DF1),
    P(8, "A", 1, 700, "punt", 4, 54, punter=P1, pr=RT1),
    P(9, "A", 1, 650, "no_play", 3, 60, passer=QB1),
    P(10, "A", 1, 600, None),
    # Q2 (16-20 inside 2:00 of the half; 21 is at 2:01)
    P(11, "A", 2, 880, "pass", 1, 40, 8, sg=1, passer=QB1, receiver=WR1),
    P(12, "A", 2, 850, "run", 2, 32, 5, rusher=RB1, assist=DF1),
    P(13, "A", 2, 800, "pass", 1, 27, 27, sg=1, passer=QB1, receiver=X1),
    P(14, "A", 2, 795, "extra_point", yl=15, kicker=K1),
    P(15, "A", 2, 790, "kickoff", yl=35, kicker=K1, kr=X1, pos="BAL"),
    P(16, "A", 2, 110, "pass", 1, 70, 15, sg=1, nh=1, passer=QB1, receiver=WR1),
    P(17, "A", 2, 90, "run", 2, 55, 6, sg=1, nh=1, rusher=X1),
    P(18, "A", 2, 60, "pass", 1, 20, 9, sg=1, nh=1, passer=QB1, receiver=WR1),
    P(19, "A", 2, 30, "qb_spike", 2, 11, passer=QB1),
    P(20, "A", 2, 25, "field_goal", 3, 11, kicker=K1),
    P(21, "A", 2, 121, "pass", 3, 74, 6, sg=1, passer=QB1, receiver=WR1),
    # Q3
    P(22, "A", 3, 900, "kickoff", yl=35, kicker=K2, kr=RT1),
    P(23, "A", 3, 870, "run", 1, 75, 2, pos="BAL", rusher=RB2, solo=DF1, ff=DF1),
    P(24, "A", 3, 840, "pass", 2, 73, 20, sg=1, passer=QB1, receiver=WR1),
    P(25, "A", 3, 800, "pass", 1, 53, 0, sg=1, passer=QB1, intc=DF2, epa=-5.2),
    P(26, "A", 3, 700, "run", 1, 8, 3, pos="BAL", gtg=1, rusher=RB2),
    P(27, "A", 3, 680, "run", 2, 5, 5, pos="BAL", gtg=1, rusher=RB2),
    P(28, "A", 3, 675, "extra_point", yl=15, pos="BAL", kicker=K2),
    P(29, "A", 3, 670, "kickoff", yl=35, kicker=K1, kr=RT2, pos="BAL"),
    # Q4 (31-36 inside 2:00)
    P(30, "A", 4, 800, "pass", 1, 70, 6, pos="BAL", sg=1, passer=QB2, receiver=WR2, qbhit=DF1),
    P(31, "A", 4, 100, "pass", 1, 60, 14, pos="BAL", sg=1, nh=1, sd=-4, wp=0.12, passer=QB2, receiver=WR2),
    P(32, "A", 4, 80, "pass", 2, 46, -5, pos="BAL", sg=1, sd=-4, wp=0.12, passer=QB2, half_sack=DF1),
    P(33, "A", 4, 40, "run", 3, 51, 2, pos="BAL", sd=-4, wp=0.12, rusher=RB2),
    P(34, "A", 4, 30, "punt", 4, 49, pos="BAL", sd=-4, punter=P2, pr=X1),
    P(35, "A", 4, 20, "qb_kneel", 1, 70, -1, sd=4, wp=0.99, rusher=QB1),
    P(36, "A", 4, 5, "qb_kneel", 2, 72, -1, sd=4, wp=0.99, rusher=QB1),
    # game B, Q1
    P(37, "B", 1, 900, "kickoff", yl=35, kicker=K3, kr=RT1),
    P(38, "B", 1, 880, "pass", 1, 75, 75, sg=1, passer=QB1, receiver=WR1, epa=6.5, wpa=0.30,
      air_yards=40.0, cpoe=35.5),
    P(39, "B", 1, 875, "extra_point", yl=15, kicker=K1),
    P(40, "B", 1, 870, "kickoff", yl=35, kicker=K1, kr=RT3, pos="CIN"),
    P(41, "B", 1, 800, "run", 1, 75, 52, pos="CIN", sd=-7, rusher=RB3),
    P(42, "B", 1, 780, "pass", 1, 23, 20, pos="CIN", sg=1, sd=-7, passer=QB3, receiver=WR3),
    P(43, "B", 1, 750, "pass", 1, 3, 0, pos="CIN", sg=1, gtg=1, sd=-7, passer=QB3, receiver=WR3),
    P(44, "B", 1, 745, "run", 2, 3, 0, pos="CIN", gtg=1, sd=-7, rusher=RB3),
    P(45, "B", 1, 740, "pass", 3, 3, 0, pos="CIN", sg=1, gtg=1, sd=-7, passer=QB3, receiver=WR3),
    P(46, "B", 1, 735, "field_goal", 4, 3, pos="CIN", gtg=1, sd=-7, kicker=K3),
    # Q2-Q4
    P(47, "B", 2, 850, "no_play", 2, 44),
    P(48, "B", 2, 840, "run", 1, 44, 7, rusher=X1),
    P(49, "B", 2, 820, "pass", 2, 37, 37, passer=X1, receiver=WR1, epa=4.1),
    P(50, "B", 2, 815, None),
    P(51, "B", 3, 600, "pass", 3, 30, 4, sg=1, passer=QB1, receiver=X1),
    P(52, "B", 3, 560, "run", 4, 26, 2, rusher=X1),
    P(53, "B", 3, 500, "pass", 1, 24, 10, sg=1, passer=QB1, receiver=X1),
    P(54, "B", 4, 400, "field_goal", 4, 14, sd=7, wp=0.85, kicker=K1),
    P(55, "B", 4, 300, "pass", 1, 75, 9, pos="CIN", sg=1, sd=-10, passer=QB3, receiver=WR3),
    P(56, "B", 4, 110, "pass", 3, 40, 6, pos="CIN", sg=1, nh=1, sd=-10, wp=0.8, passer=QB3, receiver=WR3),
    P(57, "B", 4, 3, "run", 1, 35, 1, pos="CIN", sd=-10, rusher=RB3),
    # overtime (59, 60 inside 2:00)
    P(58, "B", 5, 600, "kickoff", yl=35, kicker=K3, kr=RT1),
    P(59, "B", 5, 115, "pass", 1, 50, 20, sg=1, nh=1, wp=0.6, wpa=0.45, passer=QB1, receiver=WR1),
    P(60, "B", 5, 100, "run", 1, 30, 5, rusher=RB1),
]
REAL_PLAY_TYPES = {"pass", "run", "field_goal", "punt", "kickoff", "extra_point", "qb_kneel",
                   "qb_spike", "no_play"}  # SELECT DISTINCT play_type FROM plays, NULL excluded


def _table(con, name, cols, rows):
    """cols: {name: type}; rows: list of dicts (missing keys -> NULL)."""
    con.execute(f"CREATE TABLE {name} (" + ", ".join(f'"{c}" {t}' for c, t in cols.items()) + ")")
    if rows:
        names = list(cols)
        con.executemany(
            f"INSERT INTO {name} VALUES ({', '.join('?' * len(names))})",
            [[r.get(c) for c in names] for r in rows],
        )


def _meta(con):
    con.execute("CREATE TABLE _ingest_metadata (dataset_id VARCHAR, table_name VARCHAR, "
                "season INTEGER, row_count BIGINT, loaded_at TIMESTAMP, loader_fn VARCHAR)")
    con.execute("INSERT INTO _ingest_metadata VALUES ('pbp', 'plays', 2024, 60, now(), 'x')")


@pytest.fixture(scope="module")
def _plays_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("grp_i") / "plays.duckdb"
    con = duckdb.connect(str(path))
    _table(con, "plays", PLAY_COLS, list(reversed(ROWS)))  # physical order != game order
    _table(con, "players", {"gsis_id": "VARCHAR", "display_name": "VARCHAR",
                            "short_name": "VARCHAR", "position": "VARCHAR"},
           [{"gsis_id": k, "display_name": v, "short_name": v} for k, v in NAMES.items()])
    _meta(con)
    con.close()
    return path


@pytest.fixture
def plays_db(_plays_file, monkeypatch):
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(_plays_file))


def _search(**kw):
    kw.setdefault("season", 2024)
    return tools.nfl_search_plays(**kw)


def _pid(row):
    return int(row["desc"].split()[0].lstrip("#"))


def _pids(res):
    assert "error" not in res, res
    return {_pid(r) for r in res["rows"]}


# ── 3. play_type ─────────────────────────────────────────────────────────────────

PLAY_TYPE_PIDS = {  # hand count from ROWS
    "kickoff": {1, 15, 22, 29, 37, 40, 58}, "punt": {8, 34}, "field_goal": {20, 46, 54},
    "extra_point": {14, 28, 39}, "qb_kneel": {35, 36}, "qb_spike": {19}, "no_play": {9, 47},
    "run": {2, 5, 12, 17, 23, 26, 27, 33, 41, 44, 48, 52, 57, 60},
    "pass": {3, 4, 6, 7, 11, 13, 16, 18, 21, 24, 25, 30, 31, 32, 38, 42, 43, 45, 49, 51, 53, 55,
             56, 59},
}


def test_fixture_play_type_hand_counts_cover_sixty_rows():
    assert set(PLAY_TYPE_PIDS) == REAL_PLAY_TYPES
    assert sum(len(v) for v in PLAY_TYPE_PIDS.values()) == 58  # + 2 NULL play_type rows (10, 50)


@pytest.mark.parametrize("ptype", sorted(REAL_PLAY_TYPES))
def test_play_type_filter_returns_only_that_type(plays_db, ptype):
    res = _search(play_type=ptype)
    assert _pids(res) == PLAY_TYPE_PIDS[ptype]
    assert {r["play_type"] for r in res["rows"]} == {ptype}


@pytest.mark.parametrize("bad", ["passs", "kneel"])
def test_play_type_unknown_errors_with_allowed_list(plays_db, bad):
    res = _search(play_type=bad)
    assert "error" in res
    blob = json.dumps(res, default=str)
    assert all(v in blob for v in REAL_PLAY_TYPES)


def _schema_play_type_enum():
    from nfl_mcp import server
    tool = next(t for t in server.TOOLS if t.name == "nfl_search_plays")
    return set(tool.inputSchema["properties"]["play_type"]["enum"])


def test_schema_play_type_enum_matches_real_distinct_list():
    assert _schema_play_type_enum() == REAL_PLAY_TYPES


# ── 4. player_role ───────────────────────────────────────────────────────────────

X1_ROLES = {"passer": {49}, "rusher": {5, 17, 48, 52}, "receiver": {4, 13, 51, 53},
            "returner": {15, 34}}
X1_ROLES["any_offense"] = X1_ROLES["passer"] | X1_ROLES["rusher"] | X1_ROLES["receiver"]  # 9
X1_ROLES["any"] = X1_ROLES["any_offense"] | X1_ROLES["returner"]  # 11
ROLE_CASES = [(X1, role, pids) for role, pids in X1_ROLES.items()] + [
    # DF1: sack 6, pass defense 7, assist 12, solo+forced fumble on one play 23, qb hit 30, half sack 32
    (DF1, "defender", {6, 7, 12, 23, 30, 32}),
    (DF1, "any", {6, 7, 12, 23, 30, 32}),
    (DF2, "defender", {2, 3, 25}),  # solo 2, 3; interception 25
    (K1, "kicker", {1, 14, 15, 20, 29, 39, 40, 54}),  # kickoffs 1,15,29,40; XP 14,39; FG 20,54
    (P1, "punter", {8}),
    (P1, "kicker", set()),  # punter is not a kicker
    (RT1, "returner", {1, 8, 22, 37, 58}),
    (QB1, "rusher", {35, 36}),  # kneels
    (X1, "defender", set()),
]


@pytest.mark.parametrize("pid_, role, expected", ROLE_CASES,
                         ids=[f"{r}-{p[-2:]}" for p, r, _ in ROLE_CASES])
def test_player_role_returns_exact_rows(plays_db, pid_, role, expected):
    assert _pids(_search(player_id=pid_, player_role=role)) == expected


def test_player_role_kicker_with_play_type_field_goal(plays_db):
    assert _pids(_search(player_id=K1, player_role="kicker", play_type="field_goal")) == {20, 54}


@pytest.mark.parametrize("bad", ["quarterback", "tackler"])
def test_player_role_unknown_is_error(plays_db, bad):
    res = _search(player_id=X1, player_role=bad)
    assert "error" in res and "rows" not in res


# ── 5. situation ─────────────────────────────────────────────────────────────────

SITUATIONS = {
    "two_minute": {16, 17, 18, 19, 20, 31, 32, 33, 34, 35, 36, 56, 57, 59, 60},  # 21 is at 2:01
    "two_minute_half": {16, 17, 18, 19, 20},
    "two_minute_game": {31, 32, 33, 34, 35, 36, 56, 57, 59, 60},  # 58 is OT 10:00
    "goal_to_go": {26, 27, 43, 44, 45, 46},
}
SITUATIONS_TODAY = {
    "red_zone": {14, 18, 19, 20, 26, 27, 28, 39, 43, 44, 45, 46, 54},  # yardline_100 <= 20
    "third_down": {9, 20, 21, 33, 45, 51, 56},
    "fourth_down": {8, 34, 46, 52, 54},
}


@pytest.mark.parametrize("situation", list(SITUATIONS))
def test_situation_new_definitions(plays_db, situation):
    assert _pids(_search(situation=situation)) == SITUATIONS[situation]


@pytest.mark.parametrize("situation", list(SITUATIONS_TODAY))
def test_situation_existing_still_work(plays_db, situation):
    assert _pids(_search(situation=situation)) == SITUATIONS_TODAY[situation]


def test_situation_unknown_is_error(plays_db):  # satisfied by group H's argument checks
    assert "error" in _search(situation="garbage_time")


# ── 6. numeric and flag filters ──────────────────────────────────────────────────

FILTER_CASES = [
    ({"down": 3}, {9, 20, 21, 33, 45, 51, 56}),
    ({"down": 4}, {8, 34, 46, 52, 54}),
    ({"qtr": 5}, {58, 59, 60}),
    ({"qtr": 3}, {22, 23, 24, 25, 26, 27, 28, 29, 51, 52, 53}),
    ({"max_yardline_100": 5}, {27, 43, 44, 45, 46}),
    ({"min_yardline_100": 74, "max_yardline_100": 75}, {2, 21, 23, 38, 41, 55}),
    ({"min_score_differential": 4}, {35, 36, 54}),
    ({"max_score_differential": -8}, {55, 56, 57}),
    ({"min_score_differential": -4, "max_score_differential": -4}, {31, 32, 33, 34}),
    ({"no_huddle": True}, {4, 16, 17, 18, 31, 56, 59}),
    ({"shotgun": True, "qtr": 4}, {30, 31, 32, 55, 56}),
    ({"shotgun": False, "qtr": 4}, {33, 34, 35, 36, 54, 57}),
    ({"min_wp": 0.8}, {35, 36, 54, 56}),  # 56 sits exactly on 0.8
    ({"max_wp": 0.2}, {31, 32, 33}),
    ({"min_wp": 0.55, "max_wp": 0.9}, {54, 56, 59}),
    ({"qtr": 2, "shotgun": True, "no_huddle": True}, {16, 17, 18}),
    ({"down": 4, "max_yardline_100": 20}, {46, 54}),
]


@pytest.mark.parametrize("kw, expected", FILTER_CASES, ids=[str(k) for k, _ in FILTER_CASES])
def test_filter_narrows_to_hand_counted_rows(plays_db, kw, expected):
    assert _pids(_search(**kw)) == expected


# ── 7. sort_by ───────────────────────────────────────────────────────────────────

SORTS = {"abs_epa": [38, 25], "epa": [38, 49], "wpa": [59, 38], "yards_gained": [38, 41],
         "game_order": [1, 2]}


def test_default_order_is_abs_epa_desc(plays_db):
    rows = _search()["rows"]
    assert [_pid(r) for r in rows[:2]] == [38, 25]  # |6.5|, |-5.2|; every other |epa| <= 0.30


@pytest.mark.parametrize("sort_by", list(SORTS))
def test_sort_by_first_two_rows(plays_db, sort_by):
    res = _search(sort_by=sort_by)
    assert [_pid(r) for r in res["rows"][:2]] == SORTS[sort_by]


@pytest.mark.parametrize("bad", ["random", "epa; DROP TABLE plays"])
def test_sort_by_unknown_is_error(plays_db, bad):
    res = _search(sort_by=bad)
    assert "error" in res and "rows" not in res


# ── 8. columns ───────────────────────────────────────────────────────────────────

def test_columns_extra_appear_in_rows(plays_db):
    res = _search(play_type="pass", columns=["air_yards", "cpoe"])
    row = next(r for r in res["rows"] if _pid(r) == 38)
    assert (row["air_yards"], row["cpoe"]) == (40.0, 35.5)
    assert all("air_yards" in r and "cpoe" in r for r in res["rows"])


@pytest.mark.parametrize("cols, suggestion", [
    (["air_yard"], "air_yards"), (["cpoe", "cpoee"], "cpoe"), (['epa" FROM plays --'], None)])
def test_columns_unknown_errors_with_did_you_mean(plays_db, cols, suggestion):
    res = _search(columns=cols)
    assert "error" in res and "rows" not in res
    assert isinstance(res.get("did_you_mean"), list)
    if suggestion:
        assert suggestion in res["did_you_mean"]


# ── 9. default output ────────────────────────────────────────────────────────────

DEFAULT_KEYS = {"game_id", "play_id", "passer_player_id", "rusher_player_id",
                "receiver_player_id", "posteam", "defteam", "season", "week", "qtr", "down",
                "play_type", "desc", "epa", "wpa"}


@pytest.mark.parametrize("ptype", ["pass", "kickoff"])
def test_default_output_has_id_and_context_keys(plays_db, ptype):
    rows = _search(play_type=ptype)["rows"]
    assert rows and all(DEFAULT_KEYS <= set(r) for r in rows)


def test_default_output_ids_match_fixture(plays_db):
    row = next(r for r in _search(play_type="pass")["rows"] if _pid(r) == 49)
    assert (row["game_id"], row["play_id"], row["passer_player_id"]) == ("2024_02_KC_CIN", 490.0, X1)


# ── 10. max_rows alias ───────────────────────────────────────────────────────────

def test_max_rows_still_limits_rows(plays_db):
    assert len(_search(max_rows=3)["rows"]) == 3


@pytest.mark.parametrize("n, truncated", [(3, True), (60, False)])
def test_max_rows_alias_reports_truncation(plays_db, n, truncated):
    res = _search(max_rows=n)
    assert (res["count"], res["truncated"]) == (n, truncated)


# ── 1-2. pagination on every list tool ───────────────────────────────────────────

def _cols(spec):
    """'a b:I c:D' -> {a: VARCHAR, b: INTEGER, c: DOUBLE}."""
    types = {"I": "INTEGER", "D": "DOUBLE", "B": "BOOLEAN", "": "VARCHAR"}
    return {t.split(":")[0]: types[(t.split(":") + [""])[1]] for t in spec.split()}


SCRAMBLE = [3, 0, 6, 1, 5, 2, 4]  # insertion order != sort order
LETTERS = "ABCDEFG"
TEAMS7 = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN"]
# (tool, table, columns, row(i, letter), call kwargs, list key, id key). Sort keys are distinct and
# chosen so each tool's documented ORDER BY yields A..G.
PAGE_SPECS = [
    ("nfl_roster", "rosters", "season:I team position depth_chart_position jersey_number:I status "
     "full_name years_exp:I college height:D weight:I gsis_id pfr_id week:I game_type",
     lambda i, c: dict(season=2024, team="KC", position="WR", full_name=f"Roster {c}",
                       gsis_id=f"00-00R000{i}"), {"team": "KC", "season": 2024}, "players", "full_name"),
    ("nfl_injuries", "injuries", "season:I game_type team week:I gsis_id position full_name "
     "report_primary_injury report_secondary_injury report_status practice_primary_injury "
     "practice_secondary_injury practice_status season_type",
     lambda i, c: dict(season=2024, game_type="REG", season_type="REG", team="KC", week=1,
                       full_name=f"Inj {c}", report_status="Out"),
     {"season": 2024}, "injuries", "full_name"),
    ("nfl_schedule", "schedules", "game_id season:I game_type week:I gameday gametime away_team "
     "home_team away_score:I home_score:I result:I total:I overtime:I spread_line:D total_line:D "
     "div_game:I roof surface temp:I wind:I away_coach home_coach referee stadium location",
     lambda i, c: dict(game_id=f"2024_{i + 1:02d}_BUF_KC", season=2024, game_type="REG",
                       week=i + 1, away_team="BUF", home_team="KC"),
     {"season": 2024}, "games", "game_id"),
    ("nfl_snap_counts", "snap_counts", "game_id pfr_game_id season:I game_type week:I player "
     "pfr_player_id position team opponent offense_snaps:D offense_pct:D defense_snaps:D "
     "defense_pct:D st_snaps:D st_pct:D",
     lambda i, c: dict(game_id="2024_01_BAL_KC", season=2024, game_type="REG", week=1,
                       player=f"Snap {c}", pfr_player_id=f"SnapX{i}", team="KC", opponent="BAL",
                       offense_snaps=70.0 - i), {"season": 2024}, "snap_counts", "player"),
    ("nfl_fantasy_opportunity", "ff_opportunity", "season:I posteam week:D game_id player_id "
     "full_name position pass_attempt:D rec_attempt:D rush_attempt:D rec_air_yards:D receptions:D "
     "receptions_exp:D rec_yards_gained:D rush_yards_gained:D rec_touchdown:D rush_touchdown:D "
     "total_fantasy_points:D total_fantasy_points_exp:D total_fantasy_points_diff:D "
     "rec_attempt_team:D rush_attempt_team:D",
     lambda i, c: dict(season=2024, posteam="KC", week=1.0, player_id=f"00-00F000{i}",
                       full_name=f"Opp {c}", total_fantasy_points_exp=30.0 - i),
     {"season": 2024}, "fantasy_opportunity", "full_name"),
    ("nfl_fantasy_rankings", "ff_rankings_draft", "fp_page page_type ecr_type player id pos team "
     "ecr:D sd:D best:I worst:I bye:I scrape_date sportsdata_id tm",
     lambda i, c: dict(page_type="redraft-overall", player=f"Rank {c}", pos="WR", team="KC",
                       ecr=float(i + 1)), {}, "fantasy_rankings", "player"),
    ("nfl_search_plays", "plays", None,
     lambda i, c: dict(game_id="2024_01_BAL_KC", play_id=float(i), season=2024, week=1,
                       season_type="REG", posteam="KC", defteam="BAL", play_type="pass",
                       desc=f"Play {c}", epa=(7.0 - i) * (1 if i % 2 == 0 else -1)),
     {"season": 2024}, "rows", "desc"),
    ("nfl_td_luck", "player_td_luck", "player_id full_name position team season:I rec_td:D "
     "rec_td_exp:D rec_td_luck:D rush_td:D rush_td_exp:D rush_td_luck:D total_td_luck_score:D",
     lambda i, c: dict(player_id=f"00-00L000{i}", full_name=f"Luck {c}", team="KC", season=2024,
                       total_td_luck_score=-3.0 + i), {"season": 2024}, "td_luck", "full_name"),
    ("nfl_role_trend", "player_role_trend", "player_id full_name position team season:I week:I "
     "snap_pct:D target_share_pct:D carry_share_pct:D air_yards_share_pct:D snap_pct_3wk:D "
     "target_share_pct_3wk:D carry_share_pct_3wk:D air_yards_share_pct_3wk:D snap_pct_delta:D "
     "target_share_pct_delta:D carry_share_pct_delta:D air_yards_share_pct_delta:D",
     lambda i, c: dict(player_id=f"00-00T000{i}", full_name=f"Trend {c}", team="KC", season=2024,
                       week=5, snap_pct=50.0, snap_pct_delta=10.0 - i),
     {"season": 2024}, "role_trend", "full_name"),
    ("nfl_separation_opportunity", "player_separation_opportunity", "player_id full_name position "
     "team season:I games:I fp_diff:D fp_diff_per_game:D td_luck:D catch_luck:D target_share_pct:D "
     "air_yards_share_pct:D avg_separation:D avg_yac_above_expectation:D catch_percentage:D "
     "regression_candidate:B",
     lambda i, c: dict(player_id=f"00-00S000{i}", full_name=f"Sep {c}", team="KC", season=2024,
                       fp_diff_per_game=-5.0 + i, regression_candidate=False),
     {"season": 2024}, "separation_opportunity", "full_name"),
    ("nfl_drop_rate", "player_drop_rate", "player_id player team season:I catchable_targets:I "
     "drops:I drop_rate_pct:D contested_targets:I created_receptions:I",
     lambda i, c: dict(player_id=f"00-00D000{i}", player=f"D.{c}", team="KC", season=2024,
                       catchable_targets=50, drop_rate_pct=20.0 - i),
     {"season": 2024}, "drop_rate", "player"),
    ("nfl_contract_value", "player_contract_value", "player_id full_name position team season:I "
     "apy:D cap_pct:D total_fp:D total_fp_exp:D fp_per_million:D",
     lambda i, c: dict(player_id=f"00-00V000{i}", full_name=f"Val {c}", team="KC", season=2024,
                       apy=5.0, fp_per_million=50.0 - i),
     {"season": 2024}, "contract_value", "full_name"),
    ("nfl_injury_return", "injury_return_curve", "injury_type position week_post_return:I "
     "avg_snap_pct_recovery:D median_snap_pct_recovery:D sample_size:I",
     lambda i, c: dict(injury_type=f"Inj {c}", position="WR", week_post_return=1,
                       avg_snap_pct_recovery=90.0, sample_size=5),
     {}, "injury_return", "injury_type"),
    ("nfl_team_records", "team_records", "season:I team source_team games:I wins:I losses:I "
     "ties:I source",
     lambda i, c: dict(season=2024, team=TEAMS7[i], source_team=TEAMS7[i], games=17, wins=9,
                       losses=8, ties=0, source="nflverse"),
     {"season": 2024}, "records", "team"),
]
SPEC_IDS = [s[0] for s in PAGE_SPECS]


@pytest.fixture(scope="module")
def _page_file(tmp_path_factory):
    """Every list table with exactly 7 matching rows; rosters and plays also get 120 rows for 2023."""
    path = tmp_path_factory.mktemp("grp_i") / "pages.duckdb"
    con = duckdb.connect(str(path))
    for _tool, table, cols, make, *_ in PAGE_SPECS:
        rows = [make(i, LETTERS[i]) for i in SCRAMBLE]
        if table == "rosters":
            rows += [dict(season=2023, team="KC", position="WR", full_name=f"Old {n:03d}")
                     for n in range(120)]
        if table == "plays":
            rows += [dict(game_id="2023_01_BAL_KC", play_id=float(n), season=2023, week=1,
                          posteam="KC", defteam="BAL", play_type="run", desc=f"Old {n:03d}",
                          epa=n / 1000) for n in range(120)]
        _table(con, table, PLAY_COLS if cols is None else _cols(cols), rows)
    _table(con, "ff_rankings_week", _cols("player_name pos team rank:I ecr:D page scrape_date"), [])
    _meta(con)
    con.close()
    return path


@pytest.fixture
def page_db(_page_file, monkeypatch):
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(_page_file))


def _page(spec, **kw):
    fn = getattr(tools, spec[0], None)
    assert fn is not None, f"{spec[0]} missing"
    res = fn(**spec[4], **kw)
    assert "error" not in res, res
    return res, [r[spec[6]] for r in res[spec[5]]]


ORDER = {s[0]: [s[3](i, LETTERS[i])[s[6]] for i in range(7)] for s in PAGE_SPECS}

@pytest.mark.parametrize("spec", PAGE_SPECS, ids=SPEC_IDS)
def test_limit_three_returns_first_page_truncated(page_db, spec):
    res, ids = _page(spec, limit=3)
    assert ids == ORDER[spec[0]][:3]
    assert (res["count"], res["truncated"]) == (3, True)
    assert isinstance(res["hint"], str) and res["hint"]


@pytest.mark.parametrize("spec", PAGE_SPECS, ids=SPEC_IDS)
def test_offset_pages_follow_order(page_db, spec):
    res2, ids2 = _page(spec, limit=3, offset=3)
    res3, ids3 = _page(spec, limit=3, offset=6)
    assert (ids2, res2["count"], res2["truncated"]) == (ORDER[spec[0]][3:6], 3, True)
    assert (ids3, res3["count"], res3["truncated"]) == (ORDER[spec[0]][6:], 1, False)
    assert "hint" not in res3


@pytest.mark.parametrize("spec", PAGE_SPECS, ids=SPEC_IDS)
def test_limit_equal_to_row_count_not_truncated(page_db, spec):
    res, ids = _page(spec, limit=7)
    assert (len(ids), res["count"], res["truncated"]) == (7, 7, False)
    assert "hint" not in res


@pytest.mark.parametrize("spec", PAGE_SPECS, ids=SPEC_IDS)
def test_pages_disjoint_and_cover_all_rows(page_db, spec):
    pages = [_page(spec, limit=3, offset=o)[1] for o in (0, 3, 6)]
    flat = [x for p in pages for x in p]
    assert len(flat) == len(set(flat)) == 7
    assert set(flat) == set(ORDER[spec[0]])


@pytest.mark.parametrize("spec", PAGE_SPECS, ids=SPEC_IDS)
def test_hint_mentions_offset_and_nfl_query(page_db, spec):
    hint = _page(spec, limit=2, offset=1)[0]["hint"]
    assert "offset" in hint and "nfl_query" in hint


@pytest.mark.parametrize("bad", [0, -1, 501, "5"])
@pytest.mark.parametrize("spec", PAGE_SPECS, ids=SPEC_IDS)
def test_bad_limit_returns_error(page_db, spec, bad):
    fn = getattr(tools, spec[0])
    res = fn(**spec[4], limit=bad)
    assert set(res) >= {"error"} and spec[5] not in res


@pytest.mark.parametrize("tool, list_key", [("nfl_roster", "players"), ("nfl_search_plays", "rows")])
def test_default_limit_is_100(page_db, tool, list_key):
    res = getattr(tools, tool)(season=2023)  # 120 matching rows in the fixture
    assert (len(res[list_key]), res["count"], res["truncated"]) == (100, 100, True)


# ── Part 2: integration against the real DB ──────────────────────────────────────

def _real(sql, params=None):
    from nfl_mcp.config import get_duckdb_path
    path = get_duckdb_path()
    if not path.exists():
        pytest.skip("No DuckDB database loaded")
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(sql, params or []).fetchall()
    finally:
        con.close()


@pytest.fixture
def require_db():
    if _real("SELECT COUNT(*) FROM plays")[0][0] == 0:
        pytest.skip("plays table is empty")


integration = [pytest.mark.integration, pytest.mark.usefixtures("require_db")]


def _mark(*marks):
    def wrap(fn):
        for m in reversed(marks):
            fn = m(fn)
        return fn
    return wrap


@_mark(*integration)
def test_real_snap_counts_kc_2024_first_page_truncated():
    direct = _real("SELECT COUNT(*) FROM snap_counts WHERE team='KC' AND season=2024")[0][0]
    assert direct > 500  # 933 in grpE_v3 (all game types)
    res = tools.nfl_snap_counts(team="KC", season=2024, limit=500)
    assert (res["count"], res["truncated"]) == (500, True)


@_mark(*integration)
def test_real_snap_counts_kc_2024_two_pages_cover_every_row_once():
    direct = _real("SELECT COUNT(DISTINCT (week, pfr_player_id)) , COUNT(*) FROM snap_counts "
                   "WHERE team='KC' AND season=2024")[0]
    assert direct[0] == direct[1]  # (week, pfr_player_id) is unique here
    p1 = tools.nfl_snap_counts(team="KC", season=2024, limit=500)
    p2 = tools.nfl_snap_counts(team="KC", season=2024, limit=500, offset=500)
    assert (p2["count"], p2["truncated"]) == (direct[1] - 500, False)
    keys = [(r["week"], r["pfr_player_id"]) for r in p1["snap_counts"] + p2["snap_counts"]]
    assert len(keys) == len(set(keys)) == direct[1]


@_mark(*integration)
@pytest.mark.parametrize("team", [None, "KC"])
def test_real_roster_2024_truncation_matches_direct_count(team):
    sql = "SELECT COUNT(*) FROM rosters WHERE season=2024" + (" AND team=?" if team else "")
    direct = _real(sql, [team] if team else None)[0][0]
    kw = {"team": team} if team else {}
    res = tools.nfl_roster(season=2024, limit=500, **kw)
    assert res["count"] == min(direct, 500)
    assert res["truncated"] is (direct > 500)  # all teams: 3,216 > 500; KC alone (99) fits


@_mark(*integration)
@pytest.mark.parametrize("season_type", [None, "REG", "POST"])
def test_real_butker_kicker_field_goals_match_player_stats(season_type):
    gsis = _real("SELECT gsis_id FROM players WHERE display_name='Harrison Butker'")[0][0]
    st = " AND season_type=?" if season_type else ""
    extra = [season_type] if season_type else []
    # oracle c: field_goal rows with his kicker id; oracle b: official player_stats fg_att
    direct = _real("SELECT COUNT(*) FROM plays WHERE season=2024 AND play_type='field_goal' "
                   "AND kicker_player_id=?" + st, [gsis] + extra)[0][0]
    official = _real("SELECT SUM(fg_att) FROM player_stats WHERE season=2024 AND player_id=?" + st,
                     [gsis] + extra)[0][0]
    assert direct == official > 0
    kw = {"season_type": season_type} if season_type else {}
    res = tools.nfl_search_plays(season=2024, player_role="kicker", player_id=gsis,
                                 play_type="field_goal", limit=500, **kw)
    assert res["count"] == direct
    assert all(r["play_type"] == "field_goal" for r in res["rows"])


@_mark(*integration)
def test_real_mahomes_passer_week1_matches_direct_count():
    gsis = _real("SELECT gsis_id FROM players WHERE display_name='Patrick Mahomes'")[0][0]
    direct = _real("SELECT COUNT(*) FROM plays WHERE season=2024 AND week=1 "
                   "AND passer_player_id=?", [gsis])[0][0]
    res = tools.nfl_search_plays(season=2024, week=1, player_role="passer", player_id=gsis,
                                 limit=500)
    assert res["count"] == direct > 0
    assert all(r["passer_player_id"] == gsis for r in res["rows"])


@_mark(*integration)
def test_real_schema_enum_equals_distinct_play_types():
    real = {r[0] for r in _real("SELECT DISTINCT play_type FROM plays WHERE play_type IS NOT NULL")}
    assert real == REAL_PLAY_TYPES
    assert _schema_play_type_enum() == real


@_mark(*integration)
@pytest.mark.parametrize("ptype", sorted(REAL_PLAY_TYPES))
def test_real_play_type_accepted_and_exclusive(ptype):
    res = tools.nfl_search_plays(season=2024, week=1, play_type=ptype, max_rows=5)
    assert "error" not in res and res["rows"]
    assert {r["play_type"] for r in res["rows"]} == {ptype}
