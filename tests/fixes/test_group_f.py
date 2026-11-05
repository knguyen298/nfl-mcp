"""Group F (derived tables) tests: pin the behavior of the fixed aggregate builders.

Spec: PLAN_FIX_GROUPS_MAIN_2026-11-04.md section F; evidence: AUDIT_MAIN_2026-11-04.md
(ingest.py:534-632 team/situational/formation builders, _FANTASY_TABLES, _create_team_records).

Interface contract (beyond the shared brief)
-------------------------------------------
* nfl_mcp.ingest.rebuild_derived_tables(conn) rebuilds ALL derived tables on an open read-write
  connection: team_offense_stats, team_defense_stats, situational_stats, formation_effectiveness,
  player_td_luck, player_role_trend, player_separation_opportunity, player_drop_rate,
  player_contract_value, injury_return_curve, team_records. It runs in ONE transaction: if any
  statement fails it rolls back (previous tables stay intact) and re-raises. It issues SQL through
  conn.execute (the failure test wraps conn in a proxy). Until it exists the unit tests fall back
  to today's _create_aggregate_tables + _create_fantasy_tables + _create_team_records.
* _ingest_metadata gets exactly one row per derived table: dataset_id = table_name = <table>,
  season NULL, loader_fn 'derived'.
* team_offense_stats: one row per (team, season_year, season_type), season_type in {REG, POST}.
  Scrimmage play = play_type IN ('pass','run') AND two_point_attempt = 0 (no kickoff, punt,
  field_goal, extra_point, no_play, qb_kneel, qb_spike). Columns:
    total_plays   = scrimmage plays (sacks and scrambles included, kneels/spikes excluded)
    pass_plays    = pass_attempt=1, sack=0, two-pt=0, play_type='pass' (spikes EXCLUDED)
    pass_attempts = official nflverse attempts: pass_attempt=1, sack=0, two-pt=0, not no_play
                    (spikes INCLUDED; this is what team_stats_raw.attempts counts)
    completions, interceptions (thrown), sacks, dropbacks (qb_dropback=1, two-pt=0, not no_play)
    pass_yards    = SUM(passing_yards) over pass_attempts (sack yardage excluded)
    rush_plays    = rush_attempt=1, two-pt=0, not no_play (kneels and scrambles INCLUDED, as nflverse)
    scrambles     = qb_scramble=1 subset of rush_plays; rush_yards = SUM(rushing_yards) over rush_plays
    touchdowns    = offensive TDs: touchdown=1 AND td_team=posteam on a scrimmage play
    turnovers     = interceptions thrown + fumbles lost where fumbled_1_team = team (any play type,
                    so a muffed punt is the receiving team's turnover)
    third_down_pct = 100 * SUM(first_down=1) / scrimmage plays with down=3 (no no_play)
* team_defense_stats: one row per (team, season_year, season_type); sacks, interceptions,
  turnovers_forced (excludes the opponent's muffed punt on its own punt), def_touchdowns
  (touchdown=1 AND td_team=defteam, e.g. a pick-six).
* situational_stats: one row per (team, season_year, season_type, situation); only scrimmage run/pass
  plays (no punts, FG, kneels, spikes, no_play, two-pt). conversion_pct = first_down=1 share for the
  down buckets. New column conversion_rule (VARCHAR, non-null on every row) states the rule.
* player_td_luck, player_drop_rate, formation_effectiveness: gain season_type (REG/POST); for
  ff_opportunity-based tables POST is taken from schedules.game_type (or week > 18; both agree here).
  formation_effectiveness excludes two-point tries.
* player_contract_value: contract in force = latest year_signed <= season (ties: max apy).
* player_role_trend, injury_return_curve: snap_counts joined via players.gsis_id -> players.pfr_id ->
  snap_counts.pfr_player_id, not by name.
* nfl_team_stats(..., season_type="REG") filters by season_type; nfl_compare(compare_type="team",
  season_type=...) filters too. A team with no rows for the filter gets
  {"offense": [], "defense": [], "reason": str} under its key.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import duckdb
import pytest

from nfl_mcp import ingest, tools
from nfl_mcp.database import get_db_connection

REAL_DB = Path(os.environ.get("NFL_MCP_DB_PATH", "/tmp/grpE_v3.duckdb"))
XF = "group f: "

DERIVED = [
    "team_offense_stats", "team_defense_stats", "situational_stats", "formation_effectiveness",
    "player_td_luck", "player_role_trend", "player_separation_opportunity", "player_drop_rate",
    "player_contract_value", "injury_return_curve", "team_records",
]

# ── synthetic plays ──────────────────────────────────────────────────────────

_FLAG_COLS = [
    "pass_attempt", "rush_attempt", "complete_pass", "incomplete_pass", "sack", "qb_spike",
    "qb_kneel", "qb_scramble", "qb_dropback", "two_point_attempt", "touchdown", "pass_touchdown",
    "rush_touchdown", "return_touchdown", "interception", "fumble", "fumble_lost", "first_down",
    "penalty", "shotgun", "no_huddle", "special_teams_play", "pass", "rush",
    "third_down_converted", "third_down_failed", "fourth_down_converted", "fourth_down_failed",
]
_NUM_COLS = ["play_id", "down", "ydstogo", "yardline_100", "qtr", "yards_gained", "passing_yards",
             "rushing_yards", "receiving_yards", "epa"]
_STR_COLS = ["game_id", "season_type", "game_type", "posteam", "defteam", "play_type", "time",
             "td_team", "fumbled_1_team", "fumble_recovery_1_team", "receiver_player_id",
             "receiver_player_name", "passer_player_id", "passer_player_name",
             "rusher_player_id", "rusher_player_name"]
_INT_COLS = ["season", "week"]
PLAY_COLS = _STR_COLS + _INT_COLS + _NUM_COLS + _FLAG_COLS

REG_GAME, POST_GAME = "2024_15_KC_CLE", "2024_20_KC_HOU"


def play(pid, pt, d, togo, yl, yds=0, *, pos="KC", de="CLE", post=False, qtr=1, time="10:00", **kw):
    """One plays row with nflverse-style flags derived from play_type."""
    r = {c: 0.0 for c in _FLAG_COLS}
    r.update(game_id=POST_GAME if post else REG_GAME, play_id=float(pid), season=2024,
             week=20 if post else 15, season_type="POST" if post else "REG",
             game_type="DIV" if post else "REG", posteam=pos, defteam=de, play_type=pt,
             down=d, ydstogo=togo, yardline_100=yl, qtr=float(qtr), time=time,
             yards_gained=float(yds), epa=0.1)
    if pt == "pass":
        r.update(pass_attempt=1.0, qb_dropback=1.0, shotgun=1.0, **{"pass": 1.0},
                 passer_player_id=f"{pos}-QB", passer_player_name=f"{pos}.QB")
        if kw.pop("cmp", False):
            r.update(complete_pass=1.0, passing_yards=float(yds), receiving_yards=float(yds))
        elif not kw.get("sack") and not kw.get("interception"):
            r["incomplete_pass"] = 1.0
        if kw.get("sack"):
            r.update(passer_player_id=f"{pos}-QB")
    elif pt in ("run", "qb_kneel"):
        r.update(rush_attempt=1.0, rush=1.0, rushing_yards=float(yds), rusher_player_id=f"{pos}-RB")
        if pt == "qb_kneel":
            r["qb_kneel"] = 1.0
        if kw.get("qb_scramble"):
            r.update(qb_dropback=1.0, **{"pass": 1.0}, rush=0.0)
    elif pt == "qb_spike":
        r.update(pass_attempt=1.0, qb_spike=1.0)
    elif pt in ("punt", "kickoff", "field_goal", "extra_point"):
        r["special_teams_play"] = 1.0
    if d == 3 and pt in ("pass", "run"):
        conv = kw.get("first_down", 0.0)
        r["third_down_converted"], r["third_down_failed"] = conv, 1.0 - conv
    if d == 4 and pt in ("pass", "run"):
        conv = kw.get("first_down", 0.0)
        r["fourth_down_converted"], r["fourth_down_failed"] = conv, 1.0 - conv
    r.update({k: float(v) if k in _FLAG_COLS else v for k, v in kw.items()})
    return r


WR1 = dict(receiver_player_id="00-WR1", receiver_player_name="W.Receiver")
TD_KC = dict(touchdown=1.0, td_team="KC", first_down=1.0)


def base_plays():
    """KC 2024 REG (44 rows) + CLE REG offense (4) + KC POST (4) + HOU POST (1)."""
    P = play
    return [
        P(1, "pass", 1, 10, 75, 8, cmp=1, **WR1),
        P(2, "pass", 2, 2, 67, 0, **WR1),                       # incomplete; FTN drop
        P(3, "run", 3, 2, 67, 3, first_down=1),                 # 3rd & short converted
        P(4, "pass", 1, 10, 64, 15, cmp=1, first_down=1, **WR1),
        P(5, "pass", 1, 10, 49, 0),
        P(6, "pass", 2, 10, 49, -7, sack=1),                    # sack 1
        P(7, "pass", 3, 17, 56, 12, cmp=1),                     # 3rd & long, not converted
        P(8, "punt", 4, 5, 44, 0),
        P(9, "run", 1, 10, 80, 4, qtr=2),
        P(10, "run", 2, 6, 76, 9, qtr=2, qb_scramble=1, first_down=1),   # scramble 1
        P(11, "pass", 1, 10, 67, 25, qtr=2, cmp=1, first_down=1, **WR1),
        P(12, "no_play", 1, 10, 42, 0, qtr=2, penalty=1),
        P(13, "pass", 1, 15, 47, 0, qtr=2),
        P(14, "run", 2, 15, 47, 0, qtr=2),
        P(15, "no_play", 3, 21, 53, 0, qtr=2, penalty=1, qb_dropback=1),  # 3rd & long no_play
        P(16, "pass", 3, 21, 53, 22, qtr=2, cmp=1, first_down=1),         # 3rd & long converted
        P(17, "run", 1, 10, 31, 6, qtr=2),
        P(18, "pass", 2, 4, 25, 7, qtr=2, cmp=1, first_down=1),
        P(19, "run", 1, 10, 18, 5, qtr=2),
        P(20, "pass", 2, 5, 13, 0, qtr=2),
        P(21, "pass", 3, 5, 13, 13, qtr=2, cmp=1, pass_touchdown=1, **TD_KC),  # offensive TD 1
        P(22, "extra_point", None, None, 15, 0, qtr=2),
        P(23, "kickoff", None, None, 65, 25),                   # KC receives (nflverse posteam)
        P(24, "run", 1, 10, 70, 2, qtr=3),
        P(25, "pass", 2, 8, 68, 0, qtr=3, interception=1, touchdown=1, return_touchdown=1,
          td_team="CLE"),                                       # pick-six thrown
        P(26, "run", 1, 10, 60, 12, qtr=3, first_down=1),
        P(27, "pass", 3, 8, 48, -8, qtr=3, sack=1),             # sack 2, 3rd & long
        P(28, "punt", 4, 16, 56, 0, qtr=3, fumble=1, fumble_lost=1, fumbled_1_team="CLE",
          fumble_recovery_1_team="KC"),                         # CLE muffs KC's punt
        P(29, "run", 1, 10, 20, 4, qtr=3),
        P(30, "run", 2, 6, 16, 2, qtr=3),
        P(31, "pass", 3, 4, 14, 0, qtr=3),
        P(32, "field_goal", 4, 4, 14, 0, qtr=3),
        P(33, "run", 1, 10, 75, 3, qtr=4, time="10:00"),
        P(34, "run", 2, 7, 72, 2, qtr=4, time="09:30"),
        P(35, "pass", 3, 5, 70, 0, qtr=4, time="09:00"),
        P(36, "pass", 4, 5, 70, 9, qtr=4, time="08:30", cmp=1, first_down=1),  # 4th converted
        P(37, "run", 4, 1, 30, 0, qtr=4, time="06:00"),                         # 4th failed
        P(38, "run", 1, 1, 1, 1, qtr=4, time="05:00", rush_touchdown=1, **TD_KC),  # offensive TD 2
        P(39, "pass", None, None, 2, 2, qtr=4, time="05:00", two_point_attempt=1,
          shotgun=0, no_huddle=1, first_down=0),                # two-point pass try
        P(40, "pass", 1, 10, 52, -5, qtr=4, time="01:55", sack=1),               # sack 3
        P(41, "run", 3, 6, 45, 7, qtr=4, time="01:40", qb_scramble=1, first_down=1),  # scramble 2
        P(42, "qb_spike", 2, 10, 38, 0, qtr=4, time="01:10"),
        P(43, "qb_kneel", 1, 10, 35, -1, qtr=4, time="00:40"),
        P(44, "qb_kneel", 2, 11, 36, -1, qtr=4, time="00:05"),
        # CLE offense vs KC defense
        P(101, "pass", 1, 10, 75, 6, pos="CLE", de="KC", cmp=1),
        P(102, "pass", 2, 4, 69, -8, pos="CLE", de="KC", sack=1),
        P(103, "pass", 3, 12, 77, 0, pos="CLE", de="KC", interception=1),
        P(104, "extra_point", None, None, 15, 0, pos="CLE", de="KC", qtr=3),
        # KC POST (4 rows) and HOU POST offense (1 row)
        P(201, "pass", 1, 10, 75, 11, post=True, de="HOU", cmp=1, first_down=1, **WR1),
        P(202, "run", 1, 10, 64, 5, post=True, de="HOU"),
        P(203, "pass", 2, 5, 59, 0, post=True, de="HOU", **WR1),           # FTN drop
        P(204, "pass", 3, 5, 59, 59, post=True, de="HOU", cmp=1, pass_touchdown=1, **TD_KC),
        P(301, "pass", 1, 10, 75, 0, post=True, pos="HOU", de="KC"),
    ]


# ── synthetic DB ─────────────────────────────────────────────────────────────

def _insert(conn, table, cols_types, rows):
    conn.execute(f"CREATE TABLE {table} ({', '.join(f'{c} {t}' for c, t in cols_types)})")
    names = [c for c, _ in cols_types]
    conn.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * len(names))})",
                     [[r.get(c) for c in names] for r in rows])


def _ptype(c):
    return "VARCHAR" if c in _STR_COLS else "INTEGER" if c in _INT_COLS else "DOUBLE"


def _ff(pid, name, season, week, game_id, team="KC", pos="RB", rush_td=0.0, rec_td=0.0, fp=10.0):
    return dict(season=str(season), posteam=team, week=float(week), game_id=game_id, player_id=pid,
                full_name=name, position=pos, pass_attempt=0.0, rec_attempt=3.0, rush_attempt=5.0,
                rec_air_yards=20.0, rec_attempt_team=30.0, rush_attempt_team=25.0,
                rec_air_yards_team=200.0, rec_touchdown=rec_td, rec_touchdown_exp=0.4,
                rec_touchdown_diff=rec_td - 0.4, rush_touchdown=rush_td, rush_touchdown_exp=0.5,
                rush_touchdown_diff=rush_td - 0.5, receptions_diff=0.2, total_fantasy_points=fp,
                total_fantasy_points_exp=9.0, total_fantasy_points_diff=fp - 9.0)


def _snap(name, pfr, week, pct, season=2024, team="KC"):
    return dict(game_id=f"{season}_{week:02d}_X", pfr_game_id="x", season=season, game_type="REG",
                week=week, player=name, pfr_player_id=pfr, position="WR", team=team, opponent="CLE",
                offense_snaps=pct * 60, offense_pct=pct, defense_snaps=0.0, defense_pct=0.0,
                st_snaps=0.0, st_pct=0.0)


def _inj(gid, name, week, injury):
    return dict(season=2024, game_type="REG", season_type="REG", team="KC", week=week, gsis_id=gid,
                position="WR", full_name=name, first_name=name.split()[0], last_name=name.split()[-1],
                report_primary_injury=injury, report_status="Out", practice_primary_injury=injury,
                practice_status="DNP")


def build_db(path: Path, plays=None, extra_plays=()):
    conn = duckdb.connect(str(path))
    _insert(conn, "plays", [(c, _ptype(c)) for c in PLAY_COLS], list(plays or base_plays()) + list(extra_plays))
    sched = [  # (game_id, season, game_type, week, away, home, away_score, home_score)
        (REG_GAME, 2024, "REG", 15, "KC", "CLE", 21, 7), ("2024_16_KC_DEN", 2024, "REG", 16, "KC", "DEN", 17, 20),
        (POST_GAME, 2024, "DIV", 20, "HOU", "KC", 14, 23), ("2022_01_KC_ARI", 2022, "REG", 1, "KC", "ARI", 44, 21),
        ("2025_01_KC_LAC", 2025, "REG", 1, "KC", "LAC", 21, 27)]
    _insert(conn, "schedules", [("game_id", "VARCHAR"), ("season", "INTEGER"), ("game_type", "VARCHAR"),
            ("week", "INTEGER"), ("away_team", "VARCHAR"), ("home_team", "VARCHAR"),
            ("away_score", "INTEGER"), ("home_score", "INTEGER")],
            [dict(zip(["game_id", "season", "game_type", "week", "away_team", "home_team", "away_score",
                       "home_score"], s)) for s in sched])
    _insert(conn, "players", [(c, "VARCHAR") for c in
            ["gsis_id", "display_name", "short_name", "pfr_id", "position", "latest_team"]], [
        dict(gsis_id="00-WR1", display_name="Wade Receiver", short_name="W.Receiver", pfr_id="RecWa00", position="WR", latest_team="KC"),
        dict(gsis_id="00-OBJ", display_name="Odell Beckham Jr.", short_name="O.Beckham", pfr_id="BeckOd00", position="WR", latest_team="KC"),
        dict(gsis_id="00-RB1", display_name="Ronnie Back", short_name="R.Back", pfr_id="BackRo00", position="RB", latest_team="KC")])
    ff_rows = [
        _ff("00-RB1", "Ronnie Back", 2024, 15, REG_GAME, rush_td=1.0),
        _ff("00-RB1", "Ronnie Back", 2024, 16, "2024_16_KC_DEN", rush_td=1.0),
        _ff("00-RB1", "Ronnie Back", 2024, 20, POST_GAME, rush_td=1.0),
        _ff("00-RB1", "Ronnie Back", 2022, 1, "2022_01_KC_ARI"),
        _ff("00-RB1", "Ronnie Back", 2025, 1, "2025_01_KC_LAC"),
        _ff("00-OBJ", "Odell Beckham Jr.", 2024, 15, REG_GAME, pos="WR"),
        _ff("00-OBJ", "Odell Beckham Jr.", 2024, 16, "2024_16_KC_DEN", pos="WR"),
        _ff("00-WR1", "Wade Receiver", 2024, 15, REG_GAME, pos="WR", rec_td=1.0),
        _ff("00-WR1", "Wade Receiver", 2024, 16, "2024_16_KC_DEN", pos="WR")]
    _insert(conn, "ff_opportunity", [(c, "VARCHAR" if c in ("season", "posteam", "game_id", "player_id",
            "full_name", "position") else "DOUBLE") for c in ff_rows[0]], ff_rows)
    snaps = [
        # OBJ: snap_counts spells him "Odell Beckham" (no "Jr."); pfr id matches players.pfr_id
        *[_snap("Odell Beckham", "BeckOd00", w, 0.8) for w in (1, 2, 3)],
        _snap("Odell Beckham", "BeckOd00", 6, 0.4), _snap("Odell Beckham", "BeckOd00", 7, 0.8),
        _snap("Odell Beckham", "BeckOd00", 15, 0.5), _snap("Odell Beckham", "BeckOd00", 16, 0.7),
        # Wade Receiver: names agree everywhere (control)
        *[_snap("Wade Receiver", "RecWa00", w, 0.9) for w in (1, 2, 3)],
        _snap("Wade Receiver", "RecWa00", 5, 0.45),
        _snap("Wade Receiver", "RecWa00", 15, 0.6), _snap("Wade Receiver", "RecWa00", 16, 0.65)]
    _insert(conn, "snap_counts", [(c, "INTEGER" if c in ("season", "week") else "DOUBLE"
            if c.endswith(("_snaps", "_pct")) else "VARCHAR") for c in snaps[0]], snaps)
    injs = [_inj("00-OBJ", "Odell Beckham Jr.", 4, "Hamstring"), _inj("00-OBJ", "Odell Beckham Jr.", 5, "Hamstring"),
            _inj("00-WR1", "Wade Receiver", 4, "Ankle")]
    _insert(conn, "injuries", [(c, "INTEGER" if c in ("season", "week") else "VARCHAR") for c in injs[0]], injs)
    contracts = [  # RB1: 2020 (5.0), 2023 twice (10.0, duplicated like the real feed), 2025 active (20.0)
        dict(player="Ronnie Back", position="RB", team="KC", is_active=False, year_signed=2020, years=4, apy=5.0, apy_cap_pct=0.02, gsis_id="00-RB1"),
        dict(player="Ronnie Back", position="RB", team="KC", is_active=False, year_signed=2023, years=2, apy=10.0, apy_cap_pct=0.04, gsis_id="00-RB1"),
        dict(player="Ronnie Back", position="RB", team="KC", is_active=False, year_signed=2023, years=2, apy=10.0, apy_cap_pct=0.04, gsis_id="00-RB1"),
        dict(player="Ronnie Back", position="RB", team="KC", is_active=True, year_signed=2025, years=3, apy=20.0, apy_cap_pct=0.07, gsis_id="00-RB1")]
    _insert(conn, "contracts", [("player", "VARCHAR"), ("position", "VARCHAR"), ("team", "VARCHAR"),
            ("is_active", "BOOLEAN"), ("year_signed", "INTEGER"), ("years", "INTEGER"), ("apy", "DOUBLE"),
            ("apy_cap_pct", "DOUBLE"), ("gsis_id", "VARCHAR")], contracts)
    # FTN: WR1 catchable targets REG plays 1, 2 (drop), 4, 11 and POST plays 201, 203 (drop)
    ftn = [dict(nflverse_game_id=g, nflverse_play_id=p, season=2024, week=w, is_catchable_ball=True,
                is_drop=p in (2, 203), is_contested_ball=False, is_created_reception=False)
           for g, w, p in [(REG_GAME, 15, 1), (REG_GAME, 15, 2), (REG_GAME, 15, 4), (REG_GAME, 15, 11),
                           (POST_GAME, 20, 201), (POST_GAME, 20, 203)]]
    _insert(conn, "ftn_charting", [("nflverse_game_id", "VARCHAR"), ("nflverse_play_id", "INTEGER"),
            ("season", "INTEGER"), ("week", "INTEGER"), ("is_catchable_ball", "BOOLEAN"), ("is_drop", "BOOLEAN"),
            ("is_contested_ball", "BOOLEAN"), ("is_created_reception", "BOOLEAN")], ftn)
    _insert(conn, "nextgen_stats_receiving", [("season", "INTEGER"), ("season_type", "VARCHAR"),
            ("week", "INTEGER"), ("player_gsis_id", "VARCHAR"), ("player_display_name", "VARCHAR"),
            ("team_abbr", "VARCHAR"), ("avg_separation", "DOUBLE"), ("avg_yac_above_expectation", "DOUBLE"),
            ("catch_percentage", "DOUBLE")],
            [dict(season=2024, season_type="REG", week=15, player_gsis_id="00-WR1",
                  player_display_name="Wade Receiver", team_abbr="KC", avg_separation=3.0,
                  avg_yac_above_expectation=0.5, catch_percentage=70.0)])
    ingest._ensure_metadata_table(conn)
    return conn


def rebuild(conn):
    fn = getattr(ingest, "rebuild_derived_tables", None)
    if fn is not None:
        fn(conn)
    else:  # today's builders, so behavior that already holds can be pinned now
        ingest._create_aggregate_tables(conn)
        ingest._create_fantasy_tables(conn)
        ingest._create_team_records(conn)


def rows(conn, sql, params=()):
    rel = conn.execute(sql, list(params))
    cols = [d[0] for d in rel.description]
    return [dict(zip(cols, r)) for r in rel.fetchall()]


@pytest.fixture
def db(tmp_path):
    conn = build_db(tmp_path / "f.duckdb")
    rebuild(conn)
    yield conn
    conn.close()


def one(conn, table, team, season_type, **where):
    extra = "".join(f" AND {k} = ?" for k in where)
    found = rows(conn, f"SELECT * FROM {table} WHERE team = ? AND season_year = 2024 "
                       f"AND season_type = ?{extra}", [team, season_type, *where.values()])
    assert len(found) == 1, found
    return found[0]


# ── 1. grain ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("table", ["team_offense_stats", "team_defense_stats"])
def test_team_tables_split_reg_and_post_kc(db, table):
    keys = rows(db, f"SELECT season_year, season_type FROM {table} WHERE team='KC' ORDER BY 2 DESC")
    # KC offense: REG game vs CLE + POST game vs HOU; KC defense: CLE REG offense + HOU POST row
    assert [(r["season_year"], r["season_type"]) for r in keys] == [(2024, "REG"), (2024, "POST")]


@pytest.mark.parametrize("table", ["team_offense_stats", "team_defense_stats", "situational_stats"])
def test_team_tables_one_row_per_key(db, table):
    key = "team, season_year, season_type" + (", situation" if table == "situational_stats" else "")
    dup = db.execute(f"SELECT {key}, COUNT(*) FROM {table} GROUP BY ALL HAVING COUNT(*) > 1").fetchall()
    assert dup == []
    assert {r[0] for r in db.execute(f"SELECT DISTINCT season_type FROM {table}").fetchall()} <= {"REG", "POST"}


# ── 2-7. team_offense_stats definitions ──────────────────────────────────────

@pytest.mark.parametrize("season_type, expected", [
    # REG pass/run non-2pt rows: 1-7, 9-11, 13-14, 16-21, 24-27, 29-31, 33-38, 40-41 = 33
    ("REG", 33),
    ("POST", 4),  # rows 201-204
])
def test_total_plays_counts_scrimmage_only(db, season_type, expected):
    assert one(db, "team_offense_stats", "KC", season_type)["total_plays"] == expected


@pytest.mark.parametrize("col, expected", [
    ("pass_plays", 15),     # pass, sack=0, non-2pt: 1,2,4,5,7,11,13,16,18,20,21,25,31,35,36
    ("pass_attempts", 16),  # 15 + spike 42 (official nflverse attempt)
    ("completions", 8),     # 1,4,7,11,16,18,21,36 (2-pt completion 39 excluded)
    ("sacks", 3),           # 6, 27, 40
    ("dropbacks", 20),      # 15 passes + 3 sacks + 2 scrambles (no_play 15 and 2-pt 39 excluded)
    ("pass_yards", 111),    # 8+15+12+25+22+7+13+9; sack yards -20 and 2-pt 2 excluded
    ("interceptions", 1),   # row 25
])
def test_pass_columns_kc_reg(db, col, expected):
    assert one(db, "team_offense_stats", "KC", "REG")[col] == expected


@pytest.mark.parametrize("col, expected", [
    ("rush_plays", 17),   # 13 runs + 2 scrambles (10, 41) + 2 kneels (43, 44)
    ("rush_yards", 58),   # runs 44 + scrambles 16 + kneels -2
])
def test_rush_counts_kneels_and_scrambles(db, col, expected):
    split = _has_col(db, "team_offense_stats", "season_type")
    got = rows(db, f"SELECT SUM({col}) AS v FROM team_offense_stats WHERE team='KC'"
                   + (" AND season_type='REG'" if split else ""))[0]["v"]
    # before the split the single KC row also holds POST run 202 (1 carry, 5 yards)
    assert got == expected + (0 if split else {"rush_plays": 1, "rush_yards": 5}[col])


def test_rush_plays_exclude_two_point_run(tmp_path):
    conn = build_db(tmp_path / "f.duckdb", extra_plays=[
        play(45, "run", None, None, 2, 2, qtr=4, time="05:00", two_point_attempt=1)])
    rebuild(conn)
    assert one(conn, "team_offense_stats", "KC", "REG")["rush_plays"] == 17  # unchanged by the 2-pt run


def test_scrambles_reported_separately(db):
    assert one(db, "team_offense_stats", "KC", "REG")["scrambles"] == 2  # rows 10, 41


@pytest.mark.parametrize("season_type, expected", [("REG", 2), ("POST", 1)])  # REG 21, 38 (not 25); POST 204
def test_offense_touchdowns_exclude_pick_six(db, season_type, expected):
    assert one(db, "team_offense_stats", "KC", season_type)["touchdowns"] == expected


@pytest.mark.parametrize("team, expected", [("CLE", 1), ("KC", 0)])  # CLE returned row 25; KC none
def test_defense_touchdowns_credit_pick_six(db, team, expected):
    assert one(db, "team_defense_stats", team, "REG")["def_touchdowns"] == expected


@pytest.mark.parametrize("table, team, col, expected", [
    ("team_offense_stats", "KC", "turnovers", 1),        # INT 25 only; CLE's muff (28) is not KC's
    ("team_offense_stats", "CLE", "turnovers", 2),       # INT 103 + muff 28
    ("team_defense_stats", "CLE", "turnovers_forced", 1),  # INT 25 only; CLE did not force its muff
])
def test_muffed_punt_charged_to_receiving_team(db, table, team, col, expected):
    assert one(db, table, team, "REG")[col] == expected


@pytest.mark.parametrize("season_type, expected", [
    # REG third downs (scrimmage): 3,7,16,21,27,31,35,41 = 8; first_down=1 on 3,16,21,41 -> 50.0
    ("REG", 50.0),
    ("POST", 100.0),  # 204 only, converted
])
def test_third_down_pct_first_down_over_scrimmage(db, season_type, expected):
    assert one(db, "team_offense_stats", "KC", season_type)["third_down_pct"] == expected


@pytest.mark.parametrize("col, expected", [("sacks", 1), ("interceptions", 1)])  # rows 102, 103
def test_defense_sacks_and_ints_kc_reg(db, col, expected):
    assert one(db, "team_defense_stats", "KC", "REG")[col] == expected


# ── 8. situational_stats ─────────────────────────────────────────────────────

@pytest.mark.parametrize("situation, plays, conv", [
    ("4th Down", 2, 50.0),          # 36 (converted), 37; punts 8, 28 and FG 32 excluded
    ("3rd & Long", 3, 33.3),        # 7, 16 (converted), 27; no_play 15 excluded
    ("Two Minute Drill", 2, None),  # 40, 41; spike 42 and kneels 43, 44 excluded
])
def test_situational_buckets_kc_reg(db, situation, plays, conv):
    r = one(db, "situational_stats", "KC", "REG", situation=situation)
    assert r["plays"] == plays
    if conv is not None:
        assert r["conversion_pct"] == conv


def test_situational_every_bucket_has_conversion_rule(db):
    missing = db.execute("SELECT situation FROM situational_stats WHERE conversion_rule IS NULL "
                         "OR TRIM(conversion_rule) = ''").fetchall()
    assert db.execute("SELECT COUNT(*) FROM situational_stats").fetchone()[0] > 0
    assert missing == []


def test_situational_third_and_short_bucket_exists(db):
    # row 3 is the only 3rd & <=3 play; holds before and after the fix
    got = rows(db, "SELECT SUM(plays) AS n FROM situational_stats WHERE team='KC' AND situation='3rd & Short'")
    assert got[0]["n"] == 1


# ── 9-11. player tables ──────────────────────────────────────────────────────

@pytest.mark.parametrize("season, apy", [
    pytest.param(2022, 5.0),
    pytest.param(2024, 10.0),
    (2025, 20.0),  # active contract signed 2025 is also the one in force
])
def test_contract_value_uses_contract_in_force(db, season, apy):
    got = rows(db, "SELECT apy FROM player_contract_value WHERE player_id='00-RB1' AND season=?", [season])
    assert got and all(r["apy"] == apy for r in got), got


@pytest.mark.parametrize("player_id, pcts", [
    pytest.param("00-OBJ", [50.0, 70.0]),
    ("00-WR1", [60.0, 65.0]),  # names agree: holds today
])
def test_role_trend_snap_pct_joined_by_pfr_id(db, player_id, pcts):
    got = rows(db, "SELECT week, snap_pct FROM player_role_trend WHERE player_id=? AND season=2024 "
                   "ORDER BY week", [player_id])
    assert [r["snap_pct"] for r in got] == pcts


@pytest.mark.parametrize("injury, curve", [
    # OBJ out wks 4-5, baseline 0.8 (wks 1-3), back at 0.4 then 0.8 -> 50%, 100%
    pytest.param("hamstring", [(1, 50.0), (2, 100.0)]),
    ("ankle", [(1, 50.0)]),  # Wade out wk 4, baseline 0.9, back at 0.45; names agree
])
def test_injury_return_curve_joined_by_pfr_id(db, injury, curve):
    got = rows(db, "SELECT week_post_return, avg_snap_pct_recovery FROM injury_return_curve "
                   "WHERE injury_type=? AND position='WR' ORDER BY 1", [injury])
    assert [(r["week_post_return"], r["avg_snap_pct_recovery"]) for r in got] == curve


@pytest.mark.parametrize("season_type, rush_td", [("REG", 2.0), ("POST", 1.0)])  # wks 15+16 / wk 20
def test_td_luck_splits_reg_post(db, season_type, rush_td):
    got = rows(db, "SELECT rush_td FROM player_td_luck WHERE player_id='00-RB1' AND season=2024 "
                   "AND season_type=?", [season_type])
    assert [r["rush_td"] for r in got] == [rush_td]


@pytest.mark.parametrize("season_type, targets, drops, pct", [
    ("REG", 4, 1, 25.0),   # plays 1, 2 (drop), 4, 11
    ("POST", 2, 1, 50.0),  # plays 201, 203 (drop)
])
def test_drop_rate_splits_reg_post(db, season_type, targets, drops, pct):
    got = rows(db, "SELECT catchable_targets, drops, drop_rate_pct FROM player_drop_rate "
                   "WHERE player_id='00-WR1' AND season=2024 AND season_type=?", [season_type])
    assert [(r["catchable_targets"], r["drops"], r["drop_rate_pct"]) for r in got] == [(targets, drops, pct)]


def _formation_db(tmp_path):
    extra = [play(400 + i, "pass", None, None, 2, 0, qtr=4, two_point_attempt=1, shotgun=0, no_huddle=1)
             for i in range(4)]  # with row 39: five 2-pt tries, all NO HUDDLE (meets HAVING >= 5)
    extra += [play(210 + i, "pass", 1, 10, 50, 6, post=True, de="HOU", cmp=1) for i in range(3)]
    conn = build_db(tmp_path / "f.duckdb", extra_plays=extra)
    rebuild(conn)
    return conn


def test_formation_effectiveness_excludes_two_point_tries(tmp_path):
    conn = _formation_db(tmp_path)
    # the only NO HUDDLE pass plays are the five 2-pt tries -> no such group after the fix
    got = conn.execute("SELECT COUNT(*) FROM formation_effectiveness WHERE team='KC' "
                       "AND formation='NO HUDDLE' AND play_type='pass'").fetchone()[0]
    assert got == 0


def test_formation_effectiveness_splits_reg_post(tmp_path):
    conn = _formation_db(tmp_path)
    r = one(conn, "formation_effectiveness", "KC", "POST", formation="SHOTGUN", play_type="pass")
    assert r["plays"] == 6  # 201, 203, 204 + 210-212


# ── 12. metadata, idempotence, transaction ───────────────────────────────────

def _has_col(conn, table, col):
    return conn.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name=? "
                        "AND column_name=?", [table, col]).fetchone()[0] > 0


def test_rebuild_function_exists():
    fn = getattr(ingest, "rebuild_derived_tables", None)
    if fn is None:
        pytest.xfail(XF + "rebuild_derived_tables missing")
    assert callable(fn)


@pytest.mark.parametrize("table", [
    pytest.param(t)
    if t in DERIVED[:4] else t for t in DERIVED])
def test_metadata_one_row_per_derived_table(db, table):
    got = rows(db, "SELECT table_name, season, loader_fn FROM _ingest_metadata WHERE dataset_id=?", [table])
    assert got == [{"table_name": table, "season": None, "loader_fn": "derived"}]


def _counts(conn):
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in DERIVED}


def test_rebuild_twice_is_idempotent(db):
    first = _counts(db)
    meta = db.execute("SELECT COUNT(*) FROM _ingest_metadata").fetchone()[0]
    rebuild(db)
    assert _counts(db) == first
    assert db.execute("SELECT COUNT(*) FROM _ingest_metadata").fetchone()[0] == meta


def test_rebuild_reflects_changed_source(db):
    # deleting the POST plays must remove them from the derived offense table on rebuild
    db.execute("DELETE FROM plays WHERE season_type = 'POST'")
    rebuild(db)
    assert db.execute("SELECT SUM(total_plays) FROM team_offense_stats WHERE team='HOU'").fetchone()[0] is None


class _FailingConn:
    """Proxy that raises on the CREATE of one target table; everything else passes through."""

    def __init__(self, conn, target):
        self._conn, self._pat = conn, re.compile(rf"CREATE\b.*\b{target}", re.I | re.S)

    def execute(self, sql, *a, **k):
        if self._pat.search(sql):
            raise duckdb.Error(f"injected failure: {sql[:40]}")
        return self._conn.execute(sql, *a, **k)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _snapshot(conn):
    return {t: sorted(map(repr, conn.execute(f"SELECT * FROM {t}").fetchall())) for t in DERIVED}


@pytest.mark.parametrize("target", ["situational_stats", "player_contract_value"])
def test_failed_rebuild_keeps_previous_tables(db, target):
    fn = getattr(ingest, "rebuild_derived_tables", None)
    assert fn is not None, "rebuild_derived_tables missing"
    before = _snapshot(db)
    db.execute("DELETE FROM plays WHERE season_type = 'POST'")  # a partial rebuild would change tables
    db.execute("DELETE FROM contracts WHERE year_signed = 2025")
    with pytest.raises(Exception):
        fn(_FailingConn(db, target))
    assert _snapshot(db) == before


# ── 13. tools on the synthetic DB ────────────────────────────────────────────

@pytest.fixture
def tool_db(tmp_path, monkeypatch):
    path = tmp_path / "t.duckdb"
    conn = build_db(path)
    rebuild(conn)
    conn.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


@pytest.mark.parametrize("season_type, touchdowns", [("REG", 2), ("POST", 1)])
def test_team_stats_tool_filters_season_type_synthetic(tool_db, season_type, touchdowns):
    res = tools.nfl_team_stats("KC", 2024, side="offense", season_type=season_type)
    assert [(r["season_type"], r["touchdowns"]) for r in res["offense"]] == [(season_type, touchdowns)]


def test_compare_team_post_empty_for_non_playoff_team_synthetic(tool_db):
    res = tools.nfl_compare("KC", "CLE", compare_type="team", season=2024, season_type="POST")
    assert [r["season_type"] for r in res["KC"]["offense"]] == ["POST"]
    assert res["CLE"]["offense"] == [] and res["CLE"].get("reason")  # CLE has no POST plays


# ── Part 2: integration against the real DB ─────────────────────────────────

@pytest.fixture
def real(monkeypatch):
    if not REAL_DB.exists():
        pytest.skip(f"{REAL_DB} missing")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(REAL_DB))
    # Same opener as the tools, so the open oracle connection never clashes with theirs.
    with get_db_connection() as conn:
        yield conn


def _official(conn, team, season_type="REG"):
    """Sum of nflverse team_stats_raw per-game rows: an independent table from plays."""
    return rows(conn, """
        SELECT SUM(attempts) AS pass_attempts, SUM(completions) AS completions,
               SUM(carries) AS rush_plays, SUM(rushing_yards) AS rush_yards,
               SUM(passing_tds + rushing_tds) AS touchdowns, SUM(passing_interceptions) AS interceptions,
               SUM(def_sacks) AS def_sacks, SUM(def_interceptions) AS def_interceptions
        FROM team_stats_raw WHERE season = 2024 AND season_type = ? AND team = ?""", [season_type, team])[0]


@pytest.mark.integration
@pytest.mark.parametrize("team", ["KC", "DET", "BUF"])
@pytest.mark.parametrize("col", ["pass_attempts", "completions", "rush_plays", "rush_yards",
                                 "touchdowns", "interceptions"])
def test_offense_reg_matches_team_stats_raw(real, team, col):
    expected = _official(real, team)[col]
    got = one(real, "team_offense_stats", team, "REG")[col]
    assert abs(float(got) - float(expected)) <= 0.5, (got, expected)


@pytest.mark.integration
@pytest.mark.parametrize("team", ["KC", "DET", "BUF"])
@pytest.mark.parametrize("col, official, tol", [
    # team_stats_raw rounds half-sacks per player, so BUF 2024 REG reads 37 vs 39 sack plays; allow 2
    ("sacks", "def_sacks", 2),
    ("interceptions", "def_interceptions", 0),
])
def test_defense_reg_matches_team_stats_raw(real, team, col, official, tol):
    expected = _official(real, team)[official]
    got = one(real, "team_defense_stats", team, "REG")[col]
    assert abs(float(got) - float(expected)) <= tol, (got, expected)


@pytest.mark.integration
@pytest.mark.parametrize("team", ["KC", "DET"])
def test_team_stats_tool_one_reg_row_per_season(real, team):
    res = tools.nfl_team_stats(team, side="offense", season_type="REG")
    seasons = [r["season_year"] for r in res["offense"]]
    assert seasons and len(seasons) == len(set(seasons))
    assert {r["season_type"] for r in res["offense"]} == {"REG"}
    row24 = next(r for r in res["offense"] if r["season_year"] == 2024)
    assert row24["pass_attempts"] == _official(real, team)["pass_attempts"]


@pytest.mark.integration
def test_team_stats_tool_post_row_separate_kc_2024(real):
    res = tools.nfl_team_stats("KC", 2024, side="offense", season_type="POST")
    assert [r["season_type"] for r in res["offense"]] == ["POST"]
    assert res["offense"][0]["pass_attempts"] == _official(real, "KC", "POST")["pass_attempts"]


def _no_playoff_team_2024(conn):
    return conn.execute("""
        SELECT t FROM (SELECT home_team AS t FROM schedules WHERE season=2024 AND game_type='REG'
                       UNION SELECT away_team FROM schedules WHERE season=2024 AND game_type='REG')
        WHERE t NOT IN (SELECT home_team FROM schedules WHERE season=2024 AND game_type<>'REG'
                        UNION SELECT away_team FROM schedules WHERE season=2024 AND game_type<>'REG')
        ORDER BY t LIMIT 1""").fetchone()[0]


@pytest.mark.integration
def test_compare_team_post_differs_from_reg_kc(real):
    reg = tools.nfl_compare("KC", "DET", compare_type="team", season=2024, season_type="REG")
    post = tools.nfl_compare("KC", "DET", compare_type="team", season=2024, season_type="POST")
    assert post["KC"]["offense"][0]["pass_attempts"] == _official(real, "KC", "POST")["pass_attempts"]
    assert post["KC"]["offense"][0]["pass_attempts"] != reg["KC"]["offense"][0]["pass_attempts"]


@pytest.mark.integration
def test_compare_team_post_empty_with_reason_for_non_playoff_team(real):
    other = _no_playoff_team_2024(real)  # derived from schedules, not from the tool
    res = tools.nfl_compare("KC", other, compare_type="team", season=2024, season_type="POST")
    assert res["KC"]["offense"]
    assert res[other]["offense"] == [] and res[other].get("reason")


# ── stale (not yet rebuilt) team tables: no season_type column ───────────────

@pytest.fixture
def stale_db(tmp_path, monkeypatch):
    path = tmp_path / "stale.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE team_offense_stats (team VARCHAR, season_year INTEGER, touchdowns INTEGER)")
    conn.execute("CREATE TABLE team_defense_stats (team VARCHAR, season_year INTEGER, sacks INTEGER)")
    conn.execute("CREATE TABLE situational_stats (team VARCHAR, season_year INTEGER, situation VARCHAR)")
    conn.execute("INSERT INTO team_offense_stats VALUES ('KC', 2024, 50)")
    conn.execute("INSERT INTO team_defense_stats VALUES ('KC', 2024, 40)")
    conn.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))


def test_stale_tables_default_reg_returns_rows_with_warning(stale_db):
    res = tools.nfl_team_stats("KC", 2024, side="offense")
    assert [r["touchdowns"] for r in res["offense"]] == [50]
    assert "nfl-mcp rebuild" in res["warning"]
    cmp = tools.nfl_compare("KC", "KC", compare_type="team", season=2024)
    assert cmp["KC"]["offense"] and "nfl-mcp rebuild" in cmp["KC"]["warning"]


def test_stale_tables_explicit_post_returns_error_with_code(stale_db):
    expected = {"error": "Derived team tables are stale; run `nfl-mcp rebuild`",
                "code": "STALE_DERIVED_TABLES"}
    assert tools.nfl_team_stats("KC", 2024, season_type="POST") == expected
    assert tools.nfl_compare("KC", "BUF", compare_type="team", season=2024,
                             season_type="POST") == expected
