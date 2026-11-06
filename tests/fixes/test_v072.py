"""v0.7.2: net passing yards in the team tools (schema 4).

Source: the v0.7.2 brief. nfl_team_stats for BAL 2024 REG offense gave rush_yards 3184,
pass_yards 4189 and total_yards 7213. The parts add up to 7373. total_yards subtracts sacks
(SUM(yards_gained) over pass and run plays) and pass_yards does not (SUM(passing_yards) on
attempts), so models reported the gross passing figure next to the net total and could not
explain the gap. yards_allowed on defense is built the same way.

Every expected value comes from SQL on the raw plays table (or the nflverse team_stats_raw
table for the comparison with the published figures), never from a tool or a derived table.
The SQL sits in a comment next to the value. Values were measured on a 2024-only database
(the shape GitHub CI ingests).

Databases. Integration tests read NFL_MCP_DB_PATH when it is set, otherwise
/tmp/ci2024/v072d.duckdb (a copy of v071.duckdb upgraded by this nfl-mcp, read-only; v072,
v072b and v072c hold earlier schema-4 builds of this release: before kneels, with
SUM(yards_gained) as the total, and before spikes in the play count). The schema-3 tests
read /tmp/ci2024/v071.duckdb only. Each skips with a reason when its file is missing. The unit
tests build small databases in tmp_path and need no file.

What the gap between total_yards and rush_yards + net_pass_yards is, measured on 2024 plays:
- Yards after a lateral on a run (lateral_rushing_yards): 3 plays, 19 yards. They are in
  total_yards (yards_gained) and were not in rush_yards. rush_yards now counts them, as the
  player tools and nflverse team_stats_raw do.
- Kneels: play_type 'qb_kneel' with rush_attempt=1, 437 plays. rush_yards counts them (as the
  official rushing figure does), and total_yards and total_plays left them out (pass and run
  plays only, since group F). Official total net yards is net rushing plus net passing, so
  total_plays, total_yards and yards_per_play (and the defense play count, yards_allowed and
  yards_per_play_allowed) now count kneels. The official play count is rushes + pass attempts
  + sacks, and a spike is a pass attempt, so the play counts also count spikes. EPA,
  third-down, red-zone and explosive-play figures keep the pass and run plays only.
- Rows tagged no_play stay out of every figure. The nflverse team stats count one 2024 play
  tagged no_play (2024_15_MIA_HOU 2274, a 35-yard fake-punt run), so HOU REG differs there.
- Two plays where yards_gained holds only part of a lateral play with an enforced penalty
  (2024_02_LV_BAL 4375: -25, 2024_04_NE_SF 3004: -3). The yards credited to the rusher and
  passer on these plays are the official ones, so total_yards (and yards_allowed) is now
  rush + net pass, built from those parts, not SUM(yards_gained).

Interface contract tested here
- team_offense_stats gains sack_yards (yards lost on sacks, positive) and net_pass_yards
  (pass_yards - sack_yards). pass_yards stays gross. sacks stays. rush_yards counts rushing
  yards plus lateral_rushing_yards on rush plays.
- team_defense_stats gains rush_yards_allowed, pass_yards_allowed (gross), sack_yards (yards
  the defense gained on sacks, positive) and net_pass_yards_allowed. yards_allowed and sacks
  keep their names and meaning.
- On both tables total = rush + net pass on every row, by construction. yards_per_play is
  total / plays, with kneels and spikes in the play count. total_plays equals carries +
  attempts + sacks_suffered of the nflverse team stats for every 2024 team and season type
  except HOU REG.
- migrate.SCHEMA_VERSION is 4. The new columns are in REQUIRED_COLUMNS, both tables are in
  DERIVED_TABLES and UPGRADE_REWRITES, and a schema-3 database is upgraded without the network
  to the same tables a fresh build makes.
- nfl_team_stats and nfl_compare (team mode) return a `definitions` block keyed by the
  returned blocks ("offense", "defense"); each is a dict field -> text that says gross or net,
  sacks and what total_yards counts. side="situational" sends no definitions.
- On a database whose team tables lack the new columns, both tools still answer with the old
  fields and a `warning` that says the new fields appear after the next database update. The
  warning does not name an nfl-mcp command. Definitions leave out the fields the rows lack.
- __version__ is 0.7.2. Both tool descriptions name net_pass_yards and gross pass_yards.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
from pathlib import Path

import anyio
import duckdb
import pytest

from nfl_mcp import __version__, migrate, server, tools
from nfl_mcp import ingest as ing
from nfl_mcp.database import get_db_connection
from nfl_mcp.required_columns import REQUIRED_COLUMNS

_ENV_DB = os.environ.get("NFL_MCP_DB_PATH")
DB4 = Path(_ENV_DB or "/tmp/ci2024/v072d.duckdb")
DB3 = Path("/tmp/ci2024/v071.duckdb")
TOOLS_BY_NAME = {t.name: t for t in server.TOOLS}

NEW_OFFENSE = ("sack_yards", "net_pass_yards")
NEW_DEFENSE = ("rush_yards_allowed", "pass_yards_allowed", "sack_yards", "net_pass_yards_allowed")


# ── helpers ────────────────────────────────────────────────────────────────────

def _raw(name, args):
    """A tool call through server.call_tool: the tool's checks and the one error shape."""
    async def go():
        return await server.call_tool(name, args)
    res = anyio.run(go)
    return res.isError, json.loads(res.content[0].text)


def _desc(name):
    return TOOLS_BY_NAME[name].description


_PLAY_COLS = {
    "game_id": "VARCHAR", "play_id": "DOUBLE", "season": "INTEGER", "week": "INTEGER",
    "season_type": "VARCHAR", "posteam": "VARCHAR", "defteam": "VARCHAR", "play_type": "VARCHAR",
    "down": "DOUBLE", "ydstogo": "DOUBLE", "yardline_100": "DOUBLE", "qtr": "DOUBLE",
    "time": "VARCHAR", "shotgun": "DOUBLE", "no_huddle": "DOUBLE", "yards_gained": "DOUBLE",
    "epa": "DOUBLE", "touchdown": "DOUBLE", "td_team": "VARCHAR", "first_down": "DOUBLE",
    "fumble_lost": "DOUBLE", "fumbled_1_team": "VARCHAR", "interception": "DOUBLE",
    "rush_attempt": "DOUBLE", "pass_attempt": "DOUBLE", "sack": "DOUBLE",
    "two_point_attempt": "DOUBLE", "complete_pass": "DOUBLE", "qb_dropback": "DOUBLE",
    "qb_scramble": "DOUBLE", "qb_kneel": "DOUBLE", "qb_spike": "DOUBLE",
    "passing_yards": "DOUBLE", "rushing_yards": "DOUBLE", "lateral_rushing_yards": "DOUBLE",
}


def _p(pid, play_type, yds, *, pos="BAL", de="PIT", **kw):
    row = {"game_id": "2024_05_BAL_PIT", "play_id": float(pid), "season": 2024, "week": 5,
           "season_type": "REG", "posteam": pos, "defteam": de, "play_type": play_type,
           "down": 1.0, "ydstogo": 10.0, "yardline_100": 60.0, "qtr": 1.0, "time": "10:00",
           "shotgun": 1.0, "no_huddle": 0.0, "yards_gained": float(yds), "epa": 0.1,
           "touchdown": 0.0, "first_down": 0.0, "fumble_lost": 0.0, "interception": 0.0,
           "rush_attempt": 0.0, "pass_attempt": 0.0, "sack": 0.0, "two_point_attempt": 0.0,
           "complete_pass": 0.0, "qb_dropback": 0.0, "qb_scramble": 0.0, "qb_kneel": 0.0,
           "qb_spike": 0.0}
    row.update(kw)
    return row


# BAL offense vs PIT, 2024 REG. Hand totals:
#   pass_yards (gross)  20 + 0 + 15                      = 35 (the spike is an attempt for 0)
#   sacks 2, sack_yards 7 + 3                            = 10
#   net_pass_yards      35 - 10                          = 25
#   rush_yards          10 + (4 + 6 after a lateral) - 1 (kneel) = 19
#   total_yards         19 + 25 = 44 (rush_yards + net_pass_yards)
#   total_plays         9: plays 1-7, the spike 10 and 11 (no two-point try or no_play)
#   total = rush + net  = 19 + 25 = 44
# PIT offense: one 5-yard completion and no sack (sack_yards 0, not NULL).
_PASS = {"pass_attempt": 1.0, "qb_dropback": 1.0}
_PLAYS = [
    _p(1, "pass", 20, complete_pass=1.0, passing_yards=20.0, **_PASS),
    _p(2, "pass", 0, **_PASS),
    _p(3, "pass", -7, sack=1.0, **_PASS),
    _p(4, "pass", -3, sack=1.0, **_PASS),
    _p(5, "run", 10, rush_attempt=1.0, rushing_yards=10.0),
    _p(6, "run", 10, rush_attempt=1.0, rushing_yards=4.0, lateral_rushing_yards=6.0),
    _p(7, "qb_kneel", -1, rush_attempt=1.0, qb_kneel=1.0, rushing_yards=-1.0),
    _p(8, "pass", 2, two_point_attempt=1.0, complete_pass=1.0, passing_yards=2.0, **_PASS),
    _p(9, "no_play", 5, rush_attempt=1.0, rushing_yards=5.0),
    _p(10, "qb_spike", 0, pass_attempt=1.0, qb_spike=1.0),
    _p(11, "pass", 15, complete_pass=1.0, passing_yards=15.0, **_PASS),
    _p(12, "pass", 5, pos="PIT", de="BAL", complete_pass=1.0, passing_yards=5.0, **_PASS),
]


def _team_db(path=":memory:", plays=_PLAYS):
    con = duckdb.connect(str(path))
    ddl = ", ".join(f'"{c}" {t}' for c, t in _PLAY_COLS.items())
    con.execute(f"CREATE TABLE plays ({ddl})")
    marks = ", ".join("?" for _ in _PLAY_COLS)
    con.executemany(f"INSERT INTO plays VALUES ({marks})",
                    [[r.get(c) for c in _PLAY_COLS] for r in plays])
    with contextlib.redirect_stdout(io.StringIO()):
        ing._create_aggregate_tables(con)
    return con


def _row(con, table, team):
    rel = con.execute(f"SELECT * FROM {table} WHERE team = ? AND season_type = 'REG'", [team])
    cols = [d[0] for d in rel.description]
    (row,) = rel.fetchall()
    return dict(zip(cols, row))


def _make_stale(con):
    """Drop the schema-4 columns, as v0.7.1 built the tables."""
    for col in NEW_OFFENSE:
        con.execute(f"ALTER TABLE team_offense_stats DROP COLUMN {col}")
    for col in NEW_DEFENSE:
        con.execute(f"ALTER TABLE team_defense_stats DROP COLUMN {col}")


@pytest.fixture
def team_db(tmp_path, monkeypatch):
    path = tmp_path / "v072.duckdb"
    _team_db(path).close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


@pytest.fixture
def stale_team_db(tmp_path, monkeypatch):
    path = tmp_path / "v071.duckdb"
    con = _team_db(path)
    _make_stale(con)
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


# ═══ 1. offense: sacks, sack_yards, net_pass_yards ═════════════════════════════

def test_offense_has_gross_pass_sack_yards_and_net():
    row = _row(_team_db(), "team_offense_stats", "BAL")
    assert row["pass_yards"] == 35
    assert row["sacks"] == 2
    assert row["sack_yards"] == 10
    assert row["net_pass_yards"] == 25


def test_offense_rush_yards_count_yards_after_a_lateral():
    row = _row(_team_db(), "team_offense_stats", "BAL")
    assert row["rush_yards"] == 19
    assert row["yards_per_rush"] == round(19 / row["rush_plays"], 2)


def test_offense_total_is_rush_plus_net_with_kneels_and_spikes():
    row = _row(_team_db(), "team_offense_stats", "BAL")
    assert (row["total_plays"], row["total_yards"]) == (9, 44)
    assert row["total_yards"] == row["rush_yards"] + row["net_pass_yards"]
    assert row["yards_per_play"] == round(44 / 9, 2)


# A run like 2024_02_LV_BAL 4375: 25 rushing yards and 5 after a lateral, but yards_gained 5
# (the source keeps only part of the play). The credited yards are the official ones, so
# total_yards and yards_allowed add the 30 rushing yards, not the 5 of yards_gained:
#   rush_yards 19 + 30 = 49, total_yards 49 + 25 = 74, total_plays 10 (SUM(yards_gained) 49)
_LATERAL_PENALTY = _p(13, "run", 5, rush_attempt=1.0, rushing_yards=25.0,
                      lateral_rushing_yards=5.0)


def test_total_is_built_from_the_credited_yards_not_yards_gained():
    con = _team_db(plays=[*_PLAYS, _LATERAL_PENALTY])
    off = _row(con, "team_offense_stats", "BAL")
    assert (off["rush_yards"], off["net_pass_yards"], off["total_yards"]) == (49, 25, 74)
    assert (off["total_plays"], off["yards_per_play"]) == (10, round(74 / 10, 2))
    de = _row(con, "team_defense_stats", "PIT")
    assert (de["rush_yards_allowed"], de["yards_allowed"], de["plays_against"]) == (49, 74, 10)
    assert de["yards_per_play_allowed"] == round(74 / 10, 2)


def test_offense_analytic_metrics_leave_kneels_and_spikes_out():
    # avg_epa: every play has epa 0.1 except the kneel and the spike, set to -5 here.
    plays = [dict(r, epa=-5.0) if r["play_type"] in ("qb_kneel", "qb_spike") else r
             for r in _PLAYS]
    con = duckdb.connect(":memory:")
    ddl = ", ".join(f'"{c}" {t}' for c, t in _PLAY_COLS.items())
    con.execute(f"CREATE TABLE plays ({ddl})")
    con.executemany(f"INSERT INTO plays VALUES ({', '.join('?' for _ in _PLAY_COLS)})",
                    [[r.get(c) for c in _PLAY_COLS] for r in plays])
    with contextlib.redirect_stdout(io.StringIO()):
        ing._create_aggregate_tables(con)
    assert _row(con, "team_offense_stats", "BAL")["avg_epa"] == 0.1
    assert _row(con, "team_defense_stats", "PIT")["avg_epa_allowed"] == 0.1


def test_offense_without_sacks_has_zero_sack_yards():
    row = _row(_team_db(), "team_offense_stats", "PIT")
    assert (row["sacks"], row["sack_yards"], row["pass_yards"], row["net_pass_yards"]) == (0, 0, 5, 5)


def test_offense_column_order_puts_the_split_after_pass_yards():
    con = _team_db()
    cols = [r[0] for r in con.execute("DESCRIBE team_offense_stats").fetchall()]
    i = cols.index("pass_yards")
    assert cols[i + 1:i + 3] == ["sack_yards", "net_pass_yards"], cols


# ═══ 2. defense: the same split for yards allowed ══════════════════════════════

def test_defense_has_rush_gross_pass_sack_yards_and_net_allowed():
    row = _row(_team_db(), "team_defense_stats", "PIT")
    assert (row["plays_against"], row["yards_allowed"]) == (9, 44)
    assert row["yards_per_play_allowed"] == round(44 / 9, 2)
    assert row["rush_yards_allowed"] == 19
    assert row["pass_yards_allowed"] == 35
    assert row["sacks"] == 2
    assert row["sack_yards"] == 10
    assert row["net_pass_yards_allowed"] == 25
    assert row["yards_allowed"] == row["rush_yards_allowed"] + row["net_pass_yards_allowed"]


def test_defense_without_sacks_has_zero_sack_yards():
    row = _row(_team_db(), "team_defense_stats", "BAL")
    assert (row["sacks"], row["sack_yards"], row["pass_yards_allowed"],
            row["net_pass_yards_allowed"]) == (0, 0, 5, 5)


def test_defense_mirrors_the_opponent_offense():
    con = _team_db()
    off, de = _row(con, "team_offense_stats", "BAL"), _row(con, "team_defense_stats", "PIT")
    assert (off["rush_yards"], off["pass_yards"], off["sacks"], off["sack_yards"],
            off["net_pass_yards"], off["total_yards"]) == (
        de["rush_yards_allowed"], de["pass_yards_allowed"], de["sacks"], de["sack_yards"],
        de["net_pass_yards_allowed"], de["yards_allowed"])


# ═══ 3. schema version 4 ═══════════════════════════════════════════════════════

def test_required_columns_list_the_new_fields():
    assert set(NEW_OFFENSE) <= set(REQUIRED_COLUMNS["team_offense_stats"])
    assert set(NEW_DEFENSE) <= set(REQUIRED_COLUMNS["team_defense_stats"])


def test_schema_version_is_4_and_team_tables_are_rewritten_by_the_upgrade():
    assert migrate.SCHEMA_VERSION == 4
    for table in ("team_offense_stats", "team_defense_stats"):
        assert table in migrate.DERIVED_TABLES
        assert table in migrate.UPGRADE_REWRITES


def test_check_flags_team_tables_without_the_new_columns(tmp_path):
    con = _team_db(tmp_path / "v3.duckdb")
    _make_stale(con)
    status = migrate.check(con)
    con.close()
    derived = next(s for s in status.steps if s.name == "derived_tables")
    assert derived.needed and not derived.needs_network
    for col in NEW_OFFENSE:
        assert f"team_offense_stats.{col}" in derived.detail
    for col in NEW_DEFENSE:
        assert f"team_defense_stats.{col}" in derived.detail


def test_version_is_0_7_2():
    assert __version__ == "0.7.2"


# ═══ 4. the tools: new fields and definitions ══════════════════════════════════

def test_team_stats_returns_the_new_fields(team_db):
    res = tools.nfl_team_stats(team="BAL", season=2024, season_type="REG")
    assert "error" not in res, res
    (off,) = res["offense"]
    (de,) = res["defense"]
    assert (off["pass_yards"], off["sacks"], off["sack_yards"], off["net_pass_yards"]) == (35, 2, 10, 25)
    assert (de["pass_yards_allowed"], de["sack_yards"], de["net_pass_yards_allowed"],
            de["rush_yards_allowed"]) == (5, 0, 5, 0)
    assert "warning" not in res


def test_team_stats_definitions_cover_both_blocks(team_db):
    defs = tools.nfl_team_stats(team="BAL", season=2024, season_type="REG")["definitions"]
    assert set(defs) == {"offense", "defense"}
    for field in ("total_plays", "total_yards", "rush_yards", "pass_yards", "sacks", "avg_epa",
                  *NEW_OFFENSE):
        assert field in defs["offense"], field
    for field in ("plays_against", "yards_allowed", "sacks", "avg_epa_allowed", *NEW_DEFENSE):
        assert field in defs["defense"], field


def test_team_stats_definitions_say_gross_net_sacks_and_kneels(team_db):
    defs = tools.nfl_team_stats(team="BAL", season=2024, season_type="REG")["definitions"]
    off, de = defs["offense"], defs["defense"]
    assert "gross" in off["pass_yards"].lower() and "sack" in off["pass_yards"].lower()
    assert "net" in off["net_pass_yards"].lower() and "sack_yards" in off["net_pass_yards"]
    assert "sack" in off["total_yards"].lower() and "kneel" in off["total_yards"].lower()
    assert "lateral" in off["rush_yards"].lower() and "kneel" in off["rush_yards"].lower()
    assert "positive" in off["sack_yards"].lower()
    assert "gross" in de["pass_yards_allowed"].lower()
    assert "net" in de["net_pass_yards_allowed"].lower()
    assert "sack" in de["yards_allowed"].lower() and "kneel" in de["yards_allowed"].lower()
    for text in (off["total_plays"], de["plays_against"]):
        assert "kneel" in text.lower() and "spike" in text.lower()
    assert "without kneels or spikes" in off["avg_epa"]
    assert "without kneels or spikes" in de["avg_epa_allowed"]
    for text in (off["total_yards"], off["rush_yards"], de["yards_allowed"]):
        assert "no_play are out" in text
    assert "NFL definition of total net yards" in off["total_yards"]
    assert "NFL definition of total net yards" in de["yards_allowed"]
    for text in (off["sack_yards"], off["net_pass_yards"], de["sack_yards"],
                 de["net_pass_yards_allowed"]):
        assert "nflverse figure" in text and "official" not in text


@pytest.mark.parametrize("side, blocks", [("offense", {"offense"}), ("defense", {"defense"}),
                                          ("both", {"offense", "defense"})])
def test_team_stats_sends_only_the_returned_blocks_definitions(team_db, side, blocks):
    res = tools.nfl_team_stats(team="BAL", season=2024, side=side, season_type="REG")
    assert set(res["definitions"]) == blocks


def test_team_stats_situational_has_no_definitions(team_db):
    res = tools.nfl_team_stats(team="BAL", season=2024, side="situational", season_type="REG")
    assert "error" not in res, res
    assert "definitions" not in res


def test_compare_teams_returns_new_fields_and_definitions(team_db):
    res = tools.nfl_compare(team1="BAL", team2="PIT", season=2024, season_type="REG")
    assert "error" not in res, res
    assert res["BAL"]["offense"][0]["net_pass_yards"] == 25
    assert res["PIT"]["defense"][0]["net_pass_yards_allowed"] == 25
    assert set(res["definitions"]) == {"offense", "defense"}
    assert "net_pass_yards" in res["definitions"]["offense"]
    assert "warning" not in res


def test_team_stats_over_mcp_has_definitions(team_db):
    is_error, payload = _raw("nfl_team_stats", {"team": "BAL", "season": 2024, "season_type": "REG"})
    assert not is_error, payload
    assert payload["offense"][0]["net_pass_yards"] == 25
    assert set(payload["definitions"]) == {"offense", "defense"}


# ═══ 5. a schema-3 database: old fields and a warning ══════════════════════════

def _assert_stale_warning(text):
    assert "next database update" in text
    assert "nfl-mcp" not in text.replace("an older nfl-mcp", "")
    assert "net_pass_yards" in text and "gross" in text


def test_stale_team_stats_answers_with_old_fields_and_a_warning(stale_team_db):
    res = tools.nfl_team_stats(team="BAL", season=2024, season_type="REG")
    assert "error" not in res, res
    (off,) = res["offense"]
    (de,) = res["defense"]
    assert (off["pass_yards"], off["total_yards"], off["sacks"]) == (35, 44, 2)
    assert not set(NEW_OFFENSE) & set(off)
    assert not set(NEW_DEFENSE) & set(de)
    _assert_stale_warning(res["warning"])


def test_stale_team_stats_definitions_leave_out_missing_fields(stale_team_db):
    defs = tools.nfl_team_stats(team="BAL", season=2024, season_type="REG")["definitions"]
    assert not set(NEW_OFFENSE) & set(defs["offense"])
    assert not set(NEW_DEFENSE) & set(defs["defense"])
    assert "pass_yards" in defs["offense"] and "sacks" in defs["defense"]
    # A schema-3 table has no lateral yards in rush_yards and no kneels in the totals, so
    # their schema-4 definitions are left out.
    for field in ("rush_yards", "total_yards", "total_plays", "yards_per_play"):
        assert field not in defs["offense"], field
    assert "yards_allowed" not in defs["defense"] and "plays_against" not in defs["defense"]


def test_stale_team_stats_one_side_warns_too(stale_team_db):
    res = tools.nfl_team_stats(team="BAL", season=2024, side="defense", season_type="REG")
    _assert_stale_warning(res["warning"])


def test_stale_team_stats_situational_has_no_warning(stale_team_db):
    res = tools.nfl_team_stats(team="BAL", season=2024, side="situational", season_type="REG")
    assert "warning" not in res


def test_stale_team_stats_without_rows_has_no_warning(stale_team_db):
    res = tools.nfl_team_stats(team="BAL", season=2023, season_type="REG")
    assert res["offense"] == [] and "warning" not in res


def test_stale_compare_teams_answers_with_a_warning(stale_team_db):
    res = tools.nfl_compare(team1="BAL", team2="PIT", season=2024, season_type="REG")
    assert "error" not in res, res
    assert res["BAL"]["offense"][0]["pass_yards"] == 35
    assert "net_pass_yards" not in res["BAL"]["offense"][0]
    _assert_stale_warning(res["warning"])
    assert "net_pass_yards" not in res["definitions"]["offense"]


def test_stale_team_stats_over_mcp_is_not_an_error(stale_team_db):
    is_error, payload = _raw("nfl_team_stats", {"team": "BAL", "season": 2024, "season_type": "REG"})
    assert not is_error, payload
    assert "next database update" in payload["warning"]


# ═══ 6. descriptions ═══════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ["nfl_team_stats", "nfl_compare"])
def test_description_names_net_and_gross_passing(name):
    text = _desc(name)
    assert "net_pass_yards" in text and "gross" in text.lower(), text


@pytest.mark.parametrize("name", ["nfl_team_stats", "nfl_compare"])
def test_description_mentions_definitions(name):
    assert "definitions" in _desc(name)


# ═══ 7. real 2024 data ═════════════════════════════════════════════════════════

@pytest.fixture
def real4(monkeypatch):
    """Schema-4 2024 database, read-only."""
    if not DB4.exists():
        pytest.skip(f"{DB4} missing")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(DB4))
    with get_db_connection() as conn:
        cols = {r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'team_offense_stats'").fetchall()}
        if "net_pass_yards" not in cols:
            pytest.skip(f"{DB4} has no team_offense_stats.net_pass_yards (schema 4 not built)")
        if not conn.execute("SELECT count(*) FROM plays WHERE season = 2024").fetchone()[0]:
            pytest.skip("needs 2024 plays")
        yield conn


# Offense, 2024 REG, from plays (the same play sets as the tables):
#   SELECT posteam,
#     count(*) FILTER (WHERE sack=1 AND COALESCE(two_point_attempt,0)=0
#                      AND play_type IS DISTINCT FROM 'no_play')                       -- sacks
#     -sum(yards_gained) FILTER (same)                                                  -- sack_yards
#     sum(passing_yards) FILTER (WHERE pass_attempt=1 AND COALESCE(sack,0)=0
#                                AND COALESCE(two_point_attempt,0)=0
#                                AND play_type IS DISTINCT FROM 'no_play')              -- pass_yards
#     sum(COALESCE(rushing_yards,0)+COALESCE(lateral_rushing_yards,0)) FILTER (WHERE
#         rush_attempt=1 AND COALESCE(two_point_attempt,0)=0
#         AND play_type IS DISTINCT FROM 'no_play')                                     -- rush_yards
#     rush_yards + pass_yards - sack_yards                                              -- total_yards
#     count(*) FILTER (WHERE play_type IN ('pass','run','qb_kneel','qb_spike')
#                      AND COALESCE(two_point_attempt,0)=0)                             -- total_plays
#   FROM plays WHERE season=2024 AND season_type='REG' GROUP BY posteam
#   BAL 24 154 4189 3189 7224 1055 | PHI 45 323 3517 3048 6242 1114
#   DET 33 244 4718 2488 6962 1118 | KC  41 266 4046 1790 5570 1091
#   yards_per_play: BAL 6.85, PHI 5.6, DET 6.23, KC 5.11
#   net = pass - sack: BAL 4035, PHI 3194, DET 4474, KC 3780.
#   rush + net = total: BAL 7224, PHI 6242, DET 6962, KC 5570.
_SACK_YARDS_SQL = """-sum(yards_gained) FILTER (WHERE sack = 1 AND COALESCE(two_point_attempt, 0) = 0
                                   AND play_type IS DISTINCT FROM 'no_play')"""
_PASS_YARDS_SQL = """sum(passing_yards) FILTER (WHERE pass_attempt = 1 AND COALESCE(sack, 0) = 0
                                   AND COALESCE(two_point_attempt, 0) = 0
                                   AND play_type IS DISTINCT FROM 'no_play')"""
_RUSH_YARDS_SQL = """sum(COALESCE(rushing_yards, 0) + COALESCE(lateral_rushing_yards, 0)) FILTER (
            WHERE rush_attempt = 1 AND COALESCE(two_point_attempt, 0) = 0
            AND play_type IS DISTINCT FROM 'no_play')"""
_OFFENSE_SQL = f"""
    SELECT
        count(*) FILTER (WHERE sack = 1 AND COALESCE(two_point_attempt, 0) = 0
                         AND play_type IS DISTINCT FROM 'no_play'),
        {_SACK_YARDS_SQL},
        {_PASS_YARDS_SQL},
        {_RUSH_YARDS_SQL},
        {_RUSH_YARDS_SQL} + {_PASS_YARDS_SQL} - ({_SACK_YARDS_SQL}),
        count(*) FILTER (WHERE play_type IN ('pass', 'run', 'qb_kneel', 'qb_spike')
                         AND COALESCE(two_point_attempt, 0) = 0)
    FROM plays WHERE season = 2024 AND season_type = 'REG' AND {{side}} = ?
"""


@pytest.mark.integration
@pytest.mark.parametrize("team, sacks, sack_yards, pass_yards, rush_yards, total, plays", [
    ("BAL", 24, 154, 4189, 3189, 7224, 1055),
    ("PHI", 45, 323, 3517, 3048, 6242, 1114),
    ("DET", 33, 244, 4718, 2488, 6962, 1118),
    ("KC", 41, 266, 4046, 1790, 5570, 1091),
])
def test_offense_net_passing_matches_plays(real4, team, sacks, sack_yards, pass_yards,
                                           rush_yards, total, plays):
    expected = (sacks, sack_yards, pass_yards, rush_yards, total, plays)
    assert real4.execute(_OFFENSE_SQL.format(side="posteam"), [team]).fetchone() == expected
    res = tools.nfl_team_stats(team=team, season=2024, side="offense", season_type="REG")
    (row,) = res["offense"]
    assert (row["sacks"], row["sack_yards"], row["pass_yards"], row["rush_yards"],
            row["total_yards"], row["total_plays"]) == expected
    assert row["net_pass_yards"] == pass_yards - sack_yards
    assert row["yards_per_play"] == {"BAL": 6.85, "PHI": 5.6, "DET": 6.23, "KC": 5.11}[team]
    assert row["yards_per_play"] == round(total / plays, 2)
    assert "warning" not in res


# Defense, BAL 2024 REG, from plays (the _OFFENSE_SQL sets with defteam = 'BAL'):
#   sacks 54, sack_yards 318, pass_yards_allowed 4468, rush_yards_allowed 1361,
#   yards_allowed 5511, plays_against 1062. net_pass_yards_allowed 4468 - 318 = 4150, and
#   1361 + 4150 = 5511 (5.19 per play).
@pytest.mark.integration
def test_defense_net_passing_allowed_matches_plays(real4):
    expected = (54, 318, 4468, 1361, 5511, 1062)
    assert real4.execute(_OFFENSE_SQL.format(side="defteam"), ["BAL"]).fetchone() == expected
    res = tools.nfl_team_stats(team="BAL", season=2024, side="defense", season_type="REG")
    (row,) = res["defense"]
    assert (row["sacks"], row["sack_yards"], row["pass_yards_allowed"],
            row["rush_yards_allowed"], row["yards_allowed"], row["plays_against"]) == expected
    assert row["net_pass_yards_allowed"] == 4150
    assert row["yards_allowed"] == row["rush_yards_allowed"] + row["net_pass_yards_allowed"]


# The published nflverse team figures (team_stats_raw, one row per team and game):
#   SELECT team, sum(rushing_yards), sum(passing_yards), sum(sacks_suffered),
#          -sum(sack_yards_lost) FROM team_stats_raw
#   WHERE season=2024 AND season_type='REG' GROUP BY team
#   BAL 3189 4189 24 154 | PHI 3048 3517 45 323 | DET 2488 4718 33 244 | KC 1790 4046 41 266
@pytest.mark.integration
@pytest.mark.parametrize("team", ["BAL", "PHI", "DET", "KC"])
def test_offense_matches_nflverse_team_stats(real4, team):
    published = real4.execute(
        "SELECT sum(rushing_yards), sum(passing_yards), sum(sacks_suffered), -sum(sack_yards_lost) "
        "FROM team_stats_raw WHERE season = 2024 AND season_type = 'REG' AND team = ?",
        [team]).fetchone()
    res = tools.nfl_team_stats(team=team, season=2024, side="offense", season_type="REG")
    (row,) = res["offense"]
    assert (row["rush_yards"], row["pass_yards"], row["sacks"], row["sack_yards"]) == published


# The parts of total_yards for every 2024 team and season type, from plays with the table
# play sets (each side by posteam, and by defteam for the defense table):
#   rush part:  rushing_yards + lateral_rushing_yards on rush_attempt=1 (runs, scrambles and
#               kneels), not two-point, not no_play
#   pass part:  passing_yards on pass_attempt=1 AND sack=0, not two-point, not no_play
#   sack part:  -yards_gained on sack=1, not two-point, not no_play
#   SELECT {side}, season_type, rush, pass - sack FROM plays WHERE season=2024 GROUP BY 1, 2
# The table's rush and net passing columns must equal these parts, and its total their sum.
_PARTS_SQL = f"""
    SELECT {{side}} AS team, season_type,
        COALESCE({_RUSH_YARDS_SQL}, 0),
        COALESCE({_PASS_YARDS_SQL}, 0) - COALESCE({_SACK_YARDS_SQL}, 0)
    FROM plays WHERE season = 2024 AND {{side}} IS NOT NULL GROUP BY 1, 2
"""


@pytest.mark.integration
@pytest.mark.parametrize("table, side, total, rush, net", [
    ("team_offense_stats", "posteam", "total_yards", "rush_yards", "net_pass_yards"),
    ("team_defense_stats", "defteam", "yards_allowed", "rush_yards_allowed",
     "net_pass_yards_allowed"),
])
def test_total_is_rush_plus_net_for_every_team(real4, table, side, total, rush, net):
    parts = {(t, st): (r, n) for t, st, r, n in
             real4.execute(_PARTS_SQL.format(side=side)).fetchall()}
    rows = real4.execute(f"SELECT team, season_type, {rush}, {net}, {total} FROM {table} "
                         "WHERE season_year = 2024").fetchall()
    assert len(rows) == 46  # 32 REG + 14 POST
    assert {(t, st): (r, n) for t, st, r, n, _ in rows} == parts
    assert all(tot == r + n for _, _, r, n, tot in rows)


# SUM(yards_gained) differs from rush + net pass only on two 2024 plays, where the source
# yards_gained holds only part of a lateral play with an enforced penalty. The yards credited
# to the rusher and passer are the official ones, so total_yards uses them. Per play on plays:
#   total part:  yards_gained when play_type IN ('pass','run','qb_kneel') AND not two-point
#   minus the rush, pass and sack parts above (sack part as yards_gained, negative)
#   SELECT game_id, play_id, posteam, season_type, yards_gained part - parts FROM plays
#   WHERE season=2024 AND that difference <> 0
#   -> 2024_02_LV_BAL 4375 BAL REG -25 (run 25 + lateral 5, yards_gained 5)
#      2024_04_NE_SF  3004 NE  REG  -3 (pass 7 with a lateral, yards_gained 4)
_LATERAL_PENALTY_PLAYS = {("2024_02_LV_BAL", 4375, "BAL", "REG"): -25,
                          ("2024_04_NE_SF", 3004, "NE", "REG"): -3}
_PLAY_GAP_SQL = """
    WITH parts AS (
        SELECT game_id, CAST(play_id AS INTEGER) AS play_id, {side} AS team, season_type,
            COALESCE(CASE WHEN play_type IN ('pass', 'run', 'qb_kneel') AND {not2}
                          THEN yards_gained END, 0)
            - COALESCE(CASE WHEN rush_attempt = 1 AND {not2} AND {nnp}
                            THEN COALESCE(rushing_yards, 0) + COALESCE(lateral_rushing_yards, 0)
                       END, 0)
            - COALESCE(CASE WHEN pass_attempt = 1 AND COALESCE(sack, 0) = 0 AND {not2} AND {nnp}
                            THEN passing_yards END, 0)
            - COALESCE(CASE WHEN sack = 1 AND {not2} AND {nnp} THEN yards_gained END, 0) AS gap
        FROM plays WHERE season = 2024 AND {side} IS NOT NULL
    )
    SELECT game_id, play_id, team, season_type, gap FROM parts WHERE gap <> 0
""".format(side="{side}", not2="COALESCE(two_point_attempt, 0) = 0",
           nnp="play_type IS DISTINCT FROM 'no_play'")
# SELECT {side}, season_type, sum(yards_gained) FILTER (WHERE play_type IN
#   ('pass','run','qb_kneel') AND COALESCE(two_point_attempt,0)=0) FROM plays
#   WHERE season=2024 GROUP BY 1, 2
_YARDS_GAINED_SQL = """
    SELECT {side}, season_type, COALESCE(sum(yards_gained) FILTER (
        WHERE play_type IN ('pass', 'run', 'qb_kneel') AND COALESCE(two_point_attempt, 0) = 0), 0)
    FROM plays WHERE season = 2024 AND {side} IS NOT NULL GROUP BY 1, 2
"""


@pytest.mark.integration
@pytest.mark.parametrize("table, side, total", [
    ("team_offense_stats", "posteam", "total_yards"),
    ("team_defense_stats", "defteam", "yards_allowed"),
])
def test_yards_gained_differs_only_on_the_lateral_penalty_plays(real4, table, side, total):
    plays = {(g, p, t, st): v for g, p, t, st, v in
             real4.execute(_PLAY_GAP_SQL.format(side=side)).fetchall()}
    # On defense the same plays count for the opponent: LV for 4375, SF for 3004.
    expected = _LATERAL_PENALTY_PLAYS if side == "posteam" else {
        ("2024_02_LV_BAL", 4375, "LV", "REG"): -25, ("2024_04_NE_SF", 3004, "SF", "REG"): -3}
    assert plays == expected
    gained = {(t, st): y for t, st, y in
              real4.execute(_YARDS_GAINED_SQL.format(side=side)).fetchall()}
    rows = real4.execute(f"SELECT team, season_type, {total} FROM {table} "
                         "WHERE season_year = 2024").fetchall()
    # The table total is SUM(yards_gained) minus the gap: BAL REG 7199 + 25 = 7224.
    diff = {(t, st): gained[(t, st)] - tot for t, st, tot in rows if gained[(t, st)] != tot}
    assert diff == {(t, st): v for (_, _, t, st), v in expected.items()}


# The official play count is carries + pass attempts (spikes in) + sacks. From the nflverse
# team stats, against the table, for every 2024 team and season type:
#   SELECT team, season_type, sum(carries) + sum(attempts) + sum(sacks_suffered)
#   FROM team_stats_raw WHERE season=2024 GROUP BY 1, 2       (offense)
#   the same grouped by opponent_team                          (defense)
# They differ only where nflverse counts a play tagged no_play, which the tables leave out:
#   SELECT game_id, play_id, posteam, defteam, rushing_yards FROM plays
#   WHERE season=2024 AND play_type='no_play' AND rush_attempt=1 AND rushing_yards IS NOT NULL
#   -> 2024_15_MIA_HOU 2274 HOU MIA 35 (a fake-punt run): HOU REG offense (1057 against
#      1058) and MIA REG defense (1011 against 1012) are one play short.
_RAW_PLAYS_SQL = """
    SELECT {team}, season_type, sum(carries) + sum(attempts) + sum(sacks_suffered)
    FROM team_stats_raw WHERE season = 2024 GROUP BY 1, 2
"""


@pytest.mark.integration
@pytest.mark.parametrize("table, team, plays, expected", [
    ("team_offense_stats", "team", "total_plays", {("HOU", "REG"): (1057, 1058)}),
    ("team_defense_stats", "opponent_team", "plays_against", {("MIA", "REG"): (1011, 1012)}),
])
def test_play_count_matches_nflverse_except_the_no_play_run(real4, table, team, plays, expected):
    raw = {(t, st): n for t, st, n in real4.execute(_RAW_PLAYS_SQL.format(team=team)).fetchall()}
    rows = real4.execute(f"SELECT team, season_type, {plays} FROM {table} "
                         "WHERE season_year = 2024").fetchall()
    assert len(rows) == len(raw) == 46
    assert {(t, st): (n, raw[(t, st)]) for t, st, n in rows if n != raw[(t, st)]} == expected
    (play,) = real4.execute(
        "SELECT game_id, CAST(play_id AS INTEGER), posteam, defteam, rushing_yards FROM plays "
        "WHERE season = 2024 AND play_type = 'no_play' AND rush_attempt = 1 "
        "AND rushing_yards IS NOT NULL").fetchall()
    assert play == ("2024_15_MIA_HOU", 2274, "HOU", "MIA", 35.0)


@pytest.mark.integration
def test_league_offense_and_defense_agree(real4):
    off = real4.execute("SELECT sum(rush_yards), sum(pass_yards), sum(sack_yards), "
                        "sum(net_pass_yards), sum(total_yards) FROM team_offense_stats").fetchone()
    de = real4.execute("SELECT sum(rush_yards_allowed), sum(pass_yards_allowed), sum(sack_yards), "
                       "sum(net_pass_yards_allowed), sum(yards_allowed) FROM team_defense_stats"
                       ).fetchone()
    assert off == de


@pytest.mark.integration
def test_compare_real_teams_has_definitions(real4):
    res = tools.nfl_compare(team1="BAL", team2="PHI", season=2024, season_type="REG")
    assert res["BAL"]["offense"][0]["net_pass_yards"] == 4035
    assert res["PHI"]["offense"][0]["net_pass_yards"] == 3194
    assert set(res["definitions"]) == {"offense", "defense"}
    assert "warning" not in res


# ═══ 8. upgrade from schema 3 ══════════════════════════════════════════════════

def _team_tables(con):
    return {t: con.execute(f"SELECT * FROM {t} ORDER BY team, season_year, season_type").fetchall()
            for t in ("team_offense_stats", "team_defense_stats")}


@pytest.fixture(scope="module")
def schema3_then_upgraded(tmp_path_factory):
    """A copy of the schema-4 database made into a v0.7.1 one (the new columns dropped,
    version 3), then upgraded without the network."""
    if not DB4.exists():
        pytest.skip(f"{DB4} missing")
    src = duckdb.connect(str(DB4), read_only=True)
    try:
        cols = {r[0] for r in src.execute("DESCRIBE team_offense_stats").fetchall()}
        if "net_pass_yards" not in cols:
            pytest.skip(f"{DB4} has no team_offense_stats.net_pass_yards (schema 4 not built)")
        original = _team_tables(src)
    finally:
        src.close()
    path = tmp_path_factory.mktemp("v072") / "v3.duckdb"
    shutil.copyfile(DB4, path)
    con = duckdb.connect(str(path))
    try:
        _make_stale(con)
        # Wrong values in the columns v0.7.2 redefined, so the upgrade must recompute them.
        con.execute("UPDATE team_offense_stats SET total_plays = -1, total_yards = -1, "
                    "yards_per_play = -1, rush_yards = -1, yards_per_rush = -1")
        con.execute("UPDATE team_defense_stats SET plays_against = -1, yards_allowed = -1, "
                    "yards_per_play_allowed = -1")
        stale = _team_tables(con)
        con.execute("DELETE FROM _schema_version WHERE version > 3")
        con.execute("INSERT INTO _schema_version VALUES (3, TIMESTAMP '2026-11-06', '0.7.1')")
        before = migrate.check(con)
        with contextlib.redirect_stdout(io.StringIO()):
            result = migrate.upgrade(con, fetch=False)
        upgraded = _team_tables(con)
        version = con.execute("SELECT MAX(version) FROM _schema_version").fetchone()[0]
    finally:
        con.close()
    return {"original": original, "upgraded": upgraded, "before": before, "result": result,
            "version": version, "stale": stale}


@pytest.mark.integration
def test_v3_database_needs_the_upgrade_without_network(schema3_then_upgraded):
    before = schema3_then_upgraded["before"]
    assert (before.version, before.target, before.upgrade_needed) == (3, 4, True)
    needed = [s for s in before.steps if s.needed]
    assert [s.name for s in needed] == ["derived_tables"], needed
    assert not needed[0].needs_network


@pytest.mark.integration
def test_upgrade_from_v3_stamps_4_and_rebuilds_the_same_team_tables(schema3_then_upgraded):
    out = schema3_then_upgraded
    assert out["result"].stamped and not out["result"].failed, out["result"]
    assert out["version"] == 4
    assert out["upgraded"] == out["original"]
    # Every row held -1 in the redefined columns before the upgrade.
    assert all(-1 in row for rows in out["stale"].values() for row in rows)
    assert not any(-1 in row[3:6] for rows in out["upgraded"].values() for row in rows)


@pytest.mark.integration
def test_real_schema3_database_answers_with_a_warning(monkeypatch):
    # The v0.7.1 database as the container slot holds it until the swap.
    if not DB3.exists():
        pytest.skip(f"{DB3} missing")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(DB3))
    with get_db_connection() as conn:
        cols = {r[0] for r in conn.execute("DESCRIBE team_offense_stats").fetchall()}
    if "net_pass_yards" in cols:
        pytest.skip(f"{DB3} already has the schema-4 columns")
    res = tools.nfl_team_stats(team="BAL", season=2024, season_type="REG")
    off = next(r for r in res["offense"] if r["season_type"] == "REG")
    assert (off["pass_yards"], off["total_yards"]) == (4189, 7213)
    _assert_stale_warning(res["warning"])
