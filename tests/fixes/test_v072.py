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
/tmp/ci2024/v072.duckdb (a copy of v071.duckdb upgraded by this nfl-mcp). The schema-3 tests
read /tmp/ci2024/v071.duckdb only. Each skips with a reason when its file is missing. The unit
tests build small databases in tmp_path and need no file.

What the gap between total_yards and rush_yards + net_pass_yards is, measured on 2024 plays:
- Yards after a lateral on a run (lateral_rushing_yards): 3 plays, 19 yards. They are in
  total_yards (yards_gained) and were not in rush_yards. rush_yards now counts them, as the
  player tools and nflverse team_stats_raw do.
- Kneels: play_type 'qb_kneel' with rush_attempt=1, 437 plays. rush_yards counts them (as the
  official rushing figure does); total_yards and total_plays leave them out (pass and run
  plays only, since group F). This part of the gap stays and the definitions say so.
- Two plays where yards_gained holds only part of a lateral play with an enforced penalty
  (BAL REG -25, NE REG -3). The source data differs; nothing to fix here.

Interface contract tested here
- team_offense_stats gains sack_yards (yards lost on sacks, positive) and net_pass_yards
  (pass_yards - sack_yards). pass_yards stays gross. sacks stays. rush_yards counts rushing
  yards plus lateral_rushing_yards on rush plays.
- team_defense_stats gains rush_yards_allowed, pass_yards_allowed (gross), sack_yards (yards
  the defense gained on sacks, positive) and net_pass_yards_allowed. yards_allowed and sacks
  keep their names and meaning.
- On both tables total - (rush + net pass) equals minus the kneel yards, except where the
  source yards_gained differs on a play.
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
DB4 = Path(_ENV_DB or "/tmp/ci2024/v072.duckdb")
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
#   total_yards         20 + 0 - 7 - 3 + 10 + 10 + 15    = 45 (pass and run plays: no kneel,
#                       spike, two-point try or no_play)
#   total - (rush + net) = 45 - 44 = 1 = minus the kneel yards
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


def _team_db(path=":memory:"):
    con = duckdb.connect(str(path))
    ddl = ", ".join(f'"{c}" {t}' for c, t in _PLAY_COLS.items())
    con.execute(f"CREATE TABLE plays ({ddl})")
    marks = ", ".join("?" for _ in _PLAY_COLS)
    con.executemany(f"INSERT INTO plays VALUES ({marks})",
                    [[r.get(c) for c in _PLAY_COLS] for r in _PLAYS])
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


def test_offense_total_is_rush_plus_net_except_kneels():
    row = _row(_team_db(), "team_offense_stats", "BAL")
    assert row["total_yards"] == 45
    # The one kneel (-1) is in rush_yards and not in total_yards.
    assert row["total_yards"] - (row["rush_yards"] + row["net_pass_yards"]) == 1


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
    assert row["yards_allowed"] == 45
    assert row["rush_yards_allowed"] == 19
    assert row["pass_yards_allowed"] == 35
    assert row["sacks"] == 2
    assert row["sack_yards"] == 10
    assert row["net_pass_yards_allowed"] == 25
    assert row["yards_allowed"] - (row["rush_yards_allowed"] + row["net_pass_yards_allowed"]) == 1


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
    assert set(NEW_OFFENSE) | {"sacks"} <= set(REQUIRED_COLUMNS["team_offense_stats"])
    assert set(NEW_DEFENSE) | {"sacks"} <= set(REQUIRED_COLUMNS["team_defense_stats"])


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
    for field in ("total_yards", "rush_yards", "pass_yards", "sacks", *NEW_OFFENSE):
        assert field in defs["offense"], field
    for field in ("yards_allowed", "sacks", *NEW_DEFENSE):
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
    assert (off["pass_yards"], off["total_yards"], off["sacks"]) == (35, 45, 2)
    assert not set(NEW_OFFENSE) & set(off)
    assert not set(NEW_DEFENSE) & set(de)
    _assert_stale_warning(res["warning"])


def test_stale_team_stats_definitions_leave_out_missing_fields(stale_team_db):
    defs = tools.nfl_team_stats(team="BAL", season=2024, season_type="REG")["definitions"]
    assert not set(NEW_OFFENSE) & set(defs["offense"])
    assert not set(NEW_DEFENSE) & set(defs["defense"])
    assert "pass_yards" in defs["offense"] and "total_yards" in defs["offense"]
    # rush_yards in a schema-3 table has no lateral yards, so its schema-4 definition is left out.
    assert "rush_yards" not in defs["offense"]


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
#     sum(yards_gained) FILTER (WHERE play_type IN ('pass','run')
#                               AND COALESCE(two_point_attempt,0)=0)                    -- total_yards
#   FROM plays WHERE season=2024 AND season_type='REG' GROUP BY posteam
#   BAL 24 154 4189 3189 7213 | PHI 45 323 3517 3048 6266 | DET 33 244 4718 2488 6981
#   KC  41 266 4046 1790 5602.  net = pass - sack: BAL 4035, PHI 3194, DET 4474, KC 3780.
_OFFENSE_SQL = """
    SELECT
        count(*) FILTER (WHERE sack = 1 AND COALESCE(two_point_attempt, 0) = 0
                         AND play_type IS DISTINCT FROM 'no_play'),
        -sum(yards_gained) FILTER (WHERE sack = 1 AND COALESCE(two_point_attempt, 0) = 0
                                   AND play_type IS DISTINCT FROM 'no_play'),
        sum(passing_yards) FILTER (WHERE pass_attempt = 1 AND COALESCE(sack, 0) = 0
                                   AND COALESCE(two_point_attempt, 0) = 0
                                   AND play_type IS DISTINCT FROM 'no_play'),
        sum(COALESCE(rushing_yards, 0) + COALESCE(lateral_rushing_yards, 0)) FILTER (
            WHERE rush_attempt = 1 AND COALESCE(two_point_attempt, 0) = 0
            AND play_type IS DISTINCT FROM 'no_play'),
        sum(yards_gained) FILTER (WHERE play_type IN ('pass', 'run')
                                  AND COALESCE(two_point_attempt, 0) = 0)
    FROM plays WHERE season = 2024 AND season_type = 'REG' AND {side} = ?
"""


@pytest.mark.integration
@pytest.mark.parametrize("team, sacks, sack_yards, pass_yards, rush_yards, total", [
    ("BAL", 24, 154, 4189, 3189, 7213),
    ("PHI", 45, 323, 3517, 3048, 6266),
    ("DET", 33, 244, 4718, 2488, 6981),
    ("KC", 41, 266, 4046, 1790, 5602),
])
def test_offense_net_passing_matches_plays(real4, team, sacks, sack_yards, pass_yards,
                                           rush_yards, total):
    expected = (sacks, sack_yards, pass_yards, rush_yards, total)
    assert real4.execute(_OFFENSE_SQL.format(side="posteam"), [team]).fetchone() == expected
    res = tools.nfl_team_stats(team=team, season=2024, side="offense", season_type="REG")
    (row,) = res["offense"]
    assert (row["sacks"], row["sack_yards"], row["pass_yards"], row["rush_yards"],
            row["total_yards"]) == expected
    assert row["net_pass_yards"] == pass_yards - sack_yards
    assert "warning" not in res


# Defense, BAL 2024 REG, from plays (the _OFFENSE_SQL sets with defteam = 'BAL'):
#   sacks 54, sack_yards 318, pass_yards_allowed 4468, rush_yards_allowed 1361, yards_allowed 5521
#   net_pass_yards_allowed 4468 - 318 = 4150
@pytest.mark.integration
def test_defense_net_passing_allowed_matches_plays(real4):
    expected = (54, 318, 4468, 1361, 5521)
    assert real4.execute(_OFFENSE_SQL.format(side="defteam"), ["BAL"]).fetchone() == expected
    res = tools.nfl_team_stats(team="BAL", season=2024, side="defense", season_type="REG")
    (row,) = res["defense"]
    assert (row["sacks"], row["sack_yards"], row["pass_yards_allowed"],
            row["rush_yards_allowed"], row["yards_allowed"]) == expected
    assert row["net_pass_yards_allowed"] == 4150


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


# The rest of the gap, every 2024 team and season type:
#   total_yards - (rush_yards + net_pass_yards) + kneel yards, where kneel yards is
#   SELECT sum(yards_gained) FROM plays WHERE play_type='qb_kneel' AND rush_attempt=1
#     AND COALESCE(two_point_attempt,0)=0 AND posteam=? AND season_type=?
#   is 0 except on two plays where yards_gained holds only part of a lateral play with an
#   enforced penalty:
#   SELECT posteam, yards_gained, COALESCE(rushing_yards,0)+COALESCE(lateral_rushing_yards,0),
#          passing_yards FROM plays WHERE season=2024 AND game_id IN (...) ...
#   BAL REG: run 25 + lateral 5, yards_gained 5 (-25). NE REG: pass 7, yards_gained 4 (-3).
@pytest.mark.integration
def test_remaining_gap_is_kneels_and_two_source_plays(real4):
    kneels = {(t, st): y for t, st, y in real4.execute(
        "SELECT posteam, season_type, sum(yards_gained) FROM plays WHERE season = 2024 "
        "AND play_type = 'qb_kneel' AND rush_attempt = 1 AND COALESCE(two_point_attempt, 0) = 0 "
        "GROUP BY ALL").fetchall()}
    rows = real4.execute(
        "SELECT team, season_type, total_yards, rush_yards, net_pass_yards FROM team_offense_stats "
        "WHERE season_year = 2024").fetchall()
    assert len(rows) == 46  # 32 REG + 14 POST
    gaps = {(t, st): tot - (ru + net) + kneels.get((t, st), 0) for t, st, tot, ru, net in rows}
    assert {k: v for k, v in gaps.items() if v} == {("BAL", "REG"): -25, ("NE", "REG"): -3}


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
            "version": version}


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
