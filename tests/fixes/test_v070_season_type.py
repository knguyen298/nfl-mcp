"""v0.7.0 decisions 5 and 7: season_type is required, ALL and the playoff round codes, the
season_type split of the derived player tables (SCHEMA_VERSION 2), snap share as a percent
and integer season and week.

Source: SPEC_v0.7.0_unified_arguments.md, decisions 5 and 7.

Every expected value comes from SQL on the raw tables (plays, schedules, snap_counts,
injuries, ff_opportunity, ftn_charting, contracts), never from a tool or a derived table. The
SQL sits in a comment next to the value. Values were measured on a 2024-only database (the
shape GitHub CI ingests). The integration tests read the database at NFL_MCP_DB_PATH (default
/tmp/ci2024/v070.duckdb) read-only and skip when it is missing or has no 2024 playoff plays.
They do not skip on derived tables built by an older nfl-mcp: those must be upgraded first.

Interface contract tested here
- Season-total tools (nfl_team_stats, nfl_player_stats, nfl_compare, nfl_td_luck,
  nfl_drop_rate, nfl_separation_opportunity, nfl_contract_value): season_type enum is exactly
  [REG, POST, ALL]. Per-game tools (nfl_schedule, nfl_snap_counts, nfl_search_plays,
  nfl_injuries, nfl_ftn_charting, nfl_fantasy_opportunity, nfl_role_trend): exactly
  [REG, POST, ALL, WC, DIV, CON, SB]. season_type is in "required", has no default, and its
  description names every value and says it is required.
- A call without season_type, or with any other value, is an error. Over MCP the SDK schema
  check rejects it; the tool code rejects it too (server.call_tool without the SDK check), and
  it does so before it reads the database.
- ALL gives the REG rows and the POST rows, each labeled with season_type; no row is a
  combined total. ALL equals the union of the REG call and the POST call.
- A round code gives exactly the games of that round (schedules.game_type); plays and
  ff_opportunity get the round from schedules through game_id.
- player_role_trend, player_separation_opportunity and player_contract_value gain a
  season_type column (REQUIRED_COLUMNS lists it). The role_trend window is per season_type.
- migrate.SCHEMA_VERSION == 2. A version 1 database whose derived tables lack those columns
  is detected and fixed by `nfl-mcp upgrade` without the network.
- Snap shares are percents (0 to 100). season, season_year and week are Python ints.

Player ids used (players table): Saquon Barkley 00-0034844 (pfr BarkSa00), Travis Kelce
00-0030506 (pfr KelcTr00), A.J. Brown 00-0035676 (pfr BrowAJ00). Ids, not names, so the
name-matching change (decision 6) does not affect these tests.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
from pathlib import Path

import anyio
import duckdb
import pytest
from click.testing import CliRunner
from mcp import types

from nfl_mcp import ingest as ing
from nfl_mcp import migrate, server, tools
from nfl_mcp.cli import main
from nfl_mcp.database import get_db_connection
from nfl_mcp.required_columns import REQUIRED_COLUMNS
from nfl_mcp.validate import run_checks

REAL_DB = Path(os.environ.get("NFL_MCP_DB_PATH", "/tmp/ci2024/v070.duckdb"))

SEASON_TOTAL = ["nfl_team_stats", "nfl_player_stats", "nfl_compare", "nfl_td_luck",
                "nfl_drop_rate", "nfl_separation_opportunity", "nfl_contract_value"]
PER_GAME = ["nfl_schedule", "nfl_snap_counts", "nfl_search_plays", "nfl_injuries",
            "nfl_ftn_charting", "nfl_fantasy_opportunity", "nfl_role_trend"]
NO_SEASON_TYPE = ["nfl_team_records", "nfl_roster", "nfl_player_lookup", "nfl_fantasy_rankings",
                  "nfl_injury_return", "nfl_schema", "nfl_status", "nfl_catalog", "nfl_query"]
TOTAL_ENUM = ["REG", "POST", "ALL"]
GAME_ENUM = ["REG", "POST", "ALL", "WC", "DIV", "CON", "SB"]
ENUM = {**{t: TOTAL_ENUM for t in SEASON_TOTAL}, **{t: GAME_ENUM for t in PER_GAME}}

BARKLEY, KELCE, AJ_BROWN = "00-0034844", "00-0030506", "00-0035676"

# The other arguments a call needs, so that season_type is the only thing wrong with it.
# New v0.7.0 names (decisions 2 and 3): player_id, team1/team2.
BASE_ARGS = {
    "nfl_team_stats": {"team": "PHI"},
    "nfl_player_stats": {"player_id": BARKLEY},
    "nfl_compare": {"team1": "PHI", "team2": "KC"},
    "nfl_td_luck": {}, "nfl_drop_rate": {}, "nfl_separation_opportunity": {},
    "nfl_contract_value": {},
    **{t: {"season": 2024} for t in PER_GAME},
}

# Derived tables that gain season_type in v0.7.0 (the tables nfl_role_trend,
# nfl_separation_opportunity and nfl_contract_value read).
V2_TABLES = ["player_role_trend", "player_separation_opportunity", "player_contract_value"]


# ── helpers ────────────────────────────────────────────────────────────────────

def _handle(req):
    async def go():
        return await server._mcp_server.request_handlers[type(req)](req)
    return anyio.run(go).root


def _call(name, args):
    """A tool call through the MCP handler, with the SDK's inputSchema check."""
    return _handle(types.CallToolRequest(method="tools/call",
                                         params=types.CallToolRequestParams(name=name, arguments=args)))


def _raw(name, args):
    """A tool call through server.call_tool only: the tool code's own checks, no SDK check."""
    async def go():
        return await server.call_tool(name, args)
    return anyio.run(go)


def _text(res):
    return " ".join(c.text for c in res.content if getattr(c, "text", None))


def _ok(name, args):
    res = _call(name, args)
    assert not res.isError, _text(res)
    return json.loads(res.content[0].text)


def _all(name, key, args):
    """Every row a paged tool returns, 500 at a time."""
    rows, offset = [], 0
    while True:
        page = _ok(name, {**args, "limit": 500, "offset": offset})
        rows += page[key]
        if not page["truncated"]:
            return rows
        offset += 500


def _canon(rows):
    return sorted(json.dumps(r, sort_keys=True) for r in rows)


def _word(text, word):
    return re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text) is not None


def _schema(name):
    tool = next(t for t in _handle(types.ListToolsRequest(method="tools/list")).tools if t.name == name)
    return tool.inputSchema


def _labels(rows):
    return sorted(r["season_type"] for r in rows)


@pytest.fixture
def no_db(tmp_path, monkeypatch):
    """Point the server at a file that does not exist: these checks must not need data."""
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(tmp_path / "absent.duckdb"))


@pytest.fixture
def real(monkeypatch):
    """Oracle connection through get_db_connection (the same config the tools open)."""
    if not REAL_DB.exists():
        pytest.skip(f"{REAL_DB} missing")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(REAL_DB))
    with get_db_connection() as conn:
        n = conn.execute("SELECT COUNT(*) FROM plays WHERE season = 2024 AND season_type = 'POST'"
                         ).fetchone()[0]
        if not n:
            pytest.skip("needs 2024 playoff plays")
        yield conn


def _q(conn, sql, params=()):
    return conn.execute(sql, list(params)).fetchall()


# ═══ 1. inputSchema ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", SEASON_TOTAL + PER_GAME)
def test_season_type_is_required(name):
    assert "season_type" in _schema(name).get("required", []), name


@pytest.mark.parametrize("name", SEASON_TOTAL)
def test_season_total_enum_is_reg_post_all(name):
    assert _schema(name)["properties"]["season_type"]["enum"] == TOTAL_ENUM


@pytest.mark.parametrize("name", PER_GAME)
def test_per_game_enum_has_the_round_codes(name):
    assert _schema(name)["properties"]["season_type"]["enum"] == GAME_ENUM


@pytest.mark.parametrize("name", SEASON_TOTAL + PER_GAME)
def test_season_type_has_no_default(name):
    prop = _schema(name)["properties"]["season_type"]
    assert prop.get("type") == "string"
    assert "default" not in prop, prop


@pytest.mark.parametrize("name", SEASON_TOTAL + PER_GAME)
def test_season_type_description_names_values_and_required(name):
    text = _schema(name)["properties"]["season_type"]["description"]
    missing = [v for v in ENUM[name] if not _word(text, v)]
    assert not missing, (name, missing, text)
    assert "required" in text.lower(), text


@pytest.mark.parametrize("name", SEASON_TOTAL)
def test_season_total_description_does_not_offer_round_codes(name):
    text = _schema(name)["properties"]["season_type"]["description"]
    # A round may be named only to say it is not valid here.
    for code in ("WC", "DIV", "CON", "SB"):
        if _word(text, code):
            assert "not" in text.lower(), text


@pytest.mark.parametrize("name", NO_SEASON_TYPE)
def test_tools_without_season_type_data_take_no_season_type(name):
    assert "season_type" not in _schema(name).get("properties", {}), name


def test_tool_lists_cover_every_tool():
    listed = {t.name for t in _handle(types.ListToolsRequest(method="tools/list")).tools}
    assert listed == set(SEASON_TOTAL + PER_GAME + NO_SEASON_TYPE)


# ═══ 2. errors over MCP and in the tool code ═══════════════════════════════════

@pytest.mark.parametrize("name", SEASON_TOTAL + PER_GAME)
def test_mcp_call_without_season_type_is_an_error(no_db, name):
    res = _call(name, dict(BASE_ARGS[name]))
    assert res.isError, _text(res)
    assert "season_type" in _text(res), _text(res)


BAD_VALUES = ["Regular Season", "reg", "playoffs", "Post", "all", ""]


@pytest.mark.parametrize("value", BAD_VALUES)
@pytest.mark.parametrize("name", SEASON_TOTAL + PER_GAME)
def test_mcp_bad_season_type_lists_valid_values(no_db, name, value):
    res = _call(name, {**BASE_ARGS[name], "season_type": value})
    assert res.isError, _text(res)
    missing = [v for v in ENUM[name] if not _word(_text(res), v)]
    assert not missing, (missing, _text(res))


@pytest.mark.parametrize("code", ["WC", "DIV", "CON", "SB"])
@pytest.mark.parametrize("name", SEASON_TOTAL)
def test_mcp_round_code_on_season_total_tool_is_an_error(no_db, name, code):
    res = _call(name, {**BASE_ARGS[name], "season_type": code})
    assert res.isError, _text(res)
    missing = [v for v in TOTAL_ENUM if not _word(_text(res), v)]
    assert not missing, (missing, _text(res))


@pytest.mark.parametrize("name", SEASON_TOTAL + PER_GAME)
def test_tool_code_without_season_type_is_an_error(no_db, name):
    # Without the SDK check (a client that skips it, or a direct dispatch). The error names
    # season_type and says it is missing or required, before any database read.
    res = _raw(name, dict(BASE_ARGS[name]))
    assert res.isError, _text(res)
    text = _text(res)
    assert "season_type" in text and re.search(r"required|missing", text, re.IGNORECASE), text


@pytest.mark.parametrize("value", ["reg", "Regular Season", "playoffs", "post"])
@pytest.mark.parametrize("name", SEASON_TOTAL + PER_GAME)
def test_tool_code_is_strict_and_lists_valid_values(no_db, name, value):
    # Decision 5: values are exact and upper case; 'reg' is not REG.
    res = _raw(name, {**BASE_ARGS[name], "season_type": value})
    assert res.isError, _text(res)
    missing = [v for v in ENUM[name] if not _word(_text(res), v)]
    assert not missing, (missing, _text(res))


@pytest.mark.parametrize("name", SEASON_TOTAL)
def test_tool_code_rejects_round_code_on_season_total_tool(no_db, name):
    res = _raw(name, {**BASE_ARGS[name], "season_type": "SB"})
    assert res.isError, _text(res)
    missing = [v for v in TOTAL_ENUM if not _word(_text(res), v)]
    assert not missing, (missing, _text(res))


# ═══ 3. ALL = REG rows + POST rows, values from raw SQL ═════════════════════════

@pytest.mark.integration
def test_team_stats_all_has_one_reg_and_one_post_row_with_plays_yards(real):
    # SELECT season_type,
    #        COALESCE(SUM(COALESCE(yards_gained,0)) FILTER (WHERE play_type IN ('pass','run')
    #                 AND COALESCE(two_point_attempt,0)=0), 0),
    #        COUNT(*) FILTER (WHERE play_type IN ('pass','run') AND COALESCE(two_point_attempt,0)=0)
    # FROM plays WHERE posteam='PHI' AND season=2024 GROUP BY 1
    #   -> REG 6266, 1087; POST 1453, 247
    sql = ("SELECT season_type, COALESCE(SUM(COALESCE(yards_gained,0)) FILTER (WHERE play_type "
           "IN ('pass','run') AND COALESCE(two_point_attempt,0)=0), 0), COUNT(*) FILTER (WHERE "
           "play_type IN ('pass','run') AND COALESCE(two_point_attempt,0)=0) FROM plays "
           "WHERE posteam='PHI' AND season=2024 GROUP BY 1 ORDER BY 1")
    assert _q(real, sql) == [("POST", 1453.0, 247), ("REG", 6266.0, 1087)]
    res = _ok("nfl_team_stats", {"team": "PHI", "season": 2024, "season_type": "ALL"})
    got = sorted((r["season_type"], r["total_yards"], r["total_plays"]) for r in res["offense"])
    assert got == [("POST", 1453, 247), ("REG", 6266, 1087)]


@pytest.mark.integration
@pytest.mark.parametrize("side", ["offense", "defense"])
def test_team_stats_all_labels_each_row_and_never_combines(real, side):
    res = _ok("nfl_team_stats", {"team": "PHI", "season": 2024, "season_type": "ALL"})
    assert _labels(res[side]) == ["POST", "REG"]


@pytest.mark.integration
@pytest.mark.parametrize("side", ["offense", "defense", "situational"])
def test_team_stats_all_is_union_of_reg_and_post(real, side):
    args = {"team": "PHI", "season": 2024}
    every = _ok("nfl_team_stats", {**args, "season_type": "ALL"})[side]
    reg = _ok("nfl_team_stats", {**args, "season_type": "REG"})[side]
    post = _ok("nfl_team_stats", {**args, "season_type": "POST"})[side]
    assert reg and post
    assert {r["season_type"] for r in reg} == {"REG"} and {r["season_type"] for r in post} == {"POST"}
    assert _canon(every) == _canon(reg + post)


# SELECT season_type,
#        COUNT(*) FILTER (WHERE rusher_player_id='00-0034844' AND rush_attempt=1
#                         AND COALESCE(two_point_attempt,0)=0),
#        SUM(rushing_yards) FILTER (WHERE rusher_player_id='00-0034844' AND rush_attempt=1
#                                   AND COALESCE(two_point_attempt,0)=0),
#        SUM(lateral_rushing_yards) FILTER (WHERE lateral_rusher_player_id='00-0034844')
# FROM plays WHERE season=2024 GROUP BY 1
#   -> REG 345, 2005, NULL; POST 91, 499, NULL
BARKLEY_RUSH_SQL = (
    "SELECT season_type, COUNT(*) FILTER (WHERE rusher_player_id='00-0034844' AND rush_attempt=1 "
    "AND COALESCE(two_point_attempt,0)=0), SUM(rushing_yards) FILTER (WHERE "
    "rusher_player_id='00-0034844' AND rush_attempt=1 AND COALESCE(two_point_attempt,0)=0), "
    "SUM(lateral_rushing_yards) FILTER (WHERE lateral_rusher_player_id='00-0034844') "
    "FROM plays WHERE season=2024 GROUP BY 1 ORDER BY 1")
BARKLEY_RUSH = [("POST", 91, 499.0, None), ("REG", 345, 2005.0, None)]


@pytest.mark.integration
def test_player_stats_all_has_reg_and_post_rushing_from_plays(real):
    assert _q(real, BARKLEY_RUSH_SQL) == BARKLEY_RUSH
    res = _ok("nfl_player_stats", {"player_id": BARKLEY, "season": 2024, "season_type": "ALL"})
    got = sorted((s["season_type"], s["rushing"]["carries"], s["rushing"]["yards"])
                 for s in res["seasons"])
    assert got == [("POST", 91, 499), ("REG", 345, 2005)]


@pytest.mark.integration
def test_player_stats_all_is_union_of_reg_and_post(real):
    args = {"player_id": BARKLEY, "season": 2024}
    every = _ok("nfl_player_stats", {**args, "season_type": "ALL"})["seasons"]
    reg = _ok("nfl_player_stats", {**args, "season_type": "REG"})["seasons"]
    post = _ok("nfl_player_stats", {**args, "season_type": "POST"})["seasons"]
    assert [s["season_type"] for s in reg] == ["REG"] and [s["season_type"] for s in post] == ["POST"]
    assert _canon(every) == _canon(reg + post)


@pytest.mark.integration
def test_compare_teams_all_has_reg_and_post_rows(real):
    # PHI from the team_stats SQL above (REG 6266, POST 1453). KC:
    # same SQL with posteam='KC' -> REG 5602, POST 868
    sql = ("SELECT season_type, COALESCE(SUM(COALESCE(yards_gained,0)) FILTER (WHERE play_type "
           "IN ('pass','run') AND COALESCE(two_point_attempt,0)=0), 0) FROM plays "
           "WHERE posteam='KC' AND season=2024 GROUP BY 1 ORDER BY 1")
    assert _q(real, sql) == [("POST", 868.0), ("REG", 5602.0)]
    res = _ok("nfl_compare", {"team1": "PHI", "team2": "KC", "season": 2024, "season_type": "ALL"})
    got = {team: sorted((r["season_type"], r["total_yards"]) for r in res[team]["offense"])
           for team in ("PHI", "KC")}
    assert got == {"PHI": [("POST", 1453), ("REG", 6266)], "KC": [("POST", 868), ("REG", 5602)]}


@pytest.mark.integration
def test_compare_teams_all_is_union_of_reg_and_post(real):
    args = {"team1": "PHI", "team2": "KC", "season": 2024}
    every, reg, post = (_ok("nfl_compare", {**args, "season_type": st}) for st in ("ALL", "REG", "POST"))
    for team in ("PHI", "KC"):
        for side in ("offense", "defense"):
            assert _canon(every[team][side]) == _canon(reg[team][side] + post[team][side])


@pytest.mark.integration
def test_compare_players_all_has_reg_and_post_seasons(real):
    assert _q(real, BARKLEY_RUSH_SQL) == BARKLEY_RUSH
    args = {"player1": BARKLEY, "player2": KELCE, "season": 2024}
    every = _ok("nfl_compare", {**args, "season_type": "ALL"})
    reg = _ok("nfl_compare", {**args, "season_type": "REG"})
    post = _ok("nfl_compare", {**args, "season_type": "POST"})
    by_id = {e["player_id"]: e for e in every["players"]}
    assert sorted((s["season_type"], s["rushing"]["carries"], s["rushing"]["yards"])
                  for s in by_id[BARKLEY]["seasons"]) == [("POST", 91, 499), ("REG", 345, 2005)]
    for i in range(2):
        assert _canon(every["players"][i]["seasons"]) == _canon(
            reg["players"][i]["seasons"] + post["players"][i]["seasons"])


# ff_opportunity with the season type of its game: schedules.game_type through game_id.
FF = ("(SELECT o.*, CASE WHEN s.game_type = 'REG' THEN 'REG' ELSE 'POST' END AS st, s.game_type "
      "FROM ff_opportunity o JOIN schedules s ON s.game_id = o.game_id)")


@pytest.mark.integration
def test_td_luck_all_matches_ff_opportunity(real):
    # SELECT st, SUM(rec_touchdown), SUM(rec_touchdown_exp) FROM <FF> WHERE player_id='00-0030506'
    # GROUP BY st  -> POST 1.0, 0.36; REG 3.0, 5.92
    sql = f"SELECT st, SUM(rec_touchdown), ROUND(SUM(rec_touchdown_exp), 2) FROM {FF} " \
          "WHERE player_id='00-0030506' GROUP BY st ORDER BY st"
    assert _q(real, sql) == [("POST", 1.0, 0.36), ("REG", 3.0, 5.92)]
    rows = _all("nfl_td_luck", "td_luck", {"player_id": KELCE, "season": 2024, "season_type": "ALL"})
    got = sorted((r["season_type"], r["rec_td"], round(r["rec_td_exp"], 2)) for r in rows)
    assert got == [("POST", 1.0, 0.36), ("REG", 3.0, 5.92)]


@pytest.mark.integration
def test_drop_rate_all_matches_ftn_charting(real):
    # SELECT p.season_type, p.posteam, COUNT(*), SUM(CASE WHEN f.is_drop THEN 1 ELSE 0 END)
    # FROM ftn_charting f JOIN plays p ON f.nflverse_game_id=p.game_id AND f.nflverse_play_id=p.play_id
    # WHERE f.is_catchable_ball AND p.receiver_player_id='00-0035676' GROUP BY 1, 2
    #   -> POST PHI 15, 2; REG PHI 71, 2
    sql = ("SELECT p.season_type, p.posteam, COUNT(*), SUM(CASE WHEN f.is_drop THEN 1 ELSE 0 END) "
           "FROM ftn_charting f JOIN plays p ON f.nflverse_game_id=p.game_id AND "
           "f.nflverse_play_id=p.play_id WHERE f.is_catchable_ball AND "
           "p.receiver_player_id='00-0035676' GROUP BY 1, 2 ORDER BY 1")
    assert _q(real, sql) == [("POST", "PHI", 15, 2), ("REG", "PHI", 71, 2)]
    rows = _all("nfl_drop_rate", "drop_rate", {"player_id": AJ_BROWN, "season": 2024, "season_type": "ALL"})
    got = sorted((r["season_type"], r["team"], r["catchable_targets"], r["drops"]) for r in rows)
    assert got == [("POST", "PHI", 15, 2), ("REG", "PHI", 71, 2)]


# A.J. Brown 2024 from ff_opportunity:
# SELECT st, COUNT(DISTINCT week), SUM(total_fantasy_points_diff),
#        ROUND(SUM(total_fantasy_points_diff) / COUNT(DISTINCT week), 3),
#        ROUND(SUM(rec_touchdown_diff), 3),
#        ROUND(100.0 * SUM(rec_attempt) / SUM(rec_attempt_team), 2)
# FROM <FF> WHERE player_id='00-0035676' GROUP BY st
#   -> REG 13, 39.62, 3.048, 1.14, 31.39; POST 4, -4.02, -1.005, 0.2, 25.0
SEPARATION_SQL = (
    f"SELECT st, COUNT(DISTINCT week), ROUND(SUM(total_fantasy_points_diff), 2), "
    f"ROUND(SUM(total_fantasy_points_diff) / COUNT(DISTINCT week), 3), "
    f"ROUND(SUM(rec_touchdown_diff), 3), ROUND(100.0 * SUM(rec_attempt) / SUM(rec_attempt_team), 2) "
    f"FROM {FF} WHERE player_id='00-0035676' GROUP BY st ORDER BY st")
SEPARATION = [("POST", 4, -4.02, -1.005, 0.2, 25.0), ("REG", 13, 39.62, 3.048, 1.14, 31.39)]


@pytest.mark.integration
def test_separation_opportunity_splits_by_season_type(real):
    assert _q(real, SEPARATION_SQL) == SEPARATION
    rows = _all("nfl_separation_opportunity", "separation_opportunity",
                {"player_id": AJ_BROWN, "season": 2024, "season_type": "ALL"})
    got = sorted((r["season_type"], r["games"], round(r["fp_diff"], 2), r["fp_diff_per_game"],
                  r["td_luck"], r["target_share_pct"]) for r in rows)
    assert got == SEPARATION


@pytest.mark.integration
def test_separation_opportunity_reg_excludes_playoff_weeks(real):
    rows = _all("nfl_separation_opportunity", "separation_opportunity",
                {"player_id": AJ_BROWN, "season": 2024, "season_type": "REG"})
    assert [(r["season_type"], r["games"], r["fp_diff_per_game"]) for r in rows] == [("REG", 13, 3.048)]


# Saquon Barkley 2024. Contract in force (latest year_signed <= 2024, larger apy first):
# SELECT year_signed, apy FROM contracts WHERE gsis_id='00-0034844' AND year_signed <= 2024
# ORDER BY year_signed DESC, apy DESC LIMIT 1  -> 2024, 12.583333 (rounded to 4: 12.5833)
# SELECT st, ROUND(SUM(total_fantasy_points), 2), ROUND(SUM(total_fantasy_points_exp), 2),
#        ROUND(SUM(total_fantasy_points) / 12.583333, 2)
# FROM <FF> WHERE player_id='00-0034844' GROUP BY st
#   -> REG 355.3, 287.58, 28.24; POST 100.4, 87.61, 7.98
CONTRACT_SQL = (
    f"SELECT st, ROUND(SUM(total_fantasy_points), 2), ROUND(SUM(total_fantasy_points_exp), 2), "
    f"ROUND(SUM(total_fantasy_points) / (SELECT apy FROM contracts WHERE gsis_id='00-0034844' "
    f"AND year_signed <= 2024 ORDER BY year_signed DESC, apy DESC LIMIT 1), 2) "
    f"FROM {FF} WHERE player_id='00-0034844' GROUP BY st ORDER BY st")
CONTRACT = [("POST", 100.4, 87.61, 7.98), ("REG", 355.3, 287.58, 28.24)]


@pytest.mark.integration
def test_contract_value_splits_by_season_type(real):
    assert _q(real, "SELECT year_signed, ROUND(apy, 4) FROM contracts WHERE gsis_id='00-0034844' "
                    "AND year_signed <= 2024 ORDER BY year_signed DESC, apy DESC LIMIT 1") == [(2024, 12.5833)]
    assert _q(real, CONTRACT_SQL) == CONTRACT
    rows = _all("nfl_contract_value", "contract_value",
                {"player_id": BARKLEY, "season": 2024, "season_type": "ALL"})
    got = sorted((r["season_type"], r["total_fp"], r["total_fp_exp"], r["fp_per_million"]) for r in rows)
    assert got == CONTRACT
    assert {r["apy"] for r in rows} == {12.5833}


@pytest.mark.integration
def test_contract_value_reg_excludes_playoff_weeks(real):
    rows = _all("nfl_contract_value", "contract_value",
                {"player_id": BARKLEY, "season": 2024, "season_type": "REG"})
    assert [(r["season_type"], r["total_fp"]) for r in rows] == [("REG", 355.3)]


TOTAL_UNION = [
    ("nfl_td_luck", "td_luck", {"player_id": KELCE, "season": 2024}),
    ("nfl_td_luck", "td_luck", {"team": "KC", "season": 2024}),
    ("nfl_drop_rate", "drop_rate", {"player_id": AJ_BROWN, "season": 2024}),
    ("nfl_drop_rate", "drop_rate", {"team": "PHI", "season": 2024}),
    ("nfl_separation_opportunity", "separation_opportunity", {"player_id": AJ_BROWN, "season": 2024}),
    ("nfl_separation_opportunity", "separation_opportunity", {"team": "PHI", "season": 2024}),
    ("nfl_contract_value", "contract_value", {"player_id": BARKLEY, "season": 2024}),
    ("nfl_contract_value", "contract_value", {"team": "PHI", "season": 2024}),
]


@pytest.mark.integration
@pytest.mark.parametrize("name,key,args", TOTAL_UNION)
def test_season_total_all_is_union_of_reg_and_post(real, name, key, args):
    every = _all(name, key, {**args, "season_type": "ALL"})
    reg = _all(name, key, {**args, "season_type": "REG"})
    post = _all(name, key, {**args, "season_type": "POST"})
    assert reg and post, (len(reg), len(post))
    assert {r["season_type"] for r in reg} == {"REG"}
    assert {r["season_type"] for r in post} == {"POST"}
    assert {r["season_type"] for r in every} == {"REG", "POST"}  # no combined 'ALL' row
    assert _canon(every) == _canon(reg + post)


PER_GAME_UNION = [
    ("nfl_schedule", "games", {"team": "PHI", "season": 2024}),
    ("nfl_snap_counts", "snap_counts", {"player_id": KELCE, "season": 2024}),
    ("nfl_injuries", "injuries", {"team": "KC", "season": 2024}),
    ("nfl_fantasy_opportunity", "fantasy_opportunity", {"player_id": BARKLEY, "season": 2024}),
    ("nfl_role_trend", "role_trend", {"player_id": KELCE, "season": 2024}),
    ("nfl_search_plays", "rows", {"team": "PHI", "season": 2024, "play_type": "run",
                                  "is_touchdown": True}),
]


@pytest.mark.integration
@pytest.mark.parametrize("name,key,args", PER_GAME_UNION)
def test_per_game_all_is_union_of_reg_and_post(real, name, key, args):
    every = _all(name, key, {**args, "season_type": "ALL"})
    reg = _all(name, key, {**args, "season_type": "REG"})
    post = _all(name, key, {**args, "season_type": "POST"})
    assert reg and post, (len(reg), len(post))
    assert _canon(every) == _canon(reg + post)


@pytest.mark.integration
@pytest.mark.parametrize("name,key,args", [
    ("nfl_schedule", "games", {"team": "PHI", "season": 2024}),
    ("nfl_snap_counts", "snap_counts", {"player_id": KELCE, "season": 2024}),
])
def test_schedule_and_snap_count_rows_carry_season_type(real, name, key, args):
    # Decision 5: every row is labeled with season_type. game_type REG -> REG; WC, DIV, CON
    # and SB -> POST. SELECT DISTINCT game_type FROM schedules -> REG, WC, DIV, CON, SB.
    rows = _all(name, key, {**args, "season_type": "ALL"})
    assert rows
    assert all(r["season_type"] == ("REG" if r["game_type"] == "REG" else "POST") for r in rows)
    assert {r["season_type"] for r in rows} == {"REG", "POST"}
    post = _all(name, key, {**args, "season_type": "DIV"})
    assert post and {r["season_type"] for r in post} == {"POST"}


@pytest.mark.integration
def test_per_game_all_counts_match_raw_tables(real):
    # SELECT count(*) FROM schedules WHERE season=2024 AND (home_team='PHI' OR away_team='PHI') -> 21
    assert _q(real, "SELECT count(*) FROM schedules WHERE season=2024 AND "
                    "(home_team='PHI' OR away_team='PHI')") == [(21,)]
    assert len(_all("nfl_schedule", "games", {"team": "PHI", "season": 2024, "season_type": "ALL"})) == 21
    # SELECT count(*) FROM snap_counts WHERE pfr_player_id='KelcTr00' -> 19
    assert _q(real, "SELECT count(*) FROM snap_counts WHERE pfr_player_id='KelcTr00'") == [(19,)]
    assert len(_all("nfl_snap_counts", "snap_counts",
                    {"player_id": KELCE, "season": 2024, "season_type": "ALL"})) == 19
    # SELECT count(*) FROM injuries WHERE team='KC' AND season=2024 -> 162
    assert _q(real, "SELECT count(*) FROM injuries WHERE team='KC' AND season=2024") == [(162,)]
    assert len(_all("nfl_injuries", "injuries", {"team": "KC", "season": 2024, "season_type": "ALL"})) == 162
    # SELECT count(*) FROM plays WHERE season=2024 AND posteam='PHI' AND play_type='run' AND touchdown=1 -> 41
    assert _q(real, "SELECT count(*) FROM plays WHERE season=2024 AND posteam='PHI' AND "
                    "play_type='run' AND touchdown=1") == [(41,)]
    assert len(_all("nfl_search_plays", "rows", {"team": "PHI", "season": 2024, "play_type": "run",
                                                 "is_touchdown": True, "season_type": "ALL"})) == 41


@pytest.mark.integration
def test_ftn_charting_reg_and_post_totals_match_raw(real):
    # SELECT p.season_type, count(*) FROM ftn_charting f JOIN plays p ON f.nflverse_game_id=p.game_id
    # AND f.nflverse_play_id=p.play_id WHERE p.play_type IN ('pass','run') AND p.posteam='PHI'
    # GROUP BY 1  -> POST 248, REG 1093
    sql = ("SELECT p.season_type, count(*) FROM ftn_charting f JOIN plays p ON "
           "f.nflverse_game_id=p.game_id AND f.nflverse_play_id=p.play_id WHERE p.play_type IN "
           "('pass','run') AND p.posteam='PHI' GROUP BY 1 ORDER BY 1")
    assert _q(real, sql) == [("POST", 248), ("REG", 1093)]
    for st, n in (("REG", 1093), ("POST", 248)):
        res = _ok("nfl_ftn_charting", {"team": "PHI", "season": 2024, "season_type": st})
        assert res["total_plays"] == n, st


# ═══ 4. round codes on per-game tools ══════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.parametrize("code,n", [("WC", 6), ("DIV", 4), ("CON", 2), ("SB", 1)])
def test_schedule_round_code_returns_that_round(real, code, n):
    # SELECT game_type, count(*) FROM schedules WHERE season=2024 GROUP BY 1
    #   -> WC 6, DIV 4, CON 2, SB 1 (REG 272)
    want = sorted(r[0] for r in _q(real, "SELECT game_id FROM schedules WHERE season=2024 "
                                         "AND game_type=?", [code]))
    assert len(want) == n
    games = _all("nfl_schedule", "games", {"season": 2024, "season_type": code})
    assert sorted(g["game_id"] for g in games) == want
    assert {g["game_type"] for g in games} == {code}


@pytest.mark.integration
def test_schedule_sb_is_super_bowl_lix(real):
    # SELECT game_id, away_team, home_team, away_score, home_score FROM schedules
    # WHERE season=2024 AND game_type='SB'  -> 2024_22_KC_PHI, KC 22, PHI 40
    assert _q(real, "SELECT game_id, away_team, home_team, away_score, home_score FROM schedules "
                    "WHERE season=2024 AND game_type='SB'") == [("2024_22_KC_PHI", "KC", "PHI", 22, 40)]
    games = _ok("nfl_schedule", {"season": 2024, "season_type": "SB"})["games"]
    assert [(g["game_id"], g["away_score"], g["home_score"]) for g in games] == [("2024_22_KC_PHI", 22, 40)]


@pytest.mark.integration
def test_schedule_post_is_the_four_rounds(real):
    # SELECT count(*) FROM schedules WHERE season=2024 AND game_type IN ('WC','DIV','CON','SB') -> 13
    assert _q(real, "SELECT count(*) FROM schedules WHERE season=2024 AND "
                    "game_type IN ('WC','DIV','CON','SB')") == [(13,)]
    games = _all("nfl_schedule", "games", {"season": 2024, "season_type": "POST"})
    assert len(games) == 13 and {g["game_type"] for g in games} == {"WC", "DIV", "CON", "SB"}


@pytest.mark.integration
def test_snap_counts_sb_returns_only_super_bowl_rows(real):
    # SELECT count(*) FROM snap_counts WHERE season=2024 AND game_type='SB' -> 92
    want = sorted(_q(real, "SELECT pfr_player_id, team FROM snap_counts WHERE season=2024 "
                           "AND game_type='SB'"))
    assert len(want) == 92
    rows = _all("nfl_snap_counts", "snap_counts", {"season": 2024, "season_type": "SB"})
    assert sorted((r["pfr_player_id"], r["team"]) for r in rows) == want
    assert {r["game_type"] for r in rows} == {"SB"} and {r["week"] for r in rows} == {22}


@pytest.mark.integration
def test_search_plays_con_returns_only_the_two_conference_games(real):
    # plays.season_type is REG or POST only; the round comes from schedules through game_id.
    # SELECT game_id FROM schedules WHERE season=2024 AND game_type='CON'
    #   -> 2024_21_BUF_KC, 2024_21_WAS_PHI
    # SELECT count(*) FROM plays WHERE game_id IN (those two) -> 356
    want = sorted(_q(real, "SELECT game_id, play_id FROM plays WHERE game_id IN (SELECT game_id "
                           "FROM schedules WHERE season=2024 AND game_type='CON')"))
    assert len(want) == 356
    rows = _all("nfl_search_plays", "rows", {"season": 2024, "season_type": "CON"})
    assert {r["game_id"] for r in rows} == {"2024_21_BUF_KC", "2024_21_WAS_PHI"}
    assert sorted((r["game_id"], r["play_id"]) for r in rows) == want


@pytest.mark.integration
def test_injuries_div_returns_only_divisional_rows(real):
    # SELECT count(*) FROM injuries WHERE season=2024 AND game_type='DIV' -> 74
    want = sorted(_q(real, "SELECT full_name, team, week FROM injuries WHERE season=2024 "
                           "AND game_type='DIV'"))
    assert len(want) == 74
    rows = _all("nfl_injuries", "injuries", {"season": 2024, "season_type": "DIV"})
    assert sorted((r["full_name"], r["team"], r["week"]) for r in rows) == want


@pytest.mark.integration
def test_ftn_charting_wc_counts_wild_card_plays_only(real):
    # SELECT s.game_type, count(*) FROM ftn_charting f JOIN plays p ON f.nflverse_game_id=p.game_id
    # AND f.nflverse_play_id=p.play_id JOIN schedules s ON s.game_id=p.game_id
    # WHERE p.play_type IN ('pass','run') GROUP BY 1
    #   -> WC 694, DIV 492, CON 275, SB 119, REG 33470
    sql = ("SELECT count(*) FROM ftn_charting f JOIN plays p ON f.nflverse_game_id=p.game_id AND "
           "f.nflverse_play_id=p.play_id JOIN schedules s ON s.game_id=p.game_id "
           "WHERE p.play_type IN ('pass','run') AND s.game_type=?")
    for code, n in (("WC", 694), ("SB", 119)):
        assert _q(real, sql, [code]) == [(n,)]
        assert _ok("nfl_ftn_charting", {"season": 2024, "season_type": code})["total_plays"] == n, code


@pytest.mark.integration
def test_fantasy_opportunity_sb_returns_only_super_bowl_rows(real):
    # SELECT count(*), count(DISTINCT player_id) FROM ff_opportunity o JOIN schedules s
    # ON s.game_id=o.game_id WHERE s.game_type='SB' -> 19, 19
    want = sorted(r[0] for r in _q(real, "SELECT o.player_id FROM ff_opportunity o JOIN schedules s "
                                         "ON s.game_id=o.game_id WHERE s.game_type='SB'"))
    assert len(want) == 19
    rows = _all("nfl_fantasy_opportunity", "fantasy_opportunity", {"season": 2024, "season_type": "SB"})
    assert sorted(r["player_id"] for r in rows) == want
    assert {r["week"] for r in rows} == {22}


@pytest.mark.integration
def test_role_trend_div_returns_only_the_divisional_week(real):
    # SELECT week FROM snap_counts WHERE pfr_player_id='KelcTr00' AND game_type='DIV' -> 20
    assert _q(real, "SELECT week FROM snap_counts WHERE pfr_player_id='KelcTr00' AND game_type='DIV'") == [(20,)]
    rows = _all("nfl_role_trend", "role_trend", {"player_id": KELCE, "season": 2024, "season_type": "DIV"})
    assert [r["week"] for r in rows] == [20]


# ═══ 5. season_type on fantasy_opportunity and role_trend; derived tables ══════

# SELECT st, list(CAST(week AS INT) ORDER BY week) FROM <FF> WHERE player_id='00-0034844' GROUP BY st
#   -> REG [1,2,3,4,6,...,17] (16 weeks, no week 5 or 18); POST [19,20,21,22]
BARKLEY_WEEKS_SQL = (f"SELECT st, list(CAST(week AS INT) ORDER BY week) FROM {FF} "
                     "WHERE player_id='00-0034844' GROUP BY st ORDER BY st")
BARKLEY_REG_WEEKS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
BARKLEY_POST_WEEKS = [19, 20, 21, 22]


@pytest.mark.integration
@pytest.mark.parametrize("st,weeks", [("REG", BARKLEY_REG_WEEKS), ("POST", BARKLEY_POST_WEEKS)])
def test_fantasy_opportunity_filters_by_season_type(real, st, weeks):
    assert _q(real, BARKLEY_WEEKS_SQL) == [("POST", BARKLEY_POST_WEEKS), ("REG", BARKLEY_REG_WEEKS)]
    rows = _all("nfl_fantasy_opportunity", "fantasy_opportunity",
                {"player_id": BARKLEY, "season": 2024, "season_type": st})
    assert sorted(r["week"] for r in rows) == weeks
    assert {r["season_type"] for r in rows} == {st}


@pytest.mark.integration
def test_fantasy_opportunity_all_labels_each_week(real):
    rows = _all("nfl_fantasy_opportunity", "fantasy_opportunity",
                {"player_id": BARKLEY, "season": 2024, "season_type": "ALL"})
    got = sorted((r["week"], r["season_type"]) for r in rows)
    assert got == sorted([(w, "REG") for w in BARKLEY_REG_WEEKS] + [(w, "POST") for w in BARKLEY_POST_WEEKS])


# Travis Kelce 2024 snap share, from snap_counts:
# SELECT week, game_type, offense_pct FROM snap_counts WHERE pfr_player_id='KelcTr00' ORDER BY week
#   -> REG 1:0.89 2:0.91 3:0.77 4:0.87 5:0.77 7:0.77 8:0.79 9:0.84 10:0.84 11:0.80 12:0.89
#      13:0.90 14:0.83 15:0.89 16:0.88 17:0.89; DIV 20:0.76; CON 21:0.92; SB 22:0.95
# ff_opportunity has a Kelce row for each of these 19 weeks (FF SQL, player_id='00-0030506').
KELCE_SNAPS_SQL = ("SELECT week, offense_pct FROM snap_counts WHERE pfr_player_id='KelcTr00' "
                   "AND season=2024 ORDER BY week")
KELCE_REG_WEEKS = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


@pytest.mark.integration
@pytest.mark.parametrize("st,weeks", [("REG", KELCE_REG_WEEKS), ("POST", [20, 21, 22])])
def test_role_trend_filters_by_season_type(real, st, weeks):
    assert _q(real, f"SELECT st, list(CAST(week AS INT) ORDER BY week) FROM {FF} "
                    "WHERE player_id='00-0030506' GROUP BY st ORDER BY st") == [
        ("POST", [20, 21, 22]), ("REG", KELCE_REG_WEEKS)]
    rows = _all("nfl_role_trend", "role_trend", {"player_id": KELCE, "season": 2024, "season_type": st})
    assert sorted(r["week"] for r in rows) == weeks
    assert {r["season_type"] for r in rows} == {st}


@pytest.mark.integration
def test_role_trend_reg_window_uses_only_reg_weeks(real):
    # Week 17 trailing 3 rows: weeks 15, 16, 17 -> (89 + 88 + 89) / 3 = 88.67
    snaps = dict(_q(real, KELCE_SNAPS_SQL))
    assert (snaps[15], snaps[16], snaps[17]) == (0.89, 0.88, 0.89)
    rows = _all("nfl_role_trend", "role_trend", {"player_id": KELCE, "season": 2024, "season_type": "REG"})
    week17 = next(r for r in rows if r["week"] == 17)
    assert week17["snap_pct"] == 89.0
    assert week17["snap_pct_3wk"] == pytest.approx(88.67, abs=0.006)


@pytest.mark.integration
@pytest.mark.parametrize("st", ["POST", "ALL"])
def test_role_trend_post_window_does_not_reach_into_reg_weeks(real, st):
    # Assumption: the window is per season_type (a POST average built on REG weeks would be a
    # combined REG+POST value, which ALL must not compute).
    # POST weeks 20, 21, 22 have 76, 92, 95: 3-week averages 76.0, 84.0, 87.67.
    # (Mixing in REG weeks 16 and 17 would give 84.33 at week 20.)
    snaps = dict(_q(real, KELCE_SNAPS_SQL))
    assert (snaps[20], snaps[21], snaps[22]) == (0.76, 0.92, 0.95)
    rows = _all("nfl_role_trend", "role_trend", {"player_id": KELCE, "season": 2024, "season_type": st})
    post = {r["week"]: r for r in rows if r["season_type"] == "POST"}
    assert sorted(post) == [20, 21, 22]
    assert post[20]["snap_pct_3wk"] == pytest.approx(76.0, abs=0.006)
    assert post[20]["snap_pct_delta"] == pytest.approx(0.0, abs=0.006)
    assert post[21]["snap_pct_3wk"] == pytest.approx(84.0, abs=0.006)
    assert post[22]["snap_pct_3wk"] == pytest.approx(87.67, abs=0.006)


@pytest.mark.parametrize("table", V2_TABLES + ["player_td_luck", "player_drop_rate"])
def test_required_columns_lists_season_type_on_derived_player_tables(table):
    assert "season_type" in REQUIRED_COLUMNS[table], table


def test_required_columns_lists_what_the_season_type_filters_read():
    # Assumption: nfl_fantasy_opportunity takes the round from schedules through
    # ff_opportunity.game_id, and nfl_injuries filters on injuries.game_type.
    assert "game_id" in REQUIRED_COLUMNS["ff_opportunity"]
    assert "game_type" in REQUIRED_COLUMNS["injuries"]
    assert {"game_id", "game_type"} <= set(REQUIRED_COLUMNS["schedules"])


@pytest.mark.integration
@pytest.mark.parametrize("table", V2_TABLES)
def test_derived_table_has_reg_and_post_rows(real, table):
    got = {r[0] for r in _q(real, f"SELECT DISTINCT season_type FROM {table}")}
    assert got == {"REG", "POST"}, got


# ═══ 6. schema version 2: detection, upgrade, validate ══════════════════════════

def test_schema_version_is_2():
    assert migrate.SCHEMA_VERSION == 2


# Synthetic source tables: the columns the derived SQL in ingest.py reads (as in
# test_group_m), plus every REQUIRED_COLUMNS column of a raw table (VARCHAR when not listed
# here). Two games: 2024 week 1 (REG) and 2024 week 19 (WC). One player, P1.
# Types: V VARCHAR, I INTEGER, D DOUBLE, B BOOLEAN, L BIGINT.
SOURCES = {
    "plays": "play_id:D game_id:V home_team:V away_team:V season_type:V week:I posteam:V "
             "defteam:V yardline_100:D half_seconds_remaining:D sp:D qtr:D down:D time:V "
             "ydstogo:D desc:V play_type:V yards_gained:D shotgun:D no_huddle:D qb_dropback:D "
             "qb_scramble:D air_yards:D yards_after_catch:D td_team:V epa:D interception:D "
             "fumble_lost:D rush_attempt:D pass_attempt:D sack:D touchdown:D pass_touchdown:D "
             "rush_touchdown:D two_point_attempt:D fumble:D complete_pass:D "
             "passer_player_name:V passing_yards:D receiver_player_id:V receiver_player_name:V "
             "receiving_yards:D rusher_player_name:V rushing_yards:D fumbled_1_team:V season:I "
             "cpoe:D away_score:I home_score:I pass:D first_down:D special:D wp:D wpa:D "
             "goal_to_go:D score_differential:D passer_player_id:V rusher_player_id:V",
    "schedules": "game_id:V season:I game_type:V week:I gameday:V gametime:V away_team:V "
                 "away_score:I home_team:V home_score:I result:I overtime:I spread_line:D "
                 "total_line:D div_game:I roof:V surface:V temp:I wind:I away_coach:V "
                 "home_coach:V referee:V stadium:V",
    "rosters": "season:I team:V position:V depth_chart_position:V jersey_number:I status:V "
               "full_name:V height:D weight:I college:V years_exp:I gsis_id:V",
    "injuries": "season:I game_type:V team:V week:I gsis_id:V position:V full_name:V "
                "report_primary_injury:V report_secondary_injury:V report_status:V "
                "practice_primary_injury:V practice_secondary_injury:V practice_status:V",
    "snap_counts": "game_id:V season:I game_type:V week:I player:V pfr_player_id:V position:V "
                   "team:V opponent:V offense_snaps:D offense_pct:D defense_snaps:D "
                   "defense_pct:D st_snaps:D st_pct:D",
    "player_stats": "player_id:V player_display_name:V position:V season:I week:I "
                    "season_type:V team:V",
    "players": "gsis_id:V display_name:V pfr_id:V position:V",
    "ff_opportunity": "season:V posteam:V week:D game_id:V player_id:V full_name:V position:V "
                      "pass_attempt:D rec_attempt:D rush_attempt:D rec_air_yards:D receptions:D "
                      "receptions_exp:D rec_yards_gained:D rush_yards_gained:D rec_touchdown:D "
                      "rush_touchdown:D rec_touchdown_exp:D rush_touchdown_exp:D "
                      "total_fantasy_points:D total_fantasy_points_exp:D receptions_diff:D "
                      "rec_touchdown_diff:D rush_touchdown_diff:D total_fantasy_points_diff:D "
                      "rec_attempt_team:D rush_attempt_team:D rec_air_yards_team:D",
    "ftn_charting": "nflverse_game_id:V season:I week:I nflverse_play_id:I n_defense_box:I "
                    "is_no_huddle:B is_motion:B is_play_action:B is_screen_pass:B is_rpo:B "
                    "is_trick_play:B is_catchable_ball:B is_contested_ball:B "
                    "is_created_reception:B is_drop:B n_blitzers:I n_pass_rushers:I",
    "ff_rankings_draft": "page_type:V ecr_type:V player:V pos:V team:V ecr:D sd:D best:L "
                         "worst:L bye:L scrape_date:V",
    "ff_rankings_week": "page:V scrape_date:V player_name:V pos:V team:V rank:L ecr:D sd:D "
                        "best:L worst:L player_bye_week:L player_opponent:V note:V pos_rank:V "
                        "start_sit_grade:V",
    "nextgen_stats_receiving": "season:I season_type:V week:I avg_separation:D "
                               "catch_percentage:D yards:I avg_yac_above_expectation:D "
                               "player_gsis_id:V",
    "contracts": "player:V position:V team:V year_signed:I value:D apy:D apy_cap_pct:D gsis_id:V",
    "ff_playerids": "fantasypros_id:L gsis_id:V",
}
STATIC = {"players", "contracts", "ff_rankings_draft", "ff_rankings_week", "ff_playerids"}
SQL_TYPES = {"V": "VARCHAR", "I": "INTEGER", "D": "DOUBLE", "B": "BOOLEAN", "L": "BIGINT"}
GAMES = [{"game_id": "2024_01_BUF_KC", "week": 1, "game_type": "REG", "season_type": "REG"},
         {"game_id": "2024_19_BUF_KC", "week": 19, "game_type": "WC", "season_type": "POST"}]
P1, PFR1 = "00-0000001", "PlayOn00"


def _columns_of(table):
    cols = [tuple(c.split(":")) for c in SOURCES[table].split()]
    have = {c for c, _ in cols}
    cols += [(c, "V") for c in REQUIRED_COLUMNS.get(table, []) if c not in have]
    return cols


def _value(col, kind, game):
    if col == "season":
        return "2024" if kind == "V" else 2024
    if col == "week":
        return float(game["week"]) if kind == "D" else game["week"]
    special = {
        "game_id": game["game_id"], "nflverse_game_id": game["game_id"],
        "season_type": game["season_type"], "game_type": game["game_type"],
        "home_team": "KC", "posteam": "KC", "team": "KC", "td_team": "KC", "fumbled_1_team": "KC",
        "away_team": "BUF", "defteam": "BUF", "opponent": "BUF",
        "play_type": "pass", "position": "WR", "report_status": "Out",
        "gsis_id": P1, "player_id": P1, "player_gsis_id": P1, "receiver_player_id": P1,
        "pfr_id": PFR1, "pfr_player_id": PFR1, "display_name": "Player One",
        "full_name": "Player One", "player": "Player One",
        "year_signed": 2023, "apy": 10.0, "apy_cap_pct": 0.05,
        "offense_pct": 0.8, "defense_pct": 0.0, "st_pct": 0.1, "epa": 0.1,
        "is_catchable_ball": True, "two_point_attempt": 0.0,
    }
    if col in special:
        return special[col]
    return {"V": col, "I": 1, "D": 1.0, "B": False, "L": 1}[kind]


def _quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def build_v1(path):
    """A database as v0.6.x left it at schema version 1: every source table, the derived
    tables rebuilt, then season_type dropped from V2_TABLES (v1 derived SQL had none)."""
    con = duckdb.connect(str(path))
    try:
        ing._ensure_metadata_table(con)
        for table in SOURCES:
            cols = _columns_of(table)
            ddl = ", ".join(f'"{c}" {SQL_TYPES[k]}' for c, k in cols)
            con.execute(f'CREATE TABLE "{table}" ({ddl})')
            marks = ", ".join("?" for _ in cols)
            for game in (GAMES[:1] if table in STATIC else GAMES):
                con.execute(f'INSERT INTO "{table}" VALUES ({marks})',
                            [_value(c, k, game) for c, k in cols])
        _quiet(ing.rebuild_derived_tables, con)
        for table in V2_TABLES:
            cols = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
            if "season_type" in cols:
                con.execute(f'ALTER TABLE {table} DROP COLUMN "season_type"')
        con.execute("CREATE TABLE _schema_version "
                    "(version INTEGER, applied_at TIMESTAMP, nfl_mcp_version VARCHAR)")
        con.execute("INSERT INTO _schema_version VALUES (1, TIMESTAMP '2026-11-05', '0.6.3')")
    finally:
        con.close()
    return Path(path)


def _cli(*args):
    result = CliRunner().invoke(main, [str(a) for a in args])
    return result.exit_code, result.output


def _ro(path):
    return duckdb.connect(str(path), read_only=True)


def _no_network(*_a, **_k):
    raise RuntimeError("this upgrade must not need the network")


@pytest.fixture(scope="module")
def upgraded(tmp_path_factory):
    """One v1 database, checked, upgraded with the CLI, checked again. Network steps fail."""
    root = tmp_path_factory.mktemp("v1")
    path = build_v1(root / "v1.duckdb")
    pristine = root / "v1_pristine.duckdb"
    shutil.copyfile(path, pristine)
    out = {"path": path, "pristine": pristine}
    with pytest.MonkeyPatch.context() as mp:
        actions = getattr(migrate, "_ACTIONS", None)
        if isinstance(actions, dict):
            for name in ("raw_columns", "depth_charts_season"):
                if name in actions:
                    mp.setitem(actions, name, _no_network)
        out["check_before"] = _cli("upgrade", "--db", path, "--check")
        out["upgrade"] = _cli("upgrade", "--db", path)
        out["check_after"] = _cli("upgrade", "--db", path, "--check")
    return out


def test_v2_tables_are_derived_and_rewritten_by_the_upgrade():
    # validate lets a table shrink across an upgrade only when the upgrade rewrites it.
    assert set(V2_TABLES) <= set(migrate.DERIVED_TABLES)
    assert set(V2_TABLES) <= migrate.UPGRADE_REWRITES


def test_check_detects_v1_without_season_type_columns(tmp_path):
    path = build_v1(tmp_path / "v1.duckdb")
    con = _ro(path)
    try:
        status = migrate.check(con)
    finally:
        con.close()
    assert (status.version, status.target, status.upgrade_needed, status.newer) == (1, 2, True, False)
    needed = {s.name: s for s in status.steps if s.needed}
    assert not any(s.needs_network for s in needed.values()), needed
    assert "derived_tables" in needed, [s.name for s in status.steps]
    for table in V2_TABLES:
        assert f"{table}.season_type" in needed["derived_tables"].detail


def test_cli_check_exits_3_on_v1(upgraded):
    code, out = upgraded["check_before"]
    assert code == 3, out
    assert "1" in out and "2" in out, out


def test_cli_upgrade_exits_0_on_v1(upgraded):
    code, out = upgraded["upgrade"]
    assert code == 0, out


def test_cli_check_exits_0_after_upgrade(upgraded):
    code, out = upgraded["check_after"]
    assert code == 0, out


def test_upgrade_stamps_version_2(upgraded):
    con = _ro(upgraded["path"])
    try:
        assert con.execute("SELECT MAX(version) FROM _schema_version").fetchone()[0] == 2
    finally:
        con.close()


@pytest.mark.parametrize("table", V2_TABLES)
def test_upgrade_adds_season_type_with_reg_and_post(upgraded, table):
    con = _ro(upgraded["path"])
    try:
        cols = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
        assert set(REQUIRED_COLUMNS[table]) <= cols
        got = {r[0] for r in con.execute(f"SELECT DISTINCT season_type FROM {table}").fetchall()}
    finally:
        con.close()
    # The synthetic data has one REG game (week 1) and one WC game (week 19).
    assert got == {"REG", "POST"}


def test_upgrade_leaves_the_v1_copy_untouched(upgraded):
    con = _ro(upgraded["pristine"])
    try:
        assert con.execute("SELECT MAX(version) FROM _schema_version").fetchone()[0] == 1
    finally:
        con.close()


def _grow(path, dest, table, times=2):
    """Copy path to dest and double `table` `times` times (a current slot with more rows)."""
    shutil.copyfile(path, dest)
    con = duckdb.connect(str(dest))
    try:
        for _ in range(times):
            con.execute(f'INSERT INTO "{table}" SELECT * FROM "{table}"')
    finally:
        con.close()
    return dest


def test_validate_accepts_upgraded_slot_against_v1_slot(upgraded, tmp_path):
    current = _grow(upgraded["pristine"], tmp_path / "current.duckdb", "player_role_trend")
    results = run_checks(upgraded["path"], current)
    bad = [f"{r.name}: {r.detail}" for r in results if not r.ok]
    assert not bad, bad
    floor = next(r for r in results if r.name == "row_count_floor")
    assert "player_role_trend" in floor.detail and "allowed" in floor.detail, floor.detail


def test_validate_upgraded_slot_reports_schema_version_2(upgraded):
    results = run_checks(upgraded["path"])
    version = next(r for r in results if r.name == "schema_version")
    assert version.ok and "2" in version.detail, version.detail


def test_validate_still_fails_a_raw_shrink_across_the_upgrade(upgraded, tmp_path):
    current = _grow(upgraded["pristine"], tmp_path / "current.duckdb", "plays")
    results = run_checks(upgraded["path"], current)
    floor = next(r for r in results if r.name == "row_count_floor")
    assert not floor.ok, floor.detail
    assert "plays has 2 rows, current slot has 8" in floor.detail, floor.detail


# ═══ 7. output values ══════════════════════════════════════════════════════════
# Snap share: percent, 0 to 100. Raw snap_counts holds whole percents as fractions
# (SELECT count(*) FROM snap_counts WHERE abs(offense_pct*100 - round(offense_pct*100)) > 1e-9
#  -> 0), so round(100 * raw, 1) is the value at one decimal (and at two).

@pytest.mark.integration
def test_snap_counts_offense_pct_is_a_percent(real):
    # SELECT offense_pct, defense_pct, st_pct FROM snap_counts WHERE pfr_player_id='KelcTr00'
    # AND season=2024 AND week=1  -> 0.89, 0.0, 0.0
    assert _q(real, "SELECT offense_pct, defense_pct, st_pct FROM snap_counts WHERE "
                    "pfr_player_id='KelcTr00' AND season=2024 AND week=1") == [(0.89, 0.0, 0.0)]
    rows = _ok("nfl_snap_counts", {"player_id": KELCE, "season": 2024, "week": 1,
                                   "season_type": "REG"})["snap_counts"]
    assert [(r["offense_pct"], r["defense_pct"], r["st_pct"]) for r in rows] == [(89.0, 0.0, 0.0)]


@pytest.mark.integration
def test_snap_counts_every_pct_is_100_times_raw(real):
    raw = {(p, t): (o, d, s) for p, t, o, d, s in _q(
        real, "SELECT pfr_player_id, team, offense_pct, defense_pct, st_pct FROM snap_counts "
              "WHERE season=2024 AND game_type='SB'")}
    # SELECT max(offense_pct), max(defense_pct), max(st_pct) FROM snap_counts WHERE game_type='SB'
    #   -> 1.0, 1.0, 1.0 (so a fraction would show as 1.0, a percent as 100.0)
    assert max(v[0] for v in raw.values()) == 1.0
    rows = _all("nfl_snap_counts", "snap_counts", {"season": 2024, "season_type": "SB"})
    assert len(rows) == len(raw) == 92
    for r in rows:
        want = tuple(None if v is None else round(100 * v, 1) for v in raw[(r["pfr_player_id"], r["team"])])
        assert (r["offense_pct"], r["defense_pct"], r["st_pct"]) == want, r
        assert all(0 <= v <= 100 for v in want if v is not None)
    assert max(r["offense_pct"] for r in rows) == 100.0


@pytest.mark.integration
def test_role_trend_snap_pct_is_a_percent(real):
    assert dict(_q(real, KELCE_SNAPS_SQL))[1] == 0.89
    rows = _all("nfl_role_trend", "role_trend", {"player_id": KELCE, "season": 2024, "week": 1,
                                                  "season_type": "REG"})
    assert [r["snap_pct"] for r in rows] == [89.0]


@pytest.mark.integration
def test_role_trend_min_snap_pct_is_a_percent(real):
    # SELECT list(week ORDER BY week) FROM snap_counts WHERE pfr_player_id='KelcTr00'
    # AND season=2024 AND game_type='REG' AND offense_pct >= 0.89  -> [1, 2, 12, 13, 15, 17]
    assert _q(real, "SELECT list(week ORDER BY week) FROM snap_counts WHERE pfr_player_id='KelcTr00' "
                    "AND season=2024 AND game_type='REG' AND offense_pct >= 0.89") == [
        ([1, 2, 12, 13, 15, 17],)]
    rows = _all("nfl_role_trend", "role_trend", {"player_id": KELCE, "season": 2024,
                                                  "season_type": "REG", "min_snap_pct": 89})
    assert sorted(r["week"] for r in rows) == [1, 2, 12, 13, 15, 17]
    assert all(r["snap_pct"] >= 89 for r in rows)


def _int_fields(obj, found):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("season", "season_year", "week"):
                found.append((k, v))
            _int_fields(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _int_fields(v, found)
    return found


INT_CALLS = [
    ("nfl_schedule", {"season": 2024, "week": 1, "season_type": "REG"}),
    ("nfl_snap_counts", {"player_id": KELCE, "season": 2024, "season_type": "ALL"}),
    ("nfl_injuries", {"team": "KC", "season": 2024, "season_type": "ALL", "limit": 50}),
    ("nfl_fantasy_opportunity", {"player_id": BARKLEY, "season": 2024, "season_type": "ALL"}),
    ("nfl_role_trend", {"player_id": KELCE, "season": 2024, "season_type": "ALL"}),
    ("nfl_search_plays", {"season": 2024, "season_type": "SB", "limit": 20}),
    ("nfl_td_luck", {"player_id": KELCE, "season": 2024, "season_type": "ALL"}),
    ("nfl_drop_rate", {"player_id": AJ_BROWN, "season": 2024, "season_type": "ALL"}),
    ("nfl_separation_opportunity", {"player_id": AJ_BROWN, "season": 2024, "season_type": "ALL"}),
    ("nfl_contract_value", {"player_id": BARKLEY, "season": 2024, "season_type": "ALL"}),
    ("nfl_player_stats", {"player_id": BARKLEY, "season": 2024, "season_type": "ALL"}),
    ("nfl_team_stats", {"team": "PHI", "season": 2024, "season_type": "ALL"}),
    ("nfl_compare", {"team1": "PHI", "team2": "KC", "season": 2024, "season_type": "ALL"}),
    ("nfl_roster", {"team": "KC", "season": 2024, "limit": 5}),
    ("nfl_team_records", {"team": "KC", "season": 2024}),
]


@pytest.mark.integration
@pytest.mark.parametrize("name,args", INT_CALLS, ids=[c[0] for c in INT_CALLS])
def test_season_and_week_are_integers(real, name, args):
    found = _int_fields(_ok(name, args), [])
    assert found, f"{name} returned no season or week"
    bad = [(k, v) for k, v in found if v is not None and (type(v) is not int)]
    assert not bad, bad[:5]


@pytest.mark.integration
def test_fantasy_opportunity_season_and_week_are_python_ints(real):
    # ff_opportunity.season is VARCHAR and week is DOUBLE in the raw table:
    # SELECT data_type FROM information_schema.columns WHERE table_name='ff_opportunity'
    # AND column_name IN ('season','week')  -> VARCHAR, DOUBLE
    assert sorted(_q(real, "SELECT column_name, data_type FROM information_schema.columns WHERE "
                           "table_name='ff_opportunity' AND column_name IN ('season','week')")) == [
        ("season", "VARCHAR"), ("week", "DOUBLE")]
    res = tools.nfl_fantasy_opportunity(player_id=BARKLEY, season=2024, week=22, season_type="SB")
    rows = res["fantasy_opportunity"]
    assert [(type(r["season"]), type(r["week"])) for r in rows] == [(int, int)]
    assert (rows[0]["season"], rows[0]["week"]) == (2024, 22)
