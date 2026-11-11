"""Group L: regression suite for behavior that is correct today and must keep holding.

Every test here passes on main (v0.4.0, ab52c3d) and must still pass after each fix
group (A-K) lands. Nothing here is xfail. Behavior the audit calls wrong is not pinned
(for example passing attempts, the nfl_query regex and SELECT-prefix gate, CORS "*",
the pre-fix derived-table definitions, and the per-call 500-row cap).

Oracles
- Synthetic DuckDB files in tmp_path with hand-counted rows (unit tests, no marker).
- Official nflverse tables in /tmp/grpE_v3.duckdb (player_stats, schedules, players)
  queried directly, so each expectation comes from a different table than the tool
  reads (integration tests, skipped when the DB is missing). 2024 data only.

Interface contract assumed (beyond the shared brief)
- The dispatch table is read from server.py: a module-level dict named _DISPATCH or
  DISPATCH if one exists, otherwise the largest dict literal in server.py whose keys
  are all "nfl_*" strings. Its key set must equal the names returned by list_tools.
- list_tools returns at least the 22 tools that exist today (later groups may add
  tools such as nfl_player_lookup); names are unique; every inputSchema has
  type "object".
- Tool calls in this file go through the MCP CallToolRequest handler of
  nfl_mcp.server._mcp_server, so they survive call_tool returning either a list of
  TextContent or a CallToolResult. The first content item is JSON text of the tool dict.
- nfl_player_stats: a season row is matched on its "season" and "season_type" keys.
  Stat values are read from row["passing"] / row["receiving"] when those nested dicts
  exist (group G shape), else from the row itself (today's flat shape, which needs
  stat_type). Keys used: completions, yards, touchdowns, interceptions (passing) and
  targets, receptions, yards, touchdowns (receiving). If the tool answers
  {"ambiguous": True, "candidates": [...]}, the test re-calls with player_id set to the
  gsis_id found in player_stats.
- nfl_query keeps "rows" as a list of dicts keyed by column name, "row_count" and
  "truncated". The timeout default stays 10 s in tools._QUERY_TIMEOUT_SECONDS; the
  1 s timeout test sets both that constant and NFL_MCP_QUERY_TIMEOUT_SECONDS.
- nfl_schedule keeps the "games" key; nfl_team_records keeps "records" with wins/losses.
- nfl_schema(table=X)["columns"] keeps column_name and data_type in each entry (extra
  keys are ignored).
- Derived team tables keep the season_year column; the per-season grain is asserted on
  DISTINCT (team, season_year), so adding season_type does not break it.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import os
import re
import time
from pathlib import Path

import duckdb
import pytest

import nfl_mcp.server as server_module
from nfl_mcp import franchises, ingest, tools
from nfl_mcp.schema_pbp import _SCHEMA_CATEGORIES

REAL_DB = Path("/tmp/grpE_v3.duckdb")
# A database at the current schema version (2). grpE_v3 predates the season_type split of
# the fantasy tables, so those tools answer STALE_DERIVED_TABLES there; the minimal-call test
# needs a current database. CI's fresh 2024 ingest is one.
CURRENT_DB = Path(os.environ.get("NFL_MCP_DB_PATH", "/tmp/ci2024/v070.duckdb"))

# The 22 tools on main today. Later groups may add tools, never remove these.
TODAY_TOOLS = {
    "nfl_schema", "nfl_status", "nfl_query", "nfl_search_plays", "nfl_team_stats",
    "nfl_player_stats", "nfl_compare", "nfl_catalog", "nfl_roster", "nfl_injuries",
    "nfl_schedule", "nfl_snap_counts", "nfl_fantasy_opportunity", "nfl_fantasy_rankings",
    "nfl_ftn_charting", "nfl_td_luck", "nfl_role_trend", "nfl_separation_opportunity",
    "nfl_drop_rate", "nfl_contract_value", "nfl_injury_return", "nfl_team_records",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _oracle(sql: str, params: list | None = None) -> list[tuple]:
    """Query the real DB directly and close at once.

    The connection is closed before any tool runs so a later locked config in
    get_db_connection (group A) never meets an open connection with another config.
    """
    conn = duckdb.connect(str(REAL_DB), read_only=True)
    try:
        return conn.execute(sql, params or []).fetchall()
    finally:
        conn.close()


def _scalar(sql: str, params: list | None = None):
    return _oracle(sql, params)[0][0]


@pytest.fixture
def real_db(monkeypatch):
    if not REAL_DB.exists():
        pytest.skip(f"{REAL_DB} not present")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(REAL_DB))
    return REAL_DB


@pytest.fixture
def current_db(monkeypatch):
    if not CURRENT_DB.exists():
        pytest.skip(f"{CURRENT_DB} not present")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(CURRENT_DB))
    return CURRENT_DB


def _dispatch_names() -> list[str]:
    for attr in ("_DISPATCH", "DISPATCH"):
        table = getattr(server_module, attr, None)
        if isinstance(table, dict) and table:
            return sorted(table)
    tree = ast.parse(inspect.getsource(server_module))
    best: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and node.keys and all(
            isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value.startswith("nfl_")
            for k in node.keys
        ):
            keys = [k.value for k in node.keys]
            if len(keys) > len(best):
                best = keys
    return sorted(best)


def _call_mcp(name: str, arguments: dict):
    """Run a tool through the MCP CallToolRequest handler (input validation included)."""
    from mcp.types import CallToolRequest, CallToolRequestParams

    handler = server_module._mcp_server.request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call", params=CallToolRequestParams(name=name, arguments=arguments)
    )
    return asyncio.run(handler(request)).root


def _list_tools():
    from mcp.types import ListToolsRequest

    handler = server_module._mcp_server.request_handlers[ListToolsRequest]
    return asyncio.run(handler(ListToolsRequest(method="tools/list"))).root.tools


def _make_db(path: Path, statements: list[str]) -> Path:
    conn = duckdb.connect(str(path))
    try:
        for sql in statements:
            conn.execute(sql)
    finally:
        conn.close()
    return path


# Values for required inputSchema fields. A field missing here fails the test loudly.
_SAMPLE_ARGS = {
    "sql": "SELECT 1 AS one",
    "team": "KC",
    "player_name": "Josh Allen",
    "name": "Josh Allen",
    "player": "Josh Allen",
    "player_id": "00-0034857",
    "season_type": "ALL",
}
# Not in "required" (a name or an id, a player pair or a team pair), but a call needs one.
_ENTITY_ARGS = {
    "nfl_player_stats": {"player": "Josh Allen"},
    "nfl_player_lookup": {"player": "Josh Allen"},
    "nfl_compare": {"team1": "KC", "team2": "BUF"},
}


# ── 1. every dispatched tool answers a minimal valid call ──────────────────────

@pytest.mark.integration
@pytest.mark.parametrize("tool_name", _dispatch_names())
def test_dispatched_tool_minimal_call_returns_no_error(current_db, tool_name):
    tool = next((t for t in _list_tools() if t.name == tool_name), None)
    assert tool is not None, f"{tool_name} is dispatched but not listed"
    schema = tool.inputSchema
    missing = [f for f in schema.get("required", []) if f not in _SAMPLE_ARGS]
    assert not missing, f"no sample value for required fields {missing}"
    args = {f: _SAMPLE_ARGS[f] for f in schema.get("required", [])}
    args.update(_ENTITY_ARGS.get(tool_name, {}))
    if "season" in schema.get("properties", {}):
        args["season"] = 2024  # brief: pin integration calls to 2024
    result = _call_mcp(tool_name, args)
    assert result.isError is False, result.content[0].text
    payload = json.loads(result.content[0].text)
    assert isinstance(payload, dict)
    assert "error" not in payload, payload.get("error")


def test_dispatch_table_has_the_22_tools_of_today():
    names = _dispatch_names()
    assert len(names) == len(set(names))
    assert TODAY_TOOLS <= set(names)


# ── 2. nfl_player_stats agrees with player_stats where it is right today ───────

def _player_stats_oracle(display_name: str, season_type: str, position: str | None = None):
    pos_clause = "AND position = ?" if position else ""
    params = [display_name, season_type] + ([position] if position else [])
    rows = _oracle(
        f"""SELECT player_id, SUM(completions), SUM(passing_yards), SUM(passing_tds),
                   SUM(passing_interceptions), SUM(targets), SUM(receptions),
                   SUM(receiving_yards), SUM(receiving_tds)
            FROM player_stats
            WHERE player_display_name = ? AND season = 2024 AND season_type = ? {pos_clause}
            GROUP BY player_id""",
        params,
    )
    assert len(rows) == 1, f"player_stats has {len(rows)} ids for {display_name}"
    pid, cmp_, pyd, ptd, pint, tgt, rec, ryd, rtd = rows[0]
    return pid, {
        "passing": {"completions": cmp_, "yards": pyd, "touchdowns": ptd, "interceptions": pint},
        "receiving": {"targets": tgt, "receptions": rec, "yards": ryd, "touchdowns": rtd},
    }


def _tool_season_block(name: str, player_id: str, block: str, season_type: str, **filters):
    kwargs = {"player": name, "season_type": season_type, **filters}
    if "stat_type" in inspect.signature(tools.nfl_player_stats).parameters:
        kwargs["stat_type"] = block
    result = tools.nfl_player_stats(**kwargs)
    if result.get("ambiguous"):
        kwargs.pop("player")
        result = tools.nfl_player_stats(player_id=player_id, **kwargs)
    assert "error" not in result, result
    rows = [r for r in result["seasons"]
            if r.get("season") == 2024 and r.get("season_type") == season_type]
    assert len(rows) == 1, result["seasons"]
    row = rows[0]
    return row[block] if isinstance(row.get(block), dict) else row


_PASSING_CASES = [
    ("Josh Allen", "REG", {"season": 2024}),
    ("Josh Allen", "POST", {"season": 2024}),
    ("Lamar Jackson", "REG", {"season": 2024}),
]


@pytest.mark.integration
@pytest.mark.parametrize("name,season_type,filters", _PASSING_CASES)
@pytest.mark.parametrize("stat", ["completions", "yards", "touchdowns", "interceptions"])
def test_player_stats_passing_matches_player_stats_table(real_db, name, season_type, filters, stat):
    # Attempts are not asserted: today they include sacks and 2-pt tries (audit D.13).
    pid, expected = _player_stats_oracle(name, season_type, position="QB")
    block = _tool_season_block(name, pid, "passing", season_type, **filters)
    assert block[stat] == expected["passing"][stat]


_RECEIVING_FILTERS = [{"season": 2024}, {"season_from": 2024, "season_to": 2024}]


@pytest.mark.integration
@pytest.mark.parametrize("filters", _RECEIVING_FILTERS, ids=["season", "season_range"])
@pytest.mark.parametrize("stat", ["targets", "receptions", "yards", "touchdowns"])
def test_player_stats_receiving_chase_matches_player_stats_table(real_db, filters, stat):
    pid, expected = _player_stats_oracle("Ja'Marr Chase", "REG")
    block = _tool_season_block("Ja'Marr Chase", pid, "receiving", "REG", **filters)
    assert block[stat] == expected["receiving"][stat]


# ── 3. nfl_schedule REG count and the SB filter ────────────────────────────────

@pytest.mark.integration
@pytest.mark.parametrize("team", ["KC", "BUF", "PHI"])
def test_schedule_reg_count_matches_schedules_table(real_db, team):
    expected = _scalar(
        "SELECT COUNT(*) FROM schedules WHERE season = 2024 AND game_type = 'REG' "
        "AND (home_team = ? OR away_team = ?)", [team, team])
    assert expected == 17  # 17-game season since 2021
    result = tools.nfl_schedule(team=team, season=2024, season_type="REG")
    assert len(result["games"]) == expected
    assert {g["game_type"] for g in result["games"]} == {"REG"}


@pytest.mark.integration
@pytest.mark.parametrize("team", ["KC", "PHI"])
def test_schedule_sb_returns_the_one_super_bowl(real_db, team):
    expected_ids = [r[0] for r in _oracle(
        "SELECT game_id FROM schedules WHERE season = 2024 AND game_type = 'SB' "
        "AND (home_team = ? OR away_team = ?)", [team, team])]
    assert len(expected_ids) == 1  # KC and PHI met in the 2024 Super Bowl
    games = tools.nfl_schedule(team=team, season=2024, season_type="SB")["games"]
    assert [g["game_id"] for g in games] == expected_ids
    assert games[0]["game_type"] == "SB"


# ── 4. nfl_team_records ────────────────────────────────────────────────────────

def _schedule_record(team: str) -> tuple[int, int]:
    return _oracle(
        """SELECT
             SUM(CASE WHEN (home_team = ? AND result > 0) OR (away_team = ? AND result < 0)
                      THEN 1 ELSE 0 END),
             SUM(CASE WHEN (home_team = ? AND result < 0) OR (away_team = ? AND result > 0)
                      THEN 1 ELSE 0 END)
           FROM schedules
           WHERE season = 2024 AND game_type = 'REG' AND result IS NOT NULL
             AND (home_team = ? OR away_team = ?)""",
        [team] * 6,
    )[0]


@pytest.mark.integration
@pytest.mark.parametrize("team", ["KC", "DET", "CAR"])
def test_team_records_wins_losses_match_schedules(real_db, team):
    wins, losses = _schedule_record(team)
    rows = tools.nfl_team_records(team=team, season=2024)["records"]
    assert len(rows) == 1
    assert (rows[0]["wins"], rows[0]["losses"]) == (wins, losses)


@pytest.mark.integration
@pytest.mark.parametrize("alias,canonical", [("OAK", "LV"), ("oak", "LV"), ("SD", "LAC")])
def test_team_records_historical_code_maps_to_franchise(real_db, alias, canonical):
    wins, losses = _schedule_record(canonical)
    rows = tools.nfl_team_records(team=alias, season=2024)["records"]
    assert [r["team"] for r in rows] == [canonical]
    assert (rows[0]["wins"], rows[0]["losses"]) == (wins, losses)


@pytest.mark.parametrize("code", ["XX", "ZZZ", "QQQ"])
def test_team_records_unknown_code_returns_error(code):
    result = tools.nfl_team_records(team=code, season=2024)
    assert "error" in result
    assert code in result["error"]


# ── 5. nfl_query basics on a synthetic DB ──────────────────────────────────────

@pytest.fixture
def query_db(tmp_path, monkeypatch):
    path = _make_db(tmp_path / "q.duckdb", [
        "CREATE TABLE t (name VARCHAR, alpha INTEGER, beta DOUBLE)",
        # Inserted out of order on purpose.
        "INSERT INTO t VALUES ('c', 3, 0.3), ('a', 1, 0.1), ('e', 5, 0.5), "
        "('b', 2, 0.2), ('d', 4, 0.4)",
    ])
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


def test_query_where_order_by_returns_rows_in_order(query_db):
    result = tools.nfl_query("SELECT name FROM t WHERE alpha >= 2 ORDER BY alpha DESC")
    # alpha >= 2 keeps b,c,d,e; descending by alpha: e(5), d(4), c(3), b(2)
    assert [r["name"] for r in result["rows"]] == ["e", "d", "c", "b"]


def test_query_order_by_ascending_with_limit(query_db):
    result = tools.nfl_query("SELECT name FROM t ORDER BY beta ASC LIMIT 2")
    assert [r["name"] for r in result["rows"]] == ["a", "b"]  # beta 0.1, 0.2


@pytest.mark.parametrize("max_rows,expected_count,expected_truncated", [
    (10, 5, False),  # 5 rows < 10
    (5, 5, False),   # exactly max_rows rows is not truncated
    (3, 3, True),    # 5 rows > 3
])
def test_query_truncated_flag_and_row_count(query_db, max_rows, expected_count, expected_truncated):
    result = tools.nfl_query("SELECT * FROM t", max_rows=max_rows)
    assert result["truncated"] is expected_truncated
    assert result["row_count"] == len(result["rows"]) == expected_count


def test_query_bad_column_returns_error_with_did_you_mean(query_db):
    result = tools.nfl_query("SELECT alphx FROM t")
    assert "error" in result
    assert "alpha" in result["did_you_mean"]


def test_query_bad_column_in_where_returns_error_with_did_you_mean(query_db):
    result = tools.nfl_query("SELECT name FROM t WHERE betta > 0")
    assert "error" in result
    assert "beta" in result["did_you_mean"]


def test_query_timeout_default_is_10_seconds():
    assert getattr(tools, "_QUERY_TIMEOUT_SECONDS", None) == 10


def test_query_cross_join_times_out_when_limit_is_1_second(query_db, monkeypatch):
    monkeypatch.setattr(tools, "_QUERY_TIMEOUT_SECONDS", 1, raising=False)
    monkeypatch.setenv("NFL_MCP_QUERY_TIMEOUT_SECONDS", "1")
    start = time.monotonic()
    # 1e10 joined rows; far beyond 1 s on any machine.
    result = tools.nfl_query(
        "SELECT SUM(a.range * b.range) AS s FROM range(100000) a, range(100000) b")
    elapsed = time.monotonic() - start
    assert "error" in result
    assert "timeout" in result["error"].lower()
    assert elapsed < 6


# ── 6. nfl_schema ──────────────────────────────────────────────────────────────

def _real_tables() -> list[str]:
    if not REAL_DB.exists():
        return []
    return [r[0] for r in _oracle(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name")]


@pytest.mark.integration
@pytest.mark.parametrize("table", _real_tables() or ["<no db>"])
def test_schema_table_columns_equal_describe(real_db, table):
    expected = [(r[0], r[1]) for r in _oracle(f'DESCRIBE "{table}"')]
    result = tools.nfl_schema(table=table)
    assert "error" not in result, result
    assert [(c["column_name"], c["data_type"]) for c in result["columns"]] == expected


@pytest.mark.parametrize("category", sorted(_SCHEMA_CATEGORIES))
def test_schema_every_category_returns_text(category):
    result = tools.nfl_schema(category=category)
    assert "error" not in result
    assert isinstance(result["schema"], str) and result["schema"].strip()


@pytest.mark.parametrize("category", ["nonexistent", "EPAA", "player"])
def test_schema_unknown_category_lists_every_category(category):
    result = tools.nfl_schema(category=category)
    assert "error" in result
    assert set(result["available"]) == set(_SCHEMA_CATEGORIES)


@pytest.mark.integration
@pytest.mark.parametrize("table", ["no_such_table", "play", "Plays_2024"])
def test_schema_unknown_table_lists_the_real_tables(real_db, table):
    result = tools.nfl_schema(table=table)
    assert "error" in result
    assert set(result["available_tables"]) == set(_real_tables())


# ── 7. data integrity invariants on the real DB ────────────────────────────────

# rosters_weekly holds 34 rows with gsis_id NULL (2025-2026 only) that collide on
# (NULL, season, week); a NULL id is not a player key, so those rows are excluded.
_DUPLICATE_KEYS = [
    ("plays", "game_id, play_id", "TRUE"),
    ("schedules", "game_id", "TRUE"),
    ("player_stats", "player_id, season, week, season_type", "TRUE"),
    ("rosters_weekly", "gsis_id, season, week", "gsis_id IS NOT NULL"),
    ("participation", "nflverse_game_id, play_id", "TRUE"),
]


@pytest.mark.integration
@pytest.mark.parametrize("table,key,where", _DUPLICATE_KEYS, ids=[t for t, *_ in _DUPLICATE_KEYS])
def test_no_duplicate_keys(real_db, table, key, where):
    assert _scalar(f"SELECT COUNT(*) FROM {table} WHERE {where}") > 0
    dupes = _scalar(
        f"SELECT COUNT(*) FROM (SELECT {key} FROM {table} WHERE {where} "
        f"GROUP BY {key} HAVING COUNT(*) > 1)")
    assert dupes == 0


@pytest.mark.integration
def test_participation_joins_plays_completely(real_db):
    total, matched = _oracle(
        """SELECT COUNT(*), COUNT(p.play_id)
           FROM participation pa
           LEFT JOIN plays p ON p.game_id = pa.nflverse_game_id AND p.play_id = pa.play_id""")[0]
    assert total > 0 and matched == total


@pytest.mark.integration
def test_snap_counts_join_players_by_pfr_id_over_99_percent(real_db):
    total, matched = _oracle(
        """SELECT COUNT(*), COUNT(pl.pfr_id)
           FROM snap_counts s
           LEFT JOIN (SELECT DISTINCT pfr_id FROM players WHERE pfr_id IS NOT NULL) pl
             ON pl.pfr_id = s.pfr_player_id""")[0]
    assert total > 0 and matched / total > 0.99


@pytest.mark.integration
def test_every_play_has_a_schedule_row(real_db):
    orphans = _scalar(
        "SELECT COUNT(*) FROM plays p WHERE NOT EXISTS "
        "(SELECT 1 FROM schedules s WHERE s.game_id = p.game_id)")
    assert orphans == 0


# ── 8. server plumbing ─────────────────────────────────────────────────────────

def test_cli_serve_default_host_is_loopback():
    from nfl_mcp.cli import serve

    host = next(p for p in serve.params if p.name == "host")
    assert host.default == "127.0.0.1"


def test_entrypoint_serves_on_all_interfaces():
    script = (Path(server_module.__file__).resolve().parents[1] / "entrypoint.sh").read_text()
    serve_lines = [ln for ln in script.splitlines()
                   if "nfl-mcp serve" in ln and not ln.lstrip().startswith("#")]
    assert serve_lines, "entrypoint.sh no longer starts nfl-mcp serve"
    assert all(re.search(r"--host\s+0\.0\.0\.0\b", ln) for ln in serve_lines)


def _sse_or_json(response) -> dict:
    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
    for line in response.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:"):].strip())
    raise AssertionError(f"no JSON-RPC message in response: {response.text[:200]}")


def test_streamable_http_answers_initialize(monkeypatch):
    from starlette.testclient import TestClient

    monkeypatch.delenv("NFL_MCP_AUTH_TOKEN", raising=False)
    app = server_module.create_app()
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "group-l", "version": "0"}}}
    with TestClient(app) as client:
        resp = client.post("/mcp", json=body,
                           headers={"Accept": "application/json, text/event-stream"})
    assert resp.status_code == 200
    message = _sse_or_json(resp)
    assert message["id"] == 1
    assert message["result"]["serverInfo"]["name"] == "nfl-mcp"
    assert resp.headers.get("mcp-session-id")


def test_list_tools_names_unique_and_cover_dispatch():
    names = [t.name for t in _list_tools()]
    assert len(names) == len(set(names))
    assert TODAY_TOOLS <= set(names)
    assert set(names) == set(_dispatch_names())


def test_list_tools_every_input_schema_is_an_object():
    bad = [t.name for t in _list_tools() if t.inputSchema.get("type") != "object"]
    assert bad == []


@pytest.mark.parametrize("name,arguments", [
    ("nfl_schedule", {"season": "2024", "season_type": "REG"}),
    ("nfl_search_plays", {"season": "2024", "season_type": "REG"}),
    ("nfl_query", {"sql": "SELECT 1", "max_rows": 999}),
], ids=["schedule-season-str", "search-season-str", "query-max-rows-999"])
def test_mcp_input_validation_rejects_bad_types(name, arguments, monkeypatch):
    # Must be rejected before the tool runs, so no database is needed.
    monkeypatch.setenv("NFL_MCP_DB_PATH", "/nonexistent/never.duckdb")
    result = _call_mcp(name, arguments)
    assert result.isError is True
    assert "validation" in result.content[0].text.lower()


# ── 9. derived tables ──────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.parametrize("table", ["team_offense_stats", "team_defense_stats"])
def test_derived_team_table_covers_every_2024_team(real_db, table):
    side = "posteam" if table == "team_offense_stats" else "defteam"
    teams = {r[0] for r in _oracle(
        f"SELECT DISTINCT {side} FROM plays WHERE season = 2024 AND {side} IS NOT NULL")}
    assert len(teams) == 32
    derived = {r[0] for r in _oracle(
        f"SELECT DISTINCT team FROM {table} WHERE season_year = 2024")}
    assert derived == teams


@pytest.mark.integration
@pytest.mark.parametrize("table", ["team_offense_stats", "team_defense_stats"])
def test_derived_team_table_grain_is_team_season(real_db, table):
    expected = _scalar(
        "SELECT COUNT(*) FROM (SELECT DISTINCT posteam, season FROM plays "
        "WHERE posteam IS NOT NULL)")
    assert expected == 96  # 32 teams x seasons 2024-2026
    assert _scalar(
        f"SELECT COUNT(*) FROM (SELECT DISTINCT team, season_year FROM {table})") == expected


def _documented_situations() -> set[str]:
    expr = getattr(ingest, "_SITUATION_EXPR", None) or inspect.getsource(ingest)
    return set(re.findall(r"(?:THEN|ELSE)\s+'([^']+)'", expr))


def test_situation_buckets_documented_in_schema_text_match_ingest():
    text = _SCHEMA_CATEGORIES["aggregate_tables"]
    section = text[text.index("situational_stats"):text.index("formation_effectiveness")]
    documented = set(re.findall(r"'([^']+)'", section))
    assert documented == _documented_situations()


@pytest.mark.integration
def test_situational_stats_buckets_equal_ingest_definition(real_db):
    actual = {r[0] for r in _oracle("SELECT DISTINCT situation FROM situational_stats")}
    assert actual == _documented_situations()


# ── 10. franchises helpers ─────────────────────────────────────────────────────

_ALIAS_CASES = sorted(
    (alias, key) for key, f in franchises.FRANCHISES.items() for alias in f.aliases)


@pytest.mark.parametrize("alias,canonical", _ALIAS_CASES)
def test_canonical_team_maps_every_alias(alias, canonical):
    assert franchises.canonical_team(alias) == canonical
    assert franchises.canonical_team(alias.lower()) == canonical


# Hand-checked club names. LV 1990 is left out on purpose: the table says
# "Oakland Raiders" but the club played in Los Angeles 1982-1994.
@pytest.mark.parametrize("season,team,expected", [
    (1990, "ARI", "Phoenix Cardinals"),
    (1990, "TEN", "Houston Oilers"),
    (1990, "LAC", "San Diego Chargers"),
    (1990, "LA", "Los Angeles Rams"),
    (1990, "WAS", "Washington Redskins"),
    (2024, "ARI", "Arizona Cardinals"),
    (2024, "LV", "Las Vegas Raiders"),
    (2024, "WAS", "Washington Commanders"),
    (2024, "STL", "Los Angeles Rams"),
])
def test_era_name_for_1990_and_2024(season, team, expected):
    assert franchises.era_name(season, team) == expected


@pytest.mark.parametrize("team", sorted(franchises.FRANCHISES))
def test_availability_2024_every_franchise_active(team):
    assert franchises.availability(team, 2024) is None


# 1990 had 28 clubs. Carolina and Jacksonville began in 1995, Baltimore in 1996,
# Houston in 2002 (hand-known league history, not read from franchises.py).
@pytest.mark.parametrize("team,first_season", [("CAR", 1995), ("JAX", 1995),
                                               ("BAL", 1996), ("HOU", 2002)])
def test_availability_1990_not_yet_active(team, first_season):
    note = franchises.availability(team, 1990)
    assert note["status"] == "franchise_not_yet_active"
    assert note["first_season"] == first_season


def test_availability_1990_the_other_28_clubs_are_active():
    active = [t for t in franchises.FRANCHISES if franchises.availability(t, 1990) is None]
    assert len(active) == 28
    assert not {"CAR", "JAX", "BAL", "HOU"} & set(active)
