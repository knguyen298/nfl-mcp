"""Group N: ids on nfl_roster and nfl_injuries, season_type on nfl_td_luck and nfl_drop_rate.

Every expected value below comes from SQL run directly on the database, never from a tool
call. The SQL sits in a comment next to the value. Values were measured on a copy of the
rebuilt integration database (2024-2026) and the tests read the database at NFL_MCP_DB_PATH
(default /tmp/grpF_work.duckdb). They skip when the file is missing or has no season_type
column on the derived player tables (an unrebuilt database).

Interface contract
- nfl_roster rows gain gsis_id. nfl_injuries rows gain gsis_id and season_type. Both sit
  right after full_name. The old keys stay.
- nfl_injuries.season_type is the table value, else derived from game_type: REG is REG and
  WC, DIV, CON, SB are POST. (The 2024 rows have an empty season_type in the table.)
- nfl_td_luck and nfl_drop_rate rows gain season_type (after season). v0.7.0: both require
  season_type, exact upper case, one of REG, POST, ALL (ALL returns the REG rows and the POST
  rows, one row per type). A missing value, lower case or any other value is INVALID_ARGUMENT
  and lists the valid values. A database whose derived player tables lack season_type answers
  STALE_DERIVED_TABLES for every value (v0.7.3: the text says the server updates the database).
- ORDER BY has season_type DESC (REG first) before rowid.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nfl_mcp import discovery, server, tools
from nfl_mcp.database import get_db_connection

REAL_DB = Path(os.environ.get("NFL_MCP_DB_PATH", "/tmp/grpF_work.duckdb"))


@pytest.fixture
def real(monkeypatch):
    """Oracle connection through get_db_connection, so tool calls open the same config."""
    if not REAL_DB.exists():
        pytest.skip(f"{REAL_DB} missing")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(REAL_DB))
    with get_db_connection() as conn:
        cols = {r[0] for r in conn.execute("DESCRIBE player_td_luck").fetchall()}
        if "season_type" not in cols:
            pytest.skip("derived tables not rebuilt (no season_type)")
        yield conn


@pytest.fixture
def full(real):
    """real, but only on a multi-season database: the expected values here were measured on
    /tmp/grpF_work.duckdb (plays for 2024-2026). CI ingests 2024 only, so these skip there."""
    low, high = _one(real, "SELECT MIN(season), MAX(season) FROM plays")[0]
    if low is None or low > 2024 or high < 2026:
        pytest.skip(f"needs plays for 2024-2026; this database has {low}-{high}")
    return real


def _one(conn, sql, params=()):
    return conn.execute(sql, list(params)).fetchall()


def _all_rows(tool, key, **kwargs):
    """Every row a tool returns, read 500 at a time."""
    rows, offset = [], 0
    while True:
        res = tool(limit=500, offset=offset, **kwargs)
        rows += res[key]
        if not res["truncated"]:
            return rows
        offset += 500


# ── 1. gsis_id on roster rows ─────────────────────────────────────────────────

@pytest.mark.integration
def test_roster_row_carries_gsis_id(real):
    # SELECT gsis_id FROM rosters WHERE team='KC' AND season=2024 AND full_name='Patrick Mahomes'
    #   -> 00-0033873 (one row)
    assert _one(real, "SELECT gsis_id FROM rosters WHERE team='KC' AND season=2024 "
                      "AND full_name='Patrick Mahomes'") == [("00-0033873",)]
    res = tools.nfl_roster(team="KC", season=2024, position="QB")
    row = next(r for r in res["players"] if r["full_name"] == "Patrick Mahomes")
    assert row["gsis_id"] == "00-0033873"


@pytest.mark.integration
def test_roster_keys_keep_old_keys_and_put_gsis_id_after_name(real):
    res = tools.nfl_roster(team="KC", season=2024, limit=3)
    assert list(res["players"][0]) == [
        "full_name", "gsis_id", "position", "depth_chart_position", "team", "season",
        "jersey_number", "status", "years_exp", "college", "height", "weight"]


@pytest.mark.integration
def test_roster_gsis_id_count_matches_sql(real):
    # SELECT count(*), count(gsis_id) FROM rosters WHERE team='KC' AND season=2024  (the
    # count of rows with an id is what the tool must return, null ids stay null)
    total, with_id = _one(real, "SELECT count(*), count(gsis_id) FROM rosters "
                                "WHERE team='KC' AND season=2024")[0]
    res = tools.nfl_roster(team="KC", season=2024, limit=500)
    assert len(res["players"]) == total
    assert sum(1 for r in res["players"] if r["gsis_id"]) == with_id


@pytest.mark.integration
def test_roster_null_gsis_id_stays_null(full):
    # SELECT count(*) FROM rosters WHERE gsis_id IS NULL  -> 5 rows over 2024-2026
    assert _one(full, "SELECT count(*) FROM rosters WHERE gsis_id IS NULL") == [(5,)]
    # SELECT team, full_name FROM rosters WHERE gsis_id IS NULL AND season=2024 -> one row
    (team, name), = _one(full, "SELECT team, full_name FROM rosters "
                               "WHERE gsis_id IS NULL AND season=2024")
    res = tools.nfl_roster(team=team, season=2024, limit=500)
    assert [r["gsis_id"] for r in res["players"] if r["full_name"] == name] == [None]


# ── 2. gsis_id and season_type on injury rows ─────────────────────────────────

@pytest.mark.integration
def test_injury_row_carries_gsis_id(real):
    # SELECT gsis_id, report_status FROM injuries WHERE team='KC' AND season=2024 AND week=18
    #   AND full_name='Patrick Mahomes'  -> ('00-0033873', 'Doubtful')
    assert _one(real, "SELECT gsis_id, report_status FROM injuries WHERE team='KC' AND "
                      "season=2024 AND week=18 AND full_name='Patrick Mahomes'") == [
        ("00-0033873", "Doubtful")]
    res = tools.nfl_injuries(team="KC", season=2024, week=18, player="Mahomes", season_type="REG")
    assert [(r["gsis_id"], r["report_status"]) for r in res["injuries"]] == [
        ("00-0033873", "Doubtful")]


@pytest.mark.integration
def test_injury_keys_order(real):
    res = tools.nfl_injuries(team="KC", season=2024, week=18, player="Mahomes", season_type="REG")
    assert list(res["injuries"][0]) == [
        "full_name", "gsis_id", "position", "team", "season", "week", "season_type",
        "report_primary_injury", "report_secondary_injury", "report_status",
        "practice_primary_injury", "practice_secondary_injury", "practice_status"]


@pytest.mark.integration
def test_injury_season_type_from_table_when_present(full):
    # SELECT game_type FROM injuries WHERE season=2025 AND week=19 AND team='BUF'
    #   AND full_name='Cam Lewis'  -> 'WC' (the table's own season_type says POST too). The
    #   tool reads game_type only: a 2024-only ingest has no season_type column at all.
    assert _one(full, "SELECT game_type FROM injuries WHERE season=2025 AND "
                      "week=19 AND team='BUF' AND full_name='Cam Lewis'") == [("WC",)]
    res = tools.nfl_injuries(team="BUF", season=2025, week=19, player="Cam Lewis", season_type="POST")
    assert [r["season_type"] for r in res["injuries"]] == ["POST"]


@pytest.mark.integration
def test_injury_season_type_derived_when_table_is_null(real):
    # SELECT season_type, game_type FROM injuries WHERE season=2024 AND week=20 AND team='BAL'
    #   AND full_name='Zay Flowers'  -> (NULL, 'DIV')  so the tool says POST
    assert _one(real, "SELECT game_type FROM injuries WHERE season=2024 AND "
                      "week=20 AND team='BAL' AND full_name='Zay Flowers'") == [("DIV",)]
    res = tools.nfl_injuries(team="BAL", season=2024, week=20, player="Zay Flowers", season_type="ALL")
    assert [r["season_type"] for r in res["injuries"]] == ["POST"]
    # SELECT season_type, game_type FROM injuries WHERE season=2024 AND week=9 AND team='KC'
    #   AND full_name='Patrick Mahomes'  -> (NULL, 'REG')  so the tool says REG
    assert _one(real, "SELECT game_type FROM injuries WHERE season=2024 AND "
                      "week=9 AND team='KC' AND full_name='Patrick Mahomes'") == [("REG",)]
    res = tools.nfl_injuries(team="KC", season=2024, week=9, player="Mahomes", season_type="REG")
    assert [r["season_type"] for r in res["injuries"]] == ["REG"]


@pytest.mark.integration
def test_injury_season_type_never_empty_and_matches_game_type(real):
    # SELECT game_type, count(*) FROM injuries WHERE season=2024 GROUP BY 1 ->
    #   CON 45, DIV 74, REG 5954, SB 15, WC 127  (so POST = 261, REG = 5954)
    by_type = dict(_one(real, "SELECT game_type, count(*) FROM injuries WHERE season=2024 "
                              "GROUP BY 1"))
    assert by_type == {"CON": 45, "DIV": 74, "REG": 5954, "SB": 15, "WC": 127}
    reg, post = by_type["REG"], 45 + 74 + 15 + 127
    # Pull every 2024 row through the tool one page at a time.
    seen: dict[str, int] = {}
    offset = 0
    while True:
        res = tools.nfl_injuries(season=2024, limit=500, offset=offset, season_type="ALL")
        for r in res["injuries"]:
            seen[r["season_type"]] = seen.get(r["season_type"], 0) + 1
        if not res["truncated"]:
            break
        offset += 500
    assert seen == {"REG": reg, "POST": post}


# ── 3. season_type on nfl_td_luck ─────────────────────────────────────────────

# Matthew Stafford (00-0026498) has a REG and a POST td_luck row in 2024 and in 2025.
# SELECT season, season_type, total_td_luck_score FROM player_td_luck
#   WHERE player_id='00-0026498' ORDER BY season, season_type DESC ->
#   2024 REG -0.52, 2024 POST -0.09, 2025 REG -1.07, 2025 POST -0.05, 2026 REG -0.77
STAFFORD = "00-0026498"


@pytest.mark.integration
def test_td_luck_player_has_both_season_types(real):
    # SELECT season_type, count(*) FROM player_td_luck WHERE player_id='00-0026498'
    #   AND season=2024 GROUP BY 1  -> POST 1, REG 1
    assert dict(_one(real, "SELECT season_type, count(*) FROM player_td_luck WHERE "
                           "player_id=? AND season=2024 GROUP BY 1", [STAFFORD])) == {"POST": 1, "REG": 1}
    res = tools.nfl_td_luck(player_id=STAFFORD, season=2024, season_type="ALL")
    got = {r["season_type"]: r["total_td_luck_score"] for r in res["td_luck"]}
    # SELECT season_type, total_td_luck_score FROM player_td_luck WHERE player_id='00-0026498'
    #   AND season=2024  -> POST -0.09, REG -0.52
    assert got == {"POST": -0.09, "REG": -0.52}
    assert res["count"] == 2


@pytest.mark.integration
def test_td_luck_rows_carry_season_type_key_after_season(real):
    res = tools.nfl_td_luck(player_id=STAFFORD, season=2024, season_type="REG")
    assert list(res["td_luck"][0]) == [
        "player_id", "full_name", "position", "team", "season", "season_type", "rec_td",
        "rec_td_exp", "rec_td_luck", "rush_td", "rush_td_exp", "rush_td_luck",
        "total_td_luck_score"]


@pytest.mark.integration
def test_td_luck_all_returns_both_types(real):
    # SELECT count(*) FROM player_td_luck WHERE season=2024  -> 750 (REG 599, POST 151)
    assert _one(real, "SELECT count(*) FROM player_td_luck WHERE season=2024") == [(750,)]
    assert dict(_one(real, "SELECT season_type, count(*) FROM player_td_luck WHERE season=2024 "
                           "GROUP BY 1")) == {"REG": 599, "POST": 151}
    res = tools.nfl_td_luck(season=2024, limit=500, season_type="ALL")
    assert res["count"] == 500 and res["truncated"] is True
    res = tools.nfl_td_luck(season=2024, limit=500, offset=500, season_type="ALL")
    assert res["count"] == 250 and res["truncated"] is False


@pytest.mark.integration
@pytest.mark.parametrize("value,expected", [
    ("REG", 599), ("POST", 151)])
def test_td_luck_season_type_filter(real, value, expected):
    # SELECT count(*) FROM player_td_luck WHERE season=2024 AND season_type='REG'   -> 599
    # SELECT count(*) FROM player_td_luck WHERE season=2024 AND season_type='POST'  -> 151
    sql_value = value
    assert _one(real, "SELECT count(*) FROM player_td_luck WHERE season=2024 AND "
                      "season_type=?", [sql_value]) == [(expected,)]
    rows = _all_rows(tools.nfl_td_luck, "td_luck", season=2024, season_type=value)
    assert len(rows) == expected
    assert {r["season_type"] for r in rows} == {sql_value}


@pytest.mark.integration
def test_td_luck_filter_for_one_player(full):
    res = tools.nfl_td_luck(player_id=STAFFORD, season_type="POST")
    # SELECT season FROM player_td_luck WHERE player_id='00-0026498' AND season_type='POST'
    #   ORDER BY season  -> 2024, 2025 (the tool sorts by score, so compare as a set)
    assert sorted(r["season"] for r in res["td_luck"]) == [2024, 2025]
    res = tools.nfl_td_luck(player_id=STAFFORD, season_type="REG")
    # ... AND season_type='REG' ORDER BY season -> 2024, 2025, 2026
    assert sorted(r["season"] for r in res["td_luck"]) == [2024, 2025, 2026]


@pytest.mark.integration
def test_td_luck_order_is_stable_reg_before_post(full):
    # Stafford, no filter, ordered by score ASC: SELECT season, season_type FROM player_td_luck WHERE player_id='00-0026498'
    #   ORDER BY total_td_luck_score ASC, season, season_type DESC ->
    #   2025 REG -1.07, 2026 REG -0.77, 2024 REG -0.52, 2024 POST -0.09, 2025 POST -0.05
    expected = [tuple(r) for r in _one(full, "SELECT season, season_type FROM player_td_luck WHERE "
                                             "player_id=? ORDER BY total_td_luck_score ASC, season, "
                                             "season_type DESC", [STAFFORD])]
    assert expected == [(2025, "REG"), (2026, "REG"), (2024, "REG"), (2024, "POST"), (2025, "POST")]
    first = tools.nfl_td_luck(player_id=STAFFORD, season_type="ALL")["td_luck"]
    assert [(r["season"], r["season_type"]) for r in first] == expected
    assert tools.nfl_td_luck(player_id=STAFFORD, season_type="ALL")["td_luck"] == first


@pytest.mark.integration
def test_td_luck_paging_with_season_type(full):
    # SELECT count(*) FROM player_td_luck WHERE season=2025 AND season_type='POST'  -> 159
    assert _one(full, "SELECT count(*) FROM player_td_luck WHERE season=2025 AND "
                      "season_type='POST'") == [(159,)]
    page1 = tools.nfl_td_luck(season=2025, season_type="POST", limit=100)
    page2 = tools.nfl_td_luck(season=2025, season_type="POST", limit=100, offset=100)
    assert (page1["count"], page1["truncated"]) == (100, True)
    assert (page2["count"], page2["truncated"]) == (59, False)
    ids = [(r["player_id"], r["team"]) for r in page1["td_luck"] + page2["td_luck"]]
    assert len(ids) == len(set(ids)) == 159


# ── 4. season_type on nfl_drop_rate ───────────────────────────────────────────

# K.Juszczyk (00-0029892, SF) has REG and POST drop_rate rows in 2025.
# SELECT season, season_type, catchable_targets FROM player_drop_rate
#   WHERE player_id='00-0029892' ORDER BY season, season_type DESC ->
#   2024 REG 19, 2025 REG 24, 2025 POST 5, 2026 REG 8
JUSZCZYK = "00-0029892"


@pytest.mark.integration
def test_drop_rate_player_has_both_season_types(full):
    # SELECT season_type, catchable_targets FROM player_drop_rate WHERE player_id='00-0029892'
    #   AND season=2025  -> POST 5, REG 24
    assert dict(_one(full, "SELECT season_type, catchable_targets FROM player_drop_rate "
                           "WHERE player_id=? AND season=2025", [JUSZCZYK])) == {"POST": 5, "REG": 24}
    res = tools.nfl_drop_rate(player_id=JUSZCZYK, season=2025, season_type="ALL")
    assert {r["season_type"]: r["catchable_targets"] for r in res["drop_rate"]} == {"POST": 5, "REG": 24}
    assert res["count"] == 2


@pytest.mark.integration
def test_drop_rate_rows_carry_season_type_key_after_season(full):
    res = tools.nfl_drop_rate(player_id=JUSZCZYK, season=2025, season_type="REG")
    assert list(res["drop_rate"][0]) == [
        "player_id", "player", "team", "season", "season_type", "catchable_targets", "drops",
        "drop_rate_pct", "contested_targets", "created_receptions"]


@pytest.mark.integration
def test_drop_rate_all_returns_both_types(real):
    # SELECT count(*) FROM player_drop_rate WHERE season=2024  -> 602 (REG 486, POST 116)
    assert dict(_one(real, "SELECT season_type, count(*) FROM player_drop_rate WHERE season=2024 "
                           "GROUP BY 1")) == {"REG": 486, "POST": 116}
    first = tools.nfl_drop_rate(season=2024, limit=500, season_type="ALL")
    assert first["count"] == 500 and first["truncated"] is True
    second = tools.nfl_drop_rate(season=2024, limit=500, offset=500, season_type="ALL")
    assert second["count"] == 102 and second["truncated"] is False


@pytest.mark.integration
@pytest.mark.parametrize("value,expected", [
    ("REG", 486), ("POST", 116)])
def test_drop_rate_season_type_filter(real, value, expected):
    # SELECT count(*) FROM player_drop_rate WHERE season=2024 AND season_type='REG'   -> 486
    # SELECT count(*) FROM player_drop_rate WHERE season=2024 AND season_type='POST'  -> 116
    sql_value = value
    assert _one(real, "SELECT count(*) FROM player_drop_rate WHERE season=2024 AND "
                      "season_type=?", [sql_value]) == [(expected,)]
    rows = _all_rows(tools.nfl_drop_rate, "drop_rate", season=2024, season_type=value)
    assert len(rows) == expected
    assert {r["season_type"] for r in rows} == {sql_value}


@pytest.mark.integration
def test_drop_rate_season_type_with_min_targets(real):
    # SELECT count(*) FROM player_drop_rate WHERE season=2024 AND season_type='POST'
    #   AND catchable_targets >= 5  -> n (computed here, the tool must match)
    (n,), = _one(real, "SELECT count(*) FROM player_drop_rate WHERE season=2024 AND "
                       "season_type='POST' AND catchable_targets >= 5")
    res = tools.nfl_drop_rate(season=2024, season_type="POST", min_targets=5, limit=500)
    assert res["count"] == n


@pytest.mark.integration
def test_drop_rate_order_is_stable_reg_before_post(full):
    # SELECT season, season_type FROM player_drop_rate WHERE player_id='00-0029892'
    #   ORDER BY drop_rate_pct DESC NULLS LAST, season, season_type DESC ->
    #   all four rows have drop_rate_pct 0.0, so 2024 REG, 2025 REG, 2025 POST, 2026 REG
    expected = [tuple(r) for r in _one(full, "SELECT season, season_type FROM player_drop_rate "
                                             "WHERE player_id=? ORDER BY drop_rate_pct DESC NULLS "
                                             "LAST, season, season_type DESC", [JUSZCZYK])]
    assert expected == [(2024, "REG"), (2025, "REG"), (2025, "POST"), (2026, "REG")]
    rows = tools.nfl_drop_rate(player_id=JUSZCZYK, season_type="ALL")["drop_rate"]
    assert [(r["season"], r["season_type"]) for r in rows] == expected


@pytest.mark.integration
def test_drop_rate_paging_with_season_type(full):
    # SELECT count(*) FROM player_drop_rate WHERE season=2025 AND season_type='POST'  -> 124
    assert _one(full, "SELECT count(*) FROM player_drop_rate WHERE season=2025 AND "
                      "season_type='POST'") == [(124,)]
    page1 = tools.nfl_drop_rate(season=2025, season_type="POST", limit=100)
    page2 = tools.nfl_drop_rate(season=2025, season_type="POST", limit=100, offset=100)
    assert (page1["count"], page1["truncated"]) == (100, True)
    assert (page2["count"], page2["truncated"]) == (24, False)


# ── 5. invalid season_type (no database needed) ───────────────────────────────

@pytest.mark.parametrize("tool", [tools.nfl_td_luck, tools.nfl_drop_rate])
@pytest.mark.parametrize("value", ["SB", "WC", "bogus", "", 7, "reg", "Post", " REG ", None])
def test_invalid_season_type_is_invalid_argument(tool, value, monkeypatch, tmp_path):
    # Checked before any database read: the file does not exist.
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(tmp_path / "missing.duckdb"))
    res = tool(season=2024, season_type=value)
    assert res["code"] == "INVALID_ARGUMENT"
    assert "season_type" in res["error"] and "REG, POST, ALL" in res["error"]


# ── 6. schema, descriptions and discovery facts ───────────────────────────────

def _tool(name):
    return next(t for t in server.TOOLS if t.name == name)


@pytest.mark.parametrize("name", ["nfl_td_luck", "nfl_drop_rate"])
def test_schema_has_season_type_enum_without_default(name):
    prop = _tool(name).inputSchema["properties"]["season_type"]
    assert prop["enum"] == ["REG", "POST", "ALL"]
    assert "default" not in prop
    assert "season_type" in _tool(name).inputSchema["required"]


@pytest.mark.parametrize("name", ["nfl_td_luck", "nfl_drop_rate"])
def test_description_says_rows_carry_season_type(name):
    text = _tool(name).description
    assert "rows do not say which" not in text
    assert "each row has a season_type" in text


@pytest.mark.parametrize("name,words", [
    ("nfl_roster", ["gsis_id"]), ("nfl_injuries", ["gsis_id", "season_type"])])
def test_roster_and_injury_descriptions_name_returned_ids(name, words):
    text = _tool(name).description
    assert "do not carry it" not in text
    for word in words:
        assert "Each row carries" in text and word in text.split("Each row carries", 1)[1]


def test_discovery_facts():
    notes = " ".join(discovery.DATA_NOTES)
    assert "nfl_roster and nfl_injuries return gsis_id on every row" in notes
    assert "player_td_luck and player_drop_rate" in notes
    grain, _ = discovery.table_info("player_td_luck")
    assert "season type" in grain
    assert "game_type" in discovery.table_info("injuries")[0]


# ── 7. a database built before season_type (derived player tables not rebuilt) ─

@pytest.fixture
def stale_db(tmp_path, monkeypatch):
    """Derived player tables without season_type, two rows each (hand-made)."""
    import duckdb

    path = tmp_path / "stale.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE player_td_luck (player_id VARCHAR, full_name VARCHAR, "
                "position VARCHAR, team VARCHAR, season INTEGER, rec_td DOUBLE, rec_td_exp DOUBLE, "
                "rec_td_luck DOUBLE, rush_td DOUBLE, rush_td_exp DOUBLE, rush_td_luck DOUBLE, "
                "total_td_luck_score DOUBLE)")
    con.execute("INSERT INTO player_td_luck VALUES ('00-0000001', 'A One', 'WR', 'KC', 2024, "
                "1, 1, 0, 0, 0, 0, 0.5), ('00-0000002', 'B Two', 'RB', 'KC', 2024, 1, 1, 0, 0, 0, "
                "0, -0.5)")
    con.execute("CREATE TABLE player_drop_rate (player_id VARCHAR, player VARCHAR, team VARCHAR, "
                "season INTEGER, catchable_targets BIGINT, drops BIGINT, drop_rate_pct DOUBLE, "
                "contested_targets BIGINT, created_receptions BIGINT)")
    con.execute("INSERT INTO player_drop_rate VALUES ('00-0000001', 'A.One', 'KC', 2024, 10, 1, "
                "10.0, 1, 0), ('00-0000002', 'B.Two', 'KC', 2024, 20, 1, 5.0, 1, 0)")
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))


@pytest.mark.parametrize("tool,key", [(tools.nfl_td_luck, "td_luck"),
                                      (tools.nfl_drop_rate, "drop_rate")])
@pytest.mark.parametrize("value", ["REG", "POST", "ALL"])
def test_stale_tables_refuse_every_season_type(stale_db, tool, key, value):
    # Rows without a season_type would blend REG and POST, so no rows come back at all.
    res = tool(season=2024, season_type=value)
    assert res["code"] == "STALE_DERIVED_TABLES"
    assert "updates the database automatically" in res["error"]   # v0.7.3: no command
    assert key not in res


def test_stale_fallback_does_not_hide_other_binder_errors(tmp_path, monkeypatch):
    import duckdb

    path = tmp_path / "broken.duckdb"
    con = duckdb.connect(str(path))
    # season_type is there; full_name is not. That is a broken table, not a stale one.
    con.execute("CREATE TABLE player_td_luck (player_id VARCHAR, season INTEGER, season_type VARCHAR)")
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    res = tools.nfl_td_luck(season=2024, season_type="REG")
    assert "full_name" in res["error"] and res.get("code") != "STALE_DERIVED_TABLES"
