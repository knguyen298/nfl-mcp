"""v0.7.1: points in nfl_team_records (schema 3), games in nfl_player_stats, nfl_query hints,
the missing-player error, SQL syntax errors, whole numbers, per-block definitions and
descriptions.

Source: the v0.7.1 brief (items 1-9). Two models answered 20 questions through mcpo on v0.7.0:
both got every team points number wrong (no tool returned points), and one invented playoff
results because nfl_player_stats had no games count.

Every expected value comes from SQL on raw tables (schedules, plays, player_stats,
snap_counts), never from a tool or a derived table. The SQL sits in a comment next to the
value. Values were measured on a 2024-only database (the shape GitHub CI ingests).

Databases. Integration tests read NFL_MCP_DB_PATH when it is set. Otherwise the tests that need
schema 3 (points in team_records) read /tmp/ci2024/v071.duckdb (a copy of v070.duckdb upgraded
by this nfl-mcp) and the rest read /tmp/ci2024/v070.duckdb. Each skips with a reason when its
file is missing. The unit tests build small databases in tmp_path and need no file.

Interface contract tested here
- team_records gains points_for, points_against and point_diff (regular season, the current
  season to date; NULL on ESPN rows). nfl_team_records returns them. migrate.SCHEMA_VERSION is
  3 and a schema-2 database is upgraded without the network to the same table a fresh build
  makes. On a database whose team_records lacks the columns, nfl_team_records still returns
  the records, without points, with a `warning` that names `nfl-mcp upgrade`.
- Each nfl_player_stats season entry has `games`: distinct games of that season and season
  type in which the player has a snap_counts row with at least one snap, or a player_stats
  row. definitions["games"] names both sources.
- nfl_query: a truncated result has a `hint` that names the row limit, max_rows and its
  maximum (500), information_schema.columns and nfl_schema. DESCRIBE, SHOW and SUMMARIZE
  return up to 500 rows when max_rows is not given; other statements keep 100. The
  inputSchema has no default for max_rows (mcpo sends a schema default on every call).
- A missing player gives MISSING_PLAYER with the exact form {"player": "..."} and says that
  other names such as player_name are not accepted. The same in nfl_player_lookup,
  nfl_search_plays (player_role) and nfl_compare (the pair form).
- A DuckDB parser error gives code SQL_SYNTAX_ERROR and a hint with the DuckDB forms. The
  desc hint stays.
- nfl_search_plays returns play_id, season, week, qtr, down, ydstogo, yards_gained, touchdown
  and interception as ints; epa and wpa stay floats. nfl_injuries, nfl_schedule,
  nfl_snap_counts and nfl_roster return season (and week) as ints when the table stores DOUBLE.
- nfl_player_stats with stat_type sends only that block's definitions plus the shared ones.

Player ids (players table): Saquon Barkley 00-0034844, Travis Kelce 00-0030506, Josh Allen
00-0034857 (the Bills QB; another Josh Allen is 00-0030833), Courtland Sutton 00-0034348.
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
# Schema 3 (points in team_records) and any schema (the rest).
DB3 = Path(_ENV_DB or "/tmp/ci2024/v071.duckdb")
DB_ANY = Path(_ENV_DB or "/tmp/ci2024/v070.duckdb")
README = Path(__file__).resolve().parents[2] / "README.md"

BARKLEY, KELCE, ALLEN, SUTTON = "00-0034844", "00-0030506", "00-0034857", "00-0034348"
TOOLS_BY_NAME = {t.name: t for t in server.TOOLS}


# ── helpers ────────────────────────────────────────────────────────────────────

def _open_real(path: Path, monkeypatch):
    if not path.exists():
        pytest.skip(f"{path} missing")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return get_db_connection()


@pytest.fixture
def real(monkeypatch):
    """Any-schema 2024 database, read-only."""
    with _open_real(DB_ANY, monkeypatch) as conn:
        if not conn.execute("SELECT count(*) FROM plays WHERE season = 2024").fetchone()[0]:
            pytest.skip("needs 2024 plays")
        yield conn


@pytest.fixture
def real3(monkeypatch):
    """Schema-3 2024 database, read-only."""
    with _open_real(DB3, monkeypatch) as conn:
        cols = {r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'team_records'"
        ).fetchall()}
        if "points_for" not in cols:
            pytest.skip(f"{DB3} has no team_records.points_for (schema 3 not built)")
        yield conn


def _raw(name, args):
    """A tool call through server.call_tool: the tool's checks and the one error shape."""
    async def go():
        return await server.call_tool(name, args)
    res = anyio.run(go)
    return res.isError, json.loads(res.content[0].text)


def _table(con, name, cols: dict[str, str], rows: list[dict]):
    """Create table `name` with `cols` ({column: type}) and insert `rows` (missing keys NULL)."""
    ddl = ", ".join(f'"{c}" {t}' for c, t in cols.items())
    con.execute(f'CREATE TABLE "{name}" ({ddl})')
    marks = ", ".join("?" for _ in cols)
    for row in rows:
        con.execute(f'INSERT INTO "{name}" VALUES ({marks})', [row.get(c) for c in cols])


def _desc(name):
    return TOOLS_BY_NAME[name].description


# ═══ 1. points in nfl_team_records ═════════════════════════════════════════════

def test_team_records_requires_points_columns():
    assert {"points_for", "points_against", "point_diff"} <= set(REQUIRED_COLUMNS["team_records"])


def test_schema_version_is_3_and_team_records_is_rewritten_by_the_upgrade():
    assert migrate.SCHEMA_VERSION == 3
    assert "team_records" in migrate.DERIVED_TABLES
    assert "team_records" in migrate.UPGRADE_REWRITES


# Synthetic schedules: KC 2024 beats BUF 30-20 (REG), loses at BUF 21-24 (REG), ties DEN
# 17-17 (REG), beats BUF 40-10 in the WC round (excluded), and has an unplayed week 18 game
# (NULL scores, excluded). Hand totals, regular season:
#   KC  PF 30+21+17 = 68, PA 20+24+17 = 61, diff 7;  BUF PF 20+24 = 44, PA 30+21 = 51, diff -7
#   DEN PF 17, PA 17, diff 0.  OAK 2019 (an old code) maps to LV: PF 24, PA 16.
_SCHEDULE_COLS = {"game_id": "VARCHAR", "season": "INTEGER", "game_type": "VARCHAR",
                  "week": "INTEGER", "home_team": "VARCHAR", "away_team": "VARCHAR",
                  "home_score": "INTEGER", "away_score": "INTEGER"}
_SCHEDULE_ROWS = [dict(zip(_SCHEDULE_COLS, row)) for row in [
    ("g1", 2024, "REG", 1, "KC", "BUF", 30, 20),
    ("g2", 2024, "REG", 2, "BUF", "KC", 24, 21),
    ("g3", 2024, "REG", 3, "DEN", "KC", 17, 17),
    ("g4", 2024, "WC", 19, "KC", "BUF", 40, 10),
    ("g5", 2024, "REG", 18, "KC", "DEN", None, None),
    ("g6", 2019, "REG", 1, "OAK", "DEN", 24, 16),
]]


def _records_db(path=":memory:", espn=True):
    con = duckdb.connect(str(path))
    ing._ensure_metadata_table(con)
    _table(con, "schedules", _SCHEDULE_COLS, _SCHEDULE_ROWS)
    if espn:
        con.execute(ing._ESPN_RECORDS_DDL)
        con.execute("INSERT INTO espn_team_records (season, team, espn_abbrev, games, wins, "
                    "losses, ties) VALUES (1985, 'CHI', 'CHI', 16, 15, 1, 0)")
    return con


def _quiet_records(con):
    with contextlib.redirect_stdout(io.StringIO()):
        ing._create_team_records(con)


def test_create_team_records_adds_regular_season_points():
    con = _records_db()
    _quiet_records(con)
    got = {(r[0], r[1]): r[2:] for r in con.execute(
        "SELECT season, team, games, wins, losses, ties, points_for, points_against, point_diff "
        "FROM team_records").fetchall()}
    assert got[(2024, "KC")] == (3, 1, 1, 1, 68, 61, 7)
    assert got[(2024, "BUF")] == (2, 1, 1, 0, 44, 51, -7)
    assert got[(2024, "DEN")] == (1, 0, 0, 1, 17, 17, 0)
    assert got[(2019, "LV")] == (1, 1, 0, 0, 24, 16, 8)
    # ESPN rows have no scores: points are NULL, the record is kept.
    assert got[(1985, "CHI")] == (16, 15, 1, 0, None, None, None)


def test_team_records_points_are_integer_columns():
    con = _records_db()
    _quiet_records(con)
    types = dict(con.execute(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'team_records'"
    ).fetchall())
    for col in ("season", "points_for", "points_against", "point_diff"):
        assert types[col] == "INTEGER", (col, types[col])


def test_team_records_points_without_espn_table():
    con = _records_db(espn=False)
    _quiet_records(con)
    assert con.execute("SELECT points_for, points_against, point_diff FROM team_records "
                       "WHERE season = 2024 AND team = 'KC'").fetchone() == (68, 61, 7)


def test_check_flags_team_records_without_points(tmp_path):
    con = _records_db(tmp_path / "v2.duckdb")
    _quiet_records(con)
    for col in ("points_for", "points_against", "point_diff"):
        con.execute(f"ALTER TABLE team_records DROP COLUMN {col}")
    status = migrate.check(con)
    con.close()
    derived = next(s for s in status.steps if s.name == "derived_tables")
    assert derived.needed and not derived.needs_network
    assert "team_records.points_for" in derived.detail


@pytest.fixture
def stale_records(tmp_path, monkeypatch):
    """A database whose team_records was built by v0.7.0 (no points columns)."""
    path = tmp_path / "stale.duckdb"
    con = _records_db(path, espn=False)
    _quiet_records(con)
    for col in ("points_for", "points_against", "point_diff"):
        con.execute(f"ALTER TABLE team_records DROP COLUMN {col}")
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


def test_stale_team_records_still_returns_records_with_a_warning(stale_records):
    res = tools.nfl_team_records(team="KC", season=2024)
    assert "error" not in res, res
    (row,) = res["records"]
    assert (row["wins"], row["losses"], row["ties"]) == (1, 1, 1)
    assert not {"points_for", "points_against", "point_diff"} & set(row), row
    assert "nfl-mcp upgrade" in res["warning"] and "points" in res["warning"]


def test_stale_team_records_over_mcp_is_not_an_error(stale_records):
    is_error, payload = _raw("nfl_team_records", {"team": "KC", "season": 2024})
    assert not is_error, payload
    assert "nfl-mcp upgrade" in payload["warning"]


def test_team_records_with_points_has_no_warning(tmp_path, monkeypatch):
    path = tmp_path / "fresh.duckdb"
    con = _records_db(path)
    _quiet_records(con)
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    res = tools.nfl_team_records(season=2024)
    assert "warning" not in res
    by_team = {r["team"]: r for r in res["records"]}
    assert (by_team["KC"]["points_for"], by_team["KC"]["points_against"],
            by_team["KC"]["point_diff"]) == (68, 61, 7)


# SELECT team, sum(pf), sum(pa) FROM (SELECT home_team team, home_score pf, away_score pa
#   FROM schedules WHERE season=2024 AND game_type='REG' UNION ALL SELECT away_team, away_score,
#   home_score FROM schedules WHERE season=2024 AND game_type='REG')
# WHERE team IN ('DET','PHI') GROUP BY team  -> DET 564/342, PHI 463/303
_POINTS_SQL = """
    SELECT sum(pf), sum(pa) FROM (
        SELECT home_team AS team, home_score AS pf, away_score AS pa FROM schedules
        WHERE season = 2024 AND game_type = 'REG' AND home_score IS NOT NULL
        UNION ALL
        SELECT away_team, away_score, home_score FROM schedules
        WHERE season = 2024 AND game_type = 'REG' AND home_score IS NOT NULL
    ) WHERE team = ?
"""


@pytest.mark.integration
@pytest.mark.parametrize("team, pf, pa", [("DET", 564, 342), ("PHI", 463, 303)])
def test_team_records_points_match_schedules(real3, team, pf, pa):
    assert real3.execute(_POINTS_SQL, [team]).fetchone() == (pf, pa)
    res = tools.nfl_team_records(team=team, season=2024)
    (row,) = res["records"]
    assert (row["points_for"], row["points_against"], row["point_diff"]) == (pf, pa, pf - pa)
    assert "warning" not in res


@pytest.mark.integration
def test_every_2024_team_has_points_from_schedules(real3):
    # Every team: points from schedules (regular season, played games) equal team_records.
    expected = {r[0]: r[1:] for r in real3.execute("""
        SELECT team, sum(pf), sum(pa) FROM (
            SELECT home_team AS team, home_score AS pf, away_score AS pa FROM schedules
            WHERE season = 2024 AND game_type = 'REG' AND home_score IS NOT NULL
            UNION ALL
            SELECT away_team, away_score, home_score FROM schedules
            WHERE season = 2024 AND game_type = 'REG' AND home_score IS NOT NULL
        ) GROUP BY team""").fetchall()}
    assert len(expected) == 32
    res = tools.nfl_team_records(season=2024)
    got = {r["source_team"]: (r["points_for"], r["points_against"]) for r in res["records"]}
    assert got == expected


_RECORD_COLS = ("season, team, source_team, games, wins, losses, ties, points_for, "
                "points_against, point_diff, source")


@pytest.fixture(scope="module")
def downgraded_then_upgraded(tmp_path_factory):
    """A copy of the schema-3 database made into a v0.7.0 one (points columns dropped,
    version 2), then upgraded without the network. Returns (original rows, upgraded copy)."""
    if not DB3.exists():
        pytest.skip(f"{DB3} missing")
    src = duckdb.connect(str(DB3), read_only=True)
    try:
        cols = {r[0] for r in src.execute("DESCRIBE team_records").fetchall()}
        if "points_for" not in cols:
            pytest.skip(f"{DB3} has no team_records.points_for (schema 3 not built)")
        order = "ORDER BY season, team, source_team, source"
        original = src.execute(f"SELECT {_RECORD_COLS} FROM team_records {order}").fetchall()
    finally:
        src.close()
    path = tmp_path_factory.mktemp("v071") / "v2.duckdb"
    shutil.copyfile(DB3, path)
    con = duckdb.connect(str(path))
    try:
        for col in ("points_for", "points_against", "point_diff"):
            con.execute(f"ALTER TABLE team_records DROP COLUMN {col}")
        con.execute("DELETE FROM _schema_version WHERE version > 2")
        con.execute("INSERT INTO _schema_version VALUES (2, TIMESTAMP '2026-11-05', '0.7.0')")
        before = migrate.check(con)
        with contextlib.redirect_stdout(io.StringIO()):
            result = migrate.upgrade(con, fetch=False)
        upgraded = con.execute(f"SELECT {_RECORD_COLS} FROM team_records {order}").fetchall()
        version = con.execute("SELECT MAX(version) FROM _schema_version").fetchone()[0]
    finally:
        con.close()
    return {"original": original, "upgraded": upgraded, "before": before, "result": result,
            "version": version}


@pytest.mark.integration
def test_v2_database_needs_the_upgrade_without_network(downgraded_then_upgraded):
    before = downgraded_then_upgraded["before"]
    assert (before.version, before.target, before.upgrade_needed) == (2, 3, True)
    needed = [s for s in before.steps if s.needed]
    assert [s.name for s in needed] == ["derived_tables"], needed
    assert not needed[0].needs_network


@pytest.mark.integration
def test_upgrade_from_v2_stamps_3_and_rebuilds_the_same_team_records(downgraded_then_upgraded):
    out = downgraded_then_upgraded
    assert out["result"].stamped and not out["result"].failed, out["result"]
    assert out["version"] == 3
    assert out["upgraded"] == out["original"]


def test_team_records_description_mentions_points_and_schedule():
    text = _desc("nfl_team_records")
    for word in ("points_for", "points_against", "point_diff"):
        assert word in text, word
    low = text.lower()
    assert "regular season" in low or "regular-season" in low
    assert "to date" in low
    assert "nfl_schedule" in text and "playoff" in low


# ═══ 2. games in nfl_player_stats ══════════════════════════════════════════════
# Source chosen by comparison on 2024 (REG): player_stats has a row only for a game with a stat,
# snap_counts for every game with a snap. Josh Allen played one snap in week 18 and recorded
# no stat:
#   SELECT count(DISTINCT game_id) FROM player_stats WHERE player_id='00-0034857'
#     AND season=2024 AND season_type='REG'                                        -> 16
#   SELECT count(DISTINCT s.game_id) FROM snap_counts s JOIN players p ON p.pfr_id=s.pfr_player_id
#     WHERE p.gsis_id='00-0034857' AND s.season=2024 AND s.game_type='REG'
#     AND COALESCE(offense_snaps,0)+COALESCE(defense_snaps,0)+COALESCE(st_snaps,0) > 0  -> 17
# The official count is 17 games, so snap_counts is the main source; player_stats adds the
# few games a player has no snap row for (and the seasons before snap_counts starts, 2012).
_GAMES_SQL = """
    SELECT count(DISTINCT game_id) FROM (
        SELECT game_id FROM player_stats WHERE player_id = ? AND season = 2024 AND season_type = ?
        UNION
        SELECT s.game_id FROM snap_counts s JOIN players p ON p.pfr_id = s.pfr_player_id
        WHERE p.gsis_id = ? AND s.season = 2024
          AND (CASE WHEN s.game_type = 'REG' THEN 'REG' ELSE 'POST' END) = ?
          AND COALESCE(s.offense_snaps, 0) + COALESCE(s.defense_snaps, 0)
              + COALESCE(s.st_snaps, 0) > 0)
"""


@pytest.mark.integration
@pytest.mark.parametrize("gsis_id, reg, post", [
    # Barkley: 16 REG (rested week 18), 4 POST (WC, DIV, CON, SB).
    (BARKLEY, 16, 4),
    # Kelce: 16 REG, 3 POST (KC had the first-round bye: DIV, CON, SB).
    (KELCE, 16, 3),
    # Josh Allen: 17 REG (one snap in week 18), 3 POST. player_stats alone gives 16.
    (ALLEN, 17, 3),
    # Courtland Sutton: 17 REG (player_stats alone gives 16), 1 POST.
    (SUTTON, 17, 1),
])
def test_player_stats_games_per_season_type(real, gsis_id, reg, post):
    assert real.execute(_GAMES_SQL, [gsis_id, "REG", gsis_id, "REG"]).fetchone()[0] == reg
    assert real.execute(_GAMES_SQL, [gsis_id, "POST", gsis_id, "POST"]).fetchone()[0] == post
    res = tools.nfl_player_stats(player_id=gsis_id, season=2024, season_type="ALL")
    got = {s["season_type"]: s["games"] for s in res["seasons"]}
    assert got == {"REG": reg, "POST": post}


@pytest.mark.integration
def test_player_stats_games_is_an_int_over_mcp(real):
    is_error, payload = _raw("nfl_player_stats",
                             {"player_id": BARKLEY, "season": 2024, "season_type": "REG"})
    assert not is_error, payload
    (row,) = payload["seasons"]
    assert row["games"] == 16 and isinstance(row["games"], int)


def _games_db(path):
    """players, snap_counts and player_stats for one player, P1 (pfr PlayOn00), in 2024:
    g1 REG: snaps and a stat row; g2 REG: snaps only; g3 REG: a stat row only;
    g4 REG: a snap_counts row with zero snaps (not a game played); g5 WC: snaps only."""
    con = duckdb.connect(str(path))
    _table(con, "players", {"gsis_id": "VARCHAR", "display_name": "VARCHAR", "pfr_id": "VARCHAR"},
           [{"gsis_id": "00-0000001", "display_name": "Player One", "pfr_id": "PlayOn00"}])
    snap = {"game_id": "VARCHAR", "season": "INTEGER", "game_type": "VARCHAR",
            "week": "INTEGER", "pfr_player_id": "VARCHAR", "offense_snaps": "DOUBLE",
            "defense_snaps": "DOUBLE", "st_snaps": "DOUBLE"}
    _table(con, "snap_counts", snap, [dict(zip(snap, row)) for row in [
        ("g1", 2024, "REG", 1, "PlayOn00", 40.0, None, None),
        ("g2", 2024, "REG", 2, "PlayOn00", None, None, 3.0),
        ("g4", 2024, "REG", 4, "PlayOn00", 0.0, 0.0, 0.0),
        ("g5", 2024, "WC", 19, "PlayOn00", 50.0, None, None),
    ]])
    stats = {"player_id": "VARCHAR", "season": "INTEGER", "week": "INTEGER",
             "season_type": "VARCHAR", "game_id": "VARCHAR"}
    _table(con, "player_stats", stats, [dict(zip(stats, row)) for row in [
        ("00-0000001", 2024, 1, "REG", "g1"),
        ("00-0000001", 2024, 3, "REG", "g3"),
    ]])
    con.close()


def test_player_games_counts_snaps_and_stat_rows(tmp_path, monkeypatch):
    _games_db(tmp_path / "g.duckdb")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(tmp_path / "g.duckdb"))
    assert tools._player_games("00-0000001") == {(2024, "REG"): 3, (2024, "POST"): 1}
    assert tools._player_games("00-0000002") == {}


def test_player_games_is_empty_without_the_source_tables(tmp_path, monkeypatch):
    path = tmp_path / "none.duckdb"
    duckdb.connect(str(path)).close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    assert tools._player_games("00-0000001") == {}


@pytest.fixture
def one_player(monkeypatch):
    monkeypatch.setattr(tools, "_resolve_player",
                        lambda *a, **k: ({"gsis_id": "00-0000001", "display_name": "Player One"}, None))


def _stats_row(season, season_type):
    return {"season": season, "season_type": season_type, "attempts": 0, "carries": 3,
            "rush_yards": 12, "targets": 2, "receptions": 1, "rec_yards": 9}


def _fake_execute(games_rows, stats_rows):
    def fake(sql, params=None):
        if "snap_counts" in sql:
            return games_rows
        if "me.id" in sql:
            return stats_rows
        return []
    return fake


def test_player_stats_entry_carries_games(monkeypatch, one_player):
    monkeypatch.setattr(tools, "_execute", _fake_execute(
        [{"season": 2024, "season_type": "REG", "games": 15},
         {"season": 2024, "season_type": "POST", "games": 2}],
        [_stats_row(2024, "REG"), _stats_row(2024, "POST")]))
    res = tools.nfl_player_stats(player="Player One", season=2024, season_type="ALL")
    assert [(s["season_type"], s["games"]) for s in res["seasons"]] == [("REG", 15), ("POST", 2)]


def test_player_stats_games_is_none_when_no_source_has_the_season(monkeypatch, one_player):
    monkeypatch.setattr(tools, "_execute", _fake_execute([], [_stats_row(2005, "REG")]))
    (entry,) = tools.nfl_player_stats(player="Player One", season=2005, season_type="REG")["seasons"]
    assert entry["games"] is None


def test_player_stats_keeps_games_with_stat_type(monkeypatch, one_player):
    monkeypatch.setattr(tools, "_execute", _fake_execute(
        [{"season": 2024, "season_type": "REG", "games": 15}], [_stats_row(2024, "REG")]))
    (entry,) = tools.nfl_player_stats(player="Player One", season=2024, season_type="REG",
                                      stat_type="rushing")["seasons"]
    assert set(entry) == {"season", "season_type", "games", "rushing"}
    assert entry["games"] == 15


def test_games_definition_names_its_sources(monkeypatch, one_player):
    monkeypatch.setattr(tools, "_execute", _fake_execute([], []))
    defs = tools.nfl_player_stats(player="Player One", season_type="REG")["definitions"]
    assert "snap_counts" in defs["games"] and "player_stats" in defs["games"]


def test_player_stats_description_mentions_games():
    assert "games" in _desc("nfl_player_stats")


# ═══ 3. nfl_query: truncation hint and the DESCRIBE default ════════════════════

@pytest.fixture
def wide_db(tmp_path, monkeypatch):
    """A table with 150 columns (more than the default 100 rows of DESCRIBE)."""
    path = tmp_path / "wide.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE wide (" + ", ".join(f"c{i} INTEGER" for i in range(150)) + ")")
    con.execute("INSERT INTO wide VALUES (" + ", ".join(str(i) for i in range(150)) + ")")
    con.execute("CREATE TABLE plays (play_id DOUBLE, epa DOUBLE)")
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


@pytest.mark.parametrize("sql", ["DESCRIBE wide", "SHOW wide", "  describe wide;", "SUMMARIZE wide",
                                 "-- columns\nDESCRIBE wide"])
def test_describe_show_summarize_return_up_to_500_rows_by_default(wide_db, sql):
    res = tools.nfl_query(sql)
    assert res["row_count"] == 150 and res["truncated"] is False, res.get("row_count")
    assert "hint" not in res


def test_select_keeps_the_default_of_100_rows(wide_db):
    res = tools.nfl_query("SELECT * FROM range(300)")
    assert res["row_count"] == 100 and res["truncated"] is True


def test_truncated_result_has_a_hint_naming_the_limit_and_the_ways_out(wide_db):
    res = tools.nfl_query("DESCRIBE wide", max_rows=100)
    assert res["row_count"] == 100 and res["truncated"] is True
    hint = res["hint"]
    assert "100" in hint and "max_rows" in hint and "500" in hint
    assert "information_schema.columns" in hint and "nfl_schema" in hint


def test_truncated_at_the_maximum_does_not_suggest_raising_max_rows(wide_db):
    res = tools.nfl_query("SELECT * FROM range(1000)", max_rows=500)
    assert res["truncated"] is True and res["row_count"] == 500
    hint = res["hint"]
    assert "500" in hint
    assert "raise max_rows" not in hint.lower()
    assert "WHERE" in hint


def test_untruncated_result_has_no_hint(wide_db):
    res = tools.nfl_query("SELECT 1 AS x")
    assert res["truncated"] is False and "hint" not in res


def test_query_max_rows_has_no_schema_default():
    # mcpo turns a schema default into a value it sends on every call, which would hide
    # "max_rows not given" from the tool.
    prop = TOOLS_BY_NAME["nfl_query"].inputSchema["properties"]["max_rows"]
    assert "default" not in prop
    assert "DESCRIBE" in prop["description"] and "500" in prop["description"]


@pytest.mark.integration
def test_describe_plays_returns_every_column(real):
    # SELECT count(*) FROM information_schema.columns WHERE table_name = 'plays'  -> 373 here
    n = real.execute("SELECT count(*) FROM information_schema.columns "
                     "WHERE table_name = 'plays'").fetchone()[0]
    assert n > 100
    res = tools.nfl_query("DESCRIBE plays")
    assert res["row_count"] == min(n, 500)
    assert res["truncated"] is (n > 500)


# ═══ 4. missing player: the exact form ═════════════════════════════════════════

def _says_the_form(msg: str, arg: str = "player"):
    assert f'{{"{arg}": "' in msg, msg
    assert "player_id" in msg, msg
    assert "player_name" in msg and "not accepted" in msg, msg
    assert "proxy" in msg, msg


@pytest.mark.parametrize("call", [
    lambda: tools.nfl_player_stats(season=2024, season_type="REG"),
    lambda: tools.nfl_player_stats(player="  ", season_type="REG"),
    lambda: tools.nfl_player_lookup(),
    lambda: tools.nfl_player_lookup(player=" "),
    lambda: tools.nfl_search_plays(season=2024, season_type="REG", player_role="passer"),
])
def test_missing_player_error_shows_the_exact_form(call):
    res = call()
    assert res["code"] == "MISSING_PLAYER", res
    assert res["error"].startswith("player or player_id is required"), res["error"]
    _says_the_form(res["error"])


def test_compare_pair_error_shows_the_exact_form():
    res = tools.nfl_compare(season=2024, season_type="REG")
    msg = res["error"]
    assert '{"player1": "' in msg and '"player2": "' in msg and '{"team1": "' in msg, msg
    assert "not accepted" in msg and "proxy" in msg, msg


def test_missing_player_over_mcp(tmp_path, monkeypatch):
    path = tmp_path / "empty.duckdb"
    duckdb.connect(str(path)).close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    # What arrives when a proxy dropped an unknown player_name: no player at all.
    is_error, payload = _raw("nfl_player_stats", {"season": 2024, "season_type": "REG"})
    assert is_error and payload["code"] == "MISSING_PLAYER", payload
    _says_the_form(payload["error"])


# Open WebUI shows about the first 100 characters of a description in its tool list.
@pytest.mark.parametrize("name, form", [("nfl_player_stats", '{"player": "'),
                                        ("nfl_player_lookup", '{"player": "'),
                                        ("nfl_compare", "player1 and player2")])
def test_player_tool_names_its_argument_in_the_first_100_characters(name, form):
    head = _desc(name)[:100]
    assert form in head, head


# ═══ 5. SQL syntax errors ══════════════════════════════════════════════════════

@pytest.mark.parametrize("sql", ["SHOW COLUMNS FROM plays LIKE '%conv%'", "SELEC 1",
                                 "SELECT * FROM plays WHERE"])
def test_parser_error_has_code_and_duckdb_forms(sql):
    res = tools.nfl_query(sql)
    assert res["code"] == "SQL_SYNTAX_ERROR", res
    assert "Parser Error" in res["error"]
    hint = res["hint"]
    assert "DESCRIBE plays" in hint
    assert "information_schema.columns" in hint and "ILIKE" in hint
    assert "nfl_schema" in hint


def test_parser_error_over_mcp_is_not_tool_error(tmp_path, monkeypatch):
    path = tmp_path / "empty.duckdb"
    duckdb.connect(str(path)).close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    is_error, payload = _raw("nfl_query", {"sql": "SHOW COLUMNS FROM plays LIKE '%conv%'"})
    assert is_error
    assert payload["code"] == "SQL_SYNTAX_ERROR", payload
    assert "DESCRIBE plays" in payload["details"]["hint"]


def test_desc_hint_is_kept():
    res = tools.nfl_query("SELECT desc FROM plays")
    assert '"desc"' in res["hint"], res
    assert res["code"] == "SQL_SYNTAX_ERROR"


def test_parser_error_after_explain_has_the_code():
    res = tools.nfl_query("EXPLAIN SELEC 1")
    assert res["code"] == "SQL_SYNTAX_ERROR", res


def test_query_description_gives_the_duckdb_show_form():
    sql_prop = TOOLS_BY_NAME["nfl_query"].inputSchema["properties"]["sql"]
    text = _desc("nfl_query") + " " + sql_prop["description"]
    assert "DESCRIBE <table>" in text and "SHOW <table>" in text, text
    assert "SHOW COLUMNS" not in text


# ═══ 6. whole numbers ══════════════════════════════════════════════════════════

_INT_PLAY_COLUMNS = ("play_id", "season", "week", "qtr", "down", "ydstogo", "yards_gained",
                     "touchdown", "interception")


@pytest.fixture
def double_db(tmp_path, monkeypatch):
    """Raw tables whose integer columns are DOUBLE, as some production loads are."""
    path = tmp_path / "double.duckdb"
    con = duckdb.connect(str(path))
    play_cols = {c: "DOUBLE" for c in _INT_PLAY_COLUMNS + ("epa", "wpa")}
    play_cols.update({c: "VARCHAR" for c in (
        "game_id", "season_type", "posteam", "defteam", "play_type", "desc",
        "passer_player_name", "passer_player_id", "rusher_player_name", "rusher_player_id",
        "receiver_player_name", "receiver_player_id")})
    base = {"game_id": "2024_01_KC_BAL", "season": 2024.0, "week": 1.0, "season_type": "REG",
            "posteam": "KC", "defteam": "BAL", "qtr": 1.0, "touchdown": 0.0, "interception": 0.0}
    _table(con, "plays", play_cols, [
        {**base, "play_id": 237.0, "down": 3.0, "ydstogo": 7.0, "play_type": "pass",
         "yards_gained": 12.0, "epa": 0.53, "wpa": 0.012},
        {**base, "play_id": 1.0, "play_type": "kickoff", "yards_gained": 0.0, "epa": -0.1,
         "wpa": 0.0},
    ])
    injuries = {c: "VARCHAR" for c in REQUIRED_COLUMNS["injuries"]}
    injuries.update(gsis_id="VARCHAR", season="DOUBLE", week="DOUBLE")
    _table(con, "injuries", injuries, [{"full_name": "A B", "gsis_id": "00-1", "team": "KC",
                                        "season": 2024.0, "week": 10.0, "game_type": "REG"}])
    sched = {c: "VARCHAR" for c in REQUIRED_COLUMNS["schedules"]}
    sched.update(season="DOUBLE", week="DOUBLE")
    _table(con, "schedules", sched, [{"game_id": "2024_01_KC_BAL", "season": 2024.0, "week": 1.0,
                                      "game_type": "REG", "home_team": "KC", "away_team": "BAL"}])
    snaps = {c: "VARCHAR" for c in REQUIRED_COLUMNS["snap_counts"]}
    snaps.update(season="DOUBLE", week="DOUBLE", offense_snaps="DOUBLE", offense_pct="DOUBLE",
                 defense_snaps="DOUBLE", defense_pct="DOUBLE", st_snaps="DOUBLE", st_pct="DOUBLE")
    _table(con, "snap_counts", snaps, [{"player": "A B", "team": "KC", "season": 2024.0,
                                        "week": 1.0, "game_type": "REG", "offense_snaps": 30.0,
                                        "offense_pct": 0.5}])
    rost = {c: "VARCHAR" for c in REQUIRED_COLUMNS["rosters"]}
    rost.update(season="DOUBLE", gsis_id="VARCHAR")
    _table(con, "rosters", rost, [{"full_name": "A B", "team": "KC", "season": 2024.0,
                                   "position": "QB"}])
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


def test_search_plays_returns_integer_columns_as_ints(double_db):
    res = tools.nfl_search_plays(season=2024, season_type="REG", sort_by="game_order")
    first, second = res["rows"]
    for col in _INT_PLAY_COLUMNS:
        assert type(second[col]) is int, (col, second[col])
    assert (second["play_id"], second["qtr"], second["down"], second["touchdown"]) == (237, 1, 3, 0)
    assert first["down"] is None and first["ydstogo"] is None
    assert second["epa"] == 0.53 and isinstance(second["epa"], float)
    assert second["wpa"] == 0.012 and isinstance(second["wpa"], float)


@pytest.mark.parametrize("call, key, cols", [
    (lambda: tools.nfl_injuries(season=2024, season_type="REG"), "injuries", ("season", "week")),
    (lambda: tools.nfl_schedule(season=2024, season_type="REG"), "games", ("season", "week")),
    (lambda: tools.nfl_snap_counts(season=2024, season_type="REG"), "snap_counts", ("season", "week")),
    (lambda: tools.nfl_roster(season=2024), "players", ("season",)),
])
def test_season_and_week_are_ints_from_double_tables(double_db, call, key, cols):
    res = call()
    (row,) = res[key]
    for col in cols:
        assert type(row[col]) is int, (col, row[col])


@pytest.mark.integration
def test_search_plays_ints_on_the_real_database(real):
    # SELECT count(*) FROM plays WHERE play_id <> round(play_id) OR qtr <> round(qtr)
    #   OR down <> round(down) OR ydstogo <> round(ydstogo)
    #   OR yards_gained <> round(yards_gained)                                     -> 0
    assert real.execute(
        "SELECT count(*) FROM plays WHERE play_id <> round(play_id) OR qtr <> round(qtr) "
        "OR down <> round(down) OR ydstogo <> round(ydstogo) "
        "OR yards_gained <> round(yards_gained)").fetchone()[0] == 0
    is_error, payload = _raw("nfl_search_plays", {"season": 2024, "season_type": "REG", "limit": 50})
    assert not is_error, payload
    for row in payload["rows"]:
        for col in _INT_PLAY_COLUMNS:
            assert row[col] is None or type(row[col]) is int, (col, row[col])


# ═══ 7. definitions of the blocks in the result ════════════════════════════════

_PASSING_ONLY = {"attempts", "completions", "sacks", "dropbacks", "epa_per_attempt", "epa_per_dropback"}
_RUSHING_ONLY = {"carries", "scrambles", "yards_per_carry"}
_RECEIVING_ONLY = {"targets", "receptions"}
_SHARED = {"yards", "touchdowns", "games"}


@pytest.mark.parametrize("stat_type, own, others, label", [
    ("passing", _PASSING_ONLY, _RUSHING_ONLY | _RECEIVING_ONLY, "passing:"),
    ("rushing", _RUSHING_ONLY, _PASSING_ONLY | _RECEIVING_ONLY, "rushing:"),
    ("receiving", _RECEIVING_ONLY, _PASSING_ONLY | _RUSHING_ONLY, "receiving:"),
])
def test_stat_type_sends_only_its_definitions(monkeypatch, one_player, stat_type, own, others, label):
    monkeypatch.setattr(tools, "_execute", _fake_execute([], []))
    defs = tools.nfl_player_stats(player="Player One", season_type="REG",
                                  stat_type=stat_type)["definitions"]
    assert own | _SHARED <= set(defs)
    assert not others & set(defs), sorted(others & set(defs))
    # The shared yards and touchdowns rules keep only this block's part.
    for key in ("yards", "touchdowns"):
        assert label in defs[key]
        for other in {"passing:", "rushing:", "receiving:"} - {label}:
            assert other not in defs[key], defs[key]


def test_no_stat_type_sends_every_definition(monkeypatch, one_player):
    monkeypatch.setattr(tools, "_execute", _fake_execute([], []))
    defs = tools.nfl_player_stats(player="Player One", season_type="REG")["definitions"]
    assert _PASSING_ONLY | _RUSHING_ONLY | _RECEIVING_ONLY | _SHARED <= set(defs)
    for label in ("passing:", "rushing:", "receiving:"):
        assert label in defs["yards"] and label in defs["touchdowns"]


# ═══ 8. descriptions ═══════════════════════════════════════════════════════════

def test_team_stats_points_to_team_records_for_points():
    text = _desc("nfl_team_stats")
    assert "nfl_team_records" in text and "points" in text.lower()


def test_team_stats_points_to_schedule_for_playoff_results():
    text = _desc("nfl_team_stats")
    assert "nfl_schedule" in text and "playoff" in text.lower()


# ═══ 9. version and README ═════════════════════════════════════════════════════

def test_version_is_0_7_1():
    assert __version__ == "0.7.1"


def _readme_row(tool):
    return next(line for line in README.read_text().splitlines() if line.startswith(f"| `{tool}`"))


def test_readme_team_records_row_lists_points():
    row = _readme_row("nfl_team_records")
    assert "points_for" in row and "points_against" in row and "point_diff" in row


def test_readme_player_stats_row_lists_games():
    assert "games" in _readme_row("nfl_player_stats")
