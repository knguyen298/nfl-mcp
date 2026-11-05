"""Group J: discovery (nfl_catalog, nfl_schema, resources), freshness (data_as_of,
nfl_status), MCP tool metadata, descriptions, README and dependency pins.

Interface contract assumed by these tests (beyond the shared brief):
- nfl_catalog() -> {"tables": [entry, ...], ...}. ("datasets" is accepted as the list key
  too.) entry = {"table", "grain": non-empty str, "row_count": live COUNT(*) of the table
  (not the _ingest_metadata sum), "min_season", "max_season" (from `season` or
  `season_year`, None when the table has neither), "join_keys": [str], "derived": bool}.
  Every table in schema 'main' is listed except _ingest_metadata.
- nfl_schema() gains "relationships": [{"left_table", "left_keys", "right_table",
  "right_keys"}] (orientation free; keys are positionally paired), "data_notes": [str],
  and "derived_tables": [str] (the derived marker). Every relationship names columns that
  exist and join on the loaded data (>= 80% of non-null left rows find a match).
- nfl_schema(table=X) keeps "columns" and adds "grain" (non-empty str) and "join_keys".
  For participation the join_keys include nflverse_game_id and play_id.
- data_as_of: ISO 8601 string on every tool response, including {"error": ...}
  responses, whenever _ingest_metadata is readable. Value = MAX(loaded_at) over the
  _ingest_metadata rows of the tables the tool reads (nfl_schedule: schedules only;
  nfl_status, nfl_catalog, nfl_schema: all rows). loaded_at is UTC; a naive value or one
  with Z / +00:00 is accepted. Cached per database file: suggested helper
  nfl_mcp.tools._data_as_of(db_path: str) -> dict[str, str] (table_name -> ISO max
  loaded_at), keyed by the resolved path so a symlink swap or a new NFL_MCP_DB_PATH gives a
  fresh value. The cache test does not patch that name: it spies on duckdb.connect and
  requires that a second tool call on the same file runs no SQL touching _ingest_metadata.
- nfl_status()["datasets"]["loaded"] entries keep dataset_id, table_name, total_rows and
  gain "newest_season", "newest_week" (MAX(week) within the newest season; None when the
  table has no week column) and "last_refreshed" (ISO). plays.total_plays and
  plays.seasons stay.
- MCP: every Tool has annotations readOnlyHint=True, destructiveHint=False,
  idempotentHint=True, openWorldHint=False and an outputSchema with type "object".
  tools/call returns structuredContent equal to json.loads(content[0].text). Resources
  nfl://schema, nfl://relationships, nfl://data-notes are listed by resources/list;
  resources/read on them and on nfl://schema/plays returns non-empty text.
- Descriptions: each list tool's description contains the literal phrase "one row per"
  and names its id field (table ID_FIELDS below). nfl_query mentions "CTE" or "WITH" and
  "read-only", not "LAST RESORT". nfl_search_plays (description + inputSchema) says 1999,
  never 2013.
- nflreadpy version: after an ingest, _ingest_metadata holds the installed nflreadpy
  version either in a column named nflreadpy_version or in a row with
  dataset_id = 'nflreadpy_version'.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import tomllib
from pathlib import Path

import anyio
import duckdb
import pytest
from mcp import types

import nfl_mcp.server as server
from nfl_mcp import tools
from nfl_mcp.schema_pbp import _SCHEMA_CATEGORIES, _SCHEMA_SUMMARY

REPO = Path(__file__).resolve().parents[2]
REAL_DB = Path(os.environ.get("NFL_MCP_DB_PATH", "/tmp/grpE_v3.duckdb"))
TOOL_NAMES = [t.name for t in server.TOOLS]
TOOLS_BY_NAME = {t.name: t for t in server.TOOLS}

LIST_TOOLS = [
    "nfl_roster", "nfl_injuries", "nfl_schedule", "nfl_snap_counts", "nfl_fantasy_opportunity",
    "nfl_fantasy_rankings", "nfl_search_plays", "nfl_td_luck", "nfl_role_trend",
    "nfl_separation_opportunity", "nfl_drop_rate", "nfl_contract_value", "nfl_injury_return",
    "nfl_team_records",
]
# id field each list tool's description must name (taken from the source table columns).
ID_FIELDS = {
    "nfl_roster": "gsis_id", "nfl_injuries": "gsis_id", "nfl_schedule": "game_id",
    "nfl_snap_counts": "pfr_player_id", "nfl_fantasy_opportunity": "player_id",
    "nfl_fantasy_rankings": "fantasypros_id", "nfl_search_plays": "play_id",
    "nfl_td_luck": "player_id", "nfl_role_trend": "player_id",
    "nfl_separation_opportunity": "player_id", "nfl_drop_rate": "player_id",
    "nfl_contract_value": "player_id", "nfl_injury_return": "injury_type",
    "nfl_team_records": "team",
}
MIN_ARGS = {
    "nfl_query": {"sql": "SELECT 1 AS a"}, "nfl_search_plays": {"team": "KC", "season": 2024},
    "nfl_team_stats": {"team": "KC", "season": 2024},
    "nfl_player_stats": {"player_name": "P.Mahomes", "season": 2024},
    "nfl_compare": {"entity1": "KC", "entity2": "BUF", "season": 2024},
    "nfl_roster": {"team": "KC", "season": 2024}, "nfl_injuries": {"team": "KC", "season": 2024},
    "nfl_schedule": {"team": "KC", "season": 2024}, "nfl_snap_counts": {"team": "KC", "season": 2024},
    "nfl_fantasy_opportunity": {"team": "KC", "season": 2024},
    "nfl_ftn_charting": {"team": "KC", "season": 2024}, "nfl_td_luck": {"season": 2024},
    "nfl_role_trend": {"season": 2024}, "nfl_separation_opportunity": {"season": 2024},
    "nfl_drop_rate": {"season": 2024}, "nfl_contract_value": {"season": 2024},
    "nfl_team_records": {"team": "KC", "season": 2024}, "nfl_player_lookup": {"name": "Mahomes"},
}
DERIVED = ["team_offense_stats", "team_defense_stats", "situational_stats", "formation_effectiveness"]


def _ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


# Hand-set loaded_at stamps. Global max is team_offense_stats (derived, 12:30);
# schedules' max is 08:00:05, strictly older, so nfl_schedule must not report the global max.
STAMPS = {
    ("pbp", "plays", 2024): _ts("2026-10-01T06:00:00"),
    ("pbp", "plays", 2025): _ts("2026-10-05T12:00:00"),
    ("schedules", "schedules", 2024): _ts("2026-10-03T08:00:00"),
    ("schedules", "schedules", 2025): _ts("2026-10-03T08:00:05"),
    ("players", "players", None): _ts("2026-09-30T07:00:00"),
    ("rosters", "rosters", 2025): _ts("2026-10-02T06:00:00"),
    ("snap_counts", "snap_counts", 2025): _ts("2026-10-02T07:00:00"),
    ("participation", "participation", 2025): _ts("2026-10-02T07:30:00"),
    ("team_offense_stats", "team_offense_stats", None): _ts("2026-10-05T12:30:00"),
}
GLOBAL_MAX = _ts("2026-10-05T12:30:00")
SCHEDULES_MAX = _ts("2026-10-03T08:00:05")
PBP_MAX = _ts("2026-10-05T12:00:00")

# plays: (game_id, play_id, season, week, posteam, defteam, play_type, yards, epa)
PLAYS = [
    ("2024_01_KC_BAL", 1.0, 2024, 1, "KC", "BAL", "pass", 5, 0.2),
    ("2024_01_KC_BAL", 2.0, 2024, 1, "KC", "BAL", "run", 3, -0.1),
    ("2024_02_KC_CIN", 1.0, 2024, 2, "KC", "CIN", "pass", 12, 0.9),
    ("2024_18_KC_DEN", 1.0, 2024, 18, "DEN", "KC", "run", 1, -0.4),
    ("2025_01_KC_LAC", 1.0, 2025, 1, "KC", "LAC", "pass", 7, 0.3),
    ("2025_01_KC_LAC", 2.0, 2025, 1, "LAC", "KC", "pass", 0, -0.8),
    ("2025_02_PHI_KC", 1.0, 2025, 2, "KC", "PHI", "run", 4, 0.1),
]  # 7 plays; seasons 2024-2025; newest season 2025, newest week in it = 2 (2024 has week 18)


def _build_db(path: Path, shift: dt.timedelta = dt.timedelta(0)) -> Path:
    con = duckdb.connect(str(path))
    con.execute("""CREATE TABLE plays (game_id VARCHAR, play_id DOUBLE, season INTEGER, week INTEGER,
        season_type VARCHAR, posteam VARCHAR, defteam VARCHAR, play_type VARCHAR, yards_gained DOUBLE,
        epa DOUBLE, pass_attempt DOUBLE, rush_attempt DOUBLE, sack DOUBLE, qb_dropback DOUBLE,
        qb_scramble DOUBLE, two_point_attempt DOUBLE, touchdown DOUBLE, interception DOUBLE,
        fumble_lost DOUBLE, passer_player_name VARCHAR, rusher_player_name VARCHAR,
        receiver_player_name VARCHAR, passer_player_id VARCHAR, rusher_player_id VARCHAR,
        receiver_player_id VARCHAR, "desc" VARCHAR, game_date VARCHAR, down DOUBLE, ydstogo DOUBLE,
        yardline_100 DOUBLE, qtr DOUBLE, game_type VARCHAR)""")
    for g, pid, s, w, pos, de, pt, y, e in PLAYS:
        p = 1.0 if pt == "pass" else 0.0
        con.execute(
            "INSERT INTO plays VALUES (?,?,?,?,'REG',?,?,?,?,?,?,?,0,?,0,0,0,0,0,?,?,?,?,?,?,'d','2024-09-08',1,10,50,1,'REG')",
            [g, pid, s, w, pos, de, pt, y, e, p, 1 - p, p,
             "P.Mahomes" if p else None, None if p else "I.Pacheco", "T.Kelce" if p else None,
             "00-0033873" if p else None, None if p else "00-0038000", "00-0030506" if p else None])
    con.execute("""CREATE TABLE schedules (game_id VARCHAR, season INTEGER, game_type VARCHAR, week INTEGER,
        gameday VARCHAR, gametime VARCHAR, away_team VARCHAR, home_team VARCHAR, away_score INTEGER,
        home_score INTEGER, result INTEGER, overtime INTEGER, spread_line DOUBLE, total_line DOUBLE,
        div_game INTEGER, roof VARCHAR, surface VARCHAR, temp INTEGER, wind INTEGER, away_coach VARCHAR,
        home_coach VARCHAR, referee VARCHAR, stadium VARCHAR)""")
    for gid, s, w, a, h, as_, hs in [
        ("2024_01_KC_BAL", 2024, 1, "BAL", "KC", 20, 27), ("2024_02_KC_CIN", 2024, 2, "CIN", "KC", 25, 26),
        ("2024_18_KC_DEN", 2024, 18, "KC", "DEN", 0, 38), ("2025_01_KC_LAC", 2025, 1, "KC", "LAC", 21, 27),
        ("2025_02_PHI_KC", 2025, 2, "KC", "PHI", 17, 20), ("2025_03_KC_NYG", 2025, 3, "KC", "NYG", None, None),
    ]:  # 6 games: 3 in 2024, 3 in 2025 (one unplayed)
        con.execute("INSERT INTO schedules VALUES (?,?,'REG',?,'2025-09-07','20:20',?,?,?,?,NULL,0,-3,47,0,"
                    "'outdoors','grass',70,5,'A','B','R','S')", [gid, s, w, a, h, as_, hs])
    con.execute("CREATE TABLE players (gsis_id VARCHAR, display_name VARCHAR, pfr_id VARCHAR, position VARCHAR)")
    con.execute("INSERT INTO players VALUES ('00-0033873','Patrick Mahomes','MahoPa00','QB'),"
                "('00-0030506','Travis Kelce','KelcTr00','TE'),('00-0038000','Isiah Pacheco','PachIs00','RB')")
    con.execute("CREATE TABLE rosters (season INTEGER, team VARCHAR, position VARCHAR, depth_chart_position VARCHAR,"
                " full_name VARCHAR, gsis_id VARCHAR, jersey_number INTEGER, status VARCHAR, years_exp INTEGER,"
                " college VARCHAR, height VARCHAR, weight INTEGER)")
    con.execute("INSERT INTO rosters VALUES (2025,'KC','QB','QB','Patrick Mahomes','00-0033873',15,'ACT',8,'TT','6-2',225),"
                "(2025,'KC','TE','TE','Travis Kelce','00-0030506',87,'ACT',12,'CIN','6-5',250),"
                "(2025,'KC','RB','RB','Isiah Pacheco','00-0038000',10,'ACT',3,'RU','5-10',216),"
                "(2025,'KC','WR','WR','No Id','00-0099999',99,'ACT',1,'X','6-0',200)")  # 4 rows
    con.execute("CREATE TABLE snap_counts (game_id VARCHAR, season INTEGER, week INTEGER, player VARCHAR,"
                " pfr_player_id VARCHAR, team VARCHAR, offense_snaps DOUBLE)")
    con.execute("INSERT INTO snap_counts VALUES ('2025_01_KC_LAC',2025,1,'Travis Kelce','KelcTr00','KC',60),"
                "('2025_01_KC_LAC',2025,1,'Isiah Pacheco','PachIs00','KC',40)")
    con.execute("CREATE TABLE participation (nflverse_game_id VARCHAR, play_id DOUBLE, possession_team VARCHAR)")
    con.execute("INSERT INTO participation VALUES ('2025_01_KC_LAC',1,'KC'),('2025_01_KC_LAC',2,'LAC')")
    con.execute("CREATE TABLE team_offense_stats (team VARCHAR, season_year INTEGER, total_plays BIGINT, avg_epa DOUBLE)")
    con.execute("INSERT INTO team_offense_stats VALUES ('KC',2024,3,0.33),('KC',2025,2,0.2)")  # 2 rows, 2024-2025
    con.execute("CREATE TABLE team_defense_stats (team VARCHAR, season_year INTEGER, sacks BIGINT)")
    con.execute("INSERT INTO team_defense_stats VALUES ('KC',2025,0)")
    con.execute("CREATE TABLE situational_stats (team VARCHAR, season_year INTEGER, situation VARCHAR, plays BIGINT)")
    con.execute("INSERT INTO situational_stats VALUES ('KC',2025,'red_zone',1)")
    con.execute("CREATE TABLE formation_effectiveness (team VARCHAR, season_year INTEGER, formation VARCHAR, plays BIGINT)")
    con.execute("INSERT INTO formation_effectiveness VALUES ('KC',2025,'SHOTGUN',2)")
    con.execute("CREATE TABLE espn_team_records (season INTEGER, team VARCHAR, wins INTEGER)")  # 0 rows
    con.execute("CREATE TABLE _ingest_metadata (dataset_id VARCHAR NOT NULL, table_name VARCHAR NOT NULL,"
                " season INTEGER, row_count BIGINT, loaded_at TIMESTAMP NOT NULL, loader_fn VARCHAR)")
    counts = {"plays": {2024: 4, 2025: 3}, "schedules": {2024: 3, 2025: 3}, "players": {None: 3},
              "rosters": {2025: 10},  # deliberately stale: table holds 4 rows; catalog must count live
              "snap_counts": {2025: 2}, "participation": {2025: 2}, "team_offense_stats": {None: 2}}
    for (ds, tbl, season), ts in STAMPS.items():
        loader = "derived" if tbl == "team_offense_stats" else f"load_{ds}"
        con.execute("INSERT INTO _ingest_metadata VALUES (?,?,?,?,?,?)",
                    [ds, tbl, season, counts[tbl][season], ts + shift, loader])
    con.close()
    return path


@pytest.fixture
def synth(tmp_path, monkeypatch):
    path = _build_db(tmp_path / "synth.duckdb")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


@pytest.fixture
def real_db(monkeypatch):
    if not REAL_DB.exists():
        pytest.skip(f"real DB {REAL_DB} missing")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(REAL_DB))
    con = duckdb.connect(str(REAL_DB), read_only=True)
    yield con
    con.close()


def _as_utc_naive(value) -> dt.datetime:
    assert isinstance(value, str), f"data_as_of must be a str, got {value!r}"
    d = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if d.tzinfo is not None:
        d = d.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return d


def _catalog(resp) -> dict:
    entries = resp.get("tables", resp.get("datasets"))
    assert isinstance(entries, list), resp
    return {e["table"]: e for e in entries if isinstance(e, dict) and "table" in e}


def _count(path, sql):
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(sql).fetchone()
    finally:
        con.close()


XF = pytest.mark.xfail


# ── 1. nfl_catalog ────────────────────────────────────────────────────────────

@XF(strict=True, reason="group j: catalog lists every table")
def test_catalog_lists_every_table_except_metadata(synth):
    cat = _catalog(tools.nfl_catalog())
    expected = {"plays", "schedules", "players", "rosters", "snap_counts", "participation",
                "team_offense_stats", "team_defense_stats", "situational_stats",
                "formation_effectiveness", "espn_team_records"}  # 11 tables built above
    assert set(cat) == expected
    assert "_ingest_metadata" not in cat


@XF(strict=True, reason="group j: catalog entry shape")
def test_catalog_entries_have_contract_keys(synth):
    cat = _catalog(tools.nfl_catalog())
    assert cat, "no catalog entries with a 'table' key"
    keys = {"table", "grain", "row_count", "min_season", "max_season", "join_keys", "derived"}
    for name, e in cat.items():
        assert keys <= set(e), (name, set(e))
        assert isinstance(e["grain"], str) and e["grain"].strip(), name
        assert isinstance(e["join_keys"], list) and isinstance(e["derived"], bool), name


@XF(strict=True, reason="group j: catalog live row_count")
@pytest.mark.parametrize("table,hand", [("plays", 7), ("rosters", 4), ("team_offense_stats", 2),
                                        ("espn_team_records", 0)])
def test_catalog_row_count_matches_count_star(synth, table, hand):
    assert _count(synth, f"SELECT COUNT(*) FROM {table}")[0] == hand
    assert _catalog(tools.nfl_catalog())[table]["row_count"] == hand  # rosters metadata says 10


@XF(strict=True, reason="group j: catalog derived flag")
def test_catalog_derived_flag(synth):
    cat = _catalog(tools.nfl_catalog())
    for t in DERIVED:
        assert cat[t]["derived"] is True, t
    for t in ("plays", "schedules", "players", "espn_team_records"):
        assert cat[t]["derived"] is False, t


@XF(strict=True, reason="group j: catalog season range")
@pytest.mark.parametrize("table,col", [("plays", "season"), ("team_offense_stats", "season_year"),
                                       ("schedules", "season")])
def test_catalog_season_range_matches_min_max(synth, table, col):
    lo, hi = _count(synth, f"SELECT MIN({col}), MAX({col}) FROM {table}")  # 2024, 2025 for all three
    e = _catalog(tools.nfl_catalog())[table]
    assert (e["min_season"], e["max_season"]) == (lo, hi) == (2024, 2025)


@XF(strict=True, reason="group j: catalog join keys")
@pytest.mark.parametrize("table,keys", [("plays", {"game_id", "play_id"}), ("players", {"gsis_id"}),
                                        ("snap_counts", {"pfr_player_id"})])
def test_catalog_join_keys(synth, table, keys):
    assert keys <= set(_catalog(tools.nfl_catalog())[table]["join_keys"])


@pytest.mark.integration
@XF(strict=True, reason="group j: catalog on real DB")
def test_catalog_real_db_tables_counts_and_flags(real_db):
    names = {r[0] for r in real_db.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()}
    cat = _catalog(tools.nfl_catalog())
    assert set(cat) == names - {"_ingest_metadata"}
    for t in ("plays", "schedules", "team_offense_stats"):
        assert cat[t]["row_count"] == real_db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0], t
    lo, hi = real_db.execute("SELECT MIN(season), MAX(season) FROM plays").fetchone()
    assert (cat["plays"]["min_season"], cat["plays"]["max_season"]) == (lo, hi)
    for t in DERIVED + ["team_records", "player_td_luck"]:
        assert cat[t]["derived"] is True, t
    for t in ("plays", "players", "schedules", "espn_team_records"):
        assert cat[t]["derived"] is False, t


# ── 2. nfl_schema relationships, data notes, derived marker ───────────────────

# Verified on /tmp/grpE_v3.duckdb: participation->plays 91103/91103 rows match;
# snap_counts->players.pfr_id 63187/63301; rosters->players.gsis_id 8868/9416.
REQUIRED_RELS = [
    ("participation", ["nflverse_game_id", "play_id"], "plays", ["game_id", "play_id"]),
    ("snap_counts", ["pfr_player_id"], "players", ["pfr_id"]),
    ("rosters", ["gsis_id"], "players", ["gsis_id"]),
]


def _rel_pairs(r):
    return set(zip(r["left_keys"], r["right_keys"]))


@XF(strict=True, reason="group j: schema relationships")
@pytest.mark.parametrize("lt,lk,rt,rk", REQUIRED_RELS, ids=[r[0] for r in REQUIRED_RELS])
def test_schema_relationships_include_verified_join(synth, lt, lk, rt, rk):
    rels = tools.nfl_schema().get("relationships")
    assert isinstance(rels, list), "nfl_schema() has no relationships list"
    want = set(zip(lk, rk))
    found = any((r["left_table"], r["right_table"]) == (lt, rt) and _rel_pairs(r) == want for r in rels) or any(
        (r["left_table"], r["right_table"]) == (rt, lt) and {(b, a) for a, b in _rel_pairs(r)} == want for r in rels)
    assert found, rels


@pytest.mark.integration
@XF(strict=True, reason="group j: schema relationships join on real data")
def test_schema_relationships_really_join_on_real_db(real_db):
    rels = tools.nfl_schema().get("relationships")
    assert isinstance(rels, list) and len(rels) >= 3
    names = {r[0] for r in real_db.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    for r in rels:
        lt, rt, lk, rk = r["left_table"], r["right_table"], list(r["left_keys"]), list(r["right_keys"])
        assert lt in names and rt in names, r
        on = " AND ".join(f"l.{a} = r.{b}" for a, b in zip(lk, rk))
        nn = " AND ".join(f"l.{a} IS NOT NULL" for a in lk)
        sub = ", ".join(rk)
        n, m = real_db.execute(
            f"SELECT COUNT(*), COUNT(r.{rk[0]}) FROM {lt} l LEFT JOIN (SELECT DISTINCT {sub} FROM {rt}) r "
            f"ON {on} WHERE {nn}").fetchone()
        assert n > 0 and m / n >= 0.8, (r, n, m)


@XF(strict=True, reason="group j: data notes")
@pytest.mark.parametrize("needles", [("ftn", "manual chart"), ("next gen stats", "2016"), ("ftn", "2022")],
                         ids=["ftn_manual", "ngs_2016", "ftn_2022"])
def test_schema_data_notes_mention(synth, needles):
    notes = tools.nfl_schema().get("data_notes")
    assert isinstance(notes, list) and notes, "no data_notes"
    assert any(all(n in note.lower() for n in needles) for note in notes), notes


@XF(strict=True, reason="group j: derived marker in nfl_schema")
def test_schema_marks_derived_tables(synth):
    derived = tools.nfl_schema().get("derived_tables")
    assert isinstance(derived, list)
    assert set(DERIVED) <= set(derived) and "plays" not in derived and "players" not in derived


@XF(strict=True, reason="group j: per-table grain")
@pytest.mark.parametrize("table,col", [("team_offense_stats", "season_year"), ("participation", "nflverse_game_id"),
                                       ("rosters", "gsis_id")])
def test_schema_table_has_columns_and_grain(synth, table, col):
    r = tools.nfl_schema(table=table)
    assert col in [c["column_name"] for c in r["columns"]]
    assert isinstance(r.get("grain"), str) and r["grain"].strip(), r


@XF(strict=True, reason="group j: participation join keys")
def test_schema_participation_join_keys(synth):
    r = tools.nfl_schema(table="participation")
    assert {"nflverse_game_id", "play_id"} <= set(r.get("join_keys") or [])


def test_schema_table_lookup_still_returns_columns(synth):
    cols = [c["column_name"] for c in tools.nfl_schema(table="team_offense_stats")["columns"]]
    assert cols == ["team", "season_year", "total_plays", "avg_epa"]  # as created above


# ── 3. data_as_of ─────────────────────────────────────────────────────────────

@XF(strict=True, reason="group j: data_as_of on every response")
@pytest.mark.parametrize("name", TOOL_NAMES)
def test_data_as_of_present_on_synthetic(synth, name):
    resp = getattr(tools, name)(**MIN_ARGS.get(name, {}))
    assert isinstance(resp, dict)
    d = _as_utc_naive(resp.get("data_as_of"))
    assert min(STAMPS.values()) <= d <= GLOBAL_MAX


@pytest.mark.integration
@XF(strict=True, reason="group j: data_as_of on every response (real DB)")
@pytest.mark.parametrize("name", TOOL_NAMES)
def test_data_as_of_present_on_real_db(real_db, name):
    lo, hi = real_db.execute("SELECT MIN(loaded_at), MAX(loaded_at) FROM _ingest_metadata").fetchone()
    resp = getattr(tools, name)(**MIN_ARGS.get(name, {}))
    assert lo <= _as_utc_naive(resp.get("data_as_of")) <= hi


@XF(strict=True, reason="group j: data_as_of value")
@pytest.mark.parametrize("name,args,expected", [
    ("nfl_schedule", {"season": 2025}, SCHEDULES_MAX),  # schedules only: 08:00:05, not global 12:30
    ("nfl_schedule", {"team": "KC"}, SCHEDULES_MAX),
    ("nfl_status", {}, GLOBAL_MAX),  # reads everything
])
def test_data_as_of_equals_max_loaded_at_of_tables_read(synth, name, args, expected):
    assert _as_utc_naive(getattr(tools, name)(**args).get("data_as_of")) == expected


@pytest.mark.integration
@XF(strict=True, reason="group j: data_as_of value (real DB)")
def test_data_as_of_schedule_real_db(real_db):
    want = real_db.execute("SELECT MAX(loaded_at) FROM _ingest_metadata WHERE table_name='schedules'").fetchone()[0]
    assert _as_utc_naive(tools.nfl_schedule(team="KC", season=2024).get("data_as_of")) == want


class _SpyConn:
    def __init__(self, conn, log):
        self._c, self._log = conn, log

    def execute(self, sql, *a, **k):
        self._log.append(str(sql))
        return self._c.execute(sql, *a, **k)

    def sql(self, sql, *a, **k):
        self._log.append(str(sql))
        return self._c.sql(sql, *a, **k)

    def __getattr__(self, item):
        return getattr(self._c, item)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._c.close()


@XF(strict=True, reason="group j: data_as_of cached per file")
@pytest.mark.parametrize("first,second", [("nfl_schedule", "nfl_schedule"), ("nfl_schedule", "nfl_roster")])
def test_data_as_of_cached_second_call_skips_metadata(synth, monkeypatch, first, second):
    log: list[str] = []
    real_connect = duckdb.connect
    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: _SpyConn(real_connect(*a, **k), log))
    r1 = getattr(tools, first)(**MIN_ARGS.get(first, {}))
    assert "data_as_of" in r1
    assert any("_ingest_metadata" in s for s in log), "first call never read _ingest_metadata"
    log.clear()
    r2 = getattr(tools, second)(**MIN_ARGS.get(second, {}))
    assert "data_as_of" in r2
    assert not [s for s in log if "_ingest_metadata" in s], log


@XF(strict=True, reason="group j: _data_as_of helper")
def test_data_as_of_helper_returns_per_table_map(synth):
    fn = getattr(tools, "_data_as_of", None)
    assert fn is not None, "nfl_mcp.tools._data_as_of missing"
    m = fn(str(synth))
    assert _as_utc_naive(m["schedules"]) == SCHEDULES_MAX and _as_utc_naive(m["plays"]) == PBP_MAX


@XF(strict=True, reason="group j: data_as_of follows NFL_MCP_DB_PATH")
def test_data_as_of_changes_with_db_path(synth, tmp_path, monkeypatch):
    other = _build_db(tmp_path / "other.duckdb", shift=dt.timedelta(days=1))
    assert _as_utc_naive(tools.nfl_status().get("data_as_of")) == GLOBAL_MAX
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(other))
    assert _as_utc_naive(tools.nfl_status().get("data_as_of")) == GLOBAL_MAX + dt.timedelta(days=1)


@XF(strict=True, reason="group j: data_as_of follows symlink swap")
def test_data_as_of_changes_after_symlink_swap(tmp_path, monkeypatch):
    a = _build_db(tmp_path / "a.duckdb")
    b = _build_db(tmp_path / "b.duckdb", shift=dt.timedelta(hours=2))
    link = tmp_path / "live.duckdb"
    link.symlink_to(a)
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(link))
    assert _as_utc_naive(tools.nfl_schedule().get("data_as_of")) == SCHEDULES_MAX
    link.unlink()
    link.symlink_to(b)  # the container's A/B swap repoints the same path
    assert _as_utc_naive(tools.nfl_schedule().get("data_as_of")) == SCHEDULES_MAX + dt.timedelta(hours=2)


# ── 4. nfl_status per dataset ─────────────────────────────────────────────────

def _status_entry(resp, dataset_id):
    for e in resp["datasets"]["loaded"]:
        if e.get("dataset_id") == dataset_id:
            return e
    raise AssertionError(f"{dataset_id} missing from nfl_status")


@XF(strict=True, reason="group j: nfl_status newest season/week")
@pytest.mark.parametrize("dataset_id,season,week", [("pbp", 2025, 2), ("schedules", 2025, 3)])
def test_status_newest_season_and_week_synthetic(synth, dataset_id, season, week):
    # pbp: 2025 has weeks 1-2 (2024 has week 18, which must not win); schedules 2025 weeks 1-3.
    e = _status_entry(tools.nfl_status(), dataset_id)
    assert (e.get("newest_season"), e.get("newest_week")) == (season, week)


@XF(strict=True, reason="group j: nfl_status per-dataset refresh time")
@pytest.mark.parametrize("dataset_id,expected", [("pbp", PBP_MAX), ("schedules", SCHEDULES_MAX),
                                                 ("players", STAMPS[("players", "players", None)])])
def test_status_last_refreshed_per_dataset(synth, dataset_id, expected):
    assert _as_utc_naive(_status_entry(tools.nfl_status(), dataset_id).get("last_refreshed")) == expected


def test_status_keeps_total_plays_and_seasons(synth):
    p = tools.nfl_status()["plays"]
    assert p["total_plays"] == 7  # 7 rows in PLAYS
    assert [(s["season"], s["plays"]) for s in p["seasons"]] == [(2024, 4), (2025, 3)]


@pytest.mark.integration
@XF(strict=True, reason="group j: nfl_status newest season/week (real DB)")
def test_status_newest_pbp_week_real_db(real_db):
    s = real_db.execute("SELECT MAX(season) FROM plays").fetchone()[0]
    w = real_db.execute("SELECT MAX(week) FROM plays WHERE season = ?", [s]).fetchone()[0]
    e = _status_entry(tools.nfl_status(), "pbp")
    assert (e.get("newest_season"), e.get("newest_week")) == (s, w)


# ── 5. MCP metadata ───────────────────────────────────────────────────────────

def _handle(req):
    async def go():
        return await server._mcp_server.request_handlers[type(req)](req)
    return anyio.run(go).root


@XF(strict=True, reason="group j: tool annotations")
@pytest.mark.parametrize("hint,value", [("readOnlyHint", True), ("destructiveHint", False),
                                        ("idempotentHint", True), ("openWorldHint", False)])
def test_every_tool_has_annotation(hint, value):
    tools_listed = _handle(types.ListToolsRequest(method="tools/list")).tools
    bad = [t.name for t in tools_listed if t.annotations is None or getattr(t.annotations, hint) is not value]
    assert not bad, bad


@XF(strict=True, reason="group j: outputSchema")
def test_every_tool_has_object_output_schema():
    tools_listed = _handle(types.ListToolsRequest(method="tools/list")).tools
    bad = [t.name for t in tools_listed if not t.outputSchema or t.outputSchema.get("type") != "object"]
    assert not bad, bad


@XF(strict=True, reason="group j: structuredContent")
@pytest.mark.parametrize("name,args", [("nfl_status", {}), ("nfl_schedule", {"season": 2025})])
def test_call_tool_structured_content_matches_text(synth, name, args):
    res = _handle(types.CallToolRequest(method="tools/call",
                                        params=types.CallToolRequestParams(name=name, arguments=args)))
    assert not res.isError, res.content
    assert res.structuredContent is not None
    assert res.structuredContent == json.loads(res.content[0].text)


def _read_text(uri):
    handler = server._mcp_server.request_handlers.get(types.ReadResourceRequest)
    assert handler is not None, "no resources/read handler"
    res = _handle(types.ReadResourceRequest(method="resources/read",
                                            params=types.ReadResourceRequestParams(uri=uri)))
    return "".join(getattr(c, "text", "") for c in res.contents)


@XF(strict=True, reason="group j: resources listed")
def test_list_resources_contains_discovery_uris():
    assert types.ListResourcesRequest in server._mcp_server.request_handlers, "no resources/list handler"
    uris = {str(r.uri) for r in _handle(types.ListResourcesRequest(method="resources/list")).resources}
    assert {"nfl://schema", "nfl://relationships", "nfl://data-notes"} <= uris


@XF(strict=True, reason="group j: resources readable")
@pytest.mark.parametrize("uri", ["nfl://schema", "nfl://relationships", "nfl://data-notes"])
def test_read_resource_returns_text(synth, uri):
    assert _read_text(uri).strip()


@XF(strict=True, reason="group j: nfl://schema/plays")
def test_read_resource_plays_columns(synth):
    text = _read_text("nfl://schema/plays")
    for col in ("game_id", "play_id", "posteam", "epa", "season"):
        assert re.search(rf"\b{col}\b", text), col


# ── 6. Descriptions ───────────────────────────────────────────────────────────

def _desc(name):
    return TOOLS_BY_NAME[name].description or ""


@XF(strict=True, reason="group j: nfl_query description")
def test_query_description_drops_last_resort():
    assert "LAST RESORT" not in _desc("nfl_query").upper()


def test_query_description_mentions_cte():  # satisfied by group B's description rewrite
    assert re.search(r"\bCTE\b|\bWITH\b", _desc("nfl_query"))


def test_query_description_mentions_read_only():
    assert "read-only" in _desc("nfl_query").lower()


@XF(strict=True, reason="group j: 1999 floor in nfl_search_plays")
def test_search_plays_text_says_1999_not_2013():
    t = TOOLS_BY_NAME["nfl_search_plays"]
    text = t.description + json.dumps(t.inputSchema)
    assert "1999" in text and "2013" not in text


@XF(strict=True, reason="group j: grain in list tool descriptions")
@pytest.mark.parametrize("name", LIST_TOOLS)
def test_list_tool_description_states_grain(name):
    assert "one row per" in _desc(name).lower()


@XF(strict=True, reason="group j: id field in list tool descriptions")
@pytest.mark.parametrize("name", [n for n in LIST_TOOLS if n not in ("nfl_team_records",)])
def test_list_tool_description_names_id_field(name):
    assert re.search(rf"\b{ID_FIELDS[name]}\b", _desc(name)), ID_FIELDS[name]


def test_no_description_says_tracking_derived():
    assert not [t.name for t in server.TOOLS if "tracking-derived" in (t.description or "").lower()]


# ── 7. README ─────────────────────────────────────────────────────────────────

README = (REPO / "README.md").read_text()
PARAS = re.split(r"\n\s*\n", README)


@XF(strict=True, reason="group j: README FTN wording")
def test_readme_does_not_call_ftn_tracking_derived():
    assert not [p for p in PARAS if "FTN" in p and "tracking-derived" in p.lower()]


@XF(strict=True, reason="group j: README FTN wording")
def test_readme_says_ftn_is_manual_charting():
    assert any("FTN" in p and "manual" in p.lower() for p in PARAS)


ENV_VARS = ["NFL_MCP_DB_PATH", "NFL_MCP_DUCKDB_MEMORY_LIMIT", "NFL_MCP_DUCKDB_THREADS",
            "NFL_MCP_DUCKDB_TEMP_DIR", "NFL_MCP_AUTH_TOKEN", "NFL_MCP_CORS_ORIGINS",
            "NFL_MCP_MAX_CONCURRENT_QUERIES", "NFL_MCP_QUERY_TIMEOUT_SECONDS"]


@pytest.mark.parametrize("var", [
    v if v == "NFL_MCP_DB_PATH" else pytest.param(v, marks=XF(strict=True, reason="group j: README env var"))
    for v in ENV_VARS])
def test_readme_documents_env_var(var):
    assert re.search(rf"\b{var}\b", README)


@XF(strict=True, reason="group j: README documents every env var the code reads")
def test_readme_documents_every_env_var_read_in_code():
    read = set()
    for f in (REPO / "nfl_mcp").glob("*.py"):
        src = f.read_text()
        read |= set(re.findall(r"os\.(?:getenv|environ\.get)\(\s*[\"'](NFL_MCP_[A-Z0-9_]+)", src))
        read |= set(re.findall(r"os\.environ\[\s*[\"'](NFL_MCP_[A-Z0-9_]+)", src))
        read |= set(re.findall(r"[\"'](NFL_MCP_[A-Z0-9_]+)[\"']", src))  # helper-style reads
    assert "NFL_MCP_DB_PATH" in read
    assert sorted(v for v in read if not re.search(rf"\b{v}\b", README)) == []


# ── 8. pyproject pins and nflreadpy version in metadata ───────────────────────

DEPS = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]


def _spec(pkg):
    for d in DEPS:
        if re.match(rf"{pkg}\b(?![-_])", d):
            return d
    raise AssertionError(f"{pkg} not in dependencies")


@pytest.mark.parametrize("pkg", [
    "mcp",
    pytest.param("duckdb", marks=XF(strict=True, reason="group j: duckdb upper bound")),
    pytest.param("nflreadpy", marks=XF(strict=True, reason="group j: nflreadpy upper bound")),
])
def test_dependency_has_upper_bound(pkg):
    assert re.search(r"<\s*\d", _spec(pkg)), _spec(pkg)


def _fake_ingest(tmp_path, monkeypatch, dataset_id, loader_name, df, **kw):
    import nflreadpy
    from nfl_mcp import ingest
    monkeypatch.delenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", raising=False)
    monkeypatch.delenv("NFL_MCP_DUCKDB_THREADS", raising=False)
    monkeypatch.setattr(nflreadpy, loader_name, lambda *a, **k: df)
    path = tmp_path / "ingested.duckdb"  # tmp file only; loader is faked, no network
    ingest.run_ingest_datasets([dataset_id], skip_views=True, db_path=str(path), **kw)
    return path


def _version_recorded(path) -> bool:
    import importlib.metadata
    ver = importlib.metadata.version("nflreadpy")
    con = duckdb.connect(str(path), read_only=True)
    try:
        cols = [r[0] for r in con.execute("DESCRIBE _ingest_metadata").fetchall()]
        if "nflreadpy_version" in cols:
            vals = {r[0] for r in con.execute("SELECT nflreadpy_version FROM _ingest_metadata").fetchall()}
            return vals == {ver}
        rows = con.execute("SELECT * FROM _ingest_metadata WHERE dataset_id = 'nflreadpy_version'").fetchall()
        return any(ver in [str(v) for v in r] for r in rows)
    finally:
        con.close()


@XF(strict=True, reason="group j: nflreadpy version in metadata")
def test_ingest_records_nflreadpy_version_static(tmp_path, monkeypatch):
    import polars as pl
    df = pl.DataFrame({"team_abbr": ["KC", "BUF"], "team_name": ["Kansas City Chiefs", "Buffalo Bills"]})
    assert _version_recorded(_fake_ingest(tmp_path, monkeypatch, "teams", "load_teams", df))


@XF(strict=True, reason="group j: nflreadpy version in metadata")
def test_ingest_records_nflreadpy_version_seasonal(tmp_path, monkeypatch):
    import polars as pl
    df = pl.DataFrame({"game_id": ["2024_01_KC_BAL"], "season": [2024], "week": [1],
                       "home_team": ["KC"], "away_team": ["BAL"]})
    assert _version_recorded(_fake_ingest(tmp_path, monkeypatch, "schedules", "load_schedules", df,
                                          start=2024, end=2024))


def test_fake_ingest_writes_metadata_row(tmp_path, monkeypatch):
    import polars as pl
    df = pl.DataFrame({"team_abbr": ["KC"], "team_name": ["Kansas City Chiefs"]})
    path = _fake_ingest(tmp_path, monkeypatch, "teams", "load_teams", df)
    assert _count(path, "SELECT COUNT(*) FROM _ingest_metadata WHERE dataset_id='teams'")[0] == 1


# ── 9. schema_pbp.py consistency ──────────────────────────────────────────────

def test_schema_summary_plays_floor_is_1999():
    assert re.search(r"Table: plays \(1999 through", _SCHEMA_SUMMARY)


@XF(strict=True, reason="group j: game_context season floor says 2013")
def test_schema_game_context_season_floor_is_1999():
    line = next(l for l in _SCHEMA_CATEGORIES["game_context"].splitlines() if l.strip().startswith("season "))
    assert "1999" in line and "2013" not in line, line


def test_registry_pbp_floor_matches_summary():
    from nfl_mcp.registry import REGISTRY
    assert REGISTRY["pbp"].min_season == 1999


@pytest.mark.integration
@XF(strict=True, reason="group j: plays column count text")
def test_schema_summary_column_count_matches_describe(real_db):
    n = len(real_db.execute("DESCRIBE plays").fetchall())  # 373 on grpE_v3
    m = re.search(r"Table: plays \([^)]*?(\d+) columns\)", _SCHEMA_SUMMARY)
    assert m and int(m.group(1)) == n, (m and m.group(1), n)
