"""v0.7.0: description fixes from the audit of b1200db (SPEC_v0.7.0_unified_arguments.md,
section "Description fixes from the audit").

Behavior first, wording second:
- Where the fix is behavior (nfl_roster also matches depth_chart_position, a hint for an
  unquoted "desc", did_you_mean ranked against the queried tables, a practice_status filter
  on nfl_injuries), the test calls the tool on a real database. Expected values come from raw
  SQL, quoted next to each value, measured on the CI-style 2024-only database
  (/tmp/ci2024/duck2.duckdb, built by `nfl-mcp ingest --dataset all --start 2024 --end 2024`).
  These tests skip when the database is missing (the unit CI job has none).
- Where the fix is wording, the test checks that the false claim is gone and that a few
  required facts appear, with loose matching. No test pins a sentence.

Interface assumed beyond the spec:
- v0.7.0 argument names: nfl_player_stats takes `player`; nfl_injuries and nfl_search_plays
  require season_type (spec decision 5), so the calls here pass season_type='REG'.
- A parser error near an unquoted desc returns a `hint` field that contains '"desc"'. (The
  DuckDB error text already quotes the token, so the hint has to be a separate field.)
- did_you_mean for a column of one table (or a join) lists only columns of the tables in the
  FROM clause.
- nfl_injuries gains `practice_status`; the full stored value matches (whether the match is
  exact or partial).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from nfl_mcp import server, tools
from nfl_mcp.database import get_db_connection
from nfl_mcp.schema_pbp import _SCHEMA_CATEGORIES, _SCHEMA_SUMMARY

REAL_DB = Path(os.environ.get("NFL_MCP_DB_PATH", "/tmp/ci2024/duck2.duckdb"))
TOOLS_BY_NAME = {t.name: t for t in server.TOOLS}

# Spec decision 5: tools with season-type data and their valid season_type values.
SEASON_TOTAL_TOOLS = ["nfl_team_stats", "nfl_player_stats", "nfl_compare", "nfl_td_luck",
                      "nfl_drop_rate", "nfl_separation_opportunity", "nfl_contract_value"]
PER_GAME_TOOLS = ["nfl_schedule", "nfl_snap_counts", "nfl_search_plays", "nfl_injuries",
                  "nfl_ftn_charting", "nfl_fantasy_opportunity", "nfl_role_trend"]
SEASON_TOTAL_VALUES = ["REG", "POST", "ALL"]
PER_GAME_VALUES = ["REG", "POST", "ALL", "WC", "DIV", "CON", "SB"]

# Tools that filter by a player name today (nfl_player_stats as player_name, nfl_player_lookup
# as name; v0.7.0 renames both to player). Spec decision 6: one name rule in every tool.
NAME_TOOLS = ["nfl_search_plays", "nfl_player_stats", "nfl_player_lookup", "nfl_injuries",
              "nfl_snap_counts", "nfl_fantasy_opportunity", "nfl_fantasy_rankings",
              "nfl_ftn_charting", "nfl_td_luck", "nfl_role_trend", "nfl_separation_opportunity",
              "nfl_drop_rate", "nfl_contract_value"]
# Tools that return data rows (everything except the discovery tools and raw SQL).
DATA_TOOLS = [t.name for t in server.TOOLS
              if t.name not in ("nfl_schema", "nfl_status", "nfl_catalog", "nfl_query")]
POSITION_TOOLS = [t.name for t in server.TOOLS if "position" in t.inputSchema.get("properties", {})]

AGGREGATE_TABLES = ["team_offense_stats", "team_defense_stats", "situational_stats",
                    "formation_effectiveness"]


@pytest.fixture
def real_db(monkeypatch):
    if not REAL_DB.exists():
        pytest.skip(f"real DB {REAL_DB} missing")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(REAL_DB))
    with get_db_connection() as con:  # same opener as the tools, so the locked config matches
        yield con


def _desc(tool: str) -> str:
    return TOOLS_BY_NAME[tool].description or ""


def _props(tool: str) -> dict:
    return TOOLS_BY_NAME[tool].inputSchema.get("properties", {})


def _prop_desc(tool: str, *names: str) -> str:
    """Description of the first property in `names` that the tool has."""
    props = _props(tool)
    for name in names:
        if name in props:
            return props[name].get("description", "") or ""
    raise AssertionError(f"{tool} has none of {names}; properties: {sorted(props)}")


def _all_text(tool: str) -> str:
    """Tool description plus every property description."""
    parts = [_desc(tool)]
    parts += [p.get("description", "") or "" for p in _props(tool).values()]
    return "\n".join(parts)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _columns(con, table: str) -> set[str]:
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'main' AND table_name = ?", [table]).fetchall()
    return {r[0].lower() for r in rows}


def _word(token: str, text: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text) is not None


# ── nfl_injuries ─────────────────────────────────────────────────────────────

def test_injuries_report_status_description_drops_practice_values():
    text = _prop_desc("nfl_injuries", "report_status")
    assert "full participation" not in text.lower(), text


def test_injuries_report_status_description_names_real_values():
    # SELECT DISTINCT report_status FROM injuries -> Doubtful, Note, Out, Questionable, NULL
    text = _prop_desc("nfl_injuries", "report_status")
    for value in ("Out", "Questionable", "Doubtful", "Note"):
        assert _word(value, text), (value, text)


def test_injuries_report_status_description_states_match_rule():
    text = _prop_desc("nfl_injuries", "report_status").lower()
    assert any(w in text for w in ("partial", "exact", "substring", "contains")), text


def test_injuries_tool_description_report_status_list_has_no_practice_value():
    # The tool description lists report status values in parentheses; the list must not
    # carry practice_status values.
    lists = re.findall(r"report[ _]status(?:es)?\s*\(([^)]*)\)", _desc("nfl_injuries"), re.IGNORECASE)
    for listed in lists:
        assert "participation" not in listed.lower(), listed


def test_injuries_has_practice_status_filter():
    props = _props("nfl_injuries")
    assert "practice_status" in props, sorted(props)
    assert "participation" in (props["practice_status"].get("description") or "").lower()


def test_injuries_report_status_values_in_data_are_described(real_db):
    values = [r[0] for r in real_db.execute(
        "SELECT DISTINCT report_status FROM injuries WHERE report_status IS NOT NULL").fetchall()]
    assert values
    text = _prop_desc("nfl_injuries", "report_status")
    for value in values:
        assert _word(value, text), (value, text)


@pytest.mark.parametrize("status,expected", [
    # SELECT COUNT(*) FROM injuries WHERE team='KC' AND season=2024 AND game_type='REG'
    #   AND practice_status='Full Participation in Practice'  -> 104
    ("Full Participation in Practice", 104),
    # ... AND practice_status='Did Not Participate In Practice'  -> 19
    ("Did Not Participate In Practice", 19),
])
def test_injuries_practice_status_filter(real_db, status, expected):
    sql_count = real_db.execute(
        "SELECT COUNT(*) FROM injuries WHERE team='KC' AND season=2024 AND game_type='REG' "
        "AND practice_status=?", [status]).fetchone()[0]
    assert sql_count == expected
    r = tools.nfl_injuries(team="KC", season=2024, season_type="REG", practice_status=status,
                           limit=500)
    assert "error" not in r, r
    assert r["truncated"] is False
    assert len(r["injuries"]) == expected
    assert {row["practice_status"] for row in r["injuries"]} == {status}


# ── nfl_roster position ──────────────────────────────────────────────────────

def test_roster_position_description_is_exact_and_names_coarse_codes():
    text = _prop_desc("nfl_roster", "position")
    assert "partial match" not in text.lower(), text
    assert "exact" in text.lower(), text
    for code in ("DB", "DL", "OL", "LB"):
        assert _word(code, text), (code, text)


def test_roster_position_description_mentions_depth_chart_position():
    assert "depth_chart_position" in _all_text("nfl_roster")


@pytest.mark.parametrize("position,expected", [
    # SELECT COUNT(*) FROM rosters WHERE season=2024 AND team='KC'
    #   AND (position='CB' OR depth_chart_position='CB')  -> 10 (position='CB' alone: 0)
    ("CB", 10),
    # ... (position='T' OR depth_chart_position='T')  -> 9 (position='T' alone: 0)
    ("T", 9),
    # ... (position='DB' OR depth_chart_position='DB')  -> 18 (DB/FS 3, DB/CB 10, DB/SS 4, DB/DB 1)
    ("DB", 18),
    # ... (position='D' OR depth_chart_position='D')  -> 0: no partial match on 'D'
    ("D", 0),
])
def test_roster_position_matches_position_or_depth_chart_position(real_db, position, expected):
    sql_count = real_db.execute(
        "SELECT COUNT(*) FROM rosters WHERE season=2024 AND team='KC' "
        "AND (position=? OR depth_chart_position=?)", [position, position]).fetchone()[0]
    assert sql_count == expected
    r = tools.nfl_roster(team="KC", season=2024, position=position, limit=500)
    assert "error" not in r, r
    assert r["truncated"] is False
    assert len(r["players"]) == expected
    for row in r["players"]:
        assert position in (row["position"], row["depth_chart_position"]), row


# ── nfl_player_lookup season ─────────────────────────────────────────────────

def test_player_lookup_season_description_drops_2024_claim():
    # SELECT MIN(season) FROM rosters on the full database -> 1920.
    text = _prop_desc("nfl_player_lookup", "season")
    assert "2024 on" not in text, text


# ── nfl_query / nfl_schema: "desc" ───────────────────────────────────────────

def test_query_description_quotes_desc():
    assert '"desc"' in _all_text("nfl_query")


@pytest.mark.parametrize("category", ["play_details", "query_tips"])
def test_schema_plays_notes_quote_desc(category):
    text = tools.nfl_schema(category=category)["schema"]
    assert '"desc"' in text, category


def test_schema_never_shows_desc_unquoted():
    text = "\n".join([_SCHEMA_SUMMARY, *_SCHEMA_CATEGORIES.values()])
    # A bare lowercase desc token is the column shown unquoted; "ORDER BY x desc" is a sort.
    bare = [m.start() for m in re.finditer(r'(?<!["`\w])desc(?!["`\w])', text)
            if not re.search(r"(?i)\bby\s+\S+\s+$", text[max(0, m.start() - 40):m.start()])]
    assert not bare, [text[max(0, i - 30):i + 30] for i in bare]


@pytest.mark.parametrize("sql", [
    "SELECT desc FROM plays LIMIT 1",
    "SELECT play_id FROM plays WHERE desc ILIKE '%touchdown%' LIMIT 1",
])
def test_query_unquoted_desc_gets_hint(real_db, sql):
    r = tools.nfl_query(sql)
    assert "error" in r, r
    assert isinstance(r.get("hint"), str), r
    assert '"desc"' in r["hint"], r


def test_query_quoted_desc_and_desc_sort_work(real_db):
    assert tools.nfl_query('SELECT "desc" FROM plays LIMIT 1')["row_count"] == 1
    assert tools.nfl_query("SELECT epa FROM plays ORDER BY epa DESC LIMIT 1")["row_count"] == 1


# ── did_you_mean ranked against the queried tables ───────────────────────────

def _dym(sql: str) -> list[str]:
    r = tools.nfl_query(sql)
    assert r.get("code") == "COLUMN_NOT_FOUND", r
    dym = [s.lower() for s in r.get("did_you_mean", [])]
    assert dym, r
    return dym


@pytest.mark.parametrize("sql,tables,expected", [
    # plays has the column desc (information_schema.columns, table_name='plays').
    ("SELECT play_description FROM plays", ["plays"], "desc"),
    ("SELECT description FROM plays", ["plays"], "desc"),
    # players has display_name and no full_name.
    ("SELECT full_name FROM players", ["players"], "display_name"),
    ("SELECT play_description FROM plays JOIN schedules USING (game_id)",
     ["plays", "schedules"], "desc"),
])
def test_did_you_mean_includes_synonym_and_stays_in_queried_tables(real_db, sql, tables, expected):
    dym = _dym(sql)
    allowed = set().union(*(_columns(real_db, t) for t in tables))
    assert expected in dym, dym
    assert set(dym) <= allowed, sorted(set(dym) - allowed)


@pytest.mark.parametrize("table", ["schedules", "snap_counts"])
def test_did_you_mean_season_type_suggests_game_type_first(real_db, table):
    cols = _columns(real_db, table)
    assert "game_type" in cols and "season_type" not in cols  # schedules, snap_counts: game_type only
    dym = _dym(f"SELECT season_type FROM {table}")
    assert dym[0] == "game_type", dym
    assert set(dym) <= cols, sorted(set(dym) - cols)


def test_did_you_mean_game_status_stays_in_schedules(real_db):
    cols = _columns(real_db, "schedules")
    dym = _dym("SELECT game_status FROM schedules")
    assert set(dym) <= cols, sorted(set(dym) - cols)


# ── nfl_schema notes ─────────────────────────────────────────────────────────

def _segment(text: str, table: str) -> str:
    """Text from `table` up to the next aggregate/records table name or category list."""
    start = text.index(table)
    stops = [text.find(s, start + len(table)) for s in
             (*AGGREGATE_TABLES, "team_records", "Categories available")]
    stops = [s for s in stops if s > start]
    return text[start:min(stops) if stops else len(text)]


@pytest.mark.parametrize("table", AGGREGATE_TABLES)
def test_schema_aggregate_column_lists_include_season_type(table):
    # Each table has season_type: information_schema.columns WHERE table_name=<table>.
    for text in (_SCHEMA_CATEGORIES["aggregate_tables"], _SCHEMA_SUMMARY):
        assert "season_type" in _segment(text, table), (table, _segment(text, table))


def _schema_units() -> list[str]:
    info = tools.nfl_schema()
    units = [n for n in info.get("data_notes", [])]
    for text in (info.get("schema", ""), info.get("hint", ""),
                 tools.nfl_schema(category="query_tips")["schema"]):
        for para in re.split(r"\n\s*\n", text):
            units += [u for u in re.split(r"\n\s*•", para) if u.strip()]
    return units


@pytest.mark.parametrize("table,column", [
    # schedules and snap_counts: game_type, no season_type (information_schema.columns).
    ("schedules", "game_type"),
    ("snap_counts", "game_type"),
    # players: display_name, no full_name.
    ("players", "display_name"),
])
def test_schema_notes_name_the_column_per_table(real_db, table, column):
    units = _schema_units()
    assert any(_word(table, u) and column in u for u in units), (table, column)


# ── nfl_team_stats ───────────────────────────────────────────────────────────

def test_team_stats_does_not_claim_rankings():
    for sentence in _sentences(_desc("nfl_team_stats")):
        if "rank" in sentence.lower():
            assert "nfl_query" in sentence and "instead of nfl_query" not in sentence, sentence


def test_team_stats_states_grain():
    text = _desc("nfl_team_stats")
    assert re.search(r"\bone row per\b", text, re.IGNORECASE), text
    for token in ("team", "season_year", "season_type"):
        assert token in text, (token, text)


# ── nfl_fantasy_rankings ─────────────────────────────────────────────────────

def test_no_tool_mentions_best_ball():
    blob = json.dumps([t.model_dump(mode="json") for t in server.TOOLS]).lower()
    assert "best-ball" not in blob and "best ball" not in blob


def test_fantasy_rankings_ranking_set_drops_best_example():
    assert "'best'" not in _prop_desc("nfl_fantasy_rankings", "ranking_set")


def test_fantasy_rankings_ranking_set_examples_exist(real_db):
    # Every quoted example must match a stored ranking set:
    # SELECT page_type FROM ff_rankings_draft UNION ALL SELECT page FROM ff_rankings_week
    examples = re.findall(r"'([^']+)'", _prop_desc("nfl_fantasy_rankings", "ranking_set"))
    assert examples
    for ex in examples:
        n = real_db.execute(
            "SELECT COUNT(*) FROM (SELECT page_type AS s FROM ff_rankings_draft "
            "UNION ALL SELECT page FROM ff_rankings_week) WHERE s ILIKE ?", [f"%{ex}%"]).fetchone()[0]
        assert n > 0, ex


def test_fantasy_rankings_explains_fa_team():
    # SELECT COUNT(*) FROM ff_rankings_draft WHERE team='FA' -> 136 (free agents).
    text = _desc("nfl_fantasy_rankings") + "\n" + _prop_desc("nfl_fantasy_rankings", "team")
    assert _word("FA", text), text
    assert "free agent" in text.lower(), text


# ── nfl_injury_return ────────────────────────────────────────────────────────

def test_injury_return_has_no_diagnosis_mapping_example():
    # injury_type ILIKE '%ACL%' and '%hammy%' match no row: buckets are set at ingest.
    text = _prop_desc("nfl_injury_return", "injury_type")
    assert not re.search(r"'[^']+'\s*(->|→)", text), text
    assert "bucket" in text.lower(), text


def test_injury_return_lists_no_absent_thigh_bucket(real_db):
    # SELECT COUNT(*) FROM injury_return_curve WHERE injury_type='thigh' -> 0
    n = real_db.execute(
        "SELECT COUNT(*) FROM injury_return_curve WHERE injury_type='thigh'").fetchone()[0]
    assert n == 0
    assert not _word("thigh", _prop_desc("nfl_injury_return", "injury_type"))


def test_injury_return_lists_every_bucket_in_data(real_db):
    # SELECT DISTINCT injury_type FROM injury_return_curve
    buckets = [r[0] for r in real_db.execute(
        "SELECT DISTINCT injury_type FROM injury_return_curve").fetchall()]
    assert buckets
    text = _prop_desc("nfl_injury_return", "injury_type")
    for b in buckets:
        assert _word(b, text), b


def test_injury_return_position_states_match_rule():
    # Code: position = UPPER(?), an exact match.
    assert "exact" in _prop_desc("nfl_injury_return", "position").lower()


# ── nfl_team_records ─────────────────────────────────────────────────────────

def test_team_records_says_current_season_is_to_date():
    text = _desc("nfl_team_records").lower()
    assert re.search(r"to date|so far|in progress|partial", text), text


def test_team_records_states_required_filter():
    text = _all_text("nfl_team_records").lower()
    assert "range" in text
    assert re.search(r"\brequired\b|\brequires\b|at least one|\bone of\b|\bmust\b", text), text


# ── nfl_search_plays is_touchdown / is_turnover ──────────────────────────────

@pytest.mark.parametrize("flag", ["is_touchdown", "is_turnover"])
def test_search_plays_flag_description_says_what_false_does(flag):
    text = _prop_desc("nfl_search_plays", flag).lower()
    assert "false" in text, text


@pytest.mark.parametrize("flag,flagged", [
    # SELECT COUNT(*) FROM plays WHERE posteam='KC' AND season=2024 AND week=1
    #   AND season_type='REG'  -> 70; AND touchdown=1 -> 3
    ("is_touchdown", 3),
    # ... AND (interception=1 OR fumble_lost=1) -> 1
    ("is_turnover", 1),
])
def test_search_plays_false_flag_matches_description(real_db, flag, flagged):
    base = {"team": "KC", "season": 2024, "week": 1, "season_type": "REG", "limit": 500}
    every = tools.nfl_search_plays(**base)
    off = tools.nfl_search_plays(**base, **{flag: False})
    on = tools.nfl_search_plays(**base, **{flag: True})
    assert every["count"] == 70 and on["count"] == flagged, (every, on)
    # The description decides: false excludes the flagged plays, or false is no filter.
    text = _prop_desc("nfl_search_plays", flag).lower()
    if re.search(r"false[^.;]*(exclud|non-|without|remove|leave[s]? out|drop)", text):
        assert off["count"] == 70 - flagged
    else:
        assert re.search(r"no filter|not filter|ignored|same as omitting|every play|all plays", text), text
        assert off["count"] == 70


# ── nfl_player_stats season roster rule ──────────────────────────────────────

def test_player_stats_describes_the_roster_retry():
    text = (_desc("nfl_player_stats") + "\n"
            + _prop_desc("nfl_player_stats", "player", "player_name")).lower()
    assert "only players on a roster" not in text, text
    assert re.search(r"otherwise|fall(s)? back|if no |if none|retr|again|without the season", text), text


def test_player_stats_resolves_player_without_roster_row_in_season(real_db):
    # SELECT gsis_id FROM players WHERE display_name='Tom Brady' -> one row, '00-0019596'
    # SELECT COUNT(*) FROM rosters WHERE gsis_id='00-0019596' -> 0
    # SELECT COUNT(*) FROM plays WHERE season=2024 AND '00-0019596' IN
    #   (passer_player_id, rusher_player_id, receiver_player_id) -> 0
    assert real_db.execute(
        "SELECT COUNT(*) FROM rosters WHERE gsis_id='00-0019596' AND season=2024").fetchone()[0] == 0
    r = tools.nfl_player_stats(player="Tom Brady", season=2024, season_type="REG")
    assert "error" not in r and not r.get("ambiguous"), r
    assert r["player_id"] == "00-0019596"
    assert r["seasons"] == []


# ── nfl_query sql argument ───────────────────────────────────────────────────

def test_query_sql_argument_lists_every_allowed_statement():
    text = _prop_desc("nfl_query", "sql")
    for kw in ("SELECT", "WITH", "DESCRIBE", "SUMMARIZE", "SHOW", "EXPLAIN"):
        assert _word(kw, text), (kw, text)


@pytest.mark.parametrize("sql", [
    "SELECT 1 AS a", "WITH x AS (SELECT 1 AS a) SELECT * FROM x", "DESCRIBE schedules",
    "SUMMARIZE teams", "SHOW TABLES", "EXPLAIN SELECT 1",
])
def test_query_accepts_every_listed_statement(real_db, sql):
    r = tools.nfl_query(sql, max_rows=1)
    assert "error" not in r, r


# ── "Call this first" ────────────────────────────────────────────────────────

def test_only_one_tool_says_call_this_first():
    hits = [t.name for t in server.TOOLS if "call this first" in (t.description or "").lower()]
    assert len(hits) == 1, hits


# ── Every description: season_type values, match rule, grain ─────────────────

@pytest.mark.parametrize("tool,values",
                         [(t, SEASON_TOTAL_VALUES) for t in SEASON_TOTAL_TOOLS]
                         + [(t, PER_GAME_VALUES) for t in PER_GAME_TOOLS])
def test_description_states_season_type_values(tool, values):
    text = _desc(tool) + "\n" + _prop_desc(tool, "season_type")
    missing = [v for v in values if not _word(v, text)]
    assert not missing, (tool, missing)


@pytest.mark.parametrize("tool", NAME_TOOLS)
def test_description_states_name_match_rule(tool):
    text = _prop_desc(tool, "player", "player_name", "name").lower()
    assert "exact" in text, (tool, text)


@pytest.mark.parametrize("tool", POSITION_TOOLS)
def test_description_states_position_match_rule(tool):
    text = _prop_desc(tool, "position").lower()
    assert re.search(r"exact|partial|substring|prefix", text), (tool, text)


@pytest.mark.parametrize("tool", DATA_TOOLS)
def test_description_states_grain(tool):
    assert re.search(r"\bone (row|entry|result|candidate|record) per\b", _desc(tool),
                     re.IGNORECASE), tool
