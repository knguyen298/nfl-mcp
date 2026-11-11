"""v0.7.0 player arguments: player/player_id, nfl_compare pairs, season_from/season_to on
nfl_team_records, one name-matching rule, and ambiguity hints.

Spec: nfl-mcp-notes/SPEC_v0.7.0_unified_arguments.md, decisions 2, 3, 4 and 6.

Every expected value comes from SQL run on raw tables (plays, rosters, players, snap_counts,
injuries, schedules), never from a tool call. The SQL sits in a comment next to each value.
Values were measured on a 2024-only database (the same kind CI builds with
`nfl-mcp ingest --dataset all --start 2024 --end 2024`). The integration tests read the
database at NFL_MCP_DB_PATH (default /tmp/ci2024/v070.duckdb) and skip when it is missing or
has no 2024 plays. The fantasy ranking tables are a snapshot of the latest scrape, not season
data, so their expected counts are computed by SQL at run time rather than hard-coded.

Interface contract (assumptions where the spec is silent)
- Single-player tools: nfl_search_plays, nfl_player_stats, nfl_player_lookup, nfl_roster,
  nfl_injuries, nfl_snap_counts, nfl_fantasy_opportunity, nfl_fantasy_rankings,
  nfl_ftn_charting, nfl_td_luck, nfl_role_trend, nfl_separation_opportunity, nfl_drop_rate,
  nfl_contract_value. Each has string properties `player` and `player_id`.
- nfl_compare has player1, player2, team1, team2. None of them is in "required".
- No inputSchema has player_name, name, entity1, entity2, compare_type, from_season, to_season.
- An ambiguous name gives a non-error result (isError false) with top-level ambiguous=True,
  candidates (each with a gsis_id) and a hint string, as today. The hint names player_id in
  single-player tools and player1/player2 in nfl_compare; any argument name it mentions is in
  that tool's inputSchema. nfl_player_lookup is a search tool: several matches are its normal
  result (candidates), so it is not in the ambiguity test.
- A name that matches no player gives a non-error result with no rows, no candidates and no
  resolved player_id.
- Name rule: folded full name, last name or short name ('P.Mahomes'), compared exactly. A
  strict substring ('Mahom', 'Kelc', 'Johnso') matches nobody.
- nfl_fantasy_rankings rows carry a FantasyPros id and a name. A row belongs to the player when
  ff_playerids maps its id to his gsis_id; a row with no mapping counts when its folded name
  (generational suffix dropped) is one of his names and its position is in his group.
- Tools that need season_type under decision 5 get season_type="REG" here.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import anyio
import duckdb
import pytest
from mcp import types

from nfl_mcp import server, tools
from nfl_mcp.database import get_db_connection

REAL_DB = Path(os.environ.get("NFL_MCP_DB_PATH", "/tmp/ci2024/v070.duckdb"))

# Players used below (SELECT gsis_id, display_name, short_name, pfr_id FROM players WHERE ...)
MAHOMES = "00-0033873"   # Patrick Mahomes, P.Mahomes, MahoPa00; only 'Mahomes' in players
KELCE = "00-0030506"     # Travis Kelce, T.Kelce, KelcTr00 (Jason Kelce is J.Kelce)
BARKLEY = "00-0034844"   # Saquon Barkley, S.Barkley
HENRY = "00-0032764"     # Derrick Henry, D.Henry
# SELECT gsis_id FROM players WHERE display_name='Lamar Jackson' -> two players; both have
# snap_counts rows in 2024 (SELECT count(DISTINCT pfr_player_id) FROM snap_counts
# WHERE player='Lamar Jackson' -> 2) and 2024 roster rows.
LAMAR_JACKSONS = {"00-0034796", "00-0036152"}
# SELECT DISTINCT gsis_id FROM injuries WHERE full_name='Jaylon Jones' -> two players
JAYLON_JONESES = {"00-0037106", "00-0038407"}

SINGLE_PLAYER_TOOLS = [
    "nfl_search_plays", "nfl_player_stats", "nfl_player_lookup", "nfl_roster", "nfl_injuries",
    "nfl_snap_counts", "nfl_fantasy_opportunity", "nfl_fantasy_rankings", "nfl_ftn_charting",
    "nfl_td_luck", "nfl_role_trend", "nfl_separation_opportunity", "nfl_drop_rate",
    "nfl_contract_value",
]
# Tools that need season_type under decision 5 (the rest take none).
SEASON_TYPED = {
    "nfl_search_plays", "nfl_player_stats", "nfl_injuries", "nfl_snap_counts",
    "nfl_fantasy_opportunity", "nfl_ftn_charting", "nfl_td_luck", "nfl_role_trend",
    "nfl_separation_opportunity", "nfl_drop_rate", "nfl_contract_value", "nfl_compare",
}
# Tools that return an ambiguous result for a name matching several players.
AMBIGUITY_TOOLS = [t for t in SINGLE_PLAYER_TOOLS if t != "nfl_player_lookup"]
OLD_NAMES = ["player_name", "name", "entity1", "entity2", "compare_type", "from_season", "to_season"]
GSIS_LIKE = re.compile(r"\d{2}-\d{7}")

# SQL form of the folded name (lower case, accents, apostrophes and periods dropped, other
# separators one space, a trailing generational suffix dropped). Used only as the oracle.
def _fold_sql(col: str) -> str:
    return (
        "regexp_replace(trim(regexp_replace(regexp_replace(lower(strip_accents(" + col + ")), "
        "'[.''’`´]', '', 'g'), '[^a-z0-9]+', ' ', 'g')), ' (jr|sr|ii|iii|iv|v)$', '')"
    )


# ── fixtures and helpers ──────────────────────────────────────────────────────

@pytest.fixture
def real(monkeypatch):
    """Oracle connection through get_db_connection, so tool calls open the same config."""
    if not REAL_DB.exists():
        pytest.skip(f"{REAL_DB} missing")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(REAL_DB))
    with get_db_connection() as conn:
        if not conn.execute("SELECT count(*) FROM plays WHERE season = 2024").fetchone()[0]:
            pytest.skip("needs 2024 plays")
        yield conn


@pytest.fixture
def empty_db(tmp_path, monkeypatch):
    """An empty database: enough for call_tool's argument check, which runs before any query."""
    path = tmp_path / "empty.duckdb"
    duckdb.connect(str(path)).close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


def _one(conn, sql, params=()):
    return conn.execute(sql, list(params)).fetchone()


def _all(conn, sql, params=()):
    return conn.execute(sql, list(params)).fetchall()


def _handle(req):
    async def go():
        return await server._mcp_server.request_handlers[type(req)](req)
    return anyio.run(go).root


def _call(name: str, args: dict) -> tuple[bool, dict]:
    """(isError, payload) of one MCP tools/call. A plain-text error (the MCP library's own
    jsonschema check, e.g. a missing required property) becomes {"error": text}."""
    res = _handle(types.CallToolRequest(
        method="tools/call", params=types.CallToolRequestParams(name=name, arguments=args)))
    text = res.content[0].text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {"error": text}
    return bool(res.isError), payload


def _listed_tools() -> dict[str, types.Tool]:
    return {t.name: t for t in _handle(types.ListToolsRequest(method="tools/list")).tools}


def _props(name: str) -> dict:
    return _listed_tools()[name].inputSchema.get("properties", {})


def _args(name: str, **kwargs) -> dict:
    """kwargs plus season_type='REG' where decision 5 needs it."""
    if name in SEASON_TYPED:
        kwargs.setdefault("season_type", "REG")
    return kwargs


def _rows(payload: dict) -> list:
    """Every row a payload carries: items of top-level lists of dicts (candidates aside)."""
    out = []
    for key, value in payload.items():
        if key == "candidates":
            continue
        if isinstance(value, list):
            out += [v for v in value if isinstance(v, dict)]
    return out


def _arg_like_tokens() -> set[str]:
    """Argument names that cannot be mistaken for plain English words in a hint."""
    names = set(OLD_NAMES)
    for tool in _listed_tools().values():
        names |= set(tool.inputSchema.get("properties", {}))
    return {n for n in names if "_" in n or any(ch.isdigit() for ch in n)}


def _named(token: str, text: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text) is not None


def _all_pages(fn, key, **kwargs) -> list[dict]:
    rows, offset = [], 0
    while True:
        res = fn(limit=500, offset=offset, **kwargs)
        assert "error" not in res, res
        assert not res.get("ambiguous"), res
        rows += res[key]
        if not res["truncated"]:
            return rows
        offset += 500


# ── 1. schema ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", SINGLE_PLAYER_TOOLS)
def test_player_tool_has_player_and_player_id(name):
    props = _props(name)
    assert props.get("player", {}).get("type") == "string", (name, sorted(props))
    assert props.get("player_id", {}).get("type") == "string", (name, sorted(props))


@pytest.mark.parametrize("old", OLD_NAMES)
def test_no_input_schema_has_old_argument_name(old):
    bad = [t.name for t in _listed_tools().values()
           if old in t.inputSchema.get("properties", {})]
    assert not bad, f"{old} still in {bad}"


@pytest.mark.parametrize("name", ["nfl_player_stats", "nfl_player_lookup"])
def test_player_and_player_id_not_required(name):
    required = _listed_tools()[name].inputSchema.get("required", [])
    assert "player" not in required and "player_id" not in required, required


def test_compare_schema_has_both_pairs_and_requires_neither():
    schema = _listed_tools()["nfl_compare"].inputSchema
    props = schema.get("properties", {})
    for arg in ("player1", "player2", "team1", "team2"):
        assert props.get(arg, {}).get("type") == "string", (arg, sorted(props))
    required = set(schema.get("required", []))
    assert not required & {"player1", "player2", "team1", "team2"}, required


def test_team_records_schema_has_season_from_and_season_to():
    props = _props("nfl_team_records")
    assert props.get("season_from", {}).get("type") == "integer", sorted(props)
    assert props.get("season_to", {}).get("type") == "integer", sorted(props)


# ── 3. old names are rejected over MCP ────────────────────────────────────────

@pytest.mark.parametrize("name,args,old", [
    ("nfl_player_stats", {"player_name": "Saquon Barkley", "season_type": "REG"}, "player_name"),
    ("nfl_player_lookup", {"name": "Saquon Barkley"}, "name"),
    ("nfl_compare", {"entity1": "KC", "entity2": "PHI", "season_type": "REG"}, "entity1"),
    ("nfl_compare", {"team1": "KC", "entity2": "PHI", "season_type": "REG"}, "entity2"),
    ("nfl_compare", {"team1": "KC", "team2": "PHI", "compare_type": "team", "season_type": "REG"},
     "compare_type"),
    ("nfl_team_records", {"from_season": 2024}, "from_season"),
    ("nfl_team_records", {"team": "KC", "to_season": 2024}, "to_season"),
])
def test_old_argument_name_is_unknown_argument(empty_db, name, args, old):
    is_error, payload = _call(name, args)
    assert is_error, payload
    assert payload.get("code") == "INVALID_ARGUMENTS", payload
    assert old in payload.get("details", {}).get("unknown", []), payload
    assert f"Unknown argument '{old}'" in payload["error"] or repr(old) in payload["error"], payload


# ── 2. nfl_player_stats: player_id alone, neither argument ────────────────────

@pytest.mark.integration
def test_player_stats_player_id_alone(real):
    # Carries, rush yards (with lateral rushing yards) and rush TDs of Saquon Barkley, 2024 REG:
    # SELECT count(*) FILTER (WHERE rusher_player_id='00-0034844' AND rush_attempt=1
    #          AND COALESCE(two_point_attempt,0)=0),
    #        COALESCE(sum(rushing_yards) FILTER (WHERE <same>),0)
    #          + COALESCE(sum(lateral_rushing_yards) FILTER (WHERE lateral_rusher_player_id=
    #            '00-0034844' AND rush_attempt=1 AND COALESCE(two_point_attempt,0)=0),0),
    #        count(*) FILTER (WHERE <same> AND rush_touchdown=1)
    # FROM plays WHERE season=2024 AND season_type='REG'  -> (345, 2005, 13)
    assert _one(real, _RUSH_SQL, [BARKLEY] * 4) == (345, 2005, 13)
    is_error, res = _call("nfl_player_stats",
                          {"player_id": BARKLEY, "season": 2024, "season_type": "REG"})
    assert not is_error, res
    assert res["player_id"] == BARKLEY
    (row,) = res["seasons"]
    assert (row["season"], row["season_type"]) == (2024, "REG")
    rush = row["rushing"]
    assert (rush["carries"], rush["yards"], rush["touchdowns"]) == (345, 2005, 13)


@pytest.mark.integration
def test_player_stats_without_player_or_player_id_is_error_naming_both(real):
    is_error, payload = _call("nfl_player_stats", {"season": 2024, "season_type": "REG"})
    assert is_error, payload
    msg = payload["error"]
    assert _named("player", msg) and _named("player_id", msg), msg
    # v0.7.1: player_name appears only as a name that is not accepted (a proxy drops it).
    assert "player_name" not in msg.replace("such as player_name are not accepted", ""), msg


def test_player_stats_function_without_player_returns_error(empty_db):
    res = tools.nfl_player_stats(season_type="REG")
    assert "error" in res, res
    assert _named("player", res["error"]) and _named("player_id", res["error"]), res


# ── 4. nfl_compare pairs ──────────────────────────────────────────────────────

_RUSH_SQL = """
    SELECT count(*) FILTER (WHERE rusher_player_id = ? AND rush_attempt = 1
                                  AND COALESCE(two_point_attempt, 0) = 0),
           CAST(COALESCE(sum(rushing_yards) FILTER (WHERE rusher_player_id = ? AND rush_attempt = 1
                                  AND COALESCE(two_point_attempt, 0) = 0), 0)
              + COALESCE(sum(lateral_rushing_yards) FILTER (WHERE lateral_rusher_player_id = ?
                                  AND rush_attempt = 1 AND COALESCE(two_point_attempt, 0) = 0), 0)
                AS INTEGER),
           count(*) FILTER (WHERE rusher_player_id = ? AND rush_attempt = 1
                                  AND COALESCE(two_point_attempt, 0) = 0 AND rush_touchdown = 1)
    FROM plays WHERE season = 2024 AND season_type = 'REG'
"""


@pytest.mark.integration
@pytest.mark.parametrize("args", [
    {"player1": "Saquon Barkley"},
    {"player2": "Derrick Henry"},
    {"team1": "KC"},
    {"team2": "PHI"},
    {"player1": "Saquon Barkley", "team2": "PHI"},
    {"team1": "KC", "player2": "Derrick Henry"},
    {"player1": "Saquon Barkley", "player2": "Derrick Henry", "team1": "KC", "team2": "PHI"},
    {"player1": "Saquon Barkley", "player2": "Derrick Henry", "team1": "KC"},
    {},
], ids=["p1", "p2", "t1", "t2", "p1+t2", "t1+p2", "both", "both-partial", "none"])
def test_compare_needs_exactly_one_pair(real, args):
    is_error, payload = _call("nfl_compare", {**args, "season": 2024, "season_type": "REG"})
    assert is_error, payload
    assert payload.get("code") != "UNKNOWN_TEAM", payload
    msg = payload["error"]
    for arg in ("player1", "player2", "team1", "team2"):
        assert _named(arg, msg), (arg, msg)


def _entry(res: dict, gsis_id: str) -> dict:
    found = [p for p in res["players"] if p.get("player_id") == gsis_id]
    assert len(found) == 1, res
    return found[0]


@pytest.mark.integration
@pytest.mark.parametrize("p1,p2", [("Saquon Barkley", "Derrick Henry"), (BARKLEY, HENRY)],
                         ids=["names", "gsis_ids"])
def test_compare_players_match_raw_sql(real, p1, p2):
    # _RUSH_SQL for 00-0034844 (Saquon Barkley) -> (345, 2005, 13)
    assert _one(real, _RUSH_SQL, [BARKLEY] * 4) == (345, 2005, 13)
    # _RUSH_SQL for 00-0032764 (Derrick Henry) -> (325, 1921, 16)
    assert _one(real, _RUSH_SQL, [HENRY] * 4) == (325, 1921, 16)
    is_error, res = _call("nfl_compare", {"player1": p1, "player2": p2,
                                          "season": 2024, "season_type": "REG"})
    assert not is_error, res
    assert not res.get("ambiguous"), res
    for gid, expected in ((BARKLEY, (345, 2005, 13)), (HENRY, (325, 1921, 16))):
        (row,) = _entry(res, gid)["seasons"]
        assert (row["season"], row["season_type"]) == (2024, "REG")
        rush = row["rushing"]
        assert (rush["carries"], rush["yards"], rush["touchdowns"]) == expected


@pytest.mark.integration
@pytest.mark.parametrize("p1,p2", [("Saquon Barkley", "Derrick Henry"),
                                   ("Zzyzx Notaplayer", "Derrick Henry"),
                                   ("Barkle", "Henr")])
def test_compare_player_name_never_unknown_team(real, p1, p2):
    is_error, payload = _call("nfl_compare", {"player1": p1, "player2": p2,
                                              "season": 2024, "season_type": "REG"})
    assert "UNKNOWN_TEAM" not in json.dumps(payload), payload
    assert not is_error, payload


@pytest.mark.integration
def test_compare_substring_name_does_not_resolve(real):
    # 'Barkle' is a strict substring of Barkley: it matches nobody, so Saquon's id never appears.
    is_error, res = _call("nfl_compare", {"player1": "Barkle", "player2": "Derrick Henry",
                                          "season": 2024, "season_type": "REG"})
    assert not is_error, res
    assert not res.get("ambiguous"), res
    assert BARKLEY not in json.dumps(res), res
    _entry(res, HENRY)


@pytest.mark.integration
def test_compare_teams_pair_works(real):
    is_error, res = _call("nfl_compare", {"team1": "KC", "team2": "PHI",
                                          "season": 2024, "season_type": "REG"})
    assert not is_error, res
    text = json.dumps(res)
    assert '"KC"' in text and '"PHI"' in text, res


# ── 5. ambiguity: candidates with gsis_id, hints name real arguments ──────────

def _johnson_ids(conn) -> set[str]:
    # Every player whose folded display name, last name or short name (dots and spaces
    # removed) is 'johnson'. Candidates must come from this set.
    # SELECT gsis_id FROM players WHERE fold(display_name)='johnson' OR fold(last_name)='johnson'
    #   OR replace(fold(short_name),' ','')='johnson'   -> many (53 on 2024 rosters alone)
    rows = _all(conn, f"""
        SELECT gsis_id FROM players
        WHERE {_fold_sql('display_name')} = 'johnson' OR {_fold_sql('last_name')} = 'johnson'
           OR replace({_fold_sql('short_name')}, ' ', '') = 'johnson'""")
    return {r[0] for r in rows}


def _assert_hint_names_only_own_arguments(tool: str, hint: str):
    props = set(_props(tool))
    named = {tok for tok in _arg_like_tokens() if _named(tok, hint)}
    assert named <= props, f"{tool} hint names {sorted(named - props)}: {hint!r}"
    if tool == "nfl_compare":
        assert _named("player1", hint) or _named("player2", hint), hint
        assert not _named("player_id", hint), hint
    else:
        assert _named("player_id", hint), hint


def _ambiguity_args(tool: str, name: str) -> dict:
    if tool == "nfl_compare":
        return _args(tool, player1=name, player2="Saquon Barkley", season=2024)
    args = _args(tool, player=name)
    if "season" in _props(tool):
        args["season"] = 2024
    return args


@pytest.mark.integration
@pytest.mark.parametrize("tool", AMBIGUITY_TOOLS + ["nfl_compare"])
def test_ambiguous_last_name_returns_candidates_and_own_hint(real, tool):
    # SELECT count(DISTINCT gsis_id) FROM rosters WHERE season=2024 AND full_name LIKE '% Johnson'
    #   -> 53 on the 2024 database; counted at run time, since a re-ingest of 2024 rosters can
    #   change it. More than one is what the test needs (see _johnson_ids).
    (n,) = _one(real, "SELECT count(DISTINCT gsis_id) FROM rosters "
                      "WHERE season=2024 AND full_name LIKE '% Johnson'")
    assert n > 1, n
    oracle = _johnson_ids(real)
    is_error, res = _call(tool, _ambiguity_args(tool, "Johnson"))
    assert not is_error, res
    assert res.get("ambiguous") is True, res
    cands = res["candidates"]
    assert len(cands) >= 2, res
    ids = [c.get("gsis_id") for c in cands]
    assert all(isinstance(i, str) and i for i in ids), cands
    assert set(ids) <= oracle, set(ids) - oracle
    assert not _rows(res), "an ambiguous result must not carry rows"
    _assert_hint_names_only_own_arguments(tool, res["hint"])


@pytest.mark.integration
@pytest.mark.parametrize("tool,name,ids", [
    ("nfl_snap_counts", "Lamar Jackson", LAMAR_JACKSONS),
    ("nfl_roster", "Lamar Jackson", LAMAR_JACKSONS),
    ("nfl_injuries", "Jaylon Jones", JAYLON_JONESES),
])
def test_shared_full_name_is_ambiguous_not_merged(real, tool, name, ids):
    # snap_counts: SELECT DISTINCT pfr_player_id FROM snap_counts WHERE player='Lamar Jackson'
    #   -> JackLa00, JackLa02 (players.pfr_id of 00-0034796 and 00-0036152); 21 rows today merged
    # rosters: SELECT DISTINCT gsis_id FROM rosters WHERE full_name='Lamar Jackson' -> both ids
    # injuries: SELECT DISTINCT gsis_id FROM injuries WHERE full_name='Jaylon Jones'
    #   -> 00-0037106, 00-0038407
    table, col, idcol = {"nfl_snap_counts": ("snap_counts", "player", "pfr_player_id"),
                         "nfl_roster": ("rosters", "full_name", "gsis_id"),
                         "nfl_injuries": ("injuries", "full_name", "gsis_id")}[tool]
    assert _one(real, f"SELECT count(DISTINCT {idcol}) FROM {table} WHERE {col}=?", [name]) == (2,)
    is_error, res = _call(tool, _args(tool, player=name, season=2024))
    assert not is_error, res
    assert res.get("ambiguous") is True, res
    assert ids <= {c["gsis_id"] for c in res["candidates"]}, res["candidates"]
    assert not _rows(res), res
    _assert_hint_names_only_own_arguments(tool, res["hint"])


@pytest.mark.integration
def test_player_lookup_lists_every_match_with_gsis_id(real):
    oracle = _johnson_ids(real)
    is_error, res = _call("nfl_player_lookup", {"player": "Johnson", "season": 2024})
    assert not is_error, res
    ids = {c["gsis_id"] for c in res["candidates"]}
    assert len(ids) >= 2 and ids <= oracle, res
    if "hint" in res:
        props = set(_props("nfl_player_lookup"))
        named = {tok for tok in _arg_like_tokens() if _named(tok, res["hint"])}
        assert named <= props, res["hint"]


@pytest.mark.integration
def test_player_lookup_by_player_id(real):
    # SELECT display_name FROM players WHERE gsis_id='00-0033873' -> 'Patrick Mahomes'
    assert _one(real, "SELECT display_name FROM players WHERE gsis_id=?", [MAHOMES]) == (
        "Patrick Mahomes",)
    is_error, res = _call("nfl_player_lookup", {"player_id": MAHOMES})
    assert not is_error, res
    assert [c["gsis_id"] for c in res["candidates"]] == [MAHOMES]


# ── 6. one name-matching rule ─────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.parametrize("tool", AMBIGUITY_TOOLS)
def test_strict_substring_matches_nothing(real, tool):
    # 'Johnso' is a strict substring of Johnson. No player has it as a folded full, last or
    # short name: SELECT count(*) FROM players WHERE fold(display_name)='johnso'
    #   OR fold(last_name)='johnso' OR replace(fold(short_name),' ','')='johnso' -> 0
    assert _one(real, f"""
        SELECT count(*) FROM players
        WHERE {_fold_sql('display_name')} = 'johnso' OR {_fold_sql('last_name')} = 'johnso'
           OR replace({_fold_sql('short_name')}, ' ', '') = 'johnso'""") == (0,)
    is_error, res = _call(tool, _ambiguity_args(tool, "Johnso"))
    assert not is_error, res
    assert not res.get("ambiguous"), res
    assert not res.get("candidates"), res
    assert _rows(res) == [], res
    assert res.get("player_id") is None, res
    assert res.get("count", 0) == 0 and not res.get("total_plays"), res


@pytest.mark.integration
def test_player_lookup_substring_matches_nothing(real):
    is_error, res = _call("nfl_player_lookup", {"player": "Johnso"})
    assert not is_error, res
    assert res["candidates"] == [], res


# snap_counts: one row per game. SELECT count(*) FROM snap_counts
#   WHERE pfr_player_id='MahoPa00' AND game_type='REG'  -> 16
#   WHERE pfr_player_id='KelcTr00' AND game_type='REG'  -> 16
_SNAPS = {MAHOMES: ("MahoPa00", 16), KELCE: ("KelcTr00", 16)}


@pytest.mark.integration
@pytest.mark.parametrize("player,gid", [
    ("Patrick Mahomes", MAHOMES), ("patrick mahomes", MAHOMES), ("Mahomes", MAHOMES),
    ("P.Mahomes", MAHOMES), ("Travis Kelce", KELCE), ("T.Kelce", KELCE),
])
def test_snap_counts_exact_name_forms(real, player, gid):
    pfr, n = _SNAPS[gid]
    assert _one(real, "SELECT count(*) FROM snap_counts WHERE pfr_player_id=? AND game_type='REG'",
                [pfr]) == (n,)
    rows = _all_pages(tools.nfl_snap_counts, "snap_counts", player=player, season=2024,
                      season_type="REG")
    assert len(rows) == n
    assert {r["pfr_player_id"] for r in rows} == {pfr}


@pytest.mark.integration
@pytest.mark.parametrize("player", ["Mahom", "Kelc", "atrick Mahomes"])
def test_snap_counts_substring_matches_nothing(real, player):
    # SELECT count(*) FROM snap_counts WHERE player ILIKE '%Mahom%' -> 16+3 Mahomes rows today
    res = tools.nfl_snap_counts(player=player, season=2024, season_type="REG")
    assert "error" not in res, res
    assert not res.get("ambiguous"), res
    assert res["snap_counts"] == [], res


# injuries: one row per week. SELECT count(*) FROM injuries
#   WHERE gsis_id='00-0033873' AND game_type='REG' -> 6
#   WHERE gsis_id='00-0030506' AND game_type='REG' -> 1
_INJ = {MAHOMES: 6, KELCE: 1}


@pytest.mark.integration
@pytest.mark.parametrize("kwargs,gid", [
    ({"player": "Patrick Mahomes"}, MAHOMES), ({"player": "Mahomes"}, MAHOMES),
    ({"player": "P.Mahomes"}, MAHOMES), ({"player_id": MAHOMES}, MAHOMES),
    ({"player": "T.Kelce"}, KELCE), ({"player_id": KELCE}, KELCE),
])
def test_injuries_exact_name_forms_and_player_id(real, kwargs, gid):
    n = _INJ[gid]
    assert _one(real, "SELECT count(*) FROM injuries WHERE gsis_id=? AND game_type='REG'",
                [gid]) == (n,)
    rows = _all_pages(tools.nfl_injuries, "injuries", season=2024, season_type="REG", **kwargs)
    assert len(rows) == n
    assert {r["gsis_id"] for r in rows} == {gid}


@pytest.mark.integration
@pytest.mark.parametrize("player", ["Mahom", "Kelc"])
def test_injuries_substring_matches_nothing(real, player):
    res = tools.nfl_injuries(player=player, season=2024, season_type="REG")
    assert "error" not in res, res
    assert not res.get("ambiguous"), res
    assert res["injuries"] == [], res


def _rankings_needed(conn, scope: str) -> tuple[str, str]:
    table, col = ("ff_rankings_draft", "player") if scope == "draft" else (
        "ff_rankings_week", "player_name")
    try:
        n = _one(conn, f"SELECT count(*) FROM {table}")[0]
    except duckdb.Error:
        pytest.skip(f"{table} not loaded")
    if not n:
        pytest.skip(f"{table} is empty")
    return table, col


@pytest.mark.integration
@pytest.mark.parametrize("scope", ["draft", "week"])
@pytest.mark.parametrize("kwargs", [{"player": "Patrick Mahomes"}, {"player": "Mahomes"},
                                    {"player": "P.Mahomes"}, {"player_id": MAHOMES}],
                         ids=["full", "last", "short", "id"])
def test_fantasy_rankings_exact_name_forms(real, scope, kwargs):
    table, col = _rankings_needed(real, scope)
    # Snapshot table, so the count is measured now: rows whose folded name (suffix dropped,
    # so 'Patrick Mahomes II' counts) is 'patrick mahomes'. On the 2024 database: draft 7, week 1.
    (n,) = _one(real, f"SELECT count(*) FROM {table} WHERE {_fold_sql(col)} = 'patrick mahomes'")
    if not n:
        pytest.skip(f"no Patrick Mahomes rows in {table}")
    rows = _all_pages(tools.nfl_fantasy_rankings, "fantasy_rankings", scope=scope, **kwargs)
    assert len(rows) == n
    assert {re.sub(r" ii$", "", r["player"].lower().replace(".", "")) for r in rows} == {
        "patrick mahomes"}


@pytest.mark.integration
@pytest.mark.parametrize("scope", ["draft", "week"])
@pytest.mark.parametrize("player", ["Mahom", "Kelc"])
def test_fantasy_rankings_substring_matches_nothing(real, scope, player):
    table, col = _rankings_needed(real, scope)
    # SELECT count(*) FROM <table> WHERE <col> ILIKE '%<player>%' -> > 0 (rows the old
    # substring rule returned); the new rule returns none of them.
    (n,) = _one(real, f"SELECT count(*) FROM {table} WHERE {col} ILIKE ?", [f"%{player}%"])
    if not n:
        pytest.skip(f"no {player} rows in {table}; nothing to guard")
    res = tools.nfl_fantasy_rankings(player=player, scope=scope)
    assert "error" not in res, res
    assert not res.get("ambiguous"), res
    assert res["fantasy_rankings"] == [], res


# A player's ranking rows are those of his FantasyPros ids (ff_playerids.fantasypros_id ->
# gsis_id): ff_rankings_draft.id and ff_rankings_week.fantasypros_id. On the 2024 database:
# SELECT fantasypros_id, gsis_id, name FROM ff_playerids WHERE fantasypros_id IN (18220, 26221)
#   -> (18220, '00-0035236', 'Byron Murphy') CB MIN; (26221, '00-0039309', 'Byron Murphy') DT SEA
# SELECT DISTINCT player, id, pos, team FROM ff_rankings_draft WHERE player ILIKE 'byron murphy%'
#   -> ('Byron Murphy Jr.', 18220, 'DB', 'MIN'), ('Byron Murphy II', 26221, 'DL', 'SEA')
# Folded with the suffix dropped, both are 'byron murphy'.
BYRON_MURPHY_CB = "00-0035236"
_RANKING_TABLES = {"draft": ("ff_rankings_draft", "player", "id"),
                   "week": ("ff_rankings_week", "player_name", "fantasypros_id")}


def _mapped_rows(conn, scope, gsis_id):
    table, col, id_col = _RANKING_TABLES[scope]
    _rankings_needed(conn, scope)
    return sorted(_all(conn, f"SELECT {col}, {id_col} FROM {table} WHERE {id_col} IN "
                             "(SELECT fantasypros_id FROM ff_playerids WHERE gsis_id = ?)", [gsis_id]))


@pytest.mark.integration
@pytest.mark.parametrize("scope", ["draft", "week"])
def test_fantasy_rankings_player_id_does_not_merge_a_namesake(real, scope):
    want = _mapped_rows(real, scope, BYRON_MURPHY_CB)
    if not want:
        pytest.skip(f"no {scope} rows for Byron Murphy (CB) in this scrape")
    res = tools.nfl_fantasy_rankings(player_id=BYRON_MURPHY_CB, scope=scope, limit=500)
    assert res["player_id"] == BYRON_MURPHY_CB
    rows = res["fantasy_rankings"]
    assert sorted((r["player"], ) for r in rows) == sorted((n, ) for n, _ in want)
    assert {r["position"] for r in rows} == {"DB"}, rows  # never 'Byron Murphy II' (DL, SEA)


@pytest.mark.integration
@pytest.mark.parametrize("scope", ["draft", "week"])
@pytest.mark.parametrize("player,gsis_id", [
    # players: 'Josh Palmer' (first_name Joshua); FantasyPros stores 'Joshua Palmer' (22845).
    ("Joshua Palmer", "00-0036988"),
    # players: 'Marquise Brown'; FantasyPros stores 'Hollywood Brown' (18226).
    ("Marquise Brown", "00-0035662"),
])
def test_fantasy_rankings_finds_a_player_stored_under_another_name(real, scope, player, gsis_id):
    want = _mapped_rows(real, scope, gsis_id)
    if not want:
        pytest.skip(f"no {scope} rows for {player} in this scrape")
    res = tools.nfl_fantasy_rankings(player=player, scope=scope, limit=500)
    assert res["player_id"] == gsis_id, res
    assert sorted(r["player"] for r in res["fantasy_rankings"]) == sorted(n for n, _ in want)


@pytest.mark.integration
def test_fantasy_rankings_player_without_rows_gets_a_reason(real):
    _rankings_needed(real, "draft")
    # SELECT gsis_id FROM players WHERE display_name='Tom Brady' -> '00-0019596' (retired, unranked)
    assert _mapped_rows(real, "draft", "00-0019596") == []
    res = tools.nfl_fantasy_rankings(player="Tom Brady")
    assert res["fantasy_rankings"] == [] and res["player_id"] == "00-0019596"
    assert "Tom Brady" in res["reason"], res


# Synthetic ranking database: every branch of the id match, independent of the scrape.
RK = {"murphy_cb": "00-0000001", "murphy_dt": "00-0000002", "rubio": "00-0000003",
      "ponder_ne": "00-0000004", "ponder_tb": "00-0000005", "oldman": "00-0000006"}


@pytest.fixture
def rankings_db(tmp_path, monkeypatch):
    path = tmp_path / "rankings.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE players (gsis_id VARCHAR, display_name VARCHAR, short_name VARCHAR, "
                "first_name VARCHAR, common_first_name VARCHAR, last_name VARCHAR, position VARCHAR, "
                "latest_team VARCHAR, rookie_season INTEGER, last_season INTEGER, pfr_id VARCHAR)")
    con.executemany("INSERT INTO players VALUES (?, ?, ?, ?, ?, ?, ?, ?, 2020, 2024, NULL)", [
        (RK["murphy_cb"], "Byron Murphy", "B.Murphy", "Byron", "Byron", "Murphy", "CB", "MIN"),
        (RK["murphy_dt"], "Byron Murphy II", "B.Murphy", "Byron", "Byron", "Murphy", "DT", "SEA"),
        (RK["rubio"], "Gabe Rubio", "G.Rubio", "Gabriel", "Gabe", "Rubio", "DE", "PIT"),
        (RK["ponder_ne"], "Eli Ponder", "E.Ponder", "Eli", "Eli", "Ponder", "LB", "NE"),
        (RK["ponder_tb"], "Eli Ponder", "E.Ponder", "Eli", "Eli", "Ponder", "LB", "TB"),
        (RK["oldman"], "Tom Oldman", "T.Oldman", "Tom", "Tom", "Oldman", "QB", "NE"),
    ])
    con.execute("CREATE TABLE rosters (gsis_id VARCHAR, team VARCHAR, season INTEGER)")
    con.executemany("INSERT INTO rosters VALUES (?, ?, 2024)", [
        (RK["murphy_cb"], "MIN"), (RK["murphy_dt"], "SEA"), (RK["rubio"], "PIT"),
        (RK["ponder_ne"], "NE"), (RK["ponder_tb"], "TB"), (RK["oldman"], "NE")])
    con.execute("CREATE TABLE ff_playerids (fantasypros_id BIGINT, gsis_id VARCHAR)")
    con.executemany("INSERT INTO ff_playerids VALUES (?, ?)", [
        (101, RK["murphy_cb"]), (102, RK["murphy_dt"]), (601, RK["oldman"]), (None, RK["rubio"])])
    con.execute("CREATE TABLE ff_rankings_draft (page_type VARCHAR, ecr_type VARCHAR, player VARCHAR, "
                "id BIGINT, pos VARCHAR, team VARCHAR, ecr DOUBLE, sd DOUBLE, best BIGINT, "
                "worst BIGINT, bye BIGINT, scrape_date VARCHAR)")
    con.executemany("INSERT INTO ff_rankings_draft VALUES "
                    "('redraft-idp', 'ro', ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)", [
        ("Byron Murphy Jr.", 101, "DB", "MIN", 10.0), ("Byron Murphy II", 102, "DL", "SEA", 11.0),
        ("Gabriel Rubio", 201, "DL", "PIT", 12.0), ("Gabriel Rubio", 202, "WR", "FA", 13.0),
        ("Eli Ponder", 301, "DL", "NE", 14.0), ("Eli Ponder", 302, "DL", "FA", 15.0),
        ("Kansas City Chiefs", 401, "DST", "KC", 16.0)])
    con.execute("CREATE TABLE ff_rankings_week (page VARCHAR, fantasypros_id BIGINT, "
                "player_name VARCHAR, pos VARCHAR, team VARCHAR, rank BIGINT, ecr DOUBLE, sd DOUBLE, "
                "best BIGINT, worst BIGINT, pos_rank VARCHAR, player_opponent VARCHAR, "
                "player_bye_week BIGINT, start_sit_grade VARCHAR, note VARCHAR, scrape_date VARCHAR)")
    con.execute("INSERT INTO ff_rankings_week (page, fantasypros_id, player_name, pos, team, ecr) "
                "VALUES ('dl', 102, 'Byron Murphy II', 'DL', 'SEA', 3.0)")
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


def _ranked(**kwargs):
    res = tools.nfl_fantasy_rankings(limit=500, **kwargs)
    assert "error" not in res and not res.get("ambiguous"), res
    return res, [(r["player"], r["position"], r["team"]) for r in res["fantasy_rankings"]]


def test_fantasy_rankings_id_link_keeps_namesakes_apart(rankings_db):
    res, rows = _ranked(player_id=RK["murphy_cb"])
    assert rows == [("Byron Murphy Jr.", "DB", "MIN")] and "reason" not in res
    assert _ranked(player_id=RK["murphy_dt"])[1] == [("Byron Murphy II", "DL", "SEA")]
    assert _ranked(player_id=RK["murphy_dt"], scope="week")[1] == [("Byron Murphy II", "DL", "SEA")]
    assert _ranked(player_id=RK["murphy_cb"], scope="week")[1] == []
    assert tools.nfl_fantasy_rankings(player="Byron Murphy")["ambiguous"] is True


def test_fantasy_rankings_unlinked_row_needs_name_and_position_group(rankings_db):
    # Gabe Rubio (first_name Gabriel, DE) has no FantasyPros id: the DL row named Gabriel Rubio
    # is his; the WR of the same name is not.
    res, rows = _ranked(player="Gabe Rubio")
    assert res["player_id"] == RK["rubio"] and rows == [("Gabriel Rubio", "DL", "PIT")]


def test_fantasy_rankings_unlinked_namesakes_narrow_to_the_players_team(rankings_db):
    # Two unlinked DL rows named Eli Ponder (NE and FA). The NE player gets the NE row; the TB
    # player matches neither team, so neither row is his and the reason says why.
    assert _ranked(player_id=RK["ponder_ne"])[1] == [("Eli Ponder", "DL", "NE")]
    res, rows = _ranked(player_id=RK["ponder_tb"])
    assert rows == [] and "301, 302" in res["reason"], res


def test_fantasy_rankings_reasons_for_no_rows(rankings_db):
    res, rows = _ranked(player_id=RK["oldman"])  # linked to 601, which no ranking row has
    assert rows == [] and "Tom Oldman" in res["reason"], res
    res, rows = _ranked(player="Nobody Atall")
    assert rows == [] and "no player has that name" in res["reason"], res
    res, rows = _ranked(player_id=RK["murphy_cb"], ranking_set="dynasty")
    assert rows == [] and "Byron Murphy" in res["reason"], res


def test_fantasy_rankings_team_defense_matches_its_stored_name(rankings_db):
    res, rows = _ranked(player="Kansas City Chiefs")
    assert rows == [("Kansas City Chiefs", "DST", "KC")] and "player_id" not in res


# ── 7. nfl_roster player and player_id ────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.parametrize("kwargs,gid", [
    ({"player": "Patrick Mahomes"}, MAHOMES), ({"player": "P.Mahomes"}, MAHOMES),
    ({"player": "Mahomes"}, MAHOMES), ({"player_id": MAHOMES}, MAHOMES),
    ({"player": "T.Kelce"}, KELCE), ({"player": "Travis Kelce"}, KELCE),
    ({"player_id": KELCE}, KELCE), ({"player_id": "00-0036152"}, "00-0036152"),
])
def test_roster_player_filter_matches_sql(real, kwargs, gid):
    # SELECT gsis_id, team, season FROM rosters WHERE gsis_id=? AND season=2024
    #   00-0033873 -> [('00-0033873','KC',2024)]; 00-0030506 -> [('00-0030506','KC',2024)]
    #   00-0036152 (the other Lamar Jackson) -> one row
    expected = sorted(_all(real, "SELECT gsis_id, team, season FROM rosters "
                                 "WHERE gsis_id=? AND season=2024", [gid]))
    assert len(expected) == 1
    if gid in (MAHOMES, KELCE):
        assert expected == [(gid, "KC", 2024)]
    rows = _all_pages(tools.nfl_roster, "players", season=2024, **kwargs)
    assert sorted((r["gsis_id"], r["team"], r["season"]) for r in rows) == expected


@pytest.mark.integration
def test_roster_player_without_season_matches_sql(real):
    # SELECT count(*) FROM rosters WHERE gsis_id='00-0033873' -> 1 (2024-only database)
    (n,) = _one(real, "SELECT count(*) FROM rosters WHERE gsis_id=?", [MAHOMES])
    rows = _all_pages(tools.nfl_roster, "players", player="Patrick Mahomes")
    assert len(rows) == n and {r["gsis_id"] for r in rows} == {MAHOMES}


@pytest.mark.integration
@pytest.mark.parametrize("player", ["Mahom", "Kelc"])
def test_roster_substring_matches_nothing(real, player):
    res = tools.nfl_roster(player=player, season=2024)
    assert "error" not in res, res
    assert not res.get("ambiguous"), res
    assert res["players"] == [], res


# ── 8. nfl_team_records season_from / season_to ───────────────────────────────

# Regular-season W-L-T per team from schedules:
# WITH g AS (SELECT season, home_team team, home_score s, away_score o FROM schedules
#            WHERE game_type='REG' AND home_score IS NOT NULL
#            UNION ALL SELECT season, away_team, away_score, home_score FROM schedules
#            WHERE game_type='REG' AND home_score IS NOT NULL)
# SELECT season, team, sum(s>o), sum(s<o), sum(s=o) FROM g GROUP BY 1, 2
_RECORDS_SQL = """
    WITH g AS (
        SELECT season, home_team AS team, home_score AS s, away_score AS o FROM schedules
        WHERE game_type = 'REG' AND home_score IS NOT NULL
        UNION ALL
        SELECT season, away_team, away_score, home_score FROM schedules
        WHERE game_type = 'REG' AND home_score IS NOT NULL)
    SELECT season, team,
           CAST(sum(CASE WHEN s > o THEN 1 ELSE 0 END) AS INTEGER),
           CAST(sum(CASE WHEN s < o THEN 1 ELSE 0 END) AS INTEGER),
           CAST(sum(CASE WHEN s = o THEN 1 ELSE 0 END) AS INTEGER)
    FROM g WHERE season BETWEEN ? AND ? GROUP BY 1, 2 ORDER BY 1, 2
"""


def _records(**kwargs) -> list[dict]:
    return _all_pages(tools.nfl_team_records, "records", **kwargs)


@pytest.mark.integration
@pytest.mark.parametrize("lo,hi", [(2024, 2024), (2020, 2024), (2024, 2030)])
def test_team_records_range_including_2024(real, lo, hi):
    expected = _all(real, _RECORDS_SQL, [lo, hi])
    # _RECORDS_SQL over 2024 -> 32 teams, 272 wins in total; KC (2024,'KC',15,2,0),
    # LV (2024,'LV',4,13,0)
    assert len(expected) == 32 and sum(r[2] for r in expected) == 272
    assert (2024, "KC", 15, 2, 0) in expected and (2024, "LV", 4, 13, 0) in expected
    rows = _records(season_from=lo, season_to=hi)
    assert {r["season"] for r in rows} == {2024}
    got = sorted((r["season"], r["team"], r["wins"], r["losses"], r["ties"]) for r in rows)
    assert got == sorted(expected)


@pytest.mark.integration
@pytest.mark.parametrize("kwargs", [
    {"season_from": 2025}, {"season_from": 2025, "season_to": 2030}, {"season_to": 2023},
    {"season_from": 2020, "season_to": 2023},
    {"team": "KC", "season_from": 2025}, {"team": "KC", "season_to": 2023},
])
def test_team_records_range_excluding_2024_is_empty(real, kwargs):
    lo, hi = kwargs.get("season_from", 1900), kwargs.get("season_to", 3000)
    # SELECT count(*) FROM schedules WHERE game_type='REG' AND season BETWEEN lo AND hi -> 0
    # (schedules hold 2024 only)
    assert _one(real, "SELECT count(*) FROM schedules WHERE game_type='REG' "
                      "AND season BETWEEN ? AND ?", [lo, hi]) == (0,)
    res = tools.nfl_team_records(**kwargs)
    assert "error" not in res, res
    assert res["records"] == [], res


@pytest.mark.integration
def test_team_records_season_from_alone_is_enough(real):
    rows = _records(season_from=2024)
    # _RECORDS_SQL over 2024..2024 -> 32 rows
    assert len(rows) == len(_all(real, _RECORDS_SQL, [2024, 2024])) == 32


@pytest.mark.integration
def test_team_records_team_and_range(real):
    # _RECORDS_SQL over 2023..2024 for KC -> [(2024, 'KC', 15, 2, 0)]
    assert [r for r in _all(real, _RECORDS_SQL, [2023, 2024]) if r[1] == "KC"] == [
        (2024, "KC", 15, 2, 0)]
    rows = _records(team="KC", season_from=2023, season_to=2024)
    assert [(r["season"], r["team"], r["wins"], r["losses"], r["ties"]) for r in rows] == [
        (2024, "KC", 15, 2, 0)]


@pytest.mark.integration
def test_team_records_range_over_mcp(real):
    is_error, res = _call("nfl_team_records", {"team": "KC", "season_from": 2024, "season_to": 2024})
    assert not is_error, res
    assert [(r["season"], r["wins"], r["losses"]) for r in res["records"]] == [(2024, 15, 2)]
