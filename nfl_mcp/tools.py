"""
NFL MCP Server tools:
  nfl_schema        → returns column reference so the model knows what to query
  nfl_status        → database health: row counts, loaded seasons, tables
  nfl_query         → read-only SELECT with safety guardrails
  nfl_search_plays  → structured play search
  nfl_team_stats    → pre-aggregated team stats
  nfl_player_stats  → player stats by season
  nfl_compare       → side-by-side comparison
"""

import difflib
import logging
import re
import threading
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any, Dict

import duckdb

from .database import get_db_connection
from .franchises import (
    RECORDS_FIRST_NFLVERSE_SEASON,
    FRANCHISES,
    availability,
    canonical_team,
    era_name,
)
from .schema_pbp import _SCHEMA_SUMMARY, _SCHEMA_CATEGORIES

logger = logging.getLogger("nfl-mcp")


# ── Safety guardrails ──────────────────────────────────────────────────────────

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|UPSERT"
    r"|EXECUTE|EXEC|CALL|COPY|GRANT|REVOKE|VACUUM|ANALYZE|CLUSTER"
    r"|REINDEX|COMMENT|SECURITY|OWNER|TABLESPACE|SCHEMA"
    r"|SET|DO|LISTEN|NOTIFY|PREPARE|DEALLOCATE|LOAD|DISCARD|RESET"
    r"|pg_read_file|pg_read_binary_file|pg_write_file|pg_sleep"
    r"|lo_import|lo_export|dblink|current_setting"
    r"|pg_terminate_backend|pg_cancel_backend)\b",
    re.IGNORECASE,
)

_MAX_ROWS = 500
_QUERY_TIMEOUT_SECONDS = 10


# ── Player identity ────────────────────────────────────────────────────────────
#
# Every tool that takes a player goes through nfl_resolve_player. Names are not unique:
# `players` holds two Justin Jeffersons (00-0036322 WR MIN, 00-0041075 LB CLE) with the
# same display_name and the same short_name, so a name that matches more than one player
# is reported as ambiguous instead of merging their stats.

_SUFFIX_WORDS = {"jr", "sr", "ii", "iii", "iv", "v"}

# Last token of the surname, accent-folded and punctuation-free, so 'O'Brien',
# 'Valdes-Scantling' and 'Ah You' are searchable by the part a caller types.
_SURNAME_KEY = (
    "list_last(str_split_regex(trim(regexp_replace(regexp_replace("
    "strip_accents(lower(COALESCE(last_name, ''))), '[^a-z0-9 ]', ' ', 'g'),"
    "' +', ' ', 'g')), ' +'))"
)

_IDENTITY_COLUMNS = (
    "gsis_id, display_name, first_name, football_name, common_first_name, last_name, "
    "suffix, position, latest_team, rookie_season, last_season, pfr_id"
)


def _fold_tokens(raw: str) -> list[str]:
    """Name as lower-case accent-free word tokens, punctuation turned into spaces."""
    folded = "".join(c for c in unicodedata.normalize("NFKD", raw) if not unicodedata.combining(c))
    return re.sub("[^a-z0-9 ]", " ", folded.lower()).split()


def _name_tokens(raw: str) -> list[str]:
    """`_fold_tokens` without any trailing generational suffix (Jr., Sr., II, III).
    The suffix is matched separately, by `_suffix_token`."""
    words = _fold_tokens(raw)
    while words and words[-1] in _SUFFIX_WORDS:
        words.pop()
    return words


def _suffix_token(raw: str) -> str | None:
    """The generational suffix a name carries, as a token ('jr', 'iii'), else None.
    Both sides of a comparison go through here, so 'Marvin Harrison Jr.' can pick the
    son out of the two Marvin Harrisons. `players.suffix` is NULL for some players who
    go by Jr., and those stay ambiguous."""
    words = _fold_tokens(raw)
    return words[-1] if words and words[-1] in _SUFFIX_WORDS else None


def _first_name_matches(head: list[str], row: dict, initial: bool = True) -> bool:
    """True when the tokens before the surname name the same first name as `row`.
    With `initial` off only a spelled-out first name matches; with it on, 'J.' matches
    'Justin' too. Callers try the spelled-out pass first and fall back to initials,
    so 'D.J. Moore' finds the two D.J./DJ Moores and not every D. Moore.

    Neither pass may read a one-letter query as spelled out. plays writes 'B.Robinson'
    for Bijan Robinson and for Brian Robinson, and `players` holds a Michael Robinson
    whose first name is literally 'B.' - spelled out for him, initial for the others.
    A short given name of two or three letters is different: 'Mi.Wilson' and
    'Ja.Williams' are how plays tells Michael from Mecus and from James, so the initial
    pass accepts a prefix of the stored first name."""
    if not head:
        return True
    joined = "".join(head)
    if not initial and len(joined) == 1:
        return False
    for field in ("first_name", "football_name", "common_first_name"):
        forms = _name_tokens(row.get(field) or "")
        if not forms:
            continue
        if joined == "".join(forms) or (len(head[0]) > 1 and head[0] == forms[0]):
            return True
        if initial and (len(head) == 1 or len(head[0]) == 1) and forms[0].startswith(head[0]):
            return True
    return False


def _player_identity(row: dict) -> dict:
    """The fields that tell two same-name players apart, keyed by gsis_id."""
    name = row["display_name"]
    if row.get("suffix"):
        name = f"{name} {row['suffix']}"
    return {
        "gsis_id": row["gsis_id"],
        "full_name": name,
        "pfr_id": row.get("pfr_id"),
        "position": row.get("position"),
        "team": row.get("latest_team"),
        "first_season": row.get("rookie_season"),
        "last_season": row.get("last_season"),
    }


def _label_players(rows: list[dict], name_key: str = "full_name",
                   team_key: str = "team") -> list[dict]:
    """Add `label`, e.g. 'Justin Jefferson (WR, MIN)', to every player row."""
    for row in rows:
        facts = [str(row.get("position") or ""), str(row.get(team_key) or "")]
        tail = ", ".join(f for f in facts if f)
        row["label"] = f"{row.get(name_key)} ({tail})" if tail else str(row.get(name_key))
    return rows


def _echo_player(result: dict[str, Any], identity: dict | None) -> dict[str, Any]:
    """Tag a tool result with the player its name or id resolved to."""
    if identity:
        result["player"] = identity
    return result


def nfl_resolve_player(
    player: str | None = None,
    player_id: str | None = None,
    team: str | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    position: str | None = None,
) -> dict[str, Any]:
    """Resolve one player, or say why the name cannot pick one.

    `player_id` (a gsis_id) wins and the name is ignored. A name may be a full name
    ('Justin Jefferson'), the nflverse short form ('J.Jefferson'), or the same with a
    2-3 letter prefix ('Mi.Wilson', 'Ja.Williams') as plays writes it; 'Marvin Harrison
    Jr.' is told from his father by the suffix. A one-letter first name is a normal form
    here, so 'J.Smith' means every Smith whose first name starts with J. `team` takes any
    code the club published under - typed, current, or alias ('OAK', 'LAR', 'RAM') - and a
    latest-team match counts only when that club is a past or current member of the team
    asked for: roster tables prove the season, latest-team does not. `position` filters no
    rows here - the seven tools that resolve a player filter their own rows with it, and
    players.position uses different codes than snap_counts.position does.
    Returns exactly one of:
      {'resolved': {...}}                      - one player, use resolved['gsis_id']
      {'ambiguous': True, 'candidates': [...]} - several players, no stats to return
      {'not_found': True, 'did_you_mean': [...]}
      {'error': ...}                           - the lookup itself failed
    Never merges two ids. A name that covers several players comes back ambiguous - with
    the one exception the caller asks for: a generational suffix in the query picks the
    player who carries it, which is how 'Marvin Harrison Jr.' is not his father. A 'count'
    larger than 'candidates' means the list was cut off at 20. Never call this to build a
    list - 'candidates' holds at most 20. No caller may pass position= here: the tools that
    resolve a player narrow their own candidates with it.
    """
    tokens: list[str] = [] if player_id else _name_tokens(player or "")
    if not tokens and not player_id:
        return {"not_found": True, "query": player or "", "did_you_mean": []}

    conditions: list[str] = []
    params: list[Any] = []
    if player_id:
        conditions.append("gsis_id = ?")
        params.append(str(player_id).strip())
    else:
        conditions.append(f"{_SURNAME_KEY} = ?")
        params.append(tokens[-1])
        if season is not None:
            conditions.append("rookie_season <= ? AND last_season >= ?")
            params.extend([int(season), int(season)])
        if season_from is not None:
            conditions.append("last_season >= ?")
            params.append(int(season_from))
        if season_to is not None:
            conditions.append("rookie_season <= ?")
            params.append(int(season_to))
        if position:
            conditions.append("position ILIKE ?")
            params.append(position.upper())
        if team and team.strip():
            # A mid-season trade is in rosters_weekly and in latest_team, never in the
            # season's single rosters row, so Davante Adams 2024 answers to both LV and NYJ.
            # Every code the franchise has used is bound, so a table that stores an era code
            # ('OAK', 'SD', 'STL') matches a caller who types the current one, and the other way.
            codes = _team_codes(team)
            marks = ", ".join("?" for _ in codes)
            roster_season = " AND season = ?" if season is not None else ""
            conditions.append(
                f"(COALESCE(latest_team, '') IN ({marks}) OR gsis_id IN "
                f"(SELECT gsis_id FROM rosters WHERE team IN ({marks}){roster_season}) OR gsis_id IN "
                f"(SELECT gsis_id FROM rosters_weekly WHERE team IN ({marks}){roster_season}))"
            )
            params.extend(codes)
            for _ in range(2):
                params.extend(codes)
                if season is not None:
                    params.append(int(season))

    try:
        rows = _execute(
            f"SELECT {_IDENTITY_COLUMNS} FROM players "
            f"WHERE {' AND '.join(conditions)} "
            "ORDER BY last_season DESC NULLS LAST, gsis_id",
            params,
        )
        if player_id:
            matched = rows
        else:
            spelled_out = [r for r in rows if _first_name_matches(tokens[:-1], r, initial=False)]
            matched = spelled_out or [r for r in rows if _first_name_matches(tokens[:-1], r)]
            wanted = _suffix_token(player or "")
            if wanted and len(matched) > 1:
                # players.suffix is filled for 161 rows only, the rest carry the suffix in
                # display_name (Robert Griffin III), so both columns are checked.
                picked = [
                    r for r in matched
                    if (_suffix_token(r.get("suffix") or "") or _suffix_token(r["display_name"])) == wanted
                ]
                matched = picked or matched
        if not matched:
            near = [r["display_name"] for r in rows]
            if not near and not player_id:
                near = [
                    r["display_name"]
                    for r in _execute(
                        f"SELECT display_name FROM players WHERE {_SURNAME_KEY} LIKE ? LIMIT 100",
                        [f"{tokens[-1][:4]}%"],
                    )
                ]
            # Rank suggestions by the given name: with a shared surname every candidate
            # scores the same on the surname, and the full string falls back to alphabetical.
            return {
                "not_found": True,
                "query": player_id or player or "",
                "did_you_mean": _nearest(tokens[0] if tokens else "", near),
            }
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.exception("Player lookup failed")
        return {"error": str(e)}

    if len(matched) > 1:
        return {
            "ambiguous": True,
            "count": len(matched),
            "candidates": _label_players([_player_identity(r) for r in matched[:20]]),
            "hint": "Two or more players match. Re-run with player_id set to one gsis_id.",
        }
    return {"resolved": _label_players([_player_identity(matched[0])])[0]}


def _team_codes(team: str) -> list[str]:
    """Every code a club can be recorded under: the one the caller typed, its current one,
    then that franchise's other aliases. 'LV' reaches rows written 'OAK', and 'LA' reaches
    'STL', 'LAR' and 'RAM'. An unknown code stands alone."""
    raw = team.strip().upper()
    canon = canonical_team(raw)
    codes = [raw] if canon is None else [raw, canon, *FRANCHISES[canon].aliases]
    return list(dict.fromkeys(codes))


def _team_filter(conditions: list[str], params: list[Any], column: str, team: str | None) -> None:
    """Append a row filter on a team code the caller typed, on every code that means it.

    The resolver answers to the canonical code while the row tables carry the code of the
    era, so a filter on the raw code alone resolves the player and returns nothing - an
    empty success a caller reads as 'no data'. Tools name their own column, and a value
    that is only spaces applies no filter.
    """
    if not team or not team.strip():
        return
    codes = _team_codes(team)
    if len(codes) == 1:
        conditions.append(f"{column} = ?")
        params.append(codes[0])
    else:
        conditions.append(f"{column} IN ({', '.join('?' for _ in codes)})")
        params.extend(codes)


def _player_filter(
    sql: str | None = None,
    conditions: list[str] | None = None,
    params: list[Any] | None = None,
    *,
    player: str | None = None,
    player_id: str | None = None,
    team: str | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    position: str | None = None,
    value_key: str = "gsis_id",
    value_count: int = 1,
    missing_reason: str = "",
    must_have_player: bool = False,
) -> dict[str, Any]:
    """Resolve the player a caller named and put his id into that tool's WHERE clause.

    Returns {'identity': None} when the caller named no player, the resolver's ambiguous
    or not_found or error answer when the name cannot pick exactly one player, and
    otherwise {'identity': {...}} with `sql` appended and the id bound `value_count`
    times. Pass sql=None when the tool filters its own rows from the id afterwards.

    `position` never takes a player out of the lookup, it only narrows the candidates.
    The row tables use another vocabulary: snap_counts writes 'T' where players writes
    'OT', so a resolver that filtered on position found no player at all for
    `nfl_snap_counts(player='Trent Williams', position='T')`. So the caller's position is
    compared with the candidates afterwards, and a position that matches none of them
    leaves the answer as the resolver gave it.

    Two limits on that compare. It is exact: 'LB' does not match a stored 'OLB', 'S' does
    not match 'FS' or 'SAF', and when nothing matches the answer stays ambiguous - the
    safe direction. And it only runs over a complete list: when the resolver reported more
    matches than candidates returned (`J.Smith` reports 41 and lists 20), the candidates
    are a cut of the field and narrowing them would be a silent pick.
    """
    if not player and not player_id:
        if must_have_player:
            return {"not_found": True, "query": player or "", "did_you_mean": []}
        return {"identity": None}
    resolved = nfl_resolve_player(
        player, player_id=player_id, team=team, season=season,
        season_from=season_from, season_to=season_to,
    )
    complete = resolved.get("count") == len(resolved.get("candidates", []))
    if "ambiguous" in resolved and position and complete:
        want = position.strip().upper()
        keep = [c for c in resolved["candidates"]
                if str(c.get("position") or "").strip().upper() == want]
        if keep:
            resolved = ({"resolved": keep[0]} if len(keep) == 1 else
                        {**resolved, "candidates": keep, "count": len(keep)})
    if "resolved" not in resolved:
        return resolved
    identity = resolved["resolved"]
    value = identity.get(value_key)
    if not value:
        return {"not_found": True, "query": identity["label"], "did_you_mean": [],
                "reason": missing_reason or f"This player has no {value_key}."}
    if sql is not None and conditions is not None and params is not None:
        conditions.append(sql)
        params.extend([value] * value_count)
    return {"identity": identity}


def nfl_schema(category: str | None = None, table: str | None = None) -> Dict[str, Any]:
    """Return schema reference. Summary by default, or a specific category for detail.
    Pass table='<name>' to get the live column list for any non-pbp table."""
    if table is not None:
        try:
            rows = _execute(
                "SELECT column_name, data_type "
                "FROM information_schema.columns "
                "WHERE table_name = ? AND table_schema = 'main' "
                "ORDER BY ordinal_position",
                [table],
            )
            if rows:
                return {"table": table, "columns": rows}
            available = _execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            )
            return {
                "error": f"Table '{table}' not found in schema 'main'",
                "available_tables": [r["table_name"] for r in available],
            }
        except (duckdb.Error, ValueError, TimeoutError) as e:
            logger.error("Tool error: %s", e, exc_info=True)
            return {"error": str(e)}
    if category is None:
        try:
            tables = _execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_name NOT LIKE '\\_%' ESCAPE '\\' "
                "ORDER BY table_name"
            )
            table_names = [r["table_name"] for r in tables]
        except Exception:
            table_names = []
        return {
            "schema": _SCHEMA_SUMMARY,
            "available_tables": table_names,
            "hint": "Pass category='<name>' for full PBP column details. Pass table='<name>' for any other table's columns.",
        }
    cat = category.lower().strip()
    if cat == "all":
        full = "\n".join(v.strip() for v in _SCHEMA_CATEGORIES.values())
        return {"schema": full}
    if cat in _SCHEMA_CATEGORIES:
        return {"category": cat, "schema": _SCHEMA_CATEGORIES[cat].strip()}
    return {"error": f"Unknown category '{cat}'", "available": list(_SCHEMA_CATEGORIES.keys())}


def nfl_status() -> Dict[str, Any]:
    """Return database health: play counts, loaded seasons, and a summary of all ingested datasets."""
    try:
        total = _execute("SELECT COUNT(*) AS total_plays FROM plays")
        seasons = _execute(
            "SELECT season, season_type, COUNT(*) AS plays "
            "FROM plays GROUP BY season, season_type ORDER BY season, season_type"
        )
        min_max = _execute(
            "SELECT MIN(season) AS first_season, MAX(season) AS last_season, "
            "COUNT(DISTINCT season) AS num_seasons "
            "FROM plays"
        )
        datasets = _execute(
            "SELECT dataset_id, table_name, SUM(row_count) as total_rows "
            "FROM _ingest_metadata "
            "GROUP BY dataset_id, table_name "
            "ORDER BY dataset_id"
        )
        last_refreshed = _execute(
            "SELECT MAX(loaded_at) as last_refreshed FROM _ingest_metadata"
        )
        return {
            "plays": {
                "total_plays": total[0]["total_plays"] if total else 0,
                "season_range": min_max[0] if min_max else {},
                "seasons": seasons,
            },
            "datasets": {
                "total_loaded": len(datasets),
                "last_refreshed": last_refreshed[0]["last_refreshed"] if last_refreshed else None,
                "loaded": datasets,
            },
        }
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


# ── Self-correcting SQL errors ────────────────────────────────────────────────
#
# Raw SQL arrives from a language model, so invented column and table names are the
# most common failure. DuckDB reports them as free text whose "nearest name" list is
# spelling-based, which is useless for a name like `play_epa` (the real column is
# `epa`). These helpers re-rank against the live schema by *meaning*: in a snake_case
# name the last word is the metric and earlier words qualify it.

_COLUMN_NOT_FOUND = re.compile(r'Referenced column "(?P<name>[^"]+)" not found', re.IGNORECASE)
_TABLE_NOT_FOUND = re.compile(r"Table with name (?P<name>\S+) does not exist", re.IGNORECASE)
_QUOTED_NAMES = re.compile(r'"(?P<name>[^"]+)"')
_WORDS = re.compile(r"[^a-z0-9]+")

_COLUMN_NAMES_SQL = (
    "SELECT DISTINCT column_name FROM information_schema.columns "
    "WHERE table_schema = 'main'"
)
_TABLE_NAMES_SQL = (
    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
)


def _live_names(sql: str) -> list[str]:
    """Names straight from the live schema; empty if the lookup itself fails."""
    try:
        return [next(iter(row.values())) for row in _execute(sql)]
    except (duckdb.Error, ValueError, TimeoutError):
        return []


def _similarity(unknown: str, candidate: str) -> float:
    """Score how well `candidate` matches the name the caller guessed."""
    guessed, actual = unknown.lower(), candidate.lower()
    if guessed == actual:
        return 5.0
    g_words = [w for w in _WORDS.split(guessed) if w]
    a_words = [w for w in _WORDS.split(actual) if w]
    if not g_words or not a_words:
        return difflib.SequenceMatcher(None, guessed, actual).ratio()
    g_set, a_set = set(g_words), set(a_words)
    if g_words[-1] == a_words[-1]:
        base = 3.0
    elif g_words[-1] in a_set:
        base = 2.5
    elif g_set & a_set:
        base = 1.5
    else:
        return difflib.SequenceMatcher(None, guessed, actual).ratio()
    overlap = len(g_set & a_set) / len(g_set | a_set)
    substring = 0.4 if (guessed in actual or actual in guessed) else 0.0
    return base + overlap + substring


def _nearest(unknown: str, candidates: Iterable[str], limit: int = 5) -> list[str]:
    """Closest real names to `unknown`, best match first, ties alphabetical."""
    unique = {c.lower() for c in candidates if c and c.lower() != unknown.lower()}
    return sorted(unique, key=lambda c: (-_similarity(unknown, c), c))[:limit]


def _sql_error_detail(exc: Exception) -> Dict[str, Any]:
    """Turn a DuckDB binder/catalog error into fields the caller can act on."""
    message = str(exc)
    spelled = [m.group("name") for m in _QUOTED_NAMES.finditer(message)]

    guessed_column = _COLUMN_NOT_FOUND.search(message)
    if guessed_column:
        unknown = guessed_column.group("name")
        return {
            "code": "COLUMN_NOT_FOUND",
            "unknown_column": unknown,
            "did_you_mean": _nearest(unknown, _live_names(_COLUMN_NAMES_SQL) + spelled),
            "hint": (
                "Column names must match the schema exactly. Call "
                "nfl_schema(table='<table>') for non-pbp tables, or "
                "nfl_schema(category='<name>') for play-by-play columns."
            ),
        }

    guessed_table = _TABLE_NOT_FOUND.search(message)
    if guessed_table:
        unknown = guessed_table.group("name")
        return {
            "code": "TABLE_NOT_FOUND",
            "unknown_table": unknown,
            "did_you_mean": _nearest(unknown, _live_names(_TABLE_NAMES_SQL) + spelled),
            "hint": "Call nfl_catalog() or nfl_schema() to list the loaded tables.",
        }

    return {}


def nfl_query(sql: str, max_rows: int = 100) -> Dict[str, Any]:
    """
    Execute a read-only SQL SELECT against the nflread database.

    Safety rules:
      - Only SELECT statements allowed
      - Forbidden mutation/system keywords are blocked
      - Multiple statements (semicolons) blocked
      - Results capped at max_rows (hard max 500)
      - 10-second statement timeout

    Unknown column/table failures also return `code`, the offending name,
    `did_you_mean` (ranked against the live schema) and a `hint`.
    """
    sql = sql.strip()

    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        return {"error": "Only SELECT queries are allowed."}

    match = _FORBIDDEN.search(sql)
    if match:
        return {"error": f"Forbidden keyword: '{match.group()}'"}

    sql_no_trailing = sql.rstrip(";")
    if ";" in sql_no_trailing:
        return {"error": "Multiple statements are not allowed."}

    max_rows = min(max_rows, _MAX_ROWS)
    safe_sql = f"SELECT * FROM ({sql_no_trailing}) AS _q LIMIT {max_rows + 1}"

    try:
        rows = _execute(safe_sql)

        truncated = len(rows) > max_rows
        rows = rows[:max_rows]

        return {
            "rows":      rows,
            "row_count": len(rows),
            "truncated": truncated,
        }

    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e), **_sql_error_detail(e)}


def nfl_search_plays(
    team: str | None = None,
    opponent: str | None = None,
    player: str | None = None,
    player_id: str | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    week: int | None = None,
    season_type: str | None = None,
    play_type: str | None = None,
    situation: str | None = None,
    is_touchdown: bool = False,
    is_turnover: bool = False,
    min_yards: int | None = None,
    max_rows: int = 50,
) -> Dict[str, Any]:
    """Search for plays using structured filters instead of raw SQL."""
    conditions: list[str] = []
    params: list[Any] = []
    _team_filter(conditions, params, "posteam", team)
    _team_filter(conditions, params, "defteam", opponent)
    res = _player_filter("(passer_player_id = ? OR rusher_player_id = ? OR receiver_player_id = ?)",
                         conditions, params, player=player, player_id=player_id, team=team,
                         season=season, season_from=season_from, season_to=season_to, value_count=3)
    if "identity" not in res:
        return res
    identity = res["identity"]
    if season:
        conditions.append("season = ?")
        params.append(int(season))
    if season_from:
        conditions.append("season >= ?")
        params.append(int(season_from))
    if season_to:
        conditions.append("season <= ?")
        params.append(int(season_to))
    if week:
        conditions.append("week = ?")
        params.append(int(week))
    if season_type:
        conditions.append("season_type = ?")
        params.append(season_type)
    if play_type:
        conditions.append("play_type = ?")
        params.append(play_type)
    if situation == "red_zone":
        conditions.append("yardline_100 <= 20")
    elif situation == "third_down":
        conditions.append("down = 3")
    elif situation == "fourth_down":
        conditions.append("down = 4")
    elif situation == "two_minute":
        conditions.append("qtr = 4 AND half_seconds_remaining <= 120")
    if is_touchdown:
        conditions.append("touchdown = 1")
    if is_turnover:
        conditions.append("(interception = 1 OR fumble_lost = 1)")
    if min_yards is not None:
        conditions.append("yards_gained >= ?")
        params.append(int(min_yards))

    where = " AND ".join(conditions) if conditions else "1=1"
    max_rows = min(max_rows, _MAX_ROWS)

    sql = (
        f"SELECT season, week, posteam, defteam, down, ydstogo, play_type, "
        f"yards_gained, epa, \"desc\", passer_player_name, rusher_player_name, "
        f"receiver_player_name, passer_player_id, rusher_player_id, "
        f"receiver_player_id, touchdown, interception "
        f"FROM plays WHERE {where} "
        f"ORDER BY ABS(epa) DESC LIMIT {max_rows}"
    )

    try:
        rows = _execute(sql, params)
        if identity:
            for row in rows:
                row["label"] = identity["label"]
        return _echo_player({"rows": rows, "row_count": len(rows)}, identity)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_team_stats(
    team: str,
    season: int | None = None,
    side: str = "both",
) -> Dict[str, Any]:
    """Get pre-aggregated team stats (offense, defense, or both)."""
    team = team.upper()
    results = {}
    season_clause = ""
    season_params: list[Any] = []
    if season:
        season_clause = " AND season_year = ?"
        season_params.append(int(season))

    try:
        if side in ("offense", "both"):
            sql = f"SELECT * FROM team_offense_stats WHERE team = ?{season_clause} ORDER BY season_year"
            results["offense"] = _execute(sql, [team, *season_params])

        if side in ("defense", "both"):
            sql = f"SELECT * FROM team_defense_stats WHERE team = ?{season_clause} ORDER BY season_year"
            results["defense"] = _execute(sql, [team, *season_params])

        if side in ("situational", "both"):
            sql = f"SELECT * FROM situational_stats WHERE team = ?{season_clause} ORDER BY season_year, situation"
            results["situational"] = _execute(sql, [team, *season_params])

        return results
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_player_stats(
    player_name: str | None = None,
    player_id: str | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    season_type: str | None = None,
    stat_type: str = "passing",
) -> Dict[str, Any]:
    """Aggregate one player's stats by season, filtered and grouped by gsis_id."""
    if stat_type not in ("passing", "rushing", "receiving"):
        return {"error": f"Unknown stat_type: {stat_type}. Use 'passing', 'rushing', or 'receiving'."}

    res = _player_filter(player=player_name, player_id=player_id, season=season,
                         season_from=season_from, season_to=season_to, must_have_player=True)
    if "identity" not in res:
        return res
    identity = res["identity"]
    gsis_id = identity["gsis_id"]
    params: list[Any] = []

    if stat_type == "passing":
        sql = """
            SELECT passer_player_id AS player_id, season, season_type, COUNT(*) AS attempts,
                SUM(CASE WHEN complete_pass = 1 THEN 1 ELSE 0 END) AS completions,
                ROUND(100.0 * SUM(CASE WHEN complete_pass = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS comp_pct,
                SUM(COALESCE(passing_yards, 0)) AS yards,
                SUM(CASE WHEN pass_touchdown = 1 THEN 1 ELSE 0 END) AS touchdowns,
                SUM(CASE WHEN interception = 1 THEN 1 ELSE 0 END) AS interceptions,
                ROUND(AVG(epa), 3) AS avg_epa,
                ROUND(AVG(cpoe), 3) AS avg_cpoe,
                SUM(COALESCE(air_yards, 0)) AS air_yards,
                ROUND(AVG(air_yards), 1) AS avg_air_yards
            FROM plays
            WHERE passer_player_id = ? AND play_type = ?
        """
        params.extend([gsis_id, "pass"])
    elif stat_type == "rushing":
        sql = """
            SELECT rusher_player_id AS player_id, season, season_type, COUNT(*) AS carries,
                SUM(COALESCE(rushing_yards, 0)) AS yards,
                ROUND(AVG(yards_gained), 1) AS yards_per_carry,
                SUM(CASE WHEN rush_touchdown = 1 THEN 1 ELSE 0 END) AS touchdowns,
                SUM(CASE WHEN fumble_lost = 1 THEN 1 ELSE 0 END) AS fumbles_lost,
                ROUND(AVG(epa), 3) AS avg_epa,
                SUM(CASE WHEN yards_gained >= 10 THEN 1 ELSE 0 END) AS explosive_runs
            FROM plays
            WHERE rusher_player_id = ? AND play_type = ?
        """
        params.extend([gsis_id, "run"])
    else:
        sql = """
            SELECT receiver_player_id AS player_id, season, season_type, COUNT(*) AS targets,
                SUM(CASE WHEN complete_pass = 1 THEN 1 ELSE 0 END) AS receptions,
                SUM(COALESCE(receiving_yards, 0)) AS yards,
                ROUND(AVG(yards_after_catch), 1) AS avg_yac,
                SUM(CASE WHEN pass_touchdown = 1 THEN 1 ELSE 0 END) AS touchdowns,
                ROUND(AVG(epa), 3) AS avg_epa,
                SUM(CASE WHEN yards_gained >= 20 THEN 1 ELSE 0 END) AS explosive_plays
            FROM plays
            WHERE receiver_player_id = ? AND play_type = ?
        """
        params.extend([gsis_id, "pass"])

    if season:
        sql += " AND season = ?"
        params.append(int(season))
    if season_from:
        sql += " AND season >= ?"
        params.append(int(season_from))
    if season_to:
        sql += " AND season <= ?"
        params.append(int(season_to))
    if season_type:
        sql += " AND season_type = ?"
        params.append(season_type)
    sql += " GROUP BY player_id, season, season_type ORDER BY season, season_type"

    try:
        rows = _execute(sql, params)
        for row in rows:
            row["label"] = identity["label"]
        return {"player": identity, "stat_type": stat_type, "seasons": rows}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_compare(
    entity1: str,
    entity2: str,
    compare_type: str = "team",
    player_id1: str | None = None,
    player_id2: str | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    season_type: str | None = None,
) -> Dict[str, Any]:
    """Side-by-side comparison of two teams or players.

    Team mode returns one entry per abbreviation. Player mode returns {'players':
    [side1, side2]} in the order asked for - a list, not a dict keyed by the name, so two
    players who share a name (the two Justin Jeffersons) stay two entries. Each side
    carries 'requested' (the text typed) and 'player' with its gsis_id plus the stat
    blocks; a side the name cannot settle carries the resolver's ambiguous or not_found
    answer in place of 'player'.
    """
    e1, e2 = entity1, entity2

    try:
        if compare_type == "team":
            t1, t2 = e1.upper(), e2.upper()
            season_filter = ""
            season_params: list[Any] = []
            if season:
                season_filter += " AND season_year = ?"
                season_params.append(int(season))
            if season_from:
                season_filter += " AND season_year >= ?"
                season_params.append(int(season_from))
            if season_to:
                season_filter += " AND season_year <= ?"
                season_params.append(int(season_to))

            off1 = _execute(
                f"SELECT * FROM team_offense_stats WHERE team = ?{season_filter} ORDER BY season_year",
                [t1, *season_params],
            )
            off2 = _execute(
                f"SELECT * FROM team_offense_stats WHERE team = ?{season_filter} ORDER BY season_year",
                [t2, *season_params],
            )
            def1 = _execute(
                f"SELECT * FROM team_defense_stats WHERE team = ?{season_filter} ORDER BY season_year",
                [t1, *season_params],
            )
            def2 = _execute(
                f"SELECT * FROM team_defense_stats WHERE team = ?{season_filter} ORDER BY season_year",
                [t2, *season_params],
            )

            return {
                entity1: {"offense": off1, "defense": def1},
                entity2: {"offense": off2, "defense": def2},
            }

        elif compare_type == "player":
            season_filter = ""
            season_params: list[Any] = []
            if season:
                season_filter += " AND season = ?"
                season_params.append(int(season))
            if season_from:
                season_filter += " AND season >= ?"
                season_params.append(int(season_from))
            if season_to:
                season_filter += " AND season <= ?"
                season_params.append(int(season_to))
            if season_type:
                season_filter += " AND season_type = ?"
                season_params.append(season_type)

            sides: list[dict[str, Any]] = []
            for name, pid, requested in [
                (e1, player_id1, entity1), (e2, player_id2, entity2)
            ]:
                res = _player_filter(player=name, player_id=pid, season=season,
                                     season_from=season_from, season_to=season_to,
                                     must_have_player=True)
                if "identity" not in res:
                    sides.append({"requested": requested, **res})
                    continue
                identity = res["identity"]
                stats: dict[str, Any] = {"requested": requested, "player": identity}
                for stype, col, ptype in [
                    ("passing", "passer_player_id", "pass"),
                    ("rushing", "rusher_player_id", "run"),
                    ("receiving", "receiver_player_id", "pass"),
                ]:
                    count_sql = (
                        f"SELECT COUNT(*) AS n FROM plays "
                        f"WHERE {col} = ? AND play_type = ?{season_filter}"
                    )
                    count_row = _execute(
                        count_sql, [identity["gsis_id"], ptype, *season_params]
                    )
                    if count_row and count_row[0].get("n", 0) > 0:
                        player_result = nfl_player_stats(
                            player_id=identity["gsis_id"], season=season,
                            season_from=season_from, season_to=season_to,
                            season_type=season_type, stat_type=stype,
                        )
                        if "seasons" in player_result:
                            stats[stype] = player_result["seasons"]
                sides.append(stats)

            return {"players": sides}
        else:
            return {"error": "compare_type must be 'team' or 'player'"}

    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def _execute(sql: str, params: Sequence[Any] | None = None) -> list[dict]:
    """Execute SQL on DuckDB with timeout and cancellation.

    Runs in a worker thread with a dedicated connection, and interrupts
    the active query if it exceeds the timeout.
    """
    result = [None]
    error = [None]
    thread_conn = [None]

    def _run():
        try:
            with get_db_connection() as conn:
                thread_conn[0] = conn
                if params is None:
                    rel = conn.execute(sql)
                else:
                    rel = conn.execute(sql, params)
                columns = [desc[0] for desc in rel.description]
                result[0] = [dict(zip(columns, row)) for row in rel.fetchall()]
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=_QUERY_TIMEOUT_SECONDS)
    if t.is_alive():
        conn = thread_conn[0]
        if conn is not None:
            try:
                conn.interrupt()
            except Exception:
                logger.warning("Failed to interrupt timed out DuckDB query", exc_info=True)
        t.join(timeout=1)
        raise TimeoutError(f"Query exceeded {_QUERY_TIMEOUT_SECONDS} second timeout")
    if error[0]:
        raise error[0]
    return result[0]


def nfl_catalog() -> Dict[str, Any]:
    """List every loaded dataset with row counts and last-refresh timestamp."""
    try:
        rows = _execute("""
            SELECT
                dataset_id,
                table_name,
                SUM(row_count)   AS total_rows,
                COUNT(*)         AS seasons_loaded,
                MIN(season)      AS min_season,
                MAX(season)      AS max_season,
                MAX(loaded_at)   AS last_loaded
            FROM _ingest_metadata
            GROUP BY dataset_id, table_name
            ORDER BY dataset_id
        """)
        return {"datasets": rows, "total_datasets": len(rows)}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_roster(
    team: str | None = None,
    season: int | None = None,
    position: str | None = None,
) -> Dict[str, Any]:
    """
    Look up roster data — who was on a team's roster in a given season.
    Returns name, position, status, jersey number, experience, and measurables.
    """
    conditions, params = [], []
    if team:
        conditions.append("team = ?")
        params.append(team.upper())
    if season:
        conditions.append("season = ?")
        params.append(int(season))
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT full_name, position, depth_chart_position, team, season,
               jersey_number, status, years_exp, college, height, weight
        FROM rosters
        WHERE {where}
        ORDER BY position, full_name
        LIMIT 500
    """
    try:
        rows = _execute(sql, params if params else None)
        return {"players": rows, "count": len(rows)}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_injuries(
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    player: str | None = None,
    player_id: str | None = None,
    report_status: str | None = None,
) -> Dict[str, Any]:
    """
    Look up injury report data. Filter by team, season, week, player name or gsis_id,
    or report status (e.g. 'Out', 'Questionable', 'Doubtful', 'Full Participation').
    """
    conditions: list[str] = []
    params: list[Any] = []
    _team_filter(conditions, params, "team", team)
    if season:
        conditions.append("season = ?")
        params.append(int(season))
    if week:
        conditions.append("week = ?")
        params.append(int(week))
    res = _player_filter("gsis_id = ?", conditions, params,
                         player=player, player_id=player_id, team=team, season=season)
    if "identity" not in res:
        return res
    identity = res["identity"]
    if report_status:
        conditions.append("report_status ILIKE ?")
        params.append(f"%{report_status}%")

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT gsis_id AS player_id, full_name, position, team, season, week,
               report_primary_injury, report_secondary_injury, report_status,
               practice_primary_injury, practice_secondary_injury, practice_status
        FROM injuries
        WHERE {where}
        ORDER BY season, week, team, full_name
        LIMIT 500
    """
    try:
        rows = _execute(sql, params if params else None)
        return _echo_player(
            {"injuries": _label_players(rows), "count": len(rows)}, identity
        )
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_schedule(
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    season_type: str | None = None,
) -> Dict[str, Any]:
    """
    Look up game schedules and results. Filter by team, season, week, or season type.
    Returns scores, spread lines, weather, stadium, and coaching info.
    """
    conditions, params = [], []
    if team:
        t = team.upper()
        conditions.append("(home_team = ? OR away_team = ?)")
        params.extend([t, t])
    if season:
        conditions.append("season = ?")
        params.append(int(season))
    if week:
        conditions.append("week = ?")
        params.append(int(week))
    if season_type:
        conditions.append("game_type = ?")
        params.append(season_type.upper())

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT game_id, season, week, game_type, gameday, gametime,
               away_team, home_team, away_score, home_score, result,
               overtime, spread_line, total_line, div_game,
               roof, surface, temp, wind,
               away_coach, home_coach, referee, stadium
        FROM schedules
        WHERE {where}
        ORDER BY season, week
        LIMIT 500
    """
    try:
        rows = _execute(sql, params if params else None)
        return {"games": rows, "count": len(rows)}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


# ── Team records (1970–present) ───────────────────────────────────────────────

def nfl_team_records(
    team: str | None = None,
    season: int | None = None,
    from_season: int | None = None,
    to_season: int | None = None,
) -> Dict[str, Any]:
    """
    PREFERRED for win-loss / standings questions. Final regular-season records per
    franchise, 1970 to present. Teams are given by their CURRENT abbreviation, so the
    Raiders are 'LV' in every season; each row also carries `club` (the name the franchise
    played that season under) and `source_team` (the code the source used).

    Records before 1999 come from ESPN's opt-in ingest (run 'nfl-mcp ingest --pre1999');
    a franchise ESPN's archive does not cover is reported as record_not_available rather
    than estimated.
    """
    if not team and season is None and from_season is None and to_season is None:
        return {"error": "Provide team, season, or a from_season/to_season range."}

    key = canonical_team(team) if team else None
    if team and key is None:
        return {"error": f"Unknown team '{team}'."}

    if key and season is not None:
        note = availability(key, season)
        if note:
            return {"team": key, "season": int(season), "records": [], "count": 0, **note}

    conditions: list[str] = []
    params: list[Any] = []
    if key:
        conditions.append("team = ?")
        params.append(key)
    if season is not None:
        conditions.append("season = ?")
        params.append(int(season))
    if from_season is not None:
        conditions.append("season >= ?")
        params.append(int(from_season))
    if to_season is not None:
        conditions.append("season <= ?")
        params.append(int(to_season))
    where = " AND ".join(conditions) or "1=1"

    sql = f"""
        SELECT season, team, source_team, games, wins, losses, ties, source
        FROM team_records
        WHERE {where}
        ORDER BY season DESC, team
        LIMIT 500
    """
    try:
        rows = _execute(sql, params if params else None)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}

    for row in rows:
        row["club"] = era_name(row["season"], row["team"])

    result: Dict[str, Any] = {"records": rows, "count": len(rows)}
    if key and not rows:
        result["status"] = "record_not_available"
        if season is not None and season < RECORDS_FIRST_NFLVERSE_SEASON:
            result["reason"] = (
                "No stored record for that franchise and season. Records before "
                f"{RECORDS_FIRST_NFLVERSE_SEASON} come from the opt-in ESPN pull: "
                "run 'nfl-mcp ingest --pre1999'."
            )
        else:
            result["reason"] = "No stored record for that franchise and season."
    return result


def nfl_snap_counts(
    player: str | None = None,
    player_id: str | None = None,
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    position: str | None = None,
) -> Dict[str, Any]:
    """
    Look up snap count data — how many snaps a player or team unit played.
    Returns offense, defense, and special teams snap counts and percentages.
    """
    conditions: list[str] = []
    params: list[Any] = []
    res = _player_filter("pfr_player_id = ?", conditions, params, player=player,
                         player_id=player_id, team=team, season=season, position=position,
                         value_key="pfr_id",
                         missing_reason="This player has no PFR id, which snap_counts rows are keyed by.")
    if "identity" not in res:
        return res
    identity = res["identity"]
    _team_filter(conditions, params, "team", team)
    if season:
        conditions.append("season = ?")
        params.append(int(season))
    if week:
        conditions.append("week = ?")
        params.append(int(week))
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player, position, team, opponent, season, week, game_type,
               offense_snaps, offense_pct, defense_snaps, defense_pct,
               st_snaps, st_pct, pfr_player_id,
               (SELECT gsis_id FROM players WHERE pfr_id = snap_counts.pfr_player_id)
                   AS player_id
        FROM snap_counts
        WHERE {where}
        ORDER BY season, week, team, offense_snaps DESC NULLS LAST
        LIMIT 500
    """
    try:
        rows = _execute(sql, params if params else None)
        return _echo_player(
            {"snap_counts": _label_players(rows, name_key="player"), "count": len(rows)},
            identity,
        )
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_fantasy_opportunity(
    player: str | None = None,
    player_id: str | None = None,
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    position: str | None = None,
) -> Dict[str, Any]:
    """
    Look up fantasy football opportunity data — target share, air yards share,
    carry share, and opportunity scores per player per week. Available 2006–present.
    """
    conditions: list[str] = []
    params: list[Any] = []
    res = _player_filter("player_id = ?", conditions, params,
                         player=player, player_id=player_id, team=team, season=season,
                         position=position)
    if "identity" not in res:
        return res
    identity = res["identity"]
    _team_filter(conditions, params, "posteam", team)
    if season:
        conditions.append("season = ?")
        params.append(int(season))
    if week:
        conditions.append("week = ?")
        params.append(int(week))
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT season, week, posteam, full_name, position, player_id,
               rec_attempt, rush_attempt, rec_air_yards,
               receptions, receptions_exp, rec_yards_gained, rush_yards_gained,
               rec_touchdown, rush_touchdown,
               total_fantasy_points, total_fantasy_points_exp, total_fantasy_points_diff,
               rec_attempt_team, rush_attempt_team
        FROM ff_opportunity
        WHERE {where}
        ORDER BY season DESC, week DESC, total_fantasy_points_exp DESC NULLS LAST
        LIMIT 500
    """
    try:
        rows = _execute(sql, params if params else None)
        return _echo_player(
            {
                "fantasy_opportunity": _label_players(rows, team_key="posteam"),
                "count": len(rows),
            },
            identity,
        )
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_fantasy_rankings(
    player: str | None = None,
    player_id: str | None = None,
    position: str | None = None,
    team: str | None = None,
    scope: str = "draft",
    ranking_set: str | None = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Look up fantasy football expert consensus rankings (ECR). Two snapshots:
    scope='draft' (preseason/dynasty/best-ball draft rankings, ff_rankings_draft) and
    scope='week' (current-week start/sit rankings, ff_rankings_week). Both reflect the
    latest scrape — they are not historical/seasonal. Lower ECR = ranked higher.
    """
    scope = (scope or "draft").lower()
    if scope not in ("draft", "week"):
        return {"error": "scope must be 'draft' or 'week'"}

    try:
        limit = max(1, min(int(limit), _MAX_ROWS))
    except (TypeError, ValueError):
        limit = 50

    conditions: list[str] = []
    params: list[Any] = []
    if scope == "draft":
        table = "ff_rankings_draft"
        set_col, id_col = "page_type", "id"
        select = (
            "player, pos AS position, team, ecr, sd, best, worst, bye, "
            "page_type AS ranking_set, ecr_type, scrape_date"
        )
    else:
        table = "ff_rankings_week"
        set_col, id_col = "page", "fantasypros_id"
        select = (
            "player_name AS player, pos AS position, team, rank, ecr, sd, best, worst, "
            "pos_rank, player_opponent AS opponent, player_bye_week AS bye, "
            "start_sit_grade, note, page AS ranking_set, scrape_date"
        )

    res = _player_filter(
        f"{id_col} IN (SELECT fantasypros_id FROM ff_playerids WHERE gsis_id = ?)",
        conditions, params, player=player, player_id=player_id, team=team,
        position=position)
    if "identity" not in res:
        return res
    identity = res["identity"]
    if position:
        conditions.append("pos ILIKE ?")
        params.append(position.upper())
    _team_filter(conditions, params, "team", team)
    if ranking_set:
        conditions.append(f"{set_col} ILIKE ?")
        params.append(f"%{ranking_set}%")

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT {select}
        FROM {table}
        WHERE {where}
        ORDER BY ecr ASC NULLS LAST
        LIMIT {limit}
    """
    try:
        rows = _execute(sql, params if params else None)
        if identity and not rows and not _execute(
            "SELECT 1 FROM ff_playerids WHERE gsis_id = ? AND fantasypros_id IS NOT NULL LIMIT 1",
            [identity["gsis_id"]],
        ):
            return {"not_found": True, "query": identity["label"], "did_you_mean": [],
                    "reason": "ff_playerids maps no FantasyPros id to this player."}
        out = {"fantasy_rankings": _label_players(rows, name_key="player"), "scope": scope, "count": len(rows)}
        return _echo_player(out, identity)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_ftn_charting(
    team: str | None = None,
    opponent: str | None = None,
    player: str | None = None,
    player_id: str | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    week: int | None = None,
    season_type: str | None = None,
) -> Dict[str, Any]:
    """
    Aggregated FTN manual charting tendencies (2022–present) over offensive scrimmage
    plays (pass + run) for a filtered set: play-action, RPO, screen, no-huddle, motion,
    and trick-play usage rates, plus average defenders in the box (per scrimmage play)
    and pass rushers / blitzers (per dropback). Joins ftn_charting to plays for
    team/player/season context. Offensive perspective (team = posteam).
    """
    conditions = ["f.nflverse_game_id = p.game_id", "f.nflverse_play_id = p.play_id",
                  "p.play_type IN ('pass', 'run')"]
    params: list[Any] = []
    _team_filter(conditions, params, "p.posteam", team)
    _team_filter(conditions, params, "p.defteam", opponent)
    res = _player_filter("(p.passer_player_id = ? OR p.rusher_player_id = ? OR p.receiver_player_id = ?)",
                         conditions, params, player=player, player_id=player_id, team=team,
                         season=season, season_from=season_from, season_to=season_to, value_count=3)
    if "identity" not in res:
        return res
    identity = res["identity"]
    if season:
        conditions.append("p.season = ?")
        params.append(int(season))
    if season_from:
        conditions.append("p.season >= ?")
        params.append(int(season_from))
    if season_to:
        conditions.append("p.season <= ?")
        params.append(int(season_to))
    if week:
        conditions.append("p.week = ?")
        params.append(int(week))
    if season_type:
        conditions.append("p.season_type = ?")
        params.append(season_type.upper())

    where = " AND ".join(conditions)
    sql = f"""
        SELECT
            COUNT(*) AS total_plays,
            SUM(CASE WHEN f.is_play_action THEN 1 ELSE 0 END) AS play_action,
            SUM(CASE WHEN f.is_rpo THEN 1 ELSE 0 END) AS rpo,
            SUM(CASE WHEN f.is_screen_pass THEN 1 ELSE 0 END) AS screen_pass,
            SUM(CASE WHEN f.is_no_huddle THEN 1 ELSE 0 END) AS no_huddle,
            SUM(CASE WHEN f.is_motion THEN 1 ELSE 0 END) AS motion,
            SUM(CASE WHEN f.is_trick_play THEN 1 ELSE 0 END) AS trick_play,
            AVG(f.n_defense_box) AS avg_defenders_in_box,
            AVG(CASE WHEN p.play_type = 'pass' THEN f.n_pass_rushers END) AS avg_pass_rushers,
            AVG(CASE WHEN p.play_type = 'pass' THEN f.n_blitzers END) AS avg_blitzers
        FROM ftn_charting f, plays p
        WHERE {where}
    """
    try:
        rows = _execute(sql, params if params else None)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}

    agg = rows[0] if rows else {}
    total = agg.get("total_plays") or 0
    if not total:
        return _echo_player({"charting": {}, "total_plays": 0}, identity)

    def _pct(n: Any) -> float:
        return round(100.0 * (n or 0) / total, 1)

    def _avg(v: Any) -> float | None:
        return round(v, 2) if v is not None else None

    return _echo_player({
        "charting": {
            "total_plays": total,
            "play_action": {"plays": agg.get("play_action") or 0, "pct": _pct(agg.get("play_action"))},
            "rpo": {"plays": agg.get("rpo") or 0, "pct": _pct(agg.get("rpo"))},
            "screen_pass": {"plays": agg.get("screen_pass") or 0, "pct": _pct(agg.get("screen_pass"))},
            "no_huddle": {"plays": agg.get("no_huddle") or 0, "pct": _pct(agg.get("no_huddle"))},
            "motion": {"plays": agg.get("motion") or 0, "pct": _pct(agg.get("motion"))},
            "trick_play": {"plays": agg.get("trick_play") or 0, "pct": _pct(agg.get("trick_play"))},
            "avg_defenders_in_box": _avg(agg.get("avg_defenders_in_box")),
            "avg_pass_rushers": _avg(agg.get("avg_pass_rushers")),
            "avg_blitzers": _avg(agg.get("avg_blitzers")),
        },
        "total_plays": total,
    }, identity)


def nfl_td_luck(
    player: str | None = None,
    player_id: str | None = None,
    position: str | None = None,
    team: str | None = None,
    season: int | None = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Touchdown luck — actual vs expected TDs per player-season from the
    player_td_luck table. Negative total_td_luck_score = scored fewer TDs than
    expected (positive-regression / "unlucky" candidate); positive = over-performed.
    Default sort is most unlucky first (total_td_luck_score ASC).
    """
    try:
        limit = max(1, min(int(limit), _MAX_ROWS))
    except (TypeError, ValueError):
        limit = 50

    conditions: list[str] = []
    params: list[Any] = []
    res = _player_filter("player_id = ?", conditions, params,
                         player=player, player_id=player_id, team=team, season=season,
                         position=position)
    if "identity" not in res:
        return res
    identity = res["identity"]
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())
    _team_filter(conditions, params, "team", team)
    if season:
        conditions.append("season = ?")
        params.append(int(season))

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player_id, full_name, position, team, season,
               rec_td, rec_td_exp, rec_td_luck,
               rush_td, rush_td_exp, rush_td_luck,
               total_td_luck_score
        FROM player_td_luck
        WHERE {where}
        ORDER BY total_td_luck_score ASC NULLS LAST
        LIMIT {limit}
    """
    try:
        rows = _execute(sql, params if params else None)
        return _echo_player({"td_luck": _label_players(rows), "count": len(rows)}, identity)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_role_trend(
    player: str | None = None,
    player_id: str | None = None,
    position: str | None = None,
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    min_snap_pct: float | None = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Rolling 3-week role trend — weekly snap %, target/carry/air-yards share with a
    trailing 3-week average and the current-week delta vs that average (from
    player_role_trend). Surfaces players whose usage is rising or fading. Default
    sort is biggest snap-share gain first (snap_pct_delta DESC).
    """
    try:
        limit = max(1, min(int(limit), _MAX_ROWS))
    except (TypeError, ValueError):
        limit = 50

    conditions: list[str] = []
    params: list[Any] = []
    res = _player_filter("player_id = ?", conditions, params,
                         player=player, player_id=player_id, team=team, season=season,
                         position=position)
    if "identity" not in res:
        return res
    identity = res["identity"]
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())
    _team_filter(conditions, params, "team", team)
    if season:
        conditions.append("season = ?")
        params.append(int(season))
    if week:
        conditions.append("week = ?")
        params.append(int(week))
    if min_snap_pct is not None:
        conditions.append("snap_pct >= ?")
        params.append(float(min_snap_pct))

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player_id, full_name, position, team, season, week,
               snap_pct, target_share_pct, carry_share_pct, air_yards_share_pct,
               snap_pct_3wk, target_share_pct_3wk, carry_share_pct_3wk, air_yards_share_pct_3wk,
               snap_pct_delta, target_share_pct_delta, carry_share_pct_delta, air_yards_share_pct_delta
        FROM player_role_trend
        WHERE {where}
        ORDER BY snap_pct_delta DESC NULLS LAST
        LIMIT {limit}
    """
    try:
        rows = _execute(sql, params if params else None)
        return _echo_player({"role_trend": _label_players(rows), "count": len(rows)}, identity)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_separation_opportunity(
    player: str | None = None,
    player_id: str | None = None,
    position: str | None = None,
    team: str | None = None,
    season: int | None = None,
    regression_candidate: bool | None = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Separation-adjusted opportunity — joins Next Gen Stats receiving separation/YAC
    to fantasy opportunity per player-season (from player_separation_opportunity).
    Flags receivers getting open (avg_separation > 2.5) who under-produced
    (fp_diff_per_game < -1.5 and td_luck < -1.0) as regression_candidate. Next Gen
    Stats only goes back to 2016. Default sort is most under-producing first
    (fp_diff_per_game ASC).
    """
    try:
        limit = max(1, min(int(limit), _MAX_ROWS))
    except (TypeError, ValueError):
        limit = 50

    if season is not None and int(season) < 2016:
        return {"error": "Next Gen Stats receiving data is only available from 2016 onward"}

    conditions: list[str] = []
    params: list[Any] = []
    res = _player_filter("player_id = ?", conditions, params,
                         player=player, player_id=player_id, team=team, season=season,
                         position=position)
    if "identity" not in res:
        return res
    identity = res["identity"]
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())
    _team_filter(conditions, params, "team", team)
    if season:
        conditions.append("season = ?")
        params.append(int(season))
    if regression_candidate is not None:
        conditions.append("regression_candidate = ?")
        params.append(bool(regression_candidate))

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player_id, full_name, position, team, season, games,
               fp_diff, fp_diff_per_game, td_luck, catch_luck,
               target_share_pct, air_yards_share_pct,
               avg_separation, avg_yac_above_expectation, catch_percentage,
               regression_candidate
        FROM player_separation_opportunity
        WHERE {where}
        ORDER BY fp_diff_per_game ASC NULLS LAST
        LIMIT {limit}
    """
    try:
        rows = _execute(sql, params if params else None)
        return _echo_player(
            {"separation_opportunity": _label_players(rows), "count": len(rows)}, identity
        )
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_drop_rate(
    player: str | None = None,
    player_id: str | None = None,
    team: str | None = None,
    season: int | None = None,
    min_targets: int | None = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Drop rate — share of catchable targets dropped per receiver-season from FTN
    charting (2022+) joined to plays (player_drop_rate). Also reports contested
    targets and created receptions. Default sort is highest drop rate first
    (drop_rate_pct DESC).
    """
    try:
        limit = max(1, min(int(limit), _MAX_ROWS))
    except (TypeError, ValueError):
        limit = 50

    if season is not None and int(season) < 2022:
        return {"error": "FTN charting (drop) data is only available from 2022 onward"}

    conditions: list[str] = []
    params: list[Any] = []
    res = _player_filter("player_id = ?", conditions, params,
                         player=player, player_id=player_id, team=team, season=season)
    if "identity" not in res:
        return res
    identity = res["identity"]
    _team_filter(conditions, params, "team", team)
    if season:
        conditions.append("season = ?")
        params.append(int(season))
    if min_targets is not None:
        conditions.append("catchable_targets >= ?")
        params.append(int(min_targets))

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player_id, player, team, season,
               (SELECT position FROM players WHERE gsis_id = player_drop_rate.player_id)
                   AS position,
               catchable_targets, drops, drop_rate_pct,
               contested_targets, created_receptions
        FROM player_drop_rate
        WHERE {where}
        ORDER BY drop_rate_pct DESC NULLS LAST
        LIMIT {limit}
    """
    try:
        rows = _execute(sql, params if params else None)
        return _echo_player(
            {"drop_rate": _label_players(rows, name_key="player"), "count": len(rows)}, identity
        )
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_contract_value(
    player: str | None = None,
    player_id: str | None = None,
    position: str | None = None,
    team: str | None = None,
    season: int | None = None,
    min_apy: float | None = None,
    max_apy: float | None = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Contract value efficiency — fantasy points per $M of average per year (APY),
    using each player's active contract joined to seasonal fantasy production
    (player_contract_value). apy and cap_pct reflect the current active contract.
    Default sort is best value first (fp_per_million DESC).
    """
    try:
        limit = max(1, min(int(limit), _MAX_ROWS))
    except (TypeError, ValueError):
        limit = 50

    conditions: list[str] = []
    params: list[Any] = []
    res = _player_filter("player_id = ?", conditions, params,
                         player=player, player_id=player_id, team=team, season=season,
                         position=position)
    if "identity" not in res:
        return res
    identity = res["identity"]
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())
    _team_filter(conditions, params, "team", team)
    if season:
        conditions.append("season = ?")
        params.append(int(season))
    if min_apy is not None:
        conditions.append("apy >= ?")
        params.append(float(min_apy))
    if max_apy is not None:
        conditions.append("apy <= ?")
        params.append(float(max_apy))

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player_id, full_name, position, team, season,
               apy, cap_pct, total_fp, total_fp_exp, fp_per_million
        FROM player_contract_value
        WHERE {where}
        ORDER BY fp_per_million DESC NULLS LAST
        LIMIT {limit}
    """
    try:
        rows = _execute(sql, params if params else None)
        return _echo_player({"contract_value": _label_players(rows), "count": len(rows)}, identity)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_injury_return(
    injury_type: str | None = None,
    position: str | None = None,
    week_post_return: int | None = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Injury return curve — post-return snap-share recovery, expressed as a percent of
    each player's pre-injury baseline, at +1..+8 weeks after returning from an 'Out'
    spell, bucketed by normalized injury type and position (injury_return_curve).
    100 means fully back to baseline usage. Default sort is by weeks since return
    (week_post_return ASC). sample_size shows how many player-spells back each point.
    """
    try:
        limit = max(1, min(int(limit), _MAX_ROWS))
    except (TypeError, ValueError):
        limit = 50

    conditions, params = [], []
    if injury_type:
        conditions.append("injury_type ILIKE ?")
        params.append(f"%{injury_type}%")
    if position:
        conditions.append("position = ?")
        params.append(position.upper())
    if week_post_return is not None:
        conditions.append("week_post_return = ?")
        params.append(int(week_post_return))

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT injury_type, position, week_post_return,
               avg_snap_pct_recovery, median_snap_pct_recovery, sample_size
        FROM injury_return_curve
        WHERE {where}
        ORDER BY week_post_return ASC, injury_type, position
        LIMIT {limit}
    """
    try:
        rows = _execute(sql, params if params else None)
        return {"injury_return": rows, "count": len(rows)}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


__all__ = [
    "nfl_schema", "nfl_status", "nfl_query",
    "nfl_search_plays", "nfl_team_stats", "nfl_player_stats", "nfl_compare",
    "nfl_catalog", "nfl_roster", "nfl_injuries", "nfl_schedule", "nfl_snap_counts",
    "nfl_fantasy_opportunity", "nfl_fantasy_rankings", "nfl_ftn_charting",
    "nfl_td_luck", "nfl_role_trend", "nfl_separation_opportunity",
    "nfl_drop_rate", "nfl_contract_value", "nfl_injury_return",
    "nfl_team_records", "nfl_resolve_player",
]