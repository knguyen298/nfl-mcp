"""
NFL MCP Server tools:
  nfl_schema        → returns column reference so the model knows what to query
  nfl_status        → database health: row counts, loaded seasons, tables
  nfl_query         → one read-only SELECT/EXPLAIN statement, row-capped
  nfl_search_plays  → structured play search
  nfl_team_stats    → pre-aggregated team stats
  nfl_player_lookup → player name to gsis_id
  nfl_player_stats  → player stats by season
  nfl_compare       → side-by-side comparison
"""

import difflib
import logging
import math
import os
import re
import threading
import time
import unicodedata
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Dict

import duckdb

from .database import get_db_connection
from .franchises import (
    RECORDS_FIRST_NFLVERSE_SEASON,
    availability,
    canonical_team,
    era_name,
)
from .schema_pbp import _SCHEMA_SUMMARY, _SCHEMA_CATEGORIES

logger = logging.getLogger("nfl-mcp")


# ── Safety guardrails ──────────────────────────────────────────────────────────

_MAX_ROWS = 500
_QUERY_TIMEOUT_SECONDS = 10


# ── Player identity ────────────────────────────────────────────────────────────

def _normalize_player_name(name: str) -> str:
    """
    Convert full names to nflverse short format. Only nfl_search_plays still uses it,
    because the plays name columns hold short names. Other player tools resolve names
    through the players table (_find_players).
    'Justin Jefferson' -> 'J.Jefferson'
    'J.Jefferson'      -> 'J.Jefferson' (pass through)
    'Jefferson'        -> 'Jefferson'   (last name only, pass through)
    """
    parts = name.strip().split()
    if len(parts) == 1:
        return name
    if '.' in parts[0]:
        return name
    first = parts[0]
    last = ' '.join(parts[1:])
    return f"{first[0]}.{last}"


_GSIS_ID = re.compile(r"\d{2}-\d{7}")
_NAME_DROP = re.compile(r"[.'’`´]")
_NAME_SEP = re.compile(r"[^a-z0-9]+")
_SUFFIX_END = re.compile(r" (?:jr|sr|ii|iii|iv|v)$")
_MAX_CANDIDATES = 25


def _fold(name: str, strip_suffix: bool = False) -> str:
    """Lower-case, accent-free, punctuation-free form of a name.

    Apostrophes and periods vanish ("Ja'Marr" -> "jamarr", "St." -> "st"); hyphens and other
    separators become one space. strip_suffix drops a trailing Jr, Sr, II, III, IV or V.
    """
    text = unicodedata.normalize("NFKD", name or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = _NAME_SEP.sub(" ", _NAME_DROP.sub("", text)).strip()
    return _SUFFIX_END.sub("", text) if strip_suffix else text


def _fold_sql(col: str, strip_suffix: bool = False) -> str:
    """SQL expression that matches _fold(col)."""
    expr = (
        f"trim(regexp_replace(regexp_replace(lower(strip_accents({col})), "
        f"'[.''’`´]', '', 'g'), '[^a-z0-9]+', ' ', 'g'))"
    )
    if strip_suffix:
        expr = f"regexp_replace({expr}, ' (jr|sr|ii|iii|iv|v)$', '')"
    return expr


def _find_players(
    name: str | None = None,
    *,
    gsis_id: str | None = None,
    team: str | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    position: str | None = None,
    limit: int = _MAX_CANDIDATES,
) -> tuple[list[dict], bool]:
    """Players matching a name, narrowed by roster team and season. Returns (candidates, truncated).

    gsis_id looks one player up by id. A name matches the full name, the last name, the short name ("K.Allen"), or the first
    name plus last name, after folding accents, punctuation and a generational suffix.
    Team and season mean "has a rosters row" (in that season when both are given).
    """
    where: list[str] = []
    params: list[Any] = []
    if gsis_id is not None:
        where.append("p.gsis_id = ?")
        params.append(gsis_id)
    if name is not None:
        q = _fold(name, strip_suffix=True)
        if not q:
            return [], False
        first_last = "(p.first_name || ' ' || p.last_name)"
        common_last = "(p.common_first_name || ' ' || p.last_name)"
        where.append(
            "(" + " OR ".join([
                f"{_fold_sql('p.display_name', True)} = ?",
                f"{_fold_sql('p.last_name', True)} = ?",
                f"replace({_fold_sql('p.short_name')}, ' ', '') = ?",
                f"{_fold_sql(first_last, True)} = ?",
                f"{_fold_sql(common_last, True)} = ?",
            ]) + ")"
        )
        params.extend([q, q, q.replace(" ", ""), q, q])
    roster: list[str] = []
    roster_params: list[Any] = []
    if team:
        roster.append("r.team = ?")
        roster_params.append(team.upper())
    if season is not None:
        roster.append("r.season = ?")
        roster_params.append(int(season))
    if season_from is not None:
        roster.append("r.season >= ?")
        roster_params.append(int(season_from))
    if season_to is not None:
        roster.append("r.season <= ?")
        roster_params.append(int(season_to))
    if roster:
        where.append(
            "EXISTS (SELECT 1 FROM rosters r WHERE r.gsis_id = p.gsis_id AND "
            + " AND ".join(roster) + ")"
        )
        params.extend(roster_params)
    if position:
        where.append("upper(p.position) = ?")
        params.append(position.upper())
    sql = f"""
        SELECT p.gsis_id, p.display_name, p.short_name, p.position,
               (SELECT list(t ORDER BY first_year) FROM (
                    SELECT r.team AS t, MIN(r.season) AS first_year FROM rosters r
                    WHERE r.gsis_id = p.gsis_id AND r.team IS NOT NULL GROUP BY r.team)
               ) AS teams,
               p.latest_team,
               COALESCE(p.rookie_season, (SELECT MIN(season) FROM rosters r WHERE r.gsis_id = p.gsis_id)) AS first_season,
               COALESCE(p.last_season, (SELECT MAX(season) FROM rosters r WHERE r.gsis_id = p.gsis_id)) AS last_season,
               p.pfr_id
        FROM players p
        WHERE {' AND '.join(where) if where else '1=1'}
        ORDER BY p.last_season DESC NULLS LAST, p.display_name, p.gsis_id
        LIMIT {int(limit) + 1}
    """
    rows = _execute(sql, params)
    truncated = len(rows) > limit
    for row in rows[:limit]:
        latest = row.pop("latest_team")
        row["teams"] = row["teams"] or ([latest] if latest else [])
    return rows[:limit], truncated


def _ambiguous(candidates: list[dict], truncated: bool, entity: str | None = None) -> dict[str, Any]:
    """Response for a name that matches several players. Not an error: the caller picks an id."""
    result: dict[str, Any] = {
        "ambiguous": True,
        "candidates": candidates,
        "count": len(candidates),
        "hint": "The name matches several players. Call again with player_id set to one gsis_id.",
    }
    if entity is not None:
        result["entity"] = entity
    if truncated:
        result["truncated"] = True
    return result


def _resolve_player(
    player_name: str | None,
    player_id: str | None,
    *,
    team: str | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    entity: str | None = None,
) -> tuple[dict | None, dict[str, Any] | None]:
    """Pick one player for a tool call. Returns (player, early_response).

    player_id wins over player_name, and a name shaped like a gsis id counts as an id.
    A name matching one player gives (player, None). Several give an ambiguous response and
    an unknown id gives an error; both are returned as early_response. No match gives
    (None, None). When team or season leave nobody, a name that matches exactly one player
    overall is still used, so a player with no roster row in that season resolves.
    """
    name = None if player_name is None else str(player_name).strip()
    if not player_id and name and _GSIS_ID.fullmatch(name):
        player_id, name = name, None
    if player_id:
        pid = str(player_id).strip()
        found, _ = _find_players(gsis_id=pid, limit=1)
        if not found:
            return None, {"error": f"Unknown player_id '{pid}'", "code": "UNKNOWN_PLAYER_ID"}
        return found[0], None
    if not name:
        return None, {"error": "player_name or player_id is required", "code": "MISSING_PLAYER"}
    filters = {"team": team, "season": season, "season_from": season_from, "season_to": season_to}
    candidates, truncated = _find_players(name, **filters)
    if not candidates and any(v is not None for v in filters.values()):
        wide, _ = _find_players(name)
        if len(wide) == 1:
            candidates = wide
    if len(candidates) == 1:
        return candidates[0], None
    if candidates:
        return None, _ambiguous(candidates, truncated, entity)
    return None, None


def nfl_player_lookup(
    name: str,
    team: str | None = None,
    season: int | None = None,
    position: str | None = None,
) -> dict[str, Any]:
    """Find players by name and return their gsis_id, so other player tools can take player_id."""
    if not name or not str(name).strip():
        return {"error": "name is required", "code": "MISSING_NAME"}
    try:
        candidates, truncated = _find_players(
            str(name), team=team, season=season, position=position, limit=50)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.exception("Tool error")
        return {"error": str(e)}
    result: dict[str, Any] = {"candidates": candidates, "count": len(candidates)}
    if truncated:
        result["truncated"] = True
        result["hint"] = "More players match. Add team, season or position, or use a fuller name."
    return result


def _player_filter(
    player: str | None,
    player_id: str | None,
    name_col: str,
    *,
    team: str | None = None,
    season: int | None = None,
) -> tuple[str | None, list[Any], dict | None, dict[str, Any] | None]:
    """Player condition for a derived table that has a player_id column.

    Returns (condition, params, player, early_response). A name resolves through players:
    one match filters on player_id, several give the ambiguous response as early_response,
    none falls back to a substring match on name_col (the table's own name column).
    """
    if not player and not player_id:
        return None, [], None, None
    try:
        found, early = _resolve_player(player, player_id, team=team, season=season)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.exception("Tool error")
        return None, [], None, {"error": str(e)}
    if early is not None:
        return None, [], None, early
    if found is None:
        return f"{name_col} ILIKE ?", [f"%{player}%"], None, None
    return "player_id = ?", [found["gsis_id"]], found, None


def _with_player(result: dict[str, Any], player: dict | None) -> dict[str, Any]:
    """Add player_id and player_name to a response when the filter resolved to one player."""
    if player is not None:
        result["player_id"] = player["gsis_id"]
        result["player_name"] = player["display_name"]
    return result


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


def _unique_names(names: list[str]) -> list[str]:
    """Make column names unique the way DuckDB does: a, a_1, a_2, ..."""
    seen: set[str] = set()
    out = []
    for name in names:
        candidate, n = name, 0
        while candidate in seen:
            n += 1
            candidate = f"{name}_{n}"
        seen.add(candidate)
        out.append(candidate)
    return out


def _fetch_capped(sql: str, limit: int) -> tuple[list[dict], list[dict]]:
    """Run `sql` as written and fetch at most `limit` rows, with a timeout.

    Returns (columns, rows) where columns is [{"name", "type"}]. Duplicate
    column names are made unique (a, a_1, ...) for both columns and row keys.
    """
    timeout = _query_timeout()
    thread_conn: list[Any] = [None]
    cancelled = threading.Event()

    def _run():
        with get_db_connection() as conn:
            thread_conn[0] = conn
            if cancelled.is_set():
                return None
            cursor = conn.execute(sql)
            desc = cursor.description or []
            names = _unique_names([d[0] for d in desc])
            columns = [{"name": n, "type": str(d[1])} for n, d in zip(names, desc)]
            return columns, [dict(zip(names, r)) for r in cursor.fetchmany(limit)]

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_run)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeout:
        cancelled.set()
        for _ in range(5):
            conn = thread_conn[0]
            if conn is not None:
                try:
                    conn.interrupt()
                except (duckdb.Error, RuntimeError):
                    logger.warning("Failed to interrupt timed out DuckDB query", exc_info=True)
            if wait([future], timeout=0.2).done:
                break
            time.sleep(0.05)
        raise TimeoutError(f"Query exceeded {timeout:g} second timeout") from None
    finally:
        pool.shutdown(wait=False)


# Whitespace or comments, and an EXPLAIN option list such as (ANALYZE, FORMAT JSON).
# The option list never matches a parenthesized query: it cannot start with a
# query keyword or a nested paren.
_GAP = r"(?:\s|--[^\n]*(?:\n|\Z)|/\*.*?\*/)*"
_EXPLAIN_OPTION = r"[A-Za-z_]\w*(?:\s+[^,()]+)?"
_EXPLAIN_PREFIX = re.compile(
    rf"\A{_GAP}EXPLAIN\b(?:{_GAP}ANALYZE\b)?"
    rf"(?:{_GAP}\(\s*(?!(?:SELECT|WITH|FROM|VALUES|TABLE|PIVOT|UNPIVOT|DESCRIBE|SUMMARIZE|SHOW)\b)"
    rf"{_EXPLAIN_OPTION}(?:\s*,\s*{_EXPLAIN_OPTION})*\s*\))?",
    re.IGNORECASE | re.DOTALL,
)


def _statement_gate(sql: str) -> dict[str, Any] | None:
    """Return an error dict unless `sql` is one SELECT, or EXPLAIN [ANALYZE] of one SELECT."""
    try:
        statements = duckdb.extract_statements(sql)
    except (duckdb.Error, TypeError) as e:
        return {"error": f"Could not parse SQL: {e}"}
    if len(statements) != 1:
        return {"error": "Send exactly one statement per call."}
    stype = statements[0].type
    if stype == duckdb.StatementType.EXPLAIN:
        # EXPLAIN ANALYZE executes what it wraps, so gate the inner statement too.
        match = _EXPLAIN_PREFIX.match(sql)
        if not match:
            return {"error": "Could not read the EXPLAIN statement.", "statement_type": stype.name}
        try:
            inner = duckdb.extract_statements(sql[match.end():])
        except (duckdb.Error, TypeError) as e:
            return {"error": f"Could not parse SQL after EXPLAIN: {e}", "statement_type": stype.name}
        if len(inner) != 1:
            return {"error": "Send exactly one statement per call.", "statement_type": stype.name}
        stype = inner[0].type
    if stype != duckdb.StatementType.SELECT:
        return {
            "error": f"Only SELECT and EXPLAIN of a SELECT are allowed; got {stype.name}.",
            "statement_type": stype.name,
        }
    return None


def nfl_query(sql: str, max_rows: int = 100) -> Dict[str, Any]:
    """
    Execute one read-only SQL statement against the nflread database.

    Rules:
      - Exactly one statement, parsed by DuckDB (a trailing ';' is fine)
      - Only SELECT, or EXPLAIN [ANALYZE] of a SELECT; anything else is
        rejected with its statement_type
      - max_rows is an int from 1 to 500; larger values are clamped to 500
      - Query runs as written; max_rows + 1 rows are fetched to set `truncated`
      - 10-second timeout (NFL_MCP_QUERY_TIMEOUT_SECONDS overrides)

    Returns columns ([{name, type}]), rows, row_count and truncated.
    Unknown column/table failures also return `code`, the offending name,
    `did_you_mean` (ranked against the live schema) and a `hint`.
    """
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        return {"error": "max_rows must be an integer from 1 to 500."}
    max_rows = min(max_rows, _MAX_ROWS)

    if not isinstance(sql, str):
        return {"error": "sql must be a string"}
    rejected = _statement_gate(sql)
    if rejected:
        return rejected

    try:
        columns, rows = _fetch_capped(sql, max_rows + 1)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e), **_sql_error_detail(e)}

    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    return {
        "columns":   columns,
        "rows":      rows,
        "row_count": len(rows),
        "truncated": truncated,
    }


def nfl_search_plays(
    team: str | None = None,
    opponent: str | None = None,
    player: str | None = None,
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
    conditions = []
    params: list[Any] = []
    if team:
        conditions.append("posteam = ?")
        params.append(team)
    if opponent:
        conditions.append("defteam = ?")
        params.append(opponent)
    if player:
        player = _normalize_player_name(player)
        conditions.append(
            "(passer_player_name ILIKE ? "
            "OR rusher_player_name ILIKE ? "
            "OR receiver_player_name ILIKE ?)"
        )
        pattern = f"%{player}%"
        params.extend([pattern, pattern, pattern])
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
        f"receiver_player_name, touchdown, interception "
        f"FROM plays WHERE {where} "
        f"ORDER BY ABS(epa) DESC LIMIT {max_rows}"
    )

    try:
        rows = _execute(sql, params)
        return {"rows": rows, "row_count": len(rows)}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def _team_stats_filter(
    season: int | None,
    season_from: int | None,
    season_to: int | None,
    season_type: str | None,
) -> tuple[str, list[Any]]:
    """WHERE fragment and params for the derived team tables ('ALL' keeps both season types)."""
    clause = ""
    params: list[Any] = []
    if season:
        clause += " AND season_year = ?"
        params.append(int(season))
    if season_from:
        clause += " AND season_year >= ?"
        params.append(int(season_from))
    if season_to:
        clause += " AND season_year <= ?"
        params.append(int(season_to))
    season_type = (season_type or "REG").upper()
    if season_type != "ALL":
        clause += " AND season_type = ?"
        params.append(season_type)
    return clause, params


_STALE_TEAM_TABLES = (
    "Derived team tables predate season_type; the season_type filter was not applied "
    "and rows blend REG and POST. Run 'nfl-mcp rebuild'."
)


class _StaleDerivedTables(Exception):
    """A season_type other than REG was asked of derived tables built before season_type existed."""


_STALE_ERROR = {
    "error": "Derived team tables are stale; run `nfl-mcp rebuild`",
    "code": "STALE_DERIVED_TABLES",
}


def _team_rows(
    table: str,
    team: str,
    filters: tuple[int | None, int | None, int | None, str | None],
    tail: str = "",
) -> tuple[list[dict], bool]:
    """Rows of one derived team table, and whether the table lacks season_type (not rebuilt yet).

    On a table without season_type only the default REG falls back to unfiltered rows
    (flagged by the caller); any other season_type raises _StaleDerivedTables.
    """
    clause, params = _team_stats_filter(*filters)
    try:
        sql = (f"SELECT * FROM {table} WHERE team = ?{clause} "
               f"ORDER BY season_year, season_type DESC{tail}")
        return _execute(sql, [team, *params]), False
    except duckdb.BinderException as e:
        if "season_type" not in str(e):
            raise
        if (filters[3] or "REG").upper() != "REG":
            raise _StaleDerivedTables() from e
    clause, params = _team_stats_filter(*filters[:3], "ALL")
    sql = f"SELECT * FROM {table} WHERE team = ?{clause} ORDER BY season_year{tail}"
    return _execute(sql, [team, *params]), True


def nfl_team_stats(
    team: str,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    side: str = "both",
    season_type: str = "REG",
) -> Dict[str, Any]:
    """Get pre-aggregated team stats (offense, defense, or both) for REG, POST, or ALL games."""
    team = team.upper()
    results = {}
    filters = (season, season_from, season_to, season_type)

    try:
        stale = False
        if side in ("offense", "both"):
            results["offense"], old = _team_rows("team_offense_stats", team, filters)
            stale = stale or old

        if side in ("defense", "both"):
            results["defense"], old = _team_rows("team_defense_stats", team, filters)
            stale = stale or old

        if side in ("situational", "both"):
            results["situational"], old = _team_rows("situational_stats", team, filters, ", situation")
            stale = stale or old

        if stale:
            results["warning"] = _STALE_TEAM_TABLES
        return results
    except _StaleDerivedTables:
        return dict(_STALE_ERROR)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


_NOT_TWO_POINT = "COALESCE(two_point_attempt, 0) = 0"
_NOT_NO_PLAY = "COALESCE(play_type, '') <> 'no_play'"
_NOT_SACK = "COALESCE(sack, 0) = 0"
# Row sets for one player (me.id). Each is the definition shown to users in _STAT_DEFINITIONS.
_ATTEMPT = (f"passer_player_id = me.id AND pass_attempt = 1 AND {_NOT_SACK} "
            f"AND {_NOT_TWO_POINT} AND {_NOT_NO_PLAY}")
_SACKED = f"passer_player_id = me.id AND sack = 1 AND {_NOT_TWO_POINT} AND {_NOT_NO_PLAY}"
_DROPBACK = (f"qb_dropback = 1 AND {_NOT_TWO_POINT} AND {_NOT_NO_PLAY} "
             "AND (passer_player_id = me.id OR (rusher_player_id = me.id AND qb_scramble = 1))")
_CARRY = f"rusher_player_id = me.id AND rush_attempt = 1 AND {_NOT_TWO_POINT}"
_TARGET = (f"receiver_player_id = me.id AND pass_attempt = 1 AND {_NOT_SACK} "
           f"AND {_NOT_TWO_POINT}")

_STAT_DEFINITIONS = {
    "attempts": "pass_attempt=1 AND sack=0 AND two_point_attempt=0 AND play_type<>'no_play', "
                "passer_player_id = the player. Spikes count as attempts.",
    "completions": "attempts with complete_pass=1.",
    "yards": "passing: passing_yards on attempts; rushing: rushing_yards on carries; "
             "receiving: receiving_yards on targets.",
    "sacks": "sack=1 with the player as passer_player_id, excluding two_point_attempt=1 and no_play.",
    "dropbacks": "qb_dropback=1 AND two_point_attempt=0 AND play_type<>'no_play', where the player is "
                 "passer_player_id or is rusher_player_id on a qb_scramble=1 play "
                 "(attempts excluding spikes, plus sacks, plus scrambles).",
    "epa_per_attempt": "average epa over attempts.",
    "epa_per_dropback": "average epa over dropbacks.",
    "carries": "rush_attempt=1 AND two_point_attempt=0, rusher_player_id = the player. "
               "Kneels and scrambles count; no_play is not excluded.",
    "scrambles": "carries with qb_scramble=1.",
    "yards_per_carry": "rushing yards / carries.",
    "targets": "pass_attempt=1 AND sack=0 AND two_point_attempt=0, receiver_player_id = the player.",
    "receptions": "targets with complete_pass=1.",
    "touchdowns": "passing: pass_touchdown=1 on attempts; rushing: rush_touchdown=1 on carries; "
                  "receiving: pass_touchdown=1 on targets.",
}


def _sql_count(cond: str, extra: str | None = None) -> str:
    """SQL: number of rows in a row set (optionally narrowed by one more condition)."""
    return f"COUNT(*) FILTER (WHERE {cond}{f' AND {extra}' if extra else ''})"


def _sql_sum(expr: str, cond: str) -> str:
    return f"SUM({expr}) FILTER (WHERE {cond})"


def _sql_avg(expr: str, cond: str, digits: int = 3) -> str:
    return f"ROUND(AVG({expr}) FILTER (WHERE {cond}), {digits})"


_PLAYER_STATS_SQL = f"""
    SELECT season, season_type,
        {_sql_count(_ATTEMPT)} AS attempts,
        {_sql_count(_ATTEMPT, 'complete_pass = 1')} AS completions,
        {_sql_sum('passing_yards', _ATTEMPT)} AS pass_yards,
        {_sql_count(_ATTEMPT, 'pass_touchdown = 1')} AS pass_tds,
        {_sql_count(_ATTEMPT, 'interception = 1')} AS interceptions,
        {_sql_count(_SACKED)} AS sacks,
        {_sql_count(_DROPBACK)} AS dropbacks,
        {_sql_avg('epa', _ATTEMPT)} AS epa_per_attempt,
        {_sql_avg('epa', _DROPBACK)} AS epa_per_dropback,
        {_sql_avg('cpoe', _ATTEMPT)} AS avg_cpoe,
        {_sql_sum('air_yards', _ATTEMPT)} AS pass_air_yards,
        {_sql_count(_CARRY)} AS carries,
        {_sql_sum('rushing_yards', _CARRY)} AS rush_yards,
        {_sql_count(_CARRY, 'rush_touchdown = 1')} AS rush_tds,
        {_sql_count(_CARRY, 'qb_scramble = 1')} AS scrambles,
        {_sql_count(_CARRY, 'fumble_lost = 1')} AS fumbles_lost,
        {_sql_count(_CARRY, 'yards_gained >= 10')} AS explosive_runs,
        {_sql_avg('epa', _CARRY)} AS epa_per_carry,
        {_sql_count(_TARGET)} AS targets,
        {_sql_count(_TARGET, 'complete_pass = 1')} AS receptions,
        {_sql_sum('receiving_yards', _TARGET)} AS rec_yards,
        {_sql_count(_TARGET, 'pass_touchdown = 1')} AS rec_tds,
        {_sql_avg('yards_after_catch', _TARGET, 1)} AS avg_yac,
        {_sql_count(_TARGET, 'yards_gained >= 20')} AS explosive_plays,
        {_sql_avg('epa', _TARGET)} AS epa_per_target
    FROM plays, (SELECT CAST(? AS VARCHAR) AS id) AS me
    WHERE me.id IN (passer_player_id, rusher_player_id, receiver_player_id)
"""


def _season_row(row: dict[str, Any]) -> dict[str, Any]:
    """Nest one aggregate row into passing, rushing and receiving blocks."""
    def n(key: str) -> int:
        return int(row[key] or 0)

    attempts, carries = n("attempts"), n("carries")
    return {
        "season": row["season"],
        "season_type": row["season_type"],
        "passing": {
            "attempts": attempts,
            "completions": n("completions"),
            "yards": n("pass_yards"),
            "touchdowns": n("pass_tds"),
            "interceptions": n("interceptions"),
            "sacks": n("sacks"),
            "dropbacks": n("dropbacks"),
            "epa_per_attempt": row["epa_per_attempt"],
            "epa_per_dropback": row["epa_per_dropback"],
            "comp_pct": round(100.0 * n("completions") / attempts, 1) if attempts else None,
            "avg_cpoe": row["avg_cpoe"],
            "air_yards": n("pass_air_yards"),
        },
        "rushing": {
            "carries": carries,
            "yards": n("rush_yards"),
            "touchdowns": n("rush_tds"),
            "scrambles": n("scrambles"),
            "yards_per_carry": round(n("rush_yards") / carries, 2) if carries else None,
            "fumbles_lost": n("fumbles_lost"),
            "explosive_runs": n("explosive_runs"),
            "epa_per_carry": row["epa_per_carry"],
        },
        "receiving": {
            "targets": n("targets"),
            "receptions": n("receptions"),
            "yards": n("rec_yards"),
            "touchdowns": n("rec_tds"),
            "avg_yac": row["avg_yac"],
            "explosive_plays": n("explosive_plays"),
            "epa_per_target": row["epa_per_target"],
        },
    }


def _player_seasons(
    gsis_id: str,
    season: int | None,
    season_from: int | None,
    season_to: int | None,
    season_type: str | None,
) -> list[dict[str, Any]]:
    """Per season and season type passing, rushing and receiving stats for one gsis_id."""
    sql = _PLAYER_STATS_SQL
    params: list[Any] = [gsis_id]
    if season is not None:
        sql += " AND season = ?"
        params.append(int(season))
    if season_from is not None:
        sql += " AND season >= ?"
        params.append(int(season_from))
    if season_to is not None:
        sql += " AND season <= ?"
        params.append(int(season_to))
    if season_type:
        sql += " AND season_type = ?"
        params.append(str(season_type).upper())
    sql += " GROUP BY season, season_type ORDER BY season, season_type"
    return [_season_row(row) for row in _execute(sql, params)]


_STAT_BLOCKS = ("passing", "rushing", "receiving")


def nfl_player_stats(
    player_name: str | None = None,
    player_id: str | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    season_type: str | None = None,
    stat_type: str | None = None,
) -> dict[str, Any]:
    """Passing, rushing and receiving stats per season and season type for one player.

    A name that matches several players returns ambiguous candidates; pass player_id then.
    """
    if stat_type is not None and stat_type not in _STAT_BLOCKS:
        return {"error": f"Unknown stat_type: {stat_type}. Use 'passing', 'rushing', or 'receiving'."}
    try:
        player, early = _resolve_player(
            player_name, player_id, season=season, season_from=season_from, season_to=season_to)
        if early is not None:
            return early
        if player is None:
            return {"player_id": None, "player_name": player_name, "seasons": [],
                    "reason": f"No player matches '{player_name}'. Try nfl_player_lookup."}
        seasons = _player_seasons(player["gsis_id"], season, season_from, season_to, season_type)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}
    if stat_type:
        seasons = [{k: v for k, v in s.items() if k not in _STAT_BLOCKS or k == stat_type}
                   for s in seasons]
    return {"player_id": player["gsis_id"], "player_name": player["display_name"],
            "seasons": seasons, "definitions": dict(_STAT_DEFINITIONS)}


def _compare_teams(
    entity1: str,
    entity2: str,
    season: int | None,
    season_from: int | None,
    season_to: int | None,
    season_type: str | None,
) -> dict[str, Any]:
    """Team side of nfl_compare: offense and defense rows for each team under one filter."""
    filters = (season, season_from, season_to, season_type)
    shown = (season_type or "REG").upper()
    result: dict[str, Any] = {}
    for label in (entity1, entity2):
        team = label.upper()
        offense, old_off = _team_rows("team_offense_stats", team, filters)
        defense, old_def = _team_rows("team_defense_stats", team, filters)
        entry: dict[str, Any] = {"offense": offense, "defense": defense}
        if old_off or old_def:
            entry["warning"] = _STALE_TEAM_TABLES
        if not offense and not defense:
            scope = f" in {season}" if season else ""
            entry["reason"] = f"No {shown} team stats for {team}{scope} with the filters given"
        result[label] = entry
    return result


def _compare_players(
    entity1: str,
    entity2: str,
    season: int | None,
    season_from: int | None,
    season_to: int | None,
    season_type: str | None,
) -> dict[str, Any]:
    """Player side of nfl_compare: each entity is a name or a gsis id, resolved to one player."""
    entries = []
    for entity in (entity1, entity2):
        player, early = _resolve_player(
            entity, None, season=season, season_from=season_from, season_to=season_to,
            entity=entity)
        if early is not None:
            return early
        if player is None:
            entries.append({"player_id": None, "player_name": entity, "seasons": [],
                            "reason": f"No player matches '{entity}'. Try nfl_player_lookup."})
            continue
        entries.append({
            "player_id": player["gsis_id"],
            "player_name": player["display_name"],
            "seasons": _player_seasons(
                player["gsis_id"], season, season_from, season_to, season_type),
        })
    return {"players": entries, "definitions": dict(_STAT_DEFINITIONS)}


def nfl_compare(
    entity1: str,
    entity2: str,
    compare_type: str = "team",
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    season_type: str | None = None,
) -> Dict[str, Any]:
    """Side-by-side comparison of two teams or players.

    Team mode reads the derived team tables for season_type (REG by default, POST, or ALL).
    """
    try:
        if compare_type == "team":
            return _compare_teams(entity1, entity2, season, season_from, season_to, season_type)
        elif compare_type == "player":
            return _compare_players(entity1, entity2, season, season_from, season_to, season_type)
        else:
            return {"error": "compare_type must be 'team' or 'player'"}

    except _StaleDerivedTables:
        return dict(_STALE_ERROR)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def _query_timeout() -> float:
    """Seconds a query may run. NFL_MCP_QUERY_TIMEOUT_SECONDS is read on every call."""
    raw = os.getenv("NFL_MCP_QUERY_TIMEOUT_SECONDS")
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0
        if math.isfinite(value) and value > 0:
            return value
        logger.warning("Ignoring invalid NFL_MCP_QUERY_TIMEOUT_SECONDS=%r", raw[:40])
    return _QUERY_TIMEOUT_SECONDS


def _execute(sql: str, params: Sequence[Any] | None = None) -> list[dict]:
    """Execute SQL on DuckDB with timeout and cancellation.

    Runs in a worker thread with a dedicated connection, and interrupts
    the active query if it exceeds the timeout.
    """
    timeout = _query_timeout()
    result = [None]
    error = [None]
    thread_conn = [None]
    cancelled = threading.Event()

    def _run():
        try:
            with get_db_connection() as conn:
                thread_conn[0] = conn
                if cancelled.is_set():
                    return
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
    t.join(timeout=timeout)
    if t.is_alive():
        # Set first: a thread still opening its connection sees the flag and
        # never starts the query. Otherwise it has a connection to interrupt.
        cancelled.set()
        for _ in range(4):
            conn = thread_conn[0]
            if conn is not None:
                try:
                    conn.interrupt()
                except Exception:
                    logger.warning("Failed to interrupt timed out DuckDB query", exc_info=True)
            t.join(timeout=0.5)
            if not t.is_alive():
                break
        else:
            logger.warning("Timed out DuckDB query thread is still running after interrupt")
        raise TimeoutError(f"Query exceeded {timeout:g} second timeout")
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
    report_status: str | None = None,
) -> Dict[str, Any]:
    """
    Look up injury report data. Filter by team, season, week, player name, or
    report status (e.g. 'Out', 'Questionable', 'Doubtful', 'Full Participation').
    """
    conditions, params = [], []
    if team:
        conditions.append("team = ?")
        params.append(team.upper())
    if season:
        conditions.append("season = ?")
        params.append(int(season))
    if week:
        conditions.append("week = ?")
        params.append(int(week))
    if player:
        conditions.append("full_name ILIKE ?")
        params.append(f"%{player}%")
    if report_status:
        conditions.append("report_status ILIKE ?")
        params.append(f"%{report_status}%")

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT full_name, position, team, season, week,
               report_primary_injury, report_secondary_injury, report_status,
               practice_primary_injury, practice_secondary_injury, practice_status
        FROM injuries
        WHERE {where}
        ORDER BY season, week, team, full_name
        LIMIT 500
    """
    try:
        rows = _execute(sql, params if params else None)
        return {"injuries": rows, "count": len(rows)}
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
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    position: str | None = None,
    player_id: str | None = None,
) -> Dict[str, Any]:
    """
    Look up snap count data — how many snaps a player or team unit played.
    Returns offense, defense, and special teams snap counts and percentages.
    player matches the full names stored in snap_counts ('Travis Kelce' or 'Kelce'),
    ignoring case, accents and punctuation. player_id (gsis_id) maps through
    players.pfr_id to snap_counts.pfr_player_id.
    """
    conditions, params = [], []
    try:
        if player_id:
            found, _ = _find_players(gsis_id=str(player_id).strip(), limit=1)
            if not found:
                return {"error": f"Unknown player_id '{player_id}'", "code": "UNKNOWN_PLAYER_ID"}
            if not found[0]["pfr_id"]:
                return {"snap_counts": [], "count": 0, "player_id": found[0]["gsis_id"],
                        "reason": f"No PFR id on file for {found[0]['display_name']}, "
                                  "so snap counts cannot be matched by id."}
            conditions.append("pfr_player_id = ?")
            params.append(found[0]["pfr_id"])
        if player:
            query = _fold(str(player), strip_suffix=True)
            conditions.append(f"{_fold_sql('player')} LIKE ?")
            params.append(f"%{query}%")
        if team:
            conditions.append("team = ?")
            params.append(team.upper())
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
                   st_snaps, st_pct
            FROM snap_counts
            WHERE {where}
            ORDER BY season, week, team, offense_snaps DESC NULLS LAST
            LIMIT 500
        """
        rows = _execute(sql, params if params else None)
        return {"snap_counts": rows, "count": len(rows)}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_fantasy_opportunity(
    player: str | None = None,
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    position: str | None = None,
    player_id: str | None = None,
) -> Dict[str, Any]:
    """
    Look up fantasy football opportunity data — target share, air yards share,
    carry share, and opportunity scores per player per week. Available 2006–present.
    """
    conditions, params = [], []
    cond, cparams, resolved, early = _player_filter(
        player, player_id, "full_name", team=team, season=season)
    if early is not None:
        return early
    if cond:
        conditions.append(cond)
        params.extend(cparams)
    if team:
        conditions.append("posteam = ?")
        params.append(team.upper())
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
        return _with_player({"fantasy_opportunity": rows, "count": len(rows)}, resolved)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_fantasy_rankings(
    player: str | None = None,
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

    conditions, params = [], []
    if scope == "draft":
        table = "ff_rankings_draft"
        player_col, set_col = "player", "page_type"
        select = (
            "player, pos AS position, team, ecr, sd, best, worst, bye, "
            "page_type AS ranking_set, ecr_type, scrape_date"
        )
    else:
        table = "ff_rankings_week"
        player_col, set_col = "player_name", "page"
        select = (
            "player_name AS player, pos AS position, team, rank, ecr, sd, best, worst, "
            "pos_rank, player_opponent AS opponent, player_bye_week AS bye, "
            "start_sit_grade, note, page AS ranking_set, scrape_date"
        )

    if player:
        # The ranking tables have no gsis id, so match on names with apostrophes, periods
        # and accents folded away on both sides ("Ja'Marr Chase" finds "JaMarr Chase").
        conditions.append(f"{_fold_sql(player_col)} LIKE ?")
        params.append(f"%{_fold(str(player), strip_suffix=True)}%")
    if position:
        conditions.append("pos ILIKE ?")
        params.append(position.upper())
    if team:
        conditions.append("team = ?")
        params.append(team.upper())
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
        return {"fantasy_rankings": rows, "scope": scope, "count": len(rows)}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_ftn_charting(
    team: str | None = None,
    opponent: str | None = None,
    player: str | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    week: int | None = None,
    season_type: str | None = None,
    player_id: str | None = None,
) -> Dict[str, Any]:
    """
    Aggregated FTN manual charting tendencies (2022–present) over offensive scrimmage
    plays (pass + run) for a filtered set: play-action, RPO, screen, no-huddle, motion,
    and trick-play usage rates, plus average defenders in the box (per scrimmage play)
    and pass rushers / blitzers (per dropback). Joins ftn_charting to plays for
    team/player/season context. Offensive perspective (team = posteam).
    player (a name) or player_id (gsis_id) selects one player, matched on the passer,
    rusher and receiver id columns; a name matching several players is ambiguous.
    """
    conditions = ["f.nflverse_game_id = p.game_id", "f.nflverse_play_id = p.play_id",
                  "p.play_type IN ('pass', 'run')"]
    params: list[Any] = []
    if team:
        conditions.append("p.posteam = ?")
        params.append(team.upper())
    if opponent:
        conditions.append("p.defteam = ?")
        params.append(opponent.upper())
    resolved = None
    if player or player_id:
        try:
            resolved, early = _resolve_player(
                player, player_id, team=team, season=season,
                season_from=season_from, season_to=season_to)
        except (duckdb.Error, ValueError, TimeoutError) as e:
            logger.exception("Tool error")
            return {"error": str(e)}
        if early is not None:
            return early
        if resolved is None:
            return {"charting": {}, "total_plays": 0,
                    "reason": f"No player matches '{player}'. Try nfl_player_lookup."}
        conditions.append(
            "(p.passer_player_id = ? OR p.rusher_player_id = ? OR p.receiver_player_id = ?)")
        params.extend([resolved["gsis_id"]] * 3)
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
        return _with_player({"charting": {}, "total_plays": 0}, resolved)

    def _pct(n: Any) -> float:
        return round(100.0 * (n or 0) / total, 1)

    def _avg(v: Any) -> float | None:
        return round(v, 2) if v is not None else None

    return _with_player({
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
    }, resolved)


def nfl_td_luck(
    player: str | None = None,
    position: str | None = None,
    team: str | None = None,
    season: int | None = None,
    limit: int = 50,
    player_id: str | None = None,
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

    conditions, params = [], []
    cond, cparams, resolved, early = _player_filter(
        player, player_id, "full_name", team=team, season=season)
    if early is not None:
        return early
    if cond:
        conditions.append(cond)
        params.extend(cparams)
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())
    if team:
        conditions.append("team = ?")
        params.append(team.upper())
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
        return _with_player({"td_luck": rows, "count": len(rows)}, resolved)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_role_trend(
    player: str | None = None,
    position: str | None = None,
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    min_snap_pct: float | None = None,
    limit: int = 50,
    player_id: str | None = None,
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

    conditions, params = [], []
    cond, cparams, resolved, early = _player_filter(
        player, player_id, "full_name", team=team, season=season)
    if early is not None:
        return early
    if cond:
        conditions.append(cond)
        params.extend(cparams)
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())
    if team:
        conditions.append("team = ?")
        params.append(team.upper())
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
        return _with_player({"role_trend": rows, "count": len(rows)}, resolved)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_separation_opportunity(
    player: str | None = None,
    position: str | None = None,
    team: str | None = None,
    season: int | None = None,
    regression_candidate: bool | None = None,
    limit: int = 50,
    player_id: str | None = None,
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

    conditions, params = [], []
    cond, cparams, resolved, early = _player_filter(
        player, player_id, "full_name", team=team, season=season)
    if early is not None:
        return early
    if cond:
        conditions.append(cond)
        params.extend(cparams)
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())
    if team:
        conditions.append("team = ?")
        params.append(team.upper())
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
        return _with_player({"separation_opportunity": rows, "count": len(rows)}, resolved)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_drop_rate(
    player: str | None = None,
    team: str | None = None,
    season: int | None = None,
    min_targets: int | None = None,
    limit: int = 50,
    player_id: str | None = None,
) -> Dict[str, Any]:
    """
    Drop rate — share of catchable targets dropped per receiver-season from FTN
    charting (2022+) joined to plays (player_drop_rate). Also reports contested
    targets and created receptions. The player column holds nflverse short names
    ('J.Jefferson'); a full name in `player` resolves through players and filters by id.
    Default sort is highest drop rate first (drop_rate_pct DESC).
    """
    try:
        limit = max(1, min(int(limit), _MAX_ROWS))
    except (TypeError, ValueError):
        limit = 50

    if season is not None and int(season) < 2022:
        return {"error": "FTN charting (drop) data is only available from 2022 onward"}

    conditions, params = [], []
    cond, cparams, resolved, early = _player_filter(
        player, player_id, "player", team=team, season=season)
    if early is not None:
        return early
    if cond:
        conditions.append(cond)
        params.extend(cparams)
    if team:
        conditions.append("team = ?")
        params.append(team.upper())
    if season:
        conditions.append("season = ?")
        params.append(int(season))
    if min_targets is not None:
        conditions.append("catchable_targets >= ?")
        params.append(int(min_targets))

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player_id, player, team, season,
               catchable_targets, drops, drop_rate_pct,
               contested_targets, created_receptions
        FROM player_drop_rate
        WHERE {where}
        ORDER BY drop_rate_pct DESC NULLS LAST
        LIMIT {limit}
    """
    try:
        rows = _execute(sql, params if params else None)
        return _with_player({"drop_rate": rows, "count": len(rows)}, resolved)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_contract_value(
    player: str | None = None,
    position: str | None = None,
    team: str | None = None,
    season: int | None = None,
    min_apy: float | None = None,
    max_apy: float | None = None,
    limit: int = 50,
    player_id: str | None = None,
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

    conditions, params = [], []
    cond, cparams, resolved, early = _player_filter(
        player, player_id, "full_name", team=team, season=season)
    if early is not None:
        return early
    if cond:
        conditions.append(cond)
        params.extend(cparams)
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())
    if team:
        conditions.append("team = ?")
        params.append(team.upper())
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
        return _with_player({"contract_value": rows, "count": len(rows)}, resolved)
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
    "nfl_search_plays", "nfl_team_stats", "nfl_player_stats", "nfl_player_lookup", "nfl_compare",
    "nfl_catalog", "nfl_roster", "nfl_injuries", "nfl_schedule", "nfl_snap_counts",
    "nfl_fantasy_opportunity", "nfl_fantasy_rankings", "nfl_ftn_charting",
    "nfl_td_luck", "nfl_role_trend", "nfl_separation_opportunity",
    "nfl_drop_rate", "nfl_contract_value", "nfl_injury_return",
    "nfl_team_records",
]