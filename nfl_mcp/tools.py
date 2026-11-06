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

import datetime as dt
import difflib
import functools
import json
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

from .config import get_duckdb_path
from .database import get_db_connection
from .discovery import DATA_NOTES, DERIVED_TABLES, relationships_for, table_info
from .franchises import (
    RECORDS_FIRST_NFLVERSE_SEASON,
    availability,
    era_name,
    team_codes,
)
from .schema_pbp import _SCHEMA_SUMMARY, _SCHEMA_CATEGORIES

logger = logging.getLogger("nfl-mcp")


# ── Safety guardrails ──────────────────────────────────────────────────────────

_MAX_ROWS = 500
_QUERY_TIMEOUT_SECONDS = 10


# ── Argument handling ──────────────────────────────────────────────────────────
# Shared by every tool that filters on a team, a season, a week, a season type or a
# number. A bad value raises _BadArgument; @_checked turns it into an error dict.

class _BadArgument(Exception):
    """An argument failed validation. as_error() is the dict the tool returns."""

    def __init__(self, message: str, code: str = "INVALID_ARGUMENT"):
        super().__init__(message)
        self.code = code

    def as_error(self) -> dict[str, Any]:
        return {"error": str(self), "code": self.code}


def _checked(fn):
    """Decorator for a tool: a _BadArgument raised inside becomes {"error", "code"}."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _BadArgument as e:
            return e.as_error()
    return wrapper


def _int_arg(name: str, value: Any, minimum: int | None = None, maximum: int | None = None) -> int | None:
    """None stays None. An int, an integral float or a numeric string gives an int; 0 is a value.

    minimum and maximum are optional inclusive bounds; a value outside them is INVALID_ARGUMENT.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise _BadArgument(f"{name} must be an integer, got {value!r}")
    try:
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError
            number = int(value)
        else:
            number = int(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        raise _BadArgument(f"{name} must be an integer, got {value!r}") from None
    if (minimum is not None and number < minimum) or (maximum is not None and number > maximum):
        low = "" if minimum is None else f" at least {minimum}"
        high = "" if maximum is None else f" at most {maximum}"
        raise _BadArgument(f"{name} must be{low}{' and' if low and high else ''}{high}, got {number}")
    return number


def _num_arg(name: str, value: Any) -> float | None:
    """None stays None. A finite number or numeric string gives a float; 0 is a value."""
    if value is None:
        return None
    try:
        number = None if isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        number = None
    if number is None or not math.isfinite(number):
        raise _BadArgument(f"{name} must be a number, got {value!r}")
    return number


def _bool_arg(name: str, value: Any, default: bool | None = False) -> bool | None:
    """True or False, or default for None. The strings "true" and "false" (any case) parse."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise _BadArgument(f"{name} must be true or false, got {value!r}")


def _choice_arg(name: str, value: Any, allowed: Sequence[str]) -> str | None:
    """None stays None. Otherwise a case-insensitive match against allowed, else an error."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    raise _BadArgument(f"Unknown {name} '{value}'. Use one of: {', '.join(allowed)}.")


def _team_codes(value: Any, name: str = "team", required: bool = False) -> tuple[str, ...] | None:
    """Every stored code of the franchise a team value names, or None for None (no filter).

    Strips and upper-cases, and accepts historical codes (OAK, SD, STL, JAC, WSH), so the
    caller filters on all the codes that franchise was stored under. "" and unknown values
    raise UNKNOWN_TEAM; a missing required value raises MISSING_ARGUMENT.
    """
    if value is None:
        if required:
            raise _BadArgument(f"{name} is required", "MISSING_ARGUMENT")
        return None
    codes = team_codes(value)
    if codes is None:
        raise _BadArgument(f"Unknown team '{value}'", "UNKNOWN_TEAM")
    return codes


def _in_clause(column: str, values: Sequence[str]) -> tuple[str, list[str]]:
    """'col = ?' for one value, 'col IN (?, ?)' for several."""
    if len(values) == 1:
        return f"{column} = ?", list(values)
    return f"{column} IN ({', '.join('?' for _ in values)})", list(values)


def _team_filter(column: str, value: Any, name: str = "team") -> tuple[str, list[str]] | None:
    """(condition, params) matching a team value on column across the franchise's codes, or None."""
    codes = _team_codes(value, name)
    return None if codes is None else _in_clause(column, codes)


_PLAYOFF_ROUNDS = ("WC", "DIV", "CON", "SB")
_GAME_TYPES = ("REG", *_PLAYOFF_ROUNDS)
# season_type is required, exact and upper case. Season-total tools (derived team and player
# tables, player stats) take REG, POST or ALL; per-game tools also take one playoff round.
_TOTAL_SEASON_TYPES = ("REG", "POST", "ALL")
_GAME_SEASON_TYPES = ("REG", "POST", "ALL", *_PLAYOFF_ROUNDS)


def _season_type_arg(value: Any, allowed: Sequence[str] = _TOTAL_SEASON_TYPES) -> str:
    """The season_type of a call: one of `allowed`, exact and upper case. There is no default."""
    listed = ", ".join(allowed)
    if value is None:
        raise _BadArgument(f"season_type is required. Use one of: {listed}.")
    if not isinstance(value, str) or value not in allowed:
        raise _BadArgument(f"Unknown season_type {value!r}. Use one of: {listed}.")
    return value


def _game_types(value: Any) -> tuple[str, ...]:
    """game_type values of a per-game season_type: ALL is every type, POST the four rounds."""
    key = _season_type_arg(value, _GAME_SEASON_TYPES)
    if key == "ALL":
        return _GAME_TYPES
    return _PLAYOFF_ROUNDS if key == "POST" else (key,)


def _plays_season_type(season_type: str, prefix: str = "") -> tuple[str, list[str]] | None:
    """(clause, params) for a validated season_type on plays, or None for ALL.

    plays.season_type is REG or POST only; a round code takes its games from schedules.game_type.
    """
    if season_type == "ALL":
        return None
    if season_type in _PLAYOFF_ROUNDS:
        return f"{prefix}game_id IN (SELECT game_id FROM schedules WHERE game_type = ?)", [season_type]
    return f"{prefix}season_type = ?", [season_type]


def _range_reason(
    table: str,
    *,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    week: int | None = None,
    season_col: str = "season",
) -> str | None:
    """Why a season or week filter found nothing: it lies outside what `table` holds.

    Call it only when a query returned no rows. Returns None when the filter is inside the
    loaded range or the range cannot be read.
    """
    # table and season_col go straight into the SQL: callers must pass constants, never user input.
    if season is None and season_from is None and season_to is None and week is None:
        return None
    try:
        span = _execute(f"SELECT MIN({season_col}) AS lo, MAX({season_col}) AS hi FROM {table}")[0]
        lo, hi = span["lo"], span["hi"]
        if lo is None or hi is None:
            return None
        reasons = []
        if season is not None and not lo <= season <= hi:
            reasons.append(f"Season {season} is outside the loaded range {lo} to {hi}.")
        if season_from is not None and season_from > hi:
            reasons.append(f"season_from {season_from} is after the last loaded season ({hi}).")
        if season_to is not None and season_to < lo:
            reasons.append(f"season_to {season_to} is before the first loaded season ({lo}).")
        if week is not None:
            scoped = season is not None and lo <= season <= hi
            sql = f"SELECT MIN(week) AS lo, MAX(week) AS hi FROM {table}"
            if scoped:
                sql += f" WHERE {season_col} = ?"
            weeks = _execute(sql, [season] if scoped else None)[0]
            if weeks["lo"] is not None and not weeks["lo"] <= week <= weeks["hi"]:
                where = f" for {season}" if scoped else ""
                reasons.append(
                    f"Week {week} is outside the loaded range {weeks['lo']} to {weeks['hi']}{where}.")
        return " ".join(reasons) or None
    except Exception:  # a reason is best effort; never let it hide the real (empty) result
        logger.debug("Range lookup failed for %s", table, exc_info=True)
        return None


def _with_reason(result: dict[str, Any], rows: Any, table: str, **filters: Any) -> dict[str, Any]:
    """Add "reason" to an empty result when the season or week filter is out of range."""
    if not rows:
        reason = _range_reason(table, **filters)
        if reason:
            result["reason"] = reason
    return result


# ── Pagination ─────────────────────────────────────────────────────────────────
# Every list tool takes limit (1..500, default 100) and offset (>= 0) and reports
# count, truncated and, when more rows match, a hint.

_DEFAULT_LIMIT = 100


def _page_args(limit: Any = _DEFAULT_LIMIT, offset: Any = 0, limit_name: str = "limit") -> tuple[int, int]:
    """Validated (limit, offset). Strings are rejected, so "5" is an error, not 5."""
    checked = []
    for name, value, low, high, default in (
        (limit_name, limit, 1, _MAX_ROWS, _DEFAULT_LIMIT), ("offset", offset, 0, None, 0),
    ):
        if isinstance(value, str):
            raise _BadArgument(f"{name} must be an integer, got {value!r}")
        number = _int_arg(name, value, low, high)
        checked.append(default if number is None else number)
    return checked[0], checked[1]


def _execute_limited(
    sql: str, params: Sequence[Any] | None, limit: int, offset: int = 0
) -> tuple[list[dict], dict[str, Any]]:
    """Run `sql` (which has no LIMIT) for one page: LIMIT limit+1 OFFSET offset.

    Returns (rows, page). rows holds at most `limit` rows; page is {"count", "truncated"}
    plus a "hint" when more rows match. The extra row only tells whether another page exists.
    """
    rows = _execute(f"{sql}\nLIMIT {limit + 1} OFFSET {offset}", params or None)
    truncated = len(rows) > limit
    rows = rows[:limit]
    page: dict[str, Any] = {"count": len(rows), "truncated": truncated}
    if truncated:
        page["hint"] = (
            f"More rows match. This page has {len(rows)} rows from offset {offset}. "
            f"Call again with offset={offset + limit}, or use nfl_query for custom SQL."
        )
    return rows, page


# ── Player identity ────────────────────────────────────────────────────────────

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
    team: Sequence[str] | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    position: str | None = None,
    limit: int = _MAX_CANDIDATES,
) -> tuple[list[dict], bool]:
    """Players matching a name, narrowed by roster team and season. Returns (candidates, truncated).

    This is the one name rule of every player tool. gsis_id looks one player up by id. A name
    matches exactly, after folding case, accents, punctuation and a generational suffix, the
    full name, the last name, the short name ("K.Allen"), or the first name plus last name. A
    part of a name ("Alle") matches nobody.
    Team (a sequence of franchise codes, see _team_codes) and season mean "has a rosters row"
    (in that season when both are given).
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
        clause, codes = _in_clause("r.team", team)
        roster.append(clause)
        roster_params.extend(codes)
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


def _ambiguous(
    candidates: list[dict], truncated: bool, entity: str | None = None, id_arg: str = "player_id",
) -> dict[str, Any]:
    """Response for a name that matches several players. Not an error: the caller picks an id.

    id_arg is the argument of the calling tool that takes the chosen gsis_id.
    """
    result: dict[str, Any] = {
        "ambiguous": True,
        "candidates": candidates,
        "count": len(candidates),
        "hint": f"The name matches several players. Call again with {id_arg} set to one gsis_id.",
    }
    if entity is not None:
        result["entity"] = entity
    if truncated:
        result["truncated"] = True
    return result


# A client that names the argument player_name (or anything else) gets no player at all:
# mcpo drops an argument the schema does not list, so the error shows the exact form.
_ARGUMENT_FORM = (
    'Pass the name as player, for example {"player": "Joe Burrow"}, or a gsis_id as player_id. '
    "Other names such as player_name are not accepted, and a proxy can drop them without an "
    "error."
)
_MISSING_PLAYER = {"error": f"player or player_id is required. {_ARGUMENT_FORM}",
                   "code": "MISSING_PLAYER"}


def _resolve_player(
    player_name: str | None,
    player_id: str | None,
    *,
    team: Sequence[str] | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    entity: str | None = None,
    id_arg: str = "player_id",
) -> tuple[dict | None, dict[str, Any] | None]:
    """Pick one player for a tool call. Returns (player, early_response).

    player_id wins over player_name, and a name shaped like a gsis id counts as an id.
    A name matching one player gives (player, None). Several give an ambiguous response and
    an unknown id gives an error; both are returned as early_response. No match gives
    (None, None). When team or season leave nobody, the name is matched again without them:
    one player overall is used (so a player with no roster row in that season resolves) and
    several are returned as ambiguous. id_arg names the argument the ambiguity hint points to.
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
        return None, dict(_MISSING_PLAYER)
    filters = {"team": team, "season": season, "season_from": season_from, "season_to": season_to}
    candidates, truncated = _find_players(name, **filters)
    if not candidates and any(v is not None for v in filters.values()):
        candidates, truncated = _find_players(name)
    if len(candidates) == 1:
        return candidates[0], None
    if candidates:
        return None, _ambiguous(candidates, truncated, entity, id_arg)
    return None, None


@_checked
def nfl_player_lookup(
    player: str | None = None,
    team: str | None = None,
    season: int | None = None,
    position: str | None = None,
    player_id: str | None = None,
) -> dict[str, Any]:
    """Find players by name (or one by gsis_id) and return their gsis_id, so other player
    tools can take player_id. Several matches are the normal result, not an ambiguity."""
    pid = str(player_id).strip() if player_id is not None else ""
    name = str(player).strip() if player is not None else ""
    if not pid and _GSIS_ID.fullmatch(name):
        pid, name = name, ""
    if not pid and not name:
        return dict(_MISSING_PLAYER)
    season = _int_arg("season", season)
    try:
        if pid:
            candidates, truncated = _find_players(gsis_id=pid, limit=1)
        else:
            candidates, truncated = _find_players(
                name, team=_team_codes(team), season=season, position=position, limit=50)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.exception("Tool error")
        return {"error": str(e)}
    result: dict[str, Any] = {"candidates": candidates, "count": len(candidates)}
    if truncated:
        result["truncated"] = True
        result["hint"] = "More players match. Add team, season or position, or use a fuller name."
    return result


def _like_pattern(text: str) -> str:
    """Substring pattern for LIKE ... ESCAPE '\\' with % and _ taken literally."""
    escaped = text.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _no_player(key: str, player: Any) -> dict[str, Any]:
    """Empty page of a list tool whose player name matches nobody."""
    return {key: [], "count": 0, "truncated": False,
            "reason": f"No player matches '{player}'. Try nfl_player_lookup."}


def _one_player(
    player: str | None,
    player_id: str | None,
    key: str,
    *,
    team: Sequence[str] | None = None,
    season: int | None = None,
) -> tuple[dict | None, dict[str, Any] | None]:
    """The player filter of a list tool whose rows are listed under `key`. Returns
    (player, early_response); both are None when neither argument is given.

    The name resolves through players with the one name rule (_find_players): one match
    gives the player, so the tool filters on that player's id and never merges the rows of
    several players. Several matches give the ambiguous response, no match an empty page with
    a reason, and a resolver failure an error; each is returned as early_response.
    """
    if not player and not player_id:
        return None, None
    try:
        found, early = _resolve_player(player, player_id, team=team, season=season)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.exception("Tool error")
        return None, {"error": str(e)}
    if early is None and found is None:
        early = _no_player(key, player)
    return found, early


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
                grain, join_keys = table_info(table)
                return {"table": table, "grain": grain, "join_keys": join_keys,
                        "derived": table in DERIVED_TABLES, "columns": rows}
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
            "derived_tables": [t for t in DERIVED_TABLES if t in table_names],
            "relationships": relationships_for(set(table_names)),
            "data_notes": list(DATA_NOTES),
            "hint": "Pass category='<name>' for full PBP column details. Pass table='<name>' for any other table's "
                    "columns, grain and join keys. nfl_catalog lists every table with its row count and season range.",
        }
    cat = category.lower().strip()
    if cat == "all":
        full = "\n".join(v.strip() for v in _SCHEMA_CATEGORIES.values())
        return {"schema": full}
    if cat in _SCHEMA_CATEGORIES:
        return {"category": cat, "schema": _SCHEMA_CATEGORIES[cat].strip()}
    return {"error": f"Unknown category '{cat}'", "available": list(_SCHEMA_CATEGORIES.keys())}


def _iso_utc(value: Any) -> Any:
    """A TIMESTAMP as an ISO 8601 UTC string (fixed microseconds, so strings sort as times).
    loaded_at is stored in UTC. Any other value comes back unchanged."""
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return value.isoformat(timespec="microseconds") + "Z"
    return value


def _newest_seasons(tables: Sequence[str]) -> dict[str, tuple[Any, Any]]:
    """table -> (newest season, newest week within it) for tables with a season column.
    The week is None when the table has no week column. Best effort: {} on any failure."""
    try:
        cols: dict[str, set[str]] = {}
        for r in _execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'main' AND column_name IN ('season', 'season_year', 'week')"
        ):
            cols.setdefault(r["table_name"], set()).add(r["column_name"])
        parts = []
        for table in dict.fromkeys(tables):
            have = cols.get(table, set())
            season = "season" if "season" in have else "season_year" if "season_year" in have else None
            if season is None:
                continue
            week = (f'(SELECT TRY_CAST(MAX(week) AS INTEGER) FROM "{table}" '
                    f'WHERE {season} = (SELECT MAX({season}) FROM "{table}"))' if "week" in have else "NULL")
            parts.append(f"SELECT '{table}' AS table_name, TRY_CAST(MAX({season}) AS INTEGER) AS season, "
                         f'{week} AS week FROM "{table}"')
        if not parts:
            return {}
        return {r["table_name"]: (r["season"], r["week"]) for r in _execute(" UNION ALL ".join(parts))}
    except Exception as e:  # noqa: BLE001 - newest season is extra detail; never fail the status call for it
        logger.warning("nfl_status could not read the newest season per table: %s", e)
        return {}


def nfl_status() -> Dict[str, Any]:
    """Return database health: play counts, loaded seasons, and a summary of all ingested datasets,
    with the newest season and week and the last refresh time of each."""
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
            "SELECT dataset_id, table_name, SUM(row_count) as total_rows, "
            "MAX(loaded_at) AS last_refreshed "
            "FROM _ingest_metadata "
            "GROUP BY dataset_id, table_name "
            "ORDER BY dataset_id"
        )
        last_refreshed = _execute(
            "SELECT MAX(loaded_at) as last_refreshed FROM _ingest_metadata"
        )
        newest = _newest_seasons([d["table_name"] for d in datasets])
        for d in datasets:
            d["newest_season"], d["newest_week"] = newest.get(d["table_name"], (None, None))
            d["last_refreshed"] = _iso_utc(d.get("last_refreshed"))
        return {
            "plays": {
                "total_plays": total[0]["total_plays"] if total else 0,
                "season_range": min_max[0] if min_max else {},
                "seasons": seasons,
            },
            "datasets": {
                "total_loaded": len(datasets),
                "last_refreshed": _iso_utc(last_refreshed[0]["last_refreshed"]) if last_refreshed else None,
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
# A qualified reference (p.description) names the table, alias or CTE ('Table "p"', 'Values
# list "c"') the column is missing from.
_QUALIFIED_COLUMN_NOT_FOUND = re.compile(
    r'"(?P<table>[^"]+)" does not have a column named "(?P<name>[^"]+)"', re.IGNORECASE)
_TABLE_NOT_FOUND = re.compile(r"Table with name (?P<name>\S+) does not exist", re.IGNORECASE)
_QUOTED_NAMES = re.compile(r'"(?P<name>[^"]+)"')
_WORDS = re.compile(r"[^a-z0-9]+")

_CANDIDATE_BINDINGS = re.compile(r"Candidate bindings:(?P<names>[^\n]*)", re.IGNORECASE)
_NEAR_DESC = re.compile(r'at or near "desc"', re.IGNORECASE)
_DESC_HINT = (
    'desc is an SQL keyword, so the plays column desc (the play description) must be quoted: '
    '"desc", for example SELECT "desc" FROM plays WHERE "desc" ILIKE \'%touchdown%\'.'
)
# Any other parser error. Models reach for MySQL forms (SHOW COLUMNS FROM plays LIKE '%x%').
_SYNTAX_HINT = (
    "DuckDB could not parse the statement; nfl_query runs DuckDB SQL, not MySQL. List a "
    "table's columns with DESCRIBE plays (or SHOW plays), or search them with SELECT "
    "column_name FROM information_schema.columns WHERE table_name = 'plays' AND column_name "
    "ILIKE '%conv%'. nfl_schema(category='<name>') lists play-by-play columns by topic."
)

_COLUMN_NAMES_SQL = (
    "SELECT DISTINCT column_name FROM information_schema.columns "
    "WHERE table_schema = 'main'"
)
_TABLE_NAMES_SQL = (
    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
)
# A guessed column name (or its last word) that means a column with another name. The
# synonym is offered first when one of the queried tables has it.
_COLUMN_SYNONYMS = {
    "description": "desc",            # plays
    "full_name": "display_name",      # players (rosters and injuries do have full_name)
    "season_type": "game_type",       # schedules, snap_counts, injuries
}


def _live_names(sql: str, params: Sequence[Any] | None = None) -> list[str]:
    """Names straight from the live schema; empty if the lookup itself fails."""
    try:
        return [next(iter(row.values())) for row in _execute(sql, params)]
    except (duckdb.Error, ValueError, TimeoutError):
        return []


def _base_tables(node: Any) -> Iterable[tuple[str, str]]:
    """(table_name, alias) of every BASE_TABLE reference in a json_serialize_sql tree."""
    if isinstance(node, dict):
        if node.get("type") == "BASE_TABLE" and node.get("table_name"):
            yield node["table_name"], node.get("alias") or ""
        for value in node.values():
            yield from _base_tables(value)
    elif isinstance(node, list):
        for value in node:
            yield from _base_tables(value)


def _queried_tables(sql: str, qualifier: str | None = None) -> list[str] | None:
    """Live tables the statement reads, joined tables and the tables inside CTEs included.

    DuckDB parses the statement (json_serialize_sql); a CTE name is not a live table, so it
    drops out. qualifier is the table or alias of a qualified column (p in p.description):
    when it names one live table, only that table is returned. None when the statement cannot
    be parsed that way.
    """
    match = _EXPLAIN_PREFIX.match(sql)
    body = sql[match.end():] if match else sql
    try:
        with duckdb.connect() as conn:
            tree = json.loads(conn.execute("SELECT json_serialize_sql(?)", [body]).fetchone()[0])
    except (duckdb.Error, TypeError, ValueError):
        return None
    if tree.get("error"):
        return None
    live = {t.lower(): t for t in _live_names(_TABLE_NAMES_SQL)}
    refs = [(live[t.lower()], alias.lower()) for t, alias in _base_tables(tree) if t.lower() in live]
    if qualifier:
        q = qualifier.lower()
        named = [t for t, alias in refs if alias == q or (not alias and t.lower() == q)]
        if named:
            return named[:1]
    return list(dict.fromkeys(t for t, _ in refs))


def _suggest(unknown: str, candidates: Iterable[str], limit: int = 5) -> list[str]:
    """_nearest, with the synonym of the guessed name first when it is a candidate."""
    pool = {c.lower() for c in candidates if c}
    words = [w for w in _WORDS.split(unknown.lower()) if w]
    synonyms = [_COLUMN_SYNONYMS[k] for k in (unknown.lower(), *words[-1:]) if k in _COLUMN_SYNONYMS]
    first = [s for s in dict.fromkeys(synonyms) if s in pool]
    return (first + [n for n in _nearest(unknown, pool, limit) if n not in first])[:limit]


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


def _sql_error_detail(exc: Exception, sql: str | None = None) -> Dict[str, Any]:
    """Turn a DuckDB binder/catalog/parser error into fields the caller can act on.

    did_you_mean for a column lists only columns of the tables the statement reads (and the
    binder's own candidates, which are in scope), so a guess is never answered with a column
    of an unrelated table. When the statement cannot be parsed for its tables, every live
    column is a candidate.
    """
    message = str(exc)
    spelled = [m.group("name") for m in _QUOTED_NAMES.finditer(message)]
    near_desc = bool(_NEAR_DESC.search(message))
    if near_desc or isinstance(exc, duckdb.ParserException):
        return {"code": "SQL_SYNTAX_ERROR", "hint": _DESC_HINT if near_desc else _SYNTAX_HINT}

    guessed_column = _COLUMN_NOT_FOUND.search(message) or _QUALIFIED_COLUMN_NOT_FOUND.search(message)
    if guessed_column:
        unknown = guessed_column.group("name")
        qualifier = guessed_column.groupdict().get("table")
        tables = _queried_tables(sql, qualifier) if sql else None
        if tables is None:
            pool = _live_names(_COLUMN_NAMES_SQL)
        elif tables:
            marks = ", ".join("?" for _ in tables)
            pool = _live_names(f"{_COLUMN_NAMES_SQL} AND table_name IN ({marks})", tables)
        else:  # only subqueries or table functions: the binder candidates are the scope
            pool = []
        bound = _CANDIDATE_BINDINGS.search(message)
        if bound:
            pool += [m.group("name").rsplit(".", 1)[-1]
                     for m in _QUOTED_NAMES.finditer(bound.group("names"))]
        return {
            "code": "COLUMN_NOT_FOUND",
            "unknown_column": unknown,
            "did_you_mean": _suggest(unknown, pool),
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
# Statements that list a table's columns or describe it. A truncated one gets a hint on
# listing only the columns needed, through information_schema.columns.
_CATALOG_STATEMENT = re.compile(rf"\A{_GAP}(?:DESCRIBE|SHOW|SUMMARIZE)\b", re.IGNORECASE | re.DOTALL)
_DEFAULT_QUERY_ROWS = 100


def _truncated_hint(max_rows: int, sql: str) -> str:
    """How to see more than the max_rows rows a truncated nfl_query returned."""
    if max_rows < _MAX_ROWS:
        head = (f"Only the first {max_rows} rows are shown (max_rows={max_rows}). To see more, "
                f"raise max_rows (up to {_MAX_ROWS}), ")
    else:
        head = (f"Only the first {_MAX_ROWS} rows are shown; {_MAX_ROWS} is the most max_rows "
                "allows. ")
    if _CATALOG_STATEMENT.match(sql) or "information_schema" in sql.lower():
        columns = ("SELECT column_name FROM information_schema.columns WHERE table_name = 'plays' "
                   "AND column_name LIKE '%epa%'")
        return (f"{head}page with LIMIT and OFFSET, or list only the columns you need with WHERE, "
                f"for example {columns}. nfl_schema(category='<name>') lists play-by-play "
                "columns by topic.")
    return f"{head}add a WHERE filter, page with LIMIT and OFFSET, or aggregate with GROUP BY."


def _statement_gate(sql: str) -> dict[str, Any] | None:
    """Return an error dict unless `sql` is one SELECT, or EXPLAIN [ANALYZE] of one SELECT."""
    try:
        statements = duckdb.extract_statements(sql)
    except (duckdb.Error, TypeError) as e:
        return {"error": f"Could not parse SQL: {e}", **_sql_error_detail(e)}
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
            return {"error": f"Could not parse SQL after EXPLAIN: {e}", "statement_type": stype.name,
                    **_sql_error_detail(e)}
        if len(inner) != 1:
            return {"error": "Send exactly one statement per call.", "statement_type": stype.name}
        stype = inner[0].type
    if stype != duckdb.StatementType.SELECT:
        return {
            "error": f"Only SELECT and EXPLAIN of a SELECT are allowed; got {stype.name}.",
            "statement_type": stype.name,
        }
    return None


def nfl_query(sql: str, max_rows: int | None = None) -> Dict[str, Any]:
    """
    Execute one read-only SQL statement against the nflread database.

    Rules:
      - Exactly one statement, parsed by DuckDB (a trailing ';' is fine)
      - Only SELECT, or EXPLAIN [ANALYZE] of a SELECT; anything else is
        rejected with its statement_type
      - max_rows is an int from 1 to 500; larger values are clamped to 500. Not given
        (None): 100
      - Query runs as written; max_rows + 1 rows are fetched to set `truncated`
      - 10-second timeout (NFL_MCP_QUERY_TIMEOUT_SECONDS overrides)

    Returns columns ([{name, type}]), rows, row_count and truncated, and a `hint` when
    truncated. Unknown column/table failures also return `code`, the offending name,
    `did_you_mean` (ranked against the live schema) and a `hint`; a parser error returns
    code SQL_SYNTAX_ERROR and a `hint` with the DuckDB forms.
    """
    if max_rows is not None and (
            isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1):
        return {"error": "max_rows must be an integer from 1 to 500."}

    if not isinstance(sql, str):
        return {"error": "sql must be a string"}
    rejected = _statement_gate(sql)
    if rejected:
        return rejected
    max_rows = _DEFAULT_QUERY_ROWS if max_rows is None else min(max_rows, _MAX_ROWS)

    try:
        columns, rows = _fetch_capped(sql, max_rows + 1)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e), **_sql_error_detail(e, sql)}

    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    result = {
        "columns":   columns,
        "rows":      rows,
        "row_count": len(rows),
        "truncated": truncated,
    }
    if truncated:
        result["hint"] = _truncated_hint(max_rows, sql)
    return result


_PLAY_TYPES = ("pass", "run", "field_goal", "punt", "kickoff", "extra_point", "qb_kneel",
               "qb_spike", "no_play")
_SITUATIONS = {
    "red_zone": "yardline_100 <= 20",
    "third_down": "down = 3",
    "fourth_down": "down = 4",
    "two_minute": "half_seconds_remaining <= 120 AND (qtr = 2 OR qtr >= 4)",
    "two_minute_half": "qtr = 2 AND half_seconds_remaining <= 120",
    "two_minute_game": "qtr >= 4 AND half_seconds_remaining <= 120",
    "goal_to_go": "goal_to_go = 1",
}
# The plays id column behind each player_role. A play matches when any column matches.
_ROLE_COLUMNS = {
    "passer": ("passer",),
    "rusher": ("rusher",),
    "receiver": ("receiver",),
    "any_offense": ("passer", "rusher", "receiver"),
    "defender": (
        "sack", "half_sack_1", "half_sack_2", "solo_tackle_1", "solo_tackle_2",
        "assist_tackle_1", "assist_tackle_2", "assist_tackle_3", "assist_tackle_4",
        "tackle_with_assist_1", "tackle_with_assist_2", "interception", "pass_defense_1",
        "pass_defense_2", "forced_fumble_player_1", "forced_fumble_player_2", "qb_hit_1",
        "qb_hit_2", "tackle_for_loss_1", "tackle_for_loss_2",
    ),
    "kicker": ("kicker",),
    "punter": ("punter",),
    "returner": ("kickoff_returner", "punt_returner"),
}
_ROLE_COLUMNS["any"] = tuple(dict.fromkeys(c for cols in _ROLE_COLUMNS.values() for c in cols))
# Fixed allow-list: the key is the only user input; the SQL is never built from it.
_PLAY_SORTS = {
    "abs_epa": "ABS(epa) DESC NULLS LAST",
    "epa": "epa DESC NULLS LAST",
    "wpa": "wpa DESC NULLS LAST",
    "yards_gained": "yards_gained DESC NULLS LAST",
    "game_order": "game_id ASC, play_id ASC",
}
_PLAY_COLUMNS = (
    "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "qtr", "down", "ydstogo",
    "play_type", "yards_gained", "epa", "wpa", '"desc"', "passer_player_name",
    "passer_player_id", "rusher_player_name", "rusher_player_id", "receiver_player_name",
    "receiver_player_id", "touchdown", "interception",
)
# plays stores these as DOUBLE, but every value is whole (SELECT count(*) FROM plays WHERE
# play_id <> round(play_id) OR qtr <> round(qtr) OR ... -> 0), so they are returned as ints.
# epa and wpa are real decimals and stay DOUBLE.
_PLAY_INT_COLUMNS = frozenset({"play_id", "season", "week", "qtr", "down", "ydstogo",
                               "yards_gained", "touchdown", "interception"})
_PLAY_SELECT = ", ".join(f"CAST({c} AS INTEGER) AS {c}" if c in _PLAY_INT_COLUMNS else c
                         for c in _PLAY_COLUMNS)
_PLAYS_COLUMNS_SQL = (
    "SELECT column_name FROM information_schema.columns "
    "WHERE table_schema = 'main' AND table_name = 'plays'"
)


class _UnknownColumns(_BadArgument):
    """columns named something the plays table does not have; carries did_you_mean."""

    def __init__(self, message: str, did_you_mean: list[str]):
        super().__init__(message, "UNKNOWN_COLUMN")
        self.did_you_mean = did_you_mean

    def as_error(self) -> dict[str, Any]:
        return {**super().as_error(), "did_you_mean": self.did_you_mean}


def _extra_play_columns(columns: Any) -> list[str]:
    """Quoted extra plays columns, each checked against the live schema.

    A name that is not a live column gives UNKNOWN_COLUMN with did_you_mean. Nothing from
    the caller reaches the SQL except a name that matched a live column exactly.
    """
    if columns is None:
        return []
    if not isinstance(columns, (list, tuple)) or not all(isinstance(c, str) for c in columns):
        raise _BadArgument("columns must be a list of plays column names")
    live = {name.lower(): name for name in _live_names(_PLAYS_COLUMNS_SQL)}
    if not live:
        raise _BadArgument("Could not read the plays schema to check columns", "SCHEMA_UNAVAILABLE")
    unknown = [c for c in columns if c.strip().lower() not in live]
    if unknown:
        suggestions = list(dict.fromkeys(n for c in unknown for n in _suggest(c, live.values())))
        raise _UnknownColumns(
            f"Unknown plays column(s): {', '.join(repr(c) for c in unknown)}. "
            "Call nfl_schema(category=...) for play-by-play columns.", suggestions[:5])
    default = {c.strip('"') for c in _PLAY_COLUMNS}
    chosen = dict.fromkeys(live[c.strip().lower()] for c in columns)
    return [f'"{name}"' for name in chosen if name not in default]


@_checked
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
    max_rows: int | None = None,
    player_id: str | None = None,
    player_role: str | None = None,
    down: int | None = None,
    qtr: int | None = None,
    min_yardline_100: float | None = None,
    max_yardline_100: float | None = None,
    min_score_differential: float | None = None,
    max_score_differential: float | None = None,
    shotgun: bool | None = None,
    no_huddle: bool | None = None,
    min_wp: float | None = None,
    max_wp: float | None = None,
    sort_by: str = "abs_epa",
    columns: list[str] | None = None,
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
) -> Dict[str, Any]:
    """Search for plays using structured filters instead of raw SQL. max_rows is an alias of limit."""
    season_type = _season_type_arg(season_type, _GAME_SEASON_TYPES)
    limit, offset = _page_args(
        limit if max_rows is None else max_rows, offset, "limit" if max_rows is None else "max_rows")
    season, season_from, season_to = (_int_arg(n, v) for n, v in (
        ("season", season), ("season_from", season_from), ("season_to", season_to)))
    week, min_yards, down, qtr = (_int_arg(n, v) for n, v in (
        ("week", week), ("min_yards", min_yards), ("down", down), ("qtr", qtr)))
    play_type = _choice_arg("play_type", play_type, _PLAY_TYPES)
    situation = _choice_arg("situation", situation, tuple(_SITUATIONS))
    role = _choice_arg("player_role", player_role, tuple(_ROLE_COLUMNS))
    sort_by = _choice_arg("sort_by", sort_by, tuple(_PLAY_SORTS)) or "abs_epa"
    is_touchdown = _bool_arg("is_touchdown", is_touchdown)
    is_turnover = _bool_arg("is_turnover", is_turnover)
    shotgun, no_huddle = (_bool_arg(n, v, default=None) for n, v in (
        ("shotgun", shotgun), ("no_huddle", no_huddle)))
    ranges = [(c, op, _num_arg(n, v)) for c, op, n, v in (
        ("yardline_100", ">=", "min_yardline_100", min_yardline_100),
        ("yardline_100", "<=", "max_yardline_100", max_yardline_100),
        ("score_differential", ">=", "min_score_differential", min_score_differential),
        ("score_differential", "<=", "max_score_differential", max_score_differential),
        ("wp", ">=", "min_wp", min_wp), ("wp", "<=", "max_wp", max_wp))]
    if role and not (player or (player_id and str(player_id).strip())):
        raise _BadArgument(f"player or player_id is required with player_role. {_ARGUMENT_FORM}",
                           "MISSING_PLAYER")
    extra = _extra_play_columns(columns)

    conditions = []
    params: list[Any] = []
    for column, value, label in (("posteam", team, "team"), ("defteam", opponent, "opponent")):
        clause = _team_filter(column, value, label)
        if clause:
            conditions.append(clause[0])
            params.extend(clause[1])
    resolved = None
    gsis_id = str(player_id).strip() if player_id and str(player_id).strip() else None
    if player and gsis_id is None:
        try:
            resolved, early = _resolve_player(
                player, None, season=season, season_from=season_from, season_to=season_to)
        except (duckdb.Error, ValueError, TimeoutError) as e:
            logger.exception("Tool error")
            return {"error": str(e)}
        if early is not None:
            return early
        if resolved is None:
            return {"rows": [], "count": 0, "row_count": 0, "truncated": False,
                    "reason": f"No player matches '{player}'. Try nfl_player_lookup."}
        gsis_id = resolved["gsis_id"]
    elif gsis_id is not None:
        try:
            _, early = _resolve_player(None, gsis_id)
        except (duckdb.CatalogException, duckdb.BinderException):
            early = None  # no usable players table (a trimmed database): use the id as given
        except (duckdb.Error, ValueError, TimeoutError) as e:
            logger.exception("Tool error")
            return {"error": str(e)}
        if early is not None:
            return early
    if gsis_id is not None:
        id_columns = [f"{c}_player_id" for c in _ROLE_COLUMNS[role or "any"]]
        conditions.append("(" + " OR ".join(f"{c} = ?" for c in id_columns) + ")")
        params.extend([gsis_id] * len(id_columns))
    for column, op, value in (
        ("season", "=", season), ("season", ">=", season_from), ("season", "<=", season_to),
        ("week", "=", week),
    ):
        if value is not None:
            conditions.append(f"{column} {op} ?")
            params.append(value)
    clause = _plays_season_type(season_type)
    if clause:
        conditions.append(clause[0])
        params.extend(clause[1])
    for column, value in (("play_type", play_type), ("down", down), ("qtr", qtr)):
        if value is not None:
            conditions.append(f"{column} = ?")
            params.append(value)
    if situation:
        conditions.append(_SITUATIONS[situation])
    if is_touchdown:
        conditions.append("touchdown = 1")
    if is_turnover:
        conditions.append("(interception = 1 OR fumble_lost = 1)")
    if min_yards is not None:
        conditions.append("yards_gained >= ?")
        params.append(min_yards)
    for column, flag in (("shotgun", shotgun), ("no_huddle", no_huddle)):
        if flag is not None:
            conditions.append(f"{column} = {1 if flag else 0}")
    for column, op, value in ranges:
        if value is not None:
            conditions.append(f"{column} {op} ?")
            params.append(value)

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = (
        f"SELECT {', '.join([_PLAY_SELECT, *extra])} "
        f"FROM plays WHERE {where} "
        f"ORDER BY {_PLAY_SORTS[sort_by]}, game_id, play_id, rowid"
    )

    try:
        rows, page = _execute_limited(sql, params, limit, offset)
        result = _with_player({"rows": rows, **page, "row_count": len(rows)}, resolved)
        return _with_reason(result, rows, "plays", season=season, season_from=season_from,
                            season_to=season_to, week=week)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def _team_stats_filter(
    season: int | None,
    season_from: int | None,
    season_to: int | None,
    season_type: str | None,
) -> tuple[str, list[Any]]:
    """WHERE fragment and params for the derived team tables ('ALL' keeps both season types).

    season_type is required; a missing or invalid one raises before any query.
    """
    clause = ""
    params: list[Any] = []
    season, season_from, season_to = (_int_arg(n, v) for n, v in (
        ("season", season), ("season_from", season_from), ("season_to", season_to)))
    if season is not None:
        clause += " AND season_year = ?"
        params.append(season)
    if season_from is not None:
        clause += " AND season_year >= ?"
        params.append(season_from)
    if season_to is not None:
        clause += " AND season_year <= ?"
        params.append(season_to)
    season_type = _season_type_arg(season_type)
    if season_type != "ALL":
        clause += " AND season_type = ?"
        params.append(season_type)
    return clause, params


class _StaleDerivedTables(Exception):
    """A season_type was asked of derived tables built before season_type existed."""


_STALE_ERROR = {
    "error": "Derived team tables are stale; run `nfl-mcp rebuild`",
    "code": "STALE_DERIVED_TABLES",
}


_STALE_PLAYER_ERROR = {
    "error": "Derived player tables are stale; run `nfl-mcp upgrade`",
    "code": "STALE_DERIVED_TABLES",
}


def _season_typed_page(
    sql: str, params: list[Any], limit: int, offset: int,
) -> tuple[list[dict], dict]:
    """One page of a derived player table that must carry season_type.

    A table built before season_type existed (schema version 1) raises _StaleDerivedTables.
    """
    try:
        return _execute_limited(sql, params, limit, offset)
    except duckdb.BinderException as e:
        if 'column "season_type"' not in str(e):  # the message also quotes the SQL text
            raise
        raise _StaleDerivedTables() from e


def _team_rows(
    table: str,
    codes: Sequence[str],
    filters: tuple[int | None, int | None, int | None, str | None],
    tail: str = "",
) -> list[dict]:
    """Rows of one derived team table for a franchise's codes.

    A table built before season_type existed would blend REG and POST, so it raises
    _StaleDerivedTables for every season_type, as the derived player tables do.
    """
    clause, params = _team_stats_filter(*filters)
    team_sql, team_params = _in_clause("team", codes)
    sql = (f"SELECT * FROM {table} WHERE {team_sql}{clause} "
           f"ORDER BY season_year, season_type DESC{tail}")
    try:
        return _execute(sql, [*team_params, *params])
    except duckdb.BinderException as e:
        if 'column "season_type"' not in str(e):  # the message also quotes the SQL text
            raise
        raise _StaleDerivedTables() from e


# What each yardage field of the team tables counts, shown in the definitions block of
# nfl_team_stats and nfl_compare (team mode) for the blocks in the result.
_TEAM_DEFINITIONS = {
    "offense": {
        "total_plays": "Pass and run plays plus kneels, the official play count. Spikes, special "
                       "teams, no_play rows and two-point tries are out.",
        "total_yards": "Total net yards, the official figure: rush_yards + net_pass_yards. Sacks "
                       "and kneels are in.",
        "yards_per_play": "total_yards / total_plays.",
        "avg_epa": "Pass and run plays only, without kneels, as are pass_epa, rush_epa, "
                   "third_down_pct, red_zone_td_pct, explosive_plays and touchdowns.",
        "rush_yards": "rushing_yards plus yards after a lateral (lateral_rushing_yards) on "
                      "rush_attempt=1 plays, two-point tries and no_play out. Kneels and scrambles "
                      "are in, as in the official rushing figure.",
        "yards_per_rush": "rush_yards / rush_plays.",
        "pass_yards": "Gross passing yards: passing_yards on pass attempts (sacks, two-point tries "
                      "and no_play out). Yards lost on sacks are not subtracted.",
        "yards_per_pass": "pass_yards / pass_attempts (gross).",
        "sacks": "Sacks taken: sack=1 plays, two-point tries and no_play out.",
        "sack_yards": "Yards lost on sacks, as a positive number.",
        "net_pass_yards": "pass_yards - sack_yards: net passing yards, the official passing figure. "
                          "total_yards counts passing this way.",
    },
    "defense": {
        "plays_against": "The opponent's pass and run plays plus kneels, as in total_plays.",
        "yards_allowed": "Total net yards allowed, the official figure: rush_yards_allowed + "
                         "net_pass_yards_allowed. Sacks and kneels are in.",
        "yards_per_play_allowed": "yards_allowed / plays_against.",
        "avg_epa_allowed": "The opponent's pass and run plays only, without kneels, as is "
                           "third_down_stop_pct.",
        "rush_yards_allowed": "The opponent's rush_yards: rushing_yards plus yards after a lateral, "
                              "kneels and scrambles in.",
        "pass_yards_allowed": "Gross passing yards allowed: the opponent's passing_yards on pass "
                              "attempts. Yards gained on sacks are not subtracted.",
        "sacks": "Sacks made: sack=1 plays against the opponent, two-point tries and no_play out.",
        "sack_yards": "Yards the defense gained on sacks (the opponent's loss), as a positive number.",
        "net_pass_yards_allowed": "pass_yards_allowed - sack_yards: net passing yards allowed, the "
                                  "official figure. yards_allowed counts passing this way.",
    },
}
# A team table built before schema version 4 has no net_pass_yards (offense) or
# net_pass_yards_allowed (defense). Its rows lack these fields, or count them another way
# (no lateral yards in rush_yards, total_yards summed from yards_gained without kneels), so
# their definitions are left out.
_TEAM_V4_MARKER = {"offense": "net_pass_yards", "defense": "net_pass_yards_allowed"}
_TEAM_V4_FIELDS = {
    "offense": {"total_plays", "total_yards", "yards_per_play", "rush_yards", "yards_per_rush",
                "sack_yards", "net_pass_yards"},
    "defense": {"plays_against", "yards_allowed", "yards_per_play_allowed", "rush_yards_allowed",
                "pass_yards_allowed", "sack_yards", "net_pass_yards_allowed"},
}
_STALE_TEAM_WARNING = (
    "team_offense_stats and team_defense_stats were built by an older nfl-mcp, so sack_yards, "
    "net_pass_yards and the defense yardage split are left out. pass_yards is gross: yards lost "
    "on sacks are not subtracted. total_yards and total_plays do not count kneels yet, and "
    "total_yards is not yet rush_yards + net_pass_yards. The new fields appear after the next "
    "database update."
)


def _add_team_definitions(result: dict[str, Any], blocks: dict[str, list[dict]]) -> None:
    """Definitions of the offense and defense blocks in `blocks` (block -> rows), and the
    warning when the rows come from a table built before schema version 4."""
    present = [b for b in ("offense", "defense") if b in blocks]
    if not present:
        return
    stale = any(_TEAM_V4_MARKER[b] not in row for b in present for row in blocks[b])
    result["definitions"] = {
        b: {k: v for k, v in _TEAM_DEFINITIONS[b].items()
            if not (stale and k in _TEAM_V4_FIELDS[b])}
        for b in present}
    if stale:
        result["warning"] = _STALE_TEAM_WARNING


@_checked
def nfl_team_stats(
    team: str,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    side: str = "both",
    season_type: str | None = None,
) -> Dict[str, Any]:
    """Get pre-aggregated team stats (offense, defense, or both) for REG, POST, or ALL games."""
    _season_type_arg(season_type)
    codes = _team_codes(team, required=True)
    side = _choice_arg("side", "both" if side is None else side,
                       ("offense", "defense", "situational", "both"))
    results = {}
    filters = (season, season_from, season_to, season_type)
    _team_stats_filter(*filters)  # validate before any query

    try:
        if side in ("offense", "both"):
            results["offense"] = _team_rows("team_offense_stats", codes, filters)
        if side in ("defense", "both"):
            results["defense"] = _team_rows("team_defense_stats", codes, filters)
        if side in ("situational", "both"):
            results["situational"] = _team_rows("situational_stats", codes, filters, ", situation")
        _add_team_definitions(results, dict(results))
        return _with_reason(
            results, any(results.get(k) for k in ("offense", "defense", "situational")),
            "team_offense_stats", season=_int_arg("season", season),
            season_from=_int_arg("season_from", season_from),
            season_to=_int_arg("season_to", season_to), season_col="season_year")
    except _StaleDerivedTables:
        return dict(_STALE_ERROR)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


_NOT_TWO_POINT = "COALESCE(two_point_attempt, 0) = 0"
_NOT_NO_PLAY = "COALESCE(play_type, '') <> 'no_play'"
_NOT_SACK = "COALESCE(sack, 0) = 0"
# Row sets for one player (me.id). Each is the definition shown to users in _BLOCK_DEFINITIONS.
_ATTEMPT = (f"passer_player_id = me.id AND pass_attempt = 1 AND {_NOT_SACK} "
            f"AND {_NOT_TWO_POINT} AND {_NOT_NO_PLAY}")
_SACKED = f"passer_player_id = me.id AND sack = 1 AND {_NOT_TWO_POINT} AND {_NOT_NO_PLAY}"
_DROPBACK = (f"qb_dropback = 1 AND {_NOT_TWO_POINT} AND {_NOT_NO_PLAY} "
             "AND (passer_player_id = me.id OR (rusher_player_id = me.id AND qb_scramble = 1))")
_CARRY = f"rusher_player_id = me.id AND rush_attempt = 1 AND {_NOT_TWO_POINT}"
_PASS_TO_RECEIVER = f"pass_attempt = 1 AND {_NOT_SACK} AND {_NOT_TWO_POINT}"
_TARGET = f"receiver_player_id = me.id AND {_PASS_TO_RECEIVER}"
# A lateral after the catch: the second player gets the yards and the touchdown.
_LATERAL_RECEIVER = f"lateral_receiver_player_id = me.id AND {_PASS_TO_RECEIVER}"
_LATERAL_RUSHER = f"lateral_rusher_player_id = me.id AND rush_attempt = 1 AND {_NOT_TWO_POINT}"
# td_player_id names the scorer; it is NULL on rows without a touchdown (and on test rows).
_RECEIVING_TD = (f"({_TARGET} OR {_LATERAL_RECEIVER}) AND pass_touchdown = 1 "
                 "AND COALESCE(td_player_id, receiver_player_id) = me.id")

# What each stat counts, shown to users in the definitions block. A stat of one block is
# listed only when that block is in the result; yards and touchdowns keep the part of each
# block in the result.
_BLOCK_DEFINITIONS = {
    "passing": {
        "attempts": "pass_attempt=1 AND sack=0 AND two_point_attempt=0 AND play_type<>'no_play', "
                    "passer_player_id = the player. Spikes count as attempts. A play coded no_play "
                    "with pass_attempt=1 (for example a reversed sack ruled intentional grounding) "
                    "is excluded, so attempts can be one below the official count.",
        "completions": "attempts with complete_pass=1.",
        "sacks": "sack=1 with the player as passer_player_id, excluding two_point_attempt=1 and no_play.",
        "dropbacks": "qb_dropback=1 AND two_point_attempt=0 AND play_type<>'no_play', where the player is "
                     "passer_player_id or is rusher_player_id on a qb_scramble=1 play "
                     "(attempts excluding spikes, plus sacks, plus scrambles).",
        "epa_per_attempt": "average epa over attempts.",
        "epa_per_dropback": "average epa over dropbacks.",
    },
    "rushing": {
        "carries": "rush_attempt=1 AND two_point_attempt=0, rusher_player_id = the player. "
                   "Kneels and scrambles count; no_play is not excluded.",
        "scrambles": "carries with qb_scramble=1.",
        "yards_per_carry": "rushing yards / carries.",
    },
    "receiving": {
        "targets": "pass_attempt=1 AND sack=0 AND two_point_attempt=0, receiver_player_id = the player.",
        "receptions": "targets with complete_pass=1.",
    },
}
_SHARED_DEFINITIONS = {
    "yards": {
        "passing": "passing_yards on attempts",
        "rushing": "rushing_yards on carries plus lateral_rushing_yards when the player took a "
                   "lateral",
        "receiving": "receiving_yards on targets plus lateral_receiving_yards when the player "
                     "took a lateral",
    },
    "touchdowns": {
        "passing": "pass_touchdown=1 on attempts",
        "rushing": "rush_touchdown=1 on carries",
        "receiving": "pass_touchdown=1 on targets or lateral receptions, credited to the scorer "
                     "(td_player_id), so a receiver who lateraled to the scorer gets none",
    },
}
_GAMES_DEFINITION = (
    "distinct games of the season and season type in which the player has a snap_counts row "
    "with at least one snap (offense, defense or special teams) or a player_stats row (a game "
    "with a recorded stat). snap_counts starts in 2012, so before 2012 a game with no recorded "
    "stat is not counted. null when neither table has the season."
)


def _stat_definitions(blocks: Sequence[str]) -> dict[str, str]:
    """The definitions of the stat blocks in a result, plus yards, touchdowns and games."""
    defs: dict[str, str] = {}
    for block in blocks:
        defs.update(_BLOCK_DEFINITIONS[block])
    for key, parts in _SHARED_DEFINITIONS.items():
        defs[key] = "; ".join(f"{block}: {parts[block]}" for block in blocks) + "."
    defs["games"] = _GAMES_DEFINITION
    return defs


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
        COALESCE({_sql_sum('rushing_yards', _CARRY)}, 0)
            + COALESCE({_sql_sum('lateral_rushing_yards', _LATERAL_RUSHER)}, 0) AS rush_yards,
        {_sql_count(_CARRY, 'rush_touchdown = 1')} AS rush_tds,
        {_sql_count(_CARRY, 'qb_scramble = 1')} AS scrambles,
        {_sql_count(_CARRY, 'fumble_lost = 1')} AS fumbles_lost,
        {_sql_count(_CARRY, 'yards_gained >= 10')} AS explosive_runs,
        {_sql_avg('epa', _CARRY)} AS epa_per_carry,
        {_sql_count(_TARGET)} AS targets,
        {_sql_count(_TARGET, 'complete_pass = 1')} AS receptions,
        COALESCE({_sql_sum('receiving_yards', _TARGET)}, 0)
            + COALESCE({_sql_sum('lateral_receiving_yards', _LATERAL_RECEIVER)}, 0) AS rec_yards,
        {_sql_count(_RECEIVING_TD)} AS rec_tds,
        {_sql_avg('yards_after_catch', _TARGET, 1)} AS avg_yac,
        {_sql_count(_TARGET, 'yards_gained >= 20')} AS explosive_plays,
        {_sql_avg('epa', _TARGET)} AS epa_per_target
    FROM plays, (SELECT CAST(? AS VARCHAR) AS id) AS me
    WHERE me.id IN (passer_player_id, rusher_player_id, receiver_player_id,
                    lateral_receiver_player_id, lateral_rusher_player_id)
"""


def _season_row(row: dict[str, Any], games: int | None = None) -> dict[str, Any]:
    """Nest one aggregate row into passing, rushing and receiving blocks."""
    def n(key: str) -> int:
        return int(row[key] or 0)

    attempts, carries = n("attempts"), n("carries")
    return {
        "season": row["season"],
        "season_type": row["season_type"],
        "games": games,
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


# Games played: a game with a snap (snap_counts, 2012 on) or a recorded stat (player_stats, the
# weekly nflverse table). player_stats alone misses a game with snaps and no stat, such as Josh
# Allen's one snap in week 18 of 2024.
_PLAYER_GAMES_SQL = """
    SELECT season, season_type, COUNT(DISTINCT game_id) AS games FROM (
        SELECT season, season_type, game_id FROM player_stats
        WHERE player_id = ? AND game_id IS NOT NULL
        UNION
        SELECT s.season,
               CASE s.game_type WHEN 'REG' THEN 'REG' WHEN 'WC' THEN 'POST' WHEN 'DIV' THEN 'POST'
                    WHEN 'CON' THEN 'POST' WHEN 'SB' THEN 'POST' END,
               s.game_id
        FROM snap_counts s JOIN players p ON p.pfr_id = s.pfr_player_id
        WHERE p.gsis_id = ? AND s.game_id IS NOT NULL
          AND COALESCE(s.offense_snaps, 0) + COALESCE(s.defense_snaps, 0)
              + COALESCE(s.st_snaps, 0) > 0
    )
    GROUP BY season, season_type
"""


def _player_games(gsis_id: str) -> dict[tuple[Any, Any], int]:
    """Games played per (season, season_type) for one gsis_id; {} without the source tables."""
    try:
        rows = _execute(_PLAYER_GAMES_SQL, [gsis_id, gsis_id])
    except (duckdb.CatalogException, duckdb.BinderException):
        # A trimmed database without player_stats or snap_counts (CatalogException), or
        # without a column the query reads, such as players.pfr_id (BinderException).
        return {}
    return {(r["season"], r["season_type"]): r["games"] for r in rows}


def _player_seasons(
    gsis_id: str,
    season: int | None,
    season_from: int | None,
    season_to: int | None,
    season_type: str,
) -> list[dict[str, Any]]:
    """Per season and season type passing, rushing and receiving stats for one gsis_id.

    season_type is REG, POST or ALL; ALL gives the REG row and the POST row of each season.
    """
    sql = _PLAYER_STATS_SQL
    params: list[Any] = [gsis_id]
    season, season_from, season_to = (_int_arg(n, v) for n, v in (
        ("season", season), ("season_from", season_from), ("season_to", season_to)))
    season_type = _season_type_arg(season_type)
    if season is not None:
        sql += " AND season = ?"
        params.append(season)
    if season_from is not None:
        sql += " AND season >= ?"
        params.append(season_from)
    if season_to is not None:
        sql += " AND season <= ?"
        params.append(season_to)
    if season_type != "ALL":
        sql += " AND season_type = ?"
        params.append(season_type)
    sql += (" GROUP BY season, season_type"
            " ORDER BY season, CASE season_type WHEN 'REG' THEN 0 ELSE 1 END, season_type")
    rows = _execute(sql, params)
    games = _player_games(gsis_id) if rows else {}
    return [_season_row(row, games.get((row["season"], row["season_type"]))) for row in rows]


_STAT_BLOCKS = ("passing", "rushing", "receiving")


@_checked
def nfl_player_stats(
    player: str | None = None,
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
    season_type = _season_type_arg(season_type)
    if stat_type is not None and stat_type not in _STAT_BLOCKS:
        return {"error": f"Unknown stat_type: {stat_type}. Use 'passing', 'rushing', or 'receiving'."}
    season, season_from, season_to = (_int_arg(n, v) for n, v in (
        ("season", season), ("season_from", season_from), ("season_to", season_to)))
    try:
        found, early = _resolve_player(
            player, player_id, season=season, season_from=season_from, season_to=season_to)
        if early is not None:
            return early
        if found is None:
            return {"player_id": None, "player_name": player, "seasons": [],
                    "reason": f"No player matches '{player}'. Try nfl_player_lookup."}
        seasons = _player_seasons(found["gsis_id"], season, season_from, season_to, season_type)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}
    if stat_type:
        seasons = [{k: v for k, v in s.items() if k not in _STAT_BLOCKS or k == stat_type}
                   for s in seasons]
    result = {"player_id": found["gsis_id"], "player_name": found["display_name"],
              "seasons": seasons,
              "definitions": _stat_definitions((stat_type,) if stat_type else _STAT_BLOCKS)}
    return _with_reason(result, seasons, "plays",
                        season=season, season_from=season_from, season_to=season_to)


def _compare_teams(
    team1: str,
    team2: str,
    season: int | None,
    season_from: int | None,
    season_to: int | None,
    season_type: str | None,
) -> dict[str, Any]:
    """Team side of nfl_compare: offense and defense rows for each team under one filter."""
    filters = (season, season_from, season_to, season_type)
    _team_stats_filter(*filters)  # validate before any query
    first, second = (_team_codes(t, n, required=True)
                     for t, n in ((team1, "team1"), (team2, "team2")))
    if first[0] == second[0]:
        raise _BadArgument(
            f"'{team1}' and '{team2}' are the same franchise ({first[0]}). Compare two different teams.")
    result: dict[str, Any] = {}
    blocks: dict[str, list[dict]] = {"offense": [], "defense": []}
    for codes in (first, second):
        team = codes[0]
        offense = _team_rows("team_offense_stats", codes, filters)
        defense = _team_rows("team_defense_stats", codes, filters)
        blocks["offense"] += offense
        blocks["defense"] += defense
        entry: dict[str, Any] = {"offense": offense, "defense": defense}
        if not offense and not defense:
            scope = f" in {season}" if season is not None else ""
            entry["reason"] = f"No {season_type} team stats for {team}{scope} with the filters given"
            outside = _range_reason(
                "team_offense_stats", season_col="season_year", season=_int_arg("season", season),
                season_from=_int_arg("season_from", season_from),
                season_to=_int_arg("season_to", season_to))
            if outside:
                entry["reason"] += f". {outside}"
        result[team] = entry
    _add_team_definitions(result, blocks)
    return result


def _compare_players(
    player1: str,
    player2: str,
    season: int | None,
    season_from: int | None,
    season_to: int | None,
    season_type: str | None,
) -> dict[str, Any]:
    """Player side of nfl_compare: each entity is a name or a gsis id, resolved to one player."""
    entries = []
    for arg, entity in (("player1", player1), ("player2", player2)):
        found, early = _resolve_player(
            entity, None, season=season, season_from=season_from, season_to=season_to,
            entity=entity, id_arg=arg)
        if early is not None:
            return early
        if found is None:
            entries.append({"player_id": None, "player_name": entity, "seasons": [],
                            "reason": f"No player matches '{entity}'. Try nfl_player_lookup."})
            continue
        seasons = _player_seasons(found["gsis_id"], season, season_from, season_to, season_type)
        entries.append(_with_reason(
            {"player_id": found["gsis_id"], "player_name": found["display_name"], "seasons": seasons},
            seasons, "plays", season=_int_arg("season", season),
            season_from=_int_arg("season_from", season_from), season_to=_int_arg("season_to", season_to)))
    return {"players": entries, "definitions": _stat_definitions(_STAT_BLOCKS)}


_COMPARE_PAIRS = (
    'Use player1 and player2, or team1 and team2, for example {"player1": "Joe Burrow", '
    '"player2": "Lamar Jackson"} or {"team1": "KC", "team2": "PHI"}. Other names are not '
    "accepted, and a proxy can drop them without an error."
)


def _given(value: Any) -> bool:
    return value is not None and bool(str(value).strip())


@_checked
def nfl_compare(
    player1: str | None = None,
    player2: str | None = None,
    team1: str | None = None,
    team2: str | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    season_type: str | None = None,
) -> Dict[str, Any]:
    """Side-by-side comparison of two players (player1, player2) or two teams (team1, team2).

    Team mode reads the derived team tables for season_type (REG, POST, or ALL).
    """
    season_type = _season_type_arg(season_type)
    pair = [_given(v) for v in (player1, player2, team1, team2)]
    if pair not in ([True, True, False, False], [False, False, True, True]):
        raise _BadArgument(f"{_COMPARE_PAIRS} A call takes exactly one pair, both names of it.")
    try:
        if pair[0]:
            return _compare_players(player1, player2, season, season_from, season_to, season_type)
        return _compare_teams(team1, team2, season, season_from, season_to, season_type)
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


# ── Freshness: data_as_of ─────────────────────────────────────────────────────
# Every tool response carries data_as_of, the newest load time of the tables the tool read.
# The per-table times come from _ingest_metadata and are cached per database file, so only
# the first call on a file reads the metadata. The key holds the resolved path (a symlink
# swap changes it) plus the file's mtime and size (an in-place re-ingest changes those).

_AS_OF_CACHE: dict[tuple[str, int, int], dict[str, str]] = {}
_AS_OF_LOCK = threading.Lock()


def _data_as_of(db_path: str) -> dict[str, str]:
    """table_name -> ISO 8601 UTC time of that table's newest load. db_path is the cache key
    and must be the configured database file; {} when the file or its metadata is unreadable."""
    try:
        resolved = os.path.realpath(db_path)
        st = os.stat(resolved)
    except OSError:
        return {}
    key = (resolved, st.st_mtime_ns, st.st_size)
    with _AS_OF_LOCK:
        cached = _AS_OF_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT table_name, MAX(loaded_at) FROM _ingest_metadata GROUP BY table_name").fetchall()
        times = {name: _iso_utc(loaded) for name, loaded in rows if loaded is not None}
    except duckdb.CatalogException:  # no _ingest_metadata: cache the empty answer, log once
        logger.warning("data_as_of unavailable: the database has no _ingest_metadata table")
        times = {}
    except Exception as e:  # noqa: BLE001 - freshness is extra detail; a tool call must not fail for it
        logger.warning("data_as_of unavailable: %s", e)
        return {}
    with _AS_OF_LOCK:
        for old in [k for k in _AS_OF_CACHE if k[0] == resolved]:
            del _AS_OF_CACHE[old]
        _AS_OF_CACHE[key] = times
    return times


def _as_of(tables: Sequence[str] | None) -> str | None:
    """Newest load time over the given tables (all tables when None). When none of them has a
    metadata row, the newest load of the whole database."""
    times = _data_as_of(str(get_duckdb_path()))
    if not times:
        return None
    chosen = [times[t] for t in tables if t in times] if tables else []
    return max(chosen or times.values())


def _stamped(tables: Sequence[str] | None):
    """Decorator for a tool: add data_as_of to the dict it returns, errors included."""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            if isinstance(result, dict):
                stamp = _as_of(tables)
                if stamp:
                    result["data_as_of"] = stamp
            return result
        return wrapper
    return decorate


def nfl_catalog() -> Dict[str, Any]:
    """List every table in the database: grain, row count, season range, join keys and whether
    it is derived. Also the per-dataset load summary."""
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
        tables = _catalog_tables()
        return {"tables": tables, "total_tables": len(tables), "datasets": rows, "total_datasets": len(rows)}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def _catalog_tables() -> list[dict]:
    """One entry per table in schema 'main' except _ingest_metadata. row_count is a live
    COUNT(*), not the metadata total. One statement reads every table."""
    cols: dict[str, set[str]] = {}
    for r in _execute("SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'main'"):
        cols.setdefault(r["table_name"], set()).add(r["column_name"])
    names = sorted(t for t in cols if t != "_ingest_metadata")
    if not names:
        return []
    parts = []
    for t in names:
        season = "season" if "season" in cols[t] else "season_year" if "season_year" in cols[t] else None
        span = (f"TRY_CAST(MIN({season}) AS INTEGER) AS lo, TRY_CAST(MAX({season}) AS INTEGER) AS hi"
                if season else "NULL::INTEGER AS lo, NULL::INTEGER AS hi")
        parts.append(f"SELECT '{t}' AS t, COUNT(*) AS n, {span} FROM \"{t}\"")
    counted = {r["t"]: (r["n"], r["lo"], r["hi"]) for r in _execute(" UNION ALL ".join(parts))}
    loaded = _data_as_of(str(get_duckdb_path()))
    out = []
    for t in names:
        n, lo, hi = counted[t]
        grain, keys = table_info(t)
        out.append({"table": t, "grain": grain, "row_count": n, "min_season": lo, "max_season": hi,
                    "join_keys": keys, "derived": t in DERIVED_TABLES, "last_loaded": loaded.get(t)})
    return out


@_checked
def nfl_roster(
    team: str | None = None,
    season: int | None = None,
    position: str | None = None,
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
    player: str | None = None,
    player_id: str | None = None,
) -> Dict[str, Any]:
    """
    Look up roster data — who was on a team's roster in a given season, or the roster rows
    of one player (player or player_id). position matches position or depth_chart_position
    exactly. Returns name, gsis_id, position, status, jersey number, experience, and measurables.
    """
    limit, offset = _page_args(limit, offset)
    season = _int_arg("season", season)
    conditions, params = [], []
    clause = _team_filter("team", team)
    if clause:
        conditions.append(clause[0])
        params.extend(clause[1])
    found, early = _one_player(player, player_id, "players", team=_team_codes(team), season=season)
    if early is not None:
        return early
    if found is not None:
        conditions.append("gsis_id = ?")
        params.append(found["gsis_id"])
    if season is not None:
        conditions.append("season = ?")
        params.append(season)
    if position:
        # Coarse codes (DB, DL, OL, LB) sit in position, finer ones (CB, T, ILB) in depth_chart_position.
        conditions.append("(upper(position) = ? OR upper(depth_chart_position) = ?)")
        params.extend([position.strip().upper()] * 2)

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT full_name, gsis_id, position, depth_chart_position, team,
               CAST(season AS INTEGER) AS season, jersey_number, status, years_exp, college,
               height, weight
        FROM rosters
        WHERE {where}
        ORDER BY position, full_name, rowid
    """
    try:
        rows, page = _execute_limited(sql, params, limit, offset)
        return _with_reason(_with_player({"players": rows, **page}, found), rows, "rosters",
                            season=season)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


# season and week of a raw nflverse table. A load can store them as DOUBLE (production
# injuries did: 2024.0, 10.0), so they are cast to return ints.
_SEASON_WEEK = "CAST(season AS INTEGER) AS season, CAST(week AS INTEGER) AS week"
# season_type of a row keyed on game_type (injuries, schedules, snap_counts): REG for a
# regular-season game, POST for any of the four playoff rounds.
_SEASON_TYPE_OF_GAME = (
    "CASE game_type WHEN 'REG' THEN 'REG' WHEN 'WC' THEN 'POST' WHEN 'DIV' THEN 'POST' "
    "WHEN 'CON' THEN 'POST' WHEN 'SB' THEN 'POST' END"
)


@_checked
def nfl_injuries(
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    player: str | None = None,
    report_status: str | None = None,
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
    season_type: str | None = None,
    player_id: str | None = None,
    practice_status: str | None = None,
) -> Dict[str, Any]:
    """
    Look up injury report data. Filter by team, season, week, one player (player or
    player_id), report status (game designation: Out, Questionable, Doubtful, Note) or
    practice status (e.g. 'Full Participation in Practice'); both statuses match a part of
    the stored text, ignoring case. season_type (required) is REG, POST, ALL or one round
    (WC, DIV, CON, SB); each row carries its own season_type (REG or POST).
    """
    game_types = _game_types(season_type)
    limit, offset = _page_args(limit, offset)
    season, week = _int_arg("season", season), _int_arg("week", week)
    conditions, params = [], []
    clause = _team_filter("team", team)
    if clause:
        conditions.append(clause[0])
        params.extend(clause[1])
    found, early = _one_player(player, player_id, "injuries", team=_team_codes(team), season=season)
    if early is not None:
        return early
    if found is not None:
        conditions.append("gsis_id = ?")
        params.append(found["gsis_id"])
    if season is not None:
        conditions.append("season = ?")
        params.append(season)
    if week is not None:
        conditions.append("week = ?")
        params.append(week)
    clause = _in_clause("game_type", game_types)
    conditions.append(clause[0])
    params.extend(clause[1])
    for column, value in (("report_status", report_status), ("practice_status", practice_status)):
        if value:
            conditions.append(f"{column} ILIKE ? ESCAPE '\\'")
            params.append(_like_pattern(str(value)))

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT full_name, gsis_id, position, team, {_SEASON_WEEK},
               {_SEASON_TYPE_OF_GAME} AS season_type,
               report_primary_injury, report_secondary_injury, report_status,
               practice_primary_injury, practice_secondary_injury, practice_status
        FROM injuries
        WHERE {where}
        ORDER BY season, week, team, full_name, rowid
    """
    try:
        rows, page = _execute_limited(sql, params, limit, offset)
        return _with_reason(_with_player({"injuries": rows, **page}, found), rows, "injuries",
                            season=season, week=week)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def _team_side(row: dict, codes: Sequence[str]) -> dict:
    """A schedules row plus the game from the side of the franchise with `codes`.

    team is the code that row stores (OAK on a 2019 row, LV on a 2020 row). location is
    home, away, or neutral when schedules.location says Neutral. outcome is W, L or T for
    that team, None while the game has no score.
    """
    site = row.pop("_site")
    home = row["home_team"] in codes
    side, other = ("home", "away") if home else ("away", "home")
    team_score, opponent_score = row[f"{side}_score"], row[f"{other}_score"]
    outcome = None
    if team_score is not None and opponent_score is not None:
        outcome = "W" if team_score > opponent_score else "L" if team_score < opponent_score else "T"
    return {**row, "team": row[f"{side}_team"], "opponent": row[f"{other}_team"],
            "location": "neutral" if site == "Neutral" else side,
            "team_score": team_score, "opponent_score": opponent_score, "outcome": outcome}


@_checked
def nfl_schedule(
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    season_type: str | None = None,
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Look up game schedules and results. Filter by team, season, week, or season type
    (required: REG, POST, ALL or one round WC, DIV, CON, SB). Each game carries its
    season_type (REG or POST) and its game_type (the round).
    Returns scores, spread lines, weather, stadium, and coaching info. result is the home
    margin (home_score - away_score). With team, each row also gives the game from that
    team's side (see _team_side).
    """
    game_types = _game_types(season_type)
    limit, offset = _page_args(limit, offset)
    season, week = _int_arg("season", season), _int_arg("week", week)
    conditions, params = [], []
    codes = _team_codes(team)
    if codes:
        home, away = _in_clause("home_team", codes), _in_clause("away_team", codes)
        conditions.append(f"({home[0]} OR {away[0]})")
        params.extend([*home[1], *away[1]])
    if season is not None:
        conditions.append("season = ?")
        params.append(season)
    if week is not None:
        conditions.append("week = ?")
        params.append(week)
    clause = _in_clause("game_type", game_types)
    conditions.append(clause[0])
    params.extend(clause[1])

    where = " AND ".join(conditions)
    sql = f"""
        SELECT game_id, {_SEASON_WEEK}, {_SEASON_TYPE_OF_GAME} AS season_type, game_type,
               gameday, gametime,
               away_team, home_team, away_score, home_score, result,
               overtime, spread_line, total_line, div_game,
               roof, surface, temp, wind,
               away_coach, home_coach, referee, stadium{", location AS _site" if codes else ""}
        FROM schedules
        WHERE {where}
        ORDER BY season, week, gameday, game_id
    """
    try:
        rows, page = _execute_limited(sql, params, limit, offset)
        if codes:
            rows = [_team_side(row, codes) for row in rows]
        return _with_reason({"games": rows, **page}, rows, "schedules",
                            season=season, week=week)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


# ── Team records (1970–present) ───────────────────────────────────────────────

_RECORD_POINTS = " points_for, points_against, point_diff,"
# A database whose team_records predates schema version 3 still answers win-loss questions.
_STALE_RECORDS_WARNING = (
    "team_records was built by an older nfl-mcp and has no points_for, points_against or "
    "point_diff, so points are left out. Points appear after the next database update."
)

@_checked
def nfl_team_records(
    team: str | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    PREFERRED for win-loss / standings questions. Regular-season records per franchise,
    1970 to present: final records, and the record to date for a season still in progress.
    points_for, points_against and point_diff are regular-season points (to date in a season
    in progress; NULL for the ESPN seasons before 1999).
    Needs team, season or a season_from/season_to range. Teams are given by their CURRENT abbreviation, so the
    Raiders are 'LV' in every season; each row also carries `club` (the name the franchise
    played that season under) and `source_team` (the code the source used).

    Records before 1999 come from ESPN's opt-in ingest (run 'nfl-mcp ingest --pre1999');
    a franchise ESPN's archive does not cover is reported as record_not_available rather
    than estimated.
    """
    limit, offset = _page_args(limit, offset)
    season, season_from, season_to = (_int_arg(n, v) for n, v in (
        ("season", season), ("season_from", season_from), ("season_to", season_to)))
    if team is None and season is None and season_from is None and season_to is None:
        return {"error": "Provide team, season, or a season_from/season_to range."}

    codes = _team_codes(team)
    key = codes[0] if codes else None

    if key and season is not None:
        note = availability(key, season)
        if note:
            return {"team": key, "season": season, "records": [], "count": 0,
                    "truncated": False, **note}

    conditions: list[str] = []
    params: list[Any] = []
    if key:
        conditions.append("team = ?")
        params.append(key)
    if season is not None:
        conditions.append("season = ?")
        params.append(season)
    if season_from is not None:
        conditions.append("season >= ?")
        params.append(season_from)
    if season_to is not None:
        conditions.append("season <= ?")
        params.append(season_to)
    where = " AND ".join(conditions) or "1=1"

    def sql(points: str) -> str:
        return f"""
            SELECT season, team, source_team, games, wins, losses, ties,{points} source
            FROM team_records
            WHERE {where}
            ORDER BY season DESC, team, rowid
        """
    warning = None
    try:
        try:
            rows, page = _execute_limited(sql(_RECORD_POINTS), params, limit, offset)
        except duckdb.BinderException as e:
            if 'column "points_for"' not in str(e):  # the message also quotes the SQL text
                raise
            # team_records built before schema version 3: the records, without points.
            rows, page = _execute_limited(sql(""), params, limit, offset)
            warning = _STALE_RECORDS_WARNING
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}

    for row in rows:
        row["club"] = era_name(row["season"], row["team"])

    result: Dict[str, Any] = {"records": rows, **page}
    if warning:
        result["warning"] = warning
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


@_checked
def nfl_snap_counts(
    player: str | None = None,
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    position: str | None = None,
    player_id: str | None = None,
    season_type: str | None = None,
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Look up snap count data — how many snaps a player or team unit played.
    Returns offense, defense, and special teams snap counts and percentages.
    player resolves through players with the one name rule (exact full, last or short name)
    and player_id is a gsis_id; either maps through players.pfr_id to
    snap_counts.pfr_player_id, so one player's rows never mix with a namesake's. season_type
    is required: REG, POST (the four playoff rounds), ALL (REG and POST rows, never combined),
    or one round WC, DIV, CON, SB. offense_pct, defense_pct and st_pct are percents from 0 to 100.
    """
    game_types = _game_types(season_type)
    limit, offset = _page_args(limit, offset)
    season, week = _int_arg("season", season), _int_arg("week", week)
    conditions, params = [], []
    found, early = _one_player(player, player_id, "snap_counts", team=_team_codes(team), season=season)
    if early is not None:
        return early
    if found is not None:
        if not found["pfr_id"]:
            return {"snap_counts": [], "count": 0, "truncated": False,
                    "player_id": found["gsis_id"],
                    "reason": f"No PFR id on file for {found['display_name']}, "
                              "so snap counts cannot be matched by id."}
        conditions.append("pfr_player_id = ?")
        params.append(found["pfr_id"])
    try:
        clause = _team_filter("team", team)
        if clause:
            conditions.append(clause[0])
            params.extend(clause[1])
        if season is not None:
            conditions.append("season = ?")
            params.append(season)
        if week is not None:
            conditions.append("week = ?")
            params.append(week)
        clause = _in_clause("game_type", game_types)
        conditions.append(clause[0])
        params.extend(clause[1])
        if position:
            conditions.append("upper(position) = ?")
            params.append(position.strip().upper())

        where = " AND ".join(conditions)
        sql = f"""
            SELECT player, pfr_player_id, position, team, opponent, {_SEASON_WEEK},
                   {_SEASON_TYPE_OF_GAME} AS season_type, game_type,
                   offense_snaps, ROUND(100 * offense_pct, 1) AS offense_pct, defense_snaps,
                   ROUND(100 * defense_pct, 1) AS defense_pct,
                   st_snaps, ROUND(100 * st_pct, 1) AS st_pct
            FROM snap_counts
            WHERE {where}
            ORDER BY season, week, team, offense_snaps DESC NULLS LAST, player, pfr_player_id, rowid
        """
        rows, page = _execute_limited(sql, params, limit, offset)
        return _with_reason(_with_player({"snap_counts": rows, **page}, found), rows, "snap_counts",
                            season=season, week=week)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


# season_type of an ff_opportunity row (aliased o, joined to schedules as g): REG when the game
# is a regular-season game, POST for any playoff round. Without a schedules row the week decides.
_FF_SEASON_TYPE = ("CASE WHEN g.game_type IS NULL THEN CASE WHEN o.week > 18 THEN 'POST' ELSE 'REG' END "
                   "WHEN g.game_type = 'REG' THEN 'REG' ELSE 'POST' END")


@_checked
def nfl_fantasy_opportunity(
    player: str | None = None,
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    position: str | None = None,
    player_id: str | None = None,
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
    season_type: str | None = None,
) -> Dict[str, Any]:
    """
    Look up fantasy football opportunity data — target share, air yards share,
    carry share, and opportunity scores per player per week. Available 2006–present.
    season_type (required) is REG, POST, ALL or one round (WC, DIV, CON, SB); the round of a
    game comes from schedules through game_id. Each row carries its season_type (REG or POST).
    """
    season_type = _season_type_arg(season_type, _GAME_SEASON_TYPES)
    limit, offset = _page_args(limit, offset)
    season, week = _int_arg("season", season), _int_arg("week", week)
    team_clause = _team_filter("posteam", team)
    team = _team_codes(team)  # codes for the player resolver
    conditions, params = [], []
    resolved, early = _one_player(player, player_id, "fantasy_opportunity", team=team, season=season)
    if early is not None:
        return early
    if resolved is not None:
        conditions.append("player_id = ?")
        params.append(resolved["gsis_id"])
    if team_clause:
        conditions.append(team_clause[0])
        params.extend(team_clause[1])
    if season is not None:
        conditions.append("season = ?")
        params.append(str(season))
    if week is not None:
        conditions.append("week = ?")
        params.append(week)
    if position:
        conditions.append("upper(position) = ?")
        params.append(position.strip().upper())
    if season_type in _PLAYOFF_ROUNDS:
        conditions.append("g.game_type = ?")
        params.append(season_type)
    elif season_type != "ALL":
        conditions.append(f"{_FF_SEASON_TYPE} = ?")
        params.append(season_type)

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT CAST(season AS INTEGER) AS season, CAST(week AS INTEGER) AS week,
               {_FF_SEASON_TYPE} AS season_type, posteam, full_name, position, player_id,
               rec_attempt, rush_attempt, rec_air_yards,
               receptions, receptions_exp, rec_yards_gained, rush_yards_gained,
               rec_touchdown, rush_touchdown,
               total_fantasy_points, total_fantasy_points_exp, total_fantasy_points_diff,
               rec_attempt_team, rush_attempt_team
        FROM ff_opportunity o
        LEFT JOIN (SELECT game_id, ANY_VALUE(game_type) AS game_type FROM schedules GROUP BY game_id) g
          ON g.game_id = o.game_id
        WHERE {where}
        ORDER BY season DESC, week DESC, total_fantasy_points_exp DESC NULLS LAST, player_id, o.rowid
    """
    try:
        rows, page = _execute_limited(sql, params, limit, offset)
        return _with_reason(_with_player({"fantasy_opportunity": rows, **page}, resolved),
                            rows, "ff_opportunity", season=season, week=week)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


# FantasyPros position groups of a players.position, for ranking rows that ff_playerids does
# not link to a gsis_id. An edge rusher is DL in one source and LB in the other.
_RANKING_POSITIONS = {
    "CB": ("DB",), "S": ("DB",), "SAF": ("DB",), "FS": ("DB",), "SS": ("DB",), "DB": ("DB",),
    "DT": ("DL",), "NT": ("DL",), "DL": ("DL",), "DE": ("DL", "LB"),
    "LB": ("LB", "DL"), "OLB": ("LB", "DL"), "ILB": ("LB", "DL"), "MLB": ("LB", "DL"),
    "FB": ("RB",),
}
_PLAYER_NAME_FORMS = [
    _fold_sql("p.display_name", True),
    _fold_sql("(p.first_name || ' ' || p.last_name)", True),
    _fold_sql("(p.common_first_name || ' ' || p.last_name)", True),
]


def _ranking_ids(found: dict, table: str, id_col: str, name_col: str) -> tuple[list[int], str | None]:
    """FantasyPros ids of one resolved player's ranking rows. Returns (ids, reason).

    ff_playerids maps fantasypros_id to gsis_id, so a player with a mapping gets the rows of
    his ids and no other. For a player with none, a row counts when it has no mapping either,
    its folded name (suffix dropped) is one of his names and its position is in his position
    group ("Byron Murphy II" the DL is not "Byron Murphy" the CB). Rows of several ids are
    narrowed to a team he played for; when several ids are left, none is used and the
    reason says why.
    """
    mapped = sorted(r["fantasypros_id"] for r in _execute(
        "SELECT DISTINCT fantasypros_id FROM ff_playerids "
        "WHERE gsis_id = ? AND fantasypros_id IS NOT NULL", [found["gsis_id"]]))
    if mapped:
        return mapped, None
    position = str(found.get("position") or "").upper()
    pos_clause, pos_params = _in_clause("upper(t.pos)", _RANKING_POSITIONS.get(position, (position,)))
    rows = _execute(f"""
        SELECT DISTINCT t.{id_col} AS id, t.team
        FROM {table} t
        WHERE t.{id_col} IS NOT NULL AND {pos_clause}
          AND NOT EXISTS (SELECT 1 FROM ff_playerids x
                          WHERE x.fantasypros_id = t.{id_col} AND x.gsis_id IS NOT NULL)
          AND {_fold_sql('t.' + name_col, True)} IN (
              SELECT unnest([{', '.join(_PLAYER_NAME_FORMS)}]) FROM players p WHERE p.gsis_id = ?)
    """, [*pos_params, found["gsis_id"]])
    ids = sorted({r["id"] for r in rows})
    if len(ids) > 1:
        teams = set(found.get("teams") or [])
        ids = sorted({r["id"] for r in rows if teams & set(team_codes(r["team"]) or ())}) or ids
    if len(ids) > 1:
        return [], (f"{table} ranks several players named like {found['display_name']} "
                    "(FantasyPros ids " + ", ".join(map(str, ids)) + ") that ff_playerids does "
                    "not link to a gsis_id, so none of them is taken to be this player.")
    return ids, None


@_checked
def nfl_fantasy_rankings(
    player: str | None = None,
    position: str | None = None,
    team: str | None = None,
    scope: str = "draft",
    ranking_set: str | None = None,
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
    player_id: str | None = None,
) -> Dict[str, Any]:
    """
    Look up fantasy football expert consensus rankings (ECR). Two snapshots:
    scope='draft' (redraft, dynasty and weekly ranking sets, ff_rankings_draft) and
    scope='week' (current-week start/sit rankings, ff_rankings_week). Both reflect the
    latest scrape — they are not historical/seasonal. Lower ECR = ranked higher.
    team 'FA' is a free agent (no team).
    """
    limit, offset = _page_args(limit, offset)
    scope = (scope or "draft").lower()
    if scope not in ("draft", "week"):
        return {"error": "scope must be 'draft' or 'week'"}

    conditions, params = [], []
    if scope == "draft":
        table = "ff_rankings_draft"
        player_col, set_col, id_col = "player", "page_type", "id"
        select = (
            "player, pos AS position, team, ecr, sd, best, worst, bye, "
            "page_type AS ranking_set, ecr_type, scrape_date"
        )
    else:
        table = "ff_rankings_week"
        player_col, set_col, id_col = "player_name", "page", "fantasypros_id"
        select = (
            "player_name AS player, pos AS position, team, rank, ecr, sd, best, worst, "
            "pos_rank, player_opponent AS opponent, player_bye_week AS bye, "
            "start_sit_grade, note, page AS ranking_set, scrape_date"
        )

    found, reason = None, None
    if player or player_id:
        # The ranking tables have no gsis id. The name resolves through players with the one
        # name rule, and the rows are those of the player's FantasyPros ids (_ranking_ids). A
        # name no player has (a team defense such as 'Kansas City Chiefs') must equal a
        # stored name, folded and with the suffix dropped.
        try:
            found, early = _resolve_player(player, player_id)
            if found is not None:
                ids, reason = _ranking_ids(found, table, id_col, player_col)
        except (duckdb.Error, ValueError, TimeoutError) as e:
            logger.exception("Tool error")
            return {"error": str(e)}
        if early is not None:
            return early
        if found is None:
            conditions.append(f"{_fold_sql(player_col, True)} = ?")
            params.append(_fold(str(player), strip_suffix=True))
        elif ids:
            conditions.append(_in_clause(id_col, ids)[0])
            params.extend(ids)
        else:
            conditions.append("FALSE")
    if position:
        conditions.append("upper(pos) = ?")
        params.append(position.strip().upper())
    if isinstance(team, str) and team.strip().upper() == "FA":
        conditions.append("team = 'FA'")  # free agents: the rankings' own code, not a franchise
    else:
        team_clause = _team_filter("team", team)
        if team_clause:
            conditions.append(team_clause[0])
            params.extend(team_clause[1])
    if ranking_set:
        conditions.append(f"{set_col} ILIKE ? ESCAPE '\\'")
        params.append(_like_pattern(str(ranking_set)))

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT {select}
        FROM {table}
        WHERE {where}
        ORDER BY ecr ASC NULLS LAST, {player_col}, rowid
    """
    try:
        rows, page = _execute_limited(sql, params, limit, offset)
        result = _with_player({"fantasy_rankings": rows, "scope": scope, **page}, found)
        if not rows and (player or player_id):
            if found is None:
                reason = (f"No {table} row matches '{player}' with these filters, and no "
                          "player has that name. Try nfl_player_lookup.")
            result["reason"] = reason or (
                f"No {table} row matches {found['display_name']} ({found['gsis_id']}) with "
                "these filters. The rankings hold only the players FantasyPros ranks in its "
                "latest scrape.")
        return result
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


@_checked
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
    season_type (required) is REG, POST, ALL or one round (WC, DIV, CON, SB). ALL returns one
    result per season type in season_types and no combined total.
    """
    season_type = _season_type_arg(season_type, _GAME_SEASON_TYPES)
    conditions = ["f.nflverse_game_id = p.game_id", "f.nflverse_play_id = p.play_id",
                  "p.play_type IN ('pass', 'run')"]
    params: list[Any] = []
    season, season_from, season_to = (_int_arg(n, v) for n, v in (
        ("season", season), ("season_from", season_from), ("season_to", season_to)))
    week = _int_arg("week", week)
    for column, value, label in (("p.posteam", team, "team"), ("p.defteam", opponent, "opponent")):
        clause = _team_filter(column, value, label)
        if clause:
            conditions.append(clause[0])
            params.extend(clause[1])
    team = _team_codes(team)  # codes for the player resolver
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
    if season is not None:
        conditions.append("p.season = ?")
        params.append(season)
    if season_from is not None:
        conditions.append("p.season >= ?")
        params.append(season_from)
    if season_to is not None:
        conditions.append("p.season <= ?")
        params.append(season_to)
    if week is not None:
        conditions.append("p.week = ?")
        params.append(week)
    clause = _plays_season_type(season_type, "p.")
    if clause:
        conditions.append(clause[0])
        params.extend(clause[1])

    by_type = season_type == "ALL"
    where = " AND ".join(conditions)
    sql = f"""
        SELECT
            {"p.season_type," if by_type else ""}
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
        {"GROUP BY p.season_type ORDER BY CASE p.season_type WHEN 'REG' THEN 0 ELSE 1 END" if by_type else ""}
    """
    try:
        rows = _execute(sql, params if params else None)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}

    rows = [r for r in rows if r.get("total_plays")]
    if not rows:
        return _with_reason(_with_player({"charting": {}, "total_plays": 0}, resolved), None, "plays",
                            season=season, season_from=season_from, season_to=season_to, week=week)
    if by_type:
        # One result per season type; a total across REG and POST is never computed.
        return _with_player(
            {"season_types": [{"season_type": r["season_type"], "total_plays": r["total_plays"],
                               "charting": _charting_block(r)} for r in rows]}, resolved)
    return _with_player({"charting": _charting_block(rows[0]), "total_plays": rows[0]["total_plays"]},
                        resolved)


def _charting_block(agg: dict[str, Any]) -> dict[str, Any]:
    """The charting rates of one aggregate row of nfl_ftn_charting."""
    total = agg["total_plays"]

    def _pct(n: Any) -> float:
        return round(100.0 * (n or 0) / total, 1)

    def _avg(v: Any) -> float | None:
        return round(v, 2) if v is not None else None

    return {
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
    }


@_checked
def nfl_td_luck(
    player: str | None = None,
    position: str | None = None,
    team: str | None = None,
    season: int | None = None,
    limit: int = _DEFAULT_LIMIT,
    player_id: str | None = None,
    offset: int = 0,
    season_type: str | None = None,
) -> Dict[str, Any]:
    """
    Touchdown luck — actual vs expected TDs per player-season from the
    player_td_luck table. Negative total_td_luck_score = scored fewer TDs than
    expected (positive-regression / "unlucky" candidate); positive = over-performed.
    Default sort is most unlucky first (total_td_luck_score ASC). season_type (required) is
    REG, POST or ALL; each row has a season_type, and ALL returns the REG and POST rows
    separately, never summed.
    """
    season_type = _season_type_arg(season_type)
    limit, offset = _page_args(limit, offset)
    season = _int_arg("season", season)
    team_clause = _team_filter("team", team)
    team = _team_codes(team)  # codes for the player resolver
    conditions, params = [], []
    resolved, early = _one_player(player, player_id, "td_luck", team=team, season=season)
    if early is not None:
        return early
    if resolved is not None:
        conditions.append("player_id = ?")
        params.append(resolved["gsis_id"])
    if position:
        conditions.append("upper(position) = ?")
        params.append(position.strip().upper())
    if team_clause:
        conditions.append(team_clause[0])
        params.extend(team_clause[1])
    if season is not None:
        conditions.append("season = ?")
        params.append(season)
    if season_type != "ALL":
        conditions.append("season_type = ?")
        params.append(season_type)

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player_id, full_name, position, team, season, season_type,
               rec_td, rec_td_exp, rec_td_luck,
               rush_td, rush_td_exp, rush_td_luck,
               total_td_luck_score
        FROM player_td_luck
        WHERE {where}
        ORDER BY total_td_luck_score ASC NULLS LAST, player_id, season, season_type DESC, rowid
    """
    try:
        rows, page = _season_typed_page(sql, params, limit, offset)
        return _with_reason(_with_player({"td_luck": rows, **page}, resolved),
                            rows, "player_td_luck", season=season)
    except _StaleDerivedTables:
        return dict(_STALE_PLAYER_ERROR)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


@_checked
def nfl_role_trend(
    player: str | None = None,
    position: str | None = None,
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    min_snap_pct: float | None = None,
    limit: int = _DEFAULT_LIMIT,
    player_id: str | None = None,
    offset: int = 0,
    season_type: str | None = None,
) -> Dict[str, Any]:
    """
    Rolling 3-week role trend — weekly snap %, target/carry/air-yards share with a
    trailing 3-week average and the current-week delta vs that average (from
    player_role_trend). Surfaces players whose usage is rising or fading. Default
    sort is biggest snap-share gain first (snap_pct_delta DESC). season_type (required) is
    REG, POST, ALL or one round (WC, DIV, CON, SB). The 3-week window never mixes REG and
    POST weeks; each row has a season_type (REG or POST). Snap and share values are
    percents from 0 to 100.
    """
    season_type = _season_type_arg(season_type, _GAME_SEASON_TYPES)
    limit, offset = _page_args(limit, offset)
    season, week = _int_arg("season", season), _int_arg("week", week)
    min_snap_pct = _num_arg("min_snap_pct", min_snap_pct)
    team_clause = _team_filter("team", team)
    team = _team_codes(team)  # codes for the player resolver
    conditions, params = [], []
    resolved, early = _one_player(player, player_id, "role_trend", team=team, season=season)
    if early is not None:
        return early
    if resolved is not None:
        conditions.append("player_id = ?")
        params.append(resolved["gsis_id"])
    if position:
        conditions.append("upper(position) = ?")
        params.append(position.strip().upper())
    if team_clause:
        conditions.append(team_clause[0])
        params.extend(team_clause[1])
    if season is not None:
        conditions.append("season = ?")
        params.append(season)
    if week is not None:
        conditions.append("week = ?")
        params.append(week)
    if min_snap_pct is not None:
        conditions.append("snap_pct >= ?")
        params.append(min_snap_pct)
    if season_type in _PLAYOFF_ROUNDS:
        # The derived table keeps REG and POST only; a round is the schedules game_type of the week.
        conditions.append(
            "EXISTS (SELECT 1 FROM ff_opportunity o JOIN schedules g ON g.game_id = o.game_id "
            "WHERE o.player_id = player_role_trend.player_id "
            "AND CAST(o.season AS INTEGER) = player_role_trend.season "
            "AND CAST(o.week AS INTEGER) = player_role_trend.week AND g.game_type = ?)")
        params.append(season_type)
    elif season_type != "ALL":
        conditions.append("season_type = ?")
        params.append(season_type)

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player_id, full_name, position, team, season, season_type, week,
               snap_pct, target_share_pct, carry_share_pct, air_yards_share_pct,
               snap_pct_3wk, target_share_pct_3wk, carry_share_pct_3wk, air_yards_share_pct_3wk,
               snap_pct_delta, target_share_pct_delta, carry_share_pct_delta, air_yards_share_pct_delta
        FROM player_role_trend
        WHERE {where}
        ORDER BY snap_pct_delta DESC NULLS LAST, player_id, season, week, rowid
    """
    try:
        rows, page = _season_typed_page(sql, params, limit, offset)
        return _with_reason(_with_player({"role_trend": rows, **page}, resolved),
                            rows, "player_role_trend", season=season, week=week)
    except _StaleDerivedTables:
        return dict(_STALE_PLAYER_ERROR)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


@_checked
def nfl_separation_opportunity(
    player: str | None = None,
    position: str | None = None,
    team: str | None = None,
    season: int | None = None,
    regression_candidate: bool | None = None,
    limit: int = _DEFAULT_LIMIT,
    player_id: str | None = None,
    offset: int = 0,
    season_type: str | None = None,
) -> Dict[str, Any]:
    """
    Separation-adjusted opportunity — joins Next Gen Stats receiving separation/YAC
    to fantasy opportunity per player-season (from player_separation_opportunity).
    Flags receivers getting open (avg_separation > 2.5) who under-produced
    (fp_diff_per_game < -1.5 and td_luck < -1.0) as regression_candidate. Next Gen
    Stats only goes back to 2016. Default sort is most under-producing first
    (fp_diff_per_game ASC). season_type (required) is REG, POST or ALL; each row has a
    season_type, and ALL returns the REG and POST rows separately, never summed.
    """
    season_type = _season_type_arg(season_type)
    limit, offset = _page_args(limit, offset)
    season = _int_arg("season", season)
    if season is not None and season < 2016:
        return {"error": "Next Gen Stats receiving data is only available from 2016 onward"}

    regression_candidate = _bool_arg("regression_candidate", regression_candidate, default=None)
    team_clause = _team_filter("team", team)
    team = _team_codes(team)  # codes for the player resolver
    conditions, params = [], []
    resolved, early = _one_player(player, player_id, "separation_opportunity", team=team, season=season)
    if early is not None:
        return early
    if resolved is not None:
        conditions.append("player_id = ?")
        params.append(resolved["gsis_id"])
    if position:
        conditions.append("upper(position) = ?")
        params.append(position.strip().upper())
    if team_clause:
        conditions.append(team_clause[0])
        params.extend(team_clause[1])
    if season is not None:
        conditions.append("season = ?")
        params.append(season)
    if regression_candidate is not None:
        conditions.append("regression_candidate = ?")
        params.append(regression_candidate)
    if season_type != "ALL":
        conditions.append("season_type = ?")
        params.append(season_type)

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player_id, full_name, position, team, season, season_type, games,
               fp_diff, fp_diff_per_game, td_luck, catch_luck,
               target_share_pct, air_yards_share_pct,
               avg_separation, avg_yac_above_expectation, catch_percentage,
               regression_candidate
        FROM player_separation_opportunity
        WHERE {where}
        ORDER BY fp_diff_per_game ASC NULLS LAST, player_id, season, season_type DESC, rowid
    """
    try:
        rows, page = _season_typed_page(sql, params, limit, offset)
        return _with_reason(_with_player({"separation_opportunity": rows, **page}, resolved),
                            rows, "player_separation_opportunity", season=season)
    except _StaleDerivedTables:
        return dict(_STALE_PLAYER_ERROR)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


@_checked
def nfl_drop_rate(
    player: str | None = None,
    team: str | None = None,
    season: int | None = None,
    min_targets: int | None = None,
    limit: int = _DEFAULT_LIMIT,
    player_id: str | None = None,
    offset: int = 0,
    season_type: str | None = None,
) -> Dict[str, Any]:
    """
    Drop rate — share of catchable targets dropped per receiver-season from FTN
    charting (2022+) joined to plays (player_drop_rate). Also reports contested
    targets and created receptions. The player column holds nflverse short names
    ('J.Jefferson'); a full name in `player` resolves through players and filters by id.
    Default sort is highest drop rate first (drop_rate_pct DESC). season_type (required) is
    REG, POST or ALL; each row has a season_type, and ALL returns the REG and POST rows
    separately, never summed.
    """
    season_type = _season_type_arg(season_type)
    limit, offset = _page_args(limit, offset)
    season = _int_arg("season", season)
    if season is not None and season < 2022:
        return {"error": "FTN charting (drop) data is only available from 2022 onward"}

    min_targets = _int_arg("min_targets", min_targets)
    team_clause = _team_filter("team", team)
    team = _team_codes(team)  # codes for the player resolver
    conditions, params = [], []
    resolved, early = _one_player(player, player_id, "drop_rate", team=team, season=season)
    if early is not None:
        return early
    if resolved is not None:
        conditions.append("player_id = ?")
        params.append(resolved["gsis_id"])
    if team_clause:
        conditions.append(team_clause[0])
        params.extend(team_clause[1])
    if season is not None:
        conditions.append("season = ?")
        params.append(season)
    if season_type != "ALL":
        conditions.append("season_type = ?")
        params.append(season_type)
    if min_targets is not None:
        conditions.append("catchable_targets >= ?")
        params.append(min_targets)

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player_id, player, team, season, season_type,
               catchable_targets, drops, drop_rate_pct,
               contested_targets, created_receptions
        FROM player_drop_rate
        WHERE {where}
        ORDER BY drop_rate_pct DESC NULLS LAST, player_id, season, season_type DESC, rowid
    """
    try:
        rows, page = _season_typed_page(sql, params, limit, offset)
        return _with_reason(_with_player({"drop_rate": rows, **page}, resolved),
                            rows, "player_drop_rate", season=season)
    except _StaleDerivedTables:
        return dict(_STALE_PLAYER_ERROR)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


@_checked
def nfl_contract_value(
    player: str | None = None,
    position: str | None = None,
    team: str | None = None,
    season: int | None = None,
    min_apy: float | None = None,
    max_apy: float | None = None,
    limit: int = _DEFAULT_LIMIT,
    player_id: str | None = None,
    offset: int = 0,
    season_type: str | None = None,
) -> Dict[str, Any]:
    """
    Contract value efficiency — fantasy points per $M of average per year (APY),
    using each player's contract joined to seasonal fantasy production
    (player_contract_value). apy and cap_pct reflect the contract in force that season.
    Default sort is best value first (fp_per_million DESC). season_type (required) is REG, POST
    or ALL; each row has a season_type, and ALL returns the REG and POST rows separately, never
    summed.
    """
    season_type = _season_type_arg(season_type)
    limit, offset = _page_args(limit, offset)
    season = _int_arg("season", season)
    min_apy, max_apy = _num_arg("min_apy", min_apy), _num_arg("max_apy", max_apy)
    team_clause = _team_filter("team", team)
    team = _team_codes(team)  # codes for the player resolver
    conditions, params = [], []
    resolved, early = _one_player(player, player_id, "contract_value", team=team, season=season)
    if early is not None:
        return early
    if resolved is not None:
        conditions.append("player_id = ?")
        params.append(resolved["gsis_id"])
    if position:
        conditions.append("upper(position) = ?")
        params.append(position.strip().upper())
    if team_clause:
        conditions.append(team_clause[0])
        params.extend(team_clause[1])
    if season is not None:
        conditions.append("season = ?")
        params.append(season)
    if min_apy is not None:
        conditions.append("apy >= ?")
        params.append(min_apy)
    if max_apy is not None:
        conditions.append("apy <= ?")
        params.append(max_apy)
    if season_type != "ALL":
        conditions.append("season_type = ?")
        params.append(season_type)

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player_id, full_name, position, team, season, season_type,
               apy, cap_pct, total_fp, total_fp_exp, fp_per_million
        FROM player_contract_value
        WHERE {where}
        ORDER BY fp_per_million DESC NULLS LAST, player_id, season, season_type DESC, rowid
    """
    try:
        rows, page = _season_typed_page(sql, params, limit, offset)
        return _with_reason(_with_player({"contract_value": rows, **page}, resolved),
                            rows, "player_contract_value", season=season)
    except _StaleDerivedTables:
        return dict(_STALE_PLAYER_ERROR)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


@_checked
def nfl_injury_return(
    injury_type: str | None = None,
    position: str | None = None,
    week_post_return: int | None = None,
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Injury return curve — post-return snap-share recovery, expressed as a percent of
    each player's pre-injury baseline, at +1..+8 weeks after returning from an 'Out'
    spell, bucketed by normalized injury type and position (injury_return_curve).
    100 means fully back to baseline usage. Default sort is by weeks since return
    (week_post_return ASC). sample_size shows how many player-spells back each point.
    """
    limit, offset = _page_args(limit, offset)
    week_post_return = _int_arg("week_post_return", week_post_return)
    conditions, params = [], []
    if injury_type:
        conditions.append("injury_type ILIKE ? ESCAPE '\\'")
        params.append(_like_pattern(str(injury_type)))
    if position:
        conditions.append("position = ?")
        params.append(position.upper())
    if week_post_return is not None:
        conditions.append("week_post_return = ?")
        params.append(week_post_return)

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT injury_type, position, week_post_return,
               avg_snap_pct_recovery, median_snap_pct_recovery, sample_size
        FROM injury_return_curve
        WHERE {where}
        ORDER BY week_post_return ASC, injury_type, position, rowid
    """
    try:
        rows, page = _execute_limited(sql, params, limit, offset)
        return {"injury_return": rows, **page}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


# Tables each tool reads (None: every table). data_as_of is the newest load time among them.
# A tool that resolves a player name also reads players and rosters.
_PLAYER_TABLES = ("players", "rosters")
_TOOL_TABLES: dict[str, Sequence[str] | None] = {
    "nfl_schema": None, "nfl_status": None, "nfl_catalog": None, "nfl_query": None,
    "nfl_search_plays": ("plays", *_PLAYER_TABLES),
    "nfl_team_stats": ("team_offense_stats", "team_defense_stats", "situational_stats"),
    "nfl_player_stats": ("plays", "player_stats", "snap_counts", *_PLAYER_TABLES),
    "nfl_player_lookup": _PLAYER_TABLES,
    "nfl_compare": ("team_offense_stats", "team_defense_stats", "plays", "player_stats",
                    "snap_counts", *_PLAYER_TABLES),
    "nfl_roster": ("rosters",),
    "nfl_injuries": ("injuries",),
    "nfl_schedule": ("schedules",),
    "nfl_snap_counts": ("snap_counts", *_PLAYER_TABLES),
    "nfl_fantasy_opportunity": ("ff_opportunity", *_PLAYER_TABLES),
    "nfl_fantasy_rankings": ("ff_rankings_draft", "ff_rankings_week", "ff_playerids",
                             *_PLAYER_TABLES),
    "nfl_ftn_charting": ("ftn_charting", "plays", *_PLAYER_TABLES),
    "nfl_td_luck": ("player_td_luck", *_PLAYER_TABLES),
    "nfl_role_trend": ("player_role_trend", *_PLAYER_TABLES),
    "nfl_separation_opportunity": ("player_separation_opportunity", *_PLAYER_TABLES),
    "nfl_drop_rate": ("player_drop_rate", *_PLAYER_TABLES),
    "nfl_contract_value": ("player_contract_value", *_PLAYER_TABLES),
    "nfl_injury_return": ("injury_return_curve",),
    "nfl_team_records": ("team_records",),
}
for _name, _tables in _TOOL_TABLES.items():
    globals()[_name] = _stamped(_tables)(globals()[_name])


__all__ = [
    "nfl_schema", "nfl_status", "nfl_query",
    "nfl_search_plays", "nfl_team_stats", "nfl_player_stats", "nfl_player_lookup", "nfl_compare",
    "nfl_catalog", "nfl_roster", "nfl_injuries", "nfl_schedule", "nfl_snap_counts",
    "nfl_fantasy_opportunity", "nfl_fantasy_rankings", "nfl_ftn_charting",
    "nfl_td_luck", "nfl_role_trend", "nfl_separation_opportunity",
    "nfl_drop_rate", "nfl_contract_value", "nfl_injury_return",
    "nfl_team_records",
]