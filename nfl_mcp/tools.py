"""
NFL MCP Server tools:
  nfl_schema        → returns column reference so the model knows what to query
  nfl_status        → database health: row counts, loaded seasons, tables
  nfl_query         → one read-only SQL statement, guarded by the parser and a locked connection
  nfl_search_plays  → structured play search
  nfl_team_stats    → pre-aggregated team stats
  nfl_player_stats  → official season counting stats, summed from the week-level player_stats table
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

# Statement types the parser reports for read queries. DuckDB 2.0.0 folds SELECT, WITH,
# FROM-first, VALUES, SHOW, DESCRIBE and SUMMARIZE into SELECT.
# MULTI is a shape the parser reports for PIVOT and for some write forms that wrap it
# (CREATE TABLE ... AS PIVOT, INSERT INTO ... PIVOT, COPY (PIVOT ...) TO), so it lets a
# few writes through this list; the read-only locked connection is what stops them. What
# survives is session-scoped only: CREATE TEMP TABLE, CREATE TEMP MACRO, and EXPLAIN
# ANALYZE of SET VARIABLE or CHECKPOINT. None of it touches the database file, and the
# temp objects die with the per-call connection.
# EXPLAIN admits any statement, because EXPLAIN ANALYZE executes it — the same lock, plus
# the file lock on COPY ... TO, stops the write forms there.
# Everything else the connection cannot run — file functions, ATTACH, COPY TO, INSTALL,
# LOAD, DDL and DML — is caught here or by DuckDB.
_READ_STATEMENT_TYPES = frozenset({
    duckdb.StatementType.SELECT,
    duckdb.StatementType.MULTI,
    duckdb.StatementType.EXPLAIN,
})

_MAX_ROWS = 500
_QUERY_TIMEOUT_SECONDS = 10

# The roster status both fantasy ranking tables publish in `team`. It is a status, not a
# franchise, so `franchises.py` does not list it and no tool may treat it as a club.
_FREE_AGENT = "FA"


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
    asked for: roster tables prove the season, latest-team does not. `position` does filter
    rows here, on the name path: it is bound as `position ILIKE ?` against players.position
    with no wildcards, so it is case-insensitive but whole-code - position='WR' settles the two
    Justin Jeffersons of 2026 to 00-0036322 and position='LB' to 00-0041075, position='XX'
    answers not_found rather than handing back the unfiltered list, and 'LB' never matches a
    stored 'OLB'. It does nothing when `player_id` settles the player, because that branch
    reads by id. The tools that resolve a player filter their own rows with position too,
    because players.position uses different codes than snap_counts.position does: that is why
    _player_filter does not send position down here - 'T' would find no Trent Williams row at
    all, since players writes 'OT'.
    Returns exactly one of:
      {'resolved': {...}}                      - one player, use resolved['gsis_id']
      {'ambiguous': True, 'candidates': [...]} - several players, no stats to return
      {'not_found': True, 'did_you_mean': [...]}
      {'error': ...}                           - the lookup itself failed
    Never merges two ids. A name that covers several players comes back ambiguous - with
    the one exception the caller asks for: a generational suffix in the query picks the
    player who carries it, which is how 'Marvin Harrison Jr.' is not his father. A 'count'
    larger than 'candidates' means the list was cut off at 20. Never call this to build a
    list - 'candidates' holds at most 20.
    """
    tokens: list[str] = [] if player_id else _name_tokens(player or "")
    if not tokens and not player_id:
        return {"not_found": True, "query": player or "", "did_you_mean": []}
    if team and team.strip() and canonical_team(team) is None:
        # A code no franchise uses narrows the lookup to nothing, which reads as 'this
        # player does not exist'. Say which of the two arguments failed.
        return _unknown_team_error(team)

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


def _unknown_team_error(team: str, param: str = "team") -> dict[str, Any]:
    """The answer every tool gives for an abbreviation no franchise has ever used.

    A filter on a code that no row carries is an empty success, which a caller reads as
    'this stat does not exist'. Naming the code back and giving one current example turns
    a dead end into a re-typed query. `param` names the argument the caller got wrong, so
    a tool that takes both a club and an opponent says which of the two it refused.
    """
    return {
        "error": f"Unknown {param} '{team.strip()}'.",
        "hint": "Use a current abbreviation, e.g. KC, LV for the Raiders, LAC for the "
                "Chargers, LA for the Rams, WAS for the Commanders.",
    }


def _reversed_range(low_name: str, low: int | None, high_name: str, high: int | None) -> dict | None:
    """Refuse a season range that ends before it starts, which otherwise answers an empty success."""
    if low is not None and high is not None and int(low) > int(high):
        return {"error": f"{low_name}={low} is after {high_name}={high}. Swap the two values."}
    return None


def _coverage_reason(table: str, season: int | None = None, season_from: int | None = None,
                     season_to: int | None = None, column: str = "season") -> tuple[bool, str]:
    """Say why a read of `table` came back empty: what was asked, what the table holds, what to change.

    The first value is True when the seasons asked for lie outside what the table holds, the
    case where a list of other names or an empty list is a wrong answer rather than a thin one.
    """
    if season is not None:
        asked = f"season {season}"
    elif season_from is not None or season_to is not None:
        asked = f"seasons {season_from if season_from is not None else '...'} to " \
                f"{season_to if season_to is not None else '...'}"
    else:
        asked = "every season"
    try:
        span = _execute(f"SELECT MIN({column}) AS lo, MAX({column}) AS hi FROM {table}")
    except (duckdb.Error, TimeoutError):
        span = []
    low, high = (span[0]["lo"], span[0]["hi"]) if span else (None, None)
    if low is None or high is None:
        return True, f"No {table} rows for {asked}. The table is empty. Run `nfl-mcp ingest` to load it."
    if season is not None:
        outside = not low <= int(season) <= high
    else:
        outside = ((season_from is not None and int(season_from) > high)
                   or (season_to is not None and int(season_to) < low))
    held = f"This database holds {table} for seasons {low} to {high}."
    if outside:
        return True, (f"No {table} rows for {asked}. {held} Ask for a season in that range, or "
                      "load the seasons you need with `nfl-mcp ingest --start <first> --end <last>`.")
    return False, f"No {table} rows match these filters for {asked}. {held} Loosen the other filters."


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


# The vocabularies a `season_type` parameter accepts, one constant per vocabulary, named for the
# column behind it. _season_type_filter below refuses anything outside the one its caller names,
# and tests/test_team_codes_and_truncation.py pins each tool's hand-written schema enum in
# server.py to the same constant, so a new value is one edit and a drift is one failing test.
SEASON_TYPES = ("REG", "POST")                  # what plays, player_stats and the derived tables store
POSTSEASON_ROUNDS = ("WC", "DIV", "CON", "SB")  # the four rounds `schedules.game_type` names, in order
ALL_SCOPES = "ALL"                              # both scopes at once, as separate rows
SCHEDULE_SEASON_TYPES = (*SEASON_TYPES, *POSTSEASON_ROUNDS)  # what nfl_schedule accepts
# what nfl_separation_opportunity, nfl_td_luck and nfl_contract_value accept, and answer unprompted
SEPARATION_SEASON_TYPES = (*SEASON_TYPES, ALL_SCOPES)
SEPARATION_DEFAULT_SEASON_TYPE = SEASON_TYPES[0]


def _season_type_filter(
    conditions: list[str], params: list[Any], column: str,
    season_type: str | None, rounds: bool = False, allow_all: bool = False,
    blank_means: str | None = None,
) -> dict[str, Any] | None:
    """Append a season-type filter in the vocabulary that column actually stores.

    The tables disagree: `schedules.game_type` names the playoff round ('WC', 'DIV', 'CON', 'SB')
    while `plays` and `player_stats` only ever say 'REG' or 'POST'. Comparing
    either column to the caller's text as typed is an empty success - 'POST' matches no
    schedule row and 'WC' matches no play. With `rounds` the tool reads the schedules
    column, so 'POST' expands to the four rounds. Without it a round name is refused
    rather than widened: the caller who asked for the Super Bowl and the caller who asked
    for the wild card both got 'POST', one answer for two different questions.

    Anything outside that vocabulary is refused rather than bound. 'ALL' is the value that most
    often arrives at the wrong tool: it is a real value for nfl_separation_opportunity (see
    `allow_all`), a model carries it to the next tool, and `nfl_schedule(season_type='ALL')` came
    back with zero games - an empty answer the caller reads as an empty stat. The error names the
    set this column accepts, which is the answer to the question the caller actually asked.
    `allow_all` is for the one tool that documents 'ALL' as "both scopes": it applies no filter,
    because that tool already returns the two scopes as separate rows.

    A blank value - nothing, or a string of spaces - applies no filter, which is what the six
    tools whose default is every scope want. A tool whose documented default is one scope passes
    that default as `blank_means`, so a blank answers what the tool advertises instead of quietly
    widening: nfl_separation_opportunity documents REG, and season_type='' on it once returned
    both scopes. Returns the error to hand back, or None.
    """
    if not season_type or not season_type.strip():
        if blank_means is None:
            return None
        season_type = blank_means
    code = season_type.strip().upper()
    if allow_all and code == ALL_SCOPES:
        return None
    if rounds:
        accepted: list[str] = list(SCHEDULE_SEASON_TYPES)
    elif allow_all:
        accepted = list(SEPARATION_SEASON_TYPES)
    else:
        accepted = list(SEASON_TYPES)
    if not rounds and code in POSTSEASON_ROUNDS:
        return {
            "error": f"season_type '{code}' names a playoff round, and the table behind this "
                     f"tool stores only 'REG' or 'POST'.",
            "hint": "Use season_type='POST' for every playoff game, or nfl_schedule to pick "
                    "one round: WC, DIV, CON, SB.",
        }
    if code not in accepted:
        if allow_all:
            hint = (f"Leaving season_type out gives {SEPARATION_DEFAULT_SEASON_TYPE} only; pass "
                    f"season_type='{ALL_SCOPES}' for both scopes as separate rows.")
        else:
            hint = "Leave season_type out to get every scope this table has, as separate rows."
        return {
            "error": f"season_type '{code}' is not a value the table behind this tool stores. "
                     f"Accepted: {', '.join(accepted)}.",
            "hint": hint,
        }
    values: list[str] = list(POSTSEASON_ROUNDS) if rounds and code == "POST" else [code]
    if len(values) == 1:
        conditions.append(f"{column} = ?")
        params.append(values[0])
    else:
        conditions.append(f"{column} IN ({', '.join('?' for _ in values)})")
        params.extend(values)
    return None


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
    settle_with: tuple[str, ...] = ("player_id", "team", "position"),
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

    `settle_with` names the calling tool's own parameters that can pick one candidate, so an
    ambiguous answer tells the caller what to send next. nfl_resolve_player is not an MCP tool.
    """
    if team and team.strip() and canonical_team(team) is None:
        return _unknown_team_error(team)
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
    if "ambiguous" in resolved:
        return {**resolved, "settle_with": list(settle_with),
                "hint": "Two or more players match. Re-run with one of these set from one "
                        f"candidate: {', '.join(settle_with)}."}
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
        # Imported here like _stale_derived_shape does, so serving tools does not pay for polars.
        from .config import get_duckdb_path
        from .ingest import (
            _FRESHNESS_SQL,
            _SHAPE_STAMP_SQL,
            active_slot,
            derived_status,
        )

        freshness = _execute(_FRESHNESS_SQL)[0]
        stamp = _execute(_SHAPE_STAMP_SQL)
        derived = derived_status(stamp=stamp[0]["row_count"] if stamp else None, **freshness)
        return {
            "plays": {
                "total_plays": total[0]["total_plays"] if total else 0,
                "season_range": min_max[0] if min_max else {},
                "seasons": seasons,
            },
            "datasets": {
                "total_loaded": len(datasets),
                # The last real fetch. A rebuild writes metadata rows too, and counting them made
                # this move with no new data.
                "last_refreshed": freshness["last_fetch"],
                "loaded": datasets,
            },
            "derived": derived,
            "active_slot": active_slot(get_duckdb_path()),
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
    Execute one read-only SQL statement against the nflread database.

    Full read-only DuckDB SQL: CTEs, window functions, PIVOT, DESCRIBE,
    SUMMARIZE. One statement only; writes, DDL, COPY, INSTALL, LOAD, ATTACH
    and file-table functions are rejected (the connection itself has file and
    network access locked off). Results capped at max_rows (hard max 500),
    fetched with the query's own ORDER BY intact. 10-second statement timeout.

    Unknown column/table failures also return `code`, the offending name,
    `did_you_mean` (ranked against the live schema) and a `hint`.
    """
    try:
        statements = duckdb.extract_statements(sql)
    except (duckdb.Error, ValueError) as e:
        return {"error": str(e), **_sql_error_detail(e)}

    if len(statements) != 1:
        return {"error": f"Exactly one SQL statement is required, got {len(statements)}."}

    statement = statements[0]
    if statement.type not in _READ_STATEMENT_TYPES:
        return {"error": f"Only read-only queries are allowed, not {statement.type.name}."}

    max_rows = max(0, min(max_rows, _MAX_ROWS))

    try:
        rows = _execute(statement.query, row_limit=max_rows + 1)

        truncated = len(rows) > max_rows
        rows = rows[:max_rows]

        return {
            "sql":       statement.query.strip(),
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
    if opponent and opponent.strip() and canonical_team(opponent) is None:
        return _unknown_team_error(opponent, "opponent")
    _team_filter(conditions, params, "posteam", team)
    _team_filter(conditions, params, "defteam", opponent)
    res = _player_filter("(passer_player_id = ? OR rusher_player_id = ? OR receiver_player_id = ?)",
                         conditions, params, player=player, player_id=player_id, team=team,
                         season=season, season_from=season_from, season_to=season_to, value_count=3,
                         settle_with=("player_id", "team"))
    if "identity" not in res:
        return res
    identity = res["identity"]
    refused = _reversed_range("season_from", season_from, "season_to", season_to)
    if refused:
        return refused
    if season is not None:
        conditions.append("season = ?")
        params.append(int(season))
    if season_from is not None:
        conditions.append("season >= ?")
        params.append(int(season_from))
    if season_to is not None:
        conditions.append("season <= ?")
        params.append(int(season_to))
    if week is not None:
        conditions.append("week = ?")
        params.append(int(week))
    refused = _season_type_filter(conditions, params, "season_type", season_type)
    if refused:
        return refused
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
    max_rows = max(1, min(max_rows, _MAX_ROWS))   # a cap of zero reads as 'no rows exist'

    sql = (
        f"SELECT season, week, posteam, defteam, down, ydstogo, play_type, "
        f"yards_gained, epa, \"desc\", passer_player_name, rusher_player_name, "
        f"receiver_player_name, passer_player_id, rusher_player_id, "
        f"receiver_player_id, touchdown, interception "
        f"FROM plays WHERE {where} "
        f"ORDER BY ABS(epa) DESC"
    )

    try:
        rows, flag = _execute_limited(sql, params, max_rows)
        if identity:
            for row in rows:
                row["label"] = identity["label"]
        return _echo_player({"rows": rows, "row_count": len(rows), **flag}, identity)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


# Where the three team tables' columns come from. nfl_schema(category='aggregate_tables') names
# every column with its source; this is the same three sources in the response itself.
_TEAM_STAT_SOURCES = {
    "team_stats_raw": (
        "counting stats, summed from one row per team per game: total_plays, total_yards, rush_plays, "
        "pass_plays, completions, rush_yards, pass_yards, touchdowns, passing_tds, rushing_tds, "
        "special_teams_tds, turnovers, passing_interceptions, fumbles_lost, sacks_suffered, "
        "sack_yards, penalties, penalty_yards, sacks, interceptions; the three "
        "yards-per-play rates, which divide those same totals: yards_per_play (total_yards / "
        "total_plays), yards_per_rush (rush_yards / rush_plays), yards_per_pass (pass_yards / "
        "(pass_plays + sacks_suffered)); the defence's plays_against, yards_allowed and "
        "turnovers_forced, from the opponent's box row grouped on opponent_team; and "
        "yards_per_play_allowed, which is yards_allowed / plays_against from that same row. "
        "first_downs is the one counting column not read straight off this box: it has no "
        "first_downs column and no penalty first-down column, so the total is its "
        "passing_first_downs + rushing_first_downs plus the penalty first downs from plays "
        "(its receiving_first_downs is the same events as passing_first_downs, credited to the "
        "receiver as well as the quarterback, so adding it counts every passing first down twice). "
        "total_yards, pass_yards, yards_per_play, yards_allowed and yards_per_play_allowed are net "
        "of sacks: sack_yards is negative, so the gross figure ESPN prints is total_yards - "
        "sack_yards (Kansas City 2024 regular season 5570 net, 5836 gross). touchdowns is "
        "passing_tds + rushing_tds, the offence's own scores: defensive and return touchdowns are "
        "not in it (special_teams_tds is its own column; the box's def_tds is in team_stats_raw)"
    ),
    "schedules": "points_for, points_against, points_allowed (game_type REG is REG; WC, DIV, CON, SB are POST)",
    "plays": (
        "what no box carries, over scrimmage plays only - (pass = 1 OR rush = 1) AND epa IS NOT "
        "NULL AND two_point_attempt <> 1 AND play_type <> 'no_play': explosive_plays, avg_epa, "
        "pass_epa, rush_epa, avg_epa_allowed, and plays, avg_yards, "
        "touchdowns, avg_epa in situational_stats. Over the conversion flags instead, so kneels and "
        "spikes count too: third_down_*, and the conversions and conversion_attempts columns of "
        "situational_stats. Over drives: red_zone_trips, red_zone_touchdowns, red_zone_td_pct. "
        "Over the first_down_penalty flag, grouped by posteam, season and season_type: the penalty "
        "first downs that complete team_offense_stats.first_downs, which no box column carries"
    ),
}


_REBUILD_HINT = (
    "Rebuild the derived tables: nfl-mcp rebuild with the server stopped, or /update_db.sh in "
    "a container (builds the idle slot and swaps the symlink)"
)

# Said about the database rather than about the tool that answered: the probe names seven tables,
# and nfl_injury_return, which reads injury_return_curve, is the one that most often quotes them.
# The list stays, because it is what tells a reader which tables a rebuild has to fix.
_STALE_SHAPE_LEAD = "This database's derived tables are older than this build:"


def _stale_derived_shape() -> dict[str, Any] | None:
    """Return an error naming both rebuild paths when this database's derived shape is old.

    None means every derived table here carries the shape this code writes, so any of them can
    be read. The rule lives in nfl_mcp.ingest.shape_needs_rebuild - the same question
    `nfl-mcp rebuild --check` answers - and both reads ride on _execute so the guard stays on
    the server's read-only connection and never takes the write lock. It differs from the CLI on
    one branch only: no stamp row plus a passing probe is current here, so a hand-built test
    database and a database rebuilt just before the stamp was introduced are not nagged.

    A database ingested before this change has derived tables - the ones the ingest builds rather
    than the ones nflverse publishes - with no season_type column, so the tools that read them
    either fail on a DuckDB binder error about a column the caller never typed or, worse, answer
    with the old shape as if it were the new one: nfl_injury_return(injury_type="ankle") returned
    20 pre-season_type injury groups, `other` among them, with no error at all.

    The command has to reach a DuckDB writer. `nfl-mcp ingest --dataset teams` is wrong here: it
    takes the same write lock the server holds on the file it is reading, and it goes to the network
    for the 36-row `teams` dataset instead of rebuilding from the rows already loaded. The remedy the
    message names is `nfl-mcp rebuild`, which rebuilds from rows already loaded, but it still needs
    the write lock, so the server must be stopped first. In a container the server never stops, so
    there the path is /update_db.sh, which builds the idle slot and swaps the symlink.
    """
    # Imported here for the same reason cli.py imports it inside its commands: a reader who only
    # serves tools should not pay for polars and tqdm at import time.
    from .ingest import (
        _PROBE_PARAMS,
        _SHAPE_PROBE_SQL,
        _SHAPE_STAMP_SQL,
        DERIVED_SHAPE_VERSION,
        shape_needs_rebuild,
    )

    try:
        rows = _execute(_SHAPE_STAMP_SQL)
    except duckdb.Error:
        stamp: int | None = None  # no _ingest_metadata at all: no row to compare the shape with
    else:
        stamp = int(rows[0]["row_count"]) if rows and rows[0]["row_count"] is not None else None
    missing = [str(row["table_name"]) for row in _execute(_SHAPE_PROBE_SQL, _PROBE_PARAMS)]
    if not shape_needs_rebuild(stamp=stamp, missing=missing):
        return None
    if missing:
        names = ", ".join(sorted(missing))
        what = f"{names} {'has' if len(missing) == 1 else 'have'} no season_type column"
    else:
        what = (f"the derived-shape stamp reads {stamp}, this build writes "
                f"{DERIVED_SHAPE_VERSION}")
    return {"error": f"{_STALE_SHAPE_LEAD} {what}. {_REBUILD_HINT}"}


def nfl_team_stats(
    team: str,
    season: int | None = None,
    side: str = "both",
    season_type: str | None = None,
) -> Dict[str, Any]:
    """Get pre-aggregated team stats (offense, defense, or both).

    Each row is one team, season and season_type. A playoff game is its own row and is never added
    into the regular season: pass season_type='REG' or 'POST' to get one of them, and leave it out
    to get both, side by side. 'sources' says which table each column comes from.
    """
    if canonical_team(team) is None:
        return _unknown_team_error(team)

    conditions: list[str] = []
    params: list[Any] = []
    _team_filter(conditions, params, "team", team)
    if season is not None:
        conditions.append("season_year = ?")
        params.append(int(season))
    err = _season_type_filter(conditions, params, "season_type", season_type)
    if err is not None:
        return err
    where = " AND ".join(conditions)

    results: dict[str, Any] = {"sources": _TEAM_STAT_SOURCES}
    try:
        stale = _stale_derived_shape()
        if stale is not None:
            return stale

        if side in ("offense", "both"):
            sql = f"SELECT * FROM team_offense_stats WHERE {where} ORDER BY season_year, season_type"
            results["offense"] = _execute(sql, params)

        if side in ("defense", "both"):
            sql = f"SELECT * FROM team_defense_stats WHERE {where} ORDER BY season_year, season_type"
            results["defense"] = _execute(sql, params)

        if side in ("situational", "both"):
            sql = f"SELECT * FROM situational_stats WHERE {where} ORDER BY season_year, season_type, situation"
            results["situational"] = _execute(sql, params)

        if not any(results.get(key) for key in ("offense", "defense", "situational")):
            results["reason"] = _coverage_reason("team_offense_stats", season, column="season_year")[1]
        return results
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


# ── Official player counting stats ─────────────────────────────────────────────
#
# nfl_player_stats sums the week-level `player_stats` table (registry.py loads it with
# nflreadpy's load_player_stats, one row per player per game) instead of counting `plays`
# rows, because nflverse already applies the official definitions a play count gets wrong:
#   - a sack is in neither `attempts` nor `carries`: `sacks_suffered` carries it and
#     `sack_yards_lost` its negative yardage
#   - a kneel is a carry, a scramble is a rush, a spike IS a pass attempt, and a two-point
#     play is in neither `attempts` nor `carries` (Mahomes 2024: 619 plays rows carry
#     pass_attempt = 1, and 581 = 619 less his 36 sacks less his 2 two-point plays, with his
#     2 spikes inside the 581; team_stats_raw counts spikes the same way)
# Counting `plays` rows with play_type = 'pass' handed Patrick Mahomes 617 attempts in 2024
# where the official count is 581 (his 36 sacks), and counting play_type = 'run' handed him
# 38 carries where the official count is 58 (his 20 kneels).
#
# Every value below is one SQL expression over the week rows, written against the qualified
# column name - valid for the single table in its FROM clause - and the same string is what
# the tool returns in `metric_sources`. The response therefore names the table and column of
# every metric, and the text cannot drift from the query. The grouping keys (player_id,
# season, season_type) are not metrics and carry no source.
#
# Every rate divides summed totals; no weekly rate is averaged. `avg_cpoe` is the one metric
# that does not come from this table: `player_stats.passing_cpoe` is a weekly mean and the
# table does not record how many plays carry a cpoe value, so there is no sound weight for
# it. The season mean is read from `plays.cpoe` over the caller's own window instead.

_STAT_SOURCE_COLUMNS = re.compile(r"player_stats\.([a-z0-9_]+)")

_PLAYER_STATS_SOURCE = (
    "player_stats (week level, one row per player per game), summed per season and season_type"
)

# The one metric player_stats cannot give, and the count of plays it covers, so a caller can
# see the window it was taken over.
_CPOE_MEAN_SOURCE = (
    "AVG(plays.cpoe) over the plays that carry a value, by plays.passer_player_id, over the "
    "same season and season_type window as the rest of the row"
)
_CPOE_COUNT_SOURCE = (
    "COUNT(plays.cpoe), the number of plays that carry a value behind avg_cpoe"
)

_PLAYER_STATS_METRICS: dict[str, dict[str, str]] = {
    "passing": {
        "games": "COUNT(DISTINCT player_stats.game_id)",
        "attempts": "SUM(player_stats.attempts)",
        "completions": "SUM(player_stats.completions)",
        "comp_pct": ("ROUND(100.0 * SUM(player_stats.completions)"
                     " / NULLIF(SUM(player_stats.attempts), 0), 1)"),
        "yards": "SUM(player_stats.passing_yards)",
        "touchdowns": "SUM(player_stats.passing_tds)",
        "interceptions": "SUM(player_stats.passing_interceptions)",
        "td_pct": ("ROUND(100.0 * SUM(player_stats.passing_tds)"
                   " / NULLIF(SUM(player_stats.attempts), 0), 1)"),
        "int_pct": ("ROUND(100.0 * SUM(player_stats.passing_interceptions)"
                    " / NULLIF(SUM(player_stats.attempts), 0), 1)"),
        "sacks": "SUM(player_stats.sacks_suffered)",
        "sack_yards": "SUM(player_stats.sack_yards_lost)",
        "sack_rate": ("ROUND(SUM(player_stats.sacks_suffered)"
                      " / NULLIF(SUM(player_stats.attempts) + SUM(player_stats.sacks_suffered), 0), 3)"),
        "yards_per_attempt": ("ROUND(SUM(player_stats.passing_yards)"
                              " / NULLIF(SUM(player_stats.attempts), 0), 1)"),
        "net_yards": "SUM(player_stats.passing_yards) + SUM(player_stats.sack_yards_lost)",
        "net_yards_per_attempt": (
            "ROUND((SUM(player_stats.passing_yards) + SUM(player_stats.sack_yards_lost))"
            " / NULLIF(SUM(player_stats.attempts) + SUM(player_stats.sacks_suffered), 0), 1)"),
        "air_yards": "SUM(player_stats.passing_air_yards)",
        "avg_air_yards": ("ROUND(SUM(player_stats.passing_air_yards)"
                          " / NULLIF(SUM(player_stats.attempts), 0), 1)"),
        "yards_after_catch": "SUM(player_stats.passing_yards_after_catch)",
        "first_downs": "SUM(player_stats.passing_first_downs)",
        "pass_10_plus": "SUM(player_stats.passing_10)",
        "pass_20_plus": "SUM(player_stats.passing_20)",
        "pass_40_plus": "SUM(player_stats.passing_40)",
        "two_point_conversions": "SUM(player_stats.passing_2pt_conversions)",
        "epa": "SUM(player_stats.passing_epa)",
        "avg_epa": ("ROUND(SUM(player_stats.passing_epa)"
                    " / NULLIF(SUM(player_stats.attempts) + SUM(player_stats.sacks_suffered), 0), 3)"),
    },
    "rushing": {
        "games": "COUNT(DISTINCT player_stats.game_id)",
        "carries": "SUM(player_stats.carries)",
        "yards": "SUM(player_stats.rushing_yards)",
        "yards_per_carry": ("ROUND(SUM(player_stats.rushing_yards)"
                            " / NULLIF(SUM(player_stats.carries), 0), 1)"),
        "touchdowns": "SUM(player_stats.rushing_tds)",
        "fumbles": "SUM(player_stats.rushing_fumbles)",
        "fumbles_lost": "SUM(player_stats.rushing_fumbles_lost)",
        "first_downs": "SUM(player_stats.rushing_first_downs)",
        "rush_10_plus": "SUM(player_stats.rushing_10)",
        "rush_20_plus": "SUM(player_stats.rushing_20)",
        "rush_40_plus": "SUM(player_stats.rushing_40)",
        "two_point_conversions": "SUM(player_stats.rushing_2pt_conversions)",
        "epa": "SUM(player_stats.rushing_epa)",
        "avg_epa": ("ROUND(SUM(player_stats.rushing_epa)"
                    " / NULLIF(SUM(player_stats.carries), 0), 3)"),
    },
    "receiving": {
        "games": "COUNT(DISTINCT player_stats.game_id)",
        "targets": "SUM(player_stats.targets)",
        "receptions": "SUM(player_stats.receptions)",
        "catch_pct": ("ROUND(100.0 * SUM(player_stats.receptions)"
                      " / NULLIF(SUM(player_stats.targets), 0), 1)"),
        "yards": "SUM(player_stats.receiving_yards)",
        "touchdowns": "SUM(player_stats.receiving_tds)",
        "yards_per_reception": ("ROUND(SUM(player_stats.receiving_yards)"
                                " / NULLIF(SUM(player_stats.receptions), 0), 1)"),
        "yards_per_target": ("ROUND(SUM(player_stats.receiving_yards)"
                             " / NULLIF(SUM(player_stats.targets), 0), 1)"),
        "yards_after_catch": "SUM(player_stats.receiving_yards_after_catch)",
        "avg_yac": ("ROUND(SUM(player_stats.receiving_yards_after_catch)"
                     " / NULLIF(SUM(player_stats.receptions), 0), 1)"),
        "air_yards": "SUM(player_stats.receiving_air_yards)",
        "avg_air_yards": ("ROUND(SUM(player_stats.receiving_air_yards)"
                          " / NULLIF(SUM(player_stats.targets), 0), 1)"),
        "fumbles": "SUM(player_stats.receiving_fumbles)",
        "fumbles_lost": "SUM(player_stats.receiving_fumbles_lost)",
        "first_downs": "SUM(player_stats.receiving_first_downs)",
        "rec_10_plus": "SUM(player_stats.receiving_10)",
        "rec_20_plus": "SUM(player_stats.receiving_20)",
        "rec_40_plus": "SUM(player_stats.receiving_40)",
        "two_point_conversions": "SUM(player_stats.receiving_2pt_conversions)",
        "epa": "SUM(player_stats.receiving_epa)",
        "avg_epa": ("ROUND(SUM(player_stats.receiving_epa)"
                    " / NULLIF(SUM(player_stats.targets), 0), 3)"),
    },
}

# A season row is real when at least one official counting stat in it is non-zero: they do not move
# together, so one column cannot decide. Josh Allen 2024 REG receiving has 0 targets, 7 yards and a
# touchdown. A counting stat sums a player_stats column as it is - not a rate, not `games`, which a
# kneel-only roster spot has too (Mahomes 2024 POST receiving: 3 games, all zero), and not `epa`:
# every row `epa` alone keeps alive here is one failed two-point try, which sits outside attempts,
# carries and targets by design. nfl_player_stats drops those rows; nfl_compare gets them from it.
_PLAYER_STATS_COUNTING_METRIC = re.compile(r"^SUM\(player_stats\.[a-z0-9_]+\)$")


def _counting_keys(metrics: dict[str, str]) -> list[str]:
    """The official metrics that sum a player_stats column as it is: the ones that say a play of
    this kind happened. `epa` is derived, so it keeps no row."""
    return [key for key, formula in metrics.items()
            if _PLAYER_STATS_COUNTING_METRIC.match(formula) and key != "epa"]


def nfl_player_stats(
    player_name: str | None = None,
    player_id: str | None = None,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    season_type: str | None = None,
    stat_type: str = "passing",
    position: str | None = None,
    team: str | None = None,
) -> Dict[str, Any]:
    """Official season counting stats for one player, summed from the week-level player_stats
    table by the resolved gsis_id. Returns one row per season and season_type, dropping a season
    whose counting stats are all zero, names the column behind every metric in `metric_sources`,
    and lists metrics this database cannot provide in `unavailable`. Only `avg_cpoe` and the play
    count behind it come from plays. `position` and `team` only pick the player a name means: the
    rows still cover every club he played for."""
    if stat_type not in _PLAYER_STATS_METRICS:
        return {"error": f"Unknown stat_type: {stat_type}. Use 'passing', 'rushing', or 'receiving'."}

    refused = _reversed_range("season_from", season_from, "season_to", season_to)
    if refused:
        return refused
    res = _player_filter(player=player_name, player_id=player_id, season=season,
                         season_from=season_from, season_to=season_to, must_have_player=True,
                         team=team, position=position)
    if res.get("not_found") and (season, season_from, season_to) != (None, None, None):
        outside, reason = _coverage_reason("player_stats", season, season_from, season_to)
        if outside:
            # The names on offer would be other players of those seasons, not the one asked for.
            return {"not_found": True, "query": res.get("query"), "reason": reason}
    if "identity" not in res:
        return res
    identity = res["identity"]

    try:
        # nflverse renamed columns across versions, so ask this database what it has and keep
        # the metrics whose columns are there, instead of failing the whole call on one rename.
        have = {str(row.get("column_name")) for row in _execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'player_stats'")}
        metrics = {key: formula for key, formula in _PLAYER_STATS_METRICS[stat_type].items()
                   if all(col in have for col in _STAT_SOURCE_COLUMNS.findall(formula))}
        if not metrics:
            return {"error": f"player_stats has none of the columns behind {stat_type} stats."}
        unavailable = {key: sorted({col for col in _STAT_SOURCE_COLUMNS.findall(formula)} - have)
                       for key, formula in _PLAYER_STATS_METRICS[stat_type].items()
                       if key not in metrics}
        sources = dict(metrics)

        window = ""
        params: list[Any] = [identity["gsis_id"]]
        if season is not None:
            window += " AND season = ?"
            params.append(int(season))
        if season_from is not None:
            window += " AND season >= ?"
            params.append(int(season_from))
        if season_to is not None:
            window += " AND season <= ?"
            params.append(int(season_to))
        season_type_conditions: list[str] = []
        refused = _season_type_filter(season_type_conditions, params, "season_type",
                                      season_type)
        if refused:
            return refused
        window += "".join(f" AND {cond}" for cond in season_type_conditions)

        sql = ("SELECT player_id, season, season_type, "
               + ", ".join(f"{formula} AS {key}" for key, formula in metrics.items())
               + " FROM player_stats WHERE player_id = ?" + window
               + " GROUP BY player_id, season, season_type ORDER BY season, season_type")

        counting = _counting_keys(metrics)
        rows = [row for row in _execute(sql, params) if any(row.get(key) for key in counting)]
        if stat_type == "passing":
            # The same window, so this metric cannot quietly cover a different span than the
            # row it sits on. plays carries season_type here, so filter on it, not postseason.
            sources["avg_cpoe"] = _CPOE_MEAN_SOURCE
            sources["cpoe_plays"] = _CPOE_COUNT_SOURCE
            by_window = {(row.get("season"), row.get("season_type")): row for row in _execute(
                "SELECT season, season_type, ROUND(AVG(cpoe), 3) AS avg_cpoe, COUNT(cpoe) AS cpoe_plays"
                " FROM plays WHERE passer_player_id = ? AND cpoe IS NOT NULL" + window
                + " GROUP BY season, season_type", params)}
            for row in rows:
                got = by_window.get((row.get("season"), row.get("season_type")), {})
                row["avg_cpoe"] = got.get("avg_cpoe")
                row["cpoe_plays"] = got.get("cpoe_plays", 0)
        for row in rows:
            row["label"] = identity["label"]
        return {
            "player": identity,
            "stat_type": stat_type,
            "source": _PLAYER_STATS_SOURCE,
            "metric_sources": sources,
            "unavailable": unavailable,
            "seasons": rows,
        }
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
    position1: str | None = None,
    position2: str | None = None,
    team1: str | None = None,
    team2: str | None = None,
) -> Dict[str, Any]:
    """Side-by-side comparison of two teams or players.

    Team mode returns one entry per abbreviation, each with 'offense' and 'defense' rows and the
    'sources' map naming where every column comes from. Each row is one team, season and
    season_type, and season_type is honoured here: pass 'REG' or 'POST' and a playoff game is not
    mixed into the regular season. Player mode returns {'players': [side1, side2]} in the order
    asked for - a list, not a dict keyed by the name, so two players who share a name (the two
    Justin Jeffersons) stay two entries. Each side carries 'requested' (the text typed) and
    'player' with its gsis_id plus the stat blocks; a side the name cannot settle carries the
    resolver's ambiguous or not_found answer in place of 'player'. Each stat block is
    nfl_player_stats' rows, summed from the week-level player_stats table per season and
    season_type, with 'metric_sources' naming the player_stats column behind every metric on that
    side.
    """
    e1, e2 = entity1, entity2
    refused = _reversed_range("season_from", season_from, "season_to", season_to)
    if refused:
        return refused

    try:
        if compare_type == "team":
            for entity in (e1, e2):
                if canonical_team(entity) is None:
                    return _unknown_team_error(entity)

            # A round name is refused here too, before any query: these two tables store only
            # 'REG' or 'POST', so 'SB' would answer with no rows rather than with an error.
            err = _season_type_filter([], [], "season_type", season_type)
            if err is not None:
                return err

            stale = _stale_derived_shape()
            if stale is not None:
                return stale
            season_filter = ""
            season_params: list[Any] = []
            if season is not None:
                season_filter += " AND season_year = ?"
                season_params.append(int(season))
            if season_from is not None:
                season_filter += " AND season_year >= ?"
                season_params.append(int(season_from))
            if season_to is not None:
                season_filter += " AND season_year <= ?"
                season_params.append(int(season_to))

            out: dict[str, Any] = {}
            for entity in (e1, e2):
                conditions: list[str] = []
                params: list[Any] = []
                _team_filter(conditions, params, "team", entity)
                _season_type_filter(conditions, params, "season_type", season_type)
                where = " AND ".join(conditions) + season_filter
                args = [*params, *season_params]
                out[entity] = {
                    "offense": _execute(
                        f"SELECT * FROM team_offense_stats WHERE {where}"
                        " ORDER BY season_year, season_type", args),
                    "defense": _execute(
                        f"SELECT * FROM team_defense_stats WHERE {where}"
                        " ORDER BY season_year, season_type", args),
                    "sources": _TEAM_STAT_SOURCES,
                }

            return out

        elif compare_type == "player":
            sides: list[dict[str, Any]] = []
            for side, name, pid, requested, position, team in [
                ("1", e1, player_id1, entity1, position1, team1),
                ("2", e2, player_id2, entity2, position2, team2),
            ]:
                res = _player_filter(player=name, player_id=pid, season=season,
                                     season_from=season_from, season_to=season_to,
                                     must_have_player=True, team=team, position=position,
                                     settle_with=tuple(f"{p}{side}" for p in
                                                       ("player_id", "team", "position")))
                if "identity" not in res:
                    sides.append({"requested": requested, **res})
                    continue
                identity = res["identity"]
                stats: dict[str, Any] = {"requested": requested, "player": identity}
                sources: dict[str, Any] = {}
                for stype in ("passing", "rushing", "receiving"):
                    # One nfl_player_stats call per stat type, so the sums and the row rule read
                    # the same player_stats rows. The emptiness test this replaced counted plays
                    # rows instead, and a block with no rows left is dropped here.
                    player_result = nfl_player_stats(
                        player_id=identity["gsis_id"], season=season,
                        season_from=season_from, season_to=season_to,
                        season_type=season_type, stat_type=stype,
                    )
                    if "error" in player_result:
                        return {"error": player_result["error"]}
                    if player_result["seasons"]:
                        stats[stype] = player_result["seasons"]
                        sources[stype] = player_result["metric_sources"]
                if sources:
                    stats["source"] = _PLAYER_STATS_SOURCE
                    stats["metric_sources"] = sources
                sides.append(stats)

            return {"players": sides}
        else:
            return {"error": "compare_type must be 'team' or 'player'"}

    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def _execute(
    sql: str,
    params: Sequence[Any] | None = None,
    row_limit: int | None = None,
) -> list[dict]:
    """Execute SQL on DuckDB with timeout and cancellation.

    Runs in a worker thread with a dedicated connection, and interrupts
    the active query if it exceeds the timeout.

    With `row_limit` and no parameters the cap rides on the relation, so DuckDB plans a
    LIMIT and turns an ORDER BY that has none into a top-N sort. Fetching the cap in
    Python instead makes DuckDB build and sort every row first: over 1.33 M plays rows
    `SELECT * FROM plays ORDER BY epa DESC` hit the 10 s timeout that way and answers in
    0.3 s this way. The cap only helps a lazy relation: on PIVOT and the other MULTI and
    EXPLAIN forms, `conn.sql()` runs the statement at once and the limit then trims the
    finished result, which is no worse than the old wrap.

    The parameterized path below takes no cap. `conn.sql()` accepts parameters too, but
    every caller that passes them wants the whole result, and none passes row_limit.

    A worker that ignores the interrupt keeps its connection open until it finishes,
    which can be after this raises and after the server's limiter slot is released.
    """
    result = [None]
    error = [None]
    thread_conn = [None]

    def _run():
        try:
            with get_db_connection() as conn:
                thread_conn[0] = conn
                if params is None and row_limit is not None:
                    rel = conn.sql(sql).limit(row_limit)
                    fetched = rel.fetchall()
                else:
                    rel = conn.execute(sql) if params is None else conn.execute(sql, params)
                    fetched = rel.fetchall()
                columns = [desc[0] for desc in rel.description]
                result[0] = [dict(zip(columns, row)) for row in fetched]
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


_TRUNCATION_HINT = (
    "Rows were cut at the limit. Add filters to narrow the search, or use nfl_query to "
    "aggregate the rest."
)


def _execute_limited(
    sql: str, params: Sequence[Any] | None, limit: int
) -> tuple[list[dict], dict]:
    """Run a capped query and report the cap to the caller.

    Every list tool stops at a fixed row count, and a cap the caller cannot see reads as
    'these are all the rows': snap counts for one team in one season are more than 500
    rows, so the last weeks went missing with no sign they were ever there. Asking for one
    row more than the caller may see is how the tool finds out. The rows never carry that
    extra row, and the second value is the keys to merge into the tool's own payload -
    truncated plus a hint when rows were cut, truncated alone when they were not. The
    query must not carry its own LIMIT.
    """
    rows = _execute(f"{sql} LIMIT {int(limit) + 1}", params)
    if len(rows) > limit:
        return rows[:limit], {"truncated": True, "hint": _TRUNCATION_HINT}
    return rows, {"truncated": False}


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
    if team and team.strip() and canonical_team(team) is None:
        return _unknown_team_error(team)
    conditions: list[str] = []
    params: list[Any] = []
    _team_filter(conditions, params, "team", team)
    if season is not None:
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
    """
    try:
        rows, flag = _execute_limited(sql, params if params else None, _MAX_ROWS)
        result = {"players": rows, "count": len(rows), **flag}
        if not rows:
            result["reason"] = _coverage_reason("rosters", season)[1]
        return result
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
    if season is not None:
        conditions.append("season = ?")
        params.append(int(season))
    if week is not None:
        conditions.append("week = ?")
        params.append(int(week))
    res = _player_filter("gsis_id = ?", conditions, params,
                         player=player, player_id=player_id, team=team, season=season,
                         settle_with=("player_id", "team"))
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
    """
    try:
        rows, flag = _execute_limited(sql, params if params else None, _MAX_ROWS)
        result = {"injuries": _label_players(rows), "count": len(rows), **flag}
        if not rows:
            result["reason"] = _coverage_reason("injuries", season)[1]
        return _echo_player(result, identity)
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
    if team and team.strip() and canonical_team(team) is None:
        return _unknown_team_error(team)
    conditions, params = [], []
    if team:
        codes = _team_codes(team)
        marks = ", ".join("?" for _ in codes)
        conditions.append(f"(home_team IN ({marks}) OR away_team IN ({marks}))")
        params.extend(codes * 2)
    if season is not None:
        conditions.append("season = ?")
        params.append(int(season))
    if week is not None:
        conditions.append("week = ?")
        params.append(int(week))
    refused = _season_type_filter(conditions, params, "game_type", season_type, rounds=True)
    if refused is not None:
        return refused

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
    """
    try:
        rows, flag = _execute_limited(sql, params if params else None, _MAX_ROWS)
        return {"games": rows, "count": len(rows), **flag}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


# ── Team records (1970–present) ───────────────────────────────────────────────

# Regular-season games each club has still to play in the newest season the schedule holds. A
# count against expected_games() would call BUF and CIN 2022 unfinished: their game was called off.
_REMAINING_GAMES_SQL = """
    SELECT season, team, COUNT(*) AS remaining FROM (
        SELECT season, game_type, result, home_team AS team FROM schedules
        UNION ALL SELECT season, game_type, result, away_team FROM schedules
    )
    WHERE season = (SELECT MAX(season) FROM schedules) AND game_type = 'REG' AND result IS NULL
    GROUP BY season, team
"""


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
    refused = _reversed_range("from_season", from_season, "to_season", to_season)
    if refused:
        return refused

    key = canonical_team(team) if team else None
    if team and key is None:
        return _unknown_team_error(team)

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
    """
    try:
        stale = _stale_derived_shape()
        if stale is not None:
            return stale
        rows, flag = _execute_limited(sql, params if params else None, _MAX_ROWS)
        remaining = {(r["season"], canonical_team(r["team"])): r["remaining"]
                     for r in (_execute(_REMAINING_GAMES_SQL) if rows else [])}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}

    for row in rows:
        row["club"] = era_name(row["season"], row["team"])
        left = remaining.get((row["season"], row["team"]))
        if left:
            row["status"] = "in_progress"
            row["reason"] = (f"Season {row['season']} is in progress: {row['team']} has played "
                             f"{row['games']} of {row['games'] + left} games. The record is not final.")

    result: Dict[str, Any] = {"records": rows, "count": len(rows), **flag}
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
    if season is not None:
        conditions.append("season = ?")
        params.append(int(season))
    if week is not None:
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
    """
    try:
        rows, flag = _execute_limited(sql, params if params else None, _MAX_ROWS)
        return _echo_player(
            {"snap_counts": _label_players(rows, name_key="player"), "count": len(rows), **flag},
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
    if season is not None:
        conditions.append("season = ?")
        params.append(int(season))
    if week is not None:
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
    """
    try:
        rows, flag = _execute_limited(sql, params if params else None, _MAX_ROWS)
        return _echo_player(
            {
                "fantasy_opportunity": _label_players(rows, team_key="posteam"),
                "count": len(rows),
                **flag,
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

    # `FA` in these tables is the free-agent status, not a club, so the franchise map does
    # not know it. It filters the rows and it does not narrow the player lookup: no row in
    # `players`, `rosters` or `rosters_weekly` carries it, so binding it there would turn a
    # real free agent into a not_found.
    free_agent = (team or "").strip().upper() == _FREE_AGENT
    res = _player_filter(
        f"{id_col} IN (SELECT fantasypros_id FROM ff_playerids WHERE gsis_id = ?)",
        conditions, params, player=player, player_id=player_id,
        team=None if free_agent else team, position=position)
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
    """
    try:
        rows, flag = _execute_limited(sql, params if params else None, limit)
        if identity and not rows and not _execute(
            "SELECT 1 FROM ff_playerids WHERE gsis_id = ? AND fantasypros_id IS NOT NULL LIMIT 1",
            [identity["gsis_id"]],
        ):
            return {"not_found": True, "query": identity["label"], "did_you_mean": [],
                    "reason": "ff_playerids maps no FantasyPros id to this player."}
        out = {"fantasy_rankings": _label_players(rows, name_key="player"), "scope": scope,
               "count": len(rows), **flag}
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
    if opponent and opponent.strip() and canonical_team(opponent) is None:
        return _unknown_team_error(opponent, "opponent")
    _team_filter(conditions, params, "p.posteam", team)
    _team_filter(conditions, params, "p.defteam", opponent)
    res = _player_filter("(p.passer_player_id = ? OR p.rusher_player_id = ? OR p.receiver_player_id = ?)",
                         conditions, params, player=player, player_id=player_id, team=team,
                         season=season, season_from=season_from, season_to=season_to, value_count=3,
                         settle_with=("player_id", "team"))
    if "identity" not in res:
        return res
    identity = res["identity"]
    refused = _reversed_range("season_from", season_from, "season_to", season_to)
    if refused:
        return refused
    if season is not None:
        conditions.append("p.season = ?")
        params.append(int(season))
    if season_from is not None:
        conditions.append("p.season >= ?")
        params.append(int(season_from))
    if season_to is not None:
        conditions.append("p.season <= ?")
        params.append(int(season_to))
    if week is not None:
        conditions.append("p.week = ?")
        params.append(int(week))
    refused = _season_type_filter(conditions, params, "p.season_type", season_type)
    if refused:
        return refused

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
    season_type: str | None = SEPARATION_DEFAULT_SEASON_TYPE,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Touchdown luck — actual vs expected TDs per player-season from the
    player_td_luck table. Negative total_td_luck_score = scored fewer TDs than
    expected (positive-regression / "unlucky" candidate); positive = over-performed.
    Default sort is most unlucky first (total_td_luck_score ASC).

    Each row is one player, season and season_type, and season_type defaults to 'REG', the same
    vocabulary as nfl_separation_opportunity: pass 'POST' for the playoffs or 'ALL' for both as
    separate rows.
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
    if season is not None:
        conditions.append("season = ?")
        params.append(int(season))
    refused = _season_type_filter(conditions, params, "season_type", season_type,
                                  allow_all=True, blank_means=SEPARATION_DEFAULT_SEASON_TYPE)
    if refused is not None:
        return refused

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player_id, full_name, position, team, season, season_type,
               rec_td, rec_td_exp, rec_td_luck,
               rush_td, rush_td_exp, rush_td_luck,
               total_td_luck_score
        FROM player_td_luck
        WHERE {where}
        ORDER BY total_td_luck_score ASC NULLS LAST
    """
    try:
        stale = _stale_derived_shape()
        if stale is not None:
            return stale
        rows, flag = _execute_limited(sql, params if params else None, limit)
        result = {"td_luck": _label_players(rows), "count": len(rows), **flag}
        if not rows:
            result["reason"] = _coverage_reason("player_td_luck", season)[1]
        return _echo_player(result, identity)
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
    if season is not None:
        conditions.append("season = ?")
        params.append(int(season))
    if week is not None:
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
    """
    try:
        stale = _stale_derived_shape()
        if stale is not None:
            return stale
        rows, flag = _execute_limited(sql, params if params else None, limit)
        return _echo_player({"role_trend": _label_players(rows), "count": len(rows), **flag}, identity)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {"error": str(e)}


def nfl_separation_opportunity(
    player: str | None = None,
    player_id: str | None = None,
    position: str | None = None,
    team: str | None = None,
    season: int | None = None,
    season_type: str | None = SEPARATION_DEFAULT_SEASON_TYPE,
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

    Each row is one player, season and season_type; the regular season and the postseason are
    separate rows and are never averaged together. season_type defaults to 'REG' — the postseason
    fallback rows rest on one or two weeks of data and crowd a list that asks for every season_type.
    Pass season_type='ALL' to get both scopes in one call. A null does not reach this function over
    MCP: the parameter is declared "type": "string", and mcp.server validates arguments against
    that schema before the handler runs, so season_type=None answers 'Input validation error: None
    is not of type \'string\''.

    avg_separation, avg_yac_above_expectation and catch_percentage are NGS's own season summary row
    (week = 0) where NGS published one, and a mean of that player's weekly rows in the same
    season_type where it did not — separation weighted by targets, YAC above expectation by
    receptions, since separation is measured on every target and YAC on every catch.
    separation_source says which of the two a row used; on a weekly-rows row, fallback_weeks and
    fallback_targets give the size of the sample behind it (124 of the 206 regular-season fallback
    rows rest on a single week, averaging 1.7 weeks, because NGS published weekly rows for only 291
    receiving players in 2024). regression_candidate is null on those rows: a one-week, five-target
    sample cannot support a regression claim.
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
    if season is not None:
        conditions.append("season = ?")
        params.append(int(season))
    refused = _season_type_filter(conditions, params, "season_type", season_type,
                                  allow_all=True, blank_means=SEPARATION_DEFAULT_SEASON_TYPE)
    if refused is not None:
        return refused
    if regression_candidate is not None:
        conditions.append("regression_candidate = ?")
        params.append(bool(regression_candidate))

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player_id, full_name, position, team, season, season_type, games,
               fp_diff, fp_diff_per_game, td_luck, catch_luck,
               target_share_pct, air_yards_share_pct,
               avg_separation, avg_yac_above_expectation, catch_percentage,
               separation_source, fallback_weeks, fallback_targets, regression_candidate
        FROM player_separation_opportunity
        WHERE {where}
        ORDER BY fp_diff_per_game ASC NULLS LAST
    """
    try:
        stale = _stale_derived_shape()
        if stale is not None:
            return stale
        rows, flag = _execute_limited(sql, params if params else None, limit)
        return _echo_player(
            {"separation_opportunity": _label_players(rows), "count": len(rows), **flag}, identity
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
                         player=player, player_id=player_id, team=team, season=season,
                         settle_with=("player_id", "team"))
    if "identity" not in res:
        return res
    identity = res["identity"]
    _team_filter(conditions, params, "team", team)
    if season is not None:
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
    """
    try:
        stale = _stale_derived_shape()
        if stale is not None:
            return stale
        rows, flag = _execute_limited(sql, params if params else None, limit)
        return _echo_player(
            {"drop_rate": _label_players(rows, name_key="player"), "count": len(rows), **flag}, identity
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
    season_type: str | None = SEPARATION_DEFAULT_SEASON_TYPE,
    min_apy: float | None = None,
    max_apy: float | None = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Contract value efficiency — fantasy points per $M of average per year (APY),
    using the contract in force that season joined to seasonal fantasy production
    (player_contract_value). A contract is in force from year_signed through
    year_signed + years - 1; when two cover one season the later signing wins, and
    contract_year_signed names it. A season no contract covers keeps its row with apy,
    cap_pct and fp_per_million null; it does not borrow another season's contract.
    Each row is one season_type, 'REG' by default ('POST', or 'ALL' for both).
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
    if season is not None:
        conditions.append("season = ?")
        params.append(int(season))
    refused = _season_type_filter(conditions, params, "season_type", season_type,
                                  allow_all=True, blank_means=SEPARATION_DEFAULT_SEASON_TYPE)
    if refused is not None:
        return refused
    if min_apy is not None:
        conditions.append("apy >= ?")
        params.append(float(min_apy))
    if max_apy is not None:
        conditions.append("apy <= ?")
        params.append(float(max_apy))

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT player_id, full_name, position, team, season, season_type, contract_year_signed,
               apy, cap_pct, total_fp, total_fp_exp, fp_per_million
        FROM player_contract_value
        WHERE {where}
        ORDER BY fp_per_million DESC NULLS LAST
    """
    try:
        stale = _stale_derived_shape()
        if stale is not None:
            return stale
        rows, flag = _execute_limited(sql, params if params else None, limit)
        result = {"contract_value": _label_players(rows), "count": len(rows), **flag}
        if not rows:
            result["reason"] = _coverage_reason("player_contract_value", season)[1]
        return _echo_player(result, identity)
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
    spell, grouped by the reported injury text and position (injury_return_curve).
    100 means fully back to baseline usage. Default sort is by weeks since return
    (week_post_return ASC). sample_size shows how many player-spells back each point.
    The sample holds only players whose baseline was at least 25% of the offensive snaps:
    a backup's tiny baseline turned a promotion into a recovery of several hundred percent.
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
    """
    try:
        stale = _stale_derived_shape()
        if stale is not None:
            return stale
        rows, flag = _execute_limited(sql, params if params else None, limit)
        return {"injury_return": rows, "count": len(rows), **flag}
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