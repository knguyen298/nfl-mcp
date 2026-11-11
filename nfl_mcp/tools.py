"""
NFL MCP Server tools:
  nfl_schema        → returns column reference so the model knows what to query
  nfl_status        → database health: row counts, loaded seasons, tables
  nfl_query         → read-only SELECT with safety guardrails
  nfl_search_plays  → structured play search
  nfl_team_stats    → pre-aggregated team stats
  nfl_player_stats  → player stats by season
  nfl_compare       → side-by-side comparison

Every failure a tool returns is a dict with `error` (English text the caller can show)
and `code` (an :class:`~nfl_mcp.status.ErrorCode` member the caller can branch on), so a
client never has to read prose or DuckDB output to tell a missing database from a bad
argument or a retryable lock.
"""

import difflib
import functools
import logging
import re
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Dict, NamedTuple

import duckdb

from .database import get_db_connection
from .franchises import (
    RECORDS_FIRST_NFLVERSE_SEASON,
    RECORDS_FIRST_SEASON,
    availability,
    before_records_coverage,
    era_name,
    expected_games,
    franchise_codes,
    suggest_teams,
)
from .registry import REGISTRY
from .schema_pbp import _SCHEMA_SUMMARY, _SCHEMA_CATEGORIES
from .seasons import current_season
from .status import Availability, ErrorCode, IngestStatus

logger = logging.getLogger("nfl-mcp")


# ── Safety guardrails ──────────────────────────────────────────────────────────

# nfl_query scans the statement for these words only outside string literals, quoted
# identifiers and comments (see _mask_sql). The scan gives the model a clear early error.
# The control that stops file and network access is the locked DuckDB configuration in
# database.get_db_connection, so a name missing from these lists is not a hole by itself.
_FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE", "REPLACE",
    "UPSERT", "EXECUTE", "EXEC", "CALL", "COPY", "GRANT", "REVOKE", "VACUUM", "ANALYZE",
    "CLUSTER", "REINDEX", "COMMENT", "SECURITY", "OWNER", "TABLESPACE", "SCHEMA",
    "SET", "DO", "LISTEN", "NOTIFY", "PREPARE", "DEALLOCATE", "LOAD", "DISCARD", "RESET",
    "ATTACH", "DETACH", "INSTALL", "PRAGMA", "EXPORT", "IMPORT", "CHECKPOINT", "USE",
)
_FORBIDDEN_FUNCTIONS = (
    "pg_read_file", "pg_read_binary_file", "pg_write_file", "pg_sleep",
    "lo_import", "lo_export", "dblink", "current_setting",
    "pg_terminate_backend", "pg_cancel_backend",
    # DuckDB functions that read files or run SQL text.
    "read_text", "read_blob", "read_csv", "read_csv_auto", "sniff_csv",
    "read_parquet", "parquet_scan", "parquet_metadata", "parquet_schema",
    "parquet_file_metadata", "parquet_kv_metadata", "parquet_full_metadata",
    "parquet_bloom_probe", "read_json", "read_json_auto", "read_json_objects",
    "read_json_objects_auto", "read_ndjson", "read_ndjson_auto", "read_ndjson_objects",
    "read_duckdb", "query", "query_table", "json_execute_serialized_sql",
    "arrow_scan", "arrow_scan_dumb", "pandas_scan", "python_map_function",
)
# `glob` is also DuckDB's GLOB string-match operator (desc GLOB '*TOUCHDOWN*'), so only
# its function-call form is blocked.
_FORBIDDEN = re.compile(
    r"\b(" + "|".join(_FORBIDDEN_KEYWORDS + _FORBIDDEN_FUNCTIONS) + r"|glob(?=\s*\())\b",
    re.IGNORECASE,
)
# A quoted identifier can name a function ("read_text"('...')), so these must not hide
# inside double quotes. Keywords lose their meaning when quoted, so "set" stays allowed.
_FORBIDDEN_QUOTED_NAMES = frozenset(f.lower() for f in _FORBIDDEN_FUNCTIONS + ("glob",))

_MAX_ROWS = 500
_QUERY_TIMEOUT_SECONDS = 10

# Characters DuckDB's lexer reads as part of an identifier (ASCII word characters, `$`,
# and every non-ASCII character). A quote prefix or a `$` after one of these does not
# start a new token.
_IDENT_CHAR = re.compile(r"[A-Za-z0-9_$\u0080-\U0010ffff]")
_DOLLAR_QUOTE = re.compile(r"\$(?:[A-Za-z_\u0080-\U0010ffff][A-Za-z0-9_\u0080-\U0010ffff]*)?\$")
_LINE_END = re.compile(r"[\r\n]")
_UNREADABLE_SQL = "Unterminated or ambiguous string literal, quoted identifier or comment."


def _quoted_end(sql: str, start: int, quote: str, backslash: bool) -> int:
    """Index just past the quote that closes the quoted run opened at `start`; -1 if none.

    A doubled quote is an escaped quote. With `backslash` (DuckDB's E'...' strings) a
    backslash also escapes the next character.
    """
    i, n = start + 1, len(sql)
    while i < n:
        ch = sql[i]
        if backslash and ch == "\\":
            i += 2
        elif ch != quote:
            i += 1
        elif sql.startswith(quote, i + 1):
            i += 2
        else:
            return i + 1
    return -1


def _block_comment_end(sql: str, start: int) -> int:
    """Index just past the `*/` that closes the comment opened at `start`; -1 if none.

    DuckDB nests block comments: `/* a /* b */ c */` is one comment.
    """
    depth, i, n = 1, start + 2, len(sql)
    while i < n:
        if sql.startswith("/*", i):
            depth, i = depth + 1, i + 2
        elif sql.startswith("*/", i):
            depth, i = depth - 1, i + 2
            if depth == 0:
                return i
        else:
            i += 1
    return -1


def _sql_spans(sql: str) -> list[tuple[int, int, str]] | None:
    """Find each string literal, quoted identifier and comment, following DuckDB's lexer rules.

    Returns `(start, end, kind)` spans with kind `literal`, `name` or `comment`, or None
    when one of them is never closed.
    """
    spans: list[tuple[int, int, str]] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        after_ident = i > 0 and _IDENT_CHAR.match(sql[i - 1]) is not None
        if sql.startswith("--", i):
            line_end = _LINE_END.search(sql, i)
            start, end, kind = i, line_end.start() if line_end else n, "comment"
        elif sql.startswith("/*", i):
            start, end, kind = i, _block_comment_end(sql, i), "comment"
        elif ch == '"':
            start, end, kind = i, _quoted_end(sql, i, '"', False), "name"
        elif ch == "'":
            # E'..', B'..' and X'..' are one token when the letter does not end a longer word.
            prefix = i > 0 and sql[i - 1] in "eEbBxX" and not (
                i > 1 and _IDENT_CHAR.match(sql[i - 2])
            )
            start = i - 1 if prefix else i
            end = _quoted_end(sql, i, "'", prefix and sql[i - 1] in "eE")
            kind = "literal"
        elif ch == "$" and not after_ident and _DOLLAR_QUOTE.match(sql, i):
            tag = _DOLLAR_QUOTE.match(sql, i).group()
            close = sql.find(tag, i + len(tag))
            start, end, kind = i, close + len(tag) if close >= 0 else -1, "literal"
        else:
            i += 1
            continue
        if end < 0:
            return None
        spans.append((start, end, kind))
        i = end
    return spans


def _engine_agrees(sql: str, spans: list[tuple[int, int, str]]) -> bool:
    """Check the spans against DuckDB's own tokenizer.

    The tokenizer must start a token at every literal and quoted identifier the scanner
    found, start no token inside a literal, identifier or comment, and report no string
    literal the scanner missed. A difference means the scanner would read some text
    differently than the engine runs it, so the caller rejects the statement.
    """
    char_index, offset = {}, 0
    for idx, ch in enumerate(sql):
        char_index[offset] = idx  # the tokenizer reports UTF-8 byte offsets
        offset += len(ch.encode("utf-8"))
    tokens = {char_index[pos]: kind for pos, kind in duckdb.tokenize(sql)}
    covered = bytearray(len(sql))
    starts = set()
    for start, end, kind in spans:
        first = start if kind == "comment" else start + 1
        covered[first:end] = b"\x01" * (end - first)
        if kind != "comment":
            starts.add(start)
    if any(covered[idx] for idx in tokens):
        return False
    literals = {idx for idx, kind in tokens.items() if kind == duckdb.token_type.string_const}
    return literals <= starts <= tokens.keys()


def _mask_sql(sql: str) -> tuple[str, list[str]] | None:
    """Blank out the parts of `sql` that are not SQL code, so a text scan sees only code.

    Comments become spaces. String literals and quoted identifiers become `#` runs of the
    same length, so positions are kept and a word next to a literal stays a separate word.
    Returns the masked text and the unescaped content of each quoted identifier, or None
    when a literal, identifier or comment is unterminated or DuckDB lexes it differently.
    """
    spans = _sql_spans(sql)
    if spans is None or not _engine_agrees(sql, spans):
        return None
    chars = list(sql)
    names = []
    for start, end, kind in spans:
        chars[start:end] = (" " if kind == "comment" else "#") * (end - start)
        if kind == "name":
            names.append(sql[start + 1:end - 1].replace('""', '"'))
    return "".join(chars), names


# ── Player name normalization ──────────────────────────────────────────────────

def _normalize_player_name(name: str) -> str:
    """
    Convert full names to nflverse short format.
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


def nfl_schema(category: str | None = None, table: str | None = None) -> Dict[str, Any]:
    """Return schema reference. Summary by default, or a specific category for detail.
    Pass table='<name>' to get the live column list for any non-pbp table.

    A table that does not exist returns `code` TABLE_NOT_FOUND with `available_tables`;
    an unknown category returns INVALID_INPUT with `available_categories`. Both errors also
    carry the same list as `available`. `table` wins over `category`; a result that ignored
    `category` says so in `note`.
    If listing the tables fails, the summary still returns `available_tables: []`, plus a
    `warning` and a `code`, so a missing database is not mistaken for an empty one."""
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
                payload: Dict[str, Any] = {"table": table, "columns": rows}
                if category is not None:
                    payload["note"] = (
                        f"category='{category}' was ignored: table= returns one table's "
                        "columns. Call nfl_schema(category=...) on its own for play-by-play detail."
                    )
                return payload
            available = _execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            )
            table_names = [r["table_name"] for r in available]
            return _err(
                ErrorCode.TABLE_NOT_FOUND,
                f"Table '{table}' not found in schema 'main'",
                available_tables=table_names,
                available=table_names,
            )
        except (duckdb.Error, ValueError, TimeoutError) as e:
            logger.error("Tool error: %s", e, exc_info=True)
            return _error_from_exception(e)
    if category is None:
        payload = {
            "schema": _SCHEMA_SUMMARY,
            "available_tables": [],
            "hint": "Pass category='<name>' for full PBP column details. Pass table='<name>' for any other table's columns.",
        }
        try:
            tables = _execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_name NOT LIKE '\\_%' ESCAPE '\\' "
                "ORDER BY table_name"
            )
            table_names = [r["table_name"] for r in tables]
        except Exception as e:
            # A missing or locked database must not look like "no tables".
            logger.error("Tool error: %s", e, exc_info=True)
            detail = _error_from_exception(e)
            payload["warning"] = f"Could not list tables: {detail['error']}"
            payload["code"] = detail["code"]
            return payload
        payload["available_tables"] = table_names
        return payload
    cat = category.lower().strip()
    if cat == "all":
        full = "\n".join(v.strip() for v in _SCHEMA_CATEGORIES.values())
        return {"schema": full}
    if cat in _SCHEMA_CATEGORIES:
        return {"category": cat, "schema": _SCHEMA_CATEGORIES[cat].strip()}
    return _err(
        ErrorCode.INVALID_INPUT,
        f"Unknown category '{cat}'",
        available=list(_SCHEMA_CATEGORIES.keys()),
        available_categories=list(_SCHEMA_CATEGORIES.keys()),
    )


# ── Reading the ingest log ─────────────────────────────────────────────────────
#
# `_ingest_metadata` records how every dataset+season attempt ended: `status` holds an
# :class:`~nfl_mcp.status.IngestStatus`, `detail` a short reason. The server opens the
# database read-only and cannot add columns, so a database an older ingest wrote still has
# the six-column table. Every read below works on both shapes: it asks
# `_metadata_has_status_column` first, and on a table with no `status` column treats every
# row as `IngestStatus.LOADED` with no detail — the only kind of row that code wrote.

_METADATA_STATUS_COLUMN_SQL = (
    "SELECT 1 FROM information_schema.columns "
    "WHERE table_name = '_ingest_metadata' AND column_name = 'status'"
)

# A row means "data is in the table" when it is `loaded` or `partial`. `partial` counts as
# covered — the rows are there — and also shows up as an issue, because the season is short.
_COVERED_STATUS_SQL = "status IN (?, ?)"
_COVERED_STATUS_PARAMS = [IngestStatus.LOADED.value, IngestStatus.PARTIAL.value]

# Attempts a caller has to hear about: ones that landed nothing, and ones that landed short.
_ISSUE_STATUS_SQL = "status IN (?, ?, ?)"
_ISSUE_STATUS_PARAMS = [
    IngestStatus.FAILED.value,
    IngestStatus.EMPTY.value,
    IngestStatus.PARTIAL.value,
]

_METADATA_ISSUES_SQL = (
    "SELECT dataset_id, season, status, detail FROM _ingest_metadata "
    f"WHERE {_ISSUE_STATUS_SQL} ORDER BY dataset_id, season"
)


def _metadata_has_status_column() -> bool:
    """True when `_ingest_metadata` carries the outcome columns, False on an older database."""
    return bool(_execute(_METADATA_STATUS_COLUMN_SQL))


def _coverage_filter(has_status: bool) -> tuple[str, list[str]]:
    """SQL predicate plus its parameters that keeps only rows whose data is in the table."""
    if has_status:
        return _COVERED_STATUS_SQL, list(_COVERED_STATUS_PARAMS)
    return "1=1", []


def _ingest_issues(has_status: bool) -> list[dict]:
    """Every `failed`, `empty` and `partial` attempt as {dataset_id, season, status, detail}.

    A database with no `status` column reports none: the code that wrote it recorded a row
    only when rows landed, so it kept no record of an attempt that loaded nothing.
    """
    if not has_status:
        return []
    return _execute(_METADATA_ISSUES_SQL, list(_ISSUE_STATUS_PARAMS))


def _covered_seasons(has_status: bool) -> dict[tuple[str, str], set[int]]:
    """Seasons whose data is in the table, keyed by (dataset_id, table_name).

    Rows without a season are left out: an interior gap needs two ends, and a dataset that
    runs once (a static table) has neither.
    """
    predicate, params = _coverage_filter(has_status)
    rows = _execute(
        "SELECT DISTINCT dataset_id, table_name, season FROM _ingest_metadata "
        f"WHERE {predicate} AND season IS NOT NULL ORDER BY dataset_id, season",
        params,
    )
    seasons: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        seasons.setdefault((row["dataset_id"], row["table_name"]), set()).add(row["season"])
    return seasons


def _interior_gaps(min_season: int | None, max_season: int | None, have: set[int]) -> list[int]:
    """Seasons with no data strictly between the two ends a catalog row reports."""
    if min_season is None or max_season is None:
        return []
    return [s for s in range(min_season + 1, max_season) if s not in have]


def nfl_status() -> Dict[str, Any]:
    """Return database health: play counts, loaded seasons, and a summary of all ingested datasets.

    The play-by-play section and the datasets section are read on their own, so a database
    that holds one of them reports what it has and puts the classified failure under the
    section it could not read. Only a database that answers neither returns one top-level
    error.
    """
    plays, plays_error = _status_plays()
    datasets, datasets_error = _status_datasets()
    if plays_error and datasets_error:
        return plays_error
    payload: Dict[str, Any] = {}
    payload["plays"] = plays_error if plays_error else plays
    payload["datasets"] = datasets_error if datasets_error else datasets
    return payload


def _status_plays() -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    """Play-by-play counts for :func:`nfl_status`, as `(payload, error)`; one is empty."""
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
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {}, _error_from_exception(e)
    return {
        "total_plays": total[0]["total_plays"] if total else 0,
        "season_range": min_max[0] if min_max else {},
        "seasons": seasons,
    }, None


def _status_datasets() -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    """The ingest log for :func:`nfl_status`, as `(payload, error)`; one is empty.

    `loaded` and `last_refreshed` count only the rows whose data is in the table, so a
    season whose every attempt failed is reported in `issues` instead of as coverage.
    """
    try:
        has_status = _metadata_has_status_column()
        predicate, params = _coverage_filter(has_status)
        datasets = _execute(
            "SELECT dataset_id, table_name, SUM(row_count) as total_rows "
            "FROM _ingest_metadata "
            f"WHERE {predicate} "
            "GROUP BY dataset_id, table_name "
            "ORDER BY dataset_id",
            params,
        )
        last_refreshed = _execute(
            f"SELECT MAX(loaded_at) as last_refreshed FROM _ingest_metadata WHERE {predicate}",
            params,
        )
        issues = _ingest_issues(has_status)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return {}, _error_from_exception(e)
    return {
        "total_loaded": len(datasets),
        "last_refreshed": last_refreshed[0]["last_refreshed"] if last_refreshed else None,
        "loaded": datasets,
        "issues": issues,
    }, None


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

# DuckDB reports database-level failures as free text too. These fragments tell a
# missing file apart from a file another process is writing; if DuckDB rewords its IO
# errors, these two constants are the only place to fix.
_DB_NOT_FOUND_TEXT = "database does not exist"
_DB_LOCKED_TEXT = "Could not set lock on file"
_DB_NOT_FOUND_HINT = "The database file is missing. Run 'nfl-mcp ingest' to create it."
_DB_LOCKED_HINT = "An ingest is writing the database right now. This call can be retried later."

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
            "code": ErrorCode.COLUMN_NOT_FOUND,
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
            "code": ErrorCode.TABLE_NOT_FOUND,
            "unknown_table": unknown,
            "did_you_mean": _nearest(unknown, _live_names(_TABLE_NAMES_SQL) + spelled),
            "hint": "Call nfl_catalog() or nfl_schema() to list the loaded tables.",
        }

    return {}


def _err(code: ErrorCode, message: str, **extra: Any) -> Dict[str, Any]:
    """Build a failure payload: the text the caller sees, plus the code it branches on."""
    payload: Dict[str, Any] = {"error": message, "code": code}
    payload.update(extra)
    return payload


def _error_from_exception(exc: Exception) -> Dict[str, Any]:
    """
    Turn any exception a query raised into a failure payload.

    `error` stays ``str(exc)``; `code` says which failure it is, first match wins:
    a timeout, a missing database file, a file another process has locked, a binder or
    catalog mistake (:func:`_sql_error_detail`), otherwise an unclassified SQL error.
    """
    message = str(exc)
    if isinstance(exc, TimeoutError):
        return _err(ErrorCode.TIMEOUT, message)
    if isinstance(exc, duckdb.IOException):
        if _DB_NOT_FOUND_TEXT in message:
            return _err(ErrorCode.DB_NOT_FOUND, message, hint=_DB_NOT_FOUND_HINT)
        if _DB_LOCKED_TEXT in message:
            return _err(ErrorCode.DB_LOCKED, message, hint=_DB_LOCKED_HINT)
    detail = _sql_error_detail(exc)
    if detail:
        code = detail.pop("code")
        return _err(code, message, **detail)
    return _err(ErrorCode.SQL_ERROR, message)


def _cap(rows: list[dict], cap: int) -> tuple[list[dict], bool]:
    """Trim a fetch of `cap + 1` rows to `cap` rows and report whether the extra row came back.

    Every tool that caps its result asks the database for one row more than it returns,
    passes it through here, and reports `truncated` next to its count key. `truncated`
    is True only when more rows matched than the tool returned.
    """
    truncated = len(rows) > cap
    return rows[:cap], truncated


# ── Questions the data cannot answer ──────────────────────────────────────────
#
# A success must mean "the question was understood, and this is the data". Three checks
# run before any tool queries: the team resolver, the fixed-value check and the season
# coverage guard. Each one raises `_Refusal` with an error payload, and the
# `_refusals_as_errors` decorator on every public tool returns that payload, so a tool body
# reads as a list of checks and never repeats the "return the error" step.


class _Refusal(Exception):
    """A tool argument the data cannot answer; `payload` is the error the caller gets."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__(payload["error"])
        self.payload = payload


def _refusals_as_errors(fn: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Return a refused argument's error payload instead of raising it."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except _Refusal as refusal:
            return refusal.payload

    return wrapper


# Parameters with a fixed set of values. server.py builds each input-schema `enum` from
# these tuples, so the schema and the check below cannot disagree.
SEASON_TYPES = ("REG", "POST")
# schedules.game_type has no POST: nflverse files playoff games by round.
_PLAYOFF_GAME_TYPES = ("WC", "DIV", "CON", "SB")
SCHEDULE_SEASON_TYPES = ("REG", "POST", *_PLAYOFF_GAME_TYPES)
PLAY_TYPES = (
    "pass", "run", "field_goal", "punt", "kickoff",
    "extra_point", "qb_kneel", "qb_spike", "no_play",
)
_SITUATION_SQL = {
    "red_zone": "yardline_100 <= 20",
    "third_down": "down = 3",
    "fourth_down": "down = 4",
    "two_minute": "qtr = 4 AND half_seconds_remaining <= 120",
}
SITUATIONS = tuple(_SITUATION_SQL)
TEAM_STAT_SIDES = ("offense", "defense", "situational", "both")
STAT_TYPES = ("passing", "rushing", "receiving")
COMPARE_TYPES = ("team", "player")
RANKING_SCOPES = ("draft", "week")


def _spell_options(options: Sequence[str]) -> str:
    """'a' or 'b' / 'a', 'b', or 'c' — the list an INVALID_INPUT message offers."""
    quoted = [f"'{option}'" for option in options]
    if len(quoted) == 2:
        return f"{quoted[0]} or {quoted[1]}"
    return ", ".join(quoted[:-1]) + f", or {quoted[-1]}"


def _choice(param: str, value: Any, allowed: Sequence[str]) -> str:
    """The allowed spelling of `value`, matched without regard to case.

    An unrecognized value is refused with INVALID_INPUT. Ignoring it would widen the
    query past what was asked, and matching it literally would return an empty success.
    """
    text = str(value).strip().lower()
    for option in allowed:
        if option.lower() == text:
            return option
    raise _Refusal(_err(
        ErrorCode.INVALID_INPUT,
        f"Unknown {param}: {value}. Use {_spell_options(allowed)}.",
        parameter=param,
        allowed=list(allowed),
    ))


class _Team(NamedTuple):
    """A resolved team argument: the canonical code, and every code a table may store."""

    key: str
    codes: tuple[str, ...]


def _resolve_team(
    value: str,
    *,
    param: str = "team",
    stored_in: tuple[str, str] | None = None,
) -> _Team:
    """Resolve a team argument through `franchises`, or refuse it with UNKNOWN_TEAM.

    A known franchise resolves to its current code plus every alias, so `LV` and `OAK`
    both return the whole Raiders franchise from a table that keeps the code of the era.
    A code `franchises` does not know is still accepted when the table the tool reads
    (`stored_in` = table, column) holds it, such as FantasyPros' `FA` for free agents.
    """
    codes = franchise_codes(value)
    if codes is not None:
        return _Team(codes[0], codes)
    code = str(value).strip().upper()
    if stored_in is not None and code:
        table, column = stored_in
        try:
            stored = _execute(f"SELECT 1 FROM {table} WHERE {column} = ? LIMIT 1", [code])
        except (duckdb.Error, ValueError, TimeoutError) as e:
            logger.error("Tool error: %s", e, exc_info=True)
            raise _Refusal(_error_from_exception(e)) from e
        if stored:
            return _Team(code, (code,))
    raise _Refusal(_err(
        ErrorCode.UNKNOWN_TEAM,
        f"Unknown team '{value}'.",
        parameter=param,
        did_you_mean=suggest_teams(value),
    ))


def _match_codes(column: str, codes: Sequence[str]) -> str:
    """SQL test that `column` holds one of `codes`; the caller binds the codes in order."""
    if len(codes) == 1:
        return f"{column} = ?"
    return f"{column} IN ({', '.join('?' for _ in codes)})"


# Which registry datasets each tool's rows come from, and the name its error uses. A
# tool's first season is the latest `min_season` among them: a join has rows only where
# every source does. nfl_team_records reads franchises' own records floor instead
# (franchises.availability), so only its upper bound is checked here.
_COVERAGE: dict[str, tuple[tuple[str, ...], str]] = {
    "nfl_search_plays": (("pbp",), "Play-by-play data"),
    "nfl_team_stats": (("pbp",), "Team stats (built from play-by-play)"),
    "nfl_player_stats": (("pbp",), "Player stats (built from play-by-play)"),
    "nfl_compare": (("pbp",), "Comparison stats (built from play-by-play)"),
    "nfl_roster": (("rosters",), "Roster data"),
    "nfl_injuries": (("injuries",), "Injury report data"),
    "nfl_schedule": (("schedules",), "Schedule data"),
    "nfl_snap_counts": (("snap_counts",), "Snap count data"),
    "nfl_fantasy_opportunity": (("ff_opportunity",), "Fantasy opportunity data"),
    "nfl_ftn_charting": (("ftn_charting", "pbp"), "FTN charting data"),
    "nfl_td_luck": (("ff_opportunity",), "TD luck data"),
    "nfl_role_trend": (("ff_opportunity",), "Role trend data"),
    "nfl_separation_opportunity": (
        ("ff_opportunity", "nextgen_stats_receiving"), "Next Gen Stats receiving data",
    ),
    "nfl_drop_rate": (("ftn_charting", "pbp"), "FTN charting (drop) data"),
    "nfl_contract_value": (("ff_opportunity",), "Contract value data"),
    "nfl_team_records": ((), "Team records"),
}


def first_season(tool: str) -> int | None:
    """First season `tool` has data for, from the registry; None when it declares none."""
    floors = [REGISTRY[d].min_season for d in _COVERAGE[tool][0]]
    declared = [floor for floor in floors if floor is not None]
    return max(declared) if declared else None


def _last_season(tool: str) -> int:
    """Last season `tool` can have data for: the current season, or an earlier declared cap."""
    caps = [REGISTRY[d].max_season for d in _COVERAGE[tool][0]]
    return min([cap for cap in caps if cap is not None] + [current_season()])


def _check_coverage(
    tool: str,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
) -> None:
    """Refuse a season, or a range that lies wholly outside `tool`'s data, with OUT_OF_COVERAGE.

    A range that only overlaps the edge is answered for the seasons that exist.
    """
    label = _COVERAGE[tool][1]
    first, last = first_season(tool), _last_season(tool)
    ends_before = [int(s) for s in (season, season_to) if s is not None]
    starts_after = [int(s) for s in (season, season_from) if s is not None]
    if first is not None and any(s < first for s in ends_before):
        message = f"{label} is only available from {first} onward"
    elif any(s > last for s in starts_after):
        message = f"{label} is only available through the {last} season"
    else:
        return
    raise _Refusal(_err(
        ErrorCode.OUT_OF_COVERAGE, message, first_season=first, last_season=last,
    ))


def nfl_query(sql: str, max_rows: int = 100) -> Dict[str, Any]:
    """
    Execute a read-only SQL SELECT against the nflread database.

    Safety rules:
      - Only a SELECT statement, or a WITH ... SELECT, is allowed (leading comments are fine)
      - Forbidden mutation/system keywords and file-reading functions are blocked
      - Multiple statements (semicolons) blocked
      - Results capped at max_rows (clamped to 1..500)
      - 10-second statement timeout

    The text checks read only SQL code: string literals, quoted identifiers and comments
    are masked first (_mask_sql), so `"desc" ILIKE '%set%'` or a `;` inside a literal is
    fine. DuckDB's parser then must see exactly one SELECT. The connection itself cannot
    read files or reach the network (database.get_db_connection).

    Unknown column/table failures also return `code`, the offending name,
    `did_you_mean` (ranked against the live schema) and a `hint`. Every failure,
    including the guardrails above, also returns `code`.
    """
    sql = sql.strip()

    scanned = _mask_sql(sql)
    if scanned is None:
        return _err(ErrorCode.SQL_ERROR, _UNREADABLE_SQL)
    masked, quoted_names = scanned

    # A WITH statement must still be a query: the wrap below parses only as a subquery,
    # and DuckDB refuses a CTE that is not a SELECT.
    if not re.match(r"\s*(SELECT|WITH)\b", masked, re.IGNORECASE):
        return _err(ErrorCode.NOT_SELECT, "Only SELECT queries are allowed.")

    match = _FORBIDDEN.search(masked)
    if match:
        return _err(ErrorCode.FORBIDDEN_KEYWORD, f"Forbidden keyword: '{match.group()}'")
    for name in quoted_names:
        if name.lower() in _FORBIDDEN_QUOTED_NAMES:
            return _err(ErrorCode.FORBIDDEN_KEYWORD, f"Forbidden keyword: '{name}'")

    # Trailing semicolons, whitespace and comments are dropped; any other `;` is a second
    # statement. Dropping a trailing `--` comment also keeps it from hiding the wrap's `)`.
    body_end = len(masked.rstrip(" \t\n\r\f\v;"))
    if ";" in masked[:body_end]:
        return _err(ErrorCode.MULTIPLE_STATEMENTS, "Multiple statements are not allowed.")

    max_rows = max(1, min(max_rows, _MAX_ROWS))
    safe_sql = f"SELECT * FROM ({sql[:body_end]}) AS _q LIMIT {max_rows + 1}"

    try:
        # The engine's own parser is the last word on how many statements this is.
        statements = duckdb.extract_statements(safe_sql)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            return _err(ErrorCode.MULTIPLE_STATEMENTS, "Multiple statements are not allowed.")

        rows = _execute(safe_sql)

        rows, truncated = _cap(rows, max_rows)

        return {
            "rows":      rows,
            "row_count": len(rows),
            "truncated": truncated,
        }

    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


@_refusals_as_errors
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
    """Search for plays using structured filters instead of raw SQL.

    `team` and `opponent` accept current or former codes and match the whole franchise.
    """
    if season_type is not None:
        season_type = _choice("season_type", season_type, SEASON_TYPES)
    if play_type is not None:
        play_type = _choice("play_type", play_type, PLAY_TYPES)
    if situation is not None:
        situation = _choice("situation", situation, SITUATIONS)
    _check_coverage("nfl_search_plays", season, season_from, season_to)
    conditions = []
    params: list[Any] = []
    if team:
        codes = _resolve_team(team, stored_in=("plays", "posteam")).codes
        conditions.append(_match_codes("posteam", codes))
        params.extend(codes)
    if opponent:
        codes = _resolve_team(opponent, param="opponent", stored_in=("plays", "defteam")).codes
        conditions.append(_match_codes("defteam", codes))
        params.extend(codes)
    if player:
        player = _normalize_player_name(player)
        conditions.append(
            "(passer_player_name ILIKE ? "
            "OR rusher_player_name ILIKE ? "
            "OR receiver_player_name ILIKE ?)"
        )
        pattern = f"%{player}%"
        params.extend([pattern, pattern, pattern])
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
    if season_type is not None:
        conditions.append("season_type = ?")
        params.append(season_type)
    if play_type is not None:
        conditions.append("play_type = ?")
        params.append(play_type)
    if situation is not None:
        conditions.append(_SITUATION_SQL[situation])
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
        f"ORDER BY ABS(epa) DESC LIMIT {max_rows + 1}"
    )

    try:
        rows = _execute(sql, params)
        rows, truncated = _cap(rows, max_rows)
        return {"rows": rows, "row_count": len(rows), "truncated": truncated}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


@_refusals_as_errors
def nfl_team_stats(
    team: str,
    season: int | None = None,
    side: str = "both",
) -> Dict[str, Any]:
    """Get pre-aggregated team stats (offense, defense, or both).

    `team` accepts current or former codes and matches the whole franchise.
    """
    side = _choice("side", side, TEAM_STAT_SIDES)
    _check_coverage("nfl_team_stats", season)
    codes = _resolve_team(team, stored_in=("team_offense_stats", "team")).codes
    team_clause = _match_codes("team", codes)
    results = {}
    season_clause = ""
    season_params: list[Any] = []
    if season is not None:
        season_clause = " AND season_year = ?"
        season_params.append(int(season))

    try:
        if side in ("offense", "both"):
            sql = f"SELECT * FROM team_offense_stats WHERE {team_clause}{season_clause} ORDER BY season_year"
            results["offense"] = _execute(sql, [*codes, *season_params])

        if side in ("defense", "both"):
            sql = f"SELECT * FROM team_defense_stats WHERE {team_clause}{season_clause} ORDER BY season_year"
            results["defense"] = _execute(sql, [*codes, *season_params])

        if side in ("situational", "both"):
            sql = f"SELECT * FROM situational_stats WHERE {team_clause}{season_clause} ORDER BY season_year, situation"
            results["situational"] = _execute(sql, [*codes, *season_params])

        return results
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


# The play-by-play columns each player stat type reads, and the play type it counts.
_PLAYER_STAT_COLUMNS = {
    "passing": ("passer_player_name", "pass"),
    "rushing": ("rusher_player_name", "run"),
    "receiving": ("receiver_player_name", "pass"),
}


@_refusals_as_errors
def nfl_player_stats(
    player_name: str,
    season: int | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
    season_type: str | None = None,
    stat_type: str = "passing",
) -> Dict[str, Any]:
    """Aggregate player stats by season."""
    stat_type = _choice("stat_type", stat_type, STAT_TYPES)
    if season_type is not None:
        season_type = _choice("season_type", season_type, SEASON_TYPES)
    _check_coverage("nfl_player_stats", season, season_from, season_to)
    player_name = _normalize_player_name(player_name)
    params: list[Any] = []

    if stat_type == "passing":
        sql = """
            SELECT season, season_type, COUNT(*) AS attempts,
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
            WHERE passer_player_name ILIKE ? AND play_type = ?
        """
    elif stat_type == "rushing":
        sql = """
            SELECT season, season_type, COUNT(*) AS carries,
                SUM(COALESCE(rushing_yards, 0)) AS yards,
                ROUND(AVG(yards_gained), 1) AS yards_per_carry,
                SUM(CASE WHEN rush_touchdown = 1 THEN 1 ELSE 0 END) AS touchdowns,
                SUM(CASE WHEN fumble_lost = 1 THEN 1 ELSE 0 END) AS fumbles_lost,
                ROUND(AVG(epa), 3) AS avg_epa,
                SUM(CASE WHEN yards_gained >= 10 THEN 1 ELSE 0 END) AS explosive_runs
            FROM plays
            WHERE rusher_player_name ILIKE ? AND play_type = ?
        """
    else:
        sql = """
            SELECT season, season_type, COUNT(*) AS targets,
                SUM(CASE WHEN complete_pass = 1 THEN 1 ELSE 0 END) AS receptions,
                SUM(COALESCE(receiving_yards, 0)) AS yards,
                ROUND(AVG(yards_after_catch), 1) AS avg_yac,
                SUM(CASE WHEN pass_touchdown = 1 THEN 1 ELSE 0 END) AS touchdowns,
                ROUND(AVG(epa), 3) AS avg_epa,
                SUM(CASE WHEN yards_gained >= 20 THEN 1 ELSE 0 END) AS explosive_plays
            FROM plays
            WHERE receiver_player_name ILIKE ? AND play_type = ?
        """
    params.extend([f"%{player_name}%", _PLAYER_STAT_COLUMNS[stat_type][1]])

    if season is not None:
        sql += " AND season = ?"
        params.append(int(season))
    if season_from is not None:
        sql += " AND season >= ?"
        params.append(int(season_from))
    if season_to is not None:
        sql += " AND season <= ?"
        params.append(int(season_to))
    if season_type is not None:
        sql += " AND season_type = ?"
        params.append(season_type)
    sql += " GROUP BY season, season_type ORDER BY season, season_type"

    try:
        rows = _execute(sql, params)
        return {"player": player_name, "stat_type": stat_type, "seasons": rows}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


def _season_filter(
    column: str,
    season: int | None,
    season_from: int | None,
    season_to: int | None,
) -> tuple[str, list[Any]]:
    """` AND ...` clauses for an exact season and a range, plus their parameters."""
    sql, params = "", []
    for value, op in ((season, "="), (season_from, ">="), (season_to, "<=")):
        if value is not None:
            sql += f" AND {column} {op} ?"
            params.append(int(value))
    return sql, params


@_refusals_as_errors
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

    The result is keyed by the two entity strings as given, plus `notes`: one entry for
    each thing the comparison left out and why (a filter it could not apply, a player no
    play matched, a stat lookup that failed). Two entities that name the same team or
    player are refused with INVALID_INPUT.
    """
    compare_type = _choice("compare_type", compare_type, COMPARE_TYPES)
    if season_type is not None:
        season_type = _choice("season_type", season_type, SEASON_TYPES)
    _check_coverage("nfl_compare", season, season_from, season_to)
    if compare_type == "team":
        return _compare_teams(entity1, entity2, season, season_from, season_to, season_type)
    return _compare_players(entity1, entity2, season, season_from, season_to, season_type)


def _compare_teams(entity1, entity2, season, season_from, season_to, season_type) -> Dict[str, Any]:
    """nfl_compare in team mode: offense and defense rows for two franchises."""
    stored_in = ("team_offense_stats", "team")
    team1 = _resolve_team(entity1, param="entity1", stored_in=stored_in)
    team2 = _resolve_team(entity2, param="entity2", stored_in=stored_in)
    if team1.key == team2.key:
        raise _Refusal(_err(
            ErrorCode.INVALID_INPUT,
            f"entity1 and entity2 are the same team ({team1.key}). Compare two different teams.",
            parameter="entity2",
        ))
    season_sql, season_params = _season_filter("season_year", season, season_from, season_to)
    notes = []
    if season_type is not None:
        notes.append({"note": (
            f"season_type={season_type} was not applied: the team stats tables hold one row "
            "per team and season, with regular season and playoffs together."
        )})

    def rows(table: str, team: _Team) -> list[dict]:
        return _execute(
            f"SELECT * FROM {table} WHERE {_match_codes('team', team.codes)}{season_sql} "
            "ORDER BY season_year",
            [*team.codes, *season_params],
        )

    try:
        off1 = rows("team_offense_stats", team1)
        off2 = rows("team_offense_stats", team2)
        def1 = rows("team_defense_stats", team1)
        def2 = rows("team_defense_stats", team2)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)
    return {
        entity1: {"offense": off1, "defense": def1},
        entity2: {"offense": off2, "defense": def2},
        "notes": notes,
    }


def _compare_players(entity1, entity2, season, season_from, season_to, season_type) -> Dict[str, Any]:
    """nfl_compare in player mode: every stat type each player has plays for."""
    if _normalize_player_name(entity1).lower() == _normalize_player_name(entity2).lower():
        raise _Refusal(_err(
            ErrorCode.INVALID_INPUT,
            f"entity1 and entity2 name the same player ('{entity1}'). Compare two different players.",
            parameter="entity2",
        ))
    season_sql, season_params = _season_filter("season", season, season_from, season_to)
    if season_type is not None:
        season_sql += " AND season_type = ?"
        season_params.append(season_type)

    result: Dict[str, Any] = {}
    notes: list[dict] = []
    try:
        for label in (entity1, entity2):
            name = _normalize_player_name(label)
            stats = {}
            matched = False
            for stype, (col, ptype) in _PLAYER_STAT_COLUMNS.items():
                count_row = _execute(
                    f"SELECT COUNT(*) AS n FROM plays WHERE {col} ILIKE ? AND play_type = ?{season_sql}",
                    [f"%{name}%", ptype, *season_params],
                )
                if not (count_row and count_row[0].get("n", 0) > 0):
                    continue
                matched = True
                player_result = nfl_player_stats(
                    label, season=season, season_from=season_from,
                    season_to=season_to, season_type=season_type, stat_type=stype,
                )
                if "seasons" in player_result:
                    stats[stype] = player_result["seasons"]
                else:
                    notes.append({
                        "entity": label,
                        "stat_type": stype,
                        "code": player_result.get("code"),
                        "note": f"The {stype} lookup failed, so this comparison has no {stype} "
                                f"stats for {label}: {player_result.get('error')}",
                    })
            if not matched:
                notes.append({"entity": label, "note": (
                    f"No play matched '{label}' with these filters, so it has no stats here. "
                    "Check the spelling: names are stored in short form ('P.Mahomes')."
                )})
            result[label] = stats
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)
    result["notes"] = notes
    return result


_INTERRUPT_GRACE_SECONDS = 1.0

# Workers that were still running after a timeout and its grace join. Each timeout
# prunes the finished ones, so the length is the number of threads still running.
_stuck_workers: list[threading.Thread] = []
_stuck_workers_lock = threading.Lock()


class QueryTimeoutError(TimeoutError):
    """A query ran past the timeout. Callers that catch TimeoutError still catch it.

    `interrupted` says whether conn.interrupt() was called and returned. `worker_alive`
    says whether the worker thread was still running after the grace join.
    """

    def __init__(self, message: str, *, interrupted: bool, worker_alive: bool) -> None:
        super().__init__(message)
        self.interrupted = interrupted
        self.worker_alive = worker_alive


def _record_stuck_worker(worker: threading.Thread) -> int:
    """Remember a worker that outlived its grace join; return how many still run."""
    with _stuck_workers_lock:
        _stuck_workers[:] = [t for t in _stuck_workers if t.is_alive()]
        _stuck_workers.append(worker)
        return len(_stuck_workers)


def _execute(sql: str, params: Sequence[Any] | None = None) -> list[dict]:
    """Execute SQL on DuckDB with timeout and cancellation.

    Runs in a worker thread with a dedicated connection, and interrupts
    the active query if it exceeds the timeout.

    On timeout it raises :class:`QueryTimeoutError` with the same message as before.
    A worker that gets its connection only after the timeout does not start the query.
    A worker still running after the grace join is logged with a warning and counted.
    """
    result = [None]
    error = [None]
    thread_conn = [None]
    abandoned = [False]
    lock = threading.Lock()

    def _run():
        try:
            with get_db_connection() as conn:
                with lock:
                    if abandoned[0]:
                        return  # the caller already raised the timeout
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
        with lock:
            abandoned[0] = True
            conn = thread_conn[0]
        interrupted = False
        if conn is not None:
            try:
                conn.interrupt()
                interrupted = True
            except Exception:
                logger.warning("Failed to interrupt timed out DuckDB query", exc_info=True)
        t.join(timeout=_INTERRUPT_GRACE_SECONDS)
        worker_alive = t.is_alive()
        if worker_alive:
            running = _record_stuck_worker(t)
            logger.warning(
                "Timed out query worker still running %ss after the timeout "
                "(connected=%s, interrupted=%s); %d such worker(s) running",
                _INTERRUPT_GRACE_SECONDS, conn is not None, interrupted, running,
            )
        raise QueryTimeoutError(
            f"Query exceeded {_QUERY_TIMEOUT_SECONDS} second timeout",
            interrupted=interrupted,
            worker_alive=worker_alive,
        )
    if error[0]:
        raise error[0]
    return result[0]


def nfl_catalog() -> Dict[str, Any]:
    """List every dataset the ingest log names, with row counts, season gaps, and last refresh.

    Counts come only from rows whose data is in the table. A dataset whose every attempt
    failed still appears, with zero counts, `null` season bounds and its `issues`, so a
    failed backfill reads as a problem instead of as a dataset nobody asked for.
    """
    try:
        has_status = _metadata_has_status_column()
        predicate, params = _coverage_filter(has_status)
        # The status test is evaluated once per row in the CTE, so the two parameters bind
        # once for the whole query, and the old shape reads every row as covered.
        rows = _execute(f"""
            WITH flagged AS (
                SELECT dataset_id, table_name, season, row_count, loaded_at,
                       ({predicate}) AS covered
                FROM _ingest_metadata
            )
            SELECT
                dataset_id,
                table_name,
                COALESCE(SUM(row_count) FILTER (WHERE covered), 0) AS total_rows,
                COUNT(*)       FILTER (WHERE covered)              AS seasons_loaded,
                MIN(season)    FILTER (WHERE covered)              AS min_season,
                MAX(season)    FILTER (WHERE covered)              AS max_season,
                MAX(loaded_at) FILTER (WHERE covered)              AS last_loaded
            FROM flagged
            GROUP BY dataset_id, table_name
            ORDER BY dataset_id
        """, params)
        covered_seasons = _covered_seasons(has_status)
        issues = _ingest_issues(has_status)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)

    issues_by_dataset: Dict[str, list[dict]] = {}
    for issue in issues:
        issues_by_dataset.setdefault(issue["dataset_id"], []).append(
            {"season": issue["season"], "status": issue["status"], "detail": issue["detail"]}
        )
    for row in rows:
        have = covered_seasons.get((row["dataset_id"], row["table_name"]), set())
        row["missing_seasons"] = _interior_gaps(row["min_season"], row["max_season"], have)
        # `issues` is already ordered by season, which is the order it is reported in.
        row["issues"] = issues_by_dataset.get(row["dataset_id"], [])
    return {"datasets": rows, "total_datasets": len(rows)}


@_refusals_as_errors
def nfl_roster(
    team: str | None = None,
    season: int | None = None,
    position: str | None = None,
) -> Dict[str, Any]:
    """
    Look up roster data — who was on a team's roster in a given season.
    Returns name, position, status, jersey number, experience, and measurables.
    `truncated` is True when more rows matched than the 500 rows returned.
    """
    _check_coverage("nfl_roster", season)
    conditions, params = [], []
    if team:
        codes = _resolve_team(team, stored_in=("rosters", "team")).codes
        conditions.append(_match_codes("team", codes))
        params.extend(codes)
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
        LIMIT {_MAX_ROWS + 1}
    """
    try:
        rows = _execute(sql, params if params else None)
        rows, truncated = _cap(rows, _MAX_ROWS)
        return {"players": rows, "count": len(rows), "truncated": truncated}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


@_refusals_as_errors
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
    _check_coverage("nfl_injuries", season)
    conditions, params = [], []
    if team:
        codes = _resolve_team(team, stored_in=("injuries", "team")).codes
        conditions.append(_match_codes("team", codes))
        params.extend(codes)
    if season is not None:
        conditions.append("season = ?")
        params.append(int(season))
    if week is not None:
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
        LIMIT {_MAX_ROWS + 1}
    """
    try:
        rows = _execute(sql, params if params else None)
        rows, truncated = _cap(rows, _MAX_ROWS)
        return {"injuries": rows, "count": len(rows), "truncated": truncated}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


@_refusals_as_errors
def nfl_schedule(
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    season_type: str | None = None,
) -> Dict[str, Any]:
    """
    Look up game schedules and results. Filter by team, season, week, or season type.
    Returns scores, spread lines, weather, stadium, and coaching info.
    `truncated` is True when more rows matched than the 500 rows returned.
    `season_type` POST matches every playoff round (WC, DIV, CON, SB); a single round
    can also be asked for by its code.
    """
    if season_type is not None:
        season_type = _choice("season_type", season_type, SCHEDULE_SEASON_TYPES)
    _check_coverage("nfl_schedule", season)
    conditions, params = [], []
    if team:
        codes = _resolve_team(team, stored_in=("schedules", "home_team")).codes
        conditions.append(
            f"({_match_codes('home_team', codes)} OR {_match_codes('away_team', codes)})"
        )
        params.extend([*codes, *codes])
    if season is not None:
        conditions.append("season = ?")
        params.append(int(season))
    if week is not None:
        conditions.append("week = ?")
        params.append(int(week))
    if season_type is not None:
        game_types = _PLAYOFF_GAME_TYPES if season_type == "POST" else (season_type,)
        conditions.append(_match_codes("game_type", game_types))
        params.extend(game_types)

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
        LIMIT {_MAX_ROWS + 1}
    """
    try:
        rows = _execute(sql, params if params else None)
        rows, truncated = _cap(rows, _MAX_ROWS)
        return {"games": rows, "count": len(rows), "truncated": truncated}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


# ── Team records (1970–present) ───────────────────────────────────────────────

# The opt-in 1970-1998 records pull (see registry.py) and the value it writes in
# team_records.source. `nfl_team_records` reads the log under this id to tell "nobody has
# fetched this season yet" apart from "ESPN was fetched and does not have this franchise".
_ESPN_RECORDS_DATASET_ID = "espn_records"
_ESPN_SOURCE = "espn"


def _requested_record_span(
    season: int | None,
    from_season: int | None,
    to_season: int | None,
) -> tuple[int, int] | None:
    """The season span a records request covers, or None when it has no both ends.

    A one-ended request ("1970 onward") has no fixed answer to look up: a later ingest
    could still fill it in. A single season is a span whose ends are equal.
    """
    if season is not None:
        return (int(season), int(season))
    if from_season is not None and to_season is not None:
        return (int(min(from_season, to_season)), int(max(from_season, to_season)))
    return None


def _espn_covered_seasons(first: int, last: int) -> set[int]:
    """Seasons in `first..last` the ingest log shows the ESPN records pull already covered.

    A `loaded` or `partial` row for the `espn_records` dataset means the pull ran for that
    season, so its answer is final: a franchise ESPN's archive does not carry will not
    appear because the ingest ran again. An empty set keeps the advice to run the opt-in
    pull, and that includes a log that cannot be read at all — a tool must not stop
    pointing at the ingest just because this extra lookup failed.
    """
    try:
        predicate, params = _coverage_filter(_metadata_has_status_column())
        rows = _execute(
            "SELECT DISTINCT season FROM _ingest_metadata "
            "WHERE dataset_id = ? AND season >= ? AND season <= ? "
            f"AND ({predicate})",
            [_ESPN_RECORDS_DATASET_ID, first, last, *params],
        )
    except (duckdb.Error, ValueError, TimeoutError):
        logger.warning(
            "Could not check %s coverage in the ingest log", _ESPN_RECORDS_DATASET_ID,
            exc_info=True,
        )
        return set()
    return {row["season"] for row in rows}


def _espn_pull_covered(span: tuple[int, int] | None) -> bool:
    """True when the ingest log already holds ESPN rows for a span wholly before 1999."""
    if span is None or span[1] >= RECORDS_FIRST_NFLVERSE_SEASON:
        return False
    return bool(_espn_covered_seasons(*span))


def _missing_record(season: int | None, espn_covered: bool) -> Dict[str, Any]:
    """`status`/`reason` for a franchise-season the franchise played but no row holds."""
    note: Dict[str, Any] = {"status": Availability.RECORD_NOT_AVAILABLE}
    if espn_covered:
        note["reason"] = (
            "No stored record for that franchise and season. The ESPN records pull "
            "already covered that season and published no record for this franchise, "
            "so running the ingest again will not add it."
        )
        note["source_checked"] = _ESPN_SOURCE
    elif season is not None and season < RECORDS_FIRST_NFLVERSE_SEASON:
        note["reason"] = (
            "No stored record for that franchise and season. Records before "
            f"{RECORDS_FIRST_NFLVERSE_SEASON} come from the opt-in ESPN pull: "
            "run 'nfl-mcp ingest --pre1999'."
        )
    else:
        note["reason"] = "No stored record for that franchise and season."
    return note


def _record_gaps(key: str, span: tuple[int, int], have: set[int]) -> list[Dict[str, Any]]:
    """One entry per season of a team's range that returned no row, with the reason why.

    The range is clipped to the records floor and the current season, so it holds at most
    one entry per season since 1970.
    """
    first = max(span[0], RECORDS_FIRST_SEASON)
    last = min(span[1], current_season())
    missing = [season for season in range(first, last + 1) if season not in have]
    if not missing:
        return []
    espn = set()
    if missing[0] < RECORDS_FIRST_NFLVERSE_SEASON:
        espn = _espn_covered_seasons(missing[0], min(last, RECORDS_FIRST_NFLVERSE_SEASON - 1))
    return [
        {"season": season, **(availability(key, season) or _missing_record(season, season in espn))}
        for season in missing
    ]


def _nothing_stored(
    key: str | None,
    season: int | None,
    span: tuple[int, int] | None,
    gaps: list[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    """`status`/`reason` for a records query that matched no row."""
    if span is not None and span[1] < RECORDS_FIRST_SEASON:
        return availability(key, span[1]) if key else before_records_coverage()
    if key is None:
        reason = "No stored records for that season."
        if span is not None and span[1] < RECORDS_FIRST_NFLVERSE_SEASON:
            if not _espn_pull_covered(span):
                reason += (
                    f" Records before {RECORDS_FIRST_NFLVERSE_SEASON} come from the opt-in "
                    "ESPN pull: run 'nfl-mcp ingest --pre1999'."
                )
        return {"status": Availability.RECORD_NOT_AVAILABLE, "reason": reason}
    statuses = {gap["status"] for gap in gaps or []}
    if len(statuses) == 1 and Availability.RECORD_NOT_AVAILABLE not in statuses:
        # Every season of the range has the same franchise reason (HOU 1990-1995).
        return {k: v for k, v in gaps[0].items() if k != "season"}
    return _missing_record(season, _espn_pull_covered(span))


@_refusals_as_errors
def nfl_team_records(
    team: str | None = None,
    season: int | None = None,
    from_season: int | None = None,
    to_season: int | None = None,
) -> Dict[str, Any]:
    """
    PREFERRED for win-loss / standings questions. Regular-season records per franchise,
    1970 to present. Teams are given by their CURRENT abbreviation, so the Raiders are
    'LV' in every season; each row also carries `club` (the name the franchise played that
    season under), `source_team` (the code the source used) and `complete` (False while
    the season is still being played, so the record is to date, not final).
    `truncated` is True when more rows matched than the 500 rows returned.

    A team plus a two-ended range also returns `gaps`: one entry per season in the range
    with no row, each with a `status` and `reason`. A query that matched nothing carries a
    `status` and `reason` at the top level. A season after the current one is refused with
    OUT_OF_COVERAGE.

    Records before 1999 come from ESPN's opt-in ingest (run 'nfl-mcp ingest --pre1999');
    a franchise ESPN's archive does not cover is reported as record_not_available rather
    than estimated. Once the ingest log shows that pull covered the season, the reason says
    the source has no record for that franchise instead of telling you to ingest again.
    """
    if not team and season is None and from_season is None and to_season is None:
        return _err(ErrorCode.INVALID_INPUT, "Provide team, season, or a from_season/to_season range.")

    key = _resolve_team(team).key if team else None
    _check_coverage("nfl_team_records", season, from_season, to_season)

    if key and season is not None:
        note = availability(key, season)
        if note:
            return {"team": key, "season": int(season), "records": [], "count": 0,
                    "truncated": False, **note}

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
        LIMIT {_MAX_ROWS + 1}
    """
    try:
        rows = _execute(sql, params if params else None)
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)

    rows, truncated = _cap(rows, _MAX_ROWS)

    for row in rows:
        row["club"] = era_name(row["season"], row["team"])
        row["complete"] = row.get("games") == expected_games(row["season"])

    result: Dict[str, Any] = {"records": rows, "count": len(rows), "truncated": truncated}
    span = _requested_record_span(season, from_season, to_season)
    gaps = None
    if key and season is None and span is not None:
        gaps = _record_gaps(key, span, {row["season"] for row in rows})
        result["gaps"] = gaps
    if not rows:
        result.update(_nothing_stored(key, season, span, gaps))
    return result


@_refusals_as_errors
def nfl_snap_counts(
    player: str | None = None,
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    position: str | None = None,
) -> Dict[str, Any]:
    """
    Look up snap count data — how many snaps a player or team unit played.
    Returns offense, defense, and special teams snap counts and percentages.
    `truncated` is True when more rows matched than the 500 rows returned.
    """
    _check_coverage("nfl_snap_counts", season)
    conditions, params = [], []
    if player:
        player = _normalize_player_name(player)
        conditions.append("player ILIKE ?")
        params.append(f"%{player}%")
    if team:
        codes = _resolve_team(team, stored_in=("snap_counts", "team")).codes
        conditions.append(_match_codes("team", codes))
        params.extend(codes)
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
               st_snaps, st_pct
        FROM snap_counts
        WHERE {where}
        ORDER BY season, week, team, offense_snaps DESC NULLS LAST
        LIMIT {_MAX_ROWS + 1}
    """
    try:
        rows = _execute(sql, params if params else None)
        rows, truncated = _cap(rows, _MAX_ROWS)
        return {"snap_counts": rows, "count": len(rows), "truncated": truncated}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


@_refusals_as_errors
def nfl_fantasy_opportunity(
    player: str | None = None,
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    position: str | None = None,
) -> Dict[str, Any]:
    """
    Look up fantasy football opportunity data — target share, air yards share,
    carry share, and opportunity scores per player per week. Available 2006–present.
    """
    _check_coverage("nfl_fantasy_opportunity", season)
    conditions, params = [], []
    if player:
        conditions.append("full_name ILIKE ?")
        params.append(f"%{player}%")
    if team:
        codes = _resolve_team(team, stored_in=("ff_opportunity", "posteam")).codes
        conditions.append(_match_codes("posteam", codes))
        params.extend(codes)
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
        LIMIT {_MAX_ROWS + 1}
    """
    try:
        rows = _execute(sql, params if params else None)
        rows, truncated = _cap(rows, _MAX_ROWS)
        return {"fantasy_opportunity": rows, "count": len(rows), "truncated": truncated}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


@_refusals_as_errors
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
    scope = _choice("scope", scope or "draft", RANKING_SCOPES)

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
        conditions.append(f"{player_col} ILIKE ?")
        params.append(f"%{player}%")
    if position:
        conditions.append("pos ILIKE ?")
        params.append(position.upper())
    if team:
        codes = _resolve_team(team, stored_in=(table, "team")).codes
        conditions.append(_match_codes("team", codes))
        params.extend(codes)
    if ranking_set:
        conditions.append(f"{set_col} ILIKE ?")
        params.append(f"%{ranking_set}%")

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"""
        SELECT {select}
        FROM {table}
        WHERE {where}
        ORDER BY ecr ASC NULLS LAST
        LIMIT {limit + 1}
    """
    try:
        rows = _execute(sql, params if params else None)
        rows, truncated = _cap(rows, limit)
        return {"fantasy_rankings": rows, "scope": scope, "count": len(rows),
                "truncated": truncated}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


@_refusals_as_errors
def nfl_ftn_charting(
    team: str | None = None,
    opponent: str | None = None,
    player: str | None = None,
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
    if season_type is not None:
        season_type = _choice("season_type", season_type, SEASON_TYPES)
    _check_coverage("nfl_ftn_charting", season, season_from, season_to)
    conditions = ["f.nflverse_game_id = p.game_id", "f.nflverse_play_id = p.play_id",
                  "p.play_type IN ('pass', 'run')"]
    params: list[Any] = []
    if team:
        codes = _resolve_team(team, stored_in=("plays", "posteam")).codes
        conditions.append(_match_codes("p.posteam", codes))
        params.extend(codes)
    if opponent:
        codes = _resolve_team(opponent, param="opponent", stored_in=("plays", "defteam")).codes
        conditions.append(_match_codes("p.defteam", codes))
        params.extend(codes)
    if player:
        conditions.append(
            "(p.passer_player_name ILIKE ? OR p.rusher_player_name ILIKE ? "
            "OR p.receiver_player_name ILIKE ?)"
        )
        params.extend([f"%{player}%"] * 3)
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
    if season_type is not None:
        conditions.append("p.season_type = ?")
        params.append(season_type)

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
        return _error_from_exception(e)

    agg = rows[0] if rows else {}
    total = agg.get("total_plays") or 0
    if not total:
        return {"charting": {}, "total_plays": 0}

    def _pct(n: Any) -> float:
        return round(100.0 * (n or 0) / total, 1)

    def _avg(v: Any) -> float | None:
        return round(v, 2) if v is not None else None

    return {
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
    }


@_refusals_as_errors
def nfl_td_luck(
    player: str | None = None,
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

    _check_coverage("nfl_td_luck", season)
    conditions, params = [], []
    if player:
        conditions.append("full_name ILIKE ?")
        params.append(f"%{player}%")
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())
    if team:
        codes = _resolve_team(team, stored_in=("player_td_luck", "team")).codes
        conditions.append(_match_codes("team", codes))
        params.extend(codes)
    if season is not None:
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
        LIMIT {limit + 1}
    """
    try:
        rows = _execute(sql, params if params else None)
        rows, truncated = _cap(rows, limit)
        return {"td_luck": rows, "count": len(rows), "truncated": truncated}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


@_refusals_as_errors
def nfl_role_trend(
    player: str | None = None,
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

    _check_coverage("nfl_role_trend", season)
    conditions, params = [], []
    if player:
        conditions.append("full_name ILIKE ?")
        params.append(f"%{player}%")
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())
    if team:
        codes = _resolve_team(team, stored_in=("player_role_trend", "team")).codes
        conditions.append(_match_codes("team", codes))
        params.extend(codes)
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
        LIMIT {limit + 1}
    """
    try:
        rows = _execute(sql, params if params else None)
        rows, truncated = _cap(rows, limit)
        return {"role_trend": rows, "count": len(rows), "truncated": truncated}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


@_refusals_as_errors
def nfl_separation_opportunity(
    player: str | None = None,
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

    _check_coverage("nfl_separation_opportunity", season)
    conditions, params = [], []
    if player:
        conditions.append("full_name ILIKE ?")
        params.append(f"%{player}%")
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())
    if team:
        codes = _resolve_team(team, stored_in=("player_separation_opportunity", "team")).codes
        conditions.append(_match_codes("team", codes))
        params.extend(codes)
    if season is not None:
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
        LIMIT {limit + 1}
    """
    try:
        rows = _execute(sql, params if params else None)
        rows, truncated = _cap(rows, limit)
        return {"separation_opportunity": rows, "count": len(rows), "truncated": truncated}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


@_refusals_as_errors
def nfl_drop_rate(
    player: str | None = None,
    team: str | None = None,
    season: int | None = None,
    min_targets: int | None = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Drop rate — share of catchable targets dropped per receiver-season from FTN
    charting (2022+) joined to plays (player_drop_rate). Also reports contested
    targets and created receptions; `truncated` is True when more rows matched than
    the `limit` returns. Player names are nflverse short form
    ('J.Jefferson'). Default sort is highest drop rate first (drop_rate_pct DESC).
    """
    try:
        limit = max(1, min(int(limit), _MAX_ROWS))
    except (TypeError, ValueError):
        limit = 50

    _check_coverage("nfl_drop_rate", season)
    conditions, params = [], []
    if player:
        conditions.append("player ILIKE ?")
        params.append(f"%{player}%")
    if team:
        codes = _resolve_team(team, stored_in=("player_drop_rate", "team")).codes
        conditions.append(_match_codes("team", codes))
        params.extend(codes)
    if season is not None:
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
        LIMIT {limit + 1}
    """
    try:
        rows = _execute(sql, params if params else None)
        rows, truncated = _cap(rows, limit)
        return {"drop_rate": rows, "count": len(rows), "truncated": truncated}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


@_refusals_as_errors
def nfl_contract_value(
    player: str | None = None,
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

    _check_coverage("nfl_contract_value", season)
    conditions, params = [], []
    if player:
        conditions.append("full_name ILIKE ?")
        params.append(f"%{player}%")
    if position:
        conditions.append("position ILIKE ?")
        params.append(position.upper())
    if team:
        codes = _resolve_team(team, stored_in=("player_contract_value", "team")).codes
        conditions.append(_match_codes("team", codes))
        params.extend(codes)
    if season is not None:
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
        LIMIT {limit + 1}
    """
    try:
        rows = _execute(sql, params if params else None)
        rows, truncated = _cap(rows, limit)
        return {"contract_value": rows, "count": len(rows), "truncated": truncated}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


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
        LIMIT {limit + 1}
    """
    try:
        rows = _execute(sql, params if params else None)
        rows, truncated = _cap(rows, limit)
        return {"injury_return": rows, "count": len(rows), "truncated": truncated}
    except (duckdb.Error, ValueError, TimeoutError) as e:
        logger.error("Tool error: %s", e, exc_info=True)
        return _error_from_exception(e)


__all__ = [
    "nfl_schema", "nfl_status", "nfl_query",
    "nfl_search_plays", "nfl_team_stats", "nfl_player_stats", "nfl_compare",
    "nfl_catalog", "nfl_roster", "nfl_injuries", "nfl_schedule", "nfl_snap_counts",
    "nfl_fantasy_opportunity", "nfl_fantasy_rankings", "nfl_ftn_charting",
    "nfl_td_luck", "nfl_role_trend", "nfl_separation_opportunity",
    "nfl_drop_rate", "nfl_contract_value", "nfl_injury_return",
    "nfl_team_records",
]
