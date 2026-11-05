"""Checks a freshly ingested database slot before it replaces the active one.

`nfl-mcp validate` and update_db.sh call run_checks. Every problem with a slot (a missing
file, a corrupt file, a missing column) comes back as a failing result, never an exception.
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb

from .database import _connection_config
from .required_columns import REQUIRED_COLUMNS

# A new slot may hold up to 1 percent fewer rows than the active one before it fails.
ROW_TOLERANCE = 0.99

_PROBLEMS = (duckdb.Error, OSError, ValueError)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _open(path: Path) -> duckdb.DuckDBPyConnection:
    """Open a slot read-only with the same locked config the server uses."""
    if not path.is_file():
        raise OSError(f"database file not found at {path}")
    return duckdb.connect(str(path), read_only=True, config=_connection_config())


def _columns(conn) -> dict[str, set[str]]:
    rows = conn.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'main'"
    ).fetchall()
    found: dict[str, set[str]] = {}
    for table, column in rows:
        found.setdefault(table, set()).add(column)
    return found


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _row_count(conn, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]


def _newest_week(conn, columns: dict[str, set[str]]) -> tuple[int, int] | None:
    """Newest (season, week) pair in plays, or None when there is none or plays lacks them."""
    if not {"season", "week"} <= columns.get("plays", set()):
        return None
    row = conn.execute(
        "SELECT season, week FROM plays WHERE season IS NOT NULL AND week IS NOT NULL "
        "ORDER BY season DESC, week DESC LIMIT 1"
    ).fetchone()
    return (row[0], row[1]) if row else None


def _check_schema(columns: dict[str, set[str]]) -> list[CheckResult]:
    missing_tables = sorted(t for t in REQUIRED_COLUMNS if t not in columns)
    missing_columns = sorted(
        f"{t}.{c}" for t, cols in REQUIRED_COLUMNS.items() if t in columns
        for c in cols if c not in columns[t]
    )
    return [
        CheckResult("required_tables", not missing_tables,
                    "missing tables: " + ", ".join(missing_tables) if missing_tables
                    else f"all {len(REQUIRED_COLUMNS)} required tables present"),
        CheckResult("required_columns", not missing_columns,
                    "missing columns: " + ", ".join(missing_columns) if missing_columns
                    else "all required columns present"),
    ]


def _check_row_counts(new, new_cols, current, current_cols) -> CheckResult:
    low = []
    for table in sorted(set(new_cols) & set(current_cols)):
        new_n, current_n = _row_count(new, table), _row_count(current, table)
        if new_n < current_n * ROW_TOLERANCE:
            low.append(f"{table} has {new_n} rows, current slot has {current_n}")
    if low:
        return CheckResult("row_count_floor", False, "; ".join(low))
    return CheckResult("row_count_floor", True, "no table shrank beyond the 1 percent tolerance")


def _check_duplicates(conn, columns) -> CheckResult:
    if not {"game_id", "play_id"} <= columns.get("plays", set()):
        return CheckResult("duplicate_plays", True, "skipped: plays.game_id or play_id missing")
    extra = conn.execute(
        "SELECT COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT game_id, play_id FROM plays)) "
        "FROM plays"
    ).fetchone()[0]
    if extra:
        return CheckResult("duplicate_plays", False,
                           f"{extra} duplicate rows on (game_id, play_id) in plays")
    return CheckResult("duplicate_plays", True, "no duplicate (game_id, play_id)")


def _check_newest_week(new_newest, current_newest, has_current, columns) -> CheckResult:
    if "plays" not in columns:
        return CheckResult("newest_week", True, "skipped: plays table missing")
    if new_newest is None:
        return CheckResult("newest_week", False,
                           "cannot find the newest week: no plays row has both season and week")
    label = f"season {new_newest[0]} week {new_newest[1]}"
    if not has_current or current_newest is None:
        return CheckResult("newest_week", True, f"newest week is {label}; nothing to compare")
    if new_newest < current_newest:
        return CheckResult("newest_week", False,
                           f"newest week {label} is older than the current slot "
                           f"(season {current_newest[0]} week {current_newest[1]})")
    return CheckResult("newest_week", True, f"newest week {label}, not older than current")


def _check_null_keys(conn, columns, newest) -> CheckResult:
    if newest is None or not {"epa", "posteam"} <= columns.get("plays", set()):
        return CheckResult("null_keys", True, "skipped: no newest week or key columns")
    season, week = newest
    n, epa, posteam = conn.execute(
        "SELECT COUNT(*), COUNT(epa), COUNT(posteam) FROM plays WHERE season = ? AND week = ?",
        [season, week],
    ).fetchone()
    empty = [c for c, count in (("epa", epa), ("posteam", posteam)) if count == 0]
    if empty:
        return CheckResult("null_keys", False,
                           f"{', '.join(empty)} NULL in all {n} plays rows of season {season} week {week}")
    return CheckResult("null_keys", True, f"epa and posteam populated in season {season} week {week}")


def run_checks(new_db: Path, current_db: Path | None = None) -> list[CheckResult]:
    """Run every check on new_db, comparing with current_db when given. Never raises."""
    new_conn = current_conn = None
    results: list[CheckResult] = []
    try:
        try:
            new_conn = _open(Path(new_db))
        except _PROBLEMS as e:
            return [CheckResult("open_new", False, f"cannot open {new_db} read-only: {e}")]
        if current_db is not None:
            try:
                current_conn = _open(Path(current_db))
            except _PROBLEMS as e:
                results.append(CheckResult("open_current", False,
                                           f"cannot open {current_db} read-only: {e}"))
        try:
            results += _run(new_conn, current_conn)
        except _PROBLEMS as e:
            results.append(CheckResult("run_checks", False, f"query failed: {e}"))
        return results
    finally:
        for conn in (new_conn, current_conn):
            if conn is not None:
                conn.close()


def _run(new, current) -> list[CheckResult]:
    new_cols = _columns(new)
    results = _check_schema(new_cols)
    current_cols = _columns(current) if current is not None else {}
    if current is not None:
        results.append(_check_row_counts(new, new_cols, current, current_cols))
    results.append(_check_duplicates(new, new_cols))
    newest = _newest_week(new, new_cols)
    current_newest = _newest_week(current, current_cols) if current is not None else None
    results.append(_check_newest_week(newest, current_newest, current is not None, new_cols))
    results.append(_check_null_keys(new, new_cols, newest))
    return results
