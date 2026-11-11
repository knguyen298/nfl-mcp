"""
NFL data ingestion into DuckDB.

Supports all nflreadpy datasets via a declarative registry.
PBP ingestion preserves its existing enhanced_description logic.

Called by: nfl-mcp init / nfl-mcp ingest
"""

import contextlib
import datetime
import importlib.metadata
import os
import weakref
import duckdb
import polars as pl
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from nflreadpy.utils_date import get_current_season as _nflreadpy_current_season
from tqdm import tqdm

from .seasons import (
    FIRST_SEASON,
    EARLIEST_SEASON,
    all_seasons,
    current_season,
    season_settled_on,
)
from .registry import (
    KNOWN_SOURCES,
    PBP_DATASET_ID,
    PBP_LOADER_FN,
    PBP_TABLE_NAME,
    REGISTRY,
    SOURCE_ESPN,
    SOURCE_NFLREADPY,
)
from .status import IngestStatus


def _register(conn: duckdb.DuckDBPyConnection, name: str, table: object) -> None:
    """conn.register after closing any open result. DuckDB 2.0 refuses to register a view while
    an earlier query on the connection is not read to the end (for example, after fetchone)."""
    conn.complete()
    conn.register(name, table)


def _refresh_seasons() -> set[int]:
    """Seasons that are re-fetched on every ingest, even if already recorded as loaded.

    The in-progress season gains new games every week, so its 'already loaded' record
    only means 'loaded at some point', never 'complete'. Resolved at call time (not
    import time) so a long-running process or a test can override current_season.
    """
    return {current_season()}


def _apply_duckdb_pragmas(conn: duckdb.DuckDBPyConnection, db_path: str) -> None:
    """Bound DuckDB memory so large ingests spill to disk instead of being OOM-killed.

    Driven by env vars so containers/CI can cap memory (e.g. when baking the
    database into a Docker image) while local runs keep DuckDB's defaults:
      NFL_MCP_DUCKDB_MEMORY_LIMIT  e.g. '5GB'
      NFL_MCP_DUCKDB_THREADS       e.g. '2'
      NFL_MCP_DUCKDB_TEMP_DIR      where to spill, e.g. '/tmp/.duckdb_spill'

    The spill directory comes from the same variable nfl_mcp/database.py reads for a
    read-only connection, so the image sends the spill of a query and of an ingest to one
    place. With the variable unset the ingest spills next to the database file.
    """
    mem = os.getenv("NFL_MCP_DUCKDB_MEMORY_LIMIT")
    if mem:
        conn.execute(f"SET memory_limit='{mem}'")
        spill = (os.getenv("NFL_MCP_DUCKDB_TEMP_DIR")
                 or str(Path(db_path).parent / ".duckdb_spill"))
        conn.execute(f"SET temp_directory='{spill}'")
    threads = os.getenv("NFL_MCP_DUCKDB_THREADS")
    if threads:
        conn.execute(f"SET threads={int(threads)}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _table_exists(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    """Return True if a table named `name` exists in the main schema.

    The only way this module asks whether an object is there. Asking beats catching
    duckdb.CatalogException, because that exception also fires for a missing column or a
    missing function: a schema problem used to read as an empty database here, which made a
    season look never-loaded and made a writer try to CREATE a table that was already there.

    `name` is a bound parameter, never interpolated into the SQL. DuckDB lists views in
    information_schema.tables too, and a view answers True here, which is what every caller
    wants since a view can be SELECTed from like the table it stands in for. A table in
    another schema answers False: the ingest only ever writes to main.
    """
    row = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_name = ?",
        [name],
    ).fetchone()
    return (row[0] or 0) > 0


def _newest_season(dataset_def) -> int | None:
    """Newest season nflreadpy's own validator accepts for this dataset, None if unbounded.

    Asks nflreadpy's get_current_season (imported at module load, so a fake nflreadpy in a
    test cannot replace it) with the registry rule's roster flag, so the bound rolls over
    on the same day as the loader's check.
    """
    from .registry import NEWEST_SEASON_RULES
    if dataset_def.newest_season is None:
        return None
    roster, offset = NEWEST_SEASON_RULES[dataset_def.newest_season]
    return _nflreadpy_current_season(roster=roster) + offset


def _clip_to_window(season_list, dataset_def):
    """Drop seasons outside a dataset's own published coverage window (no network).

    Seasons outside min_season/max_season go silently. Seasons past _newest_season are not
    published yet (participation during the season, most loaders from March until Week 1):
    they are skipped with a note, not fetched, and not counted as failures.
    """
    newest = _newest_season(dataset_def)
    kept = []
    for s in season_list:
        if (dataset_def.min_season is not None and s < dataset_def.min_season) or (
                dataset_def.max_season is not None and s > dataset_def.max_season):
            continue
        if newest is not None and s > newest:
            print(f"    {dataset_def.dataset_id} {s}: not published yet, skipping")
            continue
        kept.append(s)
    return kept




def _note_failure(dataset_id: str, season: int | None, err: Exception | str) -> None:
    """Report a failure to the run in progress without writing a metadata row.

    For a failure that is not the outcome of one season's fetch: a restamp that could not
    finish, rows nothing can re-fetch, or an error outside the per-season loop. Writing a
    'failed' row with season NULL for a seasonal dataset would sit in the log for good, since
    no later success replaces a NULL-season row of a seasonal dataset. A failed season fetch
    goes through _record_attempt instead, which also writes the row.
    """
    _note_attempt(dataset_id, season, IngestStatus.FAILED, _clip_detail(str(err)), "")


def failure_labels(failures: list[str]) -> list[str]:
    """The "<dataset>" or "<dataset> <season>" of each failure, in order, each once.

    A failure is "<label>: <error>" (see IngestRunResult.failures); dataset ids hold no colon.
    """
    labels: list[str] = []
    for failure in failures:
        label = failure.split(":", 1)[0].strip()
        if label not in labels:
            labels.append(label)
    return labels


def _safe_col(name: str) -> str:
    return name.replace(".", "_").replace(" ", "_").replace("-", "_")


def _safe_rename(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({c: _safe_col(c) for c in df.columns})


def _str(val) -> str:
    if val is None:
        return ""
    s = str(val)
    return "" if s in ("None", "nan", "NaN", "") else s


def _duckdb_type_for_polars(dtype: pl.DataType) -> str:
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                 pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return "BIGINT"
    if dtype in (pl.Float32, pl.Float64):
        return "DOUBLE"
    if dtype == pl.Boolean:
        return "BOOLEAN"
    if dtype == pl.Date:
        return "DATE"
    if dtype == pl.Time:
        return "TIME"
    if dtype == pl.Datetime:
        return "TIMESTAMP"
    return "VARCHAR"


# ── Metadata table ─────────────────────────────────────────────────────────────

# One row per dataset+season attempt, tagged with an IngestStatus. A missing row is not a
# verdict: it means never tried, or that the metadata table predates this code. The four
# outcomes are recorded instead, so a gap is data the query tools can read, not a line in
# someone's terminal.
#
# `loaded` and `partial` mean the season's rows are in the database, so later runs skip it.
# `failed` and `empty` mean nothing landed, so later runs retry the season.
_DETAIL_MAX_CHARS = 300  # an exception string can carry a whole HTTP body; keep it short

# The status a row takes when nothing wrote one. The code before statuses wrote a row only
# when rows landed, so 'loaded' is true of every row it left behind. It is read from
# IngestStatus rather than spelled out again, because it goes into DDL as a SQL literal.
_LOADED_DEFAULT = IngestStatus.LOADED.value

# loader_fn on the metadata row of a table this project computes from tables it already has
# (the fantasy derived tables, team_records). Such a row is not the record of a fetch, so
# the run summary leaves it out: nothing was downloaded, and a derived table can be rebuilt
# from data that is already on disk.
DERIVED_LOADER = "derived"

# loader_fn of the ESPN pull. Like a derived row, it records no nflreadpy version.
ESPN_LOADER = "espn_records"

# Columns added to _ingest_metadata after its first release, with their DDL. An older
# database gets each one the first time an ingest or an upgrade opens it.
_ADDED_METADATA_COLUMNS = {
    "status": f"VARCHAR DEFAULT '{_LOADED_DEFAULT}'",
    "detail": "VARCHAR",
    "nflreadpy_version": "VARCHAR",
}


def _clip_detail(detail: str | None) -> str | None:
    """Trim a free-form reason to what belongs in the column; NULL stays NULL (no reason)."""
    if detail is None:
        return None
    return detail[:_DETAIL_MAX_CHARS]


def _metadata_columns(conn: duckdb.DuckDBPyConnection) -> set[str]:
    return {r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'main' AND table_name = '_ingest_metadata'"
    ).fetchall()}


def _ensure_metadata_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create _ingest_metadata if absent, and add the later columns if an old table lacks them.

    status holds an IngestStatus value; detail holds a short human-readable reason for the
    outcomes that loaded no data; nflreadpy_version names the nflreadpy that fetched the rows
    (NULL for derived rows and the ESPN pull). A database created before those columns existed
    gets them added here, and the DEFAULT backfills its rows to 'loaded' — the old code only
    ever wrote a row when rows actually landed. A column is added only when it is missing, so
    a call on an up-to-date table runs no ALTER at all: DuckDB cannot commit an ALTER after
    row changes in the same transaction.
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS _ingest_metadata (
            dataset_id  VARCHAR NOT NULL,
            table_name  VARCHAR NOT NULL,
            season      INTEGER,          -- NULL for static (non-seasonal) datasets
            row_count   BIGINT,
            loaded_at   TIMESTAMP NOT NULL,
            loader_fn   VARCHAR,
            status      VARCHAR DEFAULT '{_LOADED_DEFAULT}',
            detail      VARCHAR,          -- short reason, NULL when there is nothing to say
            nflreadpy_version VARCHAR     -- nflreadpy that fetched the rows; NULL for derived rows
        )
    """)
    have = _metadata_columns(conn)
    for column, ddl in _ADDED_METADATA_COLUMNS.items():
        if column not in have:
            conn.execute(f"ALTER TABLE _ingest_metadata ADD COLUMN {column} {ddl}")


def _stored_rows_predicate(conn: duckdb.DuckDBPyConnection) -> tuple[str, list]:
    """SQL fragment and its bound values that keeps only the rows whose data is in the table.

    A `failed` or an `empty` row records an attempt that landed nothing. A read that asks
    which seasons this database holds must skip those rows, or a season whose fetch keeps
    failing is read as a season that exists. The two callers are the restamp plan and the
    season list of a schema upgrade.

    The fragment is meant to sit at the end of a WHERE clause, so it starts with AND. It is
    empty on a database whose metadata table predates the status column: that code wrote a
    row only when rows landed, so every row it left describes loaded data.
    """
    if "status" not in _metadata_columns(conn):
        return "", []
    return " AND status IN (?, ?)", [IngestStatus.LOADED, IngestStatus.PARTIAL]


def _has_version_column(conn: duckdb.DuckDBPyConnection) -> bool:
    return "nflreadpy_version" in _metadata_columns(conn)


def _add_version_column(conn: duckdb.DuckDBPyConnection) -> None:
    """A database built before nflreadpy_version existed gets the column at the start of the next
    ingest. This must run outside a transaction: DuckDB cannot commit an ALTER after row changes."""
    if not _has_version_column(conn):
        conn.execute("ALTER TABLE _ingest_metadata ADD COLUMN nflreadpy_version VARCHAR")


def _nflreadpy_version() -> str | None:
    try:
        return importlib.metadata.version("nflreadpy")
    except importlib.metadata.PackageNotFoundError:
        return None


def _delete_metadata_row(
    conn: duckdb.DuckDBPyConnection, dataset_id: str, season: int | None
) -> None:
    """Drop the stored record for one dataset+season, if there is one (the upsert half)."""
    if season is None:
        conn.execute(
            "DELETE FROM _ingest_metadata WHERE dataset_id = ? AND season IS NULL",
            [dataset_id],
        )
    else:
        conn.execute(
            "DELETE FROM _ingest_metadata WHERE dataset_id = ? AND season = ?",
            [dataset_id, season],
        )


def _is_loaded(conn: duckdb.DuckDBPyConnection, dataset_id: str, season: int | None = None) -> bool:
    """Return True if this dataset (+ season) has rows in the database.

    Only a 'loaded' or a 'partial' record means that. A 'failed' or 'empty' record is the
    log of an attempt that put nothing in the table, so those seasons are retried on the
    next run exactly as they were when no record existed at all.

    A database with no _ingest_metadata table has loaded nothing, and says so by way of
    _table_exists. Once the table is there, a catalog error is a real problem: it is left to
    propagate instead of being answered as "not loaded", which would re-fetch the season and
    hide the problem.
    """
    if not _table_exists(conn, "_ingest_metadata"):
        return False
    if season is None:
        result = conn.execute(
            "SELECT COUNT(*) FROM _ingest_metadata "
            "WHERE dataset_id = ? AND season IS NULL AND status IN (?, ?)",
            [dataset_id, IngestStatus.LOADED, IngestStatus.PARTIAL],
        ).fetchone()
    else:
        result = conn.execute(
            "SELECT COUNT(*) FROM _ingest_metadata "
            "WHERE dataset_id = ? AND season = ? AND status IN (?, ?)",
            [dataset_id, season, IngestStatus.LOADED, IngestStatus.PARTIAL],
        ).fetchone()
    return (result[0] or 0) > 0


def _needs_refresh(conn: duckdb.DuckDBPyConnection, dataset_id: str, season: int) -> bool:
    """Return True when a season that is already recorded as loaded must be fetched again.

    A season stops changing on the day it ends, and this code calls a season ended on the
    league-year rollover (see seasons.season_settled_on). A load recorded before that day is
    not a complete season: run in January it misses the last week of the regular season, the
    playoffs and the Super Bowl. Such a season is fetched once more, and that fetch writes a
    new loaded_at, so every later run skips it. A load recorded on or after that day needs
    nothing — there is no game left to find.

    The in-progress season always refreshes; it is still gaining games.

    A season with no loaded/partial record answers False here: "never tried" and "the last
    attempt failed" are decided by the callers (see _is_loaded, and the play-by-play path
    which reads the plays table), so this function does not turn them into a re-fetch.
    """
    if season in _refresh_seasons():
        return True
    row = conn.execute(
        "SELECT max(loaded_at) FROM _ingest_metadata "
        "WHERE dataset_id = ? AND season = ? AND status IN (?, ?)",
        [dataset_id, season, IngestStatus.LOADED, IngestStatus.PARTIAL],
    ).fetchone()
    loaded_at = row[0] if row else None
    if loaded_at is None:
        return False
    # loaded_at is an instant. _record_loaded writes it from datetime.now(timezone.utc) and
    # DuckDB stores a TIMESTAMP as wall-clock time in the process's own zone, so a value read
    # back is naive. astimezone() gives that wall clock its zone back, which recovers the
    # instant, and the date is then read in UTC — the frame season_settled_on() is written in.
    utc = loaded_at.astimezone().astimezone(datetime.timezone.utc)
    return utc.date() < season_settled_on(season)


# Connections that already have an open _transaction. DuckDB raises on BEGIN inside an
# open transaction, and that error aborts the transaction it hit, so _transaction marks
# the connection it opened and a nested call does nothing: the caller that opened the
# transaction is the only one that commits or rolls back.
# WeakSet so a connection nothing else holds does not stay in the set.
_open_transactions: "weakref.WeakSet[duckdb.DuckDBPyConnection]" = weakref.WeakSet()

# Attempts written inside the open _transaction of this call stack, waiting for its COMMIT.
# None means no transaction is open, and an attempt then reaches the run at once. A
# transaction that rolls back wrote no rows, so the attempts it held are dropped: the run
# must not report a season as loaded that the database does not hold. _note_attempt fills
# this list and _transaction empties it on a successful COMMIT.
_held_notes: "ContextVar[list | None]" = ContextVar("nfl_mcp_held_ingest_notes", default=None)


@contextmanager
def _transaction(conn: duckdb.DuckDBPyConnection, rollback_failed: str = "Rollback failed"):
    """Write rows and their metadata as one unit: all of it commits, or none of it does.

    Without this, each conn.execute commits by itself, so a crash or a failing INSERT
    between the DELETE and the INSERT leaves a season with no rows while
    _ingest_metadata still says it is loaded — every later run then skips it. DuckDB
    rolls back DDL too, so a DROP + CREATE that fails partway leaves the old table.
    A ROLLBACK that fails too is printed with `rollback_failed` in front, and the original
    error is raised.

    The attempts a caller records inside the transaction reach the run in progress only
    after the COMMIT. A rollback drops them, because the rows they describe are gone. A
    nested `_transaction` on the same connection is a no-op, so the notes it makes stay in
    the outer transaction's list and commit with it.
    """
    if conn in _open_transactions:
        yield
        return
    _open_transactions.add(conn)
    token = _held_notes.set([])
    try:
        conn.execute("BEGIN")
        try:
            yield
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except Exception as rollback_err:  # noqa: BLE001 - keep the original error
                print(f"    {rollback_failed}: {rollback_err}")
            raise
        for attempt in _held_notes.get():
            _deliver_attempt(attempt)
    finally:
        _open_transactions.discard(conn)
        _held_notes.reset(token)




def _insert_metadata_row(
    conn: duckdb.DuckDBPyConnection,
    dataset_id: str,
    table_name: str,
    season: int | None,
    row_count: int,
    loader_fn: str,
    status: IngestStatus,
    detail: str | None,
) -> None:
    """INSERT one _ingest_metadata row, writing only the columns this table has.

    _ensure_metadata_table adds the later columns before any ingest writes, but the upgrade
    path can rebuild derived tables on an older table it has not altered yet; such a row
    simply has no status, detail or version to carry.
    """
    version = None if loader_fn in (DERIVED_LOADER, ESPN_LOADER) else _nflreadpy_version()
    values = {
        "dataset_id": dataset_id, "table_name": table_name, "season": season,
        "row_count": row_count, "loaded_at": datetime.datetime.now(datetime.timezone.utc),
        "loader_fn": loader_fn, "status": status, "detail": detail, "nflreadpy_version": version,
    }
    have = _metadata_columns(conn)
    columns = [c for c in values if c in have]
    conn.execute(
        f"INSERT INTO _ingest_metadata ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})",
        [values[c] for c in columns],
    )


def _record_loaded(
    conn: duckdb.DuckDBPyConnection,
    dataset_id: str,
    table_name: str,
    loader_fn: str,
    row_count: int,
    season: int | None = None,
    status: IngestStatus = IngestStatus.LOADED,
    detail: str | None = None,
) -> None:
    """Upsert a record for an attempt that put rows in the table, with the nflreadpy version
    that fetched them (derived tables and the ESPN pull are not nflreadpy loads: NULL).

    The delete and the insert commit together, so a failure cannot drop the old record
    without writing the new one. A caller already inside a _transaction keeps the record
    in that transaction instead.

    status is PARTIAL when rows landed but the season is incomplete (an ESPN pull that came
    back short of the league that year). Like LOADED it keeps the season skipped on later
    runs, because a re-fetch cannot fill a gap the publisher does not have. An attempt that
    loaded nothing is recorded by _record_attempt instead.
    """
    detail = _clip_detail(detail)
    _note_attempt(dataset_id, season, status, detail, loader_fn)
    with _transaction(conn):
        # Delete any prior record for this dataset+season before inserting fresh
        _delete_metadata_row(conn, dataset_id, season)
        _insert_metadata_row(conn, dataset_id, table_name, season, row_count, loader_fn,
                             status, detail)


def _record_attempt(
    conn: duckdb.DuckDBPyConnection,
    dataset_id: str,
    table_name: str,
    loader_fn: str,
    season: int | None,
    status: IngestStatus,
    detail: str | None = None,
) -> None:
    """Record an attempt that put no rows in the table: FAILED or EMPTY.

    The row is written with row_count 0, so it reads as "tried, nothing landed" instead of
    as data that is in the database, and the season stays not-loaded so the next run retries
    it. The reason text goes in detail, which is where a query tool can read it.

    Does nothing when a loaded or partial row already exists for that dataset+season: that
    data is still in the table, so the older record is the true one. The in-progress season
    is re-fetched on every run, so one failed fetch must not relabel the plays that are
    already stored as a failure. An earlier failed or empty row is replaced, so the record
    always describes the most recent attempt.
    """
    detail = _clip_detail(detail)
    # The run is told about the attempt before the early return below: a fetch that failed
    # against a season that already has rows is still a fetch this run made and lost, and a
    # run whose only fetch was that one must read as "nothing loaded".
    _note_attempt(dataset_id, season, status, detail, loader_fn)
    with _transaction(conn):
        if _is_loaded(conn, dataset_id, season):
            return
        _delete_metadata_row(conn, dataset_id, season)
        _insert_metadata_row(conn, dataset_id, table_name, season, 0, loader_fn, status, detail)



# ── What one run did ───────────────────────────────────────────────────────────
#
# A run of the ingest is the unit someone else acts on: `nfl-mcp ingest` turns it into an
# exit code, and the container's scheduled refresh publishes the new database only when that
# code lets it (see update_db.sh). So a run has to be able to say how it went, which is what
# the types below are for. They describe *this run's* fetches only — not the whole database,
# and not the seasons this run skipped because an earlier run already loaded them.


@dataclass(frozen=True)
class IngestAttempt:
    """One fetch this run made, and how it ended.

    season is None for a static (non-seasonal) dataset. detail is the same short reason that
    goes into the `detail` column of _ingest_metadata. source is the publisher the fetch
    asked, and it decides whether a failure of this attempt gates the run (see
    _gates_the_run).
    """

    dataset_id: str
    season: int | None
    status: IngestStatus
    detail: str | None = None
    source: str = SOURCE_NFLREADPY


@dataclass(frozen=True)
class EspnStop:
    """The ESPN records pull gave up: `failed_seasons` failed in a row, `seasons_left` were
    never asked. The seasons after the break have no metadata row at all, which is a
    different fact from "tried and failed", and this is how the run says so out loud."""

    failed_seasons: int
    seasons_left: int


@dataclass(frozen=True)
class IngestRunResult:
    """Everything one run of the ingest can say about itself."""

    attempts: list[IngestAttempt] = field(default_factory=list)
    espn_stop: EspnStop | None = None

    @property
    def counts(self) -> dict[IngestStatus, int]:
        """How many attempts ended in each status, including the statuses with none."""
        counts = {status: 0 for status in IngestStatus}
        for attempt in self.attempts:
            counts[attempt.status] += 1
        return counts

    @property
    def nothing_loaded(self) -> bool:
        """True when the run fetched at least once and not one fetch put rows in the database.

        This, and not "something failed", is the signal of a dead network. Ordinary runs
        always fail on something (play-by-play for a new season does not exist before Week 1),
        so a run that failed on every single fetch is the case worth stopping a deploy over.
        A run that only skipped already-loaded seasons attempted nothing, and is False here.
        """
        if not self.attempts:
            return False
        return not any(
            attempt.status in (IngestStatus.LOADED, IngestStatus.PARTIAL)
            for attempt in self.attempts
        )

    @property
    def any_failed(self) -> bool:
        """True when at least one attempt ended failed. `empty` is an answer, not a failure."""
        return any(attempt.status == IngestStatus.FAILED for attempt in self.attempts)

    @property
    def failures(self) -> list[str]:
        """Each failed attempt that gates the run, as "<dataset>[ <season>]: <reason>".

        This is the list the CLI prints, writes to --failures-file (as labels), and turns into
        exit 3, and the list that decides whether the schema version is stamped. A failure on
        an optional outside source is left out (see _gates_the_run) and appears in
        `optional_source_failures`. failure_labels() reads the label back from the text before
        the first colon.
        """
        return [_failure_text(a) for a in self.attempts
                if a.status == IngestStatus.FAILED and _gates_the_run(a.source)]

    @property
    def optional_source_failures(self) -> list[str]:
        """Each failed attempt on a source that does not gate the run, in the same shape."""
        return [_failure_text(a) for a in self.attempts
                if a.status == IngestStatus.FAILED and not _gates_the_run(a.source)]

    @property
    def espn_breaker_tripped(self) -> bool:
        """True when the ESPN pull stopped early instead of working through its seasons."""
        return self.espn_stop is not None


def _attempt_source(dataset_id: str) -> str:
    """The publisher an attempt belongs to, read from the registry by dataset id.

    The registry is the one place that says where a dataset's rows come from, so the run
    report does not need a second list of names. A dataset id with no registry entry (a
    caller that wrote metadata under its own name) reads as SOURCE_NFLREADPY, the source
    that gates the run.
    """
    defn = REGISTRY.get(dataset_id)
    return defn.source if defn is not None else SOURCE_NFLREADPY


def _gates_the_run(source: str) -> bool:
    """True when a failed attempt on this publisher decides the run's exit code.

    nflreadpy is the data the database is built from, so a failure there is a failure of the
    run: the CLI exits 3 and the schema version stays unstamped. An outside publisher is
    opt-in, and its rows are extra history the rest of the database does not depend on, so a
    failure there is recorded in _ingest_metadata and shown in the run summary only. ESPN is
    that source today (`nfl-mcp ingest --pre1999`). `--strict` asks for any failure at all,
    so it reads `any_failed`, which counts every source.
    """
    return source == SOURCE_NFLREADPY


def _failure_text(attempt: "IngestAttempt") -> str:
    """One failed attempt as "<dataset>[ <season>]: <reason>", the shape failure_labels reads."""
    label = (attempt.dataset_id if attempt.season is None
             else f"{attempt.dataset_id} {attempt.season}")
    return f"{label}: {attempt.detail or 'failed'}"


class _RunCollector:
    """Attempts gathered by one run, while it runs. Owned by run_ingest_datasets.

    The recording helpers write into the collector through _ACTIVE_RUN rather than through a
    module-level list, so a second run — or the next test — starts from nothing.
    """

    def __init__(self) -> None:
        self.attempts: list[IngestAttempt] = []
        self.espn_stop: EspnStop | None = None

    def note_attempt(self, attempt: "IngestAttempt") -> None:
        """Record one attempt this run made, after its write committed."""
        self.attempts.append(attempt)

    def note_espn_stop(self, failed_seasons: int, seasons_left: int) -> None:
        self.espn_stop = EspnStop(failed_seasons=failed_seasons, seasons_left=seasons_left)

    def result(self) -> IngestRunResult:
        return IngestRunResult(attempts=list(self.attempts), espn_stop=self.espn_stop)


# The run in progress, if any. ContextVar (not a global) so the value is set and restored
# around one call, and so tests that call the recording helpers on their own — which is most
# of them — simply report to nobody.
_ACTIVE_RUN: ContextVar["_RunCollector | None"] = ContextVar("nfl_mcp_active_ingest_run", default=None)


@contextmanager
def _collecting():
    """Collect the attempts of the ingest helpers called inside the block, and yield the
    collector. run_ingest_datasets does this for a whole run; the schema upgrade does it
    around each step, so a step can tell whether any of its fetches failed. The collector
    that was active before is put back on the way out."""
    collector = _RunCollector()
    token = _ACTIVE_RUN.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE_RUN.reset(token)


def _note_attempt(
    dataset_id: str,
    season: int | None,
    status: IngestStatus,
    detail: str | None,
    loader_fn: str,
) -> None:
    """Report one dataset row to the run in progress, when a run is in progress.

    Derived tables are excluded: their row records a rebuild, not a fetch, and counting one
    would let a run with no successful download still look like it loaded something.

    Inside an open _transaction the attempt waits in `_held_notes` until that transaction
    commits, so a write that rolled back never reports itself as loaded.
    """
    if loader_fn == DERIVED_LOADER:
        return
    attempt = IngestAttempt(dataset_id=dataset_id, season=season, status=status,
                            detail=detail, source=_attempt_source(dataset_id))
    held = _held_notes.get()
    if held is not None:
        held.append(attempt)
        return
    _deliver_attempt(attempt)


def _deliver_attempt(attempt: IngestAttempt) -> None:
    """Hand one attempt that really happened to the run in progress, when a run is in progress."""
    collector = _ACTIVE_RUN.get()
    if collector is None:
        return
    collector.note_attempt(attempt)


def _note_espn_stop(failed_seasons: int, seasons_left: int) -> None:
    """Report the ESPN give-up to the run in progress, when a run is in progress."""
    collector = _ACTIVE_RUN.get()
    if collector is None:
        return
    collector.note_espn_stop(failed_seasons, seasons_left)


# A dead network fails every season in the range, so the list of what went wrong is only
# useful capped: the counts say how bad it was, the first lines say what it looks like.
_SUMMARY_LINE_CAP = 10


def _print_run_summary(result: IngestRunResult) -> None:
    """Print how this run went, so the log reads as more than 'Ingestion complete!'."""
    counts = result.counts
    print("\nRun summary:")
    print(
        f"  loaded: {counts[IngestStatus.LOADED]}  "
        f"partial: {counts[IngestStatus.PARTIAL]}  "
        f"empty: {counts[IngestStatus.EMPTY]}  "
        f"failed: {counts[IngestStatus.FAILED]}"
    )
    stop = result.espn_stop
    if stop is not None:
        print(f"  ESPN pull stopped after {stop.failed_seasons} failed seasons; "
              f"{stop.seasons_left} seasons not attempted")
    unfilled = [
        a for a in result.attempts
        if a.status in (IngestStatus.FAILED, IngestStatus.EMPTY)
    ]
    for attempt in unfilled[:_SUMMARY_LINE_CAP]:
        season = "" if attempt.season is None else f" {attempt.season}"
        detail = f": {attempt.detail}" if attempt.detail else ""
        print(f"  {attempt.dataset_id}{season}: {attempt.status!s}{detail}")
    if len(unfilled) > _SUMMARY_LINE_CAP:
        print(f"  ... and {len(unfilled) - _SUMMARY_LINE_CAP} more")


# ── PBP-specific helpers (preserved from original) ────────────────────────────

def _build_enhanced_description(row: dict) -> str:
    parts = []
    try:
        d, ytg = row.get("down"), row.get("ydstogo")
        if d and _str(d):
            parts.append(f"{int(float(d))} & {int(float(ytg))}")
    except (TypeError, ValueError):
        pass

    try:
        qtr = row.get("qtr")
        if qtr and _str(qtr):
            parts.append(f"Q{int(float(qtr))}")
    except (TypeError, ValueError):
        pass

    season  = _str(row.get("season"))
    week    = _str(row.get("week"))
    stype   = _str(row.get("season_type")) or "REG"
    posteam = _str(row.get("posteam"))
    defteam = _str(row.get("defteam"))

    if week:
        try:
            parts.append(f"Week {int(float(week))} {season} ({stype}): {posteam} vs {defteam}")
        except (TypeError, ValueError):
            parts.append(f"{season} ({stype}): {posteam} vs {defteam}")
    else:
        parts.append(f"{season} ({stype}): {posteam} vs {defteam}")

    pt = _str(row.get("play_type"))
    if pt:
        parts.append(pt.upper())

    for label, key in [("QB", "passer_player_name"),
                        ("RB", "rusher_player_name"),
                        ("Receiver", "receiver_player_name")]:
        name = _str(row.get(key))
        if name:
            parts.append(f"{label}: {name}")

    desc = _str(row.get("desc"))
    if desc:
        parts.append(desc)

    tags = []
    try:
        yl = row.get("yardline_100")
        if yl is not None and float(yl) <= 20:
            tags.append("Red Zone")
    except (TypeError, ValueError):
        pass
    try:
        dv = row.get("down")
        if dv is not None:
            d = int(float(dv))
            if d == 3: tags.append("3rd Down")
            if d == 4: tags.append("4th Down")
    except (TypeError, ValueError):
        pass
    if row.get("touchdown") in (1, 1.0, True):
        tags.append("TOUCHDOWN")
    if row.get("interception") in (1, 1.0, True) or \
       row.get("fumble_lost") in (1, 1.0, True):
        tags.append("TURNOVER")
    try:
        yg = float(row.get("yards_gained") or 0)
        if (row.get("pass_attempt") in (1, 1.0) and yg >= 20) or \
           (row.get("rush_attempt") in (1, 1.0) and yg >= 10):
            tags.append("EXPLOSIVE")
    except (TypeError, ValueError):
        pass
    if stype == "POST":
        tags.append("PLAYOFFS")

    if tags:
        parts.append(f"[{', '.join(tags)}]")

    return " — ".join(parts)


def _reconcile_schema(conn: duckdb.DuckDBPyConnection, table: str, df: pl.DataFrame) -> None:
    """Add any new incoming columns to an existing table before insert-by-name."""
    existing_rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    existing_cols = {row[1] for row in existing_rows}

    added = []
    for col_name, dtype in df.schema.items():
        if col_name in existing_cols:
            continue
        duck_type = _duckdb_type_for_polars(dtype)
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col_name}" {duck_type}')
        added.append(f"{col_name} ({duck_type})")

    if added:
        print(f"    Schema drift: added {', '.join(added)}")


# ── PBP ingestion (season-by-season with enhanced_description) ─────────────────

def _create_plays_table(conn: duckdb.DuckDBPyConnection, sample_df: pl.DataFrame):
    """Create plays when it is absent, otherwise leave it exactly as it is.

    This never drops the table. --fresh replaces one requested season at a time in
    _ingest_pbp_season, after that season's fetch succeeds; a whole-table drop here would
    delete every season outside the requested range before any fetch, and a failure after
    the drop would take the rest of the table with it.
    """
    if _table_exists(conn, PBP_TABLE_NAME):
        print("  plays table already exists")
        return
    renamed = _safe_rename(sample_df.head(0))
    _register(conn, "_schema_df", renamed.to_arrow())
    conn.execute(
        f'CREATE TABLE "{PBP_TABLE_NAME}" AS '
        "SELECT *, '' AS enhanced_description FROM _schema_df"
    )
    conn.unregister("_schema_df")
    print(f"  plays table created — {len(sample_df.columns) + 1} columns")


def _ingest_pbp_season(conn: duckdb.DuckDBPyConnection, season: int) -> int:
    """Replace one season of plays; DELETE, INSERT and the metadata row commit together.

    A failed fetch or a failed write is recorded as a failed attempt and returns 0, so one
    season does not stop the run; the previous rows of that season stay.
    """
    import nflreadpy
    print(f"\n  {season}")
    try:
        df = nflreadpy.load_pbp([season])
    except Exception as e:
        print(f"    Failed to load: {e}")
        _record_attempt(conn, PBP_DATASET_ID, PBP_TABLE_NAME, PBP_LOADER_FN, season,
                        IngestStatus.FAILED, str(e))
        return 0

    print(f"    {len(df):,} plays, {len(df.columns)} columns")

    before = len(df)
    if "posteam" in df.columns and "defteam" in df.columns:
        df = df.filter(
            pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null()
        )
    dropped = before - len(df)
    if dropped:
        print(f"    Filtered {dropped:,} rows (null posteam/defteam)")

    df = _safe_rename(df)
    total = len(df)
    if total == 0:
        # Nothing to insert. The DELETE below would empty a season whose _ingest_metadata
        # row still says it is loaded, so leave the stored rows alone.
        print("    No rows left after filtering, existing rows kept")
        _record_attempt(conn, PBP_DATASET_ID, PBP_TABLE_NAME, PBP_LOADER_FN, season,
                        IngestStatus.EMPTY,
                        f"no rows to insert ({before} published, "
                        f"{dropped} dropped for null posteam/defteam)")
        return 0
    descriptions = [
        _build_enhanced_description(row)
        for row in tqdm(df.iter_rows(named=True), total=total, desc=f"    {season} descriptions")
    ]
    df = df.with_columns(pl.Series("enhanced_description", descriptions))

    # Replace, not append: the fetch above succeeded, so it is safe to drop the stale
    # copy of this season (a partial in-progress season, or a --fresh re-run). If the
    # INSERT fails the transaction rolls back and the old season stays.
    try:
        with _transaction(conn):
            _reconcile_schema(conn, PBP_TABLE_NAME, df)
            # PBP_TABLE_NAME is a plain constant ("plays"), so it needs no quoting here.
            conn.execute(f"DELETE FROM {PBP_TABLE_NAME} WHERE season = ?", [season])
            _register(conn, "_ingest_df", df.to_arrow())
            conn.execute(f"INSERT INTO {PBP_TABLE_NAME} BY NAME SELECT * FROM _ingest_df")
            _record_loaded(conn, PBP_DATASET_ID, PBP_TABLE_NAME, PBP_LOADER_FN, total, season)
    except Exception as e:  # noqa: BLE001 - soft failure, reported at the end of the run
        print(f"    Failed to write, kept the previous rows: {e}")
        _record_attempt(conn, PBP_DATASET_ID, PBP_TABLE_NAME, PBP_LOADER_FN, season,
                        IngestStatus.FAILED, str(e))
        return 0
    finally:
        with contextlib.suppress(Exception):
            conn.unregister("_ingest_df")
    print(f"    {total:,} rows inserted")
    return total


# ── Generic dataset ingestion ──────────────────────────────────────────────────

def _write_df_to_table(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    df: pl.DataFrame,
    replace: bool = False,
) -> None:
    """Create or append df into a DuckDB table.

    Existence is asked for, not inferred from a caught CatalogException: that exception
    also means a missing column or a missing function, and the old fallback answered a
    schema problem by trying to CREATE a table that was already there.
    """
    if replace:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')

    _register(conn, "_load_df", df.to_arrow())
    try:
        if _table_exists(conn, table):
            _reconcile_schema(conn, table, df)
            conn.execute(f'INSERT INTO "{table}" BY NAME SELECT * FROM _load_df')
        else:
            conn.execute(f'CREATE TABLE "{table}" AS SELECT * FROM _load_df')
    finally:
        with contextlib.suppress(Exception):
            conn.unregister("_load_df")


def _table_columns(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}


def _stamp_season(df: pl.DataFrame, season: int) -> pl.DataFrame:
    """Add a season column to a frame that has none (the file is the season we asked for)."""
    if "season" in df.columns:
        return df
    return df.with_columns(pl.lit(season, dtype=pl.Int64).alias("season"))


def _legacy_restamp_plan(conn: duckdb.DuckDBPyConnection, dataset_def) -> tuple[bool, list[int]]:
    """Find rows loaded before season stamping. Returns (has_legacy_rows, seasons to re-fetch).

    Old loads left season NULL (depth_charts 2025+ files have none) and nothing in those
    rows says which season they belong to: dt does not split cleanly (the real 2025 file
    runs through 2026-03-14, the 2026 file starts 2026-03-22). The seasons are the ones in
    metadata that have no stamped rows. Adds the season column if the table has none;
    deletes nothing (see _restamp_legacy_seasons).
    """
    table = dataset_def.table_name
    if not _table_exists(conn, table):
        return False, []
    stored, stored_params = _stored_rows_predicate(conn)
    if "season" not in _table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN season BIGINT")
    if not conn.execute(f"SELECT COUNT(*) FROM {table} WHERE season IS NULL").fetchone()[0]:
        return False, []
    seasons = sorted(r[0] for r in conn.execute(
        "SELECT DISTINCT season FROM _ingest_metadata WHERE dataset_id = ? "
        f"{stored} AND season IS NOT NULL AND season NOT IN "
        f"(SELECT DISTINCT season FROM {table} WHERE season IS NOT NULL)",
        [dataset_def.dataset_id, *stored_params],
    ).fetchall())
    return True, seasons


def _restamp_legacy_seasons(conn, dataset_def, loader, seasons: list[int]) -> dict[int, int]:
    """Re-fetch every legacy season, then swap them in. Returns {season: rows}, {} if skipped.

    Every fetch happens before anything is deleted. If one fails (or comes back empty) the
    legacy rows and their metadata stay as they are and the failure is recorded. If all
    succeed, one transaction drops the NULL-season rows and writes each season stamped.
    A second run finds no NULL rows and does nothing.
    """
    print(f"    INFO: {dataset_def.dataset_id}: re-fetching seasons {seasons} to stamp a season "
          f"on rows loaded before season stamping")
    frames: dict[int, pl.DataFrame] = {}
    for season in seasons:
        try:
            df = loader([season], **dataset_def.extra_params)
            if df is None or len(df) == 0:
                raise ValueError("no data returned")
        except Exception as e:  # noqa: BLE001 - soft failure, reported at the end of the run
            print(f"    Failed to re-fetch {dataset_def.dataset_id} {season}: {e}; "
                  f"keeping the existing rows, restamp skipped")
            _note_failure(dataset_def.dataset_id, season, e)
            return {}
        frames[season] = _stamp_season(_safe_rename(df), season)
    try:
        with _transaction(conn):
            for df in frames.values():      # new columns first: DuckDB cannot ALTER after a DELETE
                _reconcile_schema(conn, dataset_def.table_name, df)
            conn.execute(f"DELETE FROM {dataset_def.table_name} WHERE season IS NULL")
            for season, df in frames.items():
                conn.execute(
                    "DELETE FROM _ingest_metadata WHERE dataset_id = ? AND season = ?",
                    [dataset_def.dataset_id, season],
                )
                _replace_season(conn, dataset_def, df, season)
    except Exception as e:  # noqa: BLE001 - soft failure, reported at the end of the run
        print(f"    Failed to restamp {dataset_def.dataset_id}, kept the previous rows: {e}")
        _note_failure(dataset_def.dataset_id, None, e)
        return {}
    for season, df in frames.items():
        print(f"    {dataset_def.dataset_id} {season}: {len(df):,} rows restamped")
    return {season: len(df) for season, df in frames.items()}


def _replace_season(conn, dataset_def, df: pl.DataFrame, season: int) -> None:
    """DELETE the season's old rows, INSERT the new ones, upsert metadata (caller's transaction).

    The old rows are found by the dataset's delete key, never by a swallowed error: if the
    existing table has no such column the season cannot be replaced safely, so this raises.
    """
    table = dataset_def.table_name
    key = dataset_def.delete_key
    if _table_exists(conn, table):
        if key not in _table_columns(conn, table):
            raise ValueError(
                f"table {table} has no '{key}' column, cannot replace season {season}"
            )
        _reconcile_schema(conn, table, df)          # before the DELETE, see _restamp_legacy_seasons
        if dataset_def.delete_match == "id_prefix":
            conn.execute(f'DELETE FROM {table} WHERE starts_with("{key}", ?)', [f"{season}_"])
        else:
            conn.execute(f'DELETE FROM {table} WHERE "{key}" = ?', [season])
    _write_df_to_table(conn, table, df)
    _record_loaded(conn, dataset_def.dataset_id, table, dataset_def.loader_fn, len(df), season)


def _ingest_generic_dataset(
    conn: duckdb.DuckDBPyConnection,
    dataset_def,                        # DatasetDef from registry
    seasons: list[int] | None,          # None = load all via seasons=True
    fresh: bool = False,
) -> int:
    """Load a single dataset (non-pbp) into DuckDB. Returns total rows inserted.

    A season is replaced by the dataset's delete key (registry delete_key/delete_match).
    depth_charts keys by a season column the loader stamps on files that lack one
    (2025+), not by a dt range. Rows from before stamping (NULL season) are dropped and
    their seasons re-fetched in the same run, fetch first (_restamp_legacy_seasons).
    Each season's DELETE, INSERT and metadata upsert run in one transaction. Every fetch
    ends in a recorded attempt: loaded, empty (the source published no rows) or failed.
    """
    import nflreadpy

    loader = getattr(nflreadpy, dataset_def.loader_fn)
    table = dataset_def.table_name
    dataset_id = dataset_def.dataset_id
    loader_fn = dataset_def.loader_fn
    total_rows = 0
    bulk_mode = seasons is None  # pass True to loader instead of per-season list

    if not dataset_def.seasonal:
        # Static dataset — one load, replace table. Static tables are point-in-time
        # snapshots (weekly rankings, player ids, contracts, teams), so they are
        # re-fetched on every ingest; the old table is only dropped after the fetch succeeds.
        print(f"\n  {dataset_id} (static)")
        if _is_loaded(conn, dataset_id):
            print(f"    Already loaded, refreshing")

        try:
            df = loader(**dataset_def.extra_params)
        except Exception as e:
            print(f"    Failed to load: {e}")
            _record_attempt(conn, dataset_id, table, loader_fn, None, IngestStatus.FAILED, str(e))
            return 0

        if df is None or len(df) == 0:
            print(f"    No data returned")
            _record_attempt(conn, dataset_id, table, loader_fn, None, IngestStatus.EMPTY,
                            f"{loader_fn} returned no rows")
            return 0

        df = _safe_rename(df)
        print(f"    {len(df):,} rows, {len(df.columns)} columns")
        try:
            with _transaction(conn):
                _write_df_to_table(conn, table, df, replace=True)
                _record_loaded(conn, dataset_id, table, loader_fn, len(df))
        except Exception as e:  # noqa: BLE001 - soft failure, reported at the end of the run
            print(f"    Failed to write, kept the previous table: {e}")
            _record_attempt(conn, dataset_id, table, loader_fn, None, IngestStatus.FAILED, str(e))
            return 0
        total_rows = len(df)

    else:
        # Seasonal dataset — iterate per season (bulk mode resolves all_seasons() at call
        # time, so a process that runs across the March rollover sees the new season list)
        season_list = _clip_to_window(all_seasons() if bulk_mode else seasons, dataset_def)
        if dataset_def.stamp_season:
            try:
                has_legacy, legacy_seasons = _legacy_restamp_plan(conn, dataset_def)
            except Exception as e:  # noqa: BLE001 - soft failure, reported at the end of the run
                print(f"    Failed to prepare {table}: {e}")
                _note_failure(dataset_id, None, e)
                return 0
            if has_legacy and not legacy_seasons and fresh and season_list:
                # --fresh re-fetches what was asked for, so those seasons can replace the rows.
                legacy_seasons = list(season_list)
            if has_legacy and not legacy_seasons:
                # Nothing says which seasons these rows belong to, so nothing can be
                # re-fetched to replace them; deleting them would only lose data.
                n_null = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE season IS NULL").fetchone()[0]
                print(f"    WARNING: {table} has {n_null:,} rows with no season and no metadata "
                      f"saying which seasons to re-fetch; leaving them in place")
                _note_failure(dataset_id, None, RuntimeError(
                    f"{n_null} rows in {table} have no season and cannot be restamped; "
                    f"run: nfl-mcp ingest --dataset {dataset_id} --fresh"))
            elif has_legacy:
                restamped = _restamp_legacy_seasons(conn, dataset_def, loader, legacy_seasons)
                total_rows += sum(restamped.values())
                # Restamp is all or nothing. If it did not complete, write nothing for
                # these seasons now: a stamped copy would sit next to the NULL-season rows.
                season_list = [s for s in season_list if s not in legacy_seasons]
        for season in season_list:
            already = _is_loaded(conn, dataset_id, season)
            if already and not fresh and not _needs_refresh(conn, dataset_id, season):
                print(f"    {dataset_id} {season}: already loaded, skipping")
                continue
            if already:
                print(f"    {dataset_id} {season}: refreshing")

            print(f"\n  {dataset_id} {season}")
            try:
                df = loader([season], **dataset_def.extra_params)
            except Exception as e:
                print(f"    Failed to load: {e}")
                _record_attempt(conn, dataset_id, table, loader_fn, season,
                                IngestStatus.FAILED, str(e))
                continue

            if df is None or len(df) == 0:
                print(f"    No data returned")
                _record_attempt(conn, dataset_id, table, loader_fn, season,
                                IngestStatus.EMPTY, f"{loader_fn} returned no rows")
                continue

            df = _safe_rename(df)
            if dataset_def.stamp_season:
                df = _stamp_season(df, season)
            print(f"    {len(df):,} rows, {len(df.columns)} columns")

            # Replace, not append: the fetch succeeded, so drop this season's stale rows
            # (no-op on first load) and write the new ones, all or nothing.
            try:
                with _transaction(conn):
                    _replace_season(conn, dataset_def, df, season)
            except Exception as e:  # noqa: BLE001 - soft failure, reported at the end of the run
                print(f"    Failed to write, kept the previous rows: {e}")
                _record_attempt(conn, dataset_id, table, loader_fn, season,
                                IngestStatus.FAILED, str(e))
                continue
            total_rows += len(df)
            print(f"    {len(df):,} rows inserted")

    return total_rows


# ── External records (ESPN, pre-1999) ─────────────────────────────────────────

# Three seasons that failed in a row means the endpoint itself is unhappy, so the run stops
# rather than spending the rest of its timeout budget on a dead network. The count is literal
# about what "in a row" means:
#   a season ESPN was reached for and answered resets it to zero, whether the records landed,
#     came back short (partial), or came back as no records at all (empty);
#   a season that failed increases it by one;
#   a season already loaded, and so skipped without a request, leaves it exactly as it was.
# The last two rules are what let a run that resumes a half-finished backfill still trip here.
ESPN_FAILURE_LIMIT = 3

_ESPN_RECORDS_DDL = """
    CREATE TABLE IF NOT EXISTS espn_team_records (
        season          INTEGER NOT NULL,
        team            VARCHAR NOT NULL,   -- canonical current abbreviation
        espn_team_id    BIGINT,
        espn_abbrev     VARCHAR,            -- abbreviation ESPN used that season
        games           INTEGER,
        wins            INTEGER,
        losses          INTEGER,
        ties            INTEGER,
        record_summary  VARCHAR,            -- ESPN's own "12-4" string
        source_game_id  VARCHAR,            -- event the record was read from
        fetched_at      TIMESTAMP           -- when we pulled it
    )
"""


def _ingest_espn_records(
    conn: duckdb.DuckDBPyConnection,
    seasons: list[int] | None,
    fresh: bool = False,
) -> int:
    """Pull final season records for 1970-1998 from ESPN, one season at a time.

    A season is marked loaded, or partial, only when rows came back. A season that failed, or
    that ESPN has no data for, is recorded as failed, or empty, and is retried on the next run
    instead of being silently frozen. A season whose rows fell short of that year's league is
    recorded as partial: those clubs are absent from ESPN's archive for good, so the record
    should say the season is short rather than read as a complete one.

    Three seasons failing in a row stops the run: that pattern means the endpoint is down
    or rate-limiting us, not that individual seasons are empty, and it bounds how long a
    blocked network can hold up an image build. The remaining seasons stay unrecorded, and
    the fact that the loop broke early is reported to the run (see _note_espn_stop), because
    a pull that stopped and a pull that finished both return a row count.
    """
    from .espn import EspnUnavailable, fetch_season_records
    from .franchises import expected_franchise_count
    from .registry import REGISTRY

    defn = REGISTRY["espn_records"]
    season_list = _clip_to_window(all_seasons() if seasons is None else seasons, defn)
    conn.execute(_ESPN_RECORDS_DDL)
    if not season_list:
        print("\n  espn_records: nothing in range (covers 1970-1998), skipping")
        return 0

    print("\n  espn_records (external source, ~2 requests per season)")
    total = 0
    failures = 0
    for index, season in enumerate(season_list):
        if _is_loaded(conn, defn.dataset_id, season) and not fresh:
            # Already loaded, so no request was made: the failure count does not move.
            print(f"    espn_records {season}: already loaded, skipping")
            continue
        try:
            records = fetch_season_records(season)
        except EspnUnavailable as exc:
            failures += 1
            _record_attempt(conn, defn.dataset_id, defn.table_name, ESPN_LOADER,
                            season, IngestStatus.FAILED, str(exc))
            print(f"    espn_records {season}: ESPN unavailable ({exc}); retrying next run")
            if failures >= ESPN_FAILURE_LIMIT:
                _note_espn_stop(failures, len(season_list) - index - 1)
                print(f"    espn_records: {ESPN_FAILURE_LIMIT} seasons failed in a row, "
                      "stopping — run 'nfl-mcp ingest --pre1999' again later to resume")
                break
            continue
        if not records:
            _record_attempt(conn, defn.dataset_id, defn.table_name, ESPN_LOADER,
                            season, IngestStatus.EMPTY, "ESPN published no records")
            # ESPN answered this season, it simply has no records for it. That is an answer,
            # so the streak of not getting through is over.
            failures = 0
            print(f"    espn_records {season}: no records published")
            continue

        expected = expected_franchise_count(season)
        complete = len(records) == expected
        with _transaction(conn):
            conn.execute("DELETE FROM espn_team_records WHERE season = ?", [season])
            conn.executemany(
                "INSERT INTO espn_team_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (r.season, r.team, r.espn_team_id, r.espn_abbrev, r.games,
                     r.wins, r.losses, r.ties, r.record_summary, r.source_game_id, r.fetched_at)
                    for r in records
                ],
            )
            # A season short of the league that year is partial, and still skips on the next
            # run: those 34 franchise-seasons are absent from ESPN's archive permanently, so
            # a re-fetch would never fill them in.
            _record_loaded(
                conn, defn.dataset_id, defn.table_name, ESPN_LOADER, len(records), season,
                status=IngestStatus.LOADED if complete else IngestStatus.PARTIAL,
                detail=None if complete else f"{len(records)} of {expected} franchises",
            )
        total += len(records)
        # ESPN answered with rows, so whatever failed before this season is in the past.
        failures = 0

        gap = "" if complete else f"  ({expected - len(records)} franchises absent from ESPN's archive)"
        print(f"    espn_records {season}: {len(records)}/{expected} franchises{gap}")
    return total


# ── Aggregate tables (pbp-derived) ────────────────────────────────────────────

_SITUATION_EXPR = """
    CASE
        WHEN down = 4                             THEN '4th Down'
        WHEN down = 3 AND ydstogo >= 7            THEN '3rd & Long'
        WHEN down = 3 AND ydstogo <= 3            THEN '3rd & Short'
        WHEN yardline_100 <= 20                   THEN 'Red Zone'
        WHEN qtr = 4
             AND "time" IS NOT NULL
             AND regexp_matches("time", '^[0-9]+:[0-9]+')
             AND CAST(string_split("time", ':')[1] AS INT) < 2
                                                  THEN 'Two Minute Drill'
        ELSE 'Standard'
    END
""".strip()

_FORMATION_EXPR = """
    CASE
        WHEN shotgun = 1 AND no_huddle = 1 THEN 'SHOTGUN NO HUDDLE'
        WHEN shotgun = 1                   THEN 'SHOTGUN'
        WHEN no_huddle = 1                 THEN 'NO HUDDLE'
        ELSE 'UNDER CENTER'
    END
""".strip()

# What conversion_pct means in each _SITUATION_EXPR bucket, stored on every situational row.
_CONVERSION_RULE_EXPR = """
    CASE situation_label
        WHEN '4th Down'
            THEN 'Share of 4th-down pass/run plays with first_down = 1'
        WHEN '3rd & Long'
            THEN 'Share of 3rd-and-7-or-more pass/run plays with first_down = 1'
        WHEN '3rd & Short'
            THEN 'Share of 3rd-and-3-or-less pass/run plays with first_down = 1'
        WHEN 'Red Zone'
            THEN 'Share of pass/run plays inside the 20 not in the down buckets above that are offensive touchdowns'
        WHEN 'Two Minute Drill'
            THEN 'First-down rate: share of 4th-quarter pass/run plays inside 2:00 with first_down = 1'
        ELSE 'First-down rate: share of pass/run plays not in the buckets above with first_down = 1'
    END
""".strip()

# Flag columns the definitions below read. A plays table without one of them (a partial
# frame in a unit test) is padded with NULL so the SQL still binds; nflverse always has them.
_PLAYS_COLUMNS: dict[str, str] = {
    "season_type": "VARCHAR", "td_team": "VARCHAR", "fumbled_1_team": "VARCHAR",
    **{c: "DOUBLE" for c in (
        "two_point_attempt", "pass_attempt", "rush_attempt", "complete_pass", "sack",
        "interception", "fumble_lost", "qb_dropback", "qb_scramble", "first_down", "touchdown",
        "passing_yards", "rushing_yards", "lateral_rushing_yards")},
}

# Play definitions shared by every derived team table. A two-point try carries
# pass_attempt=1 or rush_attempt=1 (and pass tries carry qb_dropback=1), and some no_play
# rows carry qb_dropback=1 or qb_scramble=1, so each flag-based definition excludes both.
_NOT_2PT = "COALESCE(two_point_attempt, 0) = 0"
_NOT_NO_PLAY = "play_type IS DISTINCT FROM 'no_play'"
_SCRIMMAGE = f"(play_type IN ('pass', 'run') AND {_NOT_2PT})"
# The official play count: runs, kneels, pass attempts, spikes and sacks (carries + attempts
# + sacks in the nflverse team stats). Rows tagged no_play stay out. The analytic metrics (EPA,
# third downs, red zone, explosive plays) keep _SCRIMMAGE, without kneels or spikes.
_TOTAL_PLAY = f"(play_type IN ('pass', 'run', 'qb_kneel', 'qb_spike') AND {_NOT_2PT})"
_PASS_PLAY = f"(play_type = 'pass' AND pass_attempt = 1 AND COALESCE(sack, 0) = 0 AND {_NOT_2PT})"
_PASS_ATTEMPT = f"(pass_attempt = 1 AND COALESCE(sack, 0) = 0 AND {_NOT_2PT} AND {_NOT_NO_PLAY})"
_RUSH_PLAY = f"(rush_attempt = 1 AND {_NOT_2PT} AND {_NOT_NO_PLAY})"
_SACK = f"(sack = 1 AND {_NOT_2PT} AND {_NOT_NO_PLAY})"
_INTERCEPTION = f"(interception = 1 AND {_NOT_2PT} AND {_NOT_NO_PLAY})"
# Yardage sums shared by the offense and defense tables (0, not NULL, when no play matches).
# The total is net rushing plus net passing yards (the NFL definition of total net yards),
# built from these parts, not from SUM(yards_gained).
_RUSH_YARDS = (f"COALESCE(SUM(COALESCE(rushing_yards, 0) + COALESCE(lateral_rushing_yards, 0)) "
               f"FILTER (WHERE {_RUSH_PLAY}), 0)")
_PASS_YARDS = f"COALESCE(SUM(COALESCE(passing_yards, 0)) FILTER (WHERE {_PASS_ATTEMPT}), 0)"
# 0 - SUM keeps a sack for no loss at 0.0 (a unary minus gives -0.0).
_SACK_YARDS = f"COALESCE(0 - SUM(COALESCE(yards_gained, 0)) FILTER (WHERE {_SACK}), 0)"


def _plays_source(conn: duckdb.DuckDBPyConnection) -> str:
    """FROM-clause source for plays, padding any missing flag column with NULL."""
    have = {r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'plays'"
    ).fetchall()}
    padded = [c for c in _PLAYS_COLUMNS if c not in have]
    if padded:
        print(f"    WARNING: plays lacks column(s) {', '.join(padded)}; derived tables use NULL for them")
    pads = [f"CAST(NULL AS {_PLAYS_COLUMNS[c]}) AS {c}" for c in padded]
    return f"(SELECT *, {', '.join(pads)} FROM plays)" if pads else "plays"


def _create_aggregate_tables(conn: duckdb.DuckDBPyConnection):
    """Build the pbp-derived team tables, one row per team, season and season_type (REG/POST)."""
    print("\n  Creating aggregate tables…")
    _ensure_metadata_table(conn)
    src = _plays_source(conn)

    for table in ["team_offense_stats", "team_defense_stats",
                  "situational_stats", "formation_effectiveness"]:
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    # total_plays counts runs, kneels, pass attempts, spikes and sacks (scrambles in; special
    # teams, no_play and two-point tries out). pass_plays excludes spikes; pass_attempts is
    # the official nflverse count (spikes in). A turnover is an interception thrown or a
    # fumble lost by the team, so a muffed punt is charged to the receiving team.
    # pass_yards is gross; sack_yards (positive) is what total_yards subtracts for sacks, so
    # net_pass_yards is the nflverse net figure. rush_yards adds the yards after a lateral
    # and keeps kneels. total_yards is rush_yards + net_pass_yards (the NFL definition of
    # total net yards), and yards_per_play is total_yards / total_plays.
    conn.execute(f"""
        CREATE TABLE team_offense_stats AS
        WITH oagg AS (
            SELECT
                posteam AS team, season AS season_year, season_type,
                COUNT(*) FILTER (WHERE {_TOTAL_PLAY}) AS total_plays,
                COUNT(*) FILTER (WHERE {_RUSH_PLAY}) AS rush_plays,
                COUNT(*) FILTER (WHERE {_PASS_PLAY}) AS pass_plays,
                COUNT(*) FILTER (WHERE {_PASS_ATTEMPT}) AS pass_attempts,
                COUNT(*) FILTER (WHERE complete_pass = 1 AND {_NOT_2PT} AND {_NOT_NO_PLAY}) AS completions,
                COUNT(*) FILTER (WHERE {_INTERCEPTION}) AS interceptions,
                COUNT(*) FILTER (WHERE {_SACK}) AS sacks,
                COUNT(*) FILTER (WHERE qb_dropback = 1 AND {_NOT_2PT} AND {_NOT_NO_PLAY}) AS dropbacks,
                COUNT(*) FILTER (WHERE qb_scramble = 1 AND {_RUSH_PLAY}) AS scrambles,
                {_RUSH_YARDS} AS rush_yards,
                {_PASS_YARDS} AS pass_yards,
                {_SACK_YARDS} AS sack_yards,
                COUNT(*) FILTER (WHERE {_SCRIMMAGE} AND touchdown = 1 AND td_team = posteam) AS touchdowns,
                COUNT(*) FILTER (WHERE {_SCRIMMAGE} AND down = 3 AND first_down = 1) AS third_down_conversions,
                COUNT(*) FILTER (WHERE {_SCRIMMAGE} AND down = 3) AS third_downs,
                COUNT(*) FILTER (WHERE {_SCRIMMAGE} AND yardline_100 <= 20
                                 AND touchdown = 1 AND td_team = posteam) AS red_zone_tds,
                COUNT(*) FILTER (WHERE {_SCRIMMAGE} AND yardline_100 <= 20) AS red_zone_plays,
                COUNT(*) FILTER (WHERE ({_PASS_PLAY} AND yards_gained >= 20)
                                    OR ({_SCRIMMAGE} AND play_type = 'run' AND yards_gained >= 10)) AS explosive_plays,
                ROUND(AVG(epa) FILTER (WHERE {_SCRIMMAGE}), 3) AS avg_epa,
                ROUND(AVG(epa) FILTER (WHERE {_SCRIMMAGE} AND play_type = 'pass'), 3) AS pass_epa,
                ROUND(AVG(epa) FILTER (WHERE {_SCRIMMAGE} AND play_type = 'run'), 3) AS rush_epa
            FROM {src}
            WHERE posteam IS NOT NULL
            GROUP BY posteam, season, season_type
        ),
        fum AS (
            SELECT fumbled_1_team AS team, season AS season_year, season_type, COUNT(*) AS fumbles_lost
            FROM {src}
            WHERE fumble_lost = 1 AND fumbled_1_team IS NOT NULL AND {_NOT_NO_PLAY}
            GROUP BY fumbled_1_team, season, season_type
        )
        SELECT
            oagg.team, oagg.season_year, oagg.season_type,
            oagg.total_plays,
            oagg.rush_yards + oagg.pass_yards - oagg.sack_yards AS total_yards,
            ROUND((oagg.rush_yards + oagg.pass_yards - oagg.sack_yards)
                  / NULLIF(oagg.total_plays, 0), 2) AS yards_per_play,
            oagg.rush_plays, oagg.pass_plays, oagg.pass_attempts, oagg.completions,
            oagg.interceptions, oagg.sacks, oagg.dropbacks, oagg.scrambles,
            oagg.rush_yards, oagg.pass_yards, oagg.sack_yards,
            oagg.pass_yards - oagg.sack_yards AS net_pass_yards,
            ROUND(oagg.rush_yards / NULLIF(oagg.rush_plays, 0), 2) AS yards_per_rush,
            ROUND(oagg.pass_yards / NULLIF(oagg.pass_attempts, 0), 2) AS yards_per_pass,
            oagg.touchdowns,
            oagg.interceptions + COALESCE(fum.fumbles_lost, 0) AS turnovers,
            ROUND(100.0 * oagg.third_down_conversions / NULLIF(oagg.third_downs, 0), 1) AS third_down_pct,
            ROUND(100.0 * oagg.red_zone_tds / NULLIF(oagg.red_zone_plays, 0), 1) AS red_zone_td_pct,
            oagg.explosive_plays, oagg.avg_epa, oagg.pass_epa, oagg.rush_epa
        FROM oagg
        LEFT JOIN fum USING (team, season_year, season_type)
    """)
    print("    team_offense_stats ✓")

    # turnovers_forced counts an interception against the team plus an opponent fumble the
    # team recovered, so the receiving team's own muffed punt is not credited to it. The
    # yardage split is the offense's, seen from the defense: sack_yards is what it gained, and
    # yards_allowed is rush_yards_allowed + net_pass_yards_allowed.
    conn.execute(f"""
        CREATE TABLE team_defense_stats AS
        WITH dagg AS (
            SELECT
                defteam AS team, season AS season_year, season_type,
                COUNT(*) FILTER (WHERE {_TOTAL_PLAY}) AS plays_against,
                {_RUSH_YARDS} AS rush_yards_allowed,
                {_PASS_YARDS} AS pass_yards_allowed,
                COUNT(*) FILTER (WHERE {_SACK}) AS sacks,
                {_SACK_YARDS} AS sack_yards,
                COUNT(*) FILTER (WHERE {_INTERCEPTION}) AS interceptions,
                COUNT(*) FILTER (WHERE {_SCRIMMAGE} AND touchdown = 1 AND td_team = defteam) AS def_touchdowns,
                COUNT(*) FILTER (WHERE {_SCRIMMAGE} AND down = 3 AND COALESCE(first_down, 0) <> 1) AS third_down_stops,
                COUNT(*) FILTER (WHERE {_SCRIMMAGE} AND down = 3) AS third_downs,
                ROUND(AVG(epa) FILTER (WHERE {_SCRIMMAGE}), 3) AS avg_epa_allowed
            FROM {src}
            WHERE defteam IS NOT NULL
            GROUP BY defteam, season, season_type
        ),
        recovered AS (
            SELECT CASE WHEN fumbled_1_team = posteam THEN defteam ELSE posteam END AS team,
                   season AS season_year, season_type, COUNT(*) AS fumbles_recovered
            FROM {src}
            WHERE fumble_lost = 1 AND fumbled_1_team IN (posteam, defteam) AND {_NOT_NO_PLAY}
            GROUP BY 1, season, season_type
        )
        SELECT
            dagg.team, dagg.season_year, dagg.season_type,
            dagg.plays_against,
            dagg.rush_yards_allowed + dagg.pass_yards_allowed - dagg.sack_yards AS yards_allowed,
            ROUND((dagg.rush_yards_allowed + dagg.pass_yards_allowed - dagg.sack_yards)
                  / NULLIF(dagg.plays_against, 0), 2) AS yards_per_play_allowed,
            dagg.rush_yards_allowed, dagg.pass_yards_allowed, dagg.sacks, dagg.sack_yards,
            dagg.pass_yards_allowed - dagg.sack_yards AS net_pass_yards_allowed,
            dagg.interceptions,
            dagg.interceptions + COALESCE(recovered.fumbles_recovered, 0) AS turnovers_forced,
            dagg.def_touchdowns,
            ROUND(100.0 * dagg.third_down_stops / NULLIF(dagg.third_downs, 0), 1) AS third_down_stop_pct,
            dagg.avg_epa_allowed
        FROM dagg
        LEFT JOIN recovered USING (team, season_year, season_type)
    """)
    print("    team_defense_stats ✓")

    conn.execute(f"""
        CREATE TABLE situational_stats AS
        WITH base AS (
            SELECT *, {_SITUATION_EXPR} AS situation_label
            FROM {src}
            WHERE posteam IS NOT NULL AND down IS NOT NULL AND {_SCRIMMAGE}
        ),
        ruled AS (
            SELECT *, {_CONVERSION_RULE_EXPR} AS conversion_rule FROM base
        )
        SELECT
            posteam AS team, season AS season_year, season_type,
            situation_label AS situation,
            COUNT(*) AS plays,
            ROUND(AVG(yards_gained), 2) AS avg_yards,
            COUNT(*) FILTER (WHERE touchdown = 1 AND td_team = posteam) AS touchdowns,
            ROUND(100.0 * COUNT(*) FILTER (WHERE
                    CASE WHEN situation_label = 'Red Zone'
                         THEN touchdown = 1 AND td_team = posteam
                         ELSE first_down = 1 END)
                / NULLIF(COUNT(*), 0), 1) AS conversion_pct,
            conversion_rule,
            ROUND(AVG(epa), 3) AS avg_epa
        FROM ruled
        GROUP BY posteam, season, season_type, situation_label, conversion_rule
    """)
    print("    situational_stats ✓")

    conn.execute(f"""
        CREATE TABLE formation_effectiveness AS
        SELECT
            posteam AS team, season AS season_year, season_type,
            {_FORMATION_EXPR} AS formation,
            play_type,
            COUNT(*) AS plays,
            ROUND(AVG(yards_gained), 2) AS avg_yards,
            COUNT(*) FILTER (WHERE touchdown = 1 AND td_team = posteam) AS touchdowns,
            SUM(CASE WHEN interception=1 OR fumble_lost=1 THEN 1 ELSE 0 END) AS turnovers,
            ROUND(AVG(epa), 3) AS avg_epa
        FROM {src}
        WHERE posteam IS NOT NULL AND play_type IN ('pass','run') AND shotgun IS NOT NULL
          AND {_NOT_2PT}
        GROUP BY posteam, season, season_type, {_FORMATION_EXPR}, play_type
        HAVING COUNT(*) >= 5
    """)
    print("    formation_effectiveness ✓")

    for table in ["team_offense_stats", "team_defense_stats",
                  "situational_stats", "formation_effectiveness"]:
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        _record_loaded(conn, table, table, DERIVED_LOADER, row_count, season=None)

    print("  Aggregate tables done")


def _drop_stale_derived_table(conn: duckdb.DuckDBPyConnection, table_name: str) -> None:
    """Delete a derived table this run cannot rebuild, together with its metadata row.

    A builder skips its table when the source data is missing, and skipping is not enough:
    the copy an earlier run built stays in the database, and its _ingest_metadata row keeps
    reporting it as loaded while the tables it was computed from are gone. Table and record
    go in one transaction, so a failure cannot leave rows with no record or a record with no
    table. Does nothing when there is no such table, which is the ordinary skipped case.
    """
    if not _table_exists(conn, table_name):
        return
    with _transaction(conn):
        conn.execute(f"DROP TABLE {table_name}")
        _delete_metadata_row(conn, table_name, None)
    print(f"    {table_name} — dropped (its source data is gone)")


# Each fantasy derived table: (table_name, [required source tables], CREATE-AS SELECT body).
# All are built at ingest time so the MCP tools never compute at query time.
_FANTASY_TABLES: list[tuple[str, list[str], str]] = [
    # 1. TD luck — actual vs expected touchdowns per player-season and season_type
    #    (most "unlucky" first). {ff_typed} is ff_opportunity plus a REG/POST season_type.
    ("player_td_luck", ["ff_opportunity"], """
        SELECT
            player_id,
            ANY_VALUE(full_name)            AS full_name,
            ANY_VALUE(position)             AS position,
            ANY_VALUE(posteam)              AS team,
            CAST(season AS INTEGER)         AS season,
            season_type,
            SUM(rec_touchdown)              AS rec_td,
            SUM(rec_touchdown_exp)          AS rec_td_exp,
            SUM(rec_touchdown_diff)         AS rec_td_luck,
            SUM(rush_touchdown)             AS rush_td,
            SUM(rush_touchdown_exp)         AS rush_td_exp,
            SUM(rush_touchdown_diff)        AS rush_td_luck,
            ROUND(SUM(rec_touchdown_diff) + SUM(rush_touchdown_diff), 3) AS total_td_luck_score
        FROM {ff_typed}
        WHERE player_id IS NOT NULL
        GROUP BY player_id, CAST(season AS INTEGER), season_type
    """),

    # 2. Rolling 3-week role trend — snap / target / carry / air-yards share with
    #    a trailing 3-week average and the current-week delta vs that average. The window
    #    runs inside one season_type, so a playoff average never reaches into REG weeks.
    #    Snap counts join through players.gsis_id -> players.pfr_id -> snap_counts.pfr_player_id.
    ("player_role_trend", ["ff_opportunity", "snap_counts", "players"], """
        WITH base AS (
            SELECT
                o.player_id,
                o.full_name,
                o.position,
                o.posteam                                   AS team,
                CAST(o.season AS INTEGER)                   AS season,
                o.season_type,
                CAST(o.week AS INTEGER)                      AS week,
                ROUND(100.0 * s.offense_pct, 2)             AS snap_pct,
                ROUND(100.0 * o.rec_attempt  / NULLIF(o.rec_attempt_team, 0), 2)   AS target_share_pct,
                ROUND(100.0 * o.rush_attempt / NULLIF(o.rush_attempt_team, 0), 2)  AS carry_share_pct,
                ROUND(100.0 * o.rec_air_yards / NULLIF(o.rec_air_yards_team, 0), 2) AS air_yards_share_pct
            FROM {ff_typed} o
            LEFT JOIN (
                SELECT gsis_id, MAX(pfr_id) AS pfr_id
                FROM players
                WHERE gsis_id IS NOT NULL AND pfr_id IS NOT NULL
                GROUP BY gsis_id
            ) pl ON pl.gsis_id = o.player_id
            LEFT JOIN snap_counts s
              ON s.pfr_player_id = pl.pfr_id
             AND s.season  = CAST(o.season AS INTEGER)
             AND s.week    = CAST(o.week AS INTEGER)
            WHERE o.player_id IS NOT NULL
        )
        SELECT
            player_id, full_name, position, team, season, season_type, week,
            snap_pct, target_share_pct, carry_share_pct, air_yards_share_pct,
            ROUND(AVG(snap_pct)            OVER w, 2) AS snap_pct_3wk,
            ROUND(AVG(target_share_pct)    OVER w, 2) AS target_share_pct_3wk,
            ROUND(AVG(carry_share_pct)     OVER w, 2) AS carry_share_pct_3wk,
            ROUND(AVG(air_yards_share_pct) OVER w, 2) AS air_yards_share_pct_3wk,
            ROUND(snap_pct            - AVG(snap_pct)            OVER w, 2) AS snap_pct_delta,
            ROUND(target_share_pct    - AVG(target_share_pct)    OVER w, 2) AS target_share_pct_delta,
            ROUND(carry_share_pct     - AVG(carry_share_pct)     OVER w, 2) AS carry_share_pct_delta,
            ROUND(air_yards_share_pct - AVG(air_yards_share_pct) OVER w, 2) AS air_yards_share_pct_delta
        FROM base
        WINDOW w AS (PARTITION BY player_id, season, season_type ORDER BY week ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
    """),

    # 3. Separation-adjusted opportunity — joins Next Gen Stats receiving (2016+) to
    #    fantasy opportunity to flag separation-creators who underproduced (regression up).
    #    One row per player, season and season_type; Next Gen Stats rows join on the same type.
    ("player_separation_opportunity", ["ff_opportunity", "nextgen_stats_receiving"], """
        WITH ng AS (
            SELECT
                player_gsis_id,
                season,
                season_type,
                AVG(avg_separation)              AS avg_separation,
                AVG(avg_yac_above_expectation)   AS avg_yac_above_expectation,
                AVG(catch_percentage)            AS catch_percentage
            FROM nextgen_stats_receiving
            WHERE week > 0 AND player_gsis_id IS NOT NULL
            GROUP BY player_gsis_id, season, season_type
        ),
        opp AS (
            SELECT
                player_id,
                ANY_VALUE(full_name)   AS full_name,
                ANY_VALUE(position)    AS position,
                ANY_VALUE(posteam)     AS team,
                CAST(season AS INTEGER) AS season,
                season_type,
                COUNT(DISTINCT week)   AS games,
                SUM(total_fantasy_points_diff) AS fp_diff,
                ROUND(SUM(total_fantasy_points_diff) / NULLIF(COUNT(DISTINCT week), 0), 3) AS fp_diff_per_game,
                ROUND(SUM(rec_touchdown_diff), 3) AS td_luck,
                ROUND(SUM(receptions_diff), 3)    AS catch_luck,
                ROUND(100.0 * SUM(rec_attempt)  / NULLIF(SUM(rec_attempt_team), 0), 2)  AS target_share_pct,
                ROUND(100.0 * SUM(rec_air_yards) / NULLIF(SUM(rec_air_yards_team), 0), 2) AS air_yards_share_pct
            FROM {ff_typed}
            WHERE player_id IS NOT NULL
            GROUP BY player_id, CAST(season AS INTEGER), season_type
        )
        SELECT
            opp.player_id, opp.full_name, opp.position, opp.team, opp.season, opp.season_type,
            opp.games, opp.fp_diff, opp.fp_diff_per_game, opp.td_luck, opp.catch_luck,
            opp.target_share_pct, opp.air_yards_share_pct,
            ROUND(ng.avg_separation, 3)            AS avg_separation,
            ROUND(ng.avg_yac_above_expectation, 3) AS avg_yac_above_expectation,
            ROUND(ng.catch_percentage, 2)          AS catch_percentage,
            (ng.avg_separation > 2.5 AND opp.fp_diff_per_game < -1.5 AND opp.td_luck < -1.0) AS regression_candidate
        FROM opp
        JOIN ng ON ng.player_gsis_id = opp.player_id AND ng.season = opp.season
                      AND ng.season_type = opp.season_type
    """),

    # 4. Drop rate — catchable-target drop rate from FTN charting (2022+) joined to plays.
    ("player_drop_rate", ["ftn_charting", "plays"], """
        SELECT
            p.receiver_player_id                AS player_id,
            ANY_VALUE(p.receiver_player_name)   AS player,
            p.posteam                           AS team,
            p.season                            AS season,
            p.season_type                       AS season_type,
            COUNT(*)                            AS catchable_targets,
            SUM(CASE WHEN f.is_drop THEN 1 ELSE 0 END)              AS drops,
            ROUND(100.0 * SUM(CASE WHEN f.is_drop THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0), 2)       AS drop_rate_pct,
            SUM(CASE WHEN f.is_contested_ball THEN 1 ELSE 0 END)    AS contested_targets,
            SUM(CASE WHEN f.is_created_reception THEN 1 ELSE 0 END) AS created_receptions
        FROM ftn_charting f
        JOIN plays p
          ON f.nflverse_game_id = p.game_id
         AND f.nflverse_play_id = p.play_id
        WHERE f.is_catchable_ball = TRUE
          AND p.receiver_player_id IS NOT NULL
        GROUP BY p.receiver_player_id, p.posteam, p.season, p.season_type
    """),

    # 5. Contract value efficiency — fantasy points per $M of average per year (APY),
    #    using the contract in force that season: the latest year_signed <= season,
    #    ties broken by the larger APY. One row per player, season and season_type.
    ("player_contract_value", ["contracts", "ff_opportunity"], """
        WITH opp AS (
            SELECT
                player_id,
                ANY_VALUE(full_name)   AS full_name,
                ANY_VALUE(position)    AS position,
                ANY_VALUE(posteam)     AS team,
                CAST(season AS INTEGER) AS season,
                season_type,
                SUM(total_fantasy_points)     AS total_fp,
                SUM(total_fantasy_points_exp) AS total_fp_exp
            FROM {ff_typed}
            WHERE player_id IS NOT NULL
            GROUP BY player_id, CAST(season AS INTEGER), season_type
        ),
        ranked AS (
            SELECT
                opp.*,
                c.position     AS contract_position,
                c.apy          AS apy,
                c.apy_cap_pct  AS cap_pct,
                ROW_NUMBER() OVER (
                    PARTITION BY opp.player_id, opp.season, opp.season_type
                    ORDER BY c.year_signed DESC, c.apy DESC NULLS LAST, c.apy_cap_pct DESC NULLS LAST
                ) AS rn
            FROM opp
            JOIN contracts c
              ON c.gsis_id = opp.player_id
             AND c.year_signed <= opp.season
        )
        SELECT
            player_id,
            full_name,
            COALESCE(position, contract_position) AS position,
            team,
            season,
            season_type,
            ROUND(apy, 4)       AS apy,
            ROUND(cap_pct, 4)   AS cap_pct,
            ROUND(total_fp, 2)     AS total_fp,
            ROUND(total_fp_exp, 2) AS total_fp_exp,
            ROUND(total_fp / NULLIF(apy, 0), 2) AS fp_per_million
        FROM ranked
        WHERE rn = 1
    """),

    # 6. Injury return curve — post-return snap-share recovery (as % of pre-injury
    #    baseline) at +1..+8 weeks, bucketed by normalized injury type and position.
    #    Snap counts join through players.gsis_id -> players.pfr_id -> snap_counts.pfr_player_id.
    ("injury_return_curve", ["injuries", "snap_counts", "players"], """
        WITH outs AS (
            SELECT
                gsis_id,
                full_name,
                position,
                CAST(season AS INTEGER) AS season,
                CAST(week AS INTEGER)   AS week,
                LOWER(TRIM(COALESCE(report_primary_injury, practice_primary_injury))) AS injury_raw
            FROM injuries
            WHERE report_status = 'Out'
              AND gsis_id IS NOT NULL
              AND COALESCE(report_primary_injury, practice_primary_injury) IS NOT NULL
        ),
        typed AS (
            SELECT *,
                CASE
                    WHEN injury_raw LIKE '%hamstring%'  THEN 'hamstring'
                    WHEN injury_raw LIKE '%knee%'       THEN 'knee'
                    WHEN injury_raw LIKE '%ankle%'      THEN 'ankle'
                    WHEN injury_raw LIKE '%shoulder%'   THEN 'shoulder'
                    WHEN injury_raw LIKE '%concussion%' THEN 'concussion'
                    WHEN injury_raw LIKE '%groin%'      THEN 'groin'
                    WHEN injury_raw LIKE '%foot%'       THEN 'foot'
                    WHEN injury_raw LIKE '%calf%'       THEN 'calf'
                    WHEN injury_raw LIKE '%hip%'        THEN 'hip'
                    WHEN injury_raw LIKE '%back%'       THEN 'back'
                    WHEN injury_raw LIKE '%quad%'       THEN 'quadriceps'
                    WHEN injury_raw LIKE '%achilles%'   THEN 'achilles'
                    WHEN injury_raw LIKE '%wrist%'      THEN 'wrist'
                    WHEN injury_raw LIKE '%hand%'       THEN 'hand'
                    WHEN injury_raw LIKE '%elbow%'      THEN 'elbow'
                    WHEN injury_raw LIKE '%toe%'        THEN 'toe'
                    WHEN injury_raw LIKE '%thigh%'      THEN 'thigh'
                    WHEN injury_raw LIKE '%neck%'       THEN 'neck'
                    WHEN injury_raw LIKE '%rib%'        THEN 'ribs'
                    WHEN injury_raw LIKE '%pectoral%'   THEN 'pectoral'
                    ELSE 'other'
                END AS injury_type
            FROM outs
        ),
        islands AS (
            SELECT *,
                week - ROW_NUMBER() OVER (PARTITION BY gsis_id, season ORDER BY week) AS island
            FROM typed
        ),
        spells AS (
            SELECT
                gsis_id,
                ANY_VALUE(full_name) AS full_name,
                ANY_VALUE(position)  AS position,
                season,
                MODE(injury_type)    AS injury_type,
                MIN(week)            AS first_out_week,
                MAX(week)            AS last_out_week
            FROM islands
            GROUP BY gsis_id, season, island
        ),
        baseline AS (
            SELECT
                sp.gsis_id, pl.pfr_id, sp.position, sp.season,
                sp.injury_type, sp.first_out_week, sp.last_out_week,
                AVG(sn.offense_pct) AS baseline_pct
            FROM spells sp
            JOIN (
                SELECT gsis_id, MAX(pfr_id) AS pfr_id
                FROM players
                WHERE gsis_id IS NOT NULL AND pfr_id IS NOT NULL
                GROUP BY gsis_id
            ) pl ON pl.gsis_id = sp.gsis_id
            JOIN snap_counts sn
              ON sn.pfr_player_id = pl.pfr_id
             AND sn.season = sp.season
             AND sn.week   < sp.first_out_week
            GROUP BY sp.gsis_id, pl.pfr_id, sp.position, sp.season,
                     sp.injury_type, sp.first_out_week, sp.last_out_week
        ),
        curve AS (
            SELECT
                b.injury_type,
                b.position,
                (sn.week - b.last_out_week) AS week_post_return,
                100.0 * sn.offense_pct / NULLIF(b.baseline_pct, 0) AS recovery_pct
            FROM baseline b
            JOIN snap_counts sn
              ON sn.pfr_player_id = b.pfr_id
             AND sn.season = b.season
             AND sn.week   > b.last_out_week
             AND sn.week  <= b.last_out_week + 8
            WHERE b.baseline_pct > 0
        )
        SELECT
            injury_type,
            position,
            week_post_return,
            ROUND(AVG(recovery_pct), 1)    AS avg_snap_pct_recovery,
            ROUND(MEDIAN(recovery_pct), 1) AS median_snap_pct_recovery,
            COUNT(*)                       AS sample_size
        FROM curve
        GROUP BY injury_type, position, week_post_return
    """),
]


def _ff_opportunity_typed(conn: duckdb.DuckDBPyConnection) -> str:
    """ff_opportunity with a season_type (REG/POST) column, which the table does not carry.

    POST comes from schedules.game_type through game_id (anything but REG); without a
    schedules row the week decides (week > 18).
    """
    by_week = "CASE WHEN o.week > 18 THEN 'POST' ELSE 'REG' END"
    if not _table_exists(conn, "schedules"):
        return f"(SELECT o.*, {by_week} AS season_type FROM ff_opportunity o)"
    return f"""(
        SELECT o.*,
               CASE WHEN s.game_type IS NULL THEN {by_week}
                    WHEN s.game_type = 'REG' THEN 'REG' ELSE 'POST' END AS season_type
        FROM ff_opportunity o
        LEFT JOIN (SELECT game_id, ANY_VALUE(game_type) AS game_type FROM schedules GROUP BY game_id) s
          ON s.game_id = o.game_id
    )"""


def _create_fantasy_tables(conn: duckdb.DuckDBPyConnection):
    """Build the fantasy-analytics derived tables from already-ingested source data.

    Each table is fully recomputed (DROP + CREATE TABLE AS) and recorded in
    _ingest_metadata as a 'derived' dataset so it surfaces in nfl_catalog / nfl_status.
    Tables whose source data isn't present yet are skipped gracefully, and a copy left by an
    earlier run is dropped: it is built from data that is no longer here.
    """
    print("\n  Creating fantasy derived tables…")
    ff_typed = _ff_opportunity_typed(conn)
    for table_name, sources, body in _FANTASY_TABLES:
        missing = [s for s in sources if not _table_exists(conn, s)]
        if missing:
            print(f"    {table_name} — skipped (missing sources: {', '.join(missing)})")
            _drop_stale_derived_table(conn, table_name)
            continue
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"CREATE TABLE {table_name} AS {body.replace('{ff_typed}', ff_typed)}")
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        _record_loaded(conn, table_name, table_name, DERIVED_LOADER, row_count, season=None)
        print(f"    {table_name} ✓ ({row_count:,} rows)")
    print("  Fantasy derived tables done")


def _create_team_records(conn: duckdb.DuckDBPyConnection):
    """Build team_records: one row per franchise per season with its final regular-season mark.

    Two sources, unioned, because no single publisher spans the whole era: nflverse games
    from 1999 (aggregated from scores) and ESPN's reported records for 1970-1998 (see
    nfl_mcp/espn.py for why scores cannot be used there). Team codes are normalized to the
    canonical current abbreviation so one query spans relocations and renames.

    points_for, points_against and point_diff (schema version 3) are regular-season sums
    over the games with a score, so a season in progress has its points to date. ESPN
    rows carry no scores: their points are NULL.
    """
    from .franchises import schedule_alias_case

    statements: list[str] = []
    if _table_exists(conn, "espn_team_records"):
        statements.append("""
            SELECT CAST(season AS INTEGER) AS season, team, espn_abbrev AS source_team,
                   games, wins, losses, ties,
                   CAST(NULL AS INTEGER) AS points_for, CAST(NULL AS INTEGER) AS points_against,
                   CAST(NULL AS INTEGER) AS point_diff,
                   'espn' AS source
            FROM espn_team_records
        """)
    if _table_exists(conn, "schedules"):
        statements.append(f"""
            WITH games AS (
                SELECT season, home_team AS source_team, away_team AS opponent,
                       home_score AS points_for, away_score AS points_against
                FROM schedules
                WHERE game_type = 'REG' AND home_team IS NOT NULL
                  AND home_score IS NOT NULL AND away_score IS NOT NULL
                UNION ALL
                SELECT season, away_team, home_team, away_score, home_score
                FROM schedules
                WHERE game_type = 'REG' AND away_team IS NOT NULL
                  AND home_score IS NOT NULL AND away_score IS NOT NULL
            )
            SELECT CAST(season AS INTEGER) AS season,
                   {schedule_alias_case("source_team")} AS team, source_team,
                   COUNT(*)                                    AS games,
                   SUM(CASE WHEN points_for > points_against THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN points_for < points_against THEN 1 ELSE 0 END) AS losses,
                   SUM(CASE WHEN points_for = points_against THEN 1 ELSE 0 END) AS ties,
                   CAST(SUM(points_for) AS INTEGER)            AS points_for,
                   CAST(SUM(points_against) AS INTEGER)        AS points_against,
                   CAST(SUM(points_for - points_against) AS INTEGER) AS point_diff,
                   'nflverse' AS source
            FROM games
            GROUP BY 1, 2, 3
        """)
    if not statements:
        # team_records has two sources, and neither is here. Anything built by an earlier run
        # was built from data this database no longer has.
        print("    team_records — skipped (needs schedules or espn_team_records)")
        _drop_stale_derived_table(conn, "team_records")
        return

    conn.execute("DROP TABLE IF EXISTS team_records")
    conn.execute(f"CREATE TABLE team_records AS {statements[0]}")
    for extra in statements[1:]:
        conn.execute(f"INSERT INTO team_records {extra}")
    row_count = conn.execute("SELECT COUNT(*) FROM team_records").fetchone()[0]
    _record_loaded(conn, "team_records", "team_records", DERIVED_LOADER, row_count, season=None)
    print(f"    team_records ✓ ({row_count:,} rows)")




def rebuild_derived_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Rebuild every derived table from the loaded source data in one transaction.

    On any failure the transaction is rolled back, so the previous tables stay intact,
    and the error is re-raised. All SQL goes through conn.execute. The caller has already
    given the metadata table its later columns (run_ingest_datasets, migrate.upgrade).
    """
    with _transaction(conn, "Rollback after the failed rebuild also failed"):
        _create_aggregate_tables(conn)
        _create_fantasy_tables(conn)
        _create_team_records(conn)


def _create_indexes(conn: duckdb.DuckDBPyConnection):
    print("\n  Creating indexes…")
    for idx_sql in [
        'CREATE INDEX IF NOT EXISTS idx_plays_season      ON plays(season)',
        'CREATE INDEX IF NOT EXISTS idx_plays_posteam     ON plays(posteam)',
        'CREATE INDEX IF NOT EXISTS idx_plays_defteam     ON plays(defteam)',
        'CREATE INDEX IF NOT EXISTS idx_plays_team_season ON plays(posteam, season)',
        'CREATE INDEX IF NOT EXISTS idx_plays_game_id     ON plays(game_id)',
        'CREATE INDEX IF NOT EXISTS idx_plays_play_type   ON plays(play_type)',
    ]:
        # Deliberately broad: an index is an optimization, not data. plays gains columns over
        # time, so an index on one a given database never loaded fails here while the run
        # still has its rows. The failure is reported and the next index is tried.
        try:
            conn.execute(idx_sql)
        except Exception as e:
            print(f"    Warning: {e}")
    print("  Indexes done")


# ── Public entry points ────────────────────────────────────────────────────────

def run_ingest_datasets(
    dataset_ids: list[str],
    start: int | None = None,
    end: int | None = None,
    fresh: bool = False,
    skip_views: bool = False,
    db_path: str | None = None,
) -> IngestRunResult:
    """
    Ingest one or more datasets into DuckDB, and report what this run did.

    dataset_ids: list of keys from REGISTRY (e.g. ["pbp", "schedules"])
    start/end:   season range; if both are None, loads all available seasons.
    fresh:       re-fetch the requested seasons; rows of other seasons are kept.

    A failed fetch or season write does not stop the other datasets. The IngestRunResult
    describes the fetches this call made: each one's outcome, how many ended each way, the
    failures ("<dataset> <season>: <error>", empty on a clean run), and whether the ESPN pull
    gave up early. A season this run skipped because it was already loaded is not an
    attempt, so a run that only skipped seasons reports no attempts at all.

    The run owns the collector its recording helpers report into, and puts back whichever
    collector was active before, so one run cannot leak attempts into the next.
    """
    with _collecting() as collector:
        return _ingest_datasets(
            collector, dataset_ids=dataset_ids, start=start, end=end,
            fresh=fresh, skip_views=skip_views, db_path=db_path,
        )


def _ingest_datasets(
    collector: _RunCollector,
    dataset_ids: list[str],
    start: int | None,
    end: int | None,
    fresh: bool,
    skip_views: bool,
    db_path: str | None,
) -> IngestRunResult:
    """Do the work of one run. run_ingest_datasets owns the collector and the context."""
    import nflreadpy
    from .config import get_duckdb_path
    from .registry import REGISTRY

    bulk_mode = start is None and end is None
    if not bulk_mode:
        start = start or FIRST_SEASON
        end = end or current_season()
        if start > end:
            raise ValueError("start must be less than or equal to end")

    path = db_path or str(get_duckdb_path())
    # Build the widest range the caller asked for; each dataset is then clipped to its
    # own min_season/max_season, so asking for 1966 does not make load_pbp try 1966.
    seasons: list[int] | None = None if bulk_mode else [
        s for s in range(EARLIEST_SEASON, current_season() + 1) if start <= s <= end
    ]

    print("=" * 60)
    print("NFL MCP — Multi-Dataset Ingest")
    print("=" * 60)
    print(f"Datasets : {', '.join(dataset_ids)}")
    print(f"Seasons  : {'all available' if bulk_mode else f'{seasons[0]}–{seasons[-1]}'}")
    print(f"Database : {path}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(path)
    # The write connection stays open for the rest of the run, so every way out of the run has
    # to close it. DuckDB allows one writer per file: a connection left behind by a raised
    # exception keeps the server and the next scheduled ingest locked out until this process
    # exits.
    try:
        _apply_duckdb_pragmas(conn, path)

        _ensure_metadata_table(conn)

        pbp_ingested = 0
        has_pbp = PBP_DATASET_ID in dataset_ids

        # ── PBP (special path — enhanced_description, null filtering) ──────────
        if has_pbp:
            print("\n── Play-by-play ──")

            # Play-by-play is on a special code path, so it must clip to its own window
            # explicitly: the default backfill starts in 1966, but load_pbp only publishes
            # from 1999 and raises for anything earlier, or past get_current_season().
            pbp_seasons = _clip_to_window(
                all_seasons() if bulk_mode else seasons, REGISTRY[PBP_DATASET_ID]
            )
            if not pbp_seasons:
                # Only an explicit range can get here (--start in the off-season): nothing to do.
                print("  No published play-by-play season in range, skipping")
                has_pbp = False

        if has_pbp:
            # Discover the schema from the newest season that actually has a pbp file.
            # On kickoff day the new season can pass the validator before its file exists.
            sample_df = None
            for schema_season in reversed(pbp_seasons):
                print(f"  Discovering schema from {schema_season}…")
                try:
                    sample_df = nflreadpy.load_pbp([schema_season])
                    break
                except Exception as e:
                    print(f"    Not available ({e}); trying an earlier season")
            if sample_df is None:
                raise RuntimeError("Could not load play-by-play for any requested season")
            print(f"  {len(sample_df.columns)} columns · {len(sample_df):,} rows")

            # --fresh re-fetches the requested seasons below (each replaces its own rows); it
            # must not drop the table, which would delete seasons outside --start/--end.
            _create_plays_table(conn, sample_df)

            loaded_pbp = {
                row[0] for row in conn.execute(
                    f'SELECT DISTINCT season FROM "{PBP_TABLE_NAME}" '
                    "WHERE season IS NOT NULL"
                ).fetchall()
            } if not fresh else set()

            # A season already in plays is a season already fetched, so today's decision reads
            # the table rather than the metadata. On top of that, a season whose last fetch
            # happened before the season ended is fetched once more: run in January, that
            # season holds no playoff rows. See _needs_refresh.
            refetch = [
                s for s in pbp_seasons
                if s in loaded_pbp and _needs_refresh(conn, PBP_DATASET_ID, s)
            ]
            to_ingest   = [s for s in pbp_seasons if s not in loaded_pbp or s in refetch]
            skipped     = [s for s in pbp_seasons if s in loaded_pbp and s not in refetch]
            in_progress = _refresh_seasons()
            refreshed   = [s for s in refetch if s in in_progress]
            unfinished  = [s for s in refetch if s not in in_progress]

            if skipped:
                print(f"  Already loaded, skipping: {skipped}")
            if refreshed:
                print(f"  Current season, re-fetching: {refreshed}")
            if unfinished:
                print(f"  Last loaded before the season ended, re-fetching: {unfinished}")
            if not to_ingest:
                print("  All PBP seasons already loaded.")
            else:
                print(f"  Will ingest: {to_ingest}")
                for season in to_ingest:
                    # The season's delete, insert and metadata row commit together inside
                    # _ingest_pbp_season, so a failing season keeps its previous copy.
                    pbp_ingested += _ingest_pbp_season(conn, season)

            _create_indexes(conn)

            if not skip_views:
                _create_aggregate_tables(conn)

        # ── All other datasets ─────────────────────────────────────────────────
        other_ids = [d for d in dataset_ids if d != PBP_DATASET_ID]
        if other_ids:
            print("\n── Additional datasets ──")
            for dataset_id in other_ids:
                defn = REGISTRY.get(dataset_id)
                if defn is None:
                    print(f"\n  WARNING: unknown dataset '{dataset_id}', skipping")
                    continue
                # The source picks the ingest helper. A source with no helper is a mistake
                # in the registry, not something to guess at: sending it to nflreadpy would
                # call a loader function named after a third party and record a failure
                # against the wrong publisher.
                if defn.source == SOURCE_ESPN:
                    _ingest_espn_records(conn, seasons, fresh=fresh)
                elif defn.source == SOURCE_NFLREADPY:
                    try:
                        _ingest_generic_dataset(conn, defn, seasons, fresh=fresh)
                    except Exception as e:  # noqa: BLE001 - soft failure, reported at the end of the run
                        print(f"    Failed: {e}")
                        _note_failure(dataset_id, None, e)
                else:
                    raise ValueError(
                        f"Dataset '{dataset_id}' has source {defn.source!r}, which has no "
                        f"ingest helper (known sources: {', '.join(KNOWN_SOURCES)})"
                    )

        # ── Fantasy derived tables (built from already-ingested sources) ────────
        if not skip_views:
            _create_fantasy_tables(conn)
            _create_team_records(conn)

        result = collector.result()
        failures = result.failures
        if not failures:
            from .migrate import stamp_if_current
            if stamp_if_current(conn):
                print("\n  Schema version stamped.")
    finally:
        conn.close()

    print("\n" + "=" * 60)
    if failures:
        print(f"Ingestion finished with {len(failures)} failure(s):")
        for failure in failures:
            print(f"  FAILED {failure}")
    else:
        print("Ingestion complete!")
    if has_pbp and pbp_ingested:
        print(f"  PBP plays ingested this run: {pbp_ingested:,}")
    season_label = "all available" if bulk_mode else f"{seasons[0]}–{seasons[-1]}"
    print(f"  Seasons: {season_label}")
    print(f"  DB: {path}")
    print("=" * 60)
    _print_run_summary(result)
    return result


def run_ingest(
    start: int = FIRST_SEASON,
    end: int | None = None,
    fresh: bool = False,
    skip_views: bool = False,
    db_path: str | None = None,
) -> IngestRunResult:
    """Backward-compatible entry point — ingests PBP only. Returns the run result."""
    return run_ingest_datasets(
        dataset_ids=[PBP_DATASET_ID],
        start=start,
        end=end,
        fresh=fresh,
        skip_views=skip_views,
        db_path=db_path,
    )
