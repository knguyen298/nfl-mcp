"""
NFL data ingestion into DuckDB.

Supports all nflreadpy datasets via a declarative registry.
PBP ingestion preserves its existing enhanced_description logic.

Called by: nfl-mcp init / nfl-mcp ingest
"""

import datetime
import os
import weakref
import duckdb
import polars as pl
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
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
    SOURCE_ESPN,
    SOURCE_NFLREADPY,
)
from .status import IngestStatus


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
    """
    mem = os.getenv("NFL_MCP_DUCKDB_MEMORY_LIMIT")
    if mem:
        conn.execute(f"SET memory_limit='{mem}'")
        spill = str(Path(db_path).parent / ".duckdb_spill")
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


def _clip_to_window(season_list, dataset_def):
    """Drop seasons outside a dataset's own published coverage window (no network)."""
    return [
        s for s in season_list
        if (dataset_def.min_season is None or s >= dataset_def.min_season)
        and (dataset_def.max_season is None or s <= dataset_def.max_season)
    ]


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


def _clip_detail(detail: str | None) -> str | None:
    """Trim a free-form reason to what belongs in the column; NULL stays NULL (no reason)."""
    if detail is None:
        return None
    return detail[:_DETAIL_MAX_CHARS]


def _ensure_metadata_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create _ingest_metadata if absent, and add the outcome columns if an old table lacks them.

    status holds an IngestStatus value; detail holds a short human-readable reason for the
    outcomes that loaded no data. A database created before those columns existed gets them
    added here, and the DEFAULT backfills its rows to 'loaded' — the old code only ever
    wrote a row when rows actually landed. Running this on a table that already has both
    columns changes nothing.
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
            detail      VARCHAR           -- short reason, NULL when there is nothing to say
        )
    """)
    # CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so it cannot upgrade one.
    # IF NOT EXISTS on the ALTERs keeps this whole function safe to call twice, and gives a
    # row written by the old code its 'loaded' status.
    conn.execute(
        f"ALTER TABLE _ingest_metadata "
        f"ADD COLUMN IF NOT EXISTS status VARCHAR DEFAULT '{_LOADED_DEFAULT}'"
    )
    conn.execute(
        "ALTER TABLE _ingest_metadata ADD COLUMN IF NOT EXISTS detail VARCHAR"
    )


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


@contextmanager
def _transaction(conn: duckdb.DuckDBPyConnection):
    """Write rows and their metadata as one unit: all of it commits, or none of it does.

    Without this, each conn.execute commits by itself, so a crash or a failing INSERT
    between the DELETE and the INSERT leaves a season with no rows while
    _ingest_metadata still says it is loaded — every later run then skips it. DuckDB
    rolls back DDL too, so a DROP + CREATE that fails partway leaves the old table.
    """
    if conn in _open_transactions:
        yield
        return
    _open_transactions.add(conn)
    conn.execute("BEGIN")
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        _open_transactions.discard(conn)


_METADATA_INSERT = """
    INSERT INTO _ingest_metadata
        (dataset_id, table_name, season, row_count, loaded_at, loader_fn, status, detail)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


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
    """Upsert a record for an attempt that put rows in the table.

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
        conn.execute(
            _METADATA_INSERT,
            [dataset_id, table_name, season, row_count,
             datetime.datetime.now(datetime.timezone.utc), loader_fn,
             status, detail],
        )


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
    it. The reason text goes in detail, which is where a query tool can read it — until now
    the only record of a failure was a line printed to stdout.

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
        conn.execute(
            _METADATA_INSERT,
            [dataset_id, table_name, season, 0,
             datetime.datetime.now(datetime.timezone.utc), loader_fn,
             status, detail],
        )


# ── What one run did ───────────────────────────────────────────────────────────
#
# A run of the ingest is the unit someone else acts on: `nfl-mcp ingest` turns it into an
# exit code, and the container's scheduled refresh swaps the live database only on exit 0.
# So a run has to be able to say how it went, which is what the types below are for. They
# describe *this run's* fetches only — not the whole database, and not the seasons this run
# skipped because an earlier run already loaded them.


@dataclass(frozen=True)
class IngestAttempt:
    """One fetch this run made, and how it ended.

    season is None for a static (non-seasonal) dataset. detail is the same short reason that
    goes into the `detail` column of _ingest_metadata.
    """

    dataset_id: str
    season: int | None
    status: IngestStatus
    detail: str | None = None


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
    def espn_breaker_tripped(self) -> bool:
        """True when the ESPN pull stopped early instead of working through its seasons."""
        return self.espn_stop is not None


class _RunCollector:
    """Attempts gathered by one run, while it runs. Owned by run_ingest_datasets.

    The recording helpers write into the collector through _ACTIVE_RUN rather than through a
    module-level list, so a second run — or the next test — starts from nothing.
    """

    def __init__(self) -> None:
        self.attempts: list[IngestAttempt] = []
        self.espn_stop: EspnStop | None = None

    def note_attempt(
        self, dataset_id: str, season: int | None, status: IngestStatus, detail: str | None
    ) -> None:
        self.attempts.append(
            IngestAttempt(dataset_id=dataset_id, season=season, status=status, detail=detail)
        )

    def note_espn_stop(self, failed_seasons: int, seasons_left: int) -> None:
        self.espn_stop = EspnStop(failed_seasons=failed_seasons, seasons_left=seasons_left)

    def result(self) -> IngestRunResult:
        return IngestRunResult(attempts=list(self.attempts), espn_stop=self.espn_stop)


# The run in progress, if any. ContextVar (not a global) so the value is set and restored
# around one call, and so tests that call the recording helpers on their own — which is most
# of them — simply report to nobody.
_ACTIVE_RUN: ContextVar["_RunCollector | None"] = ContextVar("nfl_mcp_active_ingest_run", default=None)


def _note_attempt(
    dataset_id: str, season: int | None, status: IngestStatus, detail: str | None, loader_fn: str
) -> None:
    """Report one dataset row to the run in progress, when a run is in progress.

    Derived tables are excluded: their row records a rebuild, not a fetch, and counting one
    would let a run with no successful download still look like it loaded something.
    """
    collector = _ACTIVE_RUN.get()
    if collector is None or loader_fn == DERIVED_LOADER:
        return
    collector.note_attempt(dataset_id, season, status, detail)


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
    conn.register("_schema_df", renamed.to_arrow())
    conn.execute(
        f'CREATE TABLE "{PBP_TABLE_NAME}" AS '
        "SELECT *, '' AS enhanced_description FROM _schema_df"
    )
    conn.unregister("_schema_df")
    print(f"  plays table created — {len(sample_df.columns) + 1} columns")


def _ingest_pbp_season(conn: duckdb.DuckDBPyConnection, season: int) -> int:
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

    _reconcile_schema(conn, PBP_TABLE_NAME, df)
    with _transaction(conn):
        # Replace, not append: the fetch above succeeded, so it is safe to drop the stale
        # copy of this season (a partial in-progress season, or a --fresh re-run).
        conn.execute(f'DELETE FROM "{PBP_TABLE_NAME}" WHERE season = ?', [season])
        conn.register("_ingest_df", df.to_arrow())
        conn.execute(f'INSERT INTO "{PBP_TABLE_NAME}" BY NAME SELECT * FROM _ingest_df')
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
    """Create or append df into a DuckDB table."""
    if replace:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')

    # Existence is asked for, not inferred from a caught CatalogException: that exception
    # also means a missing column or a missing function, and the old fallback answered a
    # schema problem by trying to CREATE a table that was already there.
    if _table_exists(conn, table):
        _reconcile_schema(conn, table, df)
        conn.register("_load_df", df.to_arrow())
        conn.execute(f'INSERT INTO "{table}" BY NAME SELECT * FROM _load_df')
        conn.unregister("_load_df")
    else:
        conn.register("_load_df", df.to_arrow())
        conn.execute(f'CREATE TABLE "{table}" AS SELECT * FROM _load_df')
        conn.unregister("_load_df")


def _ingest_generic_dataset(
    conn: duckdb.DuckDBPyConnection,
    dataset_def,                        # DatasetDef from registry
    seasons: list[int] | None,          # None = load all via seasons=True
    fresh: bool = False,
) -> int:
    """Load a single dataset (non-pbp) into DuckDB. Returns total rows inserted."""
    import nflreadpy

    loader = getattr(nflreadpy, dataset_def.loader_fn)
    table = dataset_def.table_name
    total_rows = 0
    bulk_mode = seasons is None  # pass True to loader instead of per-season list

    if not dataset_def.seasonal:
        # Static dataset — one load, replace table. Static tables are point-in-time
        # snapshots (weekly rankings, player ids, contracts, teams), so they are
        # re-fetched on every ingest; the old table is only dropped after the fetch succeeds.
        print(f"\n  {dataset_def.dataset_id} (static)")
        if _is_loaded(conn, dataset_def.dataset_id):
            print(f"    Already loaded, refreshing")

        try:
            df = loader(**dataset_def.extra_params)
        except Exception as e:
            print(f"    Failed to load: {e}")
            _record_attempt(conn, dataset_def.dataset_id, table, dataset_def.loader_fn,
                            None, IngestStatus.FAILED, str(e))
            return 0

        if df is None or len(df) == 0:
            print(f"    No data returned")
            _record_attempt(conn, dataset_def.dataset_id, table, dataset_def.loader_fn,
                            None, IngestStatus.EMPTY, f"{dataset_def.loader_fn} returned no rows")
            return 0

        df = _safe_rename(df)
        print(f"    {len(df):,} rows, {len(df.columns)} columns")
        with _transaction(conn):
            _write_df_to_table(conn, table, df, replace=True)
            _record_loaded(conn, dataset_def.dataset_id, table, dataset_def.loader_fn, len(df))
        total_rows = len(df)

    else:
        # Seasonal dataset — iterate per season (bulk mode resolves all_seasons() at call
        # time, so a process that runs across the March rollover sees the new season list)
        season_list = _clip_to_window(
            all_seasons() if bulk_mode else seasons, dataset_def
        )
        for season in season_list:
            already = _is_loaded(conn, dataset_def.dataset_id, season)
            if already and not fresh and not _needs_refresh(conn, dataset_def.dataset_id, season):
                print(f"    {dataset_def.dataset_id} {season}: already loaded, skipping")
                continue
            if already:
                print(f"    {dataset_def.dataset_id} {season}: refreshing")

            print(f"\n  {dataset_def.dataset_id} {season}")
            try:
                df = loader([season], **dataset_def.extra_params)
            except Exception as e:
                print(f"    Failed to load: {e}")
                _record_attempt(conn, dataset_def.dataset_id, table, dataset_def.loader_fn,
                                season, IngestStatus.FAILED, str(e))
                continue

            if df is None or len(df) == 0:
                print(f"    No data returned")
                _record_attempt(conn, dataset_def.dataset_id, table, dataset_def.loader_fn,
                                season, IngestStatus.EMPTY,
                                f"{dataset_def.loader_fn} returned no rows")
                continue

            df = _safe_rename(df)
            print(f"    {len(df):,} rows, {len(df.columns)} columns")

            # Replace, not append: the fetch succeeded, so drop any stale rows for this
            # season (no-op on first load). Checked with _table_exists rather than by
            # swallowing the error: a statement that fails inside the transaction below
            # can leave that transaction unusable.
            with _transaction(conn):
                if _table_exists(conn, table):
                    conn.execute(f'DELETE FROM "{table}" WHERE season = {season}')

                _write_df_to_table(conn, table, df)
                _record_loaded(conn, dataset_def.dataset_id, table, dataset_def.loader_fn, len(df), season)
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
            _record_attempt(conn, defn.dataset_id, defn.table_name, "espn_records",
                            season, IngestStatus.FAILED, str(exc))
            print(f"    espn_records {season}: ESPN unavailable ({exc}); retrying next run")
            if failures >= ESPN_FAILURE_LIMIT:
                _note_espn_stop(failures, len(season_list) - index - 1)
                print(f"    espn_records: {ESPN_FAILURE_LIMIT} seasons failed in a row, "
                      "stopping — run 'nfl-mcp ingest --pre1999' again later to resume")
                break
            continue
        if not records:
            _record_attempt(conn, defn.dataset_id, defn.table_name, "espn_records",
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
                conn, defn.dataset_id, defn.table_name, "espn_records", len(records), season,
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


def _create_aggregate_tables(conn: duckdb.DuckDBPyConnection):
    print("\n  Creating aggregate tables…")

    for table in ["team_offense_stats", "team_defense_stats",
                  "situational_stats", "formation_effectiveness"]:
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    conn.execute("""
        CREATE TABLE team_offense_stats AS
        SELECT
            posteam AS team, season AS season_year,
            COUNT(*) AS total_plays,
            SUM(COALESCE(yards_gained,0)) AS total_yards,
            ROUND(AVG(yards_gained), 2) AS yards_per_play,
            SUM(CASE WHEN rush_attempt=1 THEN 1 ELSE 0 END) AS rush_plays,
            SUM(CASE WHEN pass_attempt=1 THEN 1 ELSE 0 END) AS pass_plays,
            SUM(CASE WHEN rush_attempt=1 THEN COALESCE(yards_gained,0) ELSE 0 END) AS rush_yards,
            SUM(CASE WHEN pass_attempt=1 THEN COALESCE(yards_gained,0) ELSE 0 END) AS pass_yards,
            ROUND(AVG(CASE WHEN rush_attempt=1 THEN yards_gained END), 2) AS yards_per_rush,
            ROUND(AVG(CASE WHEN pass_attempt=1 THEN yards_gained END), 2) AS yards_per_pass,
            SUM(CASE WHEN touchdown=1 THEN 1 ELSE 0 END) AS touchdowns,
            SUM(CASE WHEN interception=1 OR fumble_lost=1 THEN 1 ELSE 0 END) AS turnovers,
            ROUND(100.0*SUM(CASE WHEN down=3 AND yards_gained>=ydstogo THEN 1 ELSE 0 END)
                /NULLIF(SUM(CASE WHEN down=3 THEN 1 ELSE 0 END),0), 1) AS third_down_pct,
            ROUND(100.0*SUM(CASE WHEN yardline_100<=20 AND touchdown=1 THEN 1 ELSE 0 END)
                /NULLIF(SUM(CASE WHEN yardline_100<=20 THEN 1 ELSE 0 END),0), 1) AS red_zone_td_pct,
            SUM(CASE WHEN (pass_attempt=1 AND yards_gained>=20)
                       OR (rush_attempt=1  AND yards_gained>=10) THEN 1 ELSE 0 END) AS explosive_plays,
            ROUND(AVG(epa), 3) AS avg_epa,
            ROUND(AVG(CASE WHEN pass_attempt=1 THEN epa END), 3) AS pass_epa,
            ROUND(AVG(CASE WHEN rush_attempt=1  THEN epa END), 3) AS rush_epa
        FROM plays
        WHERE posteam IS NOT NULL
          AND play_type IN ('pass','run','field_goal','extra_point',
                            'punt','kickoff','qb_spike','qb_kneel','no_play')
        GROUP BY posteam, season
    """)
    print("    team_offense_stats ✓")

    conn.execute("""
        CREATE TABLE team_defense_stats AS
        SELECT
            defteam AS team, season AS season_year,
            COUNT(*) AS plays_against,
            SUM(COALESCE(yards_gained,0)) AS yards_allowed,
            ROUND(AVG(yards_gained), 2) AS yards_per_play_allowed,
            SUM(CASE WHEN sack=1         THEN 1 ELSE 0 END) AS sacks,
            SUM(CASE WHEN interception=1 THEN 1 ELSE 0 END) AS interceptions,
            SUM(CASE WHEN interception=1 OR fumble_lost=1 THEN 1 ELSE 0 END) AS turnovers_forced,
            ROUND(100.0*SUM(CASE WHEN down=3 AND yards_gained<ydstogo THEN 1 ELSE 0 END)
                /NULLIF(SUM(CASE WHEN down=3 THEN 1 ELSE 0 END),0), 1) AS third_down_stop_pct,
            ROUND(AVG(epa), 3) AS avg_epa_allowed
        FROM plays
        WHERE defteam IS NOT NULL
          AND play_type IN ('pass','run','field_goal','extra_point',
                            'punt','kickoff','qb_spike','qb_kneel','no_play')
        GROUP BY defteam, season
    """)
    print("    team_defense_stats ✓")

    conn.execute(f"""
        CREATE TABLE situational_stats AS
        WITH base AS (
            SELECT *, {_SITUATION_EXPR} AS situation_label
            FROM plays WHERE posteam IS NOT NULL AND down IS NOT NULL
        )
        SELECT
            posteam AS team, season AS season_year,
            situation_label AS situation,
            COUNT(*) AS plays,
            ROUND(AVG(yards_gained), 2) AS avg_yards,
            SUM(CASE WHEN touchdown=1 THEN 1 ELSE 0 END) AS touchdowns,
            ROUND(100.0*SUM(CASE WHEN yards_gained>=ydstogo THEN 1 ELSE 0 END)
                /NULLIF(COUNT(*),0), 1) AS conversion_pct,
            ROUND(AVG(epa), 3) AS avg_epa
        FROM base
        GROUP BY posteam, season, situation_label
    """)
    print("    situational_stats ✓")

    conn.execute(f"""
        CREATE TABLE formation_effectiveness AS
        SELECT
            posteam AS team, season AS season_year,
            {_FORMATION_EXPR} AS formation,
            play_type,
            COUNT(*) AS plays,
            ROUND(AVG(yards_gained), 2) AS avg_yards,
            SUM(CASE WHEN touchdown=1 THEN 1 ELSE 0 END) AS touchdowns,
            SUM(CASE WHEN interception=1 OR fumble_lost=1 THEN 1 ELSE 0 END) AS turnovers,
            ROUND(AVG(epa), 3) AS avg_epa
        FROM plays
        WHERE posteam IS NOT NULL AND play_type IN ('pass','run') AND shotgun IS NOT NULL
        GROUP BY posteam, season, {_FORMATION_EXPR}, play_type
        HAVING COUNT(*) >= 5
    """)
    print("    formation_effectiveness ✓")

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
    # 1. TD luck — actual vs expected touchdowns per player-season (most "unlucky" first).
    ("player_td_luck", ["ff_opportunity"], """
        SELECT
            player_id,
            ANY_VALUE(full_name)            AS full_name,
            ANY_VALUE(position)             AS position,
            ANY_VALUE(posteam)              AS team,
            CAST(season AS INTEGER)         AS season,
            SUM(rec_touchdown)              AS rec_td,
            SUM(rec_touchdown_exp)          AS rec_td_exp,
            SUM(rec_touchdown_diff)         AS rec_td_luck,
            SUM(rush_touchdown)             AS rush_td,
            SUM(rush_touchdown_exp)         AS rush_td_exp,
            SUM(rush_touchdown_diff)        AS rush_td_luck,
            ROUND(SUM(rec_touchdown_diff) + SUM(rush_touchdown_diff), 3) AS total_td_luck_score
        FROM ff_opportunity
        WHERE player_id IS NOT NULL
        GROUP BY player_id, CAST(season AS INTEGER)
    """),

    # 2. Rolling 3-week role trend — snap / target / carry / air-yards share with
    #    a trailing 3-week average and the current-week delta vs that average.
    ("player_role_trend", ["ff_opportunity", "snap_counts"], """
        WITH base AS (
            SELECT
                o.player_id,
                o.full_name,
                o.position,
                o.posteam                                   AS team,
                CAST(o.season AS INTEGER)                   AS season,
                CAST(o.week AS INTEGER)                      AS week,
                ROUND(100.0 * s.offense_pct, 2)             AS snap_pct,
                ROUND(100.0 * o.rec_attempt  / NULLIF(o.rec_attempt_team, 0), 2)   AS target_share_pct,
                ROUND(100.0 * o.rush_attempt / NULLIF(o.rush_attempt_team, 0), 2)  AS carry_share_pct,
                ROUND(100.0 * o.rec_air_yards / NULLIF(o.rec_air_yards_team, 0), 2) AS air_yards_share_pct
            FROM ff_opportunity o
            LEFT JOIN snap_counts s
              ON s.player  = o.full_name
             AND s.team    = o.posteam
             AND s.season  = CAST(o.season AS INTEGER)
             AND s.week    = CAST(o.week AS INTEGER)
            WHERE o.player_id IS NOT NULL
        )
        SELECT
            player_id, full_name, position, team, season, week,
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
        WINDOW w AS (PARTITION BY player_id, season ORDER BY week ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
    """),

    # 3. Separation-adjusted opportunity — joins Next Gen Stats receiving (2016+) to
    #    fantasy opportunity to flag separation-creators who underproduced (regression up).
    ("player_separation_opportunity", ["ff_opportunity", "nextgen_stats_receiving"], """
        WITH ng AS (
            SELECT
                player_gsis_id,
                season,
                AVG(avg_separation)              AS avg_separation,
                AVG(avg_yac_above_expectation)   AS avg_yac_above_expectation,
                AVG(catch_percentage)            AS catch_percentage
            FROM nextgen_stats_receiving
            WHERE week > 0 AND player_gsis_id IS NOT NULL
            GROUP BY player_gsis_id, season
        ),
        opp AS (
            SELECT
                player_id,
                ANY_VALUE(full_name)   AS full_name,
                ANY_VALUE(position)    AS position,
                ANY_VALUE(posteam)     AS team,
                CAST(season AS INTEGER) AS season,
                COUNT(DISTINCT week)   AS games,
                SUM(total_fantasy_points_diff) AS fp_diff,
                ROUND(SUM(total_fantasy_points_diff) / NULLIF(COUNT(DISTINCT week), 0), 3) AS fp_diff_per_game,
                ROUND(SUM(rec_touchdown_diff), 3) AS td_luck,
                ROUND(SUM(receptions_diff), 3)    AS catch_luck,
                ROUND(100.0 * SUM(rec_attempt)  / NULLIF(SUM(rec_attempt_team), 0), 2)  AS target_share_pct,
                ROUND(100.0 * SUM(rec_air_yards) / NULLIF(SUM(rec_air_yards_team), 0), 2) AS air_yards_share_pct
            FROM ff_opportunity
            WHERE player_id IS NOT NULL
            GROUP BY player_id, CAST(season AS INTEGER)
        )
        SELECT
            opp.player_id, opp.full_name, opp.position, opp.team, opp.season,
            opp.games, opp.fp_diff, opp.fp_diff_per_game, opp.td_luck, opp.catch_luck,
            opp.target_share_pct, opp.air_yards_share_pct,
            ROUND(ng.avg_separation, 3)            AS avg_separation,
            ROUND(ng.avg_yac_above_expectation, 3) AS avg_yac_above_expectation,
            ROUND(ng.catch_percentage, 2)          AS catch_percentage,
            (ng.avg_separation > 2.5 AND opp.fp_diff_per_game < -1.5 AND opp.td_luck < -1.0) AS regression_candidate
        FROM opp
        JOIN ng ON ng.player_gsis_id = opp.player_id AND ng.season = opp.season
    """),

    # 4. Drop rate — catchable-target drop rate from FTN charting (2022+) joined to plays.
    ("player_drop_rate", ["ftn_charting", PBP_TABLE_NAME], """
        SELECT
            p.receiver_player_id                AS player_id,
            ANY_VALUE(p.receiver_player_name)   AS player,
            p.posteam                           AS team,
            p.season                            AS season,
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
        GROUP BY p.receiver_player_id, p.posteam, p.season
    """),

    # 5. Contract value efficiency — fantasy points per $M of average per year (APY),
    #    using each player's active contract joined to seasonal fantasy production.
    ("player_contract_value", ["contracts", "ff_opportunity"], """
        WITH active AS (
            SELECT
                gsis_id,
                ANY_VALUE(position)   AS position,
                ANY_VALUE(team)       AS team,
                MAX(apy)              AS apy,
                MAX(apy_cap_pct)      AS cap_pct
            FROM contracts
            WHERE is_active = TRUE AND gsis_id IS NOT NULL
            GROUP BY gsis_id
        ),
        opp AS (
            SELECT
                player_id,
                ANY_VALUE(full_name)   AS full_name,
                ANY_VALUE(position)    AS position,
                ANY_VALUE(posteam)     AS team,
                CAST(season AS INTEGER) AS season,
                SUM(total_fantasy_points)     AS total_fp,
                SUM(total_fantasy_points_exp) AS total_fp_exp
            FROM ff_opportunity
            WHERE player_id IS NOT NULL
            GROUP BY player_id, CAST(season AS INTEGER)
        )
        SELECT
            opp.player_id,
            opp.full_name,
            COALESCE(opp.position, active.position) AS position,
            opp.team,
            opp.season,
            ROUND(active.apy, 4)       AS apy,
            ROUND(active.cap_pct, 4)   AS cap_pct,
            ROUND(opp.total_fp, 2)     AS total_fp,
            ROUND(opp.total_fp_exp, 2) AS total_fp_exp,
            ROUND(opp.total_fp / NULLIF(active.apy, 0), 2) AS fp_per_million
        FROM opp
        JOIN active ON active.gsis_id = opp.player_id
    """),

    # 6. Injury return curve — post-return snap-share recovery (as % of pre-injury
    #    baseline) at +1..+8 weeks, bucketed by normalized injury type and position.
    ("injury_return_curve", ["injuries", "snap_counts"], """
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
                sp.gsis_id, sp.full_name, sp.position, sp.season,
                sp.injury_type, sp.first_out_week, sp.last_out_week,
                AVG(sn.offense_pct) AS baseline_pct
            FROM spells sp
            JOIN snap_counts sn
              ON sn.player = sp.full_name
             AND sn.season = sp.season
             AND sn.week   < sp.first_out_week
            GROUP BY sp.gsis_id, sp.full_name, sp.position, sp.season,
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
              ON sn.player = b.full_name
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


def _create_fantasy_tables(conn: duckdb.DuckDBPyConnection):
    """Build the fantasy-analytics derived tables from already-ingested source data.

    Each table is fully recomputed (DROP + CREATE TABLE AS) and recorded in
    _ingest_metadata as a 'derived' dataset so it surfaces in nfl_catalog / nfl_status.
    Tables whose source data isn't present yet are skipped gracefully, and a copy left by an
    earlier run is dropped: it is built from data that is no longer here.
    """
    print("\n  Creating fantasy derived tables…")
    for table_name, sources, body in _FANTASY_TABLES:
        missing = [s for s in sources if not _table_exists(conn, s)]
        if missing:
            print(f"    {table_name} — skipped (missing sources: {', '.join(missing)})")
            _drop_stale_derived_table(conn, table_name)
            continue
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"CREATE TABLE {table_name} AS {body}")
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
    """
    from .franchises import schedule_alias_case

    statements: list[str] = []
    if _table_exists(conn, "espn_team_records"):
        statements.append("""
            SELECT season, team, espn_abbrev AS source_team, games, wins, losses, ties,
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
            SELECT season, {schedule_alias_case("source_team")} AS team, source_team,
                   COUNT(*)                                    AS games,
                   SUM(CASE WHEN points_for > points_against THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN points_for < points_against THEN 1 ELSE 0 END) AS losses,
                   SUM(CASE WHEN points_for = points_against THEN 1 ELSE 0 END) AS ties,
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

    The IngestRunResult describes the fetches this call made: each one's outcome, how many
    ended each way, and whether the ESPN pull gave up early. A season this run skipped
    because it was already loaded is not an attempt, so a run that only skipped seasons
    reports no attempts at all. A caller that wants the old behaviour ignores the return.

    The run owns the collector its recording helpers report into, and puts back whichever
    collector was active before, so one run cannot leak attempts into the next.
    """
    collector = _RunCollector()
    token = _ACTIVE_RUN.set(collector)
    try:
        return _ingest_datasets(
            collector, dataset_ids=dataset_ids, start=start, end=end,
            fresh=fresh, skip_views=skip_views, db_path=db_path,
        )
    finally:
        _ACTIVE_RUN.reset(token)


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

    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(path)
    # The write connection stays open for the rest of the run, so every way out of the run has
    # to close it. DuckDB allows one writer per file: a connection left behind by a raised
    # exception keeps the server and the next scheduled ingest locked out until this process
    # exits, which is what the RuntimeError below used to do.
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
            # from 1999 and raises for anything earlier.
            pbp_seasons = _clip_to_window(
                all_seasons() if bulk_mode else seasons, REGISTRY[PBP_DATASET_ID]
            )
            # Discover the schema from the newest season that actually has a pbp file.
            # Early in a new league year the current season has no play-by-play yet.
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

            _create_plays_table(conn, sample_df)

            # --fresh ignores the already-loaded set, so every requested season is re-fetched.
            # Each one then replaces only its own rows, after its own fetch succeeds.
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
                    # The season's delete + insert and its metadata row commit together, so a
                    # failing season leaves the previous copy of the season intact.
                    # _ingest_pbp_season's own transaction nests into this one and steps back.
                    with _transaction(conn):
                        rows = _ingest_pbp_season(conn, season)
                        if rows:
                            _record_loaded(
                                conn, PBP_DATASET_ID, PBP_TABLE_NAME, PBP_LOADER_FN,
                                rows, season,
                            )
                    pbp_ingested += rows

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
                    _ingest_generic_dataset(conn, defn, seasons, fresh=fresh)
                else:
                    raise ValueError(
                        f"Dataset '{dataset_id}' has source {defn.source!r}, which has no "
                        f"ingest helper (known sources: {', '.join(KNOWN_SOURCES)})"
                    )

        # ── Fantasy derived tables (built from already-ingested sources) ────────
        if not skip_views:
            _create_fantasy_tables(conn)
            _create_team_records(conn)
    finally:
        conn.close()

    result = collector.result()
    print("\n" + "=" * 60)
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
    """Backward-compatible entry point — ingests PBP only."""
    return run_ingest_datasets(
        dataset_ids=[PBP_DATASET_ID],
        start=start,
        end=end,
        fresh=fresh,
        skip_views=skip_views,
        db_path=db_path,
    )
