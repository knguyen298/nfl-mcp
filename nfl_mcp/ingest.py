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
import duckdb
import polars as pl
from pathlib import Path
from nflreadpy.utils_date import get_current_season as _nflreadpy_current_season
from tqdm import tqdm

from .seasons import FIRST_SEASON, EARLIEST_SEASON, all_seasons, current_season

# Computed at import time; the current season is always the last entry.
ALL_SEASONS = all_seasons()


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


# Soft failures of the current run (a fetch that raised, a season write that rolled back).
# The loaders keep going and return 0 rows; run_ingest_datasets reports this list and the
# CLI turns it into a non-zero exit.
_FAILURES: list[str] = []


def _note_failure(dataset_id: str, season: int | None, err: Exception) -> None:
    label = dataset_id if season is None else f"{dataset_id} {season}"
    _FAILURES.append(f"{label}: {err}")


@contextlib.contextmanager
def _transaction(conn: duckdb.DuckDBPyConnection):
    """BEGIN ... COMMIT, or ROLLBACK and re-raise, so a season is replaced all or nothing."""
    conn.execute("BEGIN TRANSACTION")
    try:
        yield
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except Exception as rollback_err:  # noqa: BLE001 - keep the original error
            print(f"    Rollback failed: {rollback_err}")
        raise
    conn.execute("COMMIT")


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

def _ensure_metadata_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create _ingest_metadata if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _ingest_metadata (
            dataset_id  VARCHAR NOT NULL,
            table_name  VARCHAR NOT NULL,
            season      INTEGER,          -- NULL for static (non-seasonal) datasets
            row_count   BIGINT,
            loaded_at   TIMESTAMP NOT NULL,
            loader_fn   VARCHAR,
            nflreadpy_version VARCHAR     -- nflreadpy that fetched the rows; NULL for derived rows
        )
    """)
    _add_version_column(conn)


def _has_version_column(conn: duckdb.DuckDBPyConnection) -> bool:
    return bool(conn.execute(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_name = '_ingest_metadata' AND column_name = 'nflreadpy_version'"
    ).fetchone()[0])


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


def _is_loaded(conn: duckdb.DuckDBPyConnection, dataset_id: str, season: int | None = None) -> bool:
    """Return True if this dataset (+ season) is already recorded in metadata."""
    try:
        if season is None:
            result = conn.execute(
                "SELECT COUNT(*) FROM _ingest_metadata WHERE dataset_id = ? AND season IS NULL",
                [dataset_id],
            ).fetchone()
        else:
            result = conn.execute(
                "SELECT COUNT(*) FROM _ingest_metadata WHERE dataset_id = ? AND season = ?",
                [dataset_id, season],
            ).fetchone()
        return (result[0] or 0) > 0
    except duckdb.CatalogException:
        return False


def _record_loaded(
    conn: duckdb.DuckDBPyConnection,
    dataset_id: str,
    table_name: str,
    loader_fn: str,
    row_count: int,
    season: int | None = None,
) -> None:
    """Upsert a completion record into _ingest_metadata, with the nflreadpy version that
    fetched it (derived tables and the ESPN pull are not nflreadpy loads: NULL)."""
    version = None if loader_fn in ("derived", "espn_records") else _nflreadpy_version()
    # Delete any prior record for this dataset+season before inserting fresh
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
    values = [dataset_id, table_name, season, row_count, datetime.datetime.now(datetime.timezone.utc), loader_fn]
    if _has_version_column(conn):
        conn.execute(
            "INSERT INTO _ingest_metadata (dataset_id, table_name, season, row_count, loaded_at, loader_fn, "
            "nflreadpy_version) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [*values, version],
        )
    else:  # an older database that no ingest has touched yet
        conn.execute(
            "INSERT INTO _ingest_metadata (dataset_id, table_name, season, row_count, loaded_at, loader_fn) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            values,
        )


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

def _create_plays_table(conn: duckdb.DuckDBPyConnection, sample_df: pl.DataFrame, fresh: bool = False):
    if fresh:
        conn.execute("DROP TABLE IF EXISTS plays")
        print("  Dropped existing plays table")

    renamed = _safe_rename(sample_df.head(0))
    try:
        conn.execute("SELECT 1 FROM plays LIMIT 0")
        print("  plays table already exists")
    except duckdb.CatalogException:
        _register(conn, "_schema_df", renamed.to_arrow())
        conn.execute(
            "CREATE TABLE plays AS "
            "SELECT *, '' AS enhanced_description FROM _schema_df"
        )
        conn.unregister("_schema_df")
        print(f"  plays table created — {len(sample_df.columns) + 1} columns")


def _ingest_pbp_season(conn: duckdb.DuckDBPyConnection, season: int) -> int:
    """Replace one season of plays; DELETE, INSERT and the metadata row commit together."""
    import nflreadpy
    print(f"\n  {season}")
    try:
        df = nflreadpy.load_pbp([season])
    except Exception as e:
        print(f"    Failed to load: {e}")
        _note_failure("pbp", season, e)
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
            _reconcile_schema(conn, "plays", df)
            conn.execute("DELETE FROM plays WHERE season = ?", [season])
            _register(conn, "_ingest_df", df.to_arrow())
            conn.execute("INSERT INTO plays BY NAME SELECT * FROM _ingest_df")
            if total:
                _record_loaded(conn, "pbp", "plays", "load_pbp", total, season)
    except Exception as e:  # noqa: BLE001 - soft failure, reported at the end of the run
        print(f"    Failed to write, kept the previous rows: {e}")
        _note_failure("pbp", season, e)
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
    """Create or append df into a DuckDB table."""
    if replace:
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    _register(conn, "_load_df", df.to_arrow())
    try:
        if _table_exists(conn, table):
            _reconcile_schema(conn, table, df)
            conn.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _load_df")
        else:
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _load_df")
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
    if "season" not in _table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN season BIGINT")
    if not conn.execute(f"SELECT COUNT(*) FROM {table} WHERE season IS NULL").fetchone()[0]:
        return False, []
    seasons = sorted(r[0] for r in conn.execute(
        "SELECT DISTINCT season FROM _ingest_metadata WHERE dataset_id = ? "
        f"AND season IS NOT NULL AND season NOT IN "
        f"(SELECT DISTINCT season FROM {table} WHERE season IS NOT NULL)",
        [dataset_def.dataset_id],
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
    Each season's DELETE, INSERT and metadata upsert run in one transaction.
    """
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
            _note_failure(dataset_def.dataset_id, None, e)
            return 0

        if df is None or len(df) == 0:
            print(f"    No data returned")
            return 0

        df = _safe_rename(df)
        print(f"    {len(df):,} rows, {len(df.columns)} columns")
        try:
            with _transaction(conn):
                _write_df_to_table(conn, table, df, replace=True)
                _record_loaded(conn, dataset_def.dataset_id, table, dataset_def.loader_fn, len(df))
        except Exception as e:  # noqa: BLE001 - soft failure, reported at the end of the run
            print(f"    Failed to write, kept the previous table: {e}")
            _note_failure(dataset_def.dataset_id, None, e)
            return 0
        total_rows = len(df)

    else:
        # Seasonal dataset — iterate per season (bulk mode uses ALL_SEASONS)
        season_list = _clip_to_window(ALL_SEASONS if bulk_mode else seasons, dataset_def)
        if dataset_def.stamp_season:
            try:
                has_legacy, legacy_seasons = _legacy_restamp_plan(conn, dataset_def)
            except Exception as e:  # noqa: BLE001 - soft failure, reported at the end of the run
                print(f"    Failed to prepare {table}: {e}")
                _note_failure(dataset_def.dataset_id, None, e)
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
                _note_failure(dataset_def.dataset_id, None, RuntimeError(
                    f"{n_null} rows in {table} have no season and cannot be restamped; "
                    f"run: nfl-mcp ingest --dataset {dataset_def.dataset_id} --fresh"))
            elif has_legacy:
                restamped = _restamp_legacy_seasons(conn, dataset_def, loader, legacy_seasons)
                total_rows += sum(restamped.values())
                # Restamp is all or nothing. If it did not complete, write nothing for
                # these seasons now: a stamped copy would sit next to the NULL-season rows.
                season_list = [s for s in season_list if s not in legacy_seasons]
        for season in season_list:
            already = _is_loaded(conn, dataset_def.dataset_id, season)
            if already and not fresh and season not in _refresh_seasons():
                print(f"    {dataset_def.dataset_id} {season}: already loaded, skipping")
                continue
            if already:
                print(f"    {dataset_def.dataset_id} {season}: refreshing")

            print(f"\n  {dataset_def.dataset_id} {season}")
            try:
                df = loader([season], **dataset_def.extra_params)
            except Exception as e:
                print(f"    Failed to load: {e}")
                _note_failure(dataset_def.dataset_id, season, e)
                continue

            if df is None or len(df) == 0:
                print(f"    No data returned")
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
                _note_failure(dataset_def.dataset_id, season, e)
                continue
            total_rows += len(df)
            print(f"    {len(df):,} rows inserted")

    return total_rows


# ── External records (ESPN, pre-1999) ─────────────────────────────────────────

# Three straight failures means the endpoint itself is unhappy, so the run stops rather
# than spending the rest of its timeout budget on a dead network.
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

    Seasons are only marked loaded when rows came back, so a season that failed (or that
    ESPN has no data for) is retried on the next run instead of being silently frozen.

    Three seasons failing in a row stops the run: that pattern means the endpoint is down
    or rate-limiting us, not that individual seasons are empty, and it bounds how long a
    blocked network can hold up an image build. The remaining seasons stay unrecorded.
    """
    from .espn import EspnUnavailable, fetch_season_records
    from .franchises import expected_franchise_count
    from .registry import REGISTRY

    defn = REGISTRY["espn_records"]
    season_list = _clip_to_window(ALL_SEASONS if seasons is None else seasons, defn)
    conn.execute(_ESPN_RECORDS_DDL)
    if not season_list:
        print("\n  espn_records: nothing in range (covers 1970-1998), skipping")
        return 0

    print("\n  espn_records (external source, ~2 requests per season)")
    total = 0
    failures = 0
    for season in season_list:
        if _is_loaded(conn, defn.dataset_id, season) and not fresh:
            print(f"    espn_records {season}: already loaded, skipping")
            continue
        try:
            records = fetch_season_records(season)
        except EspnUnavailable as exc:
            failures += 1
            print(f"    espn_records {season}: ESPN unavailable ({exc}); retrying next run")
            if failures >= ESPN_FAILURE_LIMIT:
                print(f"    espn_records: {ESPN_FAILURE_LIMIT} seasons failed in a row, "
                      "stopping — run 'nfl-mcp ingest --pre1999' again later to resume")
                break
            continue
        if not records:
            print(f"    espn_records {season}: no records published")
            continue

        conn.execute("DELETE FROM espn_team_records WHERE season = ?", [season])
        conn.executemany(
            "INSERT INTO espn_team_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (r.season, r.team, r.espn_team_id, r.espn_abbrev, r.games,
                 r.wins, r.losses, r.ties, r.record_summary, r.source_game_id, r.fetched_at)
                for r in records
            ],
        )
        _record_loaded(conn, defn.dataset_id, defn.table_name, "espn_records", len(records), season)
        total += len(records)
        failures = 0

        expected = expected_franchise_count(season)
        gap = "" if len(records) == expected else f"  ({expected - len(records)} franchises absent from ESPN's archive)"
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
        _record_loaded(conn, table, table, "derived", row_count, season=None)

    print("  Aggregate tables done")


def _table_exists(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    """Return True if a base table named `name` exists in the main schema."""
    row = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_name = ?",
        [name],
    ).fetchone()
    return (row[0] or 0) > 0


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
    Tables whose source data isn't present yet are skipped gracefully.
    """
    print("\n  Creating fantasy derived tables…")
    ff_typed = _ff_opportunity_typed(conn)
    for table_name, sources, body in _FANTASY_TABLES:
        missing = [s for s in sources if not _table_exists(conn, s)]
        if missing:
            print(f"    {table_name} — skipped (missing sources: {', '.join(missing)})")
            continue
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"CREATE TABLE {table_name} AS {body.replace('{ff_typed}', ff_typed)}")
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        _record_loaded(conn, table_name, table_name, "derived", row_count, season=None)
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
        print("    team_records — skipped (needs schedules or espn_team_records)")
        return

    conn.execute("DROP TABLE IF EXISTS team_records")
    conn.execute(f"CREATE TABLE team_records AS {statements[0]}")
    for extra in statements[1:]:
        conn.execute(f"INSERT INTO team_records {extra}")
    row_count = conn.execute("SELECT COUNT(*) FROM team_records").fetchone()[0]
    _record_loaded(conn, "team_records", "team_records", "derived", row_count, season=None)
    print(f"    team_records ✓ ({row_count:,} rows)")


def rebuild_derived_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Rebuild every derived table from the loaded source data in one transaction.

    On any failure the transaction is rolled back, so the previous tables stay intact,
    and the error is re-raised. All SQL goes through conn.execute.
    """
    conn.execute("BEGIN TRANSACTION")
    try:
        _create_aggregate_tables(conn)
        _create_fantasy_tables(conn)
        _create_team_records(conn)
        conn.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except duckdb.Error as rollback_error:
            print(f"    Rollback after the failed rebuild also failed: {rollback_error}")
        raise


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
) -> list[str]:
    """
    Ingest one or more datasets into DuckDB.

    dataset_ids: list of keys from REGISTRY (e.g. ["pbp", "schedules"])
    start/end:   season range; if both are None, loads all available seasons.
    fresh:       re-fetch the requested seasons; rows of other seasons are kept.

    A failed fetch or season write does not stop the other datasets. Returns the failures
    ("<dataset> <season>: <error>"), empty on a clean run; the CLI exits non-zero on any.
    """
    import nflreadpy
    from .config import get_duckdb_path
    from .registry import REGISTRY

    bulk_mode = start is None and end is None
    if not bulk_mode:
        start = start or FIRST_SEASON
        end = end or current_season()
        if start > end:
            raise ValueError("start must be less than or equal to end")

    _FAILURES.clear()
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
    _apply_duckdb_pragmas(conn, path)

    _ensure_metadata_table(conn)

    pbp_ingested = 0
    has_pbp = "pbp" in dataset_ids

    # ── PBP (special path — enhanced_description, null filtering) ──────────
    if has_pbp:
        print("\n── Play-by-play ──")

        # Play-by-play is on a special code path, so it must clip to its own window
        # explicitly: the default backfill starts in 1966, but load_pbp only publishes
        # from 1999 and raises for anything earlier, or past get_current_season().
        pbp_seasons = _clip_to_window(all_seasons() if bulk_mode else seasons, REGISTRY["pbp"])
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
                "SELECT DISTINCT season FROM plays WHERE season IS NOT NULL"
            ).fetchall()
        } if not fresh else set()

        refresh   = _refresh_seasons()
        to_ingest = [s for s in pbp_seasons if s not in loaded_pbp or s in refresh]
        skipped   = [s for s in pbp_seasons if s in loaded_pbp and s not in refresh]
        refreshed = [s for s in pbp_seasons if s in loaded_pbp and s in refresh]

        if skipped:
            print(f"  Already loaded, skipping: {skipped}")
        if refreshed:
            print(f"  Current season, re-fetching: {refreshed}")
        if not to_ingest:
            print("  All PBP seasons already loaded.")
        else:
            print(f"  Will ingest: {to_ingest}")
            for season in to_ingest:
                pbp_ingested += _ingest_pbp_season(conn, season)

        _create_indexes(conn)

        if not skip_views:
            _create_aggregate_tables(conn)

    # ── All other datasets ─────────────────────────────────────────────────
    other_ids = [d for d in dataset_ids if d != "pbp"]
    if other_ids:
        print("\n── Additional datasets ──")
        for dataset_id in other_ids:
            defn = REGISTRY.get(dataset_id)
            if defn is None:
                print(f"\n  WARNING: unknown dataset '{dataset_id}', skipping")
                continue
            if defn.source == "espn":
                _ingest_espn_records(conn, seasons, fresh=fresh)
            else:
                try:
                    _ingest_generic_dataset(conn, defn, seasons, fresh=fresh)
                except Exception as e:  # noqa: BLE001 - soft failure, reported at the end of the run
                    print(f"    Failed: {e}")
                    _note_failure(dataset_id, None, e)

    # ── Fantasy derived tables (built from already-ingested sources) ────────
    if not skip_views:
        _create_fantasy_tables(conn)
        _create_team_records(conn)

    failures = list(_FAILURES)
    if not failures:
        from .migrate import stamp_if_current
        if stamp_if_current(conn):
            print("\n  Schema version stamped.")
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
    return failures


def run_ingest(
    start: int = FIRST_SEASON,
    end: int | None = None,
    fresh: bool = False,
    skip_views: bool = False,
    db_path: str | None = None,
) -> list[str]:
    """Backward-compatible entry point — ingests PBP only. Returns the failures."""
    return run_ingest_datasets(
        dataset_ids=["pbp"],
        start=start,
        end=end,
        fresh=fresh,
        skip_views=skip_views,
        db_path=db_path,
    )
