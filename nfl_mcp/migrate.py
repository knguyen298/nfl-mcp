"""Schema version of a database file, detection of an older one, and its upgrade.

The version lives in table _schema_version, one row per stamp; the database version is
MAX(version), and no table or no rows means version 0 (v0.4.0 and the rolled-back v0.5.3).
check() reads only. upgrade() runs the steps a database needs, in order, and stamps
SCHEMA_VERSION only when no step is left. A later group that changes what the database
must hold adds a step here and raises SCHEMA_VERSION.

Version 1 is the shape after groups D, E and F: season_type and the new columns on the
derived tables, a season on every depth_charts row, and every REQUIRED_COLUMNS table.
"""

import datetime
from dataclasses import dataclass, field

from . import __version__
from .registry import REGISTRY
from .required_columns import REQUIRED_COLUMNS

SCHEMA_VERSION = 1

_VERSION_DDL = (
    "CREATE TABLE IF NOT EXISTS _schema_version "
    "(version INTEGER, applied_at TIMESTAMP, nfl_mcp_version VARCHAR)"
)

# Raw tables come from a dataset loader; the rest of REQUIRED_COLUMNS is derived.
_TABLE_DATASET = {d.table_name: d for d in REGISTRY.values()}
RAW_TABLES = [t for t in REQUIRED_COLUMNS if t in _TABLE_DATASET]
DERIVED_TABLES = [t for t in REQUIRED_COLUMNS if t not in _TABLE_DATASET and t != "_ingest_metadata"]


@dataclass(frozen=True)
class Step:
    name: str
    needed: bool
    needs_network: bool
    detail: str


@dataclass(frozen=True)
class UpgradeStatus:
    version: int
    target: int
    steps: list[Step]
    upgrade_needed: bool
    newer: bool


@dataclass
class UpgradeResult:
    before: UpgradeStatus
    after: UpgradeStatus
    ran: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    stamped: bool = False


# ── detection (read-only) ──────────────────────────────────────────────────────

def _columns(conn) -> dict[str, set[str]]:
    rows = conn.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'main'"
    ).fetchall()
    found: dict[str, set[str]] = {}
    for table, column in rows:
        found.setdefault(table, set()).add(column)
    return found


def _read_version(conn, columns: dict[str, set[str]]) -> int:
    """MAX(version) of _schema_version; 0 when the table is absent or empty."""
    if "version" not in columns.get("_schema_version", set()):
        return 0
    return conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()[0] or 0


def read_version(conn) -> int:
    """The schema version recorded in the database (0 when none)."""
    return _read_version(conn, _columns(conn))


def _gaps(columns, tables) -> tuple[list[str], list[str]]:
    """(missing tables, missing 'table.column' pairs) among `tables`, in REQUIRED_COLUMNS order."""
    missing_tables = [t for t in tables if t not in columns]
    missing_columns = [f"{t}.{c}" for t in tables if t in columns
                       for c in REQUIRED_COLUMNS[t] if c not in columns[t]]
    return missing_tables, missing_columns


def _gap_detail(missing_tables, missing_columns, ok_text) -> str:
    parts = []
    if missing_tables:
        parts.append("missing tables: " + ", ".join(missing_tables))
    if missing_columns:
        parts.append("missing columns: " + ", ".join(missing_columns))
    return "; ".join(parts) or ok_text


def _step_derived(conn, columns) -> Step:
    tables, cols = _gaps(columns, DERIVED_TABLES)
    return Step("derived_tables", bool(tables or cols), False,
                _gap_detail(tables, cols, "every derived table and column is present"))


def _depth_null_rows(conn, columns) -> int | None:
    """Rows of depth_charts without a season (all rows when no season column); None if no table."""
    if "depth_charts" not in columns:
        return None
    where = " WHERE season IS NULL" if "season" in columns["depth_charts"] else ""
    return conn.execute(f"SELECT COUNT(*) FROM depth_charts{where}").fetchone()[0]


def _step_depth(conn, columns) -> Step:
    n = _depth_null_rows(conn, columns)
    if n:
        return Step("depth_charts_season", True, True, f"{n:,} depth_charts rows have no season")
    return Step("depth_charts_season", False, True,
                "no depth_charts table" if n is None else "every depth_charts row has a season")


def _step_raw(conn, columns) -> Step:
    tables, cols = _gaps(columns, RAW_TABLES)
    return Step("raw_columns", bool(tables or cols), True,
                _gap_detail(tables, cols, "every raw table and column is present"))


_DETECT = (_step_derived, _step_depth, _step_raw)


def check(conn) -> UpgradeStatus:
    """Report the version and which steps are needed. Never writes; works read-only."""
    columns = _columns(conn)
    version = _read_version(conn, columns)
    steps = [detect(conn, columns) for detect in _DETECT]
    return UpgradeStatus(
        version=version, target=SCHEMA_VERSION, steps=steps,
        upgrade_needed=version < SCHEMA_VERSION or any(s.needed for s in steps),
        newer=version > SCHEMA_VERSION,
    )


# ── stamping ───────────────────────────────────────────────────────────────────

def _stamp(conn) -> None:
    conn.execute(_VERSION_DDL)
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    conn.execute("INSERT INTO _schema_version VALUES (?, ?, ?)", [SCHEMA_VERSION, now, __version__])


def _stampable(status: UpgradeStatus) -> bool:
    return status.version < SCHEMA_VERSION and not any(s.needed for s in status.steps)


def stamp_if_current(conn) -> bool:
    """Stamp SCHEMA_VERSION when no step is needed. True only when a row was written.

    Returns False, never raises, when the connection cannot run the check or the write.
    """
    try:
        if not _stampable(check(conn)):
            return False
        _stamp(conn)
    except Exception:  # noqa: BLE001 - a stamp is a courtesy, never a reason to fail a caller
        return False
    return True


# ── actions ────────────────────────────────────────────────────────────────────

def _run_derived(conn) -> None:
    from .ingest import rebuild_derived_tables
    rebuild_derived_tables(conn)


def _soft_failures(fn, *args, **kwargs) -> None:
    """Run fn (an ingest helper that records failures instead of raising); raise them."""
    from . import ingest
    ingest._FAILURES.clear()
    fn(*args, **kwargs)
    failures = list(ingest._FAILURES)
    ingest._FAILURES.clear()
    if failures:
        raise RuntimeError("; ".join(failures))


def _run_depth(conn) -> None:
    """Group D's fetch-first restamp: nothing is deleted unless every re-fetch succeeded."""
    from . import ingest
    _soft_failures(ingest._ingest_generic_dataset, conn, REGISTRY["depth_charts"], [], fresh=False)


def _metadata_seasons(conn, dataset_id: str) -> list[int]:
    from . import ingest
    ingest._ensure_metadata_table(conn)
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT season FROM _ingest_metadata WHERE dataset_id = ? AND season IS NOT NULL "
        "ORDER BY season", [dataset_id]).fetchall()]


def _reload_pbp(conn, seasons: list[int], table_exists: bool) -> None:
    import nflreadpy

    from . import ingest
    seasons = ingest._clip_to_window(seasons, REGISTRY["pbp"])
    if not table_exists:
        sample = None
        for season in reversed(seasons):
            try:
                sample = nflreadpy.load_pbp([season])
                break
            except Exception as e:  # noqa: BLE001 - try an earlier season
                print(f"    pbp {season} not available ({e}); trying an earlier season")
        if sample is None:
            raise RuntimeError("could not load play-by-play for any season")
        ingest._create_plays_table(conn, sample)
    for season in seasons:
        ingest._ingest_pbp_season(conn, season)


def _run_raw(conn) -> None:
    """Re-ingest each dataset behind a raw table that is missing or lacks a column.

    A table that exists reloads the seasons recorded for it in _ingest_metadata, fresh; a
    missing seasonal table loads the dataset's default seasons; a static one loads once.
    """
    from . import ingest
    columns = _columns(conn)
    missing_tables, missing_columns = _gaps(columns, RAW_TABLES)
    affected = list(dict.fromkeys(missing_tables + [c.split(".")[0] for c in missing_columns]))
    for table in affected:
        defn = _TABLE_DATASET[table]
        exists = table in columns
        print(f"  {defn.dataset_id}: re-ingesting ({'missing columns' if exists else 'table missing'})")
        seasons = _metadata_seasons(conn, defn.dataset_id) if exists and defn.seasonal else []
        if defn.seasonal and not seasons:
            seasons = ingest._clip_to_window(ingest.ALL_SEASONS, defn)
        if defn.dataset_id == "pbp":
            _soft_failures(_reload_pbp, conn, seasons, exists)
        else:
            _soft_failures(ingest._ingest_generic_dataset, conn, defn,
                           seasons if defn.seasonal else None, fresh=True)


_ACTIONS = {"derived_tables": _run_derived, "depth_charts_season": _run_depth,
            "raw_columns": _run_raw}


# ── upgrade ────────────────────────────────────────────────────────────────────

def _needs(status: UpgradeStatus, name: str) -> bool:
    return any(s.name == name and s.needed for s in status.steps)


def upgrade(conn, *, fetch: bool = True) -> UpgradeResult:
    """Run the needed steps in order, then stamp SCHEMA_VERSION when none is left.

    A failed step is recorded and the later steps still run. Steps that need the network
    are listed in `pending` and skipped when fetch is False. Refuses a newer database.
    """
    before = check(conn)
    result = UpgradeResult(before=before, after=before)
    if before.newer:
        result.failed.append(
            f"database schema version {before.version} is newer than this nfl-mcp "
            f"({before.target}); nothing was changed")
        return result

    def run(name: str) -> bool:
        try:
            _ACTIONS[name](conn)
        except Exception as e:  # noqa: BLE001 - recorded; the later steps still run
            result.failed = [f for f in result.failed if not f.startswith(f"{name}: ")]
            result.failed.append(f"{name}: {e}")
            return False
        result.failed = [f for f in result.failed if not f.startswith(f"{name}: ")]
        if name not in result.ran:
            result.ran.append(name)
        return True

    for step in before.steps:
        if not step.needed:
            continue
        if step.needs_network and not fetch:
            result.pending.append(step.name)
            continue
        run(step.name)
    # New raw data can change what the derived tables are built from (a missing source
    # table): finish in the same run rather than leaving a second one.
    if "raw_columns" in result.ran and _needs(check(conn), "derived_tables"):
        run("derived_tables")

    after = check(conn)
    if not result.failed and _stampable(after):
        _stamp(conn)
        result.stamped = True
        after = check(conn)
    result.after = after
    return result
