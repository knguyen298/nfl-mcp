"""
NFL data ingestion into DuckDB.

Supports all nflreadpy datasets via a declarative registry.
PBP ingestion preserves its existing enhanced_description logic.

Called by: nfl-mcp init / nfl-mcp ingest
"""

import datetime
import os
import duckdb
import polars as pl
from pathlib import Path
from tqdm import tqdm

from .seasons import FIRST_SEASON, EARLIEST_SEASON, all_seasons, current_season

# Computed at import time; the current season is always the last entry.
ALL_SEASONS = all_seasons()


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

def _ensure_metadata_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create _ingest_metadata if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _ingest_metadata (
            dataset_id  VARCHAR NOT NULL,
            table_name  VARCHAR NOT NULL,
            season      INTEGER,          -- NULL for static (non-seasonal) datasets
            row_count   BIGINT,
            loaded_at   TIMESTAMP NOT NULL,
            loader_fn   VARCHAR
        )
    """)


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
    """Upsert a completion record into _ingest_metadata."""
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
    conn.execute(
        """
        INSERT INTO _ingest_metadata (dataset_id, table_name, season, row_count, loaded_at, loader_fn)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [dataset_id, table_name, season, row_count, datetime.datetime.now(datetime.timezone.utc), loader_fn],
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
        conn.register("_schema_df", renamed.to_arrow())
        conn.execute(
            "CREATE TABLE plays AS "
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

    _reconcile_schema(conn, "plays", df)
    # Replace, not append: the fetch above succeeded, so it is safe to drop the stale
    # copy of this season (a partial in-progress season, or a --fresh re-run).
    conn.execute("DELETE FROM plays WHERE season = ?", [season])
    conn.register("_ingest_df", df.to_arrow())
    conn.execute("INSERT INTO plays BY NAME SELECT * FROM _ingest_df")
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

    try:
        conn.execute(f"SELECT 1 FROM {table} LIMIT 0")
        _reconcile_schema(conn, table, df)
        conn.register("_load_df", df.to_arrow())
        conn.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _load_df")
        conn.unregister("_load_df")
    except duckdb.CatalogException:
        conn.register("_load_df", df.to_arrow())
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _load_df")
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
            return 0

        if df is None or len(df) == 0:
            print(f"    No data returned")
            return 0

        df = _safe_rename(df)
        print(f"    {len(df):,} rows, {len(df.columns)} columns")
        _write_df_to_table(conn, table, df, replace=True)
        _record_loaded(conn, dataset_def.dataset_id, table, dataset_def.loader_fn, len(df))
        total_rows = len(df)

    else:
        # Seasonal dataset — iterate per season (bulk mode uses ALL_SEASONS)
        season_list = _clip_to_window(ALL_SEASONS if bulk_mode else seasons, dataset_def)
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
                continue

            if df is None or len(df) == 0:
                print(f"    No data returned")
                continue

            df = _safe_rename(df)
            print(f"    {len(df):,} rows, {len(df.columns)} columns")

            # Replace, not append: the fetch succeeded, so drop any stale rows for this
            # season (no-op on first load; table may not exist yet).
            try:
                conn.execute(f"DELETE FROM {table} WHERE season = {season}")
            except Exception:
                pass

            _write_df_to_table(conn, table, df)
            _record_loaded(conn, dataset_def.dataset_id, table, dataset_def.loader_fn, len(df), season)
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


# Every per-play figure in the three team tables uses scrimmage plays only: the standard nflfastR
# filter with two-point attempts taken back out (a two-point try carries pass or rush = 1 and an EPA
# value - all 327 of them in this data do - and is not in the regular stats) and plays nullified by
# a flag taken back out too. A no_play row is a down a penalty undid, and the same down was replayed,
# so the official box counts the replay and not that attempt: 3,879 of them across 2024 to 2026, 1,612
# in the 2024 regular season, averaging 0.02 yards and dragging avg_epa, avg_epa_allowed,
# explosive_plays and the conversion counts. What is left still includes sacks and scrambles, which
# the box charges to the pass and the run.
_SCRIMMAGE = ("(COALESCE(pass,0)=1 OR COALESCE(rush,0)=1) "
              "AND epa IS NOT NULL AND COALESCE(two_point_attempt,0)<>1 "
              "AND COALESCE(play_type,'')<>'no_play'")

# A touchdown the offense scored itself. play_by_play gives a pick-six posteam = the team that threw
# it, so counting `touchdown` handed the offense a touchdown it did not score.
_OFFENSIVE_TD = "(COALESCE(pass_touchdown,0)=1 OR COALESCE(rush_touchdown,0)=1)"

# A play nflverse scored as a down conversion or a down failure. The conversion columns count every
# play that carries one of these flags, without also requiring pass = 1 or rush = 1: 48 quarterback
# kneels and 8 spikes in the 2024 regular season carry third_down_failed = 1, and the pass-or-rush
# population dropped all of them. A punt or a field-goal attempt never carries a flag, so kicks stay
# out of the denominator. A play a penalty undid nearly always carries none either: of the 6,569
# no_play rows in the table exactly one carries a flag - game 2024_15_MIA_HOU, a Houston fake punt on
# 4th and 2 for 35 yards, fourth_down_converted = 1 - and the conversion columns do count it. That row
# is why the Houston 2024 regular-season 4th Down row of situational_stats reports 18 attempts over 17
# plays; the other 83 rows in that table where attempts exceed plays come from kneels and spikes, which
# carry a flag and are not scrimmage plays.
_DOWN_FLAGS = ("COALESCE(third_down_converted,0)=1 OR COALESCE(third_down_failed,0)=1 "
               "OR COALESCE(fourth_down_converted,0)=1 OR COALESCE(fourth_down_failed,0)=1")

# Counting stats come from team_stats_raw (nflverse's week-level team box, the same source
# player_stats is built from) and points from schedules: both already carry the official value, so a
# sack is never added to the pass attempts. Only what they do not carry - per-play rates,
# down-and-distance, red-zone trips - is built from plays, with nflfastR's own flags. Every rate
# divides a summed total by a summed total; no weekly rate is averaged. Each table gains a
# season_type column, because the postseason used to be summed into the regular season.
#
# Derived table column -> the team_stats_raw expression behind it.
_TEAM_BOX_SUMS = {
    "attempts": "attempts", "completions": "completions", "passing_yards": "passing_yards",
    "sacks_suffered": "sacks_suffered", "sack_yards": "sack_yards_lost", "carries": "carries",
    "rushing_yards": "rushing_yards", "passing_tds": "passing_tds", "rushing_tds": "rushing_tds",
    "special_teams_tds": "special_teams_tds", "passing_interceptions": "passing_interceptions",
    "fumbles_lost": "fumbles_lost_total", "penalties": "penalties",
    "penalty_yards": "penalty_yards", "def_sacks": "def_sacks",
    "def_interceptions": "def_interceptions",
    "first_downs": "passing_first_downs + rushing_first_downs + receiving_first_downs",
}

# A database with plays only - no team box, no schedule - still gets every per-play figure: this
# relation has the columns both sources carry and no rows, so those columns come back null.
_NO_SOURCES = (
    "(SELECT NULL::VARCHAR AS team, NULL::VARCHAR AS opponent_team, "
    "NULL::VARCHAR AS season_type, NULL::VARCHAR AS game_type, "
    "NULL::VARCHAR AS home_team, NULL::VARCHAR AS away_team, NULL::INTEGER AS season, "
    "NULL::INTEGER AS home_score, NULL::INTEGER AS away_score, "
    + ", ".join(f"NULL::BIGINT AS {c.strip()}"
                for c in sorted({p.strip() for e in _TEAM_BOX_SUMS.values() for p in e.split("+")}))
    + " WHERE FALSE)"
)


def _box_cte(conn: duckdb.DuckDBPyConnection) -> str:
    """One row per team, season and season_type of the week-level team box."""
    src = "team_stats_raw" if _table_exists(conn, "team_stats_raw") else _NO_SOURCES
    sums = ", ".join(f"COALESCE(SUM({e}), 0) AS {a}" for a, e in _TEAM_BOX_SUMS.items())
    return f"box AS (SELECT team, season AS season_year, season_type, {sums} FROM {src} GROUP BY 1, 2, 3)"


def _opp_box_cte(conn: duckdb.DuckDBPyConnection) -> str:
    """The same team box read from the other side: one row per defence, season and season_type,
    built by grouping the opponent's box row on opponent_team.

    The box and the per-play population do not agree on what a defence faces, and the box is the
    larger one: Kansas City 2024 regular season 1,026 plays against in the box against 1,017
    scrimmage plays in plays, its postseason 203 against 199, Philadelphia 999 against 990. A
    quarterback kneel is a carry and a spike is a pass attempt in the league's box, while
    play_by_play flags neither with pass = 1 or rush = 1, so _SCRIMMAGE drops both - and 45 of the 46
    rows of 2024 close on exactly that difference. The nullified downs are out of both counts: plays
    keeps 87 of them against Kansas City that season, and both the league's box and _SCRIMMAGE count
    the replay of the down instead. The yards run the other way by a
    handful (5,457 over those scrimmage plays against the box's 5,451), so no rate may mix the two
    populations: yards_per_play_allowed divides the two box totals, and only expected points, which
    no box carries, is read off plays."""
    src = "team_stats_raw" if _table_exists(conn, "team_stats_raw") else _NO_SOURCES
    return ("opp AS (SELECT opponent_team AS team, season AS season_year, season_type, "
            "COALESCE(SUM(attempts + sacks_suffered + carries), 0) AS plays_against, "
            "COALESCE(SUM(passing_yards + rushing_yards + sack_yards_lost), 0) AS yards_allowed, "
            "COALESCE(SUM(passing_interceptions + fumbles_lost_total), 0) AS turnovers_forced "
            f"FROM {src} GROUP BY 1, 2, 3)")


def _points_cte(conn: duckdb.DuckDBPyConnection) -> str:
    """Points for and against per team, season and season_type. schedules.game_type names the round
    (REG, WC, DIV, CON, SB); the derived tables use plays' two values, so a round is REG or POST."""
    src = "schedules" if _table_exists(conn, "schedules") else _NO_SOURCES
    round_ = "CASE WHEN game_type = 'REG' THEN 'REG' ELSE 'POST' END"
    return (f"points AS (SELECT team, season AS season_year, season_type, SUM(pts) AS points_for,"
            f" SUM(opp_pts) AS points_against FROM ("
            f"SELECT season, {round_} AS season_type, home_team AS team, home_score AS pts,"
            f" away_score AS opp_pts FROM {src}"
            f" UNION ALL "
            f"SELECT season, {round_}, away_team, away_score, home_score FROM {src})"
            f" WHERE pts IS NOT NULL GROUP BY 1, 2, 3)")


def _create_aggregate_tables(conn: duckdb.DuckDBPyConnection):
    """Build the derived team tables from plays, the week-level team box, and the schedule.

    Called after the other datasets load, so a run that ingests team_stats_raw in the same command
    is read here instead of one run later.
    """
    print("\n  Creating aggregate tables…")

    if not _table_exists(conn, "plays"):
        print("    plays not loaded yet, skipping")
        return

    for table in ["team_offense_stats", "team_defense_stats",
                  "situational_stats", "formation_effectiveness"]:
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    box, opp, points = _box_cte(conn), _opp_box_cte(conn), _points_cte(conn)

    conn.execute(f"""
        CREATE TABLE team_offense_stats AS
        WITH scrim AS (
            -- What no box carries: expected points and the explosive-play count, both per scrimmage
            -- play. The three yards-per-play figures are NOT here - they divide the box totals, see
            -- the SELECT below.
            SELECT posteam AS team, season AS season_year, season_type,
                   SUM(CASE WHEN (pass_attempt=1 AND yards_gained>=20)
                              OR (rush_attempt=1 AND yards_gained>=10) THEN 1 ELSE 0 END)
                       AS explosive_plays,
                   ROUND(AVG(epa), 3) AS avg_epa,
                   ROUND(AVG(CASE WHEN pass_attempt=1 THEN epa END), 3) AS pass_epa,
                   ROUND(AVG(CASE WHEN rush_attempt=1 THEN epa END), 3) AS rush_epa
            FROM plays WHERE posteam IS NOT NULL AND {_SCRIMMAGE} GROUP BY 1, 2, 3
        ),
        -- Down conversions over every play that carries a conversion flag, kneels and spikes
        -- included, so this population is wider than scrim's.
        downs AS (
            SELECT posteam AS team, season AS season_year, season_type,
                   SUM(CASE WHEN COALESCE(third_down_converted,0)=1 THEN 1 ELSE 0 END)
                       AS third_down_conversions,
                   SUM(CASE WHEN COALESCE(third_down_converted,0)=1
                             OR COALESCE(third_down_failed,0)=1 THEN 1 ELSE 0 END)
                       AS third_down_attempts
            FROM plays
            WHERE posteam IS NOT NULL AND COALESCE(play_type,'')<>'no_play' AND {_DOWN_FLAGS}
            GROUP BY 1, 2, 3
        ),
        -- A red zone trip is one drive that reaches the opponent's 20, not one play inside it.
        -- game_id is in the key because drive numbers restart every week.
        -- Only a football play proves the trip: the extra point is spotted at the 15 and the two-point
        -- try at the 2, so yardline_100<=20 on its own let every scoring drive prove itself through
        -- the kick after the score (1,292 extra-point rows in 2024 sit inside that bound), which
        -- counted drives that never got inside the 20 - what a long touchdown pass looks like. The
        -- two-point try is the same defect in a different costume: nflverse records it as play_type
        -- 'pass' or 'run' spotted at the 2 (99 passes and 36 runs in the 2024 regular season), so it
        -- passes the play_type test and must be excluded by its own flag, the way _SCRIMMAGE does.
        -- That takes 23 drives out of the 1,844 the 2024 regular season first reported, 1,821 left.
        -- A trip
        -- that ends in a field goal stays in the denominator: that drive did reach the 20. The score
        -- it is credited with is one the offence scored itself (_OFFENSIVE_TD - a pick-six is not
        -- one, since play_by_play files it with posteam = the team that threw the interception). The
        -- touchdown is read only from plays inside the 20, so a drive pushed back by a penalty before
        -- it scores is a trip with no touchdown: 11 drives of the 2024 regular season entered the 20
        -- and then scored from outside it, and they stay in the denominator as misses.
        red_zone AS (
            SELECT team, season_year, season_type, COUNT(*) AS red_zone_trips,
                   SUM(touchdown) AS red_zone_touchdowns
            FROM (SELECT posteam AS team, season AS season_year, season_type, game_id, drive,
                         MAX(CASE WHEN play_type IN ('pass','run','field_goal','qb_kneel','qb_spike')
                                   AND COALESCE(two_point_attempt,0)<>1
                                  THEN 1 ELSE 0 END) AS reached_20,
                         MAX(CASE WHEN {_OFFENSIVE_TD} THEN 1 ELSE 0 END) AS touchdown
                  FROM plays
                  WHERE posteam IS NOT NULL AND drive IS NOT NULL AND game_id IS NOT NULL
                    AND yardline_100<=20
                  GROUP BY 1, 2, 3, 4, 5)
            WHERE reached_20 = 1
            GROUP BY 1, 2, 3
        ),
        {box},
        {points}
        SELECT
            s.team, s.season_year, s.season_type,
            b.attempts + b.sacks_suffered + b.carries AS total_plays,
            b.rushing_yards + b.passing_yards + b.sack_yards AS total_yards,
            -- Every rate divides one box total by another, so the row cannot hold two answers for
            -- the same question. The per-play average over scrimmage plays was the second one, and
            -- it disagreed with the box the row already carried: Kansas City 2024 regular season
            -- said 5.02 yards per play against the box's 5570 / 1091 = 5.11, 4.26 yards per rush
            -- against 1790 / 450 = 3.98, 5.92 per pass against 3780 / 641 = 5.90.
            ROUND(1.0*(b.rushing_yards + b.passing_yards + b.sack_yards)
                  / NULLIF(b.attempts + b.sacks_suffered + b.carries, 0), 2) AS yards_per_play,
            b.carries AS rush_plays,
            b.attempts AS pass_plays,
            b.completions,
            b.rushing_yards AS rush_yards,
            b.passing_yards + b.sack_yards AS pass_yards,
            ROUND(1.0*b.rushing_yards / NULLIF(b.carries, 0), 2) AS yards_per_rush,
            ROUND(1.0*(b.passing_yards + b.sack_yards)
                  / NULLIF(b.attempts + b.sacks_suffered, 0), 2) AS yards_per_pass,
            b.passing_tds + b.rushing_tds AS touchdowns,
            b.passing_tds, b.rushing_tds, b.special_teams_tds,
            b.passing_interceptions + b.fumbles_lost AS turnovers,
            b.passing_interceptions, b.fumbles_lost,
            b.sacks_suffered, b.sack_yards,
            d.third_down_conversions, d.third_down_attempts,
            ROUND(100.0*d.third_down_conversions/NULLIF(d.third_down_attempts,0), 1)
                AS third_down_pct,
            rz.red_zone_trips, rz.red_zone_touchdowns,
            ROUND(100.0*rz.red_zone_touchdowns/NULLIF(rz.red_zone_trips,0), 1)
                AS red_zone_td_pct,
            b.first_downs, b.penalties, b.penalty_yards,
            s.explosive_plays, s.avg_epa, s.pass_epa, s.rush_epa,
            p.points_for, p.points_against
        FROM scrim s
        LEFT JOIN box b      USING (team, season_year, season_type)
        LEFT JOIN downs d    USING (team, season_year, season_type)
        LEFT JOIN red_zone rz USING (team, season_year, season_type)
        LEFT JOIN points p    USING (team, season_year, season_type)
    """)
    print("    team_offense_stats ✓")

    conn.execute(f"""
        CREATE TABLE team_defense_stats AS
        WITH scrim AS (
            -- Expected points, which no box carries. The counting columns come from the opponents'
            -- box rows (opp below), and so does the yards-per-play rate in the SELECT. The two
            -- populations differ and the box is the larger: Kansas City 2024 regular season 1,026
            -- plays against in the box against 1,017 scrimmage plays here, postseason 203 against
            -- 199, Philadelphia 999 against 990, because the box charges a kneel as a carry and a
            -- spike as a pass attempt while play_by_play flags neither. Before this table took its
            -- counts from the box the row read 1069 plays and 19 takeaways, and 1069 is neither
            -- population: it is the pass-or-rush plays with the 52 of Kansas City's 87 nullified
            -- downs that carried a pass or rush flag still in, and the 9 kneels and spikes still out.
            SELECT defteam AS team, season AS season_year, season_type,
                   ROUND(AVG(epa), 3) AS avg_epa_allowed
            FROM plays WHERE defteam IS NOT NULL AND {_SCRIMMAGE} GROUP BY 1, 2, 3
        ),
        downs AS (
            SELECT defteam AS team, season AS season_year, season_type,
                   SUM(CASE WHEN COALESCE(third_down_failed,0)=1 THEN 1 ELSE 0 END)
                       AS third_down_stops,
                   SUM(CASE WHEN COALESCE(third_down_converted,0)=1
                             OR COALESCE(third_down_failed,0)=1 THEN 1 ELSE 0 END)
                       AS third_down_attempts
            FROM plays
            WHERE defteam IS NOT NULL AND COALESCE(play_type,'')<>'no_play' AND {_DOWN_FLAGS}
            GROUP BY 1, 2, 3
        ),
        {opp},
        {box},
        {points}
        SELECT
            s.team, s.season_year, s.season_type,
            -- yards_per_play_allowed divides the two box totals in this same row, so the row cannot
            -- hold two answers for the same question: averaged over scrimmage plays it said 5.37 for
            -- Kansas City 2024 regular season against 5451 / 1026 = 5.31 from the opponents' box, and
            -- Baltimore's postseason 5.50 against 553 / 104 = 5.32. All 46 rows of 2024 disagreed,
            -- by as much as 0.33. Null when the box counts no plays against this defence.
            o.plays_against, o.yards_allowed,
            ROUND(1.0*o.yards_allowed / NULLIF(o.plays_against, 0), 2) AS yards_per_play_allowed,
            b.def_sacks AS sacks, b.def_interceptions AS interceptions,
            o.turnovers_forced,
            d.third_down_stops, d.third_down_attempts,
            ROUND(100.0*d.third_down_stops/NULLIF(d.third_down_attempts,0), 1)
                AS third_down_stop_pct,
            s.avg_epa_allowed, p.points_against AS points_allowed
        FROM scrim s
        LEFT JOIN opp o USING (team, season_year, season_type)
        LEFT JOIN downs d USING (team, season_year, season_type)
        LEFT JOIN box b   USING (team, season_year, season_type)
        LEFT JOIN points p USING (team, season_year, season_type)
    """)
    print("    team_defense_stats ✓")

    # A punt and a field goal are not scrimmage plays and neither carries a conversion flag, so they
    # are out of 4th down both ways; they used to sit in the denominator and pull the conversion rate
    # down to nothing useful, and conversion_pct itself used yards_gained >= ydstogo instead of the
    # conversion flags. The two conversion columns reach a wider population than the rest of the row:
    # every play that carries a flag, so a kneel or a spike that fails a third down is an attempt.
    # A play a penalty nullified carries no flag in all but one recorded case - the Houston fake punt
    # in 2024_15_MIA_HOU - which is why that 4th Down row counts 18 attempts over 17 plays.
    conn.execute(f"""
        CREATE TABLE situational_stats AS
        WITH base AS (
            SELECT *, {_SITUATION_EXPR} AS situation_label,
                   CASE WHEN {_SCRIMMAGE} THEN 1 ELSE 0 END AS scrimmage,
                   COALESCE(third_down_converted,0)+COALESCE(fourth_down_converted,0) AS converted,
                   COALESCE(third_down_converted,0)+COALESCE(third_down_failed,0)
                      +COALESCE(fourth_down_converted,0)+COALESCE(fourth_down_failed,0) AS down_attempt
            FROM plays
            WHERE posteam IS NOT NULL AND down IS NOT NULL
              AND ({_SCRIMMAGE} OR {_DOWN_FLAGS})
        )
        SELECT
            posteam AS team, season AS season_year, season_type,
            situation_label AS situation,
            -- plays, avg_yards, touchdowns and avg_epa stay on the scrimmage population; only the
            -- two conversion columns reach the wider flag population, which adds kneels and spikes.
            SUM(scrimmage) AS plays,
            ROUND(AVG(CASE WHEN scrimmage=1 THEN yards_gained END), 2) AS avg_yards,
            SUM(CASE WHEN scrimmage=1 AND {_OFFENSIVE_TD} THEN 1 ELSE 0 END) AS touchdowns,
            SUM(converted) AS conversions,
            SUM(CASE WHEN down_attempt > 0 THEN 1 ELSE 0 END) AS conversion_attempts,
            ROUND(100.0*SUM(converted)/NULLIF(SUM(CASE WHEN down_attempt > 0 THEN 1 ELSE 0 END),0), 1)
                AS conversion_pct,
            ROUND(AVG(CASE WHEN scrimmage=1 THEN epa END), 3) AS avg_epa
        FROM base
        GROUP BY posteam, season, season_type, situation_label
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
    ("player_role_trend", ["ff_opportunity", "snap_counts", "players"], """
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
            LEFT JOIN players pl
              ON pl.gsis_id = o.player_id
            LEFT JOIN snap_counts s
              ON s.pfr_player_id = pl.pfr_id
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
        -- The Next Gen Stats figures used to be AVG() of the player's weekly rows, which is wrong
        -- twice: the weekly rows are means with no play count next to them, so a 3-route game got the
        -- weight of a 12-route game, and they are incomplete (7.2 rows per player-season on average;
        -- the season summary's targets total is larger than the sum of the weekly rows for 335 of the
        -- 366 player-seasons that have both). NGS's own season summary is the week = 0 row, so that
        -- row is the figure now. It exists only for season_type = 'REG' (the POST rows are per-game,
        -- weeks 19 to 23) and not for every receiver - 69 of the 198 with REG rows in 2024 have none
        -- - so the fallback is a weighted mean of that player's weekly rows in the same
        -- season_type (separation by targets, YAC above expectation by receptions), and
        -- separation_source says which of the two a row used. ff_opportunity has no
        -- season_type, so its week column decides it: 1-18 are regular-season games and 19-22 are
        -- playoff games (checked against schedules.game_type).
        WITH ng_weekly AS (
            SELECT player_gsis_id, season, season_type,
                   -- Separation is measured on every target, so targets weight it. YAC above
                   -- expectation is a per-reception figure, so receptions weight that one.
                   SUM(avg_separation*targets)/NULLIF(SUM(targets),0) AS avg_separation,
                   SUM(avg_yac_above_expectation*receptions)/NULLIF(SUM(receptions),0) AS avg_yac_above_expectation,
                   100.0*SUM(receptions)/NULLIF(SUM(targets),0) AS catch_percentage,
                   -- The sample beside the figure, because the fallback is thin: of the 206
                   -- regular-season fallback rows 124 rest on a single weekly row and 169 on two or
                   -- fewer (1.7 weeks on average); the postseason is 83 rows, 54 of them from one
                   -- week. NGS published weekly rows for only 291 receiving players in 2024, so the
                   -- coverage is upstream and cannot be widened here.
                   COUNT(DISTINCT week) AS fallback_weeks,
                   SUM(targets) AS fallback_targets
            FROM nextgen_stats_receiving
            WHERE week > 0 AND player_gsis_id IS NOT NULL
            GROUP BY 1, 2, 3
        ),
        ng AS (
            -- No sample columns on a summary row: NGS published the season figure itself, so nothing
            -- was estimated from weekly rows and there is no sample to report.
            SELECT player_gsis_id, season, season_type, avg_separation, avg_yac_above_expectation,
                   catch_percentage, 'ngs_season_summary' AS separation_source,
                   NULL::INTEGER AS fallback_weeks, NULL::BIGINT AS fallback_targets
            FROM nextgen_stats_receiving
            WHERE week = 0 AND player_gsis_id IS NOT NULL
            UNION ALL
            SELECT player_gsis_id, season, season_type, avg_separation, avg_yac_above_expectation,
                   catch_percentage, 'weekly_rows_weighted_by_targets' AS separation_source,
                   fallback_weeks, fallback_targets
            FROM ng_weekly
        ),
        ng_preferred AS (
            -- 'ngs_season_summary' sorts before 'weekly_rows_weighted_by_targets', so the summary
            -- row wins wherever NGS published one.
            SELECT * EXCLUDE (rn) FROM (SELECT *, ROW_NUMBER() OVER (
                       PARTITION BY player_gsis_id, season, season_type ORDER BY separation_source
                   ) AS rn FROM ng) WHERE rn = 1
        ),
        opp AS (
            SELECT
                player_id,
                CASE WHEN week >= 19 THEN 'POST' ELSE 'REG' END AS season_type,
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
            GROUP BY player_id, CAST(season AS INTEGER), CASE WHEN week >= 19 THEN 'POST' ELSE 'REG' END
        )
        SELECT
            opp.player_id, opp.full_name, opp.position, opp.team, opp.season, opp.season_type,
            opp.games, opp.fp_diff, opp.fp_diff_per_game, opp.td_luck, opp.catch_luck,
            opp.target_share_pct, opp.air_yards_share_pct,
            ROUND(ng.avg_separation, 3)            AS avg_separation,
            ROUND(ng.avg_yac_above_expectation, 3) AS avg_yac_above_expectation,
            ROUND(ng.catch_percentage, 2)          AS catch_percentage,
            ng.separation_source,
            ng.fallback_weeks,
            ng.fallback_targets,
            -- Null on a fallback row: a sample of one NGS week and five targets cannot support a
            -- regression claim (DeAndre Hopkins, Baltimore 2025: 17 games, separation 2.588 from one
            -- week holding five targets), and the flag is only ever read as a claim on the season
            -- figure.
            CASE WHEN ng.separation_source = 'ngs_season_summary'
                 THEN (ng.avg_separation > 2.5 AND opp.fp_diff_per_game < -1.5 AND opp.td_luck < -1.0)
            END AS regression_candidate
        FROM opp
        JOIN ng_preferred ng
          ON ng.player_gsis_id = opp.player_id AND ng.season = opp.season
         AND ng.season_type = opp.season_type
    """),

    # 4. Drop rate — catchable-target drop rate from FTN charting (2022+) joined to plays.
    ("player_drop_rate", ["ftn_charting", "plays"], """
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
    #    baseline) at +1..+8 weeks, grouped by the injury text as reported and position.
    ("injury_return_curve", ["injuries", "snap_counts", "players", "schedules"], """
        WITH outs AS (
            SELECT
                i.gsis_id,
                pl.pfr_id,
                i.full_name,
                i.position,
                i.team,
                CAST(i.season AS INTEGER) AS season,
                CAST(i.week AS INTEGER)   AS week,
                TRIM(COALESCE(i.report_primary_injury, i.practice_primary_injury)) AS injury_type
            FROM injuries i
            LEFT JOIN players pl ON pl.gsis_id = i.gsis_id
            WHERE i.report_status = 'Out'
              AND i.gsis_id IS NOT NULL
              AND COALESCE(i.report_primary_injury, i.practice_primary_injury) IS NOT NULL
        ),
        team_games AS (
            -- One row per week the club actually played, numbered in order. A bye week
            -- has no row, so it adds no gap between the games on either side of it.
            SELECT season, team, week,
                   ROW_NUMBER() OVER (PARTITION BY season, team ORDER BY week) AS game_no
            FROM (
                SELECT CAST(season AS INTEGER) AS season, away_team AS team,
                       CAST(week AS INTEGER) AS week
                FROM schedules
                UNION
                SELECT CAST(season AS INTEGER) AS season, home_team AS team,
                       CAST(week AS INTEGER) AS week
                FROM schedules
            ) g
        ),
        islands AS (
            -- Consecutive games with no played game in between are one 'Out' spell. Team is
            -- in the partition: game_no numbers one club's games, so a trade starts a spell.
            SELECT o.*,
                   tg.game_no - ROW_NUMBER() OVER (
                       PARTITION BY o.gsis_id, o.season, o.team ORDER BY tg.game_no
                   ) AS island
            FROM outs o
            JOIN team_games tg
              ON  tg.season = o.season
              AND tg.team   = o.team
              AND tg.week   = o.week
        ),
        spells AS (
            -- More than one candidate? Take the value from the latest reported week of the
            -- spell, ties alphabetically. MODE() and ANY_VALUE() pick among ties in scan
            -- order, which changed the row set on every build: 22 spells hold two injury
            -- texts and 9 two positions (none hold two names or two pfr ids today).
            SELECT
                gsis_id,
                team,
                season,
                (MIN(STRUCT_PACK(w := -week, v := pfr_id))).v    AS pfr_id,
                (MIN(STRUCT_PACK(w := -week, v := full_name))).v AS full_name,
                (MIN(STRUCT_PACK(w := -week, v := injury_type))).v AS injury_type,
                (MIN(STRUCT_PACK(w := -week, v := position))).v    AS position,
                MIN(week)            AS first_out_week,
                MAX(week)            AS last_out_week
            FROM islands
            GROUP BY gsis_id, season, team, island
        ),
        baseline AS (
            SELECT
                sp.gsis_id, sp.pfr_id, sp.full_name, sp.position, sp.season,
                sp.injury_type, sp.first_out_week, sp.last_out_week,
                AVG(sn.offense_pct) AS baseline_pct
            FROM spells sp
            JOIN snap_counts sn
              ON sn.pfr_player_id = sp.pfr_id
             AND sn.season = sp.season
             AND sn.week   < sp.first_out_week
            GROUP BY sp.gsis_id, sp.pfr_id, sp.full_name, sp.position, sp.season,
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


def _create_fantasy_tables(conn: duckdb.DuckDBPyConnection):
    """Build the fantasy-analytics derived tables from already-ingested source data.

    Each table is fully recomputed (DROP + CREATE TABLE AS) and recorded in
    _ingest_metadata as a 'derived' dataset so it surfaces in nfl_catalog / nfl_status.
    Tables whose source data isn't present yet are skipped gracefully.
    """
    print("\n  Creating fantasy derived tables…")
    for table_name, sources, body in _FANTASY_TABLES:
        missing = [s for s in sources if not _table_exists(conn, s)]
        if missing:
            print(f"    {table_name} — skipped (missing sources: {', '.join(missing)})")
            continue
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"CREATE TABLE {table_name} AS {body}")
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
        print("    team_records — skipped (needs schedules or espn_team_records)")
        return

    conn.execute("DROP TABLE IF EXISTS team_records")
    conn.execute(f"CREATE TABLE team_records AS {statements[0]}")
    for extra in statements[1:]:
        conn.execute(f"INSERT INTO team_records {extra}")
    row_count = conn.execute("SELECT COUNT(*) FROM team_records").fetchone()[0]
    _record_loaded(conn, "team_records", "team_records", "derived", row_count, season=None)
    print(f"    team_records ✓ ({row_count:,} rows)")


# Raise this integer whenever a derived table changes columns, and add a test
# that a database stamped one version back reads as stale. The column probe in
# derived_shape_is_stale stays independent of this stamp: a database written
# before the stamp exists has no row to compare.
DERIVED_SHAPE_VERSION = 1

# Tables that gained season_type together; probing all four catches every
# database written before that change.
_PROBE_TABLES = ("team_offense_stats", "team_defense_stats",
                 "situational_stats", "player_separation_opportunity")


# The two reads the staleness question needs, as SQL rather than as a method, so
# nfl_mcp.tools can run the identical pair through its own read-only executor
# instead of opening a second kind of connection.
_SHAPE_STAMP_SQL = ("SELECT row_count FROM _ingest_metadata "
                    "WHERE dataset_id = 'derived_shape' AND season IS NULL")
_PROBE_PLACEHOLDERS = ", ".join("?" for _ in _PROBE_TABLES)
_SHAPE_PROBE_SQL = (
    f"SELECT DISTINCT table_name FROM information_schema.columns"
    f" WHERE table_name IN ({_PROBE_PLACEHOLDERS})"
    f"   AND table_name NOT IN (SELECT DISTINCT table_name FROM information_schema.columns"
    f"     WHERE table_name IN ({_PROBE_PLACEHOLDERS}) AND column_name = 'season_type')"
)
_PROBE_PARAMS = [*list(_PROBE_TABLES), *list(_PROBE_TABLES)]


def shape_needs_rebuild(*, stamp: int | None, missing: list[str]) -> bool:
    """Decide staleness from the stamp and the probe, either of which may be all you have.

    Stale when the stamp is present and older than DERIVED_SHAPE_VERSION. Stale when the
    probe names a derived table with no season_type column, because a database written before
    the stamp existed has no row to compare.

    The third answer is the one that must stay quiet: the probe passed and there is no stamp
    row at all. A hand-built test database looks like that, and so does a database rebuilt
    just before the stamp was introduced. The CLI adds that case back as stale through
    derived_shape_is_stale, because there "no stamp" means "never rebuilt by code that could
    stamp it"; a tool reading rows the probe already vouched for has nothing to rebuild.
    """
    if missing:
        return True
    if stamp is None:
        return False
    return stamp < DERIVED_SHAPE_VERSION


def derived_shape_is_stale(conn: duckdb.DuckDBPyConnection) -> bool:
    """True when the derived tables need a rebuild for the current code.

    The stamp is the cheap check; the column probe is the one that sees a
    database written before stamps existed. Keep both, not either alone. A
    probe table the database does not have passes: absent tables are
    nfl_status's job, not a reason to rebuild on every partial database.

    A database that carries no stamp row counts as stale here. This answers "rebuild this
    volume now", and a volume with no stamp was never rebuilt by code that could stamp it.
    A tool answers the weaker question - "are the rows I am about to read the wrong shape" -
    with shape_needs_rebuild directly; see nfl_mcp.tools._stale_derived_shape.
    """
    stamp = shape_stamp(conn)
    if stamp is None:
        return True
    return shape_needs_rebuild(stamp=stamp, missing=probe_missing_season_type(conn))


def shape_stamp(conn: duckdb.DuckDBPyConnection) -> int | None:
    """The derived-shape version this database carries, or None when it carries no row."""
    try:
        row = conn.execute(_SHAPE_STAMP_SQL).fetchone()
    except (duckdb.CatalogException, duckdb.BinderException):
        return None
    return int(row[0]) if row and row[0] is not None else None


def probe_missing_season_type(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Probe tables this database has that carry no season_type column."""
    return [str(row[0]) for row in conn.execute(_SHAPE_PROBE_SQL, _PROBE_PARAMS).fetchall()]


def rebuild_derived_tables(conn: duckdb.DuckDBPyConnection, db_path: str, *,
                           apply_pragmas: bool = True) -> None:
    """Rebuild every derived table from rows already in the database: no network.

    Stamps _ingest_metadata with the version these builders produce. The only
    integer columns there are row_count and season, and season stays NULL
    because the shape covers every season, so the version rides in row_count.

    One transaction, stamp included. Each builder drops a table before it recreates it, so a run
    interrupted between the two leaves the database with fewer derived tables than it started
    with - and a reader cannot see that, because the column probe only asks about tables that are
    still there. The review measured the result: a rebuild killed one second in had already
    rebuilt player_separation_opportunity, so the probe passed and every guarded tool answered
    from the still-old-shaped injury_return_curve, while `nfl-mcp rebuild --check` said stale.
    Wrapped here, a reader sees either the whole old set of tables or the whole new one, and a
    half-finished run leaves no stamp behind: the stamp upsert commits with the tables it
    describes. DuckDB takes the DROP, the CREATE TABLE AS and the metadata upsert in one
    transaction, and rolls all of them back when the connection closes without a commit, which is
    what a killed process leaves. This function starts the transaction itself, so a caller must
    hand it a connection with no transaction open: DuckDB refuses a nested BEGIN.
    """
    print("Rebuilding derived tables (no network)...")
    if apply_pragmas:
        # A caller that opened this connection already set the caps. Re-setting temp_directory is not
        # merely redundant: once DuckDB has spilled into the directory, SET temp_directory fails with
        # 'Cannot switch temporary directory after the current one has been used'. A full ingest inside
        # the container's 512MB cap spills, so a rebuild at the end of that run must pass False.
        _apply_duckdb_pragmas(conn, db_path)
    _ensure_metadata_table(conn)
    conn.begin()
    try:
        _create_aggregate_tables(conn)
        _create_fantasy_tables(conn)
        _create_team_records(conn)
        _record_loaded(conn, dataset_id="derived_shape", table_name="derived_shape",
                       loader_fn="nfl_mcp.ingest.rebuild_derived_tables",
                       row_count=DERIVED_SHAPE_VERSION)
        conn.commit()
    except BaseException:
        # Any exit but the commit above - an exception, a Ctrl-C, or a caller that dies - undoes
        # every table this run touched, so the database keeps the one shape it started with.
        conn.rollback()
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
) -> None:
    """
    Ingest one or more datasets into DuckDB.

    dataset_ids: list of keys from REGISTRY (e.g. ["pbp", "schedules"])
    start/end:   season range; if both are None, loads all available seasons.
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
        # from 1999 and raises for anything earlier.
        pbp_seasons = _clip_to_window(all_seasons() if bulk_mode else seasons, REGISTRY["pbp"])
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

        _create_plays_table(conn, sample_df, fresh=fresh)

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
                rows = _ingest_pbp_season(conn, season)
                pbp_ingested += rows
                if rows:
                    _record_loaded(conn, "pbp", "plays", "load_pbp", rows, season)

        _create_indexes(conn)

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
                _ingest_generic_dataset(conn, defn, seasons, fresh=fresh)

    # ── Derived tables (built from already-ingested sources) ────────────────
    if not skip_views:
        # After the datasets above: the team tables read team_stats_raw and schedules, so building
        # them here means a run that ingests those two is reflected in the same run. This call also
        # stamps DERIVED_SHAPE_VERSION, so a database that a plain ingest has just refreshed answers
        # `nfl-mcp rebuild --check` as current instead of asking for a rebuild it no longer needs.
        # apply_pragmas=False because this connection got its caps at line 1463 and the ingest has
        # probably spilled by now: DuckDB refuses a second SET temp_directory once the first one has
        # been used, and a container capped at 512MB crashed on exactly that after a full --fresh run.
        rebuild_derived_tables(conn, path, apply_pragmas=False)

    conn.close()
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    if has_pbp and pbp_ingested:
        print(f"  PBP plays ingested this run: {pbp_ingested:,}")
    season_label = "all available" if bulk_mode else f"{seasons[0]}–{seasons[-1]}"
    print(f"  Seasons: {season_label}")
    print(f"  DB: {path}")
    print("=" * 60)


def run_ingest(
    start: int = FIRST_SEASON,
    end: int | None = None,
    fresh: bool = False,
    skip_views: bool = False,
    db_path: str | None = None,
) -> None:
    """Backward-compatible entry point — ingests PBP only."""
    run_ingest_datasets(
        dataset_ids=["pbp"],
        start=start,
        end=end,
        fresh=fresh,
        skip_views=skip_views,
        db_path=db_path,
    )
