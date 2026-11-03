"""Unit tests for ingest helpers — no network calls, uses in-memory DuckDB."""

import datetime
import sys
import types

import duckdb
import polars as pl
import pytest

from nfl_mcp.registry import DatasetDef
from tests.team_table_fixtures import TEAM_PLAYS, load_team_sources, rows
from nfl_mcp.ingest import (
    _apply_duckdb_pragmas,
    _build_enhanced_description,
    _clip_to_window,
    _create_aggregate_tables,
    _create_fantasy_tables,
    _create_indexes,
    _create_plays_table,
    _create_team_records,
    _duckdb_type_for_polars,
    _ingest_espn_records,
    _ensure_metadata_table,
    _ESPN_RECORDS_DDL,
    _FANTASY_TABLES,
    _ingest_generic_dataset,
    _ingest_pbp_season,
    _is_loaded,
    _record_loaded,
    _reconcile_schema,
    _safe_rename,
    _FANTASY_TABLES,
    _str,
    _table_exists,
    _write_df_to_table,
    run_ingest,
    run_ingest_datasets,
    ALL_SEASONS,
    all_seasons,
    current_season,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def conn():
    """Fresh in-memory DuckDB connection per test."""
    c = duckdb.connect(":memory:")
    _ensure_metadata_table(c)
    yield c
    c.close()


def _curve_rows(conn) -> list:
    """injury_return_curve as (weeks since return, avg recovery %, sample size)."""
    return conn.execute(
        "SELECT week_post_return, avg_snap_pct_recovery, sample_size "
        "FROM injury_return_curve ORDER BY week_post_return"
    ).fetchall()


def _small_df(**cols) -> pl.DataFrame:
    """Build a tiny Polars DataFrame from keyword column-name→list pairs."""
    return pl.DataFrame(cols)


def _install_fake_nflreadpy(monkeypatch, **loaders):
    """Install a lightweight fake nflreadpy module for import-time patching."""
    fake = types.ModuleType("nflreadpy")
    for name, fn in loaders.items():
        setattr(fake, name, fn)
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)


# ── _ensure_metadata_table ─────────────────────────────────────────────────────

class TestEnsureMetadataTable:
    def test_creates_table(self):
        c = duckdb.connect(":memory:")
        _ensure_metadata_table(c)
        tables = {r[0] for r in c.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()}
        assert "_ingest_metadata" in tables
        c.close()

    def test_idempotent(self):
        c = duckdb.connect(":memory:")
        _ensure_metadata_table(c)
        _ensure_metadata_table(c)   # should not raise
        c.close()

    def test_expected_columns(self):
        c = duckdb.connect(":memory:")
        _ensure_metadata_table(c)
        cols = {r[1] for r in c.execute("PRAGMA table_info('_ingest_metadata')").fetchall()}
        assert {"dataset_id", "table_name", "season", "row_count", "loaded_at", "loader_fn"} <= cols
        c.close()


# ── _is_loaded / _record_loaded ────────────────────────────────────────────────

class TestIsLoaded:
    def test_returns_false_initially(self, conn):
        assert not _is_loaded(conn, "schedules", 2024)

    def test_returns_false_for_static_initially(self, conn):
        assert not _is_loaded(conn, "players")

    def test_returns_true_after_record_seasonal(self, conn):
        _record_loaded(conn, "schedules", "schedules", "load_schedules", 285, season=2024)
        assert _is_loaded(conn, "schedules", 2024)

    def test_returns_true_after_record_static(self, conn):
        _record_loaded(conn, "players", "players", "load_players", 24000)
        assert _is_loaded(conn, "players")

    def test_different_seasons_are_independent(self, conn):
        _record_loaded(conn, "injuries", "injuries", "load_injuries", 5000, season=2023)
        assert _is_loaded(conn, "injuries", 2023)
        assert not _is_loaded(conn, "injuries", 2024)

    def test_different_datasets_are_independent(self, conn):
        _record_loaded(conn, "schedules", "schedules", "load_schedules", 285, season=2024)
        assert not _is_loaded(conn, "rosters", 2024)

    def test_returns_false_when_metadata_table_missing(self):
        c = duckdb.connect(":memory:")
        assert _is_loaded(c, "schedules", 2024) is False
        c.close()


class TestRecordLoaded:
    def test_inserts_record(self, conn):
        _record_loaded(conn, "schedules", "schedules", "load_schedules", 285, season=2024)
        rows = conn.execute(
            "SELECT dataset_id, table_name, season, row_count, loader_fn "
            "FROM _ingest_metadata"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0] == ("schedules", "schedules", 2024, 285, "load_schedules")

    def test_upserts_on_repeat(self, conn):
        _record_loaded(conn, "schedules", "schedules", "load_schedules", 285, season=2024)
        _record_loaded(conn, "schedules", "schedules", "load_schedules", 290, season=2024)
        rows = conn.execute(
            "SELECT row_count FROM _ingest_metadata WHERE dataset_id = 'schedules'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 290     # updated, not duplicated

    def test_null_season_for_static(self, conn):
        _record_loaded(conn, "teams", "teams", "load_teams", 36)
        row = conn.execute(
            "SELECT season FROM _ingest_metadata WHERE dataset_id = 'teams'"
        ).fetchone()
        assert row[0] is None

    def test_loaded_at_is_set(self, conn):
        _record_loaded(conn, "teams", "teams", "load_teams", 36)
        ts = conn.execute(
            "SELECT loaded_at FROM _ingest_metadata WHERE dataset_id = 'teams'"
        ).fetchone()[0]
        assert isinstance(ts, datetime.datetime)
        # Timestamp should be within the last 10 seconds
        age = abs((datetime.datetime.now() - ts).total_seconds())
        assert age < 10

    def test_multiple_seasons_stored_separately(self, conn):
        for season in [2022, 2023, 2024]:
            _record_loaded(conn, "rosters", "rosters", "load_rosters", 3000, season=season)
        count = conn.execute(
            "SELECT COUNT(*) FROM _ingest_metadata WHERE dataset_id = 'rosters'"
        ).fetchone()[0]
        assert count == 3


# ── _safe_rename ───────────────────────────────────────────────────────────────

class TestSafeRename:
    def test_replaces_dots(self):
        df = pl.DataFrame({"a.b": [1]})
        result = _safe_rename(df)
        assert "a_b" in result.columns

    def test_replaces_hyphens(self):
        df = pl.DataFrame({"a-b": [1]})
        result = _safe_rename(df)
        assert "a_b" in result.columns

    def test_replaces_spaces(self):
        df = pl.DataFrame({"a b": [1]})
        result = _safe_rename(df)
        assert "a_b" in result.columns

    def test_safe_names_unchanged(self):
        df = pl.DataFrame({"season": [1], "team_abbr": [2]})
        result = _safe_rename(df)
        assert result.columns == ["season", "team_abbr"]

    def test_multiple_replacements_in_one_name(self):
        df = pl.DataFrame({"a.b-c d": [1]})
        result = _safe_rename(df)
        assert "a_b_c_d" in result.columns


# ── _duckdb_type_for_polars ────────────────────────────────────────────────────

class TestDuckdbTypeForPolars:
    @pytest.mark.parametrize("dtype,expected", [
        (pl.Int8,    "BIGINT"),
        (pl.Int16,   "BIGINT"),
        (pl.Int32,   "BIGINT"),
        (pl.Int64,   "BIGINT"),
        (pl.UInt32,  "BIGINT"),
        (pl.Float32, "DOUBLE"),
        (pl.Float64, "DOUBLE"),
        (pl.Boolean, "BOOLEAN"),
        (pl.Date,    "DATE"),
        (pl.Time,    "TIME"),
        (pl.Datetime, "TIMESTAMP"),
        (pl.Utf8,    "VARCHAR"),
    ])
    def test_type_mapping(self, dtype, expected):
        assert _duckdb_type_for_polars(dtype) == expected

    def test_unknown_type_returns_varchar(self):
        # Any type not explicitly mapped falls back to VARCHAR
        assert _duckdb_type_for_polars(pl.List(pl.Int32)) == "VARCHAR"


# ── _write_df_to_table ─────────────────────────────────────────────────────────

class TestWriteDfToTable:
    def test_creates_new_table(self, conn):
        df = pl.DataFrame({"season": [2024], "team": ["KC"]})
        _write_df_to_table(conn, "test_tbl", df)
        count = conn.execute("SELECT COUNT(*) FROM test_tbl").fetchone()[0]
        assert count == 1

    def test_appends_to_existing(self, conn):
        df1 = pl.DataFrame({"season": [2023]})
        df2 = pl.DataFrame({"season": [2024]})
        _write_df_to_table(conn, "test_tbl", df1)
        _write_df_to_table(conn, "test_tbl", df2)
        count = conn.execute("SELECT COUNT(*) FROM test_tbl").fetchone()[0]
        assert count == 2

    def test_replace_drops_and_recreates(self, conn):
        df1 = pl.DataFrame({"season": [2023]})
        df2 = pl.DataFrame({"season": [2024]})
        _write_df_to_table(conn, "test_tbl", df1)
        _write_df_to_table(conn, "test_tbl", df2, replace=True)
        rows = conn.execute("SELECT season FROM test_tbl").fetchall()
        assert rows == [(2024,)]     # old row gone


# ── _reconcile_schema ──────────────────────────────────────────────────────────

class TestReconcileSchema:
    def test_adds_missing_columns(self, conn):
        conn.execute("CREATE TABLE tbl (season BIGINT)")
        df = pl.DataFrame({"season": [2024], "week": [1], "team": ["KC"]})
        _reconcile_schema(conn, "tbl", df)
        cols = {r[1] for r in conn.execute("PRAGMA table_info('tbl')").fetchall()}
        assert {"season", "week", "team"} <= cols

    def test_ignores_existing_columns(self, conn):
        conn.execute("CREATE TABLE tbl (season BIGINT, team VARCHAR)")
        df = pl.DataFrame({"season": [2024], "team": ["KC"]})
        _reconcile_schema(conn, "tbl", df)   # should not raise
        cols = {r[1] for r in conn.execute("PRAGMA table_info('tbl')").fetchall()}
        assert cols == {"season", "team"}


# ── _str / description helpers ────────────────────────────────────────────────

class TestStringAndDescriptionHelpers:
    @pytest.mark.parametrize("value,expected", [
        (None, ""),
        ("None", ""),
        ("nan", ""),
        ("NaN", ""),
        ("", ""),
        (42, "42"),
        ("KC", "KC"),
    ])
    def test_str_normalization(self, value, expected):
        assert _str(value) == expected

    def test_build_enhanced_description_includes_context_and_tags(self):
        row = {
            "down": 3,
            "ydstogo": 8,
            "qtr": 4,
            "season": 2024,
            "week": 9,
            "season_type": "POST",
            "posteam": "KC",
            "defteam": "BAL",
            "play_type": "pass",
            "passer_player_name": "P.Mahomes",
            "rusher_player_name": None,
            "receiver_player_name": "T.Kelce",
            "desc": "Deep middle pass complete",
            "yardline_100": 12,
            "touchdown": 1,
            "interception": 1,
            "fumble_lost": 0,
            "yards_gained": 25,
            "pass_attempt": 1,
            "rush_attempt": 0,
        }
        text = _build_enhanced_description(row)
        assert "3 & 8" in text
        assert "Week 9 2024 (POST): KC vs BAL" in text
        assert "PASS" in text
        assert "QB: P.Mahomes" in text
        assert "Receiver: T.Kelce" in text
        assert "TOUCHDOWN" in text
        assert "TURNOVER" in text
        assert "EXPLOSIVE" in text
        assert "PLAYOFFS" in text

    def test_build_enhanced_description_handles_bad_numeric_values(self):
        row = {
            "down": "bad",
            "ydstogo": "bad",
            "qtr": "bad",
            "season": "2024",
            "week": "bad",
            "season_type": "REG",
            "posteam": "KC",
            "defteam": "BAL",
            "play_type": "run",
            "desc": "Simple run",
        }
        text = _build_enhanced_description(row)
        assert "2024 (REG): KC vs BAL" in text
        assert "RUN" in text
        assert "Simple run" in text

    def test_build_enhanced_description_handles_missing_week(self):
        row = {
            "season": 2024,
            "season_type": "REG",
            "posteam": "KC",
            "defteam": "BAL",
            "play_type": "run",
        }
        text = _build_enhanced_description(row)
        assert "2024 (REG): KC vs BAL" in text

    def test_build_enhanced_description_tolerates_bad_tag_inputs(self):
        row = {
            "season": 2024,
            "season_type": "REG",
            "posteam": "KC",
            "defteam": "BAL",
            "play_type": "pass",
            "yardline_100": "bad",
            "down": "bad",
            "yards_gained": "bad",
            "pass_attempt": 1,
            "rush_attempt": 0,
        }
        text = _build_enhanced_description(row)
        assert "KC vs BAL" in text


# ── create-table / pbp ingestion ───────────────────────────────────────────────

class TestPbpIngestionHelpers:
    def test_create_plays_table_creates_schema_and_enhanced_column(self, conn):
        sample = pl.DataFrame({"a.b": [1], "team name": ["KC"]})
        _create_plays_table(conn, sample, fresh=False)
        cols = {r[1] for r in conn.execute("PRAGMA table_info('plays')").fetchall()}
        assert "a_b" in cols
        assert "team_name" in cols
        assert "enhanced_description" in cols

    def test_create_plays_table_fresh_recreates_existing_table(self, conn):
        conn.execute("CREATE TABLE plays (season BIGINT)")
        conn.execute("INSERT INTO plays VALUES (2023)")
        sample = pl.DataFrame({"season": [2024], "posteam": ["KC"], "defteam": ["BAL"]})
        _create_plays_table(conn, sample, fresh=True)
        count = conn.execute("SELECT COUNT(*) FROM plays").fetchone()[0]
        assert count == 0

    def test_create_plays_table_when_already_exists_and_not_fresh(self, conn):
        conn.execute("CREATE TABLE plays (season BIGINT)")
        sample = pl.DataFrame({"season": [2024]})
        _create_plays_table(conn, sample, fresh=False)
        cols = {r[1] for r in conn.execute("PRAGMA table_info('plays')").fetchall()}
        assert "season" in cols

    def test_ingest_pbp_season_inserts_filtered_rows(self, conn, monkeypatch):
        df = pl.DataFrame({
            "season": [2024, 2024],
            "week": [1, 1],
            "season_type": ["REG", "REG"],
            "posteam": ["KC", None],
            "defteam": ["BAL", "KC"],
            "play_type": ["pass", "run"],
            "down": [3, 1],
            "ydstogo": [8, 10],
            "qtr": [4, 2],
            "yardline_100": [15, 50],
            "passer_player_name": ["P.Mahomes", None],
            "rusher_player_name": [None, "I.Pacheco"],
            "receiver_player_name": ["T.Kelce", None],
            "desc": ["Touchdown throw", "Run up middle"],
            "touchdown": [1, 0],
            "interception": [0, 0],
            "fumble_lost": [0, 0],
            "yards_gained": [24, 3],
            "pass_attempt": [1, 0],
            "rush_attempt": [0, 1],
            "game_id": ["g1", "g1"],
        })

        _create_plays_table(conn, df, fresh=False)
        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: df)

        inserted = _ingest_pbp_season(conn, 2024)
        assert inserted == 1
        row = conn.execute("SELECT posteam, enhanced_description FROM plays").fetchone()
        assert row[0] == "KC"
        assert "TOUCHDOWN" in row[1]

    def test_ingest_pbp_season_returns_zero_on_loader_error(self, conn, monkeypatch):
        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: (_ for _ in ()).throw(RuntimeError("boom")))
        assert _ingest_pbp_season(conn, 2024) == 0


# ── generic dataset ingestion ───────────────────────────────────────────────────

class TestGenericDatasetIngestion:
    def test_static_dataset_success(self, conn, monkeypatch):
        defn = DatasetDef(
            dataset_id="fake_static",
            loader_fn="load_fake_static",
            table_name="fake_static",
            seasonal=False,
            default=False,
        )
        _install_fake_nflreadpy(
            monkeypatch,
            load_fake_static=lambda **kwargs: pl.DataFrame({"team": ["KC"]}),
        )
        total = _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=False)
        assert total == 1
        assert _is_loaded(conn, "fake_static")
        assert conn.execute("SELECT COUNT(*) FROM fake_static").fetchone()[0] == 1

    def test_static_dataset_refreshes_when_already_loaded(self, conn, monkeypatch):
        defn = DatasetDef(
            dataset_id="fake_static",
            loader_fn="load_fake_static",
            table_name="fake_static",
            seasonal=False,
            default=False,
        )
        conn.execute("CREATE TABLE fake_static (x BIGINT)")
        conn.execute("INSERT INTO fake_static VALUES (1), (2)")
        _record_loaded(conn, "fake_static", "fake_static", "load_fake_static", 2)
        called = []
        _install_fake_nflreadpy(
            monkeypatch,
            load_fake_static=lambda **kwargs: called.append(True) or pl.DataFrame({"x": [99]}),
        )
        total = _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=False)
        assert total == 1
        assert called == [True]
        assert conn.execute("SELECT x FROM fake_static").fetchall() == [(99,)]

    def test_static_dataset_keeps_old_table_when_refresh_fetch_fails(self, conn, monkeypatch):
        defn = DatasetDef(
            dataset_id="fake_static",
            loader_fn="load_fake_static",
            table_name="fake_static",
            seasonal=False,
            default=False,
        )
        conn.execute("CREATE TABLE fake_static (x BIGINT)")
        conn.execute("INSERT INTO fake_static VALUES (1)")
        _record_loaded(conn, "fake_static", "fake_static", "load_fake_static", 1)
        _install_fake_nflreadpy(
            monkeypatch,
            load_fake_static=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=False) == 0
        assert conn.execute("SELECT x FROM fake_static").fetchall() == [(1,)]

    def test_static_dataset_handles_loader_error_and_empty(self, conn, monkeypatch):
        defn = DatasetDef(
            dataset_id="fake_static",
            loader_fn="load_fake_static",
            table_name="fake_static",
            seasonal=False,
            default=False,
        )
        _install_fake_nflreadpy(
            monkeypatch,
            load_fake_static=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=False) == 0

        _install_fake_nflreadpy(monkeypatch, load_fake_static=lambda **kwargs: pl.DataFrame({"x": []}))
        assert _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=False) == 0

    def test_seasonal_dataset_covers_skip_windows_and_partial_failures(self, conn, monkeypatch):
        defn = DatasetDef(
            dataset_id="fake_seasonal",
            loader_fn="load_fake_seasonal",
            table_name="fake_seasonal",
            seasonal=True,
            default=False,
            min_season=2020,
            max_season=2024,
        )
        _record_loaded(conn, "fake_seasonal", "fake_seasonal", "load_fake_seasonal", 1, season=2020)

        def loader(seasons, **kwargs):
            season = seasons[0]
            if season == 2021:
                raise RuntimeError("transient loader failure")
            if season == 2023:
                return pl.DataFrame({"season": [], "value": []})
            return pl.DataFrame({"season": [season], "value": [season]})

        _install_fake_nflreadpy(monkeypatch, load_fake_seasonal=loader)
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2030)
        total = _ingest_generic_dataset(
            conn,
            defn,
            seasons=[2019, 2020, 2021, 2023, 2024, 2025],
            fresh=False,
        )
        assert total == 1
        loaded_2024 = _is_loaded(conn, "fake_seasonal", 2024)
        loaded_2020 = _is_loaded(conn, "fake_seasonal", 2020)
        assert loaded_2024 is True
        assert loaded_2020 is True

    def test_seasonal_fresh_deletes_existing_rows_and_reinserts(self, conn, monkeypatch):
        defn = DatasetDef(
            dataset_id="fake_seasonal",
            loader_fn="load_fake_seasonal",
            table_name="fake_seasonal",
            seasonal=True,
            default=False,
        )
        conn.execute("CREATE TABLE fake_seasonal (season BIGINT, value BIGINT)")
        conn.execute("INSERT INTO fake_seasonal VALUES (2024, 1), (2024, 2), (2023, 9)")

        _install_fake_nflreadpy(
            monkeypatch,
            load_fake_seasonal=lambda seasons, **kwargs: pl.DataFrame({"season": [2024], "value": [99]}),
        )
        total = _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=True)
        assert total == 1
        rows = conn.execute(
            "SELECT season, value FROM fake_seasonal WHERE season = 2024 ORDER BY value"
        ).fetchall()
        assert rows == [(2024, 99)]

    def test_seasonal_fresh_ignores_missing_table_on_delete(self, conn, monkeypatch):
        defn = DatasetDef(
            dataset_id="fake_missing_table",
            loader_fn="load_fake_missing_table",
            table_name="fake_missing_table",
            seasonal=True,
            default=False,
        )
        _install_fake_nflreadpy(
            monkeypatch,
            load_fake_missing_table=lambda seasons, **kwargs: pl.DataFrame({"season": [2024], "value": [1]}),
        )
        total = _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=True)
        assert total == 1


    def test_seasonal_current_season_is_refetched_even_when_loaded(self, conn, monkeypatch):
        defn = DatasetDef(
            dataset_id="fake_seasonal",
            loader_fn="load_fake_seasonal",
            table_name="fake_seasonal",
            seasonal=True,
            default=False,
        )
        conn.execute("CREATE TABLE fake_seasonal (season BIGINT, value BIGINT)")
        conn.execute("INSERT INTO fake_seasonal VALUES (2024, 1), (2023, 9)")
        _record_loaded(conn, "fake_seasonal", "fake_seasonal", "load_fake_seasonal", 1, season=2024)
        _record_loaded(conn, "fake_seasonal", "fake_seasonal", "load_fake_seasonal", 1, season=2023)
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2024)

        fetched = []
        def loader(seasons, **kwargs):
            fetched.append(seasons[0])
            return pl.DataFrame({"season": [2024, 2024], "value": [1, 2]})

        _install_fake_nflreadpy(monkeypatch, load_fake_seasonal=loader)
        total = _ingest_generic_dataset(conn, defn, seasons=[2023, 2024], fresh=False)

        assert fetched == [2024]          # 2023 is complete → skipped
        assert total == 2
        rows = conn.execute("SELECT season, value FROM fake_seasonal ORDER BY season, value").fetchall()
        assert rows == [(2023, 9), (2024, 1), (2024, 2)]   # no duplicate 2024 rows

    def test_seasonal_current_season_keeps_rows_when_fetch_fails(self, conn, monkeypatch):
        defn = DatasetDef(
            dataset_id="fake_seasonal",
            loader_fn="load_fake_seasonal",
            table_name="fake_seasonal",
            seasonal=True,
            default=False,
        )
        conn.execute("CREATE TABLE fake_seasonal (season BIGINT, value BIGINT)")
        conn.execute("INSERT INTO fake_seasonal VALUES (2024, 1)")
        _record_loaded(conn, "fake_seasonal", "fake_seasonal", "load_fake_seasonal", 1, season=2024)
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2024)
        _install_fake_nflreadpy(
            monkeypatch,
            load_fake_seasonal=lambda seasons, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=False) == 0
        assert conn.execute("SELECT value FROM fake_seasonal").fetchall() == [(1,)]


class TestPbpCurrentSeasonRefresh:
    def _plays(self, conn, rows):
        df = pl.DataFrame(rows)
        _create_plays_table(conn, df)
        conn.register("_seed", df.with_columns(pl.lit("").alias("enhanced_description")).to_arrow())
        conn.execute("INSERT INTO plays BY NAME SELECT * FROM _seed")
        conn.unregister("_seed")

    def test_ingest_pbp_season_replaces_existing_rows_for_that_season(self, conn, monkeypatch):
        base = {"posteam": ["KC"], "defteam": ["BAL"], "play_id": [1], "game_id": ["g"], "week": [1]}
        self._plays(conn, {**base, "season": [2024]})
        self._plays_extra = None
        conn.execute("INSERT INTO plays BY NAME SELECT 'PHI' AS posteam, 'DAL' AS defteam, 2 AS play_id, 'h' AS game_id, 1 AS week, 2023 AS season, '' AS enhanced_description")

        new_df = pl.DataFrame({**base, "play_id": [1, 2], "posteam": ["KC", "KC"], "defteam": ["BAL", "BAL"],
                               "game_id": ["g", "g"], "week": [1, 2], "season": [2024, 2024]})
        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: new_df)

        assert _ingest_pbp_season(conn, 2024) == 2
        rows = conn.execute("SELECT season, COUNT(*) FROM plays GROUP BY season ORDER BY season").fetchall()
        assert rows == [(2023, 1), (2024, 2)]

    def test_run_ingest_datasets_refetches_current_season_pbp(self, monkeypatch, tmp_path):
        conn = TestRunIngestOrchestration._DummyConn(loaded_rows=[(2023,), (2024,)])
        pbp_calls = []
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2024)
        _install_fake_nflreadpy(
            monkeypatch,
            load_pbp=lambda seasons: pl.DataFrame({"season": [2024], "posteam": ["KC"], "defteam": ["BAL"]}),
        )
        monkeypatch.setattr("nfl_mcp.ingest.duckdb.connect", lambda path: conn)
        monkeypatch.setattr("nfl_mcp.ingest._ensure_metadata_table", lambda c: None)
        monkeypatch.setattr("nfl_mcp.ingest._create_plays_table", lambda c, df, fresh=False: None)
        monkeypatch.setattr("nfl_mcp.ingest._ingest_pbp_season", lambda c, season: pbp_calls.append(season) or 1)
        monkeypatch.setattr("nfl_mcp.ingest._record_loaded", lambda *a, **k: None)
        monkeypatch.setattr("nfl_mcp.ingest._create_indexes", lambda c: None)

        run_ingest_datasets(["pbp"], start=2023, end=2024, skip_views=True,
                            db_path=str(tmp_path / "nflread.duckdb"))
        assert pbp_calls == [2024]

    def test_schema_discovery_falls_back_when_current_season_has_no_pbp(self, monkeypatch, tmp_path):
        conn = TestRunIngestOrchestration._DummyConn(loaded_rows=[])
        asked = []
        def load_pbp(seasons):
            asked.append(seasons[0])
            if seasons[0] == 2025:
                raise FileNotFoundError("no file yet")
            return pl.DataFrame({"season": [2024], "posteam": ["KC"], "defteam": ["BAL"]})
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2025)
        _install_fake_nflreadpy(monkeypatch, load_pbp=load_pbp)
        monkeypatch.setattr("nfl_mcp.ingest.duckdb.connect", lambda path: conn)
        monkeypatch.setattr("nfl_mcp.ingest._ensure_metadata_table", lambda c: None)
        monkeypatch.setattr("nfl_mcp.ingest._create_plays_table", lambda c, df, fresh=False: None)
        monkeypatch.setattr("nfl_mcp.ingest._ingest_pbp_season", lambda c, season: 0)
        monkeypatch.setattr("nfl_mcp.ingest._create_indexes", lambda c: None)

        run_ingest_datasets(["pbp"], start=2024, end=2025, skip_views=True,
                            db_path=str(tmp_path / "nflread.duckdb"))
        assert asked[:2] == [2025, 2024]

    def test_schema_discovery_gives_up_when_no_requested_season_has_pbp(self, monkeypatch, tmp_path):
        # Every candidate raised, so there is no schema to build the table from. This has
        # to fail loudly instead of writing an empty plays table and reporting success.
        conn = TestRunIngestOrchestration._DummyConn(loaded_rows=[])
        asked = []

        def load_pbp(seasons):
            asked.append(seasons[0])
            raise FileNotFoundError("no pbp file")

        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2025)
        _install_fake_nflreadpy(monkeypatch, load_pbp=load_pbp)
        monkeypatch.setattr("nfl_mcp.ingest.duckdb.connect", lambda path: conn)
        monkeypatch.setattr("nfl_mcp.ingest._ensure_metadata_table", lambda c: None)

        with pytest.raises(RuntimeError, match="Could not load play-by-play"):
            run_ingest_datasets(["pbp"], start=2024, end=2025, skip_views=True,
                                db_path=str(tmp_path / "nflread.duckdb"))
        assert asked == [2025, 2024]


class TestSeasonHelpers:
    @pytest.mark.parametrize("date_, expected", [
        (datetime.date(2026, 9, 14), 2026),
        (datetime.date(2026, 3, 1), 2026),
        (datetime.date(2026, 2, 28), 2025),
        (datetime.date(2025, 1, 15), 2024),
    ])
    def test_current_season_rolls_over_in_march(self, date_, expected):
        assert current_season(date_) == expected

    def test_all_seasons_spans_the_oldest_dataset_and_ends_at_current(self):
        seasons = all_seasons(datetime.date(2026, 9, 14))
        assert seasons[0] == 1966 and seasons[-1] == 2026

    def test_all_seasons_constant_matches_today(self):
        assert ALL_SEASONS[-1] == current_season()


# ── per-dataset depth clipping ───────────────────────────────────────────

class TestClipToWindow:
    def test_clips_the_early_years_a_dataset_cannot_supply(self):
        from nfl_mcp.registry import REGISTRY
        deep = list(range(1920, 1931))
        assert _clip_to_window(deep, REGISTRY["pbp"]) == []
        assert _clip_to_window(deep, REGISTRY["rosters"]) == deep

    def test_clips_both_ends_when_a_max_is_declared(self):
        defn = DatasetDef(dataset_id="x", loader_fn="load_x", table_name="x",
                          seasonal=True, default=False, min_season=2016, max_season=2018)
        assert _clip_to_window(list(range(2014, 2021)), defn) == [2016, 2017, 2018]

    def test_unbounded_windows_pass_through(self):
        defn = DatasetDef(dataset_id="x", loader_fn="load_x", table_name="x",
                          seasonal=True, default=False)
        assert _clip_to_window([1920, 2026], defn) == [1920, 2026]

    def test_pbp_ingest_clips_to_play_by_play_coverage(self, monkeypatch, tmp_path):
        # A request that reaches back before 1999 must not fire load_pbp for those
        # seasons: nflreadpy raises for them, so they would be pure noise in the log.
        conn = TestRunIngestOrchestration._DummyConn(loaded_rows=[])
        pbp_calls = []
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2000)
        _install_fake_nflreadpy(
            monkeypatch,
            load_pbp=lambda seasons: pl.DataFrame({"season": [1999], "posteam": ["KC"], "defteam": ["BAL"]}),
        )
        monkeypatch.setattr("nfl_mcp.ingest.duckdb.connect", lambda path: conn)
        monkeypatch.setattr("nfl_mcp.ingest._ensure_metadata_table", lambda c: None)
        monkeypatch.setattr("nfl_mcp.ingest._create_plays_table", lambda c, df, fresh=False: None)
        monkeypatch.setattr("nfl_mcp.ingest._ingest_pbp_season", lambda c, season: pbp_calls.append(season) or 1)
        monkeypatch.setattr("nfl_mcp.ingest._record_loaded", lambda *a, **k: None)
        monkeypatch.setattr("nfl_mcp.ingest._create_indexes", lambda c: None)

        run_ingest_datasets(["pbp"], start=1995, end=2000, skip_views=True,
                            db_path=str(tmp_path / "nflread.duckdb"))
        assert pbp_calls == [1999, 2000]


# ── aggregate tables and indexes ────────────────────────────────────────────────

class TestAggregateAndIndexBuilders:
    def test_create_aggregate_tables_from_minimal_plays(self, conn):
        load_team_sources(conn, _write_df_to_table)

        _create_aggregate_tables(conn)
        created = {r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
        assert {"team_offense_stats", "team_defense_stats",
                "situational_stats", "formation_effectiveness"} <= created

    def test_aggregate_tables_are_skipped_without_plays(self, conn):
        _create_aggregate_tables(conn)  # should not raise
        assert not _table_exists(conn, "team_offense_stats")

    def test_create_indexes_handles_missing_columns_gracefully(self, conn):
        conn.execute("CREATE TABLE plays (season BIGINT)")
        _create_indexes(conn)  # should not raise even though many indexes will fail

    def test_create_indexes_on_complete_schema(self, conn):
        conn.execute(
            "CREATE TABLE plays (season BIGINT, posteam VARCHAR, defteam VARCHAR, game_id VARCHAR, play_type VARCHAR)"
        )
        _create_indexes(conn)
        idx = conn.execute(
            "SELECT COUNT(*) FROM duckdb_indexes() WHERE table_name = 'plays'"
        ).fetchone()[0]
        assert idx > 0

    def test_create_fantasy_tables_builds_present_and_skips_missing(self, conn):
        # Only ff_opportunity present → player_td_luck builds; the rest skip gracefully.
        ff = pl.DataFrame({
            "player_id": ["00-1", "00-1", "00-2"],
            "full_name": ["A Back", "A Back", "B Wr"],
            "position": ["RB", "RB", "WR"],
            "posteam": ["KC", "KC", "BAL"],
            "season": ["2024", "2024", "2024"],
            "week": [1.0, 2.0, 1.0],
            "rec_touchdown": [1.0, 0.0, 2.0],
            "rec_touchdown_exp": [1.5, 0.5, 1.0],
            "rec_touchdown_diff": [-0.5, -0.5, 1.0],
            "rush_touchdown": [1.0, 1.0, 0.0],
            "rush_touchdown_exp": [0.5, 2.0, 0.0],
            "rush_touchdown_diff": [0.5, -1.0, 0.0],
        })
        _write_df_to_table(conn, "ff_opportunity", ff, replace=True)

        _create_fantasy_tables(conn)

        created = {r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
        assert "player_td_luck" in created
        # Sources for these are absent, so they must be skipped (not created).
        assert "player_role_trend" not in created
        assert "player_drop_rate" not in created

        # The built table is recorded in metadata as a derived dataset.
        meta = conn.execute(
            "SELECT loader_fn, season FROM _ingest_metadata WHERE dataset_id = 'player_td_luck'"
        ).fetchone()
        assert meta == ("derived", None)

        # One row per player-season; total_td_luck_score aggregates rec+rush diff.
        row = conn.execute(
            "SELECT total_td_luck_score FROM player_td_luck WHERE player_id = '00-1'"
        ).fetchone()
        assert row[0] == -1.5

    def test_table_exists_reports_presence(self, conn):
        from nfl_mcp.ingest import _table_exists
        assert _table_exists(conn, "_ingest_metadata") is True
        assert _table_exists(conn, "does_not_exist") is False


class TestFantasyTablesJoinById:
    """player_role_trend and injury_return_curve must key on ids, not on names.

    Both tables used to join snap_counts by name. That lost every row where the
    fantasy name differed from the PFR name, and it let two players who share a
    name share one baseline.
    """

    @staticmethod
    def _write_sources(conn, snap_rows):
        # ff_opportunity calls the tight end "Terry Kelce"; snap_counts (PFR) and the
        # players table both use "Travis Kelce" for the same pfr id.
        _write_df_to_table(conn, "ff_opportunity", pl.DataFrame({
            "player_id": ["00-1", "00-1", "00-2"],
            "full_name": ["Terry Kelce", "Terry Kelce", "Justin Jefferson"],
            "position": ["TE", "TE", "WR"],
            "posteam": ["KC", "KC", "MIN"],
            "season": ["2024", "2024", "2024"],
            "week": [1.0, 2.0, 1.0],
            "rec_attempt": [4.0, 6.0, 8.0],
            "rec_attempt_team": [10.0, 10.0, 20.0],
            "rush_attempt": [0.0, 0.0, 0.0],
            "rush_attempt_team": [10.0, 10.0, 20.0],
            "rec_air_yards": [40.0, 60.0, 80.0],
            "rec_air_yards_team": [100.0, 100.0, 200.0],
            "rec_touchdown": [0.0, 1.0, 0.0],
            "rec_touchdown_exp": [0.5, 0.5, 0.5],
            "rec_touchdown_diff": [-0.5, 0.5, -0.5],
            "rush_touchdown": [0.0, 0.0, 0.0],
            "rush_touchdown_exp": [0.0, 0.0, 0.0],
            "rush_touchdown_diff": [0.0, 0.0, 0.0],
        }), replace=True)
        _write_df_to_table(conn, "players", pl.DataFrame({
            "gsis_id": ["00-1", "00-2"],
            "pfr_id": ["KelcTr00", "JeffJu00"],
        }), replace=True)
        _write_df_to_table(conn, "snap_counts", pl.DataFrame({
            "pfr_player_id": [r[0] for r in snap_rows],
            "player": [r[1] for r in snap_rows],
            "team": [r[2] for r in snap_rows],
            "season": [r[3] for r in snap_rows],
            "week": [r[4] for r in snap_rows],
            "offense_pct": [r[5] for r in snap_rows],
        }), replace=True)

    def test_role_trend_uses_the_id_mapping_when_the_names_differ(self, conn):
        self._write_sources(conn, [
            ("KelcTr00", "Travis Kelce", "KC", 2024, 1, 0.80),
            ("KelcTr00", "Travis Kelce", "KC", 2024, 2, 0.60),
            ("JeffJu00", "Justin Jefferson", "MIN", 2024, 1, 0.90),
        ])
        _create_fantasy_tables(conn)

        rows = conn.execute(
            "SELECT player_id, week, snap_pct FROM player_role_trend ORDER BY player_id, week"
        ).fetchall()
        assert [r[2] for r in rows] == [80.0, 60.0, 90.0]   # no nulls, right player each row

    def test_role_trend_has_no_snap_when_the_id_maps_to_nothing(self, conn):
        # The only snap row belongs to a third player, pfr JeffJu99, who happens to carry
        # the same name and team as ff row 00-2. The name join handed him those snaps;
        # the id join finds no snap row for JeffJu00 and leaves the share null.
        self._write_sources(conn, [
            ("JeffJu99", "Justin Jefferson", "MIN", 2024, 1, 0.90),
        ])
        _create_fantasy_tables(conn)

        rows = conn.execute(
            "SELECT player_id, snap_pct FROM player_role_trend ORDER BY player_id, week"
        ).fetchall()
        assert [r[1] for r in rows] == [None, None, None]

    def test_injury_return_curve_gives_same_name_players_their_own_baseline(self, conn):
        # Two players named 'Justin Jefferson'. Only the first (00-1, JeffJu00) sat out
        # weeks 3-4; the other kept playing at a flat 20% share.
        _write_df_to_table(conn, "players", pl.DataFrame({
            "gsis_id": ["00-1", "00-2"],
            "pfr_id": ["JeffJu00", "JeffJu01"],
        }), replace=True)
        _write_df_to_table(conn, "injuries", pl.DataFrame({
            "gsis_id": ["00-1", "00-1"],
            "full_name": ["Justin Jefferson", "Justin Jefferson"],
            "position": ["TE", "TE"],
            "team": ["MIN", "MIN"],
            "season": [2024, 2024],
            "week": [3, 4],
            "report_status": ["Out", "Out"],
            "report_primary_injury": ["hamstring", "hamstring"],
            "practice_primary_injury": [None, None],
        }), replace=True)
        # MIN played every week 1-6 (no bye), so weeks 3 and 4 are two consecutive
        # games and stay one spell.
        _write_df_to_table(conn, "schedules", pl.DataFrame({
            "season": [2024] * 6,
            "week": [1, 2, 3, 4, 5, 6],
            "away_team": ["MIN"] * 6,
            "home_team": ["DET"] * 6,
        }), replace=True)
        _write_df_to_table(conn, "snap_counts", pl.DataFrame({
            "pfr_player_id": ["JeffJu00", "JeffJu00", "JeffJu00", "JeffJu00",
                              "JeffJu01", "JeffJu01", "JeffJu01", "JeffJu01"],
            "player": ["Justin Jefferson"] * 8,
            "team": ["MIN"] * 8,
            "season": [2024] * 8,
            "week": [1, 2, 5, 6, 1, 2, 5, 6],
            "offense_pct": [0.8, 0.4, 0.6, 0.3, 0.2, 0.2, 0.2, 0.2],
        }), replace=True)

        _create_fantasy_tables(conn)

        curve = conn.execute(
            "SELECT week_post_return, avg_snap_pct_recovery, sample_size "
            "FROM injury_return_curve ORDER BY week_post_return"
        ).fetchall()
        # The baseline is the injured player's own pre-injury snaps (0.8, 0.4 → 0.6), so
        # +1 is his own usage and +2 is half of it, one sample each. The old name join
        # pooled both players: it gave every bucket sample_size 2 and +2 as 62.5%.
        assert curve == [(1, 100.0, 1), (2, 50.0, 1)]


class TestInjuryReturnCurveSpellsAndText:
    """The curve groups by the injury text as reported, and spells follow the schedule.

    The old build pushed each injury through a chain of LIKE checks, so 'Foot/Ankle'
    became 'ankle' (ankle is checked before foot) and any text the chain did not know
    became 'other'. It also found spells from consecutive week numbers, so a bye week
    in the middle of an injury cut one spell into two and the phantom first half handed
    the second half's return weeks a second sample.
    """

    @staticmethod
    def _write_season(conn, out_rows, game_weeks=(1, 2, 3, 4, 5, 6),
                      snaps=((1, 0.8), (2, 0.4), (5, 0.6), (6, 0.3))):
        """One MIN tight end and the schedule around him.

        out_rows: [(week, report_primary_injury, practice_primary_injury)] with
        report_status 'Out'. game_weeks: every week MIN played, so a week missing from
        it is a bye. snaps: the player's own snap share, 80%/40% before the injury
        (baseline 60%) and 60%/30% after he returns.
        """
        _write_df_to_table(conn, "players", pl.DataFrame({
            "gsis_id": ["00-1"], "pfr_id": ["TestTe00"],
        }), replace=True)
        _write_df_to_table(conn, "injuries", pl.DataFrame({
            "gsis_id": ["00-1"] * len(out_rows),
            "full_name": ["Test Tee"] * len(out_rows),
            "position": ["TE"] * len(out_rows),
            "team": ["MIN"] * len(out_rows),
            "season": [2024] * len(out_rows),
            "week": [w for w, _, _ in out_rows],
            "report_status": ["Out"] * len(out_rows),
            "report_primary_injury": [r for _, r, _ in out_rows],
            "practice_primary_injury": [p for _, _, p in out_rows],
        }), replace=True)
        _write_df_to_table(conn, "schedules", pl.DataFrame({
            "season": [2024] * len(game_weeks),
            "week": list(game_weeks),
            "away_team": ["MIN"] * len(game_weeks),
            "home_team": ["DET"] * len(game_weeks),
        }), replace=True)
        _write_df_to_table(conn, "snap_counts", pl.DataFrame({
            "pfr_player_id": ["TestTe00"] * len(snaps),
            "player": ["Test Tee"] * len(snaps),
            "team": ["MIN"] * len(snaps),
            "season": [2024] * len(snaps),
            "week": [w for w, _ in snaps],
            "offense_pct": [p for _, p in snaps],
        }), replace=True)

    def test_reported_text_is_kept_and_never_becomes_a_group(self, conn):
        self._write_season(conn, [(3, "Foot/Ankle", None), (4, "Foot/Ankle", None)])
        _create_fantasy_tables(conn)

        rows = conn.execute(
            "SELECT injury_type, week_post_return, avg_snap_pct_recovery, sample_size "
            "FROM injury_return_curve ORDER BY week_post_return"
        ).fetchall()
        # 'Foot/Ankle' is its own group. The old CASE read 'ankle' first and filed it
        # there, and never left the reported wording anywhere at all.
        assert {r[0] for r in rows} == {"Foot/Ankle"}
        assert [r[1:] for r in rows] == [(1, 100.0, 1), (2, 50.0, 1)]

    def test_unmatched_text_is_not_filed_under_other(self, conn):
        self._write_season(
            conn, [(3, None, "Not injury related - personal matter"),
                   (4, None, "Not injury related - personal matter")],
        )
        _create_fantasy_tables(conn)

        types = {r[0] for r in conn.execute(
            "SELECT injury_type FROM injury_return_curve").fetchall()}
        assert types == {"Not injury related - personal matter"}

    def test_text_is_trimmed_and_keeps_its_case(self, conn):
        self._write_season(conn, [(3, "  Knee  ", None), (4, "Knee", None)])
        _create_fantasy_tables(conn)

        types = {r[0] for r in conn.execute(
            "SELECT injury_type FROM injury_return_curve").fetchall()}
        assert types == {"Knee"}

    def test_a_bye_week_does_not_split_a_spell(self, conn):
        # Out weeks 3 and 5, MIN's bye in week 4: he never returned between them, so
        # this is one spell that ends at week 5 and one return at week 6.
        self._write_season(
            conn,
            [(3, "Knee", None), (5, "Knee", None)],
            game_weeks=(1, 2, 3, 5, 6),
            snaps=((1, 0.8), (2, 0.4), (6, 0.6)),
        )
        _create_fantasy_tables(conn)

        # The old week-number islands also made a spell of week 3 alone, and counted
        # the week-6 return twice (offset 1 and offset 3).
        assert _curve_rows(conn) == [(1, 100.0, 1)]

    def test_a_game_between_two_out_weeks_still_splits_the_spell(self, conn):
        # Same Out weeks 3 and 5, but MIN played week 4 and he was back that week, so
        # these are two spells and week 6 is the second return.
        self._write_season(
            conn,
            [(3, "Knee", None), (5, "Knee", None)],
            game_weeks=(1, 2, 3, 4, 5, 6),
            snaps=((1, 0.8), (2, 0.4), (4, 0.6), (6, 0.6)),
        )
        _create_fantasy_tables(conn)

        assert _curve_rows(conn) == [(1, 100.0, 2), (3, 100.0, 1)]

    @staticmethod
    def _write_tie_season(conn, out_rows, snaps=((1, 0.8), (2, 0.4), (6, 0.6))):
        """One MIN player, one spell, and out_rows = [(week, text, position)]."""
        _write_df_to_table(conn, "players", pl.DataFrame({
            "gsis_id": ["00-1"], "pfr_id": ["TestTe00"],
        }), replace=True)
        _write_df_to_table(conn, "injuries", pl.DataFrame({
            "gsis_id": ["00-1"] * len(out_rows),
            "full_name": ["Test Tee"] * len(out_rows),
            "position": [p for _, _, p in out_rows],
            "team": ["MIN"] * len(out_rows),
            "season": [2024] * len(out_rows),
            "week": [w for w, _, _ in out_rows],
            "report_status": ["Out"] * len(out_rows),
            "report_primary_injury": [t for _, t, _ in out_rows],
            "practice_primary_injury": [None] * len(out_rows),
        }), replace=True)
        _write_df_to_table(conn, "schedules", pl.DataFrame({
            "season": [2024] * 6,
            "week": [1, 2, 3, 4, 5, 6],
            "away_team": ["MIN"] * 6,
            "home_team": ["DET"] * 6,
        }), replace=True)
        _write_df_to_table(conn, "snap_counts", pl.DataFrame({
            "pfr_player_id": ["TestTe00"] * len(snaps),
            "player": ["Test Tee"] * len(snaps),
            "team": ["MIN"] * len(snaps),
            "season": [2024] * len(snaps),
            "week": [w for w, _ in snaps],
            "offense_pct": [p for _, p in snaps],
        }), replace=True)

    def _grouped(self, conn):
        return conn.execute(
            "SELECT injury_type, position, week_post_return, avg_snap_pct_recovery, "
            "       sample_size FROM injury_return_curve "
            "ORDER BY injury_type, position, week_post_return"
        ).fetchall()

    def test_a_spell_with_no_majority_text_uses_the_latest_week_and_stays_put(self, conn):
        # Weeks 3 and 4 report different texts and different positions, so the spell holds
        # no majority text at all and MODE() has a two-way tie to resolve.
        self._write_tie_season(conn, [
            (3, "Ribs", "TE"),
            (4, "Rib", "WR"),
        ], snaps=((1, 0.8), (2, 0.4), (5, 0.6)))
        _create_fantasy_tables(conn)
        first = self._grouped(conn)
        _create_fantasy_tables(conn)
        assert self._grouped(conn) == first
        # Both values come from week 4, the last week he was reported Out; a remaining
        # tie would break alphabetically.
        assert first == [("Rib", "WR", 1, 100.0, 1)]

    def test_the_latest_week_beats_the_most_frequent_text(self, conn):
        # 'Rib' is the mode (2 rows to 1), but week 5 says 'Ribs', and that is the text the
        # player was still out for when the spell ended.
        self._write_tie_season(conn, [
            (3, "Rib", "TE"),
            (4, "Rib", "TE"),
            (5, "Ribs", "TE"),
        ])
        _create_fantasy_tables(conn)
        assert [r[0] for r in self._grouped(conn)] == ["Ribs"]


@pytest.mark.integration
class TestInjuryReturnCurveFromRealRows:
    """Group E's two rules checked against rows the local database actually holds.

    Both tests read the loaded database read-only and build into memory, so the
    shared file is never written. They skip when the database is absent.
    """

    def test_every_group_is_a_reported_injury_text(self, rebuilt_db):
        # No value in the table may be a made-up group name: each one has to equal,
        # character for character, some injury text in the report.
        with duckdb.connect(str(rebuilt_db), read_only=True) as src:
            orphans = src.execute("""
                SELECT DISTINCT c.injury_type
                FROM injury_return_curve c
                WHERE NOT EXISTS (
                    SELECT 1 FROM injuries i
                    WHERE TRIM(COALESCE(i.report_primary_injury, i.practice_primary_injury))
                          = c.injury_type
                )
            """).fetchall()
        assert orphans == []

    def test_the_table_is_identical_on_two_builds(self, rebuilt_db):
        """Same rows every build: the real data has ties, and they must break one way.

        The review found 14 spells whose injury text has no majority and 9 that carry more
        than one position. MODE() and ANY_VALUE() read those in scan order, which gave the
        table 762, 761, 760 and 760 rows on four builds of the old code.
        """
        body = next(b for name, _s, b in _FANTASY_TABLES if name == "injury_return_curve")
        order = "ORDER BY injury_type, position, week_post_return"
        with duckdb.connect(str(rebuilt_db), read_only=True) as src:
            first = src.execute(f"CREATE TEMP TABLE curve_a AS {body}").execute(
                f"SELECT * FROM curve_a {order}").fetchall()
            second = src.execute(f"CREATE TEMP TABLE curve_b AS {body}").execute(
                f"SELECT * FROM curve_b {order}").fetchall()
        assert len(first) > 0
        assert first == second

    @staticmethod
    def _spells(db, gsis_id, season):
        """Spell boundaries for one player-season, as (team, first Out week, last).

        Runs the curve's own CTEs up to `spells` and reads them back, so the test checks
        the spelling in the table definition rather than a copy of it.
        """
        body = next(b for name, _s, b in _FANTASY_TABLES if name == "injury_return_curve")
        prefix = body.split("        baseline AS (")[0].rstrip().rstrip(",")
        with duckdb.connect(str(db), read_only=True) as src:
            if not src.execute(
                "SELECT COUNT(*) FROM injuries WHERE gsis_id = ? AND season = ? "
                "AND report_status = 'Out'", [gsis_id, season],
            ).fetchone()[0]:
                pytest.skip(f"{gsis_id} has no 'Out' rows for {season} in this database")
            return src.execute(
                prefix + "\nSELECT team, first_out_week, last_out_week FROM spells "
                         "WHERE gsis_id = ? AND season = ? ORDER BY first_out_week, team",
                [gsis_id, season],
            ).fetchall()

    def test_a_trade_starts_a_new_spell(self, rebuilt_db):
        """Marshon Lattimore, 2024: Out for New Orleans week 9, then Washington 10 to 12.

        game_no numbers one club's games, and both clubs had their bye in week 9, so his
        two spells happened to share an island and merged into one that ended at week 12.
        He never went back to New Orleans, so the New Orleans spell ends at week 9.

        The pin sits at the spell level: his 2024 snap share is 0.0 every week, so the
        baseline filter (baseline_pct > 0) drops both spells out of the curve table.
        """
        assert self._spells(rebuilt_db, "00-0033874", 2024) == [
            ("NO", 9, 9), ("WAS", 10, 12), ("WAS", 17, 18),
        ]

    def test_a_traded_player_whose_out_weeks_are_already_apart_stays_apart(self, rebuilt_db):
        # Diontae Johnson, 2024: Out for Carolina week 8, traded, Out for Baltimore week 16.
        assert self._spells(rebuilt_db, "00-0035216", 2024) == [("CAR", 8, 8), ("BAL", 16, 16)]

    def test_a_player_traded_mid_season_keeps_each_club_apart(self, rebuilt_db):
        # Sauce Gardner, 2025: Out for New York week 8, then Indianapolis 14 to 16, and
        # again at week 18 because he was listed back in week 17.
        assert self._spells(rebuilt_db, "00-0037190", 2025) == [
            ("NYJ", 8, 8), ("IND", 14, 16), ("IND", 18, 18),
        ]

    def test_no_spell_carries_two_teams(self, rebuilt_db):
        """Whole-table check: an island never mixes two clubs' game sequences."""
        body = next(b for name, _s, b in _FANTASY_TABLES if name == "injury_return_curve")
        prefix = body.split("        spells AS (")[0].rstrip().rstrip(",")
        with duckdb.connect(str(rebuilt_db), read_only=True) as src:
            mixed = src.execute(
                prefix + "\nSELECT gsis_id, season, island FROM islands "
                         "GROUP BY gsis_id, season, island HAVING COUNT(DISTINCT team) > 1"
            ).fetchall()
        assert mixed == []

    def test_one_player_spell_across_a_real_bye_week(self, conn, rebuilt_db):
        """Ashton Dulin (00-0035021), 2024: Out weeks 13 and 15, Indianapolis' bye in week 14.

        His real rows go into the in-memory database and the curve is rebuilt there. The
        filter is his id, because names in these tables are abbreviated and not unique.
        The expectation comes from the same rows: one spell that ends at week 15, so
        each snap week after it is measured once against the pre-injury baseline.
        Cutting the spell at the bye (the old rule) measured those same three weeks
        against week 13 too, at offsets 3, 4 and 5.
        """
        with duckdb.connect(str(rebuilt_db), read_only=True) as src:
            injury_rows = src.execute(
                "SELECT '00-0035021' AS gsis_id, 'Ashton Dulin' AS full_name, season, "
                "       week, team, position, report_status, report_primary_injury, "
                "       practice_primary_injury "
                "FROM injuries WHERE gsis_id = '00-0035021' AND season = 2024 "
                "ORDER BY week").fetchall()
            snaps = src.execute(
                "SELECT week, offense_pct FROM snap_counts "
                "WHERE pfr_player_id = 'DuliAs00' AND season = 2024 ORDER BY week").fetchall()
            schedule = src.execute(
                "SELECT season, week, away_team, home_team FROM schedules "
                "WHERE season = 2024 AND (away_team = 'IND' OR home_team = 'IND') "
                "ORDER BY week").fetchall()
        if not injury_rows or not snaps:
            pytest.skip("2024 injury or snap rows not loaded")

        injury_columns = ["gsis_id", "full_name", "season", "week", "team", "position",
                          "report_status", "report_primary_injury",
                          "practice_primary_injury"]
        schedule_columns = ["season", "week", "away_team", "home_team"]
        _write_df_to_table(conn, "injuries", pl.DataFrame(
            [dict(zip(injury_columns, row)) for row in injury_rows]), replace=True)
        _write_df_to_table(conn, "players", pl.DataFrame({
            "gsis_id": ["00-0035021"], "pfr_id": ["DuliAs00"],
        }), replace=True)
        _write_df_to_table(conn, "snap_counts", pl.DataFrame({
            "pfr_player_id": ["DuliAs00"] * len(snaps),
            "player": ["Ashton Dulin"] * len(snaps),
            "team": ["IND"] * len(snaps),
            "season": [2024] * len(snaps),
            "week": [w for w, _ in snaps],
            "offense_pct": [p for _, p in snaps],
        }), replace=True)
        _write_df_to_table(conn, "schedules", pl.DataFrame(
            [dict(zip(schedule_columns, row)) for row in schedule]), replace=True)
        _create_fantasy_tables(conn)

        out_weeks = sorted(row[3] for row in injury_rows if row[6] == "Out")
        game_weeks = {row[1] for row in schedule}
        first_out, last_out = min(out_weeks), max(out_weeks)
        assert any(w not in game_weeks for w in range(first_out, last_out)), \
            "expected a bye week inside the spell"

        pre = [p for w, p in snaps if w < first_out]
        baseline = sum(pre) / len(pre)
        rows = _curve_rows(conn)
        expected = sorted(
            (w - last_out, 100.0 * pct / baseline)
            for w, pct in snaps if last_out < w <= last_out + 8
        )
        assert [(w, n) for w, _, n in rows] == [(w, 1) for w, _ in expected]
        for (_, got, _), (_, want) in zip(rows, expected):
            assert abs(got - want) <= 0.1          # the table rounds to one decimal
        # Each of his snap weeks back is counted once. Cut at the bye, the old rule also
        # measured the same three weeks against week 13, so 5 samples would carry 3 weeks.
        assert sum(n for _, _, n in rows) == len([1 for w, _ in snaps if w > last_out])


# ── run_ingest_datasets orchestration ──────────────────────────────────────────

class TestRunIngestOrchestration:
    class _DummyResult:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class _DummyConn:
        def __init__(self, loaded_rows):
            self.loaded_rows = loaded_rows
            self.closed = False
            self.queries = []

        def execute(self, sql):
            self.queries.append(sql)
            return TestRunIngestOrchestration._DummyResult(self.loaded_rows)

        def close(self):
            self.closed = True

    def test_run_ingest_datasets_orchestrates_pbp_and_other_datasets(self, monkeypatch, tmp_path):
        conn = self._DummyConn(loaded_rows=[(2023,)])
        calls = {"pbp": [], "record": [], "agg": 0, "idx": 0, "other": []}

        _install_fake_nflreadpy(
            monkeypatch,
            load_pbp=lambda seasons: pl.DataFrame({"season": [2024], "posteam": ["KC"], "defteam": ["BAL"]}),
        )
        monkeypatch.setattr("nfl_mcp.ingest.duckdb.connect", lambda path: conn)
        monkeypatch.setattr("nfl_mcp.ingest._ensure_metadata_table", lambda c: None)
        monkeypatch.setattr("nfl_mcp.ingest._create_plays_table", lambda c, df, fresh=False: None)
        monkeypatch.setattr("nfl_mcp.ingest._ingest_pbp_season", lambda c, season: calls["pbp"].append(season) or 11)
        monkeypatch.setattr(
            "nfl_mcp.ingest._record_loaded",
            lambda c, dataset_id, table_name, loader_fn, row_count, season=None: calls["record"].append(
                (dataset_id, table_name, loader_fn, row_count, season)
            ),
        )
        monkeypatch.setattr("nfl_mcp.ingest._create_indexes", lambda c: calls.__setitem__("idx", calls["idx"] + 1))
        monkeypatch.setattr(
            "nfl_mcp.ingest.rebuild_derived_tables",
            lambda c, p: calls.__setitem__("rebuild", calls.get("rebuild", 0) + 1),
        )
        monkeypatch.setattr(
            "nfl_mcp.ingest._ingest_generic_dataset",
            lambda c, defn, seasons, fresh=False: calls["other"].append((defn.dataset_id, seasons, fresh)) or 5,
        )

        run_ingest_datasets(
            dataset_ids=["pbp", "unknown_dataset", "schedules"],
            start=2024,
            end=2024,
            fresh=False,
            skip_views=False,
            db_path=str(tmp_path / "nflread.duckdb"),
        )

        assert calls["pbp"] == [2024]
        assert calls["record"] == [("pbp", "plays", "load_pbp", 11, 2024)]
        assert calls["idx"] == 1
        # One call to the rebuild helper, which is what stamps DERIVED_SHAPE_VERSION.
        assert calls["rebuild"] == 1
        assert calls["other"] and calls["other"][0][0] == "schedules"
        assert conn.closed is True

    def test_run_ingest_datasets_skips_pbp_when_already_loaded(self, monkeypatch, tmp_path):
        conn = self._DummyConn(loaded_rows=[(2024,)])
        pbp_calls = []
        agg_calls = []

        _install_fake_nflreadpy(
            monkeypatch,
            load_pbp=lambda seasons: pl.DataFrame({"season": [2024], "posteam": ["KC"], "defteam": ["BAL"]}),
        )
        monkeypatch.setattr("nfl_mcp.ingest.duckdb.connect", lambda path: conn)
        monkeypatch.setattr("nfl_mcp.ingest._ensure_metadata_table", lambda c: None)
        monkeypatch.setattr("nfl_mcp.ingest._create_plays_table", lambda c, df, fresh=False: None)
        monkeypatch.setattr("nfl_mcp.ingest._ingest_pbp_season", lambda c, season: pbp_calls.append(season) or 1)
        monkeypatch.setattr("nfl_mcp.ingest._create_indexes", lambda c: None)
        monkeypatch.setattr("nfl_mcp.ingest.rebuild_derived_tables",
                            lambda c, p: agg_calls.append(True))

        run_ingest_datasets(
            dataset_ids=["pbp"],
            start=2024,
            end=2024,
            fresh=False,
            skip_views=True,
            db_path=str(tmp_path / "nflread.duckdb"),
        )

        assert pbp_calls == []
        assert agg_calls == []
        assert conn.closed is True

    def test_run_ingest_datasets_without_pbp_only_runs_generic(self, monkeypatch, tmp_path):
        conn = self._DummyConn(loaded_rows=[])
        generic_calls = []

        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: pl.DataFrame({"season": [2024]}))
        monkeypatch.setattr("nfl_mcp.ingest.duckdb.connect", lambda path: conn)
        monkeypatch.setattr("nfl_mcp.ingest._ensure_metadata_table", lambda c: None)
        monkeypatch.setattr(
            "nfl_mcp.ingest._ingest_generic_dataset",
            lambda c, defn, seasons, fresh=False: generic_calls.append(defn.dataset_id) or 1,
        )
        monkeypatch.setattr("nfl_mcp.ingest._create_aggregate_tables", lambda c: None)
        monkeypatch.setattr("nfl_mcp.ingest._create_fantasy_tables", lambda c: None)
        monkeypatch.setattr("nfl_mcp.ingest._create_team_records", lambda c: None)
        monkeypatch.setattr("nfl_mcp.ingest.rebuild_derived_tables", lambda c, p: None)

        run_ingest_datasets(
            dataset_ids=["schedules"],
            start=2024,
            end=2024,
            db_path=str(tmp_path / "nflread.duckdb"),
        )
        assert generic_calls == ["schedules"]
        assert conn.closed is True

    def test_run_ingest_datasets_routes_external_datasets_to_the_records_pull(
        self, monkeypatch, tmp_path
    ):
        conn = self._DummyConn(loaded_rows=[])
        calls = []

        _install_fake_nflreadpy(monkeypatch)
        monkeypatch.setattr("nfl_mcp.ingest.duckdb.connect", lambda path: conn)
        monkeypatch.setattr("nfl_mcp.ingest._ensure_metadata_table", lambda c: None)
        monkeypatch.setattr(
            "nfl_mcp.ingest._ingest_generic_dataset",
            lambda c, defn, seasons, fresh=False: calls.append(("generic", defn.dataset_id)) or 1,
        )
        monkeypatch.setattr(
            "nfl_mcp.ingest._ingest_espn_records",
            lambda c, seasons, fresh=False: calls.append(("espn", tuple(seasons), fresh)) or 1,
        )

        run_ingest_datasets(
            dataset_ids=["espn_records"],
            start=1996,
            end=1996,
            skip_views=True,
            db_path=str(tmp_path / "nflread.duckdb"),
        )
        # An external dataset must not go through the nflreadpy loader path.
        assert calls == [("espn", (1996,), False)]
        assert conn.closed is True

    def test_run_ingest_wrapper_calls_dataset_entrypoint(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "nfl_mcp.ingest.run_ingest_datasets",
            lambda **kwargs: captured.update(kwargs),
        )
        run_ingest(start=2020, end=2021, fresh=True, skip_views=True, db_path="/tmp/custom.duckdb")
        assert captured["dataset_ids"] == ["pbp"]
        assert captured["start"] == 2020
        assert captured["end"] == 2021
        assert captured["fresh"] is True
        assert captured["skip_views"] is True
        assert captured["db_path"] == "/tmp/custom.duckdb"


# ── run_ingest_datasets validation ────────────────────────────────────────────

class TestRunIngestValidation:
    def test_raises_for_start_greater_than_end(self):
        with pytest.raises(ValueError, match="start must be less than or equal to end"):
            run_ingest_datasets(["pbp"], start=2025, end=2020)


# ── _apply_duckdb_pragmas ─────────────────────────────────────────────────────

class TestApplyDuckdbPragmas:
    def test_applies_memory_limit_threads_and_spill(self, conn, monkeypatch, tmp_path):
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "256MB")
        monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", "1")
        db_path = str(tmp_path / "sub" / "nflread.duckdb")
        _apply_duckdb_pragmas(conn, db_path)
        mem = conn.execute("SELECT current_setting('memory_limit')").fetchone()[0]
        assert "MiB" in mem or "MB" in mem
        assert conn.execute("SELECT current_setting('threads')").fetchone()[0] == 1
        spill = conn.execute("SELECT current_setting('temp_directory')").fetchone()[0]
        assert spill.endswith(".duckdb_spill")

    def test_noop_when_env_unset(self, conn, monkeypatch):
        monkeypatch.delenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", raising=False)
        monkeypatch.delenv("NFL_MCP_DUCKDB_THREADS", raising=False)
        default_threads = conn.execute("SELECT current_setting('threads')").fetchone()[0]
        _apply_duckdb_pragmas(conn, "/tmp/nflread.duckdb")
        assert conn.execute("SELECT current_setting('threads')").fetchone()[0] == default_threads


# ── Team records: ESPN ingest + derived table ─────────────────────────────────

from nfl_mcp.espn import EspnUnavailable, SeasonRecord  # noqa: E402


def _record(season, team, summary, wins, losses, ties, *, espn_id=9, abbrev="GB", game="g1"):
    return SeasonRecord(
        season=season, team=team, wins=wins, losses=losses, ties=ties,
        games=wins + losses + ties, record_summary=summary, espn_team_id=espn_id,
        espn_abbrev=abbrev, source_game_id=game, fetched_at="1997-01-01T00:00:00Z",
    )


def _schedules_table(conn, rows):
    conn.execute("""
        CREATE TABLE schedules (
            season INTEGER, game_type VARCHAR, home_team VARCHAR, away_team VARCHAR,
            home_score INTEGER, away_score INTEGER
        )
    """)
    conn.executemany("INSERT INTO schedules VALUES (?, ?, ?, ?, ?, ?)", rows)


def _team_row(conn, team, season):
    return conn.execute(
        "SELECT games, wins, losses, ties, source, source_team FROM team_records "
        "WHERE team = ? AND season = ?", [team, season],
    ).fetchone()


class TestCreateTeamRecords:
    def test_aggregates_regular_season_games(self, conn):
        _schedules_table(conn, [
            (2024, "REG", "KC", "BAL", 26, 10),
            (2024, "REG", "BAL", "KC", 14, 17),
            (2024, "POST", "KC", "BAL", 34, 7),    # playoffs stay out of the record
            (2024, "REG", "SF", "SEA", None, None),  # unfinished game is skipped
        ])
        _create_team_records(conn)
        assert _team_row(conn, "KC", 2024) == (2, 2, 0, 0, "nflverse", "KC")
        assert _team_row(conn, "BAL", 2024) == (2, 0, 2, 0, "nflverse", "BAL")
        # One row per team per season; the unfinished game contributes nothing.
        assert conn.execute("SELECT COUNT(*) FROM team_records").fetchone()[0] == 2

    def test_counts_ties(self, conn):
        _schedules_table(conn, [(1974, "REG", "DET", "CHI", 10, 10)])
        _create_team_records(conn)
        assert _team_row(conn, "DET", 1974) == (1, 0, 0, 1, "nflverse", "DET")

    def test_normalizes_relocated_team_codes(self, conn):
        _schedules_table(conn, [
            (2019, "REG", "OAK", "SD", 24, 3),
            (2016, "REG", "STL", "TB", 31, 17),
        ])
        _create_team_records(conn)
        assert _team_row(conn, "LV", 2019) == (1, 1, 0, 0, "nflverse", "OAK")
        assert _team_row(conn, "LAC", 2019) == (1, 0, 1, 0, "nflverse", "SD")
        assert _team_row(conn, "LA", 2016) == (1, 1, 0, 0, "nflverse", "STL")

    def test_unions_the_espn_years(self, conn):
        conn.execute(_ESPN_RECORDS_DDL)
        conn.execute(
            "INSERT INTO espn_team_records VALUES "
            "(1996, 'GB', 9, 'GB', 16, 13, 3, 0, '13-3', 'g9', '1997-01-01'),"
            "(1996, 'ARI', 22, 'ARI', 16, 7, 9, 0, '7-9', 'g9', '1997-01-01')"
        )
        _schedules_table(conn, [(1999, "REG", "GB", "ARI", 27, 24)])
        _create_team_records(conn)

        assert _team_row(conn, "GB", 1996) == (16, 13, 3, 0, "espn", "GB")
        assert _team_row(conn, "GB", 1999) == (1, 1, 0, 0, "nflverse", "GB")
        seasons = [r[0] for r in conn.execute("SELECT DISTINCT season FROM team_records ORDER BY season").fetchall()]
        assert seasons == [1996, 1999]

    def test_espn_only_still_builds(self, conn):
        conn.execute(_ESPN_RECORDS_DDL)
        _create_team_records(conn)
        assert conn.execute("SELECT COUNT(*) FROM team_records").fetchone()[0] == 0

    def test_skips_when_no_source_tables(self, conn, capsys):
        _create_team_records(conn)
        assert "team_records — skipped" in capsys.readouterr().out
        assert _table_exists(conn, "team_records") is False

    def test_recorded_as_derived_dataset(self, conn):
        _schedules_table(conn, [(2024, "REG", "KC", "BAL", 26, 10)])
        _create_team_records(conn)
        meta = conn.execute(
            "SELECT loader_fn, season, row_count FROM _ingest_metadata WHERE dataset_id = 'team_records'"
        ).fetchone()
        assert meta == ("derived", None, 2)

    def test_rebuild_replaces_rows(self, conn):
        _schedules_table(conn, [(2024, "REG", "KC", "BAL", 26, 10)])
        _create_team_records(conn)
        conn.execute("INSERT INTO schedules VALUES (2024, 'REG', 'DAL', 'NYG', 20, 10)")
        _create_team_records(conn)
        assert conn.execute("SELECT COUNT(*) FROM team_records").fetchone()[0] == 4


class TestIngestEspnRecords:
    def test_stores_rows_and_marks_the_season_loaded(self, conn, monkeypatch):
        calls = []

        def fake_fetch(season, **kwargs):
            calls.append(season)
            return [_record(season, "GB", "13-3", 13, 3, 0), _record(season, "NE", "11-5", 11, 5, 0)]

        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records", fake_fetch)
        assert _ingest_espn_records(conn, [1996]) == 2

        rows = conn.execute(
            "SELECT season, team, wins, losses, ties, games, source_game_id FROM espn_team_records ORDER BY team"
        ).fetchall()
        assert rows == [(1996, "GB", 13, 3, 0, 16, "g1"), (1996, "NE", 11, 5, 0, 16, "g1")]
        assert _is_loaded(conn, "espn_records", 1996) is True
        assert calls == [1996]

    def test_already_loaded_seasons_are_skipped(self, conn, monkeypatch):
        calls = []
        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records",
                            lambda season, **kw: calls.append(season) or [_record(season, "GB", "13-3", 13, 3, 0)])
        _ingest_espn_records(conn, [1996])
        _ingest_espn_records(conn, [1996])
        assert calls == [1996]

        _ingest_espn_records(conn, [1996], fresh=True)
        assert calls == [1996, 1996]
        assert conn.execute("SELECT COUNT(*) FROM espn_team_records").fetchone()[0] == 1

    def test_requests_outside_the_window_are_dropped(self, conn, monkeypatch):
        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records",
                            lambda season, **kw: pytest.fail("must not call ESPN outside 1970-1998"))
        assert _ingest_espn_records(conn, [1999, 2024]) == 0

    def test_null_seasons_uses_the_full_window(self, conn, monkeypatch):
        calls = []
        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records",
                            lambda season, **kw: calls.append(season) or [])
        _ingest_espn_records(conn, None)
        assert calls == list(range(1970, 1999))

    def test_unavailable_espn_leaves_the_season_unrecorded(self, conn, monkeypatch, capsys):
        def boom(season, **kwargs):
            raise EspnUnavailable("ESPN request failed: 503")

        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records", boom)
        assert _ingest_espn_records(conn, [1996]) == 0
        assert _is_loaded(conn, "espn_records", 1996) is False
        assert "retrying next run" in capsys.readouterr().out

    def test_repeated_failures_stop_the_run(self, conn, monkeypatch, capsys):
        """A dead endpoint must cost three seasons, not the whole backfill."""
        calls = []

        def boom(season, **kwargs):
            calls.append(season)
            raise EspnUnavailable("ESPN request failed: timeout")

        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records", boom)
        assert _ingest_espn_records(conn, [1970, 1971, 1972, 1973, 1974]) == 0
        assert calls == [1970, 1971, 1972]
        out = capsys.readouterr().out
        assert "stopping" in out and "resume" in out
        assert _is_loaded(conn, "espn_records", 1970) is False

    def test_a_success_resets_the_failure_count(self, conn, monkeypatch):
        calls = []

        def flaky(season, **kwargs):
            calls.append(season)
            if season in (1970, 1971, 1973, 1974):
                raise EspnUnavailable("ESPN request failed: 503")
            return [_record(season, "GB", "10-4", 10, 4, 0)]

        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records", flaky)
        assert _ingest_espn_records(conn, [1970, 1971, 1972, 1973, 1974]) == 1
        assert calls == [1970, 1971, 1972, 1973, 1974]
        assert _is_loaded(conn, "espn_records", 1972) is True

    def test_empty_response_leaves_the_season_unrecorded(self, conn, monkeypatch, capsys):
        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records", lambda season, **kw: [])
        assert _ingest_espn_records(conn, [1970]) == 0
        assert _is_loaded(conn, "espn_records", 1970) is False
        assert "no records published" in capsys.readouterr().out

    def test_reports_how_much_of_the_league_espn_covered(self, conn, monkeypatch, capsys):
        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records",
                            lambda season, **kw: [_record(season, "GB", "13-3", 13, 3, 0)])
        _ingest_espn_records(conn, [1996])
        out = capsys.readouterr().out
        assert "1/30 franchises" in out
        assert "29 franchises absent" in out

    def test_complete_season_reports_no_gap(self, conn, monkeypatch, capsys):
        monkeypatch.setattr(
            "nfl_mcp.espn.fetch_season_records",
            lambda season, **kw: [_record(season, "GB", "13-3", 13, 3, 0)] * 30,
        )
        _ingest_espn_records(conn, [1996])
        assert "30/30 franchises" in capsys.readouterr().out


# ── the rebuilt team tables: every number below is added up by hand in the fixture ──────────────

class TestTeamTables:
    @pytest.fixture(autouse=True)
    def built(self, conn):
        load_team_sources(conn, _write_df_to_table)
        _create_aggregate_tables(conn)

    def _offense(self, conn, team, season_type):
        got = rows(conn, "SELECT * FROM team_offense_stats WHERE team=? AND season_year=? "
                         "AND season_type=?", team, 2024, season_type)
        assert len(got) == 1
        return got[0]

    def test_counting_stats_come_from_the_team_box(self, conn):
        kc = self._offense(conn, "KC", "REG")
        # 3 attempts + 0 sacks + 2 carries, and 14 rush + 18 pass + 0 sack yards.
        assert kc["total_plays"] == 5
        assert kc["total_yards"] == 32
        assert (kc["rush_plays"], kc["pass_plays"], kc["completions"]) == (2, 3, 2)
        assert (kc["rush_yards"], kc["pass_yards"]) == (14, 18)
        # One passing touchdown and no rushing one. Row 7's pick-six is not in either.
        assert (kc["touchdowns"], kc["passing_tds"], kc["rushing_tds"]) == (1, 1, 0)
        assert (kc["turnovers"], kc["passing_interceptions"], kc["fumbles_lost"]) == (1, 1, 0)
        assert (kc["first_downs"], kc["penalties"], kc["penalty_yards"]) == (3, 3, 20)
        # The box says the defense got 1 sack; no play in this data is a sack, and 1 still wins.
        kc_def = rows(conn, "SELECT * FROM team_defense_stats WHERE team='KC' AND season_year=2024 "
                            "AND season_type='REG'")[0]
        assert (kc_def["sacks"], kc_def["interceptions"]) == (1, 0)
        bal_def = rows(conn, "SELECT * FROM team_defense_stats WHERE team='BAL' AND season_year=2024 "
                             "AND season_type='REG'")[0]
        assert (bal_def["sacks"], bal_def["interceptions"]) == (0, 1)

    def test_points_come_from_the_schedule(self, conn):
        assert (self._offense(conn, "KC", "REG")["points_for"],
                self._offense(conn, "KC", "REG")["points_against"]) == (21, 14)
        assert (self._offense(conn, "BAL", "REG")["points_for"],
                self._offense(conn, "BAL", "REG")["points_against"]) == (14, 21)
        # The wild card is a POST row, not two more regular-season points.
        bal_post = self._offense(conn, "BAL", "POST")
        assert (bal_post["points_for"], bal_post["points_against"]) == (10, 17)

    def test_per_play_figures_use_scrimmage_plays_only(self, conn):
        kc = self._offense(conn, "KC", "REG")
        # avg_epa, pass_epa, rush_epa and explosive_plays are over rows 1, 2, 3, 4 and 7: EPA -0.3
        # over 5 plays. The punt (row 5), the field goal (row 6) and the two-point try (row 8) are
        # not in any of these.
        assert kc["avg_epa"] == pytest.approx(-0.06, abs=1e-9)
        assert kc["pass_epa"] == pytest.approx(-0.233, abs=1e-9)
        assert kc["rush_epa"] == pytest.approx(0.2, abs=1e-9)
        assert kc["explosive_plays"] == 1           # row 1, a 12-yard carry

    def test_the_three_yards_per_play_rates_divide_the_box_totals_in_the_row(self, conn):
        """The row used to hold two answers for one question: a per-scrimmage-play average next to
        the box yards and plays that did not divide into it. Kansas City here ran 20 yards over 5
        scrimmage plays (4.0) while its box row says 32 yards over 5 plays (6.4)."""
        kc = self._offense(conn, "KC", "REG")
        assert (kc["total_yards"], kc["total_plays"]) == (32, 5)
        assert kc["yards_per_play"] == 6.4
        assert (kc["rush_yards"], kc["rush_plays"], kc["yards_per_rush"]) == (14, 2, 7.0)
        # 18 pass yards (net of 0 sack yards) over 3 attempts plus 0 sacks suffered.
        assert (kc["pass_yards"], kc["pass_plays"], kc["sacks_suffered"]) == (18, 3, 0)
        assert kc["yards_per_pass"] == 6.0
        # A team whose box carries no carry gets a null, not a division by zero.
        assert self._offense(conn, "BAL", "REG")["yards_per_rush"] is None

    def test_third_down_uses_the_conversion_flags(self, conn):
        kc = self._offense(conn, "KC", "REG")
        # Rows 2 and 3 converted, row 7 failed: 2 of 3, and 66.7%. Measuring yards against
        # ydstogo instead would lose row 3 (1 yard on 3rd and 2, converted by a flag) and call it
        # 1 of 3, or 33.3%.
        assert (kc["third_down_conversions"], kc["third_down_attempts"]) == (2, 3)
        assert kc["third_down_pct"] == 66.7
        bal_def = rows(conn, "SELECT * FROM team_defense_stats WHERE team='BAL' AND season_year=2024 "
                             "AND season_type='REG'")[0]
        assert (bal_def["third_down_stops"], bal_def["third_down_attempts"]) == (1, 3)
        assert bal_def["third_down_stop_pct"] == 33.3

    def test_red_zone_rate_divides_trips_and_not_plays(self, conn):
        kc = self._offense(conn, "KC", "REG")
        # Five plays inside the 20, on two drives (row 1 alone, rows 3, 4, 7 and 8 together).
        # One of those drives ended in a touchdown that the offense scored itself.
        assert kc["red_zone_trips"] == 2
        assert kc["red_zone_touchdowns"] == 1
        assert kc["red_zone_td_pct"] == 50.0

    def test_a_punt_and_a_field_goal_are_not_fourth_down_conversions(self, conn):
        got = {r["situation"]: r for r in rows(
            conn, "SELECT * FROM situational_stats WHERE team='KC' AND season_year=2024 "
                  "AND season_type='REG'")}
        # Row 4 only. The old table counted the punt and the field goal as failed conversions,
        # which turned this row into 1 of 3.
        assert got["4th Down"]["plays"] == 1
        assert (got["4th Down"]["conversions"], got["4th Down"]["conversion_attempts"]) == (1, 1)
        assert got["4th Down"]["conversion_pct"] == 100.0
        assert got["3rd & Long"]["plays"] == 2
        assert (got["3rd & Long"]["conversions"], got["3rd & Long"]["conversion_attempts"]) == (1, 2)
        assert got["3rd & Long"]["conversion_pct"] == 50.0
        assert got["3rd & Long"]["avg_yards"] == -2.0
        # Row 3 gained 1 yard on 3rd and 2 and was converted by a flag; a yardage rule would call
        # this row 0 of 1.
        assert (got["3rd & Short"]["plays"], got["3rd & Short"]["conversions"]) == (1, 1)
        assert got["3rd & Short"]["conversion_pct"] == 100.0
        # Row 7 is a touchdown on the play - intercepted and returned - so no bucket counts it.
        assert sum(r["touchdowns"] for r in got.values()) == 1
        # A 1st-down play has no third- or fourth-down attempt, so its rate is null and not 0.
        assert got["Red Zone"]["plays"] == 1
        assert got["Red Zone"]["conversion_attempts"] == 0
        assert got["Red Zone"]["conversion_pct"] is None

    def test_the_postseason_is_its_own_row(self, conn):
        # BAL has one regular-season play (6 yards) and one postseason play (12 yards). Merged, the
        # old table said 2 plays and 9.0 yards per play.
        assert self._offense(conn, "BAL", "REG")["yards_per_play"] == 6.0
        assert self._offense(conn, "BAL", "POST")["yards_per_play"] == 12.0
        assert [r["season_type"] for r in rows(
            conn, "SELECT * FROM team_offense_stats WHERE team='KC'")] == ["REG"]
        kc_def = rows(conn, "SELECT season_type, plays_against, yards_allowed, points_allowed "
                            "FROM team_defense_stats WHERE team='KC' ORDER BY season_type")
        # The wild card gave up 12 yards and scored 17, so it allowed 10; merged, this was one row.
        assert [(r["season_type"], r["plays_against"], r["yards_allowed"], r["points_allowed"])
                for r in kc_def] == [("POST", 1, 12, 10), ("REG", 1, 6, 14)]

    def test_the_defence_counts_come_from_the_opponents_box_row(self, conn):
        """Baltimore's defence is Kansas City's box row: 3 attempts + 0 sacks + 2 carries = 5 plays
        against, 18 pass + 14 rush + 0 sack = 32 yards allowed, 1 interception + 0 fumbles lost = 1
        takeaway. Counting plays with defteam = 'BAL' instead gives the same 5 plays and 19 yards,
        because plays keeps only what scrimmage plays gained and the box keeps what the league
        charged."""
        bal_def = rows(conn, "SELECT * FROM team_defense_stats WHERE team='BAL' AND season_year=2024 "
                             "AND season_type='REG'")[0]
        assert (bal_def["plays_against"], bal_def["yards_allowed"], bal_def["turnovers_forced"]) == (
            5, 32, 1)
        # The per-play figures no box carries stay on plays against this defence; the yards-per-play
        # rate now divides the two box totals above, so Baltimore is charged 32 yards over 5 plays.
        assert bal_def["yards_per_play_allowed"] == 6.4
        assert bal_def["avg_epa_allowed"] == pytest.approx(-0.06, abs=1e-9)

    def test_a_drive_whose_only_play_inside_the_20_is_a_kick_is_not_a_trip(self, conn):
        """The extra point is spotted at the 15, so yardline_100 <= 20 on its own let a kick prove a
        trip. Real data has this shape: Kansas City 2024 regular season holds six drives whose only
        play inside the 20 is the extra point - a long touchdown pass, then the kick - and an
        extra-point row carries no touchdown flag of its own (all 1,308 of them in 2024 carry
        touchdown = 0). Each of those six drove the denominator up without a touchdown, which is why
        the league rate read 47.6. Here drive d3 is that shape, so trips stay 2 and the touchdowns
        stay 1 (50.0); counting the kick as proof would make it 3 trips and 33.3."""
        conn.execute("DROP TABLE IF EXISTS plays")
        # One extra row after the ten in the fixture: an extra point for KC on its own drive.
        extra = {"posteam": "KC", "defteam": "BAL", "season": 2024, "season_type": "REG",
                 "down": None, "ydstogo": None, "yardline_100": 15, "play_type": "extra_point",
                 "pass": 0, "rush": 0, "pass_attempt": 0, "rush_attempt": 0, "yards_gained": 0,
                 "play_id": 11, "epa": 0.0, "touchdown": 0, "pass_touchdown": 0,
                 "rush_touchdown": 0, "interception": 0, "fumble_lost": 0, "sack": 0,
                 "third_down_converted": 0, "third_down_failed": 0, "fourth_down_converted": 0,
                 "fourth_down_failed": 0, "two_point_attempt": 0, "game_id": "g1", "drive": "d3",
                 "qtr": 1, "time": "10:00", "shotgun": 0, "no_huddle": 0}
        _write_df_to_table(conn, "plays", pl.DataFrame(
            {k: [*v, extra[k]] for k, v in TEAM_PLAYS.items()}), replace=True)
        _create_aggregate_tables(conn)
        kc = self._offense(conn, "KC", "REG")
        assert (kc["red_zone_trips"], kc["red_zone_touchdowns"]) == (2, 1)
        assert kc["red_zone_td_pct"] == 50.0

    def test_a_plays_only_database_keeps_the_per_play_figures(self, conn):
        load_team_sources(conn, _write_df_to_table, with_box=False, with_schedule=False)
        _create_aggregate_tables(conn)
        kc = self._offense(conn, "KC", "REG")
        # What the box and the schedule carry is null, and so are the three rates that divide the
        # box totals; what plays carries is not.
        assert (kc["total_plays"], kc["total_yards"], kc["touchdowns"], kc["points_for"]) == (
            None, None, None, None)
        assert (kc["yards_per_play"], kc["yards_per_rush"], kc["yards_per_pass"]) == (
            None, None, None)
        assert kc["avg_epa"] == pytest.approx(-0.06, abs=1e-9)
        assert kc["third_down_pct"] == 66.7
        assert kc["red_zone_td_pct"] == 50.0
        kc_def = rows(conn, "SELECT * FROM team_defense_stats WHERE team='KC' AND season_year=2024 "
                            "AND season_type='REG'")[0]
        assert (kc_def["sacks"], kc_def["interceptions"], kc_def["points_allowed"]) == (None, None, None)
        assert kc_def["plays_against"] is None
        # No box, so no rate: 6.0 was the mean of the one Baltimore play against this defence (row 9).
        assert kc_def["yards_per_play_allowed"] is None
        assert kc_def["avg_epa_allowed"] == pytest.approx(0.05, abs=1e-9)


# ── the Next Gen Stats season summary row ───────────────────────────────────────────────────────

class TestSeparationOpportunityUsesTheSeasonSummary:
    """NGS's week = 0 row is the season figure; a mean of the weekly rows is not.

    p1 has a summary row saying 2.0 yards of separation, weekly rows saying 4.0 (2 targets) and 1.0
    (8 targets). Every one of the three numbers a mean could produce is different: 2.5 unweighted,
    1.6 weighted by targets, 2.0 from the summary. p2 has no summary row, so its figure is the
    targets-weighted mean, 1.6, and the row says so.
    """

    @pytest.fixture(autouse=True)
    def built(self, conn, monkeypatch):
        # Build only the table under test: the other fantasy tables read columns of ff_opportunity
        # this fixture does not bother to write.
        monkeypatch.setattr(
            "nfl_mcp.ingest._FANTASY_TABLES",
            [t for t in _FANTASY_TABLES if t[0] == "player_separation_opportunity"],
        )
        _write_df_to_table(conn, "nextgen_stats_receiving", pl.DataFrame({
            "player_gsis_id": ["p1", "p1", "p1", "p1", "p2", "p2"],
            "season": [2024] * 6,
            "season_type": ["REG", "REG", "REG", "POST", "REG", "REG"],
            "week": [0, 1, 2, 19, 1, 2],
            "targets": [10, 2, 8, 4, 2, 8],
            "receptions": [6, 1, 5, 2, 1, 2],
            "avg_separation": [2.0, 4.0, 1.0, 3.0, 4.0, 1.0],
            "avg_yac_above_expectation": [0.5, 1.0, -1.0, 0.0, 1.0, -1.0],
            "catch_percentage": [60.0, 50.0, 62.5, 50.0, 50.0, 25.0],
        }), replace=True)
        _write_df_to_table(conn, "ff_opportunity", pl.DataFrame({
            "player_id": ["p1", "p1", "p1", "p2"],
            "season": [2024] * 4,
            "week": [1, 2, 19, 1],
            "full_name": ["P One"] * 4,
            "position": ["WR"] * 4,
            "posteam": ["KC"] * 4,
            "rec_attempt": [4, 8, 3, 2],
            "rec_attempt_team": [20, 25, 10, 5],
            "rec_air_yards": [30, 60, 20, 10],
            "rec_air_yards_team": [150, 200, 80, 40],
            "total_fantasy_points_diff": [-1.0, -2.0, 0.5, -3.0],
            "rec_touchdown_diff": [-1.0, -1.0, 0.0, 1.0],
            "receptions_diff": [0.0, -1.0, 0.0, 0.0],
        }), replace=True)
        _create_fantasy_tables(conn)

    def _rows(self, conn):
        return rows(conn, "SELECT * FROM player_separation_opportunity ORDER BY player_id, season_type")

    def test_the_summary_row_wins_over_any_mean_of_weekly_rows(self, conn):
        p1 = [r for r in self._rows(conn) if r["player_id"] == "p1"]
        reg = next(r for r in p1 if r["season_type"] == "REG")
        assert (reg["avg_separation"], reg["avg_yac_above_expectation"]) == (2.0, 0.5)
        assert reg["catch_percentage"] == 60.0
        assert reg["separation_source"] == "ngs_season_summary"

    def test_weekly_rows_are_weighted_by_targets_when_there_is_no_summary(self, conn):
        post = [r for r in self._rows(conn) if r["player_id"] == "p1" and r["season_type"] == "POST"]
        assert [(r["avg_separation"], r["separation_source"]) for r in post] == [
            (3.0, "weekly_rows_weighted_by_targets")]
        p2 = next(r for r in self._rows(conn) if r["player_id"] == "p2")
        # (4.0*2 + 1.0*8) / 10, and (50%*2 + 25%*8) / 10 - not 2.5 and not 37.5.
        assert p2["avg_separation"] == 1.6
        assert p2["catch_percentage"] == 30.0
        assert p2["separation_source"] == "weekly_rows_weighted_by_targets"

    def test_playoff_games_are_their_own_row(self, conn):
        got = {(r["player_id"], r["season_type"]): r["games"] for r in self._rows(conn)}
        assert got == {("p1", "REG"): 2, ("p1", "POST"): 1, ("p2", "REG"): 1}
        p1 = next(r for r in self._rows(conn) if r["player_id"] == "p1" and r["season_type"] == "REG")
        assert (p1["target_share_pct"], p1["air_yards_share_pct"]) == (
            round(100.0 * 12 / 45, 2), round(100.0 * 90 / 350, 2))
