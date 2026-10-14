"""Unit tests for ingest helpers — no network calls, uses in-memory DuckDB."""

import datetime
import sys
import types

import duckdb
import polars as pl
import pytest

from nfl_mcp.registry import DatasetDef
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
    _ingest_generic_dataset,
    _ingest_pbp_season,
    _is_loaded,
    _record_loaded,
    _reconcile_schema,
    _safe_rename,
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
        plays = pl.DataFrame({
            "posteam": ["KC", "KC", "KC", "KC", "KC", "BAL"],
            "defteam": ["BAL", "BAL", "BAL", "BAL", "BAL", "KC"],
            "season": [2024, 2024, 2024, 2024, 2024, 2024],
            "yards_gained": [5, 8, 12, 20, 25, 3],
            "rush_attempt": [1, 0, 1, 0, 0, 0],
            "pass_attempt": [0, 1, 0, 1, 1, 1],
            "touchdown": [0, 0, 0, 1, 0, 0],
            "interception": [0, 0, 0, 0, 0, 1],
            "fumble_lost": [0, 0, 0, 0, 0, 0],
            "down": [1, 3, 3, 4, 3, 3],
            "ydstogo": [10, 7, 2, 1, 8, 5],
            "yardline_100": [80, 45, 18, 10, 30, 40],
            "epa": [0.1, 0.2, 0.4, 1.1, 0.8, -0.6],
            "sack": [0, 0, 0, 1, 0, 0],
            "qtr": [1, 2, 4, 4, 4, 2],
            "time": ["10:00", "08:00", "01:30", "05:00", "00:45", "09:15"],
            "shotgun": [1, 1, 1, 1, 1, 0],
            "no_huddle": [0, 1, 0, 0, 1, 0],
            "play_type": ["run", "pass", "run", "pass", "pass", "pass"],
            "game_id": ["g1", "g1", "g1", "g1", "g1", "g1"],
        })
        _write_df_to_table(conn, "plays", plays, replace=True)

        _create_aggregate_tables(conn)
        created = {r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
        assert "team_offense_stats" in created
        assert "team_defense_stats" in created
        assert "situational_stats" in created
        assert "formation_effectiveness" in created

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
            "nfl_mcp.ingest._create_aggregate_tables",
            lambda c: calls.__setitem__("agg", calls["agg"] + 1),
        )
        monkeypatch.setattr(
            "nfl_mcp.ingest._create_fantasy_tables",
            lambda c: calls.__setitem__("fantasy", calls.get("fantasy", 0) + 1),
        )
        monkeypatch.setattr(
            "nfl_mcp.ingest._create_team_records",
            lambda c: calls.__setitem__("records", calls.get("records", 0) + 1),
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
        assert calls["agg"] == 1
        assert calls.get("fantasy") == 1
        assert calls.get("records") == 1
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
        monkeypatch.setattr("nfl_mcp.ingest._create_aggregate_tables", lambda c: agg_calls.append(True))

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
        monkeypatch.setattr("nfl_mcp.ingest._create_fantasy_tables", lambda c: None)
        monkeypatch.setattr("nfl_mcp.ingest._create_team_records", lambda c: None)

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
