"""Unit tests for ingest helpers — no network calls, uses in-memory DuckDB."""

import datetime
import sys
import types

import duckdb
import polars as pl
import pytest

from nfl_mcp.registry import (
    REGISTRY,
    SOURCE_ESPN,
    SOURCE_NFLREADPY,
    DatasetDef,
)
from nfl_mcp.status import IngestStatus
from nfl_mcp import ingest as ingest_module
from nfl_mcp.ingest import (
    _apply_duckdb_pragmas,
    _build_enhanced_description,
    _clip_to_window,
    _create_aggregate_tables,
    _create_fantasy_tables,
    _create_indexes,
    _create_plays_table,
    _create_team_records,
    _DETAIL_MAX_CHARS,
    _duckdb_type_for_polars,
    _ingest_espn_records,
    _ensure_metadata_table,
    _ESPN_RECORDS_DDL,
    _ingest_generic_dataset,
    _ingest_pbp_season,
    _is_loaded,
    _open_transactions,
    _record_attempt,
    _record_loaded,
    _reconcile_schema,
    _safe_rename,
    _str,
    _table_exists,
    _transaction,
    _write_df_to_table,
    run_ingest,
    run_ingest_datasets,
    _needs_refresh,
    all_seasons,
    current_season,
    season_settled_on,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def conn():
    """Fresh in-memory DuckDB connection per test."""
    c = duckdb.connect(":memory:")
    _ensure_metadata_table(c)
    yield c
    c.close()


def _outcome(conn, dataset_id, season=None):
    """Stored outcome of one dataset+season as (status, detail, row_count), or None if unrecorded."""
    if season is None:
        return conn.execute(
            "SELECT status, detail, row_count FROM _ingest_metadata "
            "WHERE dataset_id = ? AND season IS NULL",
            [dataset_id],
        ).fetchone()
    return conn.execute(
        "SELECT status, detail, row_count FROM _ingest_metadata "
        "WHERE dataset_id = ? AND season = ?",
        [dataset_id, season],
    ).fetchone()


def _throw(exc):
    """Call this where a lambda needs to raise (lambdas cannot hold a raise statement)."""
    raise exc


def _backdate_loaded_at(conn, dataset_id, season, day):
    """Move a dataset+season's stored loaded_at back to `day` (an ISO date string).

    This is how a test reproduces a season fetched in January, before its playoffs were
    played. The stamp goes in as a UTC instant, the way _record_loaded writes one, so the
    season's date means the same thing whatever zone the test machine runs in.
    """
    conn.execute(
        "UPDATE _ingest_metadata SET loaded_at = ? WHERE dataset_id = ? AND season = ?",
        [datetime.datetime.fromisoformat(f"{day}T12:00:00+00:00"), dataset_id, season],
    )


def _stored_loaded_at(conn, dataset_id, season):
    """The loaded_at on record for one dataset+season, or None when it has none."""
    row = conn.execute(
        "SELECT loaded_at FROM _ingest_metadata WHERE dataset_id = ? AND season = ?",
        [dataset_id, season],
    ).fetchone()
    return row[0] if row else None


def _pbp_df(season, play_ids) -> pl.DataFrame:
    """A play-by-play frame for one season, shaped like what load_pbp returns."""
    ids = list(play_ids)
    return pl.DataFrame({
        "season": [season] * len(ids),
        "posteam": ["KC"] * len(ids),
        "defteam": ["BAL"] * len(ids),
        "play_id": ids,
        "game_id": ["c3"] * len(ids),
        "week": list(range(1, len(ids) + 1)),
    })


def _backdate_pbp_record(path, season, day):
    """Backdate the pbp record of one season in a database file on disk."""
    c = duckdb.connect(path)
    _backdate_loaded_at(c, "pbp", season, day)
    c.close()


def _plays_rows(path):
    """Every (season, play_id) stored in a database file on disk."""
    c = duckdb.connect(path)
    rows = c.execute("SELECT season, play_id FROM plays ORDER BY season, play_id").fetchall()
    c.close()
    return rows


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
        assert {
            "dataset_id", "table_name", "season", "row_count", "loaded_at", "loader_fn",
            "status", "detail",
        } <= cols
        c.close()

    def test_migrates_a_database_created_before_the_outcome_columns(self, tmp_path):
        """An old six-column table is upgraded in place, and its rows read as 'loaded'.

        The old code wrote a row only when rows landed, so 'loaded' is true of every row it
        left behind. Nothing else can make that claim, so the migration takes the DEFAULT.
        """
        path = str(tmp_path / "nflread.duckdb")
        conn = duckdb.connect(path)
        conn.execute("""
            CREATE TABLE _ingest_metadata (
                dataset_id VARCHAR NOT NULL,
                table_name VARCHAR NOT NULL,
                season INTEGER,
                row_count BIGINT,
                loaded_at TIMESTAMP NOT NULL,
                loader_fn VARCHAR
            )
        """)
        conn.execute(
            "INSERT INTO _ingest_metadata VALUES ('pbp', 'plays', 2020, 12345, ?, 'load_pbp')",
            [datetime.datetime(2024, 1, 1)],
        )
        old_row = conn.execute("SELECT * FROM _ingest_metadata").fetchone()
        assert len(old_row) == 6

        _ensure_metadata_table(conn)

        cols = [r[1] for r in conn.execute("PRAGMA table_info('_ingest_metadata')").fetchall()]
        assert cols[-2:] == ["status", "detail"]
        assert _outcome(conn, "pbp", 2020) == ("loaded", None, 12345)
        assert conn.execute("SELECT loaded_at FROM _ingest_metadata").fetchone()[0] == \
            datetime.datetime(2024, 1, 1)          # the row was rewritten, only widened
        assert _is_loaded(conn, "pbp", 2020) is True   # it still keeps the season skipped

        _ensure_metadata_table(conn)               # a second run changes nothing
        assert conn.execute("SELECT * FROM _ingest_metadata").fetchall() == [
            old_row + ("loaded", None)
        ]
        conn.close()

    def test_migration_is_a_no_op_on_an_up_to_date_table(self):
        conn = duckdb.connect(":memory:")
        _ensure_metadata_table(conn)
        before = [r[:4] for r in conn.execute("PRAGMA table_info('_ingest_metadata')").fetchall()]
        _ensure_metadata_table(conn)
        assert [r[:4] for r in conn.execute("PRAGMA table_info('_ingest_metadata')").fetchall()] == before
        conn.close()


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

    @pytest.mark.parametrize("status,expected", [
        (IngestStatus.LOADED, True),
        (IngestStatus.PARTIAL, True),
        (IngestStatus.FAILED, False),
        (IngestStatus.EMPTY, False),
    ])
    def test_only_loaded_and_partial_make_a_season_skip(self, conn, status, expected):
        """An attempt that loaded nothing is recorded, but does not count as loaded.

        The four statuses replace "a row exists" as the skip test, so a failed or empty
        season is retried exactly as it was when it had no record at all.
        """
        if status in (IngestStatus.LOADED, IngestStatus.PARTIAL):
            _record_loaded(conn, "schedules", "schedules", "load_schedules", 10, 2024,
                           status=status,
                           detail=None if status == IngestStatus.LOADED else "28 of 30")
        else:
            _record_attempt(conn, "schedules", "schedules", "load_schedules", 2024, status, "attempted")
        assert _is_loaded(conn, "schedules", 2024) is expected

    def test_a_failed_static_attempt_does_not_count_as_loaded(self, conn):
        _record_attempt(conn, "players", "players", "load_players", None,
                        IngestStatus.FAILED, "boom")
        assert _is_loaded(conn, "players") is False

    def test_returns_false_when_metadata_table_missing(self):
        c = duckdb.connect(":memory:")
        assert _is_loaded(c, "schedules", 2024) is False
        c.close()

    def test_a_catalog_error_on_an_existing_metadata_table_is_not_swallowed(self, conn):
        """Existence is asked for. Once the answer is yes, a catalog error stays an error.

        _ingest_metadata here is a view whose base table is gone: information_schema still
        lists the name, and reading it raises CatalogException. The old code caught exactly
        that and answered "nothing loaded", so every season was fetched again on every run
        and the schema problem was never reported to anyone.
        """
        conn.execute(
            "CREATE TABLE meta_store (dataset_id VARCHAR, table_name VARCHAR, season INTEGER,"
            " row_count BIGINT, loaded_at TIMESTAMP, loader_fn VARCHAR,"
            " status VARCHAR, detail VARCHAR)"
        )
        conn.execute("DROP TABLE _ingest_metadata")
        conn.execute("CREATE VIEW _ingest_metadata AS SELECT * FROM meta_store")
        conn.execute("DROP TABLE meta_store")

        with pytest.raises(duckdb.CatalogException):
            _is_loaded(conn, "schedules", 2024)
        with pytest.raises(duckdb.CatalogException):
            _is_loaded(conn, "players")


# ── _table_exists ──────────────────────────────────────────────────────────────

class TestTableExists:
    """The one way the ingest asks whether an object is already there."""

    def test_true_for_a_base_table(self, conn):
        conn.execute("CREATE TABLE schedules (season BIGINT)")
        assert _table_exists(conn, "schedules") is True

    def test_false_for_a_name_nothing_has_created(self, conn):
        assert _table_exists(conn, "schedules") is False

    def test_the_metadata_table_counts_once_the_run_created_it(self, conn):
        assert _table_exists(conn, "_ingest_metadata") is True

    def test_a_view_answers_true_because_that_is_what_the_callers_need(self, conn):
        """Every caller used to probe with SELECT ... LIMIT 0, which reads a view just fine.

        So a view has to answer True here, or _create_plays_table would try to CREATE over a
        name that is taken and _write_df_to_table would report a clash instead of the real
        problem.
        """
        conn.execute("CREATE TABLE base (season BIGINT)")
        conn.execute("CREATE VIEW plays AS SELECT * FROM base")
        assert _table_exists(conn, "plays") is True

    def test_a_table_in_another_schema_answers_false(self, conn):
        # The ingest writes to main and only main, so a same-named table elsewhere is not
        # "already there" and must not stop it from creating its own.
        conn.execute("CREATE SCHEMA archive")
        conn.execute("CREATE TABLE archive.plays (season BIGINT)")
        assert _table_exists(conn, "plays") is False

    def test_the_name_is_a_bound_parameter_not_sql_text(self, conn):
        # A name can carry SQL punctuation without changing the query it is looked up with.
        conn.execute("CREATE TABLE schedules (season BIGINT)")
        assert _table_exists(conn, "schedules; DROP TABLE schedules; --") is False
        assert _table_exists(conn, "schedules") is True


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

    def test_defaults_to_loaded_with_no_detail(self, conn):
        """Every existing call site writes 'loaded' and NULL without saying so."""
        _record_loaded(conn, "schedules", "schedules", "load_schedules", 285, season=2024)
        assert _outcome(conn, "schedules", 2024) == ("loaded", None, 285)

    def test_writes_an_explicit_status_and_detail(self, conn):
        _record_loaded(conn, "espn_records", "espn_team_records", "espn_records", 28, season=1996,
                       status=IngestStatus.PARTIAL, detail="28 of 30 franchises")
        assert _outcome(conn, "espn_records", 1996) == ("partial", "28 of 30 franchises", 28)

    def test_detail_is_truncated(self, conn):
        _record_loaded(conn, "x", "x", "load_x", 1, season=2024,
                       status=IngestStatus.PARTIAL, detail="y" * (_DETAIL_MAX_CHARS + 600))
        assert len(conn.execute("SELECT detail FROM _ingest_metadata").fetchone()[0]) == _DETAIL_MAX_CHARS


# ── _record_attempt ────────────────────────────────────────────────────────────

class TestRecordAttempt:
    """The record of an attempt that put no rows in the table."""

    def test_writes_a_zero_row_count_with_the_reason(self, conn):
        _record_attempt(conn, "schedules", "schedules", "load_schedules", 2024,
                        IngestStatus.FAILED, "Connection reset by peer")
        assert conn.execute(
            "SELECT dataset_id, table_name, season, row_count, loader_fn, status, detail "
            "FROM _ingest_metadata"
        ).fetchone() == (
            "schedules", "schedules", 2024, 0, "load_schedules", "failed",
            "Connection reset by peer",
        )

    def test_static_attempt_is_recorded_with_a_null_season(self, conn):
        _record_attempt(conn, "players", "players", "load_players", None,
                        IngestStatus.EMPTY, "load_players returned no rows")
        assert _outcome(conn, "players") == ("empty", "load_players returned no rows", 0)

    def test_keeps_an_existing_loaded_row(self, conn):
        """Tonight's failed fetch must not relabel plays that are still in the table.

        The in-progress season is re-fetched on every run, so this is the common case: the
        old row, with its real row_count, is the one that is true.
        """
        _record_loaded(conn, "pbp", "plays", "load_pbp", 5123, season=2024)
        _record_attempt(conn, "pbp", "plays", "load_pbp", 2024, IngestStatus.FAILED, "timeout")
        assert _outcome(conn, "pbp", 2024) == ("loaded", None, 5123)

    def test_keeps_an_existing_partial_row(self, conn):
        _record_loaded(conn, "espn_records", "espn_team_records", "espn_records", 28, season=1996,
                       status=IngestStatus.PARTIAL, detail="28 of 30 franchises")
        _record_attempt(conn, "espn_records", "espn_team_records", "espn_records", 1996,
                        IngestStatus.FAILED, "ESPN request failed: 503")
        assert _outcome(conn, "espn_records", 1996) == ("partial", "28 of 30 franchises", 28)

    def test_replaces_an_older_failed_row(self, conn):
        """With no data in the table, the newest attempt is the record worth keeping."""
        _record_attempt(conn, "rosters", "rosters", "load_rosters", 2024,
                        IngestStatus.FAILED, "first: 503")
        _record_attempt(conn, "rosters", "rosters", "load_rosters", 2024,
                        IngestStatus.EMPTY, "second: no rows")
        assert conn.execute(
            "SELECT status, detail, row_count FROM _ingest_metadata"
        ).fetchall() == [("empty", "second: no rows", 0)]

    def test_detail_is_truncated(self, conn):
        _record_attempt(conn, "rosters", "rosters", "load_rosters", 2024, IngestStatus.FAILED,
                        "x" * (_DETAIL_MAX_CHARS + 600))
        assert len(_outcome(conn, "rosters", 2024)[1]) == _DETAIL_MAX_CHARS

    def test_attempt_row_rolls_back_with_the_callers_transaction(self, conn):
        conn.execute("CREATE TABLE t (a INTEGER)")
        with pytest.raises(RuntimeError, match="boom"):
            with _transaction(conn):
                conn.execute("INSERT INTO t VALUES (1)")
                _record_attempt(conn, "schedules", "schedules", "load_schedules", 2024,
                                IngestStatus.FAILED, "boom")
                raise RuntimeError("boom")
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM _ingest_metadata").fetchone()[0] == 0


# ── _transaction ───────────────────────────────────────────────────────────────

class TestTransaction:
    """Rows and their _ingest_metadata row change together, or not at all."""

    def test_commits_on_success(self, tmp_path):
        path = str(tmp_path / "nflread.duckdb")
        c = duckdb.connect(path)
        _ensure_metadata_table(c)
        with _transaction(c):
            c.execute("CREATE TABLE t (a INTEGER)")
            c.execute("INSERT INTO t VALUES (1)")
        c.close()

        # Reopened from disk, so this only passes if the COMMIT ran.
        check = duckdb.connect(path)
        assert check.execute("SELECT a FROM t").fetchall() == [(1,)]
        check.close()

    def test_rolls_back_and_reraises_on_error(self, tmp_path):
        path = str(tmp_path / "nflread.duckdb")
        c = duckdb.connect(path)
        _ensure_metadata_table(c)
        c.execute("CREATE TABLE t (a INTEGER)")
        c.execute("INSERT INTO t VALUES (1)")
        c.close()

        c = duckdb.connect(path)
        with pytest.raises(RuntimeError, match="boom"):
            with _transaction(c):
                c.execute("DELETE FROM t")
                c.execute("INSERT INTO t VALUES (2)")
                raise RuntimeError("boom")

        assert c not in _open_transactions     # the owner released the connection
        c.execute("BEGIN")                     # nothing was left open: BEGIN is accepted
        c.execute("ROLLBACK")
        c.close()

        check = duckdb.connect(path)
        assert check.execute("SELECT a FROM t").fetchall() == [(1,)]   # delete undone, insert dropped
        check.close()

    def test_nested_transaction_defers_to_the_outer_one(self, conn):
        conn.execute("CREATE TABLE t (a INTEGER)")
        with _transaction(conn):
            conn.execute("INSERT INTO t VALUES (1)")
            with _transaction(conn):           # a second BEGIN here would raise
                conn.execute("INSERT INTO t VALUES (2)")
        assert conn.execute("SELECT a FROM t ORDER BY a").fetchall() == [(1,), (2,)]

        with pytest.raises(RuntimeError, match="boom"):
            with _transaction(conn):
                with _transaction(conn):
                    conn.execute("INSERT INTO t VALUES (3)")
                raise RuntimeError("boom")
        # The nested block cannot commit on its own, so the outer rollback undid its row.
        assert conn.execute("SELECT a FROM t ORDER BY a").fetchall() == [(1,), (2,)]


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

    def test_a_schema_problem_on_an_existing_table_is_not_answered_with_a_create(
        self, conn, monkeypatch
    ):
        """The old writer caught CatalogException and answered it by CREATE-ing the table.

        That exception is also what a missing column or a missing function raises, so a real
        schema problem ended as a "table already exists" error from a CREATE that could not
        run. The error now travels with its own message, and the rows stay.
        """
        conn.execute("CREATE TABLE test_tbl (season BIGINT)")
        conn.execute("INSERT INTO test_tbl VALUES (2024)")

        def missing_column(_conn, _table, _df):
            raise duckdb.CatalogException("Catalog Error: no such column epa")

        monkeypatch.setattr("nfl_mcp.ingest._reconcile_schema", missing_column)
        with pytest.raises(duckdb.CatalogException, match="no such column epa"):
            _write_df_to_table(conn, "test_tbl", pl.DataFrame({"season": [2025]}))

        assert conn.execute("SELECT season FROM test_tbl").fetchall() == [(2024,)]
        cols = [r[1] for r in conn.execute("PRAGMA table_info('test_tbl')").fetchall()]
        assert cols == ["season"]                      # same table, not a rebuilt one

    def test_a_view_that_cannot_take_the_insert_says_so_instead_of_clashing(self, conn):
        # Inserting into a view is the CatalogException the old fallback swallowed, then
        # repeated as a CREATE that could not run. The row lands in the base table or nowhere.
        conn.execute("CREATE TABLE base (season BIGINT)")
        conn.execute("INSERT INTO base VALUES (2024)")
        conn.execute("CREATE VIEW test_tbl AS SELECT * FROM base")
        with pytest.raises(duckdb.CatalogException, match="not an table"):
            _write_df_to_table(conn, "test_tbl", pl.DataFrame({"season": [2025]}))
        assert conn.execute("SELECT season FROM base").fetchall() == [(2024,)]


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
        _create_plays_table(conn, sample)
        cols = {r[1] for r in conn.execute("PRAGMA table_info('plays')").fetchall()}
        assert "a_b" in cols
        assert "team_name" in cols
        assert "enhanced_description" in cols

    def test_create_plays_table_keeps_an_existing_table_and_its_rows(self, conn):
        """Replaces the test that pinned the old whole-table DROP on --fresh.

        _create_plays_table used to run DROP TABLE IF EXISTS plays when fresh was true,
        which deleted seasons the run never asked for. It now only creates the table.
        """
        conn.execute("CREATE TABLE plays (season BIGINT)")
        conn.execute("INSERT INTO plays VALUES (2023)")
        sample = pl.DataFrame({"season": [2024], "posteam": ["KC"], "defteam": ["BAL"]})
        _create_plays_table(conn, sample)
        assert conn.execute("SELECT season FROM plays").fetchall() == [(2023,)]
        cols = {r[1] for r in conn.execute("PRAGMA table_info('plays')").fetchall()}
        assert cols == {"season"}

    def test_create_plays_table_when_already_exists_and_not_fresh(self, conn):
        conn.execute("CREATE TABLE plays (season BIGINT)")
        sample = pl.DataFrame({"season": [2024]})
        _create_plays_table(conn, sample)
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

        _create_plays_table(conn, df)
        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: df)

        inserted = _ingest_pbp_season(conn, 2024)
        assert inserted == 1
        row = conn.execute("SELECT posteam, enhanced_description FROM plays").fetchone()
        assert row[0] == "KC"
        assert "TOUCHDOWN" in row[1]

    def test_ingest_pbp_season_returns_zero_on_loader_error(self, conn, monkeypatch):
        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: (_ for _ in ()).throw(RuntimeError("boom")))
        assert _ingest_pbp_season(conn, 2024) == 0

    def test_ingest_pbp_season_keeps_rows_when_filtering_leaves_none(self, conn, monkeypatch):
        """A fetch that filters down to nothing must not delete the loaded season."""
        seed = pl.DataFrame({
            "season": [2024], "posteam": ["KC"], "defteam": ["BAL"],
            "play_id": [1], "game_id": ["g1"], "week": [1],
        })
        _create_plays_table(conn, seed)
        conn.register("_seed", seed.with_columns(pl.lit("").alias("enhanced_description")).to_arrow())
        conn.execute("INSERT INTO plays BY NAME SELECT * FROM _seed")
        conn.unregister("_seed")
        _record_loaded(conn, "pbp", "plays", "load_pbp", 1, season=2024)

        # The fetch succeeds, then the null posteam/defteam filter drops every row.
        empty = seed.filter(pl.col("season") > 3000)
        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: empty)

        assert _ingest_pbp_season(conn, 2024) == 0
        assert conn.execute("SELECT COUNT(*) FROM plays").fetchone()[0] == 1
        assert _is_loaded(conn, "pbp", 2024) is True     # rows and its record still agree

    def test_ingest_pbp_season_rolls_back_the_season_when_the_insert_fails(self, conn, monkeypatch):
        seed = pl.DataFrame({
            "season": [2024], "posteam": ["KC"], "defteam": ["BAL"],
            "play_id": [1], "game_id": ["g1"], "week": [1],
        })
        _create_plays_table(conn, seed)
        conn.register("_seed", seed.with_columns(pl.lit("").alias("enhanced_description")).to_arrow())
        conn.execute("INSERT INTO plays BY NAME SELECT * FROM _seed")
        conn.unregister("_seed")

        # Same column names, but text in the BIGINT season column, so the INSERT raises
        # after the DELETE for the season has already run.
        bad = pl.DataFrame({
            "season": ["not-a-season"], "posteam": ["KC"], "defteam": ["BAL"],
            "play_id": [1], "game_id": ["g1"], "week": [1],
        })
        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: bad)

        with pytest.raises(duckdb.ConversionException):
            _ingest_pbp_season(conn, 2024)

        assert conn not in _open_transactions
        rows = conn.execute("SELECT season, posteam FROM plays").fetchall()
        assert rows == [(2024, "KC")]          # the season is neither empty nor duplicated


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

    def test_seasonal_dataset_skips_loaded_seasons_and_records_the_other_attempts(self, conn, monkeypatch):
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

        calls = []
        def loader(seasons, **kwargs):
            season = seasons[0]
            calls.append(season)
            if season == 2021:
                raise RuntimeError("transient loader failure")
            if season == 2023:
                return pl.DataFrame({"season": [], "value": []})
            return pl.DataFrame({"season": [season], "value": [season]})

        _install_fake_nflreadpy(monkeypatch, load_fake_seasonal=loader)
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2030)
        seasons = [2019, 2020, 2021, 2023, 2024, 2025]
        total = _ingest_generic_dataset(conn, defn, seasons=seasons, fresh=False)
        assert total == 1
        loaded_2024 = _is_loaded(conn, "fake_seasonal", 2024)
        loaded_2020 = _is_loaded(conn, "fake_seasonal", 2020)
        assert loaded_2024 is True
        assert loaded_2020 is True
        # 2019 and 2025 clipped out of the dataset's own window, so they were never fetched.
        assert calls == [2021, 2023, 2024]

        # Every attempt is now recorded, including the two that loaded no data.
        assert _outcome(conn, "fake_seasonal", 2021) == ("failed", "transient loader failure", 0)
        assert _outcome(conn, "fake_seasonal", 2023) == (
            "empty", "load_fake_seasonal returned no rows", 0,
        )
        assert _is_loaded(conn, "fake_seasonal", 2021) is False
        assert _is_loaded(conn, "fake_seasonal", 2023) is False

        # Skipped behavior is unchanged: the loaded seasons stay skipped, the two that
        # loaded nothing are fetched again.
        calls.clear()
        assert _ingest_generic_dataset(conn, defn, seasons=seasons, fresh=False) == 0
        assert calls == [2021, 2023]

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

    def test_seasonal_insert_failure_keeps_old_rows_and_metadata(self, conn, monkeypatch):
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
        _install_fake_nflreadpy(
            monkeypatch,
            load_fake_seasonal=lambda seasons, **kwargs: pl.DataFrame({"season": [2024], "value": [99]}),
        )

        def failing_write(_conn, _table, _df, replace=False):
            raise RuntimeError("insert failed")

        # The season's rows were already deleted inside the transaction by this point.
        monkeypatch.setattr("nfl_mcp.ingest._write_df_to_table", failing_write)
        with pytest.raises(RuntimeError, match="insert failed"):
            _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=True)

        rows = conn.execute(
            "SELECT season, value FROM fake_seasonal ORDER BY season, value"
        ).fetchall()
        assert rows == [(2023, 9), (2024, 1)]        # rollback brought the old rows back
        meta = conn.execute(
            "SELECT row_count FROM _ingest_metadata WHERE dataset_id = 'fake_seasonal' AND season = 2024"
        ).fetchall()
        assert meta == [(1,)]                        # still recorded, with its old count

    def test_static_insert_failure_keeps_old_table_and_metadata(self, conn, monkeypatch):
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
            load_fake_static=lambda **kwargs: pl.DataFrame({"x": [99]}),
        )

        def drop_then_fail(_conn, _table, _df, replace=False):
            _conn.execute("DROP TABLE fake_static")   # the DDL the real writer runs first
            raise RuntimeError("insert failed")

        monkeypatch.setattr("nfl_mcp.ingest._write_df_to_table", drop_then_fail)
        with pytest.raises(RuntimeError, match="insert failed"):
            _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=False)

        # DuckDB rolls back DDL too, so the dropped table is back with its old rows.
        assert conn.execute("SELECT x FROM fake_static").fetchall() == [(1,)]
        meta = conn.execute(
            "SELECT row_count FROM _ingest_metadata WHERE dataset_id = 'fake_static'"
        ).fetchall()
        assert meta == [(1,)]

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


# ── every ingest attempt leaves a record ───────────────────────────────────

class TestAttemptRecordsAtCallSites:
    """Each outcome the ingest code can reach is written to _ingest_metadata.

    The printed message is not the record. A query tool reads the table, so the status and
    the reason have to be in it.
    """

    @staticmethod
    def _seasonal_defn():
        return DatasetDef(dataset_id="fake_seasonal", loader_fn="load_fake_seasonal",
                          table_name="fake_seasonal", seasonal=True, default=False)

    @staticmethod
    def _static_defn():
        return DatasetDef(dataset_id="fake_static", loader_fn="load_fake_static",
                          table_name="fake_static", seasonal=False, default=False)

    def test_seasonal_fetch_that_raises_is_recorded_failed(self, conn, monkeypatch):
        defn = self._seasonal_defn()
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2030)
        _install_fake_nflreadpy(
            monkeypatch,
            load_fake_seasonal=lambda seasons, **kw: _throw(RuntimeError("HTTP 502")),
        )
        assert _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=False) == 0
        assert _outcome(conn, "fake_seasonal", 2024) == ("failed", "HTTP 502", 0)

    def test_seasonal_fetch_with_no_rows_is_recorded_empty(self, conn, monkeypatch):
        defn = self._seasonal_defn()
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2030)
        _install_fake_nflreadpy(
            monkeypatch,
            load_fake_seasonal=lambda seasons, **kw: pl.DataFrame({"season": [], "value": []}),
        )
        assert _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=False) == 0
        assert _outcome(conn, "fake_seasonal", 2024) == (
            "empty", "load_fake_seasonal returned no rows", 0,
        )

    def test_failed_season_is_fetched_again_and_recorded_loaded(self, conn, monkeypatch):
        """The reason for the whole change: a gap says failed now, and is not frozen that way."""
        defn = self._seasonal_defn()
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2030)
        calls = []

        def loader(seasons, **kwargs):
            calls.append(seasons[0])
            if len(calls) == 1:
                raise RuntimeError("connection reset")
            return pl.DataFrame({"season": [2024, 2024, 2024], "value": [1, 2, 3]})

        _install_fake_nflreadpy(monkeypatch, load_fake_seasonal=loader)

        assert _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=False) == 0
        assert _outcome(conn, "fake_seasonal", 2024) == ("failed", "connection reset", 0)

        assert _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=False) == 3
        assert calls == [2024, 2024]        # retried, not skipped
        assert _outcome(conn, "fake_seasonal", 2024) == ("loaded", None, 3)
        assert _is_loaded(conn, "fake_seasonal", 2024) is True

    def test_static_fetch_that_raises_is_recorded_failed(self, conn, monkeypatch):
        defn = self._static_defn()
        _install_fake_nflreadpy(
            monkeypatch, load_fake_static=lambda **kw: _throw(RuntimeError("HTTP 502")),
        )
        assert _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=False) == 0
        assert _outcome(conn, "fake_static") == ("failed", "HTTP 502", 0)

    def test_static_fetch_with_no_rows_is_recorded_empty(self, conn, monkeypatch):
        defn = self._static_defn()
        _install_fake_nflreadpy(monkeypatch, load_fake_static=lambda **kw: pl.DataFrame({"x": []}))
        assert _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=False) == 0
        assert _outcome(conn, "fake_static") == ("empty", "load_fake_static returned no rows", 0)

    def test_static_refresh_that_fails_keeps_the_loaded_row(self, conn, monkeypatch):
        """Static tables are re-fetched on every run, so a bad night must not rewrite the record."""
        defn = self._static_defn()
        conn.execute("CREATE TABLE fake_static (x BIGINT)")
        conn.execute("INSERT INTO fake_static VALUES (1)")
        _record_loaded(conn, "fake_static", "fake_static", "load_fake_static", 1)
        _install_fake_nflreadpy(
            monkeypatch, load_fake_static=lambda **kw: _throw(RuntimeError("HTTP 502")),
        )
        assert _ingest_generic_dataset(conn, defn, seasons=[2024], fresh=False) == 0
        assert _outcome(conn, "fake_static") == ("loaded", None, 1)
        assert _is_loaded(conn, "fake_static") is True

    def test_pbp_fetch_that_raises_is_recorded_failed(self, conn, monkeypatch):
        _install_fake_nflreadpy(
            monkeypatch, load_pbp=lambda seasons: _throw(RuntimeError("connection reset")),
        )
        assert _ingest_pbp_season(conn, 2024) == 0
        assert _outcome(conn, "pbp", 2024) == ("failed", "connection reset", 0)

    def test_pbp_season_with_no_rows_left_is_recorded_empty(self, conn, monkeypatch):
        _install_fake_nflreadpy(
            monkeypatch, load_pbp=lambda seasons: pl.DataFrame({"posteam": [], "defteam": []}),
        )
        assert _ingest_pbp_season(conn, 2024) == 0
        status, detail, row_count = _outcome(conn, "pbp", 2024)
        assert (status, row_count) == ("empty", 0)
        assert "no rows to insert" in detail

    def test_pbp_attempt_leaves_a_stored_season_recorded_as_loaded(self, conn, monkeypatch):
        """The in-progress season is re-fetched every run; one failure must not call it a gap."""
        seed = pl.DataFrame({
            "season": [2024], "posteam": ["KC"], "defteam": ["BAL"],
            "play_id": [1], "game_id": ["g1"], "week": [1],
        })
        _create_plays_table(conn, seed)
        conn.register("_seed", seed.with_columns(pl.lit("").alias("enhanced_description")).to_arrow())
        conn.execute("INSERT INTO plays BY NAME SELECT * FROM _seed")
        conn.unregister("_seed")
        _record_loaded(conn, "pbp", "plays", "load_pbp", 1, season=2024)
        _install_fake_nflreadpy(
            monkeypatch, load_pbp=lambda seasons: _throw(RuntimeError("connection reset")),
        )

        assert _ingest_pbp_season(conn, 2024) == 0
        assert _outcome(conn, "pbp", 2024) == ("loaded", None, 1)
        assert conn.execute("SELECT COUNT(*) FROM plays").fetchone()[0] == 1

    def test_derived_tables_stay_recorded_as_loaded(self, conn):
        """A derived table is built from data already here, so it has no failure mode to record."""
        _schedules_table(conn, [(2024, "REG", "KC", "BAL", 26, 10)])
        _create_team_records(conn)
        assert _outcome(conn, "team_records") == ("loaded", None, 2)


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
        monkeypatch.setattr("nfl_mcp.ingest._create_plays_table", lambda c, df: None)
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
        monkeypatch.setattr("nfl_mcp.ingest._create_plays_table", lambda c, df: None)
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

    def test_all_seasons_ends_at_the_current_season(self):
        assert all_seasons()[-1] == current_season()

    def test_season_settled_on_is_march_first_of_the_next_year(self):
        assert season_settled_on(2023) == datetime.date(2024, 3, 1)
        assert season_settled_on(2024) == datetime.date(2025, 3, 1)

    @pytest.mark.parametrize("season", [2023, 2024])
    def test_a_season_stops_being_current_the_day_it_settles(self, season):
        """The two helpers share one boundary: the day a season ends is the day it stops
        being the current one, so a season is never both skipped and still in progress."""
        settled = season_settled_on(season)
        assert current_season(settled - datetime.timedelta(days=1)) == season
        assert current_season(settled) == season + 1

    def test_the_boundary_clears_february_in_a_leap_year(self):
        # 2023 ends on 1 March 2024; 29 February 2024 is still inside the 2023 season.
        assert current_season(datetime.date(2024, 2, 28)) == 2023
        assert current_season(datetime.date(2024, 2, 29)) == 2023
        assert current_season(datetime.date(2024, 3, 1)) == 2024


# ── an already-loaded season is not a finished season ──────────────────────────

class TestNeedsRefresh:
    """The one rule both ingest paths ask: must this loaded season be fetched again?"""

    @staticmethod
    def _record(conn, season, day, status=IngestStatus.LOADED):
        """Record season as loaded, stamped on `day` (None leaves the stamp at now)."""
        _record_loaded(conn, "fake", "fake_table", "load_fake", 1, season=season, status=status)
        if day:
            _backdate_loaded_at(conn, "fake", season, day)

    def test_the_season_in_progress_always_refreshes(self, conn, monkeypatch):
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2026)
        self._record(conn, 2026, None)      # loaded moments ago, still gaining games
        assert _needs_refresh(conn, "fake", 2026) is True

    def test_season_loaded_before_it_ended_refreshes(self, conn, monkeypatch):
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2026)
        self._record(conn, 2023, "2024-01-15")   # 2023 ended on 2024-03-01
        assert _needs_refresh(conn, "fake", 2023) is True

    def test_season_loaded_on_or_after_the_day_it_ended_does_not(self, conn, monkeypatch):
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2026)
        self._record(conn, 2023, "2024-03-01")   # the settle day itself: nothing left to fetch
        assert _needs_refresh(conn, "fake", 2023) is False
        self._record(conn, 2024, "2025-06-01")
        assert _needs_refresh(conn, "fake", 2024) is False

    def test_a_partial_season_is_judged_by_its_date_too(self, conn, monkeypatch):
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2026)
        self._record(conn, 1997, "1998-01-15", status=IngestStatus.PARTIAL)
        assert _needs_refresh(conn, "fake", 1997) is True

    def test_a_season_with_no_loaded_record_is_not_refreshed_here(self, conn, monkeypatch):
        """'Never tried' and 'the last attempt failed' belong to the callers, not to this rule."""
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2026)
        _record_attempt(conn, "fake", "fake_table", "load_fake", 2023,
                        IngestStatus.FAILED, "boom")
        assert _needs_refresh(conn, "fake", 2023) is False
        assert _needs_refresh(conn, "fake", 2022) is False     # no row at all


class TestStaleSeasonRefetch:
    """A season fetched in January is missing its playoffs, so it is fetched once more."""

    @staticmethod
    def _defn():
        return DatasetDef(dataset_id="fake_seasonal", loader_fn="load_fake_seasonal",
                          table_name="fake_seasonal", seasonal=True, default=False)

    @staticmethod
    def _seed(conn):
        """2023 loaded on 15 January 2024, before the season ended on 1 March."""
        conn.execute("CREATE TABLE fake_seasonal (season BIGINT, value BIGINT)")
        conn.execute("INSERT INTO fake_seasonal VALUES (2023, 9)")
        _record_loaded(conn, "fake_seasonal", "fake_seasonal", "load_fake_seasonal", 1,
                       season=2023)
        _backdate_loaded_at(conn, "fake_seasonal", 2023, "2024-01-15")

    @staticmethod
    def _rows(conn):
        return conn.execute(
            "SELECT season, value FROM fake_seasonal ORDER BY season, value"
        ).fetchall()

    def _season_loader(self, monkeypatch, fetched, *frames):
        """Install a loader that logs each season it is called for, returning `frames` in order."""
        def loader(seasons, **kwargs):
            fetched.append(seasons[0])
            return frames[len(fetched) - 1] if len(fetched) <= len(frames) else frames[-1]
        _install_fake_nflreadpy(monkeypatch, load_fake_seasonal=loader)

    def test_past_season_loaded_before_it_ended_is_refetched_once(self, conn, monkeypatch):
        defn = self._defn()
        self._seed(conn)
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2026)
        fetched = []
        self._season_loader(monkeypatch, fetched,
                            pl.DataFrame({"season": [2023, 2023], "value": [1, 2]}))

        assert _ingest_generic_dataset(conn, defn, seasons=[2023], fresh=False) == 2
        assert fetched == [2023]
        assert self._rows(conn) == [(2023, 1), (2023, 2)]        # replaced, not appended
        assert _outcome(conn, "fake_seasonal", 2023) == ("loaded", None, 2)

        # That fetch wrote a new loaded_at, so the season is settled now and is skipped.
        fetched.clear()
        assert _ingest_generic_dataset(conn, defn, seasons=[2023], fresh=False) == 0
        assert fetched == []
        assert self._rows(conn) == [(2023, 1), (2023, 2)]

    def test_past_season_loaded_after_it_ended_is_skipped(self, conn, monkeypatch):
        """Today's behavior, kept: a load from after the season's last game is complete."""
        defn = self._defn()
        conn.execute("CREATE TABLE fake_seasonal (season BIGINT, value BIGINT)")
        conn.execute("INSERT INTO fake_seasonal VALUES (2023, 9)")
        _record_loaded(conn, "fake_seasonal", "fake_seasonal", "load_fake_seasonal", 1,
                       season=2023)
        _backdate_loaded_at(conn, "fake_seasonal", 2023, "2024-03-15")
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2026)
        fetched = []
        self._season_loader(monkeypatch, fetched,
                            pl.DataFrame({"season": [2023], "value": [99]}))

        assert _ingest_generic_dataset(conn, defn, seasons=[2023], fresh=False) == 0
        assert fetched == []
        assert self._rows(conn) == [(2023, 9)]

    def test_refetch_that_fails_keeps_rows_and_record_and_is_tried_again(self, conn, monkeypatch):
        """One bad night must not cost the season: the old rows and the old stamp survive,
        and an early stamp is what asks the question again on the next run."""
        defn = self._defn()
        self._seed(conn)
        stamp = _stored_loaded_at(conn, "fake_seasonal", 2023)
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2026)
        _install_fake_nflreadpy(
            monkeypatch,
            load_fake_seasonal=lambda seasons, **kw: _throw(RuntimeError("connection reset")),
        )

        assert _ingest_generic_dataset(conn, defn, seasons=[2023], fresh=False) == 0
        assert self._rows(conn) == [(2023, 9)]
        assert _outcome(conn, "fake_seasonal", 2023) == ("loaded", None, 1)
        assert _stored_loaded_at(conn, "fake_seasonal", 2023) == stamp

        fetched = []
        self._season_loader(monkeypatch, fetched,
                            pl.DataFrame({"season": [2023, 2023], "value": [1, 2]}))
        assert _ingest_generic_dataset(conn, defn, seasons=[2023], fresh=False) == 2
        assert fetched == [2023]
        assert self._rows(conn) == [(2023, 1), (2023, 2)]


class TestStalePbpSeasonRefetch:
    """Play-by-play reads the plays table to decide 'loaded'; the season's end date now
    decides whether that copy is finished."""

    @staticmethod
    def _seed(path, season=2023, play_ids=(1,)):
        TestRunIngestOrchestration._seed_plays_db(path, _pbp_df(season, list(play_ids)))

    @staticmethod
    def _records(path):
        return TestRunIngestOrchestration._pbp_records(path)

    def test_pbp_season_loaded_before_it_ended_is_refetched_once(self, monkeypatch, tmp_path):
        path = str(tmp_path / "nflread.duckdb")
        self._seed(path)
        _backdate_pbp_record(path, 2023, "2024-01-15")
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2026)
        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: _pbp_df(2023, [7, 8]))

        run_ingest_datasets(["pbp"], start=2023, end=2023, skip_views=True, db_path=path)

        assert _plays_rows(path) == [(2023, 7), (2023, 8)]
        assert [(s, c) for s, c, _ in self._records(path)] == [(2023, 2)]

        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: _pbp_df(2023, [99]))
        run_ingest_datasets(["pbp"], start=2023, end=2023, skip_views=True, db_path=path)
        assert _plays_rows(path) == [(2023, 7), (2023, 8)]
        assert [(s, c) for s, c, _ in self._records(path)] == [(2023, 2)]

    def test_pbp_refetch_of_a_settled_season_says_why(self, monkeypatch, tmp_path, capsys):
        path = str(tmp_path / "nflread.duckdb")
        self._seed(path)
        _backdate_pbp_record(path, 2023, "2024-01-15")
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2026)
        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: _pbp_df(2023, [7]))

        run_ingest_datasets(["pbp"], start=2023, end=2023, skip_views=True, db_path=path)

        out = capsys.readouterr().out
        assert "Last loaded before the season ended, re-fetching: [2023]" in out
        assert "Current season, re-fetching" not in out

    def test_pbp_season_loaded_after_it_ended_is_skipped(self, monkeypatch, tmp_path, capsys):
        path = str(tmp_path / "nflread.duckdb")
        self._seed(path)
        _backdate_pbp_record(path, 2023, "2024-03-15")
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2026)
        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: _pbp_df(2023, [99]))

        run_ingest_datasets(["pbp"], start=2023, end=2023, skip_views=True, db_path=path)

        assert _plays_rows(path) == [(2023, 1)]
        assert "Already loaded, skipping: [2023]" in capsys.readouterr().out

    def test_pbp_refetch_that_fails_keeps_rows_and_record_and_is_tried_again(
        self, monkeypatch, tmp_path
    ):
        path = str(tmp_path / "nflread.duckdb")
        self._seed(path)
        _backdate_pbp_record(path, 2023, "2024-01-15")
        before = self._records(path)
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2026)

        asked = []

        def load_pbp(seasons):
            asked.append(seasons[0])
            if len(asked) > 1:
                # The schema-discovery pull above succeeded; this season's pull fails.
                raise RuntimeError("connection reset by peer")
            return _pbp_df(2023, [1])

        _install_fake_nflreadpy(monkeypatch, load_pbp=load_pbp)
        run_ingest_datasets(["pbp"], start=2023, end=2023, skip_views=True, db_path=path)

        assert _plays_rows(path) == [(2023, 1)]
        assert self._records(path) == before      # still loaded, still stamped in January

        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: _pbp_df(2023, [7, 8]))
        run_ingest_datasets(["pbp"], start=2023, end=2023, skip_views=True, db_path=path)
        assert _plays_rows(path) == [(2023, 7), (2023, 8)]

    def test_pbp_season_in_plays_with_no_record_is_skipped(self, monkeypatch, tmp_path):
        """A database written before _ingest_metadata existed must not reload itself whole."""
        path = str(tmp_path / "nflread.duckdb")
        self._seed(path)
        c = duckdb.connect(path)
        c.execute("DELETE FROM _ingest_metadata WHERE dataset_id = 'pbp'")
        c.close()
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2026)
        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: _pbp_df(2023, [99]))

        run_ingest_datasets(["pbp"], start=2023, end=2023, skip_views=True, db_path=path)

        assert _plays_rows(path) == [(2023, 1)]
        assert self._records(path) == []


class TestSeasonListResolvedAtCallTime:
    """The default backfill window is read when an ingest runs, never at import time.

    The server is a long-lived process: a list frozen at import would still name last
    season after 1 March, and the two ingest paths would disagree with each other.
    """

    def test_ingest_module_holds_no_import_time_season_list(self):
        assert not hasattr(ingest_module, "ALL_SEASONS")

    def test_generic_bulk_mode_uses_the_window_of_the_run(self, conn, monkeypatch):
        defn = DatasetDef(dataset_id="fake_seasonal", loader_fn="load_fake_seasonal",
                          table_name="fake_seasonal", seasonal=True, default=False,
                          min_season=1990, max_season=2030)
        fetched = []

        def loader(seasons, **kwargs):
            fetched.append(seasons[0])
            return pl.DataFrame({"season": list(seasons), "value": list(seasons)})

        _install_fake_nflreadpy(monkeypatch, load_fake_seasonal=loader)
        monkeypatch.setattr("nfl_mcp.ingest.all_seasons", lambda: [2022, 2023])
        assert _ingest_generic_dataset(conn, defn, seasons=None, fresh=False) == 2
        assert fetched == [2022, 2023]

        # A later run in the same process, after the league year rolled over.
        fetched.clear()
        monkeypatch.setattr("nfl_mcp.ingest.all_seasons", lambda: [2022, 2023, 2024])
        assert _ingest_generic_dataset(conn, defn, seasons=None, fresh=False) == 1
        assert fetched == [2024]

    def test_espn_pull_uses_the_window_of_the_run(self, conn, monkeypatch):
        calls = []
        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records",
                            lambda season, **kw: calls.append(season) or [])
        monkeypatch.setattr("nfl_mcp.ingest.all_seasons", lambda: list(range(1990, 1997)))
        assert _ingest_espn_records(conn, None) == 0
        assert calls == list(range(1990, 1997))

        calls.clear()
        monkeypatch.setattr("nfl_mcp.ingest.all_seasons", lambda: list(range(1994, 2000)))
        assert _ingest_espn_records(conn, None) == 0
        assert calls == list(range(1994, 1999))   # clipped to the dataset's 1970-1998 window


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
        monkeypatch.setattr("nfl_mcp.ingest._create_plays_table", lambda c, df: None)
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

        def fetchone(self):
            return self._rows[0] if self._rows else None

    class _DummyConn:
        """Stands in for DuckDB: answers the plays query with `loaded_rows`, the seasons
        already in the table, and a _ingest_metadata query with NULL, which is what a
        database holding no record for a season answers. A loaded season whose record tests
        need lives in a real database file instead (see _seed_plays_db).
        """

        def __init__(self, loaded_rows):
            self.loaded_rows = loaded_rows
            self.closed = False
            self.queries = []

        def execute(self, sql, params=None):
            self.queries.append(sql)
            if "_ingest_metadata" in sql:
                return TestRunIngestOrchestration._DummyResult([(None,)])
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
        monkeypatch.setattr("nfl_mcp.ingest._create_plays_table", lambda c, df: None)
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
        monkeypatch.setattr("nfl_mcp.ingest._create_plays_table", lambda c, df: None)
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

    def test_run_ingest_datasets_rejects_a_source_with_no_ingest_helper(
        self, monkeypatch, tmp_path
    ):
        """A source the router does not know is a registry mistake, not a guess.

        Falling through to the nflreadpy loader would call a function named after another
        publisher and record the outcome against the wrong dataset, so the run stops here.
        """
        conn = self._DummyConn(loaded_rows=[])
        generic = []
        espn = []

        _install_fake_nflreadpy(monkeypatch)
        monkeypatch.setattr("nfl_mcp.ingest.duckdb.connect", lambda path: conn)
        monkeypatch.setattr("nfl_mcp.ingest._ensure_metadata_table", lambda c: None)
        monkeypatch.setattr(
            "nfl_mcp.ingest._ingest_generic_dataset",
            lambda c, defn, seasons, fresh=False: generic.append(defn.dataset_id) or 1,
        )
        monkeypatch.setattr(
            "nfl_mcp.ingest._ingest_espn_records",
            lambda c, seasons, fresh=False: espn.append(seasons) or 1,
        )
        monkeypatch.setitem(
            REGISTRY, "schedules",
            DatasetDef(dataset_id="schedules", loader_fn="load_schedules",
                       table_name="schedules", seasonal=True, default=True,
                       min_season=1999, source="miascotta"),
        )

        with pytest.raises(ValueError, match="no ingest helper") as excinfo:
            run_ingest_datasets(
                dataset_ids=["schedules"], start=2024, end=2024, skip_views=True,
                db_path=str(tmp_path / "nflread.duckdb"),
            )

        assert "'schedules'" in str(excinfo.value)     # names the dataset
        assert "'miascotta'" in str(excinfo.value)     # and the source that has no helper
        assert generic == [] and espn == []
        assert conn.closed is True                     # the write lock was not left behind

    def test_the_source_constants_name_the_right_datasets(self):
        # The router branches on these two constants, so a constant that no longer matches
        # its dataset would silently send play-by-play or the records pull down the others
        # path.
        assert REGISTRY["espn_records"].source == SOURCE_ESPN
        assert REGISTRY["schedules"].source == SOURCE_NFLREADPY

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

    def test_pbp_season_rows_and_metadata_row_commit_together(self, monkeypatch, tmp_path):
        """A metadata write that fails must roll back that season's plays too."""
        path = str(tmp_path / "nflread.duckdb")
        seed = pl.DataFrame({
            "season": [2024], "posteam": ["KC"], "defteam": ["BAL"],
            "play_id": [1], "game_id": ["g1"], "week": [1],
        })
        c = duckdb.connect(path)
        _ensure_metadata_table(c)
        _create_plays_table(c, seed)
        c.register("_seed", seed.with_columns(pl.lit("").alias("enhanced_description")).to_arrow())
        c.execute("INSERT INTO plays BY NAME SELECT * FROM _seed")
        c.unregister("_seed")
        _record_loaded(c, "pbp", "plays", "load_pbp", 1, season=2024)
        c.close()

        # 2024 is the in-progress season, so run_ingest_datasets re-fetches it without --fresh.
        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2024)
        _install_fake_nflreadpy(
            monkeypatch,
            load_pbp=lambda seasons: pl.DataFrame({
                "season": [2024, 2024], "posteam": ["KC", "LV"], "defteam": ["BAL", "KC"],
                "play_id": [1, 2], "game_id": ["g1", "g2"], "week": [1, 2],
            }),
        )
        monkeypatch.setattr(
            "nfl_mcp.ingest._record_loaded",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("metadata write failed")),
        )

        with pytest.raises(RuntimeError, match="metadata write failed"):
            run_ingest_datasets(["pbp"], start=2024, end=2024, skip_views=True, db_path=path)

        check = duckdb.connect(path)
        assert check.execute("SELECT season, posteam FROM plays").fetchall() == [(2024, "KC")]
        row = check.execute(
            "SELECT row_count FROM _ingest_metadata WHERE dataset_id = 'pbp' AND season = 2024"
        ).fetchone()
        assert row == (1,)
        check.close()

    @staticmethod
    def _seed_plays_db(path, seed):
        """Write a plays table holding `seed` plus one _ingest_metadata row per season."""
        conn = duckdb.connect(path)
        _ensure_metadata_table(conn)
        _create_plays_table(conn, seed)
        conn.register("_seed", seed.with_columns(pl.lit("").alias("enhanced_description")).to_arrow())
        conn.execute("INSERT INTO plays BY NAME SELECT * FROM _seed")
        conn.unregister("_seed")
        for season in sorted(set(seed["season"].to_list())):
            rows = sum(1 for s in seed["season"].to_list() if s == season)
            _record_loaded(conn, "pbp", "plays", "load_pbp", rows, season=season)
        conn.close()

    @staticmethod
    def _pbp_records(path):
        """Every (season, row_count, loaded_at) _ingest_metadata record stored for pbp."""
        conn = duckdb.connect(path)
        rows = conn.execute(
            "SELECT season, row_count, loaded_at FROM _ingest_metadata "
            "WHERE dataset_id = 'pbp' ORDER BY season"
        ).fetchall()
        conn.close()
        return rows

    def test_fresh_pbp_replaces_only_the_requested_season(self, monkeypatch, tmp_path):
        """--fresh on 2024 leaves 2023's rows and its record alone, and does not duplicate 2024."""
        path = str(tmp_path / "nflread.duckdb")
        self._seed_plays_db(path, pl.DataFrame({
            "season":  [2023, 2024],
            "posteam": ["LV", "KC"],
            "defteam": ["KC", "BAL"],
            "play_id": [1, 2],
            "game_id": ["a1", "b2"],
            "week":    [1, 1],
        }))
        records_before = self._pbp_records(path)

        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2024)
        _install_fake_nflreadpy(
            monkeypatch,
            load_pbp=lambda seasons: pl.DataFrame({
                "season":  [2024, 2024],
                "posteam": ["KC", "KC"],
                "defteam": ["LV", "CIN"],
                "play_id": [7, 8],
                "game_id": ["c3", "c3"],
                "week":    [3, 4],
            }),
        )

        run_ingest_datasets(["pbp"], start=2024, end=2024, fresh=True, skip_views=True, db_path=path)

        conn = duckdb.connect(path)
        rows = conn.execute("SELECT season, posteam, play_id FROM plays ORDER BY play_id").fetchall()
        conn.close()
        # 2023 survived untouched; 2024 holds only the freshly fetched rows, not seed + new.
        assert rows == [(2023, "LV", 1), (2024, "KC", 7), (2024, "KC", 8)]
        after = self._pbp_records(path)
        assert [(s, c) for s, c, _ in after] == [(2023, 1), (2024, 2)]
        assert after[0] == records_before[0]        # 2023's record was not rewritten
        assert after[1][2] > records_before[1][2]   # 2024's record was refreshed

    def test_fresh_pbp_keeps_a_season_when_its_fetch_fails(self, monkeypatch, tmp_path):
        """A requested season whose fetch fails keeps the rows and record it already had."""
        path = str(tmp_path / "nflread.duckdb")
        self._seed_plays_db(path, pl.DataFrame({
            "season":  [2023, 2024],
            "posteam": ["LV", "KC"],
            "defteam": ["KC", "BAL"],
            "play_id": [1, 2],
            "game_id": ["a1", "b2"],
            "week":    [1, 1],
        }))
        records_before = self._pbp_records(path)

        asked = []

        def load_pbp(seasons):
            asked.append(list(seasons))
            if len(asked) > 1:
                # The schema-discovery pull above succeeded; this season's pull fails.
                raise RuntimeError("connection reset by peer")
            return pl.DataFrame({
                "season":  [2024],
                "posteam": ["KC"],
                "defteam": ["BAL"],
                "play_id": [7],
                "game_id": ["c3"],
                "week":    [3],
            })

        monkeypatch.setattr("nfl_mcp.ingest.current_season", lambda: 2024)
        _install_fake_nflreadpy(monkeypatch, load_pbp=load_pbp)

        run_ingest_datasets(["pbp"], start=2024, end=2024, fresh=True, skip_views=True, db_path=path)

        conn = duckdb.connect(path)
        rows = conn.execute("SELECT season, posteam, play_id FROM plays ORDER BY play_id").fetchall()
        conn.close()
        assert rows == [(2023, "LV", 1), (2024, "KC", 2)]   # nothing was dropped
        assert self._pbp_records(path) == records_before    # both records still say loaded
        assert asked == [[2024], [2024]]                    # discovered, then tried once


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
from nfl_mcp.franchises import expected_franchise_count  # noqa: E402


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

    def test_unavailable_espn_is_recorded_failed_and_fetched_again_next_run(self, conn, monkeypatch, capsys):
        """A season ESPN was down for is a recorded gap, not a silently frozen one."""
        calls = []

        def boom(season, **kwargs):
            calls.append(season)
            raise EspnUnavailable("ESPN request failed: 503")

        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records", boom)
        assert _ingest_espn_records(conn, [1996]) == 0
        assert _is_loaded(conn, "espn_records", 1996) is False
        assert _outcome(conn, "espn_records", 1996) == ("failed", "ESPN request failed: 503", 0)
        assert "retrying next run" in capsys.readouterr().out

        _ingest_espn_records(conn, [1996])
        assert calls == [1996, 1996]     # the failed row did not stop the retry

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
        # The three seasons it tried are recorded as failed; the two it never reached have
        # no record at all, which is a different fact and stays distinguishable.
        assert [
            (row[0], row[1]) for row in conn.execute(
                "SELECT season, status FROM _ingest_metadata ORDER BY season"
            ).fetchall()
        ] == [(1970, "failed"), (1971, "failed"), (1972, "failed")]
        assert _outcome(conn, "espn_records", 1973) is None

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

    def test_empty_response_is_recorded_empty_and_fetched_again_next_run(self, conn, monkeypatch, capsys):
        calls = []

        def empty(season, **kwargs):
            calls.append(season)
            return []

        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records", empty)
        assert _ingest_espn_records(conn, [1970]) == 0
        assert _is_loaded(conn, "espn_records", 1970) is False
        assert _outcome(conn, "espn_records", 1970) == ("empty", "ESPN published no records", 0)
        assert "no records published" in capsys.readouterr().out

        _ingest_espn_records(conn, [1970])
        assert calls == [1970, 1970]

    def test_metadata_write_failure_rolls_back_the_season_rows(self, conn, monkeypatch):
        conn.execute(_ESPN_RECORDS_DDL)
        conn.execute(
            "INSERT INTO espn_team_records VALUES "
            "(1996, 'GB', 9, 'GB', 16, 13, 3, 0, '13-3', 'g1', '1997-01-01')"
        )
        _record_loaded(conn, "espn_records", "espn_team_records", "espn_records", 1, season=1996)
        monkeypatch.setattr(
            "nfl_mcp.espn.fetch_season_records",
            lambda season, **kw: [_record(season, "NO", "11-5", 11, 5, 0)],
        )
        monkeypatch.setattr(
            "nfl_mcp.ingest._record_loaded",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("metadata write failed")),
        )

        with pytest.raises(RuntimeError, match="metadata write failed"):
            _ingest_espn_records(conn, [1996], fresh=True)

        # The delete + insert unit rolled back with the metadata row that failed.
        teams = conn.execute("SELECT team FROM espn_team_records WHERE season = 1996").fetchall()
        assert teams == [("GB",)]

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
        assert _outcome(conn, "espn_records", 1996) == ("loaded", None, 30)

    def test_short_season_is_recorded_partial_and_skipped_next_run(self, conn, monkeypatch):
        """Fewer records than that year's league is a partial season, and a permanent one.

        The 34 franchise-seasons neither ESPN feed covers stay partial for good, so partial
        keeps the season skipped just as loaded does — re-fetching would never fill them in.
        """
        calls = []
        monkeypatch.setattr(
            "nfl_mcp.espn.fetch_season_records",
            lambda season, **kw: calls.append(season) or [_record(season, "GB", "13-3", 13, 3, 0)],
        )
        expected = expected_franchise_count(1996)
        assert _ingest_espn_records(conn, [1996]) == 1
        assert _outcome(conn, "espn_records", 1996) == ("partial", f"1 of {expected} franchises", 1)
        assert _is_loaded(conn, "espn_records", 1996) is True

        _ingest_espn_records(conn, [1996])
        assert calls == [1996]
        assert _outcome(conn, "espn_records", 1996) == ("partial", f"1 of {expected} franchises", 1)


# ── What one run says about itself ─────────────────────────────────────────────
#
# A run is the unit `nfl-mcp ingest` turns into an exit code, and the container's scheduled
# refresh swaps the live database only on exit 0. These tests are what keeps the two ends of
# that chain honest: the attempts a run collected, the summary it printed, and the fact that
# a run that loaded nothing does not look like a run that loaded something.

from contextlib import contextmanager  # noqa: E402

from nfl_mcp.ingest import (  # noqa: E402
    _ACTIVE_RUN,
    _drop_stale_derived_table,
    _print_run_summary,
    _RunCollector,
    _SUMMARY_LINE_CAP,
    DERIVED_LOADER,
    EspnStop,
    IngestAttempt,
    IngestRunResult,
)

@contextmanager
def _collecting_run():
    """Install a collector the way run_ingest_datasets does, so a helper can report to it."""
    collector = _RunCollector()
    token = _ACTIVE_RUN.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE_RUN.reset(token)


def _attempt(dataset_id, season, status, detail=None):
    return IngestAttempt(dataset_id=dataset_id, season=season, status=status, detail=detail)


class TestIngestRunResult:
    def test_an_empty_run_loaded_nothing_but_is_not_a_failure(self):
        # A run that only skipped already-loaded seasons attempted nothing. That is not the
        # dead-network case, so it must not read as "nothing loaded".
        result = IngestRunResult()
        assert result.attempts == []
        assert result.nothing_loaded is False
        assert result.any_failed is False
        assert result.espn_breaker_tripped is False

    def test_counts_reports_every_status_including_the_ones_that_did_not_happen(self):
        result = IngestRunResult(attempts=[
            _attempt("pbp", 2024, IngestStatus.LOADED),
            _attempt("pbp", 2023, IngestStatus.FAILED, "404"),
            _attempt("pbp", 2022, IngestStatus.FAILED, "404"),
            _attempt("teams", None, IngestStatus.EMPTY, "no rows"),
        ])
        assert result.counts == {
            IngestStatus.LOADED: 1,
            IngestStatus.PARTIAL: 0,
            IngestStatus.EMPTY: 1,
            IngestStatus.FAILED: 2,
        }

    def test_a_run_where_every_fetch_failed_loaded_nothing(self):
        result = IngestRunResult(attempts=[
            _attempt("pbp", 2024, IngestStatus.FAILED, "no route to host"),
            _attempt("teams", None, IngestStatus.EMPTY, "load_teams returned no rows"),
        ])
        assert result.nothing_loaded is True
        assert result.any_failed is True

    def test_one_season_that_landed_means_the_run_loaded_something(self):
        result = IngestRunResult(attempts=[
            _attempt("pbp", 2024, IngestStatus.FAILED, "no route to host"),
            _attempt("schedules", 2024, IngestStatus.LOADED),
        ])
        assert result.nothing_loaded is False

    def test_a_partial_season_counts_as_loaded_data(self):
        result = IngestRunResult(attempts=[_attempt("espn_records", 1974, IngestStatus.PARTIAL,
                                                    "28 of 30 franchises")])
        assert result.nothing_loaded is False
        assert result.any_failed is False

    def test_an_empty_season_is_not_a_failure_for_strict_mode(self):
        # `empty` is an answer from the publisher, so --strict does not mind it.
        result = IngestRunResult(attempts=[_attempt("teams", None, IngestStatus.EMPTY)])
        assert result.any_failed is False

    def test_the_espn_stop_shows_up_as_a_tripped_breaker(self):
        assert IngestRunResult().espn_breaker_tripped is False
        stopped = IngestRunResult(espn_stop=EspnStop(failed_seasons=3, seasons_left=5))
        assert stopped.espn_breaker_tripped is True
        assert stopped.espn_stop.seasons_left == 5


class TestRunCollector:
    def test_recording_helpers_report_to_the_run_in_progress(self, conn):
        with _collecting_run() as collector:
            _record_loaded(conn, "schedules", "schedules", "load_schedules", 9, 2024)
            _record_attempt(conn, "pbp", "plays", "load_pbp", 2023,
                            IngestStatus.FAILED, "404 Not Found")
        assert [(a.dataset_id, a.season, a.status, a.detail) for a in collector.attempts] == [
            ("schedules", 2024, "loaded", None),
            ("pbp", 2023, "failed", "404 Not Found"),
        ]

    def test_a_failed_fetch_of_an_already_loaded_season_is_still_an_attempt(self, conn):
        """The row in the database stays 'loaded', but this run's fetch failed anyway.

        This is the container's normal failure mode: last week's rows are on disk, today's
        fetch of the in-progress season gets nothing, and the slot gained nothing.
        """
        _record_loaded(conn, "pbp", "plays", "load_pbp", 500, 2026)
        with _collecting_run() as collector:
            _record_attempt(conn, "pbp", "plays", "load_pbp", 2026,
                            IngestStatus.FAILED, "no route to host")
        assert [a.status for a in collector.attempts] == [IngestStatus.FAILED]
        assert _outcome(conn, "pbp", 2026)[0] == "loaded"   # the stored record still says loaded

    def test_derived_table_rows_are_not_fetch_attempts(self, conn):
        with _collecting_run() as collector:
            _record_loaded(conn, "player_td_luck", "player_td_luck", DERIVED_LOADER, 12)
            _record_loaded(conn, "team_records", "team_records", DERIVED_LOADER, 4)
        assert collector.attempts == []

    def test_a_detail_longer_than_the_column_is_clipped_in_the_run_too(self, conn):
        with _collecting_run() as collector:
            _record_attempt(conn, "pbp", "plays", "load_pbp", 2024,
                            IngestStatus.FAILED, "x" * (_DETAIL_MAX_CHARS + 50))
        assert len(collector.attempts[0].detail) == _DETAIL_MAX_CHARS

    def test_recording_outside_a_run_is_harmless(self, conn):
        # Most of this file's tests call the helpers with no run in progress.
        assert _ACTIVE_RUN.get() is None
        _record_loaded(conn, "schedules", "schedules", "load_schedules", 1, 2024)
        assert _ACTIVE_RUN.get() is None

    def test_the_collector_goes_away_when_the_run_raises(self, monkeypatch, tmp_path):
        real_connect = duckdb.connect
        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: _throw(
            RuntimeError("no pbp file")))
        monkeypatch.setattr("nfl_mcp.ingest.duckdb.connect",
                            lambda path: real_connect(":memory:"))
        with pytest.raises(RuntimeError, match="Could not load play-by-play"):
            run_ingest_datasets(["pbp"], start=2024, end=2024, skip_views=True,
                                db_path=str(tmp_path / "nflread.duckdb"))
        assert _ACTIVE_RUN.get() is None


class TestPrintRunSummary:
    def test_counts_and_every_unfilled_attempt_are_printed(self, capsys):
        _print_run_summary(IngestRunResult(attempts=[
            _attempt("pbp", 2024, IngestStatus.LOADED),
            _attempt("pbp", 2023, IngestStatus.FAILED, "404 Not Found"),
            _attempt("teams", None, IngestStatus.EMPTY, "load_teams returned no rows"),
        ]))
        out = capsys.readouterr().out
        assert "Run summary:" in out
        assert "loaded: 1" in out
        assert "empty: 1" in out
        assert "failed: 1" in out
        assert "pbp 2023: failed: 404 Not Found" in out
        # A static dataset has no season, so its line names only the dataset.
        assert "teams: empty: load_teams returned no rows" in out
        # A loaded attempt needs no line: the count already says it.
        assert "pbp 2024" not in out

    def test_an_attempt_with_no_reason_is_printed_without_one(self, capsys):
        _print_run_summary(IngestRunResult(attempts=[
            _attempt("pbp", 2024, IngestStatus.FAILED, None)]))
        assert "pbp 2024: failed" in capsys.readouterr().out

    def test_the_list_of_bad_attempts_is_capped(self, capsys):
        attempts = [
            _attempt("pbp", 2000 + i, IngestStatus.FAILED, "timeout")
            for i in range(_SUMMARY_LINE_CAP + 3)
        ]
        _print_run_summary(IngestRunResult(attempts=attempts))
        out = capsys.readouterr().out
        assert f"failed: {_SUMMARY_LINE_CAP + 3}" in out
        assert "... and 3 more" in out
        # The first _SUMMARY_LINE_CAP seasons are listed, the rest are only counted.
        assert f"pbp {2000 + _SUMMARY_LINE_CAP}" not in out
        assert f"pbp {1999 + _SUMMARY_LINE_CAP}" in out

    def test_the_cap_line_is_absent_when_the_list_fits(self, capsys):
        _print_run_summary(IngestRunResult(attempts=[
            _attempt("pbp", 2024, IngestStatus.FAILED, "timeout")]))
        assert "more" not in capsys.readouterr().out

    def test_the_summary_says_when_the_espn_pull_stopped(self, capsys):
        _print_run_summary(IngestRunResult(
            attempts=[_attempt("espn_records", 1970 + i, IngestStatus.FAILED, "503")
                      for i in range(3)],
            espn_stop=EspnStop(failed_seasons=3, seasons_left=26),
        ))
        out = capsys.readouterr().out
        assert "ESPN pull stopped after 3 failed seasons; 26 seasons not attempted" in out

    def test_a_run_that_loaded_something_says_so_in_its_summary(self, capsys):
        _print_run_summary(IngestRunResult(attempts=[_attempt("pbp", 2024, IngestStatus.LOADED)]))
        out = capsys.readouterr().out
        assert "loaded: 1" in out
        assert "failed: 0" in out


class _SpyConnection:
    """Wraps a real DuckDB connection and counts how often the run closed it."""

    def __init__(self):
        self.inner = duckdb.connect(":memory:")
        self.closes = 0

    def execute(self, *args, **kwargs):
        return self.inner.execute(*args, **kwargs)

    def register(self, *args, **kwargs):
        return self.inner.register(*args, **kwargs)

    def unregister(self, *args, **kwargs):
        return self.inner.unregister(*args, **kwargs)

    def close(self):
        self.closes += 1
        self.inner.close()


class TestRunResultEndToEnd:
    """A run's result through run_ingest_datasets, with the loaders faked."""

    def _run(self, monkeypatch, tmp_path, dataset_ids, start, end, **loaders):
        _install_fake_nflreadpy(monkeypatch, **loaders)
        return run_ingest_datasets(dataset_ids, start=start, end=end, skip_views=True,
                                   db_path=str(tmp_path / "nflread.duckdb"))

    def test_the_run_reports_each_attempt_and_its_outcome(self, monkeypatch, tmp_path, capsys):
        def load_schedules(seasons, **kwargs):
            season = seasons[0]
            if season == 2022:
                raise RuntimeError("404, reason: Not Found")
            if season == 2023:
                return _small_df(season=[], team=[])
            return _small_df(season=[season], team=["KC"])

        result = self._run(monkeypatch, tmp_path, ["schedules"], 2021, 2023,
                           load_schedules=load_schedules)

        assert [(a.dataset_id, a.season, a.status) for a in result.attempts] == [
            ("schedules", 2021, IngestStatus.LOADED),
            ("schedules", 2022, IngestStatus.FAILED),
            ("schedules", 2023, IngestStatus.EMPTY),
        ]
        assert result.counts == {
            IngestStatus.LOADED: 1, IngestStatus.PARTIAL: 0,
            IngestStatus.EMPTY: 1, IngestStatus.FAILED: 1,
        }
        assert (result.nothing_loaded, result.any_failed) == (False, True)

        out = capsys.readouterr().out
        assert "Run summary:" in out
        assert "schedules 2022: failed: 404, reason: Not Found" in out
        assert "schedules 2023: empty: load_schedules returned no rows" in out

    def test_a_run_where_everything_fails_reports_nothing_loaded(self, monkeypatch, tmp_path):
        result = self._run(monkeypatch, tmp_path, ["schedules"], 2021, 2022,
                           load_schedules=lambda seasons, **kw: _throw(
                               RuntimeError("no route to host")))
        assert [a.status for a in result.attempts] == [IngestStatus.FAILED, IngestStatus.FAILED]
        assert result.nothing_loaded is True

    def test_a_run_that_only_skipped_seasons_attempted_nothing(self, monkeypatch, tmp_path):
        _install_fake_nflreadpy(
            monkeypatch,
            load_schedules=lambda seasons, **kw: _small_df(season=seasons, team=["KC"]),
        )
        path = str(tmp_path / "nflread.duckdb")
        first = run_ingest_datasets(["schedules"], start=2021, end=2021, skip_views=True,
                                    db_path=path)
        assert [a.status for a in first.attempts] == [IngestStatus.LOADED]

        second = run_ingest_datasets(["schedules"], start=2021, end=2021, skip_views=True,
                                     db_path=path)
        # 2021 settled long ago, so this run made no request at all: nothing failed, and
        # nothing loading is not what happened either.
        assert second.attempts == []
        assert second.nothing_loaded is False

    def test_a_second_run_does_not_inherit_the_first_runs_attempts(self, monkeypatch, tmp_path):
        """The collector belongs to one run: nothing leaks between them, or into a test."""
        self._run(monkeypatch, tmp_path, ["schedules"], 2021, 2021,
                  load_schedules=lambda seasons, **kw: _small_df(season=seasons, team=["KC"]))
        # The run before this one loaded 2021, so a static dataset is the only attempt left.
        result = self._run(monkeypatch, tmp_path, ["teams"], 2021, 2021,
                           load_teams=lambda: _small_df(team=["KC"]))
        assert [(a.dataset_id, a.season, a.status) for a in result.attempts] == [
            ("teams", None, IngestStatus.LOADED)
        ]

    def test_the_run_records_the_same_outcomes_it_reports(self, monkeypatch, tmp_path):
        """What the run says and what the database says must not disagree."""
        def load_schedules(seasons, **kwargs):
            if seasons[0] == 2022:
                raise RuntimeError("404, reason: Not Found")
            return _small_df(season=seasons, team=["KC"])

        result = self._run(monkeypatch, tmp_path, ["schedules"], 2021, 2022,
                           load_schedules=load_schedules)
        path = tmp_path / "nflread.duckdb"
        conn = duckdb.connect(str(path))
        try:
            stored = dict(conn.execute(
                "SELECT season, status FROM _ingest_metadata WHERE dataset_id = 'schedules'"
            ).fetchall())
        finally:
            conn.close()
        assert stored == {2021: "loaded", 2022: "failed"}
        assert {a.season: str(a.status) for a in result.attempts} == stored

    def test_the_write_connection_is_closed_when_the_run_raises(self, monkeypatch, tmp_path):
        spy = _SpyConnection()
        monkeypatch.setattr("nfl_mcp.ingest.duckdb.connect", lambda path: spy)
        _install_fake_nflreadpy(monkeypatch, load_pbp=lambda seasons: _throw(
            RuntimeError("no pbp file")))

        with pytest.raises(RuntimeError, match="Could not load play-by-play"):
            run_ingest_datasets(["pbp"], start=2024, end=2024, skip_views=True,
                                db_path=str(tmp_path / "nflread.duckdb"))
        # DuckDB allows one writer per file, so a run that raises must still let go of it.
        assert spy.closes == 1

    def test_the_write_connection_is_closed_when_a_later_step_raises(self, monkeypatch, tmp_path):
        spy = _SpyConnection()
        monkeypatch.setattr("nfl_mcp.ingest.duckdb.connect", lambda path: spy)
        _install_fake_nflreadpy(
            monkeypatch,
            load_schedules=lambda seasons, **kw: _small_df(season=seasons, team=["KC"]),
        )
        monkeypatch.setattr("nfl_mcp.ingest._ingest_generic_dataset",
                            lambda *a, **k: _throw(RuntimeError("disk full")))

        with pytest.raises(RuntimeError, match="disk full"):
            run_ingest_datasets(["schedules"], start=2024, end=2024, skip_views=True,
                                db_path=str(tmp_path / "nflread.duckdb"))
        assert spy.closes == 1


class TestEspnFailureStreak:
    """The rule at ESPN_FAILURE_LIMIT, spelled out and tested (see AGENTS.md)."""

    def _fetch(self, conn, monkeypatch, seasons, outcome_by_season):
        calls = []

        def fake_fetch(season, **kwargs):
            calls.append(season)
            outcome = outcome_by_season[season]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records", fake_fetch)
        with _collecting_run() as collector:
            total = _ingest_espn_records(conn, list(seasons))
        return calls, total, collector.result()

    def test_an_empty_answer_resets_the_streak(self, conn, monkeypatch):
        """ESPN reaching us and having nothing is an answer, not another failure."""
        down = EspnUnavailable("ESPN request failed: 503")
        calls, total, result = self._fetch(conn, monkeypatch, [1970, 1971, 1972, 1973], {
            1970: down, 1971: [], 1972: down, 1973: down,
        })
        assert calls == [1970, 1971, 1972, 1973]      # all four asked, breaker never tripped
        assert result.espn_breaker_tripped is False
        assert total == 0
        assert [a.status for a in result.attempts] == [
            IngestStatus.FAILED, IngestStatus.EMPTY, IngestStatus.FAILED, IngestStatus.FAILED,
        ]

    def test_a_skipped_season_does_not_reset_the_streak(self, conn, monkeypatch):
        """A season already loaded asked nothing of ESPN, so the failures stay in a row."""
        _record_loaded(conn, "espn_records", "espn_team_records", "espn_records", 30, 1971)
        down = EspnUnavailable("ESPN request failed: 503")
        calls, total, result = self._fetch(conn, monkeypatch, [1970, 1971, 1972, 1973, 1974], {
            1970: down, 1972: down, 1973: down, 1974: down,
        })
        # 1971 was skipped without a request, so 1970/1972/1973 are three in a row.
        assert calls == [1970, 1972, 1973]
        assert total == 0
        assert result.espn_breaker_tripped is True
        assert result.espn_stop == EspnStop(failed_seasons=3, seasons_left=1)
        assert [a.season for a in result.attempts] == [1970, 1972, 1973]

    def test_the_stop_is_reported_even_with_no_run_in_progress(self, conn, monkeypatch):
        """Direct calls, which is how most of this file drives the pull, must not crash."""
        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records",
                            lambda season, **kw: _throw(EspnUnavailable("503")))
        assert _ACTIVE_RUN.get() is None
        assert _ingest_espn_records(conn, [1970, 1971, 1972, 1973], fresh=True) == 0

    def test_the_trip_is_visible_in_the_run_summary(self, conn, monkeypatch, capsys):
        down = EspnUnavailable("ESPN request failed: 503")
        _, _, result = self._fetch(conn, monkeypatch, [1970, 1971, 1972, 1973],
                                   {s: down for s in (1970, 1971, 1972, 1973)})
        assert result.espn_breaker_tripped is True
        _print_run_summary(result)
        assert "ESPN pull stopped after 3 failed seasons; 1 seasons not attempted" in \
            capsys.readouterr().out

    def test_a_malformed_payload_fails_one_season_and_the_run_carries_on(
        self, monkeypatch, tmp_path
    ):
        """ESPN answering with the wrong JSON shape must not abort every later dataset."""
        # A top-level list is what ESPN returns for some bad requests; reading it raised
        # AttributeError straight out of the pull before this was wrapped.
        monkeypatch.setattr("nfl_mcp.espn._default_transport", lambda url: [])
        _install_fake_nflreadpy(
            monkeypatch,
            load_schedules=lambda seasons, **kw: _small_df(season=seasons, team=["KC"]),
        )
        path = str(tmp_path / "nflread.duckdb")
        result = run_ingest_datasets(["espn_records", "schedules"], start=1998, end=1999,
                                     skip_views=True, db_path=path)

        espn, schedules = result.attempts
        assert (espn.dataset_id, espn.season, espn.status) == ("espn_records", 1998, "failed")
        assert "unexpected shape" in espn.detail
        assert (schedules.dataset_id, schedules.status) == ("schedules", IngestStatus.LOADED)
        assert result.nothing_loaded is False
        assert result.espn_breaker_tripped is False

        conn = duckdb.connect(str(path))
        try:
            assert conn.execute("SELECT COUNT(*) FROM schedules").fetchone()[0] == 1
        finally:
            conn.close()


class TestStaleDerivedTables:
    """A derived table must not outlive the tables it was built from."""

    def _ff_opportunity(self, conn):
        ff = pl.DataFrame({
            "player_id": ["00-1", "00-2"],
            "full_name": ["A Back", "B Wr"],
            "position": ["RB", "WR"],
            "posteam": ["KC", "BAL"],
            "season": ["2024", "2024"],
            "rec_touchdown": [1.0, 2.0],
            "rec_touchdown_exp": [1.5, 1.0],
            "rec_touchdown_diff": [-0.5, 1.0],
            "rush_touchdown": [1.0, 0.0],
            "rush_touchdown_exp": [0.5, 0.0],
            "rush_touchdown_diff": [0.5, 0.0],
        })
        _write_df_to_table(conn, "ff_opportunity", ff, replace=True)

    def _metadata(self, conn, dataset_id):
        return conn.execute(
            "SELECT status, row_count FROM _ingest_metadata WHERE dataset_id = ?", [dataset_id]
        ).fetchone()

    def test_a_derived_table_whose_source_is_gone_is_dropped_with_its_record(
        self, conn, capsys
    ):
        self._ff_opportunity(conn)
        _create_fantasy_tables(conn)
        assert _table_exists(conn, "player_td_luck") is True
        assert self._metadata(conn, "player_td_luck") == ("loaded", 2)

        conn.execute("DROP TABLE ff_opportunity")
        capsys.readouterr()
        _create_fantasy_tables(conn)

        assert _table_exists(conn, "player_td_luck") is False
        # Its record went with it: nfl_status must stop reporting a table that is not there.
        assert self._metadata(conn, "player_td_luck") is None
        assert "player_td_luck — dropped (its source data is gone)" in capsys.readouterr().out

    def test_skipping_a_table_that_was_never_built_is_not_a_drop(self, conn, capsys):
        _create_fantasy_tables(conn)          # nothing has sources, nothing was ever built
        out = capsys.readouterr().out
        assert "dropped" not in out

    def test_team_records_survives_while_either_source_is_still_there(self, conn, capsys):
        conn.execute(_ESPN_RECORDS_DDL)
        conn.execute(
            "INSERT INTO espn_team_records VALUES "
            "(1996, 'GB', 9, 'GB', 16, 13, 3, 0, '13-3', 'g9', '1997-01-01')"
        )
        _schedules_table(conn, [(1999, "REG", "KC", "BAL", 26, 10)])
        _create_team_records(conn)
        # One ESPN row for 1996, and KC plus BAL for the 1999 game.
        assert conn.execute("SELECT COUNT(*) FROM team_records").fetchone()[0] == 3

        conn.execute("DROP TABLE schedules")
        capsys.readouterr()
        _create_team_records(conn)

        assert _table_exists(conn, "team_records") is True
        assert conn.execute("SELECT COUNT(*) FROM team_records").fetchone()[0] == 1
        assert self._metadata(conn, "team_records") == ("loaded", 1)
        assert "dropped" not in capsys.readouterr().out

    def test_team_records_is_dropped_when_both_sources_are_gone(self, conn, capsys):
        _schedules_table(conn, [(1999, "REG", "KC", "BAL", 26, 10)])
        _create_team_records(conn)
        assert self._metadata(conn, "team_records") == ("loaded", 2)

        conn.execute("DROP TABLE schedules")
        capsys.readouterr()
        _create_team_records(conn)

        assert _table_exists(conn, "team_records") is False
        assert self._metadata(conn, "team_records") is None
        out = capsys.readouterr().out
        assert "team_records — skipped" in out
        assert "team_records — dropped (its source data is gone)" in out

    def test_dropping_a_stale_table_is_atomic(self, conn, monkeypatch):
        """If the record delete fails, the table stays, so the two cannot disagree."""
        conn.execute("CREATE TABLE player_td_luck (player_id VARCHAR)")
        _record_loaded(conn, "player_td_luck", "player_td_luck", DERIVED_LOADER, 1)
        monkeypatch.setattr(
            "nfl_mcp.ingest._delete_metadata_row",
            lambda *a, **k: _throw(RuntimeError("metadata write failed")),
        )
        with pytest.raises(RuntimeError, match="metadata write failed"):
            _drop_stale_derived_table(conn, "player_td_luck")
        assert _table_exists(conn, "player_td_luck") is True
        assert self._metadata(conn, "player_td_luck") == ("loaded", 1)
