"""The merge review: a row that landed nothing must not be read as data.

Source: the code review of merging the `state-machine` branch into `main` (v0.7.3) on
`merge/state-machine-into-main`.

`main` wrote an `_ingest_metadata` row only when rows landed. The state machine also writes
one for an attempt that ended `failed` or `empty`. That is more truth in the log, and it
broke every read that counted a row as a season. Five rules are tested here, from the review's
items 1, 2, 3, 4 and 6:

1. A `failed` or `empty` row keeps a season out of the depth_charts restamp plan and out of
   the season list `nfl-mcp upgrade` reloads. The two reads count only `loaded` and `partial`
   rows, and both work on a database whose metadata table has no `status` column.
2. `nfl-mcp ingest` exits 3 for "nothing loaded" only when at least one fetch FAILED and none
   landed. A run where every fetch came back EMPTY exits 0 and says the sources had no rows.
3. A failed ESPN season is recorded, counted and listed, and decides nothing else. It does
   not set the exit code and does not stop the schema stamp, because ESPN is the one
   opt-in third-party source. `--strict` counts it.
4. An attempt written inside a `_transaction` reaches the run report only after the COMMIT.
   A rollback drops it, so a season the database lost is not reported as loaded.
5. The ingest spills to `NFL_MCP_DUCKDB_TEMP_DIR`, the variable a query connection already
   reads, and to the database folder when it is unset.

Everything here runs in memory or in tmp_path, with nflreadpy and ESPN faked. Nothing uses
the network, a real database, or `~/.nfl-mcp`.
"""

from __future__ import annotations

import os
import sys
import types

import duckdb
import polars as pl
import pytest
from click.testing import CliRunner

from nfl_mcp import ingest as ing
from nfl_mcp import migrate
from nfl_mcp.cli import INGEST_PARTIAL_EXIT, NOTHING_LOADED_MESSAGE, main
from nfl_mcp.espn import EspnUnavailable
from nfl_mcp.ingest import (
    _apply_duckdb_pragmas,
    _ensure_metadata_table,
    _ingest_espn_records,
    _legacy_restamp_plan,
    _record_attempt,
    _record_loaded,
    _transaction,
    run_ingest_datasets,
)
from nfl_mcp.registry import REGISTRY, SOURCE_ESPN
from nfl_mcp.status import IngestStatus

# ── Helpers ────────────────────────────────────────────────────────────────────


def _small_df(**cols) -> pl.DataFrame:
    return pl.DataFrame(cols)


def _install_fake_nflreadpy(monkeypatch, **loaders):
    fake = types.ModuleType("nflreadpy")
    for name, fn in loaders.items():
        setattr(fake, name, fn)
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)


@pytest.fixture
def conn():
    """In-memory connection with the current metadata table."""
    c = duckdb.connect(":memory:")
    _ensure_metadata_table(c)
    yield c
    c.close()


@pytest.fixture
def old_conn():
    """In-memory connection with the metadata table v0.7.3 wrote: no status, no detail."""
    c = duckdb.connect(":memory:")
    c.execute(
        "CREATE TABLE _ingest_metadata ("
        "  dataset_id VARCHAR NOT NULL, table_name VARCHAR NOT NULL, season INTEGER,"
        "  row_count BIGINT, loaded_at TIMESTAMP NOT NULL, loader_fn VARCHAR)"
    )
    yield c
    c.close()


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """CliRunner with the config and the database pointed at tmp_path."""
    monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: tmp_path / "nflread.duckdb")
    monkeypatch.delenv("NFL_MCP_DB_PATH", raising=False)
    return CliRunner()


def _stub_run(monkeypatch, *attempts):
    """Replace the whole run with one that reported these attempts, and return the result."""
    from nfl_mcp.ingest import IngestRunResult

    result = IngestRunResult(attempts=list(attempts))
    monkeypatch.setattr("nfl_mcp.ingest.run_ingest_datasets", lambda **kw: result)
    return result


def _attempt(dataset_id, season, status, detail="no route to host"):
    from nfl_mcp.ingest import IngestAttempt

    return IngestAttempt(dataset_id=dataset_id, season=season, status=status, detail=detail,
                         source=ing._attempt_source(dataset_id))


def _espn_attempt(season, status=IngestStatus.FAILED, detail="ESPN request failed: 429"):
    from nfl_mcp.ingest import IngestAttempt

    return IngestAttempt(dataset_id="espn_records", season=season, status=status,
                         detail=detail, source=SOURCE_ESPN)


def _status_of(conn, dataset_id, season):
    row = conn.execute(
        "SELECT status FROM _ingest_metadata WHERE dataset_id = ? AND season = ?",
        [dataset_id, season],
    ).fetchone()
    return row[0] if row else None


# ── 1. A failed season is not a season the database has ────────────────────────


class TestRestampPlanCountsOnlyStoredSeasons:
    def test_a_failed_attempt_stays_out_of_the_plan(self, conn):
        """The reviewer's case: one loaded season and one failed attempt restamps one season.

        Before the fix 2026 entered the plan because its metadata row existed. Every run
        then asked ESPN-nflreadpy for a season that keeps failing, and the plan grew by one
        season that no fetch can fill.
        """
        conn.execute("CREATE TABLE depth_charts (team VARCHAR, season BIGINT)")
        conn.execute("INSERT INTO depth_charts VALUES ('KC', NULL), ('LV', NULL)")
        _record_loaded(conn, "depth_charts", "depth_charts", "load_depth_charts", 3, 2024)
        _record_attempt(conn, "depth_charts", "depth_charts", "load_depth_charts", 2026,
                        IngestStatus.FAILED, "404 Not Found")
        _record_attempt(conn, "depth_charts", "depth_charts", "load_depth_charts", 2025,
                        IngestStatus.EMPTY, "load_depth_charts returned no rows")

        assert _legacy_restamp_plan(conn, REGISTRY["depth_charts"]) == (True, [2024])

    def test_a_partial_season_still_enters_the_plan(self, conn):
        """`partial` means the rows are in the table, so the season is one the database has."""
        conn.execute("CREATE TABLE depth_charts (team VARCHAR, season BIGINT)")
        conn.execute("INSERT INTO depth_charts VALUES ('KC', NULL)")
        _record_loaded(conn, "depth_charts", "depth_charts", "load_depth_charts", 3, 2024,
                       status=IngestStatus.PARTIAL, detail="28 of 30")

        assert _legacy_restamp_plan(conn, REGISTRY["depth_charts"]) == (True, [2024])

    def test_a_table_with_no_status_column_still_gets_a_plan(self, old_conn):
        """Every row the old code wrote describes loaded data, so all of them are candidates."""
        old_conn.execute("CREATE TABLE depth_charts (team VARCHAR, season BIGINT)")
        old_conn.execute("INSERT INTO depth_charts VALUES ('KC', NULL)")
        for season in (2024, 2026):
            old_conn.execute(
                "INSERT INTO _ingest_metadata VALUES (?, 'depth_charts', ?, 3, "
                "TIMESTAMP '2026-01-05 00:00:00', 'load_depth_charts')",
                ["depth_charts", season],
            )

        assert _legacy_restamp_plan(old_conn, REGISTRY["depth_charts"]) == (True, [2024, 2026])

    def test_the_predicate_is_built_from_the_enums_not_a_string(self, conn):
        """The filter binds IngestStatus members, so a renamed state cannot silently drift."""
        predicate, params = ing._stored_rows_predicate(conn)
        assert predicate == " AND status IN (?, ?)"
        assert params == [IngestStatus.LOADED, IngestStatus.PARTIAL]
        assert all(isinstance(p, IngestStatus) for p in params)


class TestUpgradeSeasonList:
    def test_a_failed_attempt_is_not_a_season_to_reload(self, conn):
        _record_loaded(conn, "schedules", "schedules", "load_schedules", 9, 2024)
        _record_attempt(conn, "schedules", "schedules", "load_schedules", 2025,
                        IngestStatus.EMPTY, "no rows")
        _record_attempt(conn, "schedules", "schedules", "load_schedules", 2026,
                        IngestStatus.FAILED, "404 Not Found")

        assert migrate._metadata_seasons(conn, "schedules") == [2024]

    def test_the_upgrade_still_adds_the_columns_an_old_table_lacks(self, old_conn):
        """_metadata_seasons runs after _ensure_metadata_table, so it can filter by status."""
        old_conn.execute(
            "INSERT INTO _ingest_metadata VALUES (?, 'rosters', ?, 3, "
            "TIMESTAMP '2026-01-05 00:00:00', 'load_rosters')", ["rosters", 2019])

        assert migrate._metadata_seasons(old_conn, "rosters") == [2019]
        assert "status" in ing._metadata_columns(old_conn)


# ── 2. An all-empty run is an answer, not a failure ────────────────────────────


class TestEmptyRunExitCode:
    def test_a_run_where_every_fetch_was_empty_exits_zero(self, runner, monkeypatch):
        _stub_run(monkeypatch,
                  _attempt("schedules", 2024, IngestStatus.EMPTY, "no rows"),
                  _attempt("teams", None, IngestStatus.EMPTY, "no rows"))

        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--skip-views"])

        assert result.exit_code == 0, result.output
        assert NOTHING_LOADED_MESSAGE not in result.output
        assert "No source published new rows" in result.output

    def test_a_run_where_every_fetch_was_empty_exits_zero_with_allow_failures(
            self, runner, monkeypatch):
        _stub_run(monkeypatch, _attempt("schedules", 2024, IngestStatus.EMPTY, "no rows"))

        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--skip-views",
                                      "--allow-failures"])

        assert result.exit_code == 0, result.output

    def test_empty_and_failed_with_nothing_landed_still_exits_three(self, runner, monkeypatch):
        _stub_run(monkeypatch,
                  _attempt("schedules", 2024, IngestStatus.EMPTY, "no rows"),
                  _attempt("schedules", 2023, IngestStatus.FAILED, "no route to host"))

        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--skip-views"])

        assert result.exit_code == INGEST_PARTIAL_EXIT, result.output
        assert NOTHING_LOADED_MESSAGE in result.output
        allowed = runner.invoke(main, ["ingest", "--dataset", "schedules", "--skip-views",
                                       "--allow-failures"])
        assert allowed.exit_code == INGEST_PARTIAL_EXIT, allowed.output

    def test_a_fetched_season_that_landed_says_nothing_about_empty_sources(
            self, runner, monkeypatch):
        _stub_run(monkeypatch,
                  _attempt("schedules", 2024, IngestStatus.LOADED),
                  _attempt("teams", None, IngestStatus.EMPTY, "no rows"))

        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--skip-views"])

        assert result.exit_code == 0, result.output
        assert "No source published new rows" not in result.output

    def test_an_empty_run_reports_nothing_loaded_as_a_fact(self):
        """The property stays "this run put no rows in the database", which is what init reads."""
        from nfl_mcp.ingest import IngestRunResult

        all_empty = IngestRunResult(attempts=[
            _attempt("schedules", 2024, IngestStatus.EMPTY, "no rows")])
        assert all_empty.nothing_loaded is True
        assert all_empty.any_failed is False


# ── 3. ESPN is optional, so its failures are too ───────────────────────────────


class TestEspnFailuresDoNotGateTheRun:
    def _break_espn(self, monkeypatch):
        """Make every ESPN season fail the way a rate-limited request does."""
        monkeypatch.setattr("nfl_mcp.espn.fetch_season_records",
                            lambda season, **kw: (_ for _ in ()).throw(
                                EspnUnavailable("ESPN request failed: 429")))

    def test_the_attempt_carries_its_source(self, conn, monkeypatch, ingest_run):
        self._break_espn(monkeypatch)
        _ingest_espn_records(conn, [1998])

        attempt = ingest_run.result().attempts[0]
        assert (attempt.dataset_id, attempt.status, attempt.source) == (
            "espn_records", IngestStatus.FAILED, SOURCE_ESPN)
        assert _status_of(conn, "espn_records", 1998) == "failed"

    def test_an_espn_failure_is_left_out_of_the_gating_list(self, conn, monkeypatch,
                                                            ingest_run):
        self._break_espn(monkeypatch)
        _ingest_espn_records(conn, [1998])

        result = ingest_run.result()
        assert result.failures == []
        assert result.optional_source_failures == ["espn_records 1998: ESPN request failed: 429"]
        assert result.any_failed is True          # --strict still sees it

    def test_a_nflreadpy_failure_still_gates_the_run(self, conn, ingest_run):
        """The rule routes on the registry source, so an nflreadpy season is unchanged."""
        _record_attempt(conn, "schedules", "schedules", "load_schedules", 2024,
                        IngestStatus.FAILED, "404 Not Found")

        result = ingest_run.result()
        assert result.failures == ["schedules 2024: 404 Not Found"]
        assert result.optional_source_failures == []

    def _run_broken_espn(self, monkeypatch, tmp_path, stamps):
        """One real run: ESPN 429s on every season, schedules loads, and stamping is watched."""
        self._break_espn(monkeypatch)
        _install_fake_nflreadpy(
            monkeypatch,
            load_schedules=lambda seasons, **kw: _small_df(season=seasons, team=["KC"]),
        )
        monkeypatch.setattr("nfl_mcp.migrate.stamp_if_current",
                            lambda conn: stamps.append(True) or True)
        path = str(tmp_path / "nflread.duckdb")
        result = run_ingest_datasets(["espn_records", "schedules"], start=1997, end=1999,
                                     skip_views=True, db_path=path)
        return result, path

    def test_the_seed_build_keeps_its_stamp_when_only_espn_failed(self, monkeypatch, tmp_path,
                                                                 capsys):
        stamps: list[bool] = []
        result, _path = self._run_broken_espn(monkeypatch, tmp_path, stamps)

        assert result.failures == []
        assert result.counts[IngestStatus.FAILED] == 2       # still counted
        assert stamps == [True]                              # so the stamp ran
        out = capsys.readouterr().out
        assert "Schema version stamped." in out
        assert "espn_records 1997: failed: ESPN request failed: 429" in out
        assert "espn_records 1998: failed: ESPN request failed: 429" in out
        assert "Ingestion complete!" in out

    def test_a_nflreadpy_failure_still_stops_the_stamp(self, monkeypatch, tmp_path, capsys):
        stamps: list[bool] = []
        _install_fake_nflreadpy(
            monkeypatch,
            load_schedules=lambda seasons, **kw: (_ for _ in ()).throw(
                RuntimeError("no route to host")),
        )
        monkeypatch.setattr("nfl_mcp.migrate.stamp_if_current",
                            lambda conn: stamps.append(True) or True)

        result = run_ingest_datasets(["schedules"], start=2021, end=2022, skip_views=True,
                                     db_path=str(tmp_path / "nflread.duckdb"))

        assert result.failures == ["schedules 2021: no route to host",
                                   "schedules 2022: no route to host"]
        assert stamps == []
        assert "Schema version stamped." not in capsys.readouterr().out

    def test_the_command_exits_zero_when_only_espn_failed(self, runner, monkeypatch):
        """The image build runs `--pre1999 --allow-failures`, and must ship its stamped seed."""
        _install_fake_nflreadpy(
            monkeypatch,
            load_schedules=lambda seasons, **kw: _small_df(season=seasons, team=["KC"]),
        )
        self._break_espn(monkeypatch)

        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--pre1999",
                                      "--skip-views", "--allow-failures",
                                      "--start", "1998", "--end", "1999"])

        assert result.exit_code == 0, result.output
        assert "espn_records 1998: failed" in result.output
        assert NOTHING_LOADED_MESSAGE not in result.output

    def test_strict_counts_an_espn_failure(self, runner, monkeypatch):
        _stub_run(monkeypatch,
                  _attempt("schedules", 2024, IngestStatus.LOADED),
                  _espn_attempt(1998))

        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--pre1999",
                                      "--skip-views", "--strict"])
        assert result.exit_code == INGEST_PARTIAL_EXIT, result.output
        assert "espn_records 1998: ESPN request failed: 429" in result.output

    def test_strict_with_allow_failures_still_counts_an_espn_failure(self, runner, monkeypatch):
        _stub_run(monkeypatch,
                  _attempt("schedules", 2024, IngestStatus.LOADED),
                  _espn_attempt(1998))

        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--pre1999",
                                      "--skip-views", "--strict", "--allow-failures"])
        assert result.exit_code == INGEST_PARTIAL_EXIT, result.output

    def test_a_run_that_asked_only_for_espn_and_failed_exits_three(self, runner, monkeypatch):
        """The run asked for one source and got nothing from it: that is nothing loaded."""
        _stub_run(monkeypatch, _espn_attempt(1998))

        result = runner.invoke(main, ["ingest", "--dataset", "espn_records", "--skip-views"])

        assert result.exit_code == INGEST_PARTIAL_EXIT, result.output
        assert NOTHING_LOADED_MESSAGE in result.output

    def test_the_failures_file_lists_only_the_gating_failures(self, runner, monkeypatch,
                                                              tmp_path):
        path = tmp_path / "failed.txt"
        _stub_run(monkeypatch,
                  _attempt("schedules", 2024, IngestStatus.FAILED),
                  _espn_attempt(1998))

        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--skip-views",
                                      "--failures-file", str(path)])

        assert result.exit_code == INGEST_PARTIAL_EXIT, result.output
        assert path.read_text() == "schedules 2024\n"

    def test_the_rule_is_one_function_on_the_registry_source(self):
        assert ing._gates_the_run("nflreadpy") is True
        assert ing._gates_the_run(SOURCE_ESPN) is False
        assert ing._attempt_source("espn_records") == SOURCE_ESPN
        assert ing._attempt_source("schedules") == REGISTRY["schedules"].source
        # A name no dataset carries reads as the source that gates the run.
        assert ing._attempt_source("not_in_the_registry") == "nflreadpy"


# ── 4. A run reports a load only after it committed ────────────────────────────


class TestNotesWaitForTheCommit:
    def test_a_note_inside_a_transaction_waits_for_the_commit(self, conn, ingest_run):
        with _transaction(conn):
            _record_loaded(conn, "schedules", "schedules", "load_schedules", 9, 2024)
            assert ingest_run.result().attempts == []   # it can still roll back

        assert [(a.dataset_id, a.season, a.status)
                for a in ingest_run.result().attempts] == [("schedules", 2024, "loaded")]

    def test_a_rolled_back_transaction_drops_its_notes(self, conn, ingest_run):
        with pytest.raises(RuntimeError, match="boom"):
            with _transaction(conn):
                _record_loaded(conn, "schedules", "schedules", "load_schedules", 9, 2024)
                raise RuntimeError("boom")

        assert ingest_run.result().attempts == []
        assert _status_of(conn, "schedules", 2024) is None

    def test_a_nested_transaction_hands_its_notes_to_the_outer_commit(self, conn, ingest_run):
        with _transaction(conn):
            with _transaction(conn):
                _record_loaded(conn, "schedules", "schedules", "load_schedules", 9, 2024)
            assert ingest_run.result().attempts == []   # the inner call committed nothing
        assert [a.status for a in ingest_run.result().attempts] == [IngestStatus.LOADED]

    def test_a_note_outside_a_transaction_is_delivered_at_once(self, conn, ingest_run):
        _record_attempt(conn, "schedules", "schedules", "load_schedules", 2024,
                        IngestStatus.FAILED, "404 Not Found")
        assert [a.status for a in ingest_run.result().attempts] == [IngestStatus.FAILED]

    def test_a_second_season_that_fails_to_commit_is_not_reported_as_loaded(
            self, monkeypatch, tmp_path):
        """The rolled-back season has no rows, and the run must not say it has."""
        real_insert = ing._insert_metadata_row
        stated = {"failed": False}

        def insert_fails_once_in_2022(connection, dataset_id, table_name, season, *rest):
            """Fail the season's metadata write, then let the failure record through."""
            if season == 2022 and not stated["failed"]:
                stated["failed"] = True
                raise RuntimeError("disk full")
            return real_insert(connection, dataset_id, table_name, season, *rest)

        _install_fake_nflreadpy(
            monkeypatch,
            load_schedules=lambda seasons, **kw: _small_df(season=seasons, team=["KC"]),
        )
        monkeypatch.setattr(ing, "_insert_metadata_row", insert_fails_once_in_2022)
        path = str(tmp_path / "nflread.duckdb")

        result = run_ingest_datasets(["schedules"], start=2021, end=2022, skip_views=True,
                                     db_path=path)

        assert [(a.season, str(a.status)) for a in result.attempts] == [
            (2021, "loaded"), (2022, "failed"),
        ]
        assert result.counts[IngestStatus.LOADED] == 1
        assert result.failures == ["schedules 2022: disk full"]

        check = duckdb.connect(path)
        try:
            assert check.execute(
                "SELECT season FROM schedules ORDER BY season").fetchall() == [(2021,)]
            assert _status_of(check, "schedules", 2022) == "failed"
        finally:
            check.close()

    def test_a_clean_run_reports_the_same_attempts_it_always_did(self, monkeypatch, tmp_path):
        _install_fake_nflreadpy(
            monkeypatch,
            load_schedules=lambda seasons, **kw: _small_df(season=seasons, team=["KC"]),
        )

        result = run_ingest_datasets(["schedules"], start=2021, end=2022, skip_views=True,
                                     db_path=str(tmp_path / "nflread.duckdb"))

        assert [(a.season, str(a.status)) for a in result.attempts] == [
            (2021, "loaded"), (2022, "loaded"),
        ]
        assert result.nothing_loaded is False


# ── 6. One spill directory for the write path and the read path ────────────────


class TestWritePathSpillDirectory:
    def test_the_env_var_moves_the_spill_off_the_database_volume(self, conn, monkeypatch,
                                                                 tmp_path):
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "256MB")
        monkeypatch.setenv("NFL_MCP_DUCKDB_TEMP_DIR", str(tmp_path / "tmp" / "spill"))

        _apply_duckdb_pragmas(conn, str(tmp_path / "db" / "nflread.duckdb"))

        assert conn.execute("SELECT current_setting('temp_directory')").fetchone()[0] == \
            os.path.join(str(tmp_path), "tmp", "spill")

    def test_the_database_folder_is_the_fallback(self, conn, monkeypatch, tmp_path):
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "256MB")
        monkeypatch.delenv("NFL_MCP_DUCKDB_TEMP_DIR", raising=False)

        _apply_duckdb_pragmas(conn, str(tmp_path / "nflread.duckdb"))

        assert conn.execute("SELECT current_setting('temp_directory')").fetchone()[0] == \
            str(tmp_path / ".duckdb_spill")

    def test_nothing_is_set_without_a_memory_cap(self, conn, monkeypatch, tmp_path):
        monkeypatch.delenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", raising=False)
        monkeypatch.setenv("NFL_MCP_DUCKDB_TEMP_DIR", str(tmp_path / "spill"))
        before = conn.execute("SELECT current_setting('temp_directory')").fetchone()[0]

        _apply_duckdb_pragmas(conn, str(tmp_path / "nflread.duckdb"))

        assert conn.execute("SELECT current_setting('temp_directory')").fetchone()[0] == before


# ── tools._data_as_of: only a row whose data is in the table says how fresh it is ──

class TestDataAsOfIgnoresAttemptsThatStoredNothing:
    @pytest.fixture(autouse=True)
    def clean_cache(self, monkeypatch):
        from nfl_mcp import tools
        monkeypatch.setattr(tools, "_AS_OF_CACHE", {})

    def _database(self, tmp_path, monkeypatch, ddl, rows):
        import duckdb
        path = tmp_path / "asof.duckdb"
        conn = duckdb.connect(str(path))
        conn.execute(ddl)
        for row in rows:
            conn.execute(
                "INSERT INTO _ingest_metadata VALUES (" + ", ".join("?" * len(row)) + ")", row)
        conn.close()
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
        return path

    def test_a_later_failed_attempt_does_not_move_the_time(self, tmp_path, monkeypatch):
        import datetime as dt
        from nfl_mcp import tools
        from nfl_mcp.status import IngestStatus
        loaded = dt.datetime(2024, 1, 1, 12, tzinfo=dt.timezone.utc)
        failed = dt.datetime(2025, 6, 1, 12, tzinfo=dt.timezone.utc)
        path = self._database(
            tmp_path, monkeypatch,
            "CREATE TABLE _ingest_metadata (table_name VARCHAR, loaded_at TIMESTAMPTZ, status VARCHAR)",
            [("plays", loaded, IngestStatus.LOADED.value),
             ("plays", failed, IngestStatus.FAILED.value),
             ("rosters", failed, IngestStatus.EMPTY.value)],
        )
        assert tools._data_as_of(str(path)) == {"plays": "2024-01-01T12:00:00.000000Z"}

    def test_a_database_without_the_status_column_counts_every_row(self, tmp_path, monkeypatch):
        import datetime as dt
        from nfl_mcp import tools
        loaded = dt.datetime(2024, 1, 1, 12, tzinfo=dt.timezone.utc)
        path = self._database(
            tmp_path, monkeypatch,
            "CREATE TABLE _ingest_metadata (table_name VARCHAR, loaded_at TIMESTAMPTZ)",
            [("plays", loaded)],
        )
        assert tools._data_as_of(str(path)) == {"plays": "2024-01-01T12:00:00.000000Z"}
