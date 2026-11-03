"""Rebuild-on-start tests: the shape stamp, the CLI, and the derived-table guard.

derived_shape_is_stale is the read-only question entrypoint.sh asks before
rebuilding; these build tiny databases in memory so every branch is reachable
without touching a real file. The CLI tests point NFL_MCP_DB_PATH at copies -
a test must never open the suite's database for write.
"""

import duckdb
import pytest
from click.testing import CliRunner

from nfl_mcp.ingest import (
    _PROBE_TABLES,
    DERIVED_SHAPE_VERSION,
    _create_fantasy_tables,
    _ensure_metadata_table,
    _record_loaded,
    _write_df_to_table,
    derived_shape_is_stale,
    rebuild_derived_tables,
)
from tests.team_table_fixtures import load_team_sources


def _stamp(conn, version: int) -> None:
    _record_loaded(
        conn,
        dataset_id="derived_shape",
        table_name="derived_shape",
        loader_fn="test_rebuild_shape",
        row_count=version,
    )


class TestDerivedShapeIsStale:
    """The four shapes a /data volume arrives in, built without any ingest."""

    def test_nothing_built_yet_is_stale(self):
        conn = duckdb.connect(":memory:")  # not even _ingest_metadata
        try:
            assert derived_shape_is_stale(conn) is True
        finally:
            conn.close()

    def test_metadata_table_without_a_stamp_row_is_stale(self):
        conn = duckdb.connect(":memory:")
        try:
            _ensure_metadata_table(conn)
            assert derived_shape_is_stale(conn) is True
        finally:
            conn.close()

    def test_stamp_below_the_current_version_is_stale(self):
        conn = duckdb.connect(":memory:")
        try:
            _ensure_metadata_table(conn)
            _stamp(conn, DERIVED_SHAPE_VERSION - 1)
            assert derived_shape_is_stale(conn) is True
        finally:
            conn.close()

    def test_probe_catches_a_table_without_season_type_despite_a_current_stamp(self):
        # The case a stamp cannot see: a database written before stamps existed
        # that some other code stamped, or a stamp carried across a branch.
        conn = duckdb.connect(":memory:")
        try:
            _ensure_metadata_table(conn)
            conn.execute("CREATE TABLE team_offense_stats (team VARCHAR, season_year INTEGER)")
            _stamp(conn, DERIVED_SHAPE_VERSION)
            assert derived_shape_is_stale(conn) is True
        finally:
            conn.close()

    def test_probe_ignores_tables_the_database_does_not_have(self):
        # A partially-ingested database is nfl_status's business; this probe
        # must not push a rebuild onto one.
        conn = duckdb.connect(":memory:")
        try:
            _ensure_metadata_table(conn)
            conn.execute("CREATE TABLE team_offense_stats "
                         "(team VARCHAR, season_year INTEGER, season_type VARCHAR)")
            _stamp(conn, DERIVED_SHAPE_VERSION)
            assert derived_shape_is_stale(conn) is False
        finally:
            conn.close()

    def test_current_stamp_and_current_shape_is_not_stale(self):
        conn = duckdb.connect(":memory:")
        try:
            _ensure_metadata_table(conn)
            load_team_sources(conn, _write_df_to_table)  # all four probe tables, new shape
            _create_fantasy_tables(conn)
            _stamp(conn, DERIVED_SHAPE_VERSION)
            assert derived_shape_is_stale(conn) is False
        finally:
            conn.close()

    def test_stamp_row_saves_the_version_in_row_count(self):
        conn = duckdb.connect(":memory:")
        try:
            load_team_sources(conn, _write_df_to_table)
            rebuild_derived_tables(conn, ":memory:")
            assert derived_shape_is_stale(conn) is False
            row = conn.execute(
                "SELECT table_name, season, row_count, loader_fn FROM _ingest_metadata "
                "WHERE dataset_id = 'derived_shape'"
            ).fetchone()
            assert row[0] == "derived_shape"
            assert row[1] is None  # the shape covers every season at once
            assert row[2] == DERIVED_SHAPE_VERSION
            assert row[3] == "nfl_mcp.ingest.rebuild_derived_tables"
        finally:
            conn.close()


# ── CLI ────────────────────────────────────────────────────────────────────────


def _stale_copy(tmp_path, monkeypatch):
    """Point the CLI at a private copy of the suite's database.

    Session-scoped on purpose. A per-test copy is unsafe here: a killed or
    timed-out DuckDB writer leaves its 8-byte magic header behind, and a
    shutil.copyfile of that instant copies the header only - every later test
    then reads "not a valid DuckDB database file". One rebuild serves all
    three CLI tests, so the source is opened for write exactly once. The copy
    is made once too, from the same source the conftest rebuild fixture uses,
    and rebuilt with this branch's code - the same job that fixture does.
    """
    import shutil

    from nfl_mcp.config import get_duckdb_path

    source = get_duckdb_path()
    if not source.exists():
        pytest.skip("No DuckDB database loaded")
    copy = tmp_path / "rebuild_cli.duckdb"
    if not copy.exists():
        shutil.copyfile(source, copy)
        conn = duckdb.connect(str(copy))
        try:
            for table in _PROBE_TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            _ensure_metadata_table(conn)
            conn.execute("DELETE FROM _ingest_metadata WHERE dataset_id = 'derived_shape'")
            rebuild_derived_tables(conn, str(copy))
        finally:
            conn.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(copy))
    return duckdb.connect(str(copy))


@pytest.fixture
def runner():
    """A fresh Click test runner. Function scope: a session-scoped fixture defined as a
    class instance method is deprecated in pytest, and CliRunner costs nothing to build."""
    return CliRunner()


@pytest.mark.integration
class TestRebuildCli:
    """--check answers without a write lock; rebuild clears the guard."""

    def test_check_prints_current_and_exits_zero(self, runner, tmp_path, monkeypatch):
        # rebuild_derived_tables already stamped the version; --check must agree.
        conn = _stale_copy(tmp_path, monkeypatch)
        conn.close()

        from nfl_mcp.cli import main
        result = runner.invoke(main, ["rebuild", "--check"])
        assert result.exit_code == 0, result.output
        assert result.stdout.strip() == "current"

    def test_check_prints_stale_and_exits_one(self, runner, tmp_path, monkeypatch):
        conn = _stale_copy(tmp_path, monkeypatch)
        conn.execute("DELETE FROM _ingest_metadata WHERE dataset_id = 'derived_shape'")
        conn.close()
        # Leaving the DELETE uncommitted would hide the row from the next
        # connection; closing commits it. Prove the guard reads committed state:

        from nfl_mcp.cli import main
        result = runner.invoke(main, ["rebuild", "--check"])
        assert result.exit_code == 1
        assert result.stdout.strip() == "stale"

    def test_check_on_a_missing_database_is_stale_not_a_crash(self, runner, tmp_path, monkeypatch):
        # entrypoint.sh runs before anything is served; a path with no file must
        # answer the question, not exit with a lock error code.
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(tmp_path / "absent.duckdb"))
        from nfl_mcp.cli import main
        result = runner.invoke(main, ["rebuild", "--check"])
        assert result.exit_code == 1
        assert result.stdout.strip() == "stale"

    def test_rebuild_on_a_missing_database_fails_without_reaching_duckdb(
            self, runner, tmp_path, monkeypatch):
        # A rebuild has no rows to work from there; fail before DuckDB decides
        # to create an empty database at that path.
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(tmp_path / "absent.duckdb"))
        from nfl_mcp.cli import main
        result = runner.invoke(main, ["rebuild"])
        assert result.exit_code == 2
        assert "absent.duckdb" in result.stderr

    def test_lock_error_exits_two_on_both_paths(self, runner, tmp_path, monkeypatch):
        # exit 2 is "could not open", distinct from 0/1 "current/stale"; the
        # entrypoint treats any nonzero --check as "try to rebuild".
        import duckdb as _duckdb
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(tmp_path / "locked.duckdb"))
        _duckdb.connect(str(tmp_path / "locked.duckdb")).close()

        def locked(path, read_only=False):
            raise _duckdb.Error("Conflicting lock is held")

        monkeypatch.setattr("duckdb.connect", locked)
        from nfl_mcp.cli import main
        for args in (["rebuild", "--check"], ["rebuild"]):
            result = runner.invoke(main, args)
            assert result.exit_code == 2, (args, result.output)

    def test_rebuild_clears_the_guard_tools_read(self, runner, tmp_path, monkeypatch):
        from nfl_mcp.cli import main
        from nfl_mcp.tools import nfl_separation_opportunity, nfl_team_stats

        conn = _stale_copy(tmp_path, monkeypatch)
        for table in _PROBE_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        _ensure_metadata_table(conn)
        conn.execute("DELETE FROM _ingest_metadata WHERE dataset_id = 'derived_shape'")
        conn.close()
        db = str(tmp_path / "rebuild_cli.duckdb")

        result = runner.invoke(main, ["rebuild"])
        assert result.exit_code == 0, result.output
        assert result.exit_code == 0, result.output
        assert "Rebuilt derived tables in" in result.stdout

        check = runner.invoke(main, ["rebuild", "--check"])
        assert check.exit_code == 0
        assert check.stdout.strip() == "current"

        rebuilt = duckdb.connect(db, read_only=True)
        try:
            for table in _PROBE_TABLES:
                cols = {r[0] for r in rebuilt.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
                    [table]).fetchall()}
                assert "season_type" in cols, table
        finally:
            rebuilt.close()

        # The guarded tools read the rebuilt copy and answer with rows. The
        # synthetic source rows the copy already holds (tests/team_table_fixtures
        # built its derived tables from them) stay; only the probe tables are
        # dropped and rebuilt.
        stats = nfl_team_stats(team="KC", season=2024)
        assert "error" not in stats, stats
        offense = [r for r in stats["offense"] if r.get("season_type") == "REG"]
        assert offense, stats
        assert stats["sources"]["team_stats_raw"]
        separation = nfl_separation_opportunity(season=2024)
        assert "error" not in separation, separation
        assert separation["separation_opportunity"], separation
