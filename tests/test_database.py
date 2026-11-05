"""Unit tests for the DuckDB connection layer — no network, temp DuckDB files."""

import duckdb
import pytest

from nfl_mcp.database import _memory_limit_problem, connection_config_problem, get_db_connection


@pytest.fixture
def db_file(tmp_path):
    """A tiny on-disk DuckDB the runtime layer can open read-only."""
    path = tmp_path / "nflread.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE plays AS SELECT 1 AS x")
    con.close()
    return path


class TestConnectionConfig:
    def test_applies_all_env_caps(self, db_file, tmp_path, monkeypatch):
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(db_file))
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "256MB")
        monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", "1")
        monkeypatch.setenv("NFL_MCP_DUCKDB_TEMP_DIR", str(tmp_path / "spill"))
        with get_db_connection() as conn:
            assert conn.execute("SELECT current_setting('threads')").fetchone()[0] == 1
            mem = conn.execute("SELECT current_setting('memory_limit')").fetchone()[0]
            assert "MiB" in mem or "MB" in mem
            spill = conn.execute("SELECT current_setting('temp_directory')").fetchone()[0]
            assert spill.endswith("spill")

    def test_defaults_when_env_unset(self, db_file, monkeypatch):
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(db_file))
        for var in (
            "NFL_MCP_DUCKDB_MEMORY_LIMIT",
            "NFL_MCP_DUCKDB_THREADS",
            "NFL_MCP_DUCKDB_TEMP_DIR",
        ):
            monkeypatch.delenv(var, raising=False)
        assert connection_config_problem() is None
        with get_db_connection() as conn:
            default_threads = conn.execute("SELECT current_setting('threads')").fetchone()[0]
            ref = duckdb.connect(":memory:")
            try:
                assert default_threads == ref.execute(
                    "SELECT current_setting('threads')"
                ).fetchone()[0]
            finally:
                ref.close()


class TestMemoryProbeCache:
    def test_probe_runs_once_per_value(self, db_file, monkeypatch):
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(db_file))
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "300MB")
        _memory_limit_problem.cache_clear()
        real_connect = duckdb.connect
        probes = []

        def counting(database, *args, **kwargs):
            if database == ":memory:":
                probes.append(kwargs.get("config"))
            return real_connect(database, *args, **kwargs)

        monkeypatch.setattr(duckdb, "connect", counting)
        for _ in range(2):
            with get_db_connection() as conn:
                conn.execute("SELECT 1")
        assert probes == [{"memory_limit": "300MB"}]  # one probe for two opens


class TestGetDbConnection:
    def test_yields_readonly_connection_with_pragmas(self, db_file, monkeypatch):
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(db_file))
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "256MB")
        monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", "1")
        with get_db_connection() as conn:
            assert conn.execute("SELECT count(*) FROM plays").fetchone()[0] == 1
            assert conn.execute("SELECT current_setting('threads')").fetchone()[0] == 1
        with pytest.raises(Exception):
            conn.execute("SELECT 1")

    def test_read_only_rejects_writes(self, db_file, monkeypatch):
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(db_file))
        with get_db_connection() as conn:
            with pytest.raises(Exception):
                conn.execute("CREATE TABLE t AS SELECT 1")
