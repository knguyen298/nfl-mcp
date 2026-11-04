"""Unit tests for the read-only DuckDB layer — no network, temp DuckDB files."""

import os
import threading

import duckdb
import pytest

from nfl_mcp.database import (
    _connection_config,
    connection_config_problem,
    get_db_connection,
)

_ENV_VARS = (
    "NFL_MCP_DUCKDB_MEMORY_LIMIT",
    "NFL_MCP_DUCKDB_THREADS",
    "NFL_MCP_DUCKDB_TEMP_DIR",
)


@pytest.fixture
def db_file(tmp_path, monkeypatch):
    """A tiny on-disk DuckDB the read-only layer can open, pointed at by env."""
    path = tmp_path / "nflread.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE plays AS SELECT range AS play_id, 2020 + (range % 5)::int AS season, "
        "CASE WHEN range % 2 = 0 THEN 'KC' ELSE 'LV' END AS team, "
        "range * 1.5 AS epa FROM range(400)"
    )
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return path


class TestConnectionConfig:
    def test_locks_external_access_by_default(self, monkeypatch):
        for var in _ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        config = _connection_config()
        assert config["enable_external_access"] is False
        assert config["autoinstall_known_extensions"] is False
        assert config["autoload_known_extensions"] is False
        assert config["lock_configuration"] is True
        assert "memory_limit" not in config
        assert "threads" not in config
        assert "temp_directory" not in config

    def test_env_vars_land_in_the_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "256MB")
        monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", "1")
        monkeypatch.setenv("NFL_MCP_DUCKDB_TEMP_DIR", str(tmp_path / "spill"))
        config = _connection_config()
        assert config["memory_limit"] == "256MB"
        assert config["threads"] == 1
        assert config["temp_directory"] == str(tmp_path / "spill")


class TestGetDbConnection:
    def test_applies_env_caps_without_set(self, db_file, monkeypatch, tmp_path):
        spill = tmp_path / "spill"
        spill.mkdir()
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "64MB")
        monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", "1")
        monkeypatch.setenv("NFL_MCP_DUCKDB_TEMP_DIR", str(spill))
        with get_db_connection() as conn:
            assert conn.execute("SELECT count(*) FROM plays").fetchone()[0] == 400
            assert conn.execute("SELECT current_setting('threads')").fetchone()[0] == 1
            assert "MiB" in conn.execute("SELECT current_setting('memory_limit')").fetchone()[0]
            assert conn.execute("SELECT current_setting('temp_directory')").fetchone()[0].startswith(
                str(spill)
            )

    def test_closes_the_connection_on_exit(self, db_file):
        with get_db_connection() as conn:
            pass
        with pytest.raises(duckdb.ConnectionException):
            conn.execute("SELECT 1")

    def test_read_only_rejects_writes(self, db_file):
        with get_db_connection() as conn, pytest.raises(duckdb.Error):
            conn.execute("CREATE TABLE t AS SELECT 1")

    def test_spills_to_temp_directory_with_external_access_off(self, db_file, monkeypatch, tmp_path):
        """A 400k-row sort under a 16MB limit does not fit in memory, so DuckDB must
        write spill files. That must still work with external access locked off.
        DuckDB deletes the spill files when the connection closes, so the test
        watches the directory while the query runs."""
        spill = tmp_path / "spill"
        spill.mkdir()
        big = tmp_path / "big.duckdb"
        con = duckdb.connect(str(big))
        con.execute("CREATE TABLE t AS SELECT range a, md5(range::VARCHAR) s FROM range(400000)")
        con.close()
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(big))
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "16MB")
        monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", "1")
        monkeypatch.setenv("NFL_MCP_DUCKDB_TEMP_DIR", str(spill))

        stop = threading.Event()
        spilled = []

        def watch():
            while not stop.is_set():
                if os.listdir(spill):
                    spilled.append(True)
                stop.wait(0.005)

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            with get_db_connection() as conn:
                total = conn.execute(
                    "SELECT count(*) FROM (SELECT * FROM t ORDER BY s)"
                ).fetchone()[0]
        finally:
            stop.set()
            watcher.join(timeout=2)

        assert total == 400000
        assert spilled, "DuckDB never wrote a spill file, so the query proved nothing"


class TestConfigurationProblems:
    @pytest.mark.parametrize("var,value,named", [
        ("NFL_MCP_DUCKDB_THREADS", "two", "NFL_MCP_DUCKDB_THREADS='two'"),
        ("NFL_MCP_DUCKDB_THREADS", "0", "NFL_MCP_DUCKDB_THREADS='0'"),
        ("NFL_MCP_DUCKDB_MEMORY_LIMIT", "lots", "NFL_MCP_DUCKDB_MEMORY_LIMIT='lots'"),
    ])
    def test_a_bad_value_is_named(self, monkeypatch, var, value, named):
        for other in _ENV_VARS:
            monkeypatch.delenv(other, raising=False)
        monkeypatch.setenv(var, value)
        assert connection_config_problem().startswith(named)

    def test_good_values_pass(self, monkeypatch, tmp_path):
        monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", "2")
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "512MB")
        monkeypatch.setenv("NFL_MCP_DUCKDB_TEMP_DIR", str(tmp_path))
        assert connection_config_problem() is None

    def test_a_missing_database_says_to_ingest_and_creates_nothing(self, monkeypatch, tmp_path):
        target = tmp_path / "absent" / "nflread.duckdb"
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(target))
        with pytest.raises(duckdb.IOException, match="nfl-mcp ingest"), get_db_connection():
            pass
        assert list(tmp_path.iterdir()) == []
