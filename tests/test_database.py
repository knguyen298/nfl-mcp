"""Unit tests for the DuckDB connection layer — no network, temp DuckDB files."""

import threading
import warnings

import duckdb
import pytest

from nfl_mcp.database import (
    _lock_external_access,
    _memory_limit_problem,
    _runtime_config,
    connection_config_problem,
    get_db_connection,
)

_ENV_VARS = (
    "NFL_MCP_DUCKDB_MEMORY_LIMIT",
    "NFL_MCP_DUCKDB_THREADS",
    "NFL_MCP_DUCKDB_TEMP_DIR",
)

# A sort that needs more than 40MB: it fails with no spill directory, so it passes only
# when DuckDB can write its temporary files.
_SPILLING_SORT = (
    "SELECT count(*) FROM (SELECT md5(range::varchar) || md5(range::varchar) AS m "
    "FROM range(2000000) ORDER BY m)"
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def db_file(tmp_path, monkeypatch):
    """A tiny on-disk DuckDB the runtime layer opens read-only, set as NFL_MCP_DB_PATH."""
    path = tmp_path / "nflread.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE plays AS SELECT 1 AS x")
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


@pytest.fixture
def outside(tmp_path):
    """Files outside the database that a query must never read. Returns their directory."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.csv").write_text("a,b\nTOPSECRET,2\n")
    con = duckdb.connect(":memory:")
    con.execute(f"COPY (SELECT 'TOPSECRET' AS s) TO '{outside / 'secret.parquet'}' (FORMAT parquet)")
    con.execute(f"ATTACH '{outside / 'other.duckdb'}' AS other")
    con.execute("CREATE TABLE other.t AS SELECT 'TOPSECRET' AS s")
    con.close()
    return outside


class TestRuntimeConfig:
    def test_applies_all_env_caps(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "256MB")
        monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", "1")
        monkeypatch.setenv("NFL_MCP_DUCKDB_TEMP_DIR", str(tmp_path / "spill"))
        config = _runtime_config()
        assert config["memory_limit"] == "256MB"
        assert config["threads"] == 1
        assert config["temp_directory"] == str(tmp_path / "spill")
        conn = duckdb.connect(":memory:", config=config)
        assert conn.execute("SELECT current_setting('threads')").fetchone()[0] == 1
        mem = conn.execute("SELECT current_setting('memory_limit')").fetchone()[0]
        assert "MiB" in mem or "MB" in mem
        spill = conn.execute("SELECT current_setting('temp_directory')").fetchone()[0]
        assert spill.endswith("spill")
        conn.close()

    def test_only_the_access_settings_when_env_unset(self):
        assert _runtime_config() == {
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
            "python_enable_replacements": False,
        }


class TestConnectionConfig:
    """The connect-time config, opened through get_db_connection (from main)."""

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
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "256MB")
        monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", "1")
        with get_db_connection() as conn:
            assert conn.execute("SELECT count(*) FROM plays").fetchone()[0] == 1
            assert conn.execute("SELECT current_setting('threads')").fetchone()[0] == 1
        with pytest.raises(Exception):
            conn.execute("SELECT 1")

    def test_read_only_rejects_writes(self, db_file):
        with get_db_connection() as conn:
            with pytest.raises(Exception):
                conn.execute("CREATE TABLE t AS SELECT 1")

    def test_external_access_is_off_and_locked(self, db_file):
        with get_db_connection() as conn:
            settings = dict(conn.execute(
                "SELECT name, value FROM duckdb_settings() WHERE name IN ("
                "'enable_external_access', 'lock_configuration', "
                "'autoload_known_extensions', 'autoinstall_known_extensions')"
            ).fetchall())
        assert settings == {
            "enable_external_access": "false",
            "lock_configuration": "true",
            "autoload_known_extensions": "false",
            "autoinstall_known_extensions": "false",
        }


class TestReadPathTouchesNothing:
    """A read-only query must not create anything next to a database that is not there."""

    def test_a_missing_database_creates_no_directory(self, tmp_path, monkeypatch):
        missing = tmp_path / "data" / "nflread.duckdb"
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(missing))

        with pytest.raises(duckdb.IOException):
            with get_db_connection():
                pass

        assert not missing.parent.exists()
        assert not missing.exists()

    def test_a_missing_database_is_the_error_the_tools_layer_reports(
        self, tmp_path, monkeypatch
    ):
        """tools.py classifies this exception as DB_NOT_FOUND; the text must stay recognizable."""
        from nfl_mcp import tools
        from nfl_mcp.status import ErrorCode

        missing = tmp_path / "data" / "nflread.duckdb"
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(missing))

        with pytest.raises(duckdb.IOException) as raised:
            with get_db_connection():
                pass

        assert tools._DB_NOT_FOUND_TEXT in str(raised.value)
        assert tools._error_from_exception(raised.value)["code"] is ErrorCode.DB_NOT_FOUND


class TestSpillDirectory:
    """DuckDB's spill directory is the one thing a read may create, and it is optional."""

    def test_the_configured_directory_exists_before_a_query_runs(self, db_file, tmp_path, monkeypatch):
        spill = tmp_path / "spill"
        monkeypatch.setenv("NFL_MCP_DUCKDB_TEMP_DIR", str(spill))

        with get_db_connection():
            assert spill.is_dir()

    def test_a_directory_that_cannot_be_created_does_not_fail_a_query(self, db_file, tmp_path, monkeypatch):
        # A path DuckDB cannot make a directory under: what a read-only filesystem looks
        # like from here. The query must still answer without an explicit temp_directory.
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory")
        spill = blocked / "spill"
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "40MB")
        monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", "1")
        monkeypatch.setenv("NFL_MCP_DUCKDB_TEMP_DIR", str(spill))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with get_db_connection() as conn:
                count = conn.execute("SELECT count(*) FROM range(10)").fetchone()[0]
                temp = conn.execute("SELECT current_setting('temp_directory')").fetchone()[0]

        assert count == 10
        assert not spill.exists()
        assert str(spill) not in temp
        assert any("spill directory" in str(w.message) for w in caught)


class TestExternalAccessRefused:
    """Every way a read-only connection could reach a file or the network is refused."""

    @pytest.mark.parametrize("template", [
        "SELECT * FROM read_text('{d}/secret.csv')",
        "SELECT * FROM read_blob('{d}/secret.csv')",
        "SELECT * FROM read_csv('{d}/secret.csv')",
        "SELECT * FROM read_csv_auto('{d}/secret.csv')",
        "SELECT * FROM sniff_csv('{d}/secret.csv')",
        "SELECT * FROM '{d}/secret.csv'",
        "SELECT * FROM read_parquet('{d}/secret.parquet')",
        "SELECT * FROM parquet_scan('{d}/secret.parquet')",
        "SELECT * FROM parquet_metadata('{d}/secret.parquet')",
        "SELECT * FROM '{d}/secret.parquet'",
        "SELECT * FROM read_json('{d}/secret.csv')",
        "SELECT * FROM read_duckdb('{d}/other.duckdb')",
        "SELECT * FROM glob('{d}/*')",
        "SELECT * FROM read_csv('https://example.invalid/x.csv')",
        "SELECT * FROM read_parquet('http://127.0.0.1:9/x.parquet')",
        "ATTACH '{d}/other.duckdb' AS other (READ_ONLY)",
        "COPY plays TO '{d}/out.csv'",
        "COPY (SELECT 1) TO '{d}/out.parquet' (FORMAT parquet)",
        "EXPORT DATABASE '{d}/export'",
        "INSTALL httpfs",
        "FORCE INSTALL httpfs",
        "LOAD httpfs",
        "LOAD '{d}/evil.duckdb_extension'",
    ])
    def test_vector_is_refused(self, db_file, outside, template):
        with get_db_connection() as conn:
            with pytest.raises(duckdb.Error) as exc:
                conn.execute(template.format(d=outside)).fetchall()
        assert "TOPSECRET" not in str(exc.value)
        assert not (outside / "out.csv").exists()
        assert not (outside / "out.parquet").exists()
        assert not (outside / "export").exists()

    def test_database_stays_queryable_after_a_refusal(self, db_file, outside):
        with get_db_connection() as conn:
            with pytest.raises(duckdb.Error):
                conn.execute(f"SELECT * FROM read_text('{outside}/secret.csv')")
            assert conn.execute("SELECT count(*) FROM plays").fetchone()[0] == 1


class TestConfigurationLocked:
    @pytest.mark.parametrize("statement", [
        "SET enable_external_access = true",
        "SET GLOBAL enable_external_access = true",
        "RESET enable_external_access",
        "PRAGMA enable_external_access = true",
        "SET lock_configuration = false",
        "RESET lock_configuration",
        "SET autoload_known_extensions = true",
        "SET autoinstall_known_extensions = true",
        "SET allowed_directories = ['/']",
        "SET python_enable_replacements = true",
        "SET memory_limit = '64GB'",
        "SET temp_directory = '/'",
    ])
    def test_a_later_set_or_reset_is_refused(self, db_file, statement):
        with get_db_connection() as conn:
            with pytest.raises(duckdb.Error):
                conn.execute(statement)
            assert conn.execute("SELECT current_setting('enable_external_access')").fetchone()[0] is False
            assert conn.execute("SELECT current_setting('lock_configuration')").fetchone()[0] is True

    def test_concurrent_connections_share_the_locked_instance(self, db_file):
        # DuckDB hands every open connection to one file the same database instance, so
        # the second connection must find the lock in place instead of failing to set it.
        with get_db_connection() as first, get_db_connection() as second:
            for conn in (first, second):
                assert conn.execute("SELECT count(*) FROM plays").fetchone()[0] == 1
                assert conn.execute("SELECT current_setting('enable_external_access')").fetchone()[0] is False

    def test_a_fresh_instance_is_locked_again(self, db_file):
        with get_db_connection():
            pass
        with get_db_connection() as conn:
            assert conn.execute("SELECT current_setting('lock_configuration')").fetchone()[0] is True
            assert conn.execute("SELECT current_setting('enable_external_access')").fetchone()[0] is False

    def test_threads_opening_at_once_all_get_a_locked_connection(self, db_file):
        results, errors = [], []

        def worker():
            try:
                with get_db_connection() as conn:
                    results.append(
                        conn.execute("SELECT current_setting('enable_external_access')").fetchone()[0]
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert results == [False] * 8

    def test_refuses_to_join_an_instance_opened_with_weaker_settings(self, db_file):
        # Something else in the process opened the database with external access still on
        # and locked it. DuckDB refuses a second connection whose config differs, so
        # get_db_connection never hands out a connection to that instance.
        other = duckdb.connect(str(db_file), read_only=True, config=_runtime_config())
        other.execute("SET lock_configuration = true")
        try:
            with pytest.raises(duckdb.ConnectionException, match="different configuration"):
                with get_db_connection():
                    pass
        finally:
            other.close()

    def test_the_access_check_refuses_a_connection_with_access_still_on(self):
        # The check reads the setting back from the instance instead of trusting the config.
        conn = duckdb.connect(":memory:")
        try:
            with pytest.raises(RuntimeError, match="external access enabled"):
                _lock_external_access(conn)
        finally:
            conn.close()


class TestHeavyQueriesStillWork:
    def test_sort_spills_to_the_configured_temp_directory(self, db_file, tmp_path, monkeypatch):
        spill = tmp_path / "spill"
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "40MB")
        monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", "1")
        monkeypatch.setenv("NFL_MCP_DUCKDB_TEMP_DIR", str(spill))
        with get_db_connection() as conn:
            assert conn.execute(_SPILLING_SORT).fetchone()[0] == 2_000_000
            assert spill.is_dir()

    def test_the_same_sort_fails_without_a_spill_directory(self, tmp_path):
        # Shows the sort above needs disk: with spilling off it runs out of memory.
        conn = duckdb.connect(":memory:", config={"memory_limit": "40MB", "threads": 1, "temp_directory": ""})
        try:
            with pytest.raises(duckdb.OutOfMemoryException):
                conn.execute(_SPILLING_SORT).fetchone()
        finally:
            conn.close()

    def test_sort_spills_to_the_default_temp_directory(self, db_file, monkeypatch):
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "40MB")
        monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", "1")
        with get_db_connection() as conn:
            assert conn.execute(_SPILLING_SORT).fetchone()[0] == 2_000_000

    def test_aggregation_and_join_on_the_database(self, db_file):
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT count(*), sum(p.x) FROM plays p JOIN range(10) r ON true GROUP BY p.x"
            ).fetchone()
        assert row == (10, 10)


def test_write_connection_to_another_file_is_not_locked(db_file, tmp_path):
    # ingest.py opens its own connection with duckdb.connect(path). The lock is a setting
    # of the serve process's read-only instance, so it must not follow that connection.
    with get_db_connection():
        conn = duckdb.connect(str(tmp_path / "ingest.duckdb"))
        try:
            assert conn.execute("SELECT current_setting('enable_external_access')").fetchone()[0] is True
            conn.execute("CREATE TABLE t AS SELECT 1 AS x")
        finally:
            conn.close()
