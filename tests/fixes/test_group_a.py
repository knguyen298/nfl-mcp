"""Group A: connection hardening and dependencies.

Spec: PLAN_FIX_GROUPS_MAIN_2026-11-04.md section A; evidence AUDIT_MAIN_2026-11-04.md H1, H2, M2, M8, L1, L5.

Interface contract (beyond the shared brief)
--------------------------------------------
- nfl_mcp.database.get_db_connection() stays a context manager yielding a read-only
  DuckDB connection. It opens with config containing enable_external_access=False,
  autoinstall_known_extensions=False, autoload_known_extensions=False,
  lock_configuration=True, plus memory_limit / threads / temp_directory taken from
  NFL_MCP_DUCKDB_MEMORY_LIMIT / NFL_MCP_DUCKDB_THREADS / NFL_MCP_DUCKDB_TEMP_DIR when set.
  No env value is ever formatted into SQL.
- File and network access through the connection raises duckdb.PermissionException
  (verified on duckdb 1.5.5 with the config above). SET raises a duckdb.Error
  (InvalidInputException "configuration has been locked").
- get_db_connection() never creates directories or files. A missing DB file raises
  FileNotFoundError or a duckdb.Error (IOException); the message contains the path.
- New function nfl_mcp.database.connection_config_problem() -> str | None. Reads the
  three env vars. Returns None when all are unset or valid; otherwise a message that
  names the offending env var. Valid threads: a positive integer string. Valid memory:
  a number followed by a DuckDB unit (KB, MB, GB, TB, KiB, MiB, GiB, TiB, case-insensitive).
- `nfl-mcp serve` calls connection_config_problem() before create_app()/uvicorn.run();
  on a problem it exits non-zero and prints the message (stdout or stderr).
- `nfl-mcp doctor` opens the DB through nfl_mcp.database.get_db_connection, resolved at
  call time (import inside the function or `from . import database`). The test patches
  both nfl_mcp.database.get_db_connection and nfl_mcp.cli.get_db_connection (if present).
- Decision: CREATE TEMP TABLE is allowed on the connection. DuckDB allows it on a
  read-only DB and the locked config does not block it (verified). Rejecting it in
  nfl_query is group B's job (statement type). Temp tables do not outlive the connection.
- pytz is a declared runtime dependency in pyproject.toml [project].dependencies.

Verified on a plain duckdb.connect(path, read_only=True), duckdb 1.5.5: DuckDB itself
blocks CREATE TABLE, INSERT, ATTACH ':memory:', ATTACH of a missing file and
SET enable_external_access=true. It does NOT block ATTACH of an existing DuckDB file
(attaches and reads it), so that case is xfail here.
"""

from __future__ import annotations

import contextlib
import datetime as dt
from pathlib import Path

import duckdb
import pytest
from click.testing import CliRunner

from nfl_mcp import cli, database
from nfl_mcp.tools import nfl_query

ENV_VARS = ("NFL_MCP_DUCKDB_MEMORY_LIMIT", "NFL_MCP_DUCKDB_THREADS", "NFL_MCP_DUCKDB_TEMP_DIR")
REPO = Path(__file__).resolve().parents[2]
REAL_DB = Path("/tmp/grpE_v3.duckdb")


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _build_db(path: Path, plays_rows: list[tuple]) -> Path:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE plays (season INTEGER, week INTEGER, posteam VARCHAR, yards_gained INTEGER)")
    con.executemany("INSERT INTO plays VALUES (?, ?, ?, ?)", plays_rows)
    con.execute("CREATE TABLE tz_events (id INTEGER, ts TIMESTAMP WITH TIME ZONE)")
    con.execute("SET TimeZone='UTC'")
    con.execute("INSERT INTO tz_events VALUES (1, TIMESTAMPTZ '2024-09-08 17:00:00+00'), "
                "(2, TIMESTAMPTZ '2025-01-05 21:25:00+00')")
    con.close()
    return path


@pytest.fixture
def synth_db(tmp_path, monkeypatch):
    """3 plays: MIN 2024 wk1 +5, MIN 2024 wk2 +12, GB 2025 wk1 -3. Plus side files to try to read."""
    path = _build_db(tmp_path / "synth.duckdb",
                     [(2024, 1, "MIN", 5), (2024, 2, "MIN", 12), (2025, 1, "GB", -3)])
    (tmp_path / "side.csv").write_text("a,b\n1,2\n")
    other = duckdb.connect(str(tmp_path / "other.duckdb"))
    other.execute("CREATE TABLE z AS SELECT 42 AS a")
    other.execute(f"COPY (SELECT 1 AS a) TO '{tmp_path / 'side.parquet'}' (FORMAT PARQUET)")
    other.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


def _run(sql: str):
    with database.get_db_connection() as conn:
        return conn.execute(sql).fetchall()


def _setting(name: str):
    with database.get_db_connection() as conn:
        return conn.execute(f"SELECT current_setting('{name}')").fetchone()[0]


def _is(value, expected: bool) -> bool:
    return str(value).lower() == str(expected).lower()


# ── 1. file and network reads are blocked ─────────────────────────────────────

READS = [
    "SELECT content FROM read_text('/etc/passwd')",
    "SELECT content FROM read_text('{tmp}/side.csv')",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_csv('{tmp}/side.csv')",
    "SELECT content FROM read_blob('/etc/passwd')",
    "SELECT content FROM read_blob('{tmp}/side.csv')",
    "SELECT * FROM glob('/etc/*')",
    "SELECT * FROM glob('{tmp}/*')",
    "SELECT * FROM parquet_scan('{tmp}/side.parquet')",
    "SELECT * FROM parquet_scan('/etc/*.parquet')",
    # 127.0.0.1:9 keeps the test off the real network; today it fails with IOException.
    "SELECT * FROM read_csv('https://127.0.0.1:9/x.csv')",
    "SELECT * FROM read_parquet('https://127.0.0.1:9/x.parquet')",
]


@pytest.mark.parametrize("template", READS)
def test_file_read_blocked_with_permission_error(synth_db, tmp_path, template):
    with pytest.raises(duckdb.PermissionException):
        _run(template.format(tmp=tmp_path))


@pytest.mark.parametrize("name,expected", [
    pytest.param("enable_external_access", False),
    pytest.param("lock_configuration", True),
    pytest.param("autoinstall_known_extensions", False),
    pytest.param("autoload_known_extensions", False),
])
def test_connection_setting_locked_value(synth_db, name, expected):
    assert _is(_setting(name), expected)


# ── 2. writes, extensions, secrets and config changes are blocked ─────────────

@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_copy_to_file_blocked_and_no_file_written(synth_db, tmp_path, fmt):
    out = tmp_path / f"out.{fmt}"
    with pytest.raises(duckdb.PermissionException):
        _run(f"COPY (SELECT 1 AS a) TO '{out}' (FORMAT {fmt.upper()})")
    assert not out.exists()


def test_export_database_blocked_and_nothing_written(synth_db, tmp_path):
    out = tmp_path / "exp"
    with pytest.raises(duckdb.PermissionException):
        _run(f"EXPORT DATABASE '{out}'")
    assert not out.exists() or not any(out.iterdir())


BLOCKED = [
    pytest.param("INSTALL httpfs", duckdb.PermissionException),
    pytest.param("LOAD httpfs", duckdb.PermissionException),
    pytest.param("CREATE SECRET s (TYPE S3)", duckdb.PermissionException),
    pytest.param("SET memory_limit='1GB'", duckdb.Error),
    pytest.param("SET threads=3", duckdb.Error),
    pytest.param("ATTACH '{tmp}/other.duckdb' AS o", duckdb.Error),
    # Already blocked by DuckDB on a plain read_only connection (verified): no xfail.
    pytest.param("SET enable_external_access=true", duckdb.Error),
    pytest.param("CREATE TABLE t AS SELECT 1", duckdb.Error),
    pytest.param("INSERT INTO plays VALUES (2026, 1, 'DET', 7)", duckdb.Error),
    pytest.param("ATTACH ':memory:' AS m", duckdb.Error),
    pytest.param("ATTACH '{tmp}/nope.duckdb' AS n", duckdb.Error),
]


@pytest.mark.parametrize("template,exc", BLOCKED)
def test_statement_blocked_on_served_connection(synth_db, tmp_path, template, exc):
    with pytest.raises(exc):
        _run(template.format(tmp=tmp_path))


@pytest.mark.parametrize("sql,name,forbidden", [
    ("SET threads=3", "threads", 3),
    ("SET memory_limit='1GB'", "memory_limit", "953.6 MiB"),
])
def test_set_does_not_change_setting(synth_db, sql, name, forbidden):
    with database.get_db_connection() as conn:
        with contextlib.suppress(duckdb.Error):
            conn.execute(sql)
        assert conn.execute(f"SELECT current_setting('{name}')").fetchone()[0] != forbidden


def test_insert_leaves_row_count_unchanged(synth_db):
    with contextlib.suppress(duckdb.Error):
        _run("INSERT INTO plays VALUES (2026, 1, 'DET', 7)")
    assert _run("SELECT count(*) FROM plays")[0][0] == 3  # 3 hand-written rows


# ── 3. normal reads still work ────────────────────────────────────────────────

@pytest.mark.parametrize("sql,expected", [
    ("SELECT 1", [(1,)]),
    # 2024 rows: wk1 and wk2, both MIN -> 2
    ("SELECT count(*) FROM plays WHERE season = 2024", [(2,)]),
    # CTE: MIN 5+12=17, GB -3
    (("WITH t AS (SELECT posteam, sum(yards_gained) AS y FROM plays GROUP BY 1) "
      "SELECT posteam, y FROM t ORDER BY posteam"), [("GB", -3), ("MIN", 17)]),
    # window: running yards for MIN by week -> 5, 17
    (("SELECT week, sum(yards_gained) OVER (ORDER BY week) FROM plays "
      "WHERE posteam = 'MIN' ORDER BY week"), [(1, 5), (2, 17)]),
])
def test_select_works_on_served_connection(synth_db, sql, expected):
    assert _run(sql) == expected


def test_temp_table_allowed_within_connection(synth_db):
    with database.get_db_connection() as conn:
        conn.execute("CREATE TEMP TABLE tt AS SELECT posteam FROM plays WHERE season = 2025")
        assert conn.execute("SELECT * FROM tt").fetchall() == [("GB",)]  # one 2025 row


def test_temp_table_gone_on_next_connection(synth_db):
    with database.get_db_connection() as conn:
        conn.execute("CREATE TEMP TABLE tt AS SELECT 1 AS a")
    with pytest.raises(duckdb.CatalogException):
        _run("SELECT * FROM tt")


# ── 4. the serve path creates nothing ─────────────────────────────────────────

@pytest.mark.parametrize("parts", [("missing", "sub", "x.duckdb"), ("missing", "x.duckdb"), ("x.duckdb",)])
def test_missing_db_raises_with_path_in_message(tmp_path, monkeypatch, parts):
    path = tmp_path.joinpath(*parts)
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    with pytest.raises((FileNotFoundError, duckdb.IOException)) as exc, database.get_db_connection():
        pass
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("parts", [("missing", "sub", "x.duckdb"), ("missing", "x.duckdb")])
def test_missing_db_creates_no_directory(tmp_path, monkeypatch, parts):
    path = tmp_path.joinpath(*parts)
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    with contextlib.suppress(FileNotFoundError, duckdb.Error), database.get_db_connection():
        pass
    assert not (tmp_path / "missing").exists()


def test_missing_db_creates_no_file(tmp_path, monkeypatch):
    path = tmp_path / "x.duckdb"
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    with contextlib.suppress(FileNotFoundError, duckdb.Error), database.get_db_connection():
        pass
    assert list(tmp_path.iterdir()) == []


# ── 5. env config applied at connect time, validated, never interpolated ──────

@pytest.mark.parametrize("mem,mem_text,threads", [("256MB", "244.1 MiB", "1"), ("1GiB", "1.0 GiB", "3")])
def test_env_settings_reported_on_connection(synth_db, tmp_path, monkeypatch, mem, mem_text, threads):
    monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", mem)
    monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", threads)
    monkeypatch.setenv("NFL_MCP_DUCKDB_TEMP_DIR", str(tmp_path / "spill"))
    with database.get_db_connection() as conn:
        got = conn.execute("SELECT current_setting('memory_limit'), current_setting('threads'), "
                           "current_setting('temp_directory')").fetchone()
    # 256MB = 256e6 B / 2^20 = 244.14 MiB; 1GiB is reported as-is
    assert got == (mem_text, int(threads), str(tmp_path / "spill"))


@pytest.mark.parametrize("threads", ["1", "2"])
def test_env_settings_applied_and_config_locked(synth_db, monkeypatch, threads):
    monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", threads)
    monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "256MB")
    with database.get_db_connection() as conn:
        got = conn.execute("SELECT current_setting('threads'), current_setting('lock_configuration')").fetchone()
    assert got[0] == int(threads) and _is(got[1], True)


@pytest.mark.parametrize("var", ["NFL_MCP_DUCKDB_MEMORY_LIMIT", "NFL_MCP_DUCKDB_TEMP_DIR"])
def test_env_value_not_interpolated_into_sql(synth_db, monkeypatch, var):
    value = "1GB'; SET threads=7; SELECT '" if "MEMORY" in var else "/tmp/x'; SET threads=7; SELECT '"
    monkeypatch.setenv(var, value)
    try:
        threads = _setting("threads")
    except (ValueError, duckdb.Error):
        return  # rejecting the value outright is fine
    assert threads != 7


def _problem_fn():
    fn = getattr(database, "connection_config_problem", None)
    assert fn is not None, "nfl_mcp.database.connection_config_problem missing"
    return fn


@pytest.mark.parametrize("env", [
    {},
    {"NFL_MCP_DUCKDB_THREADS": "2", "NFL_MCP_DUCKDB_MEMORY_LIMIT": "512MB"},
    {"NFL_MCP_DUCKDB_MEMORY_LIMIT": "2GiB", "NFL_MCP_DUCKDB_TEMP_DIR": "/tmp/spill"},
])
def test_config_problem_none_for_valid_env(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert _problem_fn()() is None


@pytest.mark.parametrize("var,value", [
    ("NFL_MCP_DUCKDB_THREADS", "abc"),
    ("NFL_MCP_DUCKDB_THREADS", "0"),
    ("NFL_MCP_DUCKDB_THREADS", "-1"),
    ("NFL_MCP_DUCKDB_MEMORY_LIMIT", "lots"),
    ("NFL_MCP_DUCKDB_MEMORY_LIMIT", "1GB'; SET threads=7; SELECT '"),
])
def test_config_problem_names_bad_var(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    problem = _problem_fn()()
    assert isinstance(problem, str) and var in problem


@pytest.fixture
def fake_serve(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "create_app", lambda *a, **k: object())
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    return calls


def _output(result) -> str:
    text = result.output or ""
    with contextlib.suppress(ValueError, AttributeError):
        text += result.stderr or ""
    return text


@pytest.mark.parametrize("var,value", [
    ("NFL_MCP_DUCKDB_THREADS", "abc"),
    ("NFL_MCP_DUCKDB_THREADS", "0"),
    ("NFL_MCP_DUCKDB_MEMORY_LIMIT", "lots"),
])
def test_serve_rejects_bad_env_before_start(synth_db, monkeypatch, fake_serve, var, value):
    monkeypatch.setenv(var, value)
    result = CliRunner().invoke(cli.main, ["serve"])
    assert result.exit_code != 0
    assert fake_serve == []
    assert var in _output(result)


def test_serve_starts_with_valid_env(synth_db, monkeypatch, fake_serve):
    monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", "2")
    monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "512MB")
    result = CliRunner().invoke(cli.main, ["serve"])
    assert result.exit_code == 0, _output(result)
    assert len(fake_serve) == 1


# ── 6. doctor uses the locked connection ──────────────────────────────────────

def _patch_get_db_connection(monkeypatch, replacement):
    monkeypatch.setattr(database, "get_db_connection", replacement)
    monkeypatch.setattr(cli, "get_db_connection", replacement, raising=False)


def _run_doctor(monkeypatch):
    from nfl_mcp import config
    monkeypatch.setattr(config, "config_exists", lambda: True)
    return CliRunner().invoke(cli.main, ["doctor"])


def test_doctor_reports_synthetic_counts(synth_db, monkeypatch):
    result = _run_doctor(monkeypatch)
    # 3 hand-written plays, seasons 2024 and 2025
    assert "3 plays loaded (seasons 2024–2025)" in result.output, result.output


def test_doctor_calls_get_db_connection(synth_db, monkeypatch):
    original = database.get_db_connection
    calls = []

    @contextlib.contextmanager
    def recording():
        calls.append(1)
        with original() as conn:
            yield conn

    _patch_get_db_connection(monkeypatch, recording)
    _run_doctor(monkeypatch)
    assert calls, "doctor never called nfl_mcp.database.get_db_connection"


def test_doctor_counts_come_from_get_db_connection(synth_db, tmp_path, monkeypatch):
    # A second DB with 4 plays in 2019-2020; if doctor prints these, it used the patched function.
    alt = _build_db(tmp_path / "alt.duckdb", [(2019, 1, "MIN", 1), (2019, 2, "MIN", 2),
                                              (2020, 1, "GB", 3), (2020, 2, "GB", 4)])

    @contextlib.contextmanager
    def alt_conn():
        conn = duckdb.connect(str(alt), read_only=True)
        try:
            yield conn
        finally:
            conn.close()

    _patch_get_db_connection(monkeypatch, alt_conn)
    result = _run_doctor(monkeypatch)
    assert "4 plays loaded (seasons 2019–2020)" in result.output, result.output


def test_doctor_connection_has_external_access_disabled(synth_db, monkeypatch):
    original = database.get_db_connection
    seen = []

    @contextlib.contextmanager
    def recording():
        with original() as conn:
            seen.append(conn.execute("SELECT current_setting('enable_external_access')").fetchone()[0])
            yield conn

    _patch_get_db_connection(monkeypatch, recording)
    _run_doctor(monkeypatch)
    assert seen and all(_is(v, False) for v in seen)


# ── 7. pytz: TIMESTAMP WITH TIME ZONE round-trips ─────────────────────────────

def test_nfl_query_select_star_with_timestamptz(synth_db):
    result = nfl_query("SELECT * FROM tz_events ORDER BY id")
    assert "error" not in result, result
    assert [r["id"] for r in result["rows"]] == [1, 2]  # 2 hand-written rows


@pytest.mark.parametrize("row_id,expected", [
    (1, dt.datetime(2024, 9, 8, 17, 0, tzinfo=dt.timezone.utc)),
    (2, dt.datetime(2025, 1, 5, 21, 25, tzinfo=dt.timezone.utc)),
])
def test_nfl_query_timestamptz_value(synth_db, row_id, expected):
    result = nfl_query(f"SELECT ts FROM tz_events WHERE id = {row_id}")
    assert "error" not in result, result
    got = result["rows"][0]["ts"]
    assert isinstance(got, dt.datetime) and got.tzinfo is not None and got == expected


def test_pyproject_declares_pytz():
    tomllib = pytest.importorskip("tomllib")  # stdlib from 3.11; project supports 3.10
    deps = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]
    assert any(d.lower().replace(" ", "").startswith("pytz") for d in deps), deps


def _real_db_columns(table: str, season: int, tz_col: str) -> tuple[list[str], int]:
    """Oracle: column names from information_schema and a row count, using a cast so pytz is not needed."""
    con = duckdb.connect(str(REAL_DB), read_only=True)
    try:
        cols = [r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
            [table]).fetchall()]
        n = con.execute(f"SELECT count(*) FROM {table} WHERE season = ? AND {tz_col} IS NOT NULL",
                        [season]).fetchone()[0]
    finally:
        con.close()  # close before nfl_query opens the same file with a different config
    return cols, n


@pytest.fixture
def real_db(monkeypatch):
    if not REAL_DB.exists():
        pytest.skip("real DB missing")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(REAL_DB))


@pytest.mark.integration
# injuries.date_modified is NULL for every 2025 and 2026 row in this DB, and a NULL
# TIMESTAMPTZ does not need pytz, so each query filters to a non-NULL value.
@pytest.mark.parametrize("table,tz_col,season", [
    ("injuries", "date_modified", 2024),
    ("ftn_charting", "date_pulled", 2024),
    ("ftn_charting", "date_pulled", 2025),
])
def test_nfl_query_select_star_real_tables(real_db, table, tz_col, season):
    cols, n = _real_db_columns(table, season, tz_col)
    if n == 0:
        pytest.skip(f"{table} has no {season} rows with {tz_col}")
    result = nfl_query(f"SELECT * FROM {table} WHERE season = {season} AND {tz_col} IS NOT NULL LIMIT 1")
    assert "error" not in result, result
    assert len(result["rows"]) == 1
    assert list(result["rows"][0].keys()) == cols
