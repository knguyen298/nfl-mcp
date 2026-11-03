"""Read-only guardrails for nfl_query and for the serve connection itself."""

import asyncio
import contextlib
import json
import threading
import time
from pathlib import Path

import duckdb
import pytest

from nfl_mcp.database import get_db_connection
from nfl_mcp.tools import _READ_STATEMENT_TYPES, _execute, nfl_query

# Statements that must not run: file and table functions, extension loading,
# ATTACH, exports, DDL and DML, and any attempt to unlock the config.
# Each entry is (label, sql). A LOAD of a statically linked extension (json) still
# succeeds at runtime, so the runtime check uses an extension DuckDB must load from disk.
COPY_TARGET = "/tmp/nflmcp_dump.parquet"
BLOCKED = [
    ("read_text", "SELECT * FROM read_text('/etc/passwd')"),
    ("read_csv", "SELECT * FROM read_csv('/etc/passwd')"),
    ("read_parquet", "SELECT * FROM read_parquet('/tmp/nope.parquet')"),
    ("glob", "SELECT * FROM glob('/tmp/*')"),
    ("attach", "ATTACH ':memory:' AS other"),
    ("copy_to", f"COPY (SELECT 1 AS a) TO '{COPY_TARGET}'"),
    ("install", "INSTALL iceberg"),
    ("load", "LOAD iceberg"),
    ("create_table", "CREATE TABLE nflmcp_bad (a INTEGER)"),
    ("insert", "INSERT INTO plays VALUES (1, 2024, 'KC', 1.0)"),
    ("update", "UPDATE plays SET epa = 0"),
    ("delete", "DELETE FROM plays"),
    ("drop", "DROP TABLE plays"),
    ("set_external_access", "SET enable_external_access = true"),
]

BLOCKED_LABELS = [case[0] for case in BLOCKED]
BLOCKED_BY_LABEL = dict(BLOCKED)

# Rejected by statement type before anything runs, so these are not in BLOCKED: both
# succeed on a plain connection, they are simply not read queries.
NOT_READ_BY_TYPE = [
    ("pragma", "PRAGMA database_list"),
    ("call", "CALL range(3)"),
]

# Input that parses to no statement at all.
EMPTY_INPUTS = [
    ("empty", ""),
    ("whitespace", "   "),
    ("comment_only", "-- just a comment"),
    ("semicolon_only", ";"),
]

# pyproject.toml pins duckdb>=2.0, and that floor is what makes this true: on 2.x PIVOT
# parses as a single statement of an allowed type, while 1.x expands it to CREATE + SELECT
# and the one-statement rule rejects it. The floor has no ceiling, and 2.1.0.dev releases
# already exist, so test_pivot_is_one_allowed_statement_on_the_pinned_floor below asserts
# the parse and names the version it got, instead of skipping when it does not hold.
PIVOT_SQL = "PIVOT plays ON season USING count(*) GROUP BY team"


@pytest.fixture
def require_db():
    """Skip the live checks when no database is loaded (same rule as test_tools.py)."""
    result = nfl_query("SELECT COUNT(*) AS n FROM plays")
    if "error" in result:
        pytest.skip("No DuckDB database loaded")
    if result["rows"][0]["n"] == 0:
        pytest.skip("plays table is empty")


@pytest.fixture
def guard_db(tmp_path, monkeypatch):
    """A small plays table shaped for the blocked-write statements, via env path."""
    path = tmp_path / "nflread.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE plays AS SELECT range AS play_id, 2020 + (range % 5)::int AS season, "
        "CASE WHEN range % 2 = 0 THEN 'KC' ELSE 'LV' END AS team, "
        "range * 1.5 AS epa FROM range(400)"
    )
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


class TestBlockedStatements:
    """The locked read-only connection blocks file access; the parser blocks the rest."""

    @pytest.fixture(autouse=True)
    def remove_copy_target(self):
        Path(COPY_TARGET).unlink(missing_ok=True)
        yield
        Path(COPY_TARGET).unlink(missing_ok=True)

    @pytest.mark.parametrize("label,sql", BLOCKED, ids=BLOCKED_LABELS)
    def test_nfl_query_rejects(self, guard_db, label, sql):
        assert "error" in nfl_query(sql), f"nfl_query allowed {label}"

    @pytest.mark.parametrize("label,sql", BLOCKED, ids=BLOCKED_LABELS)
    def test_direct_connection_rejects(self, guard_db, label, sql):
        with get_db_connection() as conn, pytest.raises(duckdb.Error):
            conn.execute(sql).fetchall()
        if label == "copy_to":
            assert not Path(COPY_TARGET).exists(), "the locked connection wrote a file"

    def test_copy_to_does_not_leave_a_file(self, guard_db):
        assert "error" in nfl_query(BLOCKED_BY_LABEL["copy_to"])
        assert not Path(COPY_TARGET).exists()

    @pytest.mark.parametrize("label,sql", NOT_READ_BY_TYPE, ids=[c[0] for c in NOT_READ_BY_TYPE])
    def test_rejected_by_statement_type(self, guard_db, label, sql):
        result = nfl_query(sql)
        assert "error" in result, f"nfl_query allowed {label}"
        assert "Only read-only queries are allowed" in result["error"]

    @pytest.mark.parametrize("label,sql", EMPTY_INPUTS, ids=[c[0] for c in EMPTY_INPUTS])
    def test_input_that_parses_to_nothing_is_rejected(self, guard_db, label, sql):
        result = nfl_query(sql)
        assert result["error"] == "Exactly one SQL statement is required, got 0.", label

    def test_file_error_names_the_lock(self, guard_db):
        result = nfl_query("SELECT * FROM read_text('/etc/passwd')")
        assert "file system operations are disabled" in result["error"]

    def test_write_statement_error_names_the_type(self, guard_db):
        assert "read-only" in nfl_query("UPDATE plays SET epa = 0")["error"]

    def test_read_only_connection_still_runs_reads(self, guard_db):
        result = nfl_query("SELECT COUNT(*) AS n FROM plays")
        assert "error" not in result
        assert result["rows"][0]["n"] == 400


class TestMultiAndExplainAdmitWrites:
    """MULTI and EXPLAIN pass the allow list, so the read-only lock has to hold.

    The parser reports MULTI for PIVOT and for write forms that wrap it, and
    EXPLAIN ANALYZE executes whatever it is given. Neither is refused by the parser;
    the locked read-only connection stops them.
    """

    PIVOT_SRC = (
        "PIVOT (SELECT season AS s, team AS t, epa FROM plays) "
        "ON t USING count(*) GROUP BY s"
    )

    def _play_count(self):
        return nfl_query("SELECT COUNT(*) AS n FROM plays")["rows"][0]["n"]

    def test_create_table_as_pivot_creates_nothing(self, guard_db):
        result = nfl_query(f"CREATE TABLE nflmcp_pivot AS {self.PIVOT_SRC}")
        assert "error" in result
        assert "read-only" in result["error"]
        assert "does not exist" in nfl_query("SELECT * FROM nflmcp_pivot")["error"]

    def test_create_temp_table_as_pivot_stays_on_its_own_connection(self, guard_db):
        """The one MULTI write the lock lets through: the temp table dies with the
        per-call connection, so the next call cannot see it."""
        assert "error" not in nfl_query(f"CREATE TEMP TABLE nflmcp_tmp AS {self.PIVOT_SRC}")
        assert "error" in nfl_query("SELECT * FROM temp.nflmcp_tmp")

    def test_explain_analyze_of_a_write_writes_nothing(self, guard_db):
        before = self._play_count()
        result = nfl_query("EXPLAIN ANALYZE INSERT INTO plays VALUES (999, 2024, 'KC', 1.0)")
        assert "error" in result
        assert "read-only" in result["error"]
        assert self._play_count() == before

    def test_explain_analyze_of_copy_to_writes_no_file(self, guard_db, tmp_path):
        target = tmp_path / "explain_analyze.parquet"
        result = nfl_query(f"EXPLAIN ANALYZE COPY (SELECT 1 AS a) TO '{target}'")
        assert "error" in result
        assert "file system operations are disabled" in result["error"]
        assert not target.exists()

    def test_explain_analyze_of_a_session_change_runs(self, guard_db):
        assert "error" not in nfl_query("EXPLAIN ANALYZE SET VARIABLE nflmcp_v = 1")


class TestPreviouslyBlockedReadsNowRun:
    """Every case the old keyword regex and the SELECT-prefix check rejected."""

    def test_cte(self, guard_db):
        result = nfl_query(
            "WITH per_team AS (SELECT team, SUM(epa) AS e FROM plays GROUP BY team) "
            "SELECT team FROM per_team WHERE e > 0 ORDER BY team"
        )
        assert "error" not in result
        assert [r["team"] for r in result["rows"]] == ["KC", "LV"]

    def test_window_function(self, guard_db):
        result = nfl_query(
            "SELECT team, epa, row_number() OVER (PARTITION BY team ORDER BY epa DESC) AS rn "
            "FROM plays QUALIFY rn <= 2 ORDER BY team, rn"
        )
        assert "error" not in result
        assert [r["rn"] for r in result["rows"]] == [1, 2, 1, 2]
        assert [r["team"] for r in result["rows"]] == ["KC", "KC", "LV", "LV"]

    def test_replace_function(self, guard_db):
        result = nfl_query("SELECT replace(team, 'K', 'X') AS team FROM plays LIMIT 1")
        assert result["rows"][0]["team"] == "XC"

    def test_ilike_on_a_word_containing_set(self, guard_db):
        result = nfl_query(
            "SELECT COUNT(*) AS n FROM (SELECT 'offset' AS w) WHERE w ILIKE '%set%'"
        )
        assert "error" not in result
        assert result["rows"][0]["n"] == 1

    def test_semicolon_inside_a_string_literal(self, guard_db):
        result = nfl_query("SELECT 'a;b' AS s WHERE s ILIKE 'a;b'")
        assert "error" not in result
        assert result["rows"][0]["s"] == "a;b"

    def test_alias_named_load(self, guard_db):
        result = nfl_query("SELECT 1 AS load")
        assert "error" not in result

    def test_describe(self, guard_db):
        result = nfl_query("DESCRIBE plays")
        assert "error" not in result
        assert {r["column_name"] for r in result["rows"]} == {"play_id", "season", "team", "epa"}

    def test_summarize(self, guard_db):
        result = nfl_query("SUMMARIZE (SELECT 1 AS a)")
        assert "error" not in result

    def test_explain(self, guard_db):
        result = nfl_query("EXPLAIN SELECT team FROM plays")
        assert "error" not in result
        assert "physical_plan" in result["rows"][0]["explain_key"]

    def test_rejects_relocking_the_config(self, guard_db):
        assert "error" in nfl_query("SET memory_limit = '1GB'")

    def test_from_first_query(self, guard_db):
        result = nfl_query("FROM plays SELECT team LIMIT 2")
        assert "error" not in result
        assert result["row_count"] == 2

    def test_pivot_is_one_allowed_statement_on_the_pinned_floor(self):
        statements = duckdb.extract_statements(PIVOT_SQL)
        assert len(statements) == 1 and statements[0].type in _READ_STATEMENT_TYPES, (
            f"duckdb {duckdb.__version__} parses PIVOT as {[s.type.name for s in statements]}; "
            "nfl_query needs one read statement (see the duckdb floor in pyproject.toml)"
        )

    def test_pivot(self, guard_db):
        result = nfl_query(PIVOT_SQL)
        assert "error" not in result
        assert {r["team"] for r in result["rows"]} == {"KC", "LV"}

    def test_trailing_semicolon(self, guard_db):
        result = nfl_query("SELECT 1 AS a;")
        assert "error" not in result


class TestQueryResultShape:
    def test_order_by_survives_the_row_cap(self, guard_db):
        """No LIMIT in the statement: the cap must come from the tool, inside the query."""
        result = nfl_query("SELECT epa FROM plays ORDER BY epa DESC", max_rows=3)
        assert [round(r["epa"], 1) for r in result["rows"]] == [598.5, 597.0, 595.5]
        assert result["truncated"] is True
        assert result["row_count"] == 3

    def test_the_cap_makes_duckdb_plan_a_top_n_sort(self, guard_db):
        """Pin the mechanism, not the call shape.

        A relation capped with limit() plans 'Top N'. Fetching the cap in Python instead
        plans a full 'Order By' over the whole table, which is the slow path this replaced.
        """
        with get_db_connection() as conn:
            capped = str(conn.sql("SELECT epa FROM plays ORDER BY epa DESC").limit(3).explain())
            uncapped = str(conn.sql("SELECT epa FROM plays ORDER BY epa DESC").explain())
        assert "Top N" in capped
        assert "Top N" not in uncapped

    def test_row_cap_is_planned_by_duckdb(self, monkeypatch):
        """The cap has to reach DuckDB as a relation LIMIT, not as a Python fetch.

        A Python fetch makes DuckDB build and sort the whole result first: over 1.33 M
        plays rows `SELECT * FROM plays ORDER BY epa DESC` took 18.3 s that way and hit
        the 10 s timeout, against 0.8 s with the cap in the plan.
        """
        seen: dict[str, int] = {}
        inner = duckdb.connect()
        inner.execute("CREATE TABLE plays AS SELECT range * 1.5 AS epa FROM range(400)")

        class RelationSpy:
            def __init__(self, rel):
                self._rel = rel

            def limit(self, n):
                seen["limit"] = n
                return RelationSpy(self._rel.limit(n))

            def __getattr__(self, name):
                return getattr(self._rel, name)

        class ConnectionSpy:
            def sql(self, sql):
                return RelationSpy(inner.sql(sql))

            def execute(self, sql, params=None):
                return inner.execute(sql) if params is None else inner.execute(sql, params)

        try:
            monkeypatch.setattr(
                "nfl_mcp.tools.get_db_connection",
                lambda: contextlib.nullcontext(ConnectionSpy()),
            )
            rows = _execute("SELECT epa FROM plays ORDER BY epa DESC", row_limit=3)
        finally:
            inner.close()

        assert seen["limit"] == 3
        assert [round(r["epa"], 1) for r in rows] == [598.5, 597.0, 595.5]

    def test_returns_the_sql_it_ran(self, guard_db):
        result = nfl_query("  -- hi\n SELECT 1 AS a ;  ")
        assert result["sql"] == "SELECT 1 AS a ;"

    def test_max_rows_is_clamped(self, guard_db, monkeypatch):
        captured = {}

        def spy(sql, params=None, row_limit=None):
            captured["row_limit"] = row_limit
            return [{"a": 1}]

        monkeypatch.setattr("nfl_mcp.tools._execute", spy)
        nfl_query("SELECT 1 AS a", max_rows=9999)
        assert captured["row_limit"] == 501

    def test_negative_and_zero_max_rows_return_no_rows(self, guard_db):
        for limit in (0, -5):
            result = nfl_query("SELECT 1 AS a", max_rows=limit)
            assert result["row_count"] == 0, limit
            assert result["truncated"] is True, limit

    def test_two_statements_are_rejected(self, guard_db):
        assert "error" in nfl_query("SELECT 1; SELECT 2")

    def test_unparseable_sql_is_an_error_not_a_crash(self, guard_db):
        result = nfl_query("SELECT 'unterminated")
        assert "error" in result


class TestWriterConnectionsUnaffected:
    """get_db_connection is read-only with locked configuration, so anything that
    writes must keep opening its own connection."""

    def test_ingest_and_cli_do_not_use_get_db_connection(self):
        for path in ("nfl_mcp/ingest.py", "nfl_mcp/cli.py"):
            source = Path(path).read_text()
            assert "get_db_connection" not in source, f"{path} must keep its own connection"

    def test_ingest_still_sets_its_own_pragmas(self, tmp_path, monkeypatch):
        from nfl_mcp.ingest import _apply_duckdb_pragmas

        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "512MB")
        monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", "1")
        writer = duckdb.connect(str(tmp_path / "write.duckdb"))
        try:
            _apply_duckdb_pragmas(writer, str(tmp_path / "write.duckdb"))
            assert writer.execute("SELECT current_setting('threads')").fetchone()[0] == 1
            assert "MiB" in writer.execute("SELECT current_setting('memory_limit')").fetchone()[0]
            assert writer.execute(
                "SELECT current_setting('temp_directory')"
            ).fetchone()[0].endswith(".duckdb_spill")
            writer.execute("CREATE TABLE t (a INTEGER)")
            writer.execute("INSERT INTO t VALUES (1)")
        finally:
            writer.close()

    def test_a_writer_opens_after_a_locked_read_only_connection(self, guard_db):
        """The symlink swap in the container opens a writer next to the server."""
        with get_db_connection() as conn:
            conn.execute("SELECT 1 FROM plays").fetchall()
        writer = duckdb.connect(str(guard_db))
        try:
            writer.execute("CREATE TABLE ingested (season INTEGER)")
        finally:
            writer.close()


class TestServerStaysResponsive:
    """A slow tool must not block the event loop for the rest of the server."""

    def test_slow_query_does_not_block_a_parallel_status_call(self):
        import nfl_mcp.server as server_module

        delay = 1.2

        def slow_query(**_kwargs):
            time.sleep(delay)
            return {"rows": []}

        def fake_status():
            return {"plays": {}}

        async def scenario():
            original_query = server_module._DISPATCH["nfl_query"]
            original_status = server_module._DISPATCH["nfl_status"]
            server_module._DISPATCH["nfl_query"] = slow_query
            server_module._DISPATCH["nfl_status"] = fake_status
            try:
                started = time.perf_counter()
                await server_module.call_tool("nfl_query", {})
                return time.perf_counter() - started
            finally:
                server_module._DISPATCH["nfl_query"] = original_query
                server_module._DISPATCH["nfl_status"] = original_status

        async def with_ticker():
            gaps = []
            last = time.perf_counter()
            slow = asyncio.create_task(scenario())
            for _ in range(40):
                await asyncio.sleep(0.02)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now
            await slow
            return max(gaps)

        worst_gap = asyncio.run(with_ticker())
        assert worst_gap < 0.2, f"event loop stalled for {worst_gap:.2f}s"

    def test_dispatch_is_built_once_at_module_level(self):
        import nfl_mcp.server as server_module

        assert isinstance(server_module._DISPATCH, dict)
        assert set(server_module._DISPATCH) >= {"nfl_query", "nfl_status", "nfl_schema"}

    def test_real_slow_query_does_not_block_a_real_status_call(self, guard_db):
        """Same check without fakes: a real nfl_query that takes seconds runs
        against a real connection while nfl_status answers in parallel."""
        import nfl_mcp.server as server_module

        slow_sql = (
            "SELECT count(*) AS n FROM (SELECT md5(CAST(range AS VARCHAR)) h "
            "FROM range(2000000) ORDER BY h)"
        )

        async def scenario():
            query = asyncio.create_task(server_module.call_tool("nfl_query", {"sql": slow_sql}))
            await asyncio.sleep(0.05)
            started = time.perf_counter()
            status = await server_module.call_tool("nfl_status", None)
            waited = time.perf_counter() - started
            payload = json.loads((await query)[0].text)
            return waited, payload, json.loads(status[0].text)

        waited, query_payload, status_payload = asyncio.run(scenario())
        assert "error" not in query_payload
        assert query_payload["rows"][0]["n"] == 2000000
        # nfl_status reads a table the fixture does not have, so an error dict is a
        # fine answer — the point is that it came back while the query was still running.
        assert waited < 0.5, f"nfl_status waited {waited:.2f}s behind nfl_query"
        assert isinstance(status_payload, dict)

    def test_no_more_than_two_tools_run_at_once(self, guard_db):
        """DuckDB's memory_limit is per connection, so the limiter caps connections."""
        import nfl_mcp.server as server_module

        live = 0
        peak = 0
        lock = threading.Lock()

        def counting(**_kwargs):
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.15)
            with lock:
                live -= 1
            return {"ok": True}

        original = server_module._DISPATCH["nfl_query"]
        server_module._DISPATCH["nfl_query"] = counting
        try:
            async def fan_out():
                await asyncio.gather(
                    *[server_module.call_tool("nfl_query", {}) for _ in range(6)]
                )

            asyncio.run(fan_out())
        finally:
            server_module._DISPATCH["nfl_query"] = original

        assert server_module._MAX_CONCURRENT_TOOLS == 2
        assert peak <= 2, f"{peak} tool calls ran at once"


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflQueryLive:
    """The same guardrails against the loaded database, not a fixture."""

    def test_basic_select(self):
        result = nfl_query("SELECT COUNT(*) AS n FROM plays")
        assert result["row_count"] == 1
        assert result["rows"][0]["n"] > 0

    def test_team_filter(self):
        result = nfl_query("SELECT COUNT(*) AS n FROM plays WHERE posteam = 'KC'")
        assert result["rows"][0]["n"] > 0

    def test_aggregate_table(self):
        result = nfl_query("SELECT * FROM team_offense_stats LIMIT 1")
        assert result["row_count"] == 1
        assert "team" in result["rows"][0]

    def test_max_rows_respected(self):
        result = nfl_query("SELECT * FROM plays", max_rows=5)
        assert result["row_count"] == 5

    def test_truncation_flag(self):
        result = nfl_query("SELECT * FROM plays", max_rows=3)
        assert result["truncated"] is True

    def test_ordered_limit_returns_the_top_rows(self):
        top = nfl_query("SELECT play_id, epa FROM plays WHERE epa IS NOT NULL ORDER BY epa DESC")
        assert "error" not in top
        eps = [r["epa"] for r in top["rows"]]
        assert eps == sorted(eps, reverse=True)

        reference = nfl_query(
            "SELECT epa FROM plays WHERE epa IS NOT NULL ORDER BY epa DESC LIMIT 1"
        )
        assert eps[0] == pytest.approx(reference["rows"][0]["epa"])

    def test_cte_over_plays(self):
        result = nfl_query(
            "WITH per_team AS (SELECT posteam, COUNT(*) AS n FROM plays GROUP BY 1) "
            "SELECT posteam FROM per_team ORDER BY n DESC LIMIT 1"
        )
        assert "error" not in result
        assert result["row_count"] == 1

    def test_window_function_over_plays(self):
        result = nfl_query(
            "SELECT posteam, epa, row_number() OVER (PARTITION BY posteam ORDER BY epa DESC) AS rn "
            "FROM plays WHERE epa IS NOT NULL AND posteam IS NOT NULL QUALIFY rn = 1 LIMIT 3"
        )
        assert "error" not in result
        assert {r["rn"] for r in result["rows"]} == {1}

    def test_describe_plays(self):
        columns = {r["column_name"] for r in nfl_query("DESCRIBE plays", max_rows=500)["rows"]}
        assert {"play_id", "season", "epa", "posteam"} <= columns

    def test_pivot_over_plays(self):
        result = nfl_query("PIVOT plays ON season USING count(*) GROUP BY posteam")
        assert "error" not in result
        assert result["row_count"] > 0

    @pytest.mark.parametrize("label,sql", BLOCKED, ids=BLOCKED_LABELS)
    def test_blocked_on_the_loaded_database(self, label, sql):
        result = nfl_query(sql)
        assert "error" in result, f"loaded database allowed {label}"
