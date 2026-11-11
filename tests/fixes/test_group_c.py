"""Group C: server transport and output (server.py call_tool/create_app, tools._execute).

Spec: nfl-mcp-notes/PLAN_FIX_GROUPS_MAIN_2026-11-04.md section C; evidence: AUDIT_MAIN H3, H4, M1, M3,
M4, M7, L3, L4. Driving style follows tests/test_server.py: call_tool is awaited directly with
asyncio.run, and the Starlette app from create_app() is driven with starlette's TestClient. The two
HTTP-session test starts uvicorn on 127.0.0.1 on a free port and stops it in a finally block.

Interface contract (beyond the shared brief)
- nfl_mcp.server.to_json(obj) -> str: json.dumps(..., allow_nan=False) with no indent; NaN, +Inf and
  -Inf become null (also when nested); Decimal becomes a JSON number (Decimal('2') -> 2,
  Decimal('1.50') -> 1.5); date/datetime -> isoformat().
- call_tool(name, arguments) always returns mcp.types.CallToolResult. content[0] is TextContent with
  to_json(payload). isError is True when the tool raised, the tool name is unknown, an argument name is
  unknown, or the tool returned a dict with an "error" key.
- Error payload: one envelope for every failure. Top-level keys are the same set for every failure and
  include "error" (str message) and "code" (str). Tool-specific extras (did_you_mean, statement_type,
  hint, ...) go inside a nested key (suggested "details") so the top-level key set does not vary.
- Unknown argument names are rejected inside call_tool itself (not only by jsonschema
  additionalProperties, whose library message does not list valid names). code "INVALID_ARGUMENTS";
  the message names the bad argument and every valid argument name from the tool's inputSchema.
- Dispatch is resolved per call through module attributes of nfl_mcp.server (as today). If a
  module-level dict server._DISPATCH exists, tests patch it as well.
- Limiter: nfl_mcp.server.get_tool_limiter() -> anyio.CapacityLimiter, created lazily on first call
  (inside the running loop), reading NFL_MCP_MAX_CONCURRENT_QUERIES (default 2), cached in module
  attribute server._TOOL_LIMITER; tests reset that attribute to None. Every tool call runs in a worker
  thread (anyio.to_thread.run_sync) under that limiter.
- Timeout: NFL_MCP_QUERY_TIMEOUT_SECONDS is read on every _execute call (so monkeypatch.setenv works
  without reload); default tools._QUERY_TIMEOUT_SECONDS == 10. A timeout comes back as a dict whose
  "error" text contains "timeout".
- create_app() reads NFL_MCP_AUTH_TOKEN and NFL_MCP_CORS_ORIGINS when it is called. Auth covers every
  route, /healthz included (plan C6 "all routes"). CORS preflight (OPTIONS) is answered by the CORS
  middleware without a token. 401 for a missing, wrong or non-Bearer Authorization header. Origins are
  comma-separated, whitespace trimmed. When NFL_MCP_CORS_ORIGINS is unset, allow-origin stays "*" and a
  WARNING mentioning CORS is logged on logger "nfl-mcp" during create_app() or lifespan startup.
- Constant-time compare: the source of nfl_mcp.server contains "compare_digest" (weak proxy only: it
  proves the call exists, not that every comparison path uses it).
- GET /healthz: 200 {"ok": true} when get_db_connection() opens and SELECT 1 works; otherwise 503 with
  {"ok": false, ...}.
- Per-call log: exactly one INFO record on logger "nfl-mcp" per call whose message contains the tool
  name and key=value (or key: value) tokens duration_ms|duration, rows|row_count, truncated, and for
  failures error/code. SQL text in it is cut so the whole message is under 1000 characters.
"""

import asyncio
import contextlib
import datetime as dt
import inspect
import json
import logging
import re
import socket
import threading
import time
from decimal import Decimal

import duckdb
import pytest
from mcp.types import CallToolResult
from starlette.testclient import TestClient

import nfl_mcp.server as server
import nfl_mcp.tools as tools

TOKEN = "grpC-s3cret-token-0123456789abcdef"
INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "grpC-test", "version": "0"}},
}
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
# ~2-3 s on this host (calibrated: range(20000)^2 took 6.7 s, so 13500^2 is about 3 s).
SLOW_SQL = "SELECT sum((a.range * 7 + b.range) % 13) AS s FROM range(13500) a CROSS JOIN range(13500) b"
# For the timeout tests: about 20 times SLOW_SQL, so the timeout fires first even on a fast
# CI runner (SLOW_SQL alone finished in under 2 s on GitHub's ubuntu-latest).
LONG_SQL = "SELECT sum((a.range * 7 + b.range) % 13) AS s FROM range(60000) a CROSS JOIN range(60000) b"


# ── fixtures and helpers ────────────────────────────────────────────────────────

@pytest.fixture
def synth_db(tmp_path, monkeypatch):
    path = tmp_path / "synth.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1), (2), (3)) v(x)")
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("NFL_MCP_MAX_CONCURRENT_QUERIES", "NFL_MCP_QUERY_TIMEOUT_SECONDS",
                "NFL_MCP_AUTH_TOKEN", "NFL_MCP_CORS_ORIGINS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(server, "_TOOL_LIMITER", None, raising=False)
    yield
    monkeypatch.setattr(server, "_TOOL_LIMITER", None, raising=False)


def patch_tool(monkeypatch, name, fn):
    monkeypatch.setattr(server, name, fn)
    dispatch = getattr(server, "_DISPATCH", None)
    if isinstance(dispatch, dict):
        monkeypatch.setitem(dispatch, name, fn)


def sleeper(seconds):
    def slow_query(sql: str, max_rows: int = 100):
        time.sleep(seconds)
        return {"rows": [], "row_count": 0, "truncated": False}
    return slow_query


def fast_status():
    return {"ok": True}


def text_of(result):
    if isinstance(result, CallToolResult):
        return result.content[0].text
    return result[0].text


def call(name, args):
    return asyncio.run(server.call_tool(name, args))


def strict_loads(text):
    def _reject(token):
        raise ValueError(f"non-standard JSON constant {token}")
    return json.loads(text, parse_constant=_reject)


def first_row(payload):
    row = payload["rows"][0]
    if isinstance(row, dict):
        return row
    names = [c["name"] if isinstance(c, dict) else c for c in payload["columns"]]
    return dict(zip(names, row))


def to_json():
    fn = getattr(server, "to_json", None)
    assert fn is not None, "nfl_mcp.server.to_json missing"
    return fn


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def live_server(app):
    import uvicorn
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not srv.started and time.time() < deadline:
            time.sleep(0.05)
        assert srv.started, "uvicorn did not start"
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.should_exit = True
        thread.join(timeout=15)


# ── 1. serializer ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, expected", [
    (float("nan"), None),
    (float("inf"), None),
    (float("-inf"), None),
    (Decimal("1.50"), 1.5),
    (Decimal("2"), 2),
    (dt.date(2024, 9, 5), "2024-09-05"),
    (dt.datetime(2024, 9, 5, 20, 20, 0), "2024-09-05T20:20:00"),
])
def test_to_json_scalar_conversion(value, expected):
    out = strict_loads(to_json()({"v": value}))
    assert out == {"v": expected}


def test_to_json_decimal_is_number_not_string():
    out = to_json()({"d": Decimal("1.50")})
    assert '"1.50"' not in out and isinstance(json.loads(out)["d"], (int, float))


def test_to_json_nested_lists_and_dicts():
    obj = {"rows": [{"a": float("nan"), "b": [Decimal("0.25"), float("-inf"), {"d": dt.date(2025, 1, 2)}]}],
           "meta": {"x": [1, 2, {"y": float("inf")}]}}
    assert strict_loads(to_json()(obj)) == {
        "rows": [{"a": None, "b": [0.25, None, {"d": "2025-01-02"}]}],
        "meta": {"x": [1, 2, {"y": None}]},
    }


@pytest.mark.parametrize("obj", [{"a": 1, "b": {"c": [1, 2]}}, [{"x": 1}, {"y": [2, 3]}]])
def test_to_json_has_no_indentation(obj):
    out = to_json()(obj)
    assert not re.search(r"\n\s", out)
    assert json.loads(out) == obj


def test_call_tool_nfl_query_output_is_strict_json(synth_db, clean_env):
    sql = "SELECT 'nan'::DOUBLE AS x, 1.5::DECIMAL(10,2) AS d, DATE '2024-09-05' AS dt"
    payload = strict_loads(text_of(call("nfl_query", {"sql": sql})))
    assert first_row(payload) == {"x": None, "d": 1.5, "dt": "2024-09-05"}


def test_call_tool_infinity_is_null(synth_db, clean_env):
    sql = "SELECT 'inf'::DOUBLE AS p, '-inf'::DOUBLE AS n, 2::DECIMAL(10,0) AS whole"
    payload = strict_loads(text_of(call("nfl_query", {"sql": sql})))
    assert first_row(payload) == {"p": None, "n": None, "whole": 2}


# ── 2. error signalling ─────────────────────────────────────────────────────────

def _error_case(case, tmp_path, monkeypatch):
    if case == "unknown_tool":
        return call("no_such_tool", {})
    if case == "unknown_argument":
        return call("nfl_query", {"sql": "SELECT 1", "bogus": 1})
    if case == "sql_error":
        return call("nfl_query", {"sql": "SELECT * FROM no_such_table_xyz"})
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(tmp_path / "absent" / "missing.duckdb"))
    return call("nfl_status", {})


ERROR_CASES = ["unknown_tool", "unknown_argument", "sql_error", "missing_db"]


@pytest.mark.parametrize("case", ERROR_CASES)
def test_failure_is_flagged_is_error(case, synth_db, clean_env, tmp_path, monkeypatch):
    result = _error_case(case, tmp_path, monkeypatch)
    assert isinstance(result, CallToolResult) and result.isError is True


def test_failure_payloads_share_top_level_keys(synth_db, clean_env, tmp_path, monkeypatch):
    payloads = {c: json.loads(text_of(_error_case(c, tmp_path, monkeypatch))) for c in ERROR_CASES}
    key_sets = {c: frozenset(p) for c, p in payloads.items()}
    assert len(set(key_sets.values())) == 1, key_sets
    for p in payloads.values():
        assert isinstance(p["error"], str) and isinstance(p["code"], str)


def test_success_is_not_flagged(synth_db, clean_env):
    result = call("nfl_query", {"sql": "SELECT x FROM t"})
    assert isinstance(result, CallToolResult) and result.isError is False


@pytest.mark.parametrize("tool, args", [
    ("nfl_query", {"sql": "SELECT 1", "bogus": 1}),
    ("nfl_team_records", {"team": "GB", "bogus": 1}),
    ("nfl_schedule", {"season_type": "REG", "bogus": 1}),
])
def test_unknown_argument_lists_valid_names(tool, args, synth_db, clean_env):
    valid = set(next(t for t in server.TOOLS if t.name == tool).inputSchema["properties"])
    payload = json.loads(text_of(call(tool, args)))
    assert payload["code"] == "INVALID_ARGUMENTS"
    msg = payload["error"]
    assert "bogus" in msg
    for name in valid:
        assert re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", msg), name


# ── 3. event loop ───────────────────────────────────────────────────────────────

async def _status_while_slow(slow_args):
    slow = asyncio.create_task(server.call_tool("nfl_query", slow_args))
    await asyncio.sleep(0.2)
    t0 = time.monotonic()
    await asyncio.wait_for(server.call_tool("nfl_status", {}), timeout=10)
    status_elapsed = time.monotonic() - t0
    slow_running = not slow.done()
    await slow
    return status_elapsed, slow_running


def test_status_not_blocked_by_sleeping_tool(monkeypatch, synth_db, clean_env):
    patch_tool(monkeypatch, "nfl_query", sleeper(2.5))
    patch_tool(monkeypatch, "nfl_status", fast_status)
    elapsed, slow_running = asyncio.run(_status_while_slow({"sql": "SELECT 1"}))
    assert elapsed < 1.0 and slow_running


def test_status_not_blocked_by_slow_duckdb_query(monkeypatch, synth_db, clean_env):
    # Real _execute path: a ~3 s cross join, nfl_status replaced by a fast fake.
    patch_tool(monkeypatch, "nfl_status", fast_status)
    elapsed, slow_running = asyncio.run(_status_while_slow({"sql": SLOW_SQL}))
    assert elapsed < 1.0 and slow_running


def test_http_second_session_not_blocked(monkeypatch, synth_db, clean_env):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    patch_tool(monkeypatch, "nfl_query", sleeper(2.5))
    patch_tool(monkeypatch, "nfl_status", fast_status)

    async def session_call(url, name, args, delay):
        async with streamablehttp_client(url) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                await asyncio.sleep(delay)
                t0 = time.monotonic()
                await s.call_tool(name, args)
                return time.monotonic() - t0

    async def scenario(url):
        return await asyncio.wait_for(asyncio.gather(
            session_call(url, "nfl_query", {"sql": "SELECT 1"}, 0.0),
            session_call(url, "nfl_status", {}, 0.5),
        ), timeout=20)

    with live_server(server.create_app()) as base:
        slow_elapsed, status_elapsed = asyncio.run(scenario(base + "/mcp"))
    assert slow_elapsed >= 2.0
    assert status_elapsed < 1.0


# ── 4. concurrency limiter ──────────────────────────────────────────────────────

async def _two_slow_calls():
    t0 = time.monotonic()
    await asyncio.gather(server.call_tool("nfl_query", {"sql": "SELECT 1"}),
                         server.call_tool("nfl_query", {"sql": "SELECT 2"}))
    return time.monotonic() - t0


def test_limiter_one_runs_calls_serially(monkeypatch, synth_db, clean_env):
    # Passes today only because every call blocks the loop; must keep holding with a limiter of 1.
    monkeypatch.setenv("NFL_MCP_MAX_CONCURRENT_QUERIES", "1")
    patch_tool(monkeypatch, "nfl_query", sleeper(0.5))
    assert asyncio.run(_two_slow_calls()) >= 0.95  # 2 x 0.5 s


@pytest.mark.parametrize("limit", [None, "2"])
def test_limiter_two_overlaps_calls(limit, monkeypatch, synth_db, clean_env):
    if limit:
        monkeypatch.setenv("NFL_MCP_MAX_CONCURRENT_QUERIES", limit)  # None: default is 2
    patch_tool(monkeypatch, "nfl_query", sleeper(0.5))
    assert asyncio.run(_two_slow_calls()) < 0.85  # ~0.5 s when overlapped


@pytest.mark.parametrize("env, tokens", [(None, 2), ("1", 1), ("4", 4)])
def test_get_tool_limiter_reads_env(env, tokens, monkeypatch, clean_env):
    if env:
        monkeypatch.setenv("NFL_MCP_MAX_CONCURRENT_QUERIES", env)
    fn = getattr(server, "get_tool_limiter", None)
    assert fn is not None, "get_tool_limiter missing"

    async def total():
        return fn().total_tokens
    assert asyncio.run(total()) == tokens


# ── 5. timeout env ──────────────────────────────────────────────────────────────

def _timed_query(sql):
    t0 = time.monotonic()
    out = tools.nfl_query(sql)
    return out, time.monotonic() - t0


@pytest.mark.parametrize("seconds", [1, 2])
def test_timeout_env_cuts_slow_query(seconds, monkeypatch, synth_db, clean_env):
    monkeypatch.setenv("NFL_MCP_QUERY_TIMEOUT_SECONDS", str(seconds))
    out, elapsed = _timed_query(LONG_SQL)
    assert "timeout" in str(out.get("error", "")).lower()
    assert seconds - 0.1 <= elapsed < seconds + 1.5


def test_timeout_default_constant_is_ten():
    assert tools._QUERY_TIMEOUT_SECONDS == 10


def test_timeout_default_lets_three_second_query_finish(synth_db, clean_env):
    out, _ = _timed_query(SLOW_SQL)
    assert "error" not in out and out["row_count"] == 1


def test_no_query_thread_left_after_timeout(monkeypatch, synth_db, clean_env):
    # Env for the fixed code, module constant for today's code: both mean a 1 s timeout.
    monkeypatch.setenv("NFL_MCP_QUERY_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(tools, "_QUERY_TIMEOUT_SECONDS", 1)
    before = {t.ident for t in threading.enumerate()}
    out, _ = _timed_query(LONG_SQL)
    assert "timeout" in str(out.get("error", "")).lower()
    time.sleep(2)
    leftover = [t.name for t in threading.enumerate()
                if t.ident not in before and t.is_alive() and not t.name.startswith("AnyIO worker")]
    assert leftover == []


# ── 6. auth ─────────────────────────────────────────────────────────────────────

def _post_init(client, headers=None):
    return client.post("/mcp", json=INIT, headers={**MCP_HEADERS, **(headers or {})})


@pytest.mark.parametrize("headers", [
    None,
    {"Authorization": "Bearer wrong-token"},
    {"Authorization": "Bearer " + TOKEN[:-1] + "X"},  # same length, last char differs
    {"Authorization": TOKEN},                           # no Bearer scheme
])
def test_auth_rejects_missing_or_wrong_token(headers, monkeypatch, synth_db, clean_env):
    monkeypatch.setenv("NFL_MCP_AUTH_TOKEN", TOKEN)
    with TestClient(server.create_app()) as client:
        assert _post_init(client, headers).status_code == 401


def test_auth_accepts_right_bearer(monkeypatch, synth_db, clean_env):
    monkeypatch.setenv("NFL_MCP_AUTH_TOKEN", TOKEN)
    with TestClient(server.create_app()) as client:
        resp = _post_init(client, {"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code in (200, 202)
    assert '"serverInfo"' in resp.text


def test_no_auth_required_when_env_unset(synth_db, clean_env):
    with TestClient(server.create_app()) as client:
        resp = _post_init(client)
    assert resp.status_code == 200 and '"serverInfo"' in resp.text


@pytest.mark.parametrize("headers, status", [(None, 401), ({"Authorization": f"Bearer {TOKEN}"}, 200)])
def test_healthz_is_protected_by_auth(headers, status, monkeypatch, synth_db, clean_env):
    monkeypatch.setenv("NFL_MCP_AUTH_TOKEN", TOKEN)
    with TestClient(server.create_app()) as client:
        assert client.get("/healthz", headers=headers or {}).status_code == status


def test_preflight_needs_no_token(monkeypatch, synth_db, clean_env):
    monkeypatch.setenv("NFL_MCP_AUTH_TOKEN", TOKEN)
    with TestClient(server.create_app()) as client:
        resp = client.options("/mcp", headers={"Origin": "http://localhost:5173",
                                               "Access-Control-Request-Method": "POST"})
    assert resp.status_code == 200


def test_auth_uses_compare_digest():
    # Weak proxy: proves hmac.compare_digest is called somewhere in server.py, not that it is the only path.
    assert "compare_digest" in inspect.getsource(server)


# ── 7. CORS ─────────────────────────────────────────────────────────────────────

def _preflight(client, origin):
    return client.options("/mcp", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                           "Access-Control-Request-Headers": "content-type"})


@pytest.mark.parametrize("env", ["https://a.example,https://b.example", " https://a.example , https://b.example "])
@pytest.mark.parametrize("origin", ["https://a.example", "https://b.example"])
def test_cors_allows_listed_origin(env, origin, monkeypatch, synth_db, clean_env):
    monkeypatch.setenv("NFL_MCP_CORS_ORIGINS", env)
    with TestClient(server.create_app()) as client:
        assert _preflight(client, origin).headers.get("access-control-allow-origin") == origin


@pytest.mark.parametrize("origin", ["https://evil.example", "http://a.example"])
def test_cors_rejects_unlisted_origin(origin, monkeypatch, synth_db, clean_env):
    monkeypatch.setenv("NFL_MCP_CORS_ORIGINS", "https://a.example,https://b.example")
    with TestClient(server.create_app()) as client:
        assert "access-control-allow-origin" not in _preflight(client, origin).headers


def test_cors_unset_allows_any_origin(synth_db, clean_env):
    with TestClient(server.create_app()) as client:
        assert _preflight(client, "https://evil.example").headers["access-control-allow-origin"] == "*"


def test_cors_unset_logs_warning(synth_db, clean_env, caplog):
    with caplog.at_level(logging.WARNING, logger="nfl-mcp"):
        with TestClient(server.create_app()):
            pass
    assert any(r.name == "nfl-mcp" and r.levelno >= logging.WARNING and "cors" in r.getMessage().lower()
               for r in caplog.records)


def test_cors_set_logs_no_cors_warning(monkeypatch, synth_db, clean_env, caplog):
    monkeypatch.setenv("NFL_MCP_CORS_ORIGINS", "https://a.example")
    with caplog.at_level(logging.WARNING, logger="nfl-mcp"):
        with TestClient(server.create_app()):
            pass
    assert not any(r.levelno >= logging.WARNING and "cors" in r.getMessage().lower() for r in caplog.records)


# ── 8. /healthz ─────────────────────────────────────────────────────────────────

def test_healthz_ok_on_synthetic_db(synth_db, clean_env):
    with TestClient(server.create_app()) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200 and resp.json() == {"ok": True}


@pytest.mark.integration
def test_healthz_ok_on_real_db(monkeypatch, clean_env):
    from pathlib import Path
    real = Path("/tmp/grpE_v3.duckdb")
    if not real.exists():
        pytest.skip("real DB missing")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(real))
    with TestClient(server.create_app()) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200 and resp.json()["ok"] is True


@pytest.mark.parametrize("rel", ["missing.duckdb", "no_dir/missing.duckdb"])
def test_healthz_503_when_db_missing(rel, tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(tmp_path / rel))
    with TestClient(server.create_app()) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 503 and resp.json()["ok"] is False


# ── 9. per-call logging ─────────────────────────────────────────────────────────

def _call_records(caplog, tool):
    return [r for r in caplog.records if r.name == "nfl-mcp" and tool in r.getMessage()
            and re.search(r"duration", r.getMessage())]


@pytest.mark.parametrize("sql, rows, truncated", [
    ("SELECT x FROM t", 3, "false"),             # t has 3 hand-written rows
    ("SELECT * FROM range(150)", 100, "true"),   # 150 rows, default max_rows 100
])
def test_call_log_has_tool_duration_rows_truncated(sql, rows, truncated, synth_db, clean_env, caplog):
    with caplog.at_level(logging.INFO, logger="nfl-mcp"):
        call("nfl_query", {"sql": sql})
    recs = _call_records(caplog, "nfl_query")
    assert len(recs) == 1
    msg = recs[0].getMessage()
    assert re.search(r"duration(_ms|_s)?\s*[=:]\s*\d", msg)
    assert re.search(rf"\b(rows|row_count)\s*[=:]\s*{rows}\b", msg)
    assert re.search(rf"\btruncated\s*[=:]\s*{truncated}\b", msg, re.IGNORECASE)


def test_call_log_records_error_code(synth_db, clean_env, caplog):
    with caplog.at_level(logging.INFO, logger="nfl-mcp"):
        call("nfl_query", {"sql": "SELECT * FROM no_such_table_xyz"})
    recs = _call_records(caplog, "nfl_query")
    assert len(recs) == 1 and re.search(r"\b(error|code)\s*[=:]", recs[0].getMessage())


def test_call_log_cuts_long_sql(synth_db, clean_env, caplog):
    sql = "SELECT '" + "a" * 5000 + "' AS s"
    with caplog.at_level(logging.INFO, logger="nfl-mcp"):
        call("nfl_query", {"sql": sql})
    recs = _call_records(caplog, "nfl_query")
    assert len(recs) == 1 and len(recs[0].getMessage()) < 1000


@pytest.mark.parametrize("sql", [
    "SELECT '{a}' AS s",
    # Today the error path logs the DuckDB binder message with the full column name (tools.py:290).
    "SELECT no_such_col_{a} FROM t",
])
def test_no_log_record_carries_full_long_sql(sql, synth_db, clean_env, caplog):
    long_sql = sql.format(a="a" * 5000)
    with caplog.at_level(logging.DEBUG):
        call("nfl_query", {"sql": long_sql})
    for r in caplog.records:
        assert "a" * 1000 not in r.getMessage() + (r.exc_text or "")


def test_auth_token_never_logged(monkeypatch, synth_db, clean_env, caplog):
    monkeypatch.setenv("NFL_MCP_AUTH_TOKEN", TOKEN)
    wrong = "grpC-wrong-token-fedcba9876543210"
    with caplog.at_level(logging.DEBUG):
        with TestClient(server.create_app()) as client:
            _post_init(client)
            _post_init(client, {"Authorization": f"Bearer {wrong}"})
            _post_init(client, {"Authorization": f"Bearer {TOKEN}"})
            client.get("/healthz", headers={"Authorization": f"Bearer {TOKEN}"})
        call("nfl_query", {"sql": "SELECT 1 AS one"})
    for r in caplog.records:
        text = r.getMessage() + (r.exc_text or "")
        assert TOKEN not in text and wrong not in text


# ── 10. review fixes ────────────────────────────────────────────────────────────

def test_tool_error_dict_with_message_and_tool_keys_keeps_envelope(monkeypatch, clean_env):
    patch_tool(monkeypatch, "nfl_status", lambda: {"error": "boom", "message": "m", "tool": "t"})
    result = call("nfl_status", {})
    payload = json.loads(text_of(result))
    assert result.isError is True
    assert payload["error"] == "boom" and isinstance(payload["code"], str)
    assert payload["details"]["message"] == "m"


@pytest.mark.parametrize("raw", ["inf", "nan", "-1", "0", "abc"])
def test_timeout_env_non_finite_or_invalid_falls_back_to_default(raw, monkeypatch):
    monkeypatch.setenv("NFL_MCP_QUERY_TIMEOUT_SECONDS", raw)
    assert tools._query_timeout() == tools._QUERY_TIMEOUT_SECONDS


def test_auth_env_token_is_stripped(monkeypatch, synth_db, clean_env):
    monkeypatch.setenv("NFL_MCP_AUTH_TOKEN", f"  {TOKEN}\n")
    with TestClient(server.create_app()) as client:
        assert _post_init(client, {"Authorization": f"Bearer {TOKEN}"}).status_code == 200
        assert _post_init(client).status_code == 401


def test_auth_whitespace_only_token_means_unset_with_warning(monkeypatch, synth_db, clean_env, caplog):
    monkeypatch.setenv("NFL_MCP_AUTH_TOKEN", "   ")
    with caplog.at_level(logging.WARNING, logger="nfl-mcp"):
        with TestClient(server.create_app()) as client:
            assert _post_init(client).status_code == 200
    assert any("NFL_MCP_AUTH_TOKEN" in r.getMessage() for r in caplog.records)


def test_log_filter_survives_format_args_mismatch():
    record = logging.LogRecord("nfl-mcp", logging.INFO, __file__, 1, "mismatch %s %s", (1,), None)
    assert server._CutLongRecords().filter(record) is True


@pytest.mark.parametrize("value", [" ", "  \t\n"])
def test_serve_exits_1_on_whitespace_only_auth_token(value, monkeypatch):
    from click.testing import CliRunner
    from nfl_mcp.cli import main

    started = []
    monkeypatch.setattr("nfl_mcp.cli.uvicorn.run", lambda *a, **k: started.append(1))
    monkeypatch.setenv("NFL_MCP_AUTH_TOKEN", value)
    result = CliRunner().invoke(main, ["serve"])
    assert result.exit_code == 1
    assert "NFL_MCP_AUTH_TOKEN" in result.output
    assert started == []
