"""Tests for the MCP server: tool listing, dispatch, and the one error shape on the wire."""

import asyncio
import inspect
import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest

import nfl_mcp.server as server_module
import nfl_mcp.tools as tools
from nfl_mcp.registry import REGISTRY
from nfl_mcp.server import _error_payload, _is_error_result, _tool_error_payload, create_app
from nfl_mcp.status import Availability, ErrorCode


def _call(name, arguments):
    """Call the handler function directly; returns the CallToolResult it built."""
    return asyncio.run(server_module.call_tool(name, arguments))
from nfl_mcp.server import _error_payload, _tool_error_payload, create_app


def test_error_payload_for_unknown_tool():
    payload = _error_payload("UNKNOWN_TOOL", "Unknown tool: bad_tool", tool="bad_tool")
    assert payload["error"] == "Unknown tool: bad_tool"
    assert payload["code"] == "UNKNOWN_TOOL"
    assert payload["details"]["tool"] == "bad_tool"


def _wire(name, arguments):
    """Call through the CallToolRequest handler the SDK registered on the server."""
    handler = server_module._mcp_server.request_handlers[CallToolRequest]
    request = CallToolRequest(params=CallToolRequestParams(name=name, arguments=arguments))
    return asyncio.run(handler(request)).root


def _body(result):
    return json.loads(result.content[0].text)


def _assert_error_shape(body, code):
    assert isinstance(body["error"], str)
    assert body["code"] == code


# ── _tool_error_payload: exceptions that escaped a tool ────────────────────────

def test_error_payload_has_a_string_message_and_the_rest_under_details():
    # Merge: main's wire shape wins: error and code at the top, the rest under details.
    payload = _tool_error_payload("nfl_status", RuntimeError("boom"))
    assert payload == {
        "error": "boom",
        "code": ErrorCode.TOOL_EXECUTION_ERROR,
        "details": {"tool": "nfl_status", "type": "RuntimeError"},
    }


def test_error_payload_for_timeout_is_a_backstop():
    # No real tool reaches this branch: every tool catches TimeoutError from _execute
    # and returns its own TIMEOUT body. It stays for a tool that someday does not.
    payload = _tool_error_payload("nfl_query", TimeoutError("Query exceeded timeout"))
    assert payload["code"] == "TIMEOUT"
    assert payload["details"]["type"] == "TimeoutError"


def test_error_payload_for_invalid_arguments():
    # Merge: main's mapping wins. A TypeError that escapes a tool is reported as
    # INVALID_ARGUMENTS; the branch reported TOOL_EXECUTION_ERROR.
    payload = _tool_error_payload("nfl_query", TypeError("missing required argument"))
    assert payload["code"] == "INVALID_ARGUMENTS"


def test_every_error_code_value_is_recognised_as_an_error_body():
    for code in ErrorCode:
        assert _is_error_result({"error": "x", "code": code.value})


@pytest.mark.parametrize("result", [
    {"rows": [], "row_count": 0},
    {"error": {"code": "SQL_ERROR"}, "ok": False},   # the old nested envelope
    {"status": Availability.RECORD_NOT_AVAILABLE, "team": "CIN", "season": 1975},
    {"plays": {"error": "x", "code": ErrorCode.SQL_ERROR}, "datasets": []},
    {"error": {"offense": [], "defense": []}, "code": {"offense": [], "defense": []}},
    [{"error": "x", "code": "SQL_ERROR"}],
    "error",
    None,
])
def test_success_shapes_are_not_error_bodies(result):
    assert not _is_error_result(result)


@pytest.mark.parametrize("result", [
    {"error": "text but no code"},
    {"error": "x", "code": "NOT_A_REAL_CODE"},
])
def test_a_string_error_is_an_error_body_whatever_its_code(result):
    # Merge: main's rule wins. A string error is a failure; one with no code is reported
    # as TOOL_ERROR. The branch required a known ErrorCode value.
    assert _is_error_result(result)
    assert server_module._failure_from_result("nfl_status", result)["code"] == (
        result.get("code") or "TOOL_ERROR")


# ── call_tool ──────────────────────────────────────────────────────────────────

def test_list_tools_returns_registered_tools():
    tools_listed = asyncio.run(server_module.list_tools())
    names = {tool.name for tool in tools_listed}
    assert {"nfl_schema", "nfl_query", "nfl_schedule"}.issubset(names)


def test_list_tools_registered_with_mcp_server():
    assert ListToolsRequest in server_module._mcp_server.request_handlers


def test_server_instructions_describe_the_error_shape():
    instructions = server_module._mcp_server.instructions
    assert "`error`" in instructions and "`code`" in instructions


def test_call_tool_success_with_arguments(monkeypatch):
    monkeypatch.setattr(server_module, "nfl_query", lambda sql: {"rows": [{"sql": sql}]})
    result = _call("nfl_query", {"sql": "SELECT 1"})
    assert result.isError is False
    assert _body(result)["rows"][0]["sql"] == "SELECT 1"


def test_call_tool_accepts_none_arguments(monkeypatch):
    monkeypatch.setattr(server_module, "nfl_status", lambda: {"ok": 1})
    result = _call("nfl_status", None)
    assert result.isError is False
    assert _body(result) == {"ok": 1}


def test_call_tool_unknown_name_returns_error_payload():
    result = _call("no_such_tool", {})
    assert result.isError is True
    body = _body(result)
    _assert_error_shape(body, "UNKNOWN_TOOL")
    assert body["details"]["tool"] == "no_such_tool"


def test_call_tool_tool_exception_returns_execution_error(monkeypatch):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(server_module, "nfl_status", _boom)
    result = _call("nfl_status", None)
    assert result.isError is True
    body = _body(result)
    _assert_error_shape(body, "TOOL_EXECUTION_ERROR")
    assert body["error"] == "boom"
    # Merge: main's error shape keeps everything but error and code under details.
    assert body["details"]["tool"] == "nfl_status"
    assert body["details"]["type"] == "RuntimeError"


def test_call_tool_type_error_inside_a_tool_is_reported_as_invalid_arguments(monkeypatch):
    # Merge: main's TypeError mapping wins over the branch's TOOL_EXECUTION_ERROR.
    def _buggy(sql, max_rows=100):
        return None + 1

    monkeypatch.setattr(server_module, "nfl_query", _buggy)
    result = _call("nfl_query", {"sql": "SELECT 1"})
    assert result.isError is True
    body = _body(result)
    _assert_error_shape(body, "INVALID_ARGUMENTS")
    assert body["details"]["type"] == "TypeError"


def test_call_tool_unknown_argument_names_the_parameter():
    result = _call("nfl_roster", {"team": "KC", "colour": "red"})
    assert result.isError is True
    body = _body(result)
    _assert_error_shape(body, "INVALID_ARGUMENTS")
    assert "colour" in body["error"]


def test_call_tool_missing_argument_names_the_parameter():
    # The call does not fit the signature, so the TypeError names the parameter.
    result = _call("nfl_query", {})
    body = _body(result)
    _assert_error_shape(body, "INVALID_ARGUMENTS")
    assert "'sql'" in body["error"]


def test_call_tool_signature_check_catches_what_the_schema_does_not(monkeypatch):
    # A function that needs an argument its schema does not mark required.
    def _needs_extra(sql, extra):
        raise AssertionError("must not be called")

    monkeypatch.setattr(server_module, "nfl_query", _needs_extra)
    result = _call("nfl_query", {"sql": "SELECT 1"})
    assert result.isError is True
    body = _body(result)
    _assert_error_shape(body, "INVALID_ARGUMENTS")
    assert "'extra'" in body["error"]


def test_call_tool_schema_invalid_argument_uses_the_same_shape():
    result = _call("nfl_query", {"sql": "SELECT 1", "max_rows": "lots"})
    assert result.isError is True
    body = _body(result)
    _assert_error_shape(body, "INVALID_ARGUMENTS")
    assert body["error"].startswith("Input validation error: ")
    assert "'lots'" in body["error"]


def test_call_tool_tool_returned_error_body_is_unchanged(monkeypatch):
    returned = {"error": "Only SELECT queries are allowed.", "code": ErrorCode.NOT_SELECT}
    monkeypatch.setattr(server_module, "nfl_query", lambda sql: returned)
    result = _call("nfl_query", {"sql": "DROP TABLE plays"})
    assert result.isError is True
    body = _body(result)
    body.pop("data_as_of", None)   # main stamps every failure with the database date
    assert body == {"error": "Only SELECT queries are allowed.", "code": "NOT_SELECT",
                    "details": {"tool": "nfl_query"}}


def test_call_tool_availability_status_is_not_an_error(monkeypatch):
    answer = {"records": [], "status": Availability.FRANCHISE_INACTIVE, "count": 0}
    monkeypatch.setattr(server_module, "nfl_team_records", lambda **kw: answer)
    result = _call("nfl_team_records", {"team": "CLE", "season": 1997})
    assert result.isError is False
    assert _body(result)["status"] == "franchise_inactive"


def test_call_tool_compare_entity_named_error_is_not_an_error(monkeypatch):
    # Merge: main names the arguments player1/player2 and team1/team2.
    def _fake_compare(player1, player2, **kw):
        return {player1: {"offense": [], "defense": []}, player2: {"offense": [], "defense": []}}

    monkeypatch.setattr(server_module, "nfl_compare", _fake_compare)
    result = _call("nfl_compare", {"player1": "error", "player2": "code", "season_type": "REG"})
    assert result.isError is False
    assert set(_body(result)) == {"error", "code"}


# ── The wire: the handler the SDK registered, and a real client session ────────

def test_wire_tool_error_sets_is_error(monkeypatch):
    returned = {"error": "no such column", "code": ErrorCode.COLUMN_NOT_FOUND, "did_you_mean": ["epa"]}
    monkeypatch.setattr(server_module, "nfl_query", lambda sql, max_rows=100: returned)
    result = _wire("nfl_query", {"sql": "SELECT play_epa FROM plays"})
    assert result.isError is True
    body = _body(result)
    body.pop("data_as_of", None)   # main stamps every failure with the database date
    assert body == {"error": "no such column", "code": "COLUMN_NOT_FOUND",
                    "details": {"did_you_mean": ["epa"], "tool": "nfl_query"}}


def test_wire_success_clears_is_error(monkeypatch):
    monkeypatch.setattr(server_module, "nfl_status", lambda: {"total_plays": 3})
    result = _wire("nfl_status", {})
    assert result.isError is False
    assert _body(result) == {"total_plays": 3}


def test_wire_unknown_tool_uses_the_same_shape():
    result = _wire("no_such_tool", {})
    assert result.isError is True
    _assert_error_shape(_body(result), "UNKNOWN_TOOL")


def test_wire_schema_invalid_call_uses_the_same_shape():
    # The SDK's own check would answer with bare text; call_tool runs it instead.
    result = _wire("nfl_team_stats", {"team": "KC", "side": "special", "season_type": "REG"})
    assert result.isError is True
    body = _body(result)
    _assert_error_shape(body, "INVALID_ARGUMENTS")
    assert "'special' is not one of" in body["error"]


def test_wire_max_rows_above_the_cap_is_refused_by_the_schema():
    # Merge: main's schema declares maximum 500, so the wire refuses 9999 in the error shape.
    # The function itself still clamps (the next test, and test_safety).
    result = _wire("nfl_query", {"sql": "SELECT 1 AS n", "max_rows": 9999})
    assert result.isError is True
    body = _body(result)
    _assert_error_shape(body, "INVALID_ARGUMENTS")
    assert "9999" in body["error"]


def test_max_rows_above_the_cap_is_clamped_by_the_function(monkeypatch):
    seen = []

    def fake_fetch(sql, limit):
        # Merge: main runs the SQL as written and fetches limit rows.
        seen.append(limit)
        return [{"name": "n", "type": "INTEGER"}], [{"n": i} for i in range(600)][:limit]

    monkeypatch.setattr(tools, "_fetch_capped", fake_fetch)
    body = tools.nfl_query("SELECT 1 AS n", max_rows=9999)
    assert body["row_count"] == 500
    assert body["truncated"] is True
    assert seen == [501]


def test_client_session_sees_is_error_end_to_end(monkeypatch):
    monkeypatch.setattr(server_module, "nfl_status", lambda: {"total_plays": 3})

    async def _run():
        async with create_connected_server_and_client_session(server_module._mcp_server) as client:
            good = await client.call_tool("nfl_status", {})
            bad = await client.call_tool("nfl_query", {"sql": "SELECT 1", "max_rows": "lots"})
            return good, bad

    good, bad = asyncio.run(_run())
    assert good.isError is False
    assert json.loads(good.content[0].text) == {"total_plays": 3}
    assert bad.isError is True
    _assert_error_shape(json.loads(bad.content[0].text), "INVALID_ARGUMENTS")


def test_create_app_returns_starlette_app():
    from starlette.applications import Starlette
    app = create_app()
    assert isinstance(app, Starlette)
    route_paths = [r.path for r in app.routes]
    assert "/mcp" in route_paths


def test_create_app_handles_cors_preflight(monkeypatch):
    """Browser clients (e.g. MCP Inspector) send an OPTIONS preflight that must
    not 405. With NFL_MCP_CORS_ORIGINS unset the CORS middleware answers it
    with permissive headers."""
    from starlette.testclient import TestClient

    monkeypatch.delenv("NFL_MCP_CORS_ORIGINS", raising=False)
    app = create_app()
    with TestClient(app) as client:
        resp = client.options(
            "/mcp",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "*"
    assert "POST" in resp.headers["access-control-allow-methods"]


def test_create_app_lifespan_runs(monkeypatch):
    """Exercise the lifespan context manager (lines 480-483) with a mocked session manager."""
    import contextlib
    import asyncio
    from starlette.testclient import TestClient

    entered = []
    exited = []

    @contextlib.asynccontextmanager
    async def fake_run():
        entered.append(True)
        yield
        exited.append(True)

    # Patch StreamableHTTPSessionManager so .run() uses our fake
    import nfl_mcp.server as srv

    class FakeManager:
        def __init__(self, *args, **kwargs):
            pass
        def run(self):
            return fake_run()
        async def handle_request(self, scope, receive, send):
            pass

    monkeypatch.setattr(srv, "StreamableHTTPSessionManager", FakeManager)
    app = srv.create_app()

    with TestClient(app):
        pass  # triggers lifespan startup and shutdown

    assert entered == [True]
    assert exited == [True]


def test_team_records_is_exposed_with_its_arguments():
    tool = next(tool for tool in server_module.TOOLS if tool.name == "nfl_team_records")
    assert set(tool.inputSchema["properties"]) == {
        "team", "season", "season_from", "season_to", "limit", "offset"}


def test_call_tool_dispatches_team_records(monkeypatch):
    seen = {}

    def fake(team=None, season=None, season_from=None, season_to=None):
        seen.update(team=team, season=season, season_from=season_from, season_to=season_to)
        return {"records": [{"season": 1996, "team": "GB", "wins": 13}], "count": 1}

    monkeypatch.setattr(server_module, "nfl_team_records", fake)
    result = asyncio.run(server_module.call_tool("nfl_team_records", {"team": "OAK", "season": 1996}))
    payload = json.loads(result.content[0].text)
    assert payload["count"] == 1
    assert seen == {"team": "OAK", "season": 1996, "season_from": None, "season_to": None}


def test_dispatch_and_tools_list_the_same_names():
    """A tool listed without a handler, or a handler nobody can call, is a mistake."""
    assert set(server_module._DISPATCH) == {tool.name for tool in server_module.TOOLS}
    assert len(server_module.TOOLS) == len(server_module._DISPATCH)


def test_every_schema_property_is_a_parameter_of_its_function():
    for tool in server_module.TOOLS:
        params = inspect.signature(server_module._DISPATCH[tool.name]).parameters
        assert set(tool.inputSchema.get("properties", {})) <= set(params), tool.name
        required = {n for n, p in params.items() if p.default is inspect.Parameter.empty}
        # Merge: main's schemas also require season_type, which the function itself
        # checks (missing season_type returns INVALID_ARGUMENT). Nothing else may differ.
        schema_required = set(tool.inputSchema.get("required", []))
        assert required <= schema_required, tool.name
        assert schema_required - required <= {"season_type"}, tool.name


def test_numeric_bounds_are_only_the_row_caps():
    # Merge: main declares the row caps in the schema (1..500 rows, offset from 0); the
    # branch declared none. No other argument may carry a bound.
    for tool in server_module.TOOLS:
        for name, prop in tool.inputSchema.get("properties", {}).items():
            bounds = {k: prop[k] for k in ("minimum", "maximum") if k in prop}
            if name in ("max_rows", "limit"):
                assert bounds == {"minimum": 1, "maximum": 500}, (tool.name, name)
            elif name == "offset":
                assert bounds == {"minimum": 0}, (tool.name, name)
            else:
                assert not bounds, (tool.name, name)


def test_every_exposed_tool_has_a_handler():
    """Every listed tool answers a call with no arguments, and never with UNKNOWN_TOOL."""
    exposed = {tool.name for tool in server_module.TOOLS}
    assert "nfl_team_records" in exposed
    for name in sorted(exposed):
        result = _call(name, {})
        body = _body(result)
        # Only the server reports UNKNOWN_TOOL.
        assert result.isError is _is_error_result(body), name
        if result.isError:
            assert isinstance(body["error"], str), name
            assert body["code"] != "UNKNOWN_TOOL", name


# ── Tool descriptions the model reads before it calls anything ─────────────────

def _tool(name):
    return next(tool for tool in server_module.TOOLS if tool.name == name)


def test_only_nfl_status_tells_the_model_to_call_it_first():
    # Both health tools used to say "Call this first", which gave the model two equally
    # urgent answers. nfl_status is the first call; nfl_catalog is per-dataset coverage.
    assert [t.name for t in server_module.TOOLS if "Call this first" in t.description] == ["nfl_status"]
    assert "nfl_status is the first call" in _tool("nfl_catalog").description


def test_health_tool_descriptions_advertise_the_ingest_outcome_fields():
    assert "issues" in _tool("nfl_status").description
    catalog = _tool("nfl_catalog").description
    assert "issues" in catalog
    assert "missing_seasons" in catalog


def test_search_plays_season_description_matches_the_play_by_play_floor():
    # Play-by-play starts at the registry's pbp floor (1999); the description used to
    # say 2013, which is where a model stopped trying.
    description = _tool("nfl_search_plays").inputSchema["properties"]["season"]["description"]
    assert str(REGISTRY["pbp"].min_season) in description
    assert "2013" not in description


def test_every_team_argument_tells_the_model_which_codes_work():
    team_properties = [
        (tool.name, name, prop) for tool in server_module.TOOLS
        for name, prop in tool.inputSchema.get("properties", {}).items()
        if name in ("team", "opponent", "entity1")
    ]
    assert team_properties
    for tool_name, name, prop in team_properties:
        text = prop["description"]
        assert "OAK" in text, (tool_name, name)


def test_season_descriptions_name_the_first_season_the_tool_refuses_below():
    for tool in server_module.TOOLS:
        if tool.name not in tools._COVERAGE or tools.first_season(tool.name) is None:
            continue
        description = tool.inputSchema["properties"]["season"]["description"]
        assert f"Data starts in {tools.first_season(tool.name)}" in description, tool.name
        assert "OUT_OF_COVERAGE" in description, tool.name


def test_team_records_description_no_longer_promises_a_final_record():
    description = _tool("nfl_team_records").description
    assert "final regular-season" not in description
    assert "complete" in description
    assert "gaps" in description


def test_failure_from_result_keeps_data_as_of():
    payload = server_module._failure_from_result(
        "nfl_schedule", {"error": "boom", "code": "TOOL_ERROR", "data_as_of": "2024-09-01T00:00:00Z"})
    assert payload["data_as_of"] == "2024-09-01T00:00:00Z"
    assert payload["error"] == "boom" and "data_as_of" not in payload["details"]
