"""Tests for server error payload formatting."""

import asyncio
import json

from mcp.types import ListToolsRequest

import nfl_mcp.server as server_module
from nfl_mcp.server import _error_payload, _tool_error_payload, create_app


def test_error_payload_for_unknown_tool():
    payload = _error_payload("UNKNOWN_TOOL", "Unknown tool: bad_tool", tool="bad_tool")
    assert payload["error"] == "Unknown tool: bad_tool"
    assert payload["code"] == "UNKNOWN_TOOL"
    assert payload["details"]["tool"] == "bad_tool"


def test_error_payload_for_timeout():
    payload = _tool_error_payload("nfl_query", TimeoutError("Query exceeded timeout"))
    assert payload["code"] == "TIMEOUT"


def test_error_payload_for_invalid_arguments():
    payload = _tool_error_payload("nfl_query", TypeError("missing required argument"))
    assert payload["code"] == "INVALID_ARGUMENTS"


def test_list_tools_returns_registered_tools():
    tools = asyncio.run(server_module.list_tools())
    names = {tool.name for tool in tools}
    assert {"nfl_schema", "nfl_query", "nfl_schedule"}.issubset(names)


def test_list_tools_registered_with_mcp_server():
    assert ListToolsRequest in server_module._mcp_server.request_handlers


def test_call_tool_success_with_arguments(monkeypatch):
    monkeypatch.setattr(server_module, "nfl_query", lambda sql: {"rows": [{"sql": sql}]})
    result = asyncio.run(server_module.call_tool("nfl_query", {"sql": "SELECT 1"}))
    payload = json.loads(result.content[0].text)
    assert result.isError is False
    assert payload["rows"][0]["sql"] == "SELECT 1"


def test_call_tool_unknown_name_returns_error_payload():
    result = asyncio.run(server_module.call_tool("no_such_tool", {}))
    payload = json.loads(result.content[0].text)
    assert result.isError is True
    assert payload["code"] == "UNKNOWN_TOOL"


def test_call_tool_tool_exception_returns_execution_error(monkeypatch):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(server_module, "nfl_status", _boom)
    result = asyncio.run(server_module.call_tool("nfl_status", None))
    payload = json.loads(result.content[0].text)
    assert result.isError is True
    assert payload["code"] == "TOOL_EXECUTION_ERROR"
    assert payload["error"] == "boom"
    assert payload["details"]["tool"] == "nfl_status"


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


def test_every_exposed_tool_has_a_handler():
    """A tool listed in TOOLS with no dispatch entry would fail only at call time."""
    exposed = {tool.name for tool in server_module.TOOLS}
    assert "nfl_team_records" in exposed
    for name in sorted(exposed):
        payload = json.loads(asyncio.run(server_module.call_tool(name, {})).content[0].text)
        # Only the server reports UNKNOWN_TOOL.
        assert payload.get("code") != "UNKNOWN_TOOL", name


def test_failure_from_result_keeps_data_as_of():
    payload = server_module._failure_from_result(
        "nfl_schedule", {"error": "boom", "code": "TOOL_ERROR", "data_as_of": "2024-09-01T00:00:00Z"})
    assert payload["data_as_of"] == "2024-09-01T00:00:00Z"
    assert payload["error"] == "boom" and "data_as_of" not in payload["details"]
