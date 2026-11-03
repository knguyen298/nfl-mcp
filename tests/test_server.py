"""Tests for server error payload formatting."""

import asyncio
import inspect
import json

import pytest
from mcp.types import ListToolsRequest

import nfl_mcp.server as server_module
from nfl_mcp.server import _tool_error_payload, create_app


def test_error_payload_for_unknown_tool():
    payload = _tool_error_payload("bad_tool", ValueError("Unknown tool: bad_tool"))
    assert payload["ok"] is False
    assert payload["error"]["code"] == "UNKNOWN_TOOL"
    assert payload["error"]["tool"] == "bad_tool"


def test_error_payload_for_timeout():
    payload = _tool_error_payload("nfl_query", TimeoutError("Query exceeded timeout"))
    assert payload["error"]["code"] == "TIMEOUT"


def test_error_payload_for_invalid_arguments():
    payload = _tool_error_payload("nfl_query", TypeError("missing required argument"))
    assert payload["error"]["code"] == "INVALID_ARGUMENTS"


def test_list_tools_returns_registered_tools():
    tools = asyncio.run(server_module.list_tools())
    names = {tool.name for tool in tools}
    assert {"nfl_schema", "nfl_query", "nfl_schedule"}.issubset(names)


def test_list_tools_registered_with_mcp_server():
    assert ListToolsRequest in server_module._mcp_server.request_handlers


def test_call_tool_success_with_arguments(monkeypatch):
    monkeypatch.setitem(server_module._DISPATCH, "nfl_query", lambda sql: {"rows": [{"sql": sql}]})
    result = asyncio.run(server_module.call_tool("nfl_query", {"sql": "SELECT 1"}))
    payload = json.loads(result[0].text)
    assert payload["rows"][0]["sql"] == "SELECT 1"


def test_call_tool_unknown_name_returns_error_payload():
    result = asyncio.run(server_module.call_tool("no_such_tool", {}))
    payload = json.loads(result[0].text)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "UNKNOWN_TOOL"


def test_call_tool_tool_exception_returns_execution_error(monkeypatch):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setitem(server_module._DISPATCH, "nfl_status", _boom)
    result = asyncio.run(server_module.call_tool("nfl_status", None))
    payload = json.loads(result[0].text)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "TOOL_EXECUTION_ERROR"
    assert payload["error"]["tool"] == "nfl_status"


def test_create_app_returns_starlette_app():
    from starlette.applications import Starlette
    app = create_app()
    assert isinstance(app, Starlette)
    route_paths = [r.path for r in app.routes]
    assert "/mcp" in route_paths


def test_create_app_handles_cors_preflight():
    """Browser clients (e.g. MCP Inspector) send an OPTIONS preflight that must
    not 405. The CORS middleware should answer it with permissive headers."""
    from starlette.testclient import TestClient

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
    assert set(tool.inputSchema["properties"]) == {"team", "season", "from_season", "to_season"}


def test_player_stats_schema_lists_exactly_the_arguments_the_function_takes():
    """A parameter added to the tool without its TOOLS entry is invisible to the client."""
    tool = next(tool for tool in server_module.TOOLS if tool.name == "nfl_player_stats")
    assert set(tool.inputSchema["properties"]) == set(
        inspect.signature(server_module.nfl_player_stats).parameters)


def test_player_stat_tools_name_the_table_the_numbers_come_from():
    """Group B rule: the text the model reads says where a metric comes from."""
    described = {tool.name: tool.description for tool in server_module.TOOLS
                 if tool.name in ("nfl_player_stats", "nfl_compare")}
    assert set(described) == {"nfl_player_stats", "nfl_compare"}
    for name, text in described.items():
        assert "player_stats" in text, name
        assert "season_type" in text, name
    assert "sack" in described["nfl_player_stats"]      # the definition the old count got wrong


def test_call_tool_dispatches_team_records(monkeypatch):
    seen = {}

    def fake(team=None, season=None, from_season=None, to_season=None):
        seen.update(team=team, season=season, from_season=from_season, to_season=to_season)
        return {"records": [{"season": 1996, "team": "GB", "wins": 13}], "count": 1}

    monkeypatch.setitem(server_module._DISPATCH, "nfl_team_records", fake)
    result = asyncio.run(server_module.call_tool("nfl_team_records", {"team": "OAK", "season": 1996}))
    payload = json.loads(result[0].text)
    assert payload["count"] == 1
    assert seen == {"team": "OAK", "season": 1996, "from_season": None, "to_season": None}


def test_every_exposed_tool_has_a_handler():
    """A tool listed in TOOLS with no dispatch entry would fail only at call time."""
    exposed = {tool.name for tool in server_module.TOOLS}
    assert "nfl_team_records" in exposed
    for name in sorted(exposed):
        payload = json.loads(asyncio.run(server_module.call_tool(name, {}))[0].text)
        error = payload.get("error")
        # Tools report failures as a plain string; only the server reports UNKNOWN_TOOL.
        code = error.get("code") if isinstance(error, dict) else None
        assert code != "UNKNOWN_TOOL", name


@pytest.mark.parametrize("name", ["nfl_team_stats", "nfl_compare", "nfl_separation_opportunity"])
def test_the_tools_entry_lists_every_argument_the_function_takes(name):
    """season_type was added to all three; a parameter with no TOOLS entry is invisible."""
    tool = next(tool for tool in server_module.TOOLS if tool.name == name)
    assert set(tool.inputSchema["properties"]) == set(
        inspect.signature(getattr(server_module, name)).parameters)
    if name != "nfl_compare":
        assert "season_type" in tool.inputSchema["properties"]


def test_the_team_tool_text_names_where_the_numbers_come_from():
    """Group B rule: the text the model reads says which table or flag produced a number."""
    described = {tool.name: tool.description for tool in server_module.TOOLS
                 if tool.name in ("nfl_team_stats", "nfl_compare", "nfl_separation_opportunity")}
    assert set(described) == {"nfl_team_stats", "nfl_compare", "nfl_separation_opportunity"}
    team = described["nfl_team_stats"]
    assert "team_stats_raw" in team and "schedules" in team and "scrimmage plays" in team
    assert "season_type" in team and "playoff" in team
    # def6ef5 moved the per-play rates onto the box: yards_per_play, yards_per_rush, yards_per_pass
    # and yards_per_play_allowed divide the team_stats_raw totals in the row they sit on, and only
    # what no box carries (expected points, explosive plays) is counted from plays. The text used
    # to tell the model that any per-play figure came from plays over scrimmage plays only.
    assert "yards-per-play" in team.split("scrimmage plays")[0]
    assert "a per-play figure comes from plays" not in team
    assert "season_type" in described["nfl_compare"]
    assert "week = 0" in described["nfl_separation_opportunity"]
    assert "never averaged together" in described["nfl_separation_opportunity"]


def test_the_compare_tool_no_longer_says_team_mode_ignores_season_type():
    described = {tool.name: tool for tool in server_module.TOOLS if tool.name == "nfl_compare"}
    tool = described["nfl_compare"]
    assert "ignores" not in tool.description
    params = tool.inputSchema["properties"]["season_type"]["description"]
    assert "ignores" not in params and "Both modes apply it" in params


def test_the_separation_tool_offers_only_a_season_type_mcp_accepts():
    """Its description told the model to pass null for both scopes. mcp.server validates the
    arguments against inputSchema before the handler runs (mcp/server/lowlevel/server.py, the
    jsonschema.validate call that answers 'Input validation error: ...'), and this parameter is typed
    "string", so season_type=null never reached Python: the re-reviewer's call returned
    `Input validation error: None is not of type 'string'`. The literal 'ALL' is the way to ask for
    both scopes now, and the schema has to accept it."""
    import jsonschema

    tool = next(t for t in server_module.TOOLS if t.name == "nfl_separation_opportunity")
    spec = tool.inputSchema["properties"]["season_type"]
    assert spec["enum"] == ["REG", "POST", "ALL"]
    assert "ALL" in spec["description"]
    assert "pass null" not in spec["description"].lower()
    assert "null for both" not in tool.description.lower()
    assert "'ALL' for both" in tool.description
    jsonschema.validate({"season_type": "ALL"}, tool.inputSchema)
    jsonschema.validate({"season_type": "REG"}, tool.inputSchema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"season_type": None}, tool.inputSchema)


def test_no_other_tool_promises_a_null_its_schema_rejects():
    """A description that says pass null for a property the schema types as a string is a promise the
    server cannot keep, because mcp validates before it dispatches."""
    for tool in server_module.TOOLS:
        for name, spec in tool.inputSchema["properties"].items():
            text = (spec.get("description") or "").lower()
            promised_null = "pass null" in text or "= null" in text or "null for" in text
            if promised_null:
                assert "null" in json.dumps(spec.get("type")), f"{tool.name}.{name}"
