"""
NFL MCP Server
Tools: nfl_schema, nfl_status, nfl_query, nfl_search_plays, nfl_team_stats, nfl_player_stats, nfl_player_lookup, nfl_compare,
       nfl_catalog, nfl_roster, nfl_injuries, nfl_schedule, nfl_snap_counts, nfl_team_records
"""

import contextlib
import datetime as dt
import functools
import hmac
import json
import logging
import math
import os
import time
import traceback
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import anyio
from anyio import to_thread
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import CallToolResult, Resource, ResourceTemplate, TextContent, Tool, ToolAnnotations
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from .database import get_db_connection
from .discovery import DATA_NOTES, table_info
from .tools import (
    nfl_schema, nfl_status, nfl_query, nfl_search_plays,
    nfl_team_stats, nfl_player_stats, nfl_player_lookup, nfl_compare,
    nfl_catalog, nfl_roster, nfl_injuries, nfl_schedule, nfl_snap_counts,
    nfl_fantasy_opportunity, nfl_fantasy_rankings, nfl_ftn_charting,
    nfl_td_luck, nfl_role_trend, nfl_separation_opportunity,
    nfl_drop_rate, nfl_contract_value, nfl_injury_return,
    nfl_team_records,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nfl-mcp")

_mcp_server = Server("nfl-mcp")

_DEFAULT_MAX_CONCURRENT = 2
_LOG_FIELD_CHARS = 200
_LOG_MESSAGE_CHARS = 800
_LOG_TRACE_LINE_CHARS = 300
_TOOL_LIMITER: anyio.CapacityLimiter | None = None


class _CutLongRecords(logging.Filter):
    """Cut oversized messages and traceback lines (a pasted SQL statement can be thousands of chars)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except (TypeError, ValueError, KeyError):
            return True  # format/args mismatch: leave it for logging.handleError
        if len(message) > _LOG_MESSAGE_CHARS:
            record.msg, record.args = message[:_LOG_MESSAGE_CHARS] + "...", None
        if record.exc_info and not record.exc_text:
            lines = "".join(traceback.format_exception(*record.exc_info)).rstrip().splitlines()
            record.exc_text = "\n".join(_cut(line, _LOG_TRACE_LINE_CHARS) for line in lines)
        return True


logger.addFilter(_CutLongRecords())


def _cut(text: Any, limit: int = _LOG_FIELD_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _clean(obj: Any) -> Any:
    """Make a value strict-JSON safe: NaN/Inf -> None, Decimal -> number, dates -> ISO 8601."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Decimal):
        if not obj.is_finite():
            return None
        return int(obj) if obj == obj.to_integral_value() and obj.as_tuple().exponent >= 0 else float(obj)
    if isinstance(obj, (dt.datetime, dt.date, dt.time)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def to_json(obj: Any) -> str:
    """Serialize a tool result as compact, strict JSON (no NaN/Infinity, no indentation)."""
    return json.dumps(_clean(obj), allow_nan=False, default=str)


def get_tool_limiter() -> anyio.CapacityLimiter:
    """Return the limiter shared by all tool calls, created on first use inside the running loop."""
    global _TOOL_LIMITER
    if _TOOL_LIMITER is None:
        raw = os.getenv("NFL_MCP_MAX_CONCURRENT_QUERIES")
        try:
            tokens = int(raw) if raw else _DEFAULT_MAX_CONCURRENT
        except ValueError:
            tokens = 0
        if tokens < 1:
            logger.warning("Invalid NFL_MCP_MAX_CONCURRENT_QUERIES=%r; using %d",
                           _cut(raw, 40), _DEFAULT_MAX_CONCURRENT)
            tokens = _DEFAULT_MAX_CONCURRENT
        _TOOL_LIMITER = anyio.CapacityLimiter(tokens)
    return _TOOL_LIMITER


def _error_payload(code: str, message: str, /, **details: Any) -> dict[str, Any]:
    """The one error shape: top-level error and code, everything else under details."""
    return {"error": message, "code": code, "details": details}


def _tool_error_payload(tool_name: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, TimeoutError):
        code = "TIMEOUT"
    elif isinstance(exc, TypeError):
        code = "INVALID_ARGUMENTS"
    else:
        code = "TOOL_EXECUTION_ERROR"
    return _error_payload(code, str(exc), tool=tool_name, type=type(exc).__name__)


def _failure_from_result(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Convert a tool's {"error": ...} dict into the one error shape."""
    extras = {k: v for k, v in result.items() if k not in ("error", "code", "data_as_of")}
    message = str(result["error"])
    code = result.get("code")
    if not isinstance(code, str):
        code = "TIMEOUT" if "timeout" in message.lower() else "TOOL_ERROR"
    payload = _error_payload(code, message, **{**extras, "tool": tool_name})
    if "data_as_of" in result:
        payload["data_as_of"] = result["data_as_of"]
    return payload


def _row_count(result: Any) -> Any:
    if isinstance(result, dict):
        for key in ("row_count", "count"):
            if isinstance(result.get(key), int):
                return result[key]
        for value in result.values():
            if isinstance(value, list):
                return len(value)
    return "-"


def _log_call(name: str, arguments: dict, started: float, result: Any, failure: dict | None) -> None:
    """One INFO record per call: tool, duration, rows, truncated, and error/code on failure."""
    parts = [
        f"tool={name}",
        f"duration_ms={(time.perf_counter() - started) * 1000:.1f}",
        f"rows={_row_count(result)}",
    ]
    truncated = result.get("truncated") if isinstance(result, dict) else None
    parts.append(f"truncated={str(truncated).lower() if isinstance(truncated, bool) else '-'}")
    if failure is not None:
        parts += [f"code={failure['code']}", f"error={_cut(failure['error'])!r}"]
    if arguments:
        parts.append(f"args={_cut(to_json(arguments))}")
    logger.info(" ".join(parts))


TOOLS = [
    Tool(
        name="nfl_schema",
        description=(
            "Returns the NFL database schema. By default returns a compact summary with "
            "key columns and available categories. Pass category='<name>' for full column "
            "details on a specific section, or category='all' for everything. "
            "Pass table='<name>' to get the live column list, grain and join keys for any "
            "table. The default call also returns relationships (joins checked against the "
            "loaded data), data_notes (coverage and caveats) and derived_tables. The same "
            "facts are the MCP resources nfl://schema, nfl://relationships and nfl://data-notes."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Schema category to expand. Omit for summary. Use 'all' for full schema. "
                        "Options: game_context, teams, game_situation, play_details, timeouts, "
                        "score, boolean_outcomes, primary_players, special_teams_players, "
                        "defensive_players, fumble_players, penalties, probability_models, "
                        "epa, wpa, completion_probability, xyac, drive_data, "
                        "game_stadium_weather, vegas, aggregate_tables, query_tips"
                    ),
                },
                "table": {
                    "type": "string",
                    "description": "Table name to inspect (e.g. 'rosters', 'injuries', 'schedules'). Returns live column list from the database.",
                },
            },
        },
    ),
    Tool(
        name="nfl_status",
        description=(
            "Returns database health: play counts, loaded seasons, and a summary of all ingested "
            "datasets with row counts, the newest season and week, and the last refresh time "
            "of each. Every tool response also carries data_as_of, the newest load time of the "
            "tables that tool read. Call this first to understand what data is available."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="nfl_query",
        description=(
            "Run one read-only SQL statement. Use it for joins, aggregates and questions the "
            "other tools do not cover (nfl_search_plays, nfl_team_stats, nfl_player_stats and "
            "nfl_compare answer the common play, team and player questions faster). "
            "Call nfl_schema first, and copy column names from it, "
            "never guess them (per-play EPA is `epa`, not `play_epa`). "
            "One statement per call: SELECT (including a WITH common table expression, CTE), "
            "DESCRIBE, SUMMARIZE or EXPLAIN. Other statement types are rejected. "
            "max_rows 1-500 (default 100, larger values are clamped to 500). 10s timeout."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "One SQL SELECT or EXPLAIN statement.",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Max rows to return (default 100, max 500)",
                    "default": 100,
                    "minimum": 1,
                    "maximum": 500,
                },
            },
            "required": ["sql"],
        },
    ),
    Tool(
        name="nfl_search_plays",
        description=(
            "PREFERRED for finding specific plays. Search by player (and the role the "
            "player had in the play), team, season, week, play type, situation (red zone, "
            "third/fourth down, two-minute drill, goal to go), down, quarter, field "
            "position, score differential, shotgun, no-huddle, or win probability, plus "
            "touchdowns, turnovers or minimum yards. Returns one row per play, from 1999 on. Rows "
            "carry game_id and play_id (play_id repeats across games, so pair it with "
            "game_id) and the passer, rusher and receiver ids. Sorted by impact (ABS EPA) unless sort_by says "
            "otherwise; page with limit and offset. Use this instead of nfl_query "
            "whenever looking for plays."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": "Offensive team abbreviation (e.g. KC, PHI, BUF). Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work.",
                },
                "opponent": {
                    "type": "string",
                    "description": "Defensive team abbreviation. Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work.",
                },
                "player": {
                    "type": "string",
                    "description": (
                        "Player name, e.g. 'Mahomes', 'Patrick Mahomes' or 'J.Jefferson'. It "
                        "resolves through the players table. A name that matches several "
                        "players returns ambiguous candidates; then pass player_id. The team "
                        "argument does not narrow the name (team is the offense)."
                    ),
                },
                "player_id": {
                    "type": "string",
                    "description": "Player gsis_id (from nfl_player_lookup). Takes precedence over a name.",
                },
                "player_role": {
                    "type": "string",
                    "enum": ["passer", "rusher", "receiver", "any_offense", "defender",
                             "kicker", "punter", "returner", "any"],
                    "description": (
                        "Role the player had in the play. Needs player or player_id. passer, "
                        "rusher, receiver: that id column. any_offense: any of the three. "
                        "defender: sack, tackle, assist, interception, pass defense, forced "
                        "fumble, QB hit or tackle for loss. kicker, punter. returner: kickoff "
                        "or punt returner. any (default): every role."
                    ),
                },
                "season": {
                    "type": "integer",
                    "description": "Exact season year (1999 through the current season). Use season_from/season_to for ranges.",
                },
                "season_from": {
                    "type": "integer",
                    "description": "Start of season range (inclusive), e.g. 2020 for 'since 2020'",
                },
                "season_to": {
                    "type": "integer",
                    "description": "End of season range (inclusive)",
                },
                "week": {
                    "type": "integer",
                    "description": "Week number (1-22)",
                },
                "season_type": {
                    "type": "string",
                    "enum": ["REG", "POST"],
                    "description": "Filter by season type: REG (regular season) or POST (playoffs). Playoff rounds (WC, DIV, CON, SB) are not valid here.",
                },
                "play_type": {
                    "type": "string",
                    "enum": ["pass", "run", "field_goal", "punt", "kickoff", "extra_point",
                             "qb_kneel", "qb_spike", "no_play"],
                    "description": "Type of play. Plays with no type are never selected by this filter.",
                },
                "situation": {
                    "type": "string",
                    "enum": ["red_zone", "third_down", "fourth_down", "two_minute",
                             "two_minute_half", "two_minute_game", "goal_to_go"],
                    "description": (
                        "Game situation. red_zone: yardline_100 <= 20. third_down, fourth_down. "
                        "two_minute: 2:00 or less left in the half in quarter 2, 4 or overtime. "
                        "two_minute_half: quarter 2 only. two_minute_game: quarter 4 and overtime. "
                        "goal_to_go."
                    ),
                },
                "is_touchdown": {
                    "type": "boolean",
                    "description": "Only touchdown plays",
                },
                "is_turnover": {
                    "type": "boolean",
                    "description": "Only turnover plays (INT or fumble lost)",
                },
                "min_yards": {
                    "type": "integer",
                    "description": "Minimum yards gained",
                },
                "down": {"type": "integer", "description": "Down (1-4)."},
                "qtr": {"type": "integer", "description": "Quarter (1-4; 5 is overtime)."},
                "min_yardline_100": {
                    "type": "number",
                    "description": "Minimum yards from the opponent end zone (inclusive; 1 is the goal line, 99 is own 1).",
                },
                "max_yardline_100": {
                    "type": "number",
                    "description": "Maximum yards from the opponent end zone (inclusive).",
                },
                "min_score_differential": {
                    "type": "number",
                    "description": "Minimum offense lead (offense score minus defense score), inclusive. Negative means trailing.",
                },
                "max_score_differential": {
                    "type": "number",
                    "description": "Maximum offense lead, inclusive.",
                },
                "shotgun": {"type": "boolean", "description": "true: shotgun plays only. false: not in shotgun."},
                "no_huddle": {"type": "boolean", "description": "true: no-huddle plays only. false: huddle plays."},
                "min_wp": {"type": "number", "description": "Minimum offense win probability (0-1), inclusive."},
                "max_wp": {"type": "number", "description": "Maximum offense win probability (0-1), inclusive."},
                "sort_by": {
                    "type": "string",
                    "enum": ["abs_epa", "epa", "wpa", "yards_gained", "game_order"],
                    "default": "abs_epa",
                    "description": (
                        "abs_epa (default): biggest EPA swing first, either direction. epa, wpa, "
                        "yards_gained: highest first. game_order: game_id then play_id ascending."
                    ),
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Extra plays columns to return (for example air_yards, cpoe), checked "
                        "against the live schema. An unknown name returns an error with did_you_mean."
                    ),
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Alias of limit, kept for older callers.",
                    "minimum": 1,
                    "maximum": 500,
                },
            },
        },
    ),
    Tool(
        name="nfl_team_stats",
        description=(
            "PREFERRED for any team-level question. Returns pre-aggregated offense, "
            "defense, and situational stats. Always use this instead of nfl_query "
            "for questions like 'how did [team] do', team rankings, or team efficiency."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": "Team abbreviation (e.g. KC, PHI, BUF, BAL). Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work.",
                },
                "season": {
                    "type": "integer",
                    "description": "Season year. Omit for all available seasons.",
                },
                "season_from": {
                    "type": "integer",
                    "description": "Start of season range (inclusive), e.g. 2020 for 'since 2020'",
                },
                "season_to": {
                    "type": "integer",
                    "description": "End of season range (inclusive)",
                },
                "side": {
                    "type": "string",
                    "enum": ["offense", "defense", "situational", "both"],
                    "description": "Which stats to return (default: both offense + defense + situational)",
                    "default": "both",
                },
                "season_type": {
                    "type": "string",
                    "enum": ["REG", "POST", "ALL"],
                    "description": (
                        "REG (regular season, default), POST (playoffs), or ALL for both "
                        "(one row per season and season type, never summed together)"
                    ),
                    "default": "REG",
                },
            },
            "required": ["team"],
        },
    ),
    Tool(
        name="nfl_player_lookup",
        description=(
            "Find a player by name and get the gsis_id that the other player tools take as "
            "player_id. Use it when a name could match several players (e.g. 'Allen', 'Josh "
            "Allen', 'J.Williams'). Ignores case, accents, punctuation and Jr/II suffixes. "
            "Returns candidates with position, teams, first and last season, and pfr_id. "
            "Zero matches is count 0, not an error."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Full name, surname, or short name (e.g. 'Josh Allen', 'Allen', "
                        "'K.Allen', \"Ja'Marr Chase\")"
                    ),
                },
                "team": {
                    "type": "string",
                    "description": "Keep players with a roster row for this team (in `season` when given).",
                },
                "season": {
                    "type": "integer",
                    "description": "Keep players with a roster row in this season (rosters cover 2024 on).",
                },
                "position": {
                    "type": "string",
                    "description": "Position filter (e.g. QB, WR).",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="nfl_player_stats",
        description=(
            "PREFERRED for any player stats question. Returns season-by-season passing, "
            "rushing and receiving stats for one player, split by season type, with the "
            "definitions used (attempts exclude sacks and two-point tries; targets exclude "
            "two-point tries). Always use this instead of nfl_query for questions about a "
            "player's performance, stats, or trends. If the name matches several players the "
            "result is {ambiguous: true, candidates: [...]}: call again with player_id."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "player_name": {
                    "type": "string",
                    "description": (
                        "Player name, e.g. 'Patrick Mahomes' or 'Mahomes', or a gsis_id such as "
                        "'00-0034857'. With `season` set, only players on a roster that season "
                        "are considered. Ignored when player_id is given."
                    ),
                },
                "player_id": {
                    "type": "string",
                    "description": (
                        "Player gsis_id, e.g. '00-0034857' (from nfl_player_lookup). "
                        "Takes precedence over player_name."
                    ),
                },
                "season": {
                    "type": "integer",
                    "description": "Exact season year. Omit for all seasons.",
                },
                "season_from": {
                    "type": "integer",
                    "description": "Start of season range (inclusive), e.g. 2020 for 'since 2020'",
                },
                "season_to": {
                    "type": "integer",
                    "description": "End of season range (inclusive)",
                },
                "season_type": {
                    "type": "string",
                    "enum": ["REG", "POST"],
                    "description": "Filter by season type: REG (regular season) or POST (playoffs). Omit for one row per type. Playoff rounds (WC, DIV, CON, SB) are not valid here.",
                },
                "stat_type": {
                    "type": "string",
                    "enum": ["passing", "rushing", "receiving"],
                    "description": "Return only this block (default: all three).",
                },
            },
            "required": ["player_name"],
        },
    ),
    Tool(
        name="nfl_compare",
        description=(
            "PREFERRED for any comparison question. Side-by-side stats for two teams "
            "or two players in a single call. Always use this instead of calling "
            "nfl_team_stats or nfl_player_stats twice when the user asks to compare, "
            "rank, or contrast two teams or two players against each other."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entity1": {
                    "type": "string",
                    "description": "First team abbreviation or player name. Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work.",
                },
                "entity2": {
                    "type": "string",
                    "description": "Second team abbreviation or player name. Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work.",
                },
                "compare_type": {
                    "type": "string",
                    "enum": ["team", "player"],
                    "description": "Whether comparing teams or players (default: team)",
                    "default": "team",
                },
                "season": {
                    "type": "integer",
                    "description": "Exact season year. Omit for all seasons.",
                },
                "season_from": {
                    "type": "integer",
                    "description": "Start of season range (inclusive), e.g. 2020 for 'since 2020'",
                },
                "season_to": {
                    "type": "integer",
                    "description": "End of season range (inclusive)",
                },
                "season_type": {
                    "type": "string",
                    "enum": ["REG", "POST"],
                    "description": (
                        "Filter by season type: REG (regular season) or POST (playoffs). "
                        "Team mode defaults to REG; a team with no rows for the filter "
                        "returns empty lists and a reason."
                    ),
                },
            },
            "required": ["entity1", "entity2"],
        },
    ),
    Tool(
        name="nfl_catalog",
        description=(
            "Returns a catalog of every table in the local database: grain (what one row is), "
            "live row count, first and last season, join keys, whether the table is derived "
            "(rebuilt from other tables at ingest), and when it was last loaded. "
            "Call this first to discover what data is available beyond play-by-play."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="nfl_roster",
        description=(
            "PREFERRED for roster questions. Look up who was on a team's roster "
            "in a given season. Returns name, position, jersey number, experience, "
            "college, height, and weight. Filter by team, season, and/or position."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": "Team abbreviation (e.g. KC, PHI, BUF). Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work.",
                },
                "season": {
                    "type": "integer",
                    "description": "Season year (e.g. 2024). Omit for all seasons.",
                },
                "position": {
                    "type": "string",
                    "description": "Position filter (e.g. QB, WR, LB). Partial match supported.",
                },
            },
        },
    ),
    Tool(
        name="nfl_injuries",
        description=(
            "PREFERRED for injury report questions. Look up player injury status by "
            "team, season, week, player name, or report status (Out, Questionable, "
            "Doubtful, Full Participation). Returns primary/secondary injuries and "
            "both report and practice statuses."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": "Team abbreviation (e.g. KC, PHI). Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work.",
                },
                "season": {
                    "type": "integer",
                    "description": "Season year. Omit for all seasons.",
                },
                "week": {
                    "type": "integer",
                    "description": "Week number (1–22).",
                },
                "player": {
                    "type": "string",
                    "description": "Player name (partial match, e.g. 'Mahomes')",
                },
                "report_status": {
                    "type": "string",
                    "description": "Injury designation: 'Out', 'Questionable', 'Doubtful', 'Full Participation In Practice'",
                },
            },
        },
    ),
    Tool(
        name="nfl_schedule",
        description=(
            "PREFERRED for schedule and game result questions. Look up games by team, "
            "season, week, or season type. Returns scores, spread/total lines, "
            "weather, stadium, coaches, and referee."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": "Team abbreviation — returns all games where team is home or away. Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work.",
                },
                "season": {
                    "type": "integer",
                    "description": "Season year. Omit for all seasons.",
                },
                "week": {
                    "type": "integer",
                    "description": "Week number (1–22).",
                },
                "season_type": {
                    "type": "string",
                    "enum": ["REG", "POST", "WC", "DIV", "CON", "SB"],
                    "description": (
                        "Season type filter: REG, POST (all four playoff rounds), or one round: "
                        "WC (wild card), DIV (divisional), CON (conference), SB (Super Bowl)."
                    ),
                },
            },
        },
    ),
    Tool(
        name="nfl_snap_counts",
        description=(
            "PREFERRED for snap count questions. Look up how many offensive, defensive, "
            "and special teams snaps a player or team unit played. Filter by player, "
            "team, season, week, or position."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "player": {
                    "type": "string",
                    "description": "Player name as stored in snap counts: full name or surname, e.g. 'Travis Kelce' or 'Kelce' (not 'T.Kelce')",
                },
                "player_id": {
                    "type": "string",
                    "description": (
                        "Player gsis_id, e.g. '00-0034857'. Takes precedence over a name. "
                        "Get it from nfl_player_lookup."
                    ),
                },
                "team": {
                    "type": "string",
                    "description": "Team abbreviation (e.g. KC, PHI). Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work.",
                },
                "season": {
                    "type": "integer",
                    "description": "Season year. Omit for all seasons.",
                },
                "week": {
                    "type": "integer",
                    "description": "Week number (1–22).",
                },
                "position": {
                    "type": "string",
                    "description": "Position filter (e.g. TE, WR, CB).",
                },
                "season_type": {
                    "type": "string",
                    "enum": ["REG", "POST", "WC", "DIV", "CON", "SB"],
                    "description": (
                        "Season type filter: REG, POST (all four playoff rounds), or one round: "
                        "WC (wild card), DIV (divisional), CON (conference), SB (Super Bowl)."
                    ),
                },
            },
        },
    ),
    Tool(
        name="nfl_fantasy_opportunity",
        description=(
            "PREFERRED for fantasy football opportunity questions. Look up target share, "
            "air yards share, carry share, and opportunity scores per player per week. "
            "Requires ff_opportunity dataset. Available 2006–present. Filter by player, "
            "team, season, week, or position."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "player": {
                    "type": "string",
                    "description": "Player name, e.g. 'Justin Jefferson'. A name that matches several players returns ambiguous candidates; then pass player_id.",
                },
                "player_id": {
                    "type": "string",
                    "description": (
                        "Player gsis_id, e.g. '00-0034857'. Takes precedence over a name. "
                        "Get it from nfl_player_lookup."
                    ),
                },
                "team": {
                    "type": "string",
                    "description": "Team abbreviation (e.g. KC, PHI). Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work.",
                },
                "season": {
                    "type": "integer",
                    "description": "Season year. Omit for all seasons.",
                },
                "week": {
                    "type": "integer",
                    "description": "Week number (1–22). Omit for full season.",
                },
                "position": {
                    "type": "string",
                    "description": "Position filter (e.g. WR, RB, TE).",
                },
            },
        },
    ),
    Tool(
        name="nfl_fantasy_rankings",
        description=(
            "PREFERRED for fantasy football ranking questions (expert consensus "
            "rankings / ECR, start-sit, draft and dynasty value). Two snapshots: "
            "scope='draft' for preseason/dynasty/best-ball draft rankings, scope='week' "
            "for the current week's start/sit rankings. Lower ECR = ranked higher. "
            "These are the latest scrape, not historical. Requires ff_rankings_draft / "
            "ff_rankings_week datasets. Filter by player, position, team, or ranking_set."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "player": {
                    "type": "string",
                    "description": "Player name (partial match, e.g. 'Jefferson')",
                },
                "position": {
                    "type": "string",
                    "description": "Position filter (e.g. QB, RB, WR, TE, K, DST).",
                },
                "team": {
                    "type": "string",
                    "description": "Team abbreviation (e.g. KC, PHI). Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["draft", "week"],
                    "description": "draft = preseason/dynasty/best-ball rankings; week = current-week start/sit rankings (default: draft).",
                    "default": "draft",
                },
                "ranking_set": {
                    "type": "string",
                    "description": "Filter the ranking list/format (partial match). For draft: 'redraft', 'dynasty', 'best', or position-specific like 'redraft-rb'. For week: 'ppr', 'qb', etc.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 50, max 500).",
                    "default": 50,
                },
            },
        },
    ),
    Tool(
        name="nfl_ftn_charting",
        description=(
            "PREFERRED for play-style / scheme tendency questions (2022–present). "
            "Returns aggregated FTN manual charting rates over offensive scrimmage plays "
            "(pass + run): play-action, RPO, screen, no-huddle, pre-snap motion, and "
            "trick-play usage, plus average defenders in the box (per scrimmage play) and "
            "pass rushers / blitzers (per dropback). Use for questions like 'how often "
            "does KC run play action' or 'what motion rate did Mahomes see'. Offensive "
            "perspective (team = offense). Requires ftn_charting dataset."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": "Offensive team abbreviation (e.g. KC, PHI). Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work.",
                },
                "opponent": {
                    "type": "string",
                    "description": "Defensive team abbreviation. Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work.",
                },
                "player": {
                    "type": "string",
                    "description": "Player name for one player (passer, rusher or receiver). A name that matches several players returns ambiguous candidates; then pass player_id.",
                },
                "player_id": {
                    "type": "string",
                    "description": (
                        "Player gsis_id, e.g. '00-0034857'. Takes precedence over a name. "
                        "Get it from nfl_player_lookup."
                    ),
                },
                "season": {
                    "type": "integer",
                    "description": "Exact season year (2022–present). Use season_from/season_to for ranges.",
                },
                "season_from": {
                    "type": "integer",
                    "description": "Start of season range (inclusive).",
                },
                "season_to": {
                    "type": "integer",
                    "description": "End of season range (inclusive).",
                },
                "week": {
                    "type": "integer",
                    "description": "Week number (1–22).",
                },
                "season_type": {
                    "type": "string",
                    "enum": ["REG", "POST"],
                    "description": "Filter by season type: REG or POST. Playoff rounds (WC, DIV, CON, SB) are not valid here.",
                },
            },
        },
    ),
    Tool(
        name="nfl_td_luck",
        description=(
            "PREFERRED for touchdown luck / TD regression questions. Returns actual vs "
            "expected receiving and rushing touchdowns per player-season. Negative "
            "total_td_luck_score = scored fewer TDs than expected (positive-regression / "
            "'unlucky' bounce-back candidate); positive = over-performed (sell-high). Use "
            "for 'who was unlucky on TDs in 2024' or 'is player X due for TD regression'. "
            "Default sort is most unlucky first."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "player": {"type": "string", "description": "Player name, e.g. 'Justin Jefferson'. A name that matches several players returns ambiguous candidates; then pass player_id."},
                "player_id": {"type": "string", "description": "Player gsis_id (from nfl_player_lookup). Takes precedence over a name."},
                "position": {"type": "string", "description": "Position filter (e.g. WR, RB, TE)."},
                "team": {"type": "string", "description": "Team abbreviation (e.g. KC). Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work."},
                "season": {"type": "integer", "description": "Exact season year."},
                "limit": {"type": "integer", "description": "Max rows (default 50, max 500)."},
            },
        },
    ),
    Tool(
        name="nfl_role_trend",
        description=(
            "PREFERRED for usage-trend / breakout / fade questions. Returns weekly snap %, "
            "target share, carry share, and air-yards share with a trailing 3-week average "
            "and the current-week delta vs that average. Use for 'whose role is trending up' "
            "or 'is player X's snap share rising'. Default sort is biggest snap-share gain "
            "first (snap_pct_delta DESC)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "player": {"type": "string", "description": "Player name, e.g. 'Justin Jefferson'. A name that matches several players returns ambiguous candidates; then pass player_id."},
                "player_id": {"type": "string", "description": "Player gsis_id (from nfl_player_lookup). Takes precedence over a name."},
                "position": {"type": "string", "description": "Position filter (e.g. WR, RB)."},
                "team": {"type": "string", "description": "Team abbreviation. Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work."},
                "season": {"type": "integer", "description": "Exact season year."},
                "week": {"type": "integer", "description": "Week number."},
                "min_snap_pct": {"type": "number", "description": "Minimum snap % (0–100) for the week."},
                "limit": {"type": "integer", "description": "Max rows (default 50, max 500)."},
            },
        },
    ),
    Tool(
        name="nfl_separation_opportunity",
        description=(
            "PREFERRED for receiver regression / 'getting open but not producing' questions "
            "(2016–present). Joins Next Gen Stats separation and YAC-above-expected to fantasy "
            "opportunity per player-season. regression_candidate=true flags receivers creating "
            "separation (>2.5 yds) who under-produced (fp_diff_per_game < -1.5, td_luck < -1.0). "
            "Default sort is most under-producing first (fp_diff_per_game ASC)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "player": {"type": "string", "description": "Player name, e.g. 'Justin Jefferson'. A name that matches several players returns ambiguous candidates; then pass player_id."},
                "player_id": {"type": "string", "description": "Player gsis_id (from nfl_player_lookup). Takes precedence over a name."},
                "position": {"type": "string", "description": "Position filter (e.g. WR, TE)."},
                "team": {"type": "string", "description": "Team abbreviation. Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work."},
                "season": {"type": "integer", "description": "Exact season year (2016+)."},
                "regression_candidate": {
                    "type": "boolean",
                    "description": "If true, only return flagged positive-regression candidates.",
                },
                "limit": {"type": "integer", "description": "Max rows (default 50, max 500)."},
            },
        },
    ),
    Tool(
        name="nfl_drop_rate",
        description=(
            "PREFERRED for drops / catchable-target reliability questions (2022–present). "
            "Returns share of catchable targets dropped per receiver-season from FTN charting, "
            "plus contested targets and created receptions. Use for 'who drops the most passes' "
            "or 'what is player X's drop rate'. Full names work. "
            "Default sort is highest drop rate first."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "player": {"type": "string", "description": "Player name, e.g. 'Justin Jefferson'. A name that matches several players returns ambiguous candidates; then pass player_id."},
                "player_id": {"type": "string", "description": "Player gsis_id (from nfl_player_lookup). Takes precedence over a name."},
                "team": {"type": "string", "description": "Team abbreviation. Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work."},
                "season": {"type": "integer", "description": "Exact season year (2022+)."},
                "min_targets": {"type": "integer", "description": "Minimum catchable targets to qualify."},
                "limit": {"type": "integer", "description": "Max rows (default 50, max 500)."},
            },
        },
    ),
    Tool(
        name="nfl_contract_value",
        description=(
            "PREFERRED for contract value / cost-efficiency questions. Returns fantasy points "
            "per $M of average per year (APY) using each player's active contract joined to "
            "seasonal fantasy production. Use for 'best fantasy value per dollar' or 'is player "
            "X worth his contract'. apy and cap_pct reflect the current active contract. Default "
            "sort is best value first (fp_per_million DESC)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "player": {"type": "string", "description": "Player name, e.g. 'Justin Jefferson'. A name that matches several players returns ambiguous candidates; then pass player_id."},
                "player_id": {"type": "string", "description": "Player gsis_id (from nfl_player_lookup). Takes precedence over a name."},
                "position": {"type": "string", "description": "Position filter (e.g. RB, WR)."},
                "team": {"type": "string", "description": "Team abbreviation. Case-insensitive; historical codes (OAK, SD, STL, JAC, WSH) also work."},
                "season": {"type": "integer", "description": "Exact season year."},
                "min_apy": {"type": "number", "description": "Minimum APY in $millions."},
                "max_apy": {"type": "number", "description": "Maximum APY in $millions."},
                "limit": {"type": "integer", "description": "Max rows (default 50, max 500)."},
            },
        },
    ),
    Tool(
        name="nfl_injury_return",
        description=(
            "PREFERRED for injury recovery / return-to-form questions. Returns post-return "
            "snap-share recovery as a percent of pre-injury baseline at +1..+8 weeks after an "
            "'Out' spell, bucketed by normalized injury type (hamstring, knee, ankle, …) and "
            "position. 100 = fully back to baseline usage. Use for 'how long until players "
            "recover snaps after a hamstring injury'. Default sort is by weeks since return."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "injury_type": {
                    "type": "string",
                    "description": (
                        "Normalized injury bucket (partial match). Valid buckets: hamstring, "
                        "knee, ankle, shoulder, concussion, groin, foot, calf, hip, back, "
                        "quadriceps, achilles, wrist, hand, elbow, toe, thigh, neck, ribs, "
                        "pectoral, other. Specific diagnoses are mapped to these buckets "
                        "(e.g. 'ACL' -> knee, 'hammy' -> hamstring); anything unmatched -> other."
                    ),
                },
                "position": {"type": "string", "description": "Position filter (e.g. WR, RB)."},
                "week_post_return": {
                    "type": "integer",
                    "description": "Specific week offset after return (1–8).",
                },
                "limit": {"type": "integer", "description": "Max rows (default 50, max 500)."},
            },
        },
    ),
    Tool(
        name="nfl_team_records",
        description=(
            "PREFERRED for win-loss, record and standings questions. Returns each "
            "franchise's final regular-season record, 1970 to present. Use the team's "
            "CURRENT abbreviation ('LV' for the Raiders in every era); rows include the "
            "name the club played that season under. Pre-1999 seasons need the optional "
            "'nfl-mcp ingest --pre1999' pull and report record_not_available if absent."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": (
                        "Team abbreviation — current or historical codes are accepted "
                        "(OAK, SD, STL resolve to LV, LAC, LA). Omit for all teams."
                    ),
                },
                "season": {
                    "type": "integer",
                    "description": "Season year (1970 or later).",
                },
                "from_season": {
                    "type": "integer",
                    "description": "Start of a season range, for streaks or era comparisons.",
                },
                "to_season": {
                    "type": "integer",
                    "description": "End of a season range (inclusive).",
                },
            },
        },
    ),
]


# Every list tool pages the same way. The tool returns count, truncated and, when more
# rows match, a hint that names offset and nfl_query.
_PAGED_TOOLS = frozenset({
    "nfl_roster", "nfl_injuries", "nfl_schedule", "nfl_snap_counts", "nfl_fantasy_opportunity",
    "nfl_fantasy_rankings", "nfl_search_plays", "nfl_td_luck", "nfl_role_trend",
    "nfl_separation_opportunity", "nfl_drop_rate", "nfl_contract_value", "nfl_injury_return",
    "nfl_team_records",
})
# What one row is, and the id field the row is keyed on. Several source tables hold an id the
# tool does not return; the sentence says so.
_GRAIN_SENTENCE = {
    "nfl_roster": " Returns one row per player per team per season. The source id is gsis_id; "
                  "rows do not carry it (nfl_player_lookup returns it).",
    "nfl_injuries": " Returns one row per player per team per week. The source id is gsis_id; "
                    "rows do not carry it (nfl_player_lookup returns it).",
    "nfl_schedule": " Returns one row per game, identified by game_id.",
    "nfl_snap_counts": " Returns one row per player per game. Rows carry pfr_player_id, the "
                       "Pro Football Reference id, which is not the gsis_id (players maps one to "
                       "the other).",
    "nfl_fantasy_opportunity": " Returns one row per player per game. player_id is the gsis_id.",
    "nfl_fantasy_rankings": " Returns one row per player per ranking set. The weekly source table "
                            "has fantasypros_id; rows do not carry it, so match players by name.",
    "nfl_td_luck": " Returns one row per player per season per season type (REG and POST rows both "
                   "appear, and rows do not say which). player_id is the gsis_id.",
    "nfl_role_trend": " Returns one row per player per week. player_id is the gsis_id.",
    "nfl_separation_opportunity": " Returns one row per player per season. player_id is the gsis_id.",
    "nfl_drop_rate": " Returns one row per player per season per season type (REG and POST rows both "
                     "appear, and rows do not say which). player_id is the gsis_id.",
    "nfl_contract_value": " Returns one row per player per season. player_id is the gsis_id.",
    "nfl_injury_return": " Returns one row per injury_type, position and week_post_return.",
    "nfl_team_records": " Returns one row per franchise per season, keyed on team.",
}
_PAGE_SENTENCE = (
    " Paged: limit (1-500, default 100) and offset. The result has count and truncated; when "
    "truncated is true, hint names the next offset."
)
for _tool in TOOLS:
    if _tool.name in _PAGED_TOOLS:
        _tool.inputSchema["properties"]["limit"] = {
            "type": "integer", "minimum": 1, "maximum": 500, "default": 100,
            "description": "Max rows on this page (default 100, max 500).",
        }
        _tool.inputSchema["properties"]["offset"] = {
            "type": "integer", "minimum": 0, "default": 0,
            "description": "Rows to skip, for the next page (default 0).",
        }
        _tool.description += _GRAIN_SENTENCE.get(_tool.name, "") + _PAGE_SENTENCE


# Every tool only reads the local database. Every result is a JSON object (an error is one too).
_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "data_as_of": {
            "type": "string",
            "description": "Newest load time (ISO 8601, UTC) of the tables this tool read.",
        },
    },
}
for _tool in TOOLS:
    _tool.annotations = _READ_ONLY
    _tool.outputSchema = _OUTPUT_SCHEMA


@_mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


# ── Resources ─────────────────────────────────────────────────────────────────
# The same discovery facts as nfl_schema, for clients that read resources instead of calling tools.

_RESOURCES = [
    Resource(uri="nfl://schema", name="schema", title="Database schema",
             description="Plays summary, every table with its grain and join keys, and the derived tables.",
             mimeType="text/plain"),
    Resource(uri="nfl://relationships", name="relationships", title="Table relationships",
             description="Joins between tables, with the columns to join on. Each was checked on loaded data.",
             mimeType="application/json"),
    Resource(uri="nfl://data-notes", name="data-notes", title="Data notes",
             description="Coverage start years, id systems and other caveats.",
             mimeType="text/plain"),
]
_SCHEMA_PREFIX = "nfl://schema/"


def _table_schema_text(table: str) -> str:
    info = nfl_schema(table=table)
    if "error" in info:
        raise ValueError(info["error"])
    lines = [f"Table: {table}", f"Grain: {info['grain']}", f"Join keys: {', '.join(info['join_keys']) or 'none'}",
             f"Derived: {'yes' if info['derived'] else 'no'}", "Columns:"]
    lines += [f"  {c['column_name']} {c['data_type']}" for c in info["columns"]]
    return "\n".join(lines)


def _read_resource_text(uri: str) -> tuple[str, str]:
    """(text, mime type) of one resource. Runs in a worker thread: it reads the database."""
    if uri == "nfl://schema":
        info = nfl_schema()
        lines = [info["schema"], "", "Tables (grain; join keys):"]
        for name in info["available_tables"]:
            grain, keys = table_info(name)
            flag = " [derived]" if name in info["derived_tables"] else ""
            lines.append(f"  {name}{flag}: {grain}; keys: {', '.join(keys) or 'none'}")
        return "\n".join(lines), "text/plain"
    if uri == "nfl://relationships":
        return to_json({"relationships": nfl_schema()["relationships"]}), "application/json"
    if uri == "nfl://data-notes":
        return "\n".join(f"- {n}" for n in DATA_NOTES), "text/plain"
    if uri.startswith(_SCHEMA_PREFIX) and len(uri) > len(_SCHEMA_PREFIX):
        return _table_schema_text(uri[len(_SCHEMA_PREFIX):]), "text/plain"
    raise ValueError(f"Unknown resource: {uri}")


@_mcp_server.list_resources()
async def list_resources() -> list[Resource]:
    return _RESOURCES


@_mcp_server.list_resource_templates()
async def list_resource_templates() -> list[ResourceTemplate]:
    return [ResourceTemplate(uriTemplate=_SCHEMA_PREFIX + "{table}", name="table-schema",
                             title="Schema of one table",
                             description="Grain, join keys and live columns of one table, e.g. nfl://schema/plays.",
                             mimeType="text/plain")]


@_mcp_server.read_resource()
async def read_resource(uri: Any) -> list[ReadResourceContents]:
    text, mime = await to_thread.run_sync(functools.partial(_read_resource_text, str(uri)),
                                          limiter=get_tool_limiter())
    return [ReadResourceContents(content=text, mime_type=mime)]


@_mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> CallToolResult:
    _DISPATCH = {
        "nfl_schema":       nfl_schema,
        "nfl_status":       nfl_status,
        "nfl_query":        nfl_query,
        "nfl_search_plays": nfl_search_plays,
        "nfl_team_stats":   nfl_team_stats,
        "nfl_player_stats": nfl_player_stats,
        "nfl_player_lookup": nfl_player_lookup,
        "nfl_compare":      nfl_compare,
        "nfl_catalog":      nfl_catalog,
        "nfl_roster":       nfl_roster,
        "nfl_injuries":     nfl_injuries,
        "nfl_schedule":     nfl_schedule,
        "nfl_snap_counts":          nfl_snap_counts,
        "nfl_fantasy_opportunity":  nfl_fantasy_opportunity,
        "nfl_fantasy_rankings":     nfl_fantasy_rankings,
        "nfl_ftn_charting":         nfl_ftn_charting,
        "nfl_td_luck":              nfl_td_luck,
        "nfl_role_trend":           nfl_role_trend,
        "nfl_separation_opportunity": nfl_separation_opportunity,
        "nfl_drop_rate":            nfl_drop_rate,
        "nfl_contract_value":       nfl_contract_value,
        "nfl_injury_return":        nfl_injury_return,
        "nfl_team_records":         nfl_team_records,
    }

    started = time.perf_counter()
    args = arguments if isinstance(arguments, dict) else {}
    result: Any = None
    failure: dict[str, Any] | None = None

    fn = _DISPATCH.get(name)
    tool = next((t for t in TOOLS if t.name == name), None)
    if fn is None:
        failure = _error_payload("UNKNOWN_TOOL", f"Unknown tool: {name}", tool=name)
    elif arguments is not None and not isinstance(arguments, dict):
        failure = _error_payload("INVALID_ARGUMENTS", "Tool arguments must be a JSON object.", tool=name)
    else:
        valid = list(tool.inputSchema.get("properties", {})) if tool else None
        unknown = [a for a in args if valid is not None and a not in valid]
        if unknown:
            listed = ", ".join(valid) if valid else "none (this tool takes no arguments)"
            failure = _error_payload(
                "INVALID_ARGUMENTS",
                f"Unknown argument{'s' if len(unknown) > 1 else ''} {', '.join(repr(a) for a in unknown)} "
                f"for {name}. Valid arguments: {listed}.",
                tool=name, unknown=unknown, valid=valid,
            )

    if failure is None:
        try:
            result = await to_thread.run_sync(functools.partial(fn, **args), limiter=get_tool_limiter())
        except Exception as e:
            logger.error("Tool %s raised %s: %s", name, type(e).__name__, _cut(e), exc_info=True)
            failure = _tool_error_payload(name, e)
        else:
            if isinstance(result, dict) and "error" in result:
                failure = _failure_from_result(name, result)

    _log_call(name, args, started, result, failure)
    payload = failure if failure is not None else result
    text = to_json(payload)
    return CallToolResult(content=[TextContent(type="text", text=text)],
                          structuredContent=json.loads(text) if isinstance(payload, dict) else None,
                          isError=failure is not None)


class _BearerAuth:
    """Pure ASGI middleware: every HTTP request needs `Authorization: Bearer <token>`."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            header = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
            scheme, _, supplied = header.partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(supplied.strip().encode(), self.token):
                response = JSONResponse({"error": "Unauthorized", "code": "UNAUTHORIZED", "details": {}},
                                        status_code=401, headers={"WWW-Authenticate": "Bearer"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _healthz(request) -> JSONResponse:
    """200 when the database opens and answers SELECT 1, else 503."""
    try:
        with get_db_connection() as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception as e:
        logger.warning("Health check failed: %s", _cut(e), exc_info=True)
        return JSONResponse({"ok": False, "error": type(e).__name__}, status_code=503)
    return JSONResponse({"ok": True})


def create_app() -> Starlette:
    """Return a Starlette ASGI app that serves the MCP server over Streamable HTTP.

    NFL_MCP_AUTH_TOKEN (bearer auth on every route) and NFL_MCP_CORS_ORIGINS
    (comma-separated) are read here, when the app is built.
    """
    session_manager = StreamableHTTPSessionManager(
        app=_mcp_server,
        json_response=False,
        stateless=False,
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        logger.info("Starting NFL MCP server (Streamable HTTP)")
        async with session_manager.run():
            yield
        logger.info("NFL MCP server stopped")

    from mcp.server.fastmcp.server import StreamableHTTPASGIApp

    origins = [o.strip() for o in os.getenv("NFL_MCP_CORS_ORIGINS", "").split(",") if o.strip()]
    if not origins:
        logger.warning("NFL_MCP_CORS_ORIGINS is not set; CORS allows every origin. "
                       "Set it to a comma-separated list of origins to restrict browser access.")
        origins = ["*"]

    # Browser-based clients (e.g. MCP Inspector) send a CORS preflight OPTIONS
    # request before every call. Without this middleware those preflights get
    # 405 Method Not Allowed and the connection never establishes. Expose
    # Mcp-Session-Id so the browser can read it back from responses.
    # CORS is outermost so preflights are answered without a token and 401s carry CORS headers.
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["Mcp-Session-Id"],
        )
    ]
    raw_token = os.getenv("NFL_MCP_AUTH_TOKEN")
    token = (raw_token or "").strip()
    if raw_token and not token:
        logger.warning("NFL_MCP_AUTH_TOKEN is only whitespace; auth is disabled.")
    if token:
        middleware.append(Middleware(_BearerAuth, token=token))

    return Starlette(
        lifespan=lifespan,
        middleware=middleware,
        routes=[
            Route("/mcp", endpoint=StreamableHTTPASGIApp(session_manager)),
            Route("/healthz", endpoint=_healthz, methods=["GET"]),
        ],
    )
