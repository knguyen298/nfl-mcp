"""
NFL MCP Server
Tools: nfl_schema, nfl_status, nfl_query, nfl_search_plays, nfl_team_stats, nfl_player_stats, nfl_compare,
       nfl_catalog, nfl_roster, nfl_injuries, nfl_schedule, nfl_snap_counts, nfl_team_records
"""

import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import anyio
import anyio.to_thread
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route

from .tools import (
    nfl_schema, nfl_status, nfl_query, nfl_search_plays,
    nfl_team_stats, nfl_player_stats, nfl_compare,
    nfl_catalog, nfl_roster, nfl_injuries, nfl_schedule, nfl_snap_counts,
    nfl_fantasy_opportunity, nfl_fantasy_rankings, nfl_ftn_charting,
    nfl_td_luck, nfl_role_trend, nfl_separation_opportunity,
    nfl_drop_rate, nfl_contract_value, nfl_injury_return,
    nfl_team_records,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nfl-mcp")

_mcp_server = Server("nfl-mcp")

# DuckDB's memory_limit is per connection, so cap how many tools run at once.
_MAX_CONCURRENT_TOOLS = 2
_TOOL_LIMITER = anyio.CapacityLimiter(_MAX_CONCURRENT_TOOLS)


def _tool_error_payload(tool_name: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ValueError) and str(exc).startswith("Unknown tool:"):
        code = "UNKNOWN_TOOL"
    elif isinstance(exc, TimeoutError):
        code = "TIMEOUT"
    elif isinstance(exc, TypeError):
        code = "INVALID_ARGUMENTS"
    else:
        code = "TOOL_EXECUTION_ERROR"
    return {
        "ok": False,
        "error": {
            "code": code,
            "type": type(exc).__name__,
            "message": str(exc),
            "tool": tool_name,
        },
    }


TOOLS = [
    Tool(
        name="nfl_schema",
        description=(
            "Returns the NFL database schema. By default returns a compact summary with "
            "key columns and available categories. Pass category='<name>' for full column "
            "details on a specific section, or category='all' for everything. "
            "Pass table='<name>' to get the live column list for any non-pbp table."
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
            "Returns database health: play counts, loaded seasons, and a summary of all ingested datasets with row counts and last refresh time. Call this first to understand what data is available."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="nfl_query",
        description=(
            "Execute raw read-only DuckDB SQL when nfl_search_plays, nfl_team_stats, "
            "nfl_player_stats, and nfl_compare cannot answer the question. "
            "Accepts full read-only SQL: CTEs (WITH), window functions, subqueries, PIVOT, "
            "DESCRIBE and SUMMARIZE. One statement per call; writes and file access are "
            "rejected. "
            "Requires calling nfl_schema first, and column names must be copied from it, "
            "never guessed (per-play EPA is `epa`, not `play_epa`). "
            "500 row cap. 10s timeout."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "One read-only SQL statement.",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Max rows to return (default 100, max 500)",
                    "default": 100,
                    "maximum": 500,
                },
            },
            "required": ["sql"],
        },
    ),
    Tool(
        name="nfl_search_plays",
        description=(
            "PREFERRED for finding specific plays. Search by player, team, season, "
            "week, play type, situation, touchdowns, turnovers, or minimum yards. "
            "Returns plays sorted by impact (EPA). Use this instead of nfl_query "
            "whenever looking for plays."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": "Offensive team abbreviation (e.g. KC, PHI, BUF)",
                },
                "opponent": {
                    "type": "string",
                    "description": "Defensive team abbreviation",
                },
                "player": {
                    "type": "string",
                    "description": "Player full name or nflverse short name ('Justin Jefferson' or 'J.Jefferson'). Matches passer, rusher, or receiver by gsis_id.",
                },
                "player_id": {
                    "type": "string",
                    "description": "Player gsis_id (e.g. '00-0036322'). Overrides player. Use it when a name returns ambiguous candidates.",
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
                    "description": "Week number (1–22)",
                },
                "season_type": {
                    "type": "string",
                    "enum": ["REG", "POST"],
                    "description": "Filter by season type: REG (regular season) or POST (playoffs). plays stores only these two values, so a round name (WC, DIV, CON, SB) is an error here; use nfl_schedule to pick one round.",
                },
                "play_type": {
                    "type": "string",
                    "enum": ["pass", "run", "field_goal", "punt", "kickoff"],
                    "description": "Type of play",
                },
                "situation": {
                    "type": "string",
                    "enum": ["red_zone", "third_down", "fourth_down", "two_minute"],
                    "description": "Game situation filter",
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
                "max_rows": {
                    "type": "integer",
                    "description": "Max results (default 50, max 500)",
                    "default": 50,
                },
            },
        },
    ),
    Tool(
        name="nfl_team_stats",
        description=(
            "PREFERRED for any team-level question. Returns pre-aggregated offense, "
            "defense, and situational stats. Always use this instead of nfl_query "
            "for questions like 'how did [team] do', team rankings, or team efficiency. "
            "Each row is one team, season and season_type: a playoff game is never added into "
            "the regular season, so pass season_type='REG' or 'POST' for one of them or leave it "
            "out to get both as separate rows. Counting stats come from the week-level team box "
            "(team_stats_raw) and points from schedules; a per-play figure comes from plays over "
            "scrimmage plays only. The response carries 'sources', which names the table, column "
            "or flags behind every column."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": "Team abbreviation (e.g. KC, PHI, BUF, BAL)",
                },
                "season": {
                    "type": "integer",
                    "description": "Season year. Omit for all available seasons.",
                },
                "season_type": {
                    "type": "string",
                    "enum": ["REG", "POST"],
                    "description": "REG (regular season) or POST (playoffs). Omit to return both as separate rows - they are never summed together.",
                },
                "side": {
                    "type": "string",
                    "enum": ["offense", "defense", "situational", "both"],
                    "description": "Which stats to return (default: both offense + defense + situational)",
                    "default": "both",
                },
            },
            "required": ["team"],
        },
    ),
    Tool(
        name="nfl_player_stats",
        description=(
            "PREFERRED for any player stats question. Returns one player's official season "
            "counting stats - the numbers ESPN and Pro Football Reference show - summed from "
            "the week-level player_stats table, one row per season and season_type. A sack is "
            "in neither attempts nor carries, a kneel is a carry, a spike is a pass attempt, "
            "and a two-point play is in neither. Passing yards are gross and sack_yards is "
            "negative, so net_yards is their sum. Every rate divides summed totals, and "
            "metric_sources names the table and column behind every metric; avg_cpoe is the "
            "mean of plays.cpoe over the plays that carry a value, because player_stats stores "
            "only a weekly mean. `unavailable` lists metrics this database cannot provide, and "
            "a rate with a zero denominator is null. avg_epa divides an EPA total that counts "
            "two-point plays by attempts plus sacks, carries or targets, which do not. A season "
            "whose counting stats are all zero is left out, and nfl_compare shows the same rows. "
            "Always use this instead of nfl_query for a player's performance or stats."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "player_name": {
                    "type": "string",
                    "description": "Player full name or nflverse short name ('Justin Jefferson' or 'J.Jefferson'). Omit when player_id is given.",
                },
                "player_id": {
                    "type": "string",
                    "description": "Player gsis_id (e.g. '00-0036322'). Overrides player_name. A name that matches several players returns ambiguous plus the candidates, never merged stats.",
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
                    "description": "Filter by season type: REG (regular season) or POST (playoffs). player_stats stores only these two values, so a round name (WC, DIV, CON, SB) is an error here; use nfl_schedule to pick one round. Rows are grouped per season type either way, so omitting it does not add the two together.",
                },
                "stat_type": {
                    "type": "string",
                    "enum": ["passing", "rushing", "receiving"],
                    "description": "Type of stats to return (default: passing)",
                    "default": "passing",
                },
            },
        },
    ),
    Tool(
        name="nfl_compare",
        description=(
            "PREFERRED for any comparison question. Side-by-side stats for two teams "
            "or two players in a single call. Always use this instead of calling "
            "nfl_team_stats or nfl_player_stats twice when the user asks to compare, "
            "rank, or contrast two teams or two players against each other. Team mode "
            "returns one entry per abbreviation; player mode returns {'players': [side1, "
            "side2]} in the order requested, each side carrying 'requested' plus 'player' "
            "with its gsis_id. Each player stat block is nfl_player_stats' rows, summed from "
            "the week-level player_stats table per season and season_type, and each side names "
            "those columns in 'metric_sources'. Team mode returns one entry per abbreviation, each "
            "with 'offense' and 'defense' rows per season and season_type plus the 'sources' map, "
            "and it honours season_type: pass 'REG' or 'POST' and a playoff game stays out of the "
            "regular season. Two players who share a name are two separate "
            "sides, and a side whose name matches several players returns 'candidates' instead "
            "of stats."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entity1": {
                    "type": "string",
                    "description": "First team abbreviation or player name (full or short form)",
                },
                "entity2": {
                    "type": "string",
                    "description": "Second team abbreviation or player name (full or short form)",
                },
                "compare_type": {
                    "type": "string",
                    "enum": ["team", "player"],
                    "description": "Whether comparing teams or players (default: team)",
                    "default": "team",
                },
                "player_id1": {
                    "type": "string",
                    "description": "gsis_id for entity1 (player mode only). Overrides the name in entity1.",
                },
                "player_id2": {
                    "type": "string",
                    "description": "gsis_id for entity2 (player mode only). Overrides the name in entity2.",
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
                    "description": "Filter by season type: REG (regular season) or POST (playoffs). Both modes apply it and keep the two season types in separate rows. Both tables store only these two values, so a round name (WC, DIV, CON, SB) is an error - use nfl_schedule to pick one round.",
                },
            },
            "required": ["entity1", "entity2"],
        },
    ),
    Tool(
        name="nfl_catalog",
        description=(
            "Returns a catalog of every dataset loaded into the local database: "
            "table name, row count, seasons available, and when it was last refreshed. "
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
                    "description": "Team abbreviation (e.g. KC, PHI, BUF)",
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
                    "description": "Team abbreviation (e.g. KC, PHI)",
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
                    "description": "Player full name or nflverse short name ('Patrick Mahomes' or 'P.Mahomes')",
                },
                "player_id": {
                    "type": "string",
                    "description": "Player gsis_id (e.g. '00-0036322'). Overrides player. Use it when a name returns ambiguous candidates.",
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
                    "description": "Team abbreviation — returns all games where team is home or away.",
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
                        "Season type filter: REG, or POST (any playoff round), or one round "
                        "- WC (wild card), DIV (divisional), CON (conference championship), "
                        "SB (Super Bowl)."
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
                    "description": "Player full name or nflverse short name ('Travis Kelce' or 'T.Kelce'). snap_counts rows are keyed by PFR id, mapped through the players table.",
                },
                "player_id": {
                    "type": "string",
                    "description": "Player gsis_id (e.g. '00-0030506'). Overrides player.",
                },
                "team": {
                    "type": "string",
                    "description": "Team abbreviation (e.g. KC, PHI)",
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
                    "description": "Player full name or nflverse short name ('Justin Jefferson' or 'J.Jefferson')",
                },
                "player_id": {
                    "type": "string",
                    "description": "Player gsis_id (e.g. '00-0036322'). Overrides player.",
                },
                "team": {
                    "type": "string",
                    "description": "Team abbreviation (e.g. KC, PHI)",
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
                    "description": "Player full name or nflverse short name ('Justin Jefferson' or 'J.Jefferson'). Resolved to a gsis_id, then matched to the FantasyPros id in ff_playerids.",
                },
                "player_id": {
                    "type": "string",
                    "description": "Player gsis_id (e.g. '00-0036322'). Overrides player.",
                },
                "position": {
                    "type": "string",
                    "description": "Position filter (e.g. QB, RB, WR, TE, K, DST).",
                },
                "team": {
                    "type": "string",
                    "description": (
                        "Team abbreviation (e.g. KC, PHI). Any code the club has used works, "
                        "including 'JAC' for the Jaguars in these tables. 'FA' is accepted "
                        "and filters the free agents these tables publish."
                    ),
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
                    "description": "Offensive team abbreviation (e.g. KC, PHI).",
                },
                "opponent": {
                    "type": "string",
                    "description": "Defensive team abbreviation.",
                },
                "player": {
                    "type": "string",
                    "description": "Player full name or nflverse short name — matches passer, rusher, or receiver by gsis_id.",
                },
                "player_id": {
                    "type": "string",
                    "description": "Player gsis_id (e.g. '00-0036322'). Overrides player.",
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
                    "description": "Filter by season type: REG or POST. plays stores only these two values, so a round name (WC, DIV, CON, SB) is an error here; use nfl_schedule to pick one round.",
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
                "player": {"type": "string", "description": "Player full name or nflverse short name."},
                "player_id": {"type": "string", "description": "Player gsis_id (e.g. '00-0036322'). Overrides player."},
                "position": {"type": "string", "description": "Position filter (e.g. WR, RB, TE)."},
                "team": {"type": "string", "description": "Team abbreviation (e.g. KC)."},
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
                "player": {"type": "string", "description": "Player full name or nflverse short name."},
                "player_id": {"type": "string", "description": "Player gsis_id (e.g. '00-0036322'). Overrides player."},
                "position": {"type": "string", "description": "Position filter (e.g. WR, RB)."},
                "team": {"type": "string", "description": "Team abbreviation."},
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
            "separation (>2.5 yds) who under-produced (fp_diff_per_game < -1.5, td_luck < -1.0); it "
            "is null on fallback rows, whose sample is too small to carry the claim. "
            "Each row is one player, season and season_type - the regular season and the postseason "
            "are separate rows and are never averaged together. season_type defaults to REG; pass "
            "'ALL' for both scopes (a null is rejected by the schema, which types this as a string). "
            "The Next Gen Stats figures are that "
            "publisher's own season summary row (week = 0) where it published one; otherwise they "
            "are a mean of the weekly rows, separation weighted by targets and YAC-above-expected by "
            "receptions. separation_source names which, and fallback_weeks and fallback_targets give "
            "the sample size behind a weekly-rows row (most rest on 1-2 weeks). "
            "Default sort is most under-producing first (fp_diff_per_game ASC)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "player": {"type": "string", "description": "Player full name or nflverse short name."},
                "player_id": {"type": "string", "description": "Player gsis_id (e.g. '00-0036322'). Overrides player."},
                "position": {"type": "string", "description": "Position filter (e.g. WR, TE)."},
                "team": {"type": "string", "description": "Team abbreviation."},
                "season": {"type": "integer", "description": "Exact season year (2016+)."},
                "season_type": {
                    "type": "string",
                    "enum": ["REG", "POST", "ALL"],
                    "description": "REG (regular season, the default), POST (playoffs), or ALL for both scopes - they stay separate rows. ALL is a literal value: this parameter is typed as a string, so a null is rejected before the tool runs.",
                },
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
            "or 'what is player X's drop rate'. Player names work in full form or the "
            "nflverse short form ('Justin Jefferson' or 'J.Jefferson'). "
            "Default sort is highest drop rate first."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "player": {"type": "string", "description": "Player full name or nflverse short name."},
                "player_id": {"type": "string", "description": "Player gsis_id (e.g. '00-0036322'). Overrides player."},
                "team": {"type": "string", "description": "Team abbreviation."},
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
                "player": {"type": "string", "description": "Player full name or nflverse short name."},
                "player_id": {"type": "string", "description": "Player gsis_id (e.g. '00-0036322'). Overrides player."},
                "position": {"type": "string", "description": "Position filter (e.g. RB, WR)."},
                "team": {"type": "string", "description": "Team abbreviation."},
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


@_mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


_DISPATCH = {
    "nfl_schema":       nfl_schema,
    "nfl_status":       nfl_status,
    "nfl_query":        nfl_query,
    "nfl_search_plays": nfl_search_plays,
    "nfl_team_stats":   nfl_team_stats,
    "nfl_player_stats": nfl_player_stats,
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


@_mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    try:
        fn: Any = _DISPATCH.get(name)
        if fn is None:
            raise ValueError(f"Unknown tool: {name}")
        # Each tool opens a blocking DuckDB connection, so run it off the event loop.
        result = await anyio.to_thread.run_sync(
            lambda: fn(**(arguments or {})),
            limiter=_TOOL_LIMITER,
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    except Exception as e:
        logger.error(f"Error in {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=json.dumps(_tool_error_payload(name, e), indent=2))]


def create_app() -> Starlette:
    """Return a Starlette ASGI app that serves the MCP server over Streamable HTTP."""
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

    # Browser-based clients (e.g. MCP Inspector) send a CORS preflight OPTIONS
    # request before every call. Without this middleware those preflights get
    # 405 Method Not Allowed and the connection never establishes. Expose
    # Mcp-Session-Id so the browser can read it back from responses.
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["Mcp-Session-Id"],
        )
    ]

    return Starlette(
        lifespan=lifespan,
        middleware=middleware,
        routes=[Route("/mcp", endpoint=StreamableHTTPASGIApp(session_manager))],
    )
