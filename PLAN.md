# NFL MCP bug fixes: general rules and task groups

Give each agent the General Rules section and its own group section.

Order of work:
- Group D can start now. It touches different files.
- Group A can start now. Groups B, C, and E depend on it.
- Groups B, C, and E start after group A is merged. They edit some of the same files, so each agent rebases on main before its PR.

====================================================================
GENERAL RULES (for all groups)
====================================================================

Repository: knguyen298/nfl-mcp. Line numbers below are for main at commit ab52c3d. If the code has changed, find the same code by its function or table name.

## Scope
- Do not change authentication or CORS. Keep the CORS middleware in nfl_mcp/server.py as it is.
- Do not change update_db.sh, entrypoint.sh, the refresh schedule, the slot order, or the swap method. You may change how ingest.py builds the derived tables.
- Do only the tasks in your group. If you find another problem, write it in the PR description, and do not fix it.

## Data and definitions
- The target is the values that ESPN and Pro Football Reference show. These follow the official NFL definitions.
- Follow the data as nflverse classifies it. If a play has sack = 1, it is a sack. Make no custom categories, and no custom formulas where nflverse has a column for the stat.
- Where an nflverse stat table has the stat, return that value: player_stats and team_stats_raw (both loaded at week level, nfl_mcp/registry.py lines 132-146) and schedules. Calculate from plays only when no table has the stat. Then use the nflfastR flag columns as defined.
- For a rate across several weeks or seasons, divide the summed totals. Never average the weekly rates.
- Inspect the real table columns before you use them. Column names changed across nflverse versions.
- The local test database may hold only some seasons. In the PR, say which cases you checked against real data and which you did not.

## Code
- Keep to your group's production line budget. If you need more, stop and tell me why before you go over it.
- Add no helper, class, module, option, or error code unless a task needs it. Add a shared helper only when 3 or more call sites use it.
- When you add or change a tool parameter, update the tool schema in the TOOLS list in nfl_mcp/server.py (lines 56-741).

## Tests and PR
- Write tests for each fix.
- Branch from main, as your group says. Rebase on main before you open the PR.
- Open one PR for your group. The description says what changed, why, and how to test it.

====================================================================
GROUP A: one player ID resolver for every tool
====================================================================

Start: now. Groups B, C, and E depend on this group.
Production line budget: about 250 lines.

## The problem
Each tool matches player names in its own way. The same name gives different results in different tools: separate rows, no rows, or two players merged into one row.

Test case: there are two players named Justin Jefferson, one from 2020 and one from 2026. Both are active in 2026. They have the same full name and the same short name ("J.Jefferson"). Only gsis_id tells them apart, with position, team, and first season as labels.

On main, nfl_player_stats("Justin Jefferson", season=2026) changes the name to "J.Jefferson", matches both players, and groups by season only. It returns one row with both players' stats added together.

## Where the problems are (nfl_mcp/tools.py unless noted)
1. _normalize_player_name, lines 52-66. It changes "First Last" to "F.Last". It breaks names with suffixes ("Kenneth Walker III" becomes "K.Walker III", which does not match "K.Walker") and names that contain a period ("D.J. Moore" passes through unchanged, which does not match "D.Moore").
2. Tools that change full names to short form, then match a substring with ILIKE:
   - nfl_search_plays, lines 319-327 (passer, rusher, and receiver name columns in plays).
   - nfl_player_stats, lines 414-490. It matches with ILIKE and groups by season and season_type, so it merges players.
   - nfl_compare, player mode, lines 542-582.
   - nfl_snap_counts, line 870. snap_counts.player stores FULL names, so the short form finds nothing ("%T.Kelce%" found 0 rows, "%Travis Kelce%" found 42).
3. Tools that search short-name columns but do not change the input, so a full name finds nothing:
   - nfl_ftn_charting, lines 1048-1053.
   - nfl_drop_rate, line 1314 (player_drop_rate.player is the short name).
4. Tools that search full-name columns, so a short name finds nothing:
   - nfl_fantasy_opportunity (line 917), nfl_td_luck (1140), nfl_role_trend (1193), nfl_separation_opportunity (1256), nfl_contract_value (1366), and nfl_injuries (712).
   - nfl_fantasy_rankings, line 994. ff_rankings use FantasyPros names and IDs. ff_playerids maps those IDs to gsis_id.
5. Derived tables in nfl_mcp/ingest.py that join by name:
   - player_role_trend, lines 684-688: joins snap_counts ON s.player = o.full_name plus team, season, and week. A PFR name that differs from the ff name gives a null snap_pct.
   - injury_return_curve, lines 880-883 and 894-898: joins snap_counts by name and season only, with no team. Two players with the same name share one baseline.

## What to build
1. One resolver function that every tool above uses.
   - Input: a player_id (gsis_id), or a name with optional team, season, and position.
   - If a player_id is given, use it, and ignore the name.
   - If the name and filters match exactly one player, return that gsis_id.
   - If they match more than one player, return ambiguous: true and the candidates. Each candidate has gsis_id, full name, position, team or teams, and first and last season. The tool returns this list and no stats.
   - If they match no player, return not_found and the nearest names.
   - Never raise a general error for ambiguity. Never merge two IDs.
   - Accept a full name or a short name. Handle suffixes (Jr., Sr., II, III), periods, apostrophes, hyphens, and accents (DuckDB strip_accents()) on both sides of the compare.
   - Search the players table, and use rosters or rosters_weekly for the team and season context. Inspect the real columns of players first (look for gsis_id, display_name, short_name, position, latest_team, rookie_season, pfr_id, and similar).
2. Add an optional player_id parameter to each tool that accepts a player name.
3. Every player query filters and groups by the ID column, never by a name:
   - plays: passer_player_id, rusher_player_id, receiver_player_id.
   - ff_opportunity and the fantasy derived tables: player_id (a gsis_id).
   - injuries and contracts: gsis_id.
   - snap_counts: pfr_player_id, mapped through players.pfr_id.
   - FTN charting: through the plays IDs.
   - ff_rankings: through ff_playerids.
   Check each mapping on the real data. If an ID has no mapping, return ambiguous or not_found. Never fall back to a substring match that can merge players.
4. Every player row in the output has the ID and a label that tells same-name players apart, for example "Justin Jefferson (WR, MIN)".
5. Fix the two name joins in ingest.py (item 5) to join by ID.
6. Delete _normalize_player_name and any per-tool name code that the resolver replaces.

## Tests
- One parametrized test that runs the same inputs through every tool that accepts a player: an ambiguous name (Justin Jefferson in 2026), the same name with a team filter that makes it unique, a unique name, a short name ("J.Jefferson"), a name with a suffix, and an unknown name. Each tool must give the same resolver result.
- Find more real collisions with a query: group the player name columns and count distinct IDs. Put the query in the PR description.

====================================================================
GROUP B: return official stat values, as ESPN and PFR show them
====================================================================

Start: after group A is merged.
Production line budget: about 400 lines.
Each metric in a response, or in the schema docs (nfl_mcp/schema_pbp.py), names its source: the table and column, or the flags it uses.

## Facts about sacks (the official standard)
- A sack is not a pass attempt and not a rushing attempt. The QB is charged with the sack and the sack yards, as separate stats.
- Player passing yards are gross. They do not subtract sack yards. Team passing yards are net. They subtract sack yards.
- Completion percentage = completions / attempts. Yards per attempt = passing yards / attempts. Net yards per attempt = (passing yards - sack yards) / (attempts + sacks). Sack rate = sacks / (attempts + sacks).
- In nflfastR play-by-play, a sack row has play_type = 'pass', pass_attempt = 1, and sack = 1. That is why counting pass plays gives too many attempts.
- Kneels count as rushing attempts. Spikes count as incomplete passes. Scrambles count as rushes. Two-point attempts are not in the regular stats. player_stats already handles all of these.

## Where the problems are
1. nfl_player_stats, nfl_mcp/tools.py lines 414-490. It calculates from plays:
   - Passing counts COUNT(*) of play_type = 'pass', so sacks count as attempts. Verified: Mahomes 2024 regular season gave 617 attempts. The official count is 581, the same as player_stats, and the difference is his 36 sacks.
   - Rushing uses play_type = 'run' only, so kneels (play_type 'qb_kneel') are left out. Two-point plays may be included.
   Fix: read player_stats (week level). Filter by the resolved ID, season, and season_type, and sum per season. Return the official counting stats. Calculate rates from the summed totals. Keep the current tool parameters.
2. nfl_compare, player mode, lines 542-582: uses nfl_player_stats, so it follows the fix in item 1. Check it.
3. team_offense_stats, nfl_mcp/ingest.py lines 541-570:
   - The filter includes kickoffs, punts, field goals, extra points, spikes, kneels, and no_play rows (lines 567-568). total_plays, yards_per_play, avg_epa, and the turnover counts include special teams plays. A muffed punt counts as a turnover by the kicking team's offense.
   - touchdowns uses touchdown = 1 (line 554), which is true for any touchdown on the play. A pick-six against the offense counts as an offensive touchdown. Use pass_touchdown and rush_touchdown, or td_team = posteam.
   - third_down_pct uses yards_gained >= ydstogo (lines 556-557). Use third_down_converted and third_down_failed, which handle penalties correctly.
   - red_zone_td_pct divides by red zone plays (lines 558-559). The standard divides by red zone trips (drives that reach the opponent's 20). Use the drive columns. Check which drive columns exist.
   - It groups by posteam and season only (line 569), so playoff games are mixed in.
4. team_defense_stats, ingest.py lines 573-591: the same special teams and no_play problem. On a kickoff, defteam is the kicking team, so kick coverage counts as defense. The same mixing of season types.
5. situational_stats, ingest.py lines 594-611, with _SITUATION_EXPR at about line 509:
   - The 4th Down conversion_pct includes punts and field goals, so each punt counts as a failed conversion. Use fourth_down_converted and fourth_down_failed. Use third_down_converted and third_down_failed for 3rd down.
   - The same mixing of season types.
6. nfl_team_stats (tools.py lines 381-411) and nfl_compare in team mode (lines 506-540): add season_type. nfl_compare accepts season_type and ignores it for teams.
7. player_separation_opportunity, ingest.py lines 708-748: the ng subquery averages the weekly NGS values (week > 0) with no weights. NGS publishes a season summary row (week = 0). Use that row. Check how season_type and week 0 work in the loaded nextgen_stats_receiving table.

## How to fix the team tables
- Keep the table names, so the tools keep working, and add a season_type column to each table.
- Build the counting stats from team_stats_raw (and schedules for points) where those tables have them.
- Build what those tables do not have from plays, with the official flags. For per-play rates such as yards per play and EPA per play, use scrimmage plays only: (pass = 1 OR rush = 1) AND epa IS NOT NULL. This is the standard nflfastR filter, and it includes sacks and scrambles.

## Tests
- For two player-seasons and two team-seasons, compare the tool output to player_stats and team_stats_raw. The values must match, apart from rounding. Include one QB with many sacks.
- One team-season with a defensive or return touchdown: offensive touchdowns must not include it.
- 4th down conversion rate: punts do not count.
- season_type: regular-season values do not include playoff games.

====================================================================
GROUP C: correct filters, consistent team codes, and truncation flags
====================================================================

Start: after group A is merged. Group B edits some of the same files, so rebase carefully.
Production line budget: about 150 lines.

## Problems and where they are (nfl_mcp/tools.py unless noted)
1. The playoff filter in nfl_schedule (lines 736-778, game_type filter at lines 757-759). The tool compares season_type directly to schedules.game_type. game_type holds REG, WC, DIV, CON, and SB, so season_type='POST' returns no games (verified). Fix: POST maps to game_type IN ('WC','DIV','CON','SB'). Add WC, DIV, and CON to the enum in nfl_mcp/server.py (around line 401). Check every other place that filters on season_type or game_type for the same mismatch. plays and player_stats use REG and POST.
2. Team code case. nfl_search_plays (lines 313-318) compares team and opponent without changing them to uppercase. The other tools change them to uppercase (verified by review).
3. Team codes from before a move (OAK, SD, STL, and others). nfl_mcp/franchises.py has canonical_team() and schedule_alias_case() (the latter is used in ingest.py around line 966). Only nfl_team_records uses them. Tools that filter on the current code miss the rows that use the old code (verified by review for OAK, SD, and STL). Fix: in every tool that takes a team, change the input with canonical_team(), and match all codes that the franchise used in the data (for example, LV matches LV and OAK). Check with the real data which code each table stores for old seasons: plays (posteam and defteam), schedules, rosters, team_stats_raw, and player_stats. Return a clear error for an unknown code.
4. Silent truncation. These tools have a fixed LIMIT 500 and do not tell the client that rows were cut: nfl_roster (line 680), nfl_injuries (726), nfl_schedule (771), nfl_team_records (832), nfl_snap_counts (894), and nfl_fantasy_opportunity (943). nfl_search_plays and the tools with a limit parameter also return no flag. Example: snap counts for one team and one season are more than 500 rows, and the sort by week ascending cuts the last weeks (verified by review). Fix: one helper that runs the query with LIMIT n + 1 and returns the rows and truncated: true or false. Use it in every tool with a limit. When truncated is true, add a short hint: add filters, or use nfl_query to aggregate.
5. A wrong fact in a tool description. The nfl_search_plays schema in nfl_mcp/server.py (around line 145) says "2013 through the current season". plays starts in 1999. Correct the text.

## Tests
- nfl_schedule(team="KC", season=<a season with KC playoff games in the local data>, season_type="POST") returns the playoff games.
- A lowercase team code gives the same result as uppercase in every tool.
- An old code and the current code give the same rows, if the local data has seasons before the move. If it does not, say so in the PR.
- A query over the limit returns truncated: true, and a query under the limit returns truncated: false.

====================================================================
GROUP D: read-only database access, full read-only SQL, and a non-blocking server
====================================================================

Start: now. This group touches different files from the others.
Production line budget: about 150 lines.

## Problems and where they are
1. File access through the read-only connection. nfl_mcp/database.py line 43 opens DuckDB with read_only=True. That blocks writes, but DuckDB still allows file and network reads on a read-only connection. Verified: SELECT * FROM read_text('/etc/passwd') returns the file. The _FORBIDDEN regex (nfl_mcp/tools.py lines 35-44) does not block read_text, read_csv, read_parquet, read_json, glob, or ATTACH.
   Fix: open the serve connection with a config dict:
     enable_external_access = False
     autoinstall_known_extensions = False
     autoload_known_extensions = False
     lock_configuration = True
   plus memory_limit, threads, and temp_directory from the existing env vars, when they are set. _apply_runtime_pragmas (database.py lines 12-31) runs SET after connect, and SET fails after lock_configuration. Move those values into the config dict and delete the function. Verified: with this config, read_text raises PermissionException, and SET enable_external_access = true fails.
   Check that the ingest and CLI code (nfl_mcp/ingest.py, nfl_mcp/cli.py) open their own write connections and do not use get_db_connection. The ingest must keep working. Check that temp_directory spill still works with external access disabled.
2. nfl_query blocks valid read queries (tools.py lines 247-291).
   - Line 263: the check ^\s*SELECT\b rejects WITH (CTEs), DESCRIBE, SUMMARIZE, PIVOT, and FROM-first queries.
   - Lines 35-44 and 266: the _FORBIDDEN regex matches words anywhere, including in string literals and aliases. Verified blocked: SELECT replace(...), WHERE "desc" ILIKE '%set%', WHERE last_name = 'Do', and an alias named load. Most of its keywords are Postgres terms.
   - Lines 270-272: a semicolon inside a string literal gives "multiple statements".
   - Line 275: the query is wrapped as SELECT * FROM (<sql>) LIMIT n. SQL does not guarantee that the inner ORDER BY stays in effect after the wrap.
   Fix: delete _FORBIDDEN and the first-keyword check. Use duckdb.extract_statements(sql) to require exactly one statement, and allow only read statement types. A reviewer found that in DuckDB 2.0.0, PIVOT parses as StatementType.MULTI. Check how PIVOT, DESCRIBE, SUMMARIZE, and EXPLAIN parse in the version that pyproject.toml pins, and allow the read forms. The locked config from item 1 is the real protection. The parser check only gives a clear error. Run the statement as written, and fetch at most max_rows + 1 rows with fetchmany(), so the order stays correct. Keep the 10-second timeout and the 500-row maximum. Return the SQL with the result.
   Update the nfl_query description in nfl_mcp/server.py (lines 93-119): remove "LAST RESORT", and say that it accepts full read-only DuckDB SQL, including CTEs and window functions.
3. The event loop blocks. call_tool in nfl_mcp/server.py (lines 749-787) is async, but it calls each tool synchronously (line 782). _execute (tools.py lines 591-628) waits on thread.join() for up to 10 seconds. Verified: a 10 ms ticker showed a 10.02 s gap during one nfl_query call. During that time, the server answers no other request.
   Fix: run each tool in a worker thread (anyio.to_thread.run_sync or asyncio.to_thread), with one simple concurrency limit (default 2), because the DuckDB memory limit applies per connection. Build _DISPATCH (line 753) one time at module level.

## Tests
- These fail through nfl_query and through a direct get_db_connection() call: read_text, read_csv, read_parquet, glob, ATTACH, COPY ... TO, INSTALL, LOAD, CREATE TABLE, INSERT, UPDATE, DELETE, DROP, and SET enable_external_access = true.
- These now run: a CTE, a window function, replace(), ILIKE '%set%', a string literal that contains a semicolon, DESCRIBE plays, and PIVOT (if the pinned version allows it).
- An ordered query with a limit returns the correct first rows.
- A slow nfl_query call does not block a parallel nfl_status call.

====================================================================
GROUP E: injury return tool without custom injury groups
====================================================================

Start: after group A is merged. Group A changes the snap_counts joins in this table to ID joins. Keep those joins.
Production line budget: about 50 lines.

## The problem
nfl_injury_return (nfl_mcp/tools.py lines 1401-1444) reads injury_return_curve, built in nfl_mcp/ingest.py lines 816-910.
- Lines 832-854 put each injury into a body-part group with a chain of LIKE checks in a fixed order. The first word that matches decides the group. "Foot/Ankle" becomes ankle, because ankle is checked before foot. "Hip, Back" becomes hip, and the back part is lost. Text with no match becomes other. This is a custom classification.
- Lines 857-860 find "Out" spells from consecutive week numbers, so a bye week in the middle of an injury splits one spell into two.

Note: nfl_injuries (tools.py lines 690-733) already returns the injury text exactly as the data gives it. Do not change it.

## What to do
1. Remove the CASE that makes the groups. Use the injury text from report_primary_injury, or practice_primary_injury when the report value is null, as the data gives it. Do not change the text, except to trim spaces. Keep the original case for output, and compare without case for matching.
2. The injury_type parameter matches that text with ILIKE. Update the parameter description in nfl_mcp/server.py (around lines 688-697): remove the list of groups and the mapping examples ("ACL" -> knee, "hammy" -> hamstring), and say that the value matches the injury text in the report.
3. Build the spells with the team's game weeks from schedules, so that a bye week does not split a spell. Two Out weeks belong to one spell when the team played no game between them in which the player was not Out.
4. Group the output by injury text and position, as now.

## Tests
- "Foot/Ankle" stays "Foot/Ankle" in the output, and does not appear under "Ankle".
- A spell that goes across a bye week is one spell.
- The output has no group names from the old CASE.
