# Changelog

The version stays 0.7.3. The notes for v0.7.3 and earlier releases are in the git history,
under the tags `v0.6.3` to `v0.7.3`.

## Unreleased

The `state-machine-fixes` work, merged on top of v0.7.3.

### Tool results

- Every tool failure carries a `code` from `ErrorCode`. The server sends one error shape and
  sets `isError` on the MCP result.
- If more rows matched than a capped tool returned, the tool says `truncated: true`.
- An error raised inside a tool is reported as `TOOL_EXECUTION_ERROR`. `INVALID_ARGUMENTS` is
  only for a call that fails the input schema or does not fit the tool's signature.
- `nfl_team_records` reports `complete: true` for every season that ended. The two 2022 teams
  with a canceled game no longer read as still playing.
- The `season_from` descriptions say what happens: a start before the first season is accepted,
  and the answer begins at the first season that has data.
- `nfl_query` and the other tools stop a slow query the same way. The interrupt is sent up to
  four times, and a query that keeps running is logged and counted.
- `data_as_of` is the time of the newest load that stored rows. A failed or an empty attempt no
  longer moves it.
  `nfl_player_lookup` now always reports `truncated`.
- A season before a tool's first season is refused with `OUT_OF_COVERAGE` before any query
  runs. A season after the loaded data is still answered with no rows and a `reason`.
- The server checks tool arguments against the input schema itself, so a bad argument gets the
  `INVALID_ARGUMENTS` error shape.
- `nfl_status` and `nfl_catalog` list `issues`: loads that failed, stored no rows, or stored a
  short season. `nfl_catalog` also lists `missing_seasons`.
- `nfl_team_records` marks each row `complete`, lists `gaps` for a team over a two-ended range,
  and gives a `status` and a `reason` for a query with no stored rows. A season the ESPN pull covered
  says the source has no record, instead of advice to run the pull.
- If `nfl_schema` cannot read the table list, it keeps the summary and adds a `warning` and a
  `code`. Its two errors list their choices under `available` as well, and a `table` call that
  ignored `category` says so in `note`.
- An integer `team` value is refused as an unknown team instead of raising an exception.

### Query safety

- DuckDB's parser checks `nfl_query` first. A statement keyword is no longer refused by a word
  scan, and a column that has the name of a blocked function is allowed. A call to a blocked
  function is still refused.
- Each tool connection opens read-only with external access, extension loading and Python
  scans off, and the configuration locked.

### Ingest

- Each season loads in one transaction with its `_ingest_metadata` row. The log records how
  each attempt ended (`loaded`, `empty`, `failed`, `partial`), and a failed or empty season is
  retried on the next run.
- A season whose fetch failed or stored nothing is no longer read as a season the database has.
  The depth_charts restamp plan and the season list `nfl-mcp upgrade` reloads count only
  `loaded` and `partial` rows, so one broken season cannot hold `nfl-mcp upgrade` at exit 1 or
  stop the container's refresh on every run.
- `nfl-mcp ingest` exits 0 when every fetch came back and every source had no rows. Exit 3 for
  "nothing loaded" now needs at least one failed fetch, as it did in v0.7.3.
- A failed ESPN season (`--pre1999`) is still written to the log, counted, and listed in the run
  summary. It no longer sets the exit code and no longer stops the schema stamp, so one ESPN
  rate limit during an image build still ships a stamped seed. `--strict` counts it.
- A run reports a season only after its write commits. A season that rolled back is no longer
  counted as loaded in the run summary.
- The ingest spills DuckDB's temp files to `NFL_MCP_DUCKDB_TEMP_DIR` when that variable is set,
  which is the same variable a query connection reads. Without it the ingest spills next to the
  database file, as before.
- A season loaded before it ended is fetched again once.
- If any fetch failed, `nfl-mcp ingest --strict` exits 3, even with `--allow-failures`.
- The ESPN pull stops after a run of failed seasons, and the run reports the stop.

### Container

- `entrypoint.sh` keeps a link that resolves, so a restart serves the newest slot. A first start
  fills slot A under a scratch name and renames it into place.
- `update_db.sh` and the image build use one shared check, `validate_db.sh`, before a database
  is served. The update also refuses a slot with a leftover `.wal` file, and checks free space
  first.
- `update_db.sh` checks for a leftover `.wal` right after the ingest, before `nfl-mcp upgrade`
  opens that slot for writing and removes the file. The check before validation stays.
- `nfl-mcp serve --help` shows the default host.

### Packaging

- `jsonschema` is a declared dependency. `nfl_mcp/server.py` imports it directly, and until now
  only the `mcp` package pulled it in.
