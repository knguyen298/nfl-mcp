# Changelog

The version stays 0.7.3. The notes for v0.7.3 and earlier releases are in the git history,
under the tags `v0.6.3` to `v0.7.3`.

## Unreleased

The `state-machine-fixes` work, merged on top of v0.7.3.

### Tool results

- Every tool failure carries a `code` from `ErrorCode`. The server sends one error shape and
  sets `isError` on the MCP result.
- If more rows matched than a capped tool returned, the tool says `truncated: true`.
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
- A season loaded before it ended is fetched again once.
- If any fetch failed, `nfl-mcp ingest --strict` exits 3, even with `--allow-failures`.
- The ESPN pull stops after a run of failed seasons, and the run reports the stop.

### Container

- `entrypoint.sh` keeps a link that resolves, so a restart serves the newest slot. A first start
  fills slot A under a scratch name and renames it into place.
- `update_db.sh` and the image build use one shared check, `validate_db.sh`, before a database
  is served. The update also refuses a slot with a leftover `.wal` file, and checks free space
  first.
- `nfl-mcp serve --help` shows the default host.
