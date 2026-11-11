# AGENTS.md

Guidance for AI agents working on this codebase.

## What This Project Is

`nfl-mcp` is a local MCP server that ingests NFL data from nflreadpy into a DuckDB database and exposes it to AI clients (Claude Desktop, VS Code) via the Model Context Protocol. All data lives in a single `~/.nfl-mcp/nflread.duckdb` file. No network calls happen at query time — only at ingest time.

## Repository Layout

```
nfl_mcp/
  cli.py          # Click CLI: serve, ingest, rebuild, upgrade, validate, init, setup-client, doctor
  server.py       # MCP server: TOOLS list + _DISPATCH + call_tool handler
  tools.py        # Tool implementations (nfl_schema, nfl_status, nfl_query, ...)
  schema_pbp.py   # PBP schema strings (_SCHEMA_SUMMARY, _SCHEMA_CATEGORIES) — pure data, no logic
  ingest.py       # Data ingestion: run_ingest_datasets, idempotency, schema reconciliation
  migrate.py      # Schema version (_schema_version), the check, and `nfl-mcp upgrade`
  registry.py     # Declarative dataset definitions (REGISTRY, DatasetDef)
  espn.py         # External fetcher: 1970-1998 final season records from ESPN's scoreboard API
  franchises.py   # Franchise identity: canonical team codes, era names, availability, season shapes
  seasons.py      # Season bounds, the in-progress season, and the date a season stops changing
  status.py       # State names (Availability, ErrorCode, IngestStatus). Pure enums, no logic
  database.py     # get_db_connection() context manager (read-only DuckDB)
  config.py       # ~/.nfl-mcp/config.json read/write, get_duckdb_path()

entrypoint.sh   # Container start: keep the active link or seed slot A, then supercronic -> serve
update_db.sh    # Scheduled refresh: lock, copy active slot, ingest, validate, swap the symlink
validate_db.sh  # One database check, used by the image build and by update_db.sh
Dockerfile      # Published image (baked seed + runtime A/B updater)
Dockerfile.dev  # Same updater, no baked seed (cold ingest on first start)

tests/
  conftest.py        # Tags every test without the integration marker as a unit test
  test_registry.py   # Registry structure, coverage windows, loader functions
  test_ingest.py     # Ingest helpers using in-memory DuckDB (no network)
  test_cli.py        # CLI commands using click.testing.CliRunner
  test_config.py     # Configuration file loading, with a temporary config directory
  test_database.py   # The read-only connection layer: runtime settings and the external-access lock
  test_tools.py      # Tool functions, unit and integration (the require_db fixture)
  test_tools_unit.py # Query building, the shared input checks, and error codes (no DB)
  test_server.py     # Tool listing, dispatch, and the one error shape on the wire
  test_espn.py       # ESPN parsing and fetch logic against a fake transport (no network)
  test_franchises.py # Canonical codes, era names, availability, expected season shapes
  test_status.py     # Enum values, plain-string equality, str()/JSON serialization of states
  test_safety.py     # SQL injection / forbidden keyword guardrails
  test_dockerfile_dev.py # Image wiring: both Dockerfiles, the schedule, and the two updater scripts
  test_coverage_gaps.py  # Unit tests for lines no other test reaches
  fixes/             # Release regression tests, one file per review group or version (v0.7.x)
```

## Key Architectural Rules

**Adding a new MCP tool** changes three files:
1. `tools.py`: implement the function, add it to `__all__`, spell any fixed-value argument as a
   module-level tuple, and list the datasets its rows come from in `_COVERAGE`
2. `server.py`: add a `Tool(...)` entry to `TOOLS` and an entry to `_DISPATCH`. Build each
   fixed-value argument's `enum` with `_enum_property(<that tuple>)`, so the schema and the tool
   accept the same set
3. `tests/test_tools_unit.py`: add one call to `TOOL_ERROR_CASES`, and put the tool in either
   `ROW_CAP_CASES` or `NO_ROW_CAP`

Guard tests fail when a new tool or argument is left out of that wiring:
- `test_every_public_tool_is_walked_by_the_error_test`: every name in `__all__` needs a
  `TOOL_ERROR_CASES` entry, and each of those is run for the error shape below
- `test_every_public_tool_declares_its_row_cap_or_its_absence`: every tool is either capped
  (`ROW_CAP_CASES`) or explains why it has no row list (`NO_ROW_CAP`)
- `test_every_schema_enum_is_a_checked_choice_with_the_same_values`: every `enum` in an input
  schema must equal the tuple its tool checks, and every checked tuple must be in `CHOICE_CASES`
- `test_every_tool_with_a_season_argument_has_a_coverage_entry`: a tool that takes `season`,
  `season_from` or `from_season` needs an entry in `_COVERAGE`
- `test_every_team_argument_is_walked_by_the_team_tests`: every `team` or `opponent` argument
  needs an entry in `TEAM_ARGUMENT_CASES`

`tests/test_server.py::test_dispatch_and_tools_list_the_same_names` fails when `TOOLS` and
`_DISPATCH` hold different names.

**Adding a new dataset** requires changes in exactly one file:
1. `registry.py` — add a `DatasetDef` entry to `REGISTRY`

The ingest loop in `ingest.py` is fully generic — it reads from the registry and handles everything else (idempotency, schema reconciliation, bulk vs ranged mode).

**Exception: a dataset from outside nflreadpy.** `DatasetDef.source` defaults to
`SOURCE_NFLREADPY`. A dataset that reaches another publisher sets `source=SOURCE_ESPN` or a
new constant in `KNOWN_SOURCES`, sets `default=False`, and gives a `loader_fn` that names the
fetcher rather than an nflreadpy function. `run_ingest_datasets` routes on `source` to the
matching ingest helper. `espn_records` is the only one so far. Keep it that way unless a new
external source is worth taking off the container's scheduled refresh path.

**`min_season` is the depth knob.** Ingest clips every dataset to its own
`min_season`/`max_season`, so one dataset's early start never delays the rest and no
dataset requests a season its source never published. Two bounds in `seasons.py` set the
outer limits of what gets requested:

- `FIRST_SEASON = 1966` — the **default** backfill start (the AFL–NFL merger agreement
  year, so the final four AFL seasons of rosters are included).
- `EARLIEST_SEASON = 1920` — how deep `--start` may reach; a test asserts this equals
  `registry.earliest_season()`, the oldest declared dataset floor. Keep them distinct:
  changing the default should not silently redefine what is reachable.

- Always set `min_season` for a seasonal dataset — a test asserts none are left `None`.
- If you register something older than the default, a test will flag that the default
  clips it; update that test's expected list deliberately rather than moving `FIRST_SEASON`
  reflexively.
- Set `min_season` to the first season whose file has **rows**, not the first season the file
  exists for: nflverse publishes an empty `snap_counts_2012.parquet`, and nflreadpy happily
  returns 0 rows for it.
- `min_season` must match nflreadpy's own season validator (each `load_*.py` raises
  `Season must be between <floor> and ...`). nflreadpy is the hard ceiling — going deeper
  than its floor needs a PR upstream, not a registry change.
- Don't set `max_season` unless upstream genuinely stopped publishing; a stale cap silently
  freezes a dataset (`participation` carried `max_season=2024` while 2025 was already live).

**Never hardcode table schemas** for non-pbp tables. Use `nfl_schema(table=<name>)` which queries `information_schema.columns` at runtime. PBP schema strings live in `schema_pbp.py` because they carry human-readable annotations — not just types.

**Never spell a state name as a bare string literal.** Availability reasons, tool error codes and
ingest statuses are members of `Availability`, `ErrorCode` and `IngestStatus` in `status.py`. Add
the member there first, then import it. Each enum subclasses `str`, so a member compares equal to
its own value and `json.dumps` writes the plain string. The names move to one file. What a caller
sees does not change.

## Development Workflow

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run all tests (unit + integration if DB exists)
pytest

# Run only unit tests (no DB required)
pytest -m unit

# Run with coverage (configured in pyproject.toml, runs automatically)
pytest

# Check coverage for a specific module
pytest --cov=nfl_mcp.tools --cov-report=term-missing

# Ingest data (CI uses 2024 only; local development can use all seasons)
nfl-mcp ingest --dataset all --start 2024 --end 2024   # fast, CI-friendly
nfl-mcp ingest --dataset all                             # all seasons, slow

# After changing tool code, reinstall this checkout into the MCP client's virtual
# environment. Run it from the repository root, with the client's own pip.
<client-venv>/bin/pip install --force-reinstall .
# Then restart the client
```

## Testing Strategy

- **Unit tests** (no DB, no network, always run in CI): `test_registry`, `test_ingest`, `test_cli`,
  `test_config`, `test_database`, `test_espn`, `test_franchises`, `test_status`,
  `test_tools_unit`, `test_dockerfile_dev`, and `test_safety` except its `TestNflQueryLive` class
- **Integration tests** (`test_tools.py` classes and the `tests/fixes/` tests marked
  `@pytest.mark.integration`): require a loaded DuckDB. The `require_db` fixture skips them when
  no database is present. Some `tests/fixes/` oracles count rows of the 2024 season only, so they
  expect the CI database (`--start 2024 --end 2024`)
- `tests/conftest.py` gives every test that does not carry the `integration` marker the `unit`
  marker, so a test that reads the real database must be marked. An unmarked test runs in
  `-m unit`, passes on a developer machine that happens to hold a database, and skips its own
  assertions in CI.
- **Safety tests** (`test_safety.py`): SQL injection guardrails, no DB needed
- CI runs `nfl-mcp ingest --dataset all --start 2024 --end 2024` before tests to populate the DB
- **CI enforces 100% coverage** via `pytest -q --cov-fail-under=100` on the full suite. The floor
  lives on the CI command (not `pyproject.toml` addopts) so local unit-only runs are not held to
  the full-suite floor. New code must keep total coverage at 100% or CI fails.

When writing new tool tests, follow the existing pattern:
```python
@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestMyNewTool:
    def test_basic_filter(self):
        result = my_tool(team="KC", season=2024)
        assert "error" not in result
        assert result["count"] > 0
```

## Database

- **Path**: `~/.nfl-mcp/nflread.duckdb` (override with `NFL_MCP_DB_PATH` env var)
- **Access**: always read-only at query time via `get_db_connection()` in `database.py`
- **Metadata**: `_ingest_metadata` records one row per dataset+season attempt, with an
  `IngestStatus` in `status` and a short reason in `detail`
- **Tables**: 43 after a full build with `--pre1999`. That is 30 from the registry, 4
  play-by-play aggregates, 6 fantasy derived tables, `team_records`, `_ingest_metadata`, and
  `_schema_version`. Use `nfl_schema()` to list them or `nfl_schema(table=<name>)` for columns
- **What a tool connection can reach**: every connection opens read-only with DuckDB's
  configuration set at connect time (`_runtime_config`) and then locked
  (`_lock_external_access`). Extension autoload and auto-install are off, Python replacement
  scans are off, and `enable_external_access` is off, so a query cannot read a file on disk or
  reach the network. `lock_configuration` means a query cannot `SET` any of that back on.
- **What `nfl_query` accepts**: one SELECT, or an EXPLAIN of one SELECT. DuckDB's own parser
  decides first (`_statement_gate`): text it cannot parse is `SQL_SYNTAX_ERROR`, two statements
  are `MULTIPLE_STATEMENTS`, and any other statement is `NOT_SELECT` with its `statement_type`.
  Then `_mask_sql` blanks string literals, quoted identifiers and comments, and checks its spans
  against DuckDB's tokenizer (`_engine_agrees`). A disagreement is refused with `SQL_ERROR`. The
  scan of the masked text refuses a call to a blocked function (`_FORBIDDEN_FUNCTIONS`, or such a
  name in double quotes) with `FORBIDDEN_KEYWORD`. A statement keyword is not scanned for, because
  the parser already refused it. The SQL runs as written, through `_fetch_capped`, which fetches
  `max_rows + 1` rows. `max_rows` is an integer: below 1 is `INVALID_ARGUMENT`, above 500 is
  clamped to 500. `tests/test_safety.py` covers each of these, including the engine refusing every
  file-reading form in `_FILE_VECTORS` with the text guard switched off.

## Configuration File

`config.py` owns `~/.nfl-mcp/config.json`. Keep these rules when you change it:

- **A read writes nothing.** `load_config()` and `get_duckdb_path()` run on every tool call.
  They never rename, rewrite, or create a file or a directory. A malformed file stays as it
  is. The read path warns once per file content through `warnings.warn` (stderr) and returns
  the defaults. Never print a warning to stdout, because stdout carries the MCP protocol in
  stdio mode.
- **A write is atomic.** `save_config()` calls `write_json()`. That function writes a temp
  file in the same directory, calls `os.fsync`, and then calls `os.replace`. A failed write
  removes the temp file and leaves the old file as it was.
- **The backup belongs to the write path.** `save_config()` copies a malformed file to
  `config.json.bak` before it overwrites the file.
- **A missing database creates nothing.** `get_db_connection()` does not create the parent
  directory of the database. A missing file raises the error that the tools report as
  `DB_NOT_FOUND`. A spill directory that cannot be created does not fail the query.
- **One address for the server and the client.** `serve`, `init` and `setup-client` resolve
  the host and the port in one helper in `cli.py`. The order is the command line option, then
  the `serve_host` / `serve_port` keys, then `DEFAULT_HOST` / `DEFAULT_PORT`. `init` and
  `setup-client` save the two keys when the address differs from the default.
- **`doctor` names the state.** A missing database file and a database with no data are two
  separate failed checks. Each one prints the command that fixes it.

## Dataset Registry

Each entry in `REGISTRY` is a `DatasetDef` with:
- `loader_fn`: the nflreadpy function name (must exist as `hasattr(nflreadpy, loader_fn)`) for a
  `SOURCE_NFLREADPY` dataset. For an external dataset it names the fetcher instead
- `source`: a name from `KNOWN_SOURCES`, which is `SOURCE_NFLREADPY` by default. A dataset on any
  other source must be `default=False`, so the scheduled container refresh never calls a
  third-party API
- `seasonal`: whether it accepts a `seasons=` parameter
- `storage`: a derived property, not an argument. It returns `"append_by_season"` for a seasonal
  dataset and `"replace"` for a static one, which are the two branches
  `_ingest_generic_dataset` already takes. A registry entry cannot declare it
- `min_season`/`max_season`: clip the ingest range to avoid known 404s
- `default`: whether included in `nfl-mcp ingest` with no flags

## Ingest Idempotency

The ingest loop checks `_ingest_metadata` before every dataset+season, and it records how every
attempt ended. Re-running the same ingest command is safe. Use `--fresh` to force re-ingest.

`_ensure_metadata_table` creates the table, and adds the `status` and `detail` columns to a
 database that predates them. `status` holds one `IngestStatus` value. It is `loaded` when rows
 landed, `empty` when the source published no rows, `failed` when the fetch or the write raised,
 or `partial` when rows landed but the season is short. `detail` holds a short reason, capped at
 `_DETAIL_MAX_CHARS`, and is `NULL` when there is nothing to say. Rows written before the column
 existed default to `loaded`, which is true of them: that code wrote a row only when rows landed.
`_is_loaded` counts a season as present only when its record says `loaded` or `partial`. A
`failed` or an `empty` record is the log of an attempt that put nothing in the table, so the next
run retries that season.

**Every read that asks which seasons the database holds counts the same two statuses.**
`ingest._stored_rows_predicate` builds that filter and binds the two `IngestStatus` members, and
its two callers are `_legacy_restamp_plan` (the seasons to re-fetch for a depth_charts row with
no season) and `migrate._metadata_seasons` (the seasons `nfl-mcp upgrade` reloads). Both work on
a metadata table that has no `status` column: that code wrote a row only when rows landed, so the
predicate comes back empty. Without the filter a season whose fetch keeps failing enters the plan
forever, `nfl-mcp upgrade` exits 1 on every run, and the container's refresh stops on every run.

A season loaded before it ended is fetched again. `_needs_refresh` compares the season's
`loaded_at` against `seasons.season_settled_on(season)`, which is March 1 of the year after the
season kicks off. A load recorded before that date can miss the last week of the regular season,
the playoffs and the Super Bowl, so it is re-fetched once and the new `loaded_at` settles it. The
in-progress season is re-fetched on every run.

## Ingest Internals

- **One season is one transaction.** `_transaction` wraps each season's delete, its insert and its
  `_ingest_metadata` row, so a crash or a failing insert cannot leave a season with no rows and a
  record that says it is loaded. DuckDB rolls back DDL too. A nested `_transaction` on the same
  connection does nothing, because DuckDB refuses `BEGIN` inside an open transaction, so the
  caller that opened it is the one that commits.
- **The run hears about a season after its COMMIT.** `_record_loaded` and `_record_attempt` note
  the attempt inside the transaction that writes it. `_note_attempt` puts the note in
  `_held_notes` while a transaction is open, and `_transaction` delivers the list only after a
  successful COMMIT. A rollback drops the notes, so a season the database lost never reaches the
  run summary or `nothing_loaded`. A note made with no transaction open goes straight to the run.
- **Table existence is asked, never guessed.** `_table_exists` reads `information_schema.tables`
  with the name bound as a parameter. Nothing in `ingest.py` catches `CatalogException` to decide
  whether a table is there, because that exception also fires for a missing column or a missing
  function.
- **Source and play-by-play names are constants.** `registry.py` spells `SOURCE_NFLREADPY`,
  `SOURCE_ESPN`, `KNOWN_SOURCES`, `PBP_DATASET_ID`, `PBP_TABLE_NAME` and `PBP_LOADER_FN`. The
  router in `_ingest_datasets` picks the helper from `defn.source` and raises `ValueError` for a
  source outside `KNOWN_SOURCES`, rather than sending it to nflreadpy.
- **An attempt carries the source it fetched from.** `_note_attempt` reads it from the registry by
  dataset id (`_attempt_source`) and stores it on the `IngestAttempt`. `_gates_the_run(source)` is
  the one rule that says whether a failed attempt decides the exit code, the `--failures-file`
  list, and the schema stamp: it does for `SOURCE_NFLREADPY`, and it does not for an outside
  publisher. `IngestRunResult.failures` holds the gating ones and
  `IngestRunResult.optional_source_failures` the rest. `--strict` reads `any_failed`, which counts
  every source.
- **The ESPN breaker counts requests, not loop turns.** `ESPN_FAILURE_LIMIT` stops the pull after
  that many seasons fail in a row. A season ESPN answered resets the count to zero, whether the
  records landed, came back short, or came back as no records. A season already loaded, so never
  requested, leaves the count where it was. The run reports the give-up as an `EspnStop`, because
  a pull that stopped and a pull that finished both return a row count.
- **The write connection spills where a query connection spills.** `_apply_duckdb_pragmas`
  sets `temp_directory` to `NFL_MCP_DUCKDB_TEMP_DIR` when that variable is set, and to
  `<database dir>/.duckdb_spill` when it is not. `database.py` reads the same variable for
  every read-only connection, so the container's `/tmp/.duckdb_spill` holds the spill of a
  query and of the scheduled ingest. Both set it only when `NFL_MCP_DUCKDB_MEMORY_LIMIT` is
  set, so a run with no memory cap keeps DuckDB's own defaults. The image build passes a memory
  cap and no `NFL_MCP_DUCKDB_TEMP_DIR`, so the bake spills inside `/seed` and the `Dockerfile`
  deletes that folder after the ingest.
- **A derived table with no sources is dropped.** `_drop_stale_derived_table` deletes the table
  and its `_ingest_metadata` row together, so a copy built from data that is gone cannot keep
  reporting itself as loaded. A derived table records its rows under `loader_fn="derived"`, and
  the run summary leaves those rows out: nothing was downloaded.

## What One Ingest Run Reports

`run_ingest_datasets` returns an `IngestRunResult`. It lists every fetch that run made, each with
an `IngestStatus`, and an `EspnStop` when the ESPN pull gave up early. A season the run skipped
because an earlier run loaded it is not a fetch. A run that only skipped seasons therefore reports
no fetches, and it is not treated as a failure.

`_print_run_summary` prints the counts per status, the ESPN give-up when there was one, and the
first `_SUMMARY_LINE_CAP` seasons that landed nothing. A dead network fails every season in the
range, so the list is capped and the counts carry the size of the problem.

`nfl-mcp ingest` exits 0 when every fetch landed, 3 (`INGEST_PARTIAL_EXIT`) when a fetch failed
and its failure counts, 1 on a fatal error, and 2 on a usage error. A run where at least one
fetch failed and no fetch put rows in the database prints `NOTHING_LOADED_MESSAGE` and exits 3
even with `--allow-failures`. A run where every fetch came back and every source had no rows is
an answer, not a failure: it prints one plain line and exits 0. A season nflreadpy does not
publish yet is skipped, not failed, so play-by-play for a season before its Week 1 does not set
exit 3. `--allow-failures` turns a partial run (some fetches landed) into exit 0. `--strict`
exits 3 when any single fetch failed, even with `--allow-failures`, and counts an ESPN season
too. `--failures-file` writes the datasets and seasons whose failure decided the run, one per
line.

A failed ESPN season is the one exception. It is written to `_ingest_metadata`, counted in the
summary, and listed with the seasons that landed nothing, and it decides nothing else: it does
not set the exit code and does not stop `stamp_if_current`, so an image build that hit one ESPN
429 still ships a stamped seed. See "An attempt carries the source it fetched from" above.

That exit code decides whether a running container changes its data. `update_db.sh` treats exit 3
as a warning: the failed datasets keep their previous rows, and the script goes on to the upgrade
and the checks. Any other non-zero exit ends the script before the symlink swap, and the server
keeps the database it already had. `entrypoint.sh` runs the no-seed cold ingest with
`--allow-failures`, so a first start where a fetch failed and nothing landed exits 3, stops
before the rename, and never publishes an empty slot.

## Tool Results

- **Every tool failure carries a code.** A failure is a dict with `error` (a string) and `code`
  (an `ErrorCode` member). `tools._err` builds it, and `_error_from_exception` classifies an
  exception into `TIMEOUT`, `DB_NOT_FOUND`, `DB_LOCKED`, `COLUMN_NOT_FOUND`, `TABLE_NOT_FOUND` or
  `SQL_ERROR`. On the wire, `server.py` sends one shape: `error` and `code` at the top level, and
  every other field under `details`, with the tool name. It writes the same shape for the
  failures outside a tool: `UNKNOWN_TOOL`, `INVALID_ARGUMENTS` from the input schema or the
  signature, and `TOOL_EXECUTION_ERROR` for an exception that escaped. The handler is registered
  with `validate_input=False`, and `_argument_failure` runs the schema check and the signature
  bind itself, so a bad argument gets that shape rather than the SDK's plain text.
  `_is_error_result` recognizes a failure and sets `isError=True` on the MCP result, so a client
  sees a failure as a failure. Every response, a failure too, carries `data_as_of`.
- **A cut list says it was cut.** A tool that caps its rows asks the database for one row more
  than it returns and passes the fetch through `_cap`, which returns `truncated: True` when the
  extra row came back. The flag sits next to the tool's count key.
- **Three checks answer before any query runs.** `_team_codes` resolves a team through
  `franchises.team_codes`, so the query carries every code the franchise was stored under, and
  refuses an empty or unknown value with `UNKNOWN_TEAM` and `did_you_mean`. `JAC` is a Jaguars
  alias in `franchises.py`, and `nfl_fantasy_rankings` spells the rankings' own `FA` code.
  `_choice` matches a fixed-value argument against its tuple, ignoring case, and refuses with
  `INVALID_ARGUMENT` plus the allowed list. `season_type` is required on every tool that takes it
  and is matched exactly (`_season_type_arg`). `_check_coverage` refuses a season before the
  tool's registry floor with `OUT_OF_COVERAGE`, and answers a range that only overlaps the floor.
  A season after the loaded data is answered with no rows and a `reason`. Each check raises
  `_Refusal`, and the `_refusals_as_errors` decorator on the tool returns its payload, so a tool
  body lists its checks and never repeats the return.
- **A new tool gets those checks by the same route.** Decorate it with `@_refusals_as_errors`,
  call the helpers, spell a fixed-value argument as one of the module's tuples, and list its
  datasets in `_COVERAGE`. `server.py` builds the schema `enum` and the first-season sentence from
  the same tuples and the same registry lookup, so the schema cannot accept a value the tool
  rejects.
- **Health tools report what is missing.** `nfl_status` and `nfl_catalog` read the ingest log and
  return `issues`: every attempt recorded as `failed`, `empty` or `partial`, with its season and
  its `detail`. `nfl_catalog` also returns `missing_seasons` for each dataset, which are the
  seasons with no data strictly between the two seasons it reports. Both tools count only `loaded`
  and `partial` rows as coverage, so a dataset whose every attempt failed shows zero rows instead
  of a number nobody asked for. A question that is understood but matches no rows stays a success,
  and `nfl_team_records` answers a franchise that did not play that season with a `status` from
  `franchises.availability` rather than an error.
- **A slow query is stopped, and says so.** `_execute` runs each query in a worker thread on its
  own connection and waits `_QUERY_TIMEOUT_SECONDS`. On timeout it calls `conn.interrupt()`, so
  DuckDB stops the query, and raises `QueryTimeoutError`. The caller gets `code: TIMEOUT`. A
  worker still running after the grace join is logged with a warning and counted.

## Team Records & Franchise Identity

`franchises.py` is the single source of truth for team identity. Never map team codes by
hand in a tool or an ingest query — use it, and add the fact there if it is missing:

- `canonical_team(code)` resolves any source's abbreviation to the current one. The three
  codes nflverse still publishes for clubs that moved are aliases: `OAK`→`LV`, `SD`→`LAC`,
  `STL`→`LA`. `HOU` is **not** an alias: in nflverse it is always the Texans (2002+), while
  ESPN used `HOU` for the Oilers in 1996 — which is why ESPN rows are joined on their
  numeric franchise id, never on the abbreviation.
- `era_name(season, team)` gives the name the club played that season under (1970 `IND` is
  the Baltimore Colts). Keep the historical name in the row, never in the key.
- `availability(team, season)` is how "this team has no record" is expressed: a franchise
  that did not exist yet, or sat a season out (CLE 1996–1998), returns a status instead of
  a fabricated `0-0`. Historical records are never estimated.
- `expected_games(season)` / `expected_franchise_count(season)` encode the strike seasons
  (1982: 9 games, 1987: 15) and expansion years. They are what validate an ESPN pull: a
  cumulative record is accepted only when wins+losses+ties equals that season's length.

`RECORDS_FIRST_SEASON` (1970) is deliberately **not** `FIRST_SEASON` (1966): the records
floor is the first merged-era season with a single 26-team league, while the ingest default
reaches back to the final AFL seasons. Changing one should not move the other.

`espn.py` fetches 1970–1998 records. Keep its transport injectable — every test drives a
fake transport, and CI must not depend on ESPN. The reason it reads ESPN's reported record
instead of counting scores is documented in that module's docstring; do not "simplify" it
back to summing scores. `_ingest_espn_records` stops after `ESPN_FAILURE_LIMIT` seasons
fail in a row, which bounds how long a blocked network can hold up an image build; leave
that breaker in place.

34 of 808 franchise-seasons in the band have no record from either ESPN feed (mostly CIN
and IND before 1991). The tempting fix — sum those clubs' game scores instead — is not
free: those seasons contain exactly one hollow Monday-night `0-0` per week, so the fallback
can hand a franchise a phantom tie. If you add it, restrict it to rows the record path
cannot produce, keep it labelled as a different source, and treat a 0-0 result as suspect
rather than as a tie.

## Container Image & Data Refresh

Two independent paths keep the published image current. Change one without breaking the other.

1. **Build time (seed).** `Dockerfile` runs `nfl-mcp ingest ${INGEST_ARGS}` into
   `/seed/nflread.duckdb`, then runs `sh /src/validate_db.sh /seed/nflread.duckdb` so an
   incomplete ingest fails the build. GitHub Actions rebuilds weekly, on release, and on
   demand. This is what makes a fresh `docker pull` current.
2. **Runtime (A/B swap).** `entrypoint.sh` decides what to serve from the
   `/data/nflread.duckdb` symlink, and that link is the whole state. A link that resolves is
   left alone, so a restart keeps serving whichever slot `update_db.sh` last published, even
   when the link points at slot B or its target is mid-rewrite. A missing or dangling link is
   a first start: the script fills `/data/nflread_a.duckdb.tmp` and renames it into
   `nflread_a.duckdb`, so a copy that dies halfway never puts a part-built file where the
   server reads. A plain `/data/nflread.duckdb` file from an older image, and no slot A yet, is
   moved to slot A instead of being ingested over. With no seed at all, as in `Dockerfile.dev`,
   slot A is filled by a cold ingest into that same scratch name. The script then points the link at slot A with `ln -sfn`, starts
   `supercronic`, and `exec`s `nfl-mcp serve`. `update_db.sh` ingests into the idle slot and
   swaps the symlink. No restart is needed because `get_db_connection()` opens a fresh
   read-only connection per tool call.

`update_db.sh` publishes a newer database in this order, and every step that fails ends the run
before the swap, so the live slot keeps serving:

1. Open `/data` itself on descriptor 9 and take `flock -n` on it. A second overlapping run
   prints "Another update is already running; skipping this run." and exits 0. The lock sits on
   an open descriptor, so a run that dies releases it, and no lock file is left in `/data`. A
   missing `flock` is an error, because running unguarded is what the lock prevents.
2. Read the link, and accept only `nflread_a.duckdb` or `nflread_b.duckdb`. The check runs before
   any `rm`, because the script deletes the other slot and a loose target would delete the live
   one. The active slot must also be a regular file.
3. Delete the idle slot, its `.wal` and its `.tmp`, which is the debris of a run that died
   partway.
4. Require free space in `/data` of `NFL_MCP_SPACE_FACTOR` times the active slot's size, default
   2. The copy of the active slot and the WAL DuckDB keeps beside it both land on that
   filesystem. The image sets `NFL_MCP_DUCKDB_TEMP_DIR=/tmp/.duckdb_spill`, and both the ingest
   (`_apply_duckdb_pragmas`) and every query connection (`_runtime_config`) read that variable,
   so DuckDB spills off `/data` and is not counted here. Set the variable higher when the volume
   needs more headroom.
5. Copy the active slot to the idle slot, then run `nfl-mcp ingest` into it. Exit 3 is a
   warning that names the failed datasets. Any other failure ends the run.
6. Refuse the run when the idle slot has a `.wal` next to it right after the ingest. The check
   must come before the upgrade: the upgrade opens the slot for writing, and a write open replays
   that log and removes the evidence.
7. Run `nfl-mcp upgrade` on the idle slot, so it has the current schema version.
8. Refuse the swap when the idle slot has a `.wal` next to it. That file means a writer died
   before DuckDB closed the database, and a reader would see only the last checkpoint. The
   upgrade is the step between the two guards that also writes to the slot.
9. Run `validate_db.sh` on the idle slot, and refuse the swap when it fails or when the script is
   missing.
10. Run `nfl-mcp validate` on the idle slot against the active slot.
11. Make the idle slot read-only and swap the link with `ln -sfn`.

Rules:

- **The published `Dockerfile` must keep `ENTRYPOINT ["/entrypoint.sh"]`.** This feature once
  regressed to a no-op because the updater lived only in `Dockerfile.dev`, and
  `.github/workflows/docker.yml` builds `./Dockerfile` (no `file:` input). `tests/test_dockerfile_dev.py`
  guards this — extend it rather than deleting assertions.
- **One shared script decides whether a database is safe to serve.** `validate_db.sh` opens the
  file read-only, refuses a database with a `.wal` beside it, and refuses one with no rows in
  `plays` or no rows in `schedules`. The image build and `update_db.sh` call the same copy, so a
  build cannot pass while the running container publishes a database the build would have
  rejected. `tests/test_dockerfile_dev.py` checks that both paths call it.
- **The scheduler's own failure is one log line.** `entrypoint.sh` checks for the `supercronic`
  process two seconds after starting it. When it is gone, the script prints
  `[cron] WARNING: supercronic is not running. The database will not refresh on schedule.` and
  starts the server anyway. Nothing restarts the scheduler, and stale data beats no data.
- **Never re-ingest at container start.** A cold ingest takes 10+ minutes and blocks
  readiness behind any startup probe or health check. Copy the seed instead. Only the no-seed
  fallback ingests at start, and a published image always carries a seed. That fallback runs
  with `--allow-failures`, so a dead network exits 3 and `set -eu` stops the script before the
  rename. A cold ingest whose every source answered with no rows exits 0, and `entrypoint.sh`
  publishes that empty slot: it runs no `validate_db.sh` check. `update_db.sh` and the image
  build both run that check, and both refuse a slot with no rows in `plays` or `schedules`.
- The seed bake passes `--pre1999` (ESPN season records, ~60 requests); the runtime ingest
  in `entrypoint.sh` and `update_db.sh` must not, or every refresh depends on a third-party
  API at runtime. `test_seed_includes_pre1999_records_but_runtime_does_not` guards this.
- The image must run standalone (`docker run -p 8000:8000 <image>`) with no mounted
  volume and no external storage. `/data` and `/var/log/nflmcp` are `chmod 777` so an
  arbitrary `--user` works. Anything that requires platform-specific storage (e.g. an
  Azure Files mount) is out of scope for the image design.
- Use `ln -sfn`, not `ln -s`: the plain form fails with `File exists` on an existing
  regular file, and both scripts run `set -eu`, so the container crash-loops.
- Scripts are POSIX `/bin/sh` (no `pipefail`); `sh -n` is asserted in the tests.

## Common Pitfalls

- **Claude Desktop can't access `~/Desktop`** due to macOS sandboxing. The Claude Desktop venv must be outside Desktop — currently `~/nfl-mcp-venv`. Any code changes require `pip install --force-reinstall` into that venv.
- **DuckDB write lock**: the server opens read-only connections. Ingest opens a write connection. Don't run ingest while the server is running against the same DB file.
- **Player names are abbreviated**: `'P.Mahomes'`, `'T.Hill'` — use `ILIKE '%name%'` for fuzzy matching.
- **Team codes are historical**: `plays` and `schedules` carry the code of the era (`OAK`, `SD`, `STL`). Normalize with `franchises.canonical_team()` instead of comparing raw codes across decades, or a Raiders-through-2024 query silently returns two franchises' worth of gaps.
- **Pre-1999 game results are not in the database**, and ESPN's archived scores for 1970–2001 are not a substitute: the Monday-night game of every week is filed `0-0` with no box score. Only ESPN's cumulative per-club record is trustworthy for those years — see `espn.py`.
- **Boolean columns are `DOUBLE PRECISION` 0/1** in the plays table — filter with `= 1`, not `IS TRUE`.
- **`nfl_query` is last resort** — prefer structured tools. The tool descriptions use "PREFERRED for..." language intentionally to steer the model away from raw SQL.
- **Models invent column names** (`play_epa` for `epa`). `_sql_error_detail` in `tools.py` turns DuckDB's binder/catalog text into `code` + `unknown_column`/`unknown_table` + `did_you_mean`, re-ranked against `information_schema` by last snake_case word, because DuckDB's own "Candidate bindings" are spelling-based and miss the real metric. Keep the vocabulary hints in `schema_pbp.py` up to date — that text is what the model actually reads.

## Coverage Gaps (Known)

The full suite is at 100% line coverage on every module in `nfl_mcp/`. `ingest.py` and `cli.py` are
no longer gaps. The ingest helpers run against an in-memory DuckDB and a stubbed nflreadpy,
`tests/test_dockerfile_dev.py` runs the two container scripts against a fake `/seed`, a fake
`/data` and a fake `PATH`, and the CLI runs through `click.testing.CliRunner`. What the number
does not cover:

- **`pytest -m unit` alone reaches 99%.** The 14 lines it misses are the player, week, position
  and ranking-set filters inside `nfl_fantasy_opportunity` and `nfl_fantasy_rankings`, plus the
  fallback for a non-numeric `limit`. `TestNflFantasyOpportunity` and `TestNflFantasyRankings` in
  `tests/test_tools.py` cover them, and those classes need a database.
- **Lines, not branches.** `pyproject.toml` sets no `--cov-branch`, so a line that ran once reads
  as covered. A filter combination no test asks about leaves no mark.
- **No live network.** `nflreadpy` and ESPN are stubbed, so a change upstream shows up as data
  that stops matching, not as a failed test. CI makes one real fetch:
  `nfl-mcp ingest --dataset all --start 2024 --end 2024`.
- **`pytest` never boots the image.** It reads both Dockerfiles and runs the container scripts
  against stubs for `nfl-mcp`, `supercronic`, `du`, `df` and `flock`. `docker-build.yml` builds
  the image and needs an HTTP 200 with a `result` from the `initialize` call on `/mcp`.
