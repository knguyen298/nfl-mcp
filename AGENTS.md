# AGENTS.md

Guidance for AI agents working on this codebase.

## What This Project Is

`nfl-mcp` is a local MCP server that ingests NFL data from nflreadpy into a DuckDB database and exposes it to AI clients (Claude Desktop, VS Code) via the Model Context Protocol. All data lives in a single `~/.nfl-mcp/nflread.duckdb` file. No network calls happen at query time — only at ingest time.

## Repository Layout

```
nfl_mcp/
  cli.py          # Click CLI: serve, ingest, init, setup-client, doctor
  server.py       # MCP server: TOOLS list + _DISPATCH + call_tool handler
  tools.py        # Tool implementations (nfl_schema, nfl_status, nfl_query, ...)
  schema_pbp.py   # PBP schema strings (_SCHEMA_SUMMARY, _SCHEMA_CATEGORIES) — pure data, no logic
  ingest.py       # Data ingestion: run_ingest_datasets, idempotency, schema reconciliation
  registry.py     # Declarative dataset definitions (REGISTRY, DatasetDef)
  espn.py         # External fetcher: 1970-1998 final season records from ESPN's scoreboard API
  franchises.py   # Franchise identity: canonical team codes, era names, availability, season shapes
  database.py     # get_db_connection() context manager (read-only DuckDB)
  config.py       # ~/.nfl-mcp/config.json read/write, get_duckdb_path()

entrypoint.sh   # Container start: seed -> slot A -> symlink -> supercronic -> serve
update_db.sh    # Scheduled refresh: ingest idle slot, atomically swap the symlink
Dockerfile      # Published image (baked seed + runtime A/B updater)
Dockerfile.dev  # Same updater, no baked seed (cold ingest on first start)

tests/
  test_registry.py   # Registry structure, coverage windows, loader functions
  test_ingest.py     # Ingest helpers using in-memory DuckDB (no network)
  test_cli.py        # CLI commands using click.testing.CliRunner
  test_tools.py      # Tool functions — unit + integration (require_db fixture)
  test_espn.py       # ESPN parsing and fetch logic against a fake transport (no network)
  test_franchises.py # Canonical codes, era names, availability, expected season shapes
  test_safety.py     # SQL injection / forbidden keyword guardrails
```

## Key Architectural Rules

**Adding a new MCP tool** requires changes in exactly two files:
1. `tools.py` — implement the function, add to `__all__`
2. `server.py` — add a `Tool(...)` entry to `TOOLS` and an entry to `_DISPATCH`

A tool reports a failure as a dict with a top-level `error` string. It can add a `hint`.
`call_tool` in `server.py` sets `isError` on every answer that has that key, and it gives an
exception the same shape. An empty answer is not an error: give it a `reason` instead.

**Adding a new dataset** requires changes in exactly one file:
1. `registry.py` — add a `DatasetDef` entry to `REGISTRY`

The ingest loop in `ingest.py` is fully generic — it reads from the registry and handles everything else (idempotency, schema reconciliation, bulk vs ranged mode).

**Exception — a dataset from outside nflreadpy.** `DatasetDef.source` defaults to
`"nflreadpy"`. A dataset that reaches another publisher sets `source="<name>"`,
`default=False`, and a `loader_fn` that names the fetcher rather than an nflreadpy
function; `run_ingest_datasets` routes on `source` to the matching ingest helper.
`espn_records` is the only one so far — keep it that way unless a new external source is
worth taking off the container's scheduled refresh path.

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

# After changing tool code, reinstall into Claude Desktop's venv
pip install --force-reinstall /path/to/your/nfl-mcp/checkout
# Then restart Claude Desktop
```

## Testing Strategy

- **Unit tests** (`test_registry`, `test_ingest`, `test_cli`): no DB, no network, always run in CI
- **Integration tests** (`test_tools.py` classes marked `@pytest.mark.integration`): require a loaded DuckDB; skipped automatically via `require_db` fixture if DB is absent
- **Safety tests** (`test_safety.py`): SQL injection guardrails, no DB needed
- CI runs `nfl-mcp ingest --dataset all --start 2024 --end 2024` before tests to populate the DB
- **CI enforces 100% coverage** via `pytest -q --cov-fail-under=100` on the full suite. The floor lives on the CI command (not `pyproject.toml` addopts) so local unit-only runs (`pytest -m unit`, ~99.7%) aren't held to the full-suite floor. New code must keep total coverage at 100% or CI fails.

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
- **Metadata**: `_ingest_metadata` table tracks what datasets/seasons have been loaded
- **Tables**: 42 tables in the local database in November 2026, `_ingest_metadata` included. The
  number changes with the registry, so use `nfl_schema()` to list them or `nfl_schema(table=<name>)`
  for columns.

## Dataset Registry

Each entry in `REGISTRY` is a `DatasetDef` with:
- `loader_fn`: the nflreadpy function name (must exist as `hasattr(nflreadpy, loader_fn)`)
  for `source="nflreadpy"` datasets; for external ones it names the fetcher instead
- `source`: `"nflreadpy"` (default) or an external publisher; anything non-default must be
  `default=False` so the scheduled container refresh never calls a third-party API
- `seasonal`: whether it accepts a `seasons=` parameter
- `storage`: `"replace"` (static data) or `"append_by_season"` (time-series)
- `min_season`/`max_season`: clip the ingest range to avoid known 404s
- `default`: whether included in `nfl-mcp ingest` with no flags

## Ingest Idempotency

The ingest loop checks `_ingest_metadata` before every dataset+season. Re-running the same ingest command is safe — already-loaded combinations are skipped. Use `--fresh` to force re-ingest.

## Derived Table Shape

The derived tables (`team_offense_stats`, `team_defense_stats`, `situational_stats`, `formation_effectiveness`, the five `player_*` tables, `injury_return_curve` and `team_records`) are rebuilt from rows already in the database. `nfl-mcp ingest` rebuilds them at the end of every run, whether or not any dataset was fetched, and `nfl-mcp rebuild` rebuilds them on its own with no network at all.

No structured tool reads `formation_effectiveness`. A client reaches it only through `nfl_query`,
and `schema_pbp.py` says so. If you add a tool for it, keep it to one tool.

- `nfl_mcp/ingest.py::DERIVED_SHAPE_VERSION` describes the column shape this code writes. Raise it in the same commit that changes a derived table's columns; `rebuild_derived_tables` stamps the number into `_ingest_metadata`.
- `--check` also answers `stale` when a fetch is newer than the shape stamp (a `--skip-views` run): the
  columns are right but the numbers lag the rows. Tools do not refuse on that; `nfl_status` and
  `nfl-mcp doctor` report it.
- `nfl_mcp/cli.py::rebuild` with `--check` prints `current` or `stale` and exits 0 or 1 without a write lock, so a startup probe or a health check can ask it. `entrypoint.sh` asks it before `exec nfl-mcp serve`, because a container that pulls a new image over an existing `/data` volume keeps the slot it already had.
- A rebuild is local arithmetic, not an ingest: 1.3 s measured on a 582 MB database holding 521,905 plays. A cold ingest is still forbidden at container start — see the container section.
- `rebuild_derived_tables` wraps every drop, `CREATE TABLE AS` and the shape stamp in one transaction (`conn.begin()`), because each builder drops its table before it recreates it. An interrupted run therefore leaves the tables it started with and no stamp, and a guarded tool never sees a half-rebuilt set that the column probe vouches for. It starts the transaction itself, so a caller must hand it a connection with no transaction open: DuckDB refuses a nested BEGIN.
- Nine tools refuse to answer from a stale shape instead of guessing: `nfl_team_stats`, `nfl_compare`, `nfl_separation_opportunity`, `nfl_team_records`, `nfl_td_luck`, `nfl_role_trend`, `nfl_drop_rate`, `nfl_contract_value` and `nfl_injury_return`. Their message names the stale table or the stamp, and gives both remedies — `nfl-mcp rebuild` with the server stopped, or `/update_db.sh` in a container. `Dockerfile` runs the same probe on the seed, so an image whose build produced the old shape fails the build.

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
   `/seed/nflread.duckdb`, then validates it (no WAL, `plays`/`schedules` non-empty) so an
   incomplete ingest fails the build. GitHub Actions rebuilds weekly, on release, and on
   demand — this is what makes a fresh `docker pull` current.
2. **Runtime (A/B swap).** `entrypoint.sh` copies the seed to `/data/nflread_a.duckdb`,
   links `/data/nflread.duckdb` to slot A only when the link is missing or dangling (a restart
   keeps slot B after a swap), starts `supercronic`, then `exec` `nfl-mcp serve`.
   `update_db.sh` ingests into the idle slot and swaps the symlink only when the ingest exits 0
   (any failed fetch is non-zero), `plays` and `team_offense_stats` have rows, and
   `nfl-mcp rebuild --check` says `current` for that slot. No
   restart is needed because `get_db_connection()` opens a fresh read-only connection per
   tool call.

Rules:

- **The published `Dockerfile` must keep `ENTRYPOINT ["/entrypoint.sh"]`.** This feature once
  regressed to a no-op because the updater lived only in `Dockerfile.dev`, and
  `.github/workflows/docker.yml` builds `./Dockerfile` (no `file:` input). `tests/test_dockerfile_dev.py`
  guards this — extend it rather than deleting assertions.
- Never re-ingest at container start. A cold ingest takes minutes (the `v0.5.0` image build ran the
  full ingest, ESPN records included, in 4 m 46 s) and blocks readiness behind any startup probe
  or health check; copy the seed instead.
- The seed bake passes `--pre1999` (ESPN season records, ~60 requests); the runtime ingest
  in `entrypoint.sh` and `update_db.sh` must not, or every refresh depends on a third-party
  API at runtime. `test_seed_includes_pre1999_records_but_runtime_does_not` guards this.
- The image must run standalone (`docker run -p 8000:8000 <image>`) with no mounted
  volume and no external storage. `/data` and `/var/log/nflmcp` are `chmod 777` so an
  arbitrary `--user` works. Anything that requires platform-specific storage (e.g. an
  Azure Files mount) is out of scope for the image design.
- The server has no authentication. CORS allows every origin (`*`), and the container binds
  `0.0.0.0`. This is deliberate, because nfl-mcp is a local tool server that reads a read-only
  database. Do not publish port 8000 to an untrusted network. If you need remote access, put
  an authenticating proxy in front of it. Do not add authentication to the image.
- Use `ln -sfn`, not `ln -s`: the plain form fails with `File exists` on an existing
  regular file, and both scripts run `set -eu`, so the container crash-loops.
- Scripts are POSIX `/bin/sh` (no `pipefail`); `sh -n` is asserted in the tests.

## Common Pitfalls

- **Claude Desktop cannot read files under `~/Desktop`**, because macOS sandboxes it. Put the virtualenv that Claude Desktop runs, and the checkout you install from, outside `~/Desktop` and the other protected folders (`~/Documents`, `~/Downloads`). After a code change, run `pip install --force-reinstall` into that virtualenv and restart Claude Desktop.
- **DuckDB write lock**: the server opens read-only connections. Ingest opens a write connection. Don't run ingest while the server is running against the same DB file.
- **Player names are abbreviated** (`'P.Mahomes'`, `'T.Hill'`) and are not unique: `nfl_resolve_player` in `tools.py` settles a name into one player. It is an internal function and not an MCP tool, so no client calls it directly; every tool with a `player` parameter runs it before it queries and returns the resolved id with the rows. Filter rows by `gsis_id` (`pfr_id` in `snap_counts`), never with `ILIKE '%name%'`. A name that covers several players returns `ambiguous` plus labelled candidates. Keep `ILIKE` only as the fallback in a raw `nfl_query` over a table that has no id column.
- **Team codes are historical**: `plays` and `schedules` carry the code of the era (`OAK`, `SD`, `STL`). Normalize with `franchises.canonical_team()` instead of comparing raw codes across decades, or a Raiders-through-2024 query silently returns two franchises' worth of gaps.
- **Pre-1999 game results are not in the database**, and ESPN's archived scores for 1970–2001 are not a substitute: the Monday-night game of every week is filed `0-0` with no box score. Only ESPN's cumulative per-club record is trustworthy for those years — see `espn.py`.
- **Boolean columns are `DOUBLE PRECISION` 0/1** in the plays table — filter with `= 1`, not `IS TRUE`.
- **`nfl_query` is last resort** — prefer structured tools. The tool descriptions use "PREFERRED for..." language intentionally to steer the model away from raw SQL.
- **Models invent column names** (`play_epa` for `epa`). `_sql_error_detail` in `tools.py` turns DuckDB's binder/catalog text into `code` + `unknown_column`/`unknown_table` + `did_you_mean`, re-ranked against `information_schema` by last snake_case word, because DuckDB's own "Candidate bindings" are spelling-based and miss the real metric. Keep the vocabulary hints in `schema_pbp.py` up to date — that text is what the model actually reads.

## Coverage

CI holds the full suite at 100 percent line coverage (see Testing Strategy). There are no known
gaps. A unit-only run (`pytest -m unit`) is lower, because some lines run only against a loaded
database.
