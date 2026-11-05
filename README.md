# nfl-mcp

MCP server for NFL data (default backfill to 1966; play-by-play from 1999, rosters to 1920 on request), powered by [nflreadpy](https://github.com/nflverse/nflreadpy) and DuckDB.
Query play-by-play, rosters, injuries, stats, and more using natural language in Claude Code, VS Code, or Claude Desktop.

Ask Claude questions like:
- *"Who had the best EPA per play in 2024?"*
- *"Show me Patrick Mahomes' completion % over expected by season"*
- *"Compare 4th quarter red zone efficiency for KC vs PHI in 2023"*
- *"Which defenses had the highest sack rate in 3rd & long situations?"*
- *"Who was on IR for the Eagles in Week 10, 2023?"*
- *"Show me snap count trends for the Chiefs receiving corps in 2024"*

## Quickstart

```bash
pip install git+https://github.com/knguyen298/nfl-mcp.git
nfl-mcp init               # configure, load data, and start the server
```

`pip install nfl-mcp` works too, but installs the upstream PyPI build — this repository
publishes only the container image, so its own fixes are not on PyPI.

`init` walks you through setup and offers to start the server immediately when done. No database server to install. No credentials to manage. Data is stored locally in DuckDB.

## Run the container

The image is self-contained: the database is inside it, so there is no external storage,
database server, or credentials to configure.

```bash
docker run -d -p 8000:8000 -v nfl-mcp-data:/data \
  ghcr.io/knguyen298/nfl-mcp:latest
```

Point any MCP client at `http://localhost:8000/mcp`. The `-v` mount is **optional**:

| | Without `-v` | With `-v nfl-mcp-data:/data` |
|---|---|---|
| First start | copies the baked seed | copies the baked seed |
| Scheduled refreshes | apply immediately, live in the container's writable layer | apply immediately |
| Container restart | falls back to the image's seed | keeps the last refreshed copy |

Budget roughly **twice the database size** of free space in `/data` (the A/B slots) plus
room for ingest spill, and **2 GB of memory** if you want the scheduled ingest to have
headroom — the server itself is comfortable in 1 GB. `--user` is fine; `/data` and
`/var/log/nflmcp` are world-writable in the image.

> **The image ships with data and keeps itself current.** A complete DuckDB database is baked into the image as a **seed**, so a new container is queryable in seconds instead of spending 10+ minutes ingesting on startup. At runtime, `entrypoint.sh` copies that seed into slot A and starts `supercronic`, which runs `update_db.sh` on a schedule: ingest into the **idle slot**, then atomically swap the `/data/nflread.duckdb` symlink. The server opens a fresh read-only connection per query, so new data appears on the next request — no restart, no downtime, and a failed ingest leaves the previous copy still serving.

### How the container stays current

| Path | What it does | Cadence |
|---|---|---|
| GitHub Actions build | Re-ingests the seed baked into the image, so **whoever pulls next gets recent data** | Weekly (Thu 08:00 UTC), on release, or manually |
| In-container updater | `update_db.sh` refreshes the running container via the A/B slot swap | Sun/Mon/Wed 03:00 UTC (game-day) + Thu 06:00 UTC (corrections) |

Notes on the updater:

- **Watch `/var/log/nflmcp/nfl_update.log`.** It is the only place a failed refresh reports itself; the server keeps serving the last good copy regardless.
- **The updater inherits the runtime memory caps** (`512MB`, 1 thread), so raise the container memory to 2 GB if that log shows the ingest being killed.
- **Check what a running container actually holds** with the `nfl_status` tool: `last_refreshed` reports the newest `_ingest_metadata` timestamp.
- **Run one container per database.** Every instance keeps its own `/data` slots, so scaling out replicates both the ingest work and the storage.

**Publishing and pulling the image:**

1. The [`Publish container image`](.github/workflows/docker.yml) workflow builds `Dockerfile` and pushes it to `ghcr.io/<owner>/<repo>` — the image path follows whichever repository the workflow runs in. It runs weekly, on each GitHub release, or manually via *Actions → Run workflow*. The build ingests every default dataset, so it takes considerably longer than a normal image build.
2. To `docker pull` without credentials, make the GHCR package **public**: repo → *Packages* → `nfl-mcp` → *Package settings* → *Change visibility → Public*. Scheduled builds are **disabled on forks by default** — enable them under *Actions* if you want the weekly refresh there.
3. A release publishes the **image only**. The `Publish to PyPI` workflow is gone: PyPI's trusted publisher is registered for the upstream repository, so every release on a fork failed that job while shipping nothing. Cut a release (or *Run workflow* on `Publish container image`) and the image is what lands.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

### 1. Initialize

```bash
nfl-mcp init
```

The wizard will:
1. Configure the local DuckDB database path
2. Download the default NFL datasets (play-by-play, rosters, stats, injuries, and more)
3. Auto-configure your IDE (Claude Desktop and/or VS Code)
4. Offer to start the server immediately

Options:

```
--skip-ingest       Configure without loading data
```

### 2. Start the server

`init` offers to start the server for you. If you need to start it manually later:

```bash
nfl-mcp serve
nfl-mcp serve --port 9000
nfl-mcp serve --host 0.0.0.0
```

The server uses the [MCP Streamable HTTP](https://modelcontextprotocol.io/docs/concepts/transports) transport. Point any MCP client at `http://<host>:<port>/mcp`. The server also sends permissive CORS headers, so browser-based clients (e.g. the [MCP Inspector](https://github.com/modelcontextprotocol/inspector)) can connect — their preflight `OPTIONS` requests are answered instead of rejected.

> **Note:** The server must be running for your IDE to connect. Run `nfl-mcp serve` in a terminal and keep it open.

### 3. Verify

```bash
nfl-mcp doctor
```

Checks database connectivity, loaded data, and IDE configuration.

### 4. Manual client configuration (optional)

If you skipped IDE setup during init, or need to reconfigure:

```bash
nfl-mcp setup-client                    # auto-detect clients
nfl-mcp setup-client --client vscode    # VS Code only
nfl-mcp setup-client --client claude-desktop
```

Or configure manually. Add to `.vscode/mcp.json` (VS Code):

```json
{
  "servers": {
    "nfl": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (Claude Desktop):

```json
{
  "mcpServers": {
    "nfl": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

## CLI Reference

```
nfl-mcp init               Interactive setup wizard
nfl-mcp serve              Start the MCP server (Streamable HTTP, default port 8000)
nfl-mcp ingest             Load NFL data into the database
nfl-mcp rebuild            Rebuild the derived tables from the loaded data (no download)
nfl-mcp validate           Check a database file before it replaces the active one
nfl-mcp upgrade            Bring an older database up to the current schema
nfl-mcp setup-client       Configure IDE MCP clients
nfl-mcp doctor             Health check
```

### Serve options

```bash
nfl-mcp serve
nfl-mcp serve --port 9000
nfl-mcp serve --host 0.0.0.0
```

### Ingestion options

```bash
nfl-mcp ingest                          # default datasets, all available seasons
nfl-mcp ingest --dataset all            # every dataset
nfl-mcp ingest --dataset schedules      # one specific dataset
nfl-mcp ingest --dataset pbp --dataset injuries   # multiple datasets
nfl-mcp ingest --start 2020 --end 2024  # limit to a season range
nfl-mcp ingest --fresh                  # re-ingest even if already loaded
nfl-mcp ingest --pre1999                # also pull 1970–1998 season records from ESPN
nfl-mcp ingest --list                   # show all available dataset names
```

Ingest is **idempotent** — re-running skips completed seasons already in the database. The current season and the static snapshot tables (rankings, players, contracts, teams) are re-fetched on every run, so run `nfl-mcp ingest` after a game week to pick up new stats. The current season is computed from today's date (the league year rolls over in March), so a new season is picked up automatically.

### Upgrade an older database

```bash
nfl-mcp upgrade [--db PATH] [--check] [--no-fetch]
```

The database records a schema version. A database with no record is version 0. That is
every database built by v0.4.0 or earlier. `nfl-mcp upgrade` finds what an older database
lacks and fixes it. It never deletes the file. It runs three steps in this order:

1. Rebuild the derived tables. This needs no network.
2. Fill the `season` column of `depth_charts`. This downloads data.
3. Reload any raw table that lacks a column the tools need. This downloads data. When this
   step runs, the command rebuilds the derived tables once more afterward, because they
   are built from the raw tables.

The command stamps the current version only when nothing is left to do. Running it twice is
safe.

| Option | Effect |
|---|---|
| `--db PATH` | Upgrade this file. The default is the configured database. |
| `--check` | Change nothing. Print one line per step (`ok`, `NEEDED`, or `NEEDED (network)`) and the version. |
| `--no-fetch` | Run the steps that need no network. List the others as pending. |

Exit codes:

| Code | Meaning |
|---|---|
| 0 | The database is up to date. |
| 3 | An upgrade is needed. With `--no-fetch`, only steps that need the network are left. |
| 1 | An error, a failed step, or a database newer than this version of nfl-mcp. |

A missing file is an error. The command never creates one. The upgrade writes to the file.
In the container, the active slot is read-only on purpose. Run `update_db.sh` instead. It
upgrades the new slot before it swaps.

`nfl-mcp serve` checks the database at start and logs one warning when an upgrade is
needed. It keeps running. `nfl-mcp doctor` prints the schema version and the steps that are
needed. `nfl-mcp validate` fails a database file whose version is not the current one.

## Configuration

Every setting is an environment variable. All are optional.

| Variable | Default | What it does |
|---|---|---|
| `NFL_MCP_DB_PATH` | `~/.nfl-mcp/nflread.duckdb` (`/data/nflread.duckdb` in the container) | Path of the DuckDB file. It overrides the path in the config file. |
| `NFL_MCP_DUCKDB_MEMORY_LIMIT` | DuckDB default (`512MB` in the container) | Memory cap for every DuckDB connection. Use a number and a unit, such as `512MB` or `2GiB`. A bad value stops `serve` with exit code 1. |
| `NFL_MCP_DUCKDB_THREADS` | DuckDB default (`1` in the container) | Thread cap. Use a positive whole number. A bad value stops `serve` with exit code 1. |
| `NFL_MCP_DUCKDB_TEMP_DIR` | DuckDB default (`/tmp/.duckdb_spill` in the container) | Writable directory where DuckDB spills a large query to disk. |
| `NFL_MCP_AUTH_TOKEN` | unset | When set, every request needs `Authorization: Bearer <token>`. See Security below. |
| `NFL_MCP_CORS_ORIGINS` | unset (all origins) | Comma-separated list of origins allowed to call the server from a browser. |
| `NFL_MCP_MAX_CONCURRENT_QUERIES` | `2` | How many tool calls run at once. A bad value logs a warning and the default applies. |
| `NFL_MCP_QUERY_TIMEOUT_SECONDS` | `10` | Seconds a query may run before the server cancels it. The server reads it on every call. A bad value logs a warning and the default applies. |
| `NFL_MCP_UPGRADE_ON_START` | on | Container only. Set to `0` to turn off the upgrade check at start. See below. |

**Upgrade on start.** When the container starts, `entrypoint.sh` runs
`nfl-mcp upgrade --check` on the active database. If an upgrade is needed, it starts one
background `update_db.sh` run and does not wait for it. The server starts either way. The
update builds a new, upgraded copy in the idle slot and swaps it in. Set
`NFL_MCP_UPGRADE_ON_START=0` to skip the check.

## Security

The server is read-only. It binds to localhost unless you say otherwise.

- **Bind address.** `nfl-mcp serve` listens on `127.0.0.1`. Pass `--host 0.0.0.0` to accept
  other machines. The container does this on purpose.
- **Bearer token.** Set `NFL_MCP_AUTH_TOKEN` and every request needs the header
  `Authorization: Bearer <token>`. That includes `/healthz`. Without a token, the server has
  no authentication, so set one before you expose it beyond localhost. A token that is only
  whitespace stops `serve` with exit code 1.
- **CORS.** Set `NFL_MCP_CORS_ORIGINS` to limit which browser origins can connect. When it
  is unset, the server allows every origin and logs a warning.
- **Container user.** The container runs the server and the updater as the non-root user
  `nflmcp`.

### Read-only guards

- Each query opens a new read-only connection. Nothing in the server writes to the
  database it serves.
- The connection turns off file and network access, extension install, and extension
  autoload. Functions that read files, and the statements `COPY`, `ATTACH`, `INSTALL`,
  `LOAD`, and `SET`, all fail. The connection locks its configuration after it opens.
- `nfl_query` takes one statement per call. The statement must be a `SELECT` (a `WITH`
  query, `DESCRIBE`, or `SUMMARIZE` also counts) or `EXPLAIN` of a `SELECT`. The tool
  rejects every other statement type.
- `nfl_query` returns at most 500 rows. Every tool stops a query after the timeout above.
- Every tool carries the MCP annotations `readOnlyHint: true`, `destructiveHint: false`,
  `idempotentHint: true`, and `openWorldHint: false`.

## Datasets

All data is ingested over HTTPS at ingest time (never at query time) from three upstream
projects, via [nflreadpy](https://github.com/nflverse/nflreadpy), and stored locally in
DuckDB:

| Upstream | What it supplies |
|---|---|
| [nflverse-data](https://github.com/nflverse/nflverse-data) | play-by-play, schedules, rosters, stats, injuries, snap counts, depth charts, officials, participation, Next Gen Stats, PFR advanced stats, FTN charting, teams, players, contracts, trades, draft picks, combine |
| [DynastyProcess](https://github.com/dynastyprocess/data) | fantasy player IDs and draft/weekly rankings |
| [ffverse](https://github.com/ffverse/ffopportunity) | fantasy opportunity model |
| [ESPN](https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard) | final season win-loss records before 1999 (opt-in, see below) |

**Every nflreadpy dataset below is ingested by default** — `nfl-mcp ingest` loads the full
family so any data a client might need is already there. The single exception is
`espn_records`, which reaches a different publisher and is opt-in.

> **Season coverage: 1966 onward by default, deeper on request.** The default backfill
> starts at **1966** — the AFL–NFL merger agreement year, so the final four AFL seasons are
> included — and each dataset is then clipped to its own first available season (the
> `min_season` in `nfl_mcp/registry.py`), so a fresh install loads rosters from 1966 while
> play-by-play and the stats tables start at their 1999 source floor and Next Gen Stats at
> 2016. Pass `--start` to go deeper — `--start 1920` adds the earlier pre-merger rosters, and no
> dataset ever requests a season its source never published. `nfl-mcp ingest --list` prints
> the window for every dataset. Non-seasonal reference tables (draft picks, combine,
> contracts, players) carry their full historical record.

| Table | Loaded range |
|---|---|
| `rosters` | 1966–present (1920 with `--start 1920`) |
| `plays` | 1999–present |
| `schedules` | 1999–present |
| `player_stats` | 1999–present |
| `team_stats_raw` | 1999–present |
| `depth_charts` | 2001–present |
| `rosters_weekly` | 2002–present |
| `ff_opportunity` | 2006–present |
| `injuries` | 2009–present |
| `snap_counts` | 2013–present |
| `officials` | 2015–present |
| `nextgen_stats_*` | 2016–present |
| `participation` | 2016–present |
| `pfr_advstats_*` | 2018–present |
| `ftn_charting` | 2022–present |
| `teams` | current |
| `players` | all-time |
| `contracts` | historical |
| `trades` | historical |
| `draft_picks` | 1980–present |
| `combine` | all-time |
| `ff_playerids` | current |
| `ff_rankings_draft` | current |
| `ff_rankings_week` | current |
| `espn_team_records` | 1970–1998 (opt-in, `--pre1999`) |
| `team_records` | 1970–present once both sources are loaded (derived) |

> Older seasons are thinner than modern ones. Play-by-play carries `epa`, `wp`, and
> `shotgun` from 1999. Next Gen Stats come from player tracking and start in 2016. FTN
> charting is manual charting by FTN staff. It is not tracking data, and it starts in
> 2022. Snap counts and weekly depth also start in their own years, listed above. A query
> that spans decades mixes eras with very different column coverage. Rosters before 1966
> are available with `--start 1920` and cover the pre-merger leagues.

```bash
nfl-mcp ingest                 # load the full nflverse family (default)
nfl-mcp ingest --list          # see all dataset names
nfl-mcp ingest --dataset pbp   # load just one dataset
```

## Team records, 1970 to present

`nfl_team_records` answers win-loss and standings questions. One table is not enough,
because no publisher spans the era:

| Seasons | Source | How the record is obtained |
|---|---|---|
| 1999–present | nflverse `schedules` | wins, losses and ties counted from game scores |
| 1970–1998 | ESPN scoreboard API | the cumulative overall record ESPN reports per club |

nflverse has no games before 1999, so the earlier years need one extra, opt-in step:

```bash
nfl-mcp ingest --pre1999     # 29 seasons x 2 requests; a few minutes, idempotent
```

It is opt-in because it talks to ESPN rather than nflreadpy, so the container's scheduled
refresh never calls it and `--dataset all` on a modern season range requests nothing from
it. Re-running only fetches seasons not already recorded. The published image bakes these
records at build time, so `docker run ghcr.io/knguyen298/nfl-mcp` answers 1970+ standings
with no extra step.

If three seasons fail back to back the pull stops and says so — that pattern means the
endpoint is down, not that those seasons are empty. Run `--pre1999` again later and it picks
up where it left off.

Two things about the ESPN years are worth knowing:

* **Final records are reliable; game scores are not.** ESPN's archive is missing the
  scores for about 7% of games per season from 1970 to 2001 — always the Monday-night
  slot, filed as `Final 0-0` with no box score. Summing scores would invent ties and get
  roughly one in fifteen records wrong, so the record ESPN itself reports is used instead.
  Checked against published standings: 30 of 30 franchises correct for 1996.
* **Some franchise-seasons have no record at all.** Across 1970–1998, ESPN publishes a
  complete regular-season record for **774 of 808** franchise-seasons (96%). The gaps sit
  almost entirely with two clubs: **Cincinnati** (14 seasons, 1970–1990) and **Indianapolis**
  (14 seasons, 1970–1983). The other six are scattered — Pittsburgh in 1974 and 1997, Green
  Bay and Tampa Bay in 1977, Dallas in 1989, Washington in 1997. Eight seasons are complete:
  1985 and 1991–1996, 1998. Missing rows report `record_not_available` — never a guessed
  record — and the rest of the season still loads.
  This is not a source-selection mistake: ESPN's standings feed (`apis/v2/.../standings`) was
  measured over the same 29 seasons, agrees with all 774 records exactly, and recovers none
  of the 34 gaps — its figures for those clubs are in-season snapshots (the 1997 Steelers
  appear as 10-5, fifteen games) that fail the same schedule-length check.

Records are stored under each franchise's **current** abbreviation, so one query spans
relocations and renames: the Raiders are `LV` in 1970 and in 2024, and historical codes
(`OAK`, `SD`, `STL`) resolve to the same club. Every row also carries `club`, the name that
franchise played that season under, so `team=IND, season=1970` reports the Baltimore Colts.
A franchise that did not exist yet, or sat the season out, is reported that way rather than
as `0-0`: the Texans return `franchise_not_yet_active` for 1995, the Browns
`franchise_inactive` for 1996–1998, and any season before 1970 `before_coverage`.

```sql
SELECT season, team, wins, losses, ties FROM team_records WHERE season = 1972 AND team = 'MIA';
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `nfl_schema` | Database schema reference. The default call gives a summary, the verified table relationships, data notes, and the derived tables. Pass `category` for column detail or `table` for one table's columns, grain, and join keys. |
| `nfl_status` | Database health: total plays, loaded seasons, and per dataset the row count, newest season, newest week, and last refresh time |
| `nfl_query` | Raw SQL SELECT for custom queries (500 row cap, 10s timeout) |
| `nfl_search_plays` | Find plays by player, team, season, season type, situation, touchdowns, etc. |
| `nfl_team_stats` | Pre-aggregated team offense, defense, and situational stats |
| `nfl_player_stats` | Player stats by season and season type — passing, rushing, or receiving |
| `nfl_compare` | Side-by-side comparison of two teams or two players |
| `nfl_schedule` | Game schedule and results — scores, spread, weather, coaches |
| `nfl_team_records` | Final season win-loss records per franchise, 1970–present (1970–1998 needs `--pre1999`) |
| `nfl_roster` | Team roster by season and position. Each row has the player's `gsis_id`. |
| `nfl_injuries` | Player injury report status by team, week, and designation. Each row has the player's `gsis_id` and a `season_type` (REG or POST). |
| `nfl_snap_counts` | Offensive, defensive, and special teams snap counts per player |
| `nfl_fantasy_opportunity` | Target share, air yards share, carry share, and expected fantasy points per player per week (2006–present) |
| `nfl_fantasy_rankings` | Expert consensus rankings (ECR) — draft/dynasty/best-ball (`scope=draft`) or current-week start/sit (`scope=week`) |
| `nfl_ftn_charting` | Aggregated FTN charting tendencies (2022–present) over scrimmage plays — play-action, RPO, screen, no-huddle, motion, trick-play rates, plus box/pass-rush/blitz counts |
| `nfl_td_luck` | Actual vs expected touchdowns per player, season, and season type. It surfaces TD-regression candidates (most "unlucky" first). Each row has a `season_type` (REG or POST). Pass `season_type` to get one. Without it, both come back as separate rows. |
| `nfl_role_trend` | Rolling 3-week snap / target / carry / air-yards share with current-week delta — usage trending up or down |
| `nfl_separation_opportunity` | Next Gen Stats separation/YAC joined to fantasy opportunity (2016+) — flags receivers getting open but under-producing |
| `nfl_drop_rate` | Catchable-target drop rate per receiver, team, season, and season type from FTN charting (2022+), plus contested targets and created receptions. Each row has a `season_type` (REG or POST). Pass `season_type` to get one. Without it, both come back as separate rows. |
| `nfl_contract_value` | Fantasy points per $M of average per year (APY) — best value-for-money players |
| `nfl_injury_return` | Post-return snap-share recovery (% of pre-injury baseline) at +1..+8 weeks, by normalized injury type and position |
| `nfl_catalog` | Every table with its grain (what one row is), live row count, first and last season, join keys, derived flag, and last load time |

### What every response carries

- **`data_as_of`.** Every tool response includes `data_as_of`, an ISO 8601 UTC time. It is
  the newest load time of the tables that tool read. `nfl_schedule` reports the load time of
  `schedules` only. The server reads these times once per database file and caches them. A
  swap to a new database file, or a new load into the same file, refreshes the value. Error
  responses carry it too.
- **Structured output.** Each result is a JSON object. The MCP response sends it in
  `structuredContent` and as text. No tool declares an `outputSchema`. The mcpo proxy
  removes every result field that an `outputSchema` does not list.
- **Resources.** Clients that read MCP resources can read `nfl://schema`,
  `nfl://relationships`, and `nfl://data-notes`. They can also read `nfl://schema/<table>`,
  for example `nfl://schema/plays`. These hold the same facts as `nfl_schema`.

## Key columns in `plays`

- `epa` — expected points added (the best single-play quality metric)
- `wpa` — win probability added
- `posteam` / `defteam` — offensive/defensive team abbreviations
- `passer_player_name` / `rusher_player_name` / `receiver_player_name`
- `play_type` — `'pass'` | `'run'` | `'field_goal'` | `'punt'` | `'kickoff'` | ...
- `desc` — raw play description (use `ILIKE` for text search)

## Local Development

```bash
git clone https://github.com/ebhattad/nfl-mcp
cd nfl-mcp
pip install -e ".[dev]"

nfl-mcp ingest --dataset all --start 2024 --end 2024
nfl-mcp serve   # server available at http://localhost:8000/mcp
pytest
pytest -m unit     # unit tests
pytest -m integration  # integration tests (requires loaded DB)
```

Two dependencies have an upper bound in `pyproject.toml`. `duckdb` stays below 2, because a
new major version can change SQL behavior and the file format. `nflreadpy` stays below 0.2,
because a 0.x minor release can rename a loader or change its arguments. Each ingest records
the installed `nflreadpy` version in the `nflreadpy_version` column of `_ingest_metadata`.
Derived tables and the ESPN pull have no version, so that column is empty for them. A
database built before this column existed gets it at the next ingest.

## Troubleshooting

- `nfl-mcp doctor` is the fastest way to verify config, database, and client setup.
- If tools return database errors, run `nfl-mcp ingest` to ensure data is loaded.
- You can override the DB location with `NFL_MCP_DB_PATH=/path/to/nflread.duckdb`.
- Re-running `nfl-mcp ingest` is safe — it skips completed seasons and re-fetches only the current season and static tables.
- Stats for a game usually appear in nflverse within a few hours; play-by-play is rebuilt overnight and official stat corrections land midweek. Locally, run `nfl-mcp ingest` after that; in the container, the scheduled updater does it (log: `/var/log/nflmcp/nfl_update.log`).
- Every tool response has `data_as_of`. `nfl_status` shows the newest season, newest week, and last refresh time of each dataset.
- If `nfl-mcp serve` logs a warning that the database needs an upgrade, run `nfl-mcp upgrade`. Use `--check` first to see what it would do.
- In the container, `nfl_status`'s `last_refreshed` shows which data the endpoint is serving. If it never advances, the updater is not running — check that `ENTRYPOINT` is `/entrypoint.sh` and that `/data/nflread.duckdb` is a symlink.

## License

MIT
