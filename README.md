# nfl-mcp

MCP server for NFL data (default backfill to 1970; play-by-play from 1999, rosters to 1920 on request), powered by [nflreadpy](https://github.com/nflverse/nflreadpy) and DuckDB.
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
pip install nfl-mcp        # or: uvx nfl-mcp
nfl-mcp init               # configure, load data, and start the server
```

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
nfl-mcp ingest --list                   # show all available dataset names
```

Ingest is **idempotent** — re-running skips completed seasons already in the database. The current season and the static snapshot tables (rankings, players, contracts, teams) are re-fetched on every run, so run `nfl-mcp ingest` after a game week to pick up new stats. The current season is computed from today's date (the league year rolls over in March), so a new season is picked up automatically.

## Datasets

All data is ingested over HTTPS at ingest time (never at query time) from three upstream
projects, via [nflreadpy](https://github.com/nflverse/nflreadpy), and stored locally in
DuckDB:

| Upstream | What it supplies |
|---|---|
| [nflverse-data](https://github.com/nflverse/nflverse-data) | play-by-play, schedules, rosters, stats, injuries, snap counts, depth charts, officials, participation, Next Gen Stats, PFR advanced stats, FTN charting, teams, players, contracts, trades, draft picks, combine |
| [DynastyProcess](https://github.com/dynastyprocess/data) | fantasy player IDs and draft/weekly rankings |
| [ffverse](https://github.com/ffverse/ffopportunity) | fantasy opportunity model |

**Every dataset below is ingested by default** — `nfl-mcp ingest` loads the full family so
any data a client might need is already there.

> **Season coverage: 1970 onward by default, deeper on request.** The default backfill
> starts at **1970** and each dataset is then clipped to its own first available season (the
> `min_season` in `nfl_mcp/registry.py`), so a fresh install loads rosters from 1970 while
> play-by-play and the stats tables start at their 1999 source floor and Next Gen Stats at
> 2016. Pass `--start` to go deeper — `--start 1920` adds the pre-merger rosters, and no
> dataset ever requests a season its source never published. `nfl-mcp ingest --list` prints
> the window for every dataset. Non-seasonal reference tables (draft picks, combine,
> contracts, players) carry their full historical record.

| Table | Loaded range |
|---|---|
| `rosters` | 1970–present (1920 with `--start 1920`) |
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

> Older seasons are thinner than modern ones. Play-by-play carries `epa`, `wp`, and
> `shotgun` from 1999, but tracking-derived data (Next Gen Stats, FTN charting), snap
> counts, and weekly depth only exist from their respective start years above — so any
> query that spans decades mixes eras of very different column coverage. Pre-1970 rosters
> are available but cover the pre-merger leagues.

```bash
nfl-mcp ingest                 # load the full nflverse family (default)
nfl-mcp ingest --list          # see all dataset names
nfl-mcp ingest --dataset pbp   # load just one dataset
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `nfl_schema` | Database schema reference — compact summary by default, pass `category` for detail |
| `nfl_status` | Database health: total plays, loaded seasons, available tables |
| `nfl_query` | Raw SQL SELECT for custom queries (500 row cap, 10s timeout) |
| `nfl_search_plays` | Find plays by player, team, season, season type, situation, touchdowns, etc. |
| `nfl_team_stats` | Pre-aggregated team offense, defense, and situational stats |
| `nfl_player_stats` | Player stats by season and season type — passing, rushing, or receiving |
| `nfl_compare` | Side-by-side comparison of two teams or two players |
| `nfl_schedule` | Game schedule and results — scores, spread, weather, coaches |
| `nfl_roster` | Team roster by season and position |
| `nfl_injuries` | Player injury report status by team, week, and designation |
| `nfl_snap_counts` | Offensive, defensive, and special teams snap counts per player |
| `nfl_fantasy_opportunity` | Target share, air yards share, carry share, and expected fantasy points per player per week (2006–present) |
| `nfl_fantasy_rankings` | Expert consensus rankings (ECR) — draft/dynasty/best-ball (`scope=draft`) or current-week start/sit (`scope=week`) |
| `nfl_ftn_charting` | Aggregated FTN charting tendencies (2022–present) over scrimmage plays — play-action, RPO, screen, no-huddle, motion, trick-play rates, plus box/pass-rush/blitz counts |
| `nfl_td_luck` | Actual vs expected touchdowns per player-season — surfaces TD-regression candidates (most "unlucky" first) |
| `nfl_role_trend` | Rolling 3-week snap / target / carry / air-yards share with current-week delta — usage trending up or down |
| `nfl_separation_opportunity` | Next Gen Stats separation/YAC joined to fantasy opportunity (2016+) — flags receivers getting open but under-producing |
| `nfl_drop_rate` | Catchable-target drop rate per receiver-season from FTN charting (2022+), plus contested targets and created receptions |
| `nfl_contract_value` | Fantasy points per $M of average per year (APY) — best value-for-money players |
| `nfl_injury_return` | Post-return snap-share recovery (% of pre-injury baseline) at +1..+8 weeks, by normalized injury type and position |
| `nfl_catalog` | List all loaded tables with row counts and last refresh time |

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

## Troubleshooting

- `nfl-mcp doctor` is the fastest way to verify config, database, and client setup.
- If tools return database errors, run `nfl-mcp ingest` to ensure data is loaded.
- You can override the DB location with `NFL_MCP_DB_PATH=/path/to/nflread.duckdb`.
- Re-running `nfl-mcp ingest` is safe — it skips completed seasons and re-fetches only the current season and static tables.
- Stats for a game usually appear in nflverse within a few hours; play-by-play is rebuilt overnight and official stat corrections land midweek. Locally, run `nfl-mcp ingest` after that; in the container, the scheduled updater does it (log: `/var/log/nflmcp/nfl_update.log`).
- In the container, `nfl_status`'s `last_refreshed` shows which data the endpoint is serving. If it never advances, the updater is not running — check that `ENTRYPOINT` is `/entrypoint.sh` and that `/data/nflread.duckdb` is a symlink.

## License

MIT
