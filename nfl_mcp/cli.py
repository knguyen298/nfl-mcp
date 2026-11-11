"""
NFL MCP — command-line entry point

  nfl-mcp init                Interactive setup wizard
  nfl-mcp serve               Start the MCP server (Streamable HTTP)
  nfl-mcp ingest              Load NFL play-by-play data into the database
  nfl-mcp rebuild             Rebuild the derived tables from the loaded data
  nfl-mcp upgrade             Detect and upgrade a database made by an older nfl-mcp
  nfl-mcp validate            Check a database slot before it replaces the active one
  nfl-mcp setup-client        Configure Claude Desktop / VS Code MCP
  nfl-mcp doctor              Check that everything is working
"""

import json
import logging
import os
import sys
from pathlib import Path

import click
import uvicorn

from .config import write_json
from .server import create_app
from .seasons import FIRST_SEASON, EARLIEST_SEASON, current_season

CURRENT_SEASON = current_season()

# Said out loud when a run fetched and every fetch failed. The database is untouched in that
# case, and a caller (the container's scheduled refresh, a CI job) must not read exit code 0
# as "the data got newer".
NOTHING_LOADED_MESSAGE = "Nothing could be loaded. The database was left as it was."

# The address the server binds and the address a client is told to connect to. Both are
# resolved through _serve_settings() from these constants, so `serve` and `setup-client`
# cannot disagree, and so `http://localhost:8000/mcp` is spelled in one place.
DEFAULT_HOST = "127.0.0.1"  # bind address for `nfl-mcp serve`
DEFAULT_PORT = 8000
DEFAULT_CONNECT_HOST = "localhost"  # what a person types to reach DEFAULT_HOST or 0.0.0.0
MCP_PATH = "/mcp"  # streamable-HTTP route the server answers on

# The command that ends a "no database yet" state, written the same way in every message.
INGEST_COMMAND = "nfl-mcp ingest"


def _run_problems(result) -> tuple[bool, bool]:
    """Return (nothing_loaded, any_failed) for what a run reported.

    Both are False for None, which is what a run that was stubbed out returns: nothing ran,
    so nothing failed. Callers then do not have to know how the ingest was wired up.
    """
    if result is None:
        return False, False
    return result.nothing_loaded, result.any_failed

# `nfl-mcp ingest` exit code when the run finished but some datasets or seasons failed.
# A fatal error exits 1 and a usage error 2 (click).
INGEST_PARTIAL_EXIT = 3


def _season_window_label(min_season: int | None, max_season: int | None,
                         newest_season: str | None = None) -> str:
    """Render a dataset's coverage window for --list, e.g. '1999-present'.

    A newest_season rule that stops before the roster year (participation) never reaches
    the season in progress, so its window ends at the last completed season.
    """
    from .registry import NEWEST_SEASON_RULES
    start = str(min_season) if min_season is not None else "earliest"
    if max_season is not None:
        end = str(max_season)
    elif newest_season is not None and NEWEST_SEASON_RULES[newest_season][1] < 0:
        end = "last completed season"
    else:
        end = "present"
    return f"{start}–{end}"


def _ingest_command(start: int, end: int) -> str:
    """The exact command that loads what this run was asked to load."""
    return f"{INGEST_COMMAND} --start {start} --end {end}"


def _as_port(value: object) -> int | None:
    """A usable port number from a config file, or None when the value is not one.

    config.json is a file a person edits by hand, so a typo there must not crash a command.
    """
    try:
        port = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return port if 0 < port <= 65535 else None


def _serve_settings(config: dict, host: str | None = None,
                    port: int | None = None) -> tuple[str, int]:
    """Resolve the host and port to use, in the one precedence order that exists.

    Command line option first, then the configuration file's `serve_host` / `serve_port`
    key, then DEFAULT_HOST / DEFAULT_PORT. No environment variable sets either one today.
    `serve` passes its options; `setup-client` passes None and gets the same answer the
    server will give, which is what keeps a client URL pointing at a live address.
    """
    resolved_host = str(host or config.get("serve_host") or DEFAULT_HOST)
    resolved_port = port or _as_port(config.get("serve_port")) or DEFAULT_PORT
    return resolved_host, resolved_port


def _client_url(host: str, port: int) -> str:
    """The URL a client connects to for a server bound to this address."""
    # 0.0.0.0 and 127.0.0.1 are bind addresses, not something a client types.
    display_host = DEFAULT_CONNECT_HOST if host in ("0.0.0.0", DEFAULT_HOST) else host
    return f"http://{display_host}:{port}{MCP_PATH}"


@click.group()
def main():
    """NFL play-by-play MCP server."""


def _check_auth_token_env() -> None:
    """Exit 1 when NFL_MCP_AUTH_TOKEN is set but empty after strip: the operator meant to require auth."""
    raw = os.environ.get("NFL_MCP_AUTH_TOKEN")
    if raw and not raw.strip():
        raise click.ClickException(
            "NFL_MCP_AUTH_TOKEN is set but empty. Set a token or unset the variable."
        )


# ── serve ──────────────────────────────────────────────────────────────────────

def _exit_on_config_problem():
    """Exit non-zero, naming the env var, if NFL_MCP_DUCKDB_* is invalid."""
    from .database import connection_config_problem

    problem = connection_config_problem()
    if problem:
        click.secho(f"✗ {problem}", fg="red", err=True)
        sys.exit(1)


def _warn_if_schema_old() -> None:
    """Log one warning when the database file needs an upgrade or is newer. Never exits."""
    from . import database, migrate
    from .config import get_duckdb_path

    log = logging.getLogger("nfl-mcp")
    path = get_duckdb_path()
    if not path.is_file():
        return
    try:
        with database.get_db_connection() as conn:
            status = migrate.check(conn)
    except Exception as e:  # noqa: BLE001 - the server starts either way
        log.warning("Could not check the schema version of %s: %s", path, e)
        return
    if status.newer:
        log.warning("Database %s has schema version %d, newer than this nfl-mcp (%d); "
                    "update nfl-mcp", path, status.version, status.target)
    elif status.upgrade_needed:
        needed = [s.name for s in status.steps if s.needed]
        log.warning("Database %s is at schema version %d, this nfl-mcp expects %d%s; "
                    "run: nfl-mcp upgrade", path, status.version, status.target,
                    f" (needs: {', '.join(needed)})" if needed else "")


@main.command()
@click.option("--host", default=DEFAULT_HOST, show_default=True,
              help="Host to bind the HTTP server to. Use 0.0.0.0 for LAN access. Omit to "
                   "use the 'serve_host' key in config.json, or the default when that key "
                   "is absent.")
@click.option("--port", default=None, type=int,
              help=f"Port to listen on (default {DEFAULT_PORT}). Omit to use the "
                   "'serve_port' key in config.json, or the default when it is absent.")
@click.pass_context
def serve(ctx, host, port):
    """Start the MCP server over Streamable HTTP."""
    from .config import load_config

    # The help shows the default, but an omitted --host still lets the config key decide.
    if ctx.get_parameter_source("host") is click.core.ParameterSource.DEFAULT:
        host = None
    _exit_on_config_problem()
    _check_auth_token_env()
    _warn_if_schema_old()
    host, port = _serve_settings(load_config(), host=host, port=port)
    url = _client_url(host, port)
    click.echo(f"🏈 NFL MCP server listening on {url}")
    if host == "0.0.0.0":
        click.echo(f"   (bound to all interfaces — clients should connect via {url})")
    uvicorn.run(create_app(), host=host, port=port)


# ── ingest ─────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--dataset", "datasets", multiple=True, default=["default"],
              metavar="NAME",
              help=(
                  "Dataset(s) to load. Pass multiple times or use 'all' / 'default'. "
                  "Run with --list to see all available names."
              ))
@click.option("--start",      default=None, type=click.IntRange(EARLIEST_SEASON, CURRENT_SEASON),
              help=f"First season to load (default {FIRST_SEASON}, the oldest available being "
                   f"{EARLIEST_SEASON} for rosters). Omit to load every season each dataset "
                   "supports within that bound — each dataset is clipped to its own coverage window.")
@click.option("--end",        default=None, type=click.IntRange(EARLIEST_SEASON, CURRENT_SEASON),
              help="Last season to load (inclusive). Omit to load all available seasons.")
@click.option("--fresh",      is_flag=True,
              help="Re-ingest even if dataset+season is already recorded as loaded.")
@click.option("--skip-views", is_flag=True,
              help="Skip rebuilding PBP aggregate tables.")
@click.option("--list", "list_datasets", is_flag=True,
              help="Print all available dataset names and exit.")
@click.option("--pre1999", is_flag=True,
              help="Also pull pre-1999 season win-loss records from ESPN (opt-in external source).")
@click.option("--allow-failures", is_flag=True,
              help="Exit 0 even if a dataset or season failed to load (the failures are still "
                   "listed). For image builds and first-start loads that should not stop on "
                   "one flaky fetch.")
@click.option("--failures-file", default=None, type=click.Path(dir_okay=False),
              help="Write each dataset (or dataset and season) that failed to load to this "
                   "file, one per line. The file is empty when nothing failed. update_db.sh "
                   "uses it to name the failures in its log.")
@click.option("--strict", is_flag=True,
              help="Exit 3 when any fetch failed, even with --allow-failures. Without "
                   "--allow-failures this is already the default.")
def ingest(datasets, start, end, fresh, skip_views, list_datasets, pre1999, allow_failures,
           failures_file, strict):
    """Download NFL data via nflreadpy and load into DuckDB.

    Some failures are ordinary: play-by-play for a new season does not exist before
    Week 1. A run in which no fetch put rows in the database prints that the database
    was left as it was, and exits 3 even with --allow-failures.

    \b
    Examples:
      nfl-mcp ingest                          # default datasets, all seasons
      nfl-mcp ingest --dataset all            # every dataset, all seasons
      nfl-mcp ingest --dataset schedules      # one dataset, all seasons
      nfl-mcp ingest --dataset pbp --start 2020 --end 2024
      nfl-mcp ingest --pre1999                # defaults + ESPN 1970-1998 records
      nfl-mcp ingest --list                   # show all dataset names

    Exit codes:

    \b
      0  Every requested dataset loaded, or --allow-failures was given
         without --strict. A season the publisher has no rows for is not a
         failure.
      3  The run finished, but some datasets or seasons failed to load. Each
         failed one keeps the rows it had before. The failures are listed.
      1  The run stopped with an error, for example when no play-by-play
         season could be fetched or the database could not be opened.
      2  A usage error, for example an unknown dataset name.
    """
    from .registry import REGISTRY, DEFAULT_DATASETS, ALL_DATASETS

    if list_datasets:
        click.echo("\nAvailable datasets:\n")
        for ds_id, defn in REGISTRY.items():
            default_tag = "[default]" if defn.default else "[opt-in]"
            click.echo(f"  {ds_id:<35} {default_tag:<10}  {defn.description}")
            if defn.seasonal:
                label = _season_window_label(defn.min_season, defn.max_season, defn.newest_season)
                if defn.min_season is not None and defn.min_season < FIRST_SEASON:
                    label += f" (from {FIRST_SEASON} by default)"
                click.echo(f"  {'':<35} {'':<10}  → seasons {label}")
        click.echo()
        return

    if start is not None and end is not None and start > end:
        raise click.UsageError("--start must be less than or equal to --end.")

    # Resolve dataset aliases
    resolved: list[str] = []
    for name in datasets:
        if name == "all":
            resolved = ALL_DATASETS
            break
        elif name == "default":
            resolved.extend(DEFAULT_DATASETS)
        elif name in REGISTRY:
            resolved.append(name)
        else:
            raise click.UsageError(
                f"Unknown dataset '{name}'. Run 'nfl-mcp ingest --list' to see options."
            )
    # Deduplicate, preserve order
    seen: set[str] = set()
    dataset_ids = [d for d in resolved if not (d in seen or seen.add(d))]

    # The ESPN records pull is opt-in, so it is never part of 'default' (and therefore
    # never part of the container's scheduled refresh) unless asked for explicitly.
    if pre1999 and "espn_records" not in dataset_ids:
        dataset_ids.append("espn_records")

    from .ingest import IngestRunResult, failure_labels, run_ingest_datasets
    result = run_ingest_datasets(
        dataset_ids=dataset_ids,
        start=start,
        end=end,
        fresh=fresh,
        skip_views=skip_views,
    )
    if result is None:
        # A stubbed-out run returns None: nothing ran, so nothing failed.
        result = IngestRunResult()
    failures = result.failures
    if failures_file:
        labels = failure_labels(failures)
        Path(failures_file).write_text("".join(f"{label}\n" for label in labels))
    message = "Ingest failed for: " + "; ".join(failures)
    if result.nothing_loaded:
        # The run fetched, and no fetch put rows in the database: a dead network, or a
        # source that answered nothing. The database is no newer than before, so not even
        # --allow-failures makes this a success. entrypoint.sh stops on it rather than
        # renaming an empty first-start database into place.
        if failures:
            click.secho(f"Error: {message}", fg="red", err=True)
        click.secho(NOTHING_LOADED_MESSAGE, fg="red", err=True)
        sys.exit(INGEST_PARTIAL_EXIT)
    if failures:
        if allow_failures and not strict:
            click.echo(f"{message} (ignored: --allow-failures)")
            return
        # A distinct code: the run finished and the rest loaded. update_db.sh goes on to
        # upgrade and validate on this code and stops on any other non-zero one.
        click.secho(f"Error: {message}", fg="red", err=True)
        sys.exit(INGEST_PARTIAL_EXIT)


# ── rebuild ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--db", "db_path", default=None, type=click.Path(dir_okay=False),
              help="Database file to rebuild (default: the configured path / NFL_MCP_DB_PATH).")
def rebuild(db_path):
    """Rebuild every derived table (team, situational, fantasy, records) in one transaction.

    No data is downloaded. If any table fails the previous tables are kept and the
    command exits with an error.
    """
    import duckdb

    from .config import get_duckdb_path
    from .ingest import _apply_duckdb_pragmas, _ensure_metadata_table, rebuild_derived_tables

    path = Path(db_path).expanduser() if db_path else get_duckdb_path()
    if not path.exists():
        raise click.ClickException(f"Database not found at {path}. Run 'nfl-mcp ingest' first.")

    conn = duckdb.connect(str(path))
    try:
        _apply_duckdb_pragmas(conn, str(path))
        _ensure_metadata_table(conn)  # the builders write the status and detail columns
        rebuild_derived_tables(conn)
    except duckdb.Error as e:
        raise click.ClickException(f"Rebuild failed, previous derived tables kept: {e}")
    finally:
        conn.close()
    click.secho("Derived tables rebuilt.", fg="green")


# ── upgrade ────────────────────────────────────────────────────────────────────

def _step_line(step) -> str:
    state = "ok" if not step.needed else "NEEDED (network)" if step.needs_network else "NEEDED"
    return f"  {step.name:<22} {state}" + (f"  {step.detail}" if step.needed else "")


@main.command()
@click.option("--db", "db_path", default=None, type=click.Path(dir_okay=False),
              help="Database file (default: the configured path / NFL_MCP_DB_PATH).")
@click.option("--check", "check_only", is_flag=True,
              help="Only report what is needed (read-only). Exit 0 current, 3 upgrade needed, "
                   "1 error or a database newer than this nfl-mcp.")
@click.option("--no-fetch", is_flag=True,
              help="Skip the steps that download data; exit 3 if any are left.")
def upgrade(db_path, check_only, no_fetch):
    """Detect and upgrade a database made by an older nfl-mcp, without deleting it.

    Runs only the steps the database needs, then records the schema version once none
    is left. Running it again is safe. The served container slot is read-only on purpose:
    update_db.sh upgrades a copy of it before the swap.
    """
    import duckdb

    from . import migrate
    from .config import get_duckdb_path
    from .database import _connection_config
    from .ingest import _apply_duckdb_pragmas

    path = Path(db_path).expanduser() if db_path else get_duckdb_path()
    if not path.is_file():
        raise click.ClickException(f"Database not found at {path}. Run 'nfl-mcp ingest' first.")

    if check_only:
        try:
            conn = duckdb.connect(str(path), read_only=True, config=_connection_config())
            try:
                status = migrate.check(conn)
            finally:
                conn.close()
        except (duckdb.Error, ValueError) as e:
            raise click.ClickException(f"Cannot check {path}: {e}")
        click.echo(f"Schema version {status.version}, this nfl-mcp expects {status.target}")
        for step in status.steps:
            click.echo(_step_line(step))
        if status.newer:
            raise click.ClickException(
                f"{path} is newer than this nfl-mcp (version {status.version} > {status.target}); "
                "update nfl-mcp")
        if status.upgrade_needed:
            click.echo("Upgrade needed: run nfl-mcp upgrade")
            sys.exit(3)
        click.secho("Database is current.", fg="green")
        return

    # os.access is always true for root, so also look at the mode bits of the real file.
    if not os.access(path, os.W_OK) or path.resolve().stat().st_mode & 0o222 == 0:
        raise click.ClickException(
            f"{path} is read-only. The active container slot is read-only on purpose; "
            "the next update_db.sh run upgrades a copy of it.")
    try:
        conn = duckdb.connect(str(path))
    except duckdb.Error as e:
        raise click.ClickException(f"Cannot open {path} for writing: {e}")
    try:
        _apply_duckdb_pragmas(conn, str(path))
        result = migrate.upgrade(conn, fetch=not no_fetch)
    except duckdb.Error as e:
        raise click.ClickException(f"Upgrade of {path} failed: {e}")
    finally:
        conn.close()

    click.echo(f"Schema version {result.before.version} -> {result.after.version}, "
               f"this nfl-mcp expects {result.after.target}")
    for step in [] if result.before.newer else result.before.steps:
        if step.name in result.ran:
            state = "ran"
        elif step.name in result.pending:
            state = "skipped (--no-fetch)"
        elif step.needed:
            state = "FAILED"
        else:
            state = "ok (not needed)"
        click.echo(f"  {step.name:<22} {state}")
    for failure in result.failed:
        click.secho(f"  FAILED {failure}", fg="red")
    if result.failed:
        sys.exit(1)
    if result.stamped:
        click.secho(f"Stamped schema version {result.after.version}.", fg="green")
    if result.after.upgrade_needed:
        click.echo("Not stamped: run nfl-mcp upgrade without --no-fetch to finish.")
        sys.exit(3)


# ── validate ───────────────────────────────────────────────────────────────────

@main.command()
@click.option("--db", "db_path", required=True, type=click.Path(dir_okay=False),
              help="Database file to check (the freshly ingested slot).")
@click.option("--current", "current_path", default=None, type=click.Path(dir_okay=False),
              help="The active slot. The new file must not hold fewer rows or an older "
                   "newest week than this one.")
def validate(db_path, current_path):
    """Check a database file before it is swapped in. Exit 1 if any check fails.

    Both files are opened read-only. Checks: required tables and columns, row counts
    against the current slot, duplicate plays, newest week, and key columns in the
    newest week.
    """
    from .validate import run_checks

    results = run_checks(Path(db_path).expanduser(),
                         Path(current_path).expanduser() if current_path else None)
    for r in results:
        click.secho(f"{'ok  ' if r.ok else 'FAIL'} {r.name}: {r.detail}",
                    fg="green" if r.ok else "red")
    failed = [r for r in results if not r.ok]
    if failed:
        click.secho(f"Validation failed: {len(failed)} of {len(results)} checks.", fg="red")
        sys.exit(1)
    click.secho(f"Validation passed: {len(results)} checks.", fg="green")


# ── init ───────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--start", default=FIRST_SEASON, show_default=True,
              type=click.IntRange(EARLIEST_SEASON, CURRENT_SEASON),
              help="First season to load.")
@click.option("--end", default=CURRENT_SEASON, show_default=True,
              type=click.IntRange(EARLIEST_SEASON, CURRENT_SEASON),
              help="Last season to load.")
@click.option("--host", default=None,
              help=f"Host for the MCP server (default {DEFAULT_HOST}). Saved to "
                   "config.json when it is not the default, so 'nfl-mcp serve' binds "
                   "the same address.")
@click.option("--port", default=None, type=int,
              help=f"Port for the MCP server (default {DEFAULT_PORT}). Saved to "
                   "config.json when it is not the default, so 'nfl-mcp serve' binds "
                   "the same port.")
@click.option("--skip-ingest", is_flag=True,
              help="Skip data ingestion (configure only).")
def init(start, end, host, port, skip_ingest):
    """Interactive setup wizard — configure database, load data, set up your IDE."""
    if start > end:
        raise click.UsageError("--start must be less than or equal to --end.")
    from .config import save_config, load_config, DEFAULT_DUCKDB_PATH

    click.echo()
    click.secho("🏈 NFL MCP Setup", bold=True)
    click.echo("=" * 50)

    # ── 1. Configure database path ─────────────────────────────────────────
    config = load_config()
    # A saved path stays the default, so pressing Enter keeps it instead of resetting it.
    db_path = config.get("duckdb_path") or str(DEFAULT_DUCKDB_PATH)
    custom = click.confirm(
        f"  Database will be stored at {db_path}. Change location?", default=False
    )
    if custom:
        db_path = click.prompt("  Path", default=db_path)
    config["duckdb_path"] = db_path
    click.secho(f"  ✓ DuckDB → {db_path}", fg="green")

    # The address this wizard starts the server on, and the address it writes into the
    # client config, both come from one call. A non-default one is saved, so the next
    # 'nfl-mcp serve' binds what the client has already been told to connect to.
    host, port = _serve_settings(config, host=host, port=port)
    if (host, port) != (DEFAULT_HOST, DEFAULT_PORT):
        config["serve_host"] = host
        config["serve_port"] = port

    # ── 2. Save config ─────────────────────────────────────────────────────
    path = save_config(config)
    click.secho(f"  ✓ Config saved to {path}", fg="green")

    # ── 3. Ingest data ─────────────────────────────────────────────────────
    if not skip_ingest:
        click.echo()
        click.echo(f"  Seasons to load: {start}–{end}")
        if not click.confirm("  Download and load NFL play-by-play data now?", default=True):
            skip_ingest = True

    # Kept out here so step 5 can say what actually happened instead of claiming success.
    nothing_loaded = False
    load_failed = False
    if not skip_ingest:
        click.echo()
        from .ingest import run_ingest_datasets
        from .registry import DEFAULT_DATASETS
        try:
            result = run_ingest_datasets(
                dataset_ids=DEFAULT_DATASETS,
                start=start,
                end=end,
                fresh=False,
                skip_views=False,
            )
            nothing_loaded = _run_problems(result)[0]
        except Exception as exc:
            # The load raised, so the config just saved points at a database that does not
            # exist, or at an empty one. Say so and name the command; keep the user's file.
            load_failed = True
            click.secho(f"  ✗ Data load stopped: {exc}", fg="red")
        else:
            # Some seasons landed and some did not: the database works, so warn and go on.
            failures = [] if result is None else result.failures
            if failures and not nothing_loaded:
                click.secho("  Warning: some data failed to load: " + "; ".join(failures)
                            + ". Run 'nfl-mcp ingest' again to retry.", fg="yellow")

    # ── 4. Set up IDE client ───────────────────────────────────────────────
    click.echo()
    client_ok = True
    if click.confirm("  Configure an MCP client (Claude Desktop / VS Code)?", default=True):
        client_ok = bool(_setup_client_interactive(config))

    # ── 5. Offer to start the server ───────────────────────────────────────
    click.echo()
    if nothing_loaded or load_failed:
        click.secho(NOTHING_LOADED_MESSAGE, bold=True, fg="red")
        click.echo(f"   Load data later with:  {_ingest_command(start, end)}")
    elif client_ok:
        click.secho("🎉 Setup complete!", bold=True, fg="green")
    else:
        click.secho("⚠ Setup finished, but a client config was left unchanged.", bold=True, fg="yellow")
        click.echo("   Fix the file named above, then run:  nfl-mcp setup-client")
    click.echo()
    click.echo("   The MCP server needs to be running for your IDE to connect.")
    click.echo("   You can start it any time with:  nfl-mcp serve")
    click.echo()
    if click.confirm("  Start the server now?", default=True):
        _exit_on_config_problem()
        _check_auth_token_env()
        click.secho(f"   Starting server on {_client_url(host, port)}  (Ctrl-C to stop)", fg="cyan")
        click.echo()
        uvicorn.run(create_app(), host=host, port=port)
    else:
        click.echo("   Run  nfl-mcp serve  when you're ready.")
        click.echo()


# ── setup-client ───────────────────────────────────────────────────────────────

@main.command("setup-client")
@click.option("--client", type=click.Choice(["claude-desktop", "claude-code", "vscode", "auto"]),
              default="auto", show_default=True,
              help="Which client to configure.")
@click.option("--host", default=None,
              help=f"Host the server runs on (default {DEFAULT_HOST}). Saved to "
                   "config.json when it is not the default, so 'nfl-mcp serve' binds "
                   "the same address.")
@click.option("--port", default=None, type=int,
              help=f"Port the server runs on (default {DEFAULT_PORT}). Saved to "
                   "config.json when it is not the default, so 'nfl-mcp serve' binds "
                   "the same port.")
def setup_client(client, host, port):
    """Auto-configure Claude Desktop, Claude Code, or VS Code to use the NFL MCP server."""
    from .config import load_config, save_config
    config = load_config()
    host, port = _serve_settings(config, host=host, port=port)
    if (host, port) != (DEFAULT_HOST, DEFAULT_PORT):
        # Write down the address the client is about to be given, so the next 'nfl-mcp
        # serve' binds it instead of the built-in default and the URL keeps working.
        config["serve_host"] = host
        config["serve_port"] = port
        save_config(config)
    if client == "auto":
        configured = _setup_client_interactive(config)
    elif client == "claude-desktop":
        configured = _configure_claude_desktop(config)
    elif client == "claude-code":
        configured = _configure_claude_code(config)
    elif client == "vscode":
        configured = _configure_vscode(config)
    if not configured:
        raise click.ClickException(
            "Could not configure the selected client — a config file was left unchanged."
        )


def _setup_client_interactive(config: dict) -> bool:
    """Prompt user to pick which clients to configure.

    Returns False when a chosen client's config file could not be updated.
    """
    clients = []
    claude_path = _claude_desktop_config_path()
    if claude_path:
        clients.append(("Claude Desktop", "claude-desktop"))
    clients.append(("Claude Code (global ~/.claude/mcp.json)", "claude-code"))
    clients.append(("VS Code (.vscode/mcp.json in current directory)", "vscode"))

    all_configured = True
    for name, key in clients:
        if click.confirm(f"    Configure {name}?", default=True):
            if key == "claude-desktop":
                configured = _configure_claude_desktop(config)
            elif key == "claude-code":
                configured = _configure_claude_code(config)
            else:
                configured = _configure_vscode(config)
            if not configured:
                all_configured = False
    return all_configured


def _build_server_config(config: dict) -> dict:
    """Build the MCP server JSON block for a client config file.

    The address comes from _serve_settings(), the same call `serve` makes, so the URL a
    client is handed is the address the server binds.
    """
    host, port = _serve_settings(config)
    return {"url": _client_url(host, port)}


def _claude_desktop_config_path() -> Path | None:
    """Return Claude Desktop config path if it exists."""
    if sys.platform == "darwin":
        p = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif sys.platform == "win32":
        p = Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
    else:
        p = Path.home() / ".config" / "claude" / "claude_desktop_config.json"
    return p if p.parent.exists() else None


def _load_client_config(path: Path, servers_key: str) -> dict | None:
    """Read a client config file so the nfl entry can be merged into it.

    Returns {} when the file does not exist yet, the parsed object when the entry
    can be merged into it, and None when the file must be left alone: it is not
    valid JSON, its top level is not an object, or its servers container exists
    but is not an object. Callers treat None as a failure and never write.
    """
    if not path.exists():
        return {}
    try:
        existing = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        problem = "it is not valid JSON"
    else:
        if not isinstance(existing, dict):
            problem = "it is not a JSON object"
        elif servers_key in existing and not isinstance(existing[servers_key], dict):
            problem = f"its '{servers_key}' container is not a JSON object"
        else:
            return existing
    click.secho(f"    ✗ Left {path} unchanged — {problem}.", fg="red")
    click.echo("      Fix or remove that file, then run 'nfl-mcp setup-client' again.")
    return None


def _configure_claude_desktop(config: dict) -> bool:
    """Write or merge NFL MCP server into Claude Desktop config.

    Returns False when an existing file was left untouched.
    """
    path = _claude_desktop_config_path()
    if not path:
        click.secho("    ✗ Claude Desktop config directory not found", fg="yellow")
        return True

    existing = _load_client_config(path, "mcpServers")
    if existing is None:
        return False

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["nfl"] = _build_server_config(config)

    write_json(path, existing)
    click.secho(f"    ✓ Claude Desktop configured → {path}", fg="green")
    click.echo("      Restart Claude Desktop to pick up changes.")
    return True


def _configure_vscode(config: dict) -> bool:
    """Write or merge NFL MCP server into .vscode/mcp.json.

    Returns False when an existing file was left untouched.
    """
    vscode_dir = Path.cwd() / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    path = vscode_dir / "mcp.json"

    existing = _load_client_config(path, "servers")
    if existing is None:
        return False

    existing.setdefault("servers", {})
    existing["servers"]["nfl"] = _build_server_config(config)

    write_json(path, existing)
    click.secho(f"    ✓ VS Code configured → {path}", fg="green")
    return True


def _configure_claude_code(config: dict) -> bool:
    """Write or merge NFL MCP server into ~/.claude/mcp.json (Claude Code global config).

    Returns False when an existing file was left untouched.
    """
    path = Path.home() / ".claude" / "mcp.json"
    path.parent.mkdir(exist_ok=True)

    existing = _load_client_config(path, "mcpServers")
    if existing is None:
        return False

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["nfl"] = _build_server_config(config)

    write_json(path, existing)
    click.secho(f"    ✓ Claude Code configured → {path}", fg="green")
    click.echo("      Restart Claude Code to pick up changes.")
    return True


# ── doctor ─────────────────────────────────────────────────────────────────────

def _doctor_schema_line() -> None:
    """One line: the database schema version, the target and the steps still needed."""
    from . import database, migrate

    try:
        with database.get_db_connection() as conn:
            status = migrate.check(conn)
    except Exception:  # noqa: BLE001 - doctor reports, it does not stop
        click.secho("  – Schema version unknown (could not check)", fg="bright_black")
        return
    needed = [s.name for s in status.steps if s.needed]
    head = f"Schema version {status.version}, this nfl-mcp expects {status.target}"
    if status.newer:
        click.secho(f"  ⚠ {head}: the database is newer, update nfl-mcp", fg="yellow")
    elif status.upgrade_needed:
        click.secho(f"  ⚠ {head}: needs {', '.join(needed) or 'a version stamp'}; "
                    "run 'nfl-mcp upgrade'", fg="yellow")
    else:
        click.secho(f"  ✓ {head} (current)", fg="green")


@main.command()
def doctor():
    """Check that everything is configured and working."""
    from . import database
    from .config import config_exists, config_file_is_usable, get_duckdb_path

    click.echo()
    click.secho("🏈 NFL MCP Doctor", bold=True)
    click.echo("=" * 50)
    all_ok = True

    # 1. Config — an explicit NFL_MCP_DB_PATH stands in for a config file (containers, CI)
    if config_exists():
        if config_file_is_usable():
            click.secho("  ✓ Config file found", fg="green")
        else:
            # Left on disk on purpose, so the user's own edits are not lost. Say what it
            # costs: every path in this file is being ignored, not applied.
            click.secho(
                "  ✗ Config file exists but is not a JSON object — its settings are ignored",
                fg="red",
            )
            click.echo("    Run 'nfl-mcp init' to write a new one; nothing was moved or rewritten.")
            all_ok = False
    elif os.getenv("NFL_MCP_DB_PATH"):
        click.secho("  – No config file — the database path comes from NFL_MCP_DB_PATH", fg="yellow")
    else:
        click.secho("  ✗ No config file — run 'nfl-mcp init' first", fg="red")
        click.echo()
        sys.exit(1)

    # 2. Database — "no file" and "file with nothing loaded yet" are two different fixes,
    #    so they are two different checks. Both name the command that ends the state.
    db_path = get_duckdb_path()
    if db_path.exists():
        click.secho(f"  ✓ DuckDB file exists ({db_path})", fg="green")
        import duckdb
        try:
            with database.get_db_connection() as conn:
                count = conn.execute("SELECT COUNT(*) FROM plays").fetchone()[0]
                seasons = conn.execute(
                    "SELECT MIN(season), MAX(season) FROM plays"
                ).fetchone()
        except duckdb.CatalogException:
            # An empty database file: created, never loaded, so it has no plays table.
            click.secho(f"  ✗ Database has no data yet — no plays table in {db_path}", fg="red")
            click.echo(f"    Run '{INGEST_COMMAND}' to load data.")
            all_ok = False
        except Exception as e:
            click.secho(f"  ✗ Database error: {e}", fg="red")
            all_ok = False
        else:
            if count:
                click.secho(f"  ✓ {count:,} plays loaded (seasons {seasons[0]}–{seasons[1]})", fg="green")
            else:
                click.secho(f"  ✗ Database has no data yet — 0 plays in {db_path}", fg="red")
                click.echo(f"    Run '{INGEST_COMMAND}' to load data.")
                all_ok = False
        _doctor_schema_line()
    else:
        click.secho(f"  ✗ DuckDB file not found at {db_path}", fg="red")
        click.echo(f"    Run '{INGEST_COMMAND}' to create it.")
        all_ok = False

    # 3. Claude Desktop config
    claude_path = _claude_desktop_config_path()
    if claude_path and claude_path.exists():
        try:
            cd_config = json.loads(claude_path.read_text())
            if "nfl" in cd_config.get("mcpServers", {}):
                click.secho("  ✓ Claude Desktop configured", fg="green")
            else:
                click.secho("  ⚠ Claude Desktop config exists but 'nfl' server not found", fg="yellow")
        except Exception:
            click.secho("  ⚠ Claude Desktop config exists but is invalid JSON", fg="yellow")
    else:
        click.secho("  – Claude Desktop not detected (skipped)", fg="bright_black")

    # 4. VS Code config
    vscode_path = Path.cwd() / ".vscode" / "mcp.json"
    if vscode_path.exists():
        try:
            vs_config = json.loads(vscode_path.read_text())
            if "nfl" in vs_config.get("servers", {}):
                click.secho("  ✓ VS Code MCP configured", fg="green")
            else:
                click.secho("  ⚠ .vscode/mcp.json exists but 'nfl' server not found", fg="yellow")
        except Exception:
            click.secho("  ⚠ .vscode/mcp.json exists but is invalid JSON", fg="yellow")
    else:
        click.secho("  – VS Code MCP config not found (run 'nfl-mcp setup-client')", fg="bright_black")

    # Summary
    click.echo()
    if all_ok:
        click.secho("  All checks passed! ✅", bold=True, fg="green")
    else:
        click.secho("  Some checks failed — see above for fixes.", bold=True, fg="red")
    click.echo()
    if not all_ok:
        sys.exit(1)
