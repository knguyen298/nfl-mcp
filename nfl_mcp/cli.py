"""
NFL MCP — command-line entry point

  nfl-mcp init                Interactive setup wizard
  nfl-mcp serve               Start the MCP server (Streamable HTTP)
  nfl-mcp ingest              Load NFL play-by-play data into the database
  nfl-mcp rebuild             Rebuild the derived tables from the loaded data
  nfl-mcp validate            Check a database slot before it replaces the active one
  nfl-mcp setup-client        Configure Claude Desktop / VS Code MCP
  nfl-mcp doctor              Check that everything is working
"""

import json
import os
import sys
from pathlib import Path

import click
import uvicorn

from .server import create_app
from .seasons import FIRST_SEASON, EARLIEST_SEASON, current_season

CURRENT_SEASON = current_season()


def _season_window_label(min_season: int | None, max_season: int | None) -> str:
    """Render a dataset's coverage window for --list, e.g. '1999-present'."""
    start = str(min_season) if min_season is not None else "earliest"
    end = str(max_season) if max_season is not None else "present"
    return f"{start}–{end}"


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


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Host to bind the HTTP server to. Use 0.0.0.0 for LAN access.")
@click.option("--port", default=8000, show_default=True, type=int,
              help="Port to listen on.")
def serve(host, port):
    """Start the MCP server over Streamable HTTP."""
    _exit_on_config_problem()
    _check_auth_token_env()
    display_host = "localhost" if host == "0.0.0.0" else host
    click.echo(f"🏈 NFL MCP server listening on http://{display_host}:{port}/mcp")
    if host == "0.0.0.0":
        click.echo(f"   (bound to all interfaces — clients should connect via http://localhost:{port}/mcp)")
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
def ingest(datasets, start, end, fresh, skip_views, list_datasets, pre1999, allow_failures):
    """Download NFL data via nflreadpy and load into DuckDB.

    Examples:
      nfl-mcp ingest                          # default datasets, all seasons
      nfl-mcp ingest --dataset all            # every dataset, all seasons
      nfl-mcp ingest --dataset schedules      # one dataset, all seasons
      nfl-mcp ingest --dataset pbp --start 2020 --end 2024
      nfl-mcp ingest --pre1999                # defaults + ESPN 1970-1998 records
      nfl-mcp ingest --list                   # show all dataset names
    """
    from .registry import REGISTRY, DEFAULT_DATASETS, ALL_DATASETS

    if list_datasets:
        click.echo("\nAvailable datasets:\n")
        for ds_id, defn in REGISTRY.items():
            default_tag = "[default]" if defn.default else "[opt-in]"
            click.echo(f"  {ds_id:<35} {default_tag:<10}  {defn.description}")
            if defn.seasonal:
                label = _season_window_label(defn.min_season, defn.max_season)
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

    from .ingest import run_ingest_datasets
    failures = run_ingest_datasets(
        dataset_ids=dataset_ids,
        start=start,
        end=end,
        fresh=fresh,
        skip_views=skip_views,
    )
    if failures:
        message = "Ingest failed for: " + "; ".join(failures)
        if not allow_failures:
            raise click.ClickException(message)
        click.echo(f"{message} (ignored: --allow-failures)")


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
    from .ingest import _apply_duckdb_pragmas, rebuild_derived_tables

    path = Path(db_path).expanduser() if db_path else get_duckdb_path()
    if not path.exists():
        raise click.ClickException(f"Database not found at {path}. Run 'nfl-mcp ingest' first.")

    conn = duckdb.connect(str(path))
    try:
        _apply_duckdb_pragmas(conn, str(path))
        rebuild_derived_tables(conn)
    except duckdb.Error as e:
        raise click.ClickException(f"Rebuild failed, previous derived tables kept: {e}")
    finally:
        conn.close()
    click.secho("Derived tables rebuilt.", fg="green")


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
@click.option("--skip-ingest", is_flag=True,
              help="Skip data ingestion (configure only).")
def init(start, end, skip_ingest):
    """Interactive setup wizard — configure database, load data, set up your IDE."""
    if start > end:
        raise click.UsageError("--start must be less than or equal to --end.")
    from .config import save_config, load_config, DEFAULT_DUCKDB_PATH

    click.echo()
    click.secho("🏈 NFL MCP Setup", bold=True)
    click.echo("=" * 50)

    # ── 1. Configure database path ─────────────────────────────────────────
    config = load_config()
    db_path = str(DEFAULT_DUCKDB_PATH)
    custom = click.confirm(
        f"  Database will be stored at {db_path}. Change location?", default=False
    )
    if custom:
        db_path = click.prompt("  Path", default=db_path)
    config["duckdb_path"] = db_path
    click.secho(f"  ✓ DuckDB → {db_path}", fg="green")

    # ── 2. Save config ─────────────────────────────────────────────────────
    path = save_config(config)
    click.secho(f"  ✓ Config saved to {path}", fg="green")

    # ── 3. Ingest data ─────────────────────────────────────────────────────
    if not skip_ingest:
        click.echo()
        click.echo(f"  Seasons to load: {start}–{end}")
        if not click.confirm("  Download and load NFL play-by-play data now?", default=True):
            skip_ingest = True

    if not skip_ingest:
        click.echo()
        from .ingest import run_ingest_datasets
        from .registry import DEFAULT_DATASETS
        failures = run_ingest_datasets(
            dataset_ids=DEFAULT_DATASETS,
            start=start,
            end=end,
            fresh=False,
            skip_views=False,
        )
        if failures:
            click.secho("  Warning: some data failed to load: " + "; ".join(failures)
                        + ". Run 'nfl-mcp ingest' again to retry.", fg="yellow")

    # ── 4. Set up IDE client ───────────────────────────────────────────────
    click.echo()
    if click.confirm("  Configure an MCP client (Claude Desktop / VS Code)?", default=True):
        _setup_client_interactive(config)

    # ── 5. Offer to start the server ───────────────────────────────────────
    click.echo()
    click.secho("🎉 Setup complete!", bold=True, fg="green")
    click.echo()
    click.echo("   The MCP server needs to be running for your IDE to connect.")
    click.echo("   You can start it any time with:  nfl-mcp serve")
    click.echo()
    if click.confirm("  Start the server now?", default=True):
        _exit_on_config_problem()
        _check_auth_token_env()
        click.secho("   Starting server on http://localhost:8000/mcp  (Ctrl-C to stop)", fg="cyan")
        click.echo()
        uvicorn.run(create_app(), host="127.0.0.1", port=8000)
    else:
        click.echo("   Run  nfl-mcp serve  when you're ready.")
        click.echo()


# ── setup-client ───────────────────────────────────────────────────────────────

@main.command("setup-client")
@click.option("--client", type=click.Choice(["claude-desktop", "claude-code", "vscode", "auto"]),
              default="auto", show_default=True,
              help="Which client to configure.")
def setup_client(client):
    """Auto-configure Claude Desktop, Claude Code, or VS Code to use the NFL MCP server."""
    from .config import load_config
    config = load_config()
    if client == "auto":
        _setup_client_interactive(config)
    elif client == "claude-desktop":
        _configure_claude_desktop(config)
    elif client == "claude-code":
        _configure_claude_code(config)
    elif client == "vscode":
        _configure_vscode(config)


def _setup_client_interactive(config: dict):
    """Prompt user to pick which clients to configure."""
    clients = []
    claude_path = _claude_desktop_config_path()
    if claude_path:
        clients.append(("Claude Desktop", "claude-desktop"))
    clients.append(("Claude Code (global ~/.claude/mcp.json)", "claude-code"))
    clients.append(("VS Code (.vscode/mcp.json in current directory)", "vscode"))

    for name, key in clients:
        if click.confirm(f"    Configure {name}?", default=True):
            if key == "claude-desktop":
                _configure_claude_desktop(config)
            elif key == "claude-code":
                _configure_claude_code(config)
            else:
                _configure_vscode(config)


def _build_server_config(config: dict) -> dict:
    """Build the MCP server JSON block for a client config file."""
    host = config.get("serve_host", "localhost")
    port = config.get("serve_port", 8000)
    # 0.0.0.0 is a bind address, not a connectable URL — use localhost for clients
    display_host = "localhost" if host == "0.0.0.0" else host
    return {"url": f"http://{display_host}:{port}/mcp"}


def _claude_desktop_config_path() -> Path | None:
    """Return Claude Desktop config path if it exists."""
    if sys.platform == "darwin":
        p = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif sys.platform == "win32":
        p = Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
    else:
        p = Path.home() / ".config" / "claude" / "claude_desktop_config.json"
    return p if p.parent.exists() else None


def _configure_claude_desktop(config: dict):
    """Write or merge NFL MCP server into Claude Desktop config."""
    path = _claude_desktop_config_path()
    if not path:
        click.secho("    ✗ Claude Desktop config directory not found", fg="yellow")
        return

    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["nfl"] = _build_server_config(config)

    path.write_text(json.dumps(existing, indent=2))
    click.secho(f"    ✓ Claude Desktop configured → {path}", fg="green")
    click.echo("      Restart Claude Desktop to pick up changes.")


def _configure_vscode(config: dict):
    """Write or merge NFL MCP server into .vscode/mcp.json."""
    vscode_dir = Path.cwd() / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    path = vscode_dir / "mcp.json"

    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}

    existing.setdefault("servers", {})
    existing["servers"]["nfl"] = _build_server_config(config)

    path.write_text(json.dumps(existing, indent=2))
    click.secho(f"    ✓ VS Code configured → {path}", fg="green")


def _configure_claude_code(config: dict):
    """Write or merge NFL MCP server into ~/.claude/mcp.json (Claude Code global config)."""
    path = Path.home() / ".claude" / "mcp.json"
    path.parent.mkdir(exist_ok=True)

    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["nfl"] = _build_server_config(config)

    path.write_text(json.dumps(existing, indent=2))
    click.secho(f"    ✓ Claude Code configured → {path}", fg="green")
    click.echo("      Restart Claude Code to pick up changes.")


# ── doctor ─────────────────────────────────────────────────────────────────────

@main.command()
def doctor():
    """Check that everything is configured and working."""
    from . import database
    from .config import load_config, config_exists, get_duckdb_path

    click.echo()
    click.secho("🏈 NFL MCP Doctor", bold=True)
    click.echo("=" * 50)
    all_ok = True

    # 1. Config
    if config_exists():
        click.secho("  ✓ Config file found", fg="green")
    else:
        click.secho("  ✗ No config file — run 'nfl-mcp init' first", fg="red")
        all_ok = False
        click.echo()
        return

    # 2. Database
    db_path = get_duckdb_path()
    if db_path.exists():
        click.secho(f"  ✓ DuckDB file exists ({db_path})", fg="green")
        try:
            with database.get_db_connection() as conn:
                count = conn.execute("SELECT COUNT(*) FROM plays").fetchone()[0]
                seasons = conn.execute(
                    "SELECT MIN(season), MAX(season) FROM plays"
                ).fetchone()
            click.secho(f"  ✓ {count:,} plays loaded (seasons {seasons[0]}–{seasons[1]})", fg="green")
        except Exception as e:
            click.secho(f"  ✗ Database error: {e}", fg="red")
            all_ok = False
    else:
        click.secho(f"  ✗ DuckDB file not found at {db_path}", fg="red")
        click.echo("    Run 'nfl-mcp ingest' to load data.")
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
