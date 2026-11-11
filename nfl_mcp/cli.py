"""
NFL MCP — command-line entry point

  nfl-mcp init                Interactive setup wizard
  nfl-mcp serve               Start the MCP server (Streamable HTTP)
  nfl-mcp ingest              Load NFL play-by-play data into the database
  nfl-mcp setup-client        Configure Claude Desktop / VS Code MCP
  nfl-mcp doctor              Check that everything is working
"""

import json
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


def _season_window_label(min_season: int | None, max_season: int | None) -> str:
    """Render a dataset's coverage window for --list, e.g. '1999-present'."""
    start = str(min_season) if min_season is not None else "earliest"
    end = str(max_season) if max_season is not None else "present"
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


# ── serve ──────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--host", default=None,
              help=f"Host to bind the HTTP server to (default {DEFAULT_HOST}). "
                   "Use 0.0.0.0 for LAN access. Omit to use the 'serve_host' key in "
                   "config.json, or the default when that key is absent.")
@click.option("--port", default=None, type=int,
              help=f"Port to listen on (default {DEFAULT_PORT}). Omit to use the "
                   "'serve_port' key in config.json, or the default when it is absent.")
def serve(host, port):
    """Start the MCP server over Streamable HTTP."""
    from .config import load_config

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
@click.option("--strict", is_flag=True,
              help="Exit non-zero when any fetch failed, not only when nothing loaded at all.")
def ingest(datasets, start, end, fresh, skip_views, list_datasets, pre1999, strict):
    """Download NFL data via nflreadpy and load into DuckDB.

    Exits 0 unless every fetch this run made failed, which is the one pattern that says the
    database is no newer than it was before the run. Some failures are ordinary: play-by-play
    for a new season does not exist before Week 1, so it fails on every run from March to
    September. Use --strict to make any single failure non-zero.

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
    result = run_ingest_datasets(
        dataset_ids=dataset_ids,
        start=start,
        end=end,
        fresh=fresh,
        skip_views=skip_views,
    )

    nothing_loaded, any_failed = _run_problems(result)
    if nothing_loaded:
        click.echo(NOTHING_LOADED_MESSAGE)
        sys.exit(1)
    if strict and any_failed:
        sys.exit(1)


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

@main.command()
def doctor():
    """Check that everything is configured and working."""
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
            conn = duckdb.connect(str(db_path), read_only=True)
            try:
                count = conn.execute("SELECT COUNT(*) FROM plays").fetchone()[0]
                seasons = conn.execute(
                    "SELECT MIN(season), MAX(season) FROM plays"
                ).fetchone()
            finally:
                conn.close()
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
