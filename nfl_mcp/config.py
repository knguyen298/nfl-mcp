"""
Configuration management for NFL MCP.

Stores config at ~/.nfl-mcp/config.json.
DuckDB database lives at ~/.nfl-mcp/nflread.duckdb by default.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".nfl-mcp"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_DUCKDB_PATH = CONFIG_DIR / "nflread.duckdb"

_DEFAULT_CONFIG = {
    "duckdb_path": str(DEFAULT_DUCKDB_PATH),
}


def ensure_config_dir() -> Path:
    """Create ~/.nfl-mcp/ if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def load_config() -> dict[str, Any]:
    """Load config from disk, falling back to defaults.

    A read never changes the file. A file that is not a JSON object is reported with a
    warning and the defaults are used, so the user can still repair what they wrote.
    """
    if not CONFIG_FILE.exists():
        return {**_DEFAULT_CONFIG}
    try:
        with open(CONFIG_FILE) as f:
            stored = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        problem = str(e)
    else:
        if isinstance(stored, dict):
            return {**_DEFAULT_CONFIG, **stored}
        problem = f"it holds a JSON {type(stored).__name__}, not an object"
    warnings.warn(
        f"Cannot use the config at {CONFIG_FILE}: {problem}. Using the defaults. "
        "Fix the file, or run 'nfl-mcp init' to write a new one."
    )
    return {**_DEFAULT_CONFIG}


def save_config(config: dict[str, Any]) -> Path:
    """Write config to ~/.nfl-mcp/config.json."""
    ensure_config_dir()
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    return CONFIG_FILE


def get_duckdb_path() -> Path:
    """Return the absolute path to the DuckDB database file.

    A relative path in the config file is read from the config directory, because Claude
    Desktop usually starts the server from `/`. A relative NFL_MCP_DB_PATH is read from the
    current directory, where the shell that set it runs. Symlinks are kept: the container
    swaps the database by moving the /data/nflread.duckdb link.
    """
    # Env var override for CI / containers
    override = os.getenv("NFL_MCP_DB_PATH")
    if override:
        return Path(override).expanduser().absolute()
    return CONFIG_DIR / Path(load_config()["duckdb_path"]).expanduser()


def config_exists() -> bool:
    """Check whether a config file has been created (i.e., init has been run)."""
    return CONFIG_FILE.exists()
