"""
DuckDB connection management for NFL MCP Server.
"""

import os
import duckdb
from contextlib import contextmanager
from typing import Any

from .config import get_duckdb_path

# DuckDB still reads files and installs extensions on a read-only connection, so the
# serve connection locks its configuration at connect time instead of SET-ting it after.
_LOCKED_CONFIG: dict[str, Any] = {
    "enable_external_access": False,
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
    "lock_configuration": True,
}


def _connection_config() -> dict[str, Any]:
    """Build the locked, read-only DuckDB config for the serve process.

    In memory-constrained containers a heavy query can exceed the cgroup limit
    and OOM-kill the server (DuckDB otherwise sizes itself to host RAM). These
    env-driven caps keep queries within budget, spilling to disk when needed:
      NFL_MCP_DUCKDB_MEMORY_LIMIT  e.g. '512MB'
      NFL_MCP_DUCKDB_THREADS       e.g. '1'
      NFL_MCP_DUCKDB_TEMP_DIR      writable spill dir, e.g. '/tmp/.duckdb_spill'
    Unset vars leave DuckDB's defaults (local/dev use). A value DuckDB cannot use raises
    ValueError naming the variable; `nfl-mcp serve` asks connection_config_problem() at start.
    """
    config = dict(_LOCKED_CONFIG)
    mem = os.getenv("NFL_MCP_DUCKDB_MEMORY_LIMIT")
    if mem:
        config["memory_limit"] = mem
    threads = os.getenv("NFL_MCP_DUCKDB_THREADS")
    if threads:
        if not threads.strip().isdigit() or int(threads) < 1:
            raise ValueError(f"NFL_MCP_DUCKDB_THREADS={threads!r} is not a whole number above "
                             "zero. Set it to e.g. '1', or unset it.")
        config["threads"] = int(threads)
    spill = os.getenv("NFL_MCP_DUCKDB_TEMP_DIR")
    if spill:
        config["temp_directory"] = spill
    return config


def connection_config_problem() -> str | None:
    """Say what is wrong with the NFL_MCP_DUCKDB_* variables, or None when DuckDB accepts them.

    The values are tried on an in-memory database, so a bad one stops `serve` at start
    instead of failing every tool call while the server reports itself healthy.
    """
    try:
        config = _connection_config()
    except ValueError as e:
        return str(e)
    for key, var in (("memory_limit", "NFL_MCP_DUCKDB_MEMORY_LIMIT"),
                     ("temp_directory", "NFL_MCP_DUCKDB_TEMP_DIR")):
        if key in config:
            try:
                duckdb.connect(":memory:", config={key: config[key]}).close()
            except duckdb.Error as e:
                return f"{var}={config[key]!r} is not a value DuckDB accepts: {e}"
    return None


@contextmanager
def get_db_connection():
    """Context manager yielding a fresh read-only DuckDB connection.

    Each call opens a new connection so callers (including threaded timeouts)
    never share a connection across threads. File and network access are off
    (read_text, read_csv, glob, COPY ... TO and ATTACH all fail), so ingest and
    the CLI keep opening their own writable connections.
    """
    db_path = get_duckdb_path()
    if not db_path.exists():
        # A read creates nothing: no file and no parent directory.
        raise duckdb.IOException(
            f"No database at {db_path}. Run 'nfl-mcp ingest' to build it, or set "
            "NFL_MCP_DB_PATH to the file you built.")
    conn = duckdb.connect(str(db_path), read_only=True, config=_connection_config())
    try:
        yield conn
    finally:
        conn.close()
