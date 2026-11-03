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
    Unset vars leave DuckDB's defaults (local/dev use).
    """
    config = dict(_LOCKED_CONFIG)
    mem = os.getenv("NFL_MCP_DUCKDB_MEMORY_LIMIT")
    if mem:
        config["memory_limit"] = mem
    threads = os.getenv("NFL_MCP_DUCKDB_THREADS")
    if threads:
        config["threads"] = int(threads)
    spill = os.getenv("NFL_MCP_DUCKDB_TEMP_DIR")
    if spill:
        config["temp_directory"] = spill
    return config


@contextmanager
def get_db_connection():
    """Context manager yielding a fresh read-only DuckDB connection.

    Each call opens a new connection so callers (including threaded timeouts)
    never share a connection across threads. File and network access are off
    (read_text, read_csv, glob, COPY ... TO and ATTACH all fail), so ingest and
    the CLI keep opening their own writable connections.
    """
    db_path = get_duckdb_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path), read_only=True, config=_connection_config())
    try:
        yield conn
    finally:
        conn.close()
