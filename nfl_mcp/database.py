"""
DuckDB connection management for NFL MCP Server.
"""

import os
import threading
import warnings
from pathlib import Path

import duckdb
from contextlib import contextmanager

from .config import get_duckdb_path

# Serializes the check-then-lock in _lock_external_access. DuckDB shares one database
# instance between every open connection to the same file in this process, and the lock
# is a setting of that instance, so two threads must not race to apply it.
_LOCK_SETUP = threading.Lock()


def _runtime_config() -> dict:
    """Settings DuckDB applies when it opens the database instance for the serve process.

    Extension autoload and autoinstall are off, so a query cannot pull an extension from
    disk or from the network. Python replacement scans are off, so a table name cannot
    resolve to a Python object in the server process.

    In memory-constrained containers a heavy query can exceed the cgroup limit
    and OOM-kill the server (DuckDB otherwise sizes itself to host RAM). These
    env-driven caps keep queries within budget, spilling to disk when needed:
      NFL_MCP_DUCKDB_MEMORY_LIMIT  e.g. '512MB'
      NFL_MCP_DUCKDB_THREADS       e.g. '1'
      NFL_MCP_DUCKDB_TEMP_DIR      writable spill dir, e.g. '/tmp/.duckdb_spill'
    Unset vars leave DuckDB's defaults (local/dev use).

    These go in the connect-time config, not in SET statements: the configuration is
    locked after the first connection opens, and a SET on a later connection that shares
    the instance would fail. DuckDB applies them before the database is in use, so the
    spill directory is set before external access is turned off. DuckDB adds the spill
    directory to its allowed directories, so spilling keeps working after the lock.
    """
    config: dict = {
        "autoinstall_known_extensions": False,
        "autoload_known_extensions": False,
        "python_enable_replacements": False,
    }
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


def _lock_external_access(conn: duckdb.DuckDBPyConnection) -> None:
    """Stop queries on this database instance from reaching files or the network.

    With ``enable_external_access`` off, DuckDB refuses every file system and network
    access outside the already-attached database: ``read_text``, ``read_csv``,
    ``read_parquet``, ``glob``, a quoted file path after FROM, ``ATTACH``, ``COPY``,
    ``INSTALL``/``LOAD``, http(s) paths. ``lock_configuration`` then refuses any later
    ``SET``/``RESET``, so a query cannot turn access back on.

    Both are settings of the shared database instance. The first connection that opens
    the instance applies them; later connections find the instance locked already.
    Raises RuntimeError if the instance is locked with external access still on, because
    then this function cannot close it.
    """
    with _LOCK_SETUP:
        if not conn.execute("SELECT current_setting('lock_configuration')").fetchone()[0]:
            conn.execute("SET enable_external_access = false")
            conn.execute("SET lock_configuration = true")
    if conn.execute("SELECT current_setting('enable_external_access')").fetchone()[0]:
        raise RuntimeError(
            "DuckDB configuration is locked with external access enabled; refusing to query."
        )


@contextmanager
def get_db_connection():
    """Context manager yielding a fresh read-only DuckDB connection.

    Each call opens a new connection so callers (including threaded timeouts)
    never share a connection across threads. The connection cannot read files
    or reach the network outside the database (see _lock_external_access).
    The ingest path opens its own write connection and does not use this.

    Opening a connection creates nothing except the spill directory a large query needs,
    and only when one is configured. A missing database file raises the IOException that
    tools.py maps to `DB_NOT_FOUND`; making its parent directory would hide that behind an
    empty directory and would fail on a read-only filesystem.
    """
    db_path = get_duckdb_path()
    config = _runtime_config()
    spill = config.get("temp_directory")
    if spill and not _make_spill_directory(spill):
        del config["temp_directory"]
    conn = duckdb.connect(str(db_path), read_only=True, config=config)
    try:
        _lock_external_access(conn)
        yield conn
    finally:
        conn.close()


def _make_spill_directory(spill: str) -> bool:
    """Create DuckDB's spill directory, and say whether it is there for a query to use.

    Spilling is what keeps a heavy query inside the memory cap, so a directory that cannot
    be made is worth reporting. It is not worth failing the query over: without it DuckDB
    falls back to its own default, and a query that fits in memory still answers.
    """
    try:
        Path(spill).mkdir(parents=True, exist_ok=True)
        return True
    except OSError as exc:
        warnings.warn(
            f"Could not create the DuckDB spill directory {spill!r}: {exc}. "
            "Running without an explicit temp_directory."
        )
        return False
