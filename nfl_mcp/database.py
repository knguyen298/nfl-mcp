"""
DuckDB connection management for NFL MCP Server.
"""

import functools
import os
import re
import warnings
from contextlib import contextmanager
from pathlib import Path

import duckdb

from .config import get_duckdb_path

_MEMORY_RE = re.compile(r"\d+(\.\d+)?\s*(KB|MB|GB|TB|KiB|MiB|GiB|TiB)", re.IGNORECASE | re.ASCII)

# The text of the error a missing database file raises. tools.py matches it to report
# DB_NOT_FOUND, so the two modules share the one spelling.
DB_NOT_FOUND_TEXT = "DuckDB file not found at"


@functools.cache
def _memory_limit_problem(value: str) -> str | None:
    """Probe DuckDB once per distinct value, so '0MB' is rejected at startup."""
    probe = None
    try:
        probe = duckdb.connect(":memory:", config={"memory_limit": value})
        # DuckDB accepts the value at connect and fails on the first real query.
        probe.execute("SELECT current_setting('memory_limit')")
    except duckdb.Error as e:
        return f"NFL_MCP_DUCKDB_MEMORY_LIMIT={value!r} is not valid: {e}"
    finally:
        if probe is not None:
            probe.close()
    return None


def connection_config_problem() -> str | None:
    """Return a message naming the first invalid NFL_MCP_DUCKDB_* env var, or None.

    Valid memory limit: a number plus a DuckDB unit, e.g. '512MB' or '2GiB'.
    Valid threads: a positive integer. The temp dir is any non-empty path.
    """
    mem = os.getenv("NFL_MCP_DUCKDB_MEMORY_LIMIT")
    if mem and not _MEMORY_RE.fullmatch(mem.strip()):
        return (f"NFL_MCP_DUCKDB_MEMORY_LIMIT={mem!r} is not valid: "
                "use a number and a unit such as '512MB' or '2GiB'")
    if mem:
        problem = _memory_limit_problem(mem.strip())
        if problem:
            return problem
    threads = os.getenv("NFL_MCP_DUCKDB_THREADS")
    if threads and not (threads.strip().isascii() and threads.strip().isdigit() and int(threads) > 0):
        return f"NFL_MCP_DUCKDB_THREADS={threads!r} is not valid: use a positive whole number"
    return None


def _runtime_config() -> dict:
    """Settings for the serve process, before the access lock is added.

    Extension autoload and autoinstall are off, so a query cannot pull an extension from
    disk or from the network. Python replacement scans are off, so a table name cannot
    resolve to a Python object in the server process.

    In memory-constrained containers a heavy query can exceed the cgroup limit
    and OOM-kill the server (DuckDB otherwise sizes itself to host RAM). These
    env-driven caps keep queries within budget, spilling to disk when needed:
      NFL_MCP_DUCKDB_MEMORY_LIMIT  e.g. '512MB'
      NFL_MCP_DUCKDB_THREADS       e.g. '1'
      NFL_MCP_DUCKDB_TEMP_DIR      writable spill dir, e.g. '/tmp/.duckdb_spill'
    Unset vars leave DuckDB's defaults (local/dev use). Values go in the config
    dict, never into SQL.
    """
    config: dict = {
        "autoinstall_known_extensions": False,
        "autoload_known_extensions": False,
        "python_enable_replacements": False,
    }
    mem = os.getenv("NFL_MCP_DUCKDB_MEMORY_LIMIT")
    if mem:
        config["memory_limit"] = mem.strip()
    threads = os.getenv("NFL_MCP_DUCKDB_THREADS")
    if threads:
        config["threads"] = int(threads)
    spill = os.getenv("NFL_MCP_DUCKDB_TEMP_DIR")
    if spill:
        config["temp_directory"] = spill
    return config


def _connection_config() -> dict:
    """Build the duckdb.connect config: the runtime settings, locked down.

    ``enable_external_access`` off makes DuckDB refuse every file system and network
    access outside the attached database: ``read_text``, ``read_csv``, ``read_parquet``,
    ``glob``, a quoted file path after FROM, ``ATTACH``, ``COPY``, ``INSTALL``/``LOAD``,
    http(s) paths. ``lock_configuration`` then refuses any later ``SET``/``RESET``, so a
    query cannot turn access back on. Both go in the connect-time config: DuckDB applies
    them before the database is in use, so the spill directory is set before access is
    turned off, and DuckDB adds it to its allowed directories.

    Every connection in the process passes the same dict. DuckDB shares one database
    instance between the open connections to a file and refuses a connection whose config
    differs, so no connection can open the instance with weaker settings.
    """
    problem = connection_config_problem()
    if problem:
        raise ValueError(problem)
    config = {"enable_external_access": False, **_runtime_config()}
    config["lock_configuration"] = True
    return config


def _lock_external_access(conn: duckdb.DuckDBPyConnection) -> None:
    """Refuse a connection whose database instance still allows external access.

    The connect-time config turns access off and locks it. This reads the setting back
    from the instance, so a connection is never handed out on the word of the config alone.
    """
    if conn.execute("SELECT current_setting('enable_external_access')").fetchone()[0]:
        raise RuntimeError(
            "DuckDB configuration is locked with external access enabled; refusing to query."
        )


@contextmanager
def get_db_connection():
    """Context manager yielding a fresh read-only, locked-down DuckDB connection.

    Each call opens a new connection so callers (including threaded timeouts)
    never share a connection across threads. The connection cannot read files
    or reach the network outside the database. The ingest path opens its own
    write connection and does not use this.

    Opening a connection creates nothing except the spill directory a large query needs,
    and only when one is configured. A missing database file raises duckdb.IOException
    with the path, which tools.py reports as `DB_NOT_FOUND`; making its parent directory
    would hide that behind an empty directory and would fail on a read-only filesystem.
    """
    db_path = get_duckdb_path()
    if not db_path.is_file():
        raise duckdb.IOException(f"{DB_NOT_FOUND_TEXT} {db_path}")
    config = _connection_config()
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
