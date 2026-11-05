"""
DuckDB connection management for NFL MCP Server.
"""

import functools
import os
import re
import duckdb
from contextlib import contextmanager

from .config import get_duckdb_path

_MEMORY_RE = re.compile(r"\d+(\.\d+)?\s*(KB|MB|GB|TB|KiB|MiB|GiB|TiB)", re.IGNORECASE | re.ASCII)


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


def _connection_config() -> dict:
    """Build the duckdb.connect config: locked down, with env caps for the serve process.

    In memory-constrained containers a heavy query can exceed the cgroup limit
    and OOM-kill the server (DuckDB otherwise sizes itself to host RAM). These
    env-driven caps keep queries within budget, spilling to disk when needed:
      NFL_MCP_DUCKDB_MEMORY_LIMIT  e.g. '512MB'
      NFL_MCP_DUCKDB_THREADS       e.g. '1'
      NFL_MCP_DUCKDB_TEMP_DIR      writable spill dir, e.g. '/tmp/.duckdb_spill'
    Unset vars leave DuckDB's defaults (local/dev use). Values go in the config
    dict, never into SQL. lock_configuration stops later SET statements.
    """
    problem = connection_config_problem()
    if problem:
        raise ValueError(problem)
    config = {
        "enable_external_access": False,
        "autoinstall_known_extensions": False,
        "autoload_known_extensions": False,
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
    config["lock_configuration"] = True
    return config


@contextmanager
def get_db_connection():
    """Context manager yielding a fresh read-only, locked-down DuckDB connection.

    Each call opens a new connection so callers (including threaded timeouts)
    never share a connection across threads. Never creates the file or its
    directory; a missing file raises duckdb.IOException with the path.
    """
    db_path = get_duckdb_path()
    if not db_path.is_file():
        raise duckdb.IOException(f"DuckDB file not found at {db_path}")
    conn = duckdb.connect(str(db_path), read_only=True, config=_connection_config())
    try:
        yield conn
    finally:
        conn.close()
