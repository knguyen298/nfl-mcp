"""
Configuration management for NFL MCP.

Stores config at ~/.nfl-mcp/config.json.
DuckDB database lives at ~/.nfl-mcp/nflread.duckdb by default.

Reading the configuration never writes to disk. A running server reads it on every tool
call, so a read that renamed the file would race another call and would sweep away a file
the user has open in an editor. A malformed file is reported once per content, on stderr
(stdout carries the MCP protocol in stdio mode), and the defaults are returned. Only
:func:`save_config` changes anything on disk, and it copies a malformed file to
``config.json.bak`` before it overwrites.
"""

import json
import os
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".nfl-mcp"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_DUCKDB_PATH = CONFIG_DIR / "nflread.duckdb"

_DEFAULT_CONFIG = {
    "duckdb_path": str(DEFAULT_DUCKDB_PATH),
}

# (path, bytes) already reported, so a long-running server warns once per file state
# instead of once per tool call.
_warned_malformed: set[tuple[str, bytes]] = set()


def ensure_config_dir() -> Path:
    """Create ~/.nfl-mcp/ if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def _parse_config(raw: bytes) -> dict[str, Any] | None:
    """The JSON object these bytes hold, or None when they hold no object."""
    try:
        stored = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return stored if isinstance(stored, dict) else None


def _fallback_from_malformed_config(path: Path, raw: bytes) -> dict[str, Any]:
    """Report an unusable config file and return the defaults, without touching the disk.

    One file state is reported once per process: a server reads the file on every tool
    call, and the second call has nothing new to say.
    """
    key = (str(path), raw)
    if key not in _warned_malformed:
        _warned_malformed.add(key)
        warnings.warn(
            f"Malformed config at {path}. The file was left exactly as it is: "
            "nothing was moved or rewritten. Falling back to defaults; "
            "run 'nfl-mcp init' to regenerate config."
        )
    return {**_DEFAULT_CONFIG}


def load_config() -> dict[str, Any]:
    """Load config from disk, falling back to defaults. Reading writes nothing.

    A file that is not valid JSON, is not UTF-8 text, or parses to something other than a
    JSON object is malformed: it is reported and the defaults are returned, and the file
    stays byte-for-byte as it was. A valid object whose "duckdb_path" is not a non-empty
    string keeps its other keys and falls back to the default database path.
    """
    if not CONFIG_FILE.exists():
        return {**_DEFAULT_CONFIG}
    raw: bytes | None = None
    with suppress(OSError):  # unreadable is not the same as unusable: say which it was
        raw = CONFIG_FILE.read_bytes()
    if raw is None:
        warnings.warn(
            f"Could not read {CONFIG_FILE}. Falling back to defaults; "
            "run 'nfl-mcp init' to write a new one."
        )
        return {**_DEFAULT_CONFIG}
    stored = _parse_config(raw)
    if stored is None:
        return _fallback_from_malformed_config(CONFIG_FILE, raw)
    config = {**_DEFAULT_CONFIG, **stored}
    stored_path = config["duckdb_path"]
    if not isinstance(stored_path, str) or not stored_path:
        warnings.warn(
            f"Invalid 'duckdb_path' in {CONFIG_FILE}: {stored_path!r} is not a path. "
            f"Using the default {_DEFAULT_CONFIG['duckdb_path']}; "
            "run 'nfl-mcp init' to choose another one."
        )
        config["duckdb_path"] = _DEFAULT_CONFIG["duckdb_path"]
    return config


def _back_up_malformed_config() -> None:
    """Copy an unusable config file to config.json.bak before it is overwritten.

    This is the only place a backup is made, and it runs on a write path. A usable file is
    left alone: its contents are about to be replaced by a caller that read them.
    """
    raw: bytes | None = None
    with suppress(OSError):  # no file at all, or an unreadable one: nothing to back up
        raw = CONFIG_FILE.read_bytes()
    if raw is None or _parse_config(raw) is not None:
        return
    backup = CONFIG_FILE.with_suffix(".json.bak")
    try:
        backup.write_bytes(raw)
    except OSError as exc:
        warnings.warn(
            f"Could not back up the malformed config at {CONFIG_FILE} to {backup}: {exc}. "
            "The overwrite goes ahead; that copy is the only record of the old file."
        )


def _temp_file_for(target: Path) -> Path:
    """Temp path beside `target`, so os.replace stays on one filesystem."""
    return target.with_name(f"{target.name}.{os.getpid()}.tmp")


def write_json(target: Path, payload: dict[str, Any]) -> None:
    """Write `payload` as JSON into `target` so a failure cannot leave half a file.

    The bytes go to a temp file in the same directory, are flushed and fsynced, and only
    then does os.replace put them in place. A crash or a full disk partway through leaves
    the previous file readable instead of truncated, and the temp file is removed.
    """
    temp = _temp_file_for(target)
    try:
        with open(temp, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, target)
    except BaseException:
        with suppress(OSError):
            temp.unlink(missing_ok=True)
        raise


def save_config(config: dict[str, Any]) -> Path:
    """Write config to ~/.nfl-mcp/config.json, atomically (see :func:`write_json`).

    This is the write path, so it is where a malformed file on disk gets its backup copy.
    """
    ensure_config_dir()
    _back_up_malformed_config()
    write_json(CONFIG_FILE, config)
    return CONFIG_FILE


def get_duckdb_path() -> Path:
    """Return resolved path to the DuckDB database file."""
    # Env var override for CI / containers
    override = os.getenv("NFL_MCP_DB_PATH")
    if override:
        return Path(override).expanduser()
    return Path(load_config()["duckdb_path"]).expanduser()


def config_exists() -> bool:
    """Check whether a config file has been created (i.e., init has been run)."""
    return CONFIG_FILE.exists()


def config_file_is_usable() -> bool:
    """True when there is no config file, or when it holds a JSON object.

    A file that exists but cannot be parsed is not usable: a caller that says "config
    found" would otherwise hide the fact that it is running on defaults the user did not
    choose. Reading this tells you nothing new about the file: it changes nothing.
    """
    if not CONFIG_FILE.exists():
        return True
    raw: bytes | None = None
    with suppress(OSError):  # unreadable is not usable either
        raw = CONFIG_FILE.read_bytes()
    return raw is not None and _parse_config(raw) is not None
