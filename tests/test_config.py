"""Tests for configuration loading behavior."""

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import nfl_mcp.config as config


def _point_at_tmp(monkeypatch, tmp_path):
    """Point the config dir and file at tmp_path and make the dir, so no test reads or
    writes the real ~/.nfl-mcp."""
    config_dir = tmp_path / "nfl-mcp"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)
    return config_dir, config_file


def test_ensure_config_dir_creates_directory(tmp_path, monkeypatch):
    config_dir = tmp_path / "nfl-mcp"
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    created = config.ensure_config_dir()
    assert created == config_dir
    assert config_dir.exists()


def test_load_config_returns_defaults_when_file_missing(tmp_path, monkeypatch):
    config_dir = tmp_path / "nfl-mcp"
    config_file = config_dir / "config.json"
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)
    assert config.load_config() == {**config._DEFAULT_CONFIG}


def test_load_config_merges_valid_json(tmp_path, monkeypatch):
    config_dir = tmp_path / "nfl-mcp"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps({"duckdb_path": "/tmp/custom.duckdb", "x": 1}))

    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)

    loaded = config.load_config()
    assert loaded["duckdb_path"] == "/tmp/custom.duckdb"
    assert loaded["x"] == 1


def test_load_config_leaves_a_malformed_file_alone(tmp_path, monkeypatch):
    """A malformed file is reported and left in place: reading must not write to disk."""
    config_dir, config_file = _point_at_tmp(monkeypatch, tmp_path)
    original = b"{ this is not valid json "
    config_file.write_bytes(original)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = config.load_config()

    assert loaded == {**config._DEFAULT_CONFIG}
    assert config_file.read_bytes() == original
    assert list(config_dir.iterdir()) == [config_file]
    assert any("Malformed config" in str(w.message) for w in caught)
    assert all("Backed up to" not in str(w.message) for w in caught)


def test_reading_a_malformed_config_prints_nothing_itself(tmp_path, monkeypatch, capsys):
    """The read path speaks through warnings.warn, never by writing to stdout itself."""
    _, config_file = _point_at_tmp(monkeypatch, tmp_path)
    config_file.write_text("{ not json")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = config.load_config()
        got = config.get_duckdb_path()

    captured = capsys.readouterr()
    assert loaded == {**config._DEFAULT_CONFIG}
    assert got == config.DEFAULT_DUCKDB_PATH
    assert any("Malformed config" in str(w.message) for w in caught)
    assert captured.out == ""


def test_a_malformed_config_warns_on_stderr_and_leaves_stdout_free(tmp_path):
    """In a real process the text goes to stderr: stdout carries the MCP protocol.

    HOME points at a temp directory, so this subprocess never sees the developer's
    ~/.nfl-mcp, and it proves the file survives two reads in one process.
    """
    config_dir = tmp_path / "home" / ".nfl-mcp"
    config_dir.mkdir(parents=True)
    malformed = "{ this is not valid json"
    (config_dir / "config.json").write_text(malformed)

    env = dict(os.environ, HOME=str(tmp_path / "home"), NFL_MCP_DB_PATH="")
    script = (
        "import warnings; warnings.simplefilter('always'); "
        "from nfl_mcp.config import load_config, get_duckdb_path; "
        "load_config(); get_duckdb_path()"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), check=False,
    )

    assert proc.stdout == ""
    assert "Malformed config" in proc.stderr
    assert (config_dir / "config.json").read_text() == malformed
    assert not (config_dir / "config.json.broken").exists()
    assert not (config_dir / "config.json.bak").exists()


def test_two_reads_of_a_malformed_config_do_not_race_on_a_rename(tmp_path, monkeypatch):
    """The file keeps its identity: no read replaces it, moves it aside, or rewrites it."""
    _, config_file = _point_at_tmp(monkeypatch, tmp_path)
    config_file.write_text("nonsense")
    before = config_file.stat()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = config.load_config()
        second = config.get_duckdb_path()

    after = config_file.stat()
    assert first == {**config._DEFAULT_CONFIG}
    assert second == config.DEFAULT_DUCKDB_PATH
    assert after.st_ino == before.st_ino
    assert config_file.read_text() == "nonsense"


def test_a_malformed_file_is_reported_once_per_content(tmp_path, monkeypatch):
    """One file state is said once; a different file state is said again."""
    _, config_file = _point_at_tmp(monkeypatch, tmp_path)
    config_file.write_text("{ broken")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.load_config()
        config.load_config()
    assert sum("Malformed config" in str(w.message) for w in caught) == 1

    config_file.write_text("{ still broken, differently")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.load_config()
    assert sum("Malformed config" in str(w.message) for w in caught) == 1


@pytest.mark.parametrize("content", [b"{\xff\xfe not utf-8", b"{} {}", b""])
def test_load_config_treats_undecodable_bytes_as_malformed(tmp_path, monkeypatch, content):
    """Bytes that are not UTF-8 text are malformed, and must not raise out of the read."""
    _, config_file = _point_at_tmp(monkeypatch, tmp_path)
    config_file.write_bytes(content)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = config.load_config()

    assert loaded == {**config._DEFAULT_CONFIG}
    assert config_file.read_bytes() == content
    assert any("Malformed config" in str(w.message) for w in caught)


def test_load_config_reports_a_file_it_cannot_read(tmp_path, monkeypatch):
    """"Unreadable" is a different problem from "unusable", and it also touches nothing."""
    _, config_file = _point_at_tmp(monkeypatch, tmp_path)
    config_file.write_text('{"duckdb_path": "/data/nflread.duckdb"}')

    def _deny(self, *_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(type(config_file), "read_bytes", _deny, raising=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = config.load_config()

    assert loaded == {**config._DEFAULT_CONFIG}
    assert any("Could not read" in str(w.message) for w in caught)
    assert all("Malformed config" not in str(w.message) for w in caught)


@pytest.mark.parametrize("content", ["[]", '"x"', "1", "null", "[1, 2]"])
def test_load_config_treats_non_object_json_as_malformed(tmp_path, monkeypatch, content):
    """Valid JSON that is not an object must not reach `{**defaults, **stored}`."""
    config_dir, config_file = _point_at_tmp(monkeypatch, tmp_path)
    config_file.write_text(content)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = config.load_config()

    assert loaded == {**config._DEFAULT_CONFIG}
    assert config_file.read_text() == content
    assert not (config_dir / "config.json.broken").exists()
    assert any("Malformed config" in str(w.message) for w in caught)


@pytest.mark.parametrize("value", [None, "", 42, ["~/.nfl-mcp/nflread.duckdb"], True])
def test_load_config_ignores_a_duckdb_path_that_is_not_a_usable_string(tmp_path, monkeypatch, value):
    """A bad duckdb_path falls back to the default; the file is left in place."""
    config_dir = tmp_path / "nfl-mcp"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps({"duckdb_path": value, "serve_port": 9000}))

    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = config.load_config()

    assert loaded["duckdb_path"] == str(config.DEFAULT_DUCKDB_PATH)
    assert loaded["serve_port"] == 9000
    assert config_file.exists()
    assert not (config_dir / "config.json.broken").exists()
    assert any("Invalid 'duckdb_path'" in str(w.message) for w in caught)


def test_get_duckdb_path_survives_a_null_duckdb_path(tmp_path, monkeypatch):
    """get_duckdb_path() must not raise TypeError on a config with a null path."""
    config_dir = tmp_path / "nfl-mcp"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps({"duckdb_path": None}))

    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)
    monkeypatch.delenv("NFL_MCP_DB_PATH", raising=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = config.get_duckdb_path()

    assert got == config.DEFAULT_DUCKDB_PATH


def test_save_config_writes_file_and_returns_path(tmp_path, monkeypatch):
    config_dir = tmp_path / "nfl-mcp"
    config_file = config_dir / "config.json"
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)

    data = {"duckdb_path": "/tmp/saved.duckdb", "hello": "world"}
    returned = config.save_config(data)

    assert returned == config_file
    assert config_file.exists()
    assert json.loads(config_file.read_text()) == data


def test_save_config_keeps_the_file_format(tmp_path, monkeypatch):
    """Same bytes a caller wrote before: two-space indent, the keys in the caller's order."""
    _, config_file = _point_at_tmp(monkeypatch, tmp_path)
    data = {"duckdb_path": "/tmp/saved.duckdb", "serve_port": 9000}

    config.save_config(data)

    assert config_file.read_text() == json.dumps(data, indent=2)
    assert list(config_file.parent.iterdir()) == [config_file]


def test_save_config_leaves_the_old_file_when_the_rename_fails(tmp_path, monkeypatch):
    """A failure before os.replace must not touch the file a server is reading."""
    config_dir, config_file = _point_at_tmp(monkeypatch, tmp_path)
    original = json.dumps({"duckdb_path": "/data/old.duckdb"}, indent=2)
    config_file.write_text(original)

    def _deny(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(config.os, "replace", _deny)

    with pytest.raises(OSError):
        config.save_config({"duckdb_path": "/data/new.duckdb"})

    assert config_file.read_text() == original
    assert list(config_dir.iterdir()) == [config_file]


def test_save_config_leaves_the_old_file_when_the_write_stops_early(tmp_path, monkeypatch):
    """Same guarantee for a crash between the write and the rename (fsync raises)."""
    config_dir, config_file = _point_at_tmp(monkeypatch, tmp_path)
    original = json.dumps({"duckdb_path": "/data/old.duckdb"}, indent=2)
    config_file.write_text(original)

    def _deny(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(config.os, "fsync", _deny)

    with pytest.raises(OSError):
        config.save_config({"duckdb_path": "/data/new.duckdb"})

    assert config_file.read_text() == original
    assert list(config_dir.iterdir()) == [config_file]


def test_save_config_backs_up_a_malformed_file_before_overwriting_it(tmp_path, monkeypatch):
    """The backup belongs to the write path: init is about to overwrite what it cannot use."""
    config_dir, config_file = _point_at_tmp(monkeypatch, tmp_path)
    malformed = "{ this is not valid json"
    config_file.write_text(malformed)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.save_config({"duckdb_path": "/data/new.duckdb"})

    assert (config_dir / "config.json.bak").read_text() == malformed
    assert json.loads(config_file.read_text())["duckdb_path"] == "/data/new.duckdb"
    assert sorted(config_dir.iterdir()) == sorted(
        [config_dir / "config.json", config_dir / "config.json.bak"]
    )
    assert caught == []


def test_save_config_makes_no_backup_of_a_file_it_could_read(tmp_path, monkeypatch):
    """A usable file was read by the caller that is overwriting it: no copy is owed."""
    config_dir, config_file = _point_at_tmp(monkeypatch, tmp_path)
    config_file.write_text(json.dumps({"duckdb_path": "/data/old.duckdb"}))

    config.save_config({"duckdb_path": "/data/new.duckdb"})

    assert list(config_dir.iterdir()) == [config_file]


def test_save_config_warns_when_the_backup_cannot_be_written(tmp_path, monkeypatch):
    """Losing the copy is worth a line; it is not worth leaving the setup half-done."""
    config_dir, config_file = _point_at_tmp(monkeypatch, tmp_path)
    config_file.write_text("{ this is not valid json")

    original_write_bytes = type(config_file).write_bytes

    def _deny(self, *_args, **_kwargs):
        if self.name == "config.json.bak":
            raise OSError(13, "Permission denied")
        return original_write_bytes(self, *_args, **_kwargs)

    monkeypatch.setattr(type(config_file), "write_bytes", _deny, raising=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.save_config({"duckdb_path": "/data/new.duckdb"})

    assert any("Could not back up" in str(w.message) for w in caught)
    assert json.loads(config_file.read_text())["duckdb_path"] == "/data/new.duckdb"
    assert not (config_dir / "config.json.bak").exists()


def test_save_config_writes_when_the_old_file_cannot_be_read(tmp_path, monkeypatch):
    """An unreadable old file cannot be copied, and must not stop the write either."""
    config_dir, config_file = _point_at_tmp(monkeypatch, tmp_path)

    def _deny(self, *_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(type(config_file), "read_bytes", _deny, raising=True)

    config.save_config({"duckdb_path": "/data/new.duckdb"})

    assert json.loads(config_file.read_text())["duckdb_path"] == "/data/new.duckdb"
    assert list(config_dir.iterdir()) == [config_file]


def test_get_duckdb_path_prefers_env_override(monkeypatch):
    monkeypatch.setenv("NFL_MCP_DB_PATH", "~/.nfl-env.duckdb")
    got = config.get_duckdb_path()
    assert got == (Path.home() / ".nfl-env.duckdb")


def test_get_duckdb_path_uses_config_when_env_missing(monkeypatch):
    monkeypatch.delenv("NFL_MCP_DB_PATH", raising=False)
    monkeypatch.setattr(config, "load_config", lambda: {"duckdb_path": "~/.nfl-config.duckdb"})
    got = config.get_duckdb_path()
    assert got == (Path.home() / ".nfl-config.duckdb")


def test_config_exists_reflects_file_presence(tmp_path, monkeypatch):
    config_file = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)
    assert config.config_exists() is False
    config_file.write_text("{}")
    assert config.config_exists() is True


def test_config_file_is_usable_covers_every_state(tmp_path, monkeypatch):
    """'Exists' and 'can be used' are different answers, and doctor needs the second one."""
    _, config_file = _point_at_tmp(monkeypatch, tmp_path)

    assert config.config_file_is_usable() is True  # no file at all

    config_file.write_text(json.dumps({"duckdb_path": "/data/nflread.duckdb"}))
    assert config.config_file_is_usable() is True

    config_file.write_text("{ half an edit")
    assert config.config_file_is_usable() is False

    config_file.write_text("[1, 2]")
    assert config.config_file_is_usable() is False

    def _deny(self, *_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(type(config_file), "read_bytes", _deny, raising=True)
    assert config.config_file_is_usable() is False
