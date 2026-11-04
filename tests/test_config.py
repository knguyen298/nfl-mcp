"""Tests for configuration loading behavior."""

import json
import warnings
from pathlib import Path

import pytest

import nfl_mcp.config as config


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


@pytest.mark.parametrize("text,problem", [
    ("{ this is not valid json ", "Expecting property name"),
    ("[]", "it holds a JSON list, not an object"),
])
def test_a_bad_config_is_reported_and_left_where_it_is(tmp_path, monkeypatch, text, problem):
    """A read once renamed a malformed file to config.json.broken, and a list raised TypeError."""
    config_dir = tmp_path / "nfl-mcp"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text(text)

    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = config.load_config()

    assert loaded == {**config._DEFAULT_CONFIG}
    assert sorted(p.name for p in config_dir.iterdir()) == ["config.json"]
    assert config_file.read_text() == text
    assert any(problem in str(w.message) and "nfl-mcp init" in str(w.message) for w in caught)


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


def test_a_relative_config_path_is_read_from_the_config_directory(tmp_path, monkeypatch):
    """Claude Desktop usually starts the server from /, where data/x.duckdb means nothing."""
    monkeypatch.delenv("NFL_MCP_DB_PATH", raising=False)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "load_config", lambda: {"duckdb_path": "data/x.duckdb"})
    assert config.get_duckdb_path() == tmp_path / "data" / "x.duckdb"


def test_a_relative_env_path_is_read_from_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NFL_MCP_DB_PATH", "x.duckdb")
    assert config.get_duckdb_path() == tmp_path / "x.duckdb"
