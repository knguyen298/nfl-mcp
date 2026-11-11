"""Unit tests for CLI commands — no database or network required."""

import contextlib
import json
import os

import duckdb
import pytest
from click.testing import CliRunner

import nfl_mcp.cli as cli
import nfl_mcp.config as config
from nfl_mcp.cli import main, _season_window_label
from nfl_mcp.registry import ALL_DATASETS, DEFAULT_DATASETS, REGISTRY


@pytest.fixture
def runner():
    return CliRunner()


# ── client-config helpers aimed at a tmp directory ──────────────────────────────

def _claude_desktop_target(monkeypatch, tmp_path):
    """Point _configure_claude_desktop at tmp_path; return (file path, helper)."""
    path = tmp_path / "claude_desktop_config.json"
    monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: path)
    return path, cli._configure_claude_desktop


def _claude_code_target(monkeypatch, tmp_path):
    """Point _configure_claude_code at tmp_path; return (file path, helper)."""
    (tmp_path / ".claude").mkdir(exist_ok=True)
    path = tmp_path / ".claude" / "mcp.json"
    monkeypatch.setattr("nfl_mcp.cli.Path.home", lambda: tmp_path)
    return path, cli._configure_claude_code


def _vscode_target(monkeypatch, tmp_path):
    """Point _configure_vscode at tmp_path; return (file path, helper)."""
    (tmp_path / ".vscode").mkdir(exist_ok=True)
    path = tmp_path / ".vscode" / "mcp.json"
    monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)
    return path, cli._configure_vscode


CLIENT_TARGETS = {
    "claude-desktop": (_claude_desktop_target, "mcpServers"),
    "claude-code": (_claude_code_target, "mcpServers"),
    "vscode": (_vscode_target, "servers"),
}

# Files that parse (or do not) in ways the writer must refuse to overwrite.
UNUSABLE_CONFIG = {
    "top_level_array": lambda key: json.dumps(["mcp", "config"]),
    "top_level_string": lambda key: json.dumps("mcp config"),
    "servers_array": lambda key: json.dumps({key: ["not", "an", "object"], "theme": "dark"}),
    "servers_null": lambda key: json.dumps({key: None}),
    "servers_string": lambda key: json.dumps({key: "oops"}),
}


# ── serve ───────────────────────────────────────────────────────────────────────

class TestServe:
    @pytest.fixture(autouse=True)
    def _isolated_config(self, monkeypatch, tmp_path):
        """serve reads config.json, so point it at defaults unless a test says otherwise."""
        # Merge: serve now checks the schema of the database file first (main). A path with
        # no file skips that check, so these tests never open a real database.
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(tmp_path / "absent.duckdb"))
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {})

    def test_serve_starts_uvicorn(self, runner, monkeypatch):
        called = {}

        def fake_uvicorn_run(app, host, port):
            called["host"] = host
            called["port"] = port

        monkeypatch.setattr("nfl_mcp.cli.uvicorn.run", fake_uvicorn_run)

        result = runner.invoke(main, ["serve", "--host", "127.0.0.1", "--port", "9000"])
        assert result.exit_code == 0
        assert called.get("host") == "127.0.0.1"
        assert called.get("port") == 9000

    def test_serve_default_host_is_localhost(self, runner, monkeypatch):
        called = {}
        monkeypatch.setattr("nfl_mcp.cli.uvicorn.run", lambda app, host, port: called.update(host=host))
        runner.invoke(main, ["serve"])
        assert called.get("host") == "127.0.0.1"

    def test_serve_prints_localhost_url_when_binding_all_interfaces(self, runner, monkeypatch):
        monkeypatch.setattr("nfl_mcp.cli.uvicorn.run", lambda app, host, port: None)
        result = runner.invoke(main, ["serve", "--host", "0.0.0.0", "--port", "8000"])
        assert result.exit_code == 0
        assert "localhost:8000" in result.output
        assert "all interfaces" in result.output

    def test_serve_reads_the_address_from_the_config_file(self, runner, monkeypatch):
        called = {}
        monkeypatch.setattr(
            "nfl_mcp.cli.uvicorn.run", lambda app, host, port: called.update(host=host, port=port)
        )
        monkeypatch.setattr(
            "nfl_mcp.config.load_config",
            lambda: {"serve_host": "0.0.0.0", "serve_port": 9001},
        )

        result = runner.invoke(main, ["serve"])

        assert result.exit_code == 0, result.output
        assert called == {"host": "0.0.0.0", "port": 9001}
        assert "listening on http://localhost:9001/mcp" in result.output

    def test_a_command_line_option_beats_the_config_file(self, runner, monkeypatch):
        called = {}
        monkeypatch.setattr(
            "nfl_mcp.cli.uvicorn.run", lambda app, host, port: called.update(host=host, port=port)
        )
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {"serve_port": 9001})

        runner.invoke(main, ["serve", "--port", "9100"])

        assert called == {"host": "127.0.0.1", "port": 9100}

    def test_serve_help_documents_the_keys_it_reads(self, runner):
        """The config keys were never written down; --help is where a user looks."""
        out = runner.invoke(main, ["serve", "--help"]).output
        assert "serve_host" in out
        assert "serve_port" in out

    @pytest.mark.parametrize("stored,expected_url,expected_bind", [
        ({}, "http://localhost:8000/mcp", ("127.0.0.1", 8000)),
        ({"serve_host": "0.0.0.0"}, "http://localhost:8000/mcp", ("0.0.0.0", 8000)),
        ({"serve_port": 9001}, "http://localhost:9001/mcp", ("127.0.0.1", 9001)),
        ({"serve_host": "192.168.4.20", "serve_port": 9001},
         "http://192.168.4.20:9001/mcp", ("192.168.4.20", 9001)),
    ])
    def test_serve_and_setup_client_resolve_the_same_address(
        self, runner, monkeypatch, tmp_path, stored, expected_url, expected_bind
    ):
        """The URL a client is given is the address the server binds, from one config file."""
        called = {}
        monkeypatch.setattr(
            "nfl_mcp.cli.uvicorn.run", lambda app, host, port: called.update(host=host, port=port)
        )
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: dict(stored))
        monkeypatch.setattr("nfl_mcp.config.save_config", lambda cfg: tmp_path / "config.json")
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)

        served = runner.invoke(main, ["serve"])
        assert served.exit_code == 0, served.output
        assert (called["host"], called["port"]) == expected_bind
        assert f"listening on {expected_url}" in served.output

        configured = runner.invoke(main, ["setup-client", "--client", "vscode"])
        assert configured.exit_code == 0, configured.output
        written = json.loads((tmp_path / ".vscode" / "mcp.json").read_text())
        assert written["servers"]["nfl"]["url"] == expected_url


# ── ingest --list ──────────────────────────────────────────────────────────────

class TestIngestList:
    def test_list_exits_zero(self, runner):
        result = runner.invoke(main, ["ingest", "--list"])
        assert result.exit_code == 0

    def test_list_shows_all_dataset_ids(self, runner):
        result = runner.invoke(main, ["ingest", "--list"])
        for ds_id in ALL_DATASETS:
            assert ds_id in result.output

    def test_list_marks_default_datasets(self, runner):
        result = runner.invoke(main, ["ingest", "--list"])
        assert "[default]" in result.output

    def test_list_marks_opt_in_datasets(self, runner):
        # Datasets that reach an outside publisher are not in 'default'; the listing must say so.
        result = runner.invoke(main, ["ingest", "--list"])
        lines = result.output.splitlines()
        row = next((index for index, line in enumerate(lines) if "espn_records" in line), None)
        assert row is not None, "espn_records missing from --list"
        assert "[opt-in]" in lines[row]
        assert "1970–1998" in "\n".join(lines[row:row + 2])  # window prints on the next line

    def test_list_shows_per_dataset_season_windows(self, runner):
        result = runner.invoke(main, ["ingest", "--list"])
        assert "1999–present" in result.output   # pbp / schedules / player_stats
        assert "2022–present" in result.output   # ftn_charting

    def test_list_flags_when_the_default_start_clips_a_dataset(self, runner):
        # rosters publishes from 1920 but the default backfill starts in 1966, so the
        # listing must not imply the pre-1966 seasons load on their own.
        result = runner.invoke(main, ["ingest", "--list"])
        assert "1920–present (from 1966 by default)" in result.output

    @pytest.mark.parametrize("low,high,expected", [
        (1999, None, "1999–present"),
        (2016, 2024, "2016–2024"),
        (None, 2024, "earliest–2024"),
        (None, None, "earliest–present"),
    ])
    def test_season_window_label(self, low, high, expected):
        assert _season_window_label(low, high) == expected

    def test_list_does_not_start_ingest(self, runner, monkeypatch):
        called = []
        monkeypatch.setattr("nfl_mcp.ingest.run_ingest_datasets",
                            lambda *a, **kw: called.append(True))
        runner.invoke(main, ["ingest", "--list"])
        assert not called


# ── ingest --dataset resolution ───────────────────────────────────────────────

class TestDatasetResolution:
    def test_unknown_dataset_exits_nonzero(self, runner):
        result = runner.invoke(main, ["ingest", "--dataset", "not_a_real_dataset",
                                      "--skip-views"])
        assert result.exit_code != 0

    def test_unknown_dataset_error_message(self, runner):
        result = runner.invoke(main, ["ingest", "--dataset", "not_a_real_dataset",
                                      "--skip-views"])
        assert "Unknown dataset" in result.output
        assert "not_a_real_dataset" in result.output

    def test_all_resolves_to_every_dataset(self, runner, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "nfl_mcp.ingest.run_ingest_datasets",
            lambda dataset_ids, **kw: captured.update({"ids": dataset_ids}),
        )
        runner.invoke(main, ["ingest", "--dataset", "all"])
        assert set(captured["ids"]) == set(ALL_DATASETS)

    def test_default_resolves_to_default_datasets(self, runner, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "nfl_mcp.ingest.run_ingest_datasets",
            lambda dataset_ids, **kw: captured.update({"ids": dataset_ids}),
        )
        runner.invoke(main, ["ingest", "--dataset", "default"])
        assert set(captured["ids"]) == set(DEFAULT_DATASETS)

    def test_specific_dataset_is_passed_through(self, runner, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "nfl_mcp.ingest.run_ingest_datasets",
            lambda dataset_ids, **kw: captured.update({"ids": dataset_ids}),
        )
        runner.invoke(main, ["ingest", "--dataset", "schedules"])
        assert captured["ids"] == ["schedules"]

    def test_multiple_datasets_accumulate(self, runner, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "nfl_mcp.ingest.run_ingest_datasets",
            lambda dataset_ids, **kw: captured.update({"ids": dataset_ids}),
        )
        runner.invoke(main, ["ingest", "--dataset", "pbp", "--dataset", "injuries"])
        assert set(captured["ids"]) == {"pbp", "injuries"}

    def test_deduplication(self, runner, monkeypatch):
        """Specifying the same dataset twice should not duplicate it."""
        captured = {}
        monkeypatch.setattr(
            "nfl_mcp.ingest.run_ingest_datasets",
            lambda dataset_ids, **kw: captured.update({"ids": dataset_ids}),
        )
        runner.invoke(main, ["ingest", "--dataset", "schedules", "--dataset", "schedules"])
        assert captured["ids"].count("schedules") == 1

    def test_pre1999_adds_the_external_records_pull(self, runner, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "nfl_mcp.ingest.run_ingest_datasets",
            lambda dataset_ids, **kw: captured.update({"ids": dataset_ids}),
        )
        runner.invoke(main, ["ingest", "--dataset", "schedules", "--pre1999"])
        assert captured["ids"] == ["schedules", "espn_records"]

    def test_pre1999_does_not_duplicate(self, runner, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "nfl_mcp.ingest.run_ingest_datasets",
            lambda dataset_ids, **kw: captured.update({"ids": dataset_ids}),
        )
        runner.invoke(main, ["ingest", "--dataset", "all", "--pre1999"])
        assert captured["ids"].count("espn_records") == 1

    def test_default_ingest_leaves_the_external_pull_out(self, runner, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "nfl_mcp.ingest.run_ingest_datasets",
            lambda dataset_ids, **kw: captured.update({"ids": dataset_ids}),
        )
        runner.invoke(main, ["ingest", "--dataset", "default"])
        assert "espn_records" not in captured["ids"]


# ── ingest season range validation ────────────────────────────────────────────

class TestSeasonRangeValidation:
    def test_start_greater_than_end_exits_nonzero(self, runner):
        result = runner.invoke(main, ["ingest", "--start", "2025", "--end", "2020"])
        assert result.exit_code != 0

    def test_start_greater_than_end_error_message(self, runner):
        result = runner.invoke(main, ["ingest", "--start", "2025", "--end", "2020"])
        assert "start" in result.output.lower() or "end" in result.output.lower()

    def test_current_season_is_accepted_and_next_year_is_rejected(self, runner, monkeypatch):
        from nfl_mcp.cli import CURRENT_SEASON
        captured = {}
        monkeypatch.setattr(
            "nfl_mcp.ingest.run_ingest_datasets",
            lambda **kw: captured.update(kw),
        )
        ok = runner.invoke(main, ["ingest", "--start", str(CURRENT_SEASON), "--end", str(CURRENT_SEASON)])
        assert ok.exit_code == 0, ok.output
        assert captured["end"] == CURRENT_SEASON

        bad = runner.invoke(main, ["ingest", "--end", str(CURRENT_SEASON + 1)])
        assert bad.exit_code != 0

    def test_equal_start_end_is_valid(self, runner, monkeypatch):
        monkeypatch.setattr(
            "nfl_mcp.ingest.run_ingest_datasets",
            lambda *a, **kw: None,
        )
        result = runner.invoke(main, ["ingest", "--dataset", "schedules",
                                      "--start", "2024", "--end", "2024"])
        assert result.exit_code == 0

    def test_no_start_end_passes_none(self, runner, monkeypatch):
        """Omitting --start/--end should pass start=None, end=None (bulk mode)."""
        captured = {}
        monkeypatch.setattr(
            "nfl_mcp.ingest.run_ingest_datasets",
            lambda dataset_ids, start, end, **kw: captured.update(
                {"start": start, "end": end}
            ),
        )
        runner.invoke(main, ["ingest", "--dataset", "schedules"])
        assert captured.get("start") is None
        assert captured.get("end") is None


# ── init / setup-client routing ────────────────────────────────────────────────

class TestInitAndSetupClient:
    def test_init_start_greater_than_end_exits_nonzero(self, runner):
        result = runner.invoke(main, ["init", "--start", "2025", "--end", "2024", "--skip-ingest"])
        assert result.exit_code != 0

    def test_init_skip_ingest_saves_config_and_skips_client_setup(self, runner, monkeypatch, tmp_path):
        saved = {}
        setup_calls = []

        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {})
        def _save_config(cfg):
            saved["config"] = cfg
            return tmp_path / "config.json"
        monkeypatch.setattr(
            "nfl_mcp.config.save_config",
            _save_config,
        )
        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", lambda cfg: setup_calls.append(cfg))

        # Prompts: change DB path? -> no ; configure client? -> no ; start server? -> no
        result = runner.invoke(
            main,
            ["init", "--start", "2024", "--end", "2024", "--skip-ingest"],
            input="n\nn\nn\n",
        )

        assert result.exit_code == 0
        assert "duckdb_path" in saved["config"]
        assert setup_calls == []

    def test_init_uses_custom_db_path_prompt_value(self, runner, monkeypatch, tmp_path):
        saved = {}
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {})
        def _save_config(cfg):
            saved["config"] = cfg
            return tmp_path / "config.json"

        monkeypatch.setattr("nfl_mcp.config.save_config", _save_config)
        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", lambda cfg: None)

        result = runner.invoke(
            main,
            ["init", "--skip-ingest"],
            input="y\n/tmp/custom-nfl.duckdb\nn\nn\n",
        )

        assert result.exit_code == 0
        assert saved["config"]["duckdb_path"] == "/tmp/custom-nfl.duckdb"

    def test_init_keeps_a_saved_custom_db_path_as_the_prompt_default(self, runner, monkeypatch, tmp_path):
        """Re-running init and pressing Enter must not reset a custom path."""
        saved = {}
        monkeypatch.setattr(
            "nfl_mcp.config.load_config",
            lambda: {"duckdb_path": "/data/nfl/nflread.duckdb"},
        )
        def _save_config(cfg):
            saved["config"] = cfg
            return tmp_path / "config.json"

        monkeypatch.setattr("nfl_mcp.config.save_config", _save_config)
        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", lambda cfg: None)

        # Prompts: change DB path? -> no (accept the saved path) ; client? -> no ; server? -> no
        result = runner.invoke(
            main,
            ["init", "--skip-ingest"],
            input="n\nn\nn\n",
        )

        assert result.exit_code == 0
        assert saved["config"]["duckdb_path"] == "/data/nfl/nflread.duckdb"
        assert "Database will be stored at /data/nfl/nflread.duckdb" in result.output

    def test_init_offers_the_saved_path_when_the_user_switches_to_a_custom_one(self, runner, monkeypatch, tmp_path):
        """The custom-path prompt defaults to the saved path too."""
        saved = {}
        monkeypatch.setattr(
            "nfl_mcp.config.load_config",
            lambda: {"duckdb_path": "/data/nfl/nflread.duckdb"},
        )
        def _save_config(cfg):
            saved["config"] = cfg
            return tmp_path / "config.json"

        monkeypatch.setattr("nfl_mcp.config.save_config", _save_config)
        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", lambda cfg: None)

        # Prompts: change DB path? -> yes ; Path -> accept the default shown
        result = runner.invoke(
            main,
            ["init", "--skip-ingest"],
            input="y\n\nn\nn\n",
        )

        assert result.exit_code == 0
        assert saved["config"]["duckdb_path"] == "/data/nfl/nflread.duckdb"
        assert "[/data/nfl/nflread.duckdb]" in result.output

    def test_init_runs_ingest_when_confirmed(self, runner, monkeypatch, tmp_path):
        captured = {}

        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {})
        monkeypatch.setattr("nfl_mcp.config.save_config", lambda cfg: tmp_path / "config.json")
        monkeypatch.setattr(
            "nfl_mcp.ingest.run_ingest_datasets",
            lambda dataset_ids, start, end, fresh, skip_views: captured.update(
                {
                    "dataset_ids": dataset_ids,
                    "start": start,
                    "end": end,
                    "fresh": fresh,
                    "skip_views": skip_views,
                }
            ),
        )
        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", lambda cfg: None)

        # Prompts: change DB path? -> no ; ingest now? -> default yes ; configure client? -> no ; start server? -> no
        result = runner.invoke(
            main,
            ["init", "--start", "2024", "--end", "2024"],
            input="n\n\nn\nn\n",
        )

        assert result.exit_code == 0
        assert captured["dataset_ids"] == DEFAULT_DATASETS
        assert captured["start"] == 2024
        assert captured["end"] == 2024
        assert captured["fresh"] is False
        assert captured["skip_views"] is False

    def test_init_skips_ingest_when_user_declines(self, runner, monkeypatch, tmp_path):
        ingest_calls = []
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {})
        monkeypatch.setattr("nfl_mcp.config.save_config", lambda cfg: tmp_path / "config.json")
        monkeypatch.setattr("nfl_mcp.ingest.run_ingest_datasets", lambda *a, **k: ingest_calls.append(True))
        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", lambda cfg: None)

        result = runner.invoke(
            main,
            ["init", "--start", "2024", "--end", "2024"],
            input="n\nn\nn\nn\n",
        )

        assert result.exit_code == 0
        assert ingest_calls == []

    def test_init_configures_client_when_confirmed(self, runner, monkeypatch, tmp_path):
        setup_calls = []
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {})
        monkeypatch.setattr("nfl_mcp.config.save_config", lambda cfg: tmp_path / "config.json")
        monkeypatch.setattr(
            "nfl_mcp.cli._setup_client_interactive",
            lambda cfg: setup_calls.append(cfg) or True,
        )

        result = runner.invoke(
            main,
            ["init", "--skip-ingest"],
            input="n\ny\nn\n",
        )

        assert result.exit_code == 0
        assert len(setup_calls) == 1
        assert "Setup complete!" in result.output

    def test_init_does_not_claim_success_when_a_client_was_not_configured(self, runner, monkeypatch, tmp_path):
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {})
        monkeypatch.setattr("nfl_mcp.config.save_config", lambda cfg: tmp_path / "config.json")
        def _client_step_failed(cfg):
            print("    ✗ Left /home/u/.claude/mcp.json unchanged.")
            return False

        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", _client_step_failed)
        monkeypatch.setattr("nfl_mcp.cli.uvicorn.run", lambda app, host, port: None)

        # Prompts: change DB path? -> no ; configure client? -> yes ; start server? -> no
        result = runner.invoke(main, ["init", "--skip-ingest"], input="n\ny\nn\n")

        assert result.exit_code == 0
        assert "/home/u/.claude/mcp.json unchanged" in result.output
        assert "a client config was left unchanged" in result.output
        assert "nfl-mcp setup-client" in result.output
        assert "Setup complete!" not in result.output

    def test_init_starts_server_when_confirmed(self, runner, monkeypatch, tmp_path):
        uvicorn_calls = []
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {})
        monkeypatch.setattr("nfl_mcp.config.save_config", lambda cfg: tmp_path / "config.json")
        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", lambda cfg: None)
        monkeypatch.setattr("nfl_mcp.cli.uvicorn.run", lambda app, host, port: uvicorn_calls.append((host, port)))

        # Prompts: change DB path? -> no ; configure client? -> no ; start server? -> yes
        result = runner.invoke(
            main,
            ["init", "--skip-ingest"],
            input="n\nn\ny\n",
        )

        assert result.exit_code == 0
        assert uvicorn_calls == [("127.0.0.1", 8000)]

    def test_init_leaves_the_default_address_out_of_the_config_file(self, runner, monkeypatch, tmp_path):
        saved = {}
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {"duckdb_path": "/tmp/db.duckdb"})
        monkeypatch.setattr(
            "nfl_mcp.config.save_config", lambda cfg: saved.update(cfg) or tmp_path / "config.json"
        )
        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", lambda cfg: True)

        result = runner.invoke(main, ["init", "--skip-ingest"], input="n\nn\nn\n")

        assert result.exit_code == 0, result.output
        assert "serve_host" not in saved
        assert "serve_port" not in saved

    def test_init_saves_a_non_default_address_and_uses_it_everywhere(
        self, runner, monkeypatch, tmp_path
    ):
        """One answer for the address: saved, given to the client, and bound by the wizard."""
        saved = {}
        given_to_client = {}
        uvicorn_calls = []
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {})
        monkeypatch.setattr(
            "nfl_mcp.config.save_config", lambda cfg: saved.update(cfg) or tmp_path / "config.json"
        )
        monkeypatch.setattr(
            "nfl_mcp.cli._setup_client_interactive", lambda cfg: given_to_client.update(cfg) or True
        )
        monkeypatch.setattr(
            "nfl_mcp.cli.uvicorn.run", lambda app, host, port: uvicorn_calls.append((host, port))
        )

        # Prompts: change DB path? -> no ; configure client? -> yes ; start server? -> yes
        result = runner.invoke(
            main,
            ["init", "--skip-ingest", "--host", "0.0.0.0", "--port", "9001"],
            input="n\ny\ny\n",
        )

        assert result.exit_code == 0, result.output
        assert saved["serve_host"] == "0.0.0.0"
        assert saved["serve_port"] == 9001
        assert given_to_client["serve_port"] == 9001
        assert cli._build_server_config(given_to_client) == {"url": "http://localhost:9001/mcp"}
        assert uvicorn_calls == [("0.0.0.0", 9001)]
        assert "Starting server on http://localhost:9001/mcp" in result.output

    def test_setup_client_saves_the_address_it_gives_the_client(self, runner, monkeypatch, tmp_path):
        """A non-default port is written down, so the next 'nfl-mcp serve' binds that port."""
        saved = {}
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {"duckdb_path": "/tmp/db.duckdb"})
        monkeypatch.setattr(
            "nfl_mcp.config.save_config",
            lambda cfg: saved.update(cfg) or tmp_path / "config.json",
        )
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)

        result = runner.invoke(main, ["setup-client", "--client", "vscode", "--port", "9001"])

        assert result.exit_code == 0, result.output
        assert saved["serve_host"] == "127.0.0.1"
        assert saved["serve_port"] == 9001
        written = json.loads((tmp_path / ".vscode" / "mcp.json").read_text())
        assert written["servers"]["nfl"]["url"] == "http://localhost:9001/mcp"

    def test_setup_client_writes_no_config_for_the_default_address(self, runner, monkeypatch, tmp_path):
        saves = []
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {"duckdb_path": "/tmp/db.duckdb"})
        monkeypatch.setattr("nfl_mcp.config.save_config", lambda cfg: saves.append(cfg))
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)

        result = runner.invoke(main, ["setup-client", "--client", "vscode"])

        assert result.exit_code == 0, result.output
        assert saves == []
        written = json.loads((tmp_path / ".vscode" / "mcp.json").read_text())
        assert written["servers"]["nfl"]["url"] == "http://localhost:8000/mcp"

    def test_setup_client_help_documents_the_address_options(self, runner):
        out = runner.invoke(main, ["setup-client", "--help"]).output
        assert "--host" in out
        assert "--port" in out

    def test_setup_client_routes_auto(self, runner, monkeypatch):
        called = []
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {"duckdb_path": "/tmp/db.duckdb"})
        monkeypatch.setattr(
            "nfl_mcp.cli._setup_client_interactive",
            lambda cfg: called.append(("auto", cfg)) or True,
        )
        result = runner.invoke(main, ["setup-client", "--client", "auto"])
        assert result.exit_code == 0
        assert called and called[0][0] == "auto"

    def test_setup_client_routes_claude_desktop(self, runner, monkeypatch):
        called = []
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {"duckdb_path": "/tmp/db.duckdb"})
        monkeypatch.setattr(
            "nfl_mcp.cli._configure_claude_desktop",
            lambda cfg: called.append(("claude", cfg)) or True,
        )
        result = runner.invoke(main, ["setup-client", "--client", "claude-desktop"])
        assert result.exit_code == 0
        assert called and called[0][0] == "claude"

    def test_setup_client_routes_claude_code(self, runner, monkeypatch):
        called = []
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {"duckdb_path": "/tmp/db.duckdb"})
        monkeypatch.setattr(
            "nfl_mcp.cli._configure_claude_code",
            lambda cfg: called.append(("claude-code", cfg)) or True,
        )
        result = runner.invoke(main, ["setup-client", "--client", "claude-code"])
        assert result.exit_code == 0
        assert called and called[0][0] == "claude-code"

    def test_setup_client_routes_vscode(self, runner, monkeypatch):
        called = []
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {"duckdb_path": "/tmp/db.duckdb"})
        monkeypatch.setattr(
            "nfl_mcp.cli._configure_vscode",
            lambda cfg: called.append(("vscode", cfg)) or True,
        )
        result = runner.invoke(main, ["setup-client", "--client", "vscode"])
        assert result.exit_code == 0
        assert called and called[0][0] == "vscode"

    @pytest.mark.parametrize(
        "client,helper",
        [
            ("claude-desktop", "_configure_claude_desktop"),
            ("claude-code", "_configure_claude_code"),
            ("vscode", "_configure_vscode"),
            ("auto", "_setup_client_interactive"),
        ],
    )
    def test_setup_client_exits_nonzero_when_a_client_was_not_configured(
        self, runner, monkeypatch, client, helper
    ):
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {"duckdb_path": "/tmp/db.duckdb"})
        monkeypatch.setattr("nfl_mcp.cli." + helper, lambda cfg: False)
        result = runner.invoke(main, ["setup-client", "--client", client])
        assert result.exit_code == 1
        assert "left unchanged" in result.output


# ── CLI helpers ─────────────────────────────────────────────────────────────────

class TestServeAddressHelpers:
    """One place decides the address, so `serve` and `setup-client` cannot disagree."""

    @pytest.mark.parametrize("stored,expected", [
        ({}, "127.0.0.1"),
        ({"serve_host": None}, "127.0.0.1"),
        ({"serve_host": "0.0.0.0"}, "0.0.0.0"),
    ])
    def test_host_precedence(self, stored, expected):
        assert cli._serve_settings(stored)[0] == expected

    def test_a_command_line_host_wins_over_the_config_file(self):
        assert cli._serve_settings({"serve_host": "1.2.3.4"}, host="10.0.0.5") == (
            "10.0.0.5", 8000,
        )

    @pytest.mark.parametrize("stored,expected", [
        ({}, 8000),
        ({"serve_port": None}, 8000),
        ({"serve_port": "oops"}, 8000),
        ({"serve_port": 0}, 8000),
        ({"serve_port": 70000}, 8000),
        ({"serve_port": 9001}, 9001),
        ({"serve_port": "9001"}, 9001),
    ])
    def test_port_precedence(self, stored, expected):
        """config.json is a file a person edits, so a typo must not crash a command."""
        assert cli._serve_settings(stored)[1] == expected

    def test_a_command_line_port_wins_over_the_config_file(self):
        assert cli._serve_settings({"serve_port": 9001}, port=9100)[1] == 9100

    @pytest.mark.parametrize("host,port,expected", [
        ("127.0.0.1", 8000, "http://localhost:8000/mcp"),
        ("0.0.0.0", 9001, "http://localhost:9001/mcp"),
        ("192.168.4.20", 9001, "http://192.168.4.20:9001/mcp"),
        ("localhost", 8000, "http://localhost:8000/mcp"),
    ])
    def test_client_url_shape(self, host, port, expected):
        assert cli._client_url(host, port) == expected

    def test_ingest_command_names_the_seasons_it_was_asked_for(self):
        assert cli._ingest_command(2020, 2024) == "nfl-mcp ingest --start 2020 --end 2024"


class TestCliHelpers:
    def test_setup_client_interactive_configures_selected_clients(self, monkeypatch, tmp_path):
        configured = []
        # 3 prompts: Claude Desktop=yes, Claude Code=no, VS Code=no
        answers = iter([True, False, False])

        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: tmp_path / "claude_desktop_config.json")
        monkeypatch.setattr("nfl_mcp.cli.click.confirm", lambda *args, **kwargs: next(answers))
        monkeypatch.setattr("nfl_mcp.cli._configure_claude_desktop", lambda cfg: configured.append("claude") or True)
        monkeypatch.setattr("nfl_mcp.cli._configure_claude_code", lambda cfg: configured.append("claude-code") or True)
        monkeypatch.setattr("nfl_mcp.cli._configure_vscode", lambda cfg: configured.append("vscode") or True)

        assert cli._setup_client_interactive({"duckdb_path": "/tmp/db.duckdb"}) is True
        assert configured == ["claude"]

    def test_setup_client_interactive_configures_claude_code_branch(self, monkeypatch, tmp_path):
        configured = []
        # No Claude Desktop; 2 prompts: Claude Code=yes, VS Code=no
        answers = iter([True, False])
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.click.confirm", lambda *args, **kwargs: next(answers))
        monkeypatch.setattr("nfl_mcp.cli._configure_claude_code", lambda cfg: configured.append("claude-code") or True)
        monkeypatch.setattr("nfl_mcp.cli._configure_vscode", lambda cfg: configured.append("vscode") or True)

        assert cli._setup_client_interactive({"duckdb_path": "/tmp/db.duckdb"}) is True
        assert configured == ["claude-code"]

    def test_setup_client_interactive_configures_vscode_branch(self, monkeypatch):
        configured = []
        # No Claude Desktop; 2 prompts: Claude Code=no, VS Code=yes
        answers = iter([False, True])
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.click.confirm", lambda *args, **kwargs: next(answers))
        monkeypatch.setattr("nfl_mcp.cli._configure_claude_code", lambda cfg: configured.append("claude-code") or True)
        monkeypatch.setattr("nfl_mcp.cli._configure_vscode", lambda cfg: configured.append("vscode") or True)
        monkeypatch.setattr("nfl_mcp.cli._configure_claude_desktop", lambda cfg: configured.append("claude") or True)

        assert cli._setup_client_interactive({"duckdb_path": "/tmp/db.duckdb"}) is True
        assert configured == ["vscode"]

    def test_setup_client_interactive_reports_a_client_that_was_not_written(self, monkeypatch):
        # No Claude Desktop; both chosen clients refuse to overwrite a bad file.
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.click.confirm", lambda *args, **kwargs: True)
        monkeypatch.setattr("nfl_mcp.cli._configure_claude_code", lambda cfg: False)
        monkeypatch.setattr("nfl_mcp.cli._configure_vscode", lambda cfg: True)

        assert cli._setup_client_interactive({"duckdb_path": "/tmp/db.duckdb"}) is False

    def test_setup_client_interactive_ignores_a_client_the_user_skipped(self, monkeypatch):
        # No Claude Desktop; 2 prompts: Claude Code=no, VS Code=no.
        answers = iter([False, False])
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.click.confirm", lambda *args, **kwargs: next(answers))
        monkeypatch.setattr("nfl_mcp.cli._configure_claude_code", lambda cfg: False)
        monkeypatch.setattr("nfl_mcp.cli._configure_vscode", lambda cfg: False)

        assert cli._setup_client_interactive({"duckdb_path": "/tmp/db.duckdb"}) is True

    def test_build_server_config_default_url(self):
        result = cli._build_server_config({})
        assert result == {"url": "http://localhost:8000/mcp"}

    def test_build_server_config_custom_host_port(self):
        result = cli._build_server_config({"serve_host": "0.0.0.0", "serve_port": 9000})
        assert result == {"url": "http://localhost:9000/mcp"}

    def test_build_server_config_explicit_host(self):
        result = cli._build_server_config({"serve_host": "192.168.1.10", "serve_port": 9000})
        assert result == {"url": "http://192.168.1.10:9000/mcp"}

    @pytest.mark.parametrize("platform_name,env_var,expected_parts", [
        ("darwin", None, ("Library", "Application Support", "Claude", "claude_desktop_config.json")),
        ("linux", None, (".config", "claude", "claude_desktop_config.json")),
        ("win32", "APPDATA", ("Claude", "claude_desktop_config.json")),
    ])
    def test_claude_desktop_config_path_platforms(self, monkeypatch, tmp_path, platform_name, env_var, expected_parts):
        monkeypatch.setattr("nfl_mcp.cli.sys.platform", platform_name)
        monkeypatch.setattr("nfl_mcp.cli.Path.home", lambda: tmp_path)
        if env_var:
            appdata = tmp_path / "AppData" / "Roaming"
            monkeypatch.setenv(env_var, str(appdata))
            parent = appdata / expected_parts[0]
        else:
            parent = tmp_path.joinpath(*expected_parts[:-1])
        parent.mkdir(parents=True, exist_ok=True)

        path = cli._claude_desktop_config_path()
        assert path is not None
        assert path.name == expected_parts[-1]

    def test_claude_desktop_config_path_returns_none_if_parent_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr("nfl_mcp.cli.sys.platform", "linux")
        monkeypatch.setattr("nfl_mcp.cli.Path.home", lambda: tmp_path)
        assert cli._claude_desktop_config_path() is None

    def test_configure_claude_desktop_returns_when_path_missing(self, monkeypatch):
        # A client that is not installed is a skip, not a failed run.
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        assert cli._configure_claude_desktop({}) is True

    def test_configure_claude_desktop_leaves_invalid_json_unchanged(self, monkeypatch, tmp_path, capsys):
        path = tmp_path / "claude_desktop_config.json"
        original = '{"mcpServers": {"other": {}} oops'
        path.write_text(original)

        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: path)
        monkeypatch.setattr("nfl_mcp.cli._build_server_config", lambda cfg: {"command": "nfl-mcp", "args": ["serve"]})

        assert cli._configure_claude_desktop({}) is False
        assert path.read_text() == original
        out = capsys.readouterr().out
        assert str(path) in out
        assert "Left" in out and "unchanged" in out
        assert "run 'nfl-mcp setup-client' again" in out

    def test_configure_claude_code_writes_global_mcp_json(self, monkeypatch, tmp_path):
        monkeypatch.setattr("nfl_mcp.cli.Path.home", lambda: tmp_path)
        monkeypatch.setattr("nfl_mcp.cli._build_server_config", lambda cfg: {"url": "http://localhost:8000/mcp"})

        cli._configure_claude_code({})

        path = tmp_path / ".claude" / "mcp.json"
        assert path.exists()
        saved = json.loads(path.read_text())
        assert saved["mcpServers"]["nfl"]["url"] == "http://localhost:8000/mcp"

    def test_configure_claude_code_leaves_invalid_json_unchanged(self, monkeypatch, tmp_path, capsys):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        path = claude_dir / "mcp.json"
        original = "{invalid json"
        path.write_text(original)

        monkeypatch.setattr("nfl_mcp.cli.Path.home", lambda: tmp_path)
        monkeypatch.setattr("nfl_mcp.cli._build_server_config", lambda cfg: {"url": "http://localhost:8000/mcp"})

        assert cli._configure_claude_code({}) is False
        assert path.read_text() == original
        assert str(path) in capsys.readouterr().out

    def test_configure_vscode_leaves_invalid_json_unchanged(self, monkeypatch, tmp_path, capsys):
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir(parents=True, exist_ok=True)
        path = vscode_dir / "mcp.json"
        original = "{invalid json"
        path.write_text(original)

        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)
        monkeypatch.setattr("nfl_mcp.cli._build_server_config", lambda cfg: {"command": "nfl-mcp", "args": ["serve"]})

        assert cli._configure_vscode({}) is False
        assert path.read_text() == original
        assert str(path) in capsys.readouterr().out


# ── client config files are never destroyed ─────────────────────────────────────────────

@pytest.mark.parametrize("client", sorted(CLIENT_TARGETS))
def test_setup_client_creates_a_missing_client_file(client, monkeypatch, tmp_path):
    prepare, servers_key = CLIENT_TARGETS[client]
    path, configure = prepare(monkeypatch, tmp_path)
    monkeypatch.setattr("nfl_mcp.cli._build_server_config", lambda cfg: {"url": "http://localhost:8000/mcp"})

    assert configure({}) is True
    assert json.loads(path.read_text())[servers_key]["nfl"]["url"] == "http://localhost:8000/mcp"


@pytest.mark.parametrize("client", sorted(CLIENT_TARGETS))
def test_setup_client_keeps_every_other_entry_in_a_valid_client_file(client, monkeypatch, tmp_path):
    prepare, servers_key = CLIENT_TARGETS[client]
    path, configure = prepare(monkeypatch, tmp_path)
    path.write_text(json.dumps({servers_key: {"postgres": {"command": "pql"}}, "theme": "dark"}))
    monkeypatch.setattr("nfl_mcp.cli._build_server_config", lambda cfg: {"url": "http://localhost:8000/mcp"})

    assert configure({}) is True
    saved = json.loads(path.read_text())
    assert saved[servers_key]["nfl"]["url"] == "http://localhost:8000/mcp"
    assert saved[servers_key]["postgres"] == {"command": "pql"}
    assert saved["theme"] == "dark"


@pytest.mark.parametrize("client", sorted(CLIENT_TARGETS))
@pytest.mark.parametrize("shape", sorted(UNUSABLE_CONFIG))
def test_setup_client_refuses_to_overwrite_an_unusable_file(client, shape, monkeypatch, tmp_path, capsys):
    """Anything the writer cannot merge into is left byte-for-byte as it was."""
    prepare, servers_key = CLIENT_TARGETS[client]
    path, configure = prepare(monkeypatch, tmp_path)
    original = UNUSABLE_CONFIG[shape](servers_key)
    path.write_text(original)
    monkeypatch.setattr("nfl_mcp.cli._build_server_config", lambda cfg: {"url": "http://localhost:8000/mcp"})

    assert configure({}) is False
    assert path.read_bytes() == original.encode()
    out = capsys.readouterr().out
    assert str(path) in out
    assert "Left" in out and "unchanged" in out
    assert "Fix or remove that file" in out


def test_setup_client_refuses_a_file_with_undecodable_bytes(monkeypatch, tmp_path):
    """Text that is not valid UTF-8 cannot be parsed as JSON either."""
    prepare, _ = CLIENT_TARGETS["vscode"]
    path, configure = prepare(monkeypatch, tmp_path)
    original = b'{"servers": {}}\xff\xfe'
    path.write_bytes(original)

    assert configure({}) is False
    assert path.read_bytes() == original


@pytest.mark.parametrize("client", sorted(CLIENT_TARGETS))
def test_setup_client_leaves_the_old_file_when_a_write_fails(client, monkeypatch, tmp_path):
    """A write that stops partway must not leave a client with half a JSON file.

    os.replace is the last step of the write, so a failure there means the new bytes never
    landed: the old file is still whole and no temp file is left beside it.
    """
    prepare, servers_key = CLIENT_TARGETS[client]
    path, configure = prepare(monkeypatch, tmp_path)
    original = json.dumps({servers_key: {"postgres": {"command": "pql"}}})
    path.write_text(original)

    def _deny(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", _deny)

    with pytest.raises(OSError):
        configure({})

    assert path.read_text() == original
    assert list(path.parent.iterdir()) == [path]


# ── doctor command ──────────────────────────────────────────────────────────────

class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeHealthyDb:
    """Stand-in for duckdb.connect() that answers the two plays queries doctor runs."""

    def execute(self, sql):
        if "COUNT(*) FROM plays" in sql:
            return _FakeCursor((123456,))
        if "current_setting('enable_external_access')" in sql:
            # get_db_connection reads the lock back before it hands out the connection.
            return _FakeCursor((False,))
        return _FakeCursor((2013, 2025))

    def close(self):
        return None


class _FakeEmptyDb:
    """Stand-in for a database file that exists and has no `plays` table."""

    def execute(self, sql):
        raise duckdb.CatalogException(
            "Catalog Error: Table with name \"plays\" does not exist!"
        )

    def close(self):
        return None


class _FakeNoPlaysDb:
    """Stand-in for a database with the `plays` table and no rows in it."""

    def execute(self, sql):
        if "COUNT(*) FROM plays" in sql:
            return _FakeCursor((0,))
        return _FakeCursor((None, None))

    def close(self):
        return None


class TestDoctorCommand:
    @pytest.fixture(autouse=True)
    def _clear_db_env_var(self, monkeypatch):
        """Keep the runner's own NFL_MCP_DB_PATH (CI sets it) out of these tests."""
        monkeypatch.delenv("NFL_MCP_DB_PATH", raising=False)
        # doctor also asks whether config.json can be used; never let it look at the
        # developer's real ~/.nfl-mcp.
        monkeypatch.setattr("nfl_mcp.config.config_file_is_usable", lambda: True)

    def test_doctor_exits_nonzero_when_config_missing(self, runner, monkeypatch):
        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: False)
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 1
        assert "No config file" in result.output

    def test_doctor_says_a_config_file_that_cannot_be_parsed_is_ignored(
        self, runner, monkeypatch, tmp_path
    ):
        """A malformed file stays on disk, so doctor must not call it a finished setup."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{ half an edit, still typing")

        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
        # The real question, asked about the tmp file instead of the developer's home dir.
        monkeypatch.setattr(
            "nfl_mcp.config.config_file_is_usable",
            lambda: config._parse_config(config_file.read_bytes()) is not None,
        )
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(tmp_path / "nflread.duckdb"))

        result = runner.invoke(main, ["doctor"])

        assert result.exit_code == 1
        assert "Config file exists but is not a JSON object" in result.output
        assert "Run 'nfl-mcp init' to write a new one" in result.output
        assert "Config file found" not in result.output
        assert config_file.read_text() == "{ half an edit, still typing"

    def test_doctor_uses_the_db_path_from_the_env_var_without_a_config_file(self, runner, monkeypatch, tmp_path):
        db_path = tmp_path / "nflread.duckdb"
        db_path.write_text("placeholder")

        monkeypatch.setenv("NFL_MCP_DB_PATH", str(db_path))
        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: False)
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)
        monkeypatch.setattr("duckdb.connect", lambda *args, **kwargs: _FakeHealthyDb())

        result = runner.invoke(main, ["doctor"])

        assert result.exit_code == 0, result.output
        assert "the database path comes from NFL_MCP_DB_PATH" in result.output
        assert f"DuckDB file exists ({db_path})" in result.output
        assert "123,456 plays loaded" in result.output
        assert "Some checks failed" not in result.output
        assert "All checks passed!" in result.output

    def test_doctor_happy_path_reports_success(self, runner, monkeypatch, tmp_path):
        db_path = tmp_path / "nflread.duckdb"
        db_path.write_text("placeholder")

        claude_path = tmp_path / "claude_desktop_config.json"
        claude_path.write_text(json.dumps({"mcpServers": {"nfl": {"command": "nfl-mcp"}}}))

        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir(parents=True, exist_ok=True)
        (vscode_dir / "mcp.json").write_text(json.dumps({"servers": {"nfl": {"command": "nfl-mcp"}}}))

        class _FakeCursor:
            def __init__(self, row):
                self._row = row

            def fetchone(self):
                return self._row

        class _FakeConn:
            def execute(self, sql):
                if "COUNT(*) FROM plays" in sql:
                    return _FakeCursor((123456,))
                return _FakeCursor((2013, 2025))

            def close(self):
                return None

        @contextlib.contextmanager
        def _fake_get_db_connection():
            yield _FakeConn()

        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: db_path)
        # doctor opens the file through database.py, which holds its own reference.
        monkeypatch.setattr("nfl_mcp.database.get_duckdb_path", lambda: db_path)
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: claude_path)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)
        monkeypatch.setattr("nfl_mcp.database.get_db_connection", _fake_get_db_connection)

        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0
        assert "All checks passed!" in result.output

    def test_doctor_reports_db_error_and_failed_summary(self, runner, monkeypatch, tmp_path):
        db_path = tmp_path / "nflread.duckdb"
        db_path.write_text("placeholder")

        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: db_path)
        # doctor opens the file through database.py, which holds its own reference.
        monkeypatch.setattr("nfl_mcp.database.get_duckdb_path", lambda: db_path)
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)
        monkeypatch.setattr(
            "nfl_mcp.database.get_db_connection",
            lambda: (_ for _ in ()).throw(RuntimeError("cannot open db")),
        )

        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 1
        assert "Database error: cannot open db" in result.output
        assert "Claude Desktop not detected" in result.output
        assert "VS Code MCP config not found" in result.output
        assert "Some checks failed" in result.output

    def test_doctor_names_the_command_that_creates_a_missing_database(
        self, runner, monkeypatch, tmp_path
    ):
        """'No file' is its own check, and the advice is the command that makes one."""
        missing = tmp_path / "data" / "nflread.duckdb"

        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: missing)
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)

        result = runner.invoke(main, ["doctor"])

        assert result.exit_code == 1
        assert f"DuckDB file not found at {missing}" in result.output
        assert "Run 'nfl-mcp ingest' to create it." in result.output
        assert "has no data yet" not in result.output
        assert not missing.parent.exists()  # doctor reads, so it creates no directory

    def test_doctor_reports_a_database_with_no_plays_table_as_no_data_yet(
        self, runner, monkeypatch, tmp_path
    ):
        db_path = tmp_path / "nflread.duckdb"
        db_path.write_text("placeholder")

        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: db_path)
        # doctor opens the file through database.py, which holds its own reference.
        monkeypatch.setattr("nfl_mcp.database.get_duckdb_path", lambda: db_path)
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)
        monkeypatch.setattr("duckdb.connect", lambda *args, **kwargs: _FakeEmptyDb())

        result = runner.invoke(main, ["doctor"])

        assert result.exit_code == 1
        assert f"Database has no data yet — no plays table in {db_path}" in result.output
        assert "Run 'nfl-mcp ingest' to load data." in result.output
        assert "DuckDB file exists" in result.output
        assert "file not found" not in result.output

    def test_doctor_reports_a_database_with_zero_plays_as_no_data_yet(
        self, runner, monkeypatch, tmp_path
    ):
        db_path = tmp_path / "nflread.duckdb"
        db_path.write_text("placeholder")

        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: db_path)
        # doctor opens the file through database.py, which holds its own reference.
        monkeypatch.setattr("nfl_mcp.database.get_duckdb_path", lambda: db_path)
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)
        monkeypatch.setattr("duckdb.connect", lambda *args, **kwargs: _FakeNoPlaysDb())

        result = runner.invoke(main, ["doctor"])

        assert result.exit_code == 1
        assert f"Database has no data yet — 0 plays in {db_path}" in result.output
        assert "Run 'nfl-mcp ingest' to load data." in result.output
        assert "0 plays loaded (seasons" not in result.output

    def test_doctor_keeps_working_with_only_the_env_var_and_a_missing_file(
        self, runner, monkeypatch, tmp_path
    ):
        """The container sets NFL_MCP_DB_PATH and has no config file at all."""
        missing = tmp_path / "data" / "nflread.duckdb"

        monkeypatch.setenv("NFL_MCP_DB_PATH", str(missing))
        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: False)
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)

        result = runner.invoke(main, ["doctor"])

        assert result.exit_code == 1
        assert "the database path comes from NFL_MCP_DB_PATH" in result.output
        assert f"DuckDB file not found at {missing}" in result.output
        assert "Run 'nfl-mcp ingest' to create it." in result.output

    def test_doctor_warns_when_server_entries_missing(self, runner, monkeypatch, tmp_path):
        db_path = tmp_path / "missing.duckdb"  # intentionally does not exist

        claude_path = tmp_path / "claude_desktop_config.json"
        claude_path.write_text(json.dumps({"mcpServers": {}}))

        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir(parents=True, exist_ok=True)
        (vscode_dir / "mcp.json").write_text(json.dumps({"servers": {}}))

        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: db_path)
        # doctor opens the file through database.py, which holds its own reference.
        monkeypatch.setattr("nfl_mcp.database.get_duckdb_path", lambda: db_path)
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: claude_path)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)

        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 1
        assert "DuckDB file not found" in result.output
        assert "Claude Desktop config exists but 'nfl' server not found" in result.output
        assert ".vscode/mcp.json exists but 'nfl' server not found" in result.output
        assert "Some checks failed" in result.output

    def test_doctor_warns_on_invalid_client_json(self, runner, monkeypatch, tmp_path):
        db_path = tmp_path / "missing.duckdb"  # intentionally does not exist

        claude_path = tmp_path / "claude_desktop_config.json"
        claude_path.write_text("{ not json")

        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir(parents=True, exist_ok=True)
        (vscode_dir / "mcp.json").write_text("{ definitely not json")

        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: db_path)
        # doctor opens the file through database.py, which holds its own reference.
        monkeypatch.setattr("nfl_mcp.database.get_duckdb_path", lambda: db_path)
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: claude_path)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)

        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 1
        assert "Claude Desktop config exists but is invalid JSON" in result.output
        assert ".vscode/mcp.json exists but is invalid JSON" in result.output

    def test_doctor_warnings_about_clients_do_not_fail_the_run(self, runner, monkeypatch, tmp_path):
        db_path = tmp_path / "nflread.duckdb"
        db_path.write_text("placeholder")

        claude_path = tmp_path / "claude_desktop_config.json"
        claude_path.write_text("{ not json")

        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir(parents=True, exist_ok=True)
        (vscode_dir / "mcp.json").write_text(json.dumps({"servers": {}}))

        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: db_path)
        # doctor opens the file through database.py, which holds its own reference.
        monkeypatch.setattr("nfl_mcp.database.get_duckdb_path", lambda: db_path)
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: claude_path)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)
        monkeypatch.setattr("duckdb.connect", lambda *args, **kwargs: _FakeHealthyDb())

        result = runner.invoke(main, ["doctor"])

        assert result.exit_code == 0, result.output
        assert "Claude Desktop config exists but is invalid JSON" in result.output
        assert ".vscode/mcp.json exists but 'nfl' server not found" in result.output
        assert "All checks passed!" in result.output


# ── ingest exit codes ──────────────────────────────────────────────────────────
#
# update_db.sh runs `nfl-mcp ingest` under `set -eu` and swaps the live database only on
# exit 0. These tests pin the rule: a run where nothing at all loaded leaves the exit code
# non-zero, and a run that merely failed on some seasons does not.

from nfl_mcp.cli import INGEST_PARTIAL_EXIT, NOTHING_LOADED_MESSAGE, _run_problems  # noqa: E402
from nfl_mcp.config import DEFAULT_DUCKDB_PATH  # noqa: E402
from nfl_mcp.ingest import IngestAttempt, IngestRunResult  # noqa: E402
from nfl_mcp.status import IngestStatus  # noqa: E402


def _stub_run(monkeypatch, *statuses):
    """Replace the ingest with a run that reported these outcomes, and return its result."""
    result = IngestRunResult(attempts=[
        IngestAttempt(dataset_id="schedules", season=2024, status=status,
                      detail=None if status == IngestStatus.LOADED else "no route to host")
        for status in statuses
    ])
    monkeypatch.setattr("nfl_mcp.ingest.run_ingest_datasets", lambda **kw: result)
    return result


class TestRunProblems:
    def test_a_stubbed_out_run_neither_failed_nor_loaded_anything(self):
        assert _run_problems(None) == (False, False)

    def test_a_run_is_judged_by_its_own_attempts(self):
        loaded_then_failed = IngestRunResult(attempts=[
            IngestAttempt("schedules", 2024, IngestStatus.LOADED),
            IngestAttempt("schedules", 2023, IngestStatus.FAILED, "404"),
        ])
        assert _run_problems(loaded_then_failed) == (False, True)

        all_failed = IngestRunResult(attempts=[
            IngestAttempt("schedules", 2024, IngestStatus.FAILED, "no route to host"),
        ])
        assert _run_problems(all_failed) == (True, True)


class TestIngestExitCode:
    # Merge: main's exit codes win. A partial run exits 3 (update_db.sh goes on to upgrade and
    # validate on 3), --allow-failures turns it into 0, and a run that loaded nothing exits 3
    # even with --allow-failures. The branch exited 0 on a partial run and 1 on nothing loaded.

    def test_some_seasons_failing_exits_three_unless_failures_are_allowed(self, runner,
                                                                          monkeypatch):
        """Play-by-play for a new season fails every run from March to September.

        The run still finished, so the code is 3, not the fatal 1, and update_db.sh goes on.
        """
        _stub_run(monkeypatch, IngestStatus.LOADED, IngestStatus.FAILED)
        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--skip-views"])
        assert result.exit_code == INGEST_PARTIAL_EXIT, result.output
        assert NOTHING_LOADED_MESSAGE not in result.output
        allowed = runner.invoke(main, ["ingest", "--dataset", "schedules", "--skip-views",
                                       "--allow-failures"])
        assert allowed.exit_code == 0, allowed.output

    def test_a_run_that_loaded_nothing_exits_three(self, runner, monkeypatch):
        _stub_run(monkeypatch, IngestStatus.FAILED, IngestStatus.FAILED, IngestStatus.EMPTY)
        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--skip-views"])
        assert result.exit_code == INGEST_PARTIAL_EXIT
        assert NOTHING_LOADED_MESSAGE in result.output
        allowed = runner.invoke(main, ["ingest", "--dataset", "schedules", "--skip-views",
                                       "--allow-failures"])
        assert allowed.exit_code == INGEST_PARTIAL_EXIT

    def test_the_message_says_the_database_was_left_alone(self, runner, monkeypatch):
        _stub_run(monkeypatch, IngestStatus.FAILED)
        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--skip-views"])
        assert "Nothing could be loaded. The database was left as it was." in result.output

    def test_a_run_that_asked_for_nothing_exits_zero(self, runner, monkeypatch):
        # Every season was already loaded, so the run made no request and failed nothing.
        _stub_run(monkeypatch)
        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--skip-views"])
        assert result.exit_code == 0, result.output

    def test_empty_seasons_alone_are_not_a_failure(self, runner, monkeypatch):
        _stub_run(monkeypatch, IngestStatus.EMPTY, IngestStatus.EMPTY)
        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--skip-views"])
        # Nothing landed, so the run did load nothing, and it still says so.
        assert result.exit_code == INGEST_PARTIAL_EXIT
        assert NOTHING_LOADED_MESSAGE in result.output

    def test_strict_exits_three_when_any_attempt_failed(self, runner, monkeypatch):
        _stub_run(monkeypatch, IngestStatus.LOADED, IngestStatus.FAILED)
        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--strict"])
        assert result.exit_code == INGEST_PARTIAL_EXIT
        # --strict wins over --allow-failures.
        both = runner.invoke(main, ["ingest", "--dataset", "schedules", "--strict",
                                    "--allow-failures"])
        assert both.exit_code == INGEST_PARTIAL_EXIT

    def test_strict_tolerates_a_season_the_publisher_has_no_rows_for(self, runner, monkeypatch):
        _stub_run(monkeypatch, IngestStatus.LOADED, IngestStatus.EMPTY)
        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--strict"])
        assert result.exit_code == 0, result.output

    def test_strict_is_off_by_default(self, runner, monkeypatch):
        captured = {}

        def fake(**kw):
            captured.update(kw)
            return IngestRunResult(attempts=[
                IngestAttempt("schedules", 2024, IngestStatus.LOADED)])

        monkeypatch.setattr("nfl_mcp.ingest.run_ingest_datasets", fake)
        result = runner.invoke(main, ["ingest", "--dataset", "schedules"])
        assert result.exit_code == 0, result.output

    def test_the_list_option_never_checks_the_run(self, runner, monkeypatch):
        called = []
        monkeypatch.setattr("nfl_mcp.ingest.run_ingest_datasets",
                            lambda **kw: called.append(True))
        result = runner.invoke(main, ["ingest", "--list"])
        assert result.exit_code == 0
        assert called == []

    def test_the_strict_flag_is_documented(self, runner):
        help_text = runner.invoke(main, ["ingest", "--help"]).output
        assert "--strict" in help_text

    def _real_run(self, monkeypatch, tmp_path, loader, name="nflread.duckdb"):
        """Point the ingest at a fake nflreadpy and a tmp database, with no stub in between."""
        import sys
        import types

        fake = types.ModuleType("nflreadpy")
        fake.load_schedules = loader
        monkeypatch.setitem(sys.modules, "nflreadpy", fake)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: tmp_path / name)

    def test_a_real_run_where_everything_fails_exits_three(self, runner, monkeypatch, tmp_path):
        """The whole chain: the run finds nothing loaded, the command exits 3."""
        def dead(seasons, **kwargs):
            raise OSError("no route to host")

        self._real_run(monkeypatch, tmp_path, dead)
        result = runner.invoke(main, ["ingest", "--dataset", "schedules",
                                      "--start", "2021", "--end", "2022", "--skip-views"])
        assert result.exit_code == INGEST_PARTIAL_EXIT, result.output
        assert "failed: 2" in result.output
        assert NOTHING_LOADED_MESSAGE in result.output

    def test_a_real_run_that_loaded_one_season_exits_zero_with_allow_failures(
            self, runner, monkeypatch, tmp_path):
        import polars as pl

        def flaky(seasons, **kwargs):
            if seasons[0] == 2021:
                return pl.DataFrame({"season": [2021], "team": ["KC"]})
            raise OSError("no route to host")

        self._real_run(monkeypatch, tmp_path, flaky)
        result = runner.invoke(main, ["ingest", "--dataset", "schedules", "--allow-failures",
                                      "--start", "2021", "--end", "2022", "--skip-views"])
        assert result.exit_code == 0, result.output
        assert "loaded: 1" in result.output
        assert "schedules 2022: failed: no route to host" in result.output
        assert NOTHING_LOADED_MESSAGE not in result.output

    def test_a_real_partial_run_is_an_error_under_strict(self, runner, monkeypatch, tmp_path):
        import polars as pl

        self._real_run(monkeypatch, tmp_path,
                       lambda seasons, **kw: pl.DataFrame({"season": seasons, "team": ["KC"]}))
        ok = runner.invoke(main, ["ingest", "--dataset", "schedules",
                                  "--start", "2021", "--end", "2021", "--skip-views", "--strict"])
        assert ok.exit_code == 0, ok.output

        def dead(seasons, **kwargs):
            raise OSError("no route to host")

        # A fresh database: on the one above, 2021 is already loaded, so a dead network there
        # means nothing was attempted, which is not a failure.
        self._real_run(monkeypatch, tmp_path, dead, name="empty.duckdb")
        strict = runner.invoke(main, ["ingest", "--dataset", "schedules",
                                      "--start", "2021", "--end", "2021", "--skip-views", "--strict"])
        assert strict.exit_code == INGEST_PARTIAL_EXIT


class TestInitReportsWhatItLoaded:
    """init must not print 'Setup complete!' after a run that loaded nothing."""

    def _init(self, runner, monkeypatch, tmp_path, result, start=2024, end=2024):
        saved = {}
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {})
        monkeypatch.setattr(
            "nfl_mcp.config.save_config", lambda cfg: saved.update(cfg) or tmp_path / "config.json"
        )
        monkeypatch.setattr("nfl_mcp.ingest.run_ingest_datasets", lambda **kw: result)
        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", lambda cfg: True)
        # Prompts: change DB path? -> no ; ingest? -> yes ; client? -> no ; server? -> no
        output = runner.invoke(main, ["init", "--start", str(start), "--end", str(end)],
                               input="n\n\nn\nn\n")
        output.saved = saved
        return output

    def test_nothing_loaded_replaces_the_success_wording(self, runner, monkeypatch, tmp_path):
        result = self._init(runner, monkeypatch, tmp_path, IngestRunResult(attempts=[
            IngestAttempt("pbp", 2024, IngestStatus.FAILED, "no route to host"),
        ]))
        assert NOTHING_LOADED_MESSAGE in result.output
        assert "Setup complete!" not in result.output
        # init writes the config too, so its own exit code stays as it was.
        assert result.exit_code == 0

    def test_a_run_that_loaded_something_still_finishes_the_setup(self, runner, monkeypatch, tmp_path):
        result = self._init(runner, monkeypatch, tmp_path, IngestRunResult(attempts=[
            IngestAttempt("pbp", 2024, IngestStatus.LOADED),
            IngestAttempt("pbp", 2023, IngestStatus.FAILED, "404"),
        ]))
        assert "Setup complete!" in result.output
        assert NOTHING_LOADED_MESSAGE not in result.output

    def test_the_advice_names_the_exact_command_that_finishes_the_job(
        self, runner, monkeypatch, tmp_path
    ):
        """'Load data later' must be copy-pasteable, not a hint at a command."""
        result = self._init(runner, monkeypatch, tmp_path, IngestRunResult(attempts=[
            IngestAttempt("pbp", 2024, IngestStatus.FAILED, "no route to host"),
        ]), start=2020, end=2024)

        assert "nfl-mcp ingest --start 2020 --end 2024" in result.output

    def test_a_load_that_raised_is_reported_the_same_way(self, runner, monkeypatch, tmp_path):
        """A raised load leaves the saved config pointing at nothing; say so, keep the file."""
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {})
        saved = {}
        monkeypatch.setattr(
            "nfl_mcp.config.save_config", lambda cfg: saved.update(cfg) or tmp_path / "config.json"
        )
        def _dead(**_kw):
            raise OSError("no route to host")
        monkeypatch.setattr("nfl_mcp.ingest.run_ingest_datasets", _dead)
        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", lambda cfg: True)

        result = runner.invoke(main, ["init", "--start", "2020", "--end", "2024"],
                               input="n\n\nn\nn\n")

        assert result.exit_code == 0, result.output
        assert "Data load stopped: no route to host" in result.output
        assert NOTHING_LOADED_MESSAGE in result.output
        assert "nfl-mcp ingest --start 2020 --end 2024" in result.output
        assert "Setup complete!" not in result.output
        # The user's configuration is kept, not rolled back.
        assert saved["duckdb_path"] == str(DEFAULT_DUCKDB_PATH)

    def test_the_wizard_does_not_stop_when_the_load_raises(self, runner, monkeypatch, tmp_path):
        """The client step still runs after a failed load, so setup can still finish."""
        client_calls = []
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {})
        monkeypatch.setattr("nfl_mcp.config.save_config", lambda cfg: tmp_path / "config.json")
        def _dead(**_kw):
            raise OSError("no route to host")
        monkeypatch.setattr("nfl_mcp.ingest.run_ingest_datasets", _dead)
        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", lambda cfg: client_calls.append(cfg) or True)

        result = runner.invoke(main, ["init", "--start", "2024", "--end", "2024"],
                               input="n\n\ny\nn\n")

        assert result.exit_code == 0, result.output
        assert len(client_calls) == 1
