"""Unit tests for CLI commands — no database or network required."""

import json

import pytest
from click.testing import CliRunner

import nfl_mcp.cli as cli
from nfl_mcp.cli import main, _season_window_label
from nfl_mcp.ingest import DERIVED_SHAPE_VERSION
from nfl_mcp.registry import ALL_DATASETS, DEFAULT_DATASETS, REGISTRY


@pytest.fixture
def runner():
    return CliRunner()


# ── serve ───────────────────────────────────────────────────────────────────────

class TestServe:
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

    @pytest.mark.parametrize("var,value", [("NFL_MCP_DUCKDB_THREADS", "two"),
                                           ("NFL_MCP_DUCKDB_MEMORY_LIMIT", "lots")])
    def test_serve_refuses_a_duckdb_variable_it_cannot_use(self, runner, monkeypatch, var, value):
        """Once these made every tool fail while the server reported itself healthy."""
        started = []
        monkeypatch.setattr("nfl_mcp.cli.uvicorn.run", lambda *a, **k: started.append(True))
        monkeypatch.setenv(var, value)
        result = runner.invoke(main, ["serve"])
        assert result.exit_code == 1
        assert f"{var}='{value}'" in result.output
        assert not started


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

    @pytest.mark.parametrize("failed, code", [(0, 0), (2, 1)])
    def test_exit_status_follows_failed_fetches(self, runner, monkeypatch, failed, code):
        # update_db.sh swaps slots only on exit 0, so a missed fetch must be non-zero.
        monkeypatch.setattr("nfl_mcp.ingest.run_ingest_datasets", lambda *a, **kw: failed)
        assert runner.invoke(main, ["ingest", "--dataset", "schedules"]).exit_code == code

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
        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", lambda cfg: setup_calls.append(cfg))

        result = runner.invoke(
            main,
            ["init", "--skip-ingest"],
            input="n\ny\nn\n",
        )

        assert result.exit_code == 0
        assert len(setup_calls) == 1

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

    def test_setup_client_routes_auto(self, runner, monkeypatch):
        called = []
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {"duckdb_path": "/tmp/db.duckdb"})
        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", lambda cfg: called.append(("auto", cfg)))
        result = runner.invoke(main, ["setup-client", "--client", "auto"])
        assert result.exit_code == 0
        assert called and called[0][0] == "auto"

    def test_setup_client_routes_claude_desktop(self, runner, monkeypatch):
        called = []
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {"duckdb_path": "/tmp/db.duckdb"})
        monkeypatch.setattr("nfl_mcp.cli._configure_claude_desktop", lambda cfg: called.append(("claude", cfg)))
        result = runner.invoke(main, ["setup-client", "--client", "claude-desktop"])
        assert result.exit_code == 0
        assert called and called[0][0] == "claude"

    def test_setup_client_routes_claude_code(self, runner, monkeypatch):
        called = []
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {"duckdb_path": "/tmp/db.duckdb"})
        monkeypatch.setattr("nfl_mcp.cli._configure_claude_code", lambda cfg: called.append(("claude-code", cfg)))
        result = runner.invoke(main, ["setup-client", "--client", "claude-code"])
        assert result.exit_code == 0
        assert called and called[0][0] == "claude-code"

    def test_setup_client_routes_vscode(self, runner, monkeypatch):
        called = []
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {"duckdb_path": "/tmp/db.duckdb"})
        monkeypatch.setattr("nfl_mcp.cli._configure_vscode", lambda cfg: called.append(("vscode", cfg)))
        result = runner.invoke(main, ["setup-client", "--client", "vscode"])
        assert result.exit_code == 0
        assert called and called[0][0] == "vscode"


# ── CLI helpers ─────────────────────────────────────────────────────────────────

class TestCliHelpers:
    def test_setup_client_interactive_configures_selected_clients(self, monkeypatch, tmp_path):
        configured = []
        # 3 prompts: Claude Desktop=yes, Claude Code=no, VS Code=no
        answers = iter([True, False, False])

        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: tmp_path / "claude_desktop_config.json")
        monkeypatch.setattr("nfl_mcp.cli.click.confirm", lambda *args, **kwargs: next(answers))
        monkeypatch.setattr("nfl_mcp.cli._configure_claude_desktop", lambda cfg: configured.append("claude"))
        monkeypatch.setattr("nfl_mcp.cli._configure_claude_code", lambda cfg: configured.append("claude-code"))
        monkeypatch.setattr("nfl_mcp.cli._configure_vscode", lambda cfg: configured.append("vscode"))

        cli._setup_client_interactive({"duckdb_path": "/tmp/db.duckdb"})
        assert configured == ["claude"]

    def test_setup_client_interactive_configures_claude_code_branch(self, monkeypatch, tmp_path):
        configured = []
        # No Claude Desktop; 2 prompts: Claude Code=yes, VS Code=no
        answers = iter([True, False])
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.click.confirm", lambda *args, **kwargs: next(answers))
        monkeypatch.setattr("nfl_mcp.cli._configure_claude_code", lambda cfg: configured.append("claude-code"))
        monkeypatch.setattr("nfl_mcp.cli._configure_vscode", lambda cfg: configured.append("vscode"))

        cli._setup_client_interactive({"duckdb_path": "/tmp/db.duckdb"})
        assert configured == ["claude-code"]

    def test_setup_client_interactive_configures_vscode_branch(self, monkeypatch):
        configured = []
        # No Claude Desktop; 2 prompts: Claude Code=no, VS Code=yes
        answers = iter([False, True])
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.click.confirm", lambda *args, **kwargs: next(answers))
        monkeypatch.setattr("nfl_mcp.cli._configure_claude_code", lambda cfg: configured.append("claude-code"))
        monkeypatch.setattr("nfl_mcp.cli._configure_vscode", lambda cfg: configured.append("vscode"))
        monkeypatch.setattr("nfl_mcp.cli._configure_claude_desktop", lambda cfg: configured.append("claude"))

        cli._setup_client_interactive({"duckdb_path": "/tmp/db.duckdb"})
        assert configured == ["vscode"]

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
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        cli._configure_claude_desktop({})

    def test_configure_claude_desktop_merges_even_with_invalid_existing_json(self, monkeypatch, tmp_path):
        path = tmp_path / "claude_desktop_config.json"
        path.write_text("{invalid json")

        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: path)
        monkeypatch.setattr("nfl_mcp.cli._build_server_config", lambda cfg: {"command": "nfl-mcp", "args": ["serve"]})

        cli._configure_claude_desktop({})
        saved = json.loads(path.read_text())
        assert saved["mcpServers"]["nfl"]["command"] == "nfl-mcp"

    def test_configure_vscode_merges_even_with_invalid_existing_json(self, monkeypatch, tmp_path):
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir(parents=True, exist_ok=True)
        mcp_path = vscode_dir / "mcp.json"
        mcp_path.write_text("{invalid json")

        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)
        monkeypatch.setattr("nfl_mcp.cli._build_server_config", lambda cfg: {"command": "nfl-mcp", "args": ["serve"]})

        cli._configure_vscode({})
        saved = json.loads(mcp_path.read_text())
        assert saved["servers"]["nfl"]["command"] == "nfl-mcp"

    def test_configure_claude_code_writes_global_mcp_json(self, monkeypatch, tmp_path):
        monkeypatch.setattr("nfl_mcp.cli.Path.home", lambda: tmp_path)
        monkeypatch.setattr("nfl_mcp.cli._build_server_config", lambda cfg: {"url": "http://localhost:8000/mcp"})

        cli._configure_claude_code({})

        path = tmp_path / ".claude" / "mcp.json"
        assert path.exists()
        saved = json.loads(path.read_text())
        assert saved["mcpServers"]["nfl"]["url"] == "http://localhost:8000/mcp"

    def test_configure_claude_code_merges_with_invalid_existing_json(self, monkeypatch, tmp_path):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "mcp.json").write_text("{invalid json")

        monkeypatch.setattr("nfl_mcp.cli.Path.home", lambda: tmp_path)
        monkeypatch.setattr("nfl_mcp.cli._build_server_config", lambda cfg: {"url": "http://localhost:8000/mcp"})

        cli._configure_claude_code({})
        saved = json.loads((claude_dir / "mcp.json").read_text())
        assert saved["mcpServers"]["nfl"]["url"] == "http://localhost:8000/mcp"


# ── doctor command ──────────────────────────────────────────────────────────────

class TestDoctorCommand:
    def test_doctor_runs_from_the_database_path_alone(self, runner, monkeypatch, tmp_path):
        # A container sets NFL_MCP_DB_PATH and writes no config file; doctor used to stop there.
        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: False)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: tmp_path / "missing.duckdb")
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 1
        assert "No config file; using NFL_MCP_DB_PATH" in result.output
        assert "DuckDB file not found" in result.output

    @staticmethod
    def _stamped_db(path, *, fetched_after_build):
        import datetime

        import duckdb
        built = datetime.datetime(2026, 11, 1)
        fetched = built + datetime.timedelta(hours=1 if fetched_after_build else -1)
        conn = duckdb.connect(str(path))
        conn.execute("CREATE TABLE plays AS SELECT 2024 AS season FROM range(3)")
        conn.execute("CREATE TABLE _ingest_metadata (dataset_id VARCHAR, table_name VARCHAR, season INTEGER, "
                     "row_count BIGINT, loaded_at TIMESTAMP, loader_fn VARCHAR)")
        conn.execute("INSERT INTO _ingest_metadata VALUES "
                     "('derived_shape', 'derived_shape', NULL, ?, ?, 'nfl_mcp.ingest.rebuild_derived_tables'), "
                     "('pbp', 'plays', 2024, 3, ?, 'load_pbp')", [DERIVED_SHAPE_VERSION, built, fetched])
        conn.close()

    def test_doctor_happy_path_reports_success(self, runner, monkeypatch, tmp_path):
        slot = tmp_path / "nflread_b.duckdb"
        self._stamped_db(slot, fetched_after_build=False)
        db_path = tmp_path / "nflread.duckdb"
        db_path.symlink_to(slot.name)

        claude_path = tmp_path / "claude_desktop_config.json"
        claude_path.write_text(json.dumps({"mcpServers": {"nfl": {"command": "nfl-mcp"}}}))

        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir(parents=True, exist_ok=True)
        (vscode_dir / "mcp.json").write_text(json.dumps({"servers": {"nfl": {"command": "nfl-mcp"}}}))

        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: db_path)
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: claude_path)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)

        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0
        assert (f"Derived shape version {DERIVED_SHAPE_VERSION} "
                f"(this build writes {DERIVED_SHAPE_VERSION})") in result.output
        assert "after the last fetch" in result.output
        assert "Last real fetch: 2026-10-31 23:00:00" in result.output
        assert "Active slot: nflread_b.duckdb" in result.output
        assert "All checks passed!" in result.output

    def test_doctor_warns_of_a_null_column_and_check_stays_current(self, runner, monkeypatch, tmp_path):
        import duckdb
        import polars as pl

        from nfl_mcp.ingest import _reconcile_schema
        db_path = tmp_path / "nflread.duckdb"
        self._stamped_db(db_path, fetched_after_build=False)
        conn = duckdb.connect(str(db_path))
        _reconcile_schema(conn, "plays", pl.DataFrame({"season": [2025], "first_downs": [9]}), "season")
        conn.close()
        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: db_path)
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)

        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0
        assert ("! plays.first_downs is null for every row of seasons [2024]: re-fetch them with "
                "'nfl-mcp ingest --fresh --start 2024 --end 2024'") in result.output
        # A rebuild cannot fill the column, so the updater must still be able to swap.
        monkeypatch.setenv("NFL_MCP_DB_PATH", str(db_path))
        check = runner.invoke(main, ["rebuild", "--check"])
        assert (check.exit_code, check.stdout.strip()) == (0, "current")

    def test_doctor_fails_when_derived_tables_lag_the_rows(self, runner, monkeypatch, tmp_path):
        db_path = tmp_path / "nflread.duckdb"
        self._stamped_db(db_path, fetched_after_build=True)
        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: db_path)
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)

        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 1
        assert "before the last fetch — run 'nfl-mcp rebuild'" in result.output
        assert "Active slot: none (a plain database file)" in result.output

    def test_doctor_reports_db_error_and_failed_summary(self, runner, monkeypatch, tmp_path):
        db_path = tmp_path / "nflread.duckdb"
        db_path.write_text("placeholder")

        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: db_path)
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)
        monkeypatch.setattr(
            "duckdb.connect",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cannot open db")),
        )

        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 1
        assert "Database error: cannot open db" in result.output
        assert "Claude Desktop not detected" in result.output
        assert "VS Code MCP config not found" in result.output
        assert "Some checks failed" in result.output

    def test_doctor_warns_when_server_entries_missing(self, runner, monkeypatch, tmp_path):
        db_path = tmp_path / "missing.duckdb"  # intentionally does not exist

        claude_path = tmp_path / "claude_desktop_config.json"
        claude_path.write_text(json.dumps({"mcpServers": {}}))

        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir(parents=True, exist_ok=True)
        (vscode_dir / "mcp.json").write_text(json.dumps({"servers": {}}))

        monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: db_path)
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
        monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: claude_path)
        monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)

        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 1
        assert "Claude Desktop config exists but is invalid JSON" in result.output
        assert ".vscode/mcp.json exists but is invalid JSON" in result.output
