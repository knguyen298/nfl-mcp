"""v0.7.3: stale-table messages that name no command, and an update that survives one
failed download.

Source: the v0.7.3 brief.

1. Several tool errors told the model to run `nfl-mcp rebuild` or `nfl-mcp upgrade`. In the
   container nobody can run them: the served slot is read-only, and the user of Open WebUI
   has no shell in the container. Since v0.7.2 the records and team-stats warnings name no
   command. Now every tool-facing stale message says the database is older than this server,
   that the server updates it automatically (when the container starts and in the weekly
   update), and that the data works again after the update. The error codes and the JSON
   shape stay. The `--pre1999` hints in the tool text now name "an optional ESPN data pull",
   because the container never runs that pull and the model cannot run it either. CLI output
   (cli.py, ingest.py logs, validate.py results) keeps the command names.

2. update_db.sh has `set -eu`. `nfl-mcp ingest` exited 1 when one dataset failed to
   download, so the script stopped before `nfl-mcp upgrade`, and a new image with a new
   schema never upgraded while any download failed.

   What the ingest does (read from nfl_mcp/ingest.py and cli.py at b78f623):
   - A failed fetch writes nothing. A failed write runs in a transaction per season (pbp,
     seasonal datasets), per table (static datasets) or for all restamped seasons, and rolls
     back. The target slot starts as a copy of the active slot, so a failed dataset keeps the
     active slot's rows. These are soft failures: the run goes on and lists them.
   - A fatal error (no play-by-play season can be fetched, the database cannot be opened, a
     derived table fails to build, an ESPN write error) raises out of run_ingest_datasets.
   - Before v0.7.3 both exited 1 (ClickException or an uncaught exception).

   Interface contract tested here
   - `nfl-mcp ingest` exits 0 on a clean run (or with --allow-failures), 3 when the run
     finished with soft failures, 1 on a fatal error and 2 on a usage error. cli.INGEST_PARTIAL_EXIT
     is 3 and the help text lists the codes.
   - `--failures-file PATH` writes one "<dataset>" or "<dataset> <season>" per line, each once,
     and an empty file on a clean run. A fatal error writes no file.
   - update_db.sh on ingest exit 3 logs one WARNING line naming the failures, then runs
     upgrade and validate as before. validate still decides the swap. On any other non-zero
     exit it logs an ERROR with the exit code, runs nothing else and keeps the active slot.

3. __version__ is 0.7.3. SCHEMA_VERSION stays 4.

The unit tests build small databases in tmp_path and run update_db.sh with nfl-mcp stubbed
on PATH, the container paths rewritten to tmp_path. Nothing here touches /data.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import anyio
import duckdb
import polars as pl
import pytest
from click.testing import CliRunner

from nfl_mcp import __version__, cli, discovery, migrate, schema_pbp, server, tools
from nfl_mcp import ingest as ing
from nfl_mcp.cli import main

ROOT = Path(__file__).resolve().parents[2]

# A command a model could pass on to the user, who cannot run it in the container.
_COMMAND = re.compile(r"nfl-mcp\s+(ingest|upgrade|rebuild|validate|init|serve|doctor|setup-client)\b")


def _raw(name, args):
    """A tool call through server.call_tool: the tool's checks and the one error shape."""
    async def go():
        return await server.call_tool(name, args)
    res = anyio.run(go)
    return res.isError, json.loads(res.content[0].text)


# ═══ 1. tool-facing stale messages ═════════════════════════════════════════════

STALE_ERRORS = {"team": tools._STALE_ERROR, "player": tools._STALE_PLAYER_ERROR}


@pytest.mark.parametrize("kind", sorted(STALE_ERRORS))
def test_stale_error_keeps_its_code_and_shape(kind):
    err = STALE_ERRORS[kind]
    assert set(err) == {"error", "code"}
    assert err["code"] == "STALE_DERIVED_TABLES"


@pytest.mark.parametrize("kind", sorted(STALE_ERRORS))
def test_stale_error_names_no_command(kind):
    text = STALE_ERRORS[kind]["error"]
    assert not _COMMAND.search(text), text
    assert "`" not in text and "run" not in text.lower().split(), text


@pytest.mark.parametrize("kind", sorted(STALE_ERRORS))
def test_stale_error_says_the_server_updates_the_database(kind):
    text = STALE_ERRORS[kind]["error"]
    assert f"derived {kind} tables" in text
    assert "older version of nfl-mcp than this server" in text
    assert "updates the database automatically" in text
    assert "when the container starts" in text and "weekly database update" in text
    assert "after that update finishes" in text


@pytest.fixture
def stale_team_db(tmp_path, monkeypatch):
    """Team tables without season_type, as a database from before v0.7.0 has them."""
    path = tmp_path / "stale_team.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE team_offense_stats (team VARCHAR, season_year INTEGER, touchdowns INTEGER)")
    con.execute("CREATE TABLE team_defense_stats (team VARCHAR, season_year INTEGER, sacks INTEGER)")
    con.execute("INSERT INTO team_offense_stats VALUES ('KC', 2024, 50)")
    con.execute("INSERT INTO team_defense_stats VALUES ('KC', 2024, 40)")
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


@pytest.fixture
def stale_player_db(tmp_path, monkeypatch):
    """player_drop_rate without season_type."""
    path = tmp_path / "stale_player.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE player_drop_rate (player_id VARCHAR, player VARCHAR, team VARCHAR, "
                "season INTEGER, catchable_targets BIGINT, drops BIGINT, drop_rate_pct DOUBLE, "
                "contested_targets BIGINT, created_receptions BIGINT)")
    con.execute("INSERT INTO player_drop_rate VALUES ('00-0000001', 'A.One', 'KC', 2024, 10, 1, "
                "10.0, 1, 0)")
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


def test_stale_team_tables_return_the_new_error(stale_team_db):
    assert tools.nfl_team_stats("KC", 2024, side="offense", season_type="REG") == tools._STALE_ERROR
    assert tools.nfl_compare(team1="KC", team2="BUF", season=2024,
                             season_type="POST") == tools._STALE_ERROR


def test_stale_team_tables_over_mcp_keep_the_error_shape(stale_team_db):
    is_error, payload = _raw("nfl_team_stats", {"team": "KC", "season": 2024, "season_type": "REG"})
    assert is_error
    assert payload["code"] == "STALE_DERIVED_TABLES"
    assert payload["error"] == tools._STALE_ERROR["error"]
    assert payload["details"] == {"tool": "nfl_team_stats"}


def test_stale_player_tables_return_the_new_error(stale_player_db):
    res = tools.nfl_drop_rate(season=2024, season_type="REG")
    assert res == tools._STALE_PLAYER_ERROR
    is_error, payload = _raw("nfl_drop_rate", {"season": 2024, "season_type": "ALL"})
    assert is_error and payload["code"] == "STALE_DERIVED_TABLES"
    assert payload["error"] == tools._STALE_PLAYER_ERROR["error"]


def _model_facing_texts() -> dict[str, str]:
    """Every static text a tool hands to a model: descriptions, schemas, notes, errors, warnings."""
    texts = {f"tool {t.name}": t.description + json.dumps(t.inputSchema) for t in server.TOOLS}
    texts.update({f"data note {i}": n for i, n in enumerate(discovery.DATA_NOTES)})
    texts["schema summary"] = schema_pbp._SCHEMA_SUMMARY
    texts.update({f"schema category {k}": v for k, v in schema_pbp._SCHEMA_CATEGORIES.items()})
    for name in ("_STALE_ERROR", "_STALE_PLAYER_ERROR"):
        texts[name] = getattr(tools, name)["error"]
    for name in ("_STALE_RECORDS_WARNING", "_STALE_TEAM_WARNING"):
        texts[name] = getattr(tools, name)
    return texts


def test_no_model_facing_text_names_an_nfl_mcp_command():
    hits = {k: m.group(0) for k, v in _model_facing_texts().items() if (m := _COMMAND.search(v))}
    assert not hits, hits


def test_no_model_facing_text_names_the_pre1999_flag():
    hits = [k for k, v in _model_facing_texts().items() if "pre1999" in v]
    assert not hits, hits


def test_records_description_and_notes_still_explain_the_espn_seasons():
    desc = next(t.description for t in server.TOOLS if t.name == "nfl_team_records")
    assert "optional ESPN data pull" in desc and "record_not_available" in desc
    notes = " ".join(discovery.DATA_NOTES)
    assert "team_records goes back to 1970 through an optional ESPN data pull" in notes
    assert "1970-1998 comes from an optional" in schema_pbp._SCHEMA_SUMMARY


def test_stale_note_says_the_server_updates_the_database():
    (note,) = [n for n in discovery.DATA_NOTES if "STALE_DERIVED_TABLES" in n]
    assert "until the server updates it" in note and "automatically" in note
    assert "when the container starts" in note and "weekly database update" in note


def test_missing_pre1999_record_names_the_optional_pull_not_a_command():
    with patch("nfl_mcp.tools._execute", return_value=[]):
        res = tools.nfl_team_records(team="GB", season=1996)
    assert res["status"] == "record_not_available"
    assert "optional ESPN data pull" in res["reason"]
    assert "ESPN's archive has no record" in res["reason"]
    assert not _COMMAND.search(res["reason"]) and "pre1999" not in res["reason"]


def test_cli_messages_keep_the_command_names():
    # A human reads these on a terminal or in the update log, where the commands work.
    from nfl_mcp import validate
    con = duckdb.connect(":memory:")
    try:
        old = validate._check_schema_version(con, {})
    finally:
        con.close()
    assert not old.ok and "run nfl-mcp upgrade" in old.detail
    assert "run: nfl-mcp upgrade" in Path(cli.__file__).read_text()


# ═══ 2a. ingest exit codes and the failures file ═══════════════════════════════

def _fake_nfl(monkeypatch, **loaders):
    fake = types.ModuleType("nflreadpy")
    for name, fn in loaders.items():
        setattr(fake, name, fn)
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)


def _sched(season, n, week=None):
    return pl.DataFrame({
        "game_id": [f"{season}_{i:02d}_KC_BUF" for i in range(1, n + 1)],
        "season": [season] * n,
        "game_type": ["REG"] * n,
        "week": week if week is not None else list(range(1, n + 1)),
        "away_team": ["KC"] * n,
        "home_team": ["BUF"] * n,
    })


def _roster(season, n):
    return pl.DataFrame({
        "season": [season] * n, "team": ["KC"] * n, "position": ["QB"] * n,
        "full_name": [f"R{i}" for i in range(n)], "gsis_id": [f"00-{i}" for i in range(n)],
    })


def _boom(*a, **k):
    raise ConnectionError("simulated nflverse outage: host unreachable")


def _ingest(tmp_path, monkeypatch, args, db_name="cli.duckdb"):
    db = tmp_path / db_name
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(db))
    return CliRunner().invoke(main, ["ingest", *args, "--skip-views"]), db


def _count(db, sql):
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute(sql).fetchone()[0]
    finally:
        con.close()


TWO = ["--dataset", "schedules", "--dataset", "rosters", "--start", "2024", "--end", "2024"]


def test_partial_exit_code_is_3_and_distinct():
    assert cli.INGEST_PARTIAL_EXIT == 3
    assert cli.INGEST_PARTIAL_EXIT not in (0, 1, 2)


def test_partial_failure_exits_3_and_loads_the_rest(tmp_path, monkeypatch):
    _fake_nfl(monkeypatch, load_schedules=_boom, load_rosters=lambda s, **k: _roster(s[0], 2))
    failed = tmp_path / "failed.txt"
    result, db = _ingest(tmp_path, monkeypatch, [*TWO, "--failures-file", str(failed)])
    assert result.exit_code == 3, result.output
    assert "Ingest failed for: schedules 2024" in result.output
    assert _count(db, "SELECT COUNT(*) FROM rosters") == 2
    assert failed.read_text() == "schedules 2024\n"


def test_failed_write_rolls_back_and_keeps_the_previous_rows(tmp_path, monkeypatch):
    # The target slot is a copy of the active one: a failed season must keep its rows.
    _fake_nfl(monkeypatch, load_schedules=lambda s, **k: _sched(s[0], 2),
              load_rosters=lambda s, **k: _roster(s[0], 2))
    first, db = _ingest(tmp_path, monkeypatch, TWO)
    assert first.exit_code == 0, first.output
    # The fetch works, the INSERT fails ('week' is BIGINT), after the DELETE of the season.
    _fake_nfl(monkeypatch, load_schedules=lambda s, **k: _sched(s[0], 3, week=["x", "y", "z"]),
              load_rosters=lambda s, **k: _roster(s[0], 5))
    failed = tmp_path / "failed.txt"
    result, _ = _ingest(tmp_path, monkeypatch, [*TWO, "--fresh", "--failures-file", str(failed)])
    assert result.exit_code == 3, result.output
    assert "kept the previous rows" in result.output
    assert _count(db, "SELECT COUNT(*) FROM schedules WHERE season = 2024") == 2
    assert _count(db, "SELECT COUNT(*) FROM rosters") == 5
    assert failed.read_text() == "schedules 2024\n"


def test_failures_file_names_each_dataset_once_in_order(tmp_path, monkeypatch):
    _fake_nfl(monkeypatch, load_schedules=_boom, load_rosters=_boom)
    failed = tmp_path / "failed.txt"
    args = ["--dataset", "schedules", "--dataset", "rosters", "--start", "2023", "--end", "2024",
            "--failures-file", str(failed)]
    result, _ = _ingest(tmp_path, monkeypatch, args)
    assert result.exit_code == 3, result.output
    assert failed.read_text().splitlines() == [
        "schedules 2023", "schedules 2024", "rosters 2023", "rosters 2024"]


def test_clean_run_exits_0_with_an_empty_failures_file(tmp_path, monkeypatch):
    _fake_nfl(monkeypatch, load_schedules=lambda s, **k: _sched(s[0], 2),
              load_rosters=lambda s, **k: _roster(s[0], 2))
    failed = tmp_path / "failed.txt"
    failed.write_text("left over from an earlier run\n")
    result, _ = _ingest(tmp_path, monkeypatch, [*TWO, "--failures-file", str(failed)])
    assert result.exit_code == 0, result.output
    assert failed.read_text() == ""
    assert "Ingest failed" not in result.output


def test_allow_failures_still_exits_0_and_writes_the_file(tmp_path, monkeypatch):
    _fake_nfl(monkeypatch, load_schedules=_boom, load_rosters=lambda s, **k: _roster(s[0], 2))
    failed = tmp_path / "failed.txt"
    result, _ = _ingest(tmp_path, monkeypatch,
                        [*TWO, "--allow-failures", "--failures-file", str(failed)])
    assert result.exit_code == 0, result.output
    assert "(ignored: --allow-failures)" in result.output
    assert failed.read_text() == "schedules 2024\n"


def test_partial_failure_without_the_file_option_exits_3(tmp_path, monkeypatch):
    _fake_nfl(monkeypatch, load_schedules=_boom, load_rosters=lambda s, **k: _roster(s[0], 2))
    result, _ = _ingest(tmp_path, monkeypatch, TWO)
    assert result.exit_code == 3, result.output


def test_fatal_error_exits_1_and_writes_no_failures_file(tmp_path, monkeypatch):
    # No play-by-play season can be fetched: the run stops before any other dataset.
    _fake_nfl(monkeypatch, load_pbp=_boom, load_schedules=lambda s, **k: _sched(s[0], 2))
    failed = tmp_path / "failed.txt"
    result, _ = _ingest(tmp_path, monkeypatch, [
        "--dataset", "pbp", "--dataset", "schedules", "--start", "2024", "--end", "2024",
        "--failures-file", str(failed)])
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, RuntimeError)
    assert not failed.exists()


def test_usage_error_exits_2(tmp_path, monkeypatch):
    result, _ = _ingest(tmp_path, monkeypatch, ["--dataset", "no_such_dataset"])
    assert result.exit_code == 2, result.output


def test_help_documents_the_exit_codes_and_the_file_option():
    out = CliRunner().invoke(main, ["ingest", "--help"]).output
    assert "Exit codes:" in out
    for line in ("0  Every requested dataset loaded",
                 "3  The run finished, but some datasets or seasons failed to load",
                 "1  The run stopped with an error",
                 "2  A usage error"):
        assert line in out, out
    assert "--failures-file" in out


def test_failure_labels_take_the_label_before_the_first_colon():
    failures = ["pbp 2025: HTTP Error 404: Not Found", "pbp 2025: retry: again",
                "schedules: Binder Error: x", "rosters 2024: boom"]
    assert ing.failure_labels(failures) == ["pbp 2025", "schedules", "rosters 2024"]
    assert ing.failure_labels([]) == []


# ═══ 2b. update_db.sh on a partial and a fatal ingest ══════════════════════════

# Every call is appended to $STUB_CALLS as "<args>|<NFL_MCP_DB_PATH>". ingest writes the
# slot, writes $STUB_INGEST_FAILED to the --failures-file path and exits $STUB_INGEST_EXIT.
_STUB = r'''#!/bin/sh
printf '%s|%s\n' "$*" "${NFL_MCP_DB_PATH:-}" >> "$STUB_CALLS"
case "${1:-}" in
    validate) exit "${STUB_VALIDATE_EXIT:-0}" ;;
    upgrade)
        case "$*" in *--check*) exit "${STUB_CHECK_EXIT:-0}" ;; esac
        exit "${STUB_UPGRADE_EXIT:-0}" ;;
    ingest)
        if [ -n "${NFL_MCP_DB_PATH:-}" ]; then printf INGESTED > "$NFL_MCP_DB_PATH"; fi
        shift
        while [ $# -gt 0 ]; do
            if [ "$1" = --failures-file ] && [ -z "${STUB_NO_FILE:-}" ]; then
                printf '%b' "${STUB_INGEST_FAILED:-}" > "$2"
            fi
            shift
        done
        exit "${STUB_INGEST_EXIT:-0}" ;;
esac
exit 0
'''


def _sim(tmp_path):
    data, bin_dir = tmp_path / "data", tmp_path / "bin"
    data.mkdir(parents=True)
    bin_dir.mkdir()
    (bin_dir / "nfl-mcp").write_text(_STUB)
    (bin_dir / "nfl-mcp").chmod(0o755)
    text = (ROOT / "update_db.sh").read_text()
    assert 'DB_DIR="/data"' in text
    (tmp_path / "update_db.sh").write_text(text.replace('DB_DIR="/data"', f'DB_DIR="{data}"'))
    # Merge: update_db.sh refuses to swap without the shared validate_db.sh beside it.
    (tmp_path / "validate_db.sh").write_text("#!/bin/sh\nexit 0\n")
    (data / "nflread_a.duckdb").write_text("SEED_V1")
    os.symlink("nflread_a.duckdb", data / "nflread.duckdb")
    return {"update": tmp_path / "update_db.sh", "data": data, "bin": bin_dir,
            "calls": tmp_path / "calls.txt"}


def _run(sim, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NFL_MCP_", "STUB_"))}
    env.update(PATH=f"{sim['bin']}{os.pathsep}{os.environ['PATH']}", STUB_CALLS=str(sim["calls"]),
               **extra)
    return subprocess.run(["sh", str(sim["update"])], capture_output=True, text=True, env=env,
                          check=False, timeout=60)


def _names(sim):
    if not sim["calls"].exists():
        return []
    return [line.split("|", 1)[0].split()[0] for line in sim["calls"].read_text().splitlines()]


def _active(sim):
    return os.readlink(sim["data"] / "nflread.duckdb")


def _lines_with(out, word):
    return [line for line in out.splitlines() if word in line]


def test_partial_ingest_goes_on_to_upgrade_validate_and_swap(tmp_path):
    sim = _sim(tmp_path)
    res = _run(sim, STUB_INGEST_EXIT="3", STUB_INGEST_FAILED=r"pbp 2026\nschedules 2026\n")
    assert res.returncode == 0, res.stdout + res.stderr
    assert _names(sim) == ["ingest", "upgrade", "validate"]
    assert _active(sim) == "nflread_b.duckdb"
    (warning,) = _lines_with(res.stdout, "WARNING")
    assert "ingest failed to load: pbp 2026, schedules 2026." in warning
    assert "Continuing with the schema upgrade and validation" in warning
    assert "Update complete" in res.stdout
    assert not (sim["data"] / "nflread_b.duckdb.failed").exists()


def test_ingest_gets_the_failures_file_next_to_the_target(tmp_path):
    sim = _sim(tmp_path)
    assert _run(sim).returncode == 0
    args, path = sim["calls"].read_text().splitlines()[0].split("|", 1)
    target = sim["data"] / "nflread_b.duckdb"
    assert path == str(target)
    assert args == f"ingest --failures-file {target}.failed"


def test_partial_ingest_without_a_list_still_warns(tmp_path):
    sim = _sim(tmp_path)
    res = _run(sim, STUB_INGEST_EXIT="3", STUB_NO_FILE="1")
    assert res.returncode == 0, res.stdout + res.stderr
    (warning,) = _lines_with(res.stdout, "WARNING")
    assert "unknown, see the ingest output above" in warning
    assert _active(sim) == "nflread_b.duckdb"


def test_partial_ingest_then_failed_validation_keeps_the_active_slot(tmp_path):
    # validate stays the last check before the swap.
    sim = _sim(tmp_path)
    res = _run(sim, STUB_INGEST_EXIT="3", STUB_INGEST_FAILED=r"pbp 2026\n", STUB_VALIDATE_EXIT="1")
    assert res.returncode != 0
    assert _names(sim) == ["ingest", "upgrade", "validate"]
    assert "validation failed" in res.stdout
    assert _active(sim) == "nflread_a.duckdb"
    assert (sim["data"] / "nflread_a.duckdb").read_text() == "SEED_V1"


def test_partial_ingest_then_failed_upgrade_keeps_the_active_slot(tmp_path):
    sim = _sim(tmp_path)
    res = _run(sim, STUB_INGEST_EXIT="3", STUB_INGEST_FAILED=r"pbp 2026\n", STUB_UPGRADE_EXIT="1")
    assert res.returncode != 0
    assert _names(sim) == ["ingest", "upgrade"]
    assert _active(sim) == "nflread_a.duckdb"


@pytest.mark.parametrize("code", ["1", "2", "4", "137"])
def test_fatal_ingest_stops_and_keeps_the_active_slot(tmp_path, code):
    sim = _sim(tmp_path)
    res = _run(sim, STUB_INGEST_EXIT=code, STUB_INGEST_FAILED=r"pbp 2026\n")
    assert res.returncode == 1, res.stdout + res.stderr
    assert _names(sim) == ["ingest"]
    (error,) = _lines_with(res.stdout, "ERROR")
    assert f"ingest failed (exit {code})" in error and "Keeping nflread_a.duckdb active" in error
    assert not _lines_with(res.stdout, "WARNING")
    assert _active(sim) == "nflread_a.duckdb"
    assert (sim["data"] / "nflread_a.duckdb").read_text() == "SEED_V1"


def test_clean_ingest_logs_no_warning(tmp_path):
    sim = _sim(tmp_path)
    res = _run(sim)
    assert res.returncode == 0, res.stdout + res.stderr
    assert not _lines_with(res.stdout, "WARNING") and not _lines_with(res.stdout, "ERROR")
    assert _names(sim) == ["ingest", "upgrade", "validate"]
    assert _active(sim) == "nflread_b.duckdb"


def test_a_stale_failures_file_from_an_earlier_run_is_dropped(tmp_path):
    sim = _sim(tmp_path)
    (sim["data"] / "nflread_b.duckdb.failed").write_text("old 1999\n")
    res = _run(sim, STUB_INGEST_EXIT="3", STUB_NO_FILE="1")
    (warning,) = _lines_with(res.stdout, "WARNING")
    assert "old 1999" not in warning


def _code_lines(name):
    return [ln for ln in (ROOT / name).read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def test_script_checks_the_same_partial_code_as_the_cli():
    lines = _code_lines("update_db.sh")
    checks = [ln for ln in lines if "INGEST_RC" in ln and "-eq" in ln]
    assert len(checks) == 1, checks
    assert re.search(rf'"\$INGEST_RC" -eq {cli.INGEST_PARTIAL_EXIT}\b', checks[0]), checks[0]


def test_script_order_and_swap_are_unchanged():
    lines = _code_lines("update_db.sh")

    def first(needle):
        return next(i for i, ln in enumerate(lines) if needle in ln)
    assert (first('cp "$DB_DIR/$CURRENT" "$TARGET"') < first('chmod u+w "$TARGET"')
            < first("nfl-mcp ingest") < first("nfl-mcp upgrade --db") < first("nfl-mcp validate")
            < first('chmod a-w "$TARGET"') < first("ln -sfn"))
    assert [ln.strip() for ln in lines if "ln -" in ln] == ['ln -sfn "$(basename "$TARGET")" "$SYMLINK"']
    assert lines[0] == "set -eu"   # the first line after the #!/bin/sh shebang


def test_entrypoint_first_start_ingest_is_unchanged():
    # A cold first start still tolerates failures with --allow-failures (exit 0).
    # Merge: the branch ingests into slot A's scratch name and renames it into place, so a
    # load that dies halfway never leaves a partial file under the served name.
    lines = _code_lines("entrypoint.sh")
    assert [ln.strip() for ln in lines if "nfl-mcp ingest" in ln] == [
        'NFL_MCP_DB_PATH="$SLOT_A_TMP" nfl-mcp ingest --allow-failures']
    assert 'mv "$SLOT_A_TMP" "$SLOT_A"' in [ln.strip() for ln in lines]


@pytest.mark.parametrize("script", ["update_db.sh", "entrypoint.sh"])
def test_scripts_parse_as_posix_shell(script):
    res = subprocess.run(["sh", "-n", str(ROOT / script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


# ═══ 3. version ════════════════════════════════════════════════════════════════

def test_version_is_0_7_3():
    assert __version__ == "0.7.3"


def test_schema_version_stays_4():
    assert migrate.SCHEMA_VERSION == 4
