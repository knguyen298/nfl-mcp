"""Tests for argument, error and exit paths the other test files do not reach.

Grouped by module. Everything here is a unit test: no loaded database and no network.
Loaders, the database layer and the nflverse/ESPN calls are replaced, as in the other
test files. Where a test needs a database file it makes its own in tmp_path.
"""

import asyncio
import contextlib
import datetime as dt
import json
import logging
import sys
import threading
import types
from decimal import Decimal
from types import SimpleNamespace

import duckdb
import polars as pl
import pytest
from click.testing import CliRunner

import nfl_mcp.cli as cli
import nfl_mcp.database as database
import nfl_mcp.ingest as ingest
import nfl_mcp.migrate as migrate
import nfl_mcp.server as server
import nfl_mcp.tools as tools
import nfl_mcp.validate as validate
from nfl_mcp.cli import main
from nfl_mcp.registry import DatasetDef

pytestmark = pytest.mark.unit


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mem():
    """Fresh in-memory DuckDB connection with the metadata table."""
    c = duckdb.connect(":memory:")
    ingest._ensure_metadata_table(c)
    yield c
    c.close()


def _empty_db(path):
    """A valid, empty DuckDB file."""
    duckdb.connect(str(path)).close()
    return path


def _install_fake_nflreadpy(monkeypatch, **loaders):
    fake = types.ModuleType("nflreadpy")
    for name, fn in loaders.items():
        setattr(fake, name, fn)
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)


def _raise(exc):
    def fn(*args, **kwargs):
        raise exc
    return fn


# ── cli ─────────────────────────────────────────────────────────────────────────

class TestCliRebuild:
    def test_missing_database_exits_nonzero(self, runner, tmp_path):
        result = runner.invoke(main, ["rebuild", "--db", str(tmp_path / "none.duckdb")])
        assert result.exit_code == 1
        assert "Database not found" in result.output

    def test_rebuilds_the_derived_tables(self, runner, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr("nfl_mcp.ingest.rebuild_derived_tables", lambda conn: calls.append(conn))
        result = runner.invoke(main, ["rebuild", "--db", str(_empty_db(tmp_path / "a.duckdb"))])
        assert result.exit_code == 0
        assert "Derived tables rebuilt." in result.output
        assert len(calls) == 1

    def test_uses_the_configured_path_without_db_option(self, runner, monkeypatch, tmp_path):
        db = _empty_db(tmp_path / "configured.duckdb")
        monkeypatch.setattr("nfl_mcp.config.get_duckdb_path", lambda: db)
        monkeypatch.setattr("nfl_mcp.ingest.rebuild_derived_tables", lambda conn: None)
        assert runner.invoke(main, ["rebuild"]).exit_code == 0

    def test_failure_exits_nonzero_and_keeps_tables(self, runner, monkeypatch, tmp_path):
        monkeypatch.setattr("nfl_mcp.ingest.rebuild_derived_tables", _raise(duckdb.Error("boom")))
        result = runner.invoke(main, ["rebuild", "--db", str(_empty_db(tmp_path / "a.duckdb"))])
        assert result.exit_code == 1
        assert "Rebuild failed, previous derived tables kept" in result.output


class TestCliUpgradeErrors:
    @pytest.fixture
    def bad_db(self, tmp_path):
        path = tmp_path / "bad.duckdb"
        path.write_bytes(b"this is not a duckdb file " * 200)
        return path

    def test_check_reports_an_unreadable_file(self, runner, bad_db):
        result = runner.invoke(main, ["upgrade", "--check", "--db", str(bad_db)])
        assert result.exit_code == 1
        assert "Cannot check" in result.output

    def test_upgrade_reports_a_file_it_cannot_open_for_writing(self, runner, bad_db):
        result = runner.invoke(main, ["upgrade", "--db", str(bad_db)])
        assert result.exit_code == 1
        assert "for writing" in result.output

    def test_upgrade_reports_a_failed_step(self, runner, monkeypatch, tmp_path):
        monkeypatch.setattr("nfl_mcp.migrate.upgrade", _raise(duckdb.Error("step broke")))
        result = runner.invoke(main, ["upgrade", "--db", str(_empty_db(tmp_path / "a.duckdb"))])
        assert result.exit_code == 1
        assert "failed" in result.output and "step broke" in result.output


class TestCliInitAndDoctor:
    def test_init_warns_when_ingest_reports_failures(self, runner, monkeypatch, tmp_path):
        monkeypatch.setattr("nfl_mcp.config.load_config", lambda: {})
        monkeypatch.setattr("nfl_mcp.config.save_config", lambda cfg: tmp_path / "config.json")
        monkeypatch.setattr("nfl_mcp.ingest.run_ingest_datasets",
                            lambda **kwargs: ["schedules 2024: boom"])
        monkeypatch.setattr("nfl_mcp.cli._setup_client_interactive", lambda cfg: None)
        result = runner.invoke(main, ["init", "--start", "2024", "--end", "2024"], input="n\n\nn\nn\n")
        assert result.exit_code == 0
        assert "some data failed to load: schedules 2024: boom" in result.output

    def test_doctor_schema_line_says_a_newer_database_needs_an_update(self, monkeypatch, capsys):
        status = migrate.UpgradeStatus(version=2, target=1, steps=[], upgrade_needed=False, newer=True)

        @contextlib.contextmanager
        def fake_connection():
            yield object()

        monkeypatch.setattr("nfl_mcp.database.get_db_connection", fake_connection)
        monkeypatch.setattr("nfl_mcp.migrate.check", lambda conn: status)
        cli._doctor_schema_line()
        assert "the database is newer, update nfl-mcp" in capsys.readouterr().out


# ── database ────────────────────────────────────────────────────────────────────

class TestDatabaseConfigProblems:
    def test_memory_limit_duckdb_rejects_is_reported(self, monkeypatch):
        monkeypatch.setenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", "0MB")
        monkeypatch.delenv("NFL_MCP_DUCKDB_THREADS", raising=False)
        problem = database.connection_config_problem()
        assert problem and "NFL_MCP_DUCKDB_MEMORY_LIMIT='0MB' is not valid" in problem

    @pytest.mark.parametrize("value", ["0", "-2", "two", "1.5"])
    def test_threads_must_be_a_positive_whole_number(self, monkeypatch, value):
        monkeypatch.delenv("NFL_MCP_DUCKDB_MEMORY_LIMIT", raising=False)
        monkeypatch.setenv("NFL_MCP_DUCKDB_THREADS", value)
        problem = database.connection_config_problem()
        assert problem and "NFL_MCP_DUCKDB_THREADS" in problem and "positive whole number" in problem


# ── validate ────────────────────────────────────────────────────────────────────

class TestValidateRunChecks:
    def test_missing_current_slot_is_reported_and_the_new_one_still_checked(self, monkeypatch, tmp_path):
        monkeypatch.setattr(validate, "_run", lambda new, current: [validate.CheckResult("stub", True, "ok")])
        results = validate.run_checks(_empty_db(tmp_path / "new.duckdb"), tmp_path / "missing.duckdb")
        assert [(r.name, r.ok) for r in results] == [("open_current", False), ("stub", True)]
        assert "cannot open" in results[0].detail

    def test_a_failing_query_becomes_a_failed_check(self, monkeypatch, tmp_path):
        monkeypatch.setattr(validate, "_run", _raise(duckdb.Error("query broke")))
        results = validate.run_checks(_empty_db(tmp_path / "new.duckdb"))
        assert [(r.name, r.ok) for r in results] == [("run_checks", False)]
        assert "query failed: query broke" in results[0].detail


# ── migrate ─────────────────────────────────────────────────────────────────────

class TestMigrateReloadPbp:
    @pytest.fixture
    def recorded(self, monkeypatch):
        calls = {"create": [], "season": []}
        monkeypatch.setattr(ingest, "_clip_to_window", lambda seasons, defn: list(seasons))
        monkeypatch.setattr(ingest, "_create_plays_table", lambda conn, sample: calls["create"].append(sample))
        monkeypatch.setattr(ingest, "_ingest_pbp_season", lambda conn, season: calls["season"].append(season))
        return calls

    def test_existing_table_reloads_each_season_without_sampling(self, monkeypatch, recorded):
        _install_fake_nflreadpy(monkeypatch, load_pbp=_raise(AssertionError("must not sample")))
        migrate._reload_pbp(None, [2023, 2024], table_exists=True)
        assert recorded == {"create": [], "season": [2023, 2024]}

    def test_missing_table_is_created_from_the_newest_season_that_loads(self, monkeypatch, recorded, capsys):
        sample = pl.DataFrame({"season": [2023]})

        def load_pbp(seasons):
            if seasons == [2024]:
                raise RuntimeError("not published yet")
            return sample

        _install_fake_nflreadpy(monkeypatch, load_pbp=load_pbp)
        migrate._reload_pbp(None, [2023, 2024], table_exists=False)
        assert recorded["create"] == [sample]
        assert recorded["season"] == [2023, 2024]
        assert "pbp 2024 not available" in capsys.readouterr().out

    def test_missing_table_with_no_loadable_season_raises(self, monkeypatch, recorded):
        _install_fake_nflreadpy(monkeypatch, load_pbp=_raise(RuntimeError("down")))
        with pytest.raises(RuntimeError, match="could not load play-by-play"):
            migrate._reload_pbp(None, [2024], table_exists=False)
        assert recorded == {"create": [], "season": []}


def test_run_raw_reloads_pbp_through_the_soft_failure_wrapper(monkeypatch):
    reloads = []
    monkeypatch.setattr(migrate, "_columns", lambda conn: {})
    monkeypatch.setattr(migrate, "_gaps", lambda columns, tables: (["plays"], []))
    monkeypatch.setattr(migrate, "_reload_pbp", lambda conn, seasons, exists: reloads.append((seasons, exists)))
    monkeypatch.setattr(ingest, "_clip_to_window", lambda seasons, defn: [2023, 2024])
    migrate._run_raw(None)
    assert reloads == [([2023, 2024], False)]


# ── ingest ──────────────────────────────────────────────────────────────────────

class TestIngestHelpers:
    def test_transaction_keeps_the_original_error_when_rollback_fails(self, capsys):
        class Conn:
            def execute(self, sql):
                if sql == "ROLLBACK":
                    raise duckdb.Error("rollback broke")

        with pytest.raises(KeyError, match="original"):
            with ingest._transaction(Conn()):
                raise KeyError("original")
        assert "Rollback failed: rollback broke" in capsys.readouterr().out

    def test_nflreadpy_version_is_none_when_not_installed(self, monkeypatch):
        import importlib.metadata as metadata

        monkeypatch.setattr(metadata, "version", _raise(metadata.PackageNotFoundError("nflreadpy")))
        assert ingest._nflreadpy_version() is None

    def test_record_loaded_on_a_database_without_the_version_column(self, monkeypatch):
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE _ingest_metadata (dataset_id VARCHAR NOT NULL, table_name VARCHAR NOT NULL, "
                     "season INTEGER, row_count BIGINT, loaded_at TIMESTAMP NOT NULL, loader_fn VARCHAR)")
        ingest._record_loaded(conn, "schedules", "schedules", "load_schedules", 7, season=2024)
        assert conn.execute("SELECT dataset_id, season, row_count FROM _ingest_metadata").fetchall() \
            == [("schedules", 2024, 7)]
        conn.close()

    def test_rebuild_rollback_failure_is_reported_and_the_error_re_raised(self, monkeypatch, capsys):
        class Conn:
            def execute(self, sql):
                if sql == "ROLLBACK":
                    raise duckdb.Error("rollback broke")

        monkeypatch.setattr(ingest, "_create_aggregate_tables", _raise(RuntimeError("table broke")))
        with pytest.raises(RuntimeError, match="table broke"):
            ingest.rebuild_derived_tables(Conn())
        assert "Rollback after the failed rebuild also failed: rollback broke" in capsys.readouterr().out


class TestRestampLegacySeasons:
    DEFN = DatasetDef(dataset_id="fake_depth", loader_fn="load_fake_depth", table_name="fake_depth",
                      seasonal=True, default=False, stamp_season=True)

    def test_empty_fetch_keeps_the_rows_and_records_a_failure(self, mem, capsys):
        ingest._FAILURES.clear()
        done = ingest._restamp_legacy_seasons(
            mem, self.DEFN, lambda seasons, **kw: pl.DataFrame({"x": []}), [2024])
        assert done == {}
        assert "Failed to re-fetch fake_depth 2024: no data returned" in capsys.readouterr().out
        assert len(ingest._FAILURES) == 1
        ingest._FAILURES.clear()

    def test_a_failed_swap_is_rolled_back_and_recorded(self, mem, monkeypatch, capsys):
        mem.execute("CREATE TABLE fake_depth (x BIGINT, season BIGINT)")
        mem.execute("INSERT INTO fake_depth VALUES (1, NULL)")
        monkeypatch.setattr(ingest, "_replace_season", _raise(duckdb.Error("write broke")))
        ingest._FAILURES.clear()
        done = ingest._restamp_legacy_seasons(
            mem, self.DEFN, lambda seasons, **kw: pl.DataFrame({"x": [1]}), [2024])
        assert done == {}
        assert "Failed to restamp fake_depth, kept the previous rows: write broke" in capsys.readouterr().out
        assert mem.execute("SELECT COUNT(*) FROM fake_depth WHERE season IS NULL").fetchone()[0] == 1
        assert len(ingest._FAILURES) == 1
        ingest._FAILURES.clear()


class TestGenericDatasetWriteFailures:
    def test_static_dataset_keeps_the_old_table_when_the_write_fails(self, mem, monkeypatch, capsys):
        defn = DatasetDef(dataset_id="fake_static", loader_fn="load_fake_static", table_name="fake_static",
                          seasonal=False, default=False)
        _install_fake_nflreadpy(monkeypatch, load_fake_static=lambda **kw: pl.DataFrame({"team": ["KC"]}))
        monkeypatch.setattr(ingest, "_write_df_to_table", _raise(duckdb.Error("write broke")))
        ingest._FAILURES.clear()
        assert ingest._ingest_generic_dataset(mem, defn, seasons=[2024], fresh=False) == 0
        assert "Failed to write, kept the previous table: write broke" in capsys.readouterr().out
        assert len(ingest._FAILURES) == 1
        ingest._FAILURES.clear()

    def test_seasonal_dataset_stops_when_the_legacy_check_fails(self, mem, monkeypatch, capsys):
        defn = DatasetDef(dataset_id="fake_depth", loader_fn="load_fake_depth", table_name="fake_depth",
                          seasonal=True, default=False, stamp_season=True)
        _install_fake_nflreadpy(monkeypatch, load_fake_depth=_raise(AssertionError("must not fetch")))
        monkeypatch.setattr(ingest, "_legacy_restamp_plan", _raise(duckdb.Error("plan broke")))
        ingest._FAILURES.clear()
        assert ingest._ingest_generic_dataset(mem, defn, seasons=[2024], fresh=False) == 0
        assert "Failed to prepare fake_depth: plan broke" in capsys.readouterr().out
        assert len(ingest._FAILURES) == 1
        ingest._FAILURES.clear()


class _DummyResult:
    def fetchall(self):
        return []


class _DummyConn:
    def __init__(self):
        self.closed = False

    def execute(self, sql, params=None):
        return _DummyResult()

    def close(self):
        self.closed = True


def test_run_ingest_datasets_records_a_dataset_that_raises(monkeypatch, tmp_path, capsys):
    conn = _DummyConn()
    monkeypatch.setattr("nfl_mcp.ingest.duckdb.connect", lambda path: conn)
    monkeypatch.setattr("nfl_mcp.ingest._ensure_metadata_table", lambda c: None)
    monkeypatch.setattr("nfl_mcp.ingest._ingest_generic_dataset", _raise(RuntimeError("loader broke")))
    monkeypatch.setattr("nfl_mcp.ingest._create_fantasy_tables", lambda c: None)
    monkeypatch.setattr("nfl_mcp.ingest._create_team_records", lambda c: None)

    failures = ingest.run_ingest_datasets(
        dataset_ids=["schedules"], start=2024, end=2024, db_path=str(tmp_path / "x.duckdb"))

    assert len(failures) == 1 and "loader broke" in failures[0]
    assert "Failed: loader broke" in capsys.readouterr().out
    assert conn.closed is True
    ingest._FAILURES.clear()


# ── server ──────────────────────────────────────────────────────────────────────

class TestServerEdges:
    def test_clean_turns_a_non_finite_decimal_into_null(self):
        assert server._clean(Decimal("NaN")) is None
        assert server._clean(Decimal("Infinity")) is None

    @pytest.mark.parametrize("raw", ["abc", "0", "-3"])
    def test_invalid_concurrency_falls_back_to_the_default(self, raw, monkeypatch, caplog):
        monkeypatch.setattr(server, "_TOOL_LIMITER", None)
        monkeypatch.setenv("NFL_MCP_MAX_CONCURRENT_QUERIES", raw)

        async def tokens():
            return server.get_tool_limiter().total_tokens

        with caplog.at_level(logging.WARNING, logger="nfl-mcp"):
            assert asyncio.run(tokens()) == server._DEFAULT_MAX_CONCURRENT
        assert "Invalid NFL_MCP_MAX_CONCURRENT_QUERIES" in caplog.text
        monkeypatch.setattr(server, "_TOOL_LIMITER", None)

    def test_unknown_resource_uri_raises(self):
        with pytest.raises(ValueError, match="Unknown resource"):
            server._read_resource_text("nfl://nothing-here")

    def test_table_schema_resource_raises_when_the_table_is_unknown(self, monkeypatch):
        monkeypatch.setattr(server, "nfl_schema", lambda table=None: {"error": "Table 'zzz' not found"})
        with pytest.raises(ValueError, match="not found"):
            server._read_resource_text("nfl://schema/zzz")

    def test_resource_templates_list_the_table_schema_template(self):
        templates = asyncio.run(server.list_resource_templates())
        assert [t.uriTemplate for t in templates] == ["nfl://schema/{table}"]

    def test_call_tool_rejects_arguments_that_are_not_an_object(self):
        result = asyncio.run(server.call_tool("nfl_query", ["SELECT 1"]))
        payload = json.loads(result.content[0].text)
        assert result.isError is True
        assert payload["code"] == "INVALID_ARGUMENTS"
        assert payload["details"]["tool"] == "nfl_query"


# ── tools: argument checks ──────────────────────────────────────────────────────

class TestToolArguments:
    def test_int_arg_rejects_a_bool(self):
        with pytest.raises(tools._BadArgument, match="must be an integer"):
            tools._int_arg("season", True)

    def test_int_arg_accepts_an_integral_float_and_rejects_a_fraction(self):
        assert tools._int_arg("season", 2024.0) == 2024
        with pytest.raises(tools._BadArgument, match="must be an integer"):
            tools._int_arg("season", 2024.5)

    def test_columns_must_be_a_list_of_strings(self):
        with pytest.raises(tools._BadArgument, match="list of plays column names"):
            tools._extra_play_columns("epa")
        with pytest.raises(tools._BadArgument, match="list of plays column names"):
            tools._extra_play_columns(["epa", 3])

    def test_columns_need_a_readable_plays_schema(self, monkeypatch):
        monkeypatch.setattr(tools, "_live_names", lambda sql: [])
        with pytest.raises(tools._BadArgument) as raised:
            tools._extra_play_columns(["epa"])
        assert raised.value.as_error()["code"] == "SCHEMA_UNAVAILABLE"

    def test_team_stats_filter_rejects_an_unknown_season_type(self):
        with pytest.raises(tools._BadArgument, match="Unknown season_type 'bogus'"):
            tools._team_stats_filter(None, None, None, "bogus")

    def test_team_stats_filter_requires_a_season_type(self):
        with pytest.raises(tools._BadArgument, match="season_type is required"):
            tools._team_stats_filter(None, None, None, None)

    def test_search_plays_player_role_needs_a_player(self):
        result = tools.nfl_search_plays(player_role="passer", season_type="REG")
        assert result["code"] == "MISSING_PLAYER"


class TestRangeReason:
    @staticmethod
    def _span(monkeypatch, lo=2020, hi=2024):
        monkeypatch.setattr(tools, "_execute", lambda sql, params=None: [{"lo": lo, "hi": hi}])

    def test_season_from_after_the_last_season(self, monkeypatch):
        self._span(monkeypatch)
        reason = tools._range_reason("plays", season_from=2030)
        assert reason == "season_from 2030 is after the last loaded season (2024)."

    def test_season_to_before_the_first_season(self, monkeypatch):
        self._span(monkeypatch)
        reason = tools._range_reason("plays", season_to=2010)
        assert reason == "season_to 2010 is before the first loaded season (2020)."


class TestIsoUtc:
    def test_an_aware_timestamp_is_converted_to_utc(self):
        zone = dt.timezone(dt.timedelta(hours=-5))
        assert tools._iso_utc(dt.datetime(2024, 1, 1, 12, tzinfo=zone)) == "2024-01-01T17:00:00.000000Z"

    def test_a_naive_timestamp_is_taken_as_utc_and_other_values_pass_through(self):
        assert tools._iso_utc(dt.datetime(2024, 1, 1, 12)) == "2024-01-01T12:00:00.000000Z"
        assert tools._iso_utc("x") == "x"


def test_newest_seasons_is_empty_when_no_table_has_a_season_column(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda sql, params=None: [])
    assert tools._newest_seasons(["rosters"]) == {}


# ── tools: players ──────────────────────────────────────────────────────────────

class TestToolPlayers:
    def test_ambiguous_response_carries_the_truncated_flag(self):
        result = tools._ambiguous([{"gsis_id": "a"}, {"gsis_id": "b"}], True, "Smith")
        assert result["truncated"] is True and result["entity"] == "Smith" and result["count"] == 2

    def test_lookup_hints_when_more_players_match(self, monkeypatch):
        monkeypatch.setattr(tools, "_find_players", lambda *a, **k: ([{"gsis_id": "a"}], True))
        result = tools.nfl_player_lookup("Smith")
        assert result["truncated"] is True
        assert "Add team, season or position" in result["hint"]

    def test_lookup_takes_a_gsis_shaped_player_as_an_id(self, monkeypatch):
        seen = []
        monkeypatch.setattr(tools, "_find_players", lambda *a, **k: seen.append((a, k)) or ([], False))
        tools.nfl_player_lookup(player=" 00-0033873 ")
        assert seen == [((), {"gsis_id": "00-0033873", "limit": 1})]

    def test_one_player_returns_the_error_when_resolution_fails(self, monkeypatch):
        monkeypatch.setattr(tools, "_resolve_player", _raise(duckdb.Error("players table broke")))
        assert tools._one_player("Smith", None, "x") == (None, {"error": "players table broke"})

    def test_one_player_with_no_match_returns_no_rows_and_a_reason(self, monkeypatch):
        monkeypatch.setattr(tools, "_resolve_player", lambda *a, **k: (None, None))
        found, early = tools._one_player("Smit", None, "x")
        assert found is None
        assert early["x"] == [] and early["count"] == 0 and early["truncated"] is False
        assert "No player matches 'Smit'" in early["reason"]

    def test_one_player_without_a_player_resolves_nothing(self, monkeypatch):
        monkeypatch.setattr(tools, "_resolve_player", lambda *a, **k: pytest.fail("must not resolve"))
        assert tools._one_player(None, None, "x") == (None, None)

    def test_compare_players_reports_an_entity_that_matches_nobody(self, monkeypatch):
        monkeypatch.setattr(tools, "_resolve_player", lambda *a, **k: (None, None))
        result = tools._compare_players("Nobody One", "Nobody Two", None, None, None, None)
        assert [e["player_name"] for e in result["players"]] == ["Nobody One", "Nobody Two"]
        assert all(e["player_id"] is None and "No player matches" in e["reason"] for e in result["players"])


class TestSearchPlaysPlayerResolution:
    def test_resolver_error_by_name_is_returned(self, monkeypatch):
        monkeypatch.setattr(tools, "_resolve_player", _raise(duckdb.Error("players broke")))
        assert tools.nfl_search_plays(player="Smith", season_type="REG")["error"] == "players broke"

    def test_ambiguous_name_is_returned_as_is(self, monkeypatch):
        ambiguous = {"ambiguous": True, "candidates": [], "count": 0}
        monkeypatch.setattr(tools, "_resolve_player", lambda *a, **k: (None, ambiguous))
        assert tools.nfl_search_plays(player="Smith", season_type="REG") == ambiguous

    def test_unknown_name_gives_an_empty_result_with_a_reason(self, monkeypatch):
        monkeypatch.setattr(tools, "_resolve_player", lambda *a, **k: (None, None))
        result = tools.nfl_search_plays(player="Nobody", season_type="REG")
        assert result["rows"] == [] and "No player matches 'Nobody'" in result["reason"]

    def test_resolver_error_by_id_is_returned(self, monkeypatch):
        monkeypatch.setattr(tools, "_resolve_player", _raise(duckdb.IOException("disk broke")))
        assert tools.nfl_search_plays(player_id="00-0000001", season_type="REG")["error"] == "disk broke"


class TestSnapCountsPlayerId:
    def test_unknown_player_id(self, monkeypatch):
        monkeypatch.setattr(tools, "_find_players", lambda *a, **k: ([], False))
        result = tools.nfl_snap_counts(player_id="00-0000000", season_type="REG")
        assert result["code"] == "UNKNOWN_PLAYER_ID"

    def test_player_without_a_pfr_id_has_no_snap_counts(self, monkeypatch):
        found = [{"gsis_id": "00-0000001", "display_name": "Some Player", "pfr_id": None}]
        monkeypatch.setattr(tools, "_find_players", lambda *a, **k: (found, False))
        result = tools.nfl_snap_counts(player_id="00-0000001", season_type="REG")
        assert result["snap_counts"] == [] and result["player_id"] == "00-0000001"
        assert "No PFR id on file for Some Player" in result["reason"]


class TestFtnAndSeparationPlayers:
    def test_ftn_charting_returns_a_resolver_error(self, monkeypatch):
        monkeypatch.setattr(tools, "_resolve_player", _raise(duckdb.Error("players broke")))
        assert tools.nfl_ftn_charting(player="Smith", season_type="REG")["error"] == "players broke"

    def test_ftn_charting_for_an_unknown_player_is_empty_with_a_reason(self, monkeypatch):
        monkeypatch.setattr(tools, "_resolve_player", lambda *a, **k: (None, None))
        result = tools.nfl_ftn_charting(player="Nobody", season_type="REG")
        assert result["charting"] == {} and result["total_plays"] == 0
        assert "No player matches 'Nobody'" in result["reason"]

    def test_separation_opportunity_returns_an_ambiguous_player(self, monkeypatch):
        ambiguous = {"ambiguous": True, "candidates": [], "count": 0}
        monkeypatch.setattr(tools, "_resolve_player", lambda *a, **k: (None, ambiguous))
        assert tools.nfl_separation_opportunity(player_id="00-0000001", season_type="REG") == ambiguous


class TestFantasyRankingsPlayers:
    def test_resolver_error_is_returned(self, monkeypatch):
        monkeypatch.setattr(tools, "_resolve_player", _raise(duckdb.Error("players broke")))
        monkeypatch.setattr(tools, "_execute", lambda *a, **k: pytest.fail("rankings query must not run"))
        assert tools.nfl_fantasy_rankings(player="Smith")["error"] == "players broke"

    def test_position_is_an_exact_match(self, monkeypatch):
        captured = []
        monkeypatch.setattr(tools, "_execute", lambda sql, params=None: captured.append((sql, params)) or [])
        tools.nfl_fantasy_rankings(position=" wr ")
        sql, params = captured[0]
        assert "upper(pos) = ?" in sql and "ILIKE" not in sql
        assert params[0] == "WR"

    def test_team_fa_lists_free_agents_not_a_franchise(self, monkeypatch):
        captured = []
        monkeypatch.setattr(tools, "_execute", lambda sql, params=None: captured.append((sql, params)) or [])
        tools.nfl_fantasy_rankings(team=" fa ")
        sql, params = captured[0]
        assert "team = 'FA'" in sql
        assert not any(str(p).upper() == "FA" for p in params or [])


# ── tools: SQL error help ───────────────────────────────────────────────────────

class TestSqlErrorHelp:
    COLUMN_ERROR = 'Binder Error: Referenced column "play_epa" not found in FROM clause!'

    def test_column_error_without_the_statement_offers_every_live_column(self, monkeypatch):
        seen = []

        def fake_execute(sql, params=None):
            seen.append(params)
            return [{"column_name": c} for c in ("epa", "qb_epa", "yards_gained")]

        monkeypatch.setattr(tools, "_execute", fake_execute)
        detail = tools._sql_error_detail(duckdb.BinderException(self.COLUMN_ERROR))
        assert detail["did_you_mean"][:2] == ["epa", "qb_epa"]
        assert seen == [None]  # one lookup, not narrowed to any table

    def test_queried_tables_is_none_when_duckdb_cannot_serialize(self, monkeypatch):
        monkeypatch.setattr(tools.duckdb, "connect", _raise(duckdb.Error("no in-memory db")))
        assert tools._queried_tables("SELECT 1 FROM plays") is None

    def test_queried_tables_is_none_for_a_statement_that_does_not_parse(self, monkeypatch):
        monkeypatch.setattr(tools, "_execute", lambda *a, **k: pytest.fail("no table lookup"))
        assert tools._queried_tables("SELEC epa FROM plays") is None


# ── tools: team tables ──────────────────────────────────────────────────────────

class TestTeamTables:
    def test_team_rows_re_raises_a_binder_error_that_is_not_about_season_type(self, monkeypatch):
        monkeypatch.setattr(tools, "_execute", _raise(duckdb.BinderException("column 'zzz' not found")))
        with pytest.raises(duckdb.BinderException, match="zzz"):
            tools._team_rows("team_offense_stats", ["KC"], (None, None, None, "REG"))

    def test_compare_teams_with_no_rows_adds_the_range_reason(self, monkeypatch):
        monkeypatch.setattr(tools, "_team_rows", lambda *a, **k: [])
        monkeypatch.setattr(tools, "_range_reason", lambda *a, **k: "Season 1900 is outside the loaded range.")
        result = tools._compare_teams("KC", "BUF", 1900, None, None, "REG")
        for team in ("KC", "BUF"):  # v0.7.2: the result also holds "definitions"
            assert result[team]["reason"].endswith(". Season 1900 is outside the loaded range.")


# ── tools: statement gate ───────────────────────────────────────────────────────

class TestStatementGateExplain:
    def test_explain_the_prefix_cannot_read(self):
        # extract_statements skips the leading ';', the EXPLAIN prefix pattern does not.
        result = tools._statement_gate("; EXPLAIN SELECT 1")
        assert result["error"] == "Could not read the EXPLAIN statement."
        assert result["statement_type"] == "EXPLAIN"

    @staticmethod
    def _second_parse(monkeypatch, outcome):
        real = duckdb.extract_statements
        calls = []

        def fake(sql):
            calls.append(sql)
            if len(calls) == 1:
                return real(sql)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(duckdb, "extract_statements", fake)

    def test_inner_statement_that_does_not_parse(self, monkeypatch):
        self._second_parse(monkeypatch, duckdb.ParserException("bad inner"))
        result = tools._statement_gate("EXPLAIN SELECT 1")
        assert result["error"].startswith("Could not parse SQL after EXPLAIN: ")
        assert result["statement_type"] == "EXPLAIN"

    def test_inner_text_that_is_not_one_statement(self, monkeypatch):
        self._second_parse(monkeypatch, [])
        result = tools._statement_gate("EXPLAIN SELECT 1")
        assert result["error"] == "Send exactly one statement per call."
        assert result["statement_type"] == "EXPLAIN"


# ── tools: timeouts and cancellation ────────────────────────────────────────────

class _SlowConn:
    """Stands in for a DuckDB connection whose query outlives the timeout."""

    def __init__(self, seconds, interrupt_error=None):
        self.seconds = seconds
        self.interrupt_error = interrupt_error
        self.interrupts = 0

    def execute(self, sql, params=None):
        threading.Event().wait(self.seconds)
        return SimpleNamespace(description=[], fetchmany=lambda n: [], fetchall=lambda: [])

    def interrupt(self):
        self.interrupts += 1
        if self.interrupt_error:
            raise self.interrupt_error


def _connection_factory(conn, delay=0.0):
    """A get_db_connection stand-in that takes `delay` seconds to hand out `conn`."""
    @contextlib.contextmanager
    def factory():
        threading.Event().wait(delay)
        yield conn
    return factory


class TestTimeouts:
    def test_fetch_capped_survives_an_interrupt_that_fails(self, monkeypatch, caplog):
        conn = _SlowConn(0.6, interrupt_error=RuntimeError("cannot interrupt"))
        monkeypatch.setattr(tools, "_query_timeout", lambda: 0.05)
        monkeypatch.setattr(tools, "get_db_connection", _connection_factory(conn))
        with caplog.at_level(logging.WARNING, logger="nfl-mcp"):
            with pytest.raises(TimeoutError, match="exceeded 0.05 second timeout"):
                tools._fetch_capped("SELECT 1", 10)
        assert conn.interrupts >= 1
        assert "Failed to interrupt timed out DuckDB query" in caplog.text

    def test_fetch_capped_worker_that_wakes_after_the_timeout_never_runs_the_query(self, monkeypatch):
        conn = _SlowConn(0, interrupt_error=None)
        ran = []
        conn.execute = lambda sql, params=None: ran.append(sql)
        monkeypatch.setattr(tools, "_query_timeout", lambda: 0.05)
        monkeypatch.setattr(tools, "get_db_connection", _connection_factory(conn, delay=0.4))
        with pytest.raises(TimeoutError):
            tools._fetch_capped("SELECT 1", 10)
        threading.Event().wait(0.5)  # let the worker reach its cancelled check
        assert ran == []

    def test_execute_worker_that_wakes_after_the_timeout_never_runs_the_query(self, monkeypatch):
        conn = _SlowConn(0)
        ran = []
        conn.execute = lambda sql, params=None: ran.append(sql)
        monkeypatch.setattr(tools, "_query_timeout", lambda: 0.05)
        monkeypatch.setattr(tools, "get_db_connection", _connection_factory(conn, delay=0.3))
        with pytest.raises(TimeoutError):
            tools._execute("SELECT 1")
        threading.Event().wait(0.4)
        assert ran == []

    def test_execute_logs_when_the_worker_thread_will_not_stop(self, monkeypatch, caplog):
        class StuckThread:
            def __init__(self, target=None, daemon=None):
                pass

            def start(self):
                pass

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return True

        monkeypatch.setattr(tools, "_query_timeout", lambda: 0.01)
        monkeypatch.setattr(tools, "threading", SimpleNamespace(
            Thread=StuckThread, Event=threading.Event, Lock=threading.Lock))
        with caplog.at_level(logging.WARNING, logger="nfl-mcp"):
            with pytest.raises(TimeoutError):
                tools._execute("SELECT 1")
        assert "still running after interrupt" in caplog.text


# ── tools: data_as_of ───────────────────────────────────────────────────────────

class TestDataAsOf:
    @pytest.fixture(autouse=True)
    def clean_cache(self, monkeypatch):
        monkeypatch.setattr(tools, "_AS_OF_CACHE", {})

    def test_unexpected_metadata_error_returns_nothing_and_caches_nothing(self, monkeypatch, tmp_path, caplog):
        db = tmp_path / "x.duckdb"
        db.write_bytes(b"x")

        @contextlib.contextmanager
        def broken():
            raise RuntimeError("metadata broke")
            yield  # makes this a generator

        monkeypatch.setattr(tools, "get_db_connection", broken)
        with caplog.at_level(logging.WARNING, logger="nfl-mcp"):
            assert tools._data_as_of(str(db)) == {}
        assert "data_as_of unavailable: metadata broke" in caplog.text
        assert tools._AS_OF_CACHE == {}

    def test_a_changed_file_replaces_the_old_cache_entry(self, monkeypatch, tmp_path):
        db = tmp_path / "x.duckdb"
        db.write_bytes(b"x")
        resolved = str(db.resolve())
        tools._AS_OF_CACHE[(resolved, 1, 1)] = {"old": "stale"}
        loaded = dt.datetime(2024, 1, 1, 12, tzinfo=dt.timezone.utc)
        conn = SimpleNamespace(execute=lambda sql: SimpleNamespace(fetchall=lambda: [("plays", loaded)]))
        monkeypatch.setattr(tools, "get_db_connection", _connection_factory(conn))
        assert tools._data_as_of(str(db)) == {"plays": "2024-01-01T12:00:00.000000Z"}
        assert list(tools._AS_OF_CACHE.values()) == [{"plays": "2024-01-01T12:00:00.000000Z"}]
        assert (resolved, 1, 1) not in tools._AS_OF_CACHE


# ── lines that only the local-database tests reached (they skip on GitHub) ──────

def test_find_players_season_range_narrows_by_roster(monkeypatch):
    seen = {}

    def fake_execute(sql, params=None):
        seen["sql"], seen["params"] = sql, params
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    assert tools._find_players("Josh Allen", season_from=2020, season_to=2023) == ([], False)
    assert "r.season >= ?" in seen["sql"] and "r.season <= ?" in seen["sql"]
    assert seen["params"][-2:] == [2020, 2023]


def test_add_version_column_on_old_metadata_table():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE _ingest_metadata (dataset_id VARCHAR, table_name VARCHAR)")
    assert not ingest._has_version_column(con)
    ingest._add_version_column(con)
    assert ingest._has_version_column(con)
    ingest._add_version_column(con)  # a second run leaves the column as it is
    assert ingest._has_version_column(con)


# ── tools: derived fantasy tables built before the season_type split (schema version 1) ──

@pytest.mark.parametrize("tool,table", [
    (tools.nfl_role_trend, "player_role_trend"),
    (tools.nfl_separation_opportunity, "player_separation_opportunity"),
    (tools.nfl_contract_value, "player_contract_value"),
])
@pytest.mark.parametrize("season_type", ["REG", "POST", "ALL"])
def test_version_1_fantasy_tables_answer_stale_for_every_season_type(
        tool, table, season_type, tmp_path, monkeypatch):
    from nfl_mcp.required_columns import REQUIRED_COLUMNS
    path = tmp_path / "v1.duckdb"
    con = duckdb.connect(str(path))
    cols = [c for c in REQUIRED_COLUMNS[table] if c != "season_type"]
    con.execute(f"CREATE TABLE {table} ({', '.join(f'{c} VARCHAR' for c in cols)})")
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    res = tool(season=2024, season_type=season_type)
    assert res["code"] == "STALE_DERIVED_TABLES"
    assert "nfl-mcp upgrade" in res["error"]
