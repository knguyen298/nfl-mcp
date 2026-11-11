"""DB-independent unit tests for query construction and validation."""

import datetime
import json
import re
import threading
import time
from contextlib import contextmanager

import inspect

import duckdb
import pytest

import nfl_mcp.server as server_module
import nfl_mcp.tools as tools
from nfl_mcp.franchises import RECORDS_FIRST_NFLVERSE_SEASON
from nfl_mcp.registry import REGISTRY
from nfl_mcp.status import Availability, ErrorCode


def test_normalize_player_name_handles_full_short_and_last_name_inputs():
    assert tools._normalize_player_name("Mahomes") == "Mahomes"
    assert tools._normalize_player_name("J.Jefferson") == "J.Jefferson"
    assert tools._normalize_player_name("Justin Jefferson") == "J.Jefferson"
    assert tools._normalize_player_name("J. Jefferson") == "J. Jefferson"


def test_search_plays_uses_parameterized_filters(monkeypatch):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_search_plays(team="KC", player="Mahomes", season=2024, max_rows=5)

    assert result["row_count"] == 0
    assert "posteam = ?" in captured["sql"]
    assert "ILIKE ?" in captured["sql"]
    # The tool fetches max_rows + 1 rows to detect truncation and returns max_rows.
    assert "LIMIT 6" in captured["sql"]
    assert captured["params"][0] == "KC"
    assert "%Mahomes%" in captured["params"]
    assert 2024 in captured["params"]


def test_team_stats_uses_parameterized_team_and_season(monkeypatch):
    calls = []

    def fake_execute(sql, params=None):
        calls.append((sql, params))
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    tools.nfl_team_stats(team="kc", season=2024, side="both")

    assert len(calls) == 3
    assert all("team = ?" in sql for sql, _ in calls)
    assert all(params[0] == "KC" for _, params in calls)
    assert all(2024 in params for _, params in calls)


def test_player_stats_uses_parameterized_player_name(monkeypatch):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    tools.nfl_player_stats(player_name="P.Mahomes", stat_type="passing", season_type="REG")

    assert "ILIKE ?" in captured["sql"]
    assert captured["params"][0] == "%P.Mahomes%"
    assert captured["params"][1] == "pass"
    assert captured["params"][-1] == "REG"


def test_player_stats_normalizes_full_player_name(monkeypatch):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    tools.nfl_player_stats(player_name="Justin Jefferson", stat_type="passing")

    assert captured["params"][0] == "%J.Jefferson%"


def test_compare_team_refuses_injected_text_as_an_unknown_team_without_interpolating(monkeypatch):
    # The text is not a team, so the only query is the bound "is this code stored?" probe.
    calls = []

    def fake_execute(sql, params=None):
        calls.append((sql, params))
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_compare(
        entity1="KC'; DROP TABLE plays; --", entity2="BAL", compare_type="team", season=2024,
    )

    assert result["code"] is ErrorCode.UNKNOWN_TEAM
    assert result["parameter"] == "entity1"
    assert len(calls) == 1
    assert all("DROP TABLE" not in sql for sql, _ in calls)
    assert calls[0][1] == ["KC'; DROP TABLE PLAYS; --"]


def test_schema_table_returns_columns(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_execute",
        lambda sql, params=None: [{"column_name": "full_name", "data_type": "VARCHAR"}],
    )
    result = tools.nfl_schema(table="rosters")
    assert result["table"] == "rosters"
    assert result["columns"][0]["column_name"] == "full_name"


def test_schema_table_not_found_returns_available_tables(monkeypatch):
    def fake_execute(sql, params=None):
        if "information_schema.columns" in sql:
            return []
        return [{"table_name": "plays"}, {"table_name": "rosters"}]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_schema(table="missing_table")
    assert "not found" in result["error"]
    assert result["available_tables"] == ["plays", "rosters"]


def test_schema_table_returns_error_when_execute_raises(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(ValueError("schema boom")))
    result = tools.nfl_schema(table="rosters")
    assert result["error"] == "schema boom"


def test_schema_summary_handles_table_listing_error(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    result = tools.nfl_schema()
    assert "schema" in result
    assert result["available_tables"] == []


def test_schema_category_all_and_unknown():
    all_result = tools.nfl_schema(category="all")
    assert "schema" in all_result and isinstance(all_result["schema"], str)

    unknown = tools.nfl_schema(category="not-real")
    assert "error" in unknown
    assert "available" in unknown


def test_nfl_status_happy_path(monkeypatch):
    # The fake answers by query instead of in call order: nfl_status now probes
    # information_schema for the `status` column first, and reads `issues` only when the
    # column is there. An empty probe result is the old six-column table, which has no
    # record of a failed attempt, so `issues` is empty and no issues query runs.
    def fake_execute(sql, params=None):
        if "information_schema.columns" in sql:
            return []
        if "total_plays" in sql:
            return [{"total_plays": 123}]
        if "season_type, COUNT(*)" in sql:
            return [{"season": 2024, "season_type": "REG", "plays": 123}]
        if "first_season" in sql:
            return [{"first_season": 2024, "last_season": 2024, "num_seasons": 1}]
        if "last_refreshed" in sql:
            return [{"last_refreshed": "2026-03-01"}]
        return [{"dataset_id": "pbp", "table_name": "plays", "total_rows": 123}]

    monkeypatch.setattr(tools, "_execute", fake_execute)

    result = tools.nfl_status()
    assert result["plays"]["total_plays"] == 123
    assert result["plays"]["season_range"]["first_season"] == 2024
    assert result["datasets"]["total_loaded"] == 1
    assert result["datasets"]["last_refreshed"] == "2026-03-01"
    assert result["datasets"]["loaded"][0]["dataset_id"] == "pbp"
    assert result["datasets"]["issues"] == []


def test_nfl_status_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timeout")))
    result = tools.nfl_status()
    assert "error" in result


def test_nfl_query_reports_truncation(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_execute",
        lambda sql, params=None: [{"x": 1}, {"x": 2}, {"x": 3}],
    )
    result = tools.nfl_query("SELECT 1 AS x", max_rows=2)
    assert result["row_count"] == 2
    assert result["truncated"] is True


def test_nfl_query_returns_error_when_execute_fails(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad sql")))
    result = tools.nfl_query("SELECT 1")
    assert result["error"] == "bad sql"
    assert result["code"] == "SQL_ERROR"
    assert result["code"] is ErrorCode.SQL_ERROR


COLUMN_ERROR = (
    'Binder Error: Referenced column "play_epa" not found in FROM clause!\n'
    'Candidate bindings: "play_type", "play", "play_id", "play_type_nfl", "play_deleted"\n'
    "\n"
    "LINE 3:  ROUND(AVG(play_epa),4) epa_allowed_per_play,\n"
    "                   ^"
)

TABLE_ERROR = (
    'Catalog Error: Table with name playz does not exist!\n'
    'Did you mean "plays"?'
)

PLAY_COLUMNS = [
    "play_id", "play_type", "play_type_nfl", "play_deleted", "play", "play_clock",
    "epa", "qb_epa", "air_epa", "yac_epa", "xyac_epa", "success", "yards_gained",
]


def _failing_query(exc, schema_rows):
    """Fake _execute: answers schema lookups, raises `exc` for the caller's SQL."""

    def fake_execute(sql, params=None):
        if "information_schema" in sql:
            return schema_rows
        raise exc

    return fake_execute


def test_nfl_query_suggests_real_columns_for_an_invented_column(monkeypatch):
    monkeypatch.setattr(
        tools, "_execute", _failing_query(duckdb.BinderException(COLUMN_ERROR),
                                          [{"column_name": c} for c in PLAY_COLUMNS]),
    )
    result = tools.nfl_query("SELECT defteam, ROUND(AVG(play_epa),4) FROM plays GROUP BY 1")

    assert result["code"] == "COLUMN_NOT_FOUND"
    assert result["unknown_column"] == "play_epa"
    assert result["did_you_mean"][0] == "epa"
    assert set(result["did_you_mean"]) == {"epa", "air_epa", "qb_epa", "xyac_epa", "yac_epa"}
    assert "nfl_schema" in result["hint"]


def test_nfl_query_suggests_real_tables_for_an_invented_table(monkeypatch):
    monkeypatch.setattr(
        tools, "_execute", _failing_query(duckdb.CatalogException(TABLE_ERROR),
                                          [{"table_name": t} for t in ("plays", "schedules")]),
    )
    result = tools.nfl_query("SELECT * FROM playz")

    assert result["code"] == "TABLE_NOT_FOUND"
    assert result["unknown_table"] == "playz"
    assert result["did_you_mean"][0] == "plays"
    assert "nfl_catalog" in result["hint"]


def test_nfl_query_falls_back_to_duckdb_candidates_when_schema_lookup_fails(monkeypatch):
    def fake_execute(sql, params=None):
        if "information_schema" in sql:
            raise duckdb.Error("schema unavailable")
        raise duckdb.BinderException(COLUMN_ERROR)

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_query("SELECT AVG(play_epa) FROM plays")

    assert result["code"] == "COLUMN_NOT_FOUND"
    assert "play_type" in result["did_you_mean"]
    assert "epa" not in result["did_you_mean"]


def test_similarity_prefers_the_last_word_of_a_snake_case_name():
    assert tools._similarity("epa", "epa") == 5.0
    assert tools._similarity("play_epa", "epa") > tools._similarity("play_epa", "play_type")
    assert tools._similarity("epa_play", "play_epa") > tools._similarity("epa_play", "yards_gained")
    assert tools._similarity("abc", "xyz") < 1.0
    assert tools._similarity("_", "epa") < 1.0


def test_nearest_drops_the_guess_and_ranks_by_similarity():
    candidates = ["epa", "EPA", "", "qb_epa", "air_epa", "yac_epa", "xyac_epa", "wpa"]

    assert tools._nearest("epa", candidates) == [
        "air_epa", "qb_epa", "xyac_epa", "yac_epa", "wpa",
    ]


@pytest.mark.parametrize(
    "situation,expected_sql",
    [
        ("red_zone", "yardline_100 <= 20"),
        ("third_down", "down = 3"),
        ("fourth_down", "down = 4"),
        ("two_minute", "qtr = 4 AND half_seconds_remaining <= 120"),
    ],
)
def test_search_plays_situations(monkeypatch, situation, expected_sql):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    tools.nfl_search_plays(situation=situation, max_rows=10)
    assert expected_sql in captured["sql"]


def test_search_plays_applies_all_numeric_and_boolean_filters(monkeypatch):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_search_plays(
        opponent="BAL",
        season_from=2020,
        season_to=2021,
        week=1,
        season_type="REG",
        play_type="pass",
        is_touchdown=True,
        is_turnover=True,
        min_yards=7,
        max_rows=999,
    )

    assert result["row_count"] == 0
    assert "defteam = ?" in captured["sql"]
    assert "season >= ?" in captured["sql"]
    assert "season <= ?" in captured["sql"]
    assert "week = ?" in captured["sql"]
    assert "season_type = ?" in captured["sql"]
    assert "play_type = ?" in captured["sql"]
    assert "touchdown = 1" in captured["sql"]
    assert "(interception = 1 OR fumble_lost = 1)" in captured["sql"]
    assert "yards_gained >= ?" in captured["sql"]
    # max_rows=999 clamps to 500; the probe row makes the SQL ask for 501.
    assert "LIMIT 501" in captured["sql"]
    assert captured["params"] == ["BAL", 2020, 2021, 1, "REG", "pass", 7]


def test_search_plays_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("boom")))
    result = tools.nfl_search_plays(team="KC")
    assert "error" in result


def test_team_stats_offense_only_without_season_clause(monkeypatch):
    calls = []

    def fake_execute(sql, params=None):
        calls.append((sql, params))
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_team_stats(team="kc", side="offense")
    assert "offense" in result
    assert "defense" not in result
    assert "season_year = ?" not in calls[0][0]
    assert calls[0][1] == ["KC"]


def test_team_stats_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timeout")))
    result = tools.nfl_team_stats(team="KC")
    assert "error" in result


@pytest.mark.parametrize(
    "stat_type,expected_play_type",
    [("rushing", "run"), ("receiving", "pass")],
)
def test_player_stats_other_stat_types(monkeypatch, stat_type, expected_play_type):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return [{"season": 2024, "season_type": "REG"}]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_player_stats(
        player_name="P.Mahomes",
        stat_type=stat_type,
        season_from=2020,
        season_to=2024,
    )
    assert result["stat_type"] == stat_type
    assert captured["params"][1] == expected_play_type
    assert "season >=" in captured["sql"]
    assert "season <=" in captured["sql"]


def test_player_stats_rejects_unknown_stat_type():
    result = tools.nfl_player_stats(player_name="P.Mahomes", stat_type="kicking")
    assert "Unknown stat_type" in result["error"]


def test_player_stats_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timeout")))
    result = tools.nfl_player_stats(player_name="P.Mahomes", stat_type="passing")
    assert "error" in result


def test_compare_player_path_only_includes_stats_with_attempts(monkeypatch):
    calls = []

    def fake_execute(sql, params=None):
        calls.append((sql, params))
        if "COUNT(*) AS n" in sql and "passer_player_name" in sql:
            return [{"n": 1}]
        if "COUNT(*) AS n" in sql and "receiver_player_name" in sql:
            return [{"n": 1}]
        if "COUNT(*) AS n" in sql and "rusher_player_name" in sql:
            return [{"n": 0}]
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    monkeypatch.setattr(
        tools,
        "nfl_player_stats",
        lambda player_name, **kwargs: {"seasons": [{"player": player_name, **kwargs}]},
    )
    result = tools.nfl_compare(
        entity1="P.Mahomes",
        entity2="J.Allen",
        compare_type="player",
        season_from=2020,
        season_to=2024,
        season_type="REG",
    )
    assert "passing" in result["P.Mahomes"]
    assert "receiving" in result["P.Mahomes"]
    assert "rushing" not in result["P.Mahomes"]
    assert "season_type = ?" in calls[0][0]


def test_compare_rejects_unknown_type():
    result = tools.nfl_compare(entity1="KC", entity2="BAL", compare_type="invalid")
    assert result["error"] == "Unknown compare_type: invalid. Use 'team' or 'player'."


def test_compare_team_applies_season_range_filters(monkeypatch):
    calls = []

    def fake_execute(sql, params=None):
        calls.append((sql, params))
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_compare(
        entity1="KC",
        entity2="BAL",
        compare_type="team",
        season_from=2020,
        season_to=2021,
    )
    assert "error" not in result
    assert len(calls) == 4
    assert "season_year >= ?" in calls[0][0]
    assert "season_year <= ?" in calls[0][0]
    assert calls[0][1] == ["KC", 2020, 2021]


def test_compare_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timeout")))
    result = tools.nfl_compare(entity1="KC", entity2="BAL", compare_type="team", season=2024)
    assert "error" in result


def test_catalog_returns_dataset_summary(monkeypatch):
    # Answers by query, because nfl_catalog now asks three questions of one dataset: does
    # the log have the outcome columns, what is loaded, and which seasons have data. The
    # empty probe stands for a database written before those columns existed.
    def fake_execute(sql, params=None):
        if "information_schema.columns" in sql:
            return []
        if "seasons_loaded" in sql:
            return [{
                "dataset_id": "pbp", "table_name": "plays", "total_rows": 123,
                "seasons_loaded": 2, "min_season": 2022, "max_season": 2024,
                "last_loaded": "2026-03-01",
            }]
        return [
            {"dataset_id": "pbp", "table_name": "plays", "season": 2022},
            {"dataset_id": "pbp", "table_name": "plays", "season": 2024},
        ]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_catalog()
    assert result["total_datasets"] == 1
    assert result["datasets"][0]["dataset_id"] == "pbp"
    assert result["datasets"][0]["total_rows"] == 123
    # 2023 sits between the two ends the row reports and has no data.
    assert result["datasets"][0]["missing_seasons"] == [2023]
    assert result["datasets"][0]["issues"] == []


def test_catalog_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timeout")))
    result = tools.nfl_catalog()
    assert "error" in result


def test_roster_builds_filters_and_uppercases_inputs(monkeypatch):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return [{"full_name": "A Player"}]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_roster(team="kc", season=2024, position="wr")
    assert result["count"] == 1
    assert "team = ?" in captured["sql"]
    assert "season = ?" in captured["sql"]
    assert "position ILIKE ?" in captured["sql"]
    assert captured["params"] == ["KC", 2024, "WR"]


def test_roster_uses_none_params_when_no_filters(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        tools,
        "_execute",
        lambda sql, params=None: captured.setdefault("params", params) or [],
    )
    tools.nfl_roster()
    assert captured["params"] is None


def test_roster_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("roster timeout")))
    result = tools.nfl_roster(team="KC")
    assert result["error"] == "roster timeout"


def test_injuries_builds_filters_and_wildcards(monkeypatch):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return [{"full_name": "P.Mahomes"}]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_injuries(
        team="kc",
        season=2024,
        week=1,
        player="Mahomes",
        report_status="Out",
    )
    assert result["count"] == 1
    assert "team = ?" in captured["sql"]
    assert "report_status ILIKE ?" in captured["sql"]
    assert captured["params"] == ["KC", 2024, 1, "%Mahomes%", "%Out%"]


def test_injuries_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("injuries timeout")))
    result = tools.nfl_injuries(team="KC")
    assert result["error"] == "injuries timeout"


def test_schedule_builds_team_pair_filter_and_uppercases_season_type(monkeypatch):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return [{"game_id": "2024_01_KC_BAL"}]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_schedule(team="kc", season=2024, week=1, season_type="reg")
    assert result["count"] == 1
    assert "(home_team = ? OR away_team = ?)" in captured["sql"]
    assert "game_type = ?" in captured["sql"]
    assert captured["params"] == ["KC", "KC", 2024, 1, "REG"]


def test_schedule_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("schedule timeout")))
    result = tools.nfl_schedule(team="KC")
    assert result["error"] == "schedule timeout"


def test_snap_counts_builds_filters(monkeypatch):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return [{"player": "T.Kelce"}]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_snap_counts(
        player="Kelce",
        team="kc",
        season=2024,
        week=1,
        position="te",
    )
    assert result["count"] == 1
    assert "player ILIKE ?" in captured["sql"]
    assert "team = ?" in captured["sql"]
    assert "position ILIKE ?" in captured["sql"]
    assert captured["params"] == ["%Kelce%", "KC", 2024, 1, "TE"]


def test_snap_counts_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("snap timeout")))
    result = tools.nfl_snap_counts(team="KC")
    assert result["error"] == "snap timeout"


def test_execute_raises_worker_error(monkeypatch):
    class _BadConn:
        def execute(self, *_args, **_kwargs):
            raise ValueError("execute failed")

    @contextmanager
    def fake_get_db_connection():
        yield _BadConn()

    monkeypatch.setattr(tools, "get_db_connection", fake_get_db_connection)
    with pytest.raises(ValueError, match="execute failed"):
        tools._execute("SELECT 1")


def test_execute_returns_rows_with_and_without_params(monkeypatch):
    class _Rel:
        description = [("col_a",), ("col_b",)]

        def fetchall(self):
            return [(1, "x")]

    class _GoodConn:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params=None):
            self.calls.append((sql, params))
            return _Rel()

    conn = _GoodConn()

    @contextmanager
    def fake_get_db_connection():
        yield conn

    monkeypatch.setattr(tools, "get_db_connection", fake_get_db_connection)

    rows_without_params = tools._execute("SELECT 1")
    rows_with_params = tools._execute("SELECT ? AS col_a, ? AS col_b", [1, "x"])

    assert rows_without_params == [{"col_a": 1, "col_b": "x"}]
    assert rows_with_params == [{"col_a": 1, "col_b": "x"}]
    assert conn.calls[0][1] is None
    assert conn.calls[1][1] == [1, "x"]


class _Rel:
    description = [("x",)]

    def fetchall(self):
        return [(1,)]


def _short_timeouts(monkeypatch, grace=0.05):
    monkeypatch.setattr(tools, "_QUERY_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(tools, "_INTERRUPT_GRACE_SECONDS", grace)
    monkeypatch.setattr(tools, "_stuck_workers", [])
    warnings = []
    monkeypatch.setattr(tools.logger, "warning", lambda *a, **k: warnings.append(a))
    return warnings


def test_query_timeout_error_is_a_timeout_error():
    exc = tools.QueryTimeoutError("Query exceeded 10 second timeout", interrupted=True, worker_alive=False)
    assert isinstance(exc, TimeoutError)
    assert str(exc) == "Query exceeded 10 second timeout"
    assert (exc.interrupted, exc.worker_alive) == (True, False)


def test_execute_timeout_interrupt_warning_path(monkeypatch):
    # interrupt() itself raises; the query finishes on its own inside the grace join.
    class _SlowConn:
        def execute(self, *_args, **_kwargs):
            time.sleep(0.25)
            return _Rel()

        def interrupt(self):
            raise RuntimeError("interrupt failed")

    @contextmanager
    def fake_get_db_connection():
        yield _SlowConn()

    monkeypatch.setattr(tools, "get_db_connection", fake_get_db_connection)
    warnings = _short_timeouts(monkeypatch, grace=1.0)

    with pytest.raises(TimeoutError, match="Query exceeded") as info:
        tools._execute("SELECT 1")
    assert isinstance(info.value, tools.QueryTimeoutError)
    assert info.value.interrupted is False
    assert info.value.worker_alive is False
    assert any("Failed to interrupt" in a[0] for a in warnings)
    assert tools._stuck_workers == []


def test_execute_timeout_interrupt_delivered_stops_the_worker(monkeypatch):
    class _InterruptibleConn:
        def __init__(self):
            self.cancelled = threading.Event()

        def execute(self, *_args, **_kwargs):
            self.cancelled.wait(2)
            raise duckdb.InterruptException("INTERRUPT Error: Interrupted!")

        def interrupt(self):
            self.cancelled.set()

    @contextmanager
    def fake_get_db_connection():
        yield _InterruptibleConn()

    monkeypatch.setattr(tools, "get_db_connection", fake_get_db_connection)
    warnings = _short_timeouts(monkeypatch, grace=1.0)

    with pytest.raises(tools.QueryTimeoutError, match="Query exceeded") as info:
        tools._execute("SELECT 1")
    assert info.value.interrupted is True
    assert info.value.worker_alive is False
    assert warnings == []


def test_execute_timeout_worker_still_alive_after_grace_is_logged(monkeypatch):
    release = threading.Event()

    class _DeafConn:
        interrupts = 0

        def execute(self, *_args, **_kwargs):
            release.wait(2)
            return _Rel()

        def interrupt(self):
            _DeafConn.interrupts += 1  # returns, but the query keeps running

    @contextmanager
    def fake_get_db_connection():
        yield _DeafConn()

    monkeypatch.setattr(tools, "get_db_connection", fake_get_db_connection)
    warnings = _short_timeouts(monkeypatch)
    try:
        with pytest.raises(tools.QueryTimeoutError) as info:
            tools._execute("SELECT 1")
        assert info.value.interrupted is True
        assert info.value.worker_alive is True
        assert _DeafConn.interrupts == 1
        assert len(tools._stuck_workers) == 1
        assert any("still running" in a[0] for a in warnings)
    finally:
        release.set()
    tools._stuck_workers[0].join(2)
    assert not tools._stuck_workers[0].is_alive()


def test_execute_timeout_before_connect_never_starts_the_query(monkeypatch):
    connect = threading.Event()
    connected = threading.Event()
    executed = []

    class _Conn:
        def execute(self, *_args, **_kwargs):
            executed.append(True)
            return _Rel()

        def interrupt(self):
            raise AssertionError("no connection existed to interrupt")

    @contextmanager
    def slow_get_db_connection():
        connect.wait(2)  # stuck opening the database
        yield _Conn()
        connected.set()

    monkeypatch.setattr(tools, "get_db_connection", slow_get_db_connection)
    warnings = _short_timeouts(monkeypatch)
    try:
        with pytest.raises(tools.QueryTimeoutError) as info:
            tools._execute("SELECT 1")
        assert info.value.interrupted is False
        assert info.value.worker_alive is True
        assert any("still running" in a[0] for a in warnings)
    finally:
        connect.set()
    assert connected.wait(2)
    assert executed == []


def test_record_stuck_worker_prunes_finished_threads(monkeypatch):
    done = threading.Thread(target=lambda: None)
    done.start()
    done.join()
    monkeypatch.setattr(tools, "_stuck_workers", [done])
    alive = threading.Thread(target=lambda: None)
    alive.is_alive = lambda: True
    assert tools._record_stuck_worker(alive) == 1
    assert tools._stuck_workers == [alive]


def test_ftn_charting_builds_all_filters(monkeypatch):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return [{
            "total_plays": 10, "play_action": 4, "rpo": 1, "screen_pass": 2,
            "no_huddle": 0, "motion": 5, "trick_play": 0,
            "avg_defenders_in_box": 6.5, "avg_pass_rushers": 4.2, "avg_blitzers": 1.1,
        }]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_ftn_charting(
        team="kc", opponent="phi", player="Mahomes",
        season_from=2022, season_to=2024, week=3, season_type="reg",
    )
    sql, params = captured["sql"], captured["params"]
    assert "p.posteam = ?" in sql and "p.defteam = ?" in sql
    assert "p.passer_player_name ILIKE ?" in sql
    assert "p.season >= ?" in sql and "p.season <= ?" in sql
    assert "p.week = ?" in sql and "p.season_type = ?" in sql
    assert params[:2] == ["KC", "PHI"]
    assert params[2:5] == ["%Mahomes%"] * 3
    assert params[5:7] == [2022, 2024]
    assert params[-2:] == [3, "REG"]
    assert result["charting"]["play_action"]["pct"] == 40.0
    assert result["charting"]["motion"]["plays"] == 5


def test_ftn_charting_exact_season_filter(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        tools, "_execute",
        lambda sql, params=None: captured.update(sql=sql, params=params) or [{"total_plays": 0}],
    )
    result = tools.nfl_ftn_charting(season=2024)
    assert "p.season = ?" in captured["sql"]
    assert captured["params"] == [2024]
    assert result == {"charting": {}, "total_plays": 0}


def test_ftn_charting_handles_null_averages(monkeypatch):
    monkeypatch.setattr(
        tools, "_execute",
        lambda sql, params=None: [{
            "total_plays": 3, "play_action": None, "rpo": None, "screen_pass": None,
            "no_huddle": None, "motion": None, "trick_play": None,
            "avg_defenders_in_box": None, "avg_pass_rushers": None, "avg_blitzers": None,
        }],
    )
    result = tools.nfl_ftn_charting(team="KC")
    c = result["charting"]
    assert c["play_action"]["pct"] == 0.0
    assert c["play_action"]["plays"] == 0
    assert c["avg_defenders_in_box"] is None


def test_ftn_charting_empty_rows(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda sql, params=None: [])
    result = tools.nfl_ftn_charting(team="KC")
    assert result == {"charting": {}, "total_plays": 0}


def test_ftn_charting_returns_error_on_db_error(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(duckdb.Error("ftn boom")))
    result = tools.nfl_ftn_charting(team="KC")
    assert result["error"] == "ftn boom"


# ── Fantasy derived-table tools ────────────────────────────────────────────────

def _capture_execute(monkeypatch, rows):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return rows

    monkeypatch.setattr(tools, "_execute", fake_execute)
    return captured


def test_td_luck_builds_filters(monkeypatch):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    result = tools.nfl_td_luck(player="Jefferson", position="wr", team="min", season=2024, limit=10)
    assert result["count"] == 1
    assert "full_name ILIKE ?" in captured["sql"]
    assert "position ILIKE ?" in captured["sql"]
    assert "team = ?" in captured["sql"]
    assert "total_td_luck_score ASC" in captured["sql"]
    assert captured["params"] == ["%Jefferson%", "WR", "MIN", 2024]


def test_td_luck_invalid_limit_falls_back(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    tools.nfl_td_luck(limit="oops")
    assert "LIMIT 51" in captured["sql"]
    assert captured["params"] is None


def test_td_luck_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("td timeout")))
    assert tools.nfl_td_luck()["error"] == "td timeout"


def test_role_trend_builds_filters(monkeypatch):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    tools.nfl_role_trend(player="Kelce", position="te", team="kc", season=2024, week=8, min_snap_pct=50)
    assert "full_name ILIKE ?" in captured["sql"]
    assert "snap_pct >= ?" in captured["sql"]
    assert "snap_pct_delta DESC" in captured["sql"]
    assert captured["params"] == ["%Kelce%", "TE", "KC", 2024, 8, 50.0]


def test_role_trend_invalid_limit_falls_back(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    tools.nfl_role_trend(limit="oops")
    assert "LIMIT 51" in captured["sql"]


def test_role_trend_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("role timeout")))
    assert tools.nfl_role_trend()["error"] == "role timeout"


def test_separation_opportunity_builds_filters(monkeypatch):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    tools.nfl_separation_opportunity(player="Aiyuk", position="wr", team="sf", season=2024, regression_candidate=True)
    assert "regression_candidate = ?" in captured["sql"]
    assert "fp_diff_per_game ASC" in captured["sql"]
    assert captured["params"] == ["%Aiyuk%", "WR", "SF", 2024, True]


def test_separation_opportunity_guards_pre_2016():
    result = tools.nfl_separation_opportunity(season=2015)
    assert "2016" in result["error"]


def test_separation_opportunity_invalid_limit_falls_back(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    tools.nfl_separation_opportunity(limit="oops")
    assert "LIMIT 51" in captured["sql"]


def test_separation_opportunity_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("sep timeout")))
    assert tools.nfl_separation_opportunity()["error"] == "sep timeout"


def test_drop_rate_builds_filters(monkeypatch):
    captured = _capture_execute(monkeypatch, [{"player": "T.Pollard"}])
    tools.nfl_drop_rate(player="Pollard", team="ten", season=2024, min_targets=30)
    assert "player ILIKE ?" in captured["sql"]
    assert "catchable_targets >= ?" in captured["sql"]
    assert "drop_rate_pct DESC" in captured["sql"]
    assert captured["params"] == ["%Pollard%", "TEN", 2024, 30]


def test_drop_rate_guards_pre_2022():
    result = tools.nfl_drop_rate(season=2021)
    assert "2022" in result["error"]


def test_drop_rate_invalid_limit_falls_back(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    tools.nfl_drop_rate(limit="oops")
    assert "LIMIT 51" in captured["sql"]


def test_drop_rate_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("drop timeout")))
    assert tools.nfl_drop_rate()["error"] == "drop timeout"


def test_contract_value_builds_filters(monkeypatch):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    tools.nfl_contract_value(player="Gibbs", position="rb", team="det", season=2024, min_apy=2, max_apy=10)
    assert "apy >= ?" in captured["sql"]
    assert "apy <= ?" in captured["sql"]
    assert "fp_per_million DESC" in captured["sql"]
    assert captured["params"] == ["%Gibbs%", "RB", "DET", 2024, 2.0, 10.0]


def test_contract_value_invalid_limit_falls_back(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    tools.nfl_contract_value(limit="oops")
    assert "LIMIT 51" in captured["sql"]


def test_contract_value_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("contract timeout")))
    assert tools.nfl_contract_value()["error"] == "contract timeout"


def test_injury_return_builds_filters(monkeypatch):
    captured = _capture_execute(monkeypatch, [{"injury_type": "hamstring"}])
    tools.nfl_injury_return(injury_type="hamstring", position="wr", week_post_return=2)
    assert "injury_type ILIKE ?" in captured["sql"]
    assert "position = ?" in captured["sql"]
    assert "week_post_return = ?" in captured["sql"]
    assert "week_post_return ASC" in captured["sql"]
    assert captured["params"] == ["%hamstring%", "WR", 2]


def test_injury_return_invalid_limit_falls_back(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    tools.nfl_injury_return(limit="oops")
    assert "LIMIT 51" in captured["sql"]
    assert captured["params"] is None


def test_injury_return_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("injury timeout")))
    assert tools.nfl_injury_return()["error"] == "injury timeout"


# ── Error payloads: every failure carries a machine-readable code ──────────────

DB_MISSING = duckdb.IOException(
    'IO Error: Cannot open database "/data/nflread.duckdb" in read-only mode: '
    "database does not exist"
)

DB_LOCKED = duckdb.IOException(
    'IO Error: Could not set lock on file "/data/nflread.duckdb": '
    "Conflicting lock is held in python3"
)

DERIVED_TABLE_MISSING = duckdb.CatalogException(
    'Catalog Error: Table with name player_td_luck does not exist!\n'
    'Did you mean "plays"?'
)


def _raise(exc):
    """Fake ``_execute`` that always raises `exc`."""
    return lambda *a, **k: (_ for _ in ()).throw(exc)


def test_err_helper_keeps_the_text_and_adds_code_and_extra_fields():
    result = tools._err(ErrorCode.INVALID_INPUT, "bad input", available=["plays"])
    assert result == {
        "error": "bad input",
        "code": ErrorCode.INVALID_INPUT,
        "available": ["plays"],
    }


def test_classifier_reports_a_timeout():
    result = tools._error_from_exception(TimeoutError("Query exceeded 10 second timeout"))
    assert result["error"] == "Query exceeded 10 second timeout"
    assert result["code"] is ErrorCode.TIMEOUT
    assert "hint" not in result


def test_classifier_reports_a_missing_database_with_an_ingest_hint():
    result = tools._error_from_exception(DB_MISSING)
    assert result["error"] == str(DB_MISSING)
    assert result["code"] is ErrorCode.DB_NOT_FOUND
    assert "nfl-mcp ingest" in result["hint"]


def test_classifier_reports_a_locked_database_as_retryable():
    result = tools._error_from_exception(DB_LOCKED)
    assert result["error"] == str(DB_LOCKED)
    assert result["code"] is ErrorCode.DB_LOCKED
    assert "retried" in result["hint"].lower()


def test_classifier_reports_a_column_guess_with_suggestions(monkeypatch):
    monkeypatch.setattr(tools, "_live_names", lambda sql: ["epa", "play_type"])
    result = tools._error_from_exception(duckdb.BinderException(COLUMN_ERROR))
    assert result["code"] is ErrorCode.COLUMN_NOT_FOUND
    assert result["error"] == COLUMN_ERROR
    assert result["unknown_column"] == "play_epa"
    assert result["did_you_mean"][0] == "epa"
    assert "nfl_schema" in result["hint"]


def test_classifier_reports_a_missing_table_with_suggestions(monkeypatch):
    monkeypatch.setattr(tools, "_live_names", lambda sql: ["plays", "schedules"])
    result = tools._error_from_exception(DERIVED_TABLE_MISSING)
    assert result["code"] is ErrorCode.TABLE_NOT_FOUND
    assert result["error"] == str(DERIVED_TABLE_MISSING)
    assert result["unknown_table"] == "player_td_luck"
    assert "nfl_catalog" in result["hint"]


@pytest.mark.parametrize("exc", [
    duckdb.IOException("IO Error: unsupported file format"),
    ValueError("no idea what happened"),
    RuntimeError("kaboom"),
])
def test_classifier_falls_back_to_sql_error(exc):
    result = tools._error_from_exception(exc)
    assert result["error"] == str(exc)
    assert result["code"] is ErrorCode.SQL_ERROR
    assert "hint" not in result


def test_classifier_code_serializes_as_the_plain_string():
    payload = json.dumps(tools._error_from_exception(DB_LOCKED), indent=2)
    assert '"code": "DB_LOCKED"' in payload
    assert "ErrorCode." not in payload


# ── Early validation errors ────────────────────────────────────────────────────

VALIDATION_CASES = [
    (
        lambda: tools.nfl_query("INSERT INTO plays VALUES (1)"),
        ErrorCode.NOT_SELECT,
        "Only SELECT queries are allowed.",
    ),
    (
        lambda: tools.nfl_query("SELECT pg_sleep(100)"),
        ErrorCode.FORBIDDEN_KEYWORD,
        "Forbidden keyword: 'pg_sleep'",
    ),
    (
        lambda: tools.nfl_query("SELECT 1; SELECT 2"),
        ErrorCode.MULTIPLE_STATEMENTS,
        "Multiple statements are not allowed.",
    ),
    (
        lambda: tools.nfl_schema(category="not-real"),
        ErrorCode.INVALID_INPUT,
        "Unknown category 'not-real'",
    ),
    (
        lambda: tools.nfl_player_stats(player_name="P.Mahomes", stat_type="kicking"),
        ErrorCode.INVALID_INPUT,
        "Unknown stat_type: kicking. Use 'passing', 'rushing', or 'receiving'.",
    ),
    (
        lambda: tools.nfl_compare(entity1="KC", entity2="BAL", compare_type="invalid"),
        ErrorCode.INVALID_INPUT,
        "Unknown compare_type: invalid. Use 'team' or 'player'.",
    ),
    (
        lambda: tools.nfl_fantasy_rankings(scope="month"),
        ErrorCode.INVALID_INPUT,
        "Unknown scope: month. Use 'draft' or 'week'.",
    ),
    (
        lambda: tools.nfl_team_records(),
        ErrorCode.INVALID_INPUT,
        "Provide team, season, or a from_season/to_season range.",
    ),
    (
        lambda: tools.nfl_team_records(team="ZZZ"),
        ErrorCode.UNKNOWN_TEAM,
        "Unknown team 'ZZZ'.",
    ),
    (
        lambda: tools.nfl_separation_opportunity(season=2015),
        ErrorCode.OUT_OF_COVERAGE,
        "Next Gen Stats receiving data is only available from 2016 onward",
    ),
    (
        lambda: tools.nfl_drop_rate(season=2021),
        ErrorCode.OUT_OF_COVERAGE,
        "FTN charting (drop) data is only available from 2022 onward",
    ),
]


@pytest.mark.parametrize("call,code,message", VALIDATION_CASES)
def test_early_validation_errors_keep_their_text_and_carry_a_code(call, code, message):
    result = call()
    assert result["error"] == message
    assert result["code"] is code


def test_schema_table_not_found_carries_a_code(monkeypatch):
    def fake_execute(sql, params=None):
        if "information_schema.columns" in sql:
            return []
        return [{"table_name": "plays"}, {"table_name": "rosters"}]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_schema(table="missing_table")
    assert result["code"] is ErrorCode.TABLE_NOT_FOUND
    assert result["available_tables"] == ["plays", "rosters"]


def test_schema_summary_warns_instead_of_claiming_there_are_no_tables(monkeypatch):
    monkeypatch.setattr(tools, "_execute", _raise(DB_MISSING))
    result = tools.nfl_schema()
    assert result["available_tables"] == []
    assert result["code"] is ErrorCode.DB_NOT_FOUND
    assert "does not exist" in result["warning"]
    assert len(result["schema"]) > 200
    assert "category=" in result["hint"]


def test_schema_summary_warning_uses_the_classifier(monkeypatch):
    monkeypatch.setattr(tools, "_execute", _raise(TimeoutError("query timeout")))
    result = tools.nfl_schema()
    assert result["code"] is ErrorCode.TIMEOUT
    assert result["warning"] == "Could not list tables: query timeout"


def test_structured_tool_gets_the_column_help_nfl_query_already_had(monkeypatch):
    monkeypatch.setattr(tools, "_live_names", lambda sql: ["epa", "play_type"])
    monkeypatch.setattr(tools, "_execute", _raise(duckdb.BinderException(COLUMN_ERROR)))
    result = tools.nfl_search_plays(team="KC")
    assert result["code"] is ErrorCode.COLUMN_NOT_FOUND
    assert result["unknown_column"] == "play_epa"
    assert result["did_you_mean"][0] == "epa"


def test_structured_tool_reports_a_derived_table_that_was_never_built(monkeypatch):
    monkeypatch.setattr(tools, "_live_names", lambda sql: ["plays"])
    monkeypatch.setattr(tools, "_execute", _raise(DERIVED_TABLE_MISSING))
    result = tools.nfl_td_luck()
    assert result["code"] is ErrorCode.TABLE_NOT_FOUND
    assert result["unknown_table"] == "player_td_luck"
    assert "nfl_catalog" in result["hint"]


# ── Every public tool, forced to fail ─────────────────────────────────────────

# One call per public tool, with arguments that get past validation to `_execute`.
# A tool added to __all__ with no entry here fails the guard test below.
TOOL_ERROR_CASES = {
    "nfl_schema": {"table": "plays"},
    "nfl_status": {},
    "nfl_query": {"sql": "SELECT 1"},
    "nfl_search_plays": {"team": "KC"},
    "nfl_team_stats": {"team": "KC"},
    "nfl_player_stats": {"player_name": "P.Mahomes"},
    "nfl_compare": {"entity1": "KC", "entity2": "BAL"},
    "nfl_catalog": {},
    "nfl_roster": {"team": "KC"},
    "nfl_injuries": {"team": "KC"},
    "nfl_schedule": {"team": "KC"},
    "nfl_team_records": {"team": "KC"},
    "nfl_snap_counts": {"team": "KC"},
    "nfl_fantasy_opportunity": {"team": "KC"},
    "nfl_fantasy_rankings": {},
    "nfl_ftn_charting": {"team": "KC"},
    "nfl_td_luck": {},
    "nfl_role_trend": {},
    "nfl_separation_opportunity": {},
    "nfl_drop_rate": {},
    "nfl_contract_value": {},
    "nfl_injury_return": {},
}


def test_every_public_tool_is_walked_by_the_error_test():
    assert set(TOOL_ERROR_CASES) == set(tools.__all__)


@pytest.mark.parametrize("name", sorted(TOOL_ERROR_CASES))
def test_every_tool_failure_returns_a_string_error_and_an_error_code(monkeypatch, name):
    monkeypatch.setattr(tools, "_execute", _raise(TimeoutError("query timeout")))

    result = getattr(tools, name)(**TOOL_ERROR_CASES[name])

    assert type(result["error"]) is str
    assert result["error"] == "query timeout"
    assert isinstance(result["code"], ErrorCode)
    assert result["code"] is ErrorCode.TIMEOUT


# ── Row caps: every capped tool reports `truncated` ───────────────────────────

@pytest.mark.parametrize(
    "n_rows,expected_truncated,expected_kept",
    [(2, False, 2), (3, False, 3), (4, True, 3)],
)
def test_cap_helper_drops_the_probe_row_and_flags_truncation(n_rows, expected_truncated, expected_kept):
    rows = [{"i": i} for i in range(n_rows)]
    kept, truncated = tools._cap(rows, 3)
    assert truncated is expected_truncated
    assert kept == rows[:expected_kept]


# (tool, kwargs, row-list key, count key, cap). The cap is the most rows the tool
# returns; its SQL must ask for cap + 1 and its result must carry `truncated`.
# A new capped tool belongs here; a tool with no cap belongs in NO_ROW_CAP below.
ROW_CAP_CASES = [
    ("nfl_query", {"sql": "SELECT 1 AS x", "max_rows": 3}, "rows", "row_count", 3),
    ("nfl_search_plays", {"max_rows": 3}, "rows", "row_count", 3),
    ("nfl_team_records", {"season": 2024}, "records", "count", 500),
    ("nfl_roster", {}, "players", "count", 500),
    ("nfl_injuries", {}, "injuries", "count", 500),
    ("nfl_schedule", {}, "games", "count", 500),
    ("nfl_snap_counts", {}, "snap_counts", "count", 500),
    ("nfl_fantasy_opportunity", {}, "fantasy_opportunity", "count", 500),
    ("nfl_fantasy_rankings", {"limit": 3}, "fantasy_rankings", "count", 3),
    ("nfl_td_luck", {"limit": 3}, "td_luck", "count", 3),
    ("nfl_role_trend", {"limit": 3}, "role_trend", "count", 3),
    ("nfl_separation_opportunity", {"limit": 3}, "separation_opportunity", "count", 3),
    ("nfl_drop_rate", {"limit": 3}, "drop_rate", "count", 3),
    ("nfl_contract_value", {"limit": 3}, "contract_value", "count", 3),
    ("nfl_injury_return", {"limit": 3}, "injury_return", "count", 3),
]


def _capped_rows(n):
    """Fake `_execute` results; the extra keys satisfy nfl_team_records' era-name join."""
    return [{"i": i, "season": 2024, "team": "LV"} for i in range(n)]


@pytest.mark.parametrize("name,kwargs,list_key,count_key,cap", ROW_CAP_CASES)
def test_capped_tool_fetches_cap_plus_one_and_flags_truncation(monkeypatch, name, kwargs, list_key, count_key, cap):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        return _capped_rows(cap + 1)

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = getattr(tools, name)(**kwargs)

    assert "error" not in result
    assert re.search(rf"LIMIT {cap + 1}(?![0-9])", captured["sql"])
    assert result["truncated"] is True
    assert result[count_key] == cap
    assert len(result[list_key]) == cap
    assert [row["i"] for row in result[list_key]] == list(range(cap))


@pytest.mark.parametrize("name,kwargs,list_key,count_key,cap", ROW_CAP_CASES)
def test_capped_tool_reports_no_truncation_under_the_cap(monkeypatch, name, kwargs, list_key, count_key, cap):
    monkeypatch.setattr(tools, "_execute", lambda sql, params=None: _capped_rows(cap - 1))
    result = getattr(tools, name)(**kwargs)

    assert "error" not in result
    assert result["truncated"] is False
    assert result[count_key] == cap - 1
    assert len(result[list_key]) == cap - 1


# Tools that return no capped row list. Each reason says why `truncated` is
# meaningless there, so a new tool cannot skip the flag by accident.
NO_ROW_CAP = {
    "nfl_schema": "schema reference: column/table lists are bounded by the schema, no LIMIT clause",
    "nfl_status": "aggregate counts over every loaded season, no LIMIT clause",
    "nfl_team_stats": "one row per season (and per situation), no LIMIT clause",
    "nfl_player_stats": "GROUP BY season: one row per season, no LIMIT clause",
    "nfl_compare": "reuses the nfl_team_stats/nfl_player_stats lists, no LIMIT clause",
    "nfl_catalog": "one row per dataset in the ingest log, no LIMIT clause",
    "nfl_ftn_charting": "one aggregate row, no row list to cut",
}


def test_every_public_tool_declares_its_row_cap_or_its_absence():
    capped = {case[0] for case in ROW_CAP_CASES}
    assert capped & set(NO_ROW_CAP) == set()
    assert capped | set(NO_ROW_CAP) == set(tools.__all__)


def test_team_records_availability_early_return_carries_truncated_false():
    # The franchise-existence answers never hit the database, and they must still
    # carry the flag so every success path has the same shape.
    result = tools.nfl_team_records(team="HOU", season=1995)
    assert result["count"] == 0
    assert result["truncated"] is False


# ── Ingest outcomes read from a real database ─────────────────────────────────
#
# `_ingest_metadata` records how each dataset+season attempt ended in its `status` and
# `detail` columns. The server opens the database read-only, so it cannot add those columns
# to a database an older ingest wrote, and both table shapes have to answer the same
# questions. These tests build real DuckDB files rather than canned rows, because the SQL
# is what is under test, and point NFL_MCP_DB_PATH at them (see config.get_duckdb_path).

_METADATA_OLD_DDL = """
    CREATE TABLE _ingest_metadata (
        dataset_id VARCHAR, table_name VARCHAR, season INTEGER,
        row_count BIGINT, loaded_at TIMESTAMP, loader_fn VARCHAR
    )
"""

_METADATA_NEW_DDL = """
    CREATE TABLE _ingest_metadata (
        dataset_id VARCHAR, table_name VARCHAR, season INTEGER,
        row_count BIGINT, loaded_at TIMESTAMP, loader_fn VARCHAR,
        status VARCHAR, detail VARCHAR
    )
"""

_PLAYS_DDL = "CREATE TABLE plays (season INTEGER, season_type VARCHAR)"

_TEAM_RECORDS_DDL = """
    CREATE TABLE team_records (
        season INTEGER, team VARCHAR, source_team VARCHAR, games INTEGER,
        wins INTEGER, losses INTEGER, ties INTEGER, source VARCHAR
    )
"""

# play-by-play: an interior gap at 2024 and a short 2026. espn_records: every attempt
# landed nothing. rosters: a static dataset, so it has no season at all. Every attempt that
# landed nothing carries a LATER loaded_at than the data, so `last_refreshed` proves which
# rows count as coverage.
INGEST_LOG = [
    ("pbp", "plays", 2022, 100, "2026-01-01", "loaded", None),
    ("pbp", "plays", 2023, 200, "2026-01-01", "loaded", None),
    ("pbp", "plays", 2024, 0, "2026-04-01", "failed", "load_pbp raised"),
    ("pbp", "plays", 2025, 300, "2026-01-01", "loaded", None),
    ("pbp", "plays", 2026, 50, "2026-01-01", "partial", "30 of 32 weeks"),
    ("espn_records", "espn_team_records", 1972, 0, "2026-03-01", "empty", "ESPN listed no games"),
    ("espn_records", "espn_team_records", 1973, 0, "2026-03-02", "failed", "HTTP 500"),
    ("rosters", "rosters", None, 500, "2026-01-01", "loaded", None),
]

_PLAYS_ROWS = "INSERT INTO plays VALUES (2022, 'REG'), (2022, 'REG'), (2023, 'REG')"


def _lit(value) -> str:
    """A DuckDB literal for one test value; None becomes NULL."""
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def _log_statements(ddl: str, rows: list[tuple], *, with_status: bool) -> list[str]:
    """DDL plus one INSERT for every ingest-log row, with or without the outcome columns."""
    columns = ["dataset_id", "table_name", "season", "row_count", "loaded_at"]
    if with_status:
        columns += ["status", "detail"]
    values = []
    for row in rows:
        cells = list(row[:5]) + (list(row[5:]) if with_status else [])
        values.append("(" + ", ".join(_lit(c) for c in cells) + ")")
    return [ddl, f"INSERT INTO _ingest_metadata ({', '.join(columns)}) VALUES {', '.join(values)}"]


def _write_db(tmp_path, monkeypatch, statements: list[str]) -> None:
    """Create a real DuckDB file from `statements`, then point the tools at it."""
    path = tmp_path / "nflread.duckdb"
    conn = duckdb.connect(str(path))
    try:
        for statement in statements:
            conn.execute(statement)
    finally:
        conn.close()  # the tools open their own read-only connection
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))


def _log_db(tmp_path, monkeypatch, shape: str, *, plays: bool = True) -> None:
    """A database whose ingest log has `shape`: "old" (six columns) or "new" (eight).

    The old shape has no record of an attempt that loaded nothing, because the code that
    wrote it only recorded a season when rows landed, so those rows are left out.
    """
    if shape == "new":
        statements = _log_statements(_METADATA_NEW_DDL, INGEST_LOG, with_status=True)
    else:
        rows = [r for r in INGEST_LOG if r[5] in ("loaded", "partial")]
        statements = _log_statements(_METADATA_OLD_DDL, rows, with_status=False)
    if plays:
        statements += [_PLAYS_DDL, _PLAYS_ROWS]
    _write_db(tmp_path, monkeypatch, statements)


@pytest.fixture
def old_log_db(tmp_path, monkeypatch):
    _log_db(tmp_path, monkeypatch, "old")


@pytest.fixture
def new_log_db(tmp_path, monkeypatch):
    _log_db(tmp_path, monkeypatch, "new")


def test_catalog_on_an_old_database_keeps_the_old_counts_and_finds_the_gap(old_log_db):
    result = tools.nfl_catalog()
    assert result["total_datasets"] == 2
    assert [d["dataset_id"] for d in result["datasets"]] == ["pbp", "rosters"]
    pbp = result["datasets"][0]
    assert pbp["total_rows"] == 650
    assert pbp["seasons_loaded"] == 4
    assert (pbp["min_season"], pbp["max_season"]) == (2022, 2026)
    # Nothing was ever attempted for 2024, and the log still shows the hole between 2022
    # and 2026 because the gap comes from the seasons that have data.
    assert pbp["missing_seasons"] == [2024]
    assert pbp["issues"] == []


def test_catalog_on_a_new_database_excludes_attempts_that_landed_nothing(new_log_db):
    result = tools.nfl_catalog()
    assert result["total_datasets"] == 3
    espn, pbp, rosters = result["datasets"]
    assert (pbp["total_rows"], pbp["seasons_loaded"]) == (650, 4)
    assert (pbp["min_season"], pbp["max_season"]) == (2022, 2026)
    assert pbp["missing_seasons"] == [2024]
    assert pbp["issues"] == [
        {"season": 2024, "status": "failed", "detail": "load_pbp raised"},
        {"season": 2026, "status": "partial", "detail": "30 of 32 weeks"},
    ]
    # A partial season counts as coverage AND as an issue; a static table has no seasons.
    assert rosters["total_rows"] == 500
    assert rosters["seasons_loaded"] == 1
    assert (rosters["min_season"], rosters["max_season"]) == (None, None)
    assert (rosters["missing_seasons"], rosters["issues"]) == ([], [])
    assert espn["dataset_id"] == "espn_records"


def test_catalog_reports_a_dataset_whose_only_attempts_landed_nothing(new_log_db):
    espn = next(d for d in tools.nfl_catalog()["datasets"] if d["dataset_id"] == "espn_records")
    assert espn["total_rows"] == 0
    assert espn["seasons_loaded"] == 0
    assert (espn["min_season"], espn["max_season"]) == (None, None)
    assert espn["missing_seasons"] == []
    assert [i["season"] for i in espn["issues"]] == [1972, 1973]
    assert [i["status"] for i in espn["issues"]] == ["empty", "failed"]


def test_status_on_an_old_database_reports_the_same_health_and_no_issues(old_log_db):
    result = tools.nfl_status()
    assert "error" not in result
    assert result["plays"]["total_plays"] == 3
    assert result["datasets"]["total_loaded"] == 2
    assert [d["dataset_id"] for d in result["datasets"]["loaded"]] == ["pbp", "rosters"]
    assert result["datasets"]["loaded"][0]["total_rows"] == 650
    assert result["datasets"]["last_refreshed"] == datetime.datetime(2026, 1, 1)
    assert result["datasets"]["issues"] == []


def test_status_on_a_new_database_lists_attempts_that_landed_nothing(new_log_db):
    result = tools.nfl_status()
    loaded = result["datasets"]["loaded"]
    # espn_records never stored a row, so it is not loaded; the failed 2024 season is not
    # counted either, and play-by-play totals 650 rows over four seasons.
    assert [d["dataset_id"] for d in loaded] == ["pbp", "rosters"]
    assert loaded[0]["total_rows"] == 650
    assert result["datasets"]["total_loaded"] == 2
    # The failed and empty attempts were written later than any loaded row.
    assert result["datasets"]["last_refreshed"] == datetime.datetime(2026, 1, 1)
    assert result["datasets"]["issues"] == [
        {"dataset_id": "espn_records", "season": 1972, "status": "empty",
         "detail": "ESPN listed no games"},
        {"dataset_id": "espn_records", "season": 1973, "status": "failed", "detail": "HTTP 500"},
        {"dataset_id": "pbp", "season": 2024, "status": "failed", "detail": "load_pbp raised"},
        {"dataset_id": "pbp", "season": 2026, "status": "partial", "detail": "30 of 32 weeks"},
    ]


def test_status_reports_datasets_and_a_plays_error_when_plays_is_absent(tmp_path, monkeypatch):
    _log_db(tmp_path, monkeypatch, "new", plays=False)
    result = tools.nfl_status()
    assert "error" not in result
    assert result["datasets"]["total_loaded"] == 2
    assert result["plays"]["code"] is ErrorCode.TABLE_NOT_FOUND
    assert "plays" in result["plays"]["error"]


def test_status_reports_plays_and_a_datasets_error_when_the_log_is_absent(tmp_path, monkeypatch):
    _write_db(tmp_path, monkeypatch, [_PLAYS_DDL, _PLAYS_ROWS])
    result = tools.nfl_status()
    assert "error" not in result
    assert result["plays"]["total_plays"] == 3
    assert result["datasets"]["code"] is ErrorCode.TABLE_NOT_FOUND
    assert "_ingest_metadata" in result["datasets"]["error"]


def test_status_returns_one_error_when_neither_section_can_be_read(tmp_path, monkeypatch):
    _write_db(tmp_path, monkeypatch, [])
    result = tools.nfl_status()
    assert "plays" not in result and "datasets" not in result
    assert result["code"] is ErrorCode.TABLE_NOT_FOUND


def _records_db(tmp_path, monkeypatch, rows, *, shape: str = "new", with_log: bool = True):
    """A database with an empty `team_records` table and the given ingest log rows."""
    statements = [_TEAM_RECORDS_DDL]
    if with_log:
        ddl = _METADATA_NEW_DDL if shape == "new" else _METADATA_OLD_DDL
        statements += _log_statements(ddl, rows, with_status=shape == "new")
    _write_db(tmp_path, monkeypatch, statements)


ESPN_LOG = [
    ("espn_records", "espn_team_records", 1994, 28, "2026-01-01", "loaded", None),
    ("espn_records", "espn_team_records", 1996, 26, "2026-01-01", "partial", "28 of 30 franchises"),
    ("espn_records", "espn_team_records", 1997, 0, "2026-01-01", "failed", "HTTP 500"),
]

_PULL_ADVICE = (
    "No stored record for that franchise and season. Records before "
    f"{RECORDS_FIRST_NFLVERSE_SEASON} come from the opt-in ESPN pull: "
    "run 'nfl-mcp ingest --pre1999'."
)


def test_record_absent_after_a_partial_espn_season_says_the_source_has_none(tmp_path, monkeypatch):
    _records_db(tmp_path, monkeypatch, ESPN_LOG)
    result = tools.nfl_team_records(team="GB", season=1996)
    assert result["status"] is Availability.RECORD_NOT_AVAILABLE
    assert result["source_checked"] == "espn"
    assert "pre1999" not in result["reason"]
    assert "published no record" in result["reason"]
    assert "running the ingest again will not add it" in result["reason"]


def test_record_absent_after_a_loaded_espn_season_says_the_source_has_none(tmp_path, monkeypatch):
    _records_db(tmp_path, monkeypatch, ESPN_LOG)
    result = tools.nfl_team_records(team="CIN", season=1994)
    assert result["source_checked"] == "espn"


def test_record_absent_over_a_pre1999_range_the_pull_covered_says_the_same(tmp_path, monkeypatch):
    _records_db(tmp_path, monkeypatch, ESPN_LOG)
    result = tools.nfl_team_records(team="CIN", from_season=1990, to_season=1995)
    assert result["source_checked"] == "espn"
    assert "--pre1999" not in result["reason"]


def test_record_absent_after_a_failed_espn_season_still_points_at_the_pull(tmp_path, monkeypatch):
    _records_db(tmp_path, monkeypatch, ESPN_LOG)
    result = tools.nfl_team_records(team="GB", season=1997)
    assert "source_checked" not in result
    assert result["reason"] == _PULL_ADVICE


def test_record_absent_before_any_espn_pull_still_points_at_the_pull(tmp_path, monkeypatch):
    _records_db(tmp_path, monkeypatch, ESPN_LOG[:1])
    result = tools.nfl_team_records(team="GB", season=1996)
    assert "source_checked" not in result
    assert result["reason"] == _PULL_ADVICE


def test_record_absent_with_an_old_log_and_an_espn_row_says_the_source_has_none(tmp_path, monkeypatch):
    # The old table has no `status`, so its one espn_records row reads as loaded.
    _records_db(tmp_path, monkeypatch, ESPN_LOG, shape="old")
    result = tools.nfl_team_records(team="GB", season=1997)
    assert result["source_checked"] == "espn"


def test_record_absent_when_the_log_cannot_be_read_still_points_at_the_pull(tmp_path, monkeypatch):
    _records_db(tmp_path, monkeypatch, [], with_log=False)
    result = tools.nfl_team_records(team="GB", season=1996)
    assert "source_checked" not in result
    assert result["reason"] == _PULL_ADVICE


def test_record_absent_on_an_open_ended_range_makes_no_espn_claim(tmp_path, monkeypatch):
    _records_db(tmp_path, monkeypatch, ESPN_LOG)
    result = tools.nfl_team_records(team="GB", from_season=1970)
    assert "source_checked" not in result
    assert result["reason"] == "No stored record for that franchise and season."


def test_modern_record_absent_makes_no_espn_claim(tmp_path, monkeypatch):
    _records_db(tmp_path, monkeypatch, ESPN_LOG)
    result = tools.nfl_team_records(team="GB", season=2024)
    assert "source_checked" not in result
    assert result["reason"] == "No stored record for that franchise and season."


# ── Questions the data cannot answer ──────────────────────────────────────────
#
# A success means "the question was understood, and this is the data". The tests below
# walk every public tool through the three shared checks: the team resolver, the
# fixed-value check and the season coverage guard.


def _record_calls(monkeypatch, rows=None):
    """Replace `_execute` with a recorder that answers every query with `rows`."""
    calls = []

    def fake_execute(sql, params=None):
        calls.append((sql, list(params) if params else []))
        return list(rows or [])

    monkeypatch.setattr(tools, "_execute", fake_execute)
    return calls


def _parameters(name):
    return set(inspect.signature(getattr(tools, name)).parameters)


# Arguments the tools need before they reach any check.
_REQUIRED = {
    "nfl_team_stats": {"team": "KC"},
    "nfl_player_stats": {"player_name": "P.Mahomes"},
    "nfl_compare": {"entity1": "KC", "entity2": "BAL"},
}

_WHOLE_RAIDERS = {"LV", "OAK"}

# Every argument that names a team: (tool, argument, other kwargs, codes the query must
# carry for the Raiders). nfl_team_records reads team_records, which the ingest already
# keys by the canonical code, so it sends LV alone.
TEAM_ARGUMENT_CASES = [
    ("nfl_search_plays", "team", {}, _WHOLE_RAIDERS),
    ("nfl_search_plays", "opponent", {}, _WHOLE_RAIDERS),
    ("nfl_team_stats", "team", {}, _WHOLE_RAIDERS),
    ("nfl_compare", "entity1", {"entity2": "KC"}, _WHOLE_RAIDERS),
    ("nfl_compare", "entity2", {"entity1": "KC"}, _WHOLE_RAIDERS),
    ("nfl_roster", "team", {}, _WHOLE_RAIDERS),
    ("nfl_injuries", "team", {}, _WHOLE_RAIDERS),
    ("nfl_schedule", "team", {}, _WHOLE_RAIDERS),
    ("nfl_team_records", "team", {"season": 2024}, {"LV"}),
    ("nfl_snap_counts", "team", {}, _WHOLE_RAIDERS),
    ("nfl_fantasy_opportunity", "team", {}, _WHOLE_RAIDERS),
    ("nfl_fantasy_rankings", "team", {}, _WHOLE_RAIDERS),
    ("nfl_ftn_charting", "team", {}, _WHOLE_RAIDERS),
    ("nfl_ftn_charting", "opponent", {}, _WHOLE_RAIDERS),
    ("nfl_td_luck", "team", {}, _WHOLE_RAIDERS),
    ("nfl_role_trend", "team", {}, _WHOLE_RAIDERS),
    ("nfl_separation_opportunity", "team", {}, _WHOLE_RAIDERS),
    ("nfl_drop_rate", "team", {}, _WHOLE_RAIDERS),
    ("nfl_contract_value", "team", {}, _WHOLE_RAIDERS),
]


def test_every_team_argument_is_walked_by_the_team_tests():
    declared = {
        (name, param) for name in tools.__all__
        for param in _parameters(name) if param in ("team", "opponent")
    }
    walked = {(name, param) for name, param, _, _ in TEAM_ARGUMENT_CASES}
    assert declared <= walked
    assert walked - declared == {("nfl_compare", "entity1"), ("nfl_compare", "entity2")}


@pytest.mark.parametrize("name,param,kwargs,codes", TEAM_ARGUMENT_CASES)
def test_every_team_argument_refuses_an_unknown_team(monkeypatch, name, param, kwargs, codes):
    _record_calls(monkeypatch)
    result = getattr(tools, name)(**{**kwargs, param: "ZZZ"})
    assert result["code"] is ErrorCode.UNKNOWN_TEAM
    assert result["error"] == "Unknown team 'ZZZ'."
    assert result["parameter"] == param
    assert result["did_you_mean"] == []


@pytest.mark.parametrize("name,param,kwargs,codes", TEAM_ARGUMENT_CASES)
def test_every_team_argument_sends_every_code_of_a_relocated_franchise(
    monkeypatch, name, param, kwargs, codes,
):
    calls = _record_calls(monkeypatch)
    result = getattr(tools, name)(**{**kwargs, param: "oak"})
    assert "error" not in result
    raiders = [(sql, params) for sql, params in calls if "LV" in params]
    assert raiders, "no query carried the Raiders"
    for sql, params in raiders:
        assert set(params) & _WHOLE_RAIDERS == codes
        assert "?" in sql and "OAK" not in sql and "LV" not in sql


def test_an_unknown_team_offers_close_codes(monkeypatch):
    _record_calls(monkeypatch)
    result = tools.nfl_roster(team="KCC")
    assert result["did_you_mean"] == ["KC"]


def test_the_jaguars_reach_the_fantasypros_code(monkeypatch):
    # FantasyPros files the Jaguars as JAC; nflverse uses JAX.
    calls = _record_calls(monkeypatch)
    tools.nfl_fantasy_rankings(team="JAX")
    assert "team IN (?, ?)" in calls[-1][0]
    assert {"JAX", "JAC"} <= set(calls[-1][1])


def test_a_code_only_the_table_stores_is_accepted(monkeypatch):
    # FantasyPros lists free agents as FA, which is no franchise.
    calls = []

    def fake_execute(sql, params=None):
        calls.append((sql, params))
        return [{"1": 1}] if sql.startswith("SELECT 1 FROM ff_rankings_week") else []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_fantasy_rankings(team="fa", scope="week")
    assert "error" not in result
    assert calls[0] == ("SELECT 1 FROM ff_rankings_week WHERE team = ? LIMIT 1", ["FA"])
    assert calls[-1][1][0] == "FA"


def test_a_failed_stored_code_lookup_returns_the_classified_error(monkeypatch):
    monkeypatch.setattr(tools, "_execute", _raise(TimeoutError("probe timeout")))
    result = tools.nfl_roster(team="ZZZ")
    assert result["code"] is ErrorCode.TIMEOUT
    assert result["error"] == "probe timeout"


def test_a_team_check_runs_only_when_a_team_is_given(monkeypatch):
    calls = _record_calls(monkeypatch)
    tools.nfl_roster(team="")
    assert calls[0][1] == []


# Every parameter with a fixed set of values: (tool, other kwargs, parameter, allowed).
CHOICE_CASES = [
    ("nfl_search_plays", {}, "season_type", tools.SEASON_TYPES),
    ("nfl_search_plays", {}, "play_type", tools.PLAY_TYPES),
    ("nfl_search_plays", {}, "situation", tools.SITUATIONS),
    ("nfl_team_stats", {"team": "KC"}, "side", tools.TEAM_STAT_SIDES),
    ("nfl_player_stats", {"player_name": "P.Mahomes"}, "season_type", tools.SEASON_TYPES),
    ("nfl_player_stats", {"player_name": "P.Mahomes"}, "stat_type", tools.STAT_TYPES),
    ("nfl_compare", {"entity1": "KC", "entity2": "BAL"}, "compare_type", tools.COMPARE_TYPES),
    ("nfl_compare", {"entity1": "KC", "entity2": "BAL"}, "season_type", tools.SEASON_TYPES),
    ("nfl_schedule", {}, "season_type", tools.SCHEDULE_SEASON_TYPES),
    ("nfl_fantasy_rankings", {}, "scope", tools.RANKING_SCOPES),
    ("nfl_ftn_charting", {}, "season_type", tools.SEASON_TYPES),
]


def test_every_schema_enum_is_a_checked_choice_with_the_same_values():
    # The input schema and the tool must accept the same set, or a direct caller and an
    # MCP client get different answers to the same question.
    in_schema = {
        (tool.name, name): tuple(prop["enum"])
        for tool in server_module.TOOLS
        for name, prop in tool.inputSchema.get("properties", {}).items()
        if "enum" in prop
    }
    checked = {(name, param): tuple(allowed) for name, _, param, allowed in CHOICE_CASES}
    assert in_schema == checked


@pytest.mark.parametrize("name,kwargs,param,allowed", CHOICE_CASES)
def test_an_unrecognized_fixed_value_is_invalid_input(monkeypatch, name, kwargs, param, allowed):
    calls = _record_calls(monkeypatch)
    result = getattr(tools, name)(**{**kwargs, param: "bogus"})
    assert result["code"] is ErrorCode.INVALID_INPUT
    assert result["parameter"] == param
    assert result["allowed"] == list(allowed)
    assert result["error"].startswith(f"Unknown {param}: bogus. Use ")
    assert calls == []


@pytest.mark.parametrize("name,kwargs,param,allowed", CHOICE_CASES)
def test_every_fixed_value_is_accepted_in_any_case(monkeypatch, name, kwargs, param, allowed):
    _record_calls(monkeypatch)
    for value in allowed:
        result = getattr(tools, name)(**{**kwargs, param: value.lower()})
        assert "error" not in result, (name, param, value)


def test_search_plays_sends_the_canonical_spelling_of_a_choice(monkeypatch):
    calls = _record_calls(monkeypatch)
    tools.nfl_search_plays(season_type="post", play_type="PASS")
    assert calls[0][1] == ["POST", "pass"]


def test_schedule_post_matches_every_playoff_round(monkeypatch):
    calls = _record_calls(monkeypatch)
    tools.nfl_schedule(season_type="post")
    assert "game_type IN (?, ?, ?, ?)" in calls[0][0]
    assert calls[0][1] == ["WC", "DIV", "CON", "SB"]


def test_schedule_one_playoff_round_matches_that_round(monkeypatch):
    calls = _record_calls(monkeypatch)
    tools.nfl_schedule(season_type="sb")
    assert calls[0][1] == ["SB"]


def test_spelled_options_read_as_a_list():
    assert tools._spell_options(("a", "b")) == "'a' or 'b'"
    assert tools._spell_options(("a", "b", "c")) == "'a', 'b', or 'c'"


def test_week_zero_is_a_filter_not_a_missing_value(monkeypatch):
    calls = _record_calls(monkeypatch)
    tools.nfl_search_plays(week=0)
    assert "week = ?" in calls[0][0]
    assert calls[0][1] == [0]


# ── Season coverage ───────────────────────────────────────────────────────────

def test_every_tool_with_a_season_argument_has_a_coverage_entry():
    with_season = {
        name for name in tools.__all__
        if _parameters(name) & {"season", "season_from", "from_season"}
    }
    assert with_season == set(tools._COVERAGE)


def test_coverage_floors_come_from_the_registry():
    assert tools.first_season("nfl_search_plays") == REGISTRY["pbp"].min_season
    assert tools.first_season("nfl_roster") == REGISTRY["rosters"].min_season
    # A join has rows only where every source does.
    assert tools.first_season("nfl_separation_opportunity") == 2016
    assert tools.first_season("nfl_drop_rate") == 2022
    assert tools.first_season("nfl_team_records") is None


_FLOORED = sorted(name for name in tools._COVERAGE if tools.first_season(name) is not None)


@pytest.mark.parametrize("name", _FLOORED)
def test_a_season_before_the_dataset_is_out_of_coverage(monkeypatch, name):
    calls = _record_calls(monkeypatch)
    floor = tools.first_season(name)
    result = getattr(tools, name)(**_REQUIRED.get(name, {}), season=floor - 1)
    assert result["code"] is ErrorCode.OUT_OF_COVERAGE
    assert result["first_season"] == floor
    assert result["error"] == f"{tools._COVERAGE[name][1]} is only available from {floor} onward"
    assert calls == []
    assert "error" not in getattr(tools, name)(**_REQUIRED.get(name, {}), season=floor)


@pytest.mark.parametrize("name", sorted(tools._COVERAGE))
def test_a_season_after_the_current_one_is_out_of_coverage(monkeypatch, name):
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(tools, "current_season", lambda: 2026)
    result = getattr(tools, name)(**_REQUIRED.get(name, {}), season=2027)
    assert result["code"] is ErrorCode.OUT_OF_COVERAGE
    assert result["last_season"] == 2026
    assert "through the 2026 season" in result["error"]
    assert calls == []


@pytest.mark.parametrize("kwargs,refused", [
    ({"season_from": 1990, "season_to": 1995}, True),   # wholly before
    ({"season_to": 1990}, True),                       # ends before
    ({"season_from": 2027}, True),                     # starts after
    ({"season_from": 1990, "season_to": 2005}, False),  # overlaps the floor
    ({"season_from": 2020, "season_to": 2030}, False),  # overlaps the current season
])
def test_a_range_is_refused_only_when_it_lies_wholly_outside(monkeypatch, kwargs, refused):
    _record_calls(monkeypatch)
    monkeypatch.setattr(tools, "current_season", lambda: 2026)
    result = tools.nfl_search_plays(**kwargs)
    assert ("code" in result) is refused


def test_a_declared_upper_bound_caps_the_last_season(monkeypatch):
    monkeypatch.setitem(tools._COVERAGE, "nfl_team_records", (("espn_records",), "ESPN records"))
    assert tools._last_season("nfl_team_records") == REGISTRY["espn_records"].max_season


# ── nfl_team_records: gaps, completeness, and an empty answer ─────────────────

def test_team_range_reports_one_gap_per_missing_season(monkeypatch):
    rows = [
        {"season": 2026, "team": "KC", "source_team": "KC", "games": 6, "source": "nflverse"},
        {"season": 2024, "team": "KC", "source_team": "KC", "games": 17, "source": "nflverse"},
    ]
    _record_calls(monkeypatch, rows)
    monkeypatch.setattr(tools, "current_season", lambda: 2026)
    result = tools.nfl_team_records(team="KC", from_season=2023, to_season=2026)
    assert [gap["season"] for gap in result["gaps"]] == [2023, 2025]
    assert {gap["status"] for gap in result["gaps"]} == {Availability.RECORD_NOT_AVAILABLE}
    assert "status" not in result
    assert [(row["season"], row["complete"]) for row in result["records"]] == [
        (2026, False), (2024, True),
    ]


def test_team_range_before_the_franchise_existed_says_so(monkeypatch):
    # HOU is the Texans (2002+); the old answer was "no stored record".
    _record_calls(monkeypatch)
    result = tools.nfl_team_records(team="HOU", from_season=1990, to_season=1995)
    assert result["status"] is Availability.FRANCHISE_NOT_YET_ACTIVE
    assert result["first_season"] == 2002
    assert len(result["gaps"]) == 6
    assert all(gap["status"] is Availability.FRANCHISE_NOT_YET_ACTIVE for gap in result["gaps"])


def test_team_range_with_mixed_reasons_keeps_record_not_available_on_top(monkeypatch):
    _record_calls(monkeypatch)
    result = tools.nfl_team_records(team="CLE", from_season=1995, to_season=1997)
    assert result["status"] is Availability.RECORD_NOT_AVAILABLE
    assert [gap["status"] for gap in result["gaps"]] == [
        Availability.RECORD_NOT_AVAILABLE,
        Availability.FRANCHISE_INACTIVE,
        Availability.FRANCHISE_INACTIVE,
    ]
    assert "--pre1999" in result["gaps"][0]["reason"]


def test_team_range_gaps_are_clipped_to_the_records_floor(monkeypatch):
    _record_calls(monkeypatch)
    result = tools.nfl_team_records(team="GB", from_season=1960, to_season=1972)
    assert [gap["season"] for gap in result["gaps"]] == [1970, 1971, 1972]


def test_team_range_wholly_before_the_records_floor(monkeypatch):
    _record_calls(monkeypatch)
    gb = tools.nfl_team_records(team="GB", from_season=1950, to_season=1960)
    assert gb["status"] is Availability.BEFORE_COVERAGE
    assert gb["gaps"] == []
    sea = tools.nfl_team_records(team="SEA", from_season=1950, to_season=1960)
    assert sea["status"] is Availability.FRANCHISE_NOT_YET_ACTIVE


def test_team_range_gaps_read_espn_coverage_per_season(tmp_path, monkeypatch):
    _records_db(tmp_path, monkeypatch, ESPN_LOG)
    result = tools.nfl_team_records(team="CIN", from_season=1993, to_season=1995)
    by_season = {gap["season"]: gap for gap in result["gaps"]}
    assert by_season[1994]["source_checked"] == "espn"
    assert by_season[1993]["reason"] == _PULL_ADVICE
    assert "source_checked" not in by_season[1995]


def test_season_only_query_with_nothing_stored_says_why(monkeypatch):
    _record_calls(monkeypatch)
    before = tools.nfl_team_records(season=1960)
    assert before["status"] is Availability.BEFORE_COVERAGE
    assert before["first_season"] == 1970
    pre1999 = tools.nfl_team_records(season=1985)
    assert pre1999["status"] is Availability.RECORD_NOT_AVAILABLE
    assert "--pre1999" in pre1999["reason"]
    modern = tools.nfl_team_records(season=2020)
    assert modern["reason"] == "No stored records for that season."
    assert "gaps" not in modern


def test_season_only_query_after_the_espn_pull_drops_the_advice(tmp_path, monkeypatch):
    _records_db(tmp_path, monkeypatch, ESPN_LOG)
    result = tools.nfl_team_records(season=1994)
    assert result["reason"] == "No stored records for that season."


def test_a_season_with_rows_carries_no_status(monkeypatch):
    rows = [{"season": 2024, "team": "KC", "source_team": "KC", "games": 17, "source": "nflverse"}]
    _record_calls(monkeypatch, rows)
    result = tools.nfl_team_records(season=2024)
    assert "status" not in result
    assert result["records"][0]["complete"] is True


# ── nfl_compare: notes, and two names for one entity ──────────────────────────

def test_compare_refuses_two_codes_for_the_same_franchise(monkeypatch):
    calls = _record_calls(monkeypatch)
    result = tools.nfl_compare(entity1="LV", entity2="oak")
    assert result["code"] is ErrorCode.INVALID_INPUT
    assert "same team (LV)" in result["error"]
    assert calls == []


def test_compare_refuses_two_spellings_of_the_same_player(monkeypatch):
    result = tools.nfl_compare("Patrick Mahomes", "P.Mahomes", compare_type="player")
    assert result["code"] is ErrorCode.INVALID_INPUT
    assert "same player" in result["error"]


def test_compare_team_mode_says_it_did_not_apply_season_type(monkeypatch):
    _record_calls(monkeypatch)
    with_type = tools.nfl_compare("KC", "BAL", season_type="POST")
    assert "season_type=POST was not applied" in with_type["notes"][0]["note"]
    assert tools.nfl_compare("KC", "BAL")["notes"] == []


def test_compare_player_mode_notes_a_player_no_play_matched(monkeypatch):
    def fake_execute(sql, params=None):
        return [{"n": 3}] if "%P.Mahomes%" in params and "passer" in sql else [{"n": 0}]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    monkeypatch.setattr(tools, "nfl_player_stats", lambda *a, **k: {"seasons": [{"season": 2024}]})
    result = tools.nfl_compare("P.Mahomes", "Zz Nobody", compare_type="player")
    assert result["P.Mahomes"] == {"passing": [{"season": 2024}]}
    assert result["Zz Nobody"] == {}
    assert result["notes"] == [{"entity": "Zz Nobody", "note": result["notes"][0]["note"]}]
    assert "No play matched 'Zz Nobody'" in result["notes"][0]["note"]


def test_compare_player_mode_notes_a_failed_stat_lookup(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda sql, params=None: [{"n": 1}])
    monkeypatch.setattr(
        tools, "nfl_player_stats",
        lambda *a, stat_type, **k: (
            {"error": "slow", "code": ErrorCode.TIMEOUT} if stat_type == "rushing"
            else {"seasons": []}
        ),
    )
    result = tools.nfl_compare("P.Mahomes", "J.Allen", compare_type="player")
    failed = [note for note in result["notes"] if note.get("stat_type") == "rushing"]
    assert [note["entity"] for note in failed] == ["P.Mahomes", "J.Allen"]
    assert all(note["code"] is ErrorCode.TIMEOUT for note in failed)
    assert "rushing" not in result["P.Mahomes"]
    assert "passing" in result["P.Mahomes"]


def test_compare_player_mode_returns_a_query_failure(monkeypatch):
    monkeypatch.setattr(tools, "_execute", _raise(TimeoutError("count timeout")))
    result = tools.nfl_compare("P.Mahomes", "J.Allen", compare_type="player")
    assert result["code"] is ErrorCode.TIMEOUT


# ── nfl_schema: one list key on both errors, and an ignored category ─────────

def test_schema_table_not_found_also_lists_tables_under_available(monkeypatch):
    def fake_execute(sql, params=None):
        return [] if "information_schema.columns" in sql else [{"table_name": "plays"}]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_schema(table="nope")
    assert result["available"] == result["available_tables"] == ["plays"]


def test_schema_unknown_category_also_lists_categories_by_name():
    result = tools.nfl_schema(category="nope")
    assert result["available_categories"] == result["available"]


def test_schema_says_when_it_ignored_the_category(monkeypatch):
    monkeypatch.setattr(
        tools, "_execute",
        lambda sql, params=None: [{"column_name": "team", "data_type": "VARCHAR"}],
    )
    both = tools.nfl_schema(table="rosters", category="epa")
    assert "category='epa' was ignored" in both["note"]
    assert "note" not in tools.nfl_schema(table="rosters")
