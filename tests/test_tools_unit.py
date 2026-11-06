"""DB-independent unit tests for query construction and validation."""

import time
from collections import defaultdict
from contextlib import contextmanager

import duckdb
import pytest

import nfl_mcp.tools as tools


@pytest.fixture
def no_player_match(monkeypatch):
    """Names resolve to nobody, so a tool falls back to a substring match on its own name column."""
    monkeypatch.setattr(tools, "_find_players", lambda *a, **k: ([], False))
    monkeypatch.setattr(tools, "_table_players", lambda *a, **k: [])


MAHOMES = {"gsis_id": "00-0033873", "display_name": "Patrick Mahomes"}


@pytest.fixture
def resolves_to_mahomes(monkeypatch):
    """Record the names handed to the resolver and resolve every one of them to Mahomes."""
    seen = []

    def fake_resolve(player_name, player_id, **kwargs):
        seen.append((player_name, player_id, kwargs))
        return dict(MAHOMES), None

    monkeypatch.setattr(tools, "_resolve_player", fake_resolve)
    return seen


def test_search_plays_uses_parameterized_filters(monkeypatch, resolves_to_mahomes):
    captured = {}

    def fake_execute(sql, params=None):
        if sql.startswith("SELECT MIN("):  # empty-result range lookup, not the tool's query
            return []
        captured["sql"] = sql
        captured["params"] = params
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_search_plays(team="KC", player="Mahomes", season=2024, max_rows=5, season_type="REG")

    assert result["row_count"] == result["count"] == 0
    assert "posteam = ?" in captured["sql"]
    assert "passer_player_id = ?" in captured["sql"]
    assert "LIMIT 6 OFFSET 0" in captured["sql"]  # one extra row shows whether another page exists
    assert captured["params"][0] == "KC"
    assert MAHOMES["gsis_id"] in captured["params"]
    assert 2024 in captured["params"]


def test_team_stats_uses_parameterized_team_and_season(monkeypatch):
    calls = []

    def fake_execute(sql, params=None):
        if sql.startswith("SELECT MIN("):  # empty-result range lookup, not the tool's query
            return []
        calls.append((sql, params))
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    tools.nfl_team_stats(team="kc", season=2024, side="both", season_type="REG")

    assert len(calls) == 3
    assert all("team = ?" in sql for sql, _ in calls)
    assert all(params[0] == "KC" for _, params in calls)
    assert all(2024 in params for _, params in calls)


def test_player_stats_uses_parameterized_player_id(monkeypatch, resolves_to_mahomes):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_player_stats(player="P.Mahomes", season_type="REG")

    assert "me.id" in captured["sql"]
    assert captured["params"] == [MAHOMES["gsis_id"], "REG"]
    assert result["player_id"] == MAHOMES["gsis_id"]
    assert result["seasons"] == []


def test_player_stats_passes_full_name_to_resolver_unchanged(monkeypatch, resolves_to_mahomes):
    monkeypatch.setattr(tools, "_execute", lambda sql, params=None: [])
    tools.nfl_player_stats(player="Justin Jefferson", season_type="REG")

    assert resolves_to_mahomes[0][0] == "Justin Jefferson"


@pytest.mark.parametrize("stat_type", ["passing", "rushing", "receiving"])
def test_player_stats_stat_type_keeps_one_block(monkeypatch, resolves_to_mahomes, stat_type):
    captured = {}
    row = defaultdict(lambda: None, season=2024, season_type="REG")  # every other stat is NULL

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return [row]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_player_stats(
        player="P.Mahomes",
        stat_type=stat_type,
        season_from=2020,
        season_to=2024,
        season_type="REG",
    )
    (season,) = result["seasons"]
    assert set(season) == {"season", "season_type", stat_type}
    assert "season >=" in captured["sql"]
    assert "season <=" in captured["sql"]
    assert captured["params"] == [MAHOMES["gsis_id"], 2020, 2024, "REG"]


def test_player_stats_rejects_unknown_stat_type():
    result = tools.nfl_player_stats(player="P.Mahomes", stat_type="kicking", season_type="REG")
    assert "Unknown stat_type" in result["error"]


def test_player_stats_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timeout")))
    result = tools.nfl_player_stats(player="P.Mahomes", stat_type="passing", season_type="REG")
    assert "error" in result


def test_compare_player_path_resolves_both_entities_to_ids(monkeypatch):
    seen = []
    ids = {"P.Mahomes": "00-0033873", "00-0034857": "00-0034857"}

    def fake_resolve(player_name, player_id, **kwargs):
        seen.append((player_name, kwargs))
        return {"gsis_id": ids[player_name], "display_name": player_name}, None

    calls = []
    monkeypatch.setattr(tools, "_resolve_player", fake_resolve)
    monkeypatch.setattr(
        tools, "_execute",
        lambda sql, params=None: [] if sql.startswith("SELECT MIN(") else calls.append(params) or [])
    result = tools.nfl_compare(
        player1="P.Mahomes",
        player2="00-0034857",
        season_from=2020,
        season_to=2024,
        season_type="REG",
    )
    assert [p["player_id"] for p in result["players"]] == ["00-0033873", "00-0034857"]
    assert [s[0] for s in seen] == ["P.Mahomes", "00-0034857"]
    assert calls == [["00-0033873", 2020, 2024, "REG"], ["00-0034857", 2020, 2024, "REG"]]


def test_compare_team_does_not_interpolate_user_input(monkeypatch):
    calls = []

    def fake_execute(sql, params=None):
        calls.append((sql, params))
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_compare(
        team1="KC'; DROP TABLE plays; --", team2="BAL", season=2024, season_type="REG")

    # Group H: an unknown team code is rejected before any SQL runs, so nothing is interpolated.
    assert result["code"] == "UNKNOWN_TEAM"
    assert calls == []


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
    responses = iter([
        [{"total_plays": 123}],
        [{"season": 2024, "season_type": "REG", "plays": 123}],
        [{"first_season": 2024, "last_season": 2024, "num_seasons": 1}],
        [{"dataset_id": "pbp", "table_name": "plays", "total_rows": 123}],
        [{"last_refreshed": "2026-03-01"}],
    ])
    monkeypatch.setattr(tools, "_execute", lambda sql, params=None: next(responses))

    result = tools.nfl_status()
    assert result["plays"]["total_plays"] == 123
    assert result["plays"]["season_range"]["first_season"] == 2024
    assert result["datasets"]["total_loaded"] == 1
    assert result["datasets"]["last_refreshed"] == "2026-03-01"


def test_nfl_status_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timeout")))
    result = tools.nfl_status()
    assert "error" in result


def test_nfl_query_reports_truncation(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_fetch_capped",
        lambda sql, limit: ([{"name": "x", "type": "INTEGER"}], [{"x": 1}, {"x": 2}, {"x": 3}]),
    )
    result = tools.nfl_query("SELECT 1 AS x", max_rows=2)
    assert result["row_count"] == 2
    assert result["truncated"] is True


def test_nfl_query_returns_error_when_execute_fails(monkeypatch):
    monkeypatch.setattr(tools, "_fetch_capped", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad sql")))
    result = tools.nfl_query("SELECT 1")
    assert result["error"] == "bad sql"
    assert "code" not in result


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


def _raise_from_fetch(monkeypatch, exc):
    """Make nfl_query's own statement fail with `exc` (schema lookups still use _execute)."""

    def fake_fetch(sql, limit):
        raise exc

    monkeypatch.setattr(tools, "_fetch_capped", fake_fetch)


def test_nfl_query_suggests_real_columns_for_an_invented_column(monkeypatch):
    monkeypatch.setattr(
        tools, "_execute", _failing_query(duckdb.BinderException(COLUMN_ERROR),
                                          [{"column_name": c} for c in PLAY_COLUMNS]),
    )
    _raise_from_fetch(monkeypatch, duckdb.BinderException(COLUMN_ERROR))
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
    _raise_from_fetch(monkeypatch, duckdb.CatalogException(TABLE_ERROR))
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
    _raise_from_fetch(monkeypatch, duckdb.BinderException(COLUMN_ERROR))
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
        ("two_minute", "half_seconds_remaining <= 120 AND (qtr = 2 OR qtr >= 4)"),
    ],
)
def test_search_plays_situations(monkeypatch, situation, expected_sql):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    tools.nfl_search_plays(situation=situation, max_rows=10, season_type="REG")
    assert expected_sql in captured["sql"]


def test_search_plays_applies_all_numeric_and_boolean_filters(monkeypatch):
    captured = {}

    def fake_execute(sql, params=None):
        if sql.startswith("SELECT MIN("):  # empty-result range lookup, not the tool's query
            return []
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
        max_rows=500,
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
    assert "LIMIT 501 OFFSET 0" in captured["sql"]
    assert captured["params"] == ["BAL", 2020, 2021, 1, "REG", "pass", 7]


def test_search_plays_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("boom")))
    result = tools.nfl_search_plays(team="KC", season_type="REG")
    assert "error" in result


def test_team_stats_offense_only_without_season_clause(monkeypatch):
    calls = []

    def fake_execute(sql, params=None):
        calls.append((sql, params))
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_team_stats(team="kc", side="offense", season_type="REG")
    assert "offense" in result
    assert "defense" not in result
    assert "season_year = ?" not in calls[0][0]
    assert calls[0][1] == ["KC", "REG"]  # season_type defaults to REG


def test_team_stats_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timeout")))
    result = tools.nfl_team_stats(team="KC", season_type="REG")
    assert "error" in result


def test_compare_rejects_a_mixed_or_incomplete_pair():
    for args in ({"player1": "P.Mahomes", "team2": "BAL"}, {"team1": "KC"}, {}):
        result = tools.nfl_compare(**args, season_type="REG")
        assert result["code"] == "INVALID_ARGUMENT"
        for name in ("player1", "player2", "team1", "team2"):
            assert name in result["error"]


def test_compare_team_applies_season_range_filters(monkeypatch):
    calls = []

    def fake_execute(sql, params=None):
        if sql.startswith("SELECT MIN("):  # empty-result range lookup, not the tool's query
            return []
        calls.append((sql, params))
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_compare(
        team1="KC",
        team2="BAL",
        season_from=2020,
        season_to=2021,
        season_type="REG",
    )
    assert "error" not in result
    assert len(calls) == 4
    assert "season_year >= ?" in calls[0][0]
    assert "season_year <= ?" in calls[0][0]
    assert calls[0][1] == ["KC", 2020, 2021, "REG"]  # team mode defaults to REG


def test_compare_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timeout")))
    result = tools.nfl_compare(team1="KC", team2="BAL", season=2024, season_type="REG")
    assert "error" in result


def test_catalog_returns_dataset_summary(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_execute",
        lambda sql, params=None: ([{"dataset_id": "pbp", "table_name": "plays", "total_rows": 123}]
                                  if "_ingest_metadata" in sql else []),  # the per-table queries find none
    )
    result = tools.nfl_catalog()
    assert result["total_datasets"] == 1
    assert result["datasets"][0]["dataset_id"] == "pbp"


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
        season_type="REG",
    )
    assert result["count"] == 1
    assert "team = ?" in captured["sql"]
    assert "report_status ILIKE ?" in captured["sql"]
    assert "game_type = ?" in captured["sql"]
    assert captured["params"] == ["KC", 2024, 1, "REG", "%Mahomes%", "%Out%"]


def test_injuries_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("injuries timeout")))
    result = tools.nfl_injuries(team="KC", season_type="REG")
    assert result["error"] == "injuries timeout"


def test_schedule_builds_team_pair_filter_and_game_type(monkeypatch):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return [{"game_id": "2024_01_KC_BAL"}]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_schedule(team="kc", season=2024, week=1, season_type="REG")
    assert result["count"] == 1
    assert "(home_team = ? OR away_team = ?)" in captured["sql"]
    assert "game_type = ?" in captured["sql"]
    assert captured["params"] == ["KC", "KC", 2024, 1, "REG"]


def test_schedule_season_type_is_exact_and_upper_case(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    result = tools.nfl_schedule(team="kc", season_type="reg")
    assert result["code"] == "INVALID_ARGUMENT"
    assert "REG, POST, ALL, WC, DIV, CON, SB" in result["error"]
    assert "sql" not in captured  # rejected before any query


def test_schedule_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("schedule timeout")))
    result = tools.nfl_schedule(team="KC", season_type="REG")
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
        season_type="REG",
    )
    assert result["count"] == 1
    assert "LIKE ?" in captured["sql"]
    assert "team = ?" in captured["sql"]
    assert "position ILIKE ?" in captured["sql"]
    # The name is folded (case, accents, punctuation) but not turned into a short name.
    assert captured["params"] == ["%kelce%", "KC", 2024, 1, "REG", "TE"]


def test_snap_counts_full_name_is_not_shortened(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    tools.nfl_snap_counts(player="Travis Kelce", season_type="REG")
    assert captured["params"] == ["%travis kelce%", "REG"]


def test_snap_counts_player_id_maps_through_pfr_id(monkeypatch):
    calls = []

    def fake_execute(sql, params=None):
        if sql.startswith("SELECT MIN("):  # empty-result range lookup, not the tool's query
            return []
        calls.append((sql, params))
        if "FROM players" in sql:
            return [{"gsis_id": "00-0030506", "display_name": "Travis Kelce", "short_name": "T.Kelce",
                     "position": "TE", "teams": ["KC"], "latest_team": "KC", "first_season": 2013,
                     "last_season": 2026, "pfr_id": "KelcTr00"}]
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    tools.nfl_snap_counts(player_id="00-0030506", season=2024, season_type="REG")
    sql, params = calls[-1]
    assert "pfr_player_id = ?" in sql
    assert params == ["KelcTr00", 2024, "REG"]


def test_snap_counts_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("snap timeout")))
    result = tools.nfl_snap_counts(team="KC", season_type="REG")
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


def test_execute_timeout_interrupt_warning_path(monkeypatch):
    class _Rel:
        description = [("x",)]

        def fetchall(self):
            return [(1,)]

    class _SlowConn:
        def execute(self, *_args, **_kwargs):
            time.sleep(0.25)
            return _Rel()

        def interrupt(self):
            raise RuntimeError("interrupt failed")

    @contextmanager
    def fake_get_db_connection():
        yield _SlowConn()

    warnings = []
    monkeypatch.setattr(tools, "get_db_connection", fake_get_db_connection)
    monkeypatch.setattr(tools, "_QUERY_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(tools.logger, "warning", lambda *a, **k: warnings.append((a, k)))

    with pytest.raises(TimeoutError, match="Query exceeded"):
        tools._execute("SELECT 1")
    assert warnings


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
    monkeypatch.setattr(tools, "_resolve_player", lambda *a, **k: (dict(MAHOMES), None))
    result = tools.nfl_ftn_charting(
        team="kc", opponent="phi", player="Mahomes",
        season_from=2022, season_to=2024, week=3, season_type="REG",
    )
    sql, params = captured["sql"], captured["params"]
    assert "p.posteam = ?" in sql and "p.defteam = ?" in sql
    assert "p.passer_player_id = ?" in sql
    assert "p.season >= ?" in sql and "p.season <= ?" in sql
    assert "p.week = ?" in sql and "p.season_type = ?" in sql
    assert params[:2] == ["KC", "PHI"]
    assert params[2:5] == [MAHOMES["gsis_id"]] * 3
    assert params[5:7] == [2022, 2024]
    assert params[-2:] == [3, "REG"]
    assert result["charting"]["play_action"]["pct"] == 40.0
    assert result["charting"]["motion"]["plays"] == 5


def test_ftn_charting_exact_season_filter(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        tools, "_execute",
        lambda sql, params=None: ([] if sql.startswith("SELECT MIN(")
                                  else captured.update(sql=sql, params=params) or [{"total_plays": 0}]),
    )
    result = tools.nfl_ftn_charting(season=2024, season_type="REG")
    assert "p.season = ?" in captured["sql"]
    assert captured["params"] == [2024, "REG"]
    result.pop("data_as_of", None)  # present whenever a database file exists (group J)
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
    result = tools.nfl_ftn_charting(team="KC", season_type="REG")
    c = result["charting"]
    assert c["play_action"]["pct"] == 0.0
    assert c["play_action"]["plays"] == 0
    assert c["avg_defenders_in_box"] is None


def test_ftn_charting_empty_rows(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda sql, params=None: [])
    result = tools.nfl_ftn_charting(team="KC", season_type="REG")
    result.pop("data_as_of", None)  # present whenever a database file exists (group J)
    assert result == {"charting": {}, "total_plays": 0}


def test_ftn_charting_returns_error_on_db_error(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(duckdb.Error("ftn boom")))
    result = tools.nfl_ftn_charting(team="KC", season_type="REG")
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


def test_td_luck_builds_filters(monkeypatch, no_player_match):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    result = tools.nfl_td_luck(player="Jefferson", position="wr", team="min", season=2024, limit=10, season_type="REG")
    assert result["count"] == 1
    assert "full_name ILIKE ?" in captured["sql"]
    assert "position ILIKE ?" in captured["sql"]
    assert "team = ?" in captured["sql"]
    assert "total_td_luck_score ASC" in captured["sql"]
    assert captured["params"] == ["%Jefferson%", "WR", "MIN", 2024, "REG"]


def test_td_luck_invalid_limit_returns_error(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    result = tools.nfl_td_luck(limit="oops", season_type="REG")
    assert result["code"] == "INVALID_ARGUMENT"
    assert "sql" not in captured


def test_td_luck_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("td timeout")))
    assert tools.nfl_td_luck(season_type="REG")["error"] == "td timeout"


def test_role_trend_builds_filters(monkeypatch, no_player_match):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    tools.nfl_role_trend(player="Kelce", position="te", team="kc", season=2024, week=8, min_snap_pct=50, season_type="REG")
    assert "full_name ILIKE ?" in captured["sql"]
    assert "snap_pct >= ?" in captured["sql"]
    assert "snap_pct_delta DESC" in captured["sql"]
    assert captured["params"] == ["%Kelce%", "TE", "KC", 2024, 8, 50.0, "REG"]


def test_role_trend_invalid_limit_returns_error(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    result = tools.nfl_role_trend(limit="oops", season_type="REG")
    assert result["code"] == "INVALID_ARGUMENT"
    assert "sql" not in captured


def test_role_trend_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("role timeout")))
    assert tools.nfl_role_trend(season_type="REG")["error"] == "role timeout"


def test_separation_opportunity_builds_filters(monkeypatch, no_player_match):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    tools.nfl_separation_opportunity(player="Aiyuk", position="wr", team="sf", season=2024, regression_candidate=True, season_type="REG")
    assert "regression_candidate = ?" in captured["sql"]
    assert "fp_diff_per_game ASC" in captured["sql"]
    assert captured["params"] == ["%Aiyuk%", "WR", "SF", 2024, True, "REG"]


def test_separation_opportunity_guards_pre_2016():
    result = tools.nfl_separation_opportunity(season=2015, season_type="REG")
    assert "2016" in result["error"]


def test_separation_opportunity_invalid_limit_returns_error(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    result = tools.nfl_separation_opportunity(limit="oops", season_type="REG")
    assert result["code"] == "INVALID_ARGUMENT"
    assert "sql" not in captured


def test_separation_opportunity_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("sep timeout")))
    assert tools.nfl_separation_opportunity(season_type="REG")["error"] == "sep timeout"


def test_drop_rate_builds_filters(monkeypatch, no_player_match):
    captured = _capture_execute(monkeypatch, [{"player": "T.Pollard"}])
    tools.nfl_drop_rate(player="Pollard", team="ten", season=2024, min_targets=30, season_type="REG")
    assert "player ILIKE ?" in captured["sql"]
    assert "catchable_targets >= ?" in captured["sql"]
    assert "drop_rate_pct DESC" in captured["sql"]
    assert captured["params"] == ["%Pollard%", "TEN", 2024, "REG", 30]


def test_drop_rate_guards_pre_2022():
    result = tools.nfl_drop_rate(season=2021, season_type="REG")
    assert "2022" in result["error"]


def test_drop_rate_invalid_limit_returns_error(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    result = tools.nfl_drop_rate(limit="oops", season_type="REG")
    assert result["code"] == "INVALID_ARGUMENT"
    assert "sql" not in captured


def test_drop_rate_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("drop timeout")))
    assert tools.nfl_drop_rate(season_type="REG")["error"] == "drop timeout"


def test_contract_value_builds_filters(monkeypatch, no_player_match):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    tools.nfl_contract_value(player="Gibbs", position="rb", team="det", season=2024, min_apy=2, max_apy=10, season_type="REG")
    assert "apy >= ?" in captured["sql"]
    assert "apy <= ?" in captured["sql"]
    assert "fp_per_million DESC" in captured["sql"]
    assert captured["params"] == ["%Gibbs%", "RB", "DET", 2024, 2.0, 10.0, "REG"]


def test_contract_value_invalid_limit_returns_error(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    result = tools.nfl_contract_value(limit="oops", season_type="REG")
    assert result["code"] == "INVALID_ARGUMENT"
    assert "sql" not in captured


def test_contract_value_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("contract timeout")))
    assert tools.nfl_contract_value(season_type="REG")["error"] == "contract timeout"


def test_injury_return_builds_filters(monkeypatch):
    captured = _capture_execute(monkeypatch, [{"injury_type": "hamstring"}])
    tools.nfl_injury_return(injury_type="hamstring", position="wr", week_post_return=2)
    assert "injury_type ILIKE ?" in captured["sql"]
    assert "position = ?" in captured["sql"]
    assert "week_post_return = ?" in captured["sql"]
    assert "week_post_return ASC" in captured["sql"]
    assert captured["params"] == ["%hamstring%", "WR", 2]


def test_injury_return_invalid_limit_returns_error(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    result = tools.nfl_injury_return(limit="oops")
    assert result["code"] == "INVALID_ARGUMENT"
    assert "sql" not in captured


def test_injury_return_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("injury timeout")))
    assert tools.nfl_injury_return()["error"] == "injury timeout"


# ── Player identity helpers ────────────────────────────────────────────────────

@pytest.mark.parametrize("name, strip, expected", [
    ("Ja'Marr Chase", False, "jamarr chase"),
    ("Amon-Ra St. Brown", False, "amon ra st brown"),
    ("Audric Estimé", False, "audric estime"),
    ("Odell Beckham Jr.", True, "odell beckham"),
    ("Odell Beckham Jr.", False, "odell beckham jr"),
    ("Patrick Mahomes II", True, "patrick mahomes"),
    ("V", True, "v"),            # a lone token is a name, not a suffix
    ("K.Allen", False, "kallen"),
    ("  ", False, ""),
])
def test_fold_player_name(name, strip, expected):
    assert tools._fold(name, strip_suffix=strip) == expected


def test_player_lookup_requires_a_name():
    result = tools.nfl_player_lookup(name="  ")
    assert result["code"] == "MISSING_NAME"


def test_player_stats_without_player_is_an_error_with_code():
    result = tools.nfl_player_stats(player=None, season_type="REG")
    assert result["code"] == "MISSING_PLAYER"


def test_player_stats_returns_ambiguous_without_running_stats(monkeypatch):
    candidates = [{"gsis_id": "a", "display_name": "X"}, {"gsis_id": "b", "display_name": "X"}]
    monkeypatch.setattr(tools, "_find_players", lambda *a, **k: (candidates, False))
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: pytest.fail("stats query must not run"))
    result = tools.nfl_player_stats(player="X", season=2024, season_type="REG")
    assert result["ambiguous"] is True and result["candidates"] == candidates
    assert "error" not in result and "seasons" not in result


def test_resolver_uses_a_lone_overall_match_when_the_season_has_none(monkeypatch):
    only = {"gsis_id": "a", "display_name": "X"}
    monkeypatch.setattr(
        tools, "_find_players", lambda name=None, **k: ([], False) if k.get("season") else ([only], False))
    assert tools._resolve_player("X", None, season=2016) == (only, None)


def test_player_stats_with_no_matching_player_says_so(monkeypatch):
    monkeypatch.setattr(tools, "_find_players", lambda *a, **k: ([], False))
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: pytest.fail("stats query must not run"))
    result = tools.nfl_player_stats(player="Nobody", season=2024, season_type="REG")
    assert result["seasons"] == [] and result["player_id"] is None
    assert "No player matches 'Nobody'" in result["reason"]


# ── season_type on nfl_fantasy_opportunity and nfl_role_trend ────────────────

def test_fantasy_opportunity_filters_by_season_type(monkeypatch, no_player_match):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    tools.nfl_fantasy_opportunity(player="Kelce", team="kc", season=2024, week=3, position="te",
                                  season_type="POST")
    assert "full_name ILIKE ?" in captured["sql"] and "position ILIKE ?" in captured["sql"]
    assert "schedules" in captured["sql"] and "AS season_type" in captured["sql"]
    assert captured["params"][-4:] == ["2024", 3, "TE", "POST"]


def test_fantasy_opportunity_round_code_filters_on_the_schedule_game_type(monkeypatch, no_player_match):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    tools.nfl_fantasy_opportunity(season=2024, season_type="SB")
    assert "g.game_type = ?" in captured["sql"] and captured["params"] == ["2024", "SB"]


def test_fantasy_opportunity_all_adds_no_season_type_filter(monkeypatch, no_player_match):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    tools.nfl_fantasy_opportunity(season=2024, season_type="ALL")
    assert captured["params"] == ["2024"]
    assert "g.game_type = ?" not in captured["sql"]


def test_role_trend_round_code_filters_through_the_schedule(monkeypatch, no_player_match):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    tools.nfl_role_trend(season=2024, season_type="DIV")
    assert "EXISTS (SELECT 1 FROM ff_opportunity o JOIN schedules g" in captured["sql"]
    assert captured["params"] == [2024, "DIV"]


def test_role_trend_all_adds_no_season_type_filter(monkeypatch, no_player_match):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    tools.nfl_role_trend(season=2024, season_type="ALL")
    assert captured["params"] == [2024] and "EXISTS" not in captured["sql"]


def test_search_plays_returns_the_resolver_error_for_an_unknown_player_id(monkeypatch):
    unknown = {"error": "No player with player_id '00-9999999'", "code": "UNKNOWN_PLAYER"}
    monkeypatch.setattr(tools, "_resolve_player", lambda *a, **k: (None, unknown))
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: pytest.fail("plays query must not run"))
    assert tools.nfl_search_plays(player_id="00-9999999", season=2024, season_type="REG") == unknown
