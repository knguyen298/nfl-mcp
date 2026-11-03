"""DB-independent unit tests for query construction and validation."""

import time
from contextlib import contextmanager

import duckdb
import pytest

import nfl_mcp.tools as tools


def test_name_tokens_folds_punctuation_accents_and_suffixes():
    assert tools._name_tokens("Justin Jefferson") == ["justin", "jefferson"]
    assert tools._name_tokens("J.Jefferson") == ["j", "jefferson"]
    assert tools._name_tokens("J. Jefferson") == ["j", "jefferson"]
    assert tools._name_tokens("Kenneth Walker III") == ["kenneth", "walker"]
    assert tools._name_tokens("Willie Jefferson Jr.") == ["willie", "jefferson"]
    assert tools._name_tokens("D'vonta Smith-Sr.") == ["d", "vonta", "smith"]
    assert tools._name_tokens("Audric Estimé") == ["audric", "estime"]
    assert tools._name_tokens("  ") == []


def test_first_name_matches_full_preferred_and_initial_forms():
    row = {"first_name": "David", "football_name": None, "common_first_name": "D.J."}
    assert tools._first_name_matches(["david"], row)          # full first name
    assert not tools._first_name_matches(["d"], row, initial=False)  # one letter is an initial
    assert tools._first_name_matches(["d", "j"], row)         # 'D.J.' as two tokens
    assert tools._first_name_matches(["dj"], row)             # 'DJ' squeezed
    assert tools._first_name_matches(["d", "j"], {"first_name": "David"})  # initial, second pass
    assert not tools._first_name_matches(["d", "j"], {"first_name": "David"}, initial=False)
    assert tools._first_name_matches([], row)                 # surname-only query
    assert not tools._first_name_matches(["justin"], row)
    assert not tools._first_name_matches(["d"], {})           # nothing to compare


def test_spelled_out_pass_refuses_every_one_letter_form():
    """A stored given name of 'B.' joins to 'b', so the guard is needed twice."""
    stored_initial = {"first_name": "B.", "football_name": None, "common_first_name": None}
    assert not tools._first_name_matches(["b"], stored_initial, initial=False)  # joined form
    assert not tools._first_name_matches(["b"], {"first_name": "Brian"}, initial=False)  # first token
    assert tools._first_name_matches(["b"], stored_initial)      # the initial pass still may
    assert tools._first_name_matches(["b"], {"first_name": "Brian"})
    # the token-length guard: a 2-letter head token must not borrow another player's initial
    assert not tools._first_name_matches(["a", "st"], {"first_name": "A.J."}, initial=False)
    assert tools._first_name_matches(["a", "st"], {"first_name": "Amon-Ra"})


def test_initial_pass_accepts_the_prefix_forms_plays_writes():
    assert tools._first_name_matches(["mi"], {"first_name": "Michael"})
    assert tools._first_name_matches(["ja"], {"first_name": "Jameson"})
    assert tools._first_name_matches(["cas"], {"first_name": "Casey"})
    assert not tools._first_name_matches(["mi"], {"first_name": "Mecus"})
    assert not tools._first_name_matches(["mi"], {"first_name": "Michael"}, initial=False)
    assert not tools._first_name_matches(["mi", "x"], {"first_name": "Mecus"})  # two-token head


def test_suffix_token_reads_a_generational_suffix_on_both_sides():
    assert tools._suffix_token("Marvin Harrison Jr.") == "jr"
    assert tools._suffix_token("Robert Griffin III") == "iii"
    assert tools._suffix_token("Michael Pittman JR") == "jr"
    assert tools._suffix_token("Justin Jefferson") is None
    assert tools._suffix_token("") is None
    assert tools._name_tokens("Marvin Harrison Jr.") == ["marvin", "harrison"]


def test_resolve_player_prefers_spelled_out_names_over_initial_matches(monkeypatch):
    rows = [
        {"gsis_id": "EXACT", "display_name": "DJ Moore", "first_name": "DJ",
         "football_name": None, "common_first_name": None, "last_name": "Moore",
         "suffix": None, "position": "WR", "latest_team": "BUF",
         "rookie_season": 2022, "last_season": 2026, "pfr_id": None},
        {"gsis_id": "INITIAL", "display_name": "David Moore", "first_name": "David",
         "football_name": None, "common_first_name": None, "last_name": "Moore",
         "suffix": None, "position": "WR", "latest_team": "CAR",
         "rookie_season": 2021, "last_season": 2026, "pfr_id": None},
    ]
    monkeypatch.setattr(tools, "_execute", lambda sql, params=None: list(rows))
    # 'D.J.' is DJ Moore's name spelled out, so David Moore drops out of the candidates.
    assert tools.nfl_resolve_player("D.J. Moore")["resolved"]["gsis_id"] == "EXACT"
    # 'D. Moore' is only an initial, so both men stay in the candidate list.
    assert [c["gsis_id"] for c in tools.nfl_resolve_player("D. Moore")["candidates"]] == [
        "EXACT", "INITIAL"]


def test_label_players_shows_position_and_team_to_separate_same_names():
    rows = [
        {"full_name": "Justin Jefferson", "position": "WR", "team": "MIN"},
        {"full_name": "Justin Jefferson", "position": "LB", "team": "CLE"},
    ]
    assert [r["label"] for r in tools._label_players(rows)] == [
        "Justin Jefferson (WR, MIN)",
        "Justin Jefferson (LB, CLE)",
    ]
    (bare,) = tools._label_players([{"full_name": "Nobody Here"}], team_key="posteam")
    assert bare["label"] == "Nobody Here"


def test_resolve_player_reports_ambiguity_instead_of_merging(monkeypatch):
    def fake_execute(sql, params=None):
        return [
            {"gsis_id": "A", "display_name": "Justin Jefferson", "first_name": "Justin",
             "football_name": "Justin", "common_first_name": None, "last_name": "Jefferson",
             "suffix": None, "position": "WR", "latest_team": "MIN",
             "rookie_season": 2020, "last_season": 2026, "pfr_id": "JeffJu00"},
            {"gsis_id": "B", "display_name": "Justin Jefferson", "first_name": "Justin",
             "football_name": "Justin", "common_first_name": None, "last_name": "Jefferson",
             "suffix": None, "position": "LB", "latest_team": "CLE",
             "rookie_season": 2026, "last_season": 2026, "pfr_id": "JeffJu01"},
        ]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_resolve_player("Justin Jefferson")
    assert result["ambiguous"] is True
    assert result["count"] == 2
    assert [c["label"] for c in result["candidates"]] == [
        "Justin Jefferson (WR, MIN)", "Justin Jefferson (LB, CLE)",
    ]
    assert [c["first_season"] for c in result["candidates"]] == [2020, 2026]


def test_resolve_player_builds_id_name_and_filter_sql(monkeypatch):
    captured = []

    def fake_execute(sql, params=None):
        captured.append((sql, params))
        return [{"gsis_id": "A", "display_name": "Justin Jefferson", "first_name": "Justin",
                 "football_name": None, "common_first_name": None, "last_name": "Jefferson",
                 "suffix": "III", "position": "WR", "latest_team": "MIN",
                 "rookie_season": 2020, "last_season": 2026, "pfr_id": "JeffJu00"}]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    resolved = tools.nfl_resolve_player(
        "ignored", player_id="00-0036322",
    )["resolved"]
    assert "gsis_id = ?" in captured[0][0]
    assert captured[0][1] == ["00-0036322"]      # the id wins, the name is ignored
    assert resolved["full_name"] == "Justin Jefferson III"   # suffix stays in the name
    assert resolved["label"] == "Justin Jefferson III (WR, MIN)"

    tools.nfl_resolve_player(
        "Justin Jefferson", team="min", season=2026, season_from=2020,
        season_to=2026, position="wr",
    )
    sql, params = captured[1]
    assert "rookie_season <= ? AND last_season >= ?" in sql
    assert "last_season >= ?" in sql and "rookie_season <= ?" in sql
    assert "position ILIKE ?" in sql
    assert "SELECT gsis_id FROM rosters WHERE team IN (?) AND season = ?" in sql
    assert "SELECT gsis_id FROM rosters_weekly WHERE team IN (?) AND season = ?" in sql
    assert params == ["jefferson", 2026, 2026, 2020, 2026, "WR",
                      "MIN", "MIN", 2026, "MIN", 2026]

    tools.nfl_resolve_player("Brock Bowers", team="oak", season=2024)   # a club that moved
    sql, params = captured[2]
    assert "COALESCE(latest_team, '') IN (?, ?)" in sql
    assert "SELECT gsis_id FROM rosters WHERE team IN (?, ?) AND season = ?" in sql
    assert "SELECT gsis_id FROM rosters_weekly WHERE team IN (?, ?) AND season = ?" in sql
    assert params == ["bowers", 2024, 2024, "OAK", "LV", "OAK", "LV", 2024, "OAK", "LV", 2024]


def test_team_filter_binds_every_code_of_a_moved_club():
    conditions: list[str] = []
    params: list = []
    tools._team_filter(conditions, params, "posteam", "oak")   # a club that moved
    tools._team_filter(conditions, params, "team", "KC")       # one code, the plain form
    tools._team_filter(conditions, params, "defteam", "   ")   # spaces filter nothing
    assert conditions == ["posteam IN (?, ?)", "team = ?"]
    assert params == ["OAK", "LV", "KC"]
    assert "OAK" in tools._team_codes("LV")        # the era direction, which fix 1 added
    assert tools._team_codes("LA") == ["LA", "STL", "LAR", "RAM"]


@pytest.mark.parametrize("typed, codes", [
    ("LV", ["LV", "OAK"]), ("OAK", ["OAK", "LV"]), ("SD", ["SD", "LAC"]), ("LA", ["LA", "STL", "LAR", "RAM"]),
    ("WAS", ["WAS", "WSH", "TFN"]), ("KC", ["KC"]), ("HOU", ["HOU"]), ("zzz", ["ZZZ"]),
])
def test_team_codes_list_the_typed_code_then_the_current_one_then_the_aliases(typed, codes):
    assert tools._team_codes(typed) == codes


def test_four_codes_get_four_placeholders(monkeypatch):
    """A hardcoded two-mark list would be a parameter-count crash with 'LA'."""
    captured = []
    monkeypatch.setattr(tools, "_execute",
                        lambda sql, params=None: captured.append((sql, params)) or [])
    conditions: list[str] = []
    params: list = []
    tools._team_filter(conditions, params, "team", "LA")
    assert conditions == ["team IN (?, ?, ?, ?)"]
    tools.nfl_search_plays(team="LA", season=2024, max_rows=1)
    sql, bound = captured[0]
    assert sql.count("?") == len(bound)                        # marks and values agree
    assert bound[:4] == ["LA", "STL", "LAR", "RAM"]


def test_resolve_player_needs_a_name_or_id():
    assert tools.nfl_resolve_player()["not_found"] is True
    assert tools.nfl_resolve_player("  ")["did_you_mean"] == []


def test_resolve_player_reports_nearest_names_when_nothing_matches(monkeypatch):
    calls = []

    def fake_execute(sql, params=None):
        calls.append((sql, params))
        if "SELECT display_name FROM players" in sql:
            return [{"display_name": "Justin Jefferson"}]
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_resolve_player("Justin Jeffersonn")
    assert result["not_found"] is True
    assert result["did_you_mean"] == ["justin jefferson"]   # _nearest lowercases
    assert calls[-1][1] == ["jeff%"]      # the surname prefix feeds the suggestion search


def test_resolve_player_swallows_lookup_errors(monkeypatch):
    monkeypatch.setattr(
        tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("lookup timeout"))
    )
    assert tools.nfl_resolve_player("Justin Jefferson")["error"] == "lookup timeout"


RESOLVED_JEFFERSON = {
    "gsis_id": "00-0036322",
    "full_name": "Justin Jefferson",
    "pfr_id": "JeffJu00",
    "position": "WR",
    "team": "MIN",
    "first_season": 2020,
    "last_season": 2026,
    "label": "Justin Jefferson (WR, MIN)",
}


def stub_resolver(monkeypatch, identity=None):
    """Let every name lookup succeed without a database, so these tests read SQL only."""
    player = identity or RESOLVED_JEFFERSON
    monkeypatch.setattr(tools, "nfl_resolve_player", lambda *a, **k: {"resolved": player})
    return player


def test_search_plays_uses_parameterized_filters(monkeypatch):
    captured = {}

    def fake_execute(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    stub_resolver(monkeypatch)
    result = tools.nfl_search_plays(team="KC", player="Mahomes", season=2024, max_rows=5)

    assert result["row_count"] == 0
    assert "posteam = ?" in captured["sql"]
    assert "passer_player_id = ? OR rusher_player_id = ? OR receiver_player_id = ?" in captured["sql"]
    assert "ILIKE" not in captured["sql"]
    assert "LIMIT 6" in captured["sql"]          # one row more finds out if more exist
    assert captured["params"][0] == "KC"
    assert captured["params"][1:4] == ["00-0036322"] * 3
    assert 2024 in captured["params"]
    assert result["player"]["gsis_id"] == "00-0036322"


def _data_reads(calls):
    """The queries that read data, without the information_schema lookups the team tools run
    first to catch a database whose derived team tables predate the season_type column."""
    return [(sql, params) for sql, params in calls if "information_schema" not in sql]


def test_team_stats_uses_parameterized_team_and_season(monkeypatch):
    calls = []

    def fake_execute(sql, params=None):
        calls.append((sql, params))
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    tools.nfl_team_stats(team="kc", season=2024, side="both")

    # One column lookup per table read, then the three data queries.
    assert len(calls) == 6
    reads = _data_reads(calls)
    assert len(reads) == 3
    assert all("team = ?" in sql for sql, _ in reads)
    assert all(params[0] == "KC" for _, params in reads)
    assert all(2024 in params for _, params in reads)


def player_stats_column_rows(drop=()):
    """Answer the tool's information_schema lookup with every player_stats column the tool
    knows how to sum, less the ones this test wants missing."""
    cols = {col for metrics in tools._PLAYER_STATS_METRICS.values()
            for formula in metrics.values()
            for col in tools._STAT_SOURCE_COLUMNS.findall(formula)}
    return [{"column_name": col} for col in sorted(cols - set(drop))]


def fake_player_stats_execute(calls, rows, *, drop=(), cpoe_rows=()):
    """Record every query, answer the column lookup, hand back `rows` for the sum and
    `cpoe_rows` for the one plays query."""
    def _fake(sql, params=None):
        calls.append((sql, params))
        if "information_schema.columns" in sql:
            return player_stats_column_rows(drop=drop)
        return list(cpoe_rows) if "FROM plays" in sql else rows
    return _fake


def test_player_stats_filters_by_resolved_id(monkeypatch):
    calls = []

    monkeypatch.setattr(tools, "_execute", fake_player_stats_execute(calls, []))
    stub_resolver(monkeypatch)
    tools.nfl_player_stats(player_name="P.Mahomes", stat_type="passing", season_type="REG")

    assert "information_schema.columns" in calls[0][0]      # the column check runs first
    sql, params = calls[1]                                   # the player_stats sum
    assert "FROM player_stats" in sql
    assert "play_type" not in sql                           # no play counting left in the sums
    assert "player_id = ?" in sql
    assert "ILIKE" not in sql
    assert params[0] == "00-0036322"
    assert params[-1] == "REG"
    assert "GROUP BY player_id, season, season_type" in sql


def test_player_stats_sums_every_metric_of_the_stat_type(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_execute", fake_player_stats_execute(calls, []))
    stub_resolver(monkeypatch)
    tools.nfl_player_stats(player_name="P.Mahomes", stat_type="passing")

    sql = calls[1][0]
    for key, formula in tools._PLAYER_STATS_METRICS["passing"].items():
        assert f"{formula} AS {key}" in sql
    assert "COUNT(*) AS attempts" not in sql               # sacks are no longer attempts


def test_passing_avg_cpoe_asks_plays_for_the_callers_own_window(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_execute", fake_player_stats_execute(
        calls, [{"season": 2024, "season_type": "REG", "attempts": 581}],
        cpoe_rows=[{"season": 2024, "season_type": "REG", "avg_cpoe": 2.563, "cpoe_plays": 553}]))
    stub_resolver(monkeypatch)
    result = tools.nfl_player_stats(player_name="P.Mahomes", season_from=2020, season_to=2024,
                                    season_type="REG", stat_type="passing")

    cpoe_sql, cpoe_params = calls[-1]
    assert "FROM plays" in cpoe_sql and "AVG(cpoe)" in cpoe_sql and "cpoe IS NOT NULL" in cpoe_sql
    assert "passer_player_id = ?" in cpoe_sql and "postseason" not in cpoe_sql
    assert all(clause in cpoe_sql for clause in ("season >= ?", "season <= ?", "season_type = ?"))
    assert cpoe_params == calls[1][1]        # the same window as the summed row, exactly
    assert (result["metric_sources"]["avg_cpoe"], result["metric_sources"]["cpoe_plays"]) == (
        tools._CPOE_MEAN_SOURCE, tools._CPOE_COUNT_SOURCE)
    assert (result["seasons"][0]["avg_cpoe"], result["seasons"][0]["cpoe_plays"]) == (2.563, 553)


def test_player_stats_names_the_source_of_every_metric_it_returns(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_execute", fake_player_stats_execute(
        calls, [{"player_id": "x", "season": 2024, "season_type": "REG", "carries": 58}]))
    stub_resolver(monkeypatch)
    result = tools.nfl_player_stats("P.Mahomes", season=2024, stat_type="rushing")

    assert result["source"] == tools._PLAYER_STATS_SOURCE
    assert result["unavailable"] == {}                     # this database has every column
    assert len(calls) == 2 and "avg_cpoe" not in result["metric_sources"]  # no plays query
    sql = calls[-1][0]
    assert sql.count(" AS ") == len(result["metric_sources"])   # nothing asked for goes unnamed
    assert all(f"{formula} AS {key}" in sql
               for key, formula in result["metric_sources"].items())
    assert all("player_stats." in src for src in result["metric_sources"].values())


@pytest.mark.parametrize("dropped,missing", [
    (["sack_yards_lost"], {"sack_yards", "net_yards", "net_yards_per_attempt"}),
    (["sacks_suffered"], {"sacks", "sack_rate", "avg_epa", "net_yards_per_attempt"}),
])
def test_a_dropped_metric_is_named_with_the_column_it_misses(monkeypatch, dropped, missing):
    calls = []
    monkeypatch.setattr(tools, "_execute", fake_player_stats_execute(
        calls, [{"season": 2024}], drop=dropped))
    stub_resolver(monkeypatch)
    result = tools.nfl_player_stats("P.Mahomes", stat_type="passing")

    assert set(result["unavailable"]) == missing
    assert all(result["unavailable"][key] == dropped for key in missing)
    assert not set(result["metric_sources"]) & missing       # nothing kept that cannot be summed
    assert all(col not in calls[1][0] for col in dropped)    # and never asked for


def test_every_player_stat_metric_names_a_player_stats_column_and_divides_sums():
    """Two rules hold for every metric of every stat type: the text names at least one real
    player_stats column, and no weekly rate is averaged - a rate divides SUM() totals."""
    for stat_type, metrics in tools._PLAYER_STATS_METRICS.items():
        for key, formula in metrics.items():
            assert tools._STAT_SOURCE_COLUMNS.findall(formula), f"{stat_type}.{key} names no column"
            if "/" in formula:
                assert "SUM(" in formula.split("/", 1)[1], f"{stat_type}.{key} divides by a raw column"
    assert not [f for metrics in tools._PLAYER_STATS_METRICS.values() for f in metrics.values()
                if "AVG(" in f]
    assert not [f for metrics in tools._PLAYER_STATS_METRICS.values() for f in metrics.values()
                if "COUNT(*)" in f]


def test_player_stats_reports_a_table_with_none_of_its_columns(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda sql, params=None: [])
    stub_resolver(monkeypatch)
    assert "player_stats has none of the columns" in tools.nfl_player_stats("P.Mahomes")["error"]


def test_player_stats_labels_rows_with_the_resolved_player(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_execute", fake_player_stats_execute(calls, [{"season": 2024, "receptions": 8}]))
    stub_resolver(monkeypatch)
    result = tools.nfl_player_stats("Justin Jefferson", stat_type="receiving")
    assert result["player"]["label"] == "Justin Jefferson (WR, MIN)"
    assert result["seasons"][0]["label"] == "Justin Jefferson (WR, MIN)"


def test_player_stats_returns_the_resolver_verbatim(monkeypatch):
    monkeypatch.setattr(tools, "nfl_resolve_player", lambda *a, **k: {"ambiguous": True, "candidates": []})
    result = tools.nfl_player_stats("Justin Jefferson", season=2026)
    assert result == {"ambiguous": True, "candidates": []}


def test_compare_team_binds_the_code_and_refuses_a_non_code(monkeypatch):
    calls = []

    def fake_execute(sql, params=None):
        calls.append((sql, params))
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    result = tools.nfl_compare(entity1="KC'; DROP TABLE plays; --", entity2="BAL",
                               compare_type="team", season=2024)

    assert calls == []                              # no franchise: no query runs at all
    assert "Unknown team" in result["error"]
    assert "DROP TABLE" not in str(calls)           # the text never reached any SQL

    tools.nfl_compare(entity1="oak", entity2="BAL", compare_type="team", season=2024)
    reads = _data_reads(calls)
    assert len(calls) == 6                          # 2 staleness lookups, then 4 data reads
    assert len(reads) == 4
    assert "team IN (?, ?)" in reads[0][0]
    assert reads[0][1][:2] == ["OAK", "LV"]         # the typed code binds, with its alias
    assert "team = ?" in reads[2][0]                # a club that never moved needs one mark
    assert reads[2][1] == ["BAL", 2024]


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
        "_execute",
        lambda sql, params=None, row_limit=None: [{"x": 1}, {"x": 2}, {"x": 3}],
    )
    result = tools.nfl_query("SELECT 1 AS x", max_rows=2)
    assert result["row_count"] == 2
    assert result["truncated"] is True


def test_nfl_query_returns_error_when_execute_fails(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad sql")))
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

    def fake_execute(sql, params=None, row_limit=None):
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
    def fake_execute(sql, params=None, row_limit=None):
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
    assert "LIMIT 501" in captured["sql"]      # max_rows is capped at _MAX_ROWS
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
    reads = _data_reads(calls)
    assert len(reads) == 1
    assert "season_year = ?" not in reads[0][0]
    assert reads[0][1] == ["KC"]


def test_team_stats_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timeout")))
    result = tools.nfl_team_stats(team="KC")
    assert "error" in result


@pytest.mark.parametrize(
    "stat_type,expected_sum",
    [("rushing", "SUM(player_stats.carries) AS carries"),
     ("receiving", "SUM(player_stats.targets) AS targets")],
)
def test_player_stats_other_stat_types(monkeypatch, stat_type, expected_sum):
    calls = []

    monkeypatch.setattr(tools, "_execute", fake_player_stats_execute(
        calls, [{"season": 2024, "season_type": "REG"}]))
    stub_resolver(monkeypatch)
    result = tools.nfl_player_stats(
        player_name="P.Mahomes",
        stat_type=stat_type,
        season_from=2020,
        season_to=2024,
    )
    assert result["stat_type"] == stat_type
    assert expected_sum in calls[-1][0]
    assert "player_id = ?" in calls[-1][0]
    assert calls[-1][1][0] == "00-0036322"
    assert "season >=" in calls[-1][0]
    assert "season <=" in calls[-1][0]
    assert 2020 in calls[-1][1] and 2024 in calls[-1][1]


def test_player_stats_rejects_unknown_stat_type():
    result = tools.nfl_player_stats(player_name="P.Mahomes", stat_type="kicking")
    assert "Unknown stat_type" in result["error"]


def test_player_stats_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timeout")))
    result = tools.nfl_player_stats(player_name="P.Mahomes", stat_type="passing")
    assert "error" in result


def test_player_stats_reports_failure_of_its_own_query(monkeypatch):
    stub_resolver(monkeypatch)
    monkeypatch.setattr(
        tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("stats timeout"))
    )
    assert tools.nfl_player_stats("P.Mahomes")["error"] == "stats timeout"


def test_only_a_non_zero_counting_stat_keeps_a_row(monkeypatch):
    """Josh Allen 2024 REG receiving: 0 targets, 7 yards - a stat. Mahomes 2024 POST receiving:
    3 games, nothing else, and Curtis Samuel's playoff row with only a derived epa figure are not.
    Neither `games`, a rate, nor `epa` keeps a row."""
    calls = []
    monkeypatch.setattr(tools, "_execute", fake_player_stats_execute(calls, [
        {"season": 2024, "season_type": "REG", "targets": 0, "yards": 7, "catch_pct": 0.0},
        {"season": 2024, "season_type": "POST", "games": 3, "targets": 0, "yards": 0},
        {"season": 2024, "season_type": "OT", "games": 1, "targets": 0, "yards": 0, "epa": -0.947}]))
    stub_resolver(monkeypatch)
    result = tools.nfl_player_stats("Josh Allen", season=2024, stat_type="receiving")
    assert [(row["season_type"], row["yards"]) for row in result["seasons"]] == [("REG", 7)]


def test_the_keep_set_is_pinned_for_every_stat_type():
    """`_counting_keys` reads the formula text, so a formula rewritten with a space, a COALESCE or
    a ROUND would leave this set in silence and hide a row holding only that stat. These are the
    keys, in metric-map order; change them on purpose, never by accident."""
    assert {s: tools._counting_keys(m) for s, m in tools._PLAYER_STATS_METRICS.items()} == {
        "passing": ["attempts", "completions", "yards", "touchdowns", "interceptions", "sacks", "sack_yards", "air_yards", "yards_after_catch", "first_downs", "pass_10_plus", "pass_20_plus", "pass_40_plus", "two_point_conversions"],
        "rushing": ["carries", "yards", "touchdowns", "fumbles", "fumbles_lost", "first_downs", "rush_10_plus", "rush_20_plus", "rush_40_plus", "two_point_conversions"],
        "receiving": ["targets", "receptions", "yards", "touchdowns", "yards_after_catch", "air_yards", "fumbles", "fumbles_lost", "first_downs", "rec_10_plus", "rec_20_plus", "rec_40_plus", "two_point_conversions"],
    }


def test_compare_player_blocks_come_from_one_stats_call_per_stat_type(monkeypatch):
    stats_calls = []

    def fake_player_stats(**kwargs):
        stats_calls.append(kwargs)
        stype = kwargs["stat_type"]
        first = tools._counting_keys(tools._PLAYER_STATS_METRICS[stype])[0]
        rows = [] if stype == "rushing" else [{first: 5}]   # no rows back, no block
        return {"seasons": rows, "metric_sources": {first: f"SUM(player_stats.{first})"}}

    monkeypatch.setattr(tools, "_execute", lambda *a, **k: pytest.fail(
        "nfl_compare must not count plays for itself any more"))
    monkeypatch.setattr(
        tools,
        "nfl_resolve_player",
        lambda name, **kwargs: {"resolved": {"gsis_id": f"id-{name}", "label": name}},
    )
    monkeypatch.setattr(tools, "nfl_player_stats", fake_player_stats)
    result = tools.nfl_compare(
        entity1="P.Mahomes",
        entity2="J.Allen",
        compare_type="player",
        season=2024,
        season_type="REG",
    )
    sides = result["players"]
    assert [s["requested"] for s in sides] == ["P.Mahomes", "J.Allen"]
    assert "passing" in sides[0] and "receiving" in sides[0]
    assert "rushing" not in sides[0]                  # no rows back from nfl_player_stats, no block
    assert sides[0]["source"] == tools._PLAYER_STATS_SOURCE
    assert sides[0]["metric_sources"]["passing"] == {"attempts": "SUM(player_stats.attempts)"}
    assert [c["season_type"] for c in stats_calls] == ["REG"] * 6   # never mixed, never dropped
    assert [c["stat_type"] for c in stats_calls] == [
        "passing", "rushing", "receiving"] * 2
    assert {c["player_id"] for c in stats_calls} == {"id-P.Mahomes", "id-J.Allen"}
    assert all("season" in c and c["season"] == 2024 for c in stats_calls)


def test_compare_player_reports_a_stats_error_instead_of_an_empty_side(monkeypatch):
    monkeypatch.setattr(
        tools, "nfl_resolve_player",
        lambda name, **kwargs: {"resolved": {"gsis_id": f"id-{name}", "label": name}},
    )
    monkeypatch.setattr(tools, "nfl_player_stats", lambda **kwargs: {"error": "stats unavailable"})
    assert tools.nfl_compare("A", "B", compare_type="player") == {"error": "stats unavailable"}


def test_compare_player_reports_the_resolver_result_per_entity(monkeypatch):
    monkeypatch.setattr(
        tools, "_execute",
        lambda sql, params=None: player_stats_column_rows() if "information_schema" in sql else [],
    )
    monkeypatch.setattr(
        tools,
        "nfl_resolve_player",
        lambda name, **kwargs: (
            {"ambiguous": True, "count": 2, "candidates": []}
            if name == "Justin Jefferson" else {"resolved": {"gsis_id": "B", "label": "B"}}
        ),
    )
    result = tools.nfl_compare("Justin Jefferson", "Travis Kelce", compare_type="player")
    assert [(s["requested"], "ambiguous" in s, s.get("player", {}).get("gsis_id"))
            for s in result["players"]] == [("Justin Jefferson", True, None), ("Travis Kelce", False, "B")]
    assert "source" not in result["players"][1]      # a side with no stats claims no source


def test_compare_player_keeps_both_sides_when_the_names_are_identical(monkeypatch):
    """Two players who share a name must not overwrite each other in the result."""
    monkeypatch.setattr(tools, "_execute", lambda sql, params=None: [])
    monkeypatch.setattr(
        tools,
        "nfl_resolve_player",
        lambda name, **kwargs: {"resolved": {"gsis_id": kwargs["player_id"], "label": name}},
    )
    monkeypatch.setattr(tools, "nfl_player_stats", lambda **kwargs: {"seasons": []})
    result = tools.nfl_compare(
        "Justin Jefferson", "Justin Jefferson", compare_type="player",
        player_id1="00-0036322", player_id2="00-0041075", season=2026,
    )
    assert [s["player"]["gsis_id"] for s in result["players"]] == ["00-0036322", "00-0041075"]
    assert all(s["requested"] == "Justin Jefferson" for s in result["players"])
    assert all("source" not in s and "metric_sources" not in s for s in result["players"])


def test_compare_rejects_unknown_type():
    result = tools.nfl_compare(entity1="KC", entity2="BAL", compare_type="invalid")
    assert "compare_type must be" in result["error"]


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
    reads = _data_reads(calls)
    assert len(calls) == 6
    assert len(reads) == 4
    assert "season_year >= ?" in reads[0][0]
    assert "season_year <= ?" in reads[0][0]
    assert reads[0][1] == ["KC", 2020, 2021]


def test_compare_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timeout")))
    result = tools.nfl_compare(entity1="KC", entity2="BAL", compare_type="team", season=2024)
    assert "error" in result


def test_catalog_returns_dataset_summary(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_execute",
        lambda sql, params=None: [{"dataset_id": "pbp", "table_name": "plays", "total_rows": 123}],
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
        return [{"full_name": "Justin Jefferson", "position": "WR", "team": "MIN"}]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    stub_resolver(monkeypatch)
    result = tools.nfl_injuries(
        team="kc",
        season=2024,
        week=1,
        player="Mahomes",
        report_status="Out",
    )
    assert result["count"] == 1
    assert "team = ?" in captured["sql"]
    assert "gsis_id = ?" in captured["sql"]
    assert "report_status ILIKE ?" in captured["sql"]
    assert captured["params"] == ["KC", 2024, 1, "00-0036322", "%Out%"]
    assert result["injuries"][0]["label"] == "Justin Jefferson (WR, MIN)"


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
    assert "(home_team IN (?) OR away_team IN (?))" in captured["sql"]
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
        return [{"player": "J.Jefferson", "position": "WR", "team": "MIN"}]

    monkeypatch.setattr(tools, "_execute", fake_execute)
    stub_resolver(monkeypatch)
    result = tools.nfl_snap_counts(
        player="Kelce",
        team="kc",
        season=2024,
        week=1,
        position="te",
    )
    assert result["count"] == 1
    assert "pfr_player_id = ?" in captured["sql"]
    assert "team = ?" in captured["sql"]
    assert "position ILIKE ?" in captured["sql"]
    assert captured["params"] == ["JeffJu00", "KC", 2024, 1, "TE"]
    assert result["snap_counts"][0]["label"] == "J.Jefferson (WR, MIN)"


def test_snap_counts_without_pfr_id_is_not_found(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda sql, params=None: [{"player": "X"}])
    no_pfr = dict(RESOLVED_JEFFERSON, pfr_id=None)
    stub_resolver(monkeypatch, no_pfr)
    result = tools.nfl_snap_counts(player="Some Rookie")
    assert result["not_found"] is True
    assert "PFR id" in result["reason"]


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
    stub_resolver(monkeypatch)
    result = tools.nfl_ftn_charting(
        team="kc", opponent="phi", player="Mahomes",
        season_from=2022, season_to=2024, week=3, season_type="reg",
    )
    sql, params = captured["sql"], captured["params"]
    assert "p.posteam = ?" in sql and "p.defteam = ?" in sql
    assert "p.passer_player_id = ?" in sql
    assert "p.season >= ?" in sql and "p.season <= ?" in sql
    assert "p.week = ?" in sql and "p.season_type = ?" in sql
    assert params[:2] == ["KC", "PHI"]
    assert params[2:5] == ["00-0036322"] * 3
    assert params[5:7] == [2022, 2024]
    assert params[-2:] == [3, "REG"]
    assert result["charting"]["play_action"]["pct"] == 40.0
    assert result["charting"]["motion"]["plays"] == 5
    assert result["player"]["gsis_id"] == "00-0036322"


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


def test_ftn_charting_echoes_the_player_on_an_empty_result(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda sql, params=None: [])
    stub_resolver(monkeypatch)
    result = tools.nfl_ftn_charting(player="Justin Jefferson")
    assert result["total_plays"] == 0
    assert result["player"]["gsis_id"] == "00-0036322"


def test_ftn_charting_returns_error_on_db_error(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(duckdb.Error("ftn boom")))
    result = tools.nfl_ftn_charting(team="KC")
    assert result["error"] == "ftn boom"


# ── Fantasy derived-table tools ────────────────────────────────────────────────

def _capture_execute(monkeypatch, rows):
    captured = {}

    def fake_execute(sql, params=None):
        if "information_schema.columns WHERE table_name" in sql:
            # The stale-derived-table guard probing what it can read: answer that the table is
            # current so the call under test reaches its own query.
            return [{"column_name": "season_type"}]
        captured["sql"] = sql
        captured["params"] = params
        return rows

    monkeypatch.setattr(tools, "_execute", fake_execute)
    return captured


def test_td_luck_builds_filters(monkeypatch):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    stub_resolver(monkeypatch)
    result = tools.nfl_td_luck(player="Jefferson", position="wr", team="min", season=2024, limit=10)
    assert result["count"] == 1
    assert "player_id = ?" in captured["sql"]
    assert "position ILIKE ?" in captured["sql"]
    assert "team = ?" in captured["sql"]
    assert "total_td_luck_score ASC" in captured["sql"]
    assert captured["params"] == ["00-0036322", "WR", "MIN", 2024]


def test_td_luck_invalid_limit_falls_back(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    tools.nfl_td_luck(limit="oops")
    assert "LIMIT 51" in captured["sql"]           # the default 50, plus the probe row
    assert captured["params"] is None


def test_td_luck_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("td timeout")))
    assert tools.nfl_td_luck()["error"] == "td timeout"


def test_role_trend_builds_filters(monkeypatch):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    stub_resolver(monkeypatch)
    tools.nfl_role_trend(player="Kelce", position="te", team="kc", season=2024, week=8, min_snap_pct=50)
    assert "player_id = ?" in captured["sql"]
    assert "snap_pct >= ?" in captured["sql"]
    assert "snap_pct_delta DESC" in captured["sql"]
    assert captured["params"] == ["00-0036322", "TE", "KC", 2024, 8, 50.0]


def test_role_trend_invalid_limit_falls_back(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    tools.nfl_role_trend(limit="oops")
    assert "LIMIT 51" in captured["sql"]           # the default 50, plus the probe row


def test_role_trend_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("role timeout")))
    assert tools.nfl_role_trend()["error"] == "role timeout"


def test_separation_opportunity_builds_filters(monkeypatch):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    stub_resolver(monkeypatch)
    tools.nfl_separation_opportunity(player="Aiyuk", position="wr", team="sf", season=2024, regression_candidate=True)
    assert "player_id = ?" in captured["sql"]
    assert "regression_candidate = ?" in captured["sql"]
    assert "fp_diff_per_game ASC" in captured["sql"]
    # season_type defaults to REG, so the playoff rows do not crowd the list.
    assert "season_type = ?" in captured["sql"]
    assert captured["params"] == ["00-0036322", "WR", "SF", 2024, "REG", True]


def test_separation_opportunity_names_the_sample_behind_a_fallback_row(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    stub_resolver(monkeypatch)
    tools.nfl_separation_opportunity()
    assert "fallback_weeks" in captured["sql"] and "fallback_targets" in captured["sql"]


def test_separation_opportunity_says_what_to_rebuild(monkeypatch):
    """An old player_separation_opportunity has no season_type; the caller must hear the rebuild
    command, not DuckDB's binder error for a column they never typed."""
    monkeypatch.setattr(tools, "_execute", lambda sql, params=None: [{"column_name": "season"}])
    stub_resolver(monkeypatch)
    assert tools.nfl_separation_opportunity() == {
        "error": "player_separation_opportunity has no season_type column. "
                 "Rebuild the derived tables: nfl-mcp ingest --dataset teams"}


def test_separation_opportunity_guards_pre_2016():
    result = tools.nfl_separation_opportunity(season=2015)
    assert "2016" in result["error"]


def test_separation_opportunity_invalid_limit_falls_back(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    tools.nfl_separation_opportunity(limit="oops")
    assert "LIMIT 51" in captured["sql"]           # the default 50, plus the probe row


def test_separation_opportunity_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("sep timeout")))
    assert tools.nfl_separation_opportunity()["error"] == "sep timeout"


def test_drop_rate_builds_filters(monkeypatch):
    captured = _capture_execute(monkeypatch, [{"player": "T.Pollard"}])
    stub_resolver(monkeypatch)
    tools.nfl_drop_rate(player="Pollard", team="ten", season=2024, min_targets=30)
    assert "player_id = ?" in captured["sql"]
    assert "catchable_targets >= ?" in captured["sql"]
    assert "drop_rate_pct DESC" in captured["sql"]
    assert captured["params"] == ["00-0036322", "TEN", 2024, 30]


def test_drop_rate_guards_pre_2022():
    result = tools.nfl_drop_rate(season=2021)
    assert "2022" in result["error"]


def test_drop_rate_invalid_limit_falls_back(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    tools.nfl_drop_rate(limit="oops")
    assert "LIMIT 51" in captured["sql"]           # the default 50, plus the probe row


def test_drop_rate_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("drop timeout")))
    assert tools.nfl_drop_rate()["error"] == "drop timeout"


def test_contract_value_builds_filters(monkeypatch):
    captured = _capture_execute(monkeypatch, [{"full_name": "X"}])
    stub_resolver(monkeypatch)
    tools.nfl_contract_value(player="Gibbs", position="rb", team="det", season=2024, min_apy=2, max_apy=10)
    assert "player_id = ?" in captured["sql"]
    assert "apy >= ?" in captured["sql"]
    assert "apy <= ?" in captured["sql"]
    assert "fp_per_million DESC" in captured["sql"]
    assert captured["params"] == ["00-0036322", "RB", "DET", 2024, 2.0, 10.0]


def test_contract_value_invalid_limit_falls_back(monkeypatch):
    captured = _capture_execute(monkeypatch, [])
    tools.nfl_contract_value(limit="oops")
    assert "LIMIT 51" in captured["sql"]           # the default 50, plus the probe row


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
    assert "LIMIT 51" in captured["sql"]           # the default 50, plus the probe row
    assert captured["params"] is None


def test_injury_return_returns_error_on_timeout(monkeypatch):
    monkeypatch.setattr(tools, "_execute", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("injury timeout")))
    assert tools.nfl_injury_return()["error"] == "injury timeout"
