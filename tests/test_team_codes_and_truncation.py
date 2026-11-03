"""GROUP C: correct season-type filters, one team code per franchise, and truncation flags.

The unit classes build SQL against a recorded `_execute` and need no database. The
integration classes run against the loaded DuckDB and skip without one.

One fact the integration classes cannot prove locally: the shared database holds
`snap_counts`, `plays`, `rosters`, `schedules` and the stats tables for 2024 onward, so it
contains no `OAK`, `SD` or `STL` rows at all. The era direction is therefore proven against
a database this file builds itself (TestEraCodesOnTheirOwnDatabase), which writes 2019 rows
under the old code.
"""

import json

import duckdb
import pytest

from nfl_mcp import server, tools
from nfl_mcp.franchises import canonical_team
from nfl_mcp.tools import (
    nfl_compare,
    nfl_contract_value,
    nfl_drop_rate,
    nfl_fantasy_opportunity,
    nfl_fantasy_rankings,
    nfl_ftn_charting,
    nfl_injuries,
    nfl_injury_return,
    nfl_player_stats,
    nfl_query,
    nfl_resolve_player,
    nfl_role_trend,
    nfl_roster,
    nfl_schedule,
    nfl_schema,
    nfl_search_plays,
    nfl_separation_opportunity,
    nfl_snap_counts,
    nfl_td_luck,
    nfl_team_records,
    nfl_team_stats,
)


@pytest.fixture
def require_db(rebuilt_db):
    """Skip integration tests if no data is loaded.

    Depends on rebuilt_db, so the tools read the copy whose derived tables the code under test
    rebuilt. The shared file still holds the shape the ingest produced before those tables gained
    a season_type column, and a tool that reads it now answers with a rebuild message instead of
    rows.
    """
    result = nfl_query("SELECT COUNT(*) AS n FROM plays")
    if "error" in result:
        pytest.skip("No DuckDB database loaded")
    if result["rows"][0]["n"] == 0:
        pytest.skip("plays table is empty")


@pytest.fixture
def recorded(monkeypatch):
    """Record every query a tool builds and answer it with no rows."""
    calls: list[tuple[str, list]] = []

    def fake_execute(sql, params=None):
        calls.append((sql, list(params or [])))
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    return calls


# ── fix 1: one season_type parameter, two table vocabularies ───────────────────

# The four playoff rounds `schedules.game_type` names, in order of play.
PLAYOFF_ROUNDS = ["WC", "DIV", "CON", "SB"]


def _record_queries(monkeypatch):
    """Record every query a tool builds, answer no rows, and keep two lookups honest.

    nfl_player_stats checks which columns this database has before it filters, so that lookup
    gets one column back. The two reads of the derived-shape guard (nfl_mcp.tools
    ._stale_derived_shape) get 'nothing here is old-shaped' so they never answer for the tool.
    """
    calls: list[tuple[str, list]] = []

    def fake_execute(sql, params=None):
        calls.append((sql, list(params or [])))
        if "SELECT column_name FROM information_schema.columns" in sql:
            return [{"column_name": "attempts"}]
        return []

    monkeypatch.setattr(tools, "_execute", fake_execute)
    return calls


class TestSeasonTypeVocabulary:
    """`schedules.game_type` names the round; `plays` and `player_stats` say REG or POST."""

    def test_postseason_on_the_schedules_column_lists_the_four_rounds(self, recorded):
        result = nfl_schedule(team="KC", season=2024, season_type="POST")
        assert result["count"] == 0
        sql, params = recorded[0]
        assert "game_type IN (?, ?, ?, ?)" in sql
        assert params == ["KC", "KC", 2024, "WC", "DIV", "CON", "SB"]

    @pytest.mark.parametrize("round_code", PLAYOFF_ROUNDS)
    def test_one_round_filters_the_schedules_column_by_itself(self, recorded, round_code):
        nfl_schedule(team="KC", season_type=round_code.lower())
        sql, params = recorded[0]
        assert "game_type = ?" in sql
        assert params[-1] == round_code

    @pytest.mark.parametrize("call", [nfl_search_plays, nfl_ftn_charting])
    @pytest.mark.parametrize("round_code", PLAYOFF_ROUNDS)
    def test_a_round_name_on_a_two_word_table_is_refused_by_name(self, recorded, call,
                                                                 round_code):
        # What this refusal replaces: a round name was widened to 'POST', so the wild card and
        # the Super Bowl each answered with every postseason row the filter could reach - one
        # answer for two different questions, with nothing to tell the caller they differed.
        result = call(team="KC", season=2024, season_type=round_code.lower())
        assert f"'{round_code}'" in result["error"]
        assert "only 'REG' or 'POST'" in result["error"]
        assert "nfl_schedule" in result["hint"]
        assert recorded == []            # refused before any query was built

    def test_the_counting_stats_refuse_a_round_name_the_same_way(self, monkeypatch):
        monkeypatch.setattr(tools, "_execute",
                            lambda sql, params=None: [{"column_name": "attempts"}]
                            if "information_schema" in sql else [])
        monkeypatch.setattr(tools, "nfl_resolve_player",
                            lambda *a, **k: {"resolved": {"gsis_id": "00-0033873",
                                                          "label": "Patrick Mahomes (QB, KC)"}})
        result = nfl_player_stats(player_name="Patrick Mahomes", season=2024,
                                  season_type="SB", stat_type="passing")
        assert "'SB'" in result["error"]
        assert "only 'REG' or 'POST'" in result["error"]

    def test_a_player_comparison_refuses_a_round_name_through_the_stats_tool(self, monkeypatch):
        monkeypatch.setattr(tools, "_execute",
                            lambda sql, params=None: [{"column_name": "attempts"}]
                            if "information_schema" in sql else [])
        monkeypatch.setattr(tools, "nfl_resolve_player",
                            lambda *a, **k: {"resolved": {"gsis_id": "00-0033873",
                                                          "label": "Patrick Mahomes (QB, KC)"}})
        result = nfl_compare(entity1="Patrick Mahomes", entity2="Josh Allen",
                             compare_type="player", season=2024, season_type="CON")
        assert "'CON'" in result["error"]
        assert "only 'REG' or 'POST'" in result["error"]

    @pytest.mark.parametrize("call", [
        lambda code: nfl_team_stats(team="KC", season=2024, season_type=code),
        lambda code: nfl_compare(entity1="KC", entity2="BAL", compare_type="team",
                                 season=2024, season_type=code),
    ])
    def test_the_team_tools_refuse_a_round_name_before_they_query(self, recorded, call):
        # Group B made the derived team tables one row per season_type, storing 'REG' and 'POST'
        # like plays does, so 'SB' there answers with no rows at all instead of with the Super
        # Bowl. Refusing it is the same rule the play tools already follow.
        result = call("SB")
        assert "'SB'" in result["error"]
        assert "only 'REG' or 'POST'" in result["error"]
        assert "nfl_schedule" in result["hint"]
        assert recorded == []            # refused before any query was built

    @pytest.mark.parametrize("side", ["REG", "POST"])
    def test_the_team_tools_filter_by_the_two_words_they_store(self, recorded, side):
        result = nfl_team_stats(team="KC", season=2024, season_type=side.lower())
        assert "error" not in result
        assert any(f"season_type = ?" in sql for sql, _ in recorded)
        assert side in [p for _, params in recorded for p in params]

    def test_the_counting_stats_and_the_cpoe_window_ask_the_same_season_type(self, monkeypatch):
        calls = []

        def fake_execute(sql, params=None):
            calls.append((sql, list(params or [])))
            if "information_schema.columns" in sql:
                return [{"column_name": "attempts"}]
            return []

        monkeypatch.setattr(tools, "_execute", fake_execute)
        monkeypatch.setattr(tools, "nfl_resolve_player",
                            lambda *a, **k: {"resolved": {"gsis_id": "00-0033873",
                                                          "label": "Patrick Mahomes (QB, KC)"}})
        nfl_player_stats(player_name="Patrick Mahomes", season=2024, season_type="POST",
                         stat_type="passing")

        asked = {(sql.split("FROM ")[1].split()[0], params[-1]) for sql, params in calls
                 if "season_type" in sql}
        assert asked == {("player_stats", "POST"), ("plays", "POST")}

    def test_regular_season_is_the_same_word_in_both_tables(self, recorded):
        nfl_schedule(team="KC", season_type="reg")
        assert recorded[0][1][-1] == "REG"
        recorded.clear()
        nfl_search_plays(team="KC", season_type="reg")
        assert recorded[0][1][-1] == "REG"

    def test_the_schedule_schema_offers_every_round_the_column_stores(self):
        # A client that cannot name the round cannot ask for a wild-card game at all.
        tool = next(tool for tool in server.TOOLS if tool.name == "nfl_schedule")
        season_type = tool.inputSchema["properties"]["season_type"]
        assert season_type["enum"] == ["REG", "POST", "WC", "DIV", "CON", "SB"]
        assert all(round_code in season_type["description"] for round_code in PLAYOFF_ROUNDS)

    def test_the_play_search_schema_says_where_plays_reach(self):
        # The text said "2013 through the current season"; plays starts in 1999.
        tool = next(tool for tool in server.TOOLS if tool.name == "nfl_search_plays")
        season = tool.inputSchema["properties"]["season"]["description"]
        assert "1999" in season
        assert "2013" not in season

    def test_the_schema_text_a_model_reads_says_where_plays_reach(self):
        # Group C second pass, fix 5: nfl_schema hands this text to the model, and the
        # `season` line still said 2013 while the table starts in 1999.
        game = nfl_schema(category="game_context")["schema"]
        line = next(l for l in game.splitlines() if l.strip().startswith("season "))
        assert "1999" in line
        assert "2013" not in line

    def test_no_tool_offers_a_round_name_the_column_cannot_answer(self):
        # A client that picks a round from the schema must land on the one tool whose
        # column stores it.
        round_tools = {
            tool.name for tool in server.TOOLS
            if tool.inputSchema.get("properties", {}).get("season_type", {}).get("enum")
            and set(tool.inputSchema["properties"]["season_type"]["enum"]) & set(PLAYOFF_ROUNDS)
        }
        assert round_tools == {"nfl_schedule"}
        for tool in server.TOOLS:
            season_type = tool.inputSchema.get("properties", {}).get("season_type")
            if tool.name in ("nfl_search_plays", "nfl_player_stats", "nfl_ftn_charting",
                             "nfl_compare"):
                assert "REG" in season_type["description"], tool.name
                assert "nfl_schedule" in season_type["description"], tool.name

    def test_no_season_type_applies_no_filter(self, recorded):
        for season_type in (None, "   "):
            nfl_schedule(team="KC", season_type=season_type)
            where = recorded[0][0].split(" WHERE ")[1]
            assert "game_type" not in where
            recorded.clear()

    @pytest.mark.parametrize("call,accepted", [
        pytest.param(lambda c: nfl_search_plays(team="KC", season=2024, season_type=c),
                     "REG, POST", id="nfl_search_plays"),
        pytest.param(lambda c: nfl_team_stats(team="KC", season=2024, season_type=c),
                     "REG, POST", id="nfl_team_stats"),
        pytest.param(lambda c: nfl_player_stats(player_name="Patrick Mahomes", season=2024,
                                                season_type=c),
                     "REG, POST", id="nfl_player_stats"),
        pytest.param(lambda c: nfl_compare(entity1="KC", entity2="BAL", compare_type="team",
                                          season=2024, season_type=c),
                     "REG, POST", id="nfl_compare"),
        pytest.param(lambda c: nfl_schedule(team="KC", season=2024, season_type=c),
                     "REG, POST, WC, DIV, CON, SB", id="nfl_schedule"),
        pytest.param(lambda c: nfl_ftn_charting(team="KC", season=2024, season_type=c),
                     "REG, POST", id="nfl_ftn_charting"),
        pytest.param(lambda c: nfl_separation_opportunity(season=2024, season_type=c),
                     "REG, POST, ALL", id="nfl_separation_opportunity"),
    ])
    def test_a_value_outside_the_column_vocabulary_is_refused_not_bound(
            self, call, accepted, monkeypatch):
        """Every tool that takes season_type, on the two values that used to answer empty.

        'ALL' is a real value for nfl_separation_opportunity, so a model carries it to the next
        tool: nfl_schedule(season_type='ALL') returned zero games and nfl_team_stats(season_type=
        'playoffs') returned zero rows. Both bound a word the column never stores, and an empty
        success reads as an empty stat. Each tool must instead name the set it accepts, and skip
        the values that tool does accept.
        """
        calls = _record_queries(monkeypatch)
        monkeypatch.setattr(tools, "nfl_resolve_player",
                            lambda *a, **k: {"resolved": {"gsis_id": "00-0033873",
                                                          "label": "Patrick Mahomes (QB, KC)"}})
        wanted = accepted.split(", ")
        for bad in ("ALL", "playoffs", "pre"):
            if bad.upper() in wanted:
                continue
            result = call(bad)
            assert f"'{bad.upper()}'" in result["error"], (bad, result)
            assert f"Accepted: {accepted}." in result["error"], (bad, result)
            assert "count" not in result, (bad, result)   # not an empty success
        assert not [sql for sql, _ in calls
                    if "season_type = ?" in sql or "game_type = ?" in sql]

    def test_all_still_asks_for_both_scopes_on_the_tool_that_documents_it(self, recorded):
        """'ALL' is that tool's own contract, so it filters nothing rather than erroring."""
        result = nfl_separation_opportunity(season=2024, season_type="ALL")
        assert "error" not in result
        assert not [sql for sql, _ in recorded if "season_type = ?" in sql]

    @pytest.mark.parametrize("name,accepted", [
        pytest.param("nfl_search_plays", tools.SEASON_TYPES, id="nfl_search_plays"),
        pytest.param("nfl_team_stats", tools.SEASON_TYPES, id="nfl_team_stats"),
        pytest.param("nfl_player_stats", tools.SEASON_TYPES, id="nfl_player_stats"),
        pytest.param("nfl_compare", tools.SEASON_TYPES, id="nfl_compare"),
        pytest.param("nfl_ftn_charting", tools.SEASON_TYPES, id="nfl_ftn_charting"),
        pytest.param("nfl_schedule", tools.SCHEDULE_SEASON_TYPES, id="nfl_schedule"),
        pytest.param("nfl_separation_opportunity", tools.SEPARATION_SEASON_TYPES,
                     id="nfl_separation_opportunity"),
    ])
    def test_the_schema_enum_offers_exactly_what_the_tool_accepts(self, name, accepted):
        """A client must not be offered a value the tool refuses, nor refused one it accepts.

        `accepted` is the vocabulary the tool's own filter uses (nfl_mcp.tools
        ._season_type_filter reads the same constants), not a second list typed in here. Change
        one entry in one of those constants and this is the test that reports the schema a model
        reads no longer matches the set the code accepts.
        """
        tool = next(tool for tool in server.TOOLS if tool.name == name)
        assert tool.inputSchema["properties"]["season_type"]["enum"] == list(accepted)

    def test_the_hint_on_the_tool_that_defaults_to_one_scope_says_so(self, monkeypatch):
        """The hint told every caller to leave season_type out for every scope.

        On nfl_separation_opportunity that instruction is wrong: leaving it out gives the regular
        season, and season_type='ALL' gives both. Measured on the full database, the plain call
        answers 198 rows and season_type='ALL' answers 234.
        """
        calls = _record_queries(monkeypatch)
        result = nfl_separation_opportunity(season=2024, season_type="playoffs")
        assert result["hint"] == ("Leaving season_type out gives REG only; pass "
                                  "season_type='ALL' for both scopes as separate rows.")
        assert not [sql for sql, _ in calls if "player_separation_opportunity" in sql]

    def test_the_hint_elsewhere_still_offers_every_scope(self, monkeypatch):
        """The six tools with no scope default keep the old hint, because for them it is true."""
        _record_queries(monkeypatch)
        result = nfl_team_stats(team="KC", season=2024, season_type="playoffs")
        assert result["hint"] == ("Leave season_type out to get every scope this table has, "
                                  "as separate rows.")

    def test_a_blank_value_on_the_tool_that_defaults_answers_that_default(self, recorded):
        """season_type='' and '   ' applied no filter, so the tool answered both scopes while its
        own schema documented REG. A blank now binds the default the tool advertises."""
        for blank in ("", "   "):
            result = nfl_separation_opportunity(season=2024, season_type=blank)
            assert "error" not in result, blank
            sql, params = recorded[-1]
            assert "season_type = ?" in sql, blank
            assert params[-1] == tools.SEPARATION_DEFAULT_SEASON_TYPE, blank
            recorded.clear()

    def test_a_blank_value_elsewhere_still_applies_no_filter(self, recorded):
        """For the tools whose default is no filter, a blank stays no filter."""
        calls = {"nfl_team_stats": lambda st: nfl_team_stats(team="KC", season=2024,
                                                             season_type=st),
                 "nfl_schedule": lambda st: nfl_schedule(team="KC", season=2024, season_type=st)}
        for name, call in calls.items():
            for blank in (None, "", "   "):
                call(blank)
                sql = recorded[-1][0]
                assert "season_type = ?" not in sql, (name, blank)
                assert "game_type" not in sql.split(" WHERE ")[-1], (name, blank)
                recorded.clear()


class TestPlayoffFilterOnRealData:
    """The mismatch this fix removes was verified: season_type='POST' returned no games."""

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    def test_postseason_returns_the_playoff_games_and_nothing_else(self):
        result = nfl_schedule(team="KC", season=2024, season_type="POST")
        assert result["count"] > 0
        assert {row["game_type"] for row in result["games"]} <= set(PLAYOFF_ROUNDS)

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    def test_postseason_is_the_union_of_the_four_rounds(self):
        rounds = set()
        for round_code in PLAYOFF_ROUNDS:
            rounds.update(row["game_id"] for row in
                          nfl_schedule(season=2024, season_type=round_code)["games"])
        post = {row["game_id"] for row in nfl_schedule(season=2024, season_type="POST")["games"]}
        assert post == rounds

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    def test_regular_and_postseason_together_are_the_whole_season(self):
        reg = nfl_schedule(season=2024, season_type="REG")["count"]
        post = nfl_schedule(season=2024, season_type="POST")["count"]
        assert reg + post == nfl_schedule(season=2024)["count"]

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    def test_postseason_games_are_the_ones_the_plays_table_calls_postseason(self):
        games = {row["game_id"] for row in nfl_schedule(season=2024, season_type="POST")["games"]}
        played = nfl_query(
            "SELECT DISTINCT game_id FROM plays WHERE season = 2024 AND season_type = 'POST'")
        assert played["rows"] and games >= {row["game_id"] for row in played["rows"]}

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    def test_a_round_filter_is_a_strict_part_of_postseason(self):
        sb = nfl_schedule(season=2024, season_type="SB")
        assert {row["game_type"] for row in sb["games"]} == {"SB"}
        assert 0 < sb["count"] < nfl_schedule(season=2024, season_type="POST")["count"]


# ── fix 2: one case, whatever the caller types ─────────────────────────────────

# Every tool that takes a team, called the same way twice: once lower-case, once upper.
TEAM_TOOLS = {
    "nfl_search_plays": lambda team: nfl_search_plays(team=team, season=2024),
    "nfl_search_plays_opponent": lambda team: nfl_search_plays(opponent=team, season=2024),
    "nfl_team_stats": lambda team: nfl_team_stats(team=team, season=2024),
    "nfl_ftn_charting": lambda team: nfl_ftn_charting(team=team, season=2024),
    "nfl_ftn_charting_opponent": lambda team: nfl_ftn_charting(opponent=team, season=2024),
    "nfl_roster": lambda team: nfl_roster(team=team, season=2024),
    "nfl_injuries": lambda team: nfl_injuries(team=team, season=2024),
    "nfl_schedule": lambda team: nfl_schedule(team=team, season=2024),
    "nfl_team_records": lambda team: nfl_team_records(team=team, season=2024),
    "nfl_snap_counts": lambda team: nfl_snap_counts(team=team, season=2024),
    "nfl_fantasy_opportunity": lambda team: nfl_fantasy_opportunity(team=team, season=2024),
    "nfl_fantasy_rankings": lambda team: nfl_fantasy_rankings(team=team, limit=5),
    "nfl_td_luck": lambda team: nfl_td_luck(team=team, season=2024),
    "nfl_role_trend": lambda team: nfl_role_trend(team=team, season=2024),
    "nfl_separation_opportunity": lambda team: nfl_separation_opportunity(team=team, season=2024),
    "nfl_drop_rate": lambda team: nfl_drop_rate(team=team, season=2024),
    "nfl_contract_value": lambda team: nfl_contract_value(team=team, season=2024),
    "nfl_compare": lambda team: nfl_compare(entity1=team, entity2="BAL", compare_type="team"),
}

# These name the club they were asked for, so a blank one is a code no franchise uses.
REQUIRED_TEAM = {"nfl_team_stats", "nfl_compare", "nfl_team_records"}


class TestTeamCodeCase:
    @pytest.mark.parametrize("name", sorted(TEAM_TOOLS))
    def test_the_case_of_a_code_changes_nothing_about_the_query(self, recorded, name):
        call = TEAM_TOOLS[name]
        call("kc")
        lower = list(recorded)
        recorded.clear()
        call("KC")
        assert recorded == lower
        assert all("kc" not in str(params) for _, params in lower)   # bound upper-cased

    @pytest.mark.parametrize("name", sorted(TEAM_TOOLS))
    def test_a_code_of_only_spaces_still_applies_no_filter(self, recorded, name):
        result = TEAM_TOOLS[name]("   ")
        if name in REQUIRED_TEAM:      # a tool cannot ask about no club at all
            assert "Unknown team" in result["error"]
        else:
            assert "error" not in result


class TestCaseOnRealData:
    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    @pytest.mark.parametrize("name", sorted(TEAM_TOOLS))
    def test_a_lower_case_code_answers_with_the_same_rows(self, name):
        lower, upper = TEAM_TOOLS[name]("kc"), TEAM_TOOLS[name]("KC")
        if name == "nfl_compare":
            lower, upper = list(lower.values()), list(upper.values())
        assert lower == upper


# ── fix 3: an unknown code is an answer, not an empty result ───────────────────

class TestUnknownTeamCode:
    @pytest.mark.parametrize("name", sorted(TEAM_TOOLS))
    def test_a_code_no_franchise_uses_is_named_back(self, recorded, name):
        result = TEAM_TOOLS[name]("ZZZ")
        # Group C second pass, fix 4: the message names the argument that was wrong, so a
        # caller who typed a bad opponent is not told their team was bad.
        param = "opponent" if name.endswith("_opponent") else "team"
        assert result["error"] == f"Unknown {param} 'ZZZ'."
        assert "current abbreviation" in result["hint"]
        assert recorded == []            # refused before any query was built

    @pytest.mark.parametrize("call", [nfl_search_plays, nfl_ftn_charting])
    def test_the_opponent_side_names_the_opponent(self, recorded, call):
        assert call(opponent="ZZZ", season=2024)["error"] == "Unknown opponent 'ZZZ'."

    @pytest.mark.parametrize("call", [nfl_search_plays, nfl_ftn_charting])
    def test_the_good_side_is_not_blamed_for_the_bad_one(self, recorded, call):
        result = call(team="KC", opponent="xyz", season=2024)
        assert result["error"] == "Unknown opponent 'xyz'."
        assert "Unknown team" not in result["error"]

    def test_the_resolver_names_a_bad_club_before_it_looks(self, recorded):
        # Group C second pass, fix 4: a bad code used to narrow the candidate list to
        # nothing, so Patrick Mahomes came back as a player who does not exist.
        result = nfl_resolve_player(player="Patrick Mahomes", team="XYZ")
        assert result["error"] == "Unknown team 'XYZ'."
        assert "did_you_mean" not in result
        assert recorded == []

    def test_the_error_lists_the_codes_a_caller_is_most_likely_to_type(self):
        hint = tools._unknown_team_error("oak ")["hint"]
        assert all(code in hint for code in ("LV", "LAC", "LA", "WAS"))


# ── fix 3, the era direction: rows written under a code the club no longer uses ─

def build_era_db(path):
    """A database whose 2019 rows are written 'OAK', which no local table does any more."""
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE schedules (game_id TEXT, season INT, week INT, game_type TEXT, "
                 "gameday TEXT, gametime TEXT, away_team TEXT, home_team TEXT, away_score INT, "
                 "home_score INT, result TEXT, overtime INT, spread_line DOUBLE, "
                 "total_line DOUBLE, div_game INT, roof TEXT, surface TEXT, temp INT, "
                 "wind INT, away_coach TEXT, home_coach TEXT, referee TEXT, stadium TEXT)")
    conn.execute("INSERT INTO schedules VALUES "
                 "('2019_01_OAK_NE', 2019, 1, 'REG', '2019-09-08', '20:20', 'OAK', 'NE', 3, 33, "
                 "'NE', 0, 3.0, 47.5, 0, 'outdoors', 'grass', 66, 8, 'G.Gruden', 'B.Belichick', "
                 "'W.Meyer', 'Aztek')")
    conn.execute("INSERT INTO schedules VALUES "
                 "('2019_19_OAK_TEN', 2019, 20, 'WC', '2020-01-04', '16:35', 'TEN', 'OAK', 12, 20, "
                 "'OAK', 0, -7.0, 44.0, 0, 'outdoors', 'grass', 40, 5, 'M.Vrabel', 'J.Jones', "
                 "'J.Triggs', 'Aztek')")
    conn.execute("CREATE TABLE rosters (full_name TEXT, position TEXT, depth_chart_position TEXT, "
                 "team TEXT, season INT, jersey_number INT, status TEXT, years_exp INT, "
                 "college TEXT, height TEXT, weight INT)")
    conn.execute("INSERT INTO rosters VALUES ('Raid Example', 'T', 'LT', 'OAK', 2019, 74, "
                 "'Active', 5, 'Ohio State', '6-5', 315)")
    conn.execute("CREATE TABLE team_offense_stats (team TEXT, season_year INT, season_type TEXT, "
                 "total_plays INT)")
    conn.execute("CREATE TABLE team_defense_stats (team TEXT, season_year INT, season_type TEXT, "
                 "sacks INT)")
    conn.execute("CREATE TABLE situational_stats (team TEXT, season_year INT, season_type TEXT, "
                 "situation TEXT, plays INT)")
    for table, extra in (("team_offense_stats", "600"), ("team_defense_stats", "40")):
        conn.execute(f"INSERT INTO {table} VALUES ('OAK', 2019, 'REG', {extra})")
    conn.execute("INSERT INTO situational_stats VALUES ('OAK', 2019, 'REG', 'goal_line', 12)")
    conn.close()


@pytest.fixture
def era_db(tmp_path, monkeypatch):
    db = tmp_path / "era.duckdb"
    build_era_db(db)
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(db))
    return db


class TestEraCodesOnTheirOwnDatabase:
    """The four tools Group C moved onto the shared code list, against rows that carry the
    code of the era. Both codes must answer with the same rows, and the current one must
    reach rows the club wrote under the old name."""

    def test_the_schedule_rows_answer_to_both_codes(self, era_db):
        rows = {c: nfl_schedule(team=c, season=2019) for c in ("OAK", "LV")}
        assert rows["OAK"] == rows["LV"]
        assert [g["game_id"] for g in rows["LV"]["games"]] == ["2019_01_OAK_NE", "2019_19_OAK_TEN"]

    def test_a_postseason_answer_finds_the_era_rows_too(self, era_db):
        result = nfl_schedule(team="LV", season=2019, season_type="POST")
        assert [g["game_id"] for g in result["games"]] == ["2019_19_OAK_TEN"]

    def test_the_derived_team_tables_answer_to_both_codes(self, era_db):
        rows = {c: nfl_team_stats(team=c, season=2019) for c in ("OAK", "LV")}
        assert rows["OAK"] == rows["LV"]
        assert rows["LV"]["offense"][0]["team"] == "OAK"       # the row keeps its era code
        assert rows["LV"]["situational"][0]["situation"] == "goal_line"

    def test_the_roster_answers_to_both_codes(self, era_db):
        rows = {c: nfl_roster(team=c, season=2019) for c in ("OAK", "LV")}
        assert rows["OAK"] == rows["LV"]
        assert rows["LV"]["count"] == 1

    def test_a_comparison_of_the_same_club_under_both_codes(self, era_db):
        result = nfl_compare(entity1="LV", entity2="OAK", compare_type="team", season=2019)
        assert result["LV"]["offense"] == result["OAK"]["offense"]
        assert result["LV"]["offense"][0]["team"] == "OAK"

    def test_an_unrelated_club_still_gets_no_rows(self, era_db):
        assert nfl_schedule(team="KC", season=2019)["count"] == 0


class TestJaguarsAndFreeAgentsOnRealData:
    """Group C second pass, fix 1: the fantasy tables spell the Jaguars 'JAC' and publish
    free agents as 'FA', and the franchise gate briefly refused both."""

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    @pytest.mark.parametrize("scope, table", [
        ("draft", "ff_rankings_draft"), ("week", "ff_rankings_week"),
    ])
    def test_the_code_the_table_publishes_and_the_current_one_reach_the_same_rows(
        self, scope, table,
    ):
        stored = nfl_query(f"SELECT COUNT(*) AS n FROM {table} WHERE team = 'JAC'")
        jags = stored["rows"][0]["n"]
        assert jags > 0
        typed = nfl_fantasy_rankings(team="JAC", scope=scope, limit=tools._MAX_ROWS)
        current = nfl_fantasy_rankings(team="JAX", scope=scope, limit=tools._MAX_ROWS)
        assert typed["count"] == current["count"] == jags
        # Sorted, because the tables sort by ecr and Jaguars rows share ecr values: the two
        # queries bind the same two codes in the other order, and a tie is no difference.
        def as_rows(result):
            return sorted(json.dumps(row, sort_keys=True, default=str)
                          for row in result["fantasy_rankings"])

        assert as_rows(typed) == as_rows(current)
        assert {row["team"] for row in typed["fantasy_rankings"]} == {"JAC"}
        assert typed["truncated"] is False

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    @pytest.mark.parametrize("scope, table", [
        ("draft", "ff_rankings_draft"), ("week", "ff_rankings_week"),
    ])
    def test_the_free_agent_status_returns_the_free_agents(self, scope, table):
        stored = nfl_query(f"SELECT COUNT(*) AS n FROM {table} WHERE team = 'FA'")
        result = nfl_fantasy_rankings(team="FA", scope=scope, limit=tools._MAX_ROWS)
        assert "error" not in result
        assert result["count"] == stored["rows"][0]["n"]
        assert all(row["team"] == "FA" for row in result["fantasy_rankings"])
        if scope == "draft":
            # The draft board always carries undrafted players: 136 in both databases this
            # suite runs against. The weekly board is one week's snapshot and can hold none,
            # so an empty answer there is the tool agreeing with the table, not a lost filter.
            assert result["count"] > 0
            assert {row["team"] for row in result["fantasy_rankings"]} == {"FA"}

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    def test_free_agent_does_not_break_a_named_player(self):
        # FA filters the rows and does not narrow the lookup, because no player table
        # stores it: a free agent in the rankings still answers to his own name.
        free = nfl_fantasy_rankings(team="FA", limit=tools._MAX_ROWS)["fantasy_rankings"]
        named = next(p["player"] for p in free
                     if "resolved" in nfl_resolve_player(p["player"]))
        result = nfl_fantasy_rankings(player=named, team="FA", limit=tools._MAX_ROWS)
        assert "error" not in result
        assert result["count"] > 0
        assert all(row["team"] == "FA" for row in result["fantasy_rankings"])

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    def test_the_opportunity_table_answers_the_publishers_code_too(self):
        # ff_opportunity stores 'JAX', so the alias is what keeps a caller who types the
        # FantasyPros spelling from getting nothing.
        typed = nfl_fantasy_opportunity(team="JAC", season=2024)
        current = nfl_fantasy_opportunity(team="JAX", season=2024)
        assert typed["count"] == current["count"] > 0
        assert typed["truncated"] is False

        def as_rows(result):
            return sorted(json.dumps(row, sort_keys=True, default=str)
                          for row in result["fantasy_opportunity"])

        assert as_rows(typed) == as_rows(current)

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    def test_the_unnamed_club_is_still_refused(self):
        assert nfl_fantasy_rankings(team="ZZZ")["error"] == "Unknown team 'ZZZ'."

    @pytest.mark.parametrize("code", ["JAC", "JAX"])
    def test_both_jaguar_codes_carry_the_other(self, code):
        assert set(tools._team_codes(code)) == {"JAC", "JAX"}

    def test_free_agent_is_not_a_franchise(self):
        assert canonical_team("FA") is None
        assert tools._team_codes("FA") == ["FA"]      # reaches the rows, not a club


class TestAliasCodesOnRealData:
    """What the loaded database can say about moved clubs: it has no era rows to find."""

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    @pytest.mark.parametrize("table, column", [
        ("plays", "posteam"), ("schedules", "home_team"), ("rosters", "team"),
        ("team_stats_raw", "team"), ("snap_counts", "team"),
    ])
    def test_the_local_database_holds_no_rows_from_before_a_move(self, table, column):
        # Proves the claim in the PR: the era direction cannot be shown here, which is why
        # TestEraCodesOnTheirOwnDatabase builds its own.
        result = nfl_query(f"SELECT COUNT(*) AS n FROM {table} "
                           f"WHERE {column} IN ('OAK', 'SD', 'STL')")
        assert result["rows"][0]["n"] == 0

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    @pytest.mark.parametrize("era, current", [("OAK", "LV"), ("SD", "LAC"), ("STL", "LA")])
    def test_both_codes_of_a_moved_club_answer_alike(self, era, current):
        # The current code is the only one the local rows carry, so the alias list is what
        # keeps the old code's answer from coming back empty.
        assert (nfl_schedule(team=era)["count"] == nfl_schedule(team=current)["count"] > 0)
        assert (nfl_team_stats(team=era, season=2024)
                == nfl_team_stats(team=current, season=2024))
        assert nfl_roster(team=era, season=2024)["count"] == nfl_roster(team=current, season=2024)["count"] > 0


# ── fix 4: a capped answer says it was capped ──────────────────────────────────

# Each capped tool, the key its rows sit under, and the most rows it can hand back.
CAPPED_TOOLS = {
    "nfl_search_plays": (lambda: nfl_search_plays(season=2024, max_rows=7), "rows", 7),
    "nfl_roster": (lambda: nfl_roster(season=2024), "players", tools._MAX_ROWS),
    "nfl_injuries": (lambda: nfl_injuries(season=2024), "injuries", tools._MAX_ROWS),
    "nfl_schedule": (lambda: nfl_schedule(season=2024), "games", tools._MAX_ROWS),
    "nfl_team_records": (lambda: nfl_team_records(from_season=1970), "records", tools._MAX_ROWS),
    "nfl_snap_counts": (lambda: nfl_snap_counts(season=2024), "snap_counts", tools._MAX_ROWS),
    "nfl_fantasy_opportunity": (lambda: nfl_fantasy_opportunity(season=2024),
                                "fantasy_opportunity", tools._MAX_ROWS),
    "nfl_fantasy_rankings": (lambda: nfl_fantasy_rankings(limit=5), "fantasy_rankings", 5),
    "nfl_td_luck": (lambda: nfl_td_luck(season=2024, limit=5), "td_luck", 5),
    "nfl_role_trend": (lambda: nfl_role_trend(season=2024, limit=5), "role_trend", 5),
    "nfl_separation_opportunity": (lambda: nfl_separation_opportunity(season=2024, limit=5),
                                   "separation_opportunity", 5),
    "nfl_drop_rate": (lambda: nfl_drop_rate(limit=5), "drop_rate", 5),
    "nfl_contract_value": (lambda: nfl_contract_value(limit=5), "contract_value", 5),
    "nfl_injury_return": (lambda: nfl_injury_return(limit=2), "injury_return", 2),
}

# The tools whose own cap the loaded data goes past. `schedules` and `team_records` hold
# fewer rows than 500 for the seasons here, so they only appear in the short-answer test.
OVER_LIMIT = {name: CAPPED_TOOLS[name] for name in sorted(CAPPED_TOOLS)
              if name not in ("nfl_schedule", "nfl_team_records")}

# One team, one week, one season: answers small enough that nothing was cut.
SHORT_ANSWERS = {
    "nfl_roster": (lambda: nfl_roster(team="KC", season=2024), "players"),
    "nfl_injuries": (lambda: nfl_injuries(team="KC", season=2024, week=1), "injuries"),
    "nfl_schedule": (lambda: nfl_schedule(season=2024), "games"),
    "nfl_team_records": (lambda: nfl_team_records(from_season=2024), "records"),
}


def _fake_execute(rows):
    """A stand-in for `tools._execute` that answers the two kinds of query a capped tool runs.

    Every reader of a rebuilt table asks the derived-shape question first (nfl_mcp.tools
    ._stale_derived_shape): it reads the shape stamp out of _ingest_metadata and probes
    information_schema for a derived table with no season_type column. Both answer "nothing is
    old-shaped here", so the tool goes on to the data query the test is about. A fake that handed
    data rows back for those two instead would make the guard report a stale database and the tool
    would answer with an error instead of a truncated list.
    """
    def fake(sql, params=None):
        if "_ingest_metadata" in sql or "information_schema" in sql:
            return []
        return rows
    return fake


class TestTruncationFlag:
    @pytest.mark.parametrize("cap", [0, -1, -500])
    def test_a_cap_below_one_still_hands_back_a_row(self, monkeypatch, cap):
        # Group C second pass, fix 6: the limit tools clamp to 1 and max_rows did not, so
        # the answer was 'no rows' plus 'there were more rows than that'.
        asked = []
        monkeypatch.setattr(tools, "_execute",
                            lambda sql, params=None: asked.append(sql)
                            or [{"n": i} for i in range(2)])
        result = nfl_search_plays(season=2024, max_rows=cap)
        assert result["row_count"] == 1
        assert result["truncated"] is True
        assert "LIMIT 2" in asked[0]

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    @pytest.mark.parametrize("cap", [0, -1])
    def test_a_cap_below_one_answers_with_a_real_row(self, cap):
        result = nfl_search_plays(team="KC", season=2024, week=10, max_rows=cap)
        assert result["row_count"] == 1
        assert result["truncated"] is True

    @pytest.mark.parametrize("name", sorted(CAPPED_TOOLS))
    def test_one_more_row_than_the_caller_may_see_is_reported(self, monkeypatch, name):
        call, key, cap = CAPPED_TOOLS[name]
        monkeypatch.setattr(
            tools, "_execute",
            _fake_execute([dict({"n": i}, season=2024, team="KC") for i in range(cap + 1)]))
        result = call()
        assert len(result[key]) == cap
        assert result["truncated"] is True
        assert "nfl_query" in result["hint"]

    @pytest.mark.parametrize("name", sorted(CAPPED_TOOLS))
    def test_a_short_answer_says_nothing_was_cut(self, monkeypatch, name):
        call, key = CAPPED_TOOLS[name][0], CAPPED_TOOLS[name][1]
        monkeypatch.setattr(tools, "_execute",
                            _fake_execute([{"n": 0, "season": 2024, "team": "KC"}]))
        result = call()
        assert len(result[key]) == 1
        assert result["truncated"] is False
        assert "hint" not in result

    def test_the_helper_asks_for_one_row_more_than_the_caller_may_see(self, monkeypatch):
        asked = []
        monkeypatch.setattr(tools, "_execute",
                            lambda sql, params=None: asked.append(sql) or [{"n": i} for i in range(6)])
        rows, flag = tools._execute_limited("SELECT 1 AS x WHERE 1=0", None, 500)
        assert "LIMIT 501" in asked[0]              # the probe row, on the fixed cap
        assert [r["n"] for r in rows] == [0, 1, 2, 3, 4, 5]
        assert flag == {"truncated": False}         # 6 rows, none of them cut

    def test_the_probe_row_never_reaches_the_caller(self, monkeypatch):
        monkeypatch.setattr(tools, "_execute",
                            lambda sql, params=None: [{"n": i} for i in range(501)])
        rows, flag = tools._execute_limited("SELECT 1 AS x WHERE 1=0", None, 500)
        assert len(rows) == 500
        assert flag == {"truncated": True, "hint": tools._TRUNCATION_HINT}


class TestTruncationOnRealData:
    """Real caps: snap counts for one team in one season are 933 rows, and the old tool
    returned the first 500 of them sorted by week, so the last weeks were simply gone."""

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    @pytest.mark.parametrize("name", sorted(OVER_LIMIT))
    def test_a_query_over_the_limit_says_so(self, name):
        call, key, cap = OVER_LIMIT[name]
        result = call()
        assert len(result[key]) == cap
        assert result["truncated"] is True
        assert result["hint"] == tools._TRUNCATION_HINT

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    @pytest.mark.parametrize("name", sorted(SHORT_ANSWERS))
    def test_a_query_under_the_limit_says_it_was_not_cut(self, name):
        call, key = SHORT_ANSWERS[name]
        result = call()
        assert 0 < len(result[key]) == result["count"] < tools._MAX_ROWS
        assert result["truncated"] is False
        assert "hint" not in result

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    @pytest.mark.parametrize("cap_offset, want_rows, want_flag", [
        (-1, -1, True), (0, 0, False), (1, 0, False),
    ])
    def test_a_result_that_lands_exactly_on_the_cap_says_it_was_not_cut(
        self, cap_offset, want_rows, want_flag,
    ):
        # One team, one week: 81 plays, so a cap of 80, 81 and 82 pins the three sides of
        # `if len(rows) > limit` from one measured size. A `>=` here cuts the last row and
        # cries truncation on a complete answer.
        size = nfl_query("SELECT COUNT(*) AS n FROM plays "
                         "WHERE posteam = 'KC' AND season = 2024 AND week = 10")
        true_size = size["rows"][0]["n"]
        assert true_size > 1

        cap = true_size + cap_offset
        result = nfl_search_plays(team="KC", season=2024, week=10, max_rows=cap)
        assert result["row_count"] == true_size + want_rows
        assert len(result["rows"]) == true_size + want_rows
        assert result["truncated"] is want_flag
        assert ("hint" in result) is want_flag

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    def test_the_cap_cuts_the_rows_it_says_it_cut(self):
        result = nfl_td_luck(season=2024, limit=5)
        all_rows = nfl_query(
            "SELECT COUNT(*) AS n FROM player_td_luck WHERE season = 2024")["rows"][0]["n"]
        assert result["count"] == 5 < all_rows
        assert result["truncated"] is True

    @pytest.mark.integration
    @pytest.mark.usefixtures("require_db")
    def test_the_rows_a_capped_tool_keeps_are_the_rows_its_own_sort_puts_first(self):
        result = nfl_td_luck(season=2024, limit=3)
        top = nfl_query("SELECT player_id, full_name, position, team, season, rec_td, rec_td_exp, "
                        "rec_td_luck, rush_td, rush_td_exp, rush_td_luck, total_td_luck_score "
                        "FROM player_td_luck WHERE season = 2024 "
                        "ORDER BY total_td_luck_score ASC NULLS LAST LIMIT 3")["rows"]
        kept = [{k: v for k, v in row.items() if k != "label"} for row in result["td_luck"]]
        assert kept == top
