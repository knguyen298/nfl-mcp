"""Tests for the structured MCP tools."""

import pytest
import duckdb
from unittest.mock import patch
from nfl_mcp import tools
from nfl_mcp.ingest import (
    DERIVED_SHAPE_VERSION,
    _ensure_metadata_table,
    _record_loaded,
)
from nfl_mcp.tools import (
    nfl_schema, nfl_status, nfl_search_plays, nfl_team_stats,
    nfl_player_stats, nfl_compare, nfl_query,
    nfl_catalog, nfl_roster, nfl_injuries, nfl_schedule, nfl_snap_counts,
    nfl_fantasy_opportunity, nfl_fantasy_rankings, nfl_ftn_charting,
    nfl_td_luck, nfl_role_trend, nfl_separation_opportunity,
    nfl_drop_rate, nfl_contract_value, nfl_injury_return,
    nfl_team_records,
)


@pytest.fixture
def require_db(rebuilt_db):
    """Skip integration tests if no data is loaded.

    Depends on rebuilt_db, so the queries below read a copy of the database whose derived team
    tables have been rebuilt by the current ingest code: the databases on disk and in CI were built
    before those tables gained a season_type column, and no test may write to them.
    """
    result = nfl_query("SELECT COUNT(*) AS n FROM plays")
    if "error" in result:
        pytest.skip("No DuckDB database loaded")
    if result["rows"][0]["n"] == 0:
        pytest.skip("plays table is empty")


class TestNflSchema:
    def test_returns_summary_by_default(self):
        result = nfl_schema()
        assert "schema" in result
        assert "hint" in result
        assert len(result["schema"]) > 200

    def test_default_includes_available_tables_key(self):
        result = nfl_schema()
        assert "available_tables" in result
        assert isinstance(result["available_tables"], list)

    def test_summary_contains_key_columns(self):
        schema = nfl_schema()["schema"]
        for col in ["epa", "posteam", "defteam", "play_type", "passer_player_name"]:
            assert col in schema

    def test_category_returns_detail(self):
        result = nfl_schema(category="epa")
        assert "schema" in result
        assert "qb_epa" in result["schema"]

    def test_category_all_returns_full(self):
        result = nfl_schema(category="all")
        assert len(result["schema"]) > 5000

    def test_unknown_category_returns_error(self):
        result = nfl_schema(category="nonexistent")
        assert "error" in result
        assert "available" in result

    def test_the_team_table_docs_name_the_source_and_the_filter(self):
        """Group B rule: the schema text says where each column comes from."""
        schema = nfl_schema(category="aggregate_tables")["schema"]
        assert "season_type" in schema
        assert "team_stats_raw" in schema and "schedules" in schema
        assert "(pass = 1 OR rush = 1) AND epa IS NOT NULL AND two_point_attempt <> 1" in schema
        assert "third_down_converted" in schema and "fourth_down_converted" in schema
        assert "red_zone_trips" in schema and "game_id plus" in schema
        assert "play_type = 'no_play'" in schema
        for table in ("team_offense_stats", "team_defense_stats", "situational_stats"):
            assert table in schema


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflSchemaIntegration:
    def test_default_lists_known_tables(self):
        result = nfl_schema()
        tables = result["available_tables"]
        assert len(tables) > 0
        for expected in ("plays", "rosters", "injuries", "schedules", "snap_counts"):
            assert expected in tables

    def test_default_excludes_internal_tables(self):
        result = nfl_schema()
        for t in result["available_tables"]:
            assert not t.startswith("_"), f"Internal table '{t}' should be excluded"

    def test_table_lookup_returns_columns(self):
        result = nfl_schema(table="rosters")
        assert "error" not in result
        assert result["table"] == "rosters"
        col_names = [c["column_name"] for c in result["columns"]]
        assert "full_name" in col_names
        assert "position" in col_names
        assert "team" in col_names
        assert "season" in col_names

    def test_table_lookup_injuries(self):
        result = nfl_schema(table="injuries")
        assert "error" not in result
        col_names = [c["column_name"] for c in result["columns"]]
        assert "report_status" in col_names

    def test_table_lookup_schedules(self):
        result = nfl_schema(table="schedules")
        assert "error" not in result
        col_names = [c["column_name"] for c in result["columns"]]
        for col in ("home_team", "away_team", "home_score", "away_score"):
            assert col in col_names

    def test_unknown_table_returns_error_with_list(self):
        result = nfl_schema(table="definitely_not_a_table")
        assert "error" in result
        assert "available_tables" in result
        assert len(result["available_tables"]) > 0


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflSearchPlays:
    def test_search_by_team(self):
        result = nfl_search_plays(team="KC", season=2024, max_rows=5)
        assert "error" not in result
        assert result["row_count"] > 0
        assert all(r["posteam"] == "KC" for r in result["rows"])

    def test_search_by_player(self):
        result = nfl_search_plays(player="Mahomes", season=2024, max_rows=5)
        assert "error" not in result
        assert result["row_count"] > 0

    def test_search_touchdowns(self):
        result = nfl_search_plays(is_touchdown=True, season=2024, max_rows=5)
        assert "error" not in result
        assert all(r["touchdown"] == 1 for r in result["rows"])

    def test_search_turnovers(self):
        result = nfl_search_plays(is_turnover=True, season=2024, max_rows=5)
        assert "error" not in result
        assert result["row_count"] > 0

    def test_search_situation_red_zone(self):
        result = nfl_search_plays(situation="red_zone", team="KC", season=2024, max_rows=5)
        assert "error" not in result

    def test_search_min_yards(self):
        result = nfl_search_plays(min_yards=30, season=2024, max_rows=5)
        assert "error" not in result
        assert all(r["yards_gained"] >= 30 for r in result["rows"])

    def test_empty_search(self):
        result = nfl_search_plays(max_rows=10)
        assert "error" not in result
        assert result["row_count"] > 0


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflTeamStats:
    def test_returns_offense_and_defense(self):
        result = nfl_team_stats(team="KC", season=2024)
        assert "error" not in result
        assert "offense" in result
        assert "defense" in result
        # Two rows now, where the old table handed back one merged line: the Chiefs played 17
        # regular-season games and 3 postseason games in 2024 and they are not the same sample.
        assert [(r["season_type"], r["total_plays"]) for r in result["offense"]] == [
            ("POST", 162), ("REG", 1091)]
        assert [r["season_type"] for r in result["defense"]] == ["POST", "REG"]

    def test_offense_only(self):
        result = nfl_team_stats(team="BAL", season=2024, side="offense")
        assert "offense" in result
        assert "defense" not in result

    def test_defense_only(self):
        result = nfl_team_stats(team="BAL", season=2024, side="defense")
        assert "defense" in result
        assert "offense" not in result

    def test_includes_situational(self):
        result = nfl_team_stats(team="KC", season=2024)
        assert "situational" in result
        assert len(result["situational"]) > 0

    def test_unknown_team_names_the_code_instead_of_returning_nothing(self):
        # Group C fix 3: a code no franchise uses is an answer, not an empty result set.
        result = nfl_team_stats(team="ZZZ", season=2024)
        assert result["error"] == "Unknown team 'ZZZ'."
        assert "current abbreviation" in result["hint"]

    def test_season_type_picks_one_of_the_two_rows(self):
        reg = nfl_team_stats(team="KC", season=2024, season_type="REG")
        assert [r["season_type"] for r in reg["offense"]] == ["REG"]
        assert [r["season_type"] for r in reg["defense"]] == ["REG"]
        assert [r["season_type"] for r in reg["situational"]] == ["REG"] * len(reg["situational"])
        post = nfl_team_stats(team="KC", season=2024, side="offense", season_type="post")
        assert [(r["season_type"], r["total_plays"]) for r in post["offense"]] == [("POST", 162)]

    def test_the_response_names_the_source_of_every_column(self):
        sources = nfl_team_stats(team="KC", season=2024, side="offense")["sources"]
        assert set(sources) == {"team_stats_raw", "schedules", "plays"}
        assert "red_zone_trips" in sources["plays"]
        assert "points_for" in sources["schedules"]
        assert "scrimmage plays only" in sources["plays"]


# The four tables the staleness probe reads (nfl_mcp.ingest._PROBE_TABLES), with the
# columns their tools select and one synthetic row each, so a read path has a row.
_PROBED_TABLES = (
    ("team_offense_stats", "team VARCHAR, season_year INTEGER, total_plays INTEGER",
     "('KC', 2024, 1091"),
    ("team_defense_stats", "team VARCHAR, season_year INTEGER, sacks INTEGER",
     "('KC', 2024, 39"),
    ("situational_stats", "team VARCHAR, season_year INTEGER, situation VARCHAR",
     "('KC', 2024, '4th Down'"),
    ("player_separation_opportunity", "player_id VARCHAR, season INTEGER, avg_separation DOUBLE",
     "('00-0033873', 2024, 2.9"),
)
_ALL_PROBED = tuple(t for t, _, _ in _PROBED_TABLES)


def _derived_shape_db(tmp_path, monkeypatch, *, without: tuple[str, ...] = (),
                      stamp: int | None = None) -> None:
    """Point the tools at a database built to one derived shape, with no ingest and no network.

    `without` names the probe tables to create in the shape a database ingested before season_type
    landed has: no season_type column. All four are built the same way by default because the guard
    asks one question about the whole database - one old-shaped table is enough to make every reader
    of a derived table answer with the rebuild message.

    `stamp` writes the row `rebuild_derived_tables` leaves in _ingest_metadata, and None leaves the
    database with no _ingest_metadata table at all, which is what a hand-built test database looks
    like. Tables the tools read that are not named here simply do not exist, which is how the tests
    below tell a guard answer from a query answer.
    """
    db = tmp_path / "team_tables.duckdb"
    conn = duckdb.connect(str(db))
    for table, cols, values in _PROBED_TABLES:
        old = table in without
        extra = "" if old else ", season_type VARCHAR"
        tail = "" if old else ", 'REG'"
        conn.execute(f"CREATE TABLE {table} ({cols}{extra})")
        conn.execute(f"INSERT INTO {table} VALUES {values}{tail})")
    if stamp is not None:
        _ensure_metadata_table(conn)
        _record_loaded(conn, dataset_id="derived_shape", table_name="derived_shape",
                       loader_fn="test_tools", row_count=stamp)
    conn.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(db))
    return db


_REBUILD = ("Rebuild the derived tables: nfl-mcp rebuild with the server stopped, or "
            "/update_db.sh in a container (builds the idle slot and swaps the symlink)")
_ALL_FOUR = "player_separation_opportunity, situational_stats, team_defense_stats, team_offense_stats"
_STAMP_OLD = (f"the derived-shape stamp reads {DERIVED_SHAPE_VERSION - 1}, "
              f"this build writes {DERIVED_SHAPE_VERSION}")


class TestStaleDerivedTables:
    """A database whose derived tables predate this build cannot serve these queries.

    The tools must say what to run, not hand back DuckDB's binder error about a column the caller
    never typed, and not return rows in the old shape as if they were the new one. The named
    command must reach a DuckDB writer without the network: `nfl-mcp rebuild` with the server
    stopped, or /update_db.sh in a container. Every reader of a rebuilt table asks the same
    question `nfl-mcp rebuild --check` answers, so the answer does not depend on which table the
    tool happens to read.
    """

    def test_a_table_without_season_type_names_the_command_to_rebuild_it(self, tmp_path, monkeypatch):
        _derived_shape_db(tmp_path, monkeypatch, without=_ALL_PROBED)
        assert nfl_team_stats(team="KC", season=2024) == {
            "error": f"{_ALL_FOUR} have no season_type column. {_REBUILD}"}

    def test_the_answer_does_not_depend_on_which_table_the_tool_reads(
            self, tmp_path, monkeypatch):
        """The guard is one question about the database, so all three sides report the same gap."""
        _derived_shape_db(tmp_path, monkeypatch, without=_ALL_PROBED)
        errors = {side: nfl_team_stats(team="KC", season=2024, side=side)["error"]
                  for side in ("offense", "defense", "situational")}
        assert len(errors) == 3
        assert len(set(errors.values())) == 1
        assert errors["offense"] == f"{_ALL_FOUR} have no season_type column. {_REBUILD}"

    def test_one_old_shaped_table_stales_the_whole_database(self, tmp_path, monkeypatch):
        """player_separation_opportunity selects season_type and separation_source, so an old
        database failed here with a raw binder error and a logged traceback. The other three probe
        tables are current, and that no longer buys nfl_team_stats a read: the database needs a
        rebuild either way."""
        _derived_shape_db(tmp_path, monkeypatch, without=("player_separation_opportunity",))
        expected = {"error": "player_separation_opportunity has no season_type column. "
                             f"{_REBUILD}"}
        assert nfl_separation_opportunity() == expected
        assert nfl_team_stats(team="KC", season=2024) == expected

    def test_compare_team_names_the_command_to_rebuild_it(self, tmp_path, monkeypatch):
        _derived_shape_db(tmp_path, monkeypatch, without=_ALL_PROBED)
        assert nfl_compare(entity1="KC", entity2="BAL", compare_type="team", season=2024) == {
            "error": f"{_ALL_FOUR} have no season_type column. {_REBUILD}"}

    @pytest.mark.parametrize("tool,kwargs", [
        pytest.param(nfl_team_stats, {"team": "KC", "season": 2024}, id="nfl_team_stats"),
        pytest.param(nfl_compare,
                     {"entity1": "KC", "entity2": "BAL", "compare_type": "team"},
                     id="nfl_compare"),
        pytest.param(nfl_separation_opportunity, {}, id="nfl_separation_opportunity"),
        pytest.param(nfl_team_records, {"team": "KC"}, id="nfl_team_records"),
        pytest.param(nfl_td_luck, {}, id="nfl_td_luck"),
        pytest.param(nfl_role_trend, {}, id="nfl_role_trend"),
        pytest.param(nfl_drop_rate, {}, id="nfl_drop_rate"),
        pytest.param(nfl_contract_value, {}, id="nfl_contract_value"),
        pytest.param(nfl_injury_return, {"injury_type": "ankle"}, id="nfl_injury_return"),
    ])
    def test_an_old_stamp_answers_before_the_tool_queries(self, tool, kwargs, tmp_path, monkeypatch):
        """The stamp branch, over every tool that reads a table rebuild_derived_tables writes.

        The four probe tables are current here and the stamp row is one version back, so only the
        stamp can be what fires. None of the tables these tools read exists in the database: had a
        tool reached its own query it would hear DuckDB say the table is missing, so an answer that
        names the stamp proves the guard answered first.
        """
        _derived_shape_db(tmp_path, monkeypatch, stamp=DERIVED_SHAPE_VERSION - 1)
        assert tool(**kwargs) == {"error": f"{_STAMP_OLD}. {_REBUILD}"}

    def test_injury_return_stops_answering_with_old_shaped_rows(self, tmp_path, monkeypatch):
        """The cost the review measured, on a database in the shape that ingest left behind.

        injury_return_curve carries 20 pre-season_type injury groups including `other`, and the
        tool handed them back with no error at all. The probe tables are in that same old shape,
        so the guard now answers first.
        """
        db = _derived_shape_db(tmp_path, monkeypatch, without=_ALL_PROBED)
        conn = duckdb.connect(str(db))
        conn.execute("CREATE TABLE injury_return_curve (injury_type VARCHAR, position VARCHAR, "
                     "week_post_return INTEGER, avg_snap_pct_recovery DOUBLE, "
                     "median_snap_pct_recovery DOUBLE, sample_size INTEGER)")
        conn.execute("INSERT INTO injury_return_curve VALUES "
                     "('ankle', 'WR', 1, 82.0, 80.0, 41), ('other', 'RB', 1, 74.0, 71.0, 12)")
        conn.close()
        result = nfl_injury_return(injury_type="ankle")
        assert result == {"error": f"{_ALL_FOUR} have no season_type column. {_REBUILD}"}

    @pytest.mark.parametrize("stamp", [
        pytest.param(None, id="no_stamp_row"),
        pytest.param(DERIVED_SHAPE_VERSION, id="current_stamp_row"),
    ])
    def test_a_current_shape_is_read_as_usual(self, stamp, tmp_path, monkeypatch):
        """Probe passes with no stamp row at all, or with the current one: stay silent.

        A hand-built test database looks like the first and a database rebuilt just before the
        stamp was introduced like the second, and nagging either is wrong.
        """
        _derived_shape_db(tmp_path, monkeypatch, stamp=stamp)
        stats = nfl_team_stats(team="KC", season=2024)
        assert "error" not in stats
        assert [r["season_type"] for r in stats["offense"]] == ["REG"]
        compared = nfl_compare(entity1="KC", entity2="BAL", compare_type="team", season=2024)
        assert "error" not in compared
        assert [r["season_type"] for r in compared["KC"]["offense"]] == ["REG"]
        assert compared["BAL"]["offense"] == []
        # A tool whose table the probe does not list still reaches its own query, and DuckDB is
        # the one that says the table is missing - the guard said nothing.
        missing = nfl_injury_return(injury_type="ankle")
        assert "Rebuild the derived tables" not in missing["error"]
        assert "injury_return_curve" in missing["error"]


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestTeamTablesAreTheOfficialNumbers:
    """The rebuilt team tables against team_stats_raw, schedules, and counts run by hand.

    Every expected number below was read off the same database the test runs on, by the queries
    next to it, for the 2024 season: Kansas City (17 regular-season games, 3 postseason) and
    Philadelphia (17 and 2). The regular-season rows must equal the week-level team box exactly,
    because the box is what ESPN and Pro Football Reference print.
    """

    BOX_COLUMNS = (
        "attempts", "sacks_suffered", "carries", "completions", "rushing_yards", "passing_yards",
        "sack_yards_lost", "passing_tds", "rushing_tds", "special_teams_tds",
        "passing_interceptions", "fumbles_lost_total", "def_sacks", "def_interceptions",
        "penalties", "penalty_yards",
    )

    @staticmethod
    def box(team, season_type):
        """Ground truth: the week-level team box summed the way the derived table sums it."""
        select = ", ".join(f"COALESCE(SUM({c}), 0) AS {c}" for c in
                           TestTeamTablesAreTheOfficialNumbers.BOX_COLUMNS)
        return nfl_query(
            f"SELECT {select}, "
            f"COALESCE(SUM(passing_first_downs + rushing_first_downs + receiving_first_downs), 0) "
            f"AS first_downs "
            f"FROM team_stats_raw WHERE team='{team}' AND season=2024 AND season_type='{season_type}'"
        )["rows"][0]

    @pytest.mark.parametrize("team,plays,yards,tds,tovs", [
        ("KC", 1091, 5570, 41, 14),
        ("PHI", 1114, 6242, 53, 15),
    ])
    def test_two_team_seasons_equal_the_week_level_team_box(self, team, plays, yards, tds, tovs):
        got = nfl_team_stats(team=team, season=2024, season_type="REG", side="offense")["offense"][0]
        box = self.box(team, "REG")
        assert (got["total_plays"], got["total_yards"]) == (plays, yards)
        assert (got["touchdowns"], got["turnovers"]) == (tds, tovs)
        assert got["total_plays"] == box["attempts"] + box["sacks_suffered"] + box["carries"]
        assert got["total_yards"] == (box["rushing_yards"] + box["passing_yards"]
                                      + box["sack_yards_lost"])
        assert (got["pass_plays"], got["rush_plays"], got["completions"]) == (
            box["attempts"], box["carries"], box["completions"])
        assert (got["pass_yards"], got["rush_yards"]) == (
            box["passing_yards"] + box["sack_yards_lost"], box["rushing_yards"])
        assert (got["touchdowns"], got["sacks_suffered"], got["sack_yards"]) == (
            box["passing_tds"] + box["rushing_tds"], box["sacks_suffered"], box["sack_yards_lost"])
        assert (got["turnovers"], got["first_downs"], got["penalties"]) == (
            box["passing_interceptions"] + box["fumbles_lost_total"], box["first_downs"],
            box["penalties"])
        defense = nfl_team_stats(team=team, season=2024, season_type="REG", side="defense")["defense"][0]
        assert (defense["sacks"], defense["interceptions"]) == (
            box["def_sacks"], box["def_interceptions"])

    def test_a_sack_is_not_a_pass_attempt(self):
        """The old table counted a sack in pass_plays: Kansas City 2024 said 741."""
        got = nfl_team_stats(team="KC", season=2024, season_type="REG", side="offense")["offense"][0]
        assert got["pass_plays"] == 600
        assert got["sacks_suffered"] == 41
        assert got["sack_yards"] == -266
        assert got["pass_plays"] + got["sacks_suffered"] + got["rush_plays"] == got["total_plays"]

    def test_a_return_touchdown_is_not_an_offensive_touchdown(self):
        """Week 22, 2024 postseason: a Mahomes pass is intercepted and returned 38 yards for a
        touchdown. play_by_play files that with posteam = KC and touchdown = 1, so the old table
        credited the Chiefs' offense with a score it did not have."""
        naive = nfl_query(
            "SELECT COUNT(*) AS n FROM plays WHERE posteam='KC' AND season=2024 "
            "AND season_type='POST' AND touchdown = 1")["rows"][0]["n"]
        got = nfl_team_stats(team="KC", season=2024, season_type="POST", side="offense")["offense"][0]
        assert naive == 10
        assert got["touchdowns"] == 9
        assert (got["passing_tds"], got["rushing_tds"]) == (5, 4)
        # The situational rows add up to the offense's 9, not to 10.
        situational = nfl_team_stats(team="KC", season=2024, season_type="POST",
                                    side="situational")["situational"]
        assert sum(r["touchdowns"] for r in situational) == 9

    def test_third_down_rate_uses_the_conversion_flags(self):
        """The two counts cover every play that carries a conversion flag, so a quarterback kneel
        that fails a third down is an attempt: 48 kneels and 8 spikes carry third_down_failed = 1 in
        the 2024 regular season, and the pass-or-rush population used to drop all of them. Kansas
        City's 225 attempts become 229. A punt or a field-goal attempt carries no flag, and a play a
        penalty nullified carries none either, so nothing double counts."""
        got = nfl_team_stats(team="KC", season=2024, season_type="REG", side="offense")["offense"][0]
        assert (got["third_down_conversions"], got["third_down_attempts"]) == (111, 229)
        assert got["third_down_pct"] == 48.5
        assert got["third_down_pct"] == round(100.0 * 111 / 229, 1)
        defense = nfl_team_stats(team="KC", season=2024, season_type="REG", side="defense")["defense"][0]
        assert (defense["third_down_stops"], defense["third_down_attempts"]) == (122, 215)
        assert defense["third_down_stop_pct"] == 56.7

    def test_the_flags_and_the_yardage_do_not_agree_and_the_flags_win(self):
        """Buffalo 2024 regular season: the conversion flags credit 89 third downs out of 202
        attempts, while measuring yards against ydstogo over every 3rd-down play says 88 out of 214,
        and the same flags over the scrimmage population only says 89 out of 198. A rate is only as
        good as the rule behind it, so all three numbers are pinned."""
        counts = nfl_query(
            "SELECT SUM(CASE WHEN COALESCE(third_down_converted,0)=1 THEN 1 ELSE 0 END) AS conv, "
            "SUM(CASE WHEN COALESCE(third_down_converted,0)=1 OR COALESCE(third_down_failed,0)=1 "
            "THEN 1 ELSE 0 END) AS att, SUM(CASE WHEN yards_gained>=ydstogo THEN 1 ELSE 0 END) AS "
            "by_yards, COUNT(*) AS all_thirds FROM plays WHERE posteam='BUF' AND season=2024 "
            "AND season_type='REG' AND down=3 AND (COALESCE(pass,0)=1 OR COALESCE(rush,0)=1) "
            "AND epa IS NOT NULL AND COALESCE(two_point_attempt,0)<>1")['rows'][0]
        assert (counts["conv"], counts["att"], counts["by_yards"], counts["all_thirds"]) == (
            89, 198, 88, 214)
        got = nfl_team_stats(team="BUF", season=2024, season_type="REG", side="offense")["offense"][0]
        assert (got["third_down_conversions"], got["third_down_attempts"]) == (89, 202)
        assert got["third_down_pct"] == round(100.0 * 89 / 202, 1)

    def test_a_red_zone_rate_divides_trips_and_not_plays(self):
        """A trip is a drive that gets a football play inside the opponent's 20. Kansas City reached
        it on 67 regular-season drives in 2024 and scored a touchdown of its own on 35 (52.2%).

        The raw-source query, kept here so the literals can be re-run:
          SELECT COUNT(*) t, SUM(td) scored FROM (
            SELECT game_id, drive,
                   MAX(CASE WHEN play_type IN ('pass','run','field_goal','qb_kneel','qb_spike')
                             AND COALESCE(two_point_attempt,0)<>1
                            THEN 1 ELSE 0 END) AS reached_20,
                   MAX(CASE WHEN COALESCE(pass_touchdown,0)=1 OR COALESCE(rush_touchdown,0)=1
                            THEN 1 ELSE 0 END) AS td
            FROM plays WHERE posteam='KC' AND season=2024 AND season_type='REG'
              AND drive IS NOT NULL AND game_id IS NOT NULL AND yardline_100<=20
            GROUP BY game_id, drive) WHERE reached_20 = 1;

        Kansas City also played 276 downs inside the 20, and the old rate divided by those and
        reported 12.7. Counting a kick as proof added 6 drives that never got a football play inside
        the 20 - each one is a long touchdown pass followed by the extra point at the 15 - and the
        league figure read 2,141 trips and 47.6% for the 2024 regular season."""
        got = nfl_team_stats(team="KC", season=2024, season_type="REG", side="offense")["offense"][0]
        assert (got["red_zone_trips"], got["red_zone_touchdowns"]) == (67, 35)
        assert got["red_zone_td_pct"] == 52.2
        plays = nfl_query(
            "SELECT COUNT(*) AS n FROM plays WHERE posteam='KC' AND season=2024 "
            "AND season_type='REG' AND yardline_100<=20")["rows"][0]["n"]
        assert plays == 276
        assert got["red_zone_td_pct"] != round(100.0 * 35 / plays, 1)
        # A kick-only drive is not a trip: 6 of Kansas City's 73 old trips were that shape.
        kick_only = nfl_query(
            "SELECT COUNT(*) AS n FROM (SELECT game_id, drive, "
            "SUM(CASE WHEN play_type IN ('pass','run','field_goal','qb_kneel','qb_spike') "
            "AND COALESCE(two_point_attempt,0)<>1 THEN 1 "
            "ELSE 0 END) AS football FROM plays WHERE posteam='KC' AND season=2024 "
            "AND season_type='REG' AND drive IS NOT NULL AND game_id IS NOT NULL "
            "AND yardline_100<=20 GROUP BY game_id, drive) WHERE football = 0")["rows"][0]["n"]
        assert kick_only == 6
        phi = nfl_team_stats(team="PHI", season=2024, season_type="REG", side="offense")["offense"][0]
        assert (phi["red_zone_trips"], phi["red_zone_touchdowns"], phi["red_zone_td_pct"]) == (
            71, 38, 53.5)

    def test_a_red_zone_trip_needs_a_football_play_not_a_kick(self):
        """The extra point is spotted at the 15 and the two-point try at the 2, so yardline_100 <= 20
        on its own let the kick after a touchdown prove the trip - 1,292 extra-point rows in 2024
        sit inside that bound. Of the 6 Kansas City drives that bound used to count, every one holds
        a single extra-point row and no football play inside the 20."""
        drives = nfl_query(
            "SELECT types FROM (SELECT game_id, drive, "
            "SUM(CASE WHEN play_type IN ('pass','run','field_goal','qb_kneel','qb_spike') "
            "AND COALESCE(two_point_attempt,0)<>1 THEN 1 "
            "ELSE 0 END) AS football, LIST(play_type) AS types FROM plays WHERE posteam='KC' "
            "AND season=2024 AND season_type='REG' AND drive IS NOT NULL AND game_id IS NOT NULL "
            "AND yardline_100<=20 GROUP BY game_id, drive) WHERE football = 0")["rows"]
        assert [r["types"] for r in drives] == [["extra_point"]] * 6

    def test_a_two_point_try_does_not_prove_a_red_zone_trip(self):
        """A two-point attempt is recorded with play_type 'pass' or 'run' and is spotted at the 2, so
        it clears the football-play gate on its own and is the extra point defect again in a
        different costume: 99 passes and 36 runs in the 2024 regular season carry
        two_point_attempt = 1. The gate excludes them by that flag, the way _SCRIMMAGE does.

        The league row is what catches it, because no single team's count moves - Kansas City's 67
        trips are the same either way. The gate with the exclusion gives 1,821 trips and 1,019
        touchdowns across the 32 regular-season rows (56.0%); without it, 1,844 and 55.3%:

          SELECT COUNT(*) t, SUM(td) scored FROM (
            SELECT posteam, game_id, drive,
                   MAX(CASE WHEN play_type IN ('pass','run','field_goal','qb_kneel','qb_spike')
                             AND COALESCE(two_point_attempt,0)<>1 THEN 1 ELSE 0 END) AS reached_20,
                   MAX(CASE WHEN COALESCE(pass_touchdown,0)=1 OR COALESCE(rush_touchdown,0)=1
                            THEN 1 ELSE 0 END) AS td
            FROM plays WHERE posteam IS NOT NULL AND season=2024 AND season_type='REG'
              AND drive IS NOT NULL AND game_id IS NOT NULL AND yardline_100<=20
            GROUP BY 1, 2, 3) WHERE reached_20 = 1;
        """
        league = nfl_query(
            "SELECT SUM(red_zone_trips) AS trips, SUM(red_zone_touchdowns) AS tds, "
            "ROUND(100.0*SUM(red_zone_touchdowns)/SUM(red_zone_trips), 1) AS pct "
            "FROM team_offense_stats WHERE season_year=2024 AND season_type='REG'")["rows"][0]
        assert (league["trips"], league["tds"], league["pct"]) == (1821, 1019, 56.0)
        # 23 drives proved themselves only through the two-point try, and every one of them is
        # recorded as a pass or a run, which is why the play_type gate alone cannot drop them.
        only_two_point = nfl_query(
            "SELECT COUNT(*) AS n FROM (SELECT posteam, game_id, drive, "
            "SUM(CASE WHEN play_type IN ('pass','run','field_goal','qb_kneel','qb_spike') "
            "AND COALESCE(two_point_attempt,0)<>1 THEN 1 ELSE 0 END) AS football, "
            "SUM(CASE WHEN COALESCE(two_point_attempt,0)=1 THEN 1 ELSE 0 END) AS tries "
            "FROM plays WHERE posteam IS NOT NULL AND season=2024 AND season_type='REG' "
            "AND drive IS NOT NULL AND game_id IS NOT NULL AND yardline_100<=20 "
            "GROUP BY 1, 2, 3) WHERE football = 0 AND tries > 0")["rows"][0]["n"]
        assert only_two_point == 23

    def test_one_nullified_play_carries_a_conversion_flag(self):
        """The comment used to say a play a penalty nullified carries no flag at all. One of the
        6,569 no_play rows in the table does: game 2024_15_MIA_HOU, a Houston fake punt on 4th and 2,
        35 yards, fourth_down_converted = 1. The conversion columns count it and plays does not, so
        the Houston 2024 regular-season 4th Down row reads 18 attempts over 17 plays - the rule is
        kept as built, and the row says so."""
        flagged = nfl_query(
            "SELECT game_id, play_id, posteam, down, ydstogo, yards_gained, fourth_down_converted "
            "FROM plays WHERE play_type='no_play' AND (COALESCE(third_down_converted,0)=1 OR "
            "COALESCE(third_down_failed,0)=1 OR COALESCE(fourth_down_converted,0)=1 OR "
            "COALESCE(fourth_down_failed,0)=1)")["rows"]
        assert [(r["game_id"], r["posteam"], r["down"], r["ydstogo"], r["yards_gained"]) for r in flagged] == [
            ("2024_15_MIA_HOU", "HOU", 4.0, 2.0, 35.0)]
        fourth = next(r for r in nfl_team_stats(team="HOU", season=2024, season_type="REG",
                                                side="situational")["situational"]
                      if r["situation"] == "4th Down")
        assert (fourth["plays"], fourth["conversion_attempts"]) == (17, 18)
        # 32 rows of this table from 2024 show more attempts than plays; the other 31 are kneels and
        # spikes, which carry a flag and are not scrimmage plays, so this gap is not a counting error.
        # (Across every season in the database the same count is 84; it is scoped here so the
        # 2024-only database this suite also runs on answers the same.)
        gap = nfl_query("SELECT COUNT(*) AS n FROM situational_stats WHERE season_year=2024 "
                        "AND conversion_attempts > plays")
        assert gap["rows"][0]["n"] == 32

    def test_a_trip_scored_from_outside_the_20_counts_as_a_miss(self):
        """The touchdown check reads only plays inside the 20, so a drive that entered the 20, was
        pushed back, and scored from outside it is a trip with no touchdown. 11 drives of the 2024
        regular season have that shape and stay in the denominator as misses; the rule is deliberate
        and recorded in the comment, so this pins the count that motivated it."""
        drives = nfl_query(
            "SELECT COUNT(*) AS n FROM (SELECT posteam, game_id, drive, "
            "MAX(CASE WHEN play_type IN ('pass','run','field_goal','qb_kneel','qb_spike') "
            "AND COALESCE(two_point_attempt,0)<>1 THEN 1 ELSE 0 END) AS reached, "
            "MAX(CASE WHEN COALESCE(pass_touchdown,0)=1 OR COALESCE(rush_touchdown,0)=1 "
            "THEN 1 ELSE 0 END) AS td_inside FROM plays WHERE posteam IS NOT NULL AND season=2024 "
            "AND season_type='REG' AND drive IS NOT NULL AND game_id IS NOT NULL "
            "AND yardline_100<=20 GROUP BY 1,2,3) t JOIN (SELECT posteam, game_id, drive, "
            "MAX(CASE WHEN COALESCE(pass_touchdown,0)=1 OR COALESCE(rush_touchdown,0)=1 "
            "THEN 1 ELSE 0 END) AS td_any FROM plays WHERE posteam IS NOT NULL AND season=2024 "
            "AND season_type='REG' AND drive IS NOT NULL AND game_id IS NOT NULL GROUP BY 1,2,3) a "
            "USING (posteam, game_id, drive) WHERE t.reached=1 AND t.td_inside=0 AND a.td_any=1")
        assert drives["rows"][0]["n"] == 11

    def test_punts_and_field_goals_are_not_fourth_down_attempts(self):
        """Kansas City 2024 regular season: 118 plays are recorded on 4th down, 95 of them a punt or
        a field goal and 6 more nullified by a penalty. The situational row is the 17 plays left, of
        which the nflfastR flags credit 12 conversions out of 17 attempts."""
        down4 = nfl_query(
            "SELECT COUNT(*) AS n, SUM(CASE WHEN play_type IN ('punt','field_goal') THEN 1 ELSE 0 "
            "END) AS kicks, SUM(CASE WHEN play_type = 'no_play' THEN 1 ELSE 0 END) AS nullified "
            "FROM plays WHERE posteam='KC' AND season=2024 AND season_type='REG' "
            "AND down=4")["rows"][0]
        assert (down4["n"], down4["kicks"], down4["nullified"]) == (118, 95, 6)
        rows = nfl_team_stats(team="KC", season=2024, season_type="REG", side="situational")["situational"]
        fourth = next(r for r in rows if r["situation"] == "4th Down")
        assert (fourth["plays"], fourth["conversions"], fourth["conversion_attempts"]) == (17, 12, 17)
        assert fourth["conversion_pct"] == 70.6
        # A row with no third- or fourth-down attempt in it has a null rate, not a 0.0 rate.
        # ('Standard' and 'Red Zone' are not such rows: _SITUATION_EXPR splits 3rd down only at
        # 3rd & short and 3rd & long, so 3rd & medium lands in them.)
        two_minute = next(r for r in rows if r["situation"] == "Two Minute Drill")
        assert (two_minute["plays"], two_minute["conversion_attempts"]) == (13, 0)
        assert two_minute["conversion_pct"] is None

    def test_a_play_a_penalty_nullified_is_not_an_attempt(self):
        """play_type = 'no_play' is a down a penalty undid, and the same down was replayed, so the
        official box counts the replay and not that attempt. The 2024 regular season holds 1,612 of
        them inside the scrimmage population and they gain 35 yards between them (0.02 each).
        Kansas City's avg_epa reads 0.071 with them and 0.072 without, and the table says 0.072."""
        league = nfl_query(
            "SELECT COUNT(*) AS n, ROUND(SUM(COALESCE(yards_gained,0)), 1) AS yards FROM plays "
            "WHERE season=2024 AND season_type='REG' AND play_type = 'no_play' "
            "AND (COALESCE(pass,0)=1 OR COALESCE(rush,0)=1) AND epa IS NOT NULL "
            "AND COALESCE(two_point_attempt,0)<>1")["rows"][0]
        assert (league["n"], league["yards"]) == (1612, 35.0)
        kc = nfl_query(
            "SELECT ROUND(AVG(epa), 3) AS with_them, "
            "ROUND(AVG(CASE WHEN play_type <> 'no_play' THEN epa END), 3) AS without FROM plays "
            "WHERE posteam='KC' AND season=2024 AND season_type='REG' "
            "AND (COALESCE(pass,0)=1 OR COALESCE(rush,0)=1) AND epa IS NOT NULL "
            "AND COALESCE(two_point_attempt,0)<>1")["rows"][0]
        assert (kc["with_them"], kc["without"]) == (0.071, 0.072)
        got = nfl_team_stats(team="KC", season=2024, season_type="REG", side="offense")["offense"][0]
        assert got["avg_epa"] == 0.072

    def test_the_three_yards_per_play_rates_divide_the_totals_in_the_row(self):
        """The same row used to hold two answers for one question. Kansas City 2024 regular season
        said 5.02 yards per play, 4.26 per rush and 5.92 per pass while carrying 5570 yards over
        1091 plays, 1790 over 450 carries, and 3780 over 641 dropbacks - 5.11, 3.98 and 5.90.
        Baltimore is here too because its spread was larger: 6.65 against 6.85, 5.89 against 5.76,
        8.14 against 8.05."""
        for team, ypp, ypr, yppass in (("BAL", 6.85, 5.76, 8.05), ("KC", 5.11, 3.98, 5.9)):
            got = nfl_team_stats(team=team, season=2024, season_type="REG",
                                 side="offense")["offense"][0]
            assert (got["yards_per_play"], got["yards_per_rush"], got["yards_per_pass"]) == (
                ypp, ypr, yppass)
            assert got["yards_per_play"] == round(got["total_yards"] / got["total_plays"], 2)
            assert got["yards_per_rush"] == round(got["rush_yards"] / got["rush_plays"], 2)
            assert got["yards_per_pass"] == round(
                got["pass_yards"] / (got["pass_plays"] + got["sacks_suffered"]), 2)

    def test_the_defence_counts_come_from_the_opponents_box_row(self):
        """The defence was built from plays while the offence in front of it was built from the box,
        so a team's defence and its opponents' offence disagreed: Kansas City 2024 regular season
        showed 1069 plays against and 19 takeaways against the opponents' 1026 and 20, and the
        postseason 209 and 1 against 203 and 1. Philadelphia: 1030 and 25 against 999 and 26, and
        267 and 11 against 256 and 13."""
        for team, season_type, plays, tovs, yards in (
            ("KC", "REG", 1026, 20, 5451), ("KC", "POST", 203, 1, 1055),
            ("PHI", "REG", 999, 26, 4732), ("PHI", "POST", 256, 13, 1329),
        ):
            box = nfl_query(
                f"SELECT SUM(attempts + sacks_suffered + carries) AS plays, "
                f"SUM(passing_interceptions + fumbles_lost_total) AS tovs, "
                f"SUM(passing_yards + rushing_yards + sack_yards_lost) AS yards "
                f"FROM team_stats_raw WHERE season = 2024 AND season_type = '{season_type}' "
                f"AND opponent_team = '{team}'")["rows"][0]
            assert (box["plays"], box["tovs"], box["yards"]) == (plays, tovs, yards)
            got = nfl_team_stats(team=team, season=2024, season_type=season_type,
                                 side="defense")["defense"][0]
            assert (got["plays_against"], got["turnovers_forced"], got["yards_allowed"]) == (
                plays, tovs, yards)
        # The two populations do differ: 1017 scrimmage plays were recorded against Kansas City in
        # the 2024 regular season, against the 1026 the opponents' box rows add up to. What no box
        # carries still comes from plays.
        scrim = nfl_query(
            "SELECT COUNT(*) AS n FROM plays WHERE defteam='KC' AND season=2024 "
            "AND season_type='REG' AND (COALESCE(pass,0)=1 OR COALESCE(rush,0)=1) "
            "AND epa IS NOT NULL AND COALESCE(two_point_attempt,0)<>1 "
            "AND COALESCE(play_type,'')<>'no_play'")["rows"][0]["n"]
        assert scrim == 1017
        got = nfl_team_stats(team="KC", season=2024, season_type="REG", side="defense")["defense"][0]
        # The rate divides the two box totals already in the row: 5451 / 1026. Averaged over the
        # 1017 scrimmage plays it said 5.37, disagreeing with the yards and plays next to it.
        assert (got["yards_per_play_allowed"], got["avg_epa_allowed"]) == (5.31, 0.011)
        assert got["yards_per_play_allowed"] == round(got["yards_allowed"] / got["plays_against"], 2)

    def test_no_defence_row_holds_two_answers_for_one_question(self):
        """Every rate in team_defense_stats now divides two numbers that sit in the same row.
        yards_per_play_allowed was the exception: averaged over scrimmage plays it disagreed with the
        yards and the plays next to it on all 46 rows of 2024 - Kansas City regular season 5.37
        against 5451 / 1026 = 5.31, Baltimore postseason 5.50 against 553 / 104 = 5.32 - by as much
        as 0.33. avg_epa_allowed is the one figure left on plays: no box carries expected points, and
        the row holds no companion count that it claims to divide.

        The audit turned up one pair that is not a rate but reads like one. plays_against adds the
        opponents' sacks_suffered while sacks reports this defence's own def_sacks, and the two ends
        of one sack are not always charged the same way: 9 of the 32 rows of the 2024 regular season
        differ (Dallas 51 def_sacks against 52 sacks suffered by its opponents), by at most 2, so
        plays_against cannot be decomposed out of the row. turnovers_forced has no such problem - its        interception half equals interceptions on every one of the 46 team-season rows of 2024."""
        # Both checks run in SQL: Python rounds 117 / 208 = 56.25 to even (56.2) and DuckDB rounds
        # it away from zero (56.3), which is the value the table holds.
        bad = nfl_query(
            "SELECT COUNT(*) AS n FROM team_defense_stats WHERE season_year=2024 AND ("
            "yards_per_play_allowed IS DISTINCT FROM "
            "ROUND(1.0*yards_allowed/NULLIF(plays_against,0), 2) OR third_down_stop_pct "
            "IS DISTINCT FROM ROUND(100.0*third_down_stops/NULLIF(third_down_attempts,0), 1))")["rows"][0]["n"]
        assert bad == 0
        assert nfl_query("SELECT COUNT(*) AS n FROM team_defense_stats "
                         "WHERE season_year=2024")["rows"][0]["n"] == 46
        rows_ = nfl_query(
            "SELECT team, season_type, yards_allowed, plays_against, yards_per_play_allowed "
            "FROM team_defense_stats WHERE season_year=2024")["rows"]
        kc = next(r for r in rows_ if r["team"] == "KC" and r["season_type"] == "REG")
        assert (kc["yards_allowed"], kc["plays_against"], kc["yards_per_play_allowed"]) == (
            5451, 1026, 5.31)
        bal = next(r for r in rows_ if r["team"] == "BAL" and r["season_type"] == "POST")
        assert (bal["yards_allowed"], bal["plays_against"], bal["yards_per_play_allowed"]) == (
            553, 104, 5.32)
        sack_gap = nfl_query(
            "SELECT SUM(CASE WHEN o.ds<>p.ss THEN 1 ELSE 0 END) AS gap, MAX(ABS(o.ds-p.ss)) AS "
            "worst, SUM(CASE WHEN o.di<>p.pi THEN 1 ELSE 0 END) AS int_gap, COUNT(*) AS n "
            "FROM (SELECT team, season, season_type, SUM(def_sacks) ds, SUM(def_interceptions) di "
            "FROM team_stats_raw WHERE season=2024 GROUP BY 1,2,3) o JOIN (SELECT opponent_team "
            "team, season, season_type, SUM(sacks_suffered) ss, SUM(passing_interceptions) pi "
            "FROM team_stats_raw WHERE season=2024 GROUP BY 1,2,3) p "
            "USING (team, season, season_type)")["rows"][0]
        assert (sack_gap["gap"], sack_gap["worst"], sack_gap["int_gap"], sack_gap["n"]) == (
            9, 2, 0, 46)

    def test_a_regular_season_value_excludes_the_playoffs(self):
        reg = nfl_team_stats(team="KC", season=2024, season_type="REG", side="both")
        post = nfl_team_stats(team="KC", season=2024, season_type="POST", side="both")
        assert (reg["offense"][0]["total_plays"], post["offense"][0]["total_plays"]) == (1091, 162)
        assert (reg["offense"][0]["touchdowns"], post["offense"][0]["touchdowns"]) == (41, 9)
        assert (reg["offense"][0]["points_for"], post["offense"][0]["points_for"]) == (385, 77)
        assert (reg["defense"][0]["sacks"], post["defense"][0]["sacks"]) == (39, 12)
        # The merged number the old table gave for the same team-season.
        both = nfl_team_stats(team="KC", season=2024, side="offense")["offense"]
        assert [r["total_plays"] for r in both] == [162, 1091]
        assert sum(r["total_plays"] for r in both) == 1253

    def test_points_come_from_the_schedule(self):
        sched = nfl_query(
            "SELECT SUM(home_score) AS kc FROM (SELECT home_score FROM schedules WHERE season=2024 "
            "AND game_type='REG' AND home_team='KC' UNION ALL SELECT away_score FROM schedules "
            "WHERE season=2024 AND game_type='REG' AND away_team='KC')")["rows"][0]["kc"]
        got = nfl_team_stats(team="KC", season=2024, season_type="REG", side="both")
        assert sched == 385
        assert got["offense"][0]["points_for"] == 385
        assert got["offense"][0]["points_against"] == 326
        assert got["defense"][0]["points_allowed"] == 326


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflPlayerStats:
    """The numbers must be nflverse's official counting stats, read from the week-level
    `player_stats` table and summed per season, not counts of `plays` rows."""

    MAHOMES = "00-0033873"
    JEFFERSON = "00-0036322"
    HENRY = "00-0032764"

    @staticmethod
    def player_stats_totals(player_id, columns, season=None, season_type=None):
        """Ground truth: the same columns summed straight out of player_stats."""
        where = f"player_id = '{player_id}'"
        if season:
            where += f" AND season = {int(season)}"
        if season_type:
            where += f" AND season_type = '{season_type}'"
        select = ", ".join(f"COALESCE(SUM({col}), 0) AS {col}" for col in columns)
        return nfl_query(f"SELECT {select} FROM player_stats WHERE {where}")["rows"][0]

    def test_passing_stats(self):
        result = nfl_player_stats(player_name="P.Mahomes", stat_type="passing")
        assert "error" not in result
        assert len(result["seasons"]) > 0
        season = result["seasons"][0]
        assert "attempts" in season
        assert "completions" in season
        assert "avg_epa" in season

    def test_passing_counts_match_player_stats_for_a_qb_with_many_sacks(self):
        """Mahomes 2024: counting pass plays gave 617 attempts, his official count is 581."""
        result = nfl_player_stats(player_name="Patrick Mahomes", season=2024, season_type="REG",
                                  stat_type="passing")
        season = result["seasons"][0]
        expected = self.player_stats_totals(
            self.MAHOMES,
            ["attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
             "sacks_suffered", "sack_yards_lost", "passing_air_yards", "passing_first_downs",
             "passing_10", "passing_20", "passing_40", "passing_2pt_conversions",
             "passing_yards_after_catch"],
            season=2024, season_type="REG")

        assert (season["attempts"], season["completions"]) == (581, 392)
        assert season["attempts"] == expected["attempts"]
        assert season["yards"] == expected["passing_yards"] == 3928
        assert season["touchdowns"] == expected["passing_tds"] == 26
        assert season["interceptions"] == expected["passing_interceptions"] == 11
        assert season["sacks"] == expected["sacks_suffered"] == 36
        assert season["sack_yards"] == expected["sack_yards_lost"] == -239
        assert season["air_yards"] == expected["passing_air_yards"]
        assert season["first_downs"] == expected["passing_first_downs"]
        assert season["pass_10_plus"] == expected["passing_10"]
        assert season["pass_20_plus"] == expected["passing_20"]
        assert season["pass_40_plus"] == expected["passing_40"]
        assert season["two_point_conversions"] == expected["passing_2pt_conversions"]
        assert season["yards_after_catch"] == expected["passing_yards_after_catch"]
        assert season["games"] == 16
        # the gap the fix closes: attempts + sacks is the number counting plays produced
        assert season["attempts"] + season["sacks"] == 617

    def test_rushing_counts_include_kneels_that_plays_row_counts_miss(self):
        """Mahomes 2024: 38 `play_type = 'run'` rows, 58 official carries, the 20 others kneels."""
        result = nfl_player_stats(player_name="Patrick Mahomes", season=2024, season_type="REG",
                                  stat_type="rushing")
        season = result["seasons"][0]
        expected = self.player_stats_totals(
            self.MAHOMES, ["carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
                           "rushing_fumbles_lost", "rushing_first_downs", "rushing_10",
                           "rushing_20", "rushing_40", "rushing_2pt_conversions"],
            season=2024, season_type="REG")
        run_rows = nfl_query(
            "SELECT COUNT(*) AS n FROM plays WHERE rusher_player_id = "
            f"'{self.MAHOMES}' AND season = 2024 AND season_type = 'REG' AND play_type = 'run'"
        )["rows"][0]["n"]
        kneel_rows = nfl_query(
            "SELECT COUNT(*) AS n FROM plays WHERE rusher_player_id = "
            f"'{self.MAHOMES}' AND season = 2024 AND season_type = 'REG' AND play_type = 'qb_kneel'"
        )["rows"][0]["n"]

        assert season["carries"] == expected["carries"] == 58
        assert kneel_rows == 20 and run_rows == 38
        assert season["carries"] == run_rows + kneel_rows
        assert season["yards"] == expected["rushing_yards"] == 307
        assert season["touchdowns"] == expected["rushing_tds"] == 2
        assert season["fumbles"] == expected["rushing_fumbles"]
        assert season["fumbles_lost"] == expected["rushing_fumbles_lost"]
        assert season["first_downs"] == expected["rushing_first_downs"]
        assert season["rush_10_plus"] == expected["rushing_10"]
        assert season["yards_per_carry"] == round(307 / 58, 1)

    def test_rushing_counts_match_player_stats_for_a_running_back(self):
        result = nfl_player_stats(player_name="D.Henry", season=2024, season_type="REG",
                                  stat_type="rushing")
        season = result["seasons"][0]
        expected = self.player_stats_totals(
            self.HENRY, ["carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
                         "rushing_fumbles_lost"], season=2024, season_type="REG")

        assert season["carries"] == expected["carries"] == 325
        assert season["yards"] == expected["rushing_yards"] == 1921
        assert season["touchdowns"] == expected["rushing_tds"] == 16
        assert season["fumbles_lost"] == expected["rushing_fumbles_lost"]
        # 2024 playoff games are a separate row, not extra carries on this one
        assert season["games"] == 17

    def test_receiving_counts_match_player_stats(self):
        result = nfl_player_stats(player_name="Justin Jefferson", season=2024, season_type="REG",
                                  stat_type="receiving")
        season = result["seasons"][0]
        expected = self.player_stats_totals(
            self.JEFFERSON, ["targets", "receptions", "receiving_yards", "receiving_tds",
                             "receiving_yards_after_catch", "receiving_air_yards",
                             "receiving_first_downs", "receiving_fumbles", "receiving_10",
                             "receiving_20", "receiving_40"], season=2024, season_type="REG")

        assert (season["targets"], season["receptions"]) == (154, 103)
        assert season["targets"] == expected["targets"]
        assert season["yards"] == expected["receiving_yards"] == 1533
        assert season["touchdowns"] == expected["receiving_tds"] == 10
        assert season["yards_after_catch"] == expected["receiving_yards_after_catch"]
        assert season["air_yards"] == expected["receiving_air_yards"]
        assert season["first_downs"] == expected["receiving_first_downs"]
        assert season["rec_10_plus"] == expected["receiving_10"]
        assert season["rec_20_plus"] == expected["receiving_20"]
        assert season["rec_40_plus"] == expected["receiving_40"]
        assert season["yards_per_reception"] == round(1533 / 103, 1)
        assert season["yards_per_target"] == round(1533 / 154, 1)

    def test_avg_yac_divides_receptions_and_not_targets(self):
        """Ja'Marr Chase 2024: 787 yards after catch over 127 receptions is 6.2; over his 175
        targets it would be 4.5. Only the first is yards after catch per reception."""
        season = nfl_player_stats(player_name="Ja'Marr Chase", season=2024, season_type="REG",
                                  stat_type="receiving")["seasons"][0]
        totals = self.player_stats_totals(
            "00-0036900", ["targets", "receptions", "receiving_yards_after_catch"],
            season=2024, season_type="REG")

        assert totals == {"targets": 175, "receptions": 127, "receiving_yards_after_catch": 787}
        assert (season["targets"], season["receptions"], season["yards_after_catch"]) == (175, 127, 787)
        assert season["avg_yac"] == round(787 / 127, 1) == 6.2 != round(787 / 175, 1)

    def test_every_rate_divides_summed_totals_and_no_weekly_rate_is_averaged(self):
        result = nfl_player_stats(player_name="Patrick Mahomes", season=2024, season_type="REG",
                                  stat_type="passing")
        season = result["seasons"][0]
        att, comp, yds = season["attempts"], season["completions"], season["yards"]
        sacks, sack_yds = season["sacks"], season["sack_yards"]

        assert season["comp_pct"] == round(100.0 * comp / att, 1) == 67.5
        assert season["yards_per_attempt"] == round(yds / att, 1)
        assert season["net_yards"] == yds + sack_yds == 3689
        assert season["net_yards_per_attempt"] == round((yds + sack_yds) / (att + sacks), 1)
        assert season["sack_rate"] == round(sacks / (att + sacks), 3)
        assert season["td_pct"] == round(100.0 * season["touchdowns"] / att, 1)
        assert season["int_pct"] == round(100.0 * season["interceptions"] / att, 1)
        assert season["avg_air_yards"] == round(season["air_yards"] / att, 1)
        assert abs(season["epa"] - 71.133) < 1e-3
        assert season["avg_epa"] == round(season["epa"] / (att + sacks), 3)

    def test_avg_cpoe_covers_exactly_the_window_of_the_row_it_sits_on(self):
        """player_stats.passing_cpoe is a weekly mean and the table records no play count for it,
        so the season mean comes from plays, over the caller's own window. Drop the season_type
        filter there and the answer is 2.298 over 634 plays, which is neither value below; drop
        the grouping and every row comes back with no partner to match."""
        calls = [nfl_player_stats(player_name="Patrick Mahomes", season=2024, season_type=stype)
                 for stype in ("POST", "REG")]
        truth = nfl_query(
            "SELECT season_type, ROUND(AVG(cpoe), 3) AS cpoe, COUNT(cpoe) AS n FROM plays"
            f" WHERE passer_player_id = '{self.MAHOMES}' AND season = 2024 AND cpoe IS NOT NULL"
            " GROUP BY season_type ORDER BY season_type")["rows"]

        assert [(r["season_type"], r["cpoe"], r["n"]) for r in truth] == [
            ("POST", 0.486, 81), ("REG", 2.563, 553)]
        assert {(row["season_type"], row["cpoe_plays"]): row["avg_cpoe"]
                for call in calls for row in call["seasons"]} == {
                    ("REG", 553): 2.563, ("POST", 81): 0.486}
        assert calls[1]["metric_sources"]["avg_cpoe"] == tools._CPOE_MEAN_SOURCE

    def test_a_season_range_gives_one_row_per_season_and_each_matches_the_table(self):
        """One row per season, each equal to that season's own sum. The seasons come from the
        database, so this also passes on a single-season database."""
        seasons = [row["season"] for row in nfl_query(
            "SELECT DISTINCT season FROM player_stats WHERE player_id = "
            f"'{self.MAHOMES}' AND season_type = 'REG' ORDER BY season")["rows"]]
        result = nfl_player_stats(player_name="Patrick Mahomes", season_from=seasons[0],
                                  season_to=seasons[-1], season_type="REG", stat_type="passing")
        totals = nfl_query(
            "SELECT season, SUM(attempts) AS attempts FROM player_stats WHERE player_id = "
            f"'{self.MAHOMES}' AND season_type = 'REG' GROUP BY season ORDER BY season")["rows"]

        assert [row["season"] for row in result["seasons"]] == seasons
        assert [row["attempts"] for row in result["seasons"]] == [t["attempts"] for t in totals]
        if seasons[0] == 2024:                       # the 2024-only CI database starts here too
            assert result["seasons"][0]["attempts"] == 581

    def test_regular_season_row_excludes_playoff_games(self):
        result = nfl_player_stats(player_name="Patrick Mahomes", season=2024, stat_type="passing")
        by_type = {row["season_type"]: row for row in result["seasons"]}
        assert sorted(by_type) == ["POST", "REG"]
        assert by_type["REG"]["attempts"] == 581 and by_type["POST"]["attempts"] == 83
        assert by_type["REG"]["games"] == 16 and by_type["POST"]["games"] == 3

    def test_the_response_names_the_source_of_every_metric(self):
        result = nfl_player_stats(player_name="Patrick Mahomes", season=2024, stat_type="rushing")
        assert "player_stats" in result["source"] and "week" in result["source"]
        assert result["unavailable"] == {}              # this database has every column
        assert set(result["metric_sources"]) == {
            key for key in result["seasons"][0]
            if key not in ("player_id", "season", "season_type", "label")}
        assert all("player_stats." in src for src in result["metric_sources"].values())
        assert result["metric_sources"]["carries"] == "SUM(player_stats.carries)"

    def test_unknown_player_is_not_found(self):
        result = nfl_player_stats(player_name="ZZZZNOTAPLAYER", stat_type="passing")
        assert "error" not in result
        assert result["not_found"] is True
        assert isinstance(result["did_you_mean"], list)


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflCompare:
    def test_compare_teams(self):
        result = nfl_compare(entity1="KC", entity2="BAL", compare_type="team", season=2024)
        assert "error" not in result
        assert "KC" in result
        assert "BAL" in result
        assert "offense" in result["KC"]
        assert "defense" in result["BAL"]

    def test_compare_teams_honours_season_type(self):
        """Team mode used to take season_type and ignore it."""
        both = nfl_compare(entity1="KC", entity2="PHI", compare_type="team", season=2024)
        assert [(r["season_type"], r["total_plays"]) for r in both["KC"]["offense"]] == [
            ("POST", 162), ("REG", 1091)]
        post = nfl_compare(entity1="KC", entity2="PHI", compare_type="team", season=2024,
                           season_type="POST")
        assert [(s["offense"][0]["season_type"], s["offense"][0]["total_plays"])
                for s in (post["KC"], post["PHI"])] == [("POST", 162), ("POST", 254)]
        reg = nfl_compare(entity1="KC", entity2="PHI", compare_type="team", season=2024,
                          season_type="REG")
        assert [s["offense"][0]["total_plays"] for s in (reg["KC"], reg["PHI"])] == [1091, 1114]
        assert [s["defense"][0]["sacks"] for s in (reg["KC"], reg["PHI"])] == [39, 41]
        for side in (reg["KC"], reg["PHI"]):
            assert set(side) == {"offense", "defense", "sources"}
        assert "scrimmage plays only" in reg["KC"]["sources"]["plays"]

    def test_compare_team_filters_stack_in_the_right_order(self):
        """Season range and season type share one parameter list; the order has to match."""
        got = nfl_compare(entity1="KC", entity2="PHI", compare_type="team",
                          season_from=2024, season_to=2025, season_type="REG")
        offense = got["KC"]["offense"]
        years = [r["season_year"] for r in offense]
        # The CI database holds 2024 only, so assert the shape of the window and pin its 2024 row.
        assert years == sorted(set(years)) and set(years) <= {2024, 2025} and 2024 in years
        assert {r["season_type"] for r in offense} == {"REG"}
        assert [r["total_plays"] for r in offense if r["season_year"] == 2024] == [1091]

    def test_compare_players(self):
        result = nfl_compare(
            entity1="Patrick Mahomes", entity2="Justin Herbert",
            compare_type="player", season=2024
        )
        assert "error" not in result
        sides = result["players"]
        assert [s["requested"] for s in sides] == ["Patrick Mahomes", "Justin Herbert"]
        assert [s["player"]["gsis_id"] for s in sides] == ["00-0033873", "00-0036355"]
        assert "passing" in sides[0] and "passing" in sides[1]

    def test_compare_players_uses_the_official_player_stats_rows(self):
        """Player mode follows nfl_player_stats: same numbers, one row per season type."""
        result = nfl_compare(entity1="Patrick Mahomes", entity2="Justin Herbert",
                             compare_type="player", season=2024, season_type="REG")
        sides = result["players"]
        assert [s["passing"][0]["season_type"] for s in sides] == ["REG", "REG"]
        for side in sides:
            assert "player_stats" in side["source"] and "summed per season" in side["source"]
            assert side["metric_sources"]["passing"]["attempts"] == "SUM(player_stats.attempts)"
            assert set(side["metric_sources"]) <= {k for k in side if k in
                                                  ("passing", "rushing", "receiving")}
            for block in ("passing", "rushing", "receiving"):
                for row in side.get(block, []):
                    assert row["season_type"] == "REG"
                    assert row["player_id"] == side["player"]["gsis_id"]
        assert sides[0]["passing"][0]["attempts"] == 581      # Mahomes, 36 sacks excluded
        assert sides[0]["passing"][0]["sacks"] == 36

    def test_compare_players_without_a_season_type_keeps_the_rows_apart(self):
        result = nfl_compare(entity1="Patrick Mahomes", entity2="Justin Herbert",
                             compare_type="player", season=2024)
        side = result["players"][0]
        post, reg = side["passing"]                      # one row per season type, never one sum
        assert (post["season_type"], reg["season_type"]) == ("POST", "REG")
        assert (reg["attempts"], post["attempts"]) == (581, 83)
        assert side["passing"][0]["label"] == side["player"]["label"]

    def test_a_season_whose_counting_stats_are_all_zero_is_gone_from_both_tools(self):
        """Jefferson's 1 playoff game of passing zeros, Mahomes' 3 of receiving zeros and
        Jefferson's 1 of rushing zeros are the same case: `games` alone is no stat, and neither is
        a derived epa figure - Curtis Samuel's 3 playoff rushing games are carries 0, yards 0 and
        epa -0.947, while his 6 receptions that same postseason stay."""
        both = nfl_compare("Patrick Mahomes", "Justin Jefferson", compare_type="player", season=2024)
        post = nfl_compare("Patrick Mahomes", "Justin Jefferson", compare_type="player",
                           season=2024, season_type="POST")["players"]
        assert [(r["season_type"], r["attempts"]) for r in both["players"][1]["passing"]] == [("REG", 1)]
        assert "receiving" not in post[0] and post[0]["rushing"][0]["carries"] == 22
        assert "rushing" not in post[1] and post[1]["receiving"][0]["targets"] == 8
        samuel = nfl_player_stats(player_id="00-0033282", season=2024, season_type="POST",
                                  stat_type="rushing")
        assert samuel["seasons"] == [] and samuel["unavailable"] == {}
        side = nfl_compare(entity1="", entity2="", compare_type="player", player_id1="00-0033282",
                           season=2024, season_type="POST")["players"][0]
        assert "rushing" not in side and side["receiving"][0]["receptions"] == 6

    # A zero count that leads a block does not empty it: Josh Allen's 7-yard receiving touchdown
    # off 0 targets, Justin Watson's 2 two-point conversions and Penei Sewell's sack off 0 of them.
    @pytest.mark.parametrize("player_id,season,season_type,stat_type,expect", [
        ("00-0034857", 2024, "REG", "receiving", {"targets": 0, "yards": 7, "touchdowns": 1}),
        ("00-0034386", 2024, "POST", "receiving", {"targets": 0, "two_point_conversions": 2}),
        ("00-0032701", 2024, "REG", "receiving", {"targets": 0, "yards": 2}),
        ("00-0036880", 2024, "REG", "passing", {"attempts": 0, "sacks": 1, "sack_yards": -1}),
        ("00-0035250", 2025, "REG", "passing", {"attempts": 0, "two_point_conversions": 1}),
    ])
    def test_a_block_stays_visible_when_any_counting_stat_is_not_zero(
            self, player_id, season, season_type, stat_type, expect):
        with tools.get_db_connection() as con:
            if not con.execute(f"SELECT 1 FROM player_stats WHERE season = {season} LIMIT 1").fetchone():
                pytest.skip(f"season {season} is not in this database")   # 2025 is not in the CI DB
        rows = nfl_player_stats(player_id=player_id, season=season, season_type=season_type,
                                stat_type=stat_type)["seasons"]
        assert len(rows) == 1 and {key: rows[0][key] for key in expect} == expect
        side = nfl_compare(entity1="", entity2="", compare_type="player", player_id1=player_id,
                           season=season, season_type=season_type)["players"][0]
        assert side[stat_type] == rows       # nfl_compare shows the same row nfl_player_stats gave

    def test_compare_players_gives_the_side_it_cannot_settle_to_the_caller(self):
        # 'Lamar Jackson' is the Ravens QB and an Falcons CB in 2024, so that side
        # answers with candidates while the other side still gets its stats.
        result = nfl_compare(
            entity1="P.Mahomes", entity2="Lamar Jackson",
            compare_type="player", season=2024
        )
        sides = result["players"]
        assert sides[0]["player"]["gsis_id"] == "00-0033873"
        assert "passing" in sides[0]
        assert sides[1]["ambiguous"] is True
        assert {c["full_name"] for c in sides[1]["candidates"]} == {"Lamar Jackson"}
        assert {c["team"] for c in sides[1]["candidates"]} == {"BAL", "ATL"}

@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflStatus:
    def test_returns_plays_key(self):
        result = nfl_status()
        assert "error" not in result
        assert "plays" in result

    def test_plays_has_expected_fields(self):
        plays = nfl_status()["plays"]
        assert "total_plays" in plays
        assert "season_range" in plays
        assert plays["total_plays"] > 0

    def test_returns_datasets_key(self):
        result = nfl_status()
        assert "datasets" in result

    def test_datasets_has_expected_fields(self):
        datasets = nfl_status()["datasets"]
        assert "total_loaded" in datasets
        assert "last_refreshed" in datasets
        assert "loaded" in datasets
        assert datasets["total_loaded"] > 0
        assert isinstance(datasets["loaded"], list)

    def test_datasets_loaded_entries_have_fields(self):
        loaded = nfl_status()["datasets"]["loaded"]
        assert len(loaded) > 0
        for entry in loaded:
            assert "dataset_id" in entry
            assert "table_name" in entry
            assert "total_rows" in entry


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflCatalog:
    def test_returns_datasets_list(self):
        result = nfl_catalog()
        assert "error" not in result
        assert "datasets" in result
        assert len(result["datasets"]) > 0

    def test_includes_row_counts(self):
        result = nfl_catalog()
        for ds in result["datasets"]:
            assert "dataset_id" in ds
            assert "total_rows" in ds
            assert ds["total_rows"] > 0

    def test_total_datasets_matches_list(self):
        result = nfl_catalog()
        assert result["total_datasets"] == len(result["datasets"])


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflRoster:
    def test_filter_by_team_and_season(self):
        result = nfl_roster(team="KC", season=2024)
        assert "error" not in result
        assert result["count"] > 0
        assert all(r["team"] == "KC" for r in result["players"])
        assert all(r["season"] == 2024 for r in result["players"])

    def test_filter_by_position(self):
        result = nfl_roster(team="KC", season=2024, position="QB")
        assert "error" not in result
        assert result["count"] > 0

    def test_returns_expected_fields(self):
        result = nfl_roster(team="KC", season=2024)
        row = result["players"][0]
        for field in ("full_name", "position", "team", "season"):
            assert field in row

    def test_unknown_team_is_reported_as_unknown(self):
        result = nfl_roster(team="ZZZ", season=2024)
        assert result["error"] == "Unknown team 'ZZZ'."


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflInjuries:
    def test_filter_by_team_and_season(self):
        result = nfl_injuries(team="KC", season=2024)
        assert "error" not in result
        assert result["count"] > 0

    def test_filter_by_week(self):
        result = nfl_injuries(team="KC", season=2024, week=10)
        assert "error" not in result
        assert all(r["week"] == 10 for r in result["injuries"])

    def test_filter_by_player(self):
        result = nfl_injuries(player="Mahomes", season=2024)
        assert "error" not in result

    def test_returns_expected_fields(self):
        result = nfl_injuries(team="KC", season=2024, week=1)
        if result["count"] > 0:
            row = result["injuries"][0]
            for field in ("full_name", "position", "team", "season", "week", "report_status"):
                assert field in row


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflSchedule:
    def test_filter_by_team(self):
        result = nfl_schedule(team="KC", season=2024)
        assert "error" not in result
        assert result["count"] > 0
        assert all(
            r["home_team"] == "KC" or r["away_team"] == "KC"
            for r in result["games"]
        )

    def test_filter_by_week(self):
        result = nfl_schedule(season=2024, week=1)
        assert "error" not in result
        assert result["count"] > 0
        assert all(r["week"] == 1 for r in result["games"])

    def test_returns_expected_fields(self):
        result = nfl_schedule(team="KC", season=2024)
        row = result["games"][0]
        for field in ("game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"):
            assert field in row

    def test_season_type_filter(self):
        result = nfl_schedule(team="KC", season=2024, season_type="REG")
        assert "error" not in result
        assert all(r["game_type"] == "REG" for r in result["games"])


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflSnapCounts:
    def test_filter_by_team_and_season(self):
        result = nfl_snap_counts(team="KC", season=2024)
        assert "error" not in result
        assert result["count"] > 0

    def test_filter_by_player(self):
        result = nfl_snap_counts(player="Kelce", season=2024)
        assert "error" not in result
        assert result["count"] > 0

    def test_filter_by_week(self):
        result = nfl_snap_counts(team="KC", season=2024, week=1)
        assert "error" not in result
        assert all(r["week"] == 1 for r in result["snap_counts"])

    def test_returns_expected_fields(self):
        result = nfl_snap_counts(team="KC", season=2024, week=1)
        if result["count"] > 0:
            row = result["snap_counts"][0]
            for field in ("player", "position", "team", "season", "week", "offense_snaps", "offense_pct"):
                assert field in row


class TestNflFantasyOpportunity:
    def test_no_filters_returns_results(self):
        result = nfl_fantasy_opportunity(season=2024)
        assert "error" not in result

    def test_filter_by_player(self):
        result = nfl_fantasy_opportunity(player="Jefferson", season=2024)
        assert "error" not in result

    def test_filter_by_team_and_position(self):
        result = nfl_fantasy_opportunity(team="KC", position="WR", season=2024)
        assert "error" not in result

    def test_future_season_returns_dict(self):
        result = nfl_fantasy_opportunity(season=2099)
        assert isinstance(result, dict)

    def test_filter_by_week(self):
        result = nfl_fantasy_opportunity(team="KC", season=2024, week=1)
        assert "error" not in result

    def test_db_error_returns_error_dict(self):
        with patch("nfl_mcp.tools._execute", side_effect=duckdb.Error("boom")):
            result = nfl_fantasy_opportunity(season=2024)
        assert "error" in result
        assert "boom" in result["error"]


class TestNflFantasyRankings:
    def test_draft_scope_returns_results(self):
        result = nfl_fantasy_rankings(scope="draft")
        assert "error" not in result
        assert result["scope"] == "draft"
        assert result["count"] > 0

    def test_week_scope_returns_results(self):
        result = nfl_fantasy_rankings(scope="week")
        assert "error" not in result
        assert result["scope"] == "week"

    def test_default_scope_is_draft(self):
        result = nfl_fantasy_rankings()
        assert result["scope"] == "draft"

    def test_filter_by_position_and_team(self):
        result = nfl_fantasy_rankings(scope="draft", position="RB", team="KC")
        assert "error" not in result

    def test_filter_by_player(self):
        result = nfl_fantasy_rankings(scope="week", player="Mahomes")
        assert "error" not in result

    def test_filter_by_ranking_set(self):
        result = nfl_fantasy_rankings(scope="draft", ranking_set="dynasty")
        assert "error" not in result

    def test_draft_rows_have_expected_fields(self):
        result = nfl_fantasy_rankings(scope="draft", limit=1)
        if result["count"] > 0:
            row = result["fantasy_rankings"][0]
            for field in ("player", "position", "team", "ecr", "ranking_set"):
                assert field in row

    def test_invalid_scope_returns_error(self):
        result = nfl_fantasy_rankings(scope="bogus")
        assert "error" in result

    def test_invalid_limit_falls_back(self):
        result = nfl_fantasy_rankings(scope="draft", limit="oops")
        assert "error" not in result

    def test_limit_is_capped(self):
        result = nfl_fantasy_rankings(scope="draft", limit=99999)
        assert result["count"] <= 500

    def test_db_error_returns_error_dict(self):
        with patch("nfl_mcp.tools._execute", side_effect=duckdb.Error("boom")):
            result = nfl_fantasy_rankings(scope="draft")
        assert "error" in result
        assert "boom" in result["error"]


@pytest.mark.usefixtures("require_db")
class TestNflFtnCharting:
    def test_filter_by_team_and_season(self):
        result = nfl_ftn_charting(team="KC", season=2024)
        assert "error" not in result
        assert result["total_plays"] > 0
        c = result["charting"]
        assert 0 <= c["play_action"]["pct"] <= 100
        assert c["avg_defenders_in_box"] is not None

    def test_filter_by_player(self):
        result = nfl_ftn_charting(player="Mahomes", season=2024)
        assert "error" not in result

    def test_filter_by_opponent_and_week(self):
        result = nfl_ftn_charting(opponent="KC", season=2024, week=1, season_type="REG")
        assert "error" not in result

    def test_season_range(self):
        result = nfl_ftn_charting(team="KC", season_from=2022, season_to=2024)
        assert "error" not in result
        assert result["total_plays"] > 0

    def test_no_matching_plays_returns_empty(self):
        result = nfl_ftn_charting(team="KC", season=1999)
        assert result["total_plays"] == 0
        assert result["charting"] == {}


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflTdLuck:
    def test_no_filters_returns_results(self):
        result = nfl_td_luck(season=2024)
        assert "error" not in result
        assert result["count"] > 0

    def test_default_sorted_most_unlucky_first(self):
        result = nfl_td_luck(season=2024, limit=5)
        scores = [r["total_td_luck_score"] for r in result["td_luck"]]
        assert scores == sorted(scores)

    def test_filter_by_player(self):
        result = nfl_td_luck(player="Jefferson", season=2024)
        assert "error" not in result

    def test_filter_by_team_and_position(self):
        result = nfl_td_luck(team="KC", position="TE", season=2024)
        assert "error" not in result

    def test_rows_have_expected_fields(self):
        result = nfl_td_luck(season=2024, limit=1)
        row = result["td_luck"][0]
        for field in ("full_name", "team", "season", "rec_td_luck", "rush_td_luck", "total_td_luck_score"):
            assert field in row

    def test_invalid_limit_falls_back(self):
        result = nfl_td_luck(season=2024, limit="oops")
        assert "error" not in result

    def test_db_error_returns_error_dict(self):
        with patch("nfl_mcp.tools._execute", side_effect=duckdb.Error("boom")):
            result = nfl_td_luck(season=2024)
        assert "boom" in result["error"]


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflRoleTrend:
    def test_filter_by_season_and_week(self):
        result = nfl_role_trend(season=2024, week=8)
        assert "error" not in result
        assert result["count"] > 0

    def test_default_sorted_by_snap_delta_desc(self):
        result = nfl_role_trend(season=2024, week=8, limit=5)
        deltas = [r["snap_pct_delta"] for r in result["role_trend"] if r["snap_pct_delta"] is not None]
        assert deltas == sorted(deltas, reverse=True)

    def test_min_snap_pct_filter(self):
        result = nfl_role_trend(season=2024, week=8, min_snap_pct=50)
        for r in result["role_trend"]:
            assert r["snap_pct"] >= 50

    def test_filter_by_player_team_position(self):
        result = nfl_role_trend(player="Kelce", team="KC", position="TE", season=2024)
        assert "error" not in result

    def test_db_error_returns_error_dict(self):
        with patch("nfl_mcp.tools._execute", side_effect=duckdb.Error("boom")):
            result = nfl_role_trend(season=2024)
        assert "boom" in result["error"]


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflSeparationOpportunity:
    def test_no_filters_returns_results(self):
        result = nfl_separation_opportunity()
        assert "error" not in result
        assert result["count"] > 0

    def test_regression_candidate_filter(self):
        result = nfl_separation_opportunity(regression_candidate=True)
        for r in result["separation_opportunity"]:
            assert r["regression_candidate"] is True

    def test_pre_2016_season_guarded(self):
        result = nfl_separation_opportunity(season=2015)
        assert "error" in result
        assert "2016" in result["error"]

    def test_filter_by_player_team_position(self):
        result = nfl_separation_opportunity(player="Jefferson", team="MIN", position="WR", season=2024)
        assert "error" not in result

    def test_the_three_season_type_scopes(self):
        """The description used to say "pass null for both", and the re-reviewer ran the real MCP
        handler that way: mcp.server validates the arguments against the tool's inputSchema before
        the handler is called, and season_type is typed "string", so the answer was
        `Input validation error: None is not of type 'string'`. The literal 'ALL' is what works:
        REG is the default, POST is one scope, ALL is both."""
        one = nfl_separation_opportunity(player_id="00-0036322", season=2024)
        assert {r["season_type"] for r in one["separation_opportunity"]} == {"REG"}
        post = nfl_separation_opportunity(player_id="00-0036322", season=2024, season_type="POST")
        assert {r["season_type"] for r in post["separation_opportunity"]} == {"POST"}
        both = nfl_separation_opportunity(player_id="00-0036322", season=2024, season_type="ALL")
        assert {r["season_type"] for r in both["separation_opportunity"]} == {"REG", "POST"}
        # Lower case reaches the same place, and ALL is not passed to the database as a filter.
        lower = nfl_separation_opportunity(player_id="00-0036322", season=2024, season_type="all")
        assert len(lower["separation_opportunity"]) == len(both["separation_opportunity"])

    def test_the_ngs_season_summary_row_is_the_figure(self):
        """Justin Jefferson 2024: NGS's own week = 0 summary row says 3.079 yards of average
        separation and a 66.88 catch rate. Averaging his weekly rows unweighted gives 3.073 and
        67.48, and the old code also pulled the postseason weekly rows into that mean. season_type
        defaults to REG, so the plain call returns the regular-season row only."""
        rows = nfl_separation_opportunity(player_id="00-0036322", season=2024)["separation_opportunity"]
        assert {r["season_type"] for r in rows} == {"REG"}
        both = nfl_separation_opportunity(player_id="00-0036322", season=2024,
                                         season_type="ALL")["separation_opportunity"]
        assert {r["season_type"] for r in both} == {"REG", "POST"}
        reg = rows[0]
        summary = nfl_query(
            "SELECT avg_separation, avg_yac_above_expectation, catch_percentage "
            "FROM nextgen_stats_receiving WHERE player_gsis_id='00-0036322' AND season=2024 "
            "AND season_type='REG' AND week=0")["rows"][0]
        assert reg["separation_source"] == "ngs_season_summary"
        assert reg["avg_separation"] == round(summary["avg_separation"], 3)
        assert reg["avg_yac_above_expectation"] == round(summary["avg_yac_above_expectation"], 3)
        assert reg["catch_percentage"] == round(summary["catch_percentage"], 2)
        assert reg["avg_separation"] == 3.079
        assert reg["games"] == 17
        # A summary row is not an estimate from weekly rows, so it reports no sample.
        assert (reg["fallback_weeks"], reg["fallback_targets"]) == (None, None)

    def test_a_fallback_row_carries_its_sample_and_no_regression_claim(self):
        """A.J. Brown's 2024 postseason row is a fallback: separation 2.150, from 3 Next Gen Stats
        weeks holding 20 targets. The sample sits beside the figure, and regression_candidate is
        null, because a sample that thin cannot carry a regression claim. The thinness is upstream -
        NGS published weekly rows for only 291 receiving players in 2024 - and it is normal: of the
        2024 fallback rows, 39 of 69 regular-season rows and 24 of 36 postseason rows rest on a
        single week. On the full database the same query over every season gives 206 regular-season
        fallback rows averaging 1.7 weeks, 124 of them from one week (the caller's examples:
        DeAndre Hopkins, Baltimore 2025, 17 games, separation 2.588 from one week of five targets).
        Counts here are scoped to 2024 so both databases this suite runs on answer the same."""
        rows = nfl_separation_opportunity(player_id="00-0035676", season=2024,
                                         season_type="POST")["separation_opportunity"]
        assert [(r["separation_source"], r["avg_separation"], r["fallback_weeks"],
                 r["fallback_targets"], r["regression_candidate"]) for r in rows] == [
            ("weekly_rows_weighted_by_targets", 2.15, 3, 20, None)]
        thin = nfl_query(
            "SELECT season_type, COUNT(*) n, SUM(CASE WHEN fallback_weeks = 1 THEN 1 ELSE 0 END) "
            "one_week, SUM(CASE WHEN regression_candidate IS NOT NULL THEN 1 ELSE 0 END) claims "
            "FROM player_separation_opportunity WHERE season = 2024 AND separation_source = "
            "'weekly_rows_weighted_by_targets' GROUP BY season_type")["rows"]
        assert {r["season_type"]: (r["n"], r["one_week"], r["claims"]) for r in thin} == {
            "REG": (69, 39, 0), "POST": (36, 24, 0)}

    def test_the_fallback_yac_figure_is_weighted_by_receptions(self):
        """YAC above expectation is a per-reception figure, so receptions weight it in a fallback
        row; separation is measured on every target, so targets still weight that one. A.J. Brown's
        2024 postseason row rests on 3 weeks and 20 targets: reception-weighted YAC is 1.509, while
        weighting it by targets reads 1.050."""
        row = nfl_separation_opportunity(player_id="00-0035676", season=2024,
                                        season_type="POST")["separation_opportunity"][0]
        assert row["separation_source"] == "weekly_rows_weighted_by_targets"
        weekly = nfl_query(
            "SELECT SUM(avg_yac_above_expectation*receptions)/SUM(receptions) AS by_rec, "
            "SUM(avg_yac_above_expectation*targets)/SUM(targets) AS by_tgt, "
            "SUM(avg_separation*targets)/SUM(targets) AS sep_by_tgt, COUNT(DISTINCT week) AS n_weeks, "
            "SUM(targets) AS targets FROM nextgen_stats_receiving WHERE player_gsis_id = "
            "'00-0035676' AND season=2024 AND season_type='POST' AND week > 0")["rows"][0]
        assert (round(weekly["by_rec"], 3), round(weekly["by_tgt"], 3),
                round(weekly["sep_by_tgt"], 3)) == (1.509, 1.05, 2.15)
        assert row["avg_yac_above_expectation"] == round(weekly["by_rec"], 3)
        assert row["avg_separation"] == round(weekly["sep_by_tgt"], 3)
        assert (row["fallback_weeks"], row["fallback_targets"]) == (weekly["n_weeks"], weekly["targets"])
        assert (row["fallback_weeks"], row["fallback_targets"]) == (3, 20)

    def test_season_type_splits_the_rows(self):
        """NGS publishes no postseason summary row, so a postseason row is the weekly figure and
        says so; the regular-season row is the summary."""
        rows = nfl_separation_opportunity(player_id="00-0036322", season=2024,
                                         season_type="post")["separation_opportunity"]
        assert [(r["season_type"], r["games"], r["separation_source"]) for r in rows] == [
            ("POST", 1, "weekly_rows_weighted_by_targets")]
        assert rows[0]["avg_separation"] == 3.243

    def test_invalid_limit_falls_back(self):
        result = nfl_separation_opportunity(limit="oops")
        assert "error" not in result

    def test_db_error_returns_error_dict(self):
        with patch("nfl_mcp.tools._execute", side_effect=duckdb.Error("boom")):
            result = nfl_separation_opportunity(season=2024)
        assert "boom" in result["error"]


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflDropRate:
    def test_filter_by_season(self):
        result = nfl_drop_rate(season=2024)
        assert "error" not in result
        assert result["count"] > 0

    def test_default_sorted_highest_drop_rate_first(self):
        result = nfl_drop_rate(season=2024, min_targets=30, limit=5)
        rates = [r["drop_rate_pct"] for r in result["drop_rate"] if r["drop_rate_pct"] is not None]
        assert rates == sorted(rates, reverse=True)

    def test_pre_2022_season_guarded(self):
        result = nfl_drop_rate(season=2021)
        assert "error" in result
        assert "2022" in result["error"]

    def test_min_targets_filter(self):
        result = nfl_drop_rate(season=2024, min_targets=40)
        for r in result["drop_rate"]:
            assert r["catchable_targets"] >= 40

    def test_filter_by_player_and_team(self):
        result = nfl_drop_rate(player="Pollard", team="TEN", season=2024)
        assert "error" not in result

    def test_db_error_returns_error_dict(self):
        with patch("nfl_mcp.tools._execute", side_effect=duckdb.Error("boom")):
            result = nfl_drop_rate(season=2024)
        assert "boom" in result["error"]


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflContractValue:
    def test_no_filters_returns_results(self):
        result = nfl_contract_value(season=2024)
        assert "error" not in result
        assert result["count"] > 0

    def test_default_sorted_best_value_first(self):
        result = nfl_contract_value(season=2024, min_apy=2, limit=5)
        vals = [r["fp_per_million"] for r in result["contract_value"] if r["fp_per_million"] is not None]
        assert vals == sorted(vals, reverse=True)

    def test_apy_range_filter(self):
        result = nfl_contract_value(season=2024, min_apy=5, max_apy=10)
        for r in result["contract_value"]:
            assert 5 <= r["apy"] <= 10

    def test_filter_by_player_team_position(self):
        result = nfl_contract_value(player="Gibbs", team="DET", position="RB", season=2024)
        assert "error" not in result

    def test_invalid_limit_falls_back(self):
        result = nfl_contract_value(season=2024, limit="oops")
        assert "error" not in result

    def test_db_error_returns_error_dict(self):
        with patch("nfl_mcp.tools._execute", side_effect=duckdb.Error("boom")):
            result = nfl_contract_value(season=2024)
        assert "boom" in result["error"]


@pytest.mark.integration
class TestNflInjuryReturn:
    """Reads injury_return_curve through the rebuilt copy (tests/conftest.py).

    The shared database holds the table in the shape the ingest that built it produced,
    with the old body-part group names, so these tests rebuild it first.
    """

    def test_filter_matches_the_reported_text_any_case(self, rebuilt_db):
        # 'hamstring' is a partial, case-insensitive match; each row carries the text the
        # report gave ('Hamstring'), never the old all-lowercase bucket name.
        result = nfl_injury_return(injury_type="hamstring")
        assert "error" not in result
        assert result["count"] > 0
        names = {r["injury_type"] for r in result["injury_return"]}
        assert all("hamstring" in name.lower() for name in names)
        assert "hamstring" not in names

    def test_output_has_no_group_names_from_the_old_case(self, rebuilt_db):
        # The whole table, not the first 500 rows of it: the table holds about 760 rows,
        # so reading the tool at its maximum limit checked two thirds of what it can say.
        with duckdb.connect(str(rebuilt_db), read_only=True) as src:
            names = {r[0] for r in src.execute(
                "SELECT DISTINCT injury_type FROM injury_return_curve").fetchall()}
        assert names
        for dropped in ("other", "quadriceps", "ribs", "knee", "hamstring", "ankle"):
            assert dropped not in names

    def test_example_values_named_by_the_schema_return_rows(self, rebuilt_db):
        """Every injury text the schema offers as an example has to match something real.

        The description used to offer 'Foot/Ankle', a text no injury row carries, so a
        model that followed the example got an empty answer.
        """
        import re

        from nfl_mcp.server import TOOLS
        schema = next(t for t in TOOLS if t.name == "nfl_injury_return")
        text = schema.inputSchema["properties"]["injury_type"]["description"]
        examples = re.findall(r"'([^']+)'", text)
        assert len(examples) >= 3
        for example in examples:
            result = nfl_injury_return(injury_type=example, limit=500)
            assert result["count"] > 0, example

    def test_default_sorted_by_week_post_return(self, rebuilt_db):
        result = nfl_injury_return(injury_type="Hamstring", position="WR")
        weeks = [r["week_post_return"] for r in result["injury_return"]]
        assert weeks == sorted(weeks)

    def test_filter_by_position_and_week(self, rebuilt_db):
        result = nfl_injury_return(injury_type="Knee", position="RB", week_post_return=1)
        assert "error" not in result
        for r in result["injury_return"]:
            assert r["week_post_return"] == 1

    def test_rows_have_expected_fields(self, rebuilt_db):
        result = nfl_injury_return(injury_type="Ankle", limit=1)
        if result["count"] > 0:
            row = result["injury_return"][0]
            for field in ("injury_type", "position", "week_post_return",
                          "avg_snap_pct_recovery", "median_snap_pct_recovery", "sample_size"):
                assert field in row

    def test_invalid_limit_falls_back(self, rebuilt_db):
        result = nfl_injury_return(injury_type="Hamstring", limit="oops")
        assert "error" not in result

    def test_db_error_returns_error_dict(self):
        with patch("nfl_mcp.tools._execute", side_effect=duckdb.Error("boom")):
            result = nfl_injury_return(injury_type="Hamstring")
        assert "boom" in result["error"]


class TestInputValidation:
    def test_invalid_stat_type(self):
        result = nfl_player_stats(player_name="Mahomes", stat_type="kicking")
        assert "error" in result

    def test_invalid_compare_type(self):
        result = nfl_compare(entity1="KC", entity2="BAL", compare_type="coach")
        assert "error" in result


class TestNflTeamRecords:
    """Unit tests — franchise availability and query building, no database needed."""

    def test_requires_a_filter(self):
        assert "error" in nfl_team_records()

    def test_unknown_team(self):
        result = nfl_team_records(team="USFL-NY", season=1985)
        assert "error" in result
        assert "USFL-NY" in result["error"]

    def test_historical_codes_resolve_to_the_current_franchise(self):
        with patch("nfl_mcp.tools._execute", return_value=[]) as execute:
            nfl_team_records(team="OAK", season=2019)
        assert execute.call_args[0][1] == ["LV", 2019]

    def test_franchise_that_did_not_exist_yet(self):
        with patch("nfl_mcp.tools._execute", side_effect=AssertionError("no DB hit")):
            result = nfl_team_records(team="HOU", season=1995)
        assert result["status"] == "franchise_not_yet_active"
        assert result["first_season"] == 2002
        assert result["records"] == []

    def test_franchise_that_sat_a_season_out(self):
        result = nfl_team_records(team="CLE", season=1997)
        assert result["status"] == "franchise_inactive"
        assert "1997" in result["reason"]

    def test_before_the_covered_era(self):
        result = nfl_team_records(team="GB", season=1965)
        assert result["status"] == "before_coverage"
        assert result["first_season"] == 1970

    def test_rows_carry_the_name_played_that_season(self):
        rows = [
            {"season": 1970, "team": "IND", "source_team": "BAL", "games": 14,
             "wins": 11, "losses": 2, "ties": 1, "source": "espn"},
            {"season": 1985, "team": "IND", "source_team": "IND", "games": 14,
             "wins": 5, "losses": 11, "ties": 0, "source": "espn"},
        ]
        def fake_execute(sql, params=None):
            """Answer the derived-shape guard's two lookups with 'nothing is old-shaped' and hand
            the two records to the data query, which is the one under test."""
            if "_ingest_metadata" in sql or "information_schema" in sql:
                return []
            return rows

        with patch("nfl_mcp.tools._execute", side_effect=fake_execute):
            result = nfl_team_records(team="IND", from_season=1970, to_season=1985)
        assert [row["club"] for row in result["records"]] == [
            "Baltimore Colts", "Indianapolis Colts",
        ]
        assert result["count"] == 2

    def test_range_filters_are_passed_as_parameters(self):
        with patch("nfl_mcp.tools._execute", return_value=[]) as execute:
            nfl_team_records(team="LV", from_season=1970, to_season=1980)
        sql, params = execute.call_args[0]
        assert "season >= ?" in sql and "season <= ?" in sql
        assert params == ["LV", 1970, 1980]

    def test_all_teams_in_one_season_has_no_team_filter(self):
        with patch("nfl_mcp.tools._execute", return_value=[]) as execute:
            nfl_team_records(season=2024)
        sql, params = execute.call_args[0]
        assert "team = ?" not in sql
        assert params == [2024]

    def test_missing_pre1999_record_points_at_the_opt_in_pull(self):
        with patch("nfl_mcp.tools._execute", return_value=[]):
            result = nfl_team_records(team="GB", season=1996)
        assert result["status"] == "record_not_available"
        assert "--pre1999" in result["reason"]

    def test_missing_modern_record_makes_no_pull_suggestion(self):
        with patch("nfl_mcp.tools._execute", return_value=[]):
            result = nfl_team_records(team="GB", season=2024)
        assert result["status"] == "record_not_available"
        assert "pre1999" not in result["reason"]

    def test_database_errors_are_returned(self):
        with patch("nfl_mcp.tools._execute", side_effect=duckdb.Error("no table team_records")):
            result = nfl_team_records(season=2024)
        assert "team_records" in result["error"]


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
class TestNflTeamRecordsIntegration:
    def test_one_row_per_team(self):
        result = nfl_team_records(season=2024)
        assert "error" not in result
        assert result["count"] >= 30
        teams = {row["team"] for row in result["records"]}
        assert len(teams) == result["count"]

    def test_records_are_complete(self):
        row = nfl_team_records(team="KC", season=2024)["records"][0]
        assert row["wins"] + row["losses"] + row["ties"] == row["games"] == 17
        assert row["source"] == "nflverse"
        assert row["club"] == "Kansas City Chiefs"

    def test_alias_finds_the_same_franchise(self):
        modern = nfl_team_records(team="LV", season=2024)["records"]
        via_alias = nfl_team_records(team="OAK", season=2024)["records"]
        assert modern == via_alias

    def test_season_range_is_clipped_to_the_bounds(self):
        # CI ingests one season, so assert what a range must always guarantee rather than
        # assuming the database spans it.
        bounded = nfl_team_records(team="PHI", from_season=2000, to_season=2024)["records"]
        assert bounded
        assert {row["season"] for row in bounded} <= set(range(2000, 2025))
        assert all(row["team"] == "PHI" for row in bounded)
        unbounded = nfl_team_records(team="PHI")["records"]
        assert len(bounded) <= len(unbounded)
