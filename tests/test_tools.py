"""Tests for the structured MCP tools."""

import pytest
import duckdb
from unittest.mock import patch
from nfl_mcp import tools
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
def require_db():
    """Skip integration tests if no data is loaded."""
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
        assert len(result["offense"]) == 1

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

    def test_invalid_team_returns_empty(self):
        result = nfl_team_stats(team="ZZZ", season=2024)
        assert "error" not in result
        assert len(result["offense"]) == 0


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

    def test_unknown_team_returns_empty(self):
        result = nfl_roster(team="ZZZ", season=2024)
        assert "error" not in result
        assert result["count"] == 0


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
@pytest.mark.usefixtures("require_db")
class TestNflInjuryReturn:
    def test_filter_by_injury_type(self):
        result = nfl_injury_return(injury_type="hamstring")
        assert "error" not in result
        assert result["count"] > 0

    def test_default_sorted_by_week_post_return(self):
        result = nfl_injury_return(injury_type="hamstring", position="WR")
        weeks = [r["week_post_return"] for r in result["injury_return"]]
        assert weeks == sorted(weeks)

    def test_filter_by_position_and_week(self):
        result = nfl_injury_return(injury_type="knee", position="RB", week_post_return=1)
        assert "error" not in result
        for r in result["injury_return"]:
            assert r["week_post_return"] == 1

    def test_rows_have_expected_fields(self):
        result = nfl_injury_return(injury_type="ankle", limit=1)
        if result["count"] > 0:
            row = result["injury_return"][0]
            for field in ("injury_type", "position", "week_post_return",
                          "avg_snap_pct_recovery", "median_snap_pct_recovery", "sample_size"):
                assert field in row

    def test_invalid_limit_falls_back(self):
        result = nfl_injury_return(injury_type="hamstring", limit="oops")
        assert "error" not in result

    def test_db_error_returns_error_dict(self):
        with patch("nfl_mcp.tools._execute", side_effect=duckdb.Error("boom")):
            result = nfl_injury_return(injury_type="hamstring")
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
        with patch("nfl_mcp.tools._execute", return_value=rows):
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
