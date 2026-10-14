"""Tests for the franchise identity registry."""

import pytest

from nfl_mcp.franchises import (
    ESPN_IDS,
    FRANCHISES,
    RECORDS_FIRST_SEASON,
    availability,
    canonical_team,
    era_name,
    expected_franchise_count,
    expected_games,
    schedule_alias_case,
)


class TestCanonicalTeam:
    @pytest.mark.parametrize("raw,expected", [
        ("LV", "LV"),
        ("oak", "LV"),          # Oakland -> Las Vegas
        ("SD", "LAC"),          # San Diego -> Los Angeles Chargers
        ("STL", "LA"),          # St. Louis -> Los Angeles Rams
        ("LAR", "LA"),          # ESPN's modern Rams code
        ("WSH", "WAS"),         # ESPN's older Washington code
        (" HOU ", "HOU"),       # whitespace tolerated
    ])
    def test_resolves_source_codes(self, raw, expected):
        assert canonical_team(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "ZZZ", "USFL"])
    def test_unknown_returns_none(self, raw):
        assert canonical_team(raw) is None

    def test_every_canonical_key_resolves_to_itself(self):
        assert {canonical_team(key) for key in FRANCHISES} == set(FRANCHISES)

    def test_no_alias_is_also_a_canonical_key(self):
        # An alias that collides with a live franchise would silently merge two teams.
        for key, franchise in FRANCHISES.items():
            assert key not in franchise.aliases
            for alias in franchise.aliases:
                assert alias not in FRANCHISES, f"{alias} aliases {key} but is its own franchise"

    def test_espn_ids_are_unique_and_canonical(self):
        ids = [f.espn_id for f in FRANCHISES.values()]
        assert len(ids) == len(set(ids)) == len(ESPN_IDS)
        assert set(ESPN_IDS.values()) == set(FRANCHISES)


class TestEraName:
    @pytest.mark.parametrize("season,team,expected", [
        (1970, "IND", "Baltimore Colts"),
        (1985, "IND", "Indianapolis Colts"),
        (1970, "NE", "Boston Patriots"),
        (1971, "NE", "New England Patriots"),
        (1975, "LV", "Oakland Raiders"),
        (2024, "OAK", "Las Vegas Raiders"),   # alias resolves before the lookup
        (1996, "TEN", "Houston Oilers"),
        (1999, "TEN", "Tennessee Titans"),
        (2024, "SEA", "Seattle Seahawks"),    # franchise with no renames
    ])
    def test_names_by_season(self, season, team, expected):
        assert era_name(season, team) == expected

    def test_unknown_team(self):
        assert era_name(1990, "ZZZ") is None


class TestSeasonShapes:
    @pytest.mark.parametrize("season,games", [
        (1970, 14), (1977, 14), (1978, 16), (1981, 16),
        (1982, 9),    # strike season
        (1986, 16),
        (1987, 15),   # strike season
        (2020, 16), (2021, 17), (2025, 17),
    ])
    def test_expected_games(self, season, games):
        assert expected_games(season) == games

    @pytest.mark.parametrize("season,count", [
        (1970, 26), (1975, 26), (1976, 28), (1994, 28),
        (1995, 30), (1998, 30), (1999, 31), (2002, 32), (2025, 32),
    ])
    def test_expected_franchise_count(self, season, count):
        assert expected_franchise_count(season) == count

    def test_browns_gap_is_not_counted(self):
        assert expected_franchise_count(1996) == expected_franchise_count(1997) == 30


class TestAvailability:
    def test_active_franchise_has_no_note(self):
        assert availability("DAL", 1980) is None

    def test_expansion_franchise(self):
        note = availability("HOU", 1995)
        assert note["status"] == "franchise_not_yet_active"
        assert note["first_season"] == 2002
        assert "2002" in note["reason"]

    def test_suspended_franchise(self):
        note = availability("CLE", 1997)
        assert note["status"] == "franchise_inactive"
        assert "1997" in note["reason"]

    def test_before_coverage_floor(self):
        note = availability("GB", RECORDS_FIRST_SEASON - 1)
        assert note["status"] == "before_coverage"
        assert note["first_season"] == RECORDS_FIRST_SEASON

    def test_unknown_team(self):
        assert availability("ZZ", 1980)["status"] == "unknown_team"


class TestAliasSql:
    def test_covers_every_alias(self):
        sql = schedule_alias_case("s.team")
        for key, franchise in FRANCHISES.items():
            for alias in franchise.aliases:
                assert f"WHEN s.team = '{alias}' THEN '{key}'" in sql

    def test_passes_unmatched_codes_through(self):
        assert "ELSE team END" in schedule_alias_case("team")
