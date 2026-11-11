"""Tests for the franchise identity registry."""

import pytest

from nfl_mcp.franchises import (
    ESPN_IDS,
    FRANCHISES,
    RECORDS_FIRST_SEASON,
    availability,
    before_records_coverage,
    canonical_team,
    era_name,
    expected_franchise_count,
    expected_games,
    franchise_codes,
    schedule_alias_case,
    suggest_teams,
)


class TestCanonicalTeam:
    @pytest.mark.parametrize("raw,expected", [
        ("LV", "LV"),
        ("oak", "LV"),          # Oakland -> Las Vegas
        ("SD", "LAC"),          # San Diego -> Los Angeles Chargers
        ("STL", "LA"),          # St. Louis -> Los Angeles Rams
        ("LAR", "LA"),          # ESPN's modern Rams code
        ("WSH", "WAS"),         # ESPN's older Washington code
        ("JAC", "JAX"),         # FantasyPros' Jaguars code
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


class TestFranchiseCodes:
    @pytest.mark.parametrize("raw,expected", [
        ("LV", ("LV", "OAK")),
        ("oak", ("LV", "OAK")),      # an alias returns the whole franchise, canonical first
        ("SD", ("LAC", "SD")),
        ("KC", ("KC",)),
        ("HOU", ("HOU",)),           # the Texans; the Oilers are TEN
    ])
    def test_every_code_of_the_franchise(self, raw, expected):
        assert franchise_codes(raw) == expected

    def test_unknown_returns_none(self):
        assert franchise_codes("ZZZ") is None


class TestSuggestTeams:
    def test_close_codes_resolve_to_canonical_keys(self):
        assert suggest_teams("KCC") == ["KC"]
        assert "LV" in suggest_teams("LVR")

    def test_nothing_close_and_empty_input(self):
        assert suggest_teams("XYZ") == []
        assert suggest_teams("") == []

    def test_never_repeats_a_franchise(self):
        # OAK and LV both look like "OA"; the franchise is offered once.
        suggestions = suggest_teams("OA", limit=5)
        assert len(suggestions) == len(set(suggestions))


def test_before_records_coverage_is_what_availability_returns():
    assert availability("GB", 1965) == before_records_coverage()
