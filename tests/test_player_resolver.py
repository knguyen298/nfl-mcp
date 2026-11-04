"""GROUP A: one player id resolver shared by every tool that accepts a player.

Integration tests run against the real database. The name collisions they use are
real nflverse rows, not fixtures:

  00-0036322  Justin Jefferson  WR  MIN  2020-2026
  00-0041075  Justin Jefferson  LB  CLE  2026-2026
  00-0030235  Willie Jefferson III  LB  HOU  2013-2013  (a suffixed name)

Both Justin Jeffersons share display_name and short_name ('J.Jefferson'), so only
gsis_id tells them apart.
"""

import duckdb
import pytest

from nfl_mcp.tools import (
    nfl_compare,
    nfl_contract_value,
    nfl_drop_rate,
    nfl_fantasy_opportunity,
    nfl_fantasy_rankings,
    nfl_ftn_charting,
    nfl_injuries,
    nfl_player_stats,
    nfl_query,
    nfl_resolve_player,
    nfl_role_trend,
    nfl_search_plays,
    nfl_separation_opportunity,
    nfl_snap_counts,
    nfl_td_luck,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def require_db(rebuilt_db):
    """Skip this module when no database is loaded, like the other integration tests.

    Depends on rebuilt_db so the derived tables these tools read are the ones the current ingest
    builds - nfl_separation_opportunity selects season_type, which the file on disk predates.
    """
    result = nfl_query("SELECT COUNT(*) AS n FROM plays")
    if "error" in result or result["rows"][0]["n"] == 0:
        pytest.skip("No DuckDB database loaded", allow_module_level=True)

JJ_WR = "00-0036322"
JJ_LB = "00-0041075"
WILLIE_III = "00-0030235"

# One adapter per tool that takes a player, all with the same calling shape:
# (name, team, season, position, player_id). Tools without a parameter ignore it,
# which is what the cases below are written around.
PLAYER_TOOLS = {
    "nfl_search_plays": lambda name=None, team=None, season=None, player_id=None, **_: nfl_search_plays(
        player=name, player_id=player_id, team=team, season=season, max_rows=5),
    "nfl_player_stats": lambda name=None, team=None, season=None, player_id=None, **_: nfl_player_stats(
        player_name=name, player_id=player_id, season=season, stat_type="receiving"),
    "nfl_compare": lambda name=None, team=None, season=None, player_id=None, **_: nfl_compare(
        name or "", "Travis Kelce", compare_type="player", player_id1=player_id, season=season),
    "nfl_injuries": lambda name=None, team=None, season=None, player_id=None, **_: nfl_injuries(
        player=name, player_id=player_id, team=team, season=season),
    "nfl_snap_counts": lambda name=None, team=None, season=None, position=None, player_id=None, **_: nfl_snap_counts(
        player=name, player_id=player_id, team=team, season=season, position=position),
    "nfl_fantasy_opportunity": lambda name=None, team=None, season=None, position=None, player_id=None, **_: nfl_fantasy_opportunity(
        player=name, player_id=player_id, team=team, season=season, position=position),
    "nfl_fantasy_rankings": lambda name=None, team=None, position=None, player_id=None, **_: nfl_fantasy_rankings(
        player=name, player_id=player_id, team=team, position=position, scope="week", limit=5),
    "nfl_ftn_charting": lambda name=None, team=None, season=None, player_id=None, **_: nfl_ftn_charting(
        player=name, player_id=player_id, team=team, season=season),
    "nfl_td_luck": lambda name=None, team=None, season=None, position=None, player_id=None, **_: nfl_td_luck(
        player=name, player_id=player_id, team=team, season=season, position=position, limit=5),
    "nfl_role_trend": lambda name=None, team=None, season=None, position=None, player_id=None, **_: nfl_role_trend(
        player=name, player_id=player_id, team=team, season=season, position=position, limit=5),
    "nfl_separation_opportunity": lambda name=None, team=None, season=None, position=None, player_id=None, **_: nfl_separation_opportunity(
        player=name, player_id=player_id, team=team, season=season, position=position, limit=5),
    "nfl_drop_rate": lambda name=None, team=None, season=None, player_id=None, **_: nfl_drop_rate(
        player=name, player_id=player_id, team=team, season=season, limit=5),
    "nfl_contract_value": lambda name=None, team=None, season=None, position=None, player_id=None, **_: nfl_contract_value(
        player=name, player_id=player_id, team=team, season=season, position=position, limit=5),
}

# nfl_fantasy_rankings can only reach players ff_playerids maps to a FantasyPros id
# (the current season's players), so a retired player is reported as not_found there.
CASES = {
    "ambiguous": {"name": "Justin Jefferson", "expect": ("ambiguous", [JJ_WR, JJ_LB])},
    "unique_name": {"name": "Travis Kelce", "expect": ("resolved", "00-0030506")},
    "short_name": {"name": "J.Jefferson", "expect": ("ambiguous", None)},
    "suffix": {
        "name": "Willie Jefferson III",
        "expect": ("resolved", WILLIE_III),
        "skip": {"nfl_fantasy_rankings"},
    },
    "unknown_name": {"name": "ZZZZNOTAPLAYER", "expect": ("not_found", None)},
}


def outcome(result):
    """Reduce a tool result (or the resolver's own answer) to what it decided."""
    body = result or {}
    if body.get("ambiguous"):
        return "ambiguous", sorted(c["gsis_id"] for c in body["candidates"])
    if body.get("not_found"):
        return "not_found", None
    player = body.get("player") or body.get("resolved")
    if player:
        return "resolved", player["gsis_id"]
    return "no player filter", None


def tool_outcome(tool_name, case):
    if tool_name == "nfl_compare":
        return outcome(PLAYER_TOOLS[tool_name](**case)["players"][0])
    return outcome(PLAYER_TOOLS[tool_name](**case))


@pytest.mark.parametrize("case_name", sorted(CASES))
@pytest.mark.parametrize("tool_name", sorted(PLAYER_TOOLS))
def test_every_tool_gives_the_same_resolver_result(tool_name, case_name):
    case = CASES[case_name]
    if tool_name in case.get("skip", ()):
        pytest.skip(f"{tool_name} cannot reach this player through its id mapping")
    args = {k: v for k, v in case.items() if k not in ("expect", "skip")}
    expected = outcome(nfl_resolve_player(**{"player": args["name"], **{
        k: v for k, v in args.items() if k != "name"}}))
    assert tool_outcome(tool_name, args) == expected


@pytest.mark.parametrize("case_name", sorted(CASES))
@pytest.mark.parametrize("tool_name", sorted(PLAYER_TOOLS))
def test_case_matches_the_documented_expectation(tool_name, case_name):
    case = CASES[case_name]
    if tool_name in case.get("skip", ()):
        pytest.skip(f"{tool_name} cannot reach this player through its id mapping")
    kind, ids = case["expect"]
    got_kind, got_ids = tool_outcome(tool_name, {
        k: v for k, v in case.items() if k not in ("expect", "skip")})
    assert got_kind == kind
    if isinstance(ids, str):
        assert got_ids == ids
    elif ids is not None:
        assert got_ids == sorted(ids)


@pytest.mark.parametrize("tool_name", sorted(PLAYER_TOOLS))
def test_team_filter_makes_an_ambiguous_name_unique(tool_name):
    if tool_name in ("nfl_player_stats", "nfl_compare"):
        pytest.skip(f"{tool_name} takes no team parameter, and both Jeffersons play in 2026")
    assert tool_outcome(tool_name, {"name": "Justin Jefferson", "team": "MIN"}) == (
        "resolved", JJ_WR)


@pytest.mark.parametrize("tool_name", sorted(PLAYER_TOOLS))
def test_candidates_carry_the_labels_a_caller_needs(tool_name):
    if tool_name in ("nfl_player_stats", "nfl_compare"):
        pytest.skip(f"{tool_name} takes no team parameter, and both Jeffersons play in 2026")
    result = PLAYER_TOOLS[tool_name](name="Justin Jefferson", season=2026)
    body = result["players"][0] if tool_name == "nfl_compare" else result
    labels = {c["label"]: c["gsis_id"] for c in body["candidates"]}
    assert labels == {"Justin Jefferson (WR, MIN)": JJ_WR, "Justin Jefferson (LB, CLE)": JJ_LB}
    assert all(c["first_season"] and c["last_season"] for c in body["candidates"])
    assert not [k for k in body if k in ("seasons", "rows", "snap_counts", "td_luck")]


@pytest.mark.parametrize("tool_name", sorted(PLAYER_TOOLS))
def test_season_filter_makes_an_ambiguous_name_unique(tool_name):
    if tool_name == "nfl_fantasy_rankings":
        pytest.skip("rankings are a single latest scrape, they carry no season")
    assert tool_outcome(tool_name, {"name": "Justin Jefferson", "season": 2024}) == (
        "resolved", JJ_WR)


class TestAmbiguousNameReturnsCandidatesAndNoStats:
    def test_ambiguity_carries_the_fields_that_tell_the_players_apart(self):
        result = nfl_resolve_player("Justin Jefferson", season=2026)
        assert result["ambiguous"] is True
        assert result["count"] == 2
        wr, _ = result["candidates"]
        assert wr == {
            "gsis_id": JJ_WR,
            "full_name": "Justin Jefferson",
            "pfr_id": "JeffJu00",
            "position": "WR",
            "team": "MIN",
            "first_season": 2020,
            "last_season": 2026,
            "label": "Justin Jefferson (WR, MIN)",
        }

    def test_player_stats_refuses_to_add_two_careers_together(self):
        result = nfl_player_stats("Justin Jefferson", season=2026, stat_type="receiving")
        assert result["ambiguous"] is True
        assert "seasons" not in result

    def test_search_plays_refuses_to_mix_two_players(self):
        result = nfl_search_plays(player="Justin Jefferson", season=2026)
        assert result["ambiguous"] is True
        assert "rows" not in result

    def test_resolved_rows_carry_only_that_player(self):
        one = nfl_player_stats(player_id=JJ_WR, season=2024, stat_type="receiving")
        assert one["seasons"], one
        assert {row["player_id"] for row in one["seasons"]} == {JJ_WR}
        assert all(row["label"] == "Justin Jefferson (WR, MIN)" for row in one["seasons"])

    def test_player_id_wins_over_the_name(self):
        result = nfl_player_stats("Justin Jefferson", player_id=JJ_LB, season=2026,
                                  stat_type="receiving")
        assert result["player"]["gsis_id"] == JJ_LB        # the id decides, not the shared name
        rows = nfl_player_stats("Justin Jefferson", player_id=JJ_WR, season=2024,
                               stat_type="receiving")   # rows really came back
        assert rows["seasons"] and all(row["player_id"] == JJ_WR for row in rows["seasons"])

    def test_rows_are_labelled_in_the_arena_the_plan_asks_for(self):
        rows = nfl_fantasy_opportunity(player="Justin Jefferson", season=2024, team="MIN")["fantasy_opportunity"]
        assert rows
        assert {row["label"] for row in rows} == {"Justin Jefferson (WR, MIN)"}
        assert {row["player_id"] for row in rows} == {JJ_WR}


class TestNameForms:
    @pytest.mark.parametrize("raw", [
        "Justin Jefferson", "justin jefferson", "J.Jefferson", "J. Jefferson",
    ])
    def test_full_and_short_forms_reach_the_same_id(self, raw):
        assert nfl_resolve_player(raw, season=2024, team="MIN")["resolved"]["gsis_id"] == JJ_WR

    def test_suffix_in_the_query_matches_a_row_whose_suffix_is_a_separate_column(self):
        assert nfl_resolve_player("Willie Jefferson III")["resolved"]["gsis_id"] == WILLIE_III

    def test_resolved_name_keeps_the_suffix(self):
        assert nfl_resolve_player("Willie Jefferson")["resolved"]["full_name"] == (
            "Willie Jefferson III")

    def test_apostrophe_and_period_do_not_break_a_match(self):
        assert nfl_resolve_player("P.Mahomes")["resolved"]["gsis_id"] == (
            nfl_resolve_player("Patrick Mahomes")["resolved"]["gsis_id"])

    def test_accents_are_folded_on_both_sides(self):
        accented = nfl_resolve_player("Audric Estimé")["resolved"]["gsis_id"]
        plain = nfl_resolve_player("Audric Estime")["resolved"]["gsis_id"]
        assert accented == plain

    def test_two_real_d_j_moore_collisions_stay_separate(self):
        result = nfl_resolve_player("D.J. Moore")
        assert result["ambiguous"] is True
        assert sorted(c["full_name"] for c in result["candidates"]) == ["D.J. Moore", "DJ Moore"]

    def test_a_misspelt_first_name_suggests_the_closest_names(self):
        result = nfl_resolve_player("Justen Jefferson")
        assert result["not_found"] is True
        assert result["did_you_mean"][0] == "justin jefferson"

    def test_a_misspelt_surname_still_suggests_names(self):
        result = nfl_resolve_player("Jeffersonn")
        assert result["not_found"] is True
        assert result["did_you_mean"]

    def test_id_that_does_not_exist_is_not_found(self):
        result = nfl_resolve_player(player_id="00-9999999")
        assert result["not_found"] is True
        assert result["did_you_mean"] == []


class TestPerTableIdMappings:
    """Each tool must filter its own table through an id, never a name substring."""

    def test_snap_counts_accepts_both_name_forms_through_pfr_id(self):
        full = nfl_snap_counts(player="Travis Kelce", season=2024)
        short = nfl_snap_counts(player="T.Kelce", season=2024)
        assert full["count"] > 0
        assert full["count"] == short["count"]
        assert {row["player_id"] for row in full["snap_counts"]} == {"00-0030506"}

    def test_snap_counts_splits_same_name_players_by_pfr_id(self):
        wr = nfl_snap_counts(player="Justin Jefferson", season=2024, team="MIN")
        lb = nfl_snap_counts(player="Justin Jefferson", season=2026, team="CLE")
        assert wr["player"]["pfr_id"] == "JeffJu00"
        assert lb["player"]["pfr_id"] == "JeffJu01"
        assert wr["snap_counts"], wr                        # CI loads 2024, so the WR has rows
        assert {row["pfr_player_id"] for row in wr["snap_counts"]} == {"JeffJu00"}
        # The linebacker's rows start in 2026, a season CI does not load.
        assert {row["pfr_player_id"] for row in lb["snap_counts"]} <= {"JeffJu01"}

    def test_ftn_charting_reaches_plays_through_ids(self):
        result = nfl_ftn_charting(player="Patrick Mahomes", season=2024)
        assert result["total_plays"] > 0
        assert result["player"]["gsis_id"] == nfl_resolve_player("P.Mahomes")["resolved"]["gsis_id"]

    def test_fantasy_rankings_go_through_ff_playerids(self):
        draft = nfl_fantasy_rankings(player="Justin Jefferson", team="MIN", scope="draft")
        week = nfl_fantasy_rankings(player="Justin Jefferson", team="MIN", scope="week")
        assert draft["count"] > 0 and week["count"] > 0
        assert draft["player"]["gsis_id"] == JJ_WR
        assert {row["label"] for row in week["fantasy_rankings"]} == {"Justin Jefferson (WR, MIN)"}

    def test_rankings_say_not_found_when_the_id_has_no_fantasypros_mapping(self):
        unmapped = nfl_resolve_player(player_id="00-0027350")["resolved"]   # A.J. Jefferson
        result = nfl_fantasy_rankings(player_id=unmapped["gsis_id"])
        assert result["not_found"] is True
        assert "FantasyPros" in result["reason"]

    def test_rankings_return_no_rows_when_the_mapped_player_is_unranked(self):
        result = nfl_fantasy_rankings(player="Justin Jefferson", team="CLE", scope="week")
        assert result["player"]["gsis_id"] == JJ_LB
        assert result["count"] == 0

    def test_injuries_filter_by_gsis_id(self):
        result = nfl_injuries(player="Justin Jefferson", season=2024)
        assert result["count"] > 0
        assert {row["player_id"] for row in result["injuries"]} == {JJ_WR}

    @pytest.mark.parametrize("tool_name", [
        "nfl_td_luck", "nfl_role_trend", "nfl_separation_opportunity",
        "nfl_contract_value", "nfl_drop_rate",
    ])
    def test_derived_tables_filter_by_player_id(self, tool_name):
        result = PLAYER_TOOLS[tool_name](name="Justin Jefferson", season=2024, team="MIN")
        key = next(k for k in result if isinstance(result[k], list))
        rows = result[key]
        assert rows, result
        assert {row["player_id"] for row in rows} == {JJ_WR}


class TestCompareMode:
    def test_compare_two_players_keeps_their_ids_apart(self):
        result = nfl_compare("Justin Jefferson", "Travis Kelce", compare_type="player",
                             season=2026, player_id1=JJ_WR)
        sides = result["players"]
        assert [s["player"]["gsis_id"] for s in sides] == [JJ_WR, "00-0030506"]

    def test_compare_reports_ambiguity_for_the_entity_it_cannot_resolve(self):
        result = nfl_compare("Justin Jefferson", "Travis Kelce", compare_type="player",
                             season=2026)
        sides = result["players"]
        assert sides[0]["ambiguous"] is True
        assert sides[1]["player"]["gsis_id"] == "00-0030506"

    def test_the_two_justin_jeffersons_compare_against_each_other(self):
        """Same name, two ids: the result must not collapse to one entry (fix 5)."""
        result = nfl_compare("Justin Jefferson", "Justin Jefferson", compare_type="player",
                             player_id1=JJ_WR, player_id2=JJ_LB, season=2026)
        sides = result["players"]
        assert [s["player"]["gsis_id"] for s in sides] == [JJ_WR, JJ_LB]
        assert [s["player"]["label"] for s in sides] == [
            "Justin Jefferson (WR, MIN)", "Justin Jefferson (LB, CLE)"]
        assert {s["requested"] for s in sides} == {"Justin Jefferson"}

    def test_compare_says_which_text_each_side_came_from(self):
        result = nfl_compare("Justin Jefferson", "ZZZZNOTAPLAYER", compare_type="player",
                             player_id1=JJ_WR, season=2026)
        sides = result["players"]
        assert sides[1]["not_found"] is True
        assert sides[1]["requested"] == "ZZZZNOTAPLAYER"


class TestOneLetterGivenName:
    """A one-letter query is an initial. It may never be read as a spelled-out name.

    `players` holds eight players whose stored given name is one letter ('H.', 'C.',
    'B.'), so treating 'B' as spelled out hands 'B.Robinson' to Michael Robinson and
    drops Bijan Robinson, Brian Robinson and five others.
    """

    @pytest.mark.parametrize("raw, season, must_include", [
        ("D.Jones", 2024, "Daniel Jones"),         # 371 passing attempts in plays 2024
        ("D.Montgomery", 2024, "David Montgomery"),
        ("A.Robinson", 2024, "Allen Robinson"),
        ("J.Taylor", 2026, "Jonathan Taylor"),
        ("D.Johnson", 2024, "Diontae Johnson"),    # the old code left him out entirely
    ])
    def test_the_answer_lists_every_player_the_name_covers(self, raw, season, must_include):
        result = nfl_resolve_player(raw, season=season)
        assert result["ambiguous"] is True
        assert must_include in {c["full_name"] for c in result["candidates"]}

    def test_a_player_whose_first_name_is_written_as_an_initial_is_one_candidate(self):
        result = nfl_resolve_player("B.Robinson")   # 'B.' is Michael Robinson's first name
        assert result["count"] == 7
        assert {"Bijan Robinson", "Brian Robinson", "Michael Robinson"} <= {
            c["full_name"] for c in result["candidates"]}

    def test_a_one_letter_name_with_only_one_owner_still_resolves(self):
        assert nfl_resolve_player("H.Bell")["resolved"]["full_name"] == "Nick Bell"
        assert nfl_resolve_player("C.Collins")["count"] == 7   # Beaux Collins stores 'C.'

    def test_a_tool_refuses_to_pick_for_a_one_letter_name(self):
        result = nfl_player_stats(player_name="D.Jones", season=2024, stat_type="passing")
        assert result["ambiguous"] is True
        assert "seasons" not in result
        assert "Daniel Jones" in {c["full_name"] for c in result["candidates"]}


class TestPrefixedShortNames:
    """plays tells same-initial players apart with a 2-3 letter prefix (fix 2)."""

    @pytest.mark.parametrize("raw, season, plays_id", [
        ("Mi.Wilson", 2024, "00-0038559"),         # Michael Wilson (WR)
        ("Cas.Washington", 2024, "00-0039394"),
        ("Co.Heyward", 2024, "00-0037304"),
        ("Bri.Thomas", 2026, "00-0039893"),
    ])
    def test_a_prefixed_short_name_reaches_the_player_plays_means(self, raw, season, plays_id):
        result = nfl_resolve_player(raw, season=season)
        if "resolved" in result:
            assert result["resolved"]["gsis_id"] == plays_id
        else:
            assert plays_id in {c["gsis_id"] for c in result["candidates"]}

    def test_two_ids_for_one_prefixed_name_stay_two_candidates(self):
        result = nfl_resolve_player("Ja.Williams", season=2024)   # plays reports 00-0033948 + 00-0037240
        assert result["ambiguous"] is True
        assert {"00-0033948", "00-0037240"} <= {c["gsis_id"] for c in result["candidates"]}

    def test_an_apostrophe_first_name_is_never_read_as_a_different_one(self):
        result = nfl_resolve_player("D.Johnson", season=2024)
        names = {c["full_name"] for c in result["candidates"]}
        assert {"D'Ernest Johnson", "DJ Johnson", "Diontae Johnson"} <= names

    def test_the_id_plays_reports_is_never_left_out_of_the_answer(self):
        """A sample of the audit: no (season, name) pair loses the player plays credits."""
        sample = nfl_query(
            """SELECT season, name, player_id FROM (
                 SELECT season, passer_player_name AS name, passer_player_id AS player_id FROM plays
                 UNION SELECT season, rusher_player_name, rusher_player_id FROM plays
                 UNION SELECT season, receiver_player_name, receiver_player_id FROM plays)
               WHERE name IS NOT NULL AND player_id IS NOT NULL
               GROUP BY 1, 2, 3 ORDER BY name, season, player_id LIMIT 80""")["rows"]
        assert len(sample) == 80
        for row in sample:
            result = nfl_resolve_player(row["name"], season=int(row["season"]))
            if "resolved" in result:
                assert result["resolved"]["gsis_id"] == row["player_id"], row
            else:
                assert row["player_id"] in {c["gsis_id"] for c in result["candidates"]}, row


class TestSurnameListsLongerThanTheOldRowCap:
    """The resolver used to cut the candidate rows at 200 before it filtered names (fix 2b)."""

    def test_a_surname_the_database_holds_more_than_200_times_is_not_cut_short(self):
        assert nfl_resolve_player("Keyshawn Johnson")["resolved"]["gsis_id"] == "00-0008561"

    def test_every_player_of_that_name_is_a_candidate(self):
        result = nfl_resolve_player("Ricky Williams")   # three of them, two used to be cut
        assert result["count"] == 3
        assert sorted(c["gsis_id"] for c in result["candidates"]) == [
            "00-0017915", "00-0020739", "WIL494343"]

    def test_the_count_tells_the_truth_when_the_list_is_capped(self):
        result = nfl_resolve_player("J.Smith")
        assert result["count"] == 41
        assert len(result["candidates"]) == 20


class TestTeamFilterFollowsTrades:
    """`rosters` holds one team per player-season, so a trade needs rosters_weekly (fix 3)."""

    @pytest.mark.parametrize("code", ["LV", "NYJ"])
    def test_a_traded_player_answers_to_both_teams_that_season(self, code):
        result = nfl_resolve_player("Davante Adams", team=code, season=2024)
        assert result["resolved"]["gsis_id"] == "00-0031381"

    def test_an_old_team_abbreviation_resolves_to_the_current_franchise(self):
        assert nfl_resolve_player("Brock Bowers", team="OAK", season=2024)[
            "resolved"]["gsis_id"] == "00-0039338"     # LV, by way of franchises.canonical_team

    def test_player_team_seasons_that_only_the_weekly_roster_can_reach(self):
        """`rosters` lists one club per player-season, so a trade is invisible there.

        Pinned to 2024 because CI loads one season; the same rule finds 66 such
        player-team-seasons across the loaded 2024-2026 range."""
        rows = nfl_query(
            """SELECT COUNT(*) AS n FROM (
                 SELECT DISTINCT passer_player_id AS id, posteam AS team FROM plays
                   WHERE posteam IS NOT NULL AND passer_player_id IS NOT NULL AND season = 2024
                 UNION SELECT DISTINCT rusher_player_id, posteam FROM plays
                   WHERE posteam IS NOT NULL AND rusher_player_id IS NOT NULL AND season = 2024
                 UNION SELECT DISTINCT receiver_player_id, posteam FROM plays
                   WHERE posteam IS NOT NULL AND receiver_player_id IS NOT NULL AND season = 2024) p
               JOIN players pl ON pl.gsis_id = p.id
               WHERE COALESCE(pl.latest_team, '') <> p.team
                 AND NOT EXISTS (SELECT 1 FROM rosters r
                                 WHERE r.gsis_id = p.id AND r.team = p.team AND r.season = 2024)
                 AND EXISTS (SELECT 1 FROM rosters_weekly w
                             WHERE w.gsis_id = p.id AND w.team = p.team AND w.season = 2024)"""
        )["rows"][0]["n"]
        assert rows == 26

    def test_a_team_code_the_franchise_table_does_not_know_is_named(self):
        # Group C second pass, fix 4: the resolver used to narrow candidates to nothing and
        # report the player as missing, so a bad club code looked like a bad player name.
        assert nfl_resolve_player("Davante Adams", team="ZZZ", season=2024)["error"] \
            == "Unknown team 'ZZZ'."
        # Group C fix 3: a tool that filters rows on the code says so instead of returning 0.
        assert nfl_injuries(team="ZZZ", season=2024)["error"] == "Unknown team 'ZZZ'."


class TestPositionNarrowsCandidatesOnly:   # batch 2 fix 3
    """A caller's position narrows the candidates, it never removes a player from the lookup."""

    def test_a_position_settles_a_name_that_covers_two_positions(self):
        result = nfl_td_luck(player="Josh Allen", position="QB", season=2024)
        assert "ambiguous" not in result
        assert result["player"]["gsis_id"] == "00-0034857"

    def test_a_position_matching_no_candidate_leaves_the_pair_in_the_answer(self):
        result = nfl_td_luck(player="Josh Allen", position="DL", season=2024)
        assert result["ambiguous"] is True
        assert [c["label"] for c in result["candidates"]] == [
            "Josh Allen (QB, BUF)", "Josh Hines-Allen (DE, JAX)"]

    def test_a_position_matching_two_candidates_still_leaves_the_choice_open(self):
        result = nfl_role_trend(player="D.Johnson", position="RB", season=2024)
        assert result["ambiguous"] is True
        assert sorted(c["label"] for c in result["candidates"]) == [
            "D'Ernest Johnson (RB, NE)", "Dillon Johnson (RB, CAR)"]

    def test_a_truncated_candidate_list_is_never_narrowed(self):
        # 'J.Smith' matches 41 players and lists 20, so a match inside the cut is no answer.
        result = nfl_td_luck(player="J.Smith", position="RB")
        assert result["ambiguous"] is True
        assert result["count"] == 41 and len(result["candidates"]) == 20 and "player" not in result


class TestPositionFiltersTheResolverItself:   # review batch one, fix 4
    """`position` is bound into the resolver's own WHERE clause, whatever its docstring claimed.

    nfl_resolve_player is not in server.TOOLS, so there is no parameter description to correct
    there: the door into it is the function, and the seven tools that resolve a player each bind
    position in their own query as well. Its docstring said `position` filters no rows here while
    the code appended `position ILIKE ?`.
    """

    @pytest.mark.parametrize("typed,winner", [("WR", JJ_WR), ("lb", JJ_LB)])
    def test_a_position_settles_the_pair_inside_the_resolver(self, typed, winner):
        resolved = nfl_resolve_player("Justin Jefferson", season=2026, position=typed)
        assert resolved["resolved"]["gsis_id"] == winner

    def test_a_position_that_matches_nobody_is_not_found(self):
        result = nfl_resolve_player("Justin Jefferson", season=2026, position="XX")
        assert result["not_found"] is True
        assert "resolved" not in result

    def test_an_id_outranks_a_position(self):
        # The position clause sits on the name branch only; the id already picked the player.
        result = nfl_resolve_player("Justin Jefferson", player_id=JJ_LB, position="WR")
        assert result["resolved"]["gsis_id"] == JJ_LB

    def test_the_docstring_describes_what_the_code_does(self):
        text = nfl_resolve_player.__doc__ or ""
        assert "filters no rows" not in text
        assert "position ILIKE ?" in text


class TestPositionStaysWithItsOwnTable:
    """snap_counts uses PFR codes (T, S, DL); players uses OT, SAF, DT (fix 4)."""

    def test_snap_counts_answers_for_its_own_position_code(self):
        result = nfl_snap_counts(player="Trent Williams", position="T", season=2024)
        assert result["count"] > 0
        assert {row["position"] for row in result["snap_counts"]} == {"T"}

    def test_the_resolver_on_its_own_still_compares_against_players_position(self):
        # Documented limit: nfl_resolve_player('Trent Williams', position='T') finds no
        # player, because players.position calls him an OT. Tools no longer route
        # position through it.
        assert nfl_resolve_player("Trent Williams", position="T", season=2024)["not_found"] is True

    @pytest.mark.parametrize("tool_name", [
        "nfl_snap_counts", "nfl_fantasy_opportunity", "nfl_fantasy_rankings", "nfl_td_luck",
        "nfl_role_trend", "nfl_separation_opportunity", "nfl_contract_value",
    ])
    def test_position_still_filters_the_rows_a_tool_reads(self, tool_name):
        same_vocabulary = PLAYER_TOOLS[tool_name](
            name="Justin Jefferson", season=2024, position="WR", team="MIN")
        other_vocabulary = PLAYER_TOOLS[tool_name](
            name="Justin Jefferson", season=2024, position="DL", team="MIN")
        assert "ambiguous" not in same_vocabulary and "not_found" not in same_vocabulary
        assert "error" not in other_vocabulary
        assert other_vocabulary.get("count") == 0 or other_vocabulary.get("rows") == []


class TestSuffixPicksTheJunior:
    """A suffix in the query is a tie-breaker between a father and a son (fix 6)."""

    def test_the_query_that_names_the_junior_gets_the_junior(self):
        assert nfl_resolve_player("Marvin Harrison Jr.")["resolved"]["gsis_id"] == "00-0039849"
        assert nfl_resolve_player("Robert Griffin III")["resolved"]["gsis_id"] == "00-0029665"

    def test_without_the_suffix_the_two_stay_apart(self):
        result = nfl_resolve_player("Marvin Harrison")
        assert result["ambiguous"] is True
        assert sorted(c["full_name"] for c in result["candidates"]) == [
            "Marvin Harrison", "Marvin Harrison Jr."]

    def test_a_suffix_the_table_does_not_store_cannot_break_the_tie(self):
        # Michael Pittman Jr. has a NULL suffix column and no suffix in display_name.
        assert nfl_resolve_player("Michael Pittman Jr.")["ambiguous"] is True

    def test_a_tool_returns_rows_for_a_suffixed_player(self):
        result = nfl_player_stats(player_name="Marvin Harrison Jr.", season=2024,
                                  stat_type="receiving")
        assert result["player"]["gsis_id"] == "00-0039849"
        assert result["seasons"]


class TestCollisionsThatStayOpen:
    """PLAN_A_FIXES section 7.1 and 7.6 record these three. Each test asserts the answer a
    caller should get, so it is expected to fail until someone fixes the defect."""

    @pytest.mark.xfail(strict=True, reason="the surname match takes the last token only, so "
                                           "'Josh Allen' also matches Josh Hines-Allen (7.1)")
    def test_a_compound_surname_should_not_join_the_single_word_search(self):
        assert nfl_resolve_player("Josh Allen", season=2024)["resolved"]["gsis_id"] == "00-0034857"

    @pytest.mark.xfail(strict=True, reason="'St. Brown' folds to the surname token 'brown', so "
                                           "the whole surname is never tried (7.1)")
    def test_a_surname_that_starts_with_a_title_should_be_found(self):
        result = nfl_resolve_player("St. Brown", season=2024)
        assert sorted(c["gsis_id"] for c in result["candidates"]) == ["00-0034279", "00-0036963"]

    @pytest.mark.xfail(strict=True, reason="the echo and the labels read players.latest_team, "
                                           "so a historical query names today's club (7.6)")
    def test_a_historical_query_should_label_the_player_with_his_era_team(self):
        stats = nfl_role_trend(player="Aaron Rodgers", season=2024, limit=5)
        assert stats["role_trend"], stats
        assert stats["player"]["team"] in {row["team"] for row in stats["role_trend"]}


class TestTheSeasonClubSitsBesideTheCurrentOne:
    """team stays players.latest_team, which the xfail above records; season_team names the
    club of the season asked, from rosters, so a reader need not take today's club for it."""

    def test_a_historical_query_names_the_season_club(self):
        stats = nfl_role_trend(player="Aaron Rodgers", season=2024, limit=5)
        assert stats["role_trend"], stats
        assert stats["player"]["season_team"] in {row["team"] for row in stats["role_trend"]}
        assert stats["player"]["season_team"] != stats["player"]["team"]


class TestOneLetterHeadTokenIsNeverSpelledOut:
    """Batch 2 fix 1: a head token is compared with a stored first name only when it holds
    more than one letter, so 'A.St. Brown' cannot borrow A.J. Brown's stored 'A.J.'."""

    def test_a_short_name_with_a_title_does_not_borrow_a_player_it_does_not_name(self):
        result = nfl_resolve_player("A.St. Brown", season=2024)
        assert result["ambiguous"] is True
        assert sorted(c["gsis_id"] for c in result["candidates"]) == [
            "00-0033047", "00-0035676", "00-0036963", "00-0037175", "00-0039511"]
        assert "Amon-Ra St. Brown" in {c["full_name"] for c in result["candidates"]}

    def test_the_player_whose_short_name_is_written_out_resolves_alone(self):
        assert nfl_resolve_player("A.J. Brown", season=2024)["resolved"]["gsis_id"] == "00-0035676"


class TestEraTeamCodesReachTheSameRows:
    """Batch 2 fix 2: the row filter binds the code typed and the current one, or a query for
    a club that moved resolves the player and returns no rows."""

    @pytest.mark.parametrize("code", ["OAK", "LV"])
    def test_a_raider_answers_to_either_code_of_his_club(self, code):
        result = nfl_drop_rate(player="Brock Bowers", team=code, season=2024)
        assert result["count"] == 1
        assert result["drop_rate"][0]["team"] == "LV"

    @pytest.mark.parametrize("era, current, season", [
        ("OAK", "LV", 2024), ("SD", "LAC", 2024), ("STL", "LA", 2024),
        ("OAK", "LV", 2019), ("SD", "LAC", 2015), ("STL", "LA", 2014),
    ])
    def test_both_codes_of_a_moved_club_return_the_same_rows(self, era, current, season):
        # Only 2024 is guaranteed in the database CI builds, so non-emptiness is for 2024.
        rows = {c: nfl_search_plays(team=c, season=season, max_rows=5)["row_count"]
                for c in (era, current)}
        assert rows[era] == rows[current]
        if season == 2024:
            assert rows[era] > 0

    @pytest.mark.parametrize("era, current", [("OAK", "LV"), ("SD", "LAC"), ("STL", "LA")])
    def test_an_opponent_answers_to_either_code(self, era, current):
        assert (nfl_search_plays(opponent=era, season=2024, max_rows=50)["row_count"]
                == nfl_search_plays(opponent=current, season=2024, max_rows=50)["row_count"] > 0)

    def test_a_team_value_of_only_spaces_applies_no_filter(self):
        assert nfl_resolve_player("Patrick Mahomes", team="   ")["resolved"]["gsis_id"] == "00-0033873"
        assert nfl_search_plays(team="   ", season=2024, max_rows=5)["row_count"] == 5


def test_a_row_written_under_the_era_code_is_found_by_the_current_code(tmp_path, monkeypatch):
    """Batch 3 fix 5. The shared database holds 2024-2026, where every table already writes
    'LV', so the era direction needs a database of its own. No network, no ingest.

    latest_team is 'NYJ', not 'LV' on purpose: the player left the club. With latest_team 'LV' a
    caller who types 'LV' matches through players.latest_team and the rosters path never runs, so
    the resolver half of this test would pass even if _team_codes returned one code.
    """
    db = tmp_path / "era.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute("CREATE TABLE players (gsis_id TEXT, display_name TEXT, first_name TEXT, "
                 "football_name TEXT, common_first_name TEXT, last_name TEXT, suffix TEXT, "
                 "position TEXT, latest_team TEXT, rookie_season INT, last_season INT, pfr_id TEXT)")
    conn.execute("INSERT INTO players VALUES ('ERA01', 'Raid Example', 'Raid', NULL, NULL, "
                 "'Example', NULL, 'T', 'NYJ', 2019, 2019, 'ExamRa01')")
    for table in ("rosters", "rosters_weekly"):
        conn.execute(f"CREATE TABLE {table} AS SELECT 'ERA01' AS gsis_id, 'OAK' AS team, "
                     "2019 AS season")
    conn.execute('CREATE TABLE plays (game_id TEXT, season INT, season_type TEXT, week INT, '
                 'posteam TEXT, defteam TEXT, down INT, '
                 'ydstogo INT, play_type TEXT, yards_gained INT, epa DOUBLE, "desc" TEXT, '
                 'passer_player_name TEXT, rusher_player_name TEXT, receiver_player_name TEXT, '
                 'passer_player_id TEXT, rusher_player_id TEXT, receiver_player_id TEXT, '
                 'touchdown INT, interception INT)')
    conn.execute("INSERT INTO plays (season, week, posteam, play_type, epa, rusher_player_name, "
                 "rusher_player_id) VALUES (2019, 1, 'OAK', 'run', 0.1, 'R.Example', 'ERA01')")
    conn.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(db))
    # Two halves, each asserted for both codes over one dict, so a failure names the half and
    # shows what the other code did. The row filter reads plays, which stores 'OAK' for a 2019
    # Raider; the resolver reaches the same player through rosters.
    codes = ("OAK", "LV")
    resolved = {c: nfl_resolve_player("Raid Example", team=c, season=2019)
                    .get("resolved", {}).get("gsis_id") for c in codes}
    assert resolved == {"OAK": "ERA01", "LV": "ERA01"}
    rows = {c: [r["rusher_player_id"] for r in nfl_search_plays(team=c, season=2019)["rows"]]
            for c in codes}
    assert rows == {"OAK": ["ERA01"], "LV": ["ERA01"]}
