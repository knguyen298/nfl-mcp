"""Tests for the ESPN pre-1999 records fetch.

Each test drives a fake transport, so the parsing rules that keep the archive safe —
schedule-length validation, latest-row-wins, conference filtering — run without network.
"""

import pytest

from nfl_mcp import espn
from nfl_mcp.espn import (
    EspnUnavailable,
    _events_for_season,
    _overall_record,
    extract_records,
    fetch_season_records,
    parse_record,
    scoreboard_url,
)


def _event(event_id, date, season_year, competitors, competition_id=None):
    """Build the slice of an ESPN scoreboard event these tests care about."""
    return {
        "id": event_id,
        "date": date,
        "season": {"year": season_year},
        "competitions": [{
            "id": competition_id or f"c{event_id}",
            "competitors": [
                {"team": {"id": str(team_id), "abbreviation": abbrev, "displayName": abbrev},
                 "records": [{"name": "overall", "type": "total", "summary": summary}]}
                for team_id, abbrev, summary in competitors
            ],
        }],
    }


class TestParseRecord:
    @pytest.mark.parametrize("summary,expected", [
        ("12-4", (12, 4, 0)),
        ("11-2-1", (11, 2, 1)),
        ("0-0-1", (0, 0, 1)),
        ("16-0", (16, 0, 0)),
    ])
    def test_parses(self, summary, expected):
        assert parse_record(summary) == expected

    @pytest.mark.parametrize("summary", [None, "", "overall", "12-4-1-2", "ab-cd", "1-2-"])
    def test_rejects_junk(self, summary):
        assert parse_record(summary) is None


class TestOverallRecord:
    def test_picks_the_cumulative_overall_line(self):
        competitor = {"records": [
            {"name": "division", "type": "total", "summary": "4-2"},
            {"name": "overall", "type": "total", "summary": "12-4"},
        ]}
        assert _overall_record(competitor) == "12-4"

    @pytest.mark.parametrize("competitor", [
        {},
        {"records": []},
        {"records": [{"name": "overall", "type": "home", "summary": "6-2"}]},
        {"records": [{"name": "division", "type": "total", "summary": "4-2"}]},
    ])
    def test_missing_returns_none(self, competitor):
        assert _overall_record(competitor) is None


class TestEventSelection:
    def test_filters_by_season_and_dedupes(self):
        events = _events_for_season(
            [{"events": [
                _event("1", "2000-09-03", 2000, [(1, "ATL", "1-0")]),
                _event("2", "2001-01-07", 1999, [(2, "BUF", "8-8")]),   # other season
                _event("1", "2000-09-03", 2000, [(1, "ATL", "1-0")]),   # dup across pulls
                {"id": None},
            ]}, {"events": [
                _event("1", "2000-09-03", 2000, [(1, "ATL", "1-0")]),   # dup again
                _event("3", "2001-01-07", 2000, [(3, "CHI", "5-11")]),
            ]}],
            2000,
        )
        assert [event["id"] for event in events] == ["1", "3"]

    def test_tolerates_payloads_without_events(self):
        assert _events_for_season([{}, {"events": None}], 2000) == []


class TestExtractRecords:
    def test_accepts_a_full_length_record(self):
        records = extract_records(
            [_event("10", "1996-12-22", 1996, [(9, "GB", "13-3"), (22, "ARI", "7-9")])],
            1996,
        )
        assert {r.team: r.record_summary for r in records.values()} == {"GB": "13-3", "ARI": "7-9"}
        record = records[9]
        assert (record.wins, record.losses, record.ties, record.games) == (13, 3, 0, 16)
        assert record.espn_team_id == 9 and record.espn_abbrev == "GB"
        assert record.source_game_id == "10"

    def test_rejects_records_that_do_not_match_the_schedule(self):
        # A 1970 team played 14 games; a 16-game line is a different (or bad) cumulative row.
        records = extract_records(
            [_event("11", "1970-12-13", 1970, [(13, "OAK", "8-6-2"), (15, "MIA", "10-4")])],
            1970,
        )
        assert [r.team for r in records.values()] == ["MIA"]    # 10-4 sums to 14 ✓
        assert records[15].ties == 0

    def test_keeps_ties(self):
        records = extract_records(
            [_event("12", "1970-12-13", 1970, [(22, "CRD", "8-5-1")])], 1970,
        )
        assert (records[22].wins, records[22].losses, records[22].ties) == (8, 5, 1)
        assert records[22].team == "ARI"

    def test_ignores_conferences_and_unmapped_ids(self):
        records = extract_records(
            [_event("13", "1996-12-22", 1996, [
                (31, "AFC", "11-5"), (32, "NFC", "10-6"), (None, "??", "9-7"),
                (9999, "XX", "8-8"), (1, "ATL", "3-13"),
            ])],
            1996,
        )
        assert [r.team for r in records.values()] == ["ATL"]

    def test_latest_dated_row_wins(self):
        # ESPN keeps duplicate, contradictory rows for some 1970s games.
        records = extract_records([
            _event("20", "1970-11-01", 1970, [(13, "OAK", "3-6-1")]),
            _event("21", "1970-12-13", 1970, [(13, "OAK", "5-8-1")]),
        ], 1970)
        assert records[13].record_summary == "5-8-1"
        assert records[13].source_game_id == "21"

    def test_ignores_events_without_competitors(self):
        assert extract_records([{"id": "30", "date": "1996-09-01", "season": {"year": 1996}}], 1996) == {}


class TestFetchSeasonRecords:
    def test_pulls_both_calendar_years_and_stamps_fetch_time(self):
        seen = []

        def transport(url):
            seen.append(url)
            year = int(url.split("dates=")[1].split("&")[0])
            if year == 1996:
                return {"events": [_event("1", "1996-12-22", 1996, [(9, "GB", "13-3")])]}
            return {"events": [_event("2", "1997-01-04", 1996, [(1, "ATL", "3-13")])]} 

        records = fetch_season_records(1996, transport=transport)
        assert seen == [scoreboard_url(1996), scoreboard_url(1997)]
        assert [r.team for r in records] == ["ATL", "GB"]      # sorted by canonical team
        assert all(r.fetched_at for r in records)
        assert records[1].espn_abbrev == "GB"

    def test_empty_season_returns_empty_list(self):
        assert fetch_season_records(1970, transport=lambda url: {}) == []

    def test_retries_then_succeeds(self):
        calls = {"n": 0}
        slept = []

        def transport(url):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("connection reset")
            return {"events": [_event("1", "1996-12-22", 1996, [(9, "GB", "13-3")])]}

        records = fetch_season_records(1996, transport=transport, sleep=slept.append)
        assert [r.team for r in records] == ["GB"]
        # Two pulls per season (calendar Y and Y+1); the retry costs one extra call.
        assert calls["n"] == 3
        assert slept == [pytest.approx(espn.DEFAULT_BACKOFF)]

    def test_gives_up_after_attempts(self):
        slept = []

        def transport(url):
            raise EspnUnavailable("ESPN request failed: 503")

        with pytest.raises(EspnUnavailable, match="ESPN request failed"):
            fetch_season_records(1996, transport=transport, attempts=3, sleep=slept.append)
        assert len(slept) == 2

    def test_scoreboard_url_shape(self):
        assert scoreboard_url(1996) == (
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
            "?dates=1996&limit=1000"
        )


class TestDefaultTransport:
    def test_network_errors_become_espn_unavailable(self, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("no route to host")

        monkeypatch.setattr(espn.urllib.request, "urlopen", boom)
        with pytest.raises(EspnUnavailable, match="ESPN request failed"):
            espn._default_transport("https://example.invalid")

    def test_returns_parsed_json(self, monkeypatch):
        class Response:
            def read(self):
                return b'{"events": []}'

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["agent"] = request.get_header("User-agent")
            captured["timeout"] = timeout
            return Response()

        monkeypatch.setattr(espn.urllib.request, "urlopen", fake_urlopen)
        assert espn._default_transport("https://example.invalid/x") == {"events": []}
        assert captured["agent"] == espn.USER_AGENT
        assert captured["timeout"] == espn.DEFAULT_TIMEOUT
