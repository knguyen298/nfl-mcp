"""v0.7.1: nfl_schedule rows from one team's side.

Source: a model answered a Vikings question from nfl_schedule rows. The rows give away_team,
home_team, away_score, home_score and result (the nflverse home margin, home_score -
away_score). In a game where Minnesota was the away team, the model read the scores by
position, gave the Vikings' points to the other team and called a win a loss.

Interface contract tested here
- With `team`, each row keeps every column and adds `team` (the franchise's code as stored in
  that row, so team='OAK' on a 2020 row gives 'LV'), `opponent`, `location` ('home', 'away',
  or 'neutral' when schedules.location is 'Neutral'), `team_score`, `opponent_score` and
  `outcome` ('W', 'L' or 'T' for that team, null when the game has no score yet).
- Without `team`, the rows have none of these keys.
- The nfl_schedule description says result is the home margin and names team_score,
  opponent_score and outcome.

schedules.location on the 2024 database holds 'Home' (278 games) and 'Neutral' (7 games):
  SELECT location, count(*) FROM schedules WHERE season = 2024 GROUP BY location
    -> Home 278, Neutral 7
The 2024 season has no tie and no game without a score, so the tie, the unplayed game and the
relocated alias use a small database built in tmp_path.

Every expected value on the real database comes from SQL on the raw schedules table, in a
comment next to the value. Integration tests read NFL_MCP_DB_PATH when it is set, else
/tmp/ci2024/v071.duckdb, and skip when the file is missing.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import anyio
import duckdb
import pytest

from nfl_mcp import server, tools
from nfl_mcp.database import get_db_connection
from nfl_mcp.required_columns import REQUIRED_COLUMNS

DB = Path(os.environ.get("NFL_MCP_DB_PATH") or "/tmp/ci2024/v071.duckdb")
TEAM_KEYS = {"team", "opponent", "location", "team_score", "opponent_score", "outcome"}
TOOLS_BY_NAME = {t.name: t for t in server.TOOLS}


def _raw(name, args):
    """A tool call through server.call_tool, as an MCP client sees it."""
    async def go():
        return await server.call_tool(name, args)
    res = anyio.run(go)
    return res.isError, json.loads(res.content[0].text)


def _game(team, game_id, season=2024, season_type="ALL"):
    games = tools.nfl_schedule(team=team, season=season, season_type=season_type)["games"]
    (row,) = [g for g in games if g["game_id"] == game_id]
    return row


@pytest.fixture
def real(monkeypatch):
    """The 2024 database, read-only."""
    if not DB.exists():
        pytest.skip(f"{DB} missing")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(DB))
    with get_db_connection() as conn:
        if not conn.execute("SELECT count(*) FROM schedules WHERE season = 2024").fetchone()[0]:
            pytest.skip("needs 2024 schedules")
        yield conn


# ── the 2024 database ─────────────────────────────────────────────────────────

@pytest.mark.integration
def test_location_values_on_the_real_database(real):
    # SELECT DISTINCT location FROM schedules WHERE season = 2024  -> Home, Neutral
    got = {r[0] for r in real.execute(
        "SELECT DISTINCT location FROM schedules WHERE season = 2024").fetchall()}
    assert got == {"Home", "Neutral"}


@pytest.mark.integration
def test_away_game_scores_are_from_the_team_side(real):
    # SELECT away_team, home_team, away_score, home_score, result, location FROM schedules
    #   WHERE game_id = '2024_11_MIN_TEN'          -> MIN, TEN, 23, 13, -10, Home
    row = _game("MIN", "2024_11_MIN_TEN")
    assert (row["away_team"], row["home_team"], row["away_score"], row["home_score"],
            row["result"]) == ("MIN", "TEN", 23, 13, -10)
    assert row["team"] == "MIN" and row["opponent"] == "TEN"
    assert row["location"] == "away"
    assert row["team_score"] == row["away_score"] == 23
    assert row["opponent_score"] == row["home_score"] == 13
    assert row["outcome"] == "W"


@pytest.mark.integration
def test_away_game_over_mcp(real):
    # SELECT away_score, home_score FROM schedules WHERE game_id = '2024_01_MIN_NYG'  -> 28, 6
    is_error, payload = _raw("nfl_schedule", {"team": "min", "season": 2024, "week": 1,
                                              "season_type": "REG"})
    assert not is_error, payload
    (row,) = payload["games"]
    assert {k: row[k] for k in TEAM_KEYS} == {
        "team": "MIN", "opponent": "NYG", "location": "away", "team_score": 28,
        "opponent_score": 6, "outcome": "W"}


@pytest.mark.integration
def test_the_opponent_sees_the_same_game_from_its_side(real):
    # SELECT away_score, home_score FROM schedules WHERE game_id = '2024_11_MIN_TEN'  -> 23, 13
    row = _game("TEN", "2024_11_MIN_TEN")
    assert (row["team"], row["opponent"], row["location"]) == ("TEN", "MIN", "home")
    assert (row["team_score"], row["opponent_score"], row["outcome"]) == (13, 23, "L")


@pytest.mark.integration
def test_home_win(real):
    # SELECT away_team, home_team, away_score, home_score, location FROM schedules
    #   WHERE game_id = '2024_02_SF_MIN'           -> SF, MIN, 17, 23, Home
    row = _game("MIN", "2024_02_SF_MIN")
    assert (row["team"], row["opponent"], row["location"]) == ("MIN", "SF", "home")
    assert (row["team_score"], row["opponent_score"], row["outcome"]) == (23, 17, "W")


@pytest.mark.integration
def test_home_loss(real):
    # SELECT away_team, home_team, away_score, home_score, location FROM schedules
    #   WHERE game_id = '2024_07_DET_MIN'          -> DET, MIN, 31, 29, Home
    row = _game("MIN", "2024_07_DET_MIN")
    assert (row["team"], row["opponent"], row["location"]) == ("MIN", "DET", "home")
    assert (row["team_score"], row["opponent_score"], row["outcome"]) == (29, 31, "L")


@pytest.mark.integration
def test_listed_home_team_at_a_neutral_site(real):
    # The London game: MIN is the listed home team, the site is neutral.
    # SELECT away_team, home_team, away_score, home_score, location FROM schedules
    #   WHERE game_id = '2024_05_NYJ_MIN'          -> NYJ, MIN, 17, 23, Neutral
    row = _game("MIN", "2024_05_NYJ_MIN")
    assert (row["team"], row["opponent"], row["location"]) == ("MIN", "NYJ", "neutral")
    assert (row["team_score"], row["opponent_score"], row["outcome"]) == (23, 17, "W")


@pytest.mark.integration
@pytest.mark.parametrize("team, want", [
    # SELECT away_team, home_team, away_score, home_score, result, location FROM schedules
    #   WHERE game_type = 'SB' AND season = 2024   -> KC, PHI, 22, 40, 18, Neutral
    ("KC", ("KC", "PHI", "neutral", 22, 40, "L")),
    ("PHI", ("PHI", "KC", "neutral", 40, 22, "W")),
])
def test_super_bowl_lix_is_neutral(real, team, want):
    (row,) = tools.nfl_schedule(team=team, season=2024, season_type="SB")["games"]
    assert row["game_id"] == "2024_22_KC_PHI"
    assert row["result"] == 18  # the home margin, unchanged
    assert (row["team"], row["opponent"], row["location"], row["team_score"],
            row["opponent_score"], row["outcome"]) == want


@pytest.mark.integration
def test_no_team_filter_adds_no_team_keys(real):
    # SELECT count(*) FROM schedules WHERE season = 2024 AND week = 1 AND game_type = 'REG'  -> 16
    res = tools.nfl_schedule(season=2024, week=1, season_type="REG")
    assert res["count"] == 16
    for row in res["games"]:
        assert not TEAM_KEYS & set(row), sorted(TEAM_KEYS & set(row))
        assert {"away_team", "home_team", "away_score", "home_score", "result"} <= set(row)


@pytest.mark.integration
@pytest.mark.parametrize("team", ["MIN", "DET", "KC"])
def test_season_rows_add_up_to_the_team_record(real, team):
    # SELECT sum(CASE WHEN (home_team = 'MIN' AND home_score > away_score)
    #                   OR (away_team = 'MIN' AND away_score > home_score) THEN 1 ELSE 0 END),
    #        sum(CASE WHEN home_score = away_score THEN 1 ELSE 0 END),
    #        sum(CASE WHEN home_team = 'MIN' THEN home_score ELSE away_score END),
    #        sum(CASE WHEN home_team = 'MIN' THEN away_score ELSE home_score END)
    #   FROM schedules WHERE season = 2024 AND game_type = 'REG'
    #     AND (home_team = 'MIN' OR away_team = 'MIN')        -> 14 wins, 0 ties, 432, 332
    # team_records for MIN 2024: 14-3-0, points_for 432, points_against 332.
    raw = real.execute(
        "SELECT sum(CASE WHEN (home_team = ? AND home_score > away_score) "
        "OR (away_team = ? AND away_score > home_score) THEN 1 ELSE 0 END), "
        "sum(CASE WHEN (home_team = ? AND home_score < away_score) "
        "OR (away_team = ? AND away_score < home_score) THEN 1 ELSE 0 END), "
        "sum(CASE WHEN home_score = away_score THEN 1 ELSE 0 END), "
        "sum(CASE WHEN home_team = ? THEN home_score ELSE away_score END), "
        "sum(CASE WHEN home_team = ? THEN away_score ELSE home_score END) "
        "FROM schedules WHERE season = 2024 AND game_type = 'REG' "
        "AND (home_team = ? OR away_team = ?)", [team] * 8).fetchone()
    record = real.execute(
        "SELECT wins, losses, ties, points_for, points_against FROM team_records "
        "WHERE season = 2024 AND team = ?", [team]).fetchone()
    assert tuple(raw) == tuple(record)
    if team == "MIN":
        assert tuple(record) == (14, 3, 0, 432, 332)

    res = tools.nfl_schedule(team=team, season=2024, season_type="REG", limit=50)
    assert not res["truncated"]
    games = res["games"]
    outcomes = Counter(g["outcome"] for g in games)
    assert (outcomes["W"], outcomes["L"], outcomes["T"]) == tuple(record[:3])
    assert set(outcomes) <= {"W", "L", "T"}
    assert sum(g["team_score"] for g in games) == record[3]
    assert sum(g["opponent_score"] for g in games) == record[4]
    assert {g["team"] for g in games} == {team}


# ── a small database: tie, unplayed game, relocated franchise ─────────────────

_SCHED_TYPES = {"season": "INTEGER", "week": "INTEGER", "away_score": "INTEGER",
                "home_score": "INTEGER", "result": "INTEGER"}
_SCHED_ROWS = [
    # game_id, season, game_type, week, away, home, away_score, home_score, location
    ("2019_01_DEN_OAK", 2019, "REG", 1, "DEN", "OAK", 16, 24, "Home"),
    ("2020_02_NO_LV", 2020, "REG", 2, "NO", "LV", 24, 34, "Home"),
    ("2024_03_KC_DEN", 2024, "REG", 3, "KC", "DEN", 17, 17, "Home"),
    ("2025_01_BAL_KC", 2025, "REG", 1, "BAL", "KC", None, None, "Home"),
]


@pytest.fixture
def small_db(tmp_path, monkeypatch):
    path = tmp_path / "sched.duckdb"
    cols = {c: _SCHED_TYPES.get(c, "VARCHAR") for c in REQUIRED_COLUMNS["schedules"]}
    cols.setdefault("location", "VARCHAR")
    con = duckdb.connect(str(path))
    con.execute(f"CREATE TABLE schedules ({', '.join(f'{c} {t}' for c, t in cols.items())})")
    for game_id, season, game_type, week, away, home, a_pts, h_pts, site in _SCHED_ROWS:
        row = {"game_id": game_id, "season": season, "game_type": game_type, "week": week,
               "away_team": away, "home_team": home, "away_score": a_pts, "home_score": h_pts,
               "result": None if h_pts is None else h_pts - a_pts, "location": site}
        con.execute(f"INSERT INTO schedules VALUES ({', '.join('?' for _ in cols)})",
                    [row.get(c) for c in cols])
    con.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


def test_schedules_location_is_a_required_column():
    assert "location" in REQUIRED_COLUMNS["schedules"]


@pytest.mark.parametrize("team", ["KC", "DEN"])
def test_tie(small_db, team):
    row = _game(team, "2024_03_KC_DEN")
    assert (row["team_score"], row["opponent_score"], row["outcome"]) == (17, 17, "T")
    assert row["location"] == ("away" if team == "KC" else "home")


def test_unplayed_game_has_no_scores_and_no_outcome(small_db):
    row = _game("KC", "2025_01_BAL_KC", season=2025)
    assert (row["team"], row["opponent"], row["location"]) == ("KC", "BAL", "home")
    assert row["team_score"] is None and row["opponent_score"] is None
    assert row["outcome"] is None


@pytest.mark.parametrize("team", ["OAK", "LV", "oak"])
def test_relocated_franchise_uses_the_code_of_each_row(small_db, team):
    games = {g["game_id"]: g for g in
             tools.nfl_schedule(team=team, season_type="REG")["games"]}
    assert set(games) == {"2019_01_DEN_OAK", "2020_02_NO_LV"}
    old, new = games["2019_01_DEN_OAK"], games["2020_02_NO_LV"]
    assert (old["team"], old["opponent"], old["location"]) == ("OAK", "DEN", "home")
    assert (old["team_score"], old["opponent_score"], old["outcome"]) == (24, 16, "W")
    assert (new["team"], new["opponent"], new["location"]) == ("LV", "NO", "home")
    assert (new["team_score"], new["opponent_score"], new["outcome"]) == (34, 24, "W")


def test_team_rows_keep_every_column(small_db):
    plain = {g["game_id"]: g for g in tools.nfl_schedule(season_type="ALL")["games"]}
    for row in tools.nfl_schedule(team="KC", season_type="ALL")["games"]:
        assert set(row) == set(plain[row["game_id"]]) | TEAM_KEYS
        assert {k: row[k] for k in plain[row["game_id"]]} == plain[row["game_id"]]


def test_no_team_filter_rows_are_unchanged(small_db):
    for row in tools.nfl_schedule(season_type="ALL")["games"]:
        assert not TEAM_KEYS & set(row)
        assert "location" not in row


# ── description ───────────────────────────────────────────────────────────────

def test_description_explains_result_and_the_team_side():
    text = TOOLS_BY_NAME["nfl_schedule"].description
    assert "home_score - away_score" in text
    for key in ("team_score", "opponent_score", "outcome"):
        assert key in text, key


def test_readme_schedule_row_lists_the_team_fields():
    readme = Path(__file__).resolve().parents[2] / "README.md"
    row = next(line for line in readme.read_text().splitlines()
               if line.startswith("| `nfl_schedule`"))
    for key in ("team_score", "opponent_score", "outcome"):
        assert key in row, key
