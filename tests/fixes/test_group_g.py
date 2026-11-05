"""Group G: player identity and stat definitions.

Part 1 builds a small synthetic DuckDB (hand-counted rows) and points the tools at it
through NFL_MCP_DB_PATH. Part 2 checks nfl_player_stats against the official nflverse
player_stats table in the real database (oracle b).

Interface contract (shared contract plus the decisions below)
--------------------------------------------------------------
nfl_player_lookup(name, team=None, season=None, position=None)
    -> {"candidates": [{"gsis_id", "display_name", "short_name", "position", "teams",
        "first_season", "last_season", "pfr_id"}], "count": int}
    Matching is case-insensitive, accent-insensitive ("Estime" == "Estimé"),
    punctuation-insensitive ("Jamarr" == "Ja'Marr", "Amon Ra St Brown" == "Amon-Ra St. Brown")
    and ignores a generational suffix on either side ("Odell Beckham" == "Odell Beckham Jr.",
    "Patrick Mahomes II" == "Patrick Mahomes"). A bare surname matches every player with
    that last name; a short form ("K.Allen") matches players.short_name (Keenan and Kyle).
    No match -> count 0, no "error" key.
    team: player has a rosters row with that team (in `season` when given).
    season: player has a rosters row in that season.  position: players.position.
nfl_player_stats(player_name=None, player_id=None, season=None, season_from=None,
                 season_to=None, season_type=None)
    -> shared contract shape. Decisions:
    * player_id wins over player_name when both are given; the name is ignored.
    * A player_name that looks like a gsis id (NN-NNNNNNN) is treated as player_id.
    * Unknown player_id -> {"error": ...}.
    * When season is given, name resolution only considers players with a rosters row in
      that season (so "Josh Allen", 2024 is the BUF QB, not the C who retired in 2016).
    * "player_name" in the response is players.display_name.
    * dropbacks = qb_dropback=1 AND two_point_attempt=0 AND play_type<>'no_play' on rows
      where the player is passer_player_id, or rusher_player_id with qb_scramble=1
      (pass attempts excluding spikes + sacks + scrambles). epa_per_dropback is over the
      same rows; epa_per_attempt is over the attempt rows.
    * targets also exclude two_point_attempt=1 (the shared definition does not say so;
      checked on the real DB: with the exclusion 69/69 receivers who had a two-point
      target match player_stats.targets for 2024 REG, without it 0/69 match).
    * spikes (play_type 'qb_spike', pass_attempt=1) count as attempts: checked on the real
      DB, 39/39 passers with a spike in 2024 REG match player_stats.attempts only when the
      spike is counted.
    * yards_per_carry = rushing yards / carries.
nfl_compare(entity1, entity2, compare_type="player", season=..., season_type=...)
    entity may be a name or a gsis id. Result: {"players": [entry1, entry2]} in input
    order, each entry the nfl_player_stats shape (player_id, player_name, seasons). If an
    entity name matches several players: {"ambiguous": True, "candidates": [...],
    "entity": <that input>} and no stats.
nfl_snap_counts(player=None, player_id=None, ...): name matches snap_counts.player (full
    names) with no short-name conversion; player_id maps through players.pfr_id to
    snap_counts.pfr_player_id.
nfl_ftn_charting(player=None, player_id=None, ...): one player only, matched by id
    columns; a name with several matches -> {"ambiguous": True, "candidates": [...]}.
nfl_drop_rate / nfl_td_luck / nfl_role_trend / nfl_contract_value /
nfl_fantasy_opportunity(player=None, player_id=None, ...): player_id filters on the
    table's player_id column; a name resolves through players first (so "Ja'Marr Chase"
    finds the short-form "J.Chase" drop-rate row); several matches -> ambiguous.
nfl_fantasy_rankings(player=...): ff_rankings_* have no gsis id; the name match ignores
    apostrophes and periods on both sides ("Ja'Marr Chase" finds "JaMarr Chase").
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from nfl_mcp import tools
from nfl_mcp.config import get_duckdb_path
from nfl_mcp.database import get_db_connection

# ── ids (real gsis ids; the synthetic DB reuses them) ─────────────────────────
JA = "00-0034857"   # Josh Allen QB BUF
JA_C = "00-0030833"  # Josh Allen C, last season 2016 (real DB only)
BA = "00-0032434"   # Brandon Allen QB
KA = "00-0030279"   # Keenan Allen WR
KY = "00-0034577"   # Kyle Allen QB (short name K.Allen, same as Keenan)
JV = "00-0036997"   # Javonte Williams RB (J.Williams)
JM = "00-0037240"   # Jameson Williams WR (J.Williams)
OB = "00-0031235"   # Odell Beckham Jr.
CH = "00-0036900"   # Ja'Marr Chase
AS = "00-0036963"   # Amon-Ra St. Brown
AE = "00-0039373"   # Audric Estimé
TK = "00-0030506"   # Travis Kelce
PM = "00-0033873"   # Patrick Mahomes
SB = "00-0034844"   # Saquon Barkley
ALLENS = {JA, BA, KA, KY}


def call(_tool, **kwargs):
    fn = getattr(tools, _tool, None)
    assert fn is not None, f"{_tool} missing"
    return fn(**kwargs)


def _assert_ambiguous(res, ids=None, min_n=2):
    assert isinstance(res, dict) and "error" not in res, res
    assert res.get("ambiguous") is True, res
    got = {c["gsis_id"] for c in res["candidates"]}
    assert len(got) >= min_n
    if ids is not None:
        assert ids <= got


def _entry(res, season_type="REG"):
    return next(s for s in res["seasons"] if s["season_type"] == season_type)


# ── synthetic database ───────────────────────────────────────────────────────

def _table(con, name, cols, rows):
    con.execute(f"CREATE TABLE {name} (" + ", ".join(f'"{c}" {t}' for c, t in cols.items()) + ")")
    for r in rows:
        extra = set(r) - set(cols)
        assert not extra, (name, extra)
    if rows:
        ph = ", ".join("?" for _ in cols)
        con.executemany(f"INSERT INTO {name} VALUES ({ph})", [[r.get(c) for c in cols] for r in rows])


PLAYER_COLS = dict(
    gsis_id="VARCHAR", display_name="VARCHAR", common_first_name="VARCHAR", first_name="VARCHAR",
    last_name="VARCHAR", short_name="VARCHAR", football_name="VARCHAR", suffix="VARCHAR",
    pfr_id="VARCHAR", position_group="VARCHAR", position="VARCHAR", rookie_season="INTEGER",
    last_season="INTEGER", latest_team="VARCHAR", status="VARCHAR",
)
# gsis, display, first, last, short, pos, pfr, latest, rookie
_PLAYERS = [
    (JA, "Josh Allen", "Josh", "Allen", "J.Allen", "QB", "AlleJo02", "BUF", 2018),
    (BA, "Brandon Allen", "Brandon", "Allen", "B.Allen", "QB", "AlleBr00", "TEN", 2016),
    (KA, "Keenan Allen", "Keenan", "Allen", "K.Allen", "WR", "AlleKe00", "IND", 2013),
    (KY, "Kyle Allen", "Kyle", "Allen", "K.Allen", "QB", "AlleKy00", "PIT", 2018),
    (JV, "Javonte Williams", "Javonte", "Williams", "J.Williams", "RB", "WillJa10", "DEN", 2021),
    (JM, "Jameson Williams", "Jameson", "Williams", "J.Williams", "WR", "WillJa11", "DET", 2022),
    (OB, "Odell Beckham Jr.", "Odell", "Beckham", "O.Beckham", "WR", "BeckOd00", "MIA", 2014),
    (CH, "Ja'Marr Chase", "Ja'Marr", "Chase", "J.Chase", "WR", "ChasJa00", "CIN", 2021),
    (AS, "Amon-Ra St. Brown", "Amon-Ra", "St. Brown", "A.St. Brown", "WR", "StxxAm00", "DET", 2021),
    (AE, "Audric Estimé", "Audric", "Estimé", "A.Estime", "RB", "EstiAu00", "PIT", 2024),
    (TK, "Travis Kelce", "Travis", "Kelce", "T.Kelce", "TE", "KelcTr00", "KC", 2013),
    (PM, "Patrick Mahomes", "Patrick", "Mahomes", "P.Mahomes", "QB", "MahoPa00", "KC", 2017),
    (SB, "Saquon Barkley", "Saquon", "Barkley", "S.Barkley", "RB", "BarkSa00", "PHI", 2018),
]
# 2023: Josh BUF, Keenan LAC, Brandon CIN (Kyle has no 2023 row). 2024: everyone.
_ROSTERS = [(JA, 2023, "BUF"), (KA, 2023, "LAC"), (BA, 2023, "CIN"),
            (JA, 2024, "BUF"), (KA, 2024, "CHI"), (BA, 2024, "SF"), (KY, 2024, "PIT"),
            (JV, 2024, "DEN"), (JM, 2024, "DET"), (OB, 2024, "MIA"), (CH, 2024, "CIN"),
            (AS, 2024, "DET"), (AE, 2024, "PIT"), (TK, 2024, "KC"), (PM, 2024, "KC"),
            (SB, 2024, "PHI")]

PLAY_COLS = {c: "DOUBLE" for c in (
    "play_id pass_attempt sack qb_dropback qb_scramble qb_kneel qb_spike rush_attempt "
    "two_point_attempt complete_pass incomplete_pass interception passing_yards rushing_yards "
    "receiving_yards yards_gained air_yards yards_after_catch pass_touchdown rush_touchdown "
    "touchdown fumble_lost epa qb_epa cpoe").split()}
PLAY_COLS.update({c: "VARCHAR" for c in (
    "game_id season_type posteam defteam play_type passer_player_id passer_player_name "
    "rusher_player_id rusher_player_name receiver_player_id receiver_player_name "
    "lateral_receiver_player_id lateral_rusher_player_id td_player_id").split()})
PLAY_COLS.update(lateral_receiving_yards="DOUBLE", lateral_rushing_yards="DOUBLE")
PLAY_COLS.update(season="INTEGER", week="INTEGER")
_NAME = {JA: "J.Allen", BA: "B.Allen", KA: "K.Allen", KY: "K.Allen", JV: "J.Williams",
         JM: "J.Williams", OB: "O.Beckham", CH: "J.Chase", AE: "A.Estime", TK: "T.Kelce",
         PM: "P.Mahomes", SB: "S.Barkley"}

_PID = iter(range(1, 1000))


def _pass(passer, receiver=None, *, complete=0, yds=None, td=0, int_=0, epa=0.0, pt="pass",
          sack=0, dropback=1, two=0, spike=0, st="REG", week=1, team="BUF"):
    return dict(play_id=next(_PID), game_id=f"2024_{week:02d}_{team}", season=2024, week=week,
                season_type=st, posteam=team, defteam="OPP", play_type=pt, pass_attempt=1,
                sack=sack, qb_dropback=dropback, qb_scramble=0, qb_kneel=0, qb_spike=spike,
                rush_attempt=0, two_point_attempt=two, complete_pass=complete,
                incomplete_pass=1 - complete if not sack else 0, interception=int_,
                passing_yards=yds, receiving_yards=yds, yards_gained=yds or 0,
                pass_touchdown=td, rush_touchdown=0, touchdown=td, fumble_lost=0, epa=epa,
                qb_epa=epa, passer_player_id=passer, passer_player_name=_NAME[passer],
                receiver_player_id=receiver, receiver_player_name=_NAME.get(receiver))


def _run(rusher, *, yds, td=0, epa=0.0, pt="run", scramble=0, two=0, st="REG", week=1, team="BUF"):
    return dict(play_id=next(_PID), game_id=f"2024_{week:02d}_{team}", season=2024, week=week,
                season_type=st, posteam=team, defteam="OPP", play_type=pt, pass_attempt=0,
                sack=0, qb_dropback=scramble, qb_scramble=scramble,
                qb_kneel=1 if pt == "qb_kneel" else 0, qb_spike=0, rush_attempt=1,
                two_point_attempt=two, complete_pass=0, incomplete_pass=0, interception=0,
                rushing_yards=None if two else yds, yards_gained=yds, pass_touchdown=0,
                rush_touchdown=td, touchdown=td, fumble_lost=0, epa=epa, qb_epa=epa,
                rusher_player_id=rusher, rusher_player_name=_NAME[rusher])


# Josh Allen, 2024 REG unless noted. FTN marks which plays get an ftn_charting row.
P1 = _pass(JA, CH, complete=1, yds=20, epa=1.2)               # attempt, completion, target
P2 = _pass(JA, CH, complete=1, yds=10, td=1, epa=2.0)          # attempt, completion, TD
P3 = _pass(JA, CH, epa=-0.4)                                   # attempt, incomplete
P4 = _pass(JA, KA, int_=1, epa=-2.0)                           # attempt, interception
P5 = _pass(JA, sack=1, epa=-2.0)                               # sack: dropback, not attempt
P6 = _pass(JA, pt="qb_spike", spike=1, dropback=0, epa=-0.3)   # spike: attempt, not dropback
P7 = _pass(JA, CH, complete=1, two=1, epa=0.0)                 # two-point: nothing
P8 = _pass(JA, pt="no_play", dropback=0, epa=0.9)              # no_play: nothing
R1 = _run(JA, yds=5, epa=0.2)                                  # carry
R2 = _run(JA, yds=12, td=1, scramble=1, epa=1.2)               # carry, scramble, dropback
R3 = _run(JA, yds=-1, pt="qb_kneel", epa=-0.1)                 # kneel: carry
R4 = _run(JA, yds=2, two=1, epa=0.0)                           # two-point run: nothing
P9 = _pass(JA, CH, complete=1, yds=15, epa=0.5, st="POST", week=19)
B1 = _pass(BA, KA, complete=1, yds=8, epa=0.3, week=2, team="SF")
K1 = _pass(KY, JM, complete=1, yds=6, epa=0.1, week=3, team="PIT")
W1 = _run(JV, yds=4, team="DEN")
W2 = _run(JV, yds=6, team="DEN")
W3 = _run(JM, yds=9, team="DET")
O1 = _pass(PM, OB, complete=1, yds=7, team="KC")
E1 = _run(AE, yds=3, team="PIT")
S1 = _run(SB, yds=10, td=1, team="PHI")
T1 = _pass(PM, TK, complete=1, yds=11, team="KC")
# Laterals, in their own game (week 4) and on players no other test counts.
# L1: Mahomes -> Beckham, 4 yds, lateral to Kelce for 30 more, Kelce scores.
L1 = _pass(PM, OB, complete=1, yds=34, td=1, epa=1.0, week=4, team="KC")
L1.update(receiving_yards=4, lateral_receiver_player_id=TK, lateral_receiving_yards=30,
          td_player_id=TK)
# L2: Barkley runs 3 yds, laterals to Estime who gains 7 more.
L2 = _run(SB, yds=3, week=4, team="PHI")
L2.update(lateral_rusher_player_id=AE, lateral_rushing_yards=7)
PLAYS = [P1, P2, P3, P4, P5, P6, P7, P8, R1, R2, R3, R4, P9, B1, K1, W1, W2, W3, O1, E1, S1, T1,
         L1, L2]
FTN_PLAYS = [(P1, True), (P2, False), (P3, False), (P4, False), (P5, False), (R1, True),
             (R2, False), (P9, False), (B1, False), (K1, True)]  # (play, is_play_action)


def _build_db(path: Path) -> None:
    con = duckdb.connect(str(path))
    _table(con, "players", PLAYER_COLS, [
        dict(gsis_id=g, display_name=d, common_first_name=f, football_name=f, first_name=f,
             last_name=l, short_name=s, position=p, pfr_id=pfr, latest_team=t,
             rookie_season=r, last_season=2026, status="ACT")
        for g, d, f, l, s, p, pfr, t, r in _PLAYERS])
    _table(con, "rosters", dict(gsis_id="VARCHAR", season="INTEGER", team="VARCHAR",
                                full_name="VARCHAR", position="VARCHAR", pfr_id="VARCHAR"),
           [dict(gsis_id=g, season=s, team=t) for g, s, t in _ROSTERS])
    _table(con, "plays", PLAY_COLS, PLAYS)
    _table(con, "ftn_charting", dict(
        nflverse_game_id="VARCHAR", nflverse_play_id="INTEGER", season="INTEGER", week="INTEGER",
        is_play_action="BOOLEAN", is_rpo="BOOLEAN", is_screen_pass="BOOLEAN",
        is_no_huddle="BOOLEAN", is_motion="BOOLEAN", is_trick_play="BOOLEAN",
        n_defense_box="INTEGER", n_pass_rushers="INTEGER", n_blitzers="INTEGER",
        is_drop="BOOLEAN", is_catchable_ball="BOOLEAN"),
        [dict(nflverse_game_id=p["game_id"], nflverse_play_id=int(p["play_id"]), season=2024,
              week=p["week"], is_play_action=pa, is_rpo=False, is_screen_pass=False,
              is_no_huddle=False, is_motion=False, is_trick_play=False, n_defense_box=7,
              n_pass_rushers=4, n_blitzers=0) for p, pa in FTN_PLAYS])
    snap_cols = dict(game_id="VARCHAR", pfr_game_id="VARCHAR", season="INTEGER",
                     game_type="VARCHAR", week="INTEGER", player="VARCHAR",
                     pfr_player_id="VARCHAR", position="VARCHAR", team="VARCHAR",
                     opponent="VARCHAR", offense_snaps="DOUBLE", offense_pct="DOUBLE",
                     defense_snaps="DOUBLE", defense_pct="DOUBLE", st_snaps="DOUBLE",
                     st_pct="DOUBLE")
    _table(con, "snap_counts", snap_cols, [
        dict(season=2024, game_type="REG", week=w, player=n, pfr_player_id=pid, position=pos,
             team="KC", opponent="BAL", offense_snaps=s, offense_pct=s / 60)
        for w, n, pid, pos, s in [(1, "Travis Kelce", "KelcTr00", "TE", 48),
                                  (2, "Travis Kelce", "KelcTr00", "TE", 50),
                                  (1, "Noah Gray", "GrayNo00", "TE", 30),
                                  (2, "Noah Gray", "GrayNo00", "TE", 28),
                                  (1, "Patrick Mahomes", "MahoPa00", "QB", 60)]])
    _table(con, "player_drop_rate", dict(
        player_id="VARCHAR", player="VARCHAR", team="VARCHAR", season="INTEGER",
        catchable_targets="BIGINT", drops="BIGINT", drop_rate_pct="DOUBLE",
        contested_targets="BIGINT", created_receptions="BIGINT"),
        [dict(player_id=g, player=_NAME[g], team=t, season=2024, catchable_targets=c, drops=d,
              drop_rate_pct=round(100 * d / c, 1), contested_targets=1, created_receptions=0)
         for g, t, c, d in [(KA, "CHI", 50, 3), (CH, "CIN", 120, 4), (JV, "DEN", 40, 2),
                            (JM, "DET", 45, 1)]])
    who = [(JA, "Josh Allen", "QB", "BUF"), (KA, "Keenan Allen", "WR", "CHI"),
           (BA, "Brandon Allen", "QB", "SF"), (CH, "Ja'Marr Chase", "WR", "CIN")]
    _table(con, "player_td_luck", dict(
        player_id="VARCHAR", full_name="VARCHAR", position="VARCHAR", team="VARCHAR",
        season="INTEGER", rec_td="DOUBLE", rec_td_exp="DOUBLE", rec_td_luck="DOUBLE",
        rush_td="DOUBLE", rush_td_exp="DOUBLE", rush_td_luck="DOUBLE",
        total_td_luck_score="DOUBLE"),
        [dict(player_id=g, full_name=n, position=p, team=t, season=2024,
              total_td_luck_score=float(i)) for i, (g, n, p, t) in enumerate(who)])
    _table(con, "player_role_trend", dict(
        player_id="VARCHAR", full_name="VARCHAR", position="VARCHAR", team="VARCHAR",
        season="INTEGER", week="INTEGER",
        **{c: "DOUBLE" for c in [f"{m}{sfx}" for sfx in ("", "_3wk", "_delta") for m in (
            "snap_pct", "target_share_pct", "carry_share_pct", "air_yards_share_pct")]}),
        [dict(player_id=g, full_name=n, position=p, team=t, season=2024, week=w,
              snap_pct=80.0, snap_pct_delta=float(w)) for g, n, p, t in who for w in (1, 2)])
    _table(con, "player_contract_value", dict(
        player_id="VARCHAR", full_name="VARCHAR", position="VARCHAR", team="VARCHAR",
        season="INTEGER", apy="DOUBLE", cap_pct="DOUBLE", total_fp="DOUBLE",
        total_fp_exp="DOUBLE", fp_per_million="DOUBLE"),
        [dict(player_id=g, full_name=n, position=p, team=t, season=2024, apy=10.0,
              fp_per_million=float(i)) for i, (g, n, p, t) in enumerate(who)])
    ffo_cols = {c: "DOUBLE" for c in (
        "week rec_attempt rush_attempt rec_air_yards receptions receptions_exp "
        "rec_yards_gained rush_yards_gained rec_touchdown rush_touchdown total_fantasy_points "
        "total_fantasy_points_exp total_fantasy_points_diff rec_attempt_team "
        "rush_attempt_team").split()}
    ffo_cols.update(season="VARCHAR", posteam="VARCHAR", full_name="VARCHAR",
                    position="VARCHAR", player_id="VARCHAR")
    _table(con, "ff_opportunity", ffo_cols,
           [dict(season="2024", week=1.0, posteam=t, full_name=n, position=p, player_id=g,
                 total_fantasy_points_exp=10.0) for g, n, p, t in who])
    _table(con, "ff_rankings_week", dict(
        page="VARCHAR", player_name="VARCHAR", pos="VARCHAR", team="VARCHAR", rank="BIGINT",
        ecr="DOUBLE", sd="DOUBLE", best="BIGINT", worst="BIGINT", pos_rank="VARCHAR",
        player_opponent="VARCHAR", player_bye_week="BIGINT", start_sit_grade="VARCHAR",
        note="VARCHAR", scrape_date="VARCHAR"),
        [dict(page="ppr-wr", player_name=n, pos=p, team=t, ecr=e)
         for n, p, t, e in [("JaMarr Chase", "WR", "CIN", 1.0),
                            ("Amon-Ra St Brown", "WR", "DET", 3.0),
                            ("Keenan Allen", "WR", "CHI", 30.0)]])
    _table(con, "ff_rankings_draft", dict(
        page_type="VARCHAR", ecr_type="VARCHAR", player="VARCHAR", pos="VARCHAR",
        team="VARCHAR", ecr="DOUBLE", sd="DOUBLE", best="BIGINT", worst="BIGINT", bye="BIGINT",
        scrape_date="VARCHAR"),
        [dict(page_type="redraft-overall", player=n, pos=p, team=t, ecr=e)
         for n, p, t, e in [("Ja'Marr Chase", "WR", "CIN", 1.0),
                            ("Amon-Ra St. Brown", "WR", "DET", 5.0)]])
    con.close()


@pytest.fixture(scope="module")
def _synthetic_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("grp_g") / "synthetic.duckdb"
    _build_db(path)
    return path


@pytest.fixture
def syn(_synthetic_path, monkeypatch):
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(_synthetic_path))
    return _synthetic_path


# ── 1. nfl_player_lookup ─────────────────────────────────────────────────────

def _lookup_ids(**kw):
    res = call("nfl_player_lookup", **kw)
    assert "error" not in res, res
    assert res["count"] == len(res["candidates"])
    return {c["gsis_id"] for c in res["candidates"]}


@pytest.mark.parametrize("name", ["Allen", "allen", "ALLEN"])
def test_lookup_surname_returns_all_four_allens(syn, name):
    assert _lookup_ids(name=name) == ALLENS  # 4 Allen rows in players


def test_lookup_candidate_has_contract_keys(syn):
    res = call("nfl_player_lookup", name="Keenan Allen")
    (c,) = res["candidates"]
    assert c["gsis_id"] == KA and c["position"] == "WR" and c["pfr_id"] == "AlleKe00"
    assert {"display_name", "short_name", "teams", "first_season", "last_season"} <= set(c)


@pytest.mark.parametrize("kw, expected", [
    (dict(name="Josh Allen"), {JA}),
    (dict(name="Allen", team="CHI"), {KA}),               # only Keenan has a CHI roster row
    (dict(name="Allen", team="BUF", season=2024), {JA}),
    (dict(name="Allen", season=2023), {JA, KA, BA}),      # Kyle has no 2023 roster row
    (dict(name="Allen", position="WR"), {KA}),
    (dict(name="Allen", position="QB"), {JA, BA, KY}),
])
def test_lookup_full_name_and_filters_narrow(syn, kw, expected):
    assert _lookup_ids(**kw) == expected


@pytest.mark.parametrize("name, gid", [
    ("Odell Beckham Jr.", OB), ("Odell Beckham", OB), ("odell beckham jr", OB),
    ("Patrick Mahomes II", PM),
    ("Ja'Marr Chase", CH), ("Jamarr Chase", CH), ("JaMarr Chase", CH),
    ("Amon-Ra St. Brown", AS), ("Amon Ra St Brown", AS),
    ("Audric Estimé", AE), ("Audric Estime", AE), ("estime", AE),
])
def test_lookup_suffix_punctuation_accent_variants(syn, name, gid):
    assert _lookup_ids(name=name) == {gid}


@pytest.mark.parametrize("name", ["Zzyzx Nobody", "Allenby"])
def test_lookup_no_match_is_empty_not_error(syn, name):
    res = call("nfl_player_lookup", name=name)
    assert "error" not in res and res["count"] == 0 and res["candidates"] == []


# ── 2. nfl_player_stats identity ─────────────────────────────────────────────

@pytest.mark.parametrize("kw", [dict(player_name="Allen", season=2024),
                                dict(player_name="Allen"),
                                dict(player_name="K.Allen", season=2024)])
def test_player_stats_ambiguous_name_returns_candidates(syn, kw):
    res = call("nfl_player_stats", **kw)
    _assert_ambiguous(res, ids={KA, KY} if kw["player_name"] == "K.Allen" else ALLENS)
    assert "seasons" not in res


@pytest.mark.parametrize("kw, gid", [
    (dict(player_name="Josh Allen"), JA),
    (dict(player_id=JA), JA),
    (dict(player_name=JA), JA),                       # gsis id passed as the name
    (dict(player_name="Keenan Allen", player_id=JA), JA),  # player_id wins
    (dict(player_name="Jamarr Chase"), CH),
])
def test_player_stats_resolves_single_player(syn, kw, gid):
    res = call("nfl_player_stats", season=2024, **kw)
    assert "error" not in res and not res.get("ambiguous"), res
    assert res["player_id"] == gid
    assert res["player_name"] == dict((p[0], p[1]) for p in _PLAYERS)[gid]


@pytest.mark.parametrize("gid", ["00-9999999", "not-an-id"])
def test_player_stats_unknown_player_id_is_error(syn, gid):
    res = call("nfl_player_stats", player_id=gid)
    assert "error" in res


# ── 3. stat definitions (hand counts on the synthetic plays) ─────────────────

@pytest.fixture
def ja_reg(syn):
    res = call("nfl_player_stats", player_id=JA, season=2024, season_type="REG")
    assert "error" not in res, res
    return _entry(res)


@pytest.mark.parametrize("key, expected", [
    ("attempts", 5),        # P1 P2 P3 P4 P6(spike); not P5 sack, P7 2pt, P8 no_play
    ("completions", 2),     # P1 P2 (P7 is a two-point try)
    ("yards", 30),          # 20 + 10
    ("touchdowns", 1),      # P2
    ("interceptions", 1),   # P4
    ("sacks", 1),           # P5
    ("dropbacks", 6),       # P1-P4 + P5 sack + R2 scramble; spike/2pt/no_play excluded
])
def test_passing_counts_follow_definitions(ja_reg, key, expected):
    assert ja_reg["passing"][key] == expected


def test_epa_per_attempt_and_per_dropback_differ_with_negative_sack(ja_reg):
    # attempts: 1.2 + 2.0 - 0.4 - 2.0 - 0.3 = 0.5 over 5 -> 0.1
    assert ja_reg["passing"]["epa_per_attempt"] == pytest.approx(0.1, abs=1e-3)
    # dropbacks: 1.2 + 2.0 - 0.4 - 2.0 - 2.0(sack) + 1.2(scramble) = 0.0 over 6 -> 0.0
    assert ja_reg["passing"]["epa_per_dropback"] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("key, expected", [
    ("carries", 3),          # R1, R2 scramble, R3 kneel; not R4 two-point
    ("yards", 16),           # 5 + 12 - 1
    ("touchdowns", 1),       # R2
    ("scrambles", 1),        # R2
])
def test_rushing_counts_follow_definitions(ja_reg, key, expected):
    assert ja_reg["rushing"][key] == expected


def test_yards_per_carry_is_yards_over_carries(ja_reg):
    assert ja_reg["rushing"]["yards_per_carry"] == pytest.approx(16 / 3, abs=0.01)


@pytest.mark.parametrize("gid, expected", [
    (CH, dict(targets=3, receptions=2, yards=30, touchdowns=1)),  # P1-P3; P7 2pt excluded
    (KA, dict(targets=2, receptions=1, yards=8, touchdowns=0)),   # P4 (Josh) + B1 (Brandon)
])
def test_receiving_counts_follow_definitions(syn, gid, expected):
    res = call("nfl_player_stats", player_id=gid, season=2024, season_type="REG")
    rec = _entry(res)["receiving"]
    assert {k: rec[k] for k in expected} == expected


def test_player_stats_splits_reg_and_post(syn):
    res = call("nfl_player_stats", player_id=JA, season=2024)
    by_type = {s["season_type"]: s for s in res["seasons"]}
    assert set(by_type) == {"REG", "POST"}
    assert by_type["POST"]["passing"]["attempts"] == 1  # P9


def test_definitions_block_names_each_rule(syn):
    res = call("nfl_player_stats", player_id=JA, season=2024)
    defs = res["definitions"]
    assert isinstance(defs, dict)
    for key in ("attempts", "dropbacks", "carries", "scrambles", "targets"):
        assert key in defs and isinstance(defs[key], str) and defs[key]
    assert "sack" in defs["attempts"] and "two_point" in defs["attempts"]


@pytest.mark.parametrize("gid, block, expected", [
    # O1 (7 yds) + L1 (4 yds caught; the 30 after the lateral belong to Kelce, so is the TD)
    (OB, "receiving", dict(targets=2, receptions=2, yards=11, touchdowns=0)),
    # T1 (target, 11 yds) + 30 lateral yards and the TD from L1; L1 is not his target
    (TK, "receiving", dict(targets=1, receptions=1, yards=41, touchdowns=1)),
    # O1 + T1 + L1 are attempts; yards 7 + 11 + 34; one TD (L1)
    (PM, "passing", dict(attempts=3, completions=3, yards=52, touchdowns=1)),
    # S1 (10 yds, TD) + L2 (3 yds before the lateral); 2 carries
    (SB, "rushing", dict(carries=2, yards=13, touchdowns=1)),
    # E1 (3 yds) + 7 lateral yards from L2; the lateral is not a carry for him
    (AE, "rushing", dict(carries=1, yards=10, touchdowns=0)),
])
def test_lateral_yards_and_touchdowns_go_to_the_player_who_got_them(syn, gid, block, expected):
    res = call("nfl_player_stats", player_id=gid, season=2024, season_type="REG")
    got = _entry(res)[block]
    assert {k: got[k] for k in expected} == expected


def test_player_stats_lists_reg_before_post(syn):
    res = call("nfl_player_stats", player_id=JA, season=2024)
    assert [s["season_type"] for s in res["seasons"]] == ["REG", "POST"]


def test_definitions_note_the_reversed_no_play_pass(syn):
    defs = call("nfl_player_stats", player_id=JA, season=2024)["definitions"]
    assert "no_play" in defs["attempts"] and "one below" in defs["attempts"]
    assert "lateral" in defs["yards"] and "td_player_id" in defs["touchdowns"]


# ── name resolution when the filter leaves nobody ────────────────────────────

@pytest.mark.parametrize("name, ids", [("Allen", ALLENS), ("J.Williams", {JV, JM})])
def test_player_stats_season_without_roster_rows_is_still_ambiguous(syn, name, ids):
    res = call("nfl_player_stats", player_name=name, season=2019)  # rosters start in 2023 here
    _assert_ambiguous(res, ids=ids)


def test_player_stats_season_without_roster_rows_uses_a_lone_match(syn):
    res = call("nfl_player_stats", player_name="Patrick Mahomes", season=2019)
    assert res["player_id"] == PM and res["seasons"] == []


def test_compare_season_without_roster_rows_is_ambiguous(syn):
    res = call("nfl_compare", entity1="Allen", entity2="Patrick Mahomes", compare_type="player",
               season=2019)
    _assert_ambiguous(res, ids=ALLENS)


# ── partial names on derived tools ───────────────────────────────────────────

def test_derived_tool_partial_name_matching_several_ids_is_ambiguous(syn):
    res = call("nfl_td_luck", player="Alle", season=2024)  # "Alle" is no player's name
    _assert_ambiguous(res, ids={JA, KA, BA})
    assert not res.get("td_luck")


def test_derived_tool_partial_name_matching_one_id_filters_on_it(syn):
    res = call("nfl_td_luck", player="Chas", season=2024)
    assert {r["player_id"] for r in res["td_luck"]} == {CH}
    assert res["player_id"] == CH


@pytest.mark.parametrize("text", ["%", "_", "A_len"])
def test_derived_tool_partial_name_wildcards_are_literal(syn, text):
    assert call("nfl_td_luck", player=text)["td_luck"] == []


# ── 4. nfl_compare player mode ───────────────────────────────────────────────

def test_compare_two_same_short_name_players_by_id(syn):
    res = call("nfl_compare", entity1=JV, entity2=JM, compare_type="player", season=2024)
    players = res["players"]
    assert [p["player_id"] for p in players] == [JV, JM]
    # W1 + W2 for Javonte (2 carries, 10 yds); W3 for Jameson (1 carry, 9 yds)
    assert [_entry(p)["rushing"]["carries"] for p in players] == [2, 1]
    assert [_entry(p)["rushing"]["yards"] for p in players] == [10, 9]


@pytest.mark.parametrize("e1, e2", [("Allen", "Patrick Mahomes"), ("Patrick Mahomes", "J.Williams")])
def test_compare_ambiguous_name_returns_candidates(syn, e1, e2):
    res = call("nfl_compare", entity1=e1, entity2=e2, compare_type="player", season=2024)
    _assert_ambiguous(res)
    assert "players" not in res


# ── 5. nfl_snap_counts ───────────────────────────────────────────────────────

def _snap_players(res):
    assert "error" not in res, res
    return [r["player"] for r in res["snap_counts"]]


@pytest.mark.parametrize("kw", [dict(player="Travis Kelce"), dict(player="travis kelce"),
                                dict(player_id=TK)])
def test_snap_counts_full_name_or_id_returns_rows(syn, kw):
    assert _snap_players(call("nfl_snap_counts", season=2024, **kw)) == ["Travis Kelce"] * 2


def test_snap_counts_surname_still_works(syn):
    assert _snap_players(tools.nfl_snap_counts(player="Kelce", season=2024)) == ["Travis Kelce"] * 2


# ── 6. nfl_ftn_charting ──────────────────────────────────────────────────────

@pytest.mark.parametrize("kw", [dict(player_id=JA), dict(player="Josh Allen")])
def test_ftn_charting_one_player_by_id_or_full_name(syn, kw):
    res = call("nfl_ftn_charting", season=2024, season_type="REG", **kw)
    # P1 P2 P3 P4 P5 R1 R2 have FTN rows (P9 is POST); play action on P1 and R1
    assert res["total_plays"] == 7
    assert res["charting"]["play_action"]["plays"] == 2


def test_ftn_charting_receiver_by_id(syn):
    res = call("nfl_ftn_charting", player_id=KA, season=2024)
    assert res["total_plays"] == 2  # P4 (from Josh) + B1 (from Brandon)


@pytest.mark.parametrize("name", ["Allen", "K.Allen"])
def test_ftn_charting_ambiguous_name_not_merged(syn, name):
    res = call("nfl_ftn_charting", player=name, season=2024)
    _assert_ambiguous(res)
    assert not res.get("charting")


# ── 7. derived player tools ──────────────────────────────────────────────────

LIST_TOOLS = [
    ("nfl_drop_rate", "drop_rate", CH, "Williams", {JV, JM}),
    ("nfl_td_luck", "td_luck", KA, "Allen", {JA, KA, BA}),
    ("nfl_role_trend", "role_trend", KA, "Allen", {JA, KA, BA}),
    ("nfl_contract_value", "contract_value", KA, "Allen", {JA, KA, BA}),
    ("nfl_fantasy_opportunity", "fantasy_opportunity", KA, "Allen", {JA, KA, BA}),
]


@pytest.mark.parametrize("fn, key, gid, _n, _ids", LIST_TOOLS)
def test_derived_tool_filters_by_player_id(syn, fn, key, gid, _n, _ids):
    rows = call(fn, player_id=gid)[key]
    assert rows and {r["player_id"] for r in rows} == {gid}


@pytest.mark.parametrize("fn, key, _gid, name, ids", LIST_TOOLS)
def test_derived_tool_ambiguous_name(syn, fn, key, _gid, name, ids):
    res = call(fn, player=name)
    _assert_ambiguous(res, ids=ids)
    assert key not in res or not res[key]


@pytest.mark.parametrize("fn, key", [(f, k) for f, k, *_ in LIST_TOOLS[1:]])
def test_derived_tool_unique_full_name_keeps_working(syn, fn, key):
    rows = call(fn, player="Keenan Allen")[key]
    assert rows and {r["player_id"] for r in rows} == {KA}


@pytest.mark.parametrize("name", ["Ja'Marr Chase", "Jamarr Chase"])
def test_drop_rate_full_name_finds_short_form_row(syn, name):
    rows = tools.nfl_drop_rate(player=name)["drop_rate"]
    assert [r["player_id"] for r in rows] == [CH]


@pytest.mark.parametrize("name, stored", [("Ja'Marr Chase", "JaMarr Chase"),
                                          ("Amon-Ra St. Brown", "Amon-Ra St Brown")])
def test_fantasy_rankings_week_ignores_punctuation(syn, name, stored):
    res = tools.nfl_fantasy_rankings(player=name, scope="week")
    assert [r["player"] for r in res["fantasy_rankings"]] == [stored]


def test_fantasy_rankings_draft_apostrophe_name_works(syn):
    res = tools.nfl_fantasy_rankings(player="Ja'Marr Chase", scope="draft")
    assert [r["player"] for r in res["fantasy_rankings"]] == ["Ja'Marr Chase"]


# ── Part 2: integration against the real database ────────────────────────────

@pytest.fixture
def real_db():
    path = get_duckdb_path()
    if not Path(path).exists():
        pytest.skip("No DuckDB database loaded")
    # Same opener as the tools, so this open connection never clashes with theirs.
    with get_db_connection() as con:
        if con.execute("SELECT COUNT(*) FROM plays WHERE season = 2024").fetchone()[0] == 0:
            pytest.skip("no 2024 plays")
        yield con


def _oracle(con, gid, col):
    """Official nflverse weekly player_stats summed over 2024 REG (oracle b)."""
    return con.execute(
        f"SELECT SUM({col}) FROM player_stats WHERE player_id = ? AND season = 2024 "
        "AND season_type = 'REG'", [gid]).fetchone()[0]


ORACLE_CASES = [
    (JA, "passing", "attempts", "attempts"),
    (JA, "passing", "completions", "completions"),
    (JA, "passing", "yards", "passing_yards"),
    (JA, "passing", "touchdowns", "passing_tds"),
    (JA, "passing", "interceptions", "passing_interceptions"),
    (JA, "passing", "sacks", "sacks_suffered"),
    (JA, "rushing", "carries", "carries"),
    (JA, "rushing", "yards", "rushing_yards"),
    (SB, "rushing", "carries", "carries"),
    (SB, "rushing", "yards", "rushing_yards"),
    (SB, "rushing", "touchdowns", "rushing_tds"),
    (CH, "receiving", "targets", "targets"),
    (CH, "receiving", "receptions", "receptions"),
    (CH, "receiving", "yards", "receiving_yards"),
    (CH, "receiving", "touchdowns", "receiving_tds"),
]


@pytest.mark.integration
@pytest.mark.parametrize("gid, group, key, col", ORACLE_CASES,
                         ids=[f"{g}-{grp}-{k}" for g, grp, k, _c in ORACLE_CASES])
def test_player_stats_matches_official_2024_reg(real_db, gid, group, key, col):
    expected = _oracle(real_db, gid, col)
    res = call("nfl_player_stats", player_id=gid, season=2024, season_type="REG")
    assert _entry(res)[group][key] == pytest.approx(float(expected))


@pytest.mark.integration
def test_real_allen_2024_is_ambiguous(real_db):
    # Josh, Brandon, Keenan and Kyle Allen all have 2024 roster rows (checked in rosters).
    on_roster = {r[0] for r in real_db.execute(
        "SELECT gsis_id FROM rosters WHERE season = 2024 AND gsis_id IN (?, ?, ?, ?)",
        sorted(ALLENS)).fetchall()}
    assert on_roster == ALLENS
    res = call("nfl_player_stats", player_name="Allen", season=2024)
    _assert_ambiguous(res, ids=ALLENS, min_n=4)
    assert "seasons" not in res


@pytest.mark.integration
def test_real_josh_allen_2024_resolves_to_buf_qb(real_db):
    # players holds two "Josh Allen" rows; the C (00-0030833) last played in 2016.
    ids = {r[0] for r in real_db.execute(
        "SELECT gsis_id FROM players WHERE display_name = 'Josh Allen'").fetchall()}
    assert ids == {JA, JA_C}
    res = call("nfl_player_stats", player_name="Josh Allen", season=2024, season_type="REG")
    assert res["player_id"] == JA


@pytest.mark.integration
def test_real_lookup_josh_allen_lists_both(real_db):
    res = call("nfl_player_lookup", name="Josh Allen")
    assert {c["gsis_id"] for c in res["candidates"]} == {JA, JA_C}


@pytest.mark.integration
@pytest.mark.parametrize("kw", [dict(player="Travis Kelce"), dict(player_id=TK)])
def test_real_snap_counts_kelce_2024(real_db, kw):
    # Oracle: rows keyed by the PFR id, a different column than the name the tool matches.
    n = real_db.execute("SELECT COUNT(*) FROM snap_counts WHERE pfr_player_id = 'KelcTr00' "
                        "AND season = 2024").fetchone()[0]
    res = call("nfl_snap_counts", season=2024, **kw)
    assert n > 0 and res["count"] == n


@pytest.mark.integration
def test_real_ftn_charting_by_id_matches_id_join(real_db):
    # Oracle (c): direct join on the id columns with the tool's documented filter
    # (scrimmage pass/run plays, offensive perspective). Not circular: it keys on
    # passer/rusher/receiver ids, while today's tool keys on short names.
    n = real_db.execute(
        "SELECT COUNT(*) FROM ftn_charting f JOIN plays p ON f.nflverse_game_id = p.game_id "
        "AND f.nflverse_play_id = p.play_id WHERE p.play_type IN ('pass', 'run') "
        "AND p.season = 2024 AND ? IN (p.passer_player_id, p.rusher_player_id, "
        "p.receiver_player_id)", [JA]).fetchone()[0]
    res = call("nfl_ftn_charting", player_id=JA, season=2024)
    assert n > 0 and res["total_plays"] == n


@pytest.mark.integration
def test_real_drop_rate_full_name_chase_2024(real_db):
    res = tools.nfl_drop_rate(player="Ja'Marr Chase", season=2024)
    assert res["drop_rate"] and {r["player_id"] for r in res["drop_rate"]} == {CH}
