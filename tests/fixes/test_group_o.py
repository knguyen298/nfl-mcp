"""Group O: seasons nflreadpy does not publish yet are skipped, not failed.

Production log, 2026-11-05: the weekly ingest ended with
    FAILED participation 2026: Season must be between 2016 and 2025
so `nfl-mcp ingest` exited non-zero and update_db.sh never swapped the database.

Expected values come from reading the nflreadpy 0.1.5 validators, not from nfl_mcp:
- utils_date.get_current_season(): the year from the Thursday after Labor Day, else the
  year before. get_current_season(roster=True): the year from March 15, else the year before.
- load_participation: season <= get_current_season(roster=True) - 1.
- load_rosters, load_depth_charts: season <= get_current_season(roster=True).
- load_pbp, load_injuries, load_snap_counts, load_rosters_weekly, load_officials,
  load_nextgen_stats, load_pfr_advstats, load_ftn_charting, load_ff_opportunity:
  season <= get_current_season().
- load_player_stats, load_team_stats: no validator. Their seasons=True range ends at
  get_current_season(), and a missing file makes the downloader raise ConnectionError.
- load_schedules: no validator. It filters one full file, so a season with no games is empty.

Dates used (Labor Day 2026 is Sep 7, so the season starts Thu Sep 10, 2026; Labor Day 2027
is Sep 6, so Thu Sep 9, 2027):
  2026-11-05  in season     season 2026, roster 2026, participation 2025, ours 2026
  2027-03-05  early March   season 2026, roster 2026, participation 2025, ours 2027
  2027-04-01  off-season    season 2026, roster 2027, participation 2026, ours 2027

The real nflreadpy loaders run with their downloader replaced by a stub (no network), so
their own validators decide what a season fetch does. The date is frozen for both sides:
nflreadpy.utils_date.date for nflreadpy, nfl_mcp.ingest.current_season for nfl_mcp.
Rows are read back through nfl_mcp.database.get_db_connection.
"""

import datetime
import importlib
import re

import nflreadpy as real_nfl
import polars as pl
import pytest
from click.testing import CliRunner

from nfl_mcp import ingest as ing
from nfl_mcp.cli import _season_window_label, main
from nfl_mcp.database import get_db_connection
from nfl_mcp.registry import NEWEST_SEASON_RULES, REGISTRY
from nfl_mcp.seasons import all_seasons, current_season

_UTILS_DATE = importlib.import_module("nflreadpy.utils_date")

IN_SEASON = datetime.date(2026, 11, 5)
EARLY_MARCH = datetime.date(2027, 3, 5)
OFF_SEASON = datetime.date(2027, 4, 1)

# Grouped by the validator each nflreadpy loader runs (see the module docstring).
SEASON_BOUND = [
    "pbp", "injuries", "snap_counts", "rosters_weekly", "officials",
    "nextgen_stats_passing", "nextgen_stats_receiving", "nextgen_stats_rushing",
    "pfr_advstats_pass", "pfr_advstats_rush", "pfr_advstats_rec", "pfr_advstats_def",
    "ftn_charting", "ff_opportunity",
]
NO_VALIDATOR_SEASON_RANGE = ["player_stats", "team_stats_raw"]
ROSTER_BOUND = ["rosters", "depth_charts"]
PRIOR_ROSTER_BOUND = ["participation"]
UNBOUNDED = ["schedules", "espn_records"]

# Newest season nflreadpy accepts on each date, per group, read off the validators.
NEWEST = {
    "season": {IN_SEASON: 2026, EARLY_MARCH: 2026, OFF_SEASON: 2026,
               datetime.date(2027, 9, 8): 2026, datetime.date(2027, 9, 9): 2027},
    "roster": {IN_SEASON: 2026, EARLY_MARCH: 2026, OFF_SEASON: 2027,
               datetime.date(2027, 3, 14): 2026, datetime.date(2027, 3, 15): 2027},
    "prior_roster": {IN_SEASON: 2025, EARLY_MARCH: 2025, OFF_SEASON: 2026,
                     datetime.date(2027, 3, 14): 2025, datetime.date(2027, 3, 15): 2026},
}
GROUPS = {
    "season": SEASON_BOUND + NO_VALIDATOR_SEASON_RANGE,
    "roster": ROSTER_BOUND,
    "prior_roster": PRIOR_ROSTER_BOUND,
}
CASES = [(ds, day, newest) for group, days in NEWEST.items()
         for ds in GROUPS[group] for day, newest in days.items()]


# ── helpers ────────────────────────────────────────────────────────────────────

def _freeze(monkeypatch, day):
    """Make nflreadpy and nfl_mcp both see `day` as today."""
    class _Frozen(datetime.date):
        @classmethod
        def today(cls):
            return cls(day.year, day.month, day.day)
    monkeypatch.setattr(_UTILS_DATE, "date", _Frozen)
    monkeypatch.setattr(ing, "current_season", lambda: current_season(day))
    monkeypatch.setattr(ing, "all_seasons", lambda: all_seasons(day))


def _part(season, n=2):
    return pl.DataFrame({
        "nflverse_game_id": [f"{season}_01_KC_BUF"] * n,
        "play_id": [float(i) for i in range(1, n + 1)],
        "possession_team": ["KC"] * n,
    })


def _pbp(season, n=2):
    return pl.DataFrame({
        "game_id": [f"{season}_01_KC_BAL"] * n,
        "play_id": [float(i) for i in range(1, n + 1)],
        "season": [season] * n, "week": [1] * n, "season_type": ["REG"] * n,
        "posteam": ["KC"] * n, "defteam": ["BAL"] * n, "play_type": ["pass"] * n,
        "desc": ["x"] * n,
    })


class _StubDownloader:
    """Stands in for nflreadpy's downloader. Records every path; `fail` seasons raise."""

    def __init__(self, fail=()):
        self.paths: list[str] = []
        self.fail = set(fail)

    def download(self, repository, path, **kwargs):
        self.paths.append(path)
        m = re.search(r"_(\d{4})$", path)
        season = int(m.group(1)) if m else None
        if season in self.fail:
            raise ConnectionError(f"Failed to download {path}: 503 Server Error")
        if path.startswith("pbp_participation/"):
            return _part(season)
        if path.startswith("pbp/"):
            return _pbp(season)
        # Whole-history files (officials, nextgen_stats, schedules) cover 2000-2026.
        seasons = [season] if season is not None else list(range(2000, 2027))
        return pl.DataFrame({"season": seasons, "week": [1] * len(seasons),
                             "value": [1] * len(seasons)})

    def seasons_fetched(self, prefix):
        return sorted(int(p.rsplit("_", 1)[1]) for p in self.paths if p.startswith(prefix))


def _stub_downloads(monkeypatch, fail=()):
    """Point every registry loader's module at one stub downloader (real validators stay)."""
    stub = _StubDownloader(fail)
    for d in REGISTRY.values():
        if d.source == "nflreadpy":
            module = importlib.import_module(getattr(real_nfl, d.loader_fn).__module__)
            monkeypatch.setattr(module, "get_downloader", lambda: stub)
    return stub


def _rows(db, sql, monkeypatch):
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(db))
    with get_db_connection() as c:
        return c.execute(sql).fetchall()


def _cli(monkeypatch, db, args):
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(db))
    return CliRunner().invoke(main, ["ingest", *args, "--skip-views"])


# ── 1. the registry bound matches each nflreadpy validator ─────────────────────

def test_every_seasonal_dataset_is_classified():
    seasonal = {d.dataset_id for d in REGISTRY.values() if d.seasonal}
    listed = SEASON_BOUND + NO_VALIDATOR_SEASON_RANGE + ROSTER_BOUND + PRIOR_ROSTER_BOUND + UNBOUNDED
    assert sorted(listed) == sorted(seasonal)


@pytest.mark.parametrize("dataset_id", UNBOUNDED)
def test_unbounded_datasets_have_no_newest_season(dataset_id, monkeypatch):
    _freeze(monkeypatch, OFF_SEASON)
    assert ing._newest_season(REGISTRY[dataset_id]) is None


CASE_IDS = [f"{ds}-{day}" for ds, day, _ in CASES]


@pytest.mark.parametrize("dataset_id,day,newest", CASES, ids=CASE_IDS)
def test_newest_season_matches_nflreadpy_dates(dataset_id, day, newest, monkeypatch):
    _freeze(monkeypatch, day)
    assert ing._newest_season(REGISTRY[dataset_id]) == newest


@pytest.mark.parametrize("dataset_id,day,newest",
                         [c for c in CASES if c[0] not in NO_VALIDATOR_SEASON_RANGE],
                         ids=[i for c, i in zip(CASES, CASE_IDS) if c[0] not in NO_VALIDATOR_SEASON_RANGE])
def test_real_validator_accepts_newest_and_rejects_the_next(dataset_id, day, newest, monkeypatch):
    _freeze(monkeypatch, day)
    _stub_downloads(monkeypatch)
    defn = REGISTRY[dataset_id]
    loader = getattr(real_nfl, defn.loader_fn)
    loader([newest], **defn.extra_params)
    with pytest.raises(ValueError, match=f"Season must be between .* and {newest}$"):
        loader([newest + 1], **defn.extra_params)


@pytest.mark.parametrize("dataset_id", NO_VALIDATOR_SEASON_RANGE)
def test_stats_loaders_have_no_validator_and_would_fetch(dataset_id, monkeypatch):
    # Nothing in load_player_stats / load_team_stats stops a future season, so without the
    # registry bound the off-season run would request a file nflverse has not made yet.
    _freeze(monkeypatch, OFF_SEASON)
    stub = _stub_downloads(monkeypatch)
    defn = REGISTRY[dataset_id]
    getattr(real_nfl, defn.loader_fn)([2027], **defn.extra_params)
    assert stub.paths and stub.paths[-1].endswith("_2027")


# ── 2. the production failure: in season, participation ────────────────────────

def test_participation_in_progress_is_skipped_and_exit_is_zero(monkeypatch, tmp_path):
    _freeze(monkeypatch, IN_SEASON)
    stub = _stub_downloads(monkeypatch)
    with pytest.raises(ValueError, match="between 2016 and 2025"):
        real_nfl.load_participation([2026])         # the validator from the log
    stub.paths.clear()

    db = tmp_path / "o.duckdb"
    result = _cli(monkeypatch, db, ["--dataset", "participation"])   # bulk, like update_db.sh

    assert result.exit_code == 0, result.output
    assert "participation 2026: not published yet, skipping" in result.output
    assert "FAILED" not in result.output
    assert stub.seasons_fetched("pbp_participation/") == list(range(2016, 2026))
    assert _rows(db, "SELECT COUNT(*) FROM participation "
                     "WHERE starts_with(nflverse_game_id, '2025_')", monkeypatch) == [(2,)]
    assert _rows(db, "SELECT MAX(season) FROM _ingest_metadata "
                     "WHERE dataset_id = 'participation'", monkeypatch) == [(2025,)]


def test_run_ingest_returns_no_failures_for_the_skipped_season(monkeypatch, tmp_path):
    _freeze(monkeypatch, IN_SEASON)
    stub = _stub_downloads(monkeypatch)
    failures = ing.run_ingest_datasets(["participation"], start=2025, end=2026,
                                       skip_views=True, db_path=str(tmp_path / "o.duckdb")).failures
    assert failures == []
    assert stub.seasons_fetched("pbp_participation/") == [2025]


def test_refresh_run_fetches_current_season_but_not_unpublished_participation(monkeypatch, tmp_path):
    # Second weekly run: participation 2016-2025 is loaded, 2026 is the refresh season.
    # injuries 2026 is re-fetched (in progress), participation 2026 is still skipped.
    _freeze(monkeypatch, IN_SEASON)
    stub = _stub_downloads(monkeypatch)
    db = tmp_path / "o.duckdb"
    args = ["--dataset", "participation", "--dataset", "injuries", "--start", "2025", "--end", "2026"]
    assert _cli(monkeypatch, db, args).exit_code == 0
    stub.paths.clear()

    result = _cli(monkeypatch, db, args)
    assert result.exit_code == 0, result.output
    assert stub.seasons_fetched("pbp_participation/") == []
    assert stub.seasons_fetched("injuries/") == [2026]
    assert "participation 2026: not published yet, skipping" in result.output


# ── 3. off-season: loaders bounded by get_current_season() ─────────────────────

# On each date: datasets whose new season (2027) nflreadpy rejects, and those it accepts.
OFF_SEASON_CASES = {
    OFF_SEASON: (SEASON_BOUND + NO_VALIDATOR_SEASON_RANGE + PRIOR_ROSTER_BOUND, ROSTER_BOUND),
    EARLY_MARCH: (SEASON_BOUND + NO_VALIDATOR_SEASON_RANGE + PRIOR_ROSTER_BOUND + ROSTER_BOUND, []),
}


@pytest.mark.parametrize("day", list(OFF_SEASON_CASES), ids=str)
def test_new_league_year_skips_every_unpublished_season(day, monkeypatch, tmp_path, capsys):
    rejected, accepted = OFF_SEASON_CASES[day]
    _freeze(monkeypatch, day)
    stub = _stub_downloads(monkeypatch)
    ids = [d.dataset_id for d in REGISTRY.values() if d.default and d.seasonal]
    db = tmp_path / "o.duckdb"

    failures = ing.run_ingest_datasets(ids, start=2025, end=2027, skip_views=True, db_path=str(db)).failures

    out = capsys.readouterr().out
    assert failures == []
    for dataset_id in rejected:
        assert f"{dataset_id} 2027: not published yet, skipping" in out
    for dataset_id in accepted:
        assert f"{dataset_id} 2027: not published yet" not in out
    # Only the roster-year loaders asked for a 2027 file.
    assert sorted(p for p in stub.paths if p.endswith("_2027")) == sorted(
        {"rosters": "rosters/roster_2027", "depth_charts": "depth_charts/depth_charts_2027"}[ds]
        for ds in accepted)
    # The merged ingest log also records an attempt that found no rows (status 'empty'),
    # so only the rows that hold data count as loaded.
    loaded = _rows(db, "SELECT DISTINCT dataset_id FROM _ingest_metadata WHERE season = 2027 "
                       "AND status IN ('loaded', 'partial')", monkeypatch)
    assert sorted(r[0] for r in loaded) == sorted(accepted)


@pytest.mark.parametrize("day,loaded", [(OFF_SEASON, [(2025,), (2026,)]), (EARLY_MARCH, [(2025,)])],
                         ids=["2027-04-01", "2027-03-05"])
def test_participation_of_the_finished_season_waits_for_march_15(day, loaded, monkeypatch, tmp_path):
    _freeze(monkeypatch, day)
    _stub_downloads(monkeypatch)
    db = tmp_path / "o.duckdb"
    assert ing.run_ingest_datasets(["participation"], start=2025, end=2027, skip_views=True,
                                   db_path=str(db)).failures == []
    assert _rows(db, "SELECT season FROM _ingest_metadata WHERE dataset_id = 'participation' "
                     "ORDER BY season", monkeypatch) == loaded


def test_off_season_pbp_discovers_schema_from_the_last_season(monkeypatch, tmp_path, capsys):
    _freeze(monkeypatch, OFF_SEASON)
    stub = _stub_downloads(monkeypatch)
    db = tmp_path / "o.duckdb"
    assert ing.run_ingest_datasets(["pbp"], start=2025, end=2027, skip_views=True,
                                   db_path=str(db)).failures == []
    assert "pbp 2027: not published yet, skipping" in capsys.readouterr().out
    # schema from 2026 (never 2027), then 2025 and 2026 loaded
    assert stub.seasons_fetched("pbp/") == [2025, 2026, 2026]
    assert _rows(db, "SELECT season, COUNT(*) FROM plays GROUP BY season ORDER BY season",
                 monkeypatch) == [(2025, 2), (2026, 2)]


def test_off_season_pbp_range_with_nothing_published_is_not_a_failure(monkeypatch, tmp_path, capsys):
    _freeze(monkeypatch, OFF_SEASON)
    stub = _stub_downloads(monkeypatch)
    failures = ing.run_ingest_datasets(["pbp"], start=2027, end=2027, skip_views=True,
                                       db_path=str(tmp_path / "o.duckdb")).failures
    assert failures == []
    assert stub.paths == []
    assert "No published play-by-play season in range, skipping" in capsys.readouterr().out


def test_upgrade_reload_window_ends_at_the_last_published_pbp_season(monkeypatch):
    # migrate._run_raw and _reload_pbp clip all_seasons() through _clip_to_window.
    _freeze(monkeypatch, OFF_SEASON)
    assert ing._clip_to_window(ing.all_seasons(), REGISTRY["pbp"])[-1] == 2026
    assert ing._clip_to_window(ing.all_seasons(), REGISTRY["rosters"])[-1] == 2027


# ── 4. real failures still fail ───────────────────────────────────────────────

def test_network_error_on_a_published_season_still_exits_nonzero(monkeypatch, tmp_path):
    _freeze(monkeypatch, IN_SEASON)
    _stub_downloads(monkeypatch, fail={2025})
    result = _cli(monkeypatch, tmp_path / "o.duckdb",
                  ["--dataset", "participation", "--start", "2024", "--end", "2026"])
    assert result.exit_code != 0, result.output
    assert "FAILED participation 2025: Failed to download" in result.output
    assert "FAILED participation 2026" not in result.output
    assert "participation 2026: not published yet, skipping" in result.output


def test_write_error_on_a_published_season_still_exits_nonzero(monkeypatch, tmp_path):
    _freeze(monkeypatch, IN_SEASON)
    _stub_downloads(monkeypatch)

    def disk_full(*a, **k):
        raise OSError("No space left on device")
    monkeypatch.setattr(ing, "_write_df_to_table", disk_full)
    result = _cli(monkeypatch, tmp_path / "o.duckdb",
                  ["--dataset", "participation", "--start", "2025", "--end", "2026"])
    assert result.exit_code != 0, result.output
    assert "FAILED participation 2025: No space left on device" in result.output
    assert "FAILED participation 2026" not in result.output


# ── 5. ingest --list ───────────────────────────────────────────────────────────

def test_list_shows_participation_ends_at_the_last_completed_season():
    out = CliRunner().invoke(main, ["ingest", "--list"]).output
    lines = out.splitlines()
    after = {lines[i].split()[0]: lines[i + 1] for i in range(len(lines) - 1)
             if lines[i].strip() and "→ seasons" in lines[i + 1]}
    assert after["participation"].strip() == "→ seasons 2016–last completed season"
    assert after["pbp"].strip() == "→ seasons 1999–present"
    assert after["espn_records"].strip() == "→ seasons 1970–1998"


@pytest.mark.parametrize("low,high,rule,expected", [
    (2016, None, "roster-1", "2016–last completed season"),
    (2016, 2018, "roster-1", "2016–2018"),
    (1999, None, "season", "1999–present"),
    (1920, None, "roster", "1920–present"),
])
def test_season_window_label_with_a_newest_season_rule(low, high, rule, expected):
    assert rule in NEWEST_SEASON_RULES
    assert _season_window_label(low, high, rule) == expected
