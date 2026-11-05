"""Group D: ingest loader correctness (depth_charts, participation, DELETE handling,
per-season transactions, exit status on soft failure, --fresh scope).

Every test runs against a throwaway DuckDB file in tmp_path with a fake ``nflreadpy``
module installed in sys.modules (same pattern as tests/test_ingest.py). No network.
Expected values are hand counts of the rows written in each test.

Fake frames use the column names found in /tmp/grpE_v3.duckdb:
- depth_charts 2024 shape: season, club_code, week, game_type, depth_team, gsis_id,
  full_name, position, depth_position, formation.
- depth_charts 2025+ shape: dt (VARCHAR ISO timestamp), team, player_name, gsis_id,
  espn_id, pos_grp, pos_abb, pos_slot, pos_rank. No season/week/club_code values, and
  depth_team is NULL for every 2025+ row in the real DB, so it is left out of the fake.
  Real 2025 file dt runs 2025-08-03..2026-02-28; the 2026 file starts 2026-03-14.
- participation: nflverse_game_id ('2025_01_KC_BUF'), old_game_id, play_id (DOUBLE),
  possession_team, offense_personnel. No season column.

Interface contract (beyond the shared brief)
- _ingest_generic_dataset(conn, dataset_def, seasons, fresh=False) and
  _ingest_pbp_season(conn, season) keep their signatures.
- Re-fetching season S of depth_charts or participation replaces S only; how S is keyed
  (dt range, game-id prefix, or a season column the loader adds) is left to the fix, so
  tests identify a season's rows by gsis_id / game-id markers, never by the season column.
- Missing delete-key column on an EXISTING table: the loader raises, or leaves the table
  and _ingest_metadata for that season untouched. A missing TABLE is still a first load.
- A failure after DELETE (in _write_df_to_table or _record_loaded, or the plays INSERT)
  rolls back: the season's old rows and its old _ingest_metadata row_count remain.
  The loader may re-raise or swallow the error.
- `nfl-mcp ingest` exits non-zero when any dataset/season fetch failed, after still
  running the remaining datasets, and the output names the failed dataset id.
  run_ingest_datasets may signal this by raising at the end or by returning a failure
  list the CLI turns into the exit code.
- `--fresh` with --start/--end only drops the requested plays seasons.
"""

import inspect
import re
import sys
import types

import duckdb
import polars as pl
import pytest
from click.testing import CliRunner

from nfl_mcp import ingest as ing
from nfl_mcp.cli import main
from nfl_mcp.registry import REGISTRY, DatasetDef

XF = "group d: "


# ── helpers ────────────────────────────────────────────────────────────────────

def _fake_nfl(monkeypatch, **loaders):
    fake = types.ModuleType("nflreadpy")
    for name, fn in loaders.items():
        setattr(fake, name, fn)
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)


@pytest.fixture
def conn(tmp_path):
    c = duckdb.connect(str(tmp_path / "d.duckdb"))
    ing._ensure_metadata_table(c)
    yield c
    c.close()


def _count(c, sql):
    return c.execute(sql).fetchone()[0]


def _meta(c, dataset_id, season):
    rows = c.execute(
        "SELECT row_count FROM _ingest_metadata WHERE dataset_id = ? AND season = ?",
        [dataset_id, season],
    ).fetchall()
    return [r[0] for r in rows]


def _dc_2024(n=3):
    # 2024 shape: season/week/club_code present. gsis ids '24-*' mark these rows.
    return pl.DataFrame({
        "season": [2024] * n,
        "club_code": ["ATL"] * n,
        "week": list(range(1, n + 1)),
        "game_type": ["REG"] * n,
        "depth_team": ["1"] * n,
        "gsis_id": [f"24-{i}" for i in range(n)],
        "full_name": [f"Player {i}" for i in range(n)],
        "position": ["G"] * n,
        "depth_position": ["RG"] * n,
        "formation": ["Offense"] * n,
    })


def _dc_new(season, n=4):
    # 2025+ shape: no season/week/club_code. dt inside that season's real window.
    dts = (["2025-08-03T10:09:07Z", "2025-10-01T08:00:00Z", "2026-01-15T07:00:00Z",
            "2026-02-28T07:23:12Z"] if season == 2025 else
           ["2026-03-14T07:32:09Z", "2026-09-01T09:00:00Z", "2026-10-31T13:34:51Z",
            "2026-10-31T13:34:51Z"])
    return pl.DataFrame({
        "dt": [dts[i % 4] for i in range(n)],
        "team": ["ARI"] * n,
        "player_name": [f"P{season}-{i}" for i in range(n)],
        "gsis_id": [f"{season % 100}-{i}" for i in range(n)],
        "espn_id": [str(1000 + i) for i in range(n)],
        "pos_grp": ["Base 4-3 D"] * n,
        "pos_abb": ["LDE"] * n,
        "pos_slot": [1] * n,
        "pos_rank": [i + 1 for i in range(n)],
    })


def _part(season, games):
    """games: list of (game_suffix, n_plays). Rows marked by nflverse_game_id prefix."""
    gid, pid = [], []
    for suffix, n in games:
        gid += [f"{season}_{suffix}"] * n
        pid += [float(40 + 20 * i) for i in range(n)]
    return pl.DataFrame({
        "nflverse_game_id": gid,
        "old_game_id": [f"{season}0908{i:02d}" for i in range(len(gid))],
        "play_id": pid,
        "possession_team": ["KC"] * len(gid),
        "offense_personnel": ["1 RB, 1 TE, 3 WR"] * len(gid),
    })


def _loader_from(frames):
    """Loader returning frames[season] (callable or frame) for loader([season])."""
    def loader(seasons, **kwargs):
        f = frames[seasons[0]]
        return f() if callable(f) else f
    return loader


def _refetch(conn, monkeypatch, defn, season, mode):
    """Re-run a season the way production does: --fresh, or as the current season."""
    if mode == "current":
        monkeypatch.setattr(ing, "current_season", lambda: season)
        return ing._ingest_generic_dataset(conn, defn, [season], fresh=False)
    return ing._ingest_generic_dataset(conn, defn, [season], fresh=True)


MODES = pytest.mark.parametrize("mode", ["fresh", "current"])


# ── 1. depth_charts re-fetch does not duplicate ────────────────────────────────

class TestDepthChartsRefetch:
    defn = REGISTRY["depth_charts"]

    @MODES
    def test_depth_charts_2025_shape_loaded_twice_keeps_one_copy(self, conn, monkeypatch, mode):
        _fake_nfl(monkeypatch, load_depth_charts=_loader_from({2025: _dc_new(2025, 4)}))
        ing._ingest_generic_dataset(conn, self.defn, [2025], fresh=False)
        _refetch(conn, monkeypatch, self.defn, 2025, mode)
        # one load = 4 rows; the re-fetch must replace, not append (today: 8)
        assert _count(conn, "SELECT COUNT(*) FROM depth_charts") == 4

    @MODES
    def test_depth_charts_2024_shape_loaded_twice_keeps_one_copy(self, conn, monkeypatch, mode):
        _fake_nfl(monkeypatch, load_depth_charts=_loader_from({2024: _dc_2024(3)}))
        ing._ingest_generic_dataset(conn, self.defn, [2024], fresh=False)
        _refetch(conn, monkeypatch, self.defn, 2024, mode)
        assert _count(conn, "SELECT COUNT(*) FROM depth_charts") == 3  # 3 rows, one copy

    def test_depth_charts_mixed_db_refetch_2025_keeps_2024(self, conn, monkeypatch):
        _fake_nfl(monkeypatch, load_depth_charts=_loader_from(
            {2024: _dc_2024(3), 2025: _dc_new(2025, 4)}))
        ing._ingest_generic_dataset(conn, self.defn, [2024, 2025], fresh=False)
        ing._ingest_generic_dataset(conn, self.defn, [2025], fresh=True)
        # 2024: 3 rows untouched; 2025: 4 rows (one copy) -> 7 total
        assert _count(conn, "SELECT COUNT(*) FROM depth_charts WHERE gsis_id LIKE '24-%'") == 3
        assert _count(conn, "SELECT COUNT(*) FROM depth_charts WHERE gsis_id LIKE '25-%'") == 4

    def test_depth_charts_mixed_db_refetch_2024_keeps_2025(self, conn, monkeypatch):
        _fake_nfl(monkeypatch, load_depth_charts=_loader_from(
            {2024: _dc_2024(3), 2025: _dc_new(2025, 4)}))
        ing._ingest_generic_dataset(conn, self.defn, [2024, 2025], fresh=False)
        ing._ingest_generic_dataset(conn, self.defn, [2024], fresh=True)
        # 2024 re-fetched: 3 rows; 2025 dt-shaped rows must survive: 4
        assert _count(conn, "SELECT COUNT(*) FROM depth_charts WHERE gsis_id LIKE '24-%'") == 3
        assert _count(conn, "SELECT COUNT(*) FROM depth_charts WHERE gsis_id LIKE '25-%'") == 4

    def test_depth_charts_refetch_2026_keeps_2025_dt_rows(self, conn, monkeypatch):
        _fake_nfl(monkeypatch, load_depth_charts=_loader_from(
            {2025: _dc_new(2025, 4), 2026: _dc_new(2026, 2)}))
        ing._ingest_generic_dataset(conn, self.defn, [2025, 2026], fresh=False)
        _refetch(conn, monkeypatch, self.defn, 2026, "current")
        # 2025: 4 rows untouched; 2026: 2 rows, one copy (today 2026 doubles to 4)
        assert _count(conn, "SELECT COUNT(*) FROM depth_charts WHERE gsis_id LIKE '25-%'") == 4
        assert _count(conn, "SELECT COUNT(*) FROM depth_charts WHERE gsis_id LIKE '26-%'") == 2


# ── 2. participation re-fetch does not duplicate ───────────────────────────────

class TestParticipationRefetch:
    defn = REGISTRY["participation"]
    p25 = staticmethod(lambda: _part(2025, [("01_KC_BUF", 2), ("02_DAL_PHI", 1)]))  # 3 rows
    p24 = staticmethod(lambda: _part(2024, [("01_TEN_CHI", 2)]))                    # 2 rows

    @MODES
    def test_participation_2025_loaded_twice_keeps_one_copy(self, conn, monkeypatch, mode):
        _fake_nfl(monkeypatch, load_participation=_loader_from({2025: self.p25}))
        ing._ingest_generic_dataset(conn, self.defn, [2025], fresh=False)
        _refetch(conn, monkeypatch, self.defn, 2025, mode)
        assert _count(conn, "SELECT COUNT(*) FROM participation") == 3  # today: 6

    def test_participation_refetch_2025_with_2024_present_counts_exact(self, conn, monkeypatch):
        _fake_nfl(monkeypatch, load_participation=_loader_from({2024: self.p24, 2025: self.p25}))
        ing._ingest_generic_dataset(conn, self.defn, [2024, 2025], fresh=False)
        ing._ingest_generic_dataset(conn, self.defn, [2025], fresh=True)
        # 2024: 2 + 2025: 3 = 5 (today 8: the 2025 rows are appended again)
        assert _count(conn, "SELECT COUNT(*) FROM participation") == 5
        assert _count(conn, "SELECT COUNT(*) FROM (SELECT DISTINCT nflverse_game_id, play_id "
                            "FROM participation)") == 5

    def test_participation_2024_rows_survive_2025_refetch(self, conn, monkeypatch):
        _fake_nfl(monkeypatch, load_participation=_loader_from({2024: self.p24, 2025: self.p25}))
        ing._ingest_generic_dataset(conn, self.defn, [2024, 2025], fresh=False)
        ing._ingest_generic_dataset(conn, self.defn, [2025], fresh=True)
        # the 2025 prefix delete must not touch '2024_%' rows: 2 rows
        assert _count(conn, "SELECT COUNT(*) FROM participation "
                            "WHERE nflverse_game_id LIKE '2024\\_%' ESCAPE '\\'") == 2


# ── 3. silent DELETE failure is gone ───────────────────────────────────────────

def _missing_key_case(name):
    """(DatasetDef, pre-existing table DDL + rows, loader name, incoming frame)."""
    if name == "generic_season":
        d = DatasetDef(dataset_id="fake_nokey", loader_fn="load_fake_nokey",
                       table_name="fake_nokey", seasonal=True, default=False)
        ddl = ["CREATE TABLE fake_nokey (value BIGINT)",
               "INSERT INTO fake_nokey VALUES (1), (2)"]
        return d, ddl, pl.DataFrame({"season": [2024, 2024], "value": [1, 2]})
    d = REGISTRY["participation"]
    ddl = ["CREATE TABLE participation (play_id DOUBLE, possession_team VARCHAR)",
           "INSERT INTO participation VALUES (40.0, 'KC'), (60.0, 'KC')"]
    return d, ddl, _part(2024, [("01_TEN_CHI", 2)])


class TestDeleteFailureIsLoud:
    def test_generic_loader_source_has_no_bare_except_around_delete(self):
        src = inspect.getsource(ing._ingest_generic_dataset)
        swallowed = [m for m in re.finditer(r"except Exception:\s*\n\s*pass", src)
                     if "DELETE" in src[max(0, m.start() - 300):m.start()]]
        assert swallowed == [], "DELETE is still wrapped in `except Exception: pass`"

    @pytest.mark.parametrize("case", ["generic_season", "participation"])
    def test_missing_delete_key_column_is_not_a_silent_append(self, conn, monkeypatch, case):
        defn, ddl, df = _missing_key_case(case)
        for sql in ddl:
            conn.execute(sql)
        ing._record_loaded(conn, defn.dataset_id, defn.table_name, defn.loader_fn, 2, season=2024)
        _fake_nfl(monkeypatch, **{defn.loader_fn: lambda seasons, **k: df})
        raised = False
        try:
            ing._ingest_generic_dataset(conn, defn, [2024], fresh=True)
        except Exception:
            raised = True
        rows = _count(conn, f"SELECT COUNT(*) FROM {defn.table_name}")
        # today: DELETE fails silently, 2 new rows appended next to the 2 old -> 4
        assert raised or rows == 2, f"silent pass: {rows} rows after re-fetch"

    def test_missing_table_is_still_a_first_load(self, conn, monkeypatch):
        defn = REGISTRY["participation"]
        _fake_nfl(monkeypatch, load_participation=lambda seasons, **k: _part(2025, [("01_KC_BUF", 3)]))
        assert ing._ingest_generic_dataset(conn, defn, [2025], fresh=False) == 3
        assert _count(conn, "SELECT COUNT(*) FROM participation") == 3


# ── 4. one transaction per season ──────────────────────────────────────────────

def _sched(season, n):
    return pl.DataFrame({
        "game_id": [f"{season}_{i:02d}_KC_BUF" for i in range(1, n + 1)],
        "season": [season] * n,
        "game_type": ["REG"] * n,
        "week": list(range(1, n + 1)),
        "away_team": ["KC"] * n,
        "home_team": ["BUF"] * n,
    })


class _FailingInsertConn:
    """Proxy that raises on the plays INSERT; everything else hits the real conn."""

    def __init__(self, real):
        self._real = real

    def execute(self, sql, *args, **kwargs):
        if sql.lstrip().upper().startswith("INSERT INTO PLAYS"):
            raise RuntimeError("simulated crash between DELETE and INSERT")
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _pbp(season, n):
    return pl.DataFrame({
        "game_id": [f"{season}_01_KC_BAL"] * n,
        "play_id": [float(i) for i in range(1, n + 1)],
        "season": [season] * n,
        "week": [1] * n,
        "season_type": ["REG"] * n,
        "posteam": ["KC"] * n,
        "defteam": ["BAL"] * n,
        "play_type": ["pass"] * n,
        "desc": ["x"] * n,
    })


class TestSeasonTransaction:
    defn = REGISTRY["schedules"]

    @pytest.mark.parametrize("fail_in", ["_write_df_to_table", "_record_loaded"])
    def test_failure_after_delete_leaves_old_season(self, conn, monkeypatch, fail_in):
        _fake_nfl(monkeypatch, load_schedules=_loader_from({2024: _sched(2024, 3)}))
        ing._ingest_generic_dataset(conn, self.defn, [2024], fresh=False)   # 3 rows, meta 3
        _fake_nfl(monkeypatch, load_schedules=_loader_from({2024: _sched(2024, 5)}))
        called = []

        def boom(*a, **k):
            called.append(True)
            raise RuntimeError("simulated crash")
        monkeypatch.setattr(ing, fail_in, boom)
        try:
            ing._ingest_generic_dataset(conn, self.defn, [2024], fresh=True)
        except Exception:
            pass
        assert called, f"{fail_in} was not reached; the test did not simulate the failure"
        # nothing half-applied: old 3 rows and old metadata count 3
        assert _count(conn, "SELECT COUNT(*) FROM schedules WHERE season = 2024") == 3
        assert _meta(conn, "schedules", 2024) == [3]

    def test_pbp_insert_failure_keeps_previous_plays(self, conn, monkeypatch):
        _fake_nfl(monkeypatch, load_pbp=lambda seasons: _pbp(2024, 3))
        ing._create_plays_table(conn, _pbp(2024, 3))
        assert ing._ingest_pbp_season(conn, 2024) == 3
        ing._record_loaded(conn, "pbp", "plays", "load_pbp", 3, season=2024)
        _fake_nfl(monkeypatch, load_pbp=lambda seasons: _pbp(2024, 6))
        try:
            ing._ingest_pbp_season(_FailingInsertConn(conn), 2024)
        except Exception:
            pass
        # old 3 plays must survive; today the DELETE already removed them -> 0
        assert _count(conn, "SELECT COUNT(*) FROM plays WHERE season = 2024") == 3
        assert _meta(conn, "pbp", 2024) == [3]


# ── 5. exit status on soft failure ─────────────────────────────────────────────

def _roster(season, n):
    return pl.DataFrame({
        "season": [season] * n, "team": ["KC"] * n, "position": ["QB"] * n,
        "full_name": [f"R{i}" for i in range(n)], "gsis_id": [f"00-{i}" for i in range(n)],
    })


def _boom(*a, **k):
    raise ConnectionError("simulated nflverse outage")


def _run_cli(monkeypatch, tmp_path, args):
    db = tmp_path / "cli.duckdb"
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(db))
    result = CliRunner().invoke(main, ["ingest", *args, "--skip-views"])
    return result, db


def _db_count(db, sql):
    c = duckdb.connect(str(db), read_only=True)
    try:
        return c.execute(sql).fetchone()[0]
    finally:
        c.close()


class TestExitStatus:
    @pytest.mark.parametrize("failing, ok, ok_table", [
        ("schedules", "rosters", "rosters"),
        ("rosters", "schedules", "schedules"),
    ])
    def test_failed_fetch_continues_then_exits_nonzero(self, monkeypatch, tmp_path,
                                                       failing, ok, ok_table):
        good = {"schedules": lambda s, **k: _sched(s[0], 2), "rosters": lambda s, **k: _roster(s[0], 2)}
        _fake_nfl(monkeypatch, **{f"load_{failing}": _boom, f"load_{ok}": good[ok]})
        result, db = _run_cli(monkeypatch, tmp_path, [
            "--dataset", "schedules", "--dataset", "rosters", "--start", "2024", "--end", "2024"])
        # the other dataset still loaded: 2 rows
        assert _db_count(db, f"SELECT COUNT(*) FROM {ok_table}") == 2
        assert result.exit_code != 0, result.output
        assert failing in result.output

    def test_failed_pbp_season_exits_nonzero(self, monkeypatch, tmp_path):
        # schema discovery (2025) works, the 2024 season fetch fails -> soft failure today
        def load_pbp(seasons):
            if seasons[0] == 2024:
                raise ConnectionError("simulated outage")
            return _pbp(seasons[0], 2)
        _fake_nfl(monkeypatch, load_pbp=load_pbp)
        result, db = _run_cli(monkeypatch, tmp_path, ["--dataset", "pbp", "--start", "2024", "--end", "2025"])
        assert _db_count(db, "SELECT COUNT(*) FROM plays WHERE season = 2025") == 2
        assert result.exit_code != 0, result.output
        assert "pbp" in result.output

    def test_all_fetches_succeed_exit_zero(self, monkeypatch, tmp_path):
        _fake_nfl(monkeypatch, load_schedules=lambda s, **k: _sched(s[0], 2),
                  load_rosters=lambda s, **k: _roster(s[0], 3))
        result, db = _run_cli(monkeypatch, tmp_path, [
            "--dataset", "schedules", "--dataset", "rosters", "--start", "2024", "--end", "2024"])
        assert result.exit_code == 0, result.output
        assert _db_count(db, "SELECT COUNT(*) FROM schedules") == 2
        assert _db_count(db, "SELECT COUNT(*) FROM rosters") == 3


# ── 6. --fresh scope ───────────────────────────────────────────────────────────

class TestFreshScope:
    @staticmethod
    def _seed(monkeypatch, tmp_path):
        sizes = {2024: 3, 2025: 2}
        _fake_nfl(monkeypatch, load_pbp=lambda seasons: _pbp(seasons[0], sizes[seasons[0]]))
        result, db = _run_cli(monkeypatch, tmp_path, ["--dataset", "pbp", "--start", "2024", "--end", "2025"])
        assert result.exit_code == 0, result.output
        return sizes, db

    @pytest.mark.parametrize("target, other, other_n", [(2025, 2024, 3), (2024, 2025, 2)])
    def test_fresh_one_season_keeps_other_season_plays(self, monkeypatch, tmp_path,
                                                       target, other, other_n):
        sizes, db = self._seed(monkeypatch, tmp_path)
        sizes[target] = 4                       # the re-fetch returns 4 plays
        result, _ = _run_cli(monkeypatch, tmp_path, [
            "--dataset", "pbp", "--fresh", "--start", str(target), "--end", str(target)])
        assert result.exit_code == 0, result.output
        assert _db_count(db, f"SELECT COUNT(*) FROM plays WHERE season = {other}") == other_n
        assert _db_count(db, f"SELECT COUNT(*) FROM plays WHERE season = {target}") == 4

    def test_fresh_one_season_metadata_matches_plays(self, monkeypatch, tmp_path):
        sizes, db = self._seed(monkeypatch, tmp_path)
        sizes[2025] = 4
        _run_cli(monkeypatch, tmp_path, ["--dataset", "pbp", "--fresh", "--start", "2025", "--end", "2025"])
        c = duckdb.connect(str(db), read_only=True)
        try:
            # metadata says 2024 = 3; the plays table must still hold those 3 rows
            assert _meta(c, "pbp", 2024) == [3]
            assert _count(c, "SELECT COUNT(*) FROM plays WHERE season = 2024") == 3
        finally:
            c.close()

    def test_fresh_one_season_generic_keeps_other_season(self, monkeypatch, tmp_path):
        sizes = {2024: 2, 2025: 3}
        _fake_nfl(monkeypatch, load_schedules=lambda s, **k: _sched(s[0], sizes[s[0]]))
        _run_cli(monkeypatch, tmp_path, ["--dataset", "schedules", "--start", "2024", "--end", "2025"])
        sizes[2025] = 1
        result, db = _run_cli(monkeypatch, tmp_path, [
            "--dataset", "schedules", "--fresh", "--start", "2025", "--end", "2025"])
        assert result.exit_code == 0, result.output
        assert _db_count(db, "SELECT COUNT(*) FROM schedules WHERE season = 2024") == 2
        assert _db_count(db, "SELECT COUNT(*) FROM schedules WHERE season = 2025") == 1


# ── 7. behaviour to keep ───────────────────────────────────────────────────────

class TestKeepIdempotent:
    @pytest.mark.parametrize("dataset_id, frame, n", [
        ("schedules", lambda: _sched(2024, 3), 3),
        ("rosters", lambda: _roster(2024, 4), 4),
    ])
    @MODES
    def test_season_column_table_reload_is_idempotent(self, conn, monkeypatch, dataset_id, frame, n, mode):
        defn = REGISTRY[dataset_id]
        _fake_nfl(monkeypatch, **{defn.loader_fn: lambda s, **k: frame()})
        ing._ingest_generic_dataset(conn, defn, [2024], fresh=False)
        _refetch(conn, monkeypatch, defn, 2024, mode)
        assert _count(conn, f"SELECT COUNT(*) FROM {defn.table_name}") == n

    def test_metadata_upserts_on_reload(self, conn, monkeypatch):
        defn = REGISTRY["schedules"]
        size = {"n": 3}
        _fake_nfl(monkeypatch, load_schedules=lambda s, **k: _sched(s[0], size["n"]))
        ing._ingest_generic_dataset(conn, defn, [2024], fresh=False)
        size["n"] = 5
        ing._ingest_generic_dataset(conn, defn, [2024], fresh=True)
        assert _meta(conn, "schedules", 2024) == [5]   # one row, latest count

    def test_metadata_seasons_stay_separate(self, conn, monkeypatch):
        defn = REGISTRY["rosters"]
        _fake_nfl(monkeypatch, load_rosters=lambda s, **k: _roster(s[0], 2 if s[0] == 2024 else 3))
        ing._ingest_generic_dataset(conn, defn, [2024, 2025], fresh=False)
        ing._ingest_generic_dataset(conn, defn, [2024, 2025], fresh=True)
        assert _meta(conn, "rosters", 2024) == [2]
        assert _meta(conn, "rosters", 2025) == [3]
