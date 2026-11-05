"""Group M: schema version, detection and upgrade of an older database.

Spec: /tmp/nfl-audit/GROUP_M_SPEC.md (binding contract). The tests below define the fix.

Contract corrections (the tests follow these, not the spec text)
1. derived_tables is needed only for a structural reason: a derived table is missing, or
   lacks a column REQUIRED_COLUMNS lists for it. "version < 1" on its own does not make the
   step needed. Why: (a) the spec's Ingest section says a fresh build ends at version 1
   through stamp_if_current, which "stamps when every step is not needed"; with the
   version rule derived_tables is needed on every unstamped database, so nothing could
   ever be stamped that way. (b) upgrade(fetch=False) on /tmp/grpE_v3.duckdb leaves
   depth_charts_season pending, so the version stays 0; with the version rule the second
   run rebuilds the derived tables again, but it must be a no-op. Both real old shapes
   are caught structurally (measured): grpE_v3 lacks team_offense_stats.season_type and
   .pass_attempts, team_defense_stats.season_type and .def_touchdowns,
   situational_stats.season_type and .conversion_rule; the v0.5.3 database lacks
   pass_attempts, def_touchdowns and conversion_rule. rebuild_derived_tables rebuilds every
   derived table at once, so the fantasy tables (whose values also differ between grpE_v3
   and grpF_work: 689 player_role_trend rows, for example) are rebuilt as well.
   Consequence: a version-0 database with nothing structurally wrong reports every step
   "not needed", upgrade_needed True (version < target), and upgrade() stamps it without
   running a step.
2. stamp_if_current and the final stamp of upgrade() follow correction 1: they stamp when
   no step is needed (the version itself is not a reason).
3. A database without a depth_charts table is not held back by depth_charts_season (the
   step definition says "needed when depth_charts exists and ..."). The Ingest section's
   "unless depth_charts ... was not loaded" is read as "unless depth_charts still holds
   NULL-season rows".
4. The v0.5.3 database (~/.nfl-mcp/nflread.duckdb) has no _schema_version table, but its
   _ingest_metadata holds a row dataset_id='derived_shape', row_count 4, loader_fn
   'nfl_mcp.ingest.rebuild_derived_tables' (its own shape counter). That row is not a
   schema version: check() reports version 0 and newer False there, so upgrade does not
   refuse it as "newer".

Interface contract (shared contract plus the choices made here)
- nfl_mcp.migrate: SCHEMA_VERSION == 1; check(conn); upgrade(conn, *, fetch=True);
  stamp_if_current(conn). Status objects expose the attribute names in the spec
  (version, target, steps, upgrade_needed, newer; Step.name/needed/needs_network/detail;
  result.before/after/ran/failed/pending/stamped).
- Derived tables: the REQUIRED_COLUMNS tables that rebuild_derived_tables writes
  (DERIVED below). Raw tables: every other REQUIRED_COLUMNS table except _ingest_metadata.
  Table to dataset: the REGISTRY entry whose table_name matches (plays -> pbp).
- Step details: derived_tables names every missing table and column; depth_charts_season
  contains the NULL-season row count (digits, thousands separators allowed; a table with
  no season column counts every row); raw_columns contains "table.column" for a missing
  column and the table name for a missing table.
- Network steps fetch through the module named nflreadpy (the ingest loaders); tests put
  a fake module in sys.modules and assert its calls. raw_columns re-ingests with
  fresh=True the seasons in _ingest_metadata (exact set); a missing seasonal table loads
  the dataset's default seasons, i.e. what `nfl-mcp ingest --dataset X` loads with no
  --start/--end: ingest._clip_to_window(ingest.ALL_SEASONS, defn). A missing static table
  loads once with no seasons.
- failed entries start with the step name ("depth_charts_season: ..."). stamped is True
  only when this run wrote a _schema_version row; a run on a current database returns
  ran == [], failed == [], pending == [], stamped False and writes no row.
- CLI `nfl-mcp upgrade`: exit 0 when the database is current at the end (stamped now or
  already), 3 when only network steps are pending because of --no-fetch, 1 on a failed
  step, a newer database, an error or a missing file. --check prints one line per step
  containing the step name and "ok", "NEEDED" or "NEEDED (network)", and a line with the
  word "version" and both numbers. The read-only refusal message contains the path, the
  words "read-only" and "update_db.sh".
- serve: a warning is a logging record at WARNING or above, or an output line containing
  "warning". An old database gives exactly one warning line containing "nfl-mcp upgrade"
  and the needed step names; a newer one gives a warning containing "newer".
- doctor: one line containing "schema" (any case), the version, the target and the
  names of the needed steps.
- validate: a result named schema_version; its detail has both numbers and
  "nfl-mcp upgrade".
- update_db.sh: the upgrade line has `nfl-mcp upgrade` and `--db "$TARGET"` or
  `NFL_MCP_DB_PATH="$TARGET"`, no --check, no --no-fetch. Its guard has an echo and an
  `exit N` (N >= 1) before the validate line. The failure log line contains "upgrade"
  and "ERROR" or "fail".
- entrypoint.sh: the background start is the one non-crontab line that runs
  /update_db.sh and ends with "&", appending to /var/log/nflmcp/nfl_update.log.
- Scripts are read as text, or run with nfl-mcp stubbed on PATH and the container paths
  rewritten into tmp_path (pattern of tests/test_dockerfile_dev.py).

Measured values (read-only connections to the originals; copies for anything else)
- grpE_v3: no _schema_version table. depth_charts has a season column, 1,317,942 rows,
  1,280,630 with season NULL, 37,312 with season 2024; _ingest_metadata seasons for
  depth_charts: 2024, 2025, 2026. Every raw REQUIRED_COLUMNS table and column is present,
  so raw_columns is not needed there (no columns listed).
- v0.5.3: no _schema_version table, 1,280,630 NULL-season depth_charts rows, team tables
  have season_type but lack the three columns in correction 1; raw_columns not needed.
- grpF_work: every REQUIRED_COLUMNS table and column present; same 1,280,630 NULL rows.
- rebuild_derived_tables on a copy of grpE_v3 reproduces grpF_work's derived tables
  exactly (EXCEPT ALL both ways is empty for all 11 derived tables). Oracle rows from
  grpF_work: team_offense_stats KC 2024 REG total_plays 1067, total_yards 5602.0,
  rush_plays 450, pass_plays 598, pass_attempts 600; team_defense_stats PHI 2024 POST
  plays_against 255, yards_allowed 1330.0, sacks 16, interceptions 6, turnovers_forced
  13, def_touchdowns 1; situational_stats BUF 2025 REG plays per situation (below);
  team_records KC 2025 17 games, 6-11-0.
"""

import contextlib
import importlib
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import duckdb
import polars as pl
import pytest
from click.testing import CliRunner

from nfl_mcp import ingest as ing
from nfl_mcp.cli import main
from nfl_mcp.database import _connection_config
from nfl_mcp.registry import REGISTRY
from nfl_mcp.required_columns import REQUIRED_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
V0_DB = Path("/tmp/grpE_v3.duckdb")
V053_DB = Path.home() / ".nfl-mcp" / "nflread.duckdb"
F_DB = Path("/tmp/grpF_work.duckdb")

STEPS = ["derived_tables", "depth_charts_season", "raw_columns"]
DERIVED = ["team_offense_stats", "team_defense_stats", "situational_stats", "team_records",
           "player_td_luck", "player_role_trend", "player_separation_opportunity",
           "player_drop_rate", "player_contract_value", "injury_return_curve"]
RAW = [t for t in REQUIRED_COLUMNS if t not in DERIVED and t != "_ingest_metadata"]
EXISTING_CHECKS = {"required_tables", "required_columns", "row_count_floor",
                   "duplicate_plays", "newest_week", "null_keys"}
TEAM_TABLES = ["team_offense_stats", "team_defense_stats", "situational_stats", "team_records"]

# Measured on grpE_v3 (see the docstring): the derived columns a v0.4.0 build lacks.
V0_GAPS = {
    "team_offense_stats": ["season_type", "pass_attempts"],
    "team_defense_stats": ["season_type", "def_touchdowns"],
    "situational_stats": ["season_type", "conversion_rule"],
}
V0_NULL_DEPTH = 1_280_630


def _migrate():
    return importlib.import_module("nfl_mcp.migrate")


# ── synthetic database ─────────────────────────────────────────────────────────
#
# Source tables with the columns REQUIRED_COLUMNS lists plus every column the derived SQL in
# ingest.py reads (names taken from the real grpE_v3 tables, types too: ff_opportunity
# season is VARCHAR there). rebuild_derived_tables on one row of each builds every derived
# table with all its required columns (checked by an unmarked test below).
# Types: V VARCHAR, I INTEGER, D DOUBLE, B BOOLEAN, L BIGINT.
SCHEMA = {
    "plays": "play_id:D game_id:V home_team:V away_team:V season_type:V week:I posteam:V "
             "defteam:V yardline_100:D half_seconds_remaining:D sp:D qtr:D down:D time:V "
             "ydstogo:D desc:V play_type:V yards_gained:D shotgun:D no_huddle:D qb_dropback:D "
             "qb_scramble:D air_yards:D yards_after_catch:D td_team:V epa:D interception:D "
             "fumble_lost:D rush_attempt:D pass_attempt:D sack:D touchdown:D pass_touchdown:D "
             "rush_touchdown:D two_point_attempt:D fumble:D complete_pass:D "
             "passer_player_name:V passing_yards:D receiver_player_id:V receiver_player_name:V "
             "receiving_yards:D rusher_player_name:V rushing_yards:D fumbled_1_team:V season:I "
             "cpoe:D away_score:I home_score:I pass:D first_down:D special:D name:V fantasy:V "
             # group I: nfl_search_plays filters, sort keys and the role id columns
             "goal_to_go:D score_differential:D wp:D wpa:D passer_player_id:V "
             "rusher_player_id:V kicker_player_id:V punter_player_id:V "
             "kickoff_returner_player_id:V punt_returner_player_id:V sack_player_id:V "
             "half_sack_1_player_id:V half_sack_2_player_id:V solo_tackle_1_player_id:V "
             "solo_tackle_2_player_id:V assist_tackle_1_player_id:V assist_tackle_2_player_id:V "
             "assist_tackle_3_player_id:V assist_tackle_4_player_id:V "
             "tackle_with_assist_1_player_id:V tackle_with_assist_2_player_id:V "
             "interception_player_id:V pass_defense_1_player_id:V pass_defense_2_player_id:V "
             "forced_fumble_player_1_player_id:V forced_fumble_player_2_player_id:V "
             "qb_hit_1_player_id:V qb_hit_2_player_id:V tackle_for_loss_1_player_id:V "
             "tackle_for_loss_2_player_id:V",
    "schedules": "game_id:V season:I game_type:V week:I gameday:V gametime:V away_team:V "
                 "away_score:I home_team:V home_score:I result:I overtime:I spread_line:D "
                 "total_line:D div_game:I roof:V surface:V temp:I wind:I away_coach:V "
                 "home_coach:V referee:V stadium:V",
    "rosters": "season:I team:V position:V depth_chart_position:V jersey_number:I status:V "
               "full_name:V height:D weight:I college:V years_exp:I",
    "injuries": "season:I game_type:V team:V week:I gsis_id:V position:V full_name:V "
                "report_primary_injury:V report_secondary_injury:V report_status:V "
                "practice_primary_injury:V practice_secondary_injury:V practice_status:V "
                "season_type:V",
    "snap_counts": "game_id:V season:I game_type:V week:I player:V pfr_player_id:V position:V "
                   "team:V opponent:V offense_snaps:D offense_pct:D defense_snaps:D "
                   "defense_pct:D st_snaps:D st_pct:D",
    "player_stats": "player_id:V player_display_name:V position:V season:I week:I "
                    "season_type:V team:V",
    "players": "gsis_id:V display_name:V pfr_id:V position:V",
    "ff_opportunity": "season:V posteam:V week:D game_id:V player_id:V full_name:V position:V "
                      "pass_attempt:D rec_attempt:D rush_attempt:D rec_air_yards:D receptions:D "
                      "receptions_exp:D rec_yards_gained:D rush_yards_gained:D rec_touchdown:D "
                      "rush_touchdown:D rec_touchdown_exp:D rush_touchdown_exp:D "
                      "total_fantasy_points:D total_fantasy_points_exp:D receptions_diff:D "
                      "rec_touchdown_diff:D rush_touchdown_diff:D total_fantasy_points_diff:D "
                      "rec_attempt_team:D rush_attempt_team:D rec_air_yards_team:D",
    "ftn_charting": "nflverse_game_id:V season:I week:I nflverse_play_id:I n_defense_box:I "
                    "is_no_huddle:B is_motion:B is_play_action:B is_screen_pass:B is_rpo:B "
                    "is_trick_play:B is_catchable_ball:B is_contested_ball:B "
                    "is_created_reception:B is_drop:B n_blitzers:I n_pass_rushers:I",
    "ff_rankings_draft": "page_type:V ecr_type:V player:V pos:V team:V ecr:D sd:D best:L "
                         "worst:L bye:L scrape_date:V",
    "ff_rankings_week": "page:V scrape_date:V player_name:V pos:V team:V rank:L ecr:D sd:D "
                        "best:L worst:L player_bye_week:L player_opponent:V note:V pos_rank:V "
                        "start_sit_grade:V",
    "nextgen_stats_receiving": "season:I season_type:V week:I avg_separation:D "
                               "catch_percentage:D yards:I avg_yac_above_expectation:D "
                               "player_gsis_id:V",
    "contracts": "player:V position:V team:V year_signed:I value:D apy:D apy_cap_pct:D gsis_id:V",
}
LOADER_ONLY = {"officials": "game_id:V season:I official_name:V"}
STATIC = {"players", "contracts", "ff_rankings_draft", "ff_rankings_week"}
TABLE_DS = {"plays": "pbp"}
PL_TYPES = {"V": pl.Utf8, "I": pl.Int32, "D": pl.Float64, "B": pl.Boolean, "L": pl.Int64}
TEAM_COLS = {"posteam", "defteam", "team", "home_team", "away_team", "td_team",
             "fumbled_1_team", "opponent"}


def _value(col, kind, season):
    if col == "season":
        return str(season) if kind == "V" else season
    if col in ("game_id", "nflverse_game_id"):
        return f"{season}_01_KC_BUF"
    if col in TEAM_COLS:
        return "KC"
    if col in ("season_type", "game_type"):
        return "REG"
    if col == "play_type":
        return "pass"
    if col in ("gsis_id", "player_id", "player_gsis_id", "receiver_player_id"):
        return "00-0000001"
    if col == "position":
        return "WR"
    return {"V": col, "I": 1, "D": 1.0, "B": False, "L": 1}[kind]


def frame(table, season=2024, drop=()):
    """One row of `table` for `season`, without the columns in drop."""
    spec = {**SCHEMA, **LOADER_ONLY}[table]
    cols = [c.split(":") for c in spec.split() if c.split(":")[0] not in drop]
    return pl.DataFrame({c: [_value(c, k, season)] for c, k in cols},
                        schema={c: PL_TYPES[k] for c, k in cols})


def _dc_old(season, n=3):
    """depth_charts shape up to 2024 (season column present); gsis ids '<yy>-<i>'."""
    return pl.DataFrame({
        "season": [season] * n, "club_code": ["ATL"] * n, "week": list(range(1, n + 1)),
        "game_type": ["REG"] * n, "depth_team": ["1"] * n,
        "gsis_id": [f"{season % 100}-{i}" for i in range(n)],
        "full_name": [f"Player {i}" for i in range(n)], "position": ["G"] * n,
        "depth_position": ["RG"] * n, "formation": ["Offense"] * n,
    })


def _dc_new(season, n=4):
    """depth_charts 2025+ shape: no season column (group D's fixture shape)."""
    return pl.DataFrame({
        "dt": ["2025-08-03T10:09:07Z"] * n, "team": ["ARI"] * n,
        "player_name": [f"P{season}-{i}" for i in range(n)],
        "gsis_id": [f"{season % 100}-{i}" for i in range(n)],
        "espn_id": [str(1000 + i) for i in range(n)], "pos_grp": ["Base 4-3 D"] * n,
        "pos_abb": ["LDE"] * n, "pos_slot": [1] * n, "pos_rank": [i + 1 for i in range(n)],
    })


def _quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def stamp(con, *versions):
    con.execute("CREATE TABLE IF NOT EXISTS _schema_version "
                "(version INTEGER, applied_at TIMESTAMP, nfl_mcp_version VARCHAR)")
    for v in versions:
        con.execute("INSERT INTO _schema_version VALUES (?, TIMESTAMP '2026-11-05 00:00:00', 'test')",
                    [v])


def build_db(path, *, seasons=None, drop_tables=(), drop_columns=None, derived="current",
             depth=None, versions=None):
    """Write a synthetic database.

    seasons: {table: [seasons]} for seasonal tables (default [2024]); each season is one row
    and one _ingest_metadata row. derived: "current" (rebuild_derived_tables), "v0" (rebuild,
    then drop the V0_GAPS columns as grpE_v3 lacks them) or None (no derived tables).
    depth: None (no depth_charts), "legacy" (3 rows of 2024 stamped + 4 rows of 2025 with
    season NULL, metadata for 2024 and 2025, as the old loader left them), "stamped" (same
    rows, 2025 stamped) or "noseason" (only the 4 2025 rows, no season column).
    versions: rows for _schema_version (None: no table; (): empty table).
    """
    seasons = seasons or {}
    drop_columns = drop_columns or {}
    con = duckdb.connect(str(path))
    try:
        ing._ensure_metadata_table(con)
        for table in SCHEMA:
            if table in drop_tables:
                continue
            defn = REGISTRY[TABLE_DS.get(table, table)]
            drop = drop_columns.get(table, ())
            for season in ([None] if table in STATIC else seasons.get(table, [2024])):
                df = frame(table, season or 2024, drop)
                ing._write_df_to_table(con, table, df)
                ing._record_loaded(con, defn.dataset_id, table, defn.loader_fn, len(df), season)
        if derived:
            _quiet(ing.rebuild_derived_tables, con)
        if derived == "v0":
            for table, cols in V0_GAPS.items():
                for col in cols:
                    con.execute(f'ALTER TABLE {table} DROP COLUMN "{col}"')
        if depth in ("legacy", "stamped"):
            ing._write_df_to_table(con, "depth_charts", _dc_old(2024))
            ing._record_loaded(con, "depth_charts", "depth_charts", "load_depth_charts", 3, 2024)
        if depth:
            new = _dc_new(2025)
            if depth == "stamped":
                new = ing._stamp_season(new, 2025)
            ing._write_df_to_table(con, "depth_charts", new)
            ing._record_loaded(con, "depth_charts", "depth_charts", "load_depth_charts", 4, 2025)
        if versions is not None:
            stamp(con, *versions)
    finally:
        con.close()
    return Path(path)


def v0_synth(tmp_path, name="v0.duckdb"):
    """Version 0 with every step needed (hand-built, so every expected value is known):
    derived tables in the grpE_v3 shape, 4 NULL-season depth_charts rows (2025), rosters
    without college (metadata seasons 2023, 2024), no ff_rankings_draft table."""
    return build_db(tmp_path / name, seasons={"rosters": [2023, 2024]},
                    drop_tables={"ff_rankings_draft"}, drop_columns={"rosters": ["college"]},
                    derived="v0", depth="legacy")


def current_synth(tmp_path, name="current.duckdb", versions=(1,)):
    """Structurally current (every step not needed), stamped with `versions`."""
    return build_db(tmp_path / name, derived="current", depth="stamped", versions=versions)


def _ro(path):
    return duckdb.connect(str(path), read_only=True, config=_connection_config())


@contextlib.contextmanager
def _rw(path):
    con = duckdb.connect(str(path))
    try:
        yield con
    finally:
        con.close()


def _columns(con):
    found = {}
    for t, c in con.execute("SELECT table_name, column_name FROM information_schema.columns "
                            "WHERE table_schema = 'main'").fetchall():
        found.setdefault(t, set()).add(c)
    return found


def _versions(path):
    con = _ro(path)
    try:
        if "_schema_version" not in _columns(con):
            return []
        return [r[0] for r in con.execute("SELECT version FROM _schema_version ORDER BY version")
                .fetchall()]
    finally:
        con.close()


def _count(con, sql, params=None):
    return con.execute(sql, params or []).fetchone()[0]


def _status(path):
    con = _ro(path)
    try:
        return _migrate().check(con)
    finally:
        con.close()


def _step(status, name):
    hits = [s for s in status.steps if s.name == name]
    assert len(hits) == 1, [s.name for s in status.steps]
    return hits[0]


def _needed(status):
    return [s.name for s in status.steps if s.needed]


def _digits(text):
    """Drop thousands separators so 1,280,630 and 1_280_630 read as 1280630."""
    return re.sub(r"(?<=\d)[,_ ](?=\d{3}\b)", "", text)


def _has_word(text, word):
    return re.search(rf"(?<![\w.]){re.escape(str(word))}(?![\w])", text) is not None


# ── fake nflreadpy ─────────────────────────────────────────────────────────────

SEASONAL_LOADERS = {
    "load_pbp": "plays", "load_schedules": "schedules", "load_rosters": "rosters",
    "load_injuries": "injuries", "load_snap_counts": "snap_counts",
    "load_player_stats": "player_stats", "load_ff_opportunity": "ff_opportunity",
    "load_ftn_charting": "ftn_charting", "load_nextgen_stats": "nextgen_stats_receiving",
    "load_officials": "officials",
}
STATIC_LOADERS = {"load_players": "players", "load_contracts": "contracts"}


class FakeNfl:
    """Records every loader call as (name, seasons tuple or None, kwargs).

    fail: {loader name: None (always raise) or a set of seasons that raise}."""

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or {}

    def _hit(self, name, seasons, kw):
        self.calls.append((name, tuple(seasons) if seasons not in (None, True) else None, dict(kw)))
        if name in self.fail:
            bad = self.fail[name]
            if bad is None or any(s in bad for s in (seasons or [])):
                raise ConnectionError(f"simulated outage in {name}")

    @staticmethod
    def _seasons(seasons):
        return [ing.current_season()] if seasons in (None, True) else list(seasons)

    def module(self):
        fake = types.ModuleType("nflreadpy")

        def seasonal(name, table):
            def loader(seasons=None, **kw):
                self._hit(name, seasons, kw)
                return pl.concat([frame(table, s) for s in self._seasons(seasons)])
            return loader

        def static(name, table):
            def loader(**kw):
                self._hit(name, None, kw)
                return frame(table)
            return loader

        for name, table in SEASONAL_LOADERS.items():
            setattr(fake, name, seasonal(name, table))
        for name, table in STATIC_LOADERS.items():
            setattr(fake, name, static(name, table))

        def load_ff_rankings(type="draft", **kw):
            self._hit("load_ff_rankings", None, {"type": type, **kw})
            return frame("ff_rankings_draft" if type == "draft" else "ff_rankings_week")

        def load_depth_charts(seasons=None, **kw):
            self._hit("load_depth_charts", seasons, kw)
            parts = [_dc_new(s) if s >= 2025 else _dc_old(s) for s in self._seasons(seasons)]
            return parts[0] if len(parts) == 1 else pl.concat(parts, how="diagonal_relaxed")

        fake.load_ff_rankings = load_ff_rankings
        fake.load_depth_charts = load_depth_charts
        return fake

    def install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "nflreadpy", self.module())
        return self

    def named(self, name):
        return [c for c in self.calls if c[0] == name]

    def seasons(self, name):
        return {s for _, ss, _ in self.named(name) if ss for s in ss}


def _upgrade(path, **kw):
    with _rw(path) as con:
        return _quiet(_migrate().upgrade, con, **kw)


# ── CLI helpers ────────────────────────────────────────────────────────────────

def _cli(*args):
    result = CliRunner().invoke(main, [str(a) for a in args])
    out = result.output or ""
    try:
        err = result.stderr or ""
    except (ValueError, AttributeError):
        err = ""
    if err and err not in out:
        out += err
    return result, out


def _line(out, word):
    hits = [line for line in out.splitlines() if word in line]
    assert hits, f"no line names {word}:\n{out}"
    return hits[0]


# ── real-data fixtures ─────────────────────────────────────────────────────────

def _need(path):
    if not path.exists():
        pytest.skip(f"No DuckDB database at {path}")


@pytest.fixture(scope="session")
def f_copy(tmp_path_factory):
    """One read-only copy of grpF_work per session: the oracle for the derived tables."""
    _need(F_DB)
    dest = tmp_path_factory.mktemp("grpF") / "grpF_copy.duckdb"
    shutil.copyfile(F_DB, dest)
    dest.chmod(0o444)
    return dest


@pytest.fixture(scope="session")
def v053_copy(tmp_path_factory):
    """One read-only copy of the v0.5.3 database per session."""
    _need(V053_DB)
    dest = tmp_path_factory.mktemp("v053") / "nflread_v053.duckdb"
    shutil.copyfile(V053_DB, dest)
    dest.chmod(0o444)
    return dest


class _Once:
    """Run fn on first get() and cache the result or the error for the session."""

    def __init__(self, fn):
        self.fn, self.done, self.value, self.error = fn, False, None, None

    def get(self):
        if not self.done:
            self.done = True
            try:
                self.value = self.fn()
            except BaseException as e:  # noqa: BLE001 - re-raised on every get()
                self.error = e
        if self.error is not None:
            raise self.error
        return self.value


def _team_snapshot(con):
    return {t: con.execute(f"SELECT * FROM {t} ORDER BY ALL").fetchall() for t in TEAM_TABLES}


@pytest.fixture(scope="session")
def v0_upgraded(tmp_path_factory, f_copy):
    """grpE_v3 copied once, then upgrade(fetch=False) twice. The copy is made only when
    nfl_mcp.migrate imports, so today nothing is copied."""
    _need(V0_DB)

    def run():
        migrate = _migrate()
        dest = tmp_path_factory.mktemp("grpE") / "grpE_copy.duckdb"
        shutil.copyfile(V0_DB, dest)
        fake = FakeNfl()
        with pytest.MonkeyPatch.context() as mp:
            fake.install(mp)
            con = duckdb.connect(str(dest))
            try:
                first = _quiet(migrate.upgrade, con, fetch=False)
                snap1 = _team_snapshot(con)
                second = _quiet(migrate.upgrade, con, fetch=False)
                snap2 = _team_snapshot(con)
                null_depth = _count(con, "SELECT COUNT(*) FROM depth_charts WHERE season IS NULL")
                con.execute(f"ATTACH '{f_copy}' AS oracle (READ_ONLY)")
                diff = {t: (_count(con, f"SELECT COUNT(*) FROM (SELECT * FROM main.{t} "
                                        f"EXCEPT ALL SELECT * FROM oracle.{t})"),
                            _count(con, f"SELECT COUNT(*) FROM (SELECT * FROM oracle.{t} "
                                        f"EXCEPT ALL SELECT * FROM main.{t})"))
                        for t in TEAM_TABLES}
                con.execute("DETACH oracle")
            finally:
                con.close()
        return {"path": dest, "first": first, "second": second, "snap1": snap1,
                "snap2": snap2, "null_depth": null_depth, "diff": diff, "calls": fake.calls,
                "versions": _versions(dest)}

    return _Once(run)


# ═══ 1. current behavior the fix must keep (no marker) ═════════════════════════


def test_synthetic_schema_builds_every_derived_table_with_required_columns(tmp_path):
    # The fixture is only useful if rebuild_derived_tables accepts it: every derived table
    # of REQUIRED_COLUMNS must come out with all its listed columns.
    path = build_db(tmp_path / "s.duckdb")
    con = _ro(path)
    try:
        cols = _columns(con)
    finally:
        con.close()
    missing = sorted((t, c) for t, need in REQUIRED_COLUMNS.items()
                     for c in need if c not in cols.get(t, set()))
    assert missing == []


def test_run_ingest_on_empty_file_with_stub_loaders_builds_every_required_table(tmp_path, monkeypatch):
    path = tmp_path / "fresh.duckdb"
    fake = FakeNfl().install(monkeypatch)
    failures = _quiet(ing.run_ingest_datasets, INGEST_DATASETS, start=2024, end=2024,
                      db_path=str(path))
    assert failures == []
    con = _ro(path)
    try:
        cols = _columns(con)
    finally:
        con.close()
    assert sorted((t, c) for t, need in REQUIRED_COLUMNS.items()
                  for c in need if c not in cols.get(t, set())) == []
    assert fake.named("load_pbp")  # stubbed, never the network


def test_legacy_restamp_plan_finds_the_null_season_rows(tmp_path):
    # Group D's plan, which the depth_charts_season step reuses: 2025 has metadata and no
    # stamped rows -> (True, [2025]).
    path = build_db(tmp_path / "d.duckdb", depth="legacy")
    with _rw(path) as con:
        assert ing._legacy_restamp_plan(con, REGISTRY["depth_charts"]) == (True, [2025])
        assert _count(con, "SELECT COUNT(*) FROM depth_charts WHERE season IS NULL") == 4


def test_failed_restamp_fetch_keeps_legacy_rows(tmp_path, monkeypatch):
    # Group D's fetch-first rule: a failed re-fetch deletes nothing (4 NULL + 3 stamped).
    path = build_db(tmp_path / "d.duckdb", depth="legacy")
    FakeNfl(fail={"load_depth_charts": None}).install(monkeypatch)
    monkeypatch.setattr(ing, "current_season", lambda: 2026)
    ing._FAILURES.clear()
    with _rw(path) as con:
        _quiet(ing._ingest_generic_dataset, con, REGISTRY["depth_charts"], [2024], fresh=False)
        assert _count(con, "SELECT COUNT(*) FROM depth_charts WHERE season IS NULL") == 4
        assert _count(con, "SELECT COUNT(*) FROM depth_charts WHERE season = 2024") == 3
    assert any(f.startswith("depth_charts 2025") for f in ing._FAILURES), ing._FAILURES


def test_validate_existing_checks_pass_on_synthetic_current_slot(tmp_path):
    from nfl_mcp.validate import run_checks
    path = current_synth(tmp_path)
    results = [r for r in run_checks(path, path) if r.name in EXISTING_CHECKS]
    assert {r.name for r in results} == EXISTING_CHECKS
    assert [f"{r.name} {r.detail}" for r in results if not r.ok] == []


def test_serve_starts_on_an_old_database_today(tmp_path, monkeypatch):
    path = v0_synth(tmp_path)
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    started = []
    monkeypatch.setattr("nfl_mcp.cli.uvicorn.run", lambda app, host, port: started.append(port))
    result, out = _cli("serve")
    assert result.exit_code == 0, out
    assert started == [8000]


def test_serve_with_missing_database_starts_without_upgrade_warning(tmp_path, monkeypatch, caplog):
    result, started, warnings = _serve(monkeypatch, caplog, tmp_path / "absent.duckdb")
    assert result.exit_code == 0
    assert started
    assert warnings == []
    assert not (tmp_path / "absent.duckdb").exists()


@pytest.mark.integration
def test_grpE_v3_shape_measurements():
    # The numbers check() must report on the version-0 example.
    _need(V0_DB)
    con = _ro(V0_DB)
    try:
        cols = _columns(con)
        assert "_schema_version" not in cols
        for table, gaps in V0_GAPS.items():
            assert not set(gaps) & cols[table], table
        assert "season" in cols["depth_charts"]
        assert _count(con, "SELECT COUNT(*) FROM depth_charts WHERE season IS NULL") == V0_NULL_DEPTH
        assert _count(con, "SELECT COUNT(*) FROM depth_charts WHERE season = 2024") == 37_312
        raw_gaps = [(t, c) for t in RAW for c in REQUIRED_COLUMNS[t] if c not in cols.get(t, set())]
        assert raw_gaps == []
        derived_gaps = sorted((t, c) for t in DERIVED for c in REQUIRED_COLUMNS[t]
                              if c not in cols.get(t, set()))
        assert derived_gaps == sorted((t, c) for t, cs in V0_GAPS.items() for c in cs)
    finally:
        con.close()


@pytest.mark.integration
def test_v053_shape_measurements(v053_copy):
    con = _ro(v053_copy)
    try:
        cols = _columns(con)
        assert "_schema_version" not in cols
        assert all("season_type" in cols[t] for t in V0_GAPS)
        derived_gaps = sorted((t, c) for t in DERIVED for c in REQUIRED_COLUMNS[t]
                              if c not in cols.get(t, set()))
        assert derived_gaps == [("situational_stats", "conversion_rule"),
                                ("team_defense_stats", "def_touchdowns"),
                                ("team_offense_stats", "pass_attempts")]
        assert [(t, c) for t in RAW for c in REQUIRED_COLUMNS[t] if c not in cols.get(t, set())] == []
        assert _count(con, "SELECT COUNT(*) FROM depth_charts WHERE season IS NULL") == V0_NULL_DEPTH
        assert con.execute("SELECT row_count FROM _ingest_metadata WHERE dataset_id = 'derived_shape'"
                           ).fetchall() == [(4,)]
    finally:
        con.close()


@pytest.mark.integration
def test_grpF_work_has_every_required_column(f_copy):
    con = _ro(f_copy)
    try:
        cols = _columns(con)
    finally:
        con.close()
    assert sorted((t, c) for t, need in REQUIRED_COLUMNS.items()
                  for c in need if c not in cols.get(t, set())) == []


@pytest.mark.integration
def test_rebuild_derived_tables_on_grpE_v3_sources_reproduces_grpF_work(f_copy):
    # Copy the team tables' sources from the original (attached read-only) into memory,
    # rebuild, compare with grpF_work. Oracle row values measured in grpF_work.
    _need(V0_DB)
    con = duckdb.connect(":memory:")
    try:
        con.execute(f"ATTACH '{V0_DB}' AS src (READ_ONLY)")
        for table in ("plays", "schedules", "espn_team_records", "_ingest_metadata"):
            con.execute(f"CREATE TABLE main.{table} AS SELECT * FROM src.{table}")
        con.execute("DETACH src")
        _quiet(ing.rebuild_derived_tables, con)
        con.execute(f"ATTACH '{f_copy}' AS oracle (READ_ONLY)")
        for t in TEAM_TABLES:
            assert _count(con, f"SELECT COUNT(*) FROM (SELECT * FROM main.{t} EXCEPT ALL "
                               f"SELECT * FROM oracle.{t})") == 0, t
            assert _count(con, f"SELECT COUNT(*) FROM (SELECT * FROM oracle.{t} EXCEPT ALL "
                               f"SELECT * FROM main.{t})") == 0, t
        assert con.execute(
            "SELECT total_plays, total_yards, rush_plays, pass_plays, pass_attempts FROM "
            "main.team_offense_stats WHERE team = 'KC' AND season_year = 2024 AND season_type = 'REG'"
        ).fetchall() == [(1067, 5602.0, 450, 598, 600)]
    finally:
        con.close()


@pytest.mark.integration
def test_validate_existing_checks_pass_on_grpF_work_against_itself(f_copy):
    from nfl_mcp.validate import run_checks
    results = [r for r in run_checks(f_copy, f_copy) if r.name in EXISTING_CHECKS]
    assert {r.name for r in results} == EXISTING_CHECKS
    assert [f"{r.name} {r.detail}" for r in results if not r.ok] == []


# ═══ 2. migrate.check ══════════════════════════════════════════════════════════


def test_schema_version_constant_is_1():
    assert _migrate().SCHEMA_VERSION == 1


def test_check_v0_synthetic_reports_every_step(tmp_path):
    status = _status(v0_synth(tmp_path))
    assert (status.version, status.target, status.newer, status.upgrade_needed) == (0, 1, False, True)
    assert [s.name for s in status.steps] == STEPS
    assert [s.needs_network for s in status.steps] == [False, True, True]
    assert _needed(status) == STEPS


def test_check_v0_synthetic_details(tmp_path):
    status = _status(v0_synth(tmp_path))
    derived = _step(status, "derived_tables").detail
    for col in ("season_type", "pass_attempts", "def_touchdowns", "conversion_rule"):
        assert col in derived, derived
    # 4 rows of the 2025 frame were written without a season (build_db depth="legacy").
    assert _has_word(_digits(_step(status, "depth_charts_season").detail), 4)
    raw = _step(status, "raw_columns").detail
    assert "rosters.college" in raw and "ff_rankings_draft" in raw, raw


def test_check_current_database_needs_nothing(tmp_path):
    status = _status(current_synth(tmp_path))
    assert (status.version, status.target, status.newer, status.upgrade_needed) == (1, 1, False, False)
    assert [s.name for s in status.steps] == STEPS
    assert _needed(status) == []


@pytest.mark.parametrize("versions", [None, ()])
def test_check_structurally_current_but_unstamped_is_version_0(tmp_path, versions):
    # No table, or an empty table, is version 0. Correction 1: no step is needed, but the
    # upgrade is (version < target).
    status = _status(current_synth(tmp_path, versions=versions))
    assert (status.version, status.upgrade_needed, status.newer) == (0, True, False)
    assert _needed(status) == []


def test_check_version_is_max_of_rows(tmp_path):
    status = _status(current_synth(tmp_path, versions=(1, 2)))
    assert (status.version, status.newer) == (2, True)


def test_check_missing_derived_table_names_it(tmp_path):
    path = current_synth(tmp_path)
    with _rw(path) as con:
        con.execute("DROP TABLE player_td_luck")
    status = _status(path)
    assert _needed(status) == ["derived_tables"]
    assert "player_td_luck" in _step(status, "derived_tables").detail
    assert status.upgrade_needed


def test_check_depth_charts_without_season_column_is_needed(tmp_path):
    # 4 rows of the 2025 shape and no season column at all: every row lacks a season.
    status = _status(build_db(tmp_path / "d.duckdb", depth="noseason", versions=(1,)))
    step = _step(status, "depth_charts_season")
    assert step.needed and step.needs_network
    assert _has_word(_digits(step.detail), 4)
    assert status.upgrade_needed


def test_check_without_depth_charts_table_does_not_need_restamp(tmp_path):
    # Correction 3.
    status = _status(build_db(tmp_path / "d.duckdb", depth=None, versions=(1,)))
    assert not _step(status, "depth_charts_season").needed
    assert not status.upgrade_needed


def test_check_raw_missing_column_and_static_table(tmp_path):
    path = build_db(tmp_path / "r.duckdb", drop_tables={"ff_rankings_week"},
                    drop_columns={"rosters": ["college"]}, versions=(1,))
    status = _status(path)
    assert _needed(status) == ["raw_columns"]
    detail = _step(status, "raw_columns").detail
    assert "rosters.college" in detail and "ff_rankings_week" in detail


def test_check_writes_nothing_on_a_read_write_connection(tmp_path):
    path = v0_synth(tmp_path)
    with _rw(path) as con:
        before = _columns(con)
        _migrate().check(con)
        after = _columns(con)
    assert after == before
    assert "_schema_version" not in after


def test_check_works_read_only_and_leaves_file_untouched(tmp_path):
    path = v0_synth(tmp_path)
    path.chmod(0o444)
    stat = path.stat()
    status = _status(path)
    assert status.version == 0 and status.upgrade_needed
    assert (path.stat().st_mtime_ns, path.stat().st_size) == (stat.st_mtime_ns, stat.st_size)
    assert not Path(f"{path}.wal").exists()


# ── check on the real databases ──


@pytest.mark.integration
def test_check_grpE_v3_steps_and_version():
    _need(V0_DB)
    status = _status(V0_DB)
    assert (status.version, status.target, status.newer, status.upgrade_needed) == (0, 1, False, True)
    assert [s.name for s in status.steps] == STEPS
    assert [s.needs_network for s in status.steps] == [False, True, True]
    assert _needed(status) == ["derived_tables", "depth_charts_season"]


@pytest.mark.integration
def test_check_grpE_v3_derived_detail_names_season_type():
    _need(V0_DB)
    detail = _step(_status(V0_DB), "derived_tables").detail
    for col in ("season_type", "pass_attempts", "def_touchdowns", "conversion_rule"):
        assert col in detail, detail


@pytest.mark.integration
def test_check_grpE_v3_depth_detail_gives_null_count():
    _need(V0_DB)
    step = _step(_status(V0_DB), "depth_charts_season")
    assert step.needed and step.needs_network
    assert _has_word(_digits(step.detail), V0_NULL_DEPTH), step.detail


@pytest.mark.integration
def test_check_grpE_v3_raw_columns_not_needed():
    # Measured: no raw REQUIRED_COLUMNS table or column is missing in grpE_v3.
    _need(V0_DB)
    assert not _step(_status(V0_DB), "raw_columns").needed


@pytest.mark.integration
def test_check_grpE_v3_writes_nothing():
    _need(V0_DB)
    before = (V0_DB.stat().st_mtime_ns, V0_DB.stat().st_size)
    con = _ro(V0_DB)
    try:
        _migrate().check(con)
        _migrate().check(con)
    finally:
        con.close()
    assert (V0_DB.stat().st_mtime_ns, V0_DB.stat().st_size) == before
    assert not Path(f"{V0_DB}.wal").exists()


@pytest.mark.integration
def test_check_v053_reports_version_0_not_newer(v053_copy):
    # Correction 4: the derived_shape row (row_count 4) is not a schema version.
    status = _status(v053_copy)
    assert (status.version, status.newer, status.upgrade_needed) == (0, False, True)
    assert _needed(status) == ["derived_tables", "depth_charts_season"]


@pytest.mark.integration
def test_check_v053_details(v053_copy):
    status = _status(v053_copy)
    derived = _step(status, "derived_tables").detail
    for col in ("pass_attempts", "def_touchdowns", "conversion_rule"):
        assert col in derived, derived
    assert "season_type" not in derived, derived   # v0.5.3 already has season_type
    assert _has_word(_digits(_step(status, "depth_charts_season").detail), V0_NULL_DEPTH)


@pytest.mark.integration
def test_check_grpF_work_needs_only_the_depth_restamp(f_copy):
    status = _status(f_copy)
    assert status.version == 0
    assert _needed(status) == ["depth_charts_season"]


# ═══ 3. migrate.upgrade ════════════════════════════════════════════════════════


def test_full_upgrade_runs_every_step_and_stamps(tmp_path, monkeypatch):
    path = v0_synth(tmp_path)
    FakeNfl().install(monkeypatch)
    result = _upgrade(path)
    assert result.ran == STEPS
    assert (result.failed, result.pending, result.stamped) == ([], [], True)
    assert result.before.version == 0 and _needed(result.before) == STEPS
    assert (result.after.version, result.after.upgrade_needed) == (1, False)
    assert _needed(result.after) == []
    assert _versions(path) == [1]


def test_full_upgrade_fills_in_the_data(tmp_path, monkeypatch):
    path = v0_synth(tmp_path)
    fake = FakeNfl().install(monkeypatch)
    _upgrade(path)
    assert 2025 in fake.seasons("load_depth_charts")
    assert fake.seasons("load_rosters") == {2023, 2024}   # the metadata seasons, exactly
    assert [c[2].get("type") for c in fake.named("load_ff_rankings")] == ["draft"]
    con = _ro(path)
    try:
        cols = _columns(con)
        assert {"season_type", "pass_attempts"} <= cols["team_offense_stats"]
        assert {"season_type", "def_touchdowns"} <= cols["team_defense_stats"]
        assert {"season_type", "conversion_rule"} <= cols["situational_stats"]
        assert _count(con, "SELECT COUNT(*) FROM depth_charts WHERE season IS NULL") == 0
        # 2024: 3 rows kept; 2025: the 4 re-fetched rows, stamped.
        assert dict(con.execute("SELECT season, COUNT(*) FROM depth_charts GROUP BY 1").fetchall()) \
            == {2024: 3, 2025: 4}
        assert "college" in cols["rosters"]
        assert _count(con, "SELECT COUNT(*) FROM rosters") == 2   # one row per season, replaced
        assert _count(con, "SELECT COUNT(*) FROM ff_rankings_draft") == 1
    finally:
        con.close()


def test_check_after_full_upgrade_reports_current(tmp_path, monkeypatch):
    path = v0_synth(tmp_path)
    FakeNfl().install(monkeypatch)
    _upgrade(path)
    status = _status(path)
    assert (status.version, status.upgrade_needed, status.newer) == (1, False, False)
    assert _needed(status) == []
    con = _ro(path)
    try:
        assert con.execute("SELECT COUNT(*), COUNT(applied_at) FROM _schema_version").fetchone() == (1, 1)
    finally:
        con.close()


def test_second_upgrade_runs_nothing_and_stamps_nothing(tmp_path, monkeypatch):
    path = v0_synth(tmp_path)
    fake = FakeNfl().install(monkeypatch)
    _upgrade(path)
    n_calls = len(fake.calls)
    again = _upgrade(path)
    assert (again.ran, again.failed, again.pending, again.stamped) == ([], [], [], False)
    assert len(fake.calls) == n_calls
    assert _versions(path) == [1]


def test_failed_depth_fetch_keeps_legacy_rows_and_later_steps_run(tmp_path, monkeypatch):
    path = v0_synth(tmp_path)
    fake = FakeNfl(fail={"load_depth_charts": None}).install(monkeypatch)
    result = _upgrade(path)
    assert [f for f in result.failed if f.startswith("depth_charts_season")] == result.failed
    assert len(result.failed) == 1
    assert "derived_tables" in result.ran and "raw_columns" in result.ran
    assert "depth_charts_season" not in result.ran
    assert result.stamped is False
    assert _versions(path) == []
    assert fake.seasons("load_rosters") == {2023, 2024}
    con = _ro(path)
    try:
        assert _count(con, "SELECT COUNT(*) FROM depth_charts WHERE season IS NULL") == 4
        assert _count(con, "SELECT COUNT(*) FROM depth_charts WHERE season = 2024") == 3
        assert [r[0] for r in con.execute("SELECT row_count FROM _ingest_metadata WHERE "
                                          "dataset_id = 'depth_charts' AND season = 2025").fetchall()] == [4]
    finally:
        con.close()
    assert _step(result.after, "depth_charts_season").needed


def test_failed_upgrade_retried_later_completes(tmp_path, monkeypatch):
    path = v0_synth(tmp_path)
    FakeNfl(fail={"load_depth_charts": None}).install(monkeypatch)
    _upgrade(path)
    FakeNfl().install(monkeypatch)
    result = _upgrade(path)
    assert result.ran == ["depth_charts_season"]
    assert result.stamped and _versions(path) == [1]


def test_upgrade_no_fetch_runs_only_derived_tables(tmp_path, monkeypatch):
    path = v0_synth(tmp_path)
    fake = FakeNfl().install(monkeypatch)
    result = _upgrade(path, fetch=False)
    assert result.ran == ["derived_tables"]
    assert result.pending == ["depth_charts_season", "raw_columns"]
    assert (result.failed, result.stamped) == ([], False)
    assert fake.calls == []
    assert _versions(path) == []
    assert not _step(result.after, "derived_tables").needed


def test_upgrade_no_fetch_stamps_when_only_derived_tables_needed(tmp_path, monkeypatch):
    path = build_db(tmp_path / "d.duckdb", derived="v0")
    FakeNfl().install(monkeypatch)
    result = _upgrade(path, fetch=False)
    assert (result.ran, result.pending, result.stamped) == (["derived_tables"], [], True)
    assert _versions(path) == [1]


def test_upgrade_on_unstamped_current_database_only_stamps(tmp_path, monkeypatch):
    # Correction 1: nothing structural to do, so no step runs; the version is written.
    path = current_synth(tmp_path, versions=None)
    fake = FakeNfl().install(monkeypatch)
    result = _upgrade(path)
    assert (result.ran, result.failed, result.stamped) == ([], [], True)
    assert fake.calls == []
    assert _versions(path) == [1]


# ── raw_columns ──


def test_raw_missing_column_reingests_metadata_seasons_fresh(tmp_path, monkeypatch):
    path = build_db(tmp_path / "r.duckdb", seasons={"rosters": [2022, 2023]},
                    drop_columns={"rosters": ["college"]})
    fake = FakeNfl().install(monkeypatch)
    result = _upgrade(path)
    assert result.ran == ["raw_columns", "derived_tables"] and result.stamped
    # 2022 and 2023 are recorded as loaded and are not the current season, so only
    # fresh=True re-fetches them.
    assert fake.seasons("load_rosters") == {2022, 2023}
    assert {c[0] for c in fake.calls} == {"load_rosters"}
    con = _ro(path)
    try:
        assert "college" in _columns(con)["rosters"]
        assert dict(con.execute("SELECT season, COUNT(*) FROM rosters GROUP BY 1").fetchall()) \
            == {2022: 1, 2023: 1}
    finally:
        con.close()


def test_raw_missing_seasonal_table_loads_default_seasons(tmp_path, monkeypatch):
    path = build_db(tmp_path / "r.duckdb", drop_tables={"player_stats"})
    fake = FakeNfl().install(monkeypatch)
    result = _upgrade(path)
    defn = REGISTRY["player_stats"]
    assert fake.seasons("load_player_stats") == set(ing._clip_to_window(ing.ALL_SEASONS, defn))
    assert all(c[2] == defn.extra_params for c in fake.named("load_player_stats"))
    assert result.stamped
    con = _ro(path)
    try:
        assert set(REQUIRED_COLUMNS["player_stats"]) <= _columns(con)["player_stats"]
    finally:
        con.close()


def test_raw_missing_static_table_loads_once(tmp_path, monkeypatch):
    path = build_db(tmp_path / "r.duckdb", drop_tables={"ff_rankings_week"})
    fake = FakeNfl().install(monkeypatch)
    result = _upgrade(path)
    assert [(c[1], c[2].get("type")) for c in fake.named("load_ff_rankings")] == [(None, "week")]
    assert result.ran == ["raw_columns", "derived_tables"] and result.stamped


def test_raw_missing_column_and_table_reingest_both(tmp_path, monkeypatch):
    path = build_db(tmp_path / "r.duckdb", seasons={"snap_counts": [2023]},
                    drop_tables={"ff_rankings_draft"}, drop_columns={"snap_counts": ["st_pct"]})
    fake = FakeNfl().install(monkeypatch)
    _upgrade(path)
    assert fake.seasons("load_snap_counts") == {2023}
    assert [c[2].get("type") for c in fake.named("load_ff_rankings")] == ["draft"]
    assert _versions(path) == [1]


def test_missing_derived_source_table_is_complete_after_two_runs(tmp_path, monkeypatch):
    # players feeds player_role_trend and injury_return_curve. Whether upgrade rebuilds the
    # derived tables after raw_columns or leaves that to a second run, two runs finish it.
    path = build_db(tmp_path / "r.duckdb", drop_tables={"players"})
    FakeNfl().install(monkeypatch)
    _upgrade(path)
    _upgrade(path)
    assert _versions(path) == [1]
    con = _ro(path)
    try:
        assert {"players", "player_role_trend", "injury_return_curve"} <= set(_columns(con))
    finally:
        con.close()


def test_raw_columns_run_always_rebuilds_derived_tables_after(tmp_path, monkeypatch):
    # No structural gap in the derived tables, but the raw data was reloaded: the derived
    # tables are rebuilt after it, once, and listed once in ran.
    path = build_db(tmp_path / "r.duckdb", drop_columns={"rosters": ["college"]})
    FakeNfl().install(monkeypatch)
    order = []
    real_run = _migrate()._ACTIONS["raw_columns"]
    monkeypatch.setitem(_migrate()._ACTIONS, "raw_columns",
                        lambda conn: (order.append("raw"), real_run(conn)))
    real_rebuild = ing.rebuild_derived_tables
    monkeypatch.setattr(ing, "rebuild_derived_tables",
                        lambda conn: (order.append("derived"), real_rebuild(conn)))
    result = _upgrade(path)
    assert order == ["raw", "derived"]
    assert result.ran == ["raw_columns", "derived_tables"] and result.stamped


def test_derived_failure_before_raw_columns_is_cleared_by_the_rebuild_after(tmp_path, monkeypatch):
    path = build_db(tmp_path / "r.duckdb", derived="v0", drop_columns={"rosters": ["college"]})
    FakeNfl().install(monkeypatch)
    real_rebuild = ing.rebuild_derived_tables
    calls = []

    def flaky(conn):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("first rebuild fails")
        return real_rebuild(conn)
    monkeypatch.setattr(ing, "rebuild_derived_tables", flaky)
    result = _upgrade(path)
    assert len(calls) == 2
    assert result.failed == [] and result.ran == ["raw_columns", "derived_tables"] and result.stamped


# ── newer database ──


def test_newer_database_is_reported(tmp_path):
    status = _status(build_db(tmp_path / "n.duckdb", derived="v0", depth="legacy", versions=(2,)))
    assert (status.version, status.target, status.newer) == (2, 1, True)


def test_upgrade_refuses_newer_database_and_changes_nothing(tmp_path, monkeypatch):
    path = build_db(tmp_path / "n.duckdb", derived="v0", depth="legacy", versions=(2,))
    fake = FakeNfl().install(monkeypatch)
    result = _upgrade(path)
    assert result.failed and result.ran == [] and result.stamped is False
    assert fake.calls == []
    assert _versions(path) == [2]
    con = _ro(path)
    try:
        assert "season_type" not in _columns(con)["team_offense_stats"]
        assert _count(con, "SELECT COUNT(*) FROM depth_charts WHERE season IS NULL") == 4
    finally:
        con.close()


# ── stamp_if_current ──


def test_stamp_if_current_stamps_once(tmp_path):
    path = current_synth(tmp_path, versions=None)
    with _rw(path) as con:
        assert _migrate().stamp_if_current(con) is True
        assert _migrate().stamp_if_current(con) is False
    assert _versions(path) == [1]


def test_stamp_if_current_on_stamped_database_adds_no_row(tmp_path):
    path = current_synth(tmp_path)
    with _rw(path) as con:
        assert _migrate().stamp_if_current(con) is False
    assert _versions(path) == [1]


def test_stamp_if_current_logs_one_warning_when_the_check_raises(tmp_path, monkeypatch, caplog):
    path = current_synth(tmp_path, versions=None)
    monkeypatch.setattr(_migrate(), "check", lambda conn: (_ for _ in ()).throw(RuntimeError("boom")))
    with _rw(path) as con, caplog.at_level(logging.WARNING, logger="nfl-mcp"):
        assert _migrate().stamp_if_current(con) is False
    hits = [r for r in caplog.records if r.levelno == logging.WARNING and "boom" in r.getMessage()]
    assert len(hits) == 1, caplog.records
    assert _versions(path) == []


@pytest.mark.parametrize("kw", [
    {"depth": "legacy"},                                   # depth_charts_season needed
    {"derived": "v0"},                                     # derived_tables needed
    {"drop_columns": {"rosters": ["college"]}},            # raw_columns needed
])
def test_stamp_if_current_refuses_when_a_step_is_needed(tmp_path, kw):
    path = build_db(tmp_path / "s.duckdb", **kw)
    with _rw(path) as con:
        assert _migrate().stamp_if_current(con) is False
    assert _versions(path) == []


# ── upgrade(fetch=False) on a copy of grpE_v3 ──


@pytest.mark.integration
def test_grpE_v3_upgrade_no_fetch_runs_derived_and_leaves_depth_pending(v0_upgraded):
    r = v0_upgraded.get()
    first = r["first"]
    assert first.ran == ["derived_tables"]
    assert first.pending == ["depth_charts_season"]
    assert (first.failed, first.stamped) == ([], False)
    assert _needed(first.after) == ["depth_charts_season"]
    assert first.after.version == 0
    assert r["versions"] == []
    assert r["calls"] == []
    assert r["null_depth"] == V0_NULL_DEPTH   # fetch=False touches no depth_charts row


@pytest.mark.integration
def test_grpE_v3_upgraded_team_tables_equal_grpF_work(v0_upgraded):
    # EXCEPT ALL both ways against grpF_work (measured equal after a plain rebuild).
    assert v0_upgraded.get()["diff"] == {t: (0, 0) for t in TEAM_TABLES}


@pytest.mark.integration
def test_grpE_v3_upgraded_oracle_rows(v0_upgraded):
    snap = v0_upgraded.get()["snap1"]
    off = [r for r in snap["team_offense_stats"] if r[:3] == ("KC", 2024, "REG")]
    assert len(off) == 1 and off[0][3:9] == (1067, 5602.0, 5.25, 450, 598, 600)
    deff = [r for r in snap["team_defense_stats"] if r[:3] == ("PHI", 2024, "POST")]
    assert len(deff) == 1 and deff[0][3:10] == (255, 1330.0, 5.22, 16, 6, 13, 1)
    rec = [r for r in snap["team_records"] if r[:2] == (2025, "KC")]
    assert rec == [(2025, "KC", "KC", 17, 6, 11, 0, "nflverse")]


@pytest.mark.integration
def test_grpE_v3_upgraded_situational_rows(v0_upgraded):
    # BUF 2025 REG plays per situation, measured in grpF_work.
    r = v0_upgraded.get()
    con = _ro(r["path"])
    try:
        got = dict(con.execute("SELECT situation, plays FROM situational_stats WHERE team = 'BUF' "
                               "AND season_year = 2025 AND season_type = 'REG'").fetchall())
    finally:
        con.close()
    assert got == {"3rd & Long": 84, "3rd & Short": 77, "4th Down": 32, "Red Zone": 143,
                   "Standard": 694, "Two Minute Drill": 26}


@pytest.mark.integration
def test_grpE_v3_second_no_fetch_run_is_a_no_op(v0_upgraded):
    r = v0_upgraded.get()
    second = r["second"]
    assert (second.ran, second.failed, second.stamped) == ([], [], False)
    assert second.pending == ["depth_charts_season"]
    assert r["snap2"] == r["snap1"]
    assert r["versions"] == []


# ═══ 4. ingest stamping ════════════════════════════════════════════════════════

INGEST_DATASETS = ["pbp", "schedules", "rosters", "injuries", "snap_counts", "player_stats",
                   "players", "ff_opportunity", "ftn_charting", "ff_rankings_draft",
                   "ff_rankings_week", "nextgen_stats_receiving", "contracts"]


def _ingest(path, datasets=INGEST_DATASETS):
    return _quiet(ing.run_ingest_datasets, datasets, start=2024, end=2024, db_path=str(path))


def test_fresh_ingest_without_failures_ends_at_version_1(tmp_path, monkeypatch):
    path = tmp_path / "fresh.duckdb"
    FakeNfl().install(monkeypatch)
    assert _ingest(path) == []
    assert _versions(path) == [1]
    status = _status(path)
    assert (status.version, status.upgrade_needed) == (1, False)


def test_second_clean_ingest_adds_no_version_row(tmp_path, monkeypatch):
    path = tmp_path / "fresh.duckdb"
    FakeNfl().install(monkeypatch)
    _ingest(path)
    _ingest(path)
    assert _versions(path) == [1]


def test_ingest_with_a_failure_stays_unstamped(tmp_path, monkeypatch):
    # officials is not a required table: the structure is complete, only the failure
    # keeps the stamp off.
    path = tmp_path / "fresh.duckdb"
    FakeNfl(fail={"load_officials": None}).install(monkeypatch)
    failures = _ingest(path, INGEST_DATASETS + ["officials"])
    assert failures and any("officials" in f for f in failures)
    assert _versions(path) == []
    status = _status(path)
    assert status.version == 0 and _needed(status) == []


def test_ingest_missing_a_required_raw_table_stays_unstamped(tmp_path, monkeypatch):
    path = tmp_path / "fresh.duckdb"
    FakeNfl().install(monkeypatch)
    assert _ingest(path, [d for d in INGEST_DATASETS if d != "player_stats"]) == []
    assert _versions(path) == []
    status = _status(path)
    assert "raw_columns" in _needed(status)
    assert "player_stats" in _step(status, "raw_columns").detail


# ═══ 5. CLI upgrade ════════════════════════════════════════════════════════════


def test_cli_check_current_exits_0(tmp_path):
    result, out = _cli("upgrade", "--check", "--db", current_synth(tmp_path))
    assert result.exit_code == 0, out
    for name in STEPS:
        line = _line(out, name)
        assert re.search(r"\bok\b", line, re.IGNORECASE) and "NEEDED" not in line, line


def test_cli_check_old_exits_3_and_prints_each_step(tmp_path):
    result, out = _cli("upgrade", "--check", "--db", v0_synth(tmp_path))
    assert result.exit_code == 3, out
    derived = _line(out, "derived_tables")
    assert "NEEDED" in derived and "(network)" not in derived
    assert "NEEDED (network)" in _line(out, "depth_charts_season")
    assert "NEEDED (network)" in _line(out, "raw_columns")
    version = [line for line in out.splitlines() if re.search(r"version", line, re.IGNORECASE)]
    assert any(_has_word(v, 0) and _has_word(v, 1) for v in version), out


def test_cli_check_newer_exits_1(tmp_path):
    result, out = _cli("upgrade", "--check", "--db", current_synth(tmp_path, versions=(2,)))
    assert result.exit_code == 1, out


def test_cli_check_works_on_read_only_file(tmp_path):
    path = v0_synth(tmp_path)
    path.chmod(0o444)
    before = path.stat().st_mtime_ns
    result, out = _cli("upgrade", "--check", "--db", path)
    assert result.exit_code == 3, out
    assert path.stat().st_mtime_ns == before
    assert _versions(path) == []


def test_cli_check_default_db_is_the_configured_path(tmp_path, monkeypatch):
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(v0_synth(tmp_path)))
    result, out = _cli("upgrade", "--check")
    assert result.exit_code == 3, out


@pytest.mark.parametrize("check", [True, False])
def test_cli_missing_file_exits_1_and_creates_nothing(tmp_path, check):
    path = tmp_path / "absent.duckdb"
    result, out = _cli("upgrade", *(["--check"] if check else []), "--db", path)
    assert result.exit_code == 1, out
    assert str(path) in out
    assert not path.exists() and not Path(f"{path}.wal").exists()


def test_cli_upgrade_full_exits_0_and_stamps(tmp_path, monkeypatch):
    path = v0_synth(tmp_path)
    FakeNfl().install(monkeypatch)
    result, out = _cli("upgrade", "--db", path)
    assert result.exit_code == 0, out
    for name in STEPS:
        assert name in out
    assert _versions(path) == [1]
    again, out2 = _cli("upgrade", "--db", path)
    assert again.exit_code == 0, out2
    assert _versions(path) == [1]


def test_cli_upgrade_no_fetch_exits_3_when_network_steps_remain(tmp_path, monkeypatch):
    path = v0_synth(tmp_path)
    fake = FakeNfl().install(monkeypatch)
    result, out = _cli("upgrade", "--db", path, "--no-fetch")
    assert result.exit_code == 3, out
    assert fake.calls == []
    assert _versions(path) == []
    con = _ro(path)
    try:
        assert "season_type" in _columns(con)["team_offense_stats"]   # derived ran
    finally:
        con.close()


def test_cli_upgrade_no_fetch_exits_0_when_nothing_needs_network(tmp_path, monkeypatch):
    path = build_db(tmp_path / "d.duckdb", derived="v0")
    FakeNfl().install(monkeypatch)
    result, out = _cli("upgrade", "--db", path, "--no-fetch")
    assert result.exit_code == 0, out
    assert _versions(path) == [1]


def test_cli_upgrade_failed_step_exits_1(tmp_path, monkeypatch):
    path = v0_synth(tmp_path)
    FakeNfl(fail={"load_depth_charts": None}).install(monkeypatch)
    result, out = _cli("upgrade", "--db", path)
    assert result.exit_code == 1, out
    assert "depth_charts_season" in out
    assert _versions(path) == []


def test_cli_upgrade_newer_exits_1_and_changes_nothing(tmp_path, monkeypatch):
    path = build_db(tmp_path / "n.duckdb", derived="v0", versions=(2,))
    fake = FakeNfl().install(monkeypatch)
    result, out = _cli("upgrade", "--db", path)
    assert result.exit_code == 1, out
    assert fake.calls == [] and _versions(path) == [2]


def _read_only_or_skip(path):
    path.chmod(0o444)
    if os.access(path, os.W_OK):
        pytest.skip("running as a user that can write a 0444 file")


def test_cli_upgrade_read_only_file_is_refused(tmp_path, monkeypatch):
    path = v0_synth(tmp_path)
    _read_only_or_skip(path)
    fake = FakeNfl().install(monkeypatch)
    before = (path.stat().st_mtime_ns, path.stat().st_size)
    result, out = _cli("upgrade", "--db", path)
    assert result.exit_code == 1, out
    assert str(path) in out
    assert "read-only" in out and "update_db.sh" in out, out
    assert (path.stat().st_mtime_ns, path.stat().st_size) == before
    assert fake.calls == [] and not Path(f"{path}.wal").exists()


def test_cli_upgrade_refuses_a_file_with_no_write_bits_even_when_access_says_yes(
        tmp_path, monkeypatch):
    # Root passes os.access(W_OK) on a 0444 file; the mode bits must still refuse it.
    path = v0_synth(tmp_path)
    path.chmod(0o444)
    monkeypatch.setattr(os, "access", lambda *a, **k: True)
    fake = FakeNfl().install(monkeypatch)
    before = (path.stat().st_mtime_ns, path.stat().st_size)
    result, out = _cli("upgrade", "--db", path)
    assert result.exit_code == 1, out
    assert str(path) in out and "read-only" in out and "update_db.sh" in out
    assert (path.stat().st_mtime_ns, path.stat().st_size) == before
    assert fake.calls == [] and not Path(f"{path}.wal").exists()


def test_cli_upgrade_refuses_a_symlink_to_a_read_only_slot(tmp_path, monkeypatch):
    # The served path is a symlink; the slot behind it is the file whose mode counts.
    slot = v0_synth(tmp_path)
    slot.chmod(0o444)
    link = tmp_path / "active.duckdb"
    link.symlink_to(slot.name)
    monkeypatch.setattr(os, "access", lambda *a, **k: True)
    result, out = _cli("upgrade", "--db", link)
    assert result.exit_code == 1 and "read-only" in out, out
    assert _versions(slot) == []


@pytest.mark.integration
def test_cli_check_v053_exits_3(v053_copy):
    result, out = _cli("upgrade", "--check", "--db", v053_copy)
    assert result.exit_code == 3, out
    assert "NEEDED" in _line(out, "derived_tables")
    assert "NEEDED (network)" in _line(out, "depth_charts_season")
    assert re.search(r"\bok\b", _line(out, "raw_columns"), re.IGNORECASE)


@pytest.mark.integration
def test_cli_upgrade_refuses_read_only_v053_copy(v053_copy, monkeypatch):
    if os.access(v053_copy, os.W_OK):
        pytest.skip("running as a user that can write a 0444 file")
    FakeNfl().install(monkeypatch)
    before = (v053_copy.stat().st_mtime_ns, v053_copy.stat().st_size)
    result, out = _cli("upgrade", "--db", v053_copy)
    assert result.exit_code == 1, out
    assert str(v053_copy) in out and "read-only" in out and "update_db.sh" in out
    assert (v053_copy.stat().st_mtime_ns, v053_copy.stat().st_size) == before


# ═══ 6. serve and doctor ═══════════════════════════════════════════════════════


def _serve(monkeypatch, caplog, db_path):
    """Run serve with uvicorn.run patched. CORS origins are set so create_app logs no
    warning of its own: every warning collected comes from the startup check."""
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(db_path))
    monkeypatch.setenv("NFL_MCP_CORS_ORIGINS", "http://localhost")
    monkeypatch.delenv("NFL_MCP_AUTH_TOKEN", raising=False)
    started = []
    monkeypatch.setattr("nfl_mcp.cli.uvicorn.run", lambda app, host, port: started.append(port))
    with caplog.at_level(logging.WARNING):
        result, out = _cli("serve")
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    warnings += [line for line in out.splitlines() if "warning" in line.lower()]
    return result, started, warnings


def test_serve_old_database_logs_one_upgrade_warning(tmp_path, monkeypatch, caplog):
    path = v0_synth(tmp_path)
    path.chmod(0o444)   # serve checks read-only
    result, started, warnings = _serve(monkeypatch, caplog, path)
    assert result.exit_code == 0 and started == [8000]
    hits = [w for w in warnings if "nfl-mcp upgrade" in w]
    assert len(hits) == 1, warnings
    for name in STEPS:
        assert name in hits[0], hits[0]


def test_serve_current_database_logs_no_warning(tmp_path, monkeypatch, caplog):
    # Holds today (no check yet) and must hold after it: a current database is quiet.
    result, started, warnings = _serve(monkeypatch, caplog, current_synth(tmp_path))
    assert result.exit_code == 0 and started == [8000]
    assert warnings == []


def test_serve_newer_database_warns_and_starts(tmp_path, monkeypatch, caplog):
    result, started, warnings = _serve(monkeypatch, caplog, current_synth(tmp_path, versions=(2,)))
    assert result.exit_code == 0 and started == [8000]
    assert [w for w in warnings if "newer" in w.lower()], warnings


def test_serve_continues_when_the_check_fails(tmp_path, monkeypatch, caplog):
    # A file that exists but is not a DuckDB database: the check raises, serve warns.
    path = tmp_path / "corrupt.duckdb"
    path.write_bytes(b"not a duckdb file\n" * 64)
    result, started, warnings = _serve(monkeypatch, caplog, path)
    assert result.exit_code == 0 and started == [8000]
    assert warnings, "no warning logged for the failed check"


def test_serve_continues_when_check_raises(tmp_path, monkeypatch, caplog):
    # Patches the module attribute: serve calls migrate.check through the module.
    migrate = _migrate()

    def boom(conn):
        raise RuntimeError("simulated check failure")
    monkeypatch.setattr(migrate, "check", boom)
    result, started, warnings = _serve(monkeypatch, caplog, v0_synth(tmp_path))
    assert result.exit_code == 0 and started == [8000]
    assert warnings


def _doctor(monkeypatch, tmp_path, db_path):
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(db_path))
    monkeypatch.setattr("nfl_mcp.config.config_exists", lambda: True)
    monkeypatch.setattr("nfl_mcp.cli._claude_desktop_config_path", lambda: None)
    monkeypatch.setattr("nfl_mcp.cli.Path.cwd", lambda: tmp_path)
    result, out = _cli("doctor")
    assert result.exit_code == 0, out
    lines = [line for line in out.splitlines() if re.search(r"\bschema\b", line, re.IGNORECASE)]
    assert len(lines) == 1, out
    return lines[0]


def test_doctor_shows_old_schema_version_and_needed_steps(tmp_path, monkeypatch):
    line = _doctor(monkeypatch, tmp_path, v0_synth(tmp_path))
    assert _has_word(line, 0) and _has_word(line, 1), line
    for name in STEPS:
        assert name in line, line


def test_doctor_shows_current_schema_version(tmp_path, monkeypatch):
    line = _doctor(monkeypatch, tmp_path, current_synth(tmp_path))
    assert _has_word(line, 1), line
    assert not any(name in line for name in STEPS), line


# ═══ 7. validate schema_version ════════════════════════════════════════════════


def _schema_check(path):
    from nfl_mcp.validate import run_checks
    hits = [r for r in run_checks(path, None) if r.name == "schema_version"]
    assert len(hits) == 1
    return hits[0]


@pytest.mark.parametrize("versions", [None, ()])
def test_validate_schema_version_fails_for_version_0(tmp_path, versions):
    r = _schema_check(current_synth(tmp_path, versions=versions))
    assert not r.ok
    assert _has_word(r.detail, 0) and _has_word(r.detail, 1) and "nfl-mcp upgrade" in r.detail


def test_validate_schema_version_fails_for_newer(tmp_path):
    r = _schema_check(current_synth(tmp_path, versions=(2,)))
    assert not r.ok
    assert _has_word(r.detail, 2) and _has_word(r.detail, 1) and "nfl-mcp upgrade" in r.detail


def test_validate_schema_version_passes_for_1(tmp_path):
    from nfl_mcp.validate import run_checks
    path = current_synth(tmp_path)
    assert _schema_check(path).ok
    assert [f"{r.name} {r.detail}" for r in run_checks(path, path) if not r.ok] == []


def test_cli_validate_exit_codes_follow_schema_version(tmp_path):
    old, out_old = _cli("validate", "--db", current_synth(tmp_path, "v0.duckdb", versions=None))
    assert old.exit_code == 1 and "schema_version" in out_old, out_old
    new, out_new = _cli("validate", "--db", current_synth(tmp_path, "v1.duckdb"))
    assert new.exit_code == 0, out_new


# ═══ 8. update_db.sh ═══════════════════════════════════════════════════════════


def _lines(name):
    return (ROOT / name).read_text().splitlines()


def _find(lines, needle, start=0):
    for i in range(start, len(lines)):
        if needle in lines[i] and not lines[i].lstrip().startswith("#"):
            return i
    return -1


def _update_positions():
    lines = _lines("update_db.sh")

    def chmod(pattern):
        return next((i for i, line in enumerate(lines) if re.search(pattern, line)
                     and "$TARGET" in line and not line.lstrip().startswith("#")), -1)
    return lines, {
        "cp": _find(lines, 'cp "$DB_DIR/$CURRENT" "$TARGET"'),
        "chmod_w": chmod(r"chmod\s+u\+w\s"),
        "ingest": _find(lines, "nfl-mcp ingest"),
        "upgrade": _find(lines, "nfl-mcp upgrade"),
        "validate": _find(lines, "nfl-mcp validate"),
        "chmod_ro": chmod(r"chmod\s+a-w\s"),
        "swap": _find(lines, "ln -sfn"),
    }


def test_update_db_upgrades_between_ingest_and_validate():
    _, pos = _update_positions()
    assert pos["upgrade"] != -1, "nfl-mcp upgrade not called"
    assert pos["chmod_w"] < pos["ingest"] < pos["upgrade"] < pos["validate"] < pos["chmod_ro"] < pos["swap"]


def test_update_db_upgrade_targets_the_idle_slot_with_fetch():
    lines, pos = _update_positions()
    assert pos["upgrade"] != -1
    call = lines[pos["upgrade"]]
    assert re.search(r'--db\s+"\$TARGET"', call) or 'NFL_MCP_DB_PATH="$TARGET"' in call, call
    assert "--check" not in call and "--no-fetch" not in call, call
    assert "CURRENT" not in call and "SYMLINK" not in call, call


def test_update_db_upgrade_failure_logs_and_exits_before_validate():
    lines, pos = _update_positions()
    u, v = pos["upgrade"], pos["validate"]
    assert u != -1
    call = lines[u]
    assert not re.search(r"\|\|\s*(true|:)\s*$", call), call
    assert re.match(r"\s*if\s+!\s", call) or "||" in call or "$?" in "\n".join(lines[u + 1:u + 3]), call
    block = "\n".join(lines[u:v])
    assert re.search(r"\bexit\s+[1-9]", block), block
    assert "echo" in block, block


# ── kept (no marker) ──


def test_update_db_slot_order_unchanged():
    text = (ROOT / "update_db.sh").read_text()
    assert 'CURRENT=$(readlink "$SYMLINK")' in text
    assert 'if [ "$CURRENT" = "nflread_a.duckdb" ]; then' in text
    assert text.index('TARGET="$DB_DIR/nflread_b.duckdb"') < text.index('TARGET="$DB_DIR/nflread_a.duckdb"')


def test_update_db_chmod_lines_unchanged():
    lines, pos = _update_positions()
    assert lines[pos["chmod_w"]].strip() == 'chmod u+w "$TARGET"'
    assert lines[pos["chmod_ro"]].strip() == 'chmod a-w "$TARGET"'
    assert pos["cp"] < pos["chmod_w"] < pos["ingest"]
    assert pos["validate"] < pos["chmod_ro"] < pos["swap"]


def test_update_db_single_ln_sfn_swap_unchanged():
    lines = _lines("update_db.sh")
    swaps = [line.strip() for line in lines if "ln -" in line and not line.lstrip().startswith("#")]
    assert swaps == ['ln -sfn "$(basename "$TARGET")" "$SYMLINK"']


@pytest.mark.parametrize("cron", [
    "0 6 * * 4     /update_db.sh >> /var/log/nflmcp/nfl_update.log 2>&1",
    "0 3 * * 0,1,3 /update_db.sh >> /var/log/nflmcp/nfl_update.log 2>&1",
])
def test_entrypoint_cron_schedule_unchanged(cron):
    assert cron in _lines("entrypoint.sh")


def test_entrypoint_still_ends_with_exec_serve():
    lines = [line for line in _lines("entrypoint.sh") if line.strip() and not line.lstrip().startswith("#")]
    assert lines[-1].strip() == "exec nfl-mcp serve --host 0.0.0.0 --port 8000"


# ═══ 9. entrypoint.sh text ═════════════════════════════════════════════════════

LINK_A = 'ln -sfn "$(basename "$SLOT_A")" "$SYMLINK"'


def _entry_positions():
    lines = _lines("entrypoint.sh")
    bg = [i for i, line in enumerate(lines)
          if "update_db.sh" in line and line.rstrip().endswith("&")
          and not line.lstrip().startswith("#") and not re.match(r"\s*[0-9*]", line)]
    return lines, {
        "link": _find(lines, LINK_A),
        "check": _find(lines, "nfl-mcp upgrade --check"),
        "bg": bg,
        "serve": _find(lines, "exec nfl-mcp serve"),
    }


def test_entrypoint_checks_the_active_link_before_serve():
    lines, pos = _entry_positions()
    assert pos["check"] != -1, "nfl-mcp upgrade --check not called"
    assert pos["link"] < pos["check"] < pos["serve"]
    assert re.search(r'--db\s+"\$SYMLINK"', lines[pos["check"]]), lines[pos["check"]]


def test_entrypoint_check_has_the_opt_out():
    lines, pos = _entry_positions()
    assert pos["check"] != -1
    window = "\n".join(lines[pos["link"]:pos["check"] + 1])
    assert "NFL_MCP_UPGRADE_ON_START" in window, window
    assert re.search(r'"0"|=\s*0\b', window), window


def test_entrypoint_starts_one_background_update_after_the_check():
    lines, pos = _entry_positions()
    assert pos["check"] != -1 and len(pos["bg"]) == 1, pos
    bg = pos["bg"][0]
    assert pos["check"] < bg < pos["serve"]
    assert ">> /var/log/nflmcp/nfl_update.log" in lines[bg] and "2>&1" in lines[bg], lines[bg]


def test_entrypoint_check_cannot_fail_the_start():
    lines, pos = _entry_positions()
    assert pos["check"] != -1
    call = lines[pos["check"]]
    guarded = ("||" in call or re.match(r"\s*(if|elif|case)\b", call) or "$(" in call
               or any(re.match(r"\s*set\s+\+e", line) for line in lines[pos["link"]:pos["check"]]))
    assert guarded, call


# ═══ 10. script lifecycle with nfl-mcp stubbed ═════════════════════════════════

# Every call is appended to $STUB_CALLS as "<args>|<NFL_MCP_DB_PATH>". The stub prints
# nothing, so any "upgrade" in the script output comes from the script itself.
_STUB = r'''#!/bin/sh
printf '%s|%s\n' "$*" "${NFL_MCP_DB_PATH:-}" >> "$STUB_CALLS"
case "${1:-}" in
    validate) exit "${STUB_VALIDATE_EXIT:-0}" ;;
    upgrade)
        case "$*" in *--check*) exit "${STUB_CHECK_EXIT:-0}" ;; esac
        exit "${STUB_UPGRADE_EXIT:-0}" ;;
    ingest)
        if [ -n "${NFL_MCP_DB_PATH:-}" ]; then printf INGESTED > "$NFL_MCP_DB_PATH"; fi ;;
esac
exit 0
'''


def _sim(tmp_path):
    data, seed, bin_dir, log = (tmp_path / n for n in ("data", "seed", "bin", "log"))
    for d in (data, seed, bin_dir, log):
        d.mkdir(parents=True)
    (seed / "nflread.duckdb").write_text("SEED_V1")
    (bin_dir / "nfl-mcp").write_text(_STUB)
    (bin_dir / "supercronic").write_text("#!/bin/sh\nexit 0\n")
    bg = tmp_path / "bg_update.sh"
    bg.write_text('#!/bin/sh\necho "$$" >> "$UPDATER_MARK"\n')
    for f in (bin_dir / "nfl-mcp", bin_dir / "supercronic", bg):
        f.chmod(0o755)

    entry = (ROOT / "entrypoint.sh").read_text()
    for before, after in (('DB_DIR="/data"', f'DB_DIR="{data}"'),
                          ('SEED="/seed/nflread.duckdb"', f'SEED="{seed}/nflread.duckdb"'),
                          ("exec nfl-mcp serve --host 0.0.0.0 --port 8000", 'echo "[serve started]"'),
                          ("/var/log/nflmcp", str(log)),
                          ("/update_db.sh", str(bg))):
        assert before in entry, f"entrypoint.sh no longer contains: {before}"
        entry = entry.replace(before, after)
    updater = (ROOT / "update_db.sh").read_text().replace('DB_DIR="/data"', f'DB_DIR="{data}"')
    (tmp_path / "entrypoint.sh").write_text(entry)
    (tmp_path / "update_db.sh").write_text(updater)
    return {"entry": tmp_path / "entrypoint.sh", "update": tmp_path / "update_db.sh",
            "data": data, "bin": bin_dir, "calls": tmp_path / "calls.txt",
            "mark": tmp_path / "updater_started.txt"}


def _run_sh(sim, script, **extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("NFL_MCP_DB_PATH", "NFL_MCP_UPGRADE_ON_START")}
    env.update(PATH=f"{sim['bin']}{os.pathsep}{os.environ['PATH']}", STUB_CALLS=str(sim["calls"]),
               UPDATER_MARK=str(sim["mark"]), **extra)
    return subprocess.run(["sh", str(sim[script])], capture_output=True, text=True, env=env, check=False,
                          timeout=60)


def _calls(sim):
    if not sim["calls"].exists():
        return []
    return [line.split("|", 1) for line in sim["calls"].read_text().splitlines()]


def _active_slots(sim):
    data = sim["data"]
    (data / "nflread_a.duckdb").write_text("SEED_V1")
    os.symlink("nflread_a.duckdb", data / "nflread.duckdb")


def _wait_for(path, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return path.exists()


def test_update_db_runs_upgrade_on_target_between_ingest_and_validate(tmp_path):
    sim = _sim(tmp_path)
    _active_slots(sim)
    result = _run_sh(sim, "update")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = _calls(sim)
    names = [args.split()[0] for args, _ in calls]
    assert "upgrade" in names, calls
    assert names.index("ingest") < names.index("upgrade") < names.index("validate")
    args, _ = calls[names.index("upgrade")]
    assert f"--db {sim['data'] / 'nflread_b.duckdb'}" in args and "--check" not in args, args
    assert os.readlink(sim["data"] / "nflread.duckdb") == "nflread_b.duckdb"


@pytest.mark.parametrize("code", ["1", "3"])
def test_update_db_upgrade_failure_keeps_active_slot(tmp_path, code):
    sim = _sim(tmp_path)
    _active_slots(sim)
    result = _run_sh(sim, "update", STUB_UPGRADE_EXIT=code)
    assert result.returncode != 0
    assert os.readlink(sim["data"] / "nflread.duckdb") == "nflread_a.duckdb"
    assert (sim["data"] / "nflread_a.duckdb").read_text() == "SEED_V1"
    assert "validate" not in [args.split()[0] for args, _ in _calls(sim)]
    assert [line for line in result.stdout.splitlines()
            if re.search(r"upgrade", line, re.IGNORECASE) and re.search(r"ERROR|fail", line, re.IGNORECASE)], result.stdout


def test_entrypoint_old_database_starts_background_update_and_serves(tmp_path):
    sim = _sim(tmp_path)
    result = _run_sh(sim, "entry", STUB_CHECK_EXIT="3")
    assert result.returncode == 0, result.stderr
    assert "[serve started]" in result.stdout
    checks = [args for args, _ in _calls(sim) if args.startswith("upgrade") and "--check" in args]
    assert len(checks) == 1 and f"--db {sim['data'] / 'nflread.duckdb'}" in checks[0], checks
    assert _wait_for(sim["mark"], 10), "update_db.sh was not started"
    time.sleep(0.3)
    assert len(sim["mark"].read_text().splitlines()) == 1   # started once


def test_entrypoint_current_database_starts_no_update(tmp_path):
    sim = _sim(tmp_path)
    result = _run_sh(sim, "entry", STUB_CHECK_EXIT="0")
    assert result.returncode == 0 and "[serve started]" in result.stdout
    assert [a for a, _ in _calls(sim) if a.startswith("upgrade --check")], _calls(sim)
    assert not _wait_for(sim["mark"], 1)


def test_entrypoint_check_error_does_not_fail_the_start(tmp_path):
    sim = _sim(tmp_path)
    result = _run_sh(sim, "entry", STUB_CHECK_EXIT="1")
    assert result.returncode == 0, result.stderr
    assert "[serve started]" in result.stdout
    assert [a for a, _ in _calls(sim) if a.startswith("upgrade --check")], _calls(sim)


def test_entrypoint_opt_out_skips_the_check(tmp_path):
    sim = _sim(tmp_path)
    result = _run_sh(sim, "entry", STUB_CHECK_EXIT="3", NFL_MCP_UPGRADE_ON_START="0")
    assert result.returncode == 0 and "[serve started]" in result.stdout
    assert not [a for a, _ in _calls(sim) if a.startswith("upgrade")]
    assert not _wait_for(sim["mark"], 1)
    # The same start without the opt-out does run the check (so the test above can fail).
    sim2 = _sim(tmp_path / "again")
    _run_sh(sim2, "entry", STUB_CHECK_EXIT="0")
    assert [a for a, _ in _calls(sim2) if a.startswith("upgrade --check")]
