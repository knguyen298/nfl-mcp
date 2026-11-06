"""Group P: validate lets an upgrade from an older schema version shrink the tables it rewrites.

Production log, 2026-11-05 (active slot built by v0.4.0, schema version 0):
    FAIL row_count_floor: depth_charts has 2173176 rows, current slot has 2911328;
    injury_return_curve has 1172 rows, current slot has 1215
    ERROR: validation failed. Keeping nflread_b.duckdb active.
Both drops are the one-time effect of the version 0 -> 1 upgrade, measured on the real data:
v0.4.0 appended an extra copy of the 2025+ depth_charts rows on every refresh (old slot
2,911,328 rows, 1,863,032 distinct; new slot 1,867,684 distinct), and injury_return_curve is
a derived table whose definition changed (snap counts joined by pfr_player_id, 43 fewer rows).

Expected outcomes come from the requirement, not from nfl_mcp:
- The upgrade rewrites every derived table (DERIVED below, the tables
  rebuild_derived_tables writes) and depth_charts. When the current slot's schema version
  is below the new slot's, a shrink in one of them is named in the row_count_floor detail
  and does not fail the check.
- Every other table (plays, rosters, injuries, snap_counts, participation and the rest)
  still fails on a shrink beyond the 1 percent tolerance, across an upgrade too.
- Both slots at the same version: the check behaves as before, and the failing detail is
  the production line above, word for word.
- A current slot whose version cannot be read (a non-integer value, or a query error) gets
  no allowance: the shrink fails and the detail says the version cannot be read. A current
  slot newer than the new one gets no allowance either.
- A slot with no _schema_version table, or an empty one, is version 0.
"""

import re
from pathlib import Path

import duckdb
import pytest
from click.testing import CliRunner

from nfl_mcp.cli import main
from nfl_mcp.required_columns import REQUIRED_COLUMNS
from nfl_mcp.validate import run_checks

DERIVED = ["team_offense_stats", "team_defense_stats", "situational_stats", "team_records",
           "player_td_luck", "player_role_trend", "player_separation_opportunity",
           "player_drop_rate", "player_contract_value", "injury_return_curve"]
REWRITTEN = DERIVED + ["depth_charts"]
RAW = ["plays", "schedules", "rosters", "injuries", "snap_counts", "players", "player_stats",
       "ff_opportunity", "ftn_charting", "participation"]
# Tables outside REQUIRED_COLUMNS that the slots also hold.
EXTRA = {"depth_charts": ["season", "club_code", "gsis_id"],
         "participation": ["nflverse_game_id", "play_id", "possession_team"]}

CUR = 3  # migrate.SCHEMA_VERSION since v0.7.1 (points in team_records); 2 was v0.7.0, 1 was v0.6.x (valid old slots)
PRODUCTION_NEW = {"depth_charts": 2173176, "injury_return_curve": 1172}
PRODUCTION_CURRENT = {"depth_charts": 2911328, "injury_return_curve": 1215}
PRODUCTION_FAIL = ("depth_charts has 2173176 rows, current slot has 2911328; "
                   "injury_return_curve has 1172 rows, current slot has 1215")

# ── synthetic slot builder ─────────────────────────────────────────────────────

_EXPR = {"game_id": "'2025_09_KC_BUF'", "play_id": "CAST(range AS DOUBLE)", "season": "2025",
         "week": "9", "epa": "0.1", "posteam": "'KC'"}


def build_slot(path, *, version, rows=None):
    """Write a slot with every REQUIRED_COLUMNS table plus EXTRA, rows.get(table, 100) rows
    each (plays unique on (game_id, play_id), all in 2025 week 9).

    version: an int stamps _schema_version; None leaves the table out; () makes it empty;
    "text" stores a VARCHAR version; "error" makes reading it raise a conversion error."""
    rows = rows or {}
    con = duckdb.connect(str(path))
    try:
        for table, cols in {**REQUIRED_COLUMNS, **EXTRA}.items():
            select = ", ".join(f'{_EXPR.get(c, repr(f"{c}_x"))} AS "{c}"' for c in cols)
            con.execute(f'CREATE TABLE "{table}" AS SELECT {select} FROM range({rows.get(table, 100)})')
        if version == "error":
            con.execute("CREATE VIEW _schema_version AS SELECT CAST('one' AS INTEGER) AS version")
        elif version == "text":
            con.execute("CREATE TABLE _schema_version (version VARCHAR)")
            con.execute("INSERT INTO _schema_version VALUES ('one')")
        elif version is not None:
            con.execute("CREATE TABLE _schema_version "
                        "(version INTEGER, applied_at TIMESTAMP, nfl_mcp_version VARCHAR)")
            for v in ([] if version == () else [version]):
                con.execute("INSERT INTO _schema_version VALUES (?, TIMESTAMP '2026-11-05', 'test')", [v])
    finally:
        con.close()
    return Path(path)


@pytest.fixture
def pair(tmp_path):
    """pair(new_version, current_version, new_rows, current_rows) -> (new, current)."""
    made = []

    def factory(new_version, current_version, new_rows=None, current_rows=None):
        made.append(root := tmp_path / f"pair{len(made)}")
        root.mkdir()
        return (build_slot(root / "new.duckdb", version=new_version, rows=new_rows),
                build_slot(root / "current.duckdb", version=current_version, rows=current_rows))
    return factory


@pytest.fixture(scope="module")
def production(tmp_path_factory):
    """The production shape: current slot at version 0 (no _schema_version, as v0.4.0
    left it), new slot at the current version, with the production row counts."""
    root = tmp_path_factory.mktemp("production")
    return (build_slot(root / "new.duckdb", version=CUR, rows=PRODUCTION_NEW),
            build_slot(root / "current.duckdb", version=None, rows=PRODUCTION_CURRENT))


def floor(new, current):
    hits = [r for r in run_checks(new, current) if r.name == "row_count_floor"]
    assert len(hits) == 1
    return hits[0]


def failures(new, current):
    return [f"{r.name} {r.detail}" for r in run_checks(new, current) if not r.ok]


def has_word(text, word):
    return re.search(rf"\b{re.escape(str(word))}\b", text) is not None


def cli(*args):
    result = CliRunner().invoke(main, ["validate", *map(str, args)])
    return result.exit_code, result.output


# ── 1. the production case ─────────────────────────────────────────────────────


def test_production_upgrade_passes_every_check(production):
    assert failures(*production) == []


def test_production_upgrade_names_both_shrunk_tables_in_detail(production):
    detail = floor(*production).detail
    assert ("depth_charts has 2173176 rows, current slot has 2911328 (allowed: the current "
            f"slot is schema version 0, and the upgrade to {CUR} rewrote this table)") in detail
    assert ("injury_return_curve has 1172 rows, current slot has 1215 (allowed: the current "
            f"slot is schema version 0, and the upgrade to {CUR} rewrote this table)") in detail


def test_production_upgrade_cli_exits_zero(production):
    new, current = production
    code, out = cli("--db", new, "--current", current)
    assert code == 0, out
    assert "ok   row_count_floor: depth_charts has 2173176 rows" in out
    assert "Validation passed" in out


def test_production_shrink_at_same_version_fails_as_before(pair):
    result = floor(*pair(CUR, CUR, PRODUCTION_NEW, PRODUCTION_CURRENT))
    assert not result.ok
    assert result.detail == PRODUCTION_FAIL


def test_production_shrink_at_same_version_cli_exits_one(pair):
    new, current = pair(CUR, CUR, PRODUCTION_NEW, PRODUCTION_CURRENT)
    code, out = cli("--db", new, "--current", current)
    assert code == 1
    assert f"FAIL row_count_floor: {PRODUCTION_FAIL}" in out
    assert "Validation failed: 1 of 7 checks." in out


# ── 2. which tables the allowance covers ───────────────────────────────────────


@pytest.mark.parametrize("current_version", [None, (), 0, 1])
@pytest.mark.parametrize("table", REWRITTEN)
def test_rewritten_table_may_shrink_across_upgrade(pair, table, current_version):
    result = floor(*pair(CUR, current_version, {table: 50}))
    assert result.ok, result.detail
    assert f"{table} has 50 rows, current slot has 100 (allowed" in result.detail


@pytest.mark.parametrize("table", REWRITTEN)
def test_rewritten_table_shrink_at_same_version_fails(pair, table):
    result = floor(*pair(CUR, CUR, {table: 50}))
    assert not result.ok
    assert result.detail == f"{table} has 50 rows, current slot has 100"


@pytest.mark.parametrize("table", RAW)
def test_raw_table_shrink_across_upgrade_still_fails(pair, table):
    result = floor(*pair(CUR, None, {table: 50}))
    assert not result.ok
    assert result.detail == f"{table} has 50 rows, current slot has 100"


@pytest.mark.parametrize("table", ["plays", "rosters", "participation"])
def test_raw_table_shrink_across_upgrade_cli_exits_one(pair, table):
    new, current = pair(CUR, None, {table: 50, "depth_charts": 50})
    code, out = cli("--db", new, "--current", current)
    assert code == 1
    line = next(x for x in out.splitlines() if x.startswith("FAIL row_count_floor:"))
    assert has_word(line, table) and has_word(line, 50) and has_word(line, 100)
    # The allowed shrink is still reported next to the failing one.
    assert "depth_charts has 50 rows, current slot has 100 (allowed" in line


def test_raw_shrink_within_tolerance_across_upgrade_ok(pair):
    result = floor(*pair(CUR, None, {"plays": 99, "rosters": 99}))
    assert result.ok, result.detail
    assert "allowed" not in result.detail


def test_upgrade_without_shrink_keeps_the_usual_detail(pair):
    upgraded, same = floor(*pair(CUR, None)), floor(*pair(CUR, CUR))
    assert upgraded.ok and same.ok
    assert upgraded.detail == same.detail
    assert "allowed" not in same.detail


def test_empty_snapshot_table_still_fails_across_upgrade(pair):
    result = floor(*pair(CUR, None, {"ff_rankings_week": 0}))
    assert not result.ok
    assert has_word(result.detail, "ff_rankings_week")


# ── 3. versions that give no allowance ─────────────────────────────────────────


@pytest.mark.parametrize("current_version", ["text", "error"])
def test_unreadable_current_version_gets_no_allowance(pair, current_version):
    result = floor(*pair(CUR, current_version, {"depth_charts": 50}))
    assert not result.ok
    assert "depth_charts has 50 rows, current slot has 100" in result.detail
    assert "cannot read the current slot's schema version" in result.detail


@pytest.mark.parametrize("current_version", ["text", "error"])
def test_unreadable_current_version_cli_exits_one(pair, current_version):
    new, current = pair(CUR, current_version, {"injury_return_curve": 50})
    code, out = cli("--db", new, "--current", current)
    assert code == 1
    assert "FAIL row_count_floor: injury_return_curve has 50 rows" in out


@pytest.mark.parametrize("current_version", ["text", "error"])
def test_unreadable_current_version_without_shrink_ok(pair, current_version):
    assert floor(*pair(CUR, current_version)).ok


def test_unreadable_current_version_raw_shrink_has_plain_detail(pair):
    result = floor(*pair(CUR, "text", {"rosters": 50}))
    assert not result.ok
    assert result.detail == "rosters has 50 rows, current slot has 100"


def test_current_newer_than_new_gets_no_allowance(pair):
    result = floor(*pair(CUR, CUR + 1, {"depth_charts": 50}))
    assert not result.ok
    assert result.detail == "depth_charts has 50 rows, current slot has 100"

