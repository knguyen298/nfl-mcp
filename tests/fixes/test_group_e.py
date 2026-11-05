"""Group E: ingest validation and swap safety.

Spec: PLAN_FIX_GROUPS_MAIN_2026-11-04.md section E; evidence: AUDIT_MAIN_2026-11-04.md
(update_db.sh:42-48 has no check between ingest and swap; no chmod a-w; entrypoint.sh:35
relinks slot A on every start).

Interface contract (shared contract plus the choices made here)
- nfl_mcp.required_columns.REQUIRED_COLUMNS: dict[str, list[str]] (table -> columns).
  Covers every table the tools read (FROM/JOIN in tools.py plus the ff_rankings_* tables
  picked by literal), _ingest_metadata, players, player_stats, and the derived team tables
  team_offense_stats, team_defense_stats, situational_stats, team_records.
- nfl_mcp.validate.run_checks(new_db: Path, current_db: Path | None = None) -> list of
  results with .name (str), .ok (bool), .detail (str). It never raises on a bad slot: a
  problem is a failing result. It opens both files read-only (the current slot is a-w).
- Tests locate a failure by text: a failing result whose f"{name} {detail}" contains the
  expected words. Suggested names: required_tables, required_columns, row_count_floor,
  duplicate_plays, newest_week, null_keys.
- Row-count floor: for every table present in BOTH slots except the snapshot tables
  (SNAPSHOT_TABLES: ff_rankings_draft, ff_rankings_week, which need only be non-empty),
  new rows >= current rows * 0.99 (1 percent tolerance). Tests use 199/200 (ok) and
  190/200 (fail), so any tolerance in [0.5%, 5%) passes. Failure detail names the table and both counts. A table absent from
  the current slot is not checked. current_db=None -> floor check absent or ok.
- Duplicate check on plays (game_id, play_id). Count = COUNT(*) - COUNT(DISTINCT key), i.e.
  the rows that would have to be removed; the detail contains that number.
- Newest week is measured on plays as the max (season, week) pair, compared as a pair (2026
  week 1 is newer than 2025 week 22). New older than current -> fail. current_db=None ->
  check absent or ok.
- Null keys: in the newest (season, week) of the new plays, epa and posteam must each have
  at least one non-null value; the detail names the all-null column. If season or week is
  NULL in every plays row, some check fails and names "season" (newest week cannot be found).
- CLI: `nfl-mcp validate --db PATH [--current PATH]`, exit 0 when all ok, 1 when any check
  fails, and the failing check text is printed. Missing --db file -> non-zero exit.
- update_db.sh: `nfl-mcp validate --db "$TARGET" --current <active slot>` sits between the
  ingest line and the `ln -sfn` swap. Accepted guards: (a) `if ! ... nfl-mcp validate ...`
  with a non-zero `exit` before the matching `fi` and before the swap; (b) `... || exit N`
  or `|| { ...; exit N; }`; (c) `if nfl-mcp validate ...; then` with the swap inside the
  then-branch; (d) `set -e` before it and a plain call (no ||, &&, pipe, `!`,
  or `if`). `chmod a-w "$TARGET"` (or ugo-w / -w / 444) after ingest and before the swap.
  A `chmod u+w "$TARGET"` (any +w or 6xx mode) after the `cp` into the target and before
  the ingest: cp copies the a-w mode of the active slot, so the copy is read-only.
- entrypoint.sh: the `ln -sfn` to slot A runs only when "$SYMLINK" is not already a link:
  `[ ! -L "$SYMLINK" ]` (or ! -e / ! -h) on the line before, on the same line with &&, as
  `[ -L "$SYMLINK" ] ||`, or as the condition of the enclosing `if`/`elif`.
- Per the brief the scripts are only read as text, never executed.
"""

import importlib
import os
import re
import shutil
from pathlib import Path

import duckdb
import pytest
from click.testing import CliRunner

from nfl_mcp.cli import main

ROOT = Path(__file__).resolve().parents[2]
REAL_DB = Path(os.environ.get("NFL_MCP_DB_PATH", "/tmp/grpE_v3.duckdb"))

# ── synthetic slot builder ─────────────────────────────────────────────────────

# Minimal columns per table, taken from the SQL in nfl_mcp/tools.py. The builder unions in
# whatever REQUIRED_COLUMNS lists once it exists, so a "good" slot has every required column.
BASE_SCHEMA = {
    "plays": ["game_id", "play_id", "season", "week", "epa", "posteam", "season_type"],
    "schedules": ["game_id", "season", "week", "game_type", "home_team", "away_team"],
    "players": ["gsis_id", "display_name", "position"],
    "rosters": ["full_name", "position", "team", "season"],
    "snap_counts": ["season", "week", "team", "offense_snaps"],
    "player_stats": ["player_id", "season", "week", "season_type"],
    "injuries": ["season", "week", "team", "full_name"],
    "ff_opportunity": ["season", "week"],
    "ftn_charting": ["season", "week"],
    "ff_rankings_draft": ["player"],
    "ff_rankings_week": ["player_name"],
    "team_offense_stats": ["season_year", "team"],
    "team_defense_stats": ["season_year", "team"],
    "situational_stats": ["season_year", "team"],
    "team_records": ["season", "team"],
    "player_td_luck": ["season"],
    "player_role_trend": ["season"],
    "player_separation_opportunity": ["season"],
    "player_drop_rate": ["season"],
    "player_contract_value": ["season"],
    "injury_return_curve": ["injury_type", "position"],
    "_ingest_metadata": ["dataset_id", "table_name", "season", "row_count", "loaded_at", "loader_fn"],
}
TYPES = {"season": "INTEGER", "season_year": "INTEGER", "week": "INTEGER", "play_id": "DOUBLE", "epa": "DOUBLE", "row_count": "BIGINT"}
PLAY_COLS = ("game_id", "play_id", "season", "week", "epa", "posteam")


def _required_module():
    try:
        return importlib.import_module("nfl_mcp.required_columns")
    except ImportError:
        return None


def _schema():
    schema = {t: list(c) for t, c in BASE_SCHEMA.items()}
    rc = getattr(_required_module(), "REQUIRED_COLUMNS", None)
    if isinstance(rc, dict):
        for table, cols in rc.items():
            existing = schema.setdefault(table, [])
            existing.extend(c for c in cols if c not in existing)
    return schema


def make_plays(n=200, weeks=((2025, 8), (2025, 9)), epa=0.1, posteam="KC", start_id=1):
    """n plays spread round-robin over weeks; (game_id, play_id) unique."""
    rows = []
    for i in range(n):
        season, week = weeks[i % len(weeks)]
        rows.append((f"{season}_{week:02d}_KC_BUF", float(start_id + i), season, week, epa, posteam))
    return rows


def build_slot(path, plays, *, drop_tables=(), drop_columns=None, row_counts=None):
    """Write a DuckDB slot. Non-plays tables get row_counts.get(table, 3) rows stamped with
    the newest (season, week) in plays so every table agrees on recency."""
    drop_columns = drop_columns or {}
    row_counts = row_counts or {}
    keyed = [(p[2], p[3]) for p in plays if p[2] is not None and p[3] is not None]
    newest = max(keyed) if keyed else (2025, 9)
    con = duckdb.connect(str(path))
    try:
        for table, cols in _schema().items():
            if table in drop_tables:
                continue
            cols = [c for c in cols if c not in drop_columns.get(table, ())]
            ddl = ", ".join(f'"{c}" {TYPES.get(c, "VARCHAR")}' for c in cols)
            con.execute(f'CREATE TABLE "{table}" ({ddl})')
            if table == "plays":
                rows = []
                for i, p in enumerate(plays):
                    given = dict(zip(PLAY_COLS, p))
                    rows.append([given[c] if c in given else _value(c, i, newest) for c in cols])
            else:
                rows = [[_value(c, i, newest) for c in cols] for i in range(row_counts.get(table, 3))]
            if rows:
                marks = ", ".join("?" for _ in cols)
                con.executemany(f'INSERT INTO "{table}" VALUES ({marks})', rows)
    finally:
        con.close()
    return path


def _value(col, i, newest):
    if col in ("season", "season_year"):
        return newest[0]
    if col == "week":
        return newest[1]
    if col in ("season_type", "game_type"):
        return "REG"
    if col in TYPES:
        return i
    return f"{col}_{i}"


@pytest.fixture
def slots(tmp_path):
    """Return a factory: slots(name, plays, **kw) -> Path in tmp_path."""
    def factory(name, plays=None, **kw):
        return build_slot(tmp_path / f"{name}.duckdb", make_plays() if plays is None else plays, **kw)
    return factory


def run_checks(new, current=None):
    mod = importlib.import_module("nfl_mcp.validate")
    fn = getattr(mod, "run_checks", None)
    assert fn is not None, "nfl_mcp.validate.run_checks missing"
    results = list(fn(Path(new), Path(current) if current is not None else None))
    assert results, "run_checks returned no results"
    for r in results:
        assert hasattr(r, "name") and hasattr(r, "ok") and hasattr(r, "detail")
    return results


def failures(results):
    return [f"{r.name} {r.detail}" for r in results if not r.ok]


def assert_all_ok(results):
    assert failures(results) == []


def assert_fails_with(results, *words):
    hits = [t for t in failures(results) if all(re.search(rf"\b{re.escape(str(w))}\b", t) for w in words)]
    assert hits, f"no failing check mentions {words}; failures: {failures(results)}"


def invoke_validate(*args):
    result = CliRunner().invoke(main, ["validate", *map(str, args)])
    try:
        extra = result.stderr
    except (ValueError, AttributeError):
        extra = ""
    return result, (result.output or "") + (extra or "")


# ── 1. REQUIRED_COLUMNS ────────────────────────────────────────────────────────


def _required_columns():
    mod = _required_module()
    assert mod is not None, "nfl_mcp.required_columns missing"
    rc = getattr(mod, "REQUIRED_COLUMNS", None)
    assert rc is not None, "REQUIRED_COLUMNS missing"
    return rc


def test_required_columns_is_dict_of_table_to_column_list():
    rc = _required_columns()
    assert isinstance(rc, dict) and rc
    for table, cols in rc.items():
        assert isinstance(table, str)
        assert isinstance(cols, list) and cols, table
        assert all(isinstance(c, str) for c in cols), table
        assert len(cols) == len(set(cols)), f"duplicate column in {table}"


@pytest.mark.parametrize("table", [
    "plays", "schedules", "players", "rosters", "snap_counts", "player_stats", "injuries",
    "team_offense_stats", "team_defense_stats", "situational_stats", "team_records",
])
def test_required_columns_includes_table(table):
    assert table in _required_columns()


def _tables_in_tools_sql():
    src = (ROOT / "nfl_mcp" / "tools.py").read_text()
    names = set(re.findall(r"\b(?:FROM|JOIN)\s+([a-z_][a-z0-9_]*)", src))
    names |= set(re.findall(r'table\s*=\s*"([a-z_][a-z0-9_]*)"', src))
    return names - {"information_schema"}


def test_required_columns_cover_every_table_in_tools_sql():
    # Oracle: table names parsed from the tools.py source text, not from the new module.
    missing = sorted(_tables_in_tools_sql() - set(_required_columns()))
    assert missing == []


@pytest.mark.parametrize("table,column", [
    ("plays", "game_id"), ("plays", "play_id"), ("plays", "season"), ("plays", "week"),
    ("plays", "epa"), ("plays", "posteam"), ("schedules", "home_team"),
    ("schedules", "game_type"), ("rosters", "full_name"), ("snap_counts", "offense_snaps"),
    ("injuries", "full_name"),
])
def test_required_columns_include_column_tools_select(table, column):
    # Columns read straight from the SELECT/WHERE text in tools.py.
    assert column in _required_columns().get(table, [])


@pytest.fixture
def require_db():
    if not REAL_DB.exists():
        pytest.skip(f"No DuckDB database at {REAL_DB}")
    con = duckdb.connect(str(REAL_DB), read_only=True)
    try:
        if con.execute("SELECT COUNT(*) FROM plays").fetchone()[0] == 0:
            pytest.skip("plays table is empty")
    finally:
        con.close()


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
def test_every_required_column_exists_in_loaded_db():
    rc = _required_columns()
    con = duckdb.connect(str(REAL_DB), read_only=True)
    try:
        live = set(con.execute(
            "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'main'"
        ).fetchall())
    finally:
        con.close()
    missing = sorted((t, c) for t, cols in rc.items() for c in cols if (t, c) not in live)
    assert missing == []


@pytest.fixture(scope="module")
def real_copy(tmp_path_factory):
    """Read-only copy of the real DB so run_checks can never touch the shared file."""
    if not REAL_DB.exists():
        pytest.skip(f"No DuckDB database at {REAL_DB}")
    dest = tmp_path_factory.mktemp("real") / "real_copy.duckdb"
    shutil.copyfile(REAL_DB, dest)
    dest.chmod(0o444)
    return dest


@pytest.mark.integration
@pytest.mark.usefixtures("require_db")
@pytest.mark.parametrize("with_current", [False, True])
def test_run_checks_passes_on_loaded_db(real_copy, with_current):
    # The loaded DB has 0 duplicate (game_id, play_id) (audit, re-checkable) and is compared
    # with itself as the current slot, so every check must pass.
    assert_all_ok(run_checks(real_copy, real_copy if with_current else None))


# ── 2. good slot passes ────────────────────────────────────────────────────────


@pytest.mark.parametrize("with_current", [False, True])
def test_run_checks_good_slot_all_ok(slots, with_current):
    new = slots("new")
    current = slots("current") if with_current else None
    assert_all_ok(run_checks(new, current))


def test_run_checks_works_on_read_only_slots(slots):
    # The active slot is a-w after a swap; validate must open both files read_only.
    new, current = slots("new"), slots("current")
    new.chmod(0o444)
    current.chmod(0o444)
    before = (new.stat().st_mtime_ns, current.stat().st_mtime_ns)
    assert_all_ok(run_checks(new, current))
    assert (new.stat().st_mtime_ns, current.stat().st_mtime_ns) == before


@pytest.mark.parametrize("with_current", [False, True])
def test_cli_validate_good_slot_exits_zero(slots, with_current):
    args = ["--db", slots("new")] + (["--current", slots("current")] if with_current else [])
    result, out = invoke_validate(*args)
    assert result.exit_code == 0, out


# ── 3. missing table / column ──────────────────────────────────────────────────


@pytest.mark.parametrize("table", ["plays", "schedules", "players", "team_offense_stats"])
def test_missing_required_table_fails_naming_table(slots, table):
    assert_fails_with(run_checks(slots("new", drop_tables={table})), table)


@pytest.mark.parametrize("table,column", [("plays", "epa"), ("schedules", "home_team"), ("rosters", "full_name")])
def test_missing_required_column_fails_naming_table_and_column(slots, table, column):
    assert_fails_with(run_checks(slots("new", drop_columns={table: [column]})), table, column)


@pytest.mark.parametrize("kw,words", [
    ({"drop_tables": {"schedules"}}, ["schedules"]),
    ({"drop_columns": {"plays": ["posteam"]}}, ["plays", "posteam"]),
])
def test_cli_validate_bad_slot_exits_one_and_prints_failure(slots, kw, words):
    result, out = invoke_validate("--db", slots("new", **kw))
    assert result.exit_code == 1, out
    for word in words:
        assert word in out


def test_cli_validate_missing_db_file_exits_nonzero(tmp_path):
    result, out = invoke_validate("--db", tmp_path / "absent.duckdb")
    assert result.exit_code not in (0, None), out
    assert "No such command" not in out


# ── 4. row-count floor ─────────────────────────────────────────────────────────


def test_row_floor_plays_drop_beyond_tolerance_fails(slots):
    new, current = slots("new", make_plays(190)), slots("current", make_plays(200))
    # 190 < 200 * 0.99 = 198 -> fail naming plays, 190 and 200.
    assert_fails_with(run_checks(new, current), "plays", 190, 200)


def test_row_floor_other_table_drop_fails(slots):
    new = slots("new", row_counts={"player_stats": 40})
    current = slots("current", row_counts={"player_stats": 50})
    # 40 < 50 * 0.99 -> fail naming player_stats, 40 and 50.
    assert_fails_with(run_checks(new, current), "player_stats", 40, 50)


@pytest.mark.parametrize("new_n", [199, 200, 260])
def test_row_floor_within_tolerance_or_growth_ok(slots, new_n):
    # 199/200 is a 0.5 percent drop, inside the 1 percent tolerance; 200 and 260 do not drop.
    new, current = slots("new", make_plays(new_n)), slots("current", make_plays(200))
    assert_all_ok(run_checks(new, current))


@pytest.mark.parametrize("table", ["snap_counts", "team_records"])
def test_row_floor_ignores_table_missing_from_current(slots, table):
    new, current = slots("new"), slots("current", drop_tables={table})
    assert_all_ok(run_checks(new, current))


@pytest.mark.parametrize("table", ["ff_rankings_draft", "ff_rankings_week"])
def test_row_floor_snapshot_table_may_shrink_by_20_percent(slots, table):
    new = slots("new", row_counts={table: 80})
    current = slots("current", row_counts={table: 100})
    assert_all_ok(run_checks(new, current))


@pytest.mark.parametrize("table", ["ff_rankings_draft", "ff_rankings_week"])
def test_row_floor_empty_snapshot_table_fails_naming_table(slots, table):
    new = slots("new", row_counts={table: 0})
    current = slots("current", row_counts={table: 100})
    assert_fails_with(run_checks(new, current), table)


def test_row_floor_non_snapshot_table_shrinking_5_percent_fails(slots):
    new = slots("new", row_counts={"injuries": 95})
    current = slots("current", row_counts={"injuries": 100})
    assert_fails_with(run_checks(new, current), "injuries", 95, 100)


# ── 5. duplicate (game_id, play_id) ────────────────────────────────────────────


def test_duplicate_plays_three_doubled_keys_fails_with_count(slots):
    plays = make_plays(200)
    plays += plays[:3]  # 203 rows, 200 distinct keys -> 3 duplicate rows
    results = run_checks(slots("new", plays))
    assert_fails_with(results, 3)
    assert any("dup" in t.lower() for t in failures(results))


def test_duplicate_plays_tripled_key_counts_extra_rows(slots):
    plays = make_plays(200)
    plays += [plays[0], plays[0], plays[1], plays[2]]  # 204 rows, 200 distinct -> 4 extra
    assert_fails_with(run_checks(slots("new", plays)), 4)


def test_same_play_id_in_different_games_is_not_duplicate(slots):
    # play_id 1.0 in two different games is two distinct keys -> no duplicate.
    plays = make_plays(100, weeks=((2025, 9),)) + make_plays(100, weeks=((2025, 8),))
    assert_all_ok(run_checks(slots("new", plays)))


# ── 6. newest week regression ──────────────────────────────────────────────────


@pytest.mark.parametrize("new_weeks,current_weeks", [
    (((2025, 8),), ((2025, 8), (2025, 9))),       # current has week 9, new stops at 8
    (((2025, 22),), ((2026, 1),)),                 # older season, higher week number
])
def test_newest_week_older_than_current_fails(slots, new_weeks, current_weeks):
    new = slots("new", make_plays(200, weeks=new_weeks))
    current = slots("current", make_plays(200, weeks=current_weeks))
    assert_fails_with(run_checks(new, current), "week")


@pytest.mark.parametrize("new_weeks,current_weeks", [
    (((2025, 8), (2025, 9)), ((2025, 8), (2025, 9))),   # equal
    (((2025, 9), (2025, 10)), ((2025, 8), (2025, 9))),  # one week newer
    (((2026, 1),), ((2025, 22),)),                       # new season, lower week number
])
def test_newest_week_equal_or_newer_ok(slots, new_weeks, current_weeks):
    new = slots("new", make_plays(200, weeks=new_weeks))
    current = slots("current", make_plays(200, weeks=current_weeks))
    assert_all_ok(run_checks(new, current))


def test_newest_week_without_current_slot_ok(slots):
    assert_all_ok(run_checks(slots("new", make_plays(200, weeks=((2025, 8),)))))


# ── 7. null key columns in newest week ─────────────────────────────────────────


def _null_newest(plays, cols, keep_one=False):
    """Set cols to NULL on every (2025, 9) row; keep_one leaves the first such row intact."""
    out, kept = [], not keep_one
    for p in plays:
        row = list(p)
        if (row[2], row[3]) == (2025, 9):
            if kept:
                for c in cols:
                    row[PLAY_COLS.index(c)] = None
            kept = True
        out.append(tuple(row))
    return out


@pytest.mark.parametrize("cols", [("epa",), ("posteam",), ("epa", "posteam")])
def test_null_key_column_in_newest_week_fails(slots, cols):
    plays = _null_newest(make_plays(200), cols)  # all 100 week-9 rows NULL in cols
    results = run_checks(slots("new", plays))
    for col in cols:
        assert_fails_with(results, col)


def test_plays_with_no_season_or_week_fails(slots):
    plays = [(g, pid, None, None, None, None) for g, pid, *_ in make_plays(200)]
    assert_fails_with(run_checks(slots("new", plays)), "season")


@pytest.mark.parametrize("cols", [("epa",), ("epa", "posteam")])
def test_one_non_null_row_in_newest_week_ok(slots, cols):
    plays = _null_newest(make_plays(200), cols, keep_one=True)  # 99 NULL, 1 non-null
    assert_all_ok(run_checks(slots("new", plays)))


def test_null_epa_in_older_week_only_ok(slots):
    # Week 8 rows all NULL epa; newest week 9 is fully populated -> ok.
    plays = [(g, pid, s, w, None if w == 8 else e, t) for g, pid, s, w, e, t in make_plays(200)]
    assert_all_ok(run_checks(slots("new", plays)))


# ── script helpers ─────────────────────────────────────────────────────────────


def _lines(name):
    return (ROOT / name).read_text().splitlines()


def _find(lines, needle, start=0):
    for i in range(start, len(lines)):
        if needle in lines[i] and not lines[i].lstrip().startswith("#"):
            return i
    return -1


def _positions():
    lines = _lines("update_db.sh")
    return lines, {
        "cp": _find(lines, 'cp "$DB_DIR/$CURRENT" "$TARGET"'),
        "ingest": _find(lines, "nfl-mcp ingest"),
        "validate": _find(lines, "nfl-mcp validate"),
        "swap": _find(lines, "ln -sfn"),
    }


# ── 8. update_db.sh validate and chmod ─────────────────────────────────────────


def test_update_db_validates_between_ingest_and_swap():
    _, pos = _positions()
    assert pos["validate"] != -1, "nfl-mcp validate not called"
    assert 0 <= pos["ingest"] < pos["validate"] < pos["swap"]


def test_update_db_validates_target_against_current_slot():
    lines, pos = _positions()
    assert pos["validate"] != -1
    call = lines[pos["validate"]]
    assert re.search(r'--db\s+"?\$TARGET"?', call), call
    assert re.search(r"--current\s+\S*(CURRENT|SYMLINK)", call), call


def _guard_form(lines, v, swap):
    call = lines[v]
    if re.search(r"\|\|\s*(true|:)\b", call):
        return None
    between = "\n".join(lines[v:swap])
    if re.match(r"\s*if\s+!\s", call):
        fi = next((i for i in range(v + 1, swap) if lines[i].strip() == "fi"), -1)
        if fi != -1 and re.search(r"\bexit\s+[1-9]", "\n".join(lines[v:fi])):
            return "if"
        return None
    if re.match(r"\s*if\s", call):
        # `if nfl-mcp validate ...; then ... ln -sfn ...; fi`: the swap sits inside the then-branch.
        between_lines = lines[v + 1:swap]
        closed = any(s.strip() in ("fi", "else") or s.strip().startswith("elif") for s in between_lines)
        return "if-then" if not closed else None
    if "||" in call:
        after = between.split("||", 1)[1]
        return "or" if re.search(r"\bexit\s+[1-9]", after) else None
    errexit = any(re.match(r"\s*set\s+(-[a-z]*e[a-z]*\b|-o\s+errexit)", line) for line in lines[:v])
    plain = not re.search(r"&&|\|", call) and not re.match(r"\s*(if|!|while|until)\b", call)
    return "set-e" if errexit and plain else None


def test_update_db_swap_only_reached_when_validate_succeeds():
    lines, pos = _positions()
    assert pos["validate"] != -1
    assert _guard_form(lines, pos["validate"], pos["swap"]) in ("if", "if-then", "or", "set-e")


def test_update_db_makes_target_read_only_before_swap():
    lines, pos = _positions()
    i = next((k for k, line in enumerate(lines) if re.search(r"chmod\s+(a-w|ugo-w|-w|0?444)\s", line)
              and "$TARGET" in line and not line.lstrip().startswith("#")), -1)
    assert i != -1, "no chmod a-w on $TARGET"
    assert pos["ingest"] < i < pos["swap"]


def test_update_db_restores_write_permission_after_copy_before_ingest():
    # cp keeps the a-w mode of the active slot, so the restore must follow the copy.
    lines, pos = _positions()
    i = next((k for k, line in enumerate(lines) if re.search(r"chmod\s+([ugoa]*\+w|0?[67][0-7]{2})\s", line)
              and "$TARGET" in line and not line.lstrip().startswith("#")), -1)
    assert i != -1, "no chmod u+w on $TARGET"
    assert pos["cp"] < i < pos["ingest"]


# ── 9. entrypoint.sh symlink guard ─────────────────────────────────────────────

LINK_A = 'ln -sfn "$(basename "$SLOT_A")" "$SYMLINK"'
ABSENT = r'\[\s+!\s+-[Leh]\s+"\$SYMLINK"\s+\]'


def _link_guard():
    """Return the guarding condition text for the slot-A link, or None."""
    lines = _lines("entrypoint.sh")
    n = _find(lines, LINK_A)
    assert n != -1, "slot-A ln -sfn line missing"
    line = lines[n]
    if re.search(ABSENT + r"\s*&&", line) or re.search(r'\[\s+-[Lh]\s+"\$SYMLINK"\s+\]\s*\|\|', line):
        return line
    prev = next((lines[i] for i in range(n - 1, -1, -1) if lines[i].strip() and not lines[i].lstrip().startswith("#")), "")
    if re.search(ABSENT, prev) and re.match(r"\s*(if|elif)\b", prev):
        return prev
    depth = 0
    for i in range(n - 1, -1, -1):
        s = lines[i].strip()
        if s == "fi" or s.startswith("fi ") or s.startswith("fi;"):
            depth += 1
        elif re.match(r"(if|elif)\b", s) and depth == 0:
            return s
        elif s.startswith("if ") and depth > 0:
            depth -= 1
    return None


def test_entrypoint_slot_a_link_is_guarded():
    assert _link_guard() is not None


def test_entrypoint_link_guard_checks_symlink_absence():
    guard = _link_guard() or ""
    assert re.search(ABSENT, guard) or re.search(r'\[\s+-[Lh]\s+"\$SYMLINK"\s+\]\s*\|\|', guard), guard


def test_entrypoint_still_links_slot_a_and_reports_active_db():
    entry = (ROOT / "entrypoint.sh").read_text()
    assert entry.count(LINK_A) == 1
    assert 'echo "[init] Active DB: $(readlink "$SYMLINK")"' in entry


# ── 10. regressions that must keep holding ─────────────────────────────────────


def test_update_db_ingests_the_idle_slot():
    lines, pos = _positions()
    text = "\n".join(lines)
    assert 'CURRENT=$(readlink "$SYMLINK")' in text
    assert 'if [ "$CURRENT" = "nflread_a.duckdb" ]; then' in text
    assert 'TARGET="$DB_DIR/nflread_b.duckdb"' in text and 'TARGET="$DB_DIR/nflread_a.duckdb"' in text
    ingest = lines[pos["ingest"]]
    assert 'NFL_MCP_DB_PATH="$TARGET"' in ingest and "CURRENT" not in ingest and "SYMLINK" not in ingest


def test_update_db_swaps_with_ln_sfn_after_ingest():
    lines, pos = _positions()
    swaps = [line.strip() for line in lines if "ln -" in line and not line.lstrip().startswith("#")]
    assert swaps == ['ln -sfn "$(basename "$TARGET")" "$SYMLINK"']
    assert 0 <= pos["cp"] < pos["ingest"] < pos["swap"]


@pytest.mark.parametrize("cron", [
    "0 6 * * 4     /update_db.sh >> /var/log/nflmcp/nfl_update.log 2>&1",
    "0 3 * * 0,1,3 /update_db.sh >> /var/log/nflmcp/nfl_update.log 2>&1",
])
def test_entrypoint_cron_schedule_unchanged(cron):
    assert cron in _lines("entrypoint.sh")
