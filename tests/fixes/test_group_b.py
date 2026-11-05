"""Group B: nfl_query statement handling.

Every test runs against a synthetic DuckDB built in tmp_path, so each expected
value is hand-counted from the rows below (see PLAYS).

Interface contract (shared contract plus the choices made here)
- nfl_query(sql, max_rows=100) -> {"columns": [{"name": str, "type": str}],
  "rows": [dict], "row_count": int, "truncated": bool}.
  "type" is the DuckDB type name as str(cursor.description[i][1]) gives it on
  duckdb 1.5.5, e.g. "VARCHAR", "DOUBLE", "INTEGER", "BIGINT".
- The statement gate is duckdb.extract_statements(sql): exactly one statement,
  and its type must be SELECT or EXPLAIN. On duckdb 1.5.5 DESCRIBE, SUMMARIZE,
  SHOW, PIVOT with an IN list, VALUES, FROM-first and PRAGMA table functions all
  parse as SELECT, so they are allowed.
- Rejected statement: {"error": str, "statement_type": <StatementType.name>}.
  Names on duckdb 1.5.5: INSERT, UPDATE, DELETE, CREATE (also CREATE TEMP TABLE,
  which is rejected: the allowlist is SELECT and EXPLAIN only), DROP, ALTER,
  COPY, ATTACH, SET (also "PRAGMA threads=1"), PRAGMA ("PRAGMA enable_profiling"),
  CALL, LOAD (both INSTALL and LOAD), EXPORT. The error text names the type.
- More than one statement: {"error": ...} whose text contains "one statement".
  A single trailing ";" (optionally followed by a comment) is one statement.
- No outer wrapper: the user's SQL runs as written and the cap is applied by
  fetching max_rows + 1 rows. Error text must not contain the old "AS _q LIMIT".
- max_rows: int in 1..500. Values above 500 are CLAMPED to 500 (no error), which
  keeps the documented "hard max 500" and tests/test_safety.py::test_max_rows_clamped.
  0, negatives, non-int (str, float, None) return {"error": ...} whose text
  contains "max_rows"; the function never raises.
- Unknown column/table keep the existing _sql_error_detail fields: code,
  unknown_column / unknown_table, did_you_mean, hint.
- Timeout stays enforced through nfl_mcp.tools._QUERY_TIMEOUT_SECONDS; the
  error text contains "timeout".

Not pinned here: "PIVOT plays ON posteam USING count(*)" (no IN list) parses on
duckdb 1.5.5 as TWO statements (CREATE + SELECT), so a one-statement gate
rejects it. Tests use PIVOT with an explicit IN list instead.
"""

import duckdb
import pytest

from nfl_mcp import tools
from nfl_mcp.tools import nfl_query

# ── Synthetic data ─────────────────────────────────────────────────────────────

PLAYS_COLUMNS = [
    "game_id", "play_id", "season", "week", "posteam", "defteam",
    "play_type", "epa", "yards_gained", "desc",
]

# (game_id, play_id, season, week, posteam, defteam, play_type, epa, yards, desc)
PLAYS = [
    ("G1", 1, 2024, 1, "MIN", "GB",  "pass",     0.50,  8, "pass short right to Jefferson"),
    ("G1", 2, 2024, 1, "MIN", "GB",  "run",     -0.30,  1, "run up the middle"),
    ("G1", 3, 2024, 1, "GB",  "MIN", "pass",     1.20, 25, "pass deep left, touchdown"),
    ("G1", 4, 2024, 1, "GB",  "MIN", "no_play",  0.00,  0, "offsetting penalties, no play"),
    ("G2", 1, 2024, 2, "OAK", "KC",  "pass",     2.10, 40, "pass deep middle"),
    ("G2", 2, 2024, 2, "OAK", "KC",  "run",     -1.10, -3, "run left end, tackled for loss"),
    ("G2", 3, 2024, 2, "KC",  "OAK", "pass",     0.80, 12, "shotgun set, pass short left"),
    ("G2", 4, 2024, 2, "KC",  "OAK", "run",      0.10,  3, "run right guard"),
    ("G3", 1, 2025, 1, "LV",  "DEN", "pass",    -0.70,  0, "pass incomplete"),
    ("G3", 2, 2025, 1, "LV",  "DEN", "run",      1.50, 18, "run right end"),
    ("G3", 3, 2025, 1, "DEN", "LV",  "pass",    -2.00, -8, "sacked"),
    ("G3", 4, 2025, 1, "DEN", "LV",  "punt",     0.30,  0, "punt 45 yards"),
]
# Hand facts used below:
#   12 rows; season 2024 = 8 rows, 2025 = 4 rows; every posteam has exactly 2 rows.
#   epa ranking (desc): G2/1 2.10, G3/2 1.50, G1/3 1.20, G2/3 0.80, G1/1 0.50, ...
#   yards by posteam: MIN 9, GB 25, OAK 37, KC 15, LV 18, DEN -8.
#   desc ILIKE '%set%': G1/4 "offsetting", G2/3 "shotgun set" -> 2 rows.

PLAYERS = [
    ("00-0000001", "Alex Do", "Do"),
    ("00-0000002", "Jamie Doe", "Doe"),
    ("00-0000003", "Chris Dorsett", "Dorsett"),
]


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "group_b.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute(
        'CREATE TABLE plays (game_id VARCHAR, play_id INTEGER, season INTEGER, '
        'week INTEGER, posteam VARCHAR, defteam VARCHAR, play_type VARCHAR, '
        'epa DOUBLE, yards_gained INTEGER, "desc" VARCHAR)'
    )
    conn.executemany("INSERT INTO plays VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", PLAYS)
    conn.execute("CREATE TABLE players (gsis_id VARCHAR, display_name VARCHAR, last_name VARCHAR)")
    conn.executemany("INSERT INTO players VALUES (?, ?, ?)", PLAYERS)
    conn.execute("CREATE TABLE plays150 AS SELECT range::INTEGER AS i FROM range(150)")
    conn.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return path


def _plays_count(path) -> int:
    conn = duckdb.connect(str(path), read_only=True)
    try:
        return conn.execute("SELECT count(*) FROM plays").fetchone()[0]
    finally:
        conn.close()


def _ok(result):
    assert "error" not in result, result
    return result["rows"]


# ── 1. Valid read SQL the old regex blocked ────────────────────────────────────



def test_cte_runs(db):
    rows = _ok(nfl_query(
        "WITH t AS (SELECT posteam, sum(yards_gained) AS y FROM plays GROUP BY posteam) "
        "SELECT posteam, y FROM t ORDER BY y DESC LIMIT 1"
    ))
    assert rows == [{"posteam": "OAK", "y": 37}]  # OAK 40 + -3 = 37, the max


def test_window_row_number_per_posteam(db):
    rows = _ok(nfl_query(
        "SELECT * FROM (SELECT posteam, game_id, play_id, "
        "ROW_NUMBER() OVER (PARTITION BY posteam ORDER BY epa DESC) AS rn FROM plays) "
        "WHERE rn = 1"
    ))
    # Best play per team by epa, hand-picked from PLAYS: 6 teams, 6 rows.
    assert {(r["posteam"], r["game_id"], r["play_id"]) for r in rows} == {
        ("MIN", "G1", 1), ("GB", "G1", 3), ("OAK", "G2", 1),
        ("KC", "G2", 3), ("LV", "G3", 2), ("DEN", "G3", 4),
    }


def test_replace_function_runs(db):
    rows = _ok(nfl_query(
        "SELECT replace(posteam, 'OAK', 'LV') AS team, count(*) AS n FROM plays "
        "GROUP BY 1 HAVING team = 'LV'"
    ))
    assert rows == [{"team": "LV", "n": 4}]  # 2 OAK + 2 LV


def test_ilike_set_pattern_runs(db):
    rows = _ok(nfl_query(
        "SELECT game_id, play_id FROM plays WHERE \"desc\" ILIKE '%set%' ORDER BY game_id"
    ))
    assert [(r["game_id"], r["play_id"]) for r in rows] == [("G1", 4), ("G2", 3)]  # 2 rows


def test_last_name_do_runs(db):
    rows = _ok(nfl_query("SELECT gsis_id FROM players WHERE last_name = 'Do'"))
    assert rows == [{"gsis_id": "00-0000001"}]  # only Alex Do; Doe/Dorsett excluded


def test_alias_named_load_runs(db):
    rows = _ok(nfl_query("SELECT COUNT(*) AS load FROM plays WHERE season = 2024"))
    assert rows == [{"load": 8}]  # 8 rows in 2024


@pytest.mark.parametrize("sql,expected", [
    ("SELECT 'a;b' AS x", [{"x": "a;b"}]),
    ("SELECT count(*) AS n FROM plays WHERE \"desc\" <> 'drop; table'", [{"n": 12}]),  # no desc matches
])
def test_semicolon_inside_string_literal(db, sql, expected):
    assert _ok(nfl_query(sql)) == expected


@pytest.mark.parametrize("sql,expected", [
    ("SELECT count(*) AS n FROM plays -- all plays", [{"n": 12}]),
    ("SELECT 1 AS x -- note", [{"x": 1}]),
])
def test_trailing_line_comment(db, sql, expected):
    assert _ok(nfl_query(sql)) == expected


def test_from_first_syntax(db):
    rows = _ok(nfl_query("FROM plays SELECT count(*) AS n WHERE season = 2025"))
    assert rows == [{"n": 4}]  # 4 rows in 2025


def test_values_clause(db):
    result = nfl_query("VALUES (1, 'a'), (2, 'b')")
    assert result.get("row_count") == 2, result
    assert [list(r.values()) for r in result["rows"]] == [[1, "a"], [2, "b"]]


@pytest.mark.parametrize("sql", [
    "-- leading comment\nSELECT count(*) AS n FROM plays WHERE posteam = 'MIN'",
    "/* block comment */ SELECT count(*) AS n FROM plays WHERE posteam = 'MIN'",
])
def test_leading_comment(db, sql):
    assert _ok(nfl_query(sql)) == [{"n": 2}]  # MIN has 2 rows


@pytest.mark.parametrize("sql", [
    pytest.param("PIVOT (SELECT posteam, yards_gained FROM plays) ON posteam IN ('MIN', 'OAK') "
                 "USING sum(yards_gained)",
                 id="pivot_statement"),
    pytest.param("SELECT * FROM (SELECT posteam, yards_gained FROM plays) "
                 "PIVOT (sum(yards_gained) FOR posteam IN ('MIN', 'OAK'))", id="pivot_sql_standard"),
])
def test_pivot_with_in_list(db, sql):
    rows = _ok(nfl_query(sql))
    assert len(rows) == 1
    assert (rows[0]["MIN"], rows[0]["OAK"]) == (9, 37)  # MIN 8+1, OAK 40-3


def test_describe_plays(db):
    rows = _ok(nfl_query("DESCRIBE plays"))
    assert [r["column_name"] for r in rows] == PLAYS_COLUMNS


def test_summarize_plays(db):
    rows = _ok(nfl_query("SUMMARIZE plays"))
    assert sorted(r["column_name"] for r in rows) == sorted(PLAYS_COLUMNS)  # one row per column


def test_show_tables(db):
    rows = _ok(nfl_query("SHOW TABLES"))
    assert {r["name"] for r in rows} == {"plays", "players", "plays150"}


def test_explain_select_runs(db):
    result = nfl_query("EXPLAIN SELECT 1")
    assert "error" not in result, result
    assert "rows" in result  # plan text deliberately not asserted


# ── 2. Rejected statement types ────────────────────────────────────────────────

REJECTED = [
    ("INSERT INTO plays (game_id, play_id) VALUES ('G9', 9)", "INSERT"),
    ("UPDATE plays SET epa = 0", "UPDATE"),
    ("DELETE FROM plays", "DELETE"),
    ("CREATE TABLE t2 AS SELECT * FROM plays", "CREATE"),
    ("CREATE TEMP TABLE t3 AS SELECT * FROM plays", "CREATE"),
    ("DROP TABLE plays", "DROP"),
    ("ALTER TABLE plays ADD COLUMN z INTEGER", "ALTER"),
    ("COPY plays TO '{tmp}/out.csv'", "COPY"),
    ("ATTACH '{tmp}/other.duckdb' AS other", "ATTACH"),
    ("EXPORT DATABASE '{tmp}/export_dir'", "EXPORT"),
    ("SET threads = 1", "SET"),
    ("PRAGMA threads = 1", "SET"),
    ("PRAGMA enable_profiling", "PRAGMA"),
    ("CALL pragma_version()", "CALL"),
    ("INSTALL httpfs", "LOAD"),
    ("LOAD httpfs", "LOAD"),
]
REJECTED_IDS = [s.split()[0] + "_" + t for s, t in REJECTED]


@pytest.mark.parametrize("sql,stype", REJECTED, ids=REJECTED_IDS)
def test_rejected_statement_names_type(db, tmp_path, sql, stype):
    result = nfl_query(sql.format(tmp=tmp_path))
    assert "error" in result
    assert result.get("statement_type") == stype
    assert stype in result["error"].upper()


@pytest.mark.parametrize("sql,stype", REJECTED, ids=REJECTED_IDS)
def test_rejected_statement_changes_nothing(db, tmp_path, sql, stype):
    result = nfl_query(sql.format(tmp=tmp_path))
    assert "error" in result
    assert _plays_count(db) == 12  # unchanged from PLAYS
    assert not (tmp_path / "out.csv").exists()
    assert not (tmp_path / "other.duckdb").exists()
    assert not (tmp_path / "export_dir").exists()


# ── 3. One statement only ──────────────────────────────────────────────────────

@pytest.mark.parametrize("sql", ["SELECT 1; SELECT 2", "SELECT 1; DROP TABLE plays"])
def test_two_statements_rejected_mentions_one_statement(db, sql):
    result = nfl_query(sql)
    assert "one statement" in result.get("error", "").lower(), result


@pytest.mark.parametrize("sql", ["SELECT 1; SELECT 2", "SELECT 1; DROP TABLE plays"])
def test_two_statements_rejected_table_intact(db, sql):
    assert "error" in nfl_query(sql)
    assert _plays_count(db) == 12


@pytest.mark.parametrize("sql", [
    "SELECT count(*) AS n FROM plays;",
    "SELECT count(*) AS n FROM plays;   \n",
])
def test_single_trailing_semicolon_runs(db, sql):
    assert _ok(nfl_query(sql)) == [{"n": 12}]


def test_trailing_semicolon_then_comment_runs(db):
    assert _ok(nfl_query("SELECT count(*) AS n FROM plays; -- done")) == [{"n": 12}]


# ── 4. Row cap ─────────────────────────────────────────────────────────────────

def test_order_by_epa_top3(db):
    result = nfl_query("SELECT game_id, play_id FROM plays ORDER BY epa DESC", max_rows=3)
    # Hand ranking: 2.10 G2/1, 1.50 G3/2, 1.20 G1/3
    assert [(r["game_id"], r["play_id"]) for r in result["rows"]] == [("G2", 1), ("G3", 2), ("G1", 3)]
    assert result["truncated"] is True  # 12 rows > 3


def test_inner_limit_honoured(db):
    result = nfl_query("SELECT play_id FROM plays ORDER BY epa ASC LIMIT 2", max_rows=5)
    # Lowest epa: -2.00 G3/3, -1.10 G2/2
    assert [r["play_id"] for r in result["rows"]] == [3, 2]
    assert result["truncated"] is False


def test_exactly_max_rows_not_truncated(db):
    result = nfl_query("SELECT * FROM plays", max_rows=12)
    assert (result["row_count"], result["truncated"]) == (12, False)  # table has exactly 12


def test_one_over_max_rows_truncated(db):
    result = nfl_query("SELECT * FROM plays", max_rows=11)
    assert (result["row_count"], len(result["rows"]), result["truncated"]) == (11, 11, True)


def test_default_cap_is_100(db):
    result = nfl_query("SELECT i FROM plays150 ORDER BY i")
    assert (result["row_count"], result["truncated"]) == (100, True)  # 150 rows, default 100
    assert result["rows"][-1] == {"i": 99}


@pytest.mark.parametrize("n", [600, 9999])
def test_max_rows_above_500_clamped(db, n):
    result = nfl_query("SELECT range AS i FROM range(700)", max_rows=n)
    assert (result["row_count"], result["truncated"]) == (500, True)  # 700 rows, clamp 500


@pytest.mark.parametrize("bad", [0, -5, "7", None, 2.5], ids=["zero", "neg", "str", "none", "float"])
def test_invalid_max_rows_returns_error(db, bad):
    result = nfl_query("SELECT * FROM plays", max_rows=bad)
    assert "max_rows" in result.get("error", ""), result


# ── 5. columns metadata ────────────────────────────────────────────────────────



def test_columns_names_and_types(db):
    result = nfl_query("SELECT game_id, play_id, epa FROM plays LIMIT 2")
    assert result["columns"] == [
        {"name": "game_id", "type": "VARCHAR"},
        {"name": "play_id", "type": "INTEGER"},
        {"name": "epa", "type": "DOUBLE"},
    ]


def test_columns_present_on_zero_rows(db):
    result = nfl_query("SELECT posteam, epa FROM plays WHERE season = 1999")
    assert result["rows"] == [] and result["row_count"] == 0  # no 1999 rows
    assert [c["name"] for c in result["columns"]] == ["posteam", "epa"]


@pytest.mark.parametrize("sql", [
    "SELECT * FROM plays",
    "SELECT season, count(*) AS n FROM plays GROUP BY season",
    "SELECT * FROM players WHERE false",
])
def test_columns_match_row_keys(db, sql):
    result = nfl_query(sql)
    names = [c["name"] for c in result["columns"]]
    assert names and all(isinstance(c["type"], str) and c["type"] for c in result["columns"])
    for row in result["rows"]:
        assert list(row) == names


def test_duplicate_column_names_made_unique(db):
    result = nfl_query("SELECT 1 AS a, 2 AS a")
    assert [c["name"] for c in result["columns"]] == ["a", "a_1"]
    assert result["rows"] == [{"a": 1, "a_1": 2}]


def test_self_join_keeps_every_column(db):
    result = nfl_query(
        "SELECT * FROM plays a JOIN plays b ON a.game_id = b.game_id AND a.play_id = b.play_id"
    )
    names = [c["name"] for c in result["columns"]]
    assert len(names) == 2 * len(PLAYS_COLUMNS) == len(set(names))  # 10 + 10, all unique
    assert result["row_count"] == 12  # each play joins only itself
    for row in result["rows"]:
        assert list(row) == names
        assert row["game_id"] == row["game_id_1"] and row["epa"] == row["epa_1"]


# ── 6. Error detail ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sql,unknown,expected", [
    ("SELECT epaa FROM plays", "epaa", "epa"),
    ("SELECT yards_gain FROM plays", "yards_gain", "yards_gained"),
])
def test_bad_column_did_you_mean(db, sql, unknown, expected):
    result = nfl_query(sql)
    assert result.get("code") == "COLUMN_NOT_FOUND", result
    assert result["unknown_column"] == unknown
    assert expected in result["did_you_mean"]
    assert "hint" in result


def test_bad_table_did_you_mean(db):
    result = nfl_query("SELECT * FROM player")
    assert result.get("code") == "TABLE_NOT_FOUND", result
    assert "players" in result["did_you_mean"]


def test_error_text_has_no_wrapper(db):
    result = nfl_query("SELECT epaa FROM plays")
    assert "error" in result
    assert "AS _q LIMIT" not in result["error"]


# ── 7. Timeout ─────────────────────────────────────────────────────────────────

def test_timeout_enforced(db, monkeypatch):
    monkeypatch.setattr(tools, "_QUERY_TIMEOUT_SECONDS", 1)
    monkeypatch.setenv("NFL_MCP_QUERY_TIMEOUT_SECONDS", "1")
    result = nfl_query(
        "SELECT sum(a.range + b.range) AS s FROM range(1000000) a CROSS JOIN range(1000000) b"
    )
    assert "timeout" in result.get("error", "").lower(), result


# ── 8. EXPLAIN gate, non-string sql ────────────────────────────────────────────

@pytest.mark.parametrize("sql", [
    "EXPLAIN SELECT count(*) FROM plays",
    "EXPLAIN ANALYZE SELECT count(*) FROM plays",
    "/* c */ explain analyze SELECT count(*) FROM plays;",
    "EXPLAIN ANALYZE (SELECT 1)",
    "EXPLAIN (SELECT 1) UNION ALL SELECT 2",
    "EXPLAIN (ANALYZE, FORMAT JSON) SELECT 1",
])
def test_explain_of_select_allowed(db, sql):
    assert "error" not in nfl_query(sql)


@pytest.mark.parametrize("sql,stype", [
    ("EXPLAIN COPY (SELECT 1) TO '{tmp}/out.csv'", "COPY"),
    ("EXPLAIN ANALYZE COPY (SELECT 1) TO '{tmp}/out.csv'", "COPY"),
    ("EXPLAIN ANALYZE CREATE TEMP TABLE z AS SELECT 1", "CREATE"),
    ("explain (analyze) INSERT INTO plays SELECT * FROM plays", "INSERT"),
    ("EXPLAIN (ANALYZE) COPY (SELECT 1) TO '{tmp}/out.csv'", "COPY"),
    ("EXPLAIN (ANALYZE, FORMAT JSON) COPY (SELECT 1) TO '{tmp}/out.csv'", "COPY"),
    ("EXPLAIN(FORMAT JSON) COPY (SELECT 1) TO '{tmp}/out.csv'", "COPY"),
    ("EXPLAIN /* x */ ANALYZE COPY (SELECT 1) TO '{tmp}/out.csv'", "COPY"),
])
def test_explain_of_write_rejected(db, tmp_path, sql, stype):
    result = nfl_query(sql.format(tmp=tmp_path))
    assert result.get("statement_type") == stype, result
    assert stype in result["error"].upper()
    assert _plays_count(db) == 12
    assert not (tmp_path / "out.csv").exists()


@pytest.mark.parametrize("bad", [None, 123])
def test_non_string_sql_returns_error(db, bad):
    assert nfl_query(bad) == {"error": "sql must be a string"}


def test_infinite_timeout_env_falls_back(monkeypatch):
    monkeypatch.setenv("NFL_MCP_QUERY_TIMEOUT_SECONDS", "inf")
    assert tools._query_timeout() == tools._QUERY_TIMEOUT_SECONDS
