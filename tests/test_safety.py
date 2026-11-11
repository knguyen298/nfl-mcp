"""Tests for SQL safety guardrails in nfl_mcp.tools."""

import duckdb
import pytest

from nfl_mcp import tools
from nfl_mcp.status import ErrorCode
from nfl_mcp.tools import (
    _FORBIDDEN,
    _FORBIDDEN_FUNCTIONS,
    _mask_sql,
    nfl_query,
)

# Statement keywords the branch's text scan used to block. DuckDB's parser now decides the
# statement type (_statement_gate), so these words only matter as test input.
_STATEMENT_WORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE", "REPLACE",
    "UPSERT", "EXECUTE", "EXEC", "CALL", "COPY", "GRANT", "REVOKE", "VACUUM",
    "CLUSTER", "REINDEX", "COMMENT", "SECURITY", "OWNER", "TABLESPACE", "SCHEMA",
    "SET", "DO", "LISTEN", "NOTIFY", "PREPARE", "DEALLOCATE", "LOAD", "DISCARD", "RESET",
    "ATTACH", "DETACH", "INSTALL", "PRAGMA", "EXPORT", "IMPORT", "CHECKPOINT", "USE",
)


# ── Forbidden function detection ───────────────────────────────────────────────

class TestForbiddenRegex:
    """The text scan blocks calls to file-reading and SQL-running functions."""

    def test_statement_keywords_are_left_to_the_parser(self):
        for keyword in _STATEMENT_WORDS + ("ANALYZE",):
            assert _FORBIDDEN.search(keyword) is None, keyword
            assert _FORBIDDEN.search(keyword.lower()) is None, keyword

    @pytest.mark.parametrize("keyword", [
        "pg_read_file", "pg_read_binary_file", "pg_write_file",
        "pg_sleep", "lo_import", "lo_export", "dblink",
        "current_setting", "pg_terminate_backend", "pg_cancel_backend",
        "read_text", "read_blob", "read_csv", "read_csv_auto", "sniff_csv",
        "read_parquet", "parquet_scan", "parquet_metadata", "parquet_schema",
        "parquet_file_metadata", "parquet_kv_metadata", "parquet_full_metadata",
        "parquet_bloom_probe", "read_json", "read_json_auto", "read_json_objects",
        "read_json_objects_auto", "read_ndjson", "read_ndjson_auto",
        "read_ndjson_objects", "read_duckdb", "query", "query_table",
        "json_execute_serialized_sql", "arrow_scan", "arrow_scan_dumb",
        "pandas_scan", "python_map_function",
    ])
    def test_blocks_dangerous_functions(self, keyword):
        assert _FORBIDDEN.search(f"{keyword}('x')"), f"Should block function: {keyword}"
        assert _FORBIDDEN.search(f"{keyword.upper()} ('x')"), keyword
        assert _FORBIDDEN.search(keyword) is None, f"A bare name is not a call: {keyword}"

    def test_allows_select(self):
        assert _FORBIDDEN.search("SELECT") is None

    def test_allows_common_sql_words(self):
        for word in ["FROM", "WHERE", "GROUP", "ORDER", "HAVING", "JOIN",
                     "LEFT", "RIGHT", "INNER", "OUTER", "CASE", "WHEN",
                     "THEN", "ELSE", "END", "AS", "ON", "AND", "OR", "NOT",
                     "IN", "BETWEEN", "LIKE", "ILIKE", "IS", "NULL",
                     "COUNT", "SUM", "AVG", "MIN", "MAX", "ROUND",
                     "COALESCE", "NULLIF", "DISTINCT", "LIMIT", "OFFSET",
                     "WITH", "GLOB", "desc", "settings", "offset_yards", "reset_x"]:
            assert _FORBIDDEN.search(word) is None, f"Should allow: {word}"

    def test_list_and_pattern_agree(self):
        """The pattern is built from the list: every listed function is blocked as a call."""
        for word in _FORBIDDEN_FUNCTIONS:
            assert _FORBIDDEN.match(word + "(").group() == word, word

    def test_glob_blocked_only_as_a_function_call(self):
        """GLOB is also a string-match operator, so only glob( is blocked."""
        assert _FORBIDDEN.search("SELECT * FROM glob('/tmp/*')")
        assert _FORBIDDEN.search("SELECT * FROM GLOB ('/tmp/*')")
        assert _FORBIDDEN.search("SELECT * FROM glob   ('/tmp/*')")
        assert _FORBIDDEN.search("WHERE \"desc\" GLOB '*TOUCHDOWN*'") is None


# ── nfl_query guardrails ──────────────────────────────────────────────────────

class TestNflQueryGuardrails:
    """Test that nfl_query blocks dangerous queries."""

    def test_rejects_non_select(self):
        result = nfl_query("INSERT INTO plays VALUES (1)")
        assert "error" in result
        assert result["statement_type"] == "INSERT"

    def test_rejects_delete(self):
        result = nfl_query("DELETE FROM plays")
        assert "error" in result

    def test_rejects_drop(self):
        result = nfl_query("SELECT 1; DROP TABLE plays")
        assert "error" in result

    def test_rejects_multiple_statements(self):
        result = nfl_query("SELECT 1; SELECT 2")
        assert "error" in result
        assert "one statement" in result["error"]

    def test_rejects_update_in_subquery(self):
        result = nfl_query("SELECT * FROM (UPDATE plays SET season=0) AS x")
        assert "error" in result

    def test_rejects_set_statement(self):
        result = nfl_query("SELECT * FROM plays; SET statement_timeout = '0'")
        assert "error" in result

    def test_rejects_copy(self):
        result = nfl_query("SELECT 1; COPY plays TO '/tmp/dump'")
        assert "error" in result

    def test_rejects_pg_read_file(self):
        result = nfl_query("SELECT pg_read_file('/etc/passwd')")
        assert "error" in result

    def test_rejects_pg_read_binary_file(self):
        result = nfl_query("SELECT pg_read_binary_file('/etc/passwd')")
        assert "error" in result

    def test_allows_trailing_semicolon(self):
        # A single trailing semicolon is one statement, not rejected
        result = nfl_query("SELECT 1;")
        # Should either succeed or fail on execution, not on validation
        assert "one statement" not in result.get("error", "")

    def test_max_rows_clamped(self):
        result = nfl_query("SELECT 1", max_rows=9999)
        # Should not error on max_rows being too large — it gets clamped
        assert "error" not in result or "max_rows" not in result.get("error", "")

    def test_non_string_sql_is_an_invalid_argument(self):
        result = nfl_query(None)
        assert (result["error"], result["code"]) == ("sql must be a string", "INVALID_ARGUMENT")

    @pytest.mark.parametrize("sql,code", [
        ("INSERT INTO plays VALUES (1)", "NOT_SELECT"),
        ("DELETE FROM plays", "NOT_SELECT"),
        ("SELECT pg_sleep(100)", "FORBIDDEN_KEYWORD"),
        ("SELECT 1; SELECT 2", "MULTIPLE_STATEMENTS"),
    ])
    def test_guardrail_errors_carry_a_machine_readable_code(self, sql, code):
        """A caller branches on `code` instead of reading the English `error` text."""
        result = nfl_query(sql)
        assert result["code"] == code
        assert isinstance(result["error"], str)


# ── nfl_query with live DuckDB ────────────────────────────────────────────────

@pytest.mark.integration
class TestNflQueryLive:
    """Integration tests that require the DuckDB database to be loaded."""

    @pytest.fixture(autouse=True)
    def _check_db(self):
        """Skip if no data is loaded."""
        result = nfl_query("SELECT COUNT(*) AS n FROM plays")
        if "error" in result:
            pytest.skip("No DuckDB database loaded — run 'nfl-mcp ingest' first")
        if result["rows"][0]["n"] == 0:
            pytest.skip("plays table is empty")

    def test_basic_select(self):
        result = nfl_query("SELECT COUNT(*) AS n FROM plays")
        assert result["row_count"] == 1
        assert result["rows"][0]["n"] > 0

    def test_team_filter(self):
        result = nfl_query("SELECT COUNT(*) AS n FROM plays WHERE posteam = 'KC'")
        assert result["rows"][0]["n"] > 0

    def test_aggregate_table(self):
        result = nfl_query("SELECT * FROM team_offense_stats LIMIT 1")
        assert result["row_count"] == 1
        assert "team" in result["rows"][0]

    def test_max_rows_respected(self):
        result = nfl_query("SELECT * FROM plays", max_rows=5)
        assert result["row_count"] <= 5

    def test_truncation_flag(self):
        result = nfl_query("SELECT * FROM plays", max_rows=3)
        assert result["truncated"] is True


# ── Lexical masking ───────────────────────────────────────────────────────────

class _Captured(list):
    """The SQL text of each run, with the row limit of each run in `limits`."""

    def __init__(self):
        super().__init__()
        self.limits = []


@pytest.fixture
def captured(monkeypatch):
    """Replace _fetch_capped: record the SQL nfl_query would run, return one row."""
    seen = _Captured()

    def fake_fetch(sql, limit):
        seen.append(sql)
        seen.limits.append(limit)
        return [{"name": "ok", "type": "INTEGER"}], [{"ok": 1}]

    monkeypatch.setattr(tools, "_fetch_capped", fake_fetch)
    return seen


class TestMaskSql:
    """_mask_sql blanks literals, quoted names and comments the way DuckDB lexes them."""

    # (sql, the parts DuckDB does not run as code). A literal or quoted name becomes a
    # run of `#`, a comment becomes spaces; everything else is kept as is.
    @pytest.mark.parametrize("sql,masked_parts", [
        ("SELECT 'set'", ["'set'"]),
        ("SELECT 'it''s; set'", ["'it''s; set'"]),
        ('SELECT "de""sc"', ['"de""sc"']),
        ("SELECT 1 -- set; x", ["-- set; x"]),
        ("SELECT 1 -- set\n, 2", ["-- set"]),
        ("SELECT 1 -- set\r, 2", ["-- set"]),
        ("SELECT /* a /* set */ b */ 1", ["/* a /* set */ b */"]),
        ("SELECT E'a\\'b; set'", ["E'a\\'b; set'"]),
        ("SELECT e'a\\\\', 1", ["e'a\\\\'"]),
        ("SELECT X'1F', B'01'", ["X'1F'", "B'01'"]),
        ("SELECT 'a\\', 1", ["'a\\'"]),
        ("SELECT $$it's; set$$", ["$$it's; set$$"]),
        ("SELECT $t$ a $$ set $t$", ["$t$ a $$ set $t$"]),
        ("SELECT 1 AS a$b$, 2", []),
        ("SELECT 'é; set', 1", ["'é; set'"]),
        ("SELECT 'a'||'b'", ["'a'", "'b'"]),
        ("SELECT abcE'x'", ["'x'"]),
    ])
    def test_masks_what_duckdb_does_not_run_as_code(self, sql, masked_parts):
        expected = sql
        for part in masked_parts:
            fill = " " if part.startswith(("--", "/*")) else "#"
            expected = expected.replace(part, fill * len(part), 1)
        masked, _ = _mask_sql(sql)
        assert masked == expected
        assert len(masked) == len(sql)

    def test_returns_the_unescaped_quoted_names(self):
        assert _mask_sql('SELECT "a""b", "desc" FROM t')[1] == ['a"b', "desc"]

    @pytest.mark.parametrize("sql", [
        "SELECT 'abc",
        "SELECT 'it''s",
        "SELECT \"desc",
        "SELECT 1 /* x",
        "SELECT 1 /* a /* b */",
        "SELECT $$ x",
        "SELECT $t$ x $u$",
        "SELECT E'abc\\'",
    ])
    def test_unterminated_construct_returns_none(self, sql):
        assert _mask_sql(sql) is None

    def test_disagreement_with_the_engine_returns_none(self):
        # DuckDB lexes 1E'x' as the number 1 and the E-string E'x'; the scanner reads the
        # E as part of the number. They disagree on where the literal starts.
        assert _mask_sql("SELECT 1E'x'") is None

    def test_engine_token_inside_a_masked_span_returns_none(self, monkeypatch):
        # A tokenizer that starts a token inside the comment: the engine would run that
        # text as code while the scanner skipped it.
        monkeypatch.setattr(duckdb, "tokenize", lambda sql: [(0, duckdb.token_type.keyword),
                                                             (12, duckdb.token_type.keyword)])
        assert _mask_sql("SELECT 1 /* SET */") is None

    def test_comment_the_engine_does_not_see_returns_none(self, monkeypatch):
        # DuckDB 2 reports a comment token at the start of each comment. A comment the
        # engine does not report is text the engine may run as code.
        monkeypatch.setattr(duckdb, "tokenize", lambda sql: [(0, duckdb.token_type.keyword),
                                                             (7, duckdb.token_type.numeric_const)])
        assert _mask_sql("SELECT 1 /* x */") is None

    def test_comment_token_the_scanner_missed_returns_none(self, monkeypatch):
        monkeypatch.setattr(duckdb, "tokenize", lambda sql: [(0, duckdb.token_type.keyword),
                                                             (7, duckdb.token_type.comment)])
        assert _mask_sql("SELECT 1") is None

    def test_literal_the_engine_does_not_see_returns_none(self, monkeypatch):
        monkeypatch.setattr(duckdb, "tokenize", lambda sql: [(0, duckdb.token_type.keyword)])
        assert _mask_sql("SELECT 'x'") is None


# ── nfl_query: what is now allowed ──────────────────────────────────────────

class TestNflQueryAllows:
    @pytest.mark.parametrize("sql", [
        "SELECT \"desc\" FROM plays WHERE \"desc\" ILIKE '%set%'",
        "SELECT \"desc\" FROM plays WHERE \"desc\" ILIKE '%Call%'",
        "SELECT * FROM plays WHERE \"desc\" ILIKE '%DROP TABLE%' OR \"desc\" ILIKE '%delete%'",
        "SELECT 'a;b' AS s",
        "SELECT 'a -- b /* c' AS s",
        "SELECT 'it''s; set' AS s",
        "SELECT 1 AS \"set\"",
        "SELECT 1 AS \"drop zone\"",
        "SELECT 1 AS \"read_text_column\"",
        "WITH kc AS (SELECT * FROM plays WHERE posteam = 'KC') SELECT count(*) FROM kc",
        "with a AS (SELECT 1 AS x), b AS (SELECT x FROM a) SELECT * FROM b",
        "-- top 5\nSELECT 1",
        "/* note */ SELECT 1",
        "/* a /* nested */ b */\n-- line\nWITH a AS (SELECT 1) SELECT * FROM a",
        "SELECT $$ ; set $$ AS s",
        "SELECT E'it\\'s; set' AS s",
        "SELECT 1 FROM plays WHERE \"desc\" GLOB '*TOUCHDOWN*'",
        # DuckDB's FROM-first form is one SELECT to the parser
        "FROM plays",
        # a column or alias named like a blocked function is not a call
        "SELECT query, current_setting FROM plays",
        "SELECT 1 AS read_csv",
        "SELECT set",
    ])
    def test_passes_the_guard(self, captured, sql):
        result = nfl_query(sql)
        assert "error" not in result, result
        assert len(captured) == 1

    def test_runs_as_written_and_fetches_one_row_more(self, captured):
        sql = "SELECT 'a;b' AS s ; -- done\n/* end */"
        nfl_query(sql, max_rows=5)
        assert captured == [sql]
        assert captured.limits == [6]

    def test_trailing_line_comment_is_kept(self, captured):
        nfl_query("SELECT 1 AS x -- note", max_rows=5)
        assert captured == ["SELECT 1 AS x -- note"]


# ── nfl_query: bypass attempts ────────────────────────────────────────────────

class TestNflQueryBypassAttempts:
    @pytest.mark.parametrize("sql,code", [
        # a keyword separated from its call or its statement by a comment
        ("SELECT * FROM read_text/**/('/etc/passwd')", ErrorCode.FORBIDDEN_KEYWORD),
        ("SELECT * FROM read_text--x\n('/etc/passwd')", ErrorCode.FORBIDDEN_KEYWORD),
        ("SELECT 1 /**/;/**/ SELECT 2", ErrorCode.MULTIPLE_STATEMENTS),
        ("SELECT 1 --x\n; SELECT 2", ErrorCode.MULTIPLE_STATEMENTS),
        ("SELECT 1 --x\r; SELECT 2", ErrorCode.MULTIPLE_STATEMENTS),
        # hidden after a literal with an escaped quote
        ("SELECT 'it''s', read_text('/etc/passwd')", ErrorCode.FORBIDDEN_KEYWORD),
        ("SELECT E'it\\'s', read_text('/etc/passwd')", ErrorCode.FORBIDDEN_KEYWORD),
        # in a plain string a backslash is not an escape: the literal ends at the 2nd quote
        ("SELECT 'a\\', read_text('/etc/passwd') --'", ErrorCode.FORBIDDEN_KEYWORD),
        ("SELECT \"a\"\"\", read_text('/etc/passwd')", ErrorCode.FORBIDDEN_KEYWORD),
        # mixed case
        ("SeLeCt * FrOm ReAd_TeXt('/etc/passwd')", ErrorCode.FORBIDDEN_KEYWORD),
        ("select 1 WHERE 1 = 1 oR 1 IN (sElEcT 1 FROM plays) UnIoN SELECT 1; dRoP TABLE plays",
         ErrorCode.MULTIPLE_STATEMENTS),
        # inside a CTE
        ("WITH x AS (SELECT * FROM read_csv('/etc/passwd')) SELECT * FROM x",
         ErrorCode.FORBIDDEN_KEYWORD),
        # a second statement after a literal that contains a quote
        ("SELECT 'O''Brien'; SELECT 2", ErrorCode.MULTIPLE_STATEMENTS),
        ("SELECT E'\\''; SELECT 2", ErrorCode.MULTIPLE_STATEMENTS),
        ("SELECT \"a;\"\"b\"; SELECT 2", ErrorCode.MULTIPLE_STATEMENTS),
        # dollar-quoted strings
        ("SELECT $t$ ; $t$; SELECT 2", ErrorCode.MULTIPLE_STATEMENTS),
        ("SELECT $$x$$, read_text('/etc/passwd')", ErrorCode.FORBIDDEN_KEYWORD),
        # a quoted function name
        ("SELECT * FROM \"read_text\"('/etc/passwd')", ErrorCode.FORBIDDEN_KEYWORD),
        ("SELECT * FROM main.\"READ_CSV\"('/etc/passwd')", ErrorCode.FORBIDDEN_KEYWORD),
        ("SELECT * FROM \"glob\"('/tmp/*')", ErrorCode.FORBIDDEN_KEYWORD),
        ("SELECT * FROM query('SELECT 1')", ErrorCode.FORBIDDEN_KEYWORD),
        # unterminated literal, identifier or comment: the parser refuses it
        ("SELECT 'abc", ErrorCode.SQL_SYNTAX_ERROR),
        ("SELECT 1 /* ; SELECT 2", ErrorCode.SQL_SYNTAX_ERROR),
        ("SELECT \"desc FROM plays", ErrorCode.SQL_SYNTAX_ERROR),
        ("SELECT $$ ; SELECT 2", ErrorCode.SQL_SYNTAX_ERROR),
        ("SELECT E'\\'; SELECT 2", ErrorCode.SQL_SYNTAX_ERROR),
        ("SELECT 1E'x'", ErrorCode.SQL_SYNTAX_ERROR),
        # not a query
        ("/* SELECT */ DELETE FROM plays", ErrorCode.NOT_SELECT),
        ("-- SELECT\nINSERT INTO plays VALUES (1)", ErrorCode.NOT_SELECT),
        ("'SELECT' FROM plays", ErrorCode.SQL_SYNTAX_ERROR),
        ("SET enable_external_access = true", ErrorCode.NOT_SELECT),
        ("RESET lock_configuration", ErrorCode.NOT_SELECT),
        ("PRAGMA enable_external_access = true", ErrorCode.NOT_SELECT),
        ("", ErrorCode.MULTIPLE_STATEMENTS),
        ("SELECT 1; SET enable_external_access = true", ErrorCode.MULTIPLE_STATEMENTS),
        ("SELECT 1; RESET lock_configuration", ErrorCode.MULTIPLE_STATEMENTS),
        ("EXPLAIN ANALYZE DELETE FROM plays", ErrorCode.NOT_SELECT),
    ])
    def test_rejected_before_execution(self, captured, sql, code):
        result = nfl_query(sql)
        assert result["code"] == code, result
        assert isinstance(result["error"], str)
        assert captured == []

    def test_unreadable_message(self, captured, monkeypatch):
        # The parser accepts the text, but the tokenizer starts a token inside the comment,
        # so the scanner cannot trust its reading and refuses.
        monkeypatch.setattr(duckdb, "tokenize", lambda sql: [(0, duckdb.token_type.keyword),
                                                             (12, duckdb.token_type.keyword)])
        result = nfl_query("SELECT 1 /* SET */")
        assert (result["error"], result["code"]) == (
            "Unterminated or ambiguous string literal, quoted identifier or comment.",
            "SQL_ERROR",
        )
        assert captured == []

    def test_forbidden_message_names_the_original_text(self, captured):
        assert nfl_query("SELECT * FROM ReAd_TeXt('x')")["error"] == "Forbidden keyword: 'ReAd_TeXt'"
        assert nfl_query('SELECT * FROM "Read_Text"(\'x\')')["error"] == "Forbidden keyword: 'Read_Text'"


class TestDenylistProperty:
    """Each denylisted word is fine inside a literal and rejected as a bare token."""

    @pytest.mark.parametrize("word", _STATEMENT_WORDS + _FORBIDDEN_FUNCTIONS + ("glob",))
    def test_inside_a_literal_is_allowed(self, captured, word):
        for sql in (
            f"SELECT \"desc\" FROM plays WHERE \"desc\" ILIKE '%{word}%'",
            f"SELECT '{word.lower()}; {word}(1) -- /*' AS s",
            f"SELECT $${word}$$ AS s",
            f"SELECT 1 /* {word} */ AS s -- {word}",
        ):
            assert "error" not in nfl_query(sql), sql

    @pytest.mark.parametrize("word", _FORBIDDEN_FUNCTIONS)
    def test_as_a_call_is_rejected(self, captured, word):
        for sql in (f"SELECT {word}('x')", f"SELECT 1 FROM plays WHERE {word.lower()} (1) = 1"):
            result = nfl_query(sql)
            assert result["code"] == ErrorCode.FORBIDDEN_KEYWORD, sql
        assert captured == []

    @pytest.mark.parametrize("sql,statement_type", [
        ("INSERT INTO plays VALUES (1)", "INSERT"),
        ("UPDATE plays SET a = 1", "UPDATE"),
        ("DELETE FROM plays", "DELETE"),
        ("DROP TABLE plays", "DROP"),
        ("CREATE TABLE t (a INT)", "CREATE"),
        ("ALTER TABLE plays ADD COLUMN x INT", "ALTER"),
        ("COPY plays TO '/tmp/x'", "COPY"),
        ("SET threads = 1", "SET"),
        ("RESET threads", "SET"),
        ("USE memory", "SET"),
        ("PRAGMA version", "PRAGMA"),
        ("ATTACH 'x.db'", "ATTACH"),
        ("DETACH x", "DETACH"),
        ("INSTALL httpfs", "LOAD"),
        ("LOAD httpfs", "LOAD"),
        ("EXPORT DATABASE '/tmp/x'", "EXPORT"),
        ("CHECKPOINT", "CALL"),
        ("CALL pragma_version()", "CALL"),
        ("VACUUM", "VACUUM"),
        ("PREPARE p AS SELECT 1", "PREPARE"),
        ("EXECUTE p", "EXECUTE"),
    ])
    def test_statement_keyword_is_refused_by_the_parser(self, captured, sql, statement_type):
        result = nfl_query(sql)
        assert result["code"] == ErrorCode.NOT_SELECT, sql
        assert result["statement_type"] == statement_type
        assert captured == []

    @pytest.mark.parametrize("word", _FORBIDDEN_FUNCTIONS + ("glob",))
    def test_function_as_a_quoted_name_is_rejected(self, captured, word):
        result = nfl_query(f'SELECT * FROM "{word}"(\'x\')')
        assert result["code"] == ErrorCode.FORBIDDEN_KEYWORD
        assert captured == []

    def test_glob_call_is_rejected(self, captured):
        assert nfl_query("SELECT * FROM glob('/tmp/*')")["code"] == ErrorCode.FORBIDDEN_KEYWORD


# ── nfl_query: max_rows clamp ─────────────────────────────────────────────────

class TestMaxRowsClamp:
    @pytest.mark.parametrize("max_rows", [0, -1, -500])
    def test_non_positive_is_an_invalid_argument(self, captured, max_rows):
        result = nfl_query("SELECT 1 AS x", max_rows=max_rows)
        assert result["code"] == ErrorCode.INVALID_ARGUMENT
        assert captured == []

    def test_one_row_back_of_two_is_truncated(self, monkeypatch):
        monkeypatch.setattr(tools, "_fetch_capped",
                            lambda sql, limit: ([{"name": "x", "type": "INTEGER"}],
                                                [{"x": 1}, {"x": 2}][:limit]))
        result = nfl_query("SELECT 1 AS x", max_rows=1)
        assert (result["rows"], result["row_count"], result["truncated"]) == ([{"x": 1}], 1, True)

    def test_zero_rows_back_is_not_truncated(self, monkeypatch):
        monkeypatch.setattr(tools, "_fetch_capped", lambda sql, limit: ([], []))
        result = nfl_query("SELECT 1 WHERE false", max_rows=1)
        assert (result["rows"], result["row_count"], result["truncated"]) == ([], 0, False)

    def test_large_clamps_to_500(self, captured):
        nfl_query("SELECT 1", max_rows=10_000)
        assert captured.limits == [501]


# ── nfl_query: the engine parser backstop ─────────────────────────────────────

class _FakeStatement:
    def __init__(self, type_):
        self.type = type_


class TestParserBackstop:
    """DuckDB's parser must see exactly one SELECT, whatever the text checks concluded."""

    @pytest.mark.parametrize("statements,code", [
        ([_FakeStatement(duckdb.StatementType.SELECT)] * 2, ErrorCode.MULTIPLE_STATEMENTS),
        ([_FakeStatement(duckdb.StatementType.SET)], ErrorCode.NOT_SELECT),
        ([], ErrorCode.MULTIPLE_STATEMENTS),
    ])
    def test_rejects_unless_one_select(self, captured, monkeypatch, statements, code):
        monkeypatch.setattr(duckdb, "extract_statements", lambda sql: statements)
        result = nfl_query("SELECT 1")
        assert result["code"] == code
        assert captured == []

    def test_parse_error_is_a_syntax_error(self, captured):
        result = nfl_query("SELECT FROM WHERE")
        assert result["code"] == ErrorCode.SQL_SYNTAX_ERROR
        assert "Parser Error" in result["error"]
        assert captured == []


# ── nfl_query end to end against a temporary database ─────────────────────────

_PLAY_DESCRIPTIONS = (
    "(1, 'KC', 'P.Mahomes pass short; set up by a play-action fake'),"
    " (2, 'KC', 'Call reversed -- after review /* booth */'),"
    " (3, 'BUF', 'J.Allen scrambles, O''Brien tackles'),"
    " (4, 'BUF', 'Timeout #1 by BUF')"
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """A temporary database with a plays table, plus a secret file outside it."""
    for var in ("NFL_MCP_DUCKDB_MEMORY_LIMIT", "NFL_MCP_DUCKDB_THREADS", "NFL_MCP_DUCKDB_TEMP_DIR"):
        monkeypatch.delenv(var, raising=False)
    path = tmp_path / "nflread.duckdb"
    con = duckdb.connect(str(path))
    con.execute(f"CREATE TABLE plays AS SELECT * FROM (VALUES {_PLAY_DESCRIPTIONS}) "
                "AS v(play_id, posteam, \"desc\")")
    con.close()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.csv").write_text("a,b\nTOPSECRET,2\n")
    mem = duckdb.connect(":memory:")
    mem.execute(f"COPY (SELECT 'TOPSECRET' AS s) TO '{outside / 'secret.parquet'}' (FORMAT parquet)")
    mem.execute(f"COPY (SELECT 'TOPSECRET' AS s) TO '{outside / 'secret.json'}'")
    mem.close()
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(path))
    return outside


@pytest.fixture
def no_text_guard(monkeypatch):
    """Turn the function scan off, so the engine setting is the only thing under test."""
    monkeypatch.setattr(tools, "_FORBIDDEN", __import__("re").compile(r"(?!x)x"))
    monkeypatch.setattr(tools, "_FORBIDDEN_QUOTED_NAMES", frozenset())


_FILE_VECTORS = [
    "SELECT * FROM read_text('{d}/secret.csv')",
    "SELECT * FROM read_blob('{d}/secret.csv')",
    "SELECT * FROM read_csv('{d}/secret.csv')",
    "SELECT * FROM read_csv_auto('{d}/secret.csv')",
    "SELECT * FROM sniff_csv('{d}/secret.csv')",
    "SELECT * FROM read_parquet('{d}/secret.parquet')",
    "SELECT * FROM parquet_scan('{d}/secret.parquet')",
    "SELECT * FROM parquet_metadata('{d}/secret.parquet')",
    "SELECT * FROM read_json('{d}/secret.json')",
    "SELECT * FROM read_json_auto('{d}/secret.json')",
    "SELECT * FROM glob('{d}/*')",
    "SELECT * FROM '{d}/secret.csv'",
    "SELECT * FROM \"{d}/secret.csv\"",
    "SELECT * FROM '{d}/secret.parquet'",
    "SELECT * FROM query('SELECT * FROM read_text(''{d}/secret.csv'')')",
    "SELECT * FROM read_csv('https://example.invalid/x.csv')",
    "SELECT * FROM read_parquet('http://127.0.0.1:9/x.parquet')",
    "WITH s AS (SELECT * FROM read_text('{d}/secret.csv')) SELECT * FROM s",
]


class TestNflQueryEngineRefusesExternalAccess:
    @pytest.mark.parametrize("template", _FILE_VECTORS)
    def test_text_guard_or_engine_refuses(self, temp_db, template):
        result = nfl_query(template.format(d=temp_db))
        assert "error" in result and "code" in result
        assert "TOPSECRET" not in repr(result)
        assert "secret.csv\\n" not in repr(result)

    @pytest.mark.parametrize("template", _FILE_VECTORS)
    def test_engine_refuses_with_the_text_guard_off(self, temp_db, no_text_guard, template):
        result = nfl_query(template.format(d=temp_db))
        assert "error" in result and "code" in result
        assert "TOPSECRET" not in repr(result)
        assert "rows" not in result

    def test_engine_error_says_file_access_is_disabled(self, temp_db, no_text_guard):
        result = nfl_query(f"SELECT * FROM read_text('{temp_db}/secret.csv')")
        assert result["code"] == ErrorCode.SQL_ERROR
        assert "disabled by configuration" in result["error"]

    def test_data_modifying_cte_fails_closed(self, temp_db, no_text_guard):
        result = nfl_query("WITH x AS (DELETE FROM plays RETURNING *) SELECT * FROM x")
        assert result["code"] == ErrorCode.SQL_ERROR
        assert nfl_query("SELECT count(*) AS n FROM plays")["rows"] == [{"n": 4}]

    @pytest.mark.parametrize("sql", [
        "SELECT current_setting('enable_external_access') AS v",
        "SELECT value AS v FROM duckdb_settings() WHERE name = 'lock_configuration'",
    ])
    def test_settings_read_back_locked_through_nfl_query(self, temp_db, no_text_guard, sql):
        rows = nfl_query(sql)["rows"]
        assert rows[0]["v"] in (False, "true")


class TestNflQueryEndToEnd:
    @pytest.mark.parametrize("sql,expected", [
        ("SELECT play_id FROM plays WHERE \"desc\" ILIKE '%set%' ORDER BY 1", [1]),
        ("SELECT play_id FROM plays WHERE \"desc\" ILIKE '%Call%' ORDER BY 1", [2]),
        ("SELECT play_id FROM plays WHERE \"desc\" LIKE '%; set%'", [1]),
        ("SELECT play_id FROM plays WHERE \"desc\" LIKE '%-- after%' AND \"desc\" LIKE '%/* booth%'", [2]),
        ("SELECT play_id FROM plays WHERE \"desc\" ILIKE '%O''Brien%'", [3]),
        ("SELECT play_id FROM plays WHERE \"desc\" GLOB '*#1*'", [4]),
        ("WITH k AS (SELECT * FROM plays WHERE posteam = 'KC') SELECT play_id FROM k ORDER BY 1", [1, 2]),
        ("-- KC plays\nSELECT play_id FROM plays WHERE posteam = 'KC' ORDER BY 1;", [1, 2]),
        ("/* a /* b */ c */ SELECT play_id AS \"set\" FROM plays WHERE play_id = 3", [3]),
        ("SELECT play_id FROM plays WHERE $$BUF$$ = posteam ORDER BY 1 -- trailing", [3, 4]),
        ("SELECT play_id FROM plays WHERE E'K\\x43' = posteam ORDER BY 1", [1, 2]),
    ])
    def test_normal_queries_work(self, temp_db, sql, expected):
        result = nfl_query(sql)
        assert "error" not in result, result
        assert [list(r.values())[0] for r in result["rows"]] == expected

    def test_max_rows_one_returns_one_row(self, temp_db):
        result = nfl_query("SELECT play_id FROM plays ORDER BY 1", max_rows=1)
        assert (result["rows"], result["row_count"], result["truncated"]) == (
            [{"play_id": 1}], 1, True)

    def test_max_rows_one_with_one_match_is_not_truncated(self, temp_db):
        result = nfl_query("SELECT play_id FROM plays WHERE play_id = 4", max_rows=1)
        assert (result["rows"], result["row_count"], result["truncated"]) == (
            [{"play_id": 4}], 1, False)

    def test_data_modifying_cte_fails_closed_with_the_guard_on(self, temp_db):
        result = nfl_query("WITH x AS (DELETE FROM plays RETURNING *) SELECT * FROM x")
        assert result["code"] == ErrorCode.SQL_ERROR
        assert nfl_query("SELECT count(*) AS n FROM plays")["rows"] == [{"n": 4}]
