"""Tests for SQL safety guardrails in nfl_mcp.tools."""

import pytest
from nfl_mcp.tools import nfl_query


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

    def test_allows_trailing_semicolon(self):
        # A single trailing semicolon is one statement, not rejected
        result = nfl_query("SELECT 1;")
        # Should either succeed or fail on execution, not on validation
        assert "one statement" not in result.get("error", "")

    def test_max_rows_clamped(self):
        result = nfl_query("SELECT 1", max_rows=9999)
        # Should not error on max_rows being too large — it gets clamped
        assert "error" not in result or "max_rows" not in result.get("error", "")


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
