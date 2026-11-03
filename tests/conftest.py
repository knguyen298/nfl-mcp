"""Pytest marker normalization for unit and integration test selection."""

import shutil

import duckdb
import pytest


def pytest_collection_modifyitems(items):
    """Auto-tag unmarked tests as unit tests."""
    for item in items:
        if "integration" not in item.keywords:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session")
def rebuilt_db(tmp_path_factory):
    """Point the tools at a copy of the database whose derived tables have been rebuilt.

    The databases this suite runs against - the local one, and CI's 2024-only one - were both
    ingested before the team tables gained a season_type column, and a test must not write to
    either: the server holds the local file open read-only and the ingest lock is one writer deep.
    So copy whichever database the path resolves to, rebuild the derived tables in the copy, and
    run the integration tests against that. On a database ingested after this change the rebuild
    recomputes the same rows the ingest produced.
    """
    from nfl_mcp.config import get_duckdb_path
    from nfl_mcp.ingest import _create_aggregate_tables, _create_fantasy_tables

    source = get_duckdb_path()
    if not source.exists():
        pytest.skip(f"no database at {source}")
    copy = tmp_path_factory.mktemp("rebuilt") / "nflread.duckdb"
    shutil.copyfile(source, copy)
    conn = duckdb.connect(str(copy))
    try:
        _create_aggregate_tables(conn)
        _create_fantasy_tables(conn)
    finally:
        conn.close()

    patcher = pytest.MonkeyPatch()
    patcher.setenv("NFL_MCP_DB_PATH", str(copy))
    yield copy
    patcher.undo()
