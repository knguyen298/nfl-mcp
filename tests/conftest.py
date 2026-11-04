"""Pytest marker normalization for unit and integration test selection."""

import shutil

import duckdb
import pytest

from nfl_mcp.config import get_duckdb_path
from nfl_mcp.ingest import rebuild_derived_tables


def pytest_collection_modifyitems(items):
    """Auto-tag unmarked tests as unit tests."""
    for item in items:
        if "integration" not in item.keywords:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session")
def rebuilt_db_copy(tmp_path_factory):
    """Copy the database the path resolves to into a temp file and rebuild its derived tables.

    The databases this suite runs against - the local one, and CI's 2024-only one - were both
    ingested before the derived tables gained the columns this change adds, and a test must not
    write to either: the server holds the local file open read-only and the ingest lock is one
    writer deep. So copy whichever database the path resolves to, rebuild the derived tables in
    the copy with the code under test, and run the integration tests against that. On a database
    ingested after this change the rebuild recomputes the same rows the ingest produced. Returns
    None when no database is loaded.

    It calls rebuild_derived_tables rather than the two builders, because the rebuild is what also
    stamps the derived shape with this build's DERIVED_SHAPE_VERSION. A copy that only had its tables
    rebuilt keeps the stamp of the database it came from, and the nine tools that refuse to answer
    from a stale shape then refuse to answer here - which is what happened when the stamp went to 2
    for the corrected first_downs.
    """
    source = get_duckdb_path()
    if not source.exists():
        return None
    copy = tmp_path_factory.mktemp("rebuilt") / "nflread.duckdb"
    shutil.copyfile(source, copy)
    conn = duckdb.connect(str(copy))
    try:
        rebuild_derived_tables(conn, str(copy))
    finally:
        conn.close()
    return copy


@pytest.fixture(scope="session")
def rebuilt_db(rebuilt_db_copy):
    """Point the tools at the rebuilt copy for the duration of the session.

    Session scope because tests/test_player_resolver.py asks for rebuilt_db from a module-scoped
    fixture, and pytest refuses a function-scoped fixture inside a module-scoped one. The
    environment is repointed with a MonkeyPatch owned here and undone when the session ends.
    """
    if rebuilt_db_copy is None:
        pytest.skip("No DuckDB database loaded")
    patcher = pytest.MonkeyPatch()
    patcher.setenv("NFL_MCP_DB_PATH", str(rebuilt_db_copy))
    yield rebuilt_db_copy
    patcher.undo()
