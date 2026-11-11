"""Pytest marker normalization for unit and integration test selection."""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-tag unmarked tests as unit tests."""
    for item in items:
        if "integration" not in item.keywords:
            item.add_marker(pytest.mark.unit)


@pytest.fixture
def ingest_run():
    """Collect the attempts the ingest helpers make inside a test, the way a whole run does.

    The merge replaced the module-level ingest._FAILURES list with a collector per run
    (ingest._collecting). A test that calls an ingest helper directly reads what failed
    from ingest_run.result().failures.
    """
    from nfl_mcp import ingest

    with ingest._collecting() as run:
        yield run
