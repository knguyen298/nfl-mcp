"""The one integration test that reads the database on disk as it is, with no rebuilt copy.

Every other integration test runs against a copy that conftest.rebuilt_db_copy rebuilt with the
code under test. That is why the suite stayed green at 100 percent coverage while the shared
database answered all nine guarded tools with the staleness refusal. This test opens the file the
server would open, read-only, and fails if any of those tools refuses.
"""

import pytest

from nfl_mcp import tools
from nfl_mcp.config import get_duckdb_path

# Read at import, before the session fixture rebuilt_db points NFL_MCP_DB_PATH at its copy.
_ON_DISK = get_duckdb_path()

_GUARDED = {
    "nfl_team_stats": lambda: tools.nfl_team_stats(team="KC"),
    "nfl_compare": lambda: tools.nfl_compare("KC", "BUF"),
    "nfl_team_records": lambda: tools.nfl_team_records(team="KC"),
    "nfl_td_luck": lambda: tools.nfl_td_luck(limit=1),
    "nfl_role_trend": lambda: tools.nfl_role_trend(limit=1),
    "nfl_separation_opportunity": lambda: tools.nfl_separation_opportunity(limit=1),
    "nfl_drop_rate": lambda: tools.nfl_drop_rate(limit=1),
    "nfl_contract_value": lambda: tools.nfl_contract_value(limit=1),
    "nfl_injury_return": lambda: tools.nfl_injury_return(limit=1),
}


@pytest.mark.integration
@pytest.mark.parametrize("name", sorted(_GUARDED))
def test_the_database_on_disk_is_not_refused(name, monkeypatch):
    if not _ON_DISK.exists():
        pytest.skip(f"No DuckDB database at {_ON_DISK}")
    monkeypatch.setenv("NFL_MCP_DB_PATH", str(_ON_DISK))

    result = _GUARDED[name]()

    assert tools._STALE_SHAPE_LEAD not in str(result.get("error", ""))
