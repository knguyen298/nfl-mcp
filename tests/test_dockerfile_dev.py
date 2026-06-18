from pathlib import Path


def test_dev_dockerfile_installs_reconciled_local_package():
    dockerfile = Path("Dockerfile.dev").read_text()

    assert "uv pip install --system nfl-mcp" not in dockerfile
    assert "site-packages/nfl_mcp" not in dockerfile
    assert "COPY . /src" in dockerfile
    assert "uv pip install --system --no-cache ." in dockerfile


def test_dev_dockerfile_uses_runtime_data_location_and_update_entrypoint():
    dockerfile = Path("Dockerfile.dev").read_text()

    assert "ENV NFL_MCP_DB_PATH=/data/nflread.duckdb" in dockerfile
    assert "COPY entrypoint.sh /entrypoint.sh" in dockerfile
    assert "COPY update_db.sh /update_db.sh" in dockerfile
    assert "chmod +x /entrypoint.sh /update_db.sh" in dockerfile
    assert "ENTRYPOINT [\"/entrypoint.sh\"]" in dockerfile
    assert "supercronic" in dockerfile
    assert "uv pip install --system --no-cache ." in dockerfile


def test_entrypoint_initializes_slot_a_and_starts_scheduler_and_server():
    entrypoint = Path("entrypoint.sh").read_text()

    assert entrypoint.startswith("#!/bin/sh\n")
    assert "NFL_MCP_DB_PATH=\"$DB_DIR/nflread_a.duckdb\" nfl-mcp ingest" in entrypoint
    assert "ln -s nflread_a.duckdb \"$SYMLINK\"" in entrypoint
    assert "supercronic /tmp/crontab &" in entrypoint
    assert "exec nfl-mcp serve --host 0.0.0.0 --port 8000" in entrypoint


def test_update_script_ingests_inactive_slot_and_swaps_symlink():
    script = Path("update_db.sh").read_text()

    assert script.startswith("#!/bin/sh\n")
    assert "CURRENT=$(readlink \"$SYMLINK\")" in script
    assert "TARGET=\"$DB_DIR/nflread_b.duckdb\"" in script
    assert "TARGET=\"$DB_DIR/nflread_a.duckdb\"" in script
    assert "cp \"$DB_DIR/$CURRENT\" \"$TARGET\"" in script
    assert "NFL_MCP_DB_PATH=\"$TARGET\" nfl-mcp ingest" in script
    assert "ln -sfn \"$(basename \"$TARGET\")\" \"$SYMLINK\"" in script


def test_database_update_scripts_are_posix_sh_compatible():
    for script_name in ("entrypoint.sh", "update_db.sh"):
        script = Path(script_name).read_text()
        assert script.startswith("#!/bin/sh\n")
        assert "pipefail" not in script
