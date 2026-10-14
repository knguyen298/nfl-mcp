"""Container image + database-updater wiring tests.

These assert on file contents rather than building images: the fork's auto-updating
database once regressed to a no-op simply because the published Dockerfile stopped using
entrypoint.sh, and a text assertion catches exactly that class of drift.
"""

import os
import subprocess
from pathlib import Path


def _read(name: str) -> str:
    return Path(name).read_text()


# ── dev image ──────────────────────────────────────────────────────────────────


def test_dev_dockerfile_installs_reconciled_local_package():
    dockerfile = _read("Dockerfile.dev")

    assert "uv pip install --system nfl-mcp" not in dockerfile
    assert "site-packages/nfl_mcp" not in dockerfile
    assert "COPY . /src" in dockerfile
    assert "uv pip install --system --no-cache ." in dockerfile


def test_dev_dockerfile_uses_runtime_data_location_and_update_entrypoint():
    dockerfile = _read("Dockerfile.dev")

    assert "ENV NFL_MCP_DB_PATH=/data/nflread.duckdb" in dockerfile
    assert "COPY entrypoint.sh /entrypoint.sh" in dockerfile
    assert "COPY update_db.sh /update_db.sh" in dockerfile
    assert "chmod +x /entrypoint.sh /update_db.sh" in dockerfile
    assert "ENTRYPOINT [\"/entrypoint.sh\"]" in dockerfile
    assert "supercronic" in dockerfile
    assert "uv pip install --system --no-cache ." in dockerfile


# ── published image must carry the updater ─────────────────────────────────────

class TestPublishedImageCarriesUpdater:
    """The auto-updating database has to live in the image GitHub Actions publishes."""

    def test_publish_workflow_builds_this_dockerfile(self):
        # build-push-action with no `file:` input builds ./Dockerfile, which is the only
        # reason these assertions matter for the shipped image.
        workflow = _read(".github/workflows/docker.yml")
        assert "file: Dockerfile.dev" not in workflow

    def test_installs_the_scheduler(self):
        dockerfile = _read("Dockerfile")
        assert "supercronic" in dockerfile
        assert "SUPERCRONIC_VERSION" in dockerfile

    def test_uses_the_update_entrypoint(self):
        dockerfile = _read("Dockerfile")
        assert "COPY entrypoint.sh /entrypoint.sh" in dockerfile
        assert "COPY update_db.sh /update_db.sh" in dockerfile
        assert "ENTRYPOINT [\"/entrypoint.sh\"]" in dockerfile
        assert 'ENTRYPOINT ["nfl-mcp", "serve"' not in dockerfile

    def test_seeds_the_image_and_serves_from_the_data_symlink(self):
        dockerfile = _read("Dockerfile")
        # Baked database is a seed under /seed; /data holds the swappable A/B slots.
        assert "NFL_MCP_DB_PATH=/seed/nflread.duckdb" in dockerfile
        assert "ENV NFL_MCP_DB_PATH=/data/nflread.duckdb" in dockerfile

    def test_runs_standalone_with_no_mounted_storage(self):
        # The image is the deliverable: no external database, no assumed volume mount,
        # and /data writable by an arbitrary --user.
        dockerfile = _read("Dockerfile")
        assert "VOLUME" not in dockerfile
        assert "chmod 777 /data /var/log/nflmcp" in dockerfile

    def test_build_fails_on_a_silently_incomplete_seed(self):
        dockerfile = _read("Dockerfile")
        assert "test ! -e /seed/nflread.duckdb.wal" in dockerfile
        assert "('plays','schedules')" in dockerfile

    def test_dockerfile_and_entrypoint_agree_on_the_served_path(self):
        # The drift that disabled auto-updating was these two paths diverging.
        served = 'ENV NFL_MCP_DB_PATH=/data/nflread.duckdb'
        assert served in _read("Dockerfile")
        assert 'SYMLINK="$DB_DIR/nflread.duckdb"' in _read("entrypoint.sh")
        assert 'DB_DIR="/data"' in _read("entrypoint.sh")


# ── entrypoint: seed, then serve, then keep fresh ──────────────────────────────

class TestEntrypoint:
    def test_seeds_slot_a_instead_of_ingesting_on_startup(self):
        # A cold ingest at container start takes 10+ minutes and blocks readiness behind
        # any startup probe, so the seed must be copied instead.
        entrypoint = _read("entrypoint.sh")
        assert 'cp "$SEED" "$SLOT_A"' in entrypoint
        assert 'SEED="/seed/nflread.duckdb"' in entrypoint

    def test_cold_ingest_is_only_the_no_seed_fallback(self):
        entrypoint = _read("entrypoint.sh")
        assert 'NFL_MCP_DB_PATH="$SLOT_A" nfl-mcp ingest' in entrypoint
        assert 'elif [ -f "$SEED" ]' in entrypoint

    def test_adopts_a_legacy_baked_database(self):
        # Images that baked the DB straight into /data must not be re-ingested over.
        assert 'mv "$SYMLINK" "$SLOT_A"' in _read("entrypoint.sh")

    def test_links_the_active_slot_idempotently(self):
        entrypoint = _read("entrypoint.sh")
        assert 'ln -sfn "$(basename "$SLOT_A")" "$SYMLINK"' in entrypoint
        # Plain `ln -s` fails with "File exists" when /data/nflread.duckdb is already a
        # regular file, which under set -eu crash-loops the container.
        assert 'ln -s nflread_a.duckdb "$SYMLINK"' not in entrypoint

    def test_starts_scheduler_then_serves(self):
        entrypoint = _read("entrypoint.sh")
        assert "supercronic /tmp/crontab &" in entrypoint
        assert "exec nfl-mcp serve --host 0.0.0.0 --port 8000" in entrypoint


# ── update_db.sh: ingest idle slot, then swap ─────────────────────────────────

class TestUpdateScript:
    def test_ingests_inactive_slot_and_swaps_symlink(self):
        script = _read("update_db.sh")

        assert script.startswith("#!/bin/sh\n")
        assert "CURRENT=$(readlink \"$SYMLINK\")" in script
        assert "TARGET=\"$DB_DIR/nflread_b.duckdb\"" in script
        assert "TARGET=\"$DB_DIR/nflread_a.duckdb\"" in script
        assert "cp \"$DB_DIR/$CURRENT\" \"$TARGET\"" in script
        assert "NFL_MCP_DB_PATH=\"$TARGET\" nfl-mcp ingest" in script
        assert "ln -sfn \"$(basename \"$TARGET\")\" \"$SYMLINK\"" in script

    def test_clears_the_idle_slot_before_reusing_it(self):
        script = _read("update_db.sh")
        assert 'rm -f "$TARGET" "$TARGET.wal"' in script
        # The cleanup must come before the copy that overwrites it.
        assert script.index('rm -f "$TARGET"') < script.index('cp "$DB_DIR/$CURRENT"')

    def test_aborts_on_insufficient_disk_before_the_copy(self):
        script = _read("update_db.sh")
        assert "FREE_KB" in script and "NEEDED_KB" in script
        assert script.index("NEEDED_KB") < script.index('cp "$DB_DIR/$CURRENT"')

    def test_symlink_swap_is_what_publishes_new_data(self):
        # Documents why no restart is needed: a fresh read-only connection per query.
        assert "fresh read-only DuckDB connection per" in _read("update_db.sh")
        assert "read_only=True" in _read("nfl_mcp/database.py")


# ── shell hygiene ──────────────────────────────────────────────────────────────

def test_database_update_scripts_are_posix_sh_compatible():
    for script_name in ("entrypoint.sh", "update_db.sh"):
        script = _read(script_name)
        assert script.startswith("#!/bin/sh\n")
        assert "pipefail" not in script


def test_update_scripts_parse_as_posix_shell():
    for script_name in ("entrypoint.sh", "update_db.sh"):
        result = subprocess.run(["sh", "-n", script_name], capture_output=True, text=True)
        assert result.returncode == 0, f"{script_name}: {result.stderr.strip()}"


# ── functional simulation of the container lifecycle ───────────────────────────
#
# The init branches above are asserted as text, but the real failure modes are runtime
# ones (a link that already exists, a directory layout the branch didn't anticipate).
# These run the actual scripts with the container paths rewritten to a temp dir and
# nfl-mcp / supercronic stubbed on PATH.

_NFL_MCP_STUB = (
    '#!/bin/sh\n'
    'echo "[stub nfl-mcp $*]"\n'
    'if [ -n "${NFL_MCP_DB_PATH:-}" ]; then printf INGESTED > "$NFL_MCP_DB_PATH"; fi\n'
)


def _sim(tmp_path, *, with_seed=True, existing_baked=False):
    """Materialise a fake container root and return (entrypoint, update_db, paths)."""
    root = Path(__file__).resolve().parents[1]
    data, seed, bin_dir = tmp_path / "data", tmp_path / "seed", tmp_path / "bin"
    data.mkdir(parents=True)
    seed.mkdir()
    bin_dir.mkdir()

    if with_seed:
        (seed / "nflread.duckdb").write_text("SEED_V1")
    if existing_baked:
        # An image that baked the database straight into /data (pre-seed layout).
        (data / "nflread.duckdb").write_text("OLD_BAKED")

    (bin_dir / "nfl-mcp").write_text(_NFL_MCP_STUB)
    (bin_dir / "supercronic").write_text('#!/bin/sh\necho "[stub supercronic]"\n')
    for tool in ("nfl-mcp", "supercronic"):
        (bin_dir / tool).chmod(0o755)

    entrypoint = (root / "entrypoint.sh").read_text()
    for before, after in (
        ('DB_DIR="/data"', f'DB_DIR="{data}"'),
        ('SEED="/seed/nflread.duckdb"', f'SEED="{seed}/nflread.duckdb"'),
        ("mkdir -p /var/log/nflmcp 2>/dev/null || true", ":"),
        ("exec nfl-mcp serve --host 0.0.0.0 --port 8000", 'echo "[serve started]"'),
    ):
        assert before in entrypoint, f"entrypoint.sh no longer contains: {before}"
        entrypoint = entrypoint.replace(before, after)

    updater = (root / "update_db.sh").read_text().replace('DB_DIR="/data"', f'DB_DIR="{data}"')

    entrypoint_path = tmp_path / "entrypoint.sh"
    updater_path = tmp_path / "update_db.sh"
    entrypoint_path.write_text(entrypoint)
    updater_path.write_text(updater)
    return entrypoint_path, updater_path, data, bin_dir


def _run(script, bin_dir):
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    result = subprocess.run(["sh", str(script)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout


class TestContainerLifecycle:
    def test_first_start_seeds_slot_a_and_serves(self, tmp_path):
        entrypoint, _, data, bin_dir = _sim(tmp_path)
        out = _run(entrypoint, bin_dir)

        assert "Copying baked seed into slot A" in out
        assert "[serve started]" in out
        assert (data / "nflread_a.duckdb").read_text() == "SEED_V1"
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"

    def test_restart_does_not_re_ingest_or_fail(self, tmp_path):
        entrypoint, _, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        (data / "nflread_a.duckdb").write_text("KEEP_ME")

        out = _run(entrypoint, bin_dir)
        assert "Copying baked seed" not in out
        assert (data / "nflread_a.duckdb").read_text() == "KEEP_ME"

    def test_adopts_a_baked_database_from_an_older_image(self, tmp_path):
        entrypoint, _, data, bin_dir = _sim(tmp_path, existing_baked=True)
        out = _run(entrypoint, bin_dir)

        # Must adopt, not crash on `ln -s` over an existing file, and not re-ingest.
        assert "Adopted existing /data database as slot A" in out
        assert (data / "nflread_a.duckdb").read_text() == "OLD_BAKED"
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"

    def test_falls_back_to_cold_ingest_without_a_seed(self, tmp_path):
        entrypoint, _, data, bin_dir = _sim(tmp_path, with_seed=False)
        out = _run(entrypoint, bin_dir)

        assert "No seed found; ingesting into slot A" in out
        assert (data / "nflread_a.duckdb").read_text() == "INGESTED"

    def test_updater_alternates_slots_and_keeps_serving(self, tmp_path):
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)

        _run(updater, bin_dir)
        assert os.readlink(data / "nflread.duckdb") == "nflread_b.duckdb"
        assert (data / "nflread_a.duckdb").exists()   # previous copy still on disk

        _run(updater, bin_dir)
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"
        assert (data / "nflread_b.duckdb").exists()

    def test_updater_refuses_to_run_without_the_symlink(self, tmp_path):
        _, updater, data, bin_dir = _sim(tmp_path)
        env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
        result = subprocess.run(["sh", str(updater)], capture_output=True, text=True, env=env)

        assert result.returncode != 0
        assert "is not a symlink" in result.stdout
