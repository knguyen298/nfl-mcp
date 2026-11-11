"""Container image + database-updater wiring tests.

These assert on file contents rather than building images: the fork's auto-updating
database once regressed to a no-op simply because the published Dockerfile stopped using
entrypoint.sh, and a text assertion catches exactly that class of drift.
"""

import fcntl
import os
import re
import shutil
import subprocess
import sys
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


def test_both_dockerfiles_open_the_data_and_log_directories():
    # The image must run with an arbitrary `--user`, so the directory the A/B slots live in
    # has to be writable by that user, not only the log directory. Dockerfile.dev once kept
    # the chmod for the log directory alone, which made the dev container fail to seed.
    for name in ("Dockerfile", "Dockerfile.dev"):
        assert "chmod 777 /data /var/log/nflmcp" in _read(name), name


def test_dev_dockerfile_matches_the_published_runtime_wiring():
    # The only intended difference is the baked seed. Everything the published image gives a
    # running container — scripts, scheduler, directories, env, port, entrypoint — has to be
    # in the dev image too, or testing the dev image proves nothing about the shipped one.
    published, dev = _read("Dockerfile"), _read("Dockerfile.dev")

    shared = [
        "COPY entrypoint.sh /entrypoint.sh",
        "COPY update_db.sh /update_db.sh",
        "COPY validate_db.sh /validate_db.sh",
        "chmod +x /entrypoint.sh /update_db.sh /validate_db.sh",
        "chmod 777 /data /var/log/nflmcp",
        "ENV NFL_MCP_DB_PATH=/data/nflread.duckdb",
        "ENV NFL_MCP_DUCKDB_MEMORY_LIMIT=512MB",
        "NFL_MCP_DUCKDB_THREADS=1",
        "NFL_MCP_DUCKDB_TEMP_DIR=/tmp/.duckdb_spill",
        "EXPOSE 8000",
        'ENTRYPOINT ["/entrypoint.sh"]',
        "supercronic",
    ]
    for line in shared:
        assert line in published, f"published Dockerfile lost: {line}"
        assert line in dev, f"Dockerfile.dev is missing: {line}"

    # And the one intended difference is still the only one: no seed baked into dev.
    assert "/seed" not in dev
    assert "nfl-mcp ingest" not in dev
    assert "nfl-mcp ingest --pre1999 ${INGEST_ARGS}" in published


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
        # The row check now lives in the shared validator; the build calls it inside the same
        # RUN, so a non-zero exit there still fails the build.
        assert "sh /src/validate_db.sh /seed/nflread.duckdb" in dockerfile
        assert dockerfile.index("nfl-mcp ingest --pre1999") < dockerfile.index(
            "sh /src/validate_db.sh /seed/nflread.duckdb"
        )
        assert "('plays','schedules')" in _read("validate_db.sh")
        # No `|| true` anywhere in the bake: a failing check has to stop the build.
        assert "|| true" not in dockerfile

    def test_seed_includes_pre1999_records_but_runtime_does_not(self):
        # The published image answers 1970+ standings out of the box, while a running
        # container never calls ESPN: the opt-in flag belongs to the build only.
        assert "nfl-mcp ingest --pre1999 ${INGEST_ARGS}" in _read("Dockerfile")
        for script in ("entrypoint.sh", "update_db.sh"):
            contents = _read(script)
            assert "nfl-mcp ingest" in contents
            assert "--pre1999" not in contents

    def test_dockerfile_and_entrypoint_agree_on_the_served_path(self):
        # The drift that disabled auto-updating was these two paths diverging.
        served = 'ENV NFL_MCP_DB_PATH=/data/nflread.duckdb'
        assert served in _read("Dockerfile")
        assert 'SYMLINK="$DB_DIR/nflread.duckdb"' in _read("entrypoint.sh")
        assert 'DB_DIR="/data"' in _read("entrypoint.sh")


# ── one validator for both paths that publish a database ───────────────────

class TestSharedValidator:
    """validate_db.sh is the single answer to "is this database safe to serve?"."""

    def test_checks_the_wal_and_the_core_tables(self):
        script = _read("validate_db.sh")

        assert script.startswith("#!/bin/sh\n")
        assert "set -eu" in script
        assert "pipefail" not in script
        # A database that was not closed cleanly is never safe to publish.
        assert 'if [ -e "$DB.wal" ]' in script
        assert "read_only=True" in script
        assert "('plays','schedules')" in script

    def test_both_publish_paths_call_the_one_copy(self):
        assert "sh /src/validate_db.sh /seed/nflread.duckdb" in _read("Dockerfile")

        updater = _read("update_db.sh")
        assert 'VALIDATE_DB="$(dirname "$0")/validate_db.sh"' in updater
        assert 'sh "$VALIDATE_DB" "$TARGET"' in updater
        assert 'if ! sh "$VALIDATE_DB" "$TARGET"; then' in updater
        # One copy of the table list: the callers ask, the validator knows what it checks.
        # Merge: main's comments mention plays, so only code lines are checked.
        code = "\n".join(line for line in updater.splitlines()
                         if not line.lstrip().startswith("#"))
        assert "plays" not in code
        assert "schedules" not in code

    def test_validator_is_copied_into_both_images(self):
        for name in ("Dockerfile", "Dockerfile.dev"):
            assert "COPY validate_db.sh /validate_db.sh" in _read(name), name

    def test_validator_parses_as_posix_shell(self):
        result = subprocess.run(["sh", "-n", "validate_db.sh"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr.strip()


# ── entrypoint: seed, then serve, then keep fresh ──────────────────────────────

class TestEntrypoint:
    def test_seeds_slot_a_instead_of_ingesting_on_startup(self):
        # A cold ingest at container start takes 10+ minutes and blocks readiness behind
        # any startup probe, so the seed must be copied instead. The copy fills a scratch
        # name and is renamed into place, so a seed copy that dies halfway is never the
        # file the server serves.
        entrypoint = _read("entrypoint.sh")
        assert 'cp "$SEED" "$SLOT_A_TMP"' in entrypoint
        assert 'mv "$SLOT_A_TMP" "$SLOT_A"' in entrypoint
        assert 'SEED="/seed/nflread.duckdb"' in entrypoint
        assert 'cp "$SEED" "$SLOT_A"' not in entrypoint

    def test_cold_ingest_is_only_the_no_seed_fallback(self):
        entrypoint = _read("entrypoint.sh")
        # The ingest writes the scratch name too, and only a successful one is renamed.
        assert 'NFL_MCP_DB_PATH="$SLOT_A_TMP" nfl-mcp ingest' in entrypoint
        assert 'elif [ -f "$SEED" ]' in entrypoint

    def test_keeps_a_live_symlink_instead_of_forcing_slot_a(self):
        # update_db.sh alternates the link between slots. Re-pointing it at slot A on
        # every start would serve the older slot after an update, and a restart during
        # an in-place rewrite of slot A would serve a half-written file.
        entrypoint = _read("entrypoint.sh")
        # Merge: written as "missing or dangling" so main's guard test reads it too.
        guard = '[ ! -L "$SYMLINK" ] || [ ! -e "$SYMLINK" ]'
        assert guard in entrypoint
        # The guard decides first; the re-link to slot A belongs to the repair branch.
        assert entrypoint.index(guard) < entrypoint.index('ln -sfn "$(basename "$SLOT_A")"')

    def test_clears_a_scratch_file_left_by_an_interrupted_start(self):
        entrypoint = _read("entrypoint.sh")
        assert 'rm -f "$SLOT_A_TMP" "$SLOT_A_TMP.wal"' in entrypoint
        assert entrypoint.index('rm -f "$SLOT_A_TMP"') < entrypoint.index('cp "$SEED" "$SLOT_A_TMP"')

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

    def test_notices_a_scheduler_that_died_and_keeps_serving(self):
        # Nothing supervises supercronic, so the only sign of a dead scheduler would be data
        # that quietly stops getting newer. Say so once, in the container log.
        entrypoint = _read("entrypoint.sh")

        assert "SCHEDULER_PID=$!" in entrypoint
        assert 'kill -0 "$SCHEDULER_PID"' in entrypoint
        assert "WARNING: supercronic is not running" in entrypoint
        # The grace period has to come before the check, and the check before the server:
        # a dead scheduler must never stop the container from serving the data it has.
        assert entrypoint.index("sleep") < entrypoint.index('kill -0 "$SCHEDULER_PID"')
        assert entrypoint.index("WARNING: supercronic is not running") < entrypoint.index(
            "exec nfl-mcp serve"
        )
        assert "exit 1" not in entrypoint
        assert "sleep 2" in entrypoint   # the grace period; _sim shortens it to keep tests fast
        # No supervisor was added: nothing waits on the scheduler or restarts it.
        assert " wait\n" not in entrypoint

    def test_scheduler_output_reaches_the_container_log(self):
        # supercronic is started without a redirect, so its own output (which job ran, which
        # one exited non-zero) goes to the container's stdout/stderr and shows in `docker
        # logs`. The updater's own output keeps going to the documented log file.
        entrypoint = _read("entrypoint.sh")
        assert "supercronic /tmp/crontab &" in entrypoint
        assert "supercronic /tmp/crontab >" not in entrypoint
        assert "/update_db.sh >> /var/log/nflmcp/nfl_update.log 2>&1" in entrypoint


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

    def test_space_requirement_covers_the_copy_and_the_wal(self):
        # The run needs the copy plus DuckDB's temp/spill files and the .wal it writes next
        # to the database, so the requirement is the active slot times a factor, not the
        # slot size plus a fixed guess. Measured on the filesystem $DB_DIR lives on.
        script = _read("update_db.sh")

        assert 'SPACE_FACTOR="${NFL_MCP_SPACE_FACTOR:-2}"' in script
        assert "NEEDED_KB=$(( ACTIVE_KB * SPACE_FACTOR ))" in script
        assert 'du -sk "$DB_DIR/$CURRENT"' in script
        assert 'df -Pk "$DB_DIR"' in script
        assert "1048576" not in script
        # Asked for before the copy, so a short volume cannot waste the run.
        assert script.index("SPACE_FACTOR") < script.index('cp "$DB_DIR/$CURRENT"')

    def test_clears_debris_from_a_crashed_run_before_it_writes(self):
        # One crashed run used to leave slot B's .wal and .tmp behind forever; the .wal is
        # what made a later update refuse to publish, and the files are what filled the disk.
        script = _read("update_db.sh")

        assert 'rm -f "$TARGET" "$TARGET.wal" "$TARGET.tmp"' in script
        assert script.index('rm -f "$TARGET"') < script.index("SPACE_FACTOR")
        assert script.index('rm -f "$TARGET"') < script.index('cp "$DB_DIR/$CURRENT"')

    def test_checks_the_new_slot_before_it_publishes_it(self):
        script = _read("update_db.sh")

        # A leftover .wal means the ingest died before DuckDB closed the file. One guard sits
        # right after the ingest, before the upgrade can replay the file, and one sits again
        # before the swap, after the steps that also write to the slot.
        assert script.count('if [ -e "$TARGET.wal" ]') == 2
        assert "did not close the database cleanly" in script
        ingest = script.index('NFL_MCP_DB_PATH="$TARGET" nfl-mcp ingest')
        upgrade = script.index('NFL_MCP_DB_PATH="$TARGET" nfl-mcp upgrade')
        guards = [i for i in range(len(script)) if script.startswith('if [ -e "$TARGET.wal" ]', i)]
        assert ingest < guards[0] < upgrade < guards[1]
        assert script.index('if [ -e "$TARGET.wal" ]') < script.index('ln -sfn "$(basename "$TARGET")"')
        # Validation and both WAL guards decide before the swap.
        assert script.index('sh "$VALIDATE_DB" "$TARGET"') < script.index('ln -sfn "$(basename "$TARGET")"')
        assert script.index('NFL_MCP_DB_PATH="$TARGET" nfl-mcp ingest') < script.index(
            'sh "$VALIDATE_DB" "$TARGET"'
        )

    def test_symlink_swap_is_what_publishes_new_data(self):
        # Documents why no restart is needed: a fresh read-only connection per query.
        assert "fresh read-only DuckDB connection per" in _read("update_db.sh")
        assert "read_only=True" in _read("nfl_mcp/database.py")

    def test_locks_one_run_at_a_time_before_it_reads_the_link(self):
        script = _read("update_db.sh")

        # Merge: main's lock is on the /data directory itself, so no lock file can be left
        # behind owned by another user.
        assert 'exec 9<"$DB_DIR"' in script
        assert "flock -n 9" in script
        # The lock has to be held before the script works out which slot it will delete.
        assert script.index("flock -n 9") < script.index('CURRENT=$(readlink "$SYMLINK")')
        assert script.index("flock -n 9") < script.index('rm -f "$TARGET"')
        # A skipped overlap is not a failure, so the cron log stays quiet about it.
        assert 'Another update is already running; skipping this run."\n    exit 0' in script
        # Debian ships flock(1) in util-linux, an Essential package, so the slim image has
        # it. If it is ever missing, failing beats every later run reporting a skip.
        assert "command -v flock" in script
        # A lock directory would stay behind after a killed container and block every
        # later run, so the script must lock an open file instead of making a directory.
        assert "mkdir" not in script

    def test_accepts_only_the_two_slot_names_and_checks_the_active_file(self):
        script = _read("update_db.sh")

        # Merge: an if/elif chain (main's tests read the first test), with no catch-all
        # branch that picks a slot: any other value ends the run.
        assert 'if [ "$CURRENT" = "nflread_a.duckdb" ]; then' in script
        assert 'elif [ "$CURRENT" = "nflread_b.duckdb" ]; then' in script
        assert "unexpected active link" in script
        assert "is not a regular file" in script
        # Both guards decide before the idle slot is deleted, so a rejected run removes
        # nothing. The old if/else sent every other value to the slot-A branch, which is
        # where a hand-made link would have the live database deleted.
        for guard in ("unexpected active link", "is not a regular file"):
            assert script.index(guard) < script.index('rm -f "$TARGET"')
        chain = script[script.index('if [ "$CURRENT" = "nflread_a.duckdb" ]'):]
        chain = chain[:chain.index("\nfi\n")]
        assert chain.count("TARGET=") == 2
        assert chain.index("else\n") < chain.index("unexpected active link") < chain.index("exit 1")


# ── build workflow: what triggers it, and what it accepts ───────────────────────


def _workflow(name=".github/workflows/docker-build.yml"):
    return _read(name)


def _paths_block(workflow):
    """The `paths:` list of the pull_request trigger, as text."""
    return workflow.split("paths:", 1)[1].split("\n\n", 1)[0]


class TestBuildWorkflow:
    """The job that proves the image still builds has to notice image changes and fail on a
    container that comes up broken."""

    def test_paths_filter_lists_every_file_the_image_copies(self):
        paths = _paths_block(_workflow())

        # Everything that changes the image or how it runs, not just the Dockerfiles.
        for name in _IMAGE_TRIGGERING_FILES:
            assert f"- {name}\n" in paths, f"a change to {name} would not trigger a build check"
        assert "- nfl_mcp/**\n" in paths

    def test_paths_filter_covers_the_scripts_both_dockerfiles_copy(self):
        # Derived from the images themselves, so a new COPYed script cannot be left out of
        # the filter by accident.
        copied = set()
        for name in ("Dockerfile", "Dockerfile.dev"):
            copied |= {
                p for p in re.findall(r"^COPY\s+(\S+)\s+/\S+", _read(name), re.M) if p != "."
            }

        assert {"entrypoint.sh", "update_db.sh", "validate_db.sh"} <= copied
        paths = _paths_block(_workflow())
        for name in sorted(copied):
            assert f"- {name}\n" in paths, f"{name} is copied into the image but not watched"

    def test_the_schedule_lives_in_a_file_the_filter_covers(self):
        # No crontab file is copied today: entrypoint.sh writes /tmp/crontab, and that file
        # is in the filter. If the schedule moves to its own file, list it in both places.
        entrypoint = _read("entrypoint.sh")
        assert "0 6 * * 4" in entrypoint
        assert "0 3 * * 0,1,3" in entrypoint
        assert "- entrypoint.sh\n" in _paths_block(_workflow())
        assert not Path("crontab").exists() or "- crontab\n" in _paths_block(_workflow())

    def test_smoke_step_requires_a_success_status(self):
        step = _workflow().split("Smoke test", 1)[1]

        assert 'if [ "$code" = "200" ]' in step
        # The check this replaced accepted any HTTP response at all, so a container that came
        # up answering 500 passed the build.
        assert '[ -n "$code" ]' not in step
        assert '"$code" != "000"' not in step
        # Success is decided by the loop, so the step cannot exit 0 on a bad response.
        assert "exit 0" not in step
        assert 'exit "$status"' in step

    def test_smoke_step_retries_for_a_bounded_time(self):
        step = _workflow().split("Smoke test", 1)[1]

        assert "for attempt in $(seq 1 30)" in step
        assert "sleep 2" in step
        assert "break" in step
        # A hung server must not hang the job: curl itself is given a per-call timeout.
        assert "-m 10" in step
        # And a failure says what the container did instead of passing quietly.
        assert "docker logs nfl-mcp-ci" in step

    def test_smoke_step_probes_only_endpoints_the_server_serves(self):
        # No invented health endpoint: every URL the step probes is a Route create_app makes.
        routes = set(re.findall(r'Route\("(/[^"]*)"', _read("nfl_mcp/server.py")))
        assert routes, "no routes found in nfl_mcp/server.py"

        probed = set(re.findall(r"http://localhost:8000(/[^\s\"']+)", _workflow()))
        assert probed, "the smoke step probes nothing"
        assert probed <= routes, f"probed {sorted(probed)} but the server serves {sorted(routes)}"

    def test_publish_workflow_still_has_no_path_filter(self):
        # docker.yml runs on release, schedule and dispatch, so a paths filter there would
        # only ever apply to `release` and is deliberately absent.
        assert "paths:" not in _read(".github/workflows/docker.yml")


# ── shell hygiene ──────────────────────────────────────────────────────────────

def test_database_update_scripts_are_posix_sh_compatible():
    for script_name in ("entrypoint.sh", "update_db.sh", "validate_db.sh"):
        script = _read(script_name)
        assert script.startswith("#!/bin/sh\n")
        assert "pipefail" not in script


def test_update_scripts_parse_as_posix_shell():
    for script_name in ("entrypoint.sh", "update_db.sh", "validate_db.sh"):
        result = subprocess.run(["sh", "-n", script_name], capture_output=True, text=True)
        assert result.returncode == 0, f"{script_name}: {result.stderr.strip()}"


# ── functional simulation of the container lifecycle ───────────────────────────
#
# The init branches above are asserted as text, but the real failure modes are runtime
# ones (a link that already exists, a directory layout the branch didn't anticipate).
# These run the actual scripts with the container paths rewritten to a temp dir and
# nfl-mcp / supercronic stubbed on PATH.

# Only `ingest` writes the database; `validate` exits with $STUB_VALIDATE_EXIT (default 0).
_NFL_MCP_STUB = (
    '#!/bin/sh\n'
    'echo "[stub nfl-mcp $*]"\n'
    'if [ "${1:-}" = validate ]; then exit "${STUB_VALIDATE_EXIT:-0}"; fi\n'
    'if [ "${1:-}" = ingest ] && [ -n "${NFL_MCP_DB_PATH:-}" ]; then\n'
    '    printf INGESTED > "$NFL_MCP_DB_PATH"\n'
    'fi\n'
)

# A tool that writes some bytes and then dies, standing in for a copy or an ingest
# interrupted halfway (SIGKILL, dropped network, full disk).
_PARTIAL_THEN_FAIL = {
    "cp": '#!/bin/sh\nprintf PARTIAL > "$2"\nexit 1\n',        # destination is argv 2
    "nfl-mcp": '#!/bin/sh\nprintf PARTIAL > "$NFL_MCP_DB_PATH"\nexit 1\n',
}

# `nfl-mcp ingest` that loads nothing and reports it (exit 1), as `nfl_mcp/ingest.py` does
# when every requested season failed. It writes no database at all, unlike the stub above.
_INGEST_FAILS_WITHOUT_WRITING = '#!/bin/sh\necho "[stub nfl-mcp] nothing loaded"\nexit 1\n'

# An ingest that fills the slot and then dies before DuckDB closes it: the case the WAL
# guard in update_db.sh exists for, since DuckDB replays that file only on a write open.
_INGEST_LEAVES_A_WAL = (
    '#!/bin/sh\n'
    'printf INGESTED > "$NFL_MCP_DB_PATH"\n'
    'printf WAL > "$NFL_MCP_DB_PATH.wal"\n'
)

# Same, and every `nfl-mcp` call is written to $STUB_CALL_LOG, so a test can see how far the
# updater got before it stopped.
_INGEST_LEAVES_A_WAL_AND_LOGS_CALLS = (
    '#!/bin/sh\n'
    'if [ -n "${STUB_CALL_LOG:-}" ]; then echo "$1" >> "$STUB_CALL_LOG"; fi\n'
    'if [ "${1:-}" = ingest ]; then\n'
    '    printf INGESTED > "$NFL_MCP_DB_PATH"\n'
    '    printf WAL > "$NFL_MCP_DB_PATH.wal"\n'
    'fi\n'
)

# An ingest that closes cleanly and an upgrade that does not: the upgrade opens the slot for
# writing, so this is the one step between the ingest guard and the swap that can leave a WAL.
_UPGRADE_LEAVES_A_WAL = (
    '#!/bin/sh\n'
    'if [ "${1:-}" = ingest ] && [ -n "${NFL_MCP_DB_PATH:-}" ]; then\n'
    '    printf INGESTED > "$NFL_MCP_DB_PATH"\n'
    'fi\n'
    'if [ "${1:-}" = upgrade ]; then\n'
    '    printf WAL > "${NFL_MCP_DB_PATH:-/dev/null}.wal"\n'
    'fi\n'
)

# Disk space is measured with du/df, so both are stubbed: a test then reasons about
# deliberate numbers instead of about whatever the CI runner's volume has left. The active
# slot reads as 1000 KB (so the default 2x factor asks for 2000 KB) on a volume with 1 GB.
_DU_STUB = '#!/bin/sh\necho 1000\n'
_DF_STUB_FMT = (
    '#!/bin/sh\n'
    'echo "Filesystem 1024-blocks Used Available Capacity Mounted on"\n'
    'echo "sim 2000000 1000000 {available} 66% /data"\n'
)
_DF_STUB = _DF_STUB_FMT.format(available=999000)

# validate_db.sh is called by name from next to the updater, so the simulation supplies it.
_VALIDATE_PASSES = '#!/bin/sh\necho "[stub validate $1]"\nexit 0\n'
_VALIDATE_FAILS = '#!/bin/sh\necho "[stub validate $1] no rows"\nexit 1\n'

# supercronic that stays up (the normal case) and the default stub below, which exits at
# once and so stands in for a scheduler that failed to start. The live one lets go of the
# captured stdout pipe before it blocks, or the test would wait on it.
_SUPERCRONIC_STAYS_UP = (
    '#!/bin/sh\n'
    'echo "[stub supercronic running]"\n'
    'exec >/dev/null 2>&1\n'
    'exec sleep 4\n'
)

# util-linux' flock(1) is what update_db.sh locks with, and Debian ships it as part of the
# base image. Hosts without it (macOS) get this shim, which covers the only form the
# script uses and takes the same kernel lock the real binary takes.
_FLOCK_SHIM = (
    'import fcntl, sys\n'
    'if sys.argv[1:2] != ["-n"] or len(sys.argv) != 3:\n'
    '    sys.exit("flock shim: only `flock -n <fd>` is supported")\n'
    'try:\n'
    '    fcntl.flock(int(sys.argv[2]), fcntl.LOCK_EX | fcntl.LOCK_NB)\n'
    'except OSError:\n'
    '    sys.exit(1)\n'
)


_IMAGE_TRIGGERING_FILES = (
    "Dockerfile",
    "Dockerfile.dev",
    ".dockerignore",
    "pyproject.toml",
    "entrypoint.sh",
    "update_db.sh",
    "validate_db.sh",
    "crontab",
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
    (bin_dir / "du").write_text(_DU_STUB)
    (bin_dir / "df").write_text(_DF_STUB)
    if shutil.which("flock") is None:
        # macOS has no flock(1); the updater needs one, so give the fake PATH a stand-in.
        (bin_dir / "flock").write_text("#!" + sys.executable + "\n" + _FLOCK_SHIM)
    for tool in bin_dir.iterdir():
        tool.chmod(0o755)

    entrypoint = (root / "entrypoint.sh").read_text()
    for before, after in (
        ('DB_DIR="/data"', f'DB_DIR="{data}"'),
        ('SEED="/seed/nflread.duckdb"', f'SEED="{seed}/nflread.duckdb"'),
        ("mkdir -p /var/log/nflmcp 2>/dev/null || true", ":"),
        # The scheduler grace period, shortened so ~20 simulated starts do not add half a
        # minute to the suite. Long enough for a stub that exits at once to be reaped.
        ("sleep 2", "sleep 0.4"),
        ("exec nfl-mcp serve --host 0.0.0.0 --port 8000", 'echo "[serve started]"'),
    ):
        assert before in entrypoint, f"entrypoint.sh no longer contains: {before}"
        entrypoint = entrypoint.replace(before, after)

    updater = (root / "update_db.sh").read_text().replace('DB_DIR="/data"', f'DB_DIR="{data}"')

    entrypoint_path = tmp_path / "entrypoint.sh"
    updater_path = tmp_path / "update_db.sh"
    entrypoint_path.write_text(entrypoint)
    updater_path.write_text(updater)
    # update_db.sh looks for the validator next to itself, which is how /validate_db.sh
    # sits next of /update_db.sh in the image.
    (tmp_path / "validate_db.sh").write_text(_VALIDATE_PASSES)
    return entrypoint_path, updater_path, data, bin_dir


def _env(bin_dir, env_extra=None, **extra_env):
    """Inherited environment minus NFL_MCP_DB_PATH, so a stub can never write to a path the
    outer test run exported."""
    env = {k: v for k, v in os.environ.items() if k != "NFL_MCP_DB_PATH"}
    return {**env, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            **(env_extra or {}), **extra_env}


def _run_raw(script, bin_dir, env_extra=None, **extra_env):
    """Run a script and hand back the result, for the paths expected to fail."""
    return subprocess.run(["sh", str(script)], capture_output=True, text=True,
                          env=_env(bin_dir, env_extra, **extra_env))


def _run(script, bin_dir, env_extra=None, **extra_env):
    result = _run_raw(script, bin_dir, env_extra, **extra_env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _stub_failing_tool(bin_dir, name):
    """Put `name` on PATH so it writes its destination halfway, then exits non-zero."""
    tool = bin_dir / name
    tool.write_text(_PARTIAL_THEN_FAIL[name])
    tool.chmod(0o755)


def _stub_tool(bin_dir, name, script):
    """Replace a command on the fake PATH with `script`."""
    tool = bin_dir / name
    tool.write_text(script)
    tool.chmod(0o755)
    return tool


def _stub_free_space(bin_dir, available_kb):
    """Report `available_kb` KB free, whatever the host volume actually has."""
    _stub_tool(bin_dir, "df", _DF_STUB_FMT.format(available=available_kb))


def _stub_validator(tmp_path, script):
    """Replace the simulated validator that sits next to the simulated updater."""
    (tmp_path / "validate_db.sh").write_text(script)


def _hold_lock(data):
    """Take update_db.sh's own lock (in the rewritten $DB_DIR) from this process.

    Returns the open handle; closing it releases the lock, the same way the kernel
    releases it when the updater dies.
    """
    handle = _DirLock(os.open(data, os.O_RDONLY))   # the script locks the directory itself
    try:
        fcntl.flock(handle.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise
    return handle


class _DirLock:
    """An open descriptor on a directory. Closing it releases a flock taken on it."""

    def __init__(self, fd):
        self.fd = fd

    def close(self):
        os.close(self.fd)


def _lock_is_free(data):
    """True when no process holds the updater's lock. Never blocks."""
    try:
        handle = _hold_lock(data)
    except OSError:
        return False
    handle.close()
    return True


def _slots(data):
    """Contents and modification times of both slots, for "nothing was touched" checks."""
    state = {}
    for name in (
        "nflread_a.duckdb",
        "nflread_a.duckdb.wal",
        "nflread_a.duckdb.tmp",
        "nflread_b.duckdb",
        "nflread_b.duckdb.wal",
        "nflread_b.duckdb.tmp",
    ):
        path = data / name
        state[name] = (path.read_bytes(), path.stat().st_mtime_ns) if path.exists() else None
    return state


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

    def test_restart_after_an_update_keeps_the_link_on_slot_b(self, tmp_path):
        # The regression: a restart re-linked to slot A, so the server went backwards to
        # the pre-update copy and the next update deleted the newer slot B.
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _run(updater, bin_dir)
        assert os.readlink(data / "nflread.duckdb") == "nflread_b.duckdb"
        slot_b = data / "nflread_b.duckdb"
        contents, modified = slot_b.read_text(), slot_b.stat().st_mtime_ns

        out = _run(entrypoint, bin_dir)

        assert os.readlink(data / "nflread.duckdb") == "nflread_b.duckdb"
        assert (slot_b.read_text(), slot_b.stat().st_mtime_ns) == (contents, modified)
        assert "Copying baked seed" not in out
        assert "ingesting into slot A" not in out

    def test_restart_on_slot_b_tolerates_a_missing_slot_a(self, tmp_path):
        # Slot A is only the first-start seed; a live link to B must not trigger a
        # re-seed just because A is gone.
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _run(updater, bin_dir)
        (data / "nflread_a.duckdb").unlink()

        out = _run(entrypoint, bin_dir)

        assert os.readlink(data / "nflread.duckdb") == "nflread_b.duckdb"
        assert not (data / "nflread_a.duckdb").exists()
        assert "Copying baked seed" not in out

    def test_restart_on_slot_b_ignores_a_half_written_slot_a(self, tmp_path):
        # A restart while the updater rewrote slot A must keep serving B, not adopt the
        # truncated A (and its leftover WAL) as active.
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _run(updater, bin_dir)
        (data / "nflread_a.duckdb").write_text("HALF_WRITTEN")
        (data / "nflread_a.duckdb.wal").write_text("WAL")

        _run(entrypoint, bin_dir)

        assert os.readlink(data / "nflread.duckdb") == "nflread_b.duckdb"
        assert (data / "nflread_a.duckdb").read_text() == "HALF_WRITTEN"

    def test_dangling_symlink_falls_back_to_slot_a(self, tmp_path):
        # Link says B, B is gone (e.g. the volume was reset): repair to A, don't serve a
        # target that does not exist.
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _run(updater, bin_dir)
        (data / "nflread_b.duckdb").unlink()

        out = _run(entrypoint, bin_dir)

        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"
        assert (data / "nflread_a.duckdb").read_text() == "SEED_V1"
        assert "keeping existing link" not in out

    def test_dangling_symlink_with_no_slot_a_is_seeded(self, tmp_path):
        entrypoint, _, data, bin_dir = _sim(tmp_path)
        os.symlink("nflread_b.duckdb", data / "nflread.duckdb")

        out = _run(entrypoint, bin_dir)

        assert "Copying baked seed into slot A" in out
        assert (data / "nflread_a.duckdb").read_text() == "SEED_V1"
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"

    def test_failed_seed_copy_leaves_no_slot_a(self, tmp_path):
        entrypoint, _, data, bin_dir = _sim(tmp_path)
        _stub_failing_tool(bin_dir, "cp")

        result = _run_raw(entrypoint, bin_dir)

        assert result.returncode != 0
        assert not (data / "nflread_a.duckdb").exists()
        assert (data / "nflread_a.duckdb.tmp").read_text() == "PARTIAL"

    def test_failed_cold_ingest_leaves_no_slot_a(self, tmp_path):
        entrypoint, _, data, bin_dir = _sim(tmp_path, with_seed=False)
        _stub_failing_tool(bin_dir, "nfl-mcp")

        result = _run_raw(entrypoint, bin_dir)

        assert result.returncode != 0
        assert not (data / "nflread_a.duckdb").exists()
        assert (data / "nflread_a.duckdb.tmp").read_text() == "PARTIAL"

    def test_updater_alternates_slots_and_keeps_serving(self, tmp_path):
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)

        _run(updater, bin_dir)
        assert os.readlink(data / "nflread.duckdb") == "nflread_b.duckdb"
        assert (data / "nflread_a.duckdb").exists()   # previous copy still on disk

        _run(updater, bin_dir)
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"
        assert (data / "nflread_b.duckdb").exists()

    def test_updater_makes_the_new_slot_read_only_and_can_reuse_it(self, tmp_path):
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)

        _run(updater, bin_dir)
        assert (data / "nflread_b.duckdb").stat().st_mode & 0o222 == 0

        # B is now active and read-only; the next pass copies it into A and must still write.
        _run(updater, bin_dir)
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"
        assert (data / "nflread_a.duckdb").read_text() == "INGESTED"

    def test_updater_keeps_the_active_slot_when_validation_fails(self, tmp_path):
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        result = subprocess.run(["sh", str(updater)], capture_output=True, text=True,
                                env=_env(bin_dir, STUB_VALIDATE_EXIT="1"))

        assert result.returncode != 0
        assert "validation failed" in result.stdout
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"
        assert (data / "nflread_a.duckdb").read_text() == "SEED_V1"

    def test_restart_after_an_update_keeps_the_swapped_slot(self, tmp_path):
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _run(updater, bin_dir)

        _run(entrypoint, bin_dir)
        assert os.readlink(data / "nflread.duckdb") == "nflread_b.duckdb"

    def test_cold_ingest_tolerates_failed_datasets(self, tmp_path):
        entrypoint, _, _, bin_dir = _sim(tmp_path, with_seed=False)
        assert "[stub nfl-mcp ingest --allow-failures]" in _run(entrypoint, bin_dir)

    def test_updater_refuses_to_run_without_the_symlink(self, tmp_path):
        _, updater, data, bin_dir = _sim(tmp_path)
        result = subprocess.run(["sh", str(updater)], capture_output=True, text=True,
                                env=_env(bin_dir))

        assert result.returncode != 0
        assert "is not a symlink" in result.stdout

    def test_second_update_skips_while_another_holds_the_lock(self, tmp_path):
        # A cron run that overlaps the boot-time update must not touch the idle slot.
        import fcntl

        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        fd = os.open(data, os.O_RDONLY)   # the script locks the directory itself
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            out = _run(updater, bin_dir)   # exits 0, like a run with nothing to do

            assert "Another update is already running" in out
            assert "[stub nfl-mcp" not in out
            assert not (data / "nflread_b.duckdb").exists()
            assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"
        finally:
            os.close(fd)
        # Lock released: the next run proceeds.
        _run(updater, bin_dir)
        assert os.readlink(data / "nflread.duckdb") == "nflread_b.duckdb"

    def test_updater_skips_while_another_run_holds_the_lock(self, tmp_path):
        # Two crontab entries, or cron plus a manual `docker exec`, otherwise pick the
        # same idle slot and delete the file the other one is writing.
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        link, slots = os.readlink(data / "nflread.duckdb"), _slots(data)

        holder = _hold_lock(data)
        try:
            result = _run_raw(updater, bin_dir)
        finally:
            holder.close()

        assert result.returncode == 0                      # an overlap is not a failure
        assert "Another update is already running" in result.stdout
        assert os.readlink(data / "nflread.duckdb") == link   # nothing swapped
        assert _slots(data) == slots                        # nothing deleted or copied
        assert not (data / "nflread_b.duckdb").exists()

    def test_updater_releases_the_lock_when_it_finishes(self, tmp_path):
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _run(updater, bin_dir)

        assert _lock_is_free(data)

    def test_updater_releases_the_lock_when_the_ingest_fails(self, tmp_path):
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _stub_failing_tool(bin_dir, "nfl-mcp")

        assert _run_raw(updater, bin_dir).returncode != 0
        assert _lock_is_free(data)

    def test_updater_refuses_a_link_that_holds_an_absolute_path(self, tmp_path):
        # readlink returns the target verbatim, so an absolute link is not a slot name.
        # The old else-branch read it as "not slot A" and deleted slot A: the live file.
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        target = data / "nflread_a.duckdb"
        os.remove(data / "nflread.duckdb")
        os.symlink(str(target), data / "nflread.duckdb")
        slots = _slots(data)

        result = _run_raw(updater, bin_dir)

        assert result.returncode == 1
        assert "unexpected active link" in result.stdout
        assert str(target) in result.stdout                  # the error names what it saw
        assert _slots(data) == slots                         # slot A survived
        assert os.readlink(data / "nflread.duckdb") == str(target)

    def test_updater_refuses_a_link_to_an_unexpected_name(self, tmp_path):
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _run(updater, bin_dir)                               # live on slot B now
        os.remove(data / "nflread.duckdb")
        os.symlink("active.duckdb", data / "nflread.duckdb")  # made by hand
        slots = _slots(data)

        result = _run_raw(updater, bin_dir)

        assert result.returncode == 1
        assert "unexpected active link" in result.stdout
        assert "active.duckdb" in result.stdout
        assert _slots(data) == slots                         # both slots untouched
        assert (data / "nflread_a.duckdb").exists()
        assert (data / "nflread_b.duckdb").exists()

    def test_updater_refuses_when_the_active_slot_file_is_missing(self, tmp_path):
        # Slot A is the idle slot here. Deleting it before noticing that B is gone would
        # have removed the only database on disk, then failed the copy anyway.
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _run(updater, bin_dir)
        (data / "nflread_b.duckdb").unlink()
        slots = _slots(data)

        result = _run_raw(updater, bin_dir)

        assert result.returncode == 1
        assert "active slot" in result.stdout and "is not a regular file" in result.stdout
        assert _slots(data) == slots                         # nothing deleted
        assert (data / "nflread_a.duckdb").read_text() == "SEED_V1"
        assert os.readlink(data / "nflread.duckdb") == "nflread_b.duckdb"

    def test_updater_keeps_the_link_on_the_old_slot_when_the_ingest_fails(self, tmp_path):
        # A half-built idle slot must never be published: the server keeps the old data.
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _stub_failing_tool(bin_dir, "nfl-mcp")

        result = _run_raw(updater, bin_dir)

        assert result.returncode != 0
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"
        assert (data / "nflread_a.duckdb").read_text() == "SEED_V1"
        assert (data / "nflread_b.duckdb").read_text() == "PARTIAL"


# ── things that go wrong at runtime: disk, a crashed run, a bad new slot ────────


class TestContainerLifecycleKeepsServing:
    """Every guard has to leave the container serving the database it already had."""

    def test_dead_scheduler_warns_and_still_serves(self, tmp_path):
        # The stub supercronic _sim installs exits at once, which is a scheduler that never
        # started. Losing the schedule is bad; refusing to serve the data is worse.
        entrypoint, _, data, bin_dir = _sim(tmp_path)
        out = _run(entrypoint, bin_dir)

        assert "WARNING: supercronic is not running" in out
        assert "[serve started]" in out
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"

    def test_a_scheduler_that_stays_up_is_not_warned_about(self, tmp_path):
        entrypoint, _, data, bin_dir = _sim(tmp_path)
        _stub_tool(bin_dir, "supercronic", _SUPERCRONIC_STAYS_UP)

        out = _run(entrypoint, bin_dir)

        assert "WARNING" not in out
        assert "Scheduler started (pid" in out
        assert "[serve started]" in out
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"

    def test_updater_stops_for_too_little_disk_and_touches_nothing(self, tmp_path):
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        link, slots = os.readlink(data / "nflread.duckdb"), _slots(data)

        # du reads the active slot as 1000 KB, so the default 2x factor asks for 2000 KB.
        _stub_free_space(bin_dir, 1000)
        result = _run_raw(updater, bin_dir)

        assert result.returncode != 0
        assert "need 2000KB free" in result.stdout
        assert "2x the 1000KB active slot" in result.stdout
        assert os.readlink(data / "nflread.duckdb") == link
        assert _slots(data) == slots
        assert not (data / "nflread_b.duckdb").exists()      # nothing copied

    def test_the_space_factor_is_overridable(self, tmp_path):
        # Same volume, same slot: 1500 KB free is short at 2x and enough at 1x.
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _stub_free_space(bin_dir, 1500)

        assert _run_raw(updater, bin_dir).returncode != 0
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"

        out = _run(updater, bin_dir, {"NFL_MCP_SPACE_FACTOR": "1"})
        assert "Update complete" in out
        assert os.readlink(data / "nflread.duckdb") == "nflread_b.duckdb"

    def test_updater_refuses_to_publish_a_slot_with_a_wal(self, tmp_path):
        # An ingest that dies before DuckDB closes the file leaves <slot>.wal. Swapping to it
        # would serve the last checkpoint and quietly hide everything after it.
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _stub_tool(bin_dir, "nfl-mcp", _INGEST_LEAVES_A_WAL)

        result = _run_raw(updater, bin_dir)

        assert result.returncode != 0
        assert "nflread_b.duckdb.wal exists" in result.stdout
        assert "Keeping nflread_a.duckdb active" in result.stdout
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"
        assert (data / "nflread_a.duckdb").read_text() == "SEED_V1"
        assert (data / "nflread_b.duckdb.wal").exists()

    def test_the_wal_guard_runs_before_the_schema_upgrade(self, tmp_path):
        """`nfl-mcp upgrade` opens the slot for writing, and that replays and deletes the WAL.

        A guard after the upgrade cannot tell that the ingest never closed its file: by then
        the evidence is gone. So the updater must refuse the run right after the ingest, and
        must not spend the upgrade on a slot it will not publish.
        """
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _stub_tool(bin_dir, "nfl-mcp", _INGEST_LEAVES_A_WAL_AND_LOGS_CALLS)
        calls = tmp_path / "calls.log"

        result = _run_raw(updater, bin_dir, STUB_CALL_LOG=str(calls))

        assert result.returncode != 0
        assert "nflread_b.duckdb.wal exists after the ingest" in result.stdout
        assert calls.read_text().split() == ["ingest"]     # the upgrade was never asked
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"

    def test_a_wal_the_upgrade_left_also_blocks_the_swap(self, tmp_path):
        """The second guard stays, because the upgrade writes to the slot too."""
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _stub_tool(bin_dir, "nfl-mcp", _UPGRADE_LEAVES_A_WAL)

        result = _run_raw(updater, bin_dir)

        assert result.returncode != 0
        assert "nflread_b.duckdb.wal exists after the schema upgrade" in result.stdout
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"

    def test_updater_clears_debris_from_a_crashed_run(self, tmp_path):
        # Without this, the files one crash left behind would block every later update: the
        # .wal trips the guard above and the half-built slot is what fills the volume.
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        for name, body in (
            ("nflread_b.duckdb", "STALE"),
            ("nflread_b.duckdb.wal", "STALE_WAL"),
            ("nflread_b.duckdb.tmp", "STALE_TMP"),
        ):
            (data / name).write_text(body)

        out = _run(updater, bin_dir)

        assert "Update complete" in out
        assert (data / "nflread_b.duckdb").read_text() == "INGESTED"
        assert not (data / "nflread_b.duckdb.wal").exists()
        assert not (data / "nflread_b.duckdb.tmp").exists()
        assert os.readlink(data / "nflread.duckdb") == "nflread_b.duckdb"

    def test_updater_keeps_the_active_slot_when_validation_fails(self, tmp_path):
        # An ingest can exit 0 having loaded nothing. The same check the image build runs on
        # the seed has to hold the swap.
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _stub_validator(tmp_path, _VALIDATE_FAILS)

        result = _run_raw(updater, bin_dir)

        assert result.returncode != 0
        assert "failed validation" in result.stdout
        assert "Keeping nflread_a.duckdb active" in result.stdout
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"
        assert (data / "nflread_a.duckdb").read_text() == "SEED_V1"

    def test_updater_refuses_to_publish_when_the_validator_is_missing(self, tmp_path):
        # No validator means no proof the new slot has rows, so the swap is not worth it.
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        (tmp_path / "validate_db.sh").unlink()

        result = _run_raw(updater, bin_dir)

        assert result.returncode != 0
        assert "validator" in result.stdout and "is missing" in result.stdout
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"

    def test_updater_keeps_the_link_when_the_ingest_exits_non_zero(self, tmp_path):
        # `nfl-mcp ingest` exits 1 when a run loaded nothing at all (network down). Under
        # `set -eu` that has to stop the run before the swap, and the lock has to be freed.
        entrypoint, updater, data, bin_dir = _sim(tmp_path)
        _run(entrypoint, bin_dir)
        _stub_tool(bin_dir, "nfl-mcp", _INGEST_FAILS_WITHOUT_WRITING)

        result = _run_raw(updater, bin_dir)

        assert result.returncode != 0
        assert os.readlink(data / "nflread.duckdb") == "nflread_a.duckdb"
        assert (data / "nflread_a.duckdb").read_text() == "SEED_V1"
        assert _lock_is_free(data)


class TestRootStartDropsPrivileges:
    """entrypoint.sh started as root chowns a foreign-owned /data, then re-runs as nflmcp.

    Nothing runs as root: `id`, `find`, `chown` and `setpriv` are stubs. The `find` stub reports
    the directory itself as foreign-owned unless the owner is nflmcp.
    """

    @staticmethod
    def _stubs(bin_dir, tmp_path, *, owner):
        log = tmp_path / "calls.log"
        stubs = {
            "id": '[ "$1" = -u ] && echo 0\n',
            "find": f'[ {owner} = nflmcp ] || echo "$1"\n',
            "chown": 'echo "chown $*" >> "$CALL_LOG"\n',
            "setpriv": 'echo "setpriv $*" >> "$CALL_LOG"\n',
        }
        for name, body in stubs.items():
            (bin_dir / name).write_text("#!/bin/sh\n" + body)
            (bin_dir / name).chmod(0o755)
        return log

    def test_foreign_owner_is_chowned_then_script_reruns_as_nflmcp(self, tmp_path):
        entrypoint, _, data, bin_dir = _sim(tmp_path)
        log = self._stubs(bin_dir, tmp_path, owner="root")
        out = _run(entrypoint, bin_dir, CALL_LOG=str(log))

        calls = log.read_text().splitlines()
        assert calls == [
            f"chown -R nflmcp:nflmcp {data}",
            f"setpriv --reuid=nflmcp --regid=nflmcp --init-groups {entrypoint}",
        ]
        assert "changing owner" in out
        # The stub setpriv does not re-run the script, so nothing below the drop executed.
        assert "[serve started]" not in out
        assert not (data / "nflread.duckdb").exists()

    def test_correct_owner_is_not_chowned_again(self, tmp_path):
        entrypoint, _, _, bin_dir = _sim(tmp_path)
        log = self._stubs(bin_dir, tmp_path, owner="nflmcp")
        _run(entrypoint, bin_dir, CALL_LOG=str(log))

        calls = log.read_text().splitlines()
        assert len(calls) == 1 and calls[0].startswith("setpriv --reuid=nflmcp")

    def test_non_root_start_skips_the_drop(self, tmp_path):
        entrypoint, _, data, bin_dir = _sim(tmp_path)
        out = _run(entrypoint, bin_dir)   # real `id -u` is not 0 here
        assert "[serve started]" in out and "changing owner" not in out
