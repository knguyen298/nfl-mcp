#!/bin/sh
set -eu

# Publishes a newer database without a gap and without ever serving a half-written file.
# The $DB_DIR/nflread.duckdb symlink is the state: its target names the active slot. Read
# it once, accept only the two slot names, write the *other* slot, then rename the link.
# Two slots also keep the one DuckDB writer per file apart from the server's read-only
# connections, which is why nothing here ever writes to the active slot.

DB_DIR="/data"
SYMLINK="$DB_DIR/nflread.duckdb"

# One update at a time: a cron run that starts while the boot-time update is still working
# would pick the same idle slot and delete the file the first run has open. The lock is on
# the directory itself, so no lock file can be left behind owned by another user. It sits
# on an open file descriptor, so the kernel releases it when this process dies.
exec 9<"$DB_DIR"
if ! command -v flock >/dev/null 2>&1; then
    # Without flock every run would look like an overlap and skip. Fail where the log shows it.
    echo "[$(date -u)] ERROR: flock is missing, refusing to update without a lock"
    exit 1
fi
if ! flock -n 9; then
    echo "[$(date -u)] Another update is already running; skipping this run."
    exit 0
fi

if [ ! -L "$SYMLINK" ]; then
    echo "[$(date -u)] ERROR: $SYMLINK is not a symlink. Has entrypoint.sh run first-run init?"
    exit 1
fi

CURRENT=$(readlink "$SYMLINK")

# Only these two values are legal, and the check runs before any `rm`: the slot picked
# below is the one this script deletes, so a loose fallback (an absolute link target, a
# hand-made name, a typo) would delete the live database instead of the idle one.
if [ "$CURRENT" = "nflread_a.duckdb" ]; then
    TARGET="$DB_DIR/nflread_b.duckdb"
elif [ "$CURRENT" = "nflread_b.duckdb" ]; then
    TARGET="$DB_DIR/nflread_a.duckdb"
else
    echo "[$(date -u)] ERROR: unexpected active link '$CURRENT', expected nflread_a.duckdb or nflread_b.duckdb"
    exit 1
fi

# The active slot is the source of the copy and the fallback if this run fails, so it has
# to be a real file before the idle slot is dropped.
if [ ! -f "$DB_DIR/$CURRENT" ]; then
    echo "[$(date -u)] ERROR: active slot '$DB_DIR/$CURRENT' is not a regular file"
    exit 1
fi

# The ingest writes the datasets that failed to load here, one per line.
FAILED_LIST="$TARGET.failed"

echo "[$(date -u)] Active DB : $CURRENT"
echo "[$(date -u)] Target DB : $TARGET"

# Clear the files a run left behind when it died partway: the half-built idle slot, the
# WAL file, the scratch file and the failure list. If they stay, they block every later
# update and fill the volume. Removing them first also frees the space measured below.
rm -f "$TARGET" "$TARGET.wal" "$TARGET.tmp" "$FAILED_LIST"

# Ask for the space this run needs before it writes anything:
#   * one full copy of the active slot, into the idle slot, and
#   * room for the WAL file DuckDB keeps next to the database while the load writes.
# The active slot is the only size a running container can measure, so the requirement is
# that size times a factor. Set NFL_MCP_SPACE_FACTOR to a larger number when the volume
# needs more headroom. DuckDB writes its spill files to the directory in
# NFL_MCP_DUCKDB_TEMP_DIR, which both the ingest and a query connection read. The image
# sets it to /tmp/.duckdb_spill, on a different filesystem, so this check covers the copy
# and the WAL on the filesystem the server reads.
SPACE_FACTOR="${NFL_MCP_SPACE_FACTOR:-2}"
ACTIVE_KB=$(du -sk "$DB_DIR/$CURRENT" | cut -f1)
NEEDED_KB=$(( ACTIVE_KB * SPACE_FACTOR ))
FREE_KB=$(df -Pk "$DB_DIR" | awk 'NR==2 {print $4}')
if [ "$FREE_KB" -lt "$NEEDED_KB" ]; then
    echo "[$(date -u)] ERROR: need ${NEEDED_KB}KB free on $DB_DIR (${SPACE_FACTOR}x the ${ACTIVE_KB}KB active slot), have ${FREE_KB}KB. Skipping this run."
    exit 1
fi

echo "[$(date -u)] Copying active DB to target slot..."
cp "$DB_DIR/$CURRENT" "$TARGET"
# cp keeps the read-only mode the active slot got when it was swapped in; the ingest
# needs to write to this copy.
chmod u+w "$TARGET"

# Incremental: completed seasons are skipped; the current season and static
# snapshot tables (rankings, players, contracts) are re-fetched every run.
# Exit 3 means the run finished but some datasets or seasons failed to load. Each of them
# rolled back and keeps the rows of the active slot this copy came from, so the update
# goes on: the schema upgrade below must not wait for every download to succeed, and
# validate still decides the swap. Any other non-zero exit is a fatal error and stops here.
# A run where at least one fetch failed and none put rows in the database also exits 3: the
# copy then holds the same data as the active slot, so the checks below pass or fail it
# exactly as they would the active slot. A run where every source answered with no rows is
# not a failure and exits 0.
echo "[$(date -u)] Running incremental ingest..."
INGEST_RC=0
NFL_MCP_DB_PATH="$TARGET" nfl-mcp ingest --failures-file "$FAILED_LIST" || INGEST_RC=$?
if [ "$INGEST_RC" -eq 3 ]; then
    FAILED=$(awk 'NR > 1 { printf ", " } { printf "%s", $0 }' "$FAILED_LIST" 2>/dev/null || true)
    echo "[$(date -u)] WARNING: ingest failed to load: ${FAILED:-unknown, see the ingest output above}." \
        "They keep their previous rows. Continuing with the schema upgrade and validation."
elif [ "$INGEST_RC" -ne 0 ]; then
    echo "[$(date -u)] ERROR: ingest failed (exit $INGEST_RC). Keeping $CURRENT active."
    exit 1
fi
rm -f "$FAILED_LIST"

# A .wal next to the slot right after the ingest means the load process died before DuckDB
# closed the file. Check it here, before the upgrade: the upgrade opens the slot for writing,
# and a write open replays that log and deletes it. Waiting for the check below would then
# publish a slot whose ingest never closed cleanly.
if [ -e "$TARGET.wal" ]; then
    echo "[$(date -u)] ERROR: $TARGET.wal exists after the ingest, the load did not close the database cleanly. Keeping $CURRENT active."
    exit 1
fi

# Bring a slot copied from an older database up to the current schema (derived tables,
# depth_charts season, missing raw columns) and record the version. Exit 3 means a step
# is still pending, which validate would only catch later, so any non-zero stops here.
echo "[$(date -u)] Upgrading $(basename "$TARGET") to the current schema..."
if ! NFL_MCP_DB_PATH="$TARGET" nfl-mcp upgrade --db "$TARGET"; then
    echo "[$(date -u)] ERROR: schema upgrade failed or is incomplete. Keeping $CURRENT active."
    exit 1
fi

# A .wal next to the new slot means a writer died before DuckDB closed the file. If this
# run published that file, the server answers from the last checkpoint. Stop here, and
# name the file that blocked the swap. The ingest has its own guard above. This one covers
# the steps between that guard and the swap, the schema upgrade first of all.
if [ -e "$TARGET.wal" ]; then
    echo "[$(date -u)] ERROR: $TARGET.wal exists after the schema upgrade, a writer did not close the database cleanly. Keeping $CURRENT active."
    exit 1
fi

# Run the same check the image build runs on the seed, from one shared script: the file
# opens read-only and the core tables have rows. If the new slot fails it, keep the old
# slot live.
VALIDATE_DB="$(dirname "$0")/validate_db.sh"
if [ ! -f "$VALIDATE_DB" ]; then
    echo "[$(date -u)] ERROR: validator '$VALIDATE_DB' is missing, refusing to publish an unvalidated database"
    exit 1
fi
if ! sh "$VALIDATE_DB" "$TARGET"; then
    echo "[$(date -u)] ERROR: $TARGET failed validation. Keeping $CURRENT active."
    exit 1
fi

# Then compare the new slot with the active one: missing tables or columns, fewer rows or
# an older newest week than the active slot, duplicate plays, or empty key columns. On
# failure the active slot stays in place and the new slot is left on disk for inspection.
echo "[$(date -u)] Validating $(basename "$TARGET") against $CURRENT..."
if ! nfl-mcp validate --db "$TARGET" --current "$DB_DIR/$CURRENT"; then
    echo "[$(date -u)] ERROR: validation failed. Keeping $CURRENT active."
    exit 1
fi

# Read-only from here on: nothing can modify the slot the server is about to read.
# The next ingest into this slot restores write permission after its copy.
chmod a-w "$TARGET"

# Atomic rename of the link. The server opens a fresh read-only DuckDB connection per
# tool call (see nfl_mcp/database.py), so the next query reads the new slot — no restart,
# and any in-flight query keeps its handle on the old file until it finishes.
echo "[$(date -u)] Swapping symlink to $(basename "$TARGET")..."
ln -sfn "$(basename "$TARGET")" "$SYMLINK"

echo "[$(date -u)] Update complete. Active DB is now $(basename "$TARGET")."
