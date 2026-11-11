#!/bin/sh
set -eu

# Publishes a newer database without a gap and without ever serving a half-written file.
# The $DB_DIR/nflread.duckdb symlink is the state: its target names the active slot. Read
# it once, accept only the two slot names, write the *other* slot, then rename the link.
# Two slots also keep the one DuckDB writer per file apart from the server's read-only
# connections, which is why nothing here ever writes to the active slot.

DB_DIR="/data"
SYMLINK="$DB_DIR/nflread.duckdb"

# One updater at a time. Two overlapping runs (two crontab entries, or cron plus a manual
# `docker exec`) both pick the same idle slot: the second deletes the file the first is
# writing, and the first then publishes the second's half-built file. The lock sits on an
# open file descriptor, so the kernel releases it when this process dies. A lock directory
# would stay behind after a killed container and block every later run.
LOCK_FILE="$DB_DIR/.update.lock"
exec 9>"$LOCK_FILE"
if ! command -v flock >/dev/null 2>&1; then
    # Without flock every run would look like an overlap and skip. Fail where the log shows it.
    echo "[$(date -u)] ERROR: flock is missing, refusing to update without a lock"
    exit 1
fi
if ! flock -n 9; then
    echo "[$(date -u)] Another update is running, skipping this one"
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
case "$CURRENT" in
    nflread_a.duckdb) TARGET="$DB_DIR/nflread_b.duckdb" ;;
    nflread_b.duckdb) TARGET="$DB_DIR/nflread_a.duckdb" ;;
    *)
        echo "[$(date -u)] ERROR: unexpected active link '$CURRENT', expected nflread_a.duckdb or nflread_b.duckdb"
        exit 1
        ;;
esac

# The active slot is the source of the copy and the fallback if this run fails, so it has
# to be a real file before the idle slot is dropped.
if [ ! -f "$DB_DIR/$CURRENT" ]; then
    echo "[$(date -u)] ERROR: active slot '$DB_DIR/$CURRENT' is not a regular file"
    exit 1
fi

echo "[$(date -u)] Active DB : $CURRENT"
echo "[$(date -u)] Target DB : $TARGET"

# Clear the files a run left behind when it died partway: the half-built idle slot, the
# WAL file, and the scratch file. If they stay, they block every later update and fill the
# volume. Removing them first also frees the space that the check below measures.
rm -f "$TARGET" "$TARGET.wal" "$TARGET.tmp"

# Ask for the space this run needs before it writes anything:
#   * one full copy of the active slot, into the idle slot, and
#   * room for DuckDB's temp files and the WAL file it keeps next to the database while
#     the load writes.
# The active slot is the only size a running container can measure, so the requirement is
# that size times a factor. Set NFL_MCP_SPACE_FACTOR to a larger number when the volume
# needs more headroom. The image puts DuckDB's temp directory in /tmp, on a different
# filesystem, so this check covers the copy and the WAL on the filesystem the server reads.
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

# Incremental: completed seasons are skipped; the current season and static
# snapshot tables (rankings, players, contracts) are re-fetched every run.
echo "[$(date -u)] Running incremental ingest..."
NFL_MCP_DB_PATH="$TARGET" nfl-mcp ingest

# A .wal next to the new slot means the load process died before DuckDB closed the file.
# If this run published that file, the server answers from the last checkpoint. Stop here,
# and name the file that blocked the swap.
if [ -e "$TARGET.wal" ]; then
    echo "[$(date -u)] ERROR: $TARGET.wal exists, the ingest did not close the database cleanly. Keeping $CURRENT active."
    exit 1
fi

# Run the same check the image build runs on the seed, from one shared script. A load can
# exit 0 and still add no rows. If the new slot fails this check, keep the old slot live.
VALIDATE_DB="$(dirname "$0")/validate_db.sh"
if [ ! -f "$VALIDATE_DB" ]; then
    echo "[$(date -u)] ERROR: validator '$VALIDATE_DB' is missing, refusing to publish an unvalidated database"
    exit 1
fi
if ! sh "$VALIDATE_DB" "$TARGET"; then
    echo "[$(date -u)] ERROR: $TARGET failed validation. Keeping $CURRENT active."
    exit 1
fi

# Atomic rename of the link. The server opens a fresh read-only DuckDB connection per
# tool call (see nfl_mcp/database.py), so the next query reads the new slot — no restart,
# and any in-flight query keeps its handle on the old file until it finishes.
echo "[$(date -u)] Swapping symlink to $(basename "$TARGET")..."
ln -sfn "$(basename "$TARGET")" "$SYMLINK"

echo "[$(date -u)] Update complete. Active DB is now $(basename "$TARGET")."
