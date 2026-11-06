#!/bin/sh
set -eu

DB_DIR="/data"
SYMLINK="$DB_DIR/nflread.duckdb"

if [ ! -L "$SYMLINK" ]; then
    echo "[$(date -u)] ERROR: $SYMLINK is not a symlink. Has entrypoint.sh run first-run init?"
    exit 1
fi

# One update at a time: a cron run that starts while the boot-time update is still working
# would pick the same idle slot and delete the file the first run has open. The lock is on
# the directory itself, so no lock file can be left behind owned by another user.
exec 9<"$DB_DIR"
if ! flock -n 9; then
    echo "[$(date -u)] Another update is already running; skipping this run."
    exit 0
fi

CURRENT=$(readlink "$SYMLINK")

if [ "$CURRENT" = "nflread_a.duckdb" ]; then
    TARGET="$DB_DIR/nflread_b.duckdb"
else
    TARGET="$DB_DIR/nflread_a.duckdb"
fi

# The ingest writes the datasets that failed to load here, one per line.
FAILED_LIST="$TARGET.failed"

echo "[$(date -u)] Active DB : $CURRENT"
echo "[$(date -u)] Target DB : $TARGET"

# The idle slot is disposable: drop it (and any WAL left by an interrupted run) so a
# failed previous pass can neither be mistaken for fresh data nor fill the disk.
rm -f "$TARGET" "$TARGET.wal" "$FAILED_LIST"

# Fail before the copy rather than partway through it — a truncated DuckDB file in the
# idle slot is harmless, but a full disk mid-ingest wastes the whole run.
NEEDED_KB=$(( $(du -sk "$DB_DIR/$CURRENT" | cut -f1) + 1048576 ))   # copy + 1GB headroom
FREE_KB=$(df -Pk "$DB_DIR" | awk 'NR==2 {print $4}')
if [ "$FREE_KB" -lt "$NEEDED_KB" ]; then
    echo "[$(date -u)] ERROR: need ${NEEDED_KB}KB free on $DB_DIR, have ${FREE_KB}KB. Skipping this run."
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
echo "[$(date -u)] Running incremental ingest..."
INGEST_RC=0
NFL_MCP_DB_PATH="$TARGET" nfl-mcp ingest --failures-file "$FAILED_LIST" || INGEST_RC=$?
if [ "$INGEST_RC" -eq 3 ]; then
    FAILED=$(awk 'NR > 1 { printf ", " } { printf "%s", $0 }' "$FAILED_LIST" 2>/dev/null || true)
    echo "[$(date -u)] WARNING: ingest could not load: ${FAILED:-unknown, see the ingest output above}." \
        "They keep their previous rows. Continuing with the schema upgrade and validation."
elif [ "$INGEST_RC" -ne 0 ]; then
    echo "[$(date -u)] ERROR: ingest failed (exit $INGEST_RC). Keeping $CURRENT active."
    exit 1
fi
rm -f "$FAILED_LIST"

# Bring a slot copied from an older database up to the current schema (derived tables,
# depth_charts season, missing raw columns) and record the version. Exit 3 means a step
# is still pending, which validate would only catch later, so any non-zero stops here.
echo "[$(date -u)] Upgrading $(basename "$TARGET") to the current schema..."
if ! NFL_MCP_DB_PATH="$TARGET" nfl-mcp upgrade --db "$TARGET"; then
    echo "[$(date -u)] ERROR: schema upgrade failed or is incomplete. Keeping $CURRENT active."
    exit 1
fi

# Refuse to publish a bad slot: missing tables or columns, fewer rows or an older newest
# week than the active slot, duplicate plays, or empty key columns. On failure the active
# slot stays in place and the new slot is left on disk for inspection.
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
