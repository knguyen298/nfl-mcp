#!/bin/sh
set -eu

DB_DIR="/data"
SYMLINK="$DB_DIR/nflread.duckdb"

if [ ! -L "$SYMLINK" ]; then
    echo "[$(date -u)] ERROR: $SYMLINK is not a symlink. Has entrypoint.sh run first-run init?"
    exit 1
fi

CURRENT=$(readlink "$SYMLINK")

if [ "$CURRENT" = "nflread_a.duckdb" ]; then
    TARGET="$DB_DIR/nflread_b.duckdb"
else
    TARGET="$DB_DIR/nflread_a.duckdb"
fi

echo "[$(date -u)] Active DB : $CURRENT"
echo "[$(date -u)] Target DB : $TARGET"

# The idle slot is disposable: drop it (and any WAL left by an interrupted run) so a
# failed previous pass can neither be mistaken for fresh data nor fill the disk.
rm -f "$TARGET" "$TARGET.wal"

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

# Incremental: completed seasons are skipped; the current season and static
# snapshot tables (rankings, players, contracts) are re-fetched every run.
echo "[$(date -u)] Running incremental ingest..."
NFL_MCP_DB_PATH="$TARGET" nfl-mcp ingest

# Atomic rename of the link. The server opens a fresh read-only DuckDB connection per
# tool call (see nfl_mcp/database.py), so the next query reads the new slot — no restart,
# and any in-flight query keeps its handle on the old file until it finishes.
echo "[$(date -u)] Swapping symlink to $(basename "$TARGET")..."
ln -sfn "$(basename "$TARGET")" "$SYMLINK"

echo "[$(date -u)] Update complete. Active DB is now $(basename "$TARGET")."
