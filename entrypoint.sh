#!/bin/sh
set -eu

DB_DIR="/data"
SYMLINK="$DB_DIR/nflread.duckdb"
SLOT_A="$DB_DIR/nflread_a.duckdb"
# Same-directory scratch name for slot A, so publishing it is a rename, not a copy.
SLOT_A_TMP="$SLOT_A.tmp"
SEED="/seed/nflread.duckdb"

mkdir -p "$DB_DIR"
# Best-effort: the image creates this at build time, and a read-only root filesystem
# must not abort startup over a log directory.
mkdir -p /var/log/nflmcp 2>/dev/null || true

# A symlink that resolves is the answer to "which slot is active?" — update_db.sh
# alternates it between slot A and slot B, so re-pointing it at slot A on every start
# would serve an older database after an update and let the next update delete the
# newer one. It can also point at a slot another run was mid-way through rewriting.
# Leave such a link alone; only a missing or dangling link needs first-start repair.
if [ -L "$SYMLINK" ] && [ -e "$SYMLINK" ]; then
    echo "[init] Active DB: $(readlink "$SYMLINK") (keeping existing link)"
else
    # Seed slot A rather than ingesting on first start. The published image bakes a
    # database at /seed, so a cold container becomes queryable in seconds. A full ingest
    # here would take 10+ minutes, which blocks readiness on any platform with a startup
    # probe or health check, and makes every `docker run` from scratch wait for the network.
    # Clear the scratch name first: a previous start may have died partway through it.
    rm -f "$SLOT_A_TMP" "$SLOT_A_TMP.wal"

    if [ ! -f "$SLOT_A" ]; then
        if [ -f "$SYMLINK" ] && [ ! -L "$SYMLINK" ]; then
            # Older images baked the database directly into /data; adopt it as slot A
            # instead of re-ingesting on top of it.
            mv "$SYMLINK" "$SLOT_A"
            echo "[init] Adopted existing /data database as slot A"
        elif [ -f "$SEED" ]; then
            echo "[init] Copying baked seed into slot A..."
            # Fill the scratch name, then rename. A copy or ingest that dies halfway must
            # never leave a partial file under the name the server serves.
            cp "$SEED" "$SLOT_A_TMP"
            mv "$SLOT_A_TMP" "$SLOT_A"
        else
            # No seed (e.g. the dev image): fall back to a cold ingest.
            echo "[init] No seed found; ingesting into slot A (slow startup)..."
            NFL_MCP_DB_PATH="$SLOT_A_TMP" nfl-mcp ingest
            mv "$SLOT_A_TMP" "$SLOT_A"
        fi
    fi

    # Relative link: the symlink lives in $DB_DIR, so the target name resolves there.
    ln -sfn "$(basename "$SLOT_A")" "$SYMLINK"
    echo "[init] Active DB: $(readlink "$SYMLINK")"
fi

cat > /tmp/crontab << 'EOF'
# Thursday 6AM UTC: corrected weekly data and all default datasets.
0 6 * * 4     /update_db.sh >> /var/log/nflmcp/nfl_update.log 2>&1

# Sun/Mon/Wed 3AM UTC: overnight game-day updates.
0 3 * * 0,1,3 /update_db.sh >> /var/log/nflmcp/nfl_update.log 2>&1
EOF

# supercronic reports what it does (which job started, which job failed) on its own output,
# and its output is this script's output, so `docker logs` shows it. Each crontab line above
# sends the updater's own output to /var/log/nflmcp/nfl_update.log.
# Nothing restarts the scheduler. If it fails to start, the only sign is data that stops
# getting newer. So the script looks for the process once, a moment after start, and writes
# one clear line. Serving old data is better than not serving at all, so this part does not
# exit.
supercronic /tmp/crontab &
SCHEDULER_PID=$!
# Two seconds: long enough for a bad crontab file or a missing supercronic to fail.
sleep 2
if kill -0 "$SCHEDULER_PID" 2>/dev/null; then
    echo "[cron] Scheduler started (pid $SCHEDULER_PID)."
else
    echo "[cron] WARNING: supercronic is not running. The database will not refresh on schedule."
fi

echo "[server] Starting NFL MCP server..."
exec nfl-mcp serve --host 0.0.0.0 --port 8000
