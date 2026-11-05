#!/bin/sh
set -eu

DB_DIR="/data"
SYMLINK="$DB_DIR/nflread.duckdb"
SLOT_A="$DB_DIR/nflread_a.duckdb"
SEED="/seed/nflread.duckdb"

mkdir -p "$DB_DIR"
# Best-effort: the image creates this at build time, and a read-only root filesystem
# must not abort startup over a log directory.
mkdir -p /var/log/nflmcp 2>/dev/null || true

# Seed slot A rather than ingesting on first start. The published image bakes a
# database at /seed, so a cold container becomes queryable in seconds. A full ingest
# here would take 10+ minutes, which blocks readiness on any platform with a startup
# probe or health check, and makes every `docker run` from scratch wait for the network.
if [ ! -f "$SLOT_A" ]; then
    if [ -f "$SYMLINK" ] && [ ! -L "$SYMLINK" ]; then
        # Older images baked the database directly into /data; adopt it as slot A
        # instead of re-ingesting on top of it.
        mv "$SYMLINK" "$SLOT_A"
        echo "[init] Adopted existing /data database as slot A"
    elif [ -f "$SEED" ]; then
        echo "[init] Copying baked seed into slot A..."
        cp "$SEED" "$SLOT_A"
    else
        # No seed (e.g. the dev image): fall back to a cold ingest.
        echo "[init] No seed found; ingesting into slot A (slow startup)..."
        # --allow-failures: one flaky season must not crash-loop a first boot. The ingest
        # prints "Ingest failed for: ..." naming each failure; the weekly update fills gaps.
        NFL_MCP_DB_PATH="$SLOT_A" nfl-mcp ingest --allow-failures
    fi
fi

# Relative link: the symlink lives in $DB_DIR, so the target name resolves there. Only
# create it when absent: after an update swapped to slot B, a restart must keep serving
# B, not relink to the older slot A.
if [ ! -L "$SYMLINK" ]; then
    ln -sfn "$(basename "$SLOT_A")" "$SYMLINK"
fi
echo "[init] Active DB: $(readlink "$SYMLINK")"

cat > /tmp/crontab << 'EOF'
# Thursday 6AM UTC: corrected weekly data and all default datasets.
0 6 * * 4     /update_db.sh >> /var/log/nflmcp/nfl_update.log 2>&1

# Sun/Mon/Wed 3AM UTC: overnight game-day updates.
0 3 * * 0,1,3 /update_db.sh >> /var/log/nflmcp/nfl_update.log 2>&1
EOF

supercronic /tmp/crontab &
echo "[cron] Scheduler started."

# A database made by an older release still serves, but may lack columns the tools use.
# The active slot is read-only, so the check only reads; on exit 3 one update_db.sh run
# upgrades a copy and swaps it in. Opt out with NFL_MCP_UPGRADE_ON_START=0. Nothing here
# may block or fail the start.
if [ "${NFL_MCP_UPGRADE_ON_START:-1}" != "0" ]; then
    UPGRADE_RC=0
    nfl-mcp upgrade --check --db "$SYMLINK" || UPGRADE_RC=$?
    if [ "$UPGRADE_RC" -eq 3 ]; then
        echo "[upgrade] Active DB needs a schema upgrade; starting update_db.sh in the background."
        /update_db.sh >> /var/log/nflmcp/nfl_update.log 2>&1 &
    elif [ "$UPGRADE_RC" -ne 0 ]; then
        echo "[upgrade] Schema check exited $UPGRADE_RC; starting without an upgrade."
    fi
fi

echo "[server] Starting NFL MCP server..."
exec nfl-mcp serve --host 0.0.0.0 --port 8000
