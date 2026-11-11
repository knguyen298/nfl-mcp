#!/bin/sh
set -eu

# One answer to "is this database safe to serve?". Two callers use it: the image build,
# which checks the seed copy in /seed, and update_db.sh, which checks each new idle slot.
# One copy is on purpose. If the two checks differ, a build can pass while a running
# container publishes a database with no rows in it.
#
# Usage: sh validate_db.sh <path-to-duckdb>
# The script exits 0 only when the file opens read-only and the core tables have rows.

if [ "$#" -ne 1 ]; then
    echo "[validate] usage: sh validate_db.sh <path-to-duckdb>" >&2
    exit 2
fi

DB="$1"

if [ ! -f "$DB" ]; then
    echo "[validate] ERROR: $DB is not a regular file" >&2
    exit 1
fi

# When DuckDB does not close a database cleanly, it leaves <file>.wal next to the file.
# DuckDB replays that log only when it opens the file for writing. So a reader sees the
# last checkpoint and nothing newer. Do not publish a file with a leftover log beside it.
if [ -e "$DB.wal" ]; then
    echo "[validate] ERROR: $DB.wal exists - the database was not closed cleanly" >&2
    exit 1
fi

# The script opens the database read-only, so the check itself cannot create a WAL.
# `plays` and `schedules` come from the slow play-by-play job. When both tables have rows,
# the load did the work it was meant to do. When one table has no rows, the load stopped
# early. `plays` holds one row per play. `schedules` holds one row per scheduled game.
# The image ships the command `python`. The script falls back to `python3` so it also runs
# on a host that has only the versioned name, such as a developer machine.
PYTHON=python
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3
"$PYTHON" - "$DB" <<'PY'
import duckdb
import sys

path = sys.argv[1]
tables = ('plays','schedules')
conn = duckdb.connect(path, read_only=True)
counts = {t: conn.execute("SELECT count(*) FROM " + t).fetchone()[0] for t in tables}
conn.close()
empty = [t for t in tables if counts[t] == 0]
if empty:
    sys.exit("[validate] ERROR: %s has no rows in %s" % (path, empty))
print("[validate] OK: %s rows=%s" % (path, counts))
PY
