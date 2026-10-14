FROM python:3.12-slim

ARG SUPERCRONIC_VERSION=0.2.33

# supercronic runs the in-container database updater (see entrypoint.sh). curl is only
# needed to fetch it and is removed with the apt lists afterwards.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://github.com/aptible/supercronic/releases/download/v${SUPERCRONIC_VERSION}/supercronic-linux-amd64" \
        -o /usr/local/bin/supercronic \
    && chmod +x /usr/local/bin/supercronic

# Install nfl-mcp from local source so the image always reflects the committed
# code (including the DuckDB memory bounds used during the baked ingest).
WORKDIR /src
COPY . /src
RUN pip install --no-cache-dir uv && uv pip install --system --no-cache .

# The database the server opens at runtime: a symlink managed by entrypoint.sh that
# points at whichever slot (nflread_a / nflread_b) is currently active.
ENV NFL_MCP_DB_PATH=/data/nflread.duckdb

# Seed the image with a pre-built database at /seed, then let the container keep itself
# fresh. Two refresh paths, and both matter:
#
#   * Build time (this RUN + the weekly GitHub Actions cron): whoever pulls the image gets
#     recent data immediately, and the server is queryable in seconds instead of doing a
#     10+ minute cold ingest that would trip the platform startup probe (~60s).
#   * Runtime (entrypoint.sh + update_db.sh under supercronic): a long-running container
#     copies /seed into slot A, then ingests into the idle slot and atomically swaps the
#     symlink. The server opens a fresh read-only connection per query, so the swap takes
#     effect on the next request with no restart.
#
# Memory caps are passed inline (not as ENV) so the heavy build-time ingest stays
# within the CI runner's RAM (spilling to disk) without leaking the large limit
# into the lightweight runtime serve process configured further down.
#
# After ingest we validate the seed: no leftover WAL, and the core tables that the
# slow PBP path produces actually have rows — so a silently-incomplete ingest fails
# the build instead of shipping a broken image.
#
# INGEST_ARGS is empty by default, so production/release builds bake the default window:
# 1970 onward, clipped per dataset (play-by-play and the stats tables from their 1999
# source floor). That is a multi-GB build; pass e.g. INGEST_ARGS=--start 1999 for a smaller
# image, or a single season in CI to validate the build pipeline quickly without the bake.
ARG INGEST_ARGS=""
RUN mkdir -p /seed /data /var/log/nflmcp \
 && chmod 777 /data /var/log/nflmcp \
 && NFL_MCP_DB_PATH=/seed/nflread.duckdb \
    NFL_MCP_DUCKDB_MEMORY_LIMIT=3GB \
    NFL_MCP_DUCKDB_THREADS=2 \
    POLARS_MAX_THREADS=2 \
    nfl-mcp ingest ${INGEST_ARGS} \
 && rm -rf /seed/.duckdb_spill /data/.duckdb_spill /root/.cache \
 && test ! -e /seed/nflread.duckdb.wal \
 && python -c "import duckdb,sys; c=duckdb.connect('/seed/nflread.duckdb',read_only=True); rows={t:c.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in ('plays','schedules')}; print('baked rows:',rows); sys.exit(0 if all(v>0 for v in rows.values()) else 1)"

# A/B slot management plus the schedule that drives it.
COPY entrypoint.sh /entrypoint.sh
COPY update_db.sh /update_db.sh
RUN chmod +x /entrypoint.sh /update_db.sh

# Runtime resource caps for the small serve container (e.g. 1Gi). Heavy queries
# spill to /tmp instead of OOM-killing the process. Note the updater inherits these,
# so give the container >=2Gi if the scheduled ingest OOMs.
ENV NFL_MCP_DUCKDB_MEMORY_LIMIT=512MB \
    NFL_MCP_DUCKDB_THREADS=1 \
    NFL_MCP_DUCKDB_TEMP_DIR=/tmp/.duckdb_spill

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
