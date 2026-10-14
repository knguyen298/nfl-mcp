"""
Season helpers shared by the CLI and the ingest pipeline.

Kept dependency-free so the CLI can import it without pulling in DuckDB / polars.
"""

import datetime

# Two distinct bounds, deliberately kept apart:
#
#   EARLIEST_SEASON — how deep the upstream sources actually go (rosters, 1920).
#                     The lower bound for --start; nothing can be loaded below it.
#   FIRST_SEASON    — the default backfill start. 1970, the merger era, so a fresh
#                     install gets a century of rosters without the pre-merger leagues.
#
# Every dataset is additionally clipped to its own registry min_season/max_season, so the
# default starts *later* than 1970 for anything whose source began publishing after it
# (play-by-play and all stats tables start 1999, Next Gen Stats 2016, FTN charting 2022).
# Passing --start 1920 reaches the pre-1970 rosters; tests assert EARLIEST_SEASON stays
# equal to registry.earliest_season().
EARLIEST_SEASON = 1920
FIRST_SEASON = 1970


def current_season(today: datetime.date | None = None) -> int:
    """Return the NFL season that is in progress or most recently completed.

    Follows the nflverse convention: a new season begins in March, when the league
    year rolls over (free agency, draft, new schedules, new rosters). From March through
    February of the following year the season is labelled by the year it kicks off in.
    Data sources that do not exist yet for a new season (e.g. play-by-play before
    Week 1) fail to load gracefully and are retried on the next ingest.
    """
    today = today or datetime.date.today()
    return today.year if today.month >= 3 else today.year - 1


def all_seasons(today: datetime.date | None = None) -> list[int]:
    """Every season from FIRST_SEASON through the current one, inclusive.

    This is the default backfill window, and the widest one any dataset could want inside
    it. Callers must clip it to the per-dataset coverage window before loading (see
    registry min_season/max_season).
    """
    return list(range(FIRST_SEASON, current_season(today) + 1))
