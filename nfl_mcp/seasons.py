"""
Season helpers shared by the CLI and the ingest pipeline.

Kept dependency-free so the CLI can import it without pulling in DuckDB / polars.
"""

import datetime

# Two distinct bounds, deliberately kept apart:
#
#   EARLIEST_SEASON — how deep the upstream sources actually go (rosters, 1920).
#                     The lower bound for --start; nothing can be loaded below it.
#   FIRST_SEASON    — the default backfill start. 1966, the AFL–NFL merger agreement year,
#                     so a fresh install gets pre-merger rosters as well as every merged season.
#
# Every dataset is additionally clipped to its own registry min_season/max_season, so the
# default starts *later* than 1966 for anything whose source began publishing after it
# (play-by-play and all stats tables start 1999, Next Gen Stats 2016, FTN charting 2022).
# Passing --start 1920 reaches the pre-1966 rosters; tests assert EARLIEST_SEASON stays
# equal to registry.earliest_season().
EARLIEST_SEASON = 1920
FIRST_SEASON = 1966

# The league year rolls over on the 1st of this month: schedules, rosters and the season
# label all change then. Both helpers below are written from this one constant, so the day
# a season stops being "current" and the day it is called finished cannot drift apart.
SEASON_ROLLOVER_MONTH = 3


def season_settled_on(season: int) -> datetime.date:
    """Return the first date on which `season`'s record can no longer change.

    A season plays out into February of the year after it kicks off (the last regular-season
    week, the playoffs, the Super Bowl). From the league-year rollover on — March 1 of
    `season + 1` — every game it will ever have is finished, so data loaded on or after that
    date is complete. Data loaded before it may be missing those last games.
    """
    return datetime.date(season + 1, SEASON_ROLLOVER_MONTH, 1)


def current_season(today: datetime.date | None = None) -> int:
    """Return the NFL season that is in progress or most recently completed.

    Follows the nflverse convention: a new season begins in March, when the league
    year rolls over (free agency, draft, new schedules, new rosters). From March through
    February of the following year the season is labelled by the year it kicks off in.
    Data sources that do not exist yet for a new season (e.g. play-by-play before
    Week 1, participation until the season ends) are skipped by the ingest using each
    dataset's registry newest_season rule, and loaded once nflreadpy accepts them.
    """
    today = today or datetime.date.today()
    # This year's season is current once last year's has settled — the same March 1 boundary
    # season_settled_on() gives callers who ask whether a season is finished.
    return today.year if today >= season_settled_on(today.year - 1) else today.year - 1


def all_seasons(today: datetime.date | None = None) -> list[int]:
    """Every season from FIRST_SEASON through the current one, inclusive.

    This is the default backfill window, and the widest one any dataset could want inside
    it. Callers must clip it to the per-dataset coverage window before loading (see
    registry min_season/max_season).
    """
    return list(range(FIRST_SEASON, current_season(today) + 1))
