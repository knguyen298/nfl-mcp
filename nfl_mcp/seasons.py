"""
Season helpers shared by the CLI and the ingest pipeline.

Kept dependency-free so the CLI can import it without pulling in DuckDB / polars.
"""

import datetime

FIRST_SEASON = 2013   # earliest season where every nflverse source is complete
EARLIEST_SEASON = 1999  # earliest season any nflverse dataset (e.g. schedules) covers


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
    """Every season from FIRST_SEASON through the current one, inclusive."""
    return list(range(FIRST_SEASON, current_season(today) + 1))
