"""
Franchise identity — one canonical answer to "which team is this?".

Sources disagree.  nflverse labels games with the abbreviation that was current at the
time (``OAK``, ``SD``, ``STL``).  ESPN mixes eras with retroactive names (``LAR`` and
``TEN`` in 1970, and the 1970 Baltimore Colts filed under Indianapolis).  Two franchises
moved cities inside our coverage window and one stopped playing for three seasons.

Every table and tool therefore normalizes to the **current** abbreviation, using the
codes nflverse uses today (``LV``, ``LAC``, ``LA``, ``WAS``, ``TEN`` …), so a query for
the Raiders returns 1970 and 2024 in one result set.  The historical name is recoverable
per season through :func:`era_name`.

The tables here are our own knowledge of NFL franchise lineage — not source data — so
they can live in the repo even though the records themselves are fetched from ESPN.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Records coverage boundaries.  RECORDS_FIRST_SEASON is the first merged-era season with
# a single 26-team league; ESPN's archive supplies final records from here to 1998, and
# nflverse's own schedules take over from 1999.
RECORDS_FIRST_SEASON = 1970
RECORDS_LAST_ESPN_SEASON = 1998
RECORDS_FIRST_NFLVERSE_SEASON = 1999


@dataclass(frozen=True)
class Franchise:
    """A franchise as one continuous entity, whatever it was called."""

    key: str                      # canonical current abbreviation (nflverse code)
    name: str                     # current club name
    espn_id: int | None = None    # ESPN franchise id — stable across relocations
    aliases: tuple[str, ...] = ()  # abbreviations other sources have used for this club
    first_season: int = 1920      # first season the franchise played
    inactive: tuple[int, ...] = ()  # seasons the franchise existed but did not play
    eras: tuple[tuple[int, str], ...] = field(default_factory=tuple)  # (first season, club name)

    def era_name(self, season: int) -> str:
        """Club name for `season`, falling back to the current name."""
        match = self.name
        for start, label in sorted(self.eras):
            if season >= start:
                match = label
        return match


# Keyed by canonical abbreviation.  ESPN ids come from ESPN's own payloads (31 and 32 are
# the AFC/NFC conference placeholders, not franchises, and are deliberately absent).
FRANCHISES: dict[str, Franchise] = {f.key: f for f in (
    Franchise("ARI", "Arizona Cardinals", 22, first_season=1920,
              eras=((1920, "Chicago Cardinals"), (1960, "St. Louis Cardinals"),
                    (1988, "Phoenix Cardinals"), (1994, "Arizona Cardinals"))),
    Franchise("ATL", "Atlanta Falcons", 1, first_season=1966),
    Franchise("BAL", "Baltimore Ravens", 33, first_season=1996),
    Franchise("BUF", "Buffalo Bills", 2, first_season=1960),
    Franchise("CAR", "Carolina Panthers", 29, first_season=1995),
    Franchise("CHI", "Chicago Bears", 3, first_season=1920,
              eras=((1920, "Decatur Staleys"), (1922, "Chicago Staleys"), (1923, "Chicago Bears"))),
    Franchise("CIN", "Cincinnati Bengals", 4, first_season=1968),
    # The Browns' name, history and records stayed in Cleveland while the players moved to
    # Baltimore, so the franchise is continuous — it simply did not play 1996–1998.
    Franchise("CLE", "Cleveland Browns", 5, first_season=1946, inactive=(1996, 1997, 1998)),
    Franchise("DAL", "Dallas Cowboys", 6, first_season=1960),
    Franchise("DEN", "Denver Broncos", 7, first_season=1960),
    Franchise("DET", "Detroit Lions", 8, first_season=1930,
              eras=((1930, "Portsmouth Spartans"), (1934, "Detroit Lions"))),
    Franchise("GB", "Green Bay Packers", 9, first_season=1921),
    Franchise("HOU", "Houston Texans", 34, first_season=2002),
    Franchise("IND", "Indianapolis Colts", 11, first_season=1953,
              eras=((1953, "Baltimore Colts"), (1984, "Indianapolis Colts"))),
    Franchise("JAX", "Jacksonville Jaguars", 30, first_season=1995),
    Franchise("KC", "Kansas City Chiefs", 12, first_season=1960,
              eras=((1960, "Dallas Texans"), (1963, "Kansas City Chiefs"))),
    Franchise("LA", "Los Angeles Rams", 14, aliases=("STL", "LAR", "RAM"), first_season=1937,
              eras=((1937, "Cleveland Rams"), (1946, "Los Angeles Rams"),
                    (1995, "St. Louis Rams"), (2016, "Los Angeles Rams"))),
    Franchise("LAC", "Los Angeles Chargers", 24, aliases=("SD",), first_season=1960,
              eras=((1960, "Los Angeles Chargers"), (1961, "San Diego Chargers"),
                    (2017, "Los Angeles Chargers"))),
    Franchise("LV", "Las Vegas Raiders", 13, aliases=("OAK",), first_season=1960,
              eras=((1960, "Oakland Raiders"), (2020, "Las Vegas Raiders"))),
    Franchise("MIA", "Miami Dolphins", 15, first_season=1966),
    Franchise("MIN", "Minnesota Vikings", 16, first_season=1961),
    Franchise("NE", "New England Patriots", 17, first_season=1960,
              eras=((1960, "Boston Patriots"), (1971, "New England Patriots"))),
    Franchise("NO", "New Orleans Saints", 18, first_season=1967),
    Franchise("NYG", "New York Giants", 19, first_season=1925),
    Franchise("NYJ", "New York Jets", 20, first_season=1960,
              eras=((1960, "New York Titans"), (1963, "New York Jets"))),
    Franchise("PHI", "Philadelphia Eagles", 21, first_season=1933),
    Franchise("PIT", "Pittsburgh Steelers", 23, first_season=1933,
              eras=((1933, "Pittsburgh Pirates"), (1940, "Pittsburgh Steelers"))),
    Franchise("SEA", "Seattle Seahawks", 26, first_season=1976),
    Franchise("SF", "San Francisco 49ers", 25, first_season=1946),
    Franchise("TB", "Tampa Bay Buccaneers", 27, first_season=1976),
    Franchise("TEN", "Tennessee Titans", 10, first_season=1960,
              eras=((1960, "Houston Oilers"), (1997, "Tennessee Oilers"),
                    (1999, "Tennessee Titans"))),
    Franchise("WAS", "Washington Commanders", 28, aliases=("WSH", "TFN"), first_season=1932,
              eras=((1932, "Washington Redskins"), (2020, "Washington Football Team"),
                    (2022, "Washington Commanders"))),
)}

# ESPN franchise id -> canonical key.  This is the only join key used for ESPN data,
# because ESPN's abbreviations drift (id 10 is "HOU" in 1996 and "TEN" in 1970).
ESPN_IDS: dict[int, str] = {
    f.espn_id: f.key for f in FRANCHISES.values() if f.espn_id is not None
}

_ALIASES: dict[str, str] = {
    alias: key for key, f in FRANCHISES.items() for alias in f.aliases
}


def canonical_team(value: str) -> str | None:
    """Resolve any source's team abbreviation to the canonical current one.

    Returns None for anything unrecognised, so callers can surface a clean error instead
    of silently querying a team that does not exist.
    """
    if not value:
        return None
    code = value.strip().upper()
    if code in FRANCHISES:
        return code
    return _ALIASES.get(code)


def era_name(season: int, team: str) -> str | None:
    """The club's name in `season` ("Baltimore Colts" for 1970), or None if unknown."""
    franchise = FRANCHISES.get(canonical_team(team) or "")
    return franchise.era_name(season) if franchise else None


def expected_games(season: int) -> int:
    """Scheduled regular-season games per team, including the two strike seasons."""
    if season <= 1977:
        return 14
    if season == 1982:      # 9-game strike season
        return 9
    if season == 1987:      # 15-game strike season
        return 15
    if season <= 2020:
        return 16
    return 17


def expected_franchise_count(season: int) -> int:
    """How many franchises played in `season` — used to flag incomplete ingest runs."""
    count = 0
    for franchise in FRANCHISES.values():
        if season < franchise.first_season or season in franchise.inactive:
            continue
        count += 1
    return count


def availability(team: str, season: int) -> dict | None:
    """Why a franchise has no record for a season, or None if it should have one.

    Historical records are never estimated: a franchise that did not exist yet, or sat a
    season out, is reported that way rather than as a 0-0 record.
    """
    key = canonical_team(team)
    if key is None:
        return {"status": "unknown_team", "reason": f"Unknown team '{team}'."}
    franchise = FRANCHISES[key]
    if season < franchise.first_season:
        return {
            "status": "franchise_not_yet_active",
            "reason": f"{franchise.name} did not exist until {franchise.first_season}.",
            "first_season": franchise.first_season,
        }
    if season in franchise.inactive:
        return {
            "status": "franchise_inactive",
            "reason": f"{franchise.name} did not play in {season}.",
        }
    if season < RECORDS_FIRST_SEASON:
        return {
            "status": "before_coverage",
            "reason": f"Team records start in {RECORDS_FIRST_SEASON}, the first merged-era season.",
            "first_season": RECORDS_FIRST_SEASON,
        }
    return None


def schedule_alias_case(column: str = "source_team") -> str:
    """SQL CASE expression mapping nflverse era abbreviations onto canonical keys.

    Built from the same alias table the Python helpers use, so the ingest SQL and the
    tools can never drift apart.
    """
    arms = " ".join(
        f"WHEN {column} = '{src}' THEN '{canon}'"
        for src, canon in sorted(_ALIASES.items())
    )
    return f"CASE {arms} ELSE {column} END"
