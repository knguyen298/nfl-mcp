"""
ESPN's public scoreboard feed, used for one thing only: **final season win-loss records
before 1999**, which nflverse does not publish.

Findings this module is built around, all measured rather than assumed:

* ``dates=YYYY`` returns a *calendar* year, so an NFL season needs two pulls (Y and Y+1)
  and rows are attributed with each event's own ``season.year``.
* ``seasontype`` is ignored on year queries, and ``week.number`` on them is month-relative,
  so neither can separate preseason / regular / postseason.
* ESPN's archived gamebooks are missing scores for roughly 7% of games per season from
  1970 to 2001 — always the Monday-night slot, reported as ``Final 0-0`` with no box
  score at all. Summing scores would invent ties and hand out wrong records.
* The per-competitor ``records`` entry (``name="overall"``, ``type="total"``) survives that
  hole. It holds the cumulative season record, so a franchise's final regular-season
  standing can be read straight off any of its games. Validated: 30/30 franchises for
  1996 against published standings, 30/31 teams for 2000 against nflverse.
* A record is only accepted when wins+losses+ties equals that season's scheduled length,
  which is what keeps postseason rows (and any other stray cumulative line) out.

Records that no game carries are reported as missing, never estimated.
"""

from __future__ import annotations

import dataclasses
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from . import __version__
from .franchises import ESPN_IDS, expected_games

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
USER_AGENT = f"nfl-mcp/{__version__} (+https://github.com/knguyen298/nfl-mcp)"
DEFAULT_TIMEOUT = 30.0
DEFAULT_ATTEMPTS = 2
DEFAULT_BACKOFF = 1.5


class EspnUnavailable(RuntimeError):
    """Raised when ESPN could not be reached, or answered with something unusable."""


@dataclass(frozen=True)
class SeasonRecord:
    """One franchise's final regular-season record, as ESPN reported it."""

    season: int
    team: str                 # canonical current abbreviation
    wins: int
    losses: int
    ties: int
    games: int
    record_summary: str       # ESPN's own "12-4" / "11-2-1" string
    espn_team_id: int
    espn_abbrev: str          # abbreviation ESPN used for that season
    source_game_id: str       # the event the record was read from
    fetched_at: str           # UTC ISO timestamp


Transport = Callable[[str], dict]


def _default_transport(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise EspnUnavailable(f"ESPN request failed: {exc}") from exc


def scoreboard_url(year: int) -> str:
    return f"{SCOREBOARD_URL}?dates={year}&limit=1000"


def parse_record(summary: str | None) -> tuple[int, int, int] | None:
    """Parse ESPN's "12-4" / "11-2-1" form into (wins, losses, ties)."""
    if not summary:
        return None
    parts = summary.split("-")
    if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts):
        return None
    wins, losses = int(parts[0]), int(parts[1])
    ties = int(parts[2]) if len(parts) == 3 else 0
    return wins, losses, ties


def _overall_record(competitor: dict) -> str | None:
    for record in competitor.get("records") or []:
        if record.get("name") == "overall" and record.get("type") == "total":
            return record.get("summary")
    return None


def _events_for_season(payloads: Iterable[dict], season: int) -> list[dict]:
    """Events belonging to `season`, de-duplicated across calendar-year pulls."""
    seen: set[str] = set()
    events: list[dict] = []
    for payload in payloads:
        for event in payload.get("events") or []:
            if (event.get("season") or {}).get("year") != season:
                continue
            event_id = event.get("id")
            if not event_id or event_id in seen:
                continue
            seen.add(event_id)
            events.append(event)
    return events


def extract_records(events: Iterable[dict], season: int) -> dict[int, SeasonRecord]:
    """Latest cumulative record per franchise id whose game count matches the schedule."""
    wanted = expected_games(season)
    best: dict[int, tuple[str, SeasonRecord]] = {}
    for event in events:
        for competitor in (event.get("competitions") or [{}])[0].get("competitors") or []:
            team = competitor.get("team") or {}
            raw_id = team.get("id")
            if raw_id is None or not str(raw_id).isdigit():
                continue
            team_id = int(raw_id)
            if team_id not in ESPN_IDS:
                continue  # conference placeholders (AFC/NFC) and anything unrecognised
            summary = _overall_record(competitor)
            parsed = parse_record(summary)
            if parsed is None or sum(parsed) != wanted:
                continue
            record = SeasonRecord(
                season=season,
                team=ESPN_IDS[team_id],
                wins=parsed[0], losses=parsed[1], ties=parsed[2], games=sum(parsed),
                record_summary=summary,
                espn_team_id=team_id,
                espn_abbrev=team.get("abbreviation") or "",
                source_game_id=event.get("id") or "",
                fetched_at="",  # stamped by the caller
            )
            date = event.get("date") or ""
            if team_id not in best or date > best[team_id][0]:
                # ESPN keeps duplicate, contradictory rows for some 1970s games; the
                # latest dated row that matches the schedule length is the usable one.
                best[team_id] = (date, record)
    return {team_id: entry[1] for team_id, entry in best.items()}


def fetch_season_records(
    season: int,
    *,
    transport: Transport | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
    sleep: Callable[[float], None] = time.sleep,
) -> list[SeasonRecord]:
    """Fetch every franchise's final regular-season record for one season.

    Raises EspnUnavailable if either calendar-year pull fails after `attempts` tries, so
    the caller can leave the season unrecorded and retry on the next run.
    """
    send = transport or _default_transport
    payloads = [_pull(send, scoreboard_url(year), attempts, backoff, sleep) for year in (season, season + 1)]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    records = extract_records(_events_for_season(payloads, season), season)
    return [
        dataclasses.replace(record, fetched_at=now)
        for _, record in sorted(records.items(), key=lambda item: item[1].team)
    ]


def _pull(
    send: Transport,
    url: str,
    attempts: int,
    backoff: float,
    sleep: Callable[[float], None],
) -> dict:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return send(url)
        except Exception as exc:  # transport-specific failures all mean "try again"
            last = exc
            if attempt + 1 < attempts:
                sleep(backoff * (attempt + 1))
    raise EspnUnavailable(f"ESPN request failed for {url}: {last}")
