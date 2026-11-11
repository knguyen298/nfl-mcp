"""
Named lifecycle states — one place that spells every state, status and error code.

Three lifecycles live here: why a franchise-season has no record (:class:`Availability`),
how a tool call failed (:class:`ErrorCode`), and how one dataset+season ingest ended
(:class:`IngestStatus`).  Members subclass ``str``, so they compare equal to the plain
strings they replace and ``json.dumps`` writes the plain value.  Each class also defines
``__str__``: ``requires-python`` is 3.10, ``enum.StrEnum`` only exists in 3.11, and
without it ``f"{member}"`` would render ``Availability.FRANCHISE_INACTIVE``.

This module imports nothing from the rest of the package, so any module can import it
without a circular import.
"""

from __future__ import annotations

from enum import Enum


class Availability(str, Enum):
    """Reasons a franchise-season has no record, returned by ``franchises.availability``."""

    UNKNOWN_TEAM = "unknown_team"  # abbreviation resolves to no franchise at all
    FRANCHISE_NOT_YET_ACTIVE = "franchise_not_yet_active"  # franchise did not exist yet
    FRANCHISE_INACTIVE = "franchise_inactive"  # existed, but sat this season out
    BEFORE_COVERAGE = "before_coverage"  # season predates our records floor
    RECORD_NOT_AVAILABLE = "record_not_available"  # no publisher gave this record

    def __str__(self) -> str:
        return self.value


class ErrorCode(str, Enum):
    """How a tool call failed, reported in the ``code`` field of an error payload."""

    # Written by server.call_tool when the tool call did not complete.
    UNKNOWN_TOOL = "UNKNOWN_TOOL"  # no tool registered under the requested name
    TIMEOUT = "TIMEOUT"  # a query ran past the timeout (tools, or a raised TimeoutError)
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"  # arguments fail the input schema or signature
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"  # any other exception from a tool
    # Written by tools._sql_error_detail from DuckDB's own binder/catalog message.
    COLUMN_NOT_FOUND = "COLUMN_NOT_FOUND"  # table exists, that column does not
    TABLE_NOT_FOUND = "TABLE_NOT_FOUND"  # no such table in the database
    # Reserved for the nfl_query guardrails and the ingest / CLI error paths.
    NOT_SELECT = "NOT_SELECT"  # statement is not a SELECT
    FORBIDDEN_KEYWORD = "FORBIDDEN_KEYWORD"  # statement uses a blocked keyword
    MULTIPLE_STATEMENTS = "MULTIPLE_STATEMENTS"  # request packs several statements
    SQL_ERROR = "SQL_ERROR"  # SQL failed for a reason we cannot classify
    DB_NOT_FOUND = "DB_NOT_FOUND"  # database file is missing
    DB_LOCKED = "DB_LOCKED"  # another process holds the write lock
    INVALID_INPUT = "INVALID_INPUT"  # a tool argument is malformed
    UNKNOWN_TEAM = "UNKNOWN_TEAM"  # a tool was asked about an unknown team
    OUT_OF_COVERAGE = "OUT_OF_COVERAGE"  # season is outside what we publish

    def __str__(self) -> str:
        return self.value


class IngestStatus(str, Enum):
    """Outcome of one dataset+season row in the ``_ingest_metadata`` table."""

    LOADED = "loaded"  # rows were written for the season
    EMPTY = "empty"  # the source published no rows for the season
    FAILED = "failed"  # the fetch or the write raised
    PARTIAL = "partial"  # some rows landed, the season is incomplete

    def __str__(self) -> str:
        return self.value
