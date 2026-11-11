"""Unit tests for nfl_mcp.status — the one place state names are spelled.

tests/conftest.py auto-tags unmarked tests as unit tests, so these need no marker and
no loaded database.  The point of the file is that the refactor changed no value: every
member still has to look like the string literal it replaced, in comparisons, in
``str()``, in f-strings and in JSON.
"""

import inspect
import json

import duckdb
import pytest

from nfl_mcp import status, tools
from nfl_mcp.franchises import availability
from nfl_mcp.server import _error_body, _tool_error_payload
from nfl_mcp.status import Availability, ErrorCode, IngestStatus

# Every member with the exact string a caller used to see.  Spelled out rather than
# derived from `.value`, so renaming a value fails here instead of passing silently.
SPELLINGS = [
    (Availability.UNKNOWN_TEAM, "unknown_team"),
    (Availability.FRANCHISE_NOT_YET_ACTIVE, "franchise_not_yet_active"),
    (Availability.FRANCHISE_INACTIVE, "franchise_inactive"),
    (Availability.BEFORE_COVERAGE, "before_coverage"),
    (Availability.RECORD_NOT_AVAILABLE, "record_not_available"),
    (ErrorCode.UNKNOWN_TOOL, "UNKNOWN_TOOL"),
    (ErrorCode.TIMEOUT, "TIMEOUT"),
    (ErrorCode.INVALID_ARGUMENTS, "INVALID_ARGUMENTS"),
    (ErrorCode.TOOL_EXECUTION_ERROR, "TOOL_EXECUTION_ERROR"),
    (ErrorCode.COLUMN_NOT_FOUND, "COLUMN_NOT_FOUND"),
    (ErrorCode.TABLE_NOT_FOUND, "TABLE_NOT_FOUND"),
    (ErrorCode.NOT_SELECT, "NOT_SELECT"),
    (ErrorCode.FORBIDDEN_KEYWORD, "FORBIDDEN_KEYWORD"),
    (ErrorCode.MULTIPLE_STATEMENTS, "MULTIPLE_STATEMENTS"),
    (ErrorCode.SQL_ERROR, "SQL_ERROR"),
    (ErrorCode.DB_NOT_FOUND, "DB_NOT_FOUND"),
    (ErrorCode.DB_LOCKED, "DB_LOCKED"),
    (ErrorCode.INVALID_INPUT, "INVALID_INPUT"),
    (ErrorCode.UNKNOWN_TEAM, "UNKNOWN_TEAM"),
    (ErrorCode.OUT_OF_COVERAGE, "OUT_OF_COVERAGE"),
    (IngestStatus.LOADED, "loaded"),
    (IngestStatus.EMPTY, "empty"),
    (IngestStatus.FAILED, "failed"),
    (IngestStatus.PARTIAL, "partial"),
]


@pytest.mark.parametrize("member,spelling", SPELLINGS)
def test_member_compares_equal_to_the_plain_string(member, spelling):
    assert member == spelling
    assert spelling == member
    assert isinstance(member, str)


@pytest.mark.parametrize("member,spelling", SPELLINGS)
def test_str_and_fstring_give_the_plain_value(member, spelling):
    assert str(member) == spelling
    assert f"{member}" == spelling
    assert type(member).__str__(member) == spelling


@pytest.mark.parametrize("member,spelling", SPELLINGS)
def test_json_dumps_writes_the_plain_value(member, spelling):
    assert json.dumps({"status": member}) == json.dumps({"status": spelling})
    assert json.dumps({member: 1}) == json.dumps({spelling: 1})
    # The server's error path serializes without `default=str`; a str subclass survives.
    assert json.dumps({"status": member}, indent=2) == f'{{\n  "status": "{spelling}"\n}}'


def test_availability_values_are_the_agreed_set():
    assert {member.value for member in Availability} == {
        "unknown_team",
        "franchise_not_yet_active",
        "franchise_inactive",
        "before_coverage",
        "record_not_available",
    }


def test_ingest_status_values_are_the_agreed_set():
    assert {member.value for member in IngestStatus} == {"loaded", "empty", "failed", "partial"}


def test_error_code_values_match_their_names():
    assert [member.value for member in ErrorCode] == [member.name for member in ErrorCode]


def test_each_class_names_its_own_lifecycle():
    assert {type(member) for member, _ in SPELLINGS} == {Availability, ErrorCode, IngestStatus}
    for cls in (Availability, ErrorCode, IngestStatus):
        assert cls.__doc__


def test_status_module_does_not_import_the_package():
    """status.py stays a leaf so any module can import it without a cycle."""
    source = inspect.getsource(status)
    assert "from ." not in source
    assert "import nfl_mcp" not in source


class TestAvailabilityReturnsMembers:
    def test_unknown_team(self):
        note = availability("ZZ", 1980)
        assert isinstance(note["status"], Availability)
        assert note["status"] is Availability.UNKNOWN_TEAM
        assert note["status"] == "unknown_team"

    def test_franchise_not_yet_active(self):
        note = availability("HOU", 1970)
        assert note["status"] is Availability.FRANCHISE_NOT_YET_ACTIVE
        assert note["status"] == "franchise_not_yet_active"

    def test_franchise_inactive(self):
        note = availability("CLE", 1997)
        assert note["status"] is Availability.FRANCHISE_INACTIVE
        assert note["status"] == "franchise_inactive"

    def test_before_coverage(self):
        note = availability("KC", 1968)
        assert note["status"] is Availability.BEFORE_COVERAGE
        assert note["status"] == "before_coverage"

    def test_loadable_season_has_no_note(self):
        assert availability("KC", 2024) is None

    def test_note_serializes_to_plain_values(self):
        payload = json.dumps(availability("CLE", 1997), indent=2, default=str)
        assert '"status": "franchise_inactive"' in payload
        assert "Availability." not in payload


class TestErrorPayloadsKeepTheOldStrings:
    # Only an exception that escaped a tool reaches _tool_error_payload now. UNKNOWN_TOOL
    # and INVALID_ARGUMENTS are written by call_tool before the tool runs (_error_body),
    # and a TypeError from inside a tool is a tool bug, not an argument error.
    @pytest.mark.parametrize("exc,code", [
        (TimeoutError("Query exceeded timeout"), "TIMEOUT"),
        (TypeError("unsupported operand"), "TOOL_EXECUTION_ERROR"),
        (RuntimeError("boom"), "TOOL_EXECUTION_ERROR"),
    ])
    def test_tool_error_payload_code(self, exc, code):
        payload = _tool_error_payload("nfl_query", exc)
        assert isinstance(payload["code"], ErrorCode)
        assert payload["code"] is ErrorCode(code)
        assert payload["code"] == code
        assert payload["error"] == str(exc)

    @pytest.mark.parametrize("code", [ErrorCode.UNKNOWN_TOOL, ErrorCode.INVALID_ARGUMENTS])
    def test_server_error_body_code(self, code):
        payload = _error_body("nfl_query", code, "message")
        assert payload["code"] is code
        assert json.dumps(payload).count(f'"{code.value}"') == 1

    def test_tool_error_payload_serializes_to_plain_value(self):
        payload = json.dumps(_tool_error_payload("nfl_query", TimeoutError("slow")), indent=2)
        assert '"code": "TIMEOUT"' in payload
        assert "ErrorCode." not in payload

    def test_column_not_found(self, monkeypatch):
        monkeypatch.setattr(tools, "_live_names", lambda sql: ["epa", "play_type"])
        exc = duckdb.BinderException('Binder Error: Referenced column "play_epa" not found!')
        detail = tools._sql_error_detail(exc)
        assert detail["code"] is ErrorCode.COLUMN_NOT_FOUND
        assert detail["code"] == "COLUMN_NOT_FOUND"
        assert detail["unknown_column"] == "play_epa"

    def test_table_not_found(self, monkeypatch):
        monkeypatch.setattr(tools, "_live_names", lambda sql: ["plays", "schedules"])
        exc = duckdb.CatalogException("Catalog Error: Table with name playz does not exist!")
        detail = tools._sql_error_detail(exc)
        assert detail["code"] is ErrorCode.TABLE_NOT_FOUND
        assert detail["code"] == "TABLE_NOT_FOUND"
        assert detail["unknown_table"] == "playz"

    def test_unclassifiable_error_has_no_code(self, monkeypatch):
        monkeypatch.setattr(tools, "_live_names", lambda sql: [])
        assert tools._sql_error_detail(duckdb.Error("syntax error at or near \"SELEKT\"")) == {}

    def test_sql_detail_serializes_to_plain_value(self, monkeypatch):
        monkeypatch.setattr(tools, "_live_names", lambda sql: ["plays"])
        exc = duckdb.CatalogException("Catalog Error: Table with name playz does not exist!")
        assert '"code": "TABLE_NOT_FOUND"' in json.dumps(tools._sql_error_detail(exc), indent=2)
