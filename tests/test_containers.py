"""Result and trace containers round-trip against the Stage 1 schema oracle."""

from conftest import load_fixture, validate_against
from local_judge import (
    AttemptRecord,
    ErrorObject,
    ResultEntry,
    ResultStatus,
    TraceRecord,
)


def test_result_entry_from_question_error_fixture():
    fixture = load_fixture("malformed-question-sibling.response.json")
    urgency = fixture["results"]["urgency"]
    entry = ResultEntry.from_dict(urgency)
    assert entry.status is ResultStatus.QUESTION_ERROR
    assert entry.answer is None
    assert entry.error is not None and entry.error.code == "INVALID_QUESTION"
    assert entry.error.path == ""
    assert entry.requested_samples == 1
    assert list(entry.trace.attempts) == []
    validate_against(entry.to_dict(), "#/$defs/result")
    assert entry.to_dict() == urgency


def test_trace_record_round_trip_from_choice_fixture():
    fixture = load_fixture("choice-valid.response.json")
    trace = fixture["results"]["department"]["trace"]
    record = TraceRecord.from_dict(trace)
    assert record.parent_trace_id is None
    assert record.policy_provenance == "caller-declared-unverified"
    assert record.attempts[0].parsed_value == {"choice": "technical"}
    assert record.attempts[0].terminal_error is None
    validate_against(record.to_dict(), "#/$defs/trace")
    assert record.to_dict() == trace


def test_answered_result_round_trip():
    fixture = load_fixture("noul-decile.response.json")
    entry = ResultEntry.from_dict(fixture["results"]["is_refund"])
    assert entry.status is ResultStatus.ANSWERED
    assert entry.answer == {"noul": 0.86}
    assert entry.agreement == 0.6666666666666666
    validate_against(entry.to_dict(), "#/$defs/result")
    assert entry.to_dict() == fixture["results"]["is_refund"]


def test_error_object_enforces_path_rule():
    assert ErrorObject(code="MODEL_TIMEOUT", path="", message="m")
    assert ErrorObject(code="UNKNOWN_FIELD", path="/policy/authority", message="m")
    import pytest

    with pytest.raises(ValueError):
        ErrorObject(code="MODEL_TIMEOUT", path="/model", message="m")
    with pytest.raises(ValueError):
        ErrorObject(code="UNKNOWN_FIELD", path="", message="m")
    with pytest.raises(ValueError):
        ErrorObject(code="MODEL_TIMEOUT", path="not-a-pointer", message="m")
    with pytest.raises(ValueError):
        AttemptRecord(
            raw_output="",
            parsed_value=None,
            validation_outcome="timeout",
            timestamp="2026-09-22T12:00:00Z",
            terminal_error=ErrorObject(code="MALFORMED_JSON", path="", message="m"),
        )
