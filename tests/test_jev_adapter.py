"""Jev adapter: input conversion, mapping, refusal, disclosure (issue #16)."""

import json

from conftest import TEST_PROFILE_SUPPORTED, load_fixture, validate_against
from local_judge import (
    AttemptRecord,
    ErrorObject,
    RequestValidator,
    ResultEntry,
    ResultStatus,
    TraceRecord,
)
from local_judge.adapter import JevAdapter
from local_judge.ports import RawAttempt, TransportOutcome


def adapter():
    return JevAdapter(
        validator=RequestValidator({"qwen3:8b": TEST_PROFILE_SUPPORTED}),
        model_profiles={"qwen3:8b": TEST_PROFILE_SUPPORTED},
    )


def native_answered(qid, trace_id_suffix, answer, requested_samples=3, agreement=1.0):
    from test_interfaces import minimal_trace

    trace = TraceRecord.from_dict({**minimal_trace().to_dict(), "trace_id": f"00000000-0000-4000-8000-{trace_id_suffix}"})
    return ResultEntry(
        type_=answer_type(answer),
        status=ResultStatus.ANSWERED,
        answer=answer,
        agreement=agreement,
        requested_samples=requested_samples,
        error=None,
        trace=trace,
    )


def answer_type(answer):
    if "choice" in answer:
        return "choice"
    if "score" in answer:
        return "score"
    return "noul"


def jev_request():
    return load_fixture("jev-adapter-success.request.json")


def test_input_converts_to_native_envelope_with_fixed_defaults():
    envelope = adapter().convert_input(jev_request())
    assert envelope.policy.version == "jev-adapter-v1"
    assert envelope.inference.sample_count == 3
    assert envelope.inference.temperature == 0
    assert envelope.inference.timeout_ms == 30000
    assert envelope.state == jev_request()["state"]


def test_converted_input_passes_the_native_schema():
    converted = adapter().convert_input(jev_request())
    from conftest import validate_against as va

    va(_envelope_dict(converted), "#/$defs/requestEnvelope")


def _envelope_dict(envelope):
    d = {
        "contract_version": "v1",
        "state": envelope.state,
        "model": envelope.model,
        "policy": {"version": envelope.policy.version},
        "inference": {
            "sample_count": envelope.inference.sample_count,
            "temperature": envelope.inference.temperature,
            "timeout_ms": envelope.inference.timeout_ms,
        },
        "questions": {qid: q.raw for qid, q in envelope.questions.items()},
    }
    return d


def test_unknown_profile_rejected_as_structural():
    from local_judge import StructuralCode

    def unreachable_runner(envelope):
        raise AssertionError("transport reached for an unknown profile")

    result = adapter().evaluate({"state": "s", "model": "jev-latest",
                                 "questions": {"q": {"type": "noul", "instructions": "i"}}},
                                unreachable_runner)
    assert result["answers"] is None and result["local_judge"] is None
    assert result["error"]["code"] == StructuralCode.UNSUPPORTED_LOCAL_MODEL.value


def test_complete_native_result_maps_to_jev_answers():
    choice_answer = {"choice": "technical", "vote_share": {"billing": 0, "technical": 1}}
    noul_answer = {"noul": 0.9}
    results = {
        "department": native_answered("department", "0000000000A1", choice_answer),
        "is_refund": native_answered("is_refund", "0000000000A2", noul_answer),
    }
    result = adapter().evaluate_with_results(jev_request(), results)
    assert result["error"] is None
    assert result["answers"]["department"]["confidence"] == 1.0
    assert result["answers"]["department"]["probabilities"] == {"billing": 0, "technical": 1}
    assert result["answers"]["is_refund"] == {"type": "noul", "noul": 0.9}
    assert result["local_judge"]["confidence_disclosure"] == (
        "confidence is local repeated-sample agreement, not Jev or calibrated confidence"
    )
    validate_against(result, "#/$defs/jevAdapterResult")


def test_unmappable_result_never_invents_answers():
    from local_judge.ports import RawAttempt as _  # noqa: F401
    from test_interfaces import minimal_trace

    fallback = ResultEntry(
        type_="choice",
        status=ResultStatus.QUESTION_ERROR,
        answer=None,
        agreement=None,
        requested_samples=3,
        error=ErrorObject(code="INVALID_QUESTION", path="", message="bad question"),
        trace=TraceRecord.from_dict({**minimal_trace().to_dict(), "trace_id": "00000000-0000-4000-8000-0000000000B1"}),
    )
    results = {"q": fallback}
    request = {"state": "s", "model": "qwen3:8b", "questions": {"q": {"type": "noul", "instructions": "i"}}}
    result = adapter().evaluate_with_results(request, results)
    assert result["answers"] is None
    assert result["error"]["code"] == "JEV_ADAPTER_UNMAPPABLE_RESULT"
    assert result["local_judge"]["traces"]["q"] is not None
    validate_against(result, "#/$defs/jevAdapterResult")


def test_score_mapping_carries_legend_probabilities_and_disclosed_confidence():
    score_answer = {
        "score": 1.5,
        "legend": {"0": "Cosmetic", "1": "Workaround", "2": "Blocking"},
        "vote_share": {"0": 0, "1": 0.5, "2": 0.5},
    }
    request = {
        "state": {"ticket": "Checkout degraded."},
        "model": "qwen3:8b",
        "questions": {"severity": {"type": "score", "instructions": "Severity?", "criteria": ["Cosmetic", "Workaround", "Blocking"]}},
    }
    results = {"severity": native_answered("severity", "0000000000C1", score_answer, agreement=0.5)}
    result = adapter().evaluate_with_results(request, results)
    assert result["answers"]["severity"] == {
        "type": "score",
        "score": 1.5,
        "probabilities": {"0": 0, "1": 0.5, "2": 0.5},
        "legend": {"0": "Cosmetic", "1": "Workaround", "2": "Blocking"},
        "confidence": 0.5,
    }
    validate_against(result, "#/$defs/jevAdapterResult")


def test_missing_requested_question_refuses():
    results = {"department": native_answered("department", "0000000000D1", {"choice": "billing", "vote_share": {"billing": 1}})}
    request = {
        "state": "s",
        "model": "qwen3:8b",
        "questions": {
            "department": {"type": "choice", "instructions": "i", "criteria": {"billing": "b"}},
            "missing": {"type": "noul", "instructions": "i"},
        },
    }
    result = adapter().evaluate_with_results(request, results)
    assert result["answers"] is None
    assert result["error"]["code"] == "JEV_ADAPTER_UNMAPPABLE_RESULT"


def test_unrequested_result_refuses():
    results = {
        "department": native_answered("department", "0000000000D2", {"choice": "billing", "vote_share": {"billing": 1}}),
        "ghost": native_answered("ghost", "0000000000D3", {"noul": 0.5}),
    }
    result = adapter().evaluate_with_results(jev_request(), results)
    assert result["answers"] is None
    assert result["error"]["code"] == "JEV_ADAPTER_UNMAPPABLE_RESULT"


def test_answered_single_sample_without_agreement_refuses():
    results = {"department": native_answered("department", "0000000000D4", {"choice": "billing", "vote_share": {"billing": 1}}, requested_samples=1, agreement=None)}
    result = adapter().evaluate_with_results(jev_request(), results)
    assert result["answers"] is None
    assert result["error"]["code"] == "JEV_ADAPTER_UNMAPPABLE_RESULT"


def test_missing_jev_fields_report_missing_field():
    result = adapter().evaluate({}, None)
    assert result["answers"] is None and result["local_judge"] is None
    assert result["error"]["code"] == "MISSING_FIELD"
    assert result["error"]["path"] == "/state"


def test_non_string_jev_key_refuses_without_crashing():
    result = adapter().evaluate({1: "x", "state": "s", "model": "qwen3:8b",
                                 "questions": {"q": {"type": "noul", "instructions": "i"}}}, None)
    assert result["error"]["code"] == "UNKNOWN_FIELD"


def test_stage_seven_adapter_fixtures_validate_against_the_schema():
    success = load_fixture("jev-adapter-success.result.json")
    validate_against(success, "#/$defs/jevAdapterResult")
    unmappable = load_fixture("jev-adapter-unmappable.result.json")
    validate_against(unmappable, "#/$defs/jevAdapterResult")
    validate_against(load_fixture("jev-adapter-success.request.json"), "#/$defs/jevInput")


def test_wrong_typed_result_for_a_question_refuses():
    results = {
        "is_refund": native_answered("is_refund", "0000000000E1", {"choice": "yes", "vote_share": {"yes": 1}}),
    }
    request = {
        "state": "s",
        "model": "qwen3:8b",
        "questions": {"is_refund": {"type": "noul", "instructions": "i"}},
    }
    result = adapter().evaluate_with_results(request, results)
    assert result["answers"] is None
    assert result["error"]["code"] == "JEV_ADAPTER_UNMAPPABLE_RESULT"
    validate_against(result, "#/$defs/jevAdapterResult")
