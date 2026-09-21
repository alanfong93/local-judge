"""Every Stage 1 structural code gets its prescribed closed rejection (issue #8)."""

import pytest

from conftest import TEST_PROFILE_SUPPORTED, load_fixture
from local_judge import ModelProfile, RequestValidator, StructuralCode, StructuralError


def validator(profiles=None):
    registry = {"qwen3:8b": TEST_PROFILE_SUPPORTED, "seedless": TEST_PROFILE_SUPPORTED - {"seed"}}
    registry.update(profiles or {})
    return RequestValidator(registry)


def base_request(**overrides):
    req = {
        "contract_version": "v1",
        "state": {"ticket": "Example evidence"},
        "model": "qwen3:8b",
        "policy": {"version": "support-triage-2026-09-21"},
        "inference": {"sample_count": 1, "temperature": 0, "timeout_ms": 30000},
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "Payments", "technical": "Bugs"},
            }
        },
    }
    req.update(overrides)
    return req


def expect_rejection(raw, code, path):
    with pytest.raises(StructuralError) as excinfo:
        validator().parse(raw)
    err = excinfo.value.error
    assert err.code == code, f"expected {code}, got {err.code}: {err.message}"
    assert err.path == path, f"expected path {path!r}, got {err.path!r}"
    assert err.message
    return err


def test_unknown_envelope_field_fixture_is_rejected():
    raw = load_fixture("structural-rejection.request.json")
    expect_rejection(raw, StructuralCode.UNKNOWN_FIELD, "/temperature")


def test_malformed_json():
    expect_rejection("{not json", StructuralCode.MALFORMED_JSON, "")


def test_non_json_constants_are_malformed():
    expect_rejection(
        '{"contract_version": "v1", "state": {"x": NaN}, "model": "qwen3:8b",'
        ' "policy": {"version": "p"}, "inference": {}, "questions": {"q": {"type": "noul", "instructions": "i"}}}',
        StructuralCode.MALFORMED_JSON,
        "",
    )


def test_unsupported_contract_version_including_missing():
    expect_rejection(base_request(contract_version="v2"), StructuralCode.UNSUPPORTED_CONTRACT_VERSION, "/contract_version")
    expect_rejection(base_request(contract_version=1), StructuralCode.UNSUPPORTED_CONTRACT_VERSION, "/contract_version")
    req = base_request()
    del req["contract_version"]
    expect_rejection(req, StructuralCode.UNSUPPORTED_CONTRACT_VERSION, "/contract_version")


def test_missing_and_invalid_state():
    req = base_request()
    del req["state"]
    expect_rejection(req, StructuralCode.MISSING_FIELD, "/state")
    expect_rejection(base_request(state=17), StructuralCode.INVALID_FIELD, "/state")


def test_missing_and_invalid_model():
    req = base_request()
    del req["model"]
    expect_rejection(req, StructuralCode.MISSING_FIELD, "/model")
    expect_rejection(base_request(model=""), StructuralCode.INVALID_FIELD, "/model")


def test_unknown_model_profile():
    expect_rejection(
        base_request(model="jev-latest"), StructuralCode.UNSUPPORTED_LOCAL_MODEL, "/model"
    )


def test_policy_is_closed_and_version_required():
    req = base_request()
    req["policy"] = {"version": "p", "authority": "internal"}
    expect_rejection(req, StructuralCode.UNKNOWN_FIELD, "/policy/authority")
    expect_rejection(base_request(policy={}), StructuralCode.MISSING_FIELD, "/policy/version")
    expect_rejection(base_request(policy={"version": ""}), StructuralCode.INVALID_FIELD, "/policy/version")
    req2 = base_request()
    del req2["policy"]
    expect_rejection(req2, StructuralCode.MISSING_FIELD, "/policy")


def test_inference_is_closed_with_bounds():
    req = base_request()
    req["inference"] = {"bogus": 1}
    expect_rejection(req, StructuralCode.UNKNOWN_FIELD, "/inference/bogus")
    req = base_request()
    req["inference"] = {"sample_count": 6}
    expect_rejection(req, StructuralCode.INVALID_FIELD, "/inference/sample_count")
    req = base_request()
    req["inference"] = {"temperature": 3}
    expect_rejection(req, StructuralCode.INVALID_FIELD, "/inference/temperature")
    req = base_request()
    req["inference"] = {"timeout_ms": 0}
    expect_rejection(req, StructuralCode.INVALID_FIELD, "/inference/timeout_ms")
    req = base_request()
    del req["inference"]
    expect_rejection(req, StructuralCode.MISSING_FIELD, "/inference")


def test_unsupported_inference_setting_before_model_calls():
    expect_rejection(
        base_request(model="seedless", inference={"seed": 7}),
        StructuralCode.UNSUPPORTED_INFERENCE_SETTING,
        "/inference/seed",
    )


def test_invalid_questions_map():
    expect_rejection(base_request(questions=[]), StructuralCode.INVALID_QUESTIONS_MAP, "/questions")
    expect_rejection(base_request(questions={}), StructuralCode.INVALID_QUESTIONS_MAP, "/questions")
    req = base_request()
    del req["questions"]
    expect_rejection(req, StructuralCode.MISSING_FIELD, "/questions")


def test_duplicate_question_id_in_raw_json():
    raw = (
        '{"contract_version": "v1", "state": "s", "model": "qwen3:8b",'
        ' "policy": {"version": "p"}, "inference": {},'
        ' "questions": {"a": {"type": "noul", "instructions": "i"},'
        '              "a": {"type": "noul", "instructions": "j"}}}'
    )
    expect_rejection(raw, StructuralCode.DUPLICATE_QUESTION_ID, "")


def test_duplicate_inside_a_question_object_is_not_structural():
    """A dup inside one question's body is that question's typed concern, not a whole-request fault."""
    raw = (
        '{"contract_version": "v1", "state": "s", "model": "qwen3:8b",'
        ' "policy": {"version": "p"}, "inference": {},'
        ' "questions": {"a": {"type": "noul", "instructions": "i", "instructions": "j"}}}'
    )
    envelope = validator().parse(raw)
    assert envelope.questions["a"].raw["instructions"] == "j"


def test_duplicate_beats_question_count():
    """A raw request with both defects reports the duplicate: dup is checked before count."""
    members = ", ".join(
        f'"q{i}": {{"type": "noul", "instructions": "yes?"}}' for i in range(1, 66)
    )
    raw = (
        '{"contract_version": "v1", "state": "s", "model": "qwen3:8b",'
        ' "policy": {"version": "p"}, "inference": {},'
        ' "questions": {' + members + ', "q1": {"type": "noul", "instructions": "dup"}}}'
    )
    expect_rejection(raw, StructuralCode.DUPLICATE_QUESTION_ID, "")


def test_root_duplicate_field_names_the_member():
    expect_rejection(
        '{"contract_version": "v1", "contract_version": "v2", "state": "s", "model": "m",'
        ' "policy": {"version": "p"}, "inference": {}, "questions": {"q": {"type": "noul", "instructions": "i"}}}',
        StructuralCode.INVALID_FIELD,
        "/contract_version",
    )


def test_oversize_request_and_question_count():
    expect_rejection(
        base_request(state="x" * (256 * 1024)),
        StructuralCode.REQUEST_TOO_LARGE,
        "",
    )
    questions = {
        f"q{i}": {"type": "noul", "instructions": "yes?"} for i in range(65)
    }
    expect_rejection(base_request(questions=questions), StructuralCode.REQUEST_TOO_LARGE, "")


def test_escaped_pointer_tokens_for_hostile_keys():
    req = base_request()
    req["policy"] = {"version": "p", "a/b~c": 1}
    expect_rejection(req, StructuralCode.UNKNOWN_FIELD, "/policy/a~1b~0c")


def test_rejection_response_container_matches_published_schema():
    from local_judge import ErrorObject, RejectionResponse

    fixture_error = load_fixture("structural-rejection.response.json")["error"]
    response = RejectionResponse(
        contract_version="v1", model="qwen3:8b", error=ErrorObject(**fixture_error)
    )
    as_dict = response.to_dict()
    from conftest import validate_against

    validate_against(as_dict, "#/$defs/rejectedResponse")
    assert as_dict == load_fixture("structural-rejection.response.json")


def test_non_object_request_body_is_malformed():
    expect_rejection("[1, 2, 3]", StructuralCode.MALFORMED_JSON, "")



def test_oversize_is_rejected_before_parsing():
    """The encoded-size cap fires on the bytes, even when the body is also malformed."""
    expect_rejection(
        '{"state": "' + "x" * (256 * 1024),  # over the cap AND not valid JSON (unterminated)
        StructuralCode.REQUEST_TOO_LARGE,
        "",
    )
    expect_rejection('{"state": "' + "x" * (256 * 1024) + '"}', StructuralCode.REQUEST_TOO_LARGE, "")
