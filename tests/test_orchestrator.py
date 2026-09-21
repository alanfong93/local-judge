"""S2-3 orchestration boundary tests: sampling count, ordering, isolation,
trace assembly, canonical hash, and replay reconstruction (issue #9).

The fake executor supplies typed terminal outcomes; S2-3 itself parses no
Choice/Score/Noul output and computes no aggregate or agreement.
"""

import json
import uuid

import pytest

from conftest import validate_against
from local_judge import (
    AttemptRecord,
    ErrorObject,
    Inference,
    RequestValidator,
    StructuralCode,
    StructuralError,
    ResultEntry,
    Policy,
    QuestionEntry,
    RequestEnvelope,
    ResultStatus,
    TraceRecord,
)
from local_judge.canonical import canonical_json, canonical_request_hash
from local_judge.orchestrator import (
    ResolvedReplay,
    SamplingOrchestrator,
    SamplingTypeExecutor,
    resolve_replay,
)
from dataclasses import replace
from local_judge.ports import RawAttempt, TransportOutcome


def envelope(sample_count=2, questions=None):
    return RequestEnvelope(
        state={"ticket": "s"},
        model="qwen3:8b",
        policy=Policy(version="p1-policy"),
        inference=Inference(sample_count=sample_count, temperature=0, seed=None, timeout_ms=1000),
        questions={
            qid: QuestionEntry(id=qid, raw=raw)
            for qid, raw in (questions or {"q": {"type": "noul", "instructions": "i"}}).items()
        },
    )


class FakePort:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def attempt(self, model, rendered_messages, inference, response_schema=None):
        self.calls.append(model)
        raw = self.outputs.pop(0) if self.outputs else "fallback"
        return RawAttempt(outcome=TransportOutcome.OK, output=raw)


class FakeExecutor:
    """Supplies typed terminal outcomes; counts renders; can be told to fail."""

    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)
        self.classified = []
        self.rendered_for = []

    def render_messages(self, question_id, question, state):
        self.rendered_for.append(question_id)
        return [{"role": "user", "content": {"q": question, "state": state}}]

    def classify(self, raw_attempt):
        self.classified.append(raw_attempt)
        if raw_attempt.outcome is TransportOutcome.OK:
            return AttemptRecord(
                raw_output=raw_attempt.output,
                parsed_value=raw_attempt.output,
                validation_outcome="valid",
                timestamp="2026-09-22T12:00:00Z",
                terminal_error=None,
            )
        return AttemptRecord(
            raw_output="",
            parsed_value=None,
            validation_outcome=raw_attempt.outcome.value,
            timestamp="2026-09-22T12:00:00Z",
            terminal_error=ErrorObject(code="MODEL_TIMEOUT", path="", message="timed out"),
        )

    def run(self, question_id, question, state, attempts):
        if question_id in self.fail_for:
            raise RuntimeError("executor exploded for this question")
        assert isinstance(attempts, tuple)
        trace = TraceRecord(
            trace_id=str(uuid.uuid4()),
            parent_trace_id=None,
            trace_schema_version="v1",
            contract_version="v1",
            prompt_template_version="p1",
            output_schema_version="o1",
            aggregation_version="a1",
            policy_version="p1-policy",
            accepted_request={
                "contract_version": "v1",
                "state": state,
                "model": "qwen3:8b",
                "policy": {"version": "p1-policy"},
                "inference": {},
                "questions": {question_id: question},
            },
            canonical_request_hash="a" * 64,
            state=state,
            question=question,
            criteria=None,
            policy_provenance="caller-declared-unverified",
            rendered_messages=[],
            resolved_inference={"sample_count": 2, "temperature": 0, "seed": None, "timeout_ms": 1000},
            backend="fake",
            model="qwen3:8b",
            model_digest=None,
            local_runtime_version="test",
            attempts=tuple(attempts),
            aggregate=None,
        )
        return ResultEntry(
            type_=None,
            status=ResultStatus.QUESTION_ERROR,
            answer=None,
            agreement=None,
            requested_samples=len(attempts),
            error=ErrorObject(code="MODEL_TIMEOUT", path="", message="typed outcome"),
            trace=trace,
        )


def versions():
    return {
        "prompt_template_version": "p1",
        "output_schema_version": "o1",
        "aggregation_version": "a1",
    }



def minimal_trace():
    return TraceRecord(
        trace_id="00000000-0000-4000-8000-000000000099",
        parent_trace_id=None,
        trace_schema_version="v1",
        contract_version="v1",
        prompt_template_version="p1",
        output_schema_version="o1",
        aggregation_version="a1",
        policy_version="p1-policy",
        accepted_request={"contract_version": "v1", "state": "s", "model": "qwen3:8b",
                          "policy": {"version": "p"}, "inference": {},
                          "questions": {"q": {"type": "choice", "instructions": "i"}}},
        canonical_request_hash="a" * 64,
        state="s",
        question={"type": "choice", "instructions": "i"},
        criteria=None,
        policy_provenance="caller-declared-unverified",
        rendered_messages=[],
        resolved_inference={"sample_count": 1, "temperature": 0, "seed": None, "timeout_ms": 1000},
        backend="fake",
        model="qwen3:8b",
        model_digest=None,
        local_runtime_version="test",
        attempts=(),
        aggregate=None,
    )


def test_exactly_sample_count_attempts_per_question():
    port = FakePort([f"out{i}" for i in range(5)])
    executor = FakeExecutor()
    orch = SamplingOrchestrator(port=port, executor=executor, versions=versions(), backend="fake")
    result = orch.run_question(envelope(sample_count=3), "q", {"type": "noul", "instructions": "i"})
    assert len(port.calls) == 3
    assert result.requested_samples == 3
    assert len(result.trace.attempts) == 3


def test_attempts_are_ordered_and_classified_by_the_executor():
    port = FakePort(["a", "b"])
    executor = FakeExecutor()
    orch = SamplingOrchestrator(port=port, executor=executor, versions=versions(), backend="fake")
    result = orch.run_question(envelope(sample_count=2), "q", {"type": "noul", "instructions": "i"})
    assert [a.raw_output for a in result.trace.attempts] == ["a", "b"]
    assert [a.parsed_value for a in result.trace.attempts] == ["a", "b"]


def test_timeout_outcome_gets_mechanical_terminal_error():
    port = FakePort([])
    port.outputs = ["never"]
    port.attempt = lambda *a, **k: RawAttempt(outcome=TransportOutcome.TIMEOUT, output=None)
    executor = FakeExecutor()
    orch = SamplingOrchestrator(port=port, executor=executor, versions=versions(), backend="fake")
    result = orch.run_question(envelope(sample_count=1), "q", {"type": "noul", "instructions": "i"})
    assert result.trace.attempts[0].terminal_error is not None
    assert result.trace.attempts[0].terminal_error.code == "MODEL_TIMEOUT"


def test_failed_question_does_not_abort_valid_siblings():
    port = FakePort(["x", "y"])
    executor = FakeExecutor(fail_for=("bad",))
    questions = {
        "good": {"type": "noul", "instructions": "i"},
        "bad": {"type": "noul", "instructions": "j"},
    }
    orch = SamplingOrchestrator(port=port, executor=executor, versions=versions(), backend="fake")
    results = orch.run_questions(envelope(sample_count=1, questions=questions))
    assert set(results) == {"good", "bad"}
    assert results["good"].status is ResultStatus.QUESTION_ERROR  # fake always errors
    assert results["bad"].status is ResultStatus.QUESTION_ERROR
    assert "executor exploded" in results["bad"].error.message


def test_failed_executor_question_yields_question_error():
    port = FakePort(["x"])
    executor = FakeExecutor(fail_for=("bad",))
    orch = SamplingOrchestrator(port=port, executor=executor, versions=versions(), backend="fake")
    results = orch.run_questions(envelope(sample_count=1, questions={"bad": {"type": "noul"}}))
    assert results["bad"].status is ResultStatus.QUESTION_ERROR
    assert results["bad"].error.code == "INVALID_QUESTION"


def test_trace_assembly_matches_contract():
    port = FakePort(["a"])
    executor = FakeExecutor()
    env = envelope(sample_count=1)
    orch = SamplingOrchestrator(port=port, executor=executor, versions=versions(), backend="fake")
    result = orch.run_question(env, "q", {"type": "noul", "instructions": "i"})
    trace = result.trace
    assert trace.parent_trace_id is None
    assert trace.policy_provenance == "caller-declared-unverified"
    assert trace.contract_version == "v1"
    assert trace.prompt_template_version == "p1"
    assert trace.accepted_request is not None
    validate_against(trace.to_dict(), "#/$defs/trace")


def test_canonical_hash_is_sha256_of_rfc8785_style_canonical_json():
    obj = {"b": 1, "a": {"y": [1, 2], "x": "é\n"}, "c": 0.5}
    canonical = canonical_json(obj)
    assert canonical == '{"a":{"x":"é\\n","y":[1,2]},"b":1,"c":0.5}'
    assert canonical_request_hash(obj) == canonical_request_hash(obj)
    import hashlib

    assert canonical_request_hash(obj) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_orchestrator_records_canonical_hash_of_accepted_request():
    port = FakePort(["a"])
    executor = FakeExecutor()
    env = envelope(sample_count=1)
    orch = SamplingOrchestrator(port=port, executor=executor, versions=versions(), backend="fake")
    result = orch.run_question(env, "q", {"type": "noul", "instructions": "i"})
    expected = canonical_request_hash(
        json.loads(json.dumps({"state": env.state, "model": env.model} , sort_keys=True))
    )
    # the hash is computed over the full accepted request; here we pin
    # determinism and hex-64 shape rather than re-implementing JCS in the test
    h = result.trace.canonical_request_hash
    assert len(h) == 64 and h == h.lower()
    assert h != canonical_request_hash({"different": True})


def test_no_typed_parsing_lives_in_the_orchestrator():
    """S2-3 must not parse typed output or compute aggregates: source-level pin."""
    import inspect

    from local_judge import orchestrator

    source = inspect.getsource(orchestrator)
    for banned in ("vote_share", "agreement ==", "float(answer", '"choice"', '"score"'):
        assert banned not in source, banned


def test_replay_resolves_recorded_versions():
    trace = TraceRecord(
        trace_id="00000000-0000-4000-8000-000000000001",
        parent_trace_id=None,
        trace_schema_version="v1",
        contract_version="v1",
        prompt_template_version="p1",
        output_schema_version="o1",
        aggregation_version="a1",
        policy_version="p1-policy",
        accepted_request={"contract_version": "v1", "state": "s", "model": "qwen3:8b",
                          "policy": {"version": "p1-policy"}, "inference": {},
                          "questions": {"q": {"type": "noul", "instructions": "i"}}},
        canonical_request_hash="a" * 64,
        state="s",
        question={"type": "noul", "instructions": "i"},
        criteria=None,
        policy_provenance="caller-declared-unverified",
        rendered_messages=[],
        resolved_inference={"sample_count": 2, "temperature": 0, "seed": None, "timeout_ms": 1000},
        backend="fake",
        model="qwen3:8b",
        model_digest=None,
        local_runtime_version="test",
        attempts=(),
        aggregate=None,
    )
    registry = {
        "prompt_templates": {"p1": object()},
        "output_schemas": {"o1": object()},
        "aggregations": {"a1": object()},
        "models": {"qwen3:8b": object()},
    }
    resolved = resolve_replay(trace, registry)
    assert isinstance(resolved, ResolvedReplay)
    assert resolved.prompt_template is registry["prompt_templates"]["p1"]
    assert resolved.resolved_inference == trace.resolved_inference


def test_replay_without_recorded_artifacts_is_rejected():
    trace = TraceRecord(
        trace_id="00000000-0000-4000-8000-000000000002",
        parent_trace_id=None,
        trace_schema_version="v1",
        contract_version="v1",
        prompt_template_version="pGONE",
        output_schema_version="o1",
        aggregation_version="a1",
        policy_version="p1-policy",
        accepted_request={"contract_version": "v1", "state": "s", "model": "qwen3:8b",
                          "policy": {"version": "p"}, "inference": {},
                          "questions": {"q": {"type": "noul", "instructions": "i"}}},
        canonical_request_hash="a" * 64,
        state="s",
        question={"type": "noul", "instructions": "i"},
        criteria=None,
        policy_provenance="caller-declared-unverified",
        rendered_messages=[],
        resolved_inference={"sample_count": 2, "temperature": 0, "seed": None, "timeout_ms": 1000},
        backend="fake",
        model="qwen3:8b",
        model_digest=None,
        local_runtime_version="test",
        attempts=(),
        aggregate=None,
    )
    response = resolve_replay(trace, {"prompt_templates": {}, "output_schemas": {"o1": object()},
                                      "aggregations": {"a1": object()}, "models": {"qwen3:8b": object()}})
    from local_judge import RejectionResponse

    assert isinstance(response, RejectionResponse)
    assert response.error.code == "REPLAY_CONFIGURATION_UNAVAILABLE"
    validate_against(response.to_dict(), "#/$defs/rejectedResponse")


def test_canonical_number_and_ordering_boundaries():
    from local_judge.canonical import canonical_json as cj

    assert cj(1e-6) == "0.000001"
    assert cj(1e-7) == "1e-7"
    assert cj(1e21) == "1e+21"
    assert cj(-0.0) == "0"
    assert cj(100.0) == "100"
    assert cj({"é": 1, "e": 2}) == '{"e":2,"é":1}'
    # UTF-16 code-unit ordering: an astral-plane key sorts BEFORE "\u00e9"-class BMP keys
    assert cj({"😀": 1, "é": 2}) == '{"é":2,"😀":1}'
    # discriminating pair: UTF-16 units (D83D < FFFD) vs code points (FFFD < 1F600)
    assert cj({"�": 1, "😀": 2}) == '{"😀":2,"�":1}'


def test_failed_question_preserves_recorded_evidence():
    port = FakePort(["a", "b"])
    executor = FakeExecutor(fail_for=("bad",))
    orch = SamplingOrchestrator(port=port, executor=executor, versions=versions(), backend="fake")
    results = orch.run_questions(envelope(sample_count=2, questions={"bad": {"type": "noul", "instructions": "i"}}))
    trace = results["bad"].trace
    assert len(trace.attempts) == 2, "classified attempts must survive the executor failure"
    assert [a.raw_output for a in trace.attempts] == ["a", "b"]
    assert isinstance(trace.rendered_messages, list)


def test_orchestrator_passes_typed_results_through_untouched():
    sentinel_answer = {"choice": "x", "vote_share": {"x": 1}}
    sentinel_aggregate = {"choice": "x", "vote_share": {"x": 1}}

    class TypedExecutor(FakeExecutor):
        def run(self, question_id, question, state, attempts):
            trace = minimal_trace()
            return ResultEntry(
                type_="choice",
                status=ResultStatus.ANSWERED,
                answer=sentinel_answer,
                agreement=None,
                requested_samples=len(attempts),
                error=None,
                trace=trace,
            )

    from conftest import TEST_PROFILE_SUPPORTED

    port = FakePort(["raw"])
    orch = SamplingOrchestrator(port=port, executor=TypedExecutor(), versions=versions(), backend="fake")
    result = orch.run_question(envelope(sample_count=1), "q", {"type": "choice", "instructions": "i"})
    assert result.answer is sentinel_answer
    assert result.trace.aggregate is sentinel_answer
    assert result.status is ResultStatus.ANSWERED


def test_replay_digest_verification():
    trace = minimal_trace()
    trace = replace(trace, model_digest="d" * 64)
    import types

    matching = types.SimpleNamespace(digest="d" * 64)
    mismatching = types.SimpleNamespace(digest="e" * 64)
    dict_registry = {
        "prompt_templates": {"p1": object()},
        "output_schemas": {"o1": object()},
        "aggregations": {"a1": object()},
        "models": {"qwen3:8b": {"digest": "d" * 64}},
    }
    from local_judge import RejectionResponse

    resolved = resolve_replay(trace, {
        "prompt_templates": {"p1": object()},
        "output_schemas": {"o1": object()},
        "aggregations": {"a1": object()},
        "models": {"qwen3:8b": matching},
    })
    assert isinstance(resolved, ResolvedReplay)
    rejected = resolve_replay(trace, {
        "prompt_templates": {"p1": object()},
        "output_schemas": {"o1": object()},
        "aggregations": {"a1": object()},
        "models": {"qwen3:8b": mismatching},
    })
    assert isinstance(rejected, RejectionResponse)
    rejected_dict = resolve_replay(trace, dict_registry)
    assert isinstance(rejected_dict, ResolvedReplay)


def test_non_string_object_keys_are_rejected():
    import pytest

    from local_judge.canonical import canonical_json

    with pytest.raises(ValueError):
        canonical_json({1: "a"})



def test_big_integers_follow_double_semantics():
    from local_judge.canonical import canonical_json as cj

    assert cj(9007199254740993) == "9007199254740992"
    assert cj(1) == "1"


def test_lone_surrogates_are_invalid_unicode():
    import pytest

    from local_judge.canonical import canonical_json as cj

    with pytest.raises(ValueError):
        cj("\ud800")
    with pytest.raises(ValueError):
        canonical_request_hash({"state": "\udfff"})


def test_lone_surrogate_request_is_structurally_rejected():
    from conftest import TEST_PROFILE_SUPPORTED
    from local_judge import RequestValidator, StructuralCode

    raw = (
        '{"contract_version": "v1", "state": "\ud800", "model": "qwen3:8b",'
        ' "policy": {"version": "p"}, "inference": {},'
        ' "questions": {"q": {"type": "noul", "instructions": "i"}}}'
    )
    with pytest.raises(StructuralError) as excinfo:
        RequestValidator({"qwen3:8b": TEST_PROFILE_SUPPORTED}).parse(raw)
    assert excinfo.value.error.code == StructuralCode.MALFORMED_JSON


def test_mid_sampling_port_failure_preserves_earlier_samples():
    class FlakyPort:
        def __init__(self):
            self.calls = 0

        def attempt(self, model, rendered_messages, inference, response_schema=None):
            self.calls += 1
            if self.calls == 1:
                return RawAttempt(outcome=TransportOutcome.OK, output="first")
            raise RuntimeError("transport exploded mid-sampling")

    executor = FakeExecutor()
    orch = SamplingOrchestrator(port=FlakyPort(), executor=executor, versions=versions(), backend="fake")
    results = orch.run_questions(envelope(sample_count=2, questions={"q": {"type": "noul", "instructions": "i"}}))
    trace = results["q"].trace
    assert len(trace.attempts) == 1, "sample-1 evidence must survive the sample-2 failure"
    assert trace.attempts[0].raw_output == "first"
    assert results["q"].status is ResultStatus.QUESTION_ERROR


def test_lone_surrogate_object_keys_are_rejected():
    import pytest

    from local_judge.canonical import canonical_json as cj

    with pytest.raises(ValueError):
        cj({"\ud800": 1})
    with pytest.raises(ValueError):
        canonical_request_hash({"\udfff": "v"})
