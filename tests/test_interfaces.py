"""The model port and type executor interfaces exist and are satisfiable."""

from local_judge import (
    Inference,
    ResultEntry,
    ResultStatus,
    TraceRecord,
)
from local_judge.executor import TypeExecutor
from local_judge.ports import LocalModelPort, RawAttempt, TransportOutcome


def minimal_trace() -> TraceRecord:
    """A valid closed trace a Stage 3 executor could emit."""
    return TraceRecord.from_dict(
        {
            "trace_id": "00000000-0000-4000-8000-000000000010",
            "parent_trace_id": None,
            "trace_schema_version": "v1",
            "contract_version": "v1",
            "prompt_template_version": "p1",
            "output_schema_version": "o1",
            "aggregation_version": "a1",
            "policy_version": "p",
            "accepted_request": {
                "contract_version": "v1",
                "state": "s",
                "model": "qwen3:8b",
                "policy": {"version": "p"},
                "inference": {},
                "questions": {"q": {"type": "noul", "instructions": "i"}},
            },
            "canonical_request_hash": "a" * 64,
            "state": "s",
            "question": {"type": "noul", "instructions": "i"},
            "criteria": None,
            "policy_provenance": "caller-declared-unverified",
            "rendered_messages": [],
            "resolved_inference": {"sample_count": 1, "temperature": 0, "seed": None, "timeout_ms": 30000},
            "backend": "ollama",
            "model": "qwen3:8b",
            "model_digest": None,
            "local_runtime_version": "test",
            "attempts": [],
            "aggregate": None,
        }
    )


class FakePort:
    def attempt(self, model, rendered_messages, inference):
        return RawAttempt(outcome=TransportOutcome.OK, output='{"choice":"technical"}')


class FakeExecutor:
    def run(self, question_id, question, state, attempts):
        return ResultEntry(
            type_="choice",
            status=ResultStatus.ANSWERED,
            answer={"choice": "technical", "vote_share": {"technical": 1}},
            agreement=None,
            requested_samples=1,
            error=None,
            trace=minimal_trace(),
        )


def test_fake_port_satisfies_protocol():
    port = FakePort()
    assert isinstance(port, LocalModelPort)
    attempt = port.attempt("qwen3:8b", [], Inference())
    assert attempt.outcome is TransportOutcome.OK
    assert attempt.output == '{"choice":"technical"}'


def test_fake_executor_satisfies_protocol():
    executor = FakeExecutor()
    assert isinstance(executor, TypeExecutor)
    entry = executor.run("q", {}, "state", [])
    assert entry.status is ResultStatus.ANSWERED
    assert isinstance(entry.trace, TraceRecord)


def test_result_requires_a_real_trace():
    import pytest

    with pytest.raises(ValueError):
        ResultEntry(
            type_="choice",
            status=ResultStatus.ANSWERED,
            answer=None,
            agreement=None,
            requested_samples=1,
            error=None,
            trace=None,
        )
