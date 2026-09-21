"""Closed contract containers for the native v1 boundary.

The envelope models preserve caller-supplied state, policy, and questions as
opaque data. Result, trace, and attempt containers mirror the Stage 1 schema
exactly: from_dict rejects unknown members (closed), to_dict reproduces the
contract shape byte-for-byte in meaning.
"""

from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Any, Mapping

_UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"
)

from local_judge.errors import (
    ErrorObject,
    INABILITY_CODES,
    QUESTION_ERROR_CODES,
)

_SAMPLE_TERMINAL_CODES = QUESTION_ERROR_CODES | INABILITY_CODES


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_resolved_inference(d):
    sample_count = d.get("sample_count")
    if not _is_int(sample_count) or not 1 <= sample_count <= 5:
        raise ValueError("resolved_inference.sample_count must be an integer from 1 through 5")
    temperature = d.get("temperature")
    if not _is_number(temperature) or not 0 <= temperature <= 2:
        raise ValueError("resolved_inference.temperature must be a number from 0 through 2")
    seed = d.get("seed")
    if seed is not None and not _is_int(seed):
        raise ValueError("resolved_inference.seed must be an integer or null")
    timeout_ms = d.get("timeout_ms")
    if not _is_int(timeout_ms) or not 1 <= timeout_ms <= 120000:
        raise ValueError("resolved_inference.timeout_ms must be an integer from 1 through 120000")
    if set(d) != {"sample_count", "temperature", "seed", "timeout_ms"}:
        raise ValueError("resolved_inference must be the closed resolved settings object")


def _validate_choice_answer(answer):
    if not isinstance(answer, Mapping) or set(answer) != {"choice", "vote_share"}:
        raise ValueError("choice answer must be a closed object with choice and vote_share")
    if not isinstance(answer["choice"], str) or not answer["choice"]:
        raise ValueError("choice answer.choice must be a nonempty string")
    _validate_vote_share(answer["vote_share"])


def _validate_vote_share(vote_share):
    if not isinstance(vote_share, Mapping) or len(vote_share) < 1:
        raise ValueError("vote_share must be a nonempty object")
    for key, value in vote_share.items():
        if not _is_number(value) or not 0 <= value <= 1:
            raise ValueError("vote_share values must be numbers from 0 through 1")


def _validate_score_answer(answer):
    if not isinstance(answer, Mapping) or set(answer) != {"score", "legend", "vote_share"}:
        raise ValueError("score answer must be a closed object with score, legend, and vote_share")
    if not _is_number(answer["score"]) or not 0 <= answer["score"] <= 9:
        raise ValueError("score answer.score must be a number from 0 through 9")
    for keyed in ("legend", "vote_share"):
        m = answer[keyed]
        if not isinstance(m, Mapping):
            raise ValueError(f"score answer.{keyed} must be an object")
        for key, value in m.items():
            if not re.fullmatch(r"[0-9]", key):
                raise ValueError(f"score answer.{keyed} keys must be single decimal digits")
            if keyed == "legend":
                if value is not None and not isinstance(value, (str, list, dict)):
                    raise ValueError("score answer.legend values must be JSONContent")
            elif not _is_number(value) or not 0 <= value <= 1:
                raise ValueError("score answer.vote_share values must be numbers from 0 through 1")


def _validate_noul_answer(answer):
    if not isinstance(answer, Mapping) or set(answer) != {"noul"}:
        raise ValueError("noul answer must be a closed object with noul")
    if not _is_number(answer["noul"]) or not 0 <= answer["noul"] <= 1:
        raise ValueError("noul answer.noul must be a number from 0 through 1")


def _validate_typed_answer(type_, answer):
    if type_ == "choice":
        _validate_choice_answer(answer)
    elif type_ == "score":
        _validate_score_answer(answer)
    elif type_ == "noul":
        _validate_noul_answer(answer)
    else:
        raise ValueError(f"unknown question type: {type_!r}")


def _validate_answer_by_keys(answer):
    """Trace aggregates carry no type tag; the closed key set identifies the answer."""
    if not isinstance(answer, Mapping):
        raise ValueError("aggregate must be an answer object or null")
    keys = set(answer)
    if "choice" in keys or "noul" in keys or "score" in keys:
        if keys == {"noul"}:
            _validate_noul_answer(answer)
        elif keys == {"score", "legend", "vote_share"}:
            _validate_score_answer(answer)
        elif keys == {"choice", "vote_share"}:
            _validate_choice_answer(answer)
        else:
            raise ValueError(f"aggregate does not match any closed answer shape: {sorted(keys)}")
    else:
        raise ValueError(f"aggregate does not match any closed answer shape: {sorted(keys)}")


def _validate_rendered_messages(messages):
    for message in messages:
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError("rendered messages must be closed objects with role and content")
        if not isinstance(message["role"], str) or not message["role"]:
            raise ValueError("rendered message role must be a nonempty string")
        if message["content"] is not None and not isinstance(message["content"], (str, list, dict)):
            raise ValueError("rendered message content must be JSONContent")

_RESULT_KEYS = frozenset(
    {"type", "status", "answer", "agreement", "requested_samples", "error", "trace"}
)
_ATTEMPT_KEYS = frozenset(
    {"raw_output", "parsed_value", "validation_outcome", "timestamp", "terminal_error"}
)
_TRACE_KEYS = frozenset(
    {
        "trace_id",
        "parent_trace_id",
        "trace_schema_version",
        "contract_version",
        "prompt_template_version",
        "output_schema_version",
        "aggregation_version",
        "policy_version",
        "accepted_request",
        "canonical_request_hash",
        "state",
        "question",
        "criteria",
        "policy_provenance",
        "rendered_messages",
        "resolved_inference",
        "backend",
        "model",
        "model_digest",
        "local_runtime_version",
        "attempts",
        "aggregate",
    }
)


class ResultStatus(StrEnum):
    ANSWERED = "answered"
    INABILITY_TO_ANSWER = "inability_to_answer"
    QUESTION_ERROR = "question_error"


@dataclass(frozen=True)
class Policy:
    """Caller-declared, unauthenticated policy. version is never registry-checked."""

    version: str


@dataclass(frozen=True)
class Inference:
    """Closed inference settings with the contract defaults."""

    sample_count: int = 1
    temperature: float = 0
    seed: int | None = None
    timeout_ms: int = 30000


@dataclass(frozen=True)
class QuestionEntry:
    """One submitted question, opaque at this layer: id plus the raw object."""

    id: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class RequestEnvelope:
    """A structurally valid native v1 evaluation envelope."""

    state: Any
    model: str
    policy: Policy
    inference: Inference
    questions: Mapping[str, QuestionEntry]


@dataclass(frozen=True)
class AttemptRecord:
    """One sampling attempt, in order."""

    raw_output: str
    parsed_value: Any
    validation_outcome: str
    timestamp: str
    terminal_error: ErrorObject | None

    def __post_init__(self) -> None:
        if not isinstance(self.raw_output, str):
            raise ValueError("attempt raw_output must be a string")
        if not isinstance(self.validation_outcome, str) or not self.validation_outcome:
            raise ValueError("attempt validation_outcome must be a nonempty string")
        if self.terminal_error is not None and self.terminal_error.code not in _SAMPLE_TERMINAL_CODES:
            raise ValueError(
                f"an attempt can only end in a sample-level error or inability reason, got {self.terminal_error.code!r}"
            )
        if not _TIMESTAMP_PATTERN.match(self.timestamp):
            raise ValueError(f"attempt timestamp must be RFC 3339, got {self.timestamp!r}")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AttemptRecord":
        if not isinstance(d, Mapping):
            raise ValueError("an attempt record must be a JSON object")
        if set(d) != _ATTEMPT_KEYS:
            raise ValueError(f"attempt members must be exactly {_ATTEMPT_KEYS}, got {sorted(d)}")
        terminal = d["terminal_error"]
        return cls(
            raw_output=d["raw_output"],
            parsed_value=d["parsed_value"],
            validation_outcome=d["validation_outcome"],
            timestamp=d["timestamp"],
            terminal_error=ErrorObject(**terminal) if terminal is not None else None,
        )

    def to_dict(self) -> dict:
        return {
            "raw_output": self.raw_output,
            "parsed_value": self.parsed_value,
            "validation_outcome": self.validation_outcome,
            "timestamp": self.timestamp,
            "terminal_error": self.terminal_error.to_dict() if self.terminal_error is not None else None,
        }


@dataclass(frozen=True)
class TraceRecord:
    """Inline, self-contained replay record; policy provenance is always unverified."""

    trace_id: str
    parent_trace_id: str | None
    trace_schema_version: str
    contract_version: str
    prompt_template_version: str
    output_schema_version: str
    aggregation_version: str
    policy_version: str
    accepted_request: Mapping[str, Any]
    canonical_request_hash: str
    state: Any
    question: Mapping[str, Any]
    criteria: Any
    policy_provenance: str
    rendered_messages: Any
    resolved_inference: Mapping[str, Any]
    backend: str
    model: str
    model_digest: str | None
    local_runtime_version: str
    attempts: tuple[AttemptRecord, ...] = field(default=())
    aggregate: Any = None

    def __post_init__(self) -> None:
        for name in (
            "trace_schema_version",
            "prompt_template_version",
            "output_schema_version",
            "aggregation_version",
            "policy_version",
            "backend",
            "model",
            "local_runtime_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"trace member {name} must be a nonempty string")
        if not isinstance(self.state, (str, list, dict)):
            raise ValueError("trace state must be a JSON string, object, or array")
        if not isinstance(self.accepted_request, Mapping) or set(self.accepted_request) != {
            "contract_version",
            "state",
            "model",
            "policy",
            "inference",
            "questions",
        }:
            raise ValueError("trace accepted_request must be the closed accepted request object")
        if not isinstance(self.question, Mapping):
            raise ValueError("trace question must be the submitted question object")
        if self.criteria is not None and not isinstance(self.criteria, (str, list, dict)):
            raise ValueError("trace criteria must be JSONContent or null")
        if not isinstance(self.rendered_messages, list):
            raise ValueError("trace rendered_messages must be a list")
        _validate_rendered_messages(self.rendered_messages)
        if self.aggregate is not None:
            _validate_answer_by_keys(self.aggregate)
        if not isinstance(self.resolved_inference, Mapping):
            raise ValueError("trace resolved_inference must be the resolved settings object")
        _require_resolved_inference(self.resolved_inference)
        if not _UUID_PATTERN.match(self.trace_id):
            raise ValueError(f"trace_id must be a UUID, got {self.trace_id!r}")
        if self.parent_trace_id is not None and not _UUID_PATTERN.match(self.parent_trace_id):
            raise ValueError(f"parent_trace_id must be a UUID or null, got {self.parent_trace_id!r}")
        if not _HASH_PATTERN.match(self.canonical_request_hash):
            raise ValueError(f"canonical_request_hash must be lowercase hex SHA-256, got {self.canonical_request_hash!r}")
        if self.policy_provenance != "caller-declared-unverified":
            raise ValueError("policy_provenance is fixed to caller-declared-unverified")
        if self.contract_version != "v1":
            raise ValueError("trace contract_version must be exactly 'v1'")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TraceRecord":
        if not isinstance(d, Mapping):
            raise ValueError("a trace record must be a JSON object")
        if set(d) != _TRACE_KEYS:
            raise ValueError(f"trace members must be exactly {_TRACE_KEYS}, got {sorted(d)}")
        return cls(
            trace_id=d["trace_id"],
            parent_trace_id=d["parent_trace_id"],
            trace_schema_version=d["trace_schema_version"],
            contract_version=d["contract_version"],
            prompt_template_version=d["prompt_template_version"],
            output_schema_version=d["output_schema_version"],
            aggregation_version=d["aggregation_version"],
            policy_version=d["policy_version"],
            accepted_request=d["accepted_request"],
            canonical_request_hash=d["canonical_request_hash"],
            state=d["state"],
            question=d["question"],
            criteria=d["criteria"],
            policy_provenance=d["policy_provenance"],
            rendered_messages=d["rendered_messages"],
            resolved_inference=d["resolved_inference"],
            backend=d["backend"],
            model=d["model"],
            model_digest=d["model_digest"],
            local_runtime_version=d["local_runtime_version"],
            attempts=tuple(AttemptRecord.from_dict(a) for a in d["attempts"]),
            aggregate=d["aggregate"],
        )

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "parent_trace_id": self.parent_trace_id,
            "trace_schema_version": self.trace_schema_version,
            "contract_version": self.contract_version,
            "prompt_template_version": self.prompt_template_version,
            "output_schema_version": self.output_schema_version,
            "aggregation_version": self.aggregation_version,
            "policy_version": self.policy_version,
            "accepted_request": self.accepted_request,
            "canonical_request_hash": self.canonical_request_hash,
            "state": self.state,
            "question": self.question,
            "criteria": self.criteria,
            "policy_provenance": self.policy_provenance,
            "rendered_messages": self.rendered_messages,
            "resolved_inference": self.resolved_inference,
            "backend": self.backend,
            "model": self.model,
            "model_digest": self.model_digest,
            "local_runtime_version": self.local_runtime_version,
            "attempts": [a.to_dict() for a in self.attempts],
            "aggregate": self.aggregate,
        }


@dataclass(frozen=True)
class ResultEntry:
    """One entry of the keyed results map."""

    type_: str | None
    status: ResultStatus
    answer: Any
    agreement: float | None
    requested_samples: int
    error: ErrorObject | None
    trace: TraceRecord

    def __post_init__(self) -> None:
        if not isinstance(self.trace, TraceRecord):
            raise ValueError("every result carries an inline trace record")
        if not _is_int(self.requested_samples) or not 1 <= self.requested_samples <= 5:
            raise ValueError("requested_samples must be an integer from 1 through 5")
        if not isinstance(self.status, ResultStatus):
            raise ValueError("status must be a ResultStatus")
        if self.type_ is not None and self.type_ not in ("choice", "score", "noul"):
            raise ValueError(f"unknown question type: {self.type_!r}")
        answered = self.status is ResultStatus.ANSWERED
        if answered:
            if self.answer is None:
                raise ValueError("answered results carry a type-specific answer")
            if self.error is not None:
                raise ValueError("answered results carry no error")
            if self.type_ is None:
                raise ValueError("answered results know their type")
            _validate_typed_answer(self.type_, self.answer)
            if self.requested_samples == 1:
                if self.agreement is not None:
                    raise ValueError("agreement is null when one sample was requested")
            else:
                if not _is_number(self.agreement) or not 0 <= self.agreement <= 1:
                    raise ValueError("agreement must be a number from 0 through 1 for answered multi-sample results")
        else:
            if self.answer is not None:
                raise ValueError("non-answered results never carry an answer")
            if self.error is None:
                raise ValueError("non-answered results carry the terminal error")
            if self.agreement is not None:
                raise ValueError("agreement is null for non-answered results")
            expected = (
                QUESTION_ERROR_CODES if self.status is ResultStatus.QUESTION_ERROR else INABILITY_CODES
            )
            if self.error.code not in expected:
                raise ValueError(f"{self.status.value} cannot carry error code {self.error.code!r}")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ResultEntry":
        if not isinstance(d, Mapping):
            raise ValueError("a result entry must be a JSON object")
        if set(d) != _RESULT_KEYS:
            raise ValueError(f"result members must be exactly {_RESULT_KEYS}, got {sorted(d)}")
        return cls(
            type_=d["type"],
            status=ResultStatus(d["status"]),
            answer=d["answer"],
            agreement=d["agreement"],
            requested_samples=d["requested_samples"],
            error=ErrorObject(**d["error"]) if d["error"] is not None else None,
            trace=TraceRecord.from_dict(d["trace"]),
        )

    def to_dict(self) -> dict:
        return {
            "type": self.type_,
            "status": self.status.value,
            "answer": self.answer,
            "agreement": self.agreement,
            "requested_samples": self.requested_samples,
            "error": self.error.to_dict() if self.error is not None else None,
            "trace": self.trace.to_dict(),
        }


@dataclass(frozen=True)
class RejectionResponse:
    """Native rejected response: empty results and one error object."""

    contract_version: str | None
    model: str | None
    error: ErrorObject

    def to_dict(self) -> dict:
        return {
            "contract_version": self.contract_version,
            "model": self.model,
            "status": "rejected",
            "results": {},
            "error": self.error.to_dict(),
        }


@dataclass(frozen=True)
class CompletedResponse:
    """Native completed response for a valid envelope."""

    contract_version: str
    model: str
    results: Mapping[str, ResultEntry]

    def to_dict(self) -> dict:
        return {
            "contract_version": self.contract_version,
            "model": self.model,
            "status": "completed",
            "results": {qid: entry.to_dict() for qid, entry in self.results.items()},
            "error": None,
        }
