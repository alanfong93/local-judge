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
        if self.terminal_error is not None and self.terminal_error.code not in _SAMPLE_TERMINAL_CODES:
            raise ValueError(
                f"an attempt can only end in a sample-level error or inability reason, got {self.terminal_error.code!r}"
            )
        if not _TIMESTAMP_PATTERN.match(self.timestamp):
            raise ValueError(f"attempt timestamp must be RFC 3339, got {self.timestamp!r}")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AttemptRecord":
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

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ResultEntry":
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
