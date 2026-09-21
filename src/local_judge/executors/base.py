"""NativeTypeExecutor: the S3-0 boundary shared by Choice, Score, and Noul.

Owns the versioned prompt compile, local revalidation of every raw model
result against the type-specific sample union, and ordered attempt/terminal
recording. Typed aggregation and agreement stay with the per-type subclasses
supplied via the `aggregate` hook (issues S3-1/2/3).
"""

import json
import time
import uuid
from typing import Any, Callable, Mapping

from local_judge.errors import ErrorObject
from local_judge.models import AttemptRecord, ResultEntry, ResultStatus, TraceRecord
from local_judge.ports import RawAttempt, TransportOutcome
from local_judge.prompt import VersionedPromptCompiler

INABILITY_CODES = ("INSUFFICIENT_EVIDENCE", "AMBIGUOUS_EVIDENCE", "UNSUPPORTED_QUESTION")

_TRANSPORT_TERMINAL = {
    TransportOutcome.TIMEOUT: "MODEL_TIMEOUT",
    TransportOutcome.UNAVAILABLE: "MODEL_UNAVAILABLE",
    TransportOutcome.CONTEXT_OVERFLOW: "CONTEXT_LIMIT_EXCEEDED",
    TransportOutcome.MALFORMED_RESPONSE: "INVALID_MODEL_OUTPUT",
}

# A valid closed envelope used as the placeholder accepted request inside a
# bare executor-level trace; orchestration replaces the whole trace with the
# real accepted request.
_PLACEHOLDER_ACCEPTED_REQUEST = {
    "contract_version": "v1",
    "state": "-",
    "model": "-",
    "policy": {"version": "-"},
    "inference": {},
    "questions": {"-": {"type": "noul", "instructions": "-"}},
}


class InabilitySignal(Exception):
    """Raised by an aggregate to end a question as inability_to_answer."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _now() -> str:
    seconds, fraction = divmod(time.time(), 1)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds)) + f".{int(fraction * 1000):03d}Z"


class NativeTypeExecutor:
    """Base executor: compile, sample-union revalidation, terminal recording."""

    def __init__(
        self,
        question_type: str,
        criteria: Any,
        aggregate: Callable[[list], Any],
        backend: str = "unspecified",
        model: str = "unspecified",
        versions: Mapping[str, str] | None = None,
    ) -> None:
        self.question_type = question_type
        self.criteria = criteria
        self.aggregate = aggregate
        self.backend = backend
        self.model = model
        supplied = dict(versions or {})
        self.template_version = supplied.get("prompt_template_version", "prompt-1")
        self.output_schema_version = supplied.get("output_schema_version", "schema-1")

    def versions(self) -> dict:
        return {
            "prompt_template_version": self.template_version,
            "output_schema_version": self.output_schema_version,
        }

    def render_messages(self, question_id: str, question: Mapping[str, Any], state: Any) -> list:
        compiler = VersionedPromptCompiler(
            template_version=self.template_version,
            output_schema_version=self.output_schema_version,
        )
        return compiler.render(question_id, question, state)[0]

    def output_schema(self, question: Mapping[str, Any]) -> dict:
        compiler = VersionedPromptCompiler(
            template_version=self.template_version,
            output_schema_version=self.output_schema_version,
        )
        return compiler.render("validation", question, None)[1]

    def classify_sample(self, raw: RawAttempt, question: Mapping[str, Any] | None = None) -> AttemptRecord:
        if raw.outcome is not TransportOutcome.OK:
            code = _TRANSPORT_TERMINAL.get(raw.outcome.value)
            terminal = ErrorObject(code=code, path="", message=f"transport outcome: {raw.outcome.value}")
            return AttemptRecord(
                raw_output=raw.output or "",
                parsed_value=None,
                validation_outcome=raw.outcome.value,
                timestamp=_now(),
                terminal_error=terminal,
            )
        parsed, failure = self._revalidate(raw.output)
        if failure is not None:
            code, message = failure
            terminal = ErrorObject(code=code, path="", message=message)
            return AttemptRecord(
                raw_output=raw.output,
                parsed_value=None,
                validation_outcome="invalid",
                timestamp=_now(),
                terminal_error=terminal,
            )
        return AttemptRecord(
            raw_output=raw.output,
            parsed_value=parsed,
            validation_outcome="valid",
            timestamp=_now(),
            terminal_error=None,
        )

    def _revalidate(self, output: str):
        """Local revalidation against the type-specific sample union."""
        try:
            value = json.loads(output)
        except ValueError:
            return None, ("INVALID_MODEL_OUTPUT", "the model output is not one JSON value")
        if isinstance(value, Mapping) and set(value) == {"reason"} and value["reason"] in INABILITY_CODES:
            return None, (value["reason"], "the sample declared an explicit inability")
        qtype = self.question_type
        if qtype == "choice":
            menu = self.criteria or {}
            if isinstance(value, str) and value in menu:
                return value, None
        if qtype == "score":
            rubric = self.criteria or []
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(rubric):
                return value, None
        if qtype == "noul":
            if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1:
                return float(value), None
        return None, ("INVALID_MODEL_OUTPUT", "the sample does not match the sample union for this question type")

    def run(self, question_id: str, question: Mapping[str, Any], state: Any, attempts) -> ResultEntry:
        records = tuple(attempts)
        aggregate_samples = []
        for record in records:
            if record.terminal_error is None:
                aggregate_samples.append(record.parsed_value)
            elif record.terminal_error.code in INABILITY_CODES:
                return self._result(
                    question, records, ResultStatus.INABILITY_TO_ANSWER,
                    error=record.terminal_error, answer=None, agreement=None,
                )
            else:
                return self._result(
                    question, records, ResultStatus.QUESTION_ERROR,
                    error=record.terminal_error, answer=None, agreement=None,
                )
        agreement = self.agreement_for(aggregate_samples) if aggregate_samples else None
        try:
            outcome = self.aggregate(aggregate_samples)
        except InabilitySignal as signal:
            inability = ErrorObject(code=signal.code, path="", message="the aggregate declared an inability")
            return self._result(
                question, records, ResultStatus.INABILITY_TO_ANSWER, error=inability, answer=None,
                agreement=None,
            )
        if isinstance(outcome, tuple) and len(outcome) == 2 and outcome[0] == "inability":
            if outcome[1] not in INABILITY_CODES:
                raise ValueError(f"aggregate declared unknown inability code: {outcome[1]!r}")
            inability = ErrorObject(code=outcome[1], path="", message="the aggregate declared an inability")
            return self._result(
                question, records, ResultStatus.INABILITY_TO_ANSWER, error=inability, answer=None,
                agreement=None,
            )
        if outcome is None:
            return self._result(
                question, records, ResultStatus.QUESTION_ERROR,
                error=ErrorObject(code="INVALID_QUESTION", path="", message="no aggregate was produced"),
                answer=None, agreement=None,
            )
        return self._result(
            question, records, ResultStatus.ANSWERED, answer=outcome, error=None,
            agreement=agreement if agreement is not None else None,
        )

    def agreement_for(self, parsed_samples: list) -> float | None:
        """Per-type agreement; the base never computes one."""
        return None

    def _result(self, question, records, status, answer=None, error=None, agreement=None):
        trace = TraceRecord(
            trace_id=str(uuid.uuid4()),
            parent_trace_id=None,
            trace_schema_version="v1",
            contract_version="v1",
            prompt_template_version=self.template_version,
            output_schema_version=self.output_schema_version,
            aggregation_version="a1",
            policy_version="unspecified",
            accepted_request=dict(_PLACEHOLDER_ACCEPTED_REQUEST),
            canonical_request_hash="0" * 64,
            state="-",
            question=dict(question),
            criteria=None,
            policy_provenance="caller-declared-unverified",
            rendered_messages=[],
            resolved_inference={"sample_count": 1, "temperature": 0, "seed": None, "timeout_ms": 30000},
            backend=self.backend,
            model=self.model,
            model_digest=None,
            local_runtime_version="unspecified",
            attempts=list(records),
            aggregate=answer if status is ResultStatus.ANSWERED else None,
        )
        return ResultEntry(
            type_=self.question_type,
            status=status,
            answer=answer,
            agreement=agreement,
            requested_samples=len(records),
            error=error,
            trace=trace,
        )
