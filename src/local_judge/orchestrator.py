"""Type-agnostic sampling orchestration, inline trace assembly, and replay
resolution (issue #9, docs/CONTRACT.md 'Question Isolation' and 'Trace and Replay').

The orchestrator runs exactly the validated sample count per accepted question,
keeps questions isolated, and assembles the inline trace. Typed parsing,
invalid-model-output classification, inability, aggregation, and agreement
belong to the type executor; this module computes none of them. Traces stay
in-memory; nothing is persisted.
"""

import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from local_judge.canonical import canonical_request_hash
from local_judge.errors import ErrorObject
from local_judge.models import (
    AttemptRecord,
    Inference,
    RejectionResponse,
    RequestEnvelope,
    ResultEntry,
    ResultStatus,
    TraceRecord,
)
from local_judge.ports import RawAttempt

def _rfc3339_now(index: int) -> str:
    seconds, fraction = divmod(time.time(), 1)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds)) + f".{int(fraction * 1000):03d}Z"


@dataclass(frozen=True)
class ResolvedReplay:
    """A replay whose recorded versions all resolved; no stochastic promise is made."""

    prompt_template: Any
    output_schema: Any
    aggregation: Any
    model_artifact: Any
    accepted_request: Mapping[str, Any]
    resolved_inference: Mapping[str, Any]
    parent_trace_id: str


class SamplingTypeExecutor:
    """What a Stage 3 executor provides on top of typed result production."""

    def render_messages(self, question_id: str, question: Mapping[str, Any], state: Any) -> list:
        raise NotImplementedError

    def classify(self, raw_attempt: RawAttempt) -> AttemptRecord:
        raise NotImplementedError

    def run(self, question_id: str, question: Mapping[str, Any], state: Any, attempts) -> ResultEntry:
        raise NotImplementedError


class SamplingOrchestrator:
    """Runs each accepted question in isolation against one port and one executor."""

    def __init__(
        self,
        port,
        executor,
        versions: Mapping[str, str],
        backend: str,
        local_runtime_version: str = "local-judge",
        model_digest: str | None = None,
    ) -> None:
        self._port = port
        self._executor = executor
        self._versions = dict(versions)
        self._backend = backend
        self._local_runtime_version = local_runtime_version
        self._model_digest = model_digest

    def run_questions(self, envelope: RequestEnvelope) -> dict[str, ResultEntry]:
        results = {}
        for question_id, entry in envelope.questions.items():
            results[question_id] = self.run_question(envelope, question_id, entry.raw)
        return results

    def run_question(self, envelope: RequestEnvelope, question_id: str, question: Mapping[str, Any]) -> ResultEntry:
        accepted_request = self._accepted_request(envelope)
        rendered_messages: list = []
        attempt_records: tuple = ()
        try:
            messages = self._executor.render_messages(question_id, question, envelope.state)
            rendered_messages = list(messages)
            raw_attempts: list[RawAttempt] = []
            for _ in range(envelope.inference.sample_count):
                raw_attempts.append(
                    self._port.attempt(
                        envelope.model,
                        messages,
                        envelope.inference,
                    )
                )
            attempt_records = tuple(self._executor.classify(raw) for raw in raw_attempts)
            result = self._executor.run(question_id, question, envelope.state, attempt_records)
        except Exception as exc:  # isolation: one question never aborts its siblings
            return self._question_error_fallback(
                envelope, question_id, question, accepted_request, exc,
                attempt_records=attempt_records, rendered_messages=rendered_messages,
            )
        trace = self._build_trace(
            envelope,
            question,
            accepted_request,
            attempts=attempt_records,
            rendered_messages=rendered_messages,
            aggregate=result.answer,
            trace_id=result.trace.trace_id,
        )
        return replace(result, trace=trace)

    def _build_trace(self, envelope, question, accepted_request, attempts, rendered_messages, aggregate, trace_id):
        return TraceRecord(
            trace_id=trace_id,
            parent_trace_id=None,
            trace_schema_version="v1",
            contract_version="v1",
            prompt_template_version=self._versions["prompt_template_version"],
            output_schema_version=self._versions["output_schema_version"],
            aggregation_version=self._versions["aggregation_version"],
            policy_version=envelope.policy.version,
            accepted_request=accepted_request,
            canonical_request_hash=canonical_request_hash(accepted_request),
            state=envelope.state,
            question=question,
            criteria=question.get("criteria") if isinstance(question, Mapping) else None,
            policy_provenance="caller-declared-unverified",
            rendered_messages=list(rendered_messages),
            resolved_inference={
                "sample_count": envelope.inference.sample_count,
                "temperature": envelope.inference.temperature,
                "seed": envelope.inference.seed,
                "timeout_ms": envelope.inference.timeout_ms,
            },
            backend=self._backend,
            model=envelope.model,
            model_digest=self._model_digest,
            local_runtime_version=self._local_runtime_version,
            attempts=tuple(attempts),
            aggregate=aggregate,
        )

    def _question_error_fallback(
        self,
        envelope,
        question_id,
        question,
        accepted_request,
        exc: Exception,
        attempt_records: tuple = (),
        rendered_messages: list | None = None,
    ) -> ResultEntry:
        """Isolation boundary: an executor crash becomes that question's error only."""
        trace = TraceRecord(
            trace_id=str(uuid.uuid4()),
            parent_trace_id=None,
            trace_schema_version="v1",
            contract_version="v1",
            prompt_template_version=self._versions["prompt_template_version"],
            output_schema_version=self._versions["output_schema_version"],
            aggregation_version=self._versions["aggregation_version"],
            policy_version=envelope.policy.version,
            accepted_request=accepted_request,
            canonical_request_hash=canonical_request_hash(accepted_request),
            state=envelope.state,
            question=question,
            criteria=question.get("criteria") if isinstance(question, Mapping) else None,
            policy_provenance="caller-declared-unverified",
            rendered_messages=list(rendered_messages or []),
            resolved_inference={
                "sample_count": envelope.inference.sample_count,
                "temperature": envelope.inference.temperature,
                "seed": envelope.inference.seed,
                "timeout_ms": envelope.inference.timeout_ms,
            },
            backend=self._backend,
            model=envelope.model,
            model_digest=self._model_digest,
            local_runtime_version=self._local_runtime_version,
            attempts=tuple(attempt_records),
            aggregate=None,
        )
        return ResultEntry(
            type_=None,
            status=ResultStatus.QUESTION_ERROR,
            answer=None,
            agreement=None,
            requested_samples=envelope.inference.sample_count,
            error=ErrorObject(code="INVALID_QUESTION", path="", message=str(exc)),
            trace=trace,
        )

    def _accepted_request(self, envelope: RequestEnvelope) -> dict:
        return {
            "contract_version": "v1",
            "state": envelope.state,
            "model": envelope.model,
            "policy": {"version": envelope.policy.version},
            "inference": self._inference_dict(envelope.inference),
            "questions": {qid: entry.raw for qid, entry in envelope.questions.items()},
        }

    @staticmethod
    def _inference_dict(inference: Inference) -> dict:
        d = {}
        d["sample_count"] = inference.sample_count
        d["temperature"] = inference.temperature
        if inference.seed is not None:
            d["seed"] = inference.seed
        d["timeout_ms"] = inference.timeout_ms
        return d


def resolve_replay(trace: TraceRecord, artifact_registry: Mapping[str, Mapping[str, Any]]):
    """Reconstruct replay inputs from recorded versions only.

    Resolves the recorded prompt-template, output-schema, and aggregation
    artifacts plus the model profile. Any missing versioned artifact yields the
    Stage 1 REPLAY_CONFIGURATION_UNAVAILABLE rejection instead of substituting
    current defaults. Replay is a new evaluation: equal stochastic output is
    never promised.
    """
    missing = (
        trace.prompt_template_version not in artifact_registry.get("prompt_templates", {})
        or trace.output_schema_version not in artifact_registry.get("output_schemas", {})
        or trace.aggregation_version not in artifact_registry.get("aggregations", {})
        or trace.model not in artifact_registry.get("models", {})
    )
    model_artifact = artifact_registry.get("models", {}).get(trace.model)
    if not missing and trace.model_digest is not None:
        recorded = getattr(model_artifact, "digest", None)
        if recorded is None and isinstance(model_artifact, Mapping):
            recorded = model_artifact.get("digest")
        if recorded != trace.model_digest:
            missing = True  # the resolved artifact is not the recorded model
    if missing:
        return RejectionResponse(
            contract_version="v1",
            model=trace.model,
            error=ErrorObject(
                code="REPLAY_CONFIGURATION_UNAVAILABLE",
                path="",
                message="the recorded versioned configuration is not available for replay",
            ),
        )
    return ResolvedReplay(
        prompt_template=artifact_registry["prompt_templates"][trace.prompt_template_version],
        output_schema=artifact_registry["output_schemas"][trace.output_schema_version],
        aggregation=artifact_registry["aggregations"][trace.aggregation_version],
        model_artifact=artifact_registry["models"][trace.model],
        accepted_request=trace.accepted_request,
        resolved_inference=trace.resolved_inference,
        parent_trace_id=trace.trace_id,
    )
