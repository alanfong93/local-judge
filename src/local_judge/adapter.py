"""The documented Jev compatibility adapter (docs/CONTRACT.md 'Jev Adapter').

Accepts the documented Jev-shaped input map (closed object: state, model,
questions), validates it with the native structural rules, converts it to a
native v1 evaluation with fixed adapter defaults, and maps only fully
answered native results to the documented Jev answer shapes. Every failure
refuses rather than invents; agreement is disclosed as agreement, never as
Jev or calibrated confidence.
"""

from typing import Any, Mapping

from local_judge.errors import ErrorObject, StructuralCode, StructuralError
from local_judge.models import RequestEnvelope, ResultEntry, ResultStatus
from local_judge.validation import RequestValidator


class JevAdapter:
    POLICY_VERSION = "jev-adapter-v1"
    SAMPLE_COUNT = 3
    TEMPERATURE = 0
    TIMEOUT_MS = 30000
    DISCLOSURE = "confidence is local repeated-sample agreement, not Jev or calibrated confidence"

    _INPUT_KEYS = frozenset({"state", "model", "questions"})

    def __init__(self, validator: RequestValidator, model_profiles: Mapping[str, Any]) -> None:
        self._validator = validator
        self._model_profiles = model_profiles

    def convert_input(self, jev_input: Mapping[str, Any]) -> RequestEnvelope:
        """Validate the Jev-shaped input with native structural rules, then convert.

        Unknown, missing, or invalid input members surface the native structural
        codes before any conversion or model call; JEV_ADAPTER_UNMAPPABLE_RESULT
        is never used for input validation.
        """
        for key in jev_input:
            if key not in self._INPUT_KEYS:
                label = key if isinstance(key, str) else str(key)
                escape = label.replace("~", "~0").replace("/", "~1")
                raise StructuralError(
                    StructuralCode.UNKNOWN_FIELD, f"unknown Jev input field: {label!r}", f"/{escape}"
                )
        # absent members stay absent so the validator reports MISSING_FIELD
        # (not INVALID_FIELD for an injected None); present members are type-checked
        native = {
            "contract_version": "v1",
            "policy": {"version": self.POLICY_VERSION},
            "inference": {
                "sample_count": self.SAMPLE_COUNT,
                "temperature": self.TEMPERATURE,
                "timeout_ms": self.TIMEOUT_MS,
            },
        }
        for key in ("state", "model", "questions"):
            if key in jev_input:
                native[key] = jev_input[key]
        return self._validator.parse(native)

    def evaluate(self, jev_input: Mapping[str, Any], runner) -> dict:
        """One adapter evaluation: structural validation, native run, mapping."""
        try:
            envelope = self.convert_input(jev_input)
        except StructuralError as exc:
            return self._structural_failure(exc)
        results = runner(envelope)
        return self.map_results(jev_input, results)

    def evaluate_with_results(self, jev_input: Mapping[str, Any], results: Mapping[str, ResultEntry]) -> dict:
        """Map caller-supplied native results (fake-runner test seam)."""
        try:
            self.convert_input(jev_input)
        except StructuralError as exc:
            return self._structural_failure(exc)
        return self.map_results(jev_input, results)

    def map_results(self, jev_input: Mapping[str, Any], results: Mapping[str, ResultEntry]) -> dict:
        requested = set(jev_input.get("questions", {}))
        traces = {}
        answers = {}
        mappable = True
        refused = False
        for question_id in requested:
            entry = results.get(question_id)
            if entry is None:
                refused = True  # a requested question was not answered
                continue
            traces[question_id] = entry.trace.to_dict()
            requested_type = jev_input.get("questions", {}).get(question_id, {}).get("type")
            if entry.type_ != requested_type:
                refused = True  # a result of the wrong type is never mapped
                continue
            if entry.status is not ResultStatus.ANSWERED or entry.answer is None:
                refused = True
                continue
            if entry.agreement is None:
                # missing agreement (e.g. a single-sample result) cannot be disclosed
                refused = True
                continue
            answers[question_id] = self._convert_answer(entry)
        if refused or set(results) != requested:
            refused = True  # unrequested entries or dropped questions refuse the whole result
        if refused:
            return {
                "answers": None,
                "local_judge": self._local_judge(traces),
                "error": ErrorObject(
                    code="JEV_ADAPTER_UNMAPPABLE_RESULT",
                    path="",
                    message="a requested question did not answer; partial results are never mapped",
                ).to_dict(),
            }
        return {
            "answers": answers,
            "local_judge": self._local_judge(traces),
            "error": None,
        }

    def _convert_answer(self, entry: ResultEntry) -> dict:
        answer = entry.answer
        if entry.type_ == "choice":
            return {
                "type": "choice",
                "choice": answer["choice"],
                "probabilities": answer["vote_share"],
                "confidence": entry.agreement,
            }
        if entry.type_ == "score":
            return {
                "type": "score",
                "score": answer["score"],
                "probabilities": answer["vote_share"],
                "legend": answer["legend"],
                "confidence": entry.agreement,
            }
        return {"type": "noul", "noul": answer["noul"]}

    def _local_judge(self, traces: Mapping[str, dict]) -> dict:
        return {
            "contract_version": "v1",
            "traces": dict(traces),
            "confidence_disclosure": self.DISCLOSURE,
        }

    @staticmethod
    def _structural_failure(exc: StructuralError) -> dict:
        return {"answers": None, "local_judge": None, "error": exc.error.to_dict()}
