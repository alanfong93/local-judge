"""Whole-request structural validation of the native v1 envelope.

Rejects before any model call with the Stage 1 structural codes, in a fixed
order (docs/CONTRACT.md 'Structural Rejection'). state, policy content, and
question content are preserved as opaque data; this layer never interprets
their content as engine configuration or authority.

Each check order is pinned by tests/test_structural_rejection.py:
malformed JSON, oversize payload, duplicate top-level fields, contract
version, unknown/missing/invalid envelope fields (state, model, policy,
inference), unknown model profile, unsupported inference settings, questions
map shape, duplicate raw question IDs, question count.
"""

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Union

from local_judge.errors import StructuralCode, StructuralError
from local_judge.models import Inference, Policy, QuestionEntry, RequestEnvelope

MAX_ENCODED_BYTES = 256 * 1024
MAX_QUESTIONS = 64

_ENVELOPE_KEYS = frozenset(
    {"contract_version", "state", "model", "policy", "inference", "questions"}
)
_INFERENCE_KEYS = frozenset({"sample_count", "temperature", "seed", "timeout_ms"})


class _DupDict(dict):
    """dict that remembers which keys appeared more than once in the raw JSON."""

    dups: frozenset


def _reject_constant(name: str) -> None:
    raise ValueError(f"{name} is not JSON")


def _strict_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is not a finite JSON number")
    return value


def _escape(token: str) -> str:
    """RFC 6901 escape for one reference token."""
    return token.replace("~", "~0").replace("/", "~1")


def _pairs_hook(pairs: list[tuple[str, Any]]) -> _DupDict:
    keys = [k for k, _ in pairs]
    d = _DupDict(pairs)
    d.dups = frozenset(k for k in keys if keys.count(k) > 1)
    return d


@dataclass(frozen=True)
class ModelProfile:
    """A configured local model profile and the inference settings it can honor."""

    name: str
    supported_inference_settings: frozenset[str]


_ProfileSetting = Union[frozenset[str], ModelProfile]


class RequestValidator:
    """Parses a raw request into a RequestEnvelope or raises StructuralError."""

    def __init__(self, model_profiles: Mapping[str, _ProfileSetting] | None = None) -> None:
        self._profiles: dict[str, frozenset[str]] = {}
        for name, spec in (model_profiles or {}).items():
            if isinstance(spec, ModelProfile):
                self._profiles[name] = spec.supported_inference_settings
            else:
                self._profiles[name] = frozenset(spec)

    def parse(self, raw: Union[bytes, bytearray, str, Mapping[str, Any]]) -> RequestEnvelope:
        if isinstance(raw, Mapping):
            try:
                raw = json.dumps(raw, ensure_ascii=False, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError, OverflowError):
                raise StructuralError(
                    StructuralCode.MALFORMED_JSON, "the request mapping is not JSON-representable"
                ) from None
        elif isinstance(raw, str):
            raw = raw.encode("utf-8")
        try:
            doc = json.loads(
                raw,
                object_pairs_hook=_pairs_hook,
                parse_constant=_reject_constant,
                parse_float=_strict_float,
            )
        except (UnicodeDecodeError, ValueError):
            raise StructuralError(
                StructuralCode.MALFORMED_JSON, "the raw request cannot be decoded as one JSON value"
            ) from None
        if len(raw) > MAX_ENCODED_BYTES:
            raise StructuralError(
                StructuralCode.REQUEST_TOO_LARGE,
                "the encoded request exceeds 256 KiB",
            )
        if not isinstance(doc, _DupDict):
            raise StructuralError(
                StructuralCode.MALFORMED_JSON, "the request body must be one JSON object"
            )
        if doc.dups:
            first_dup = sorted(doc.dups)[0]
            raise StructuralError(
                StructuralCode.INVALID_FIELD,
                f"duplicate top-level field: {first_dup!r}",
                f"/{_escape(first_dup)}",
            )

        if "contract_version" not in doc or doc["contract_version"] != "v1":
            raise StructuralError(
                StructuralCode.UNSUPPORTED_CONTRACT_VERSION,
                "contract_version must be present and exactly 'v1'",
                "/contract_version",
            )

        for key in doc:
            if key not in _ENVELOPE_KEYS:
                raise StructuralError(
                    StructuralCode.UNKNOWN_FIELD, f"unknown top-level field: {key!r}", f"/{_escape(key)}"
                )

        self._require(doc, "state")
        state = doc["state"]
        if not isinstance(state, (str, list, dict)):
            raise StructuralError(
                StructuralCode.INVALID_FIELD, "state must be a JSON string, object, or array", "/state"
            )

        self._require(doc, "model")
        model = doc["model"]
        if not isinstance(model, str) or not model:
            raise StructuralError(StructuralCode.INVALID_FIELD, "model must be a nonempty string", "/model")

        self._require(doc, "policy")
        policy_doc = doc["policy"]
        if not isinstance(policy_doc, _DupDict):
            raise StructuralError(StructuralCode.INVALID_FIELD, "policy must be an object", "/policy")
        if policy_doc.dups:
            first_dup = sorted(policy_doc.dups)[0]
            raise StructuralError(
                StructuralCode.INVALID_FIELD,
                f"duplicate policy field: {first_dup!r}",
                f"/policy/{_escape(first_dup)}",
            )
        self._require(policy_doc, "version", "/policy")
        version = policy_doc["version"]
        if not isinstance(version, str) or not version:
            raise StructuralError(
                StructuralCode.INVALID_FIELD, "policy.version must be a nonempty string", "/policy/version"
            )
        for key in policy_doc:
            if key != "version":
                raise StructuralError(
                    StructuralCode.UNKNOWN_FIELD, f"unknown policy field: {key!r}", f"/policy/{_escape(key)}"
                )
        policy = Policy(version=version)

        self._require(doc, "inference")
        inference_doc = doc["inference"]
        if not isinstance(inference_doc, _DupDict):
            raise StructuralError(StructuralCode.INVALID_FIELD, "inference must be an object", "/inference")
        if inference_doc.dups:
            first_dup = sorted(inference_doc.dups)[0]
            raise StructuralError(
                StructuralCode.INVALID_FIELD,
                f"duplicate inference field: {first_dup!r}",
                f"/inference/{_escape(first_dup)}",
            )
        for key in inference_doc:
            if key not in _INFERENCE_KEYS:
                raise StructuralError(
                    StructuralCode.UNKNOWN_FIELD, f"unknown inference field: {key!r}", f"/inference/{_escape(key)}"
                )
        inference = self._parse_inference(inference_doc)

        if model not in self._profiles:
            raise StructuralError(
                StructuralCode.UNSUPPORTED_LOCAL_MODEL,
                f"requested model is not a configured local profile: {model!r}",
                "/model",
            )
        if doc["inference"].get("seed") is not None and "seed" not in self._profiles[model]:
            raise StructuralError(
                StructuralCode.UNSUPPORTED_INFERENCE_SETTING,
                "the selected profile cannot honor the requested inference setting: seed",
                "/inference/seed",
            )

        self._require(doc, "questions")
        questions_doc = doc["questions"]
        if not isinstance(questions_doc, _DupDict):
            raise StructuralError(
                StructuralCode.INVALID_QUESTIONS_MAP, "questions must be a nonempty JSON object", "/questions"
            )
        if len(questions_doc) == 0:
            raise StructuralError(
                StructuralCode.INVALID_QUESTIONS_MAP, "questions must not be empty", "/questions"
            )
        if questions_doc.dups:
            raise StructuralError(
                StructuralCode.DUPLICATE_QUESTION_ID,
                "duplicate question ID in raw JSON",
            )
        if len(questions_doc) > MAX_QUESTIONS:
            raise StructuralError(
                StructuralCode.REQUEST_TOO_LARGE,
                f"a request may contain at most {MAX_QUESTIONS} questions",
            )
        for qid, qdoc in questions_doc.items():
            if not isinstance(qid, str) or not qid:
                raise StructuralError(
                    StructuralCode.INVALID_FIELD, "question IDs must be nonempty strings", "/questions"
                )
            if not isinstance(qdoc, _DupDict):
                raise StructuralError(
                    StructuralCode.INVALID_FIELD, f"question {qid!r} must be an object", f"/questions/{_escape(qid)}"
                )
        questions = {qid: QuestionEntry(id=qid, raw=qdoc) for qid, qdoc in questions_doc.items()}

        return RequestEnvelope(
            state=state, model=model, policy=policy, inference=inference, questions=questions
        )

    def _parse_inference(self, doc: Mapping[str, Any]) -> Inference:
        sample_count = doc.get("sample_count", 1)
        if not _is_int(sample_count) or not 1 <= sample_count <= 5:
            raise StructuralError(
                StructuralCode.INVALID_FIELD, "sample_count must be an integer from 1 through 5",
                "/inference/sample_count",
            )
        temperature = doc.get("temperature", 0)
        if not _is_number(temperature) or not 0 <= temperature <= 2:
            raise StructuralError(
                StructuralCode.INVALID_FIELD, "temperature must be a number from 0 through 2",
                "/inference/temperature",
            )
        seed = doc.get("seed")
        if seed is not None and not _is_int(seed):
            raise StructuralError(
                StructuralCode.INVALID_FIELD, "seed must be an integer", "/inference/seed"
            )
        timeout_ms = doc.get("timeout_ms", 30000)
        if not _is_int(timeout_ms) or not 1 <= timeout_ms <= 120000:
            raise StructuralError(
                StructuralCode.INVALID_FIELD, "timeout_ms must be an integer from 1 through 120000",
                "/inference/timeout_ms",
            )
        return Inference(
            sample_count=sample_count,
            temperature=temperature,
            seed=seed,
            timeout_ms=timeout_ms,
        )

    @staticmethod
    def _require(obj: Mapping[str, Any], key: str, prefix: str = "") -> None:
        if key not in obj:
            path = f"{prefix}/{key}" if prefix else f"/{key}"
            raise StructuralError(StructuralCode.MISSING_FIELD, f"required field is absent: {key!r}", path)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
