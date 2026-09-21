"""Structural error vocabulary: codes, the closed error object, and the exception.

Path rule (docs/CONTRACT.md 'Validation and Execution'): the empty string for
exactly the codes enumerated in EMPTY_PATH_CODES, an RFC 6901 JSON Pointer to
the offending field for every other code.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import re

RFC3339_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")


def is_rfc3339(value: str) -> bool:
    """RFC 3339 shape AND a calendar-valid instant (2026-99-99 is not a date)."""
    if not isinstance(value, str) or not RFC3339_PATTERN.match(value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False

_POINTER_PATTERN = re.compile(r"^(?:/(?:[^/~]|~[01])*)+$")

EMPTY_PATH_CODES = frozenset(
    {
        "MALFORMED_JSON",
        "REQUEST_TOO_LARGE",
        "DUPLICATE_QUESTION_ID",
        "INVALID_QUESTION",
        "INVALID_MODEL_OUTPUT",
        "MODEL_TIMEOUT",
        "MODEL_UNAVAILABLE",
        "CONTEXT_LIMIT_EXCEEDED",
        "INSUFFICIENT_EVIDENCE",
        "AMBIGUOUS_EVIDENCE",
        "UNSUPPORTED_QUESTION",
        "AGGREGATION_TIE",
        "REPLAY_CONFIGURATION_UNAVAILABLE",
        "JEV_ADAPTER_UNMAPPABLE_RESULT",
    }
)

FIELD_PATH_CODES = frozenset(
    {
        "UNSUPPORTED_CONTRACT_VERSION",
        "UNKNOWN_FIELD",
        "MISSING_FIELD",
        "INVALID_FIELD",
        "INVALID_QUESTIONS_MAP",
        "UNSUPPORTED_INFERENCE_SETTING",
        "UNSUPPORTED_LOCAL_MODEL",
    }
)


QUESTION_ERROR_CODES = frozenset(
    {"INVALID_QUESTION", "INVALID_MODEL_OUTPUT", "MODEL_TIMEOUT", "MODEL_UNAVAILABLE", "CONTEXT_LIMIT_EXCEEDED"}
)

INABILITY_CODES = frozenset(
    {"INSUFFICIENT_EVIDENCE", "AMBIGUOUS_EVIDENCE", "UNSUPPORTED_QUESTION", "AGGREGATION_TIE"}
)

class StructuralCode(StrEnum):
    """Whole-request structural codes; rejection happens before any model call."""

    MALFORMED_JSON = "MALFORMED_JSON"
    UNSUPPORTED_CONTRACT_VERSION = "UNSUPPORTED_CONTRACT_VERSION"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_FIELD = "INVALID_FIELD"
    INVALID_QUESTIONS_MAP = "INVALID_QUESTIONS_MAP"
    DUPLICATE_QUESTION_ID = "DUPLICATE_QUESTION_ID"
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"
    UNSUPPORTED_INFERENCE_SETTING = "UNSUPPORTED_INFERENCE_SETTING"
    UNSUPPORTED_LOCAL_MODEL = "UNSUPPORTED_LOCAL_MODEL"


@dataclass(frozen=True)
class ErrorObject:
    """Closed error shape: code, path, message - and nothing else."""

    code: str
    path: str
    message: str

    def __post_init__(self) -> None:
        if self.code not in EMPTY_PATH_CODES and self.code not in FIELD_PATH_CODES:
            raise ValueError(f"unknown error code: {self.code!r}")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be a nonempty string")
        if self.code in EMPTY_PATH_CODES:
            if self.path != "":
                raise ValueError(f"{self.code} carries an empty path, got {self.path!r}")
        else:
            if not _POINTER_PATTERN.match(self.path):
                raise ValueError(f"{self.code} requires an RFC 6901 JSON Pointer, got {self.path!r}")

    def to_dict(self) -> dict:
        return {"code": self.code, "path": self.path, "message": self.message}


class StructuralError(Exception):
    """A whole-request structural rejection raised before any model call."""

    def __init__(self, code: str, message: str, path: str = "") -> None:
        self.error = ErrorObject(code=code, path=path, message=message)
        super().__init__(f"{code} at {path!r}: {message}")
