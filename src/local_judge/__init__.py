"""local-judge native v1 core boundary.

Importing this package performs no network, model, filesystem, or storage
access. Public surface only; internal modules may change.
"""

from local_judge.errors import ErrorObject, StructuralCode, StructuralError
from local_judge.endpoint import OpenAICompatibleModelPort, OpenAICompatibleProfile
from local_judge.models import (
    AttemptRecord,
    CompletedResponse,
    Inference,
    Policy,
    QuestionEntry,
    RequestEnvelope,
    RejectionResponse,
    ResultEntry,
    ResultStatus,
    TraceRecord,
)
from local_judge.ports import ModelPort
from local_judge.validation import ModelProfile, RequestValidator

__all__ = [
    "AttemptRecord",
    "CompletedResponse",
    "ErrorObject",
    "Inference",
    "ModelProfile",
    "ModelPort",
    "OpenAICompatibleModelPort",
    "OpenAICompatibleProfile",
    "Policy",
    "QuestionEntry",
    "RequestEnvelope",
    "RequestValidator",
    "RejectionResponse",
    "ResultEntry",
    "ResultStatus",
    "StructuralCode",
    "StructuralError",
    "TraceRecord",
]
