"""Raw single-attempt local-model port and transport outcomes.

The port returns raw backend output and a transport outcome only. It never
parses model output into typed answers, never classifies inability, and never
aggregates: that is the type executor's job (docs/CONTRACT.md 'Sample Union').
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from local_judge.models import Inference


class TransportOutcome(StrEnum):
    """Transport-level outcome of one backend attempt."""

    OK = "ok"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    CONTEXT_OVERFLOW = "context_overflow"


@dataclass(frozen=True)
class RawAttempt:
    """One raw backend attempt: output text and transport outcome, nothing else."""

    outcome: TransportOutcome
    output: str | None


@runtime_checkable
class LocalModelPort(Protocol):
    """Raw single-attempt port over a configured local model profile."""

    def attempt(
        self,
        model: str,
        rendered_messages: Sequence[Mapping[str, Any]],
        inference: Inference,
    ) -> RawAttempt: ...
