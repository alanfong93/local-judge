"""Pluggable type-executor interface.

The executor owns everything typed about one question: parsing raw attempts,
invalid-model-output classification, inability, aggregate, and agreement
(docs/CONTRACT.md 'Sample Union' and 'Results and Aggregation'). The core
never interprets question content itself.
"""

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from local_judge.models import ResultEntry
from local_judge.ports import RawAttempt


@runtime_checkable
class TypeExecutor(Protocol):
    """Turns raw attempts for one submitted question into its result entry."""

    def run(
        self,
        question_id: str,
        question: Mapping[str, Any],
        state: Any,
        attempts: Sequence[RawAttempt],
    ) -> ResultEntry: ...
