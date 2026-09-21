"""Score judgments: ordered-rubric validation, aggregation, agreement (issue #15).

The native Score is an empirical expected zero-based position on the caller's
ordered rubric (docs/CONTRACT.md 'Score' and 'Results and Aggregation'). A
tied mode remains answered because the expected position is defined.
"""

from typing import Mapping

from local_judge.executors.base import NativeTypeExecutor


class ScoreExecutor(NativeTypeExecutor):
    def __init__(self, criteria) -> None:
        if isinstance(criteria, (str, Mapping)) or criteria is None:
            raise ValueError("score criteria must be an ordered array of 2 through 10 values")
        rubric = list(criteria)
        if not 2 <= len(rubric) <= 10:
            raise ValueError("score criteria must be an ordered array of 2 through 10 values")
        if any(item is None or not isinstance(item, (str, list, dict)) for item in rubric):
            raise ValueError("score criteria values must be JSONContent")
        super().__init__(
            question_type="score",
            criteria=rubric,
            aggregate=self.aggregate_from_parsed,
        )

    def aggregate_from_parsed(self, parsed_samples: list) -> dict:
        rubric = list(self.criteria)
        n = len(parsed_samples)
        counts = {level: 0 for level in range(len(rubric))}
        for value in parsed_samples:
            counts[value] += 1
        vote_share = {str(level): counts[level] / n for level in range(len(rubric))}
        score = sum(level * vote_share[str(level)] for level in range(len(rubric)))
        legend = {str(level): criterion for level, criterion in enumerate(rubric)}
        return {"score": score, "legend": legend, "vote_share": vote_share}

    def agreement_for(self, parsed_samples: list) -> float | None:
        n = len(parsed_samples)
        if n <= 1:
            return None
        counts: dict = {}
        for value in parsed_samples:
            counts[value] = counts.get(value, 0) + 1
        return max(counts.values()) / n
