"""Noul judgments: mean aggregation and decile-bucket agreement (issue #14).

The native Noul estimates whether a proposition is true from 0 through 1
(docs/CONTRACT.md 'Noul'). Its mean stays an estimate; its quantized
consistency stays agreement — never confidence, correctness probability, or
degree scoring.
"""

from math import floor

from local_judge.executors.base import NativeTypeExecutor


class NoulExecutor(NativeTypeExecutor):
    def __init__(self, criteria=None) -> None:
        super().__init__(
            question_type="noul",
            criteria=criteria,
            aggregate=self.aggregate_from_parsed,
        )

    def aggregate_from_parsed(self, parsed_samples: list) -> dict:
        mean = sum(parsed_samples) / len(parsed_samples)
        return {"noul": mean}

    def agreement_for(self, parsed_samples: list) -> float | None:
        n = len(parsed_samples)
        if n <= 1:
            return None
        buckets = {}
        for value in parsed_samples:
            bucket = min(floor(10 * value), 9)
            buckets[bucket] = buckets.get(bucket, 0) + 1
        return max(buckets.values()) / n
