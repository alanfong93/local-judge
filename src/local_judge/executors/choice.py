"""Choice judgments: menu validation, plurality winner, vote shares (issue #12).

A Choice answer proves only that the selected id belonged to the supplied
menu (docs/CONTRACT.md 'Choice'). The winner is the unique largest count; a
tie for the largest count is inability_to_answer with AGGREGATION_TIE.
"""

from local_judge.executors.base import NativeTypeExecutor


class ChoiceExecutor(NativeTypeExecutor):
    def __init__(self, criteria) -> None:
        if not isinstance(criteria, Mapping):
            raise ValueError("choice criteria must be a map of 2 through 255 option ids")
        if not 2 <= len(criteria) <= 255:
            raise ValueError("choice criteria must contain 2 through 255 options")
        for option_id, description in criteria.items():
            if not isinstance(option_id, str) or not option_id:
                raise ValueError("choice option ids must be nonempty strings")
            if description is not None and not isinstance(description, (str, list, dict)):
                raise ValueError("choice option descriptions must be JSONContent or null")
        super().__init__(
            question_type="choice",
            criteria=criteria,
            aggregate=self.aggregate_from_parsed,
        )

    def aggregate_from_parsed(self, parsed_samples: list) -> dict | tuple:
        menu = list(self.criteria)
        n = len(parsed_samples)
        counts = {option: 0 for option in menu}
        for value in parsed_samples:
            counts[value] += 1
        largest = max(counts.values())
        winners = [option for option, count in counts.items() if count == largest]
        if len(winners) != 1:
            return ("inability", "AGGREGATION_TIE")
        vote_share = {option: counts[option] / n for option in menu}
        return {"choice": winners[0], "vote_share": vote_share}

    def agreement_for(self, parsed_samples: list) -> float | None:
        n = len(parsed_samples)
        if n <= 1:
            return None
        counts: dict = {}
        for value in parsed_samples:
            counts[value] = counts.get(value, 0) + 1
        return max(counts.values()) / n
