"""S3-1: Choice judgments through the shared boundary (issue #12, practice:tdd).

Equations under test (docs/CONTRACT.md 'Results and Aggregation'):
vote_share[option] = count(option)/n for every supplied option (zero included);
the winner is the unique largest count — a tie is inability_to_answer (AGGREGATION_TIE);
agreement = max(option count)/n, null when n = 1.
"""

import pytest

from local_judge.executors.choice import ChoiceExecutor
from local_judge.ports import RawAttempt, TransportOutcome
from local_judge import ResultStatus

MENU = {"billing": "Payments", "technical": "Bugs"}


def executor():
    return ChoiceExecutor(criteria=MENU)


def ok(text):
    return RawAttempt(outcome=TransportOutcome.OK, output=text)


def test_valid_sample_is_a_menu_id():
    record = executor().classify_sample(ok('"technical"'))
    assert record.terminal_error is None
    assert record.parsed_value == "technical"


def test_non_menu_ids_are_invalid_model_output():
    for bad in ('"plumbing"', '"Billing"', '""'):
        record = executor().classify_sample(ok(bad))
        assert record.terminal_error is not None, bad
        assert record.terminal_error.code == "INVALID_MODEL_OUTPUT", bad


def test_aggregate_counts_every_menu_option_including_zero_votes():
    answer = executor().aggregate_from_parsed(["technical", "billing", "technical"])
    assert answer["choice"] == "technical"
    assert answer["vote_share"] == {"billing": 1 / 3, "technical": 2 / 3}


def test_unique_largest_count_wins():
    answer = executor().aggregate_from_parsed(["billing", "technical", "technical"])
    assert answer["choice"] == "technical"


def test_tie_is_inability():
    answer = executor().aggregate_from_parsed(["billing", "technical"])
    assert answer == ("inability", "AGGREGATION_TIE")


def test_agreement_is_largest_count_over_n():
    assert executor().agreement_for(["billing", "technical", "technical"]) == pytest.approx(2 / 3)


def test_agreement_null_for_single_sample():
    assert executor().agreement_for(["technical"]) is None


def test_run_answered_multi_sample():
    e = executor()
    attempts = [e.classify_sample(ok(t)) for t in ('"billing"', '"technical"', '"technical"')]
    result = e.run("department", {"type": "choice", "criteria": MENU}, "s", attempts)
    assert result.status is ResultStatus.ANSWERED
    assert result.answer["choice"] == "technical"
    assert result.answer["vote_share"] == {"billing": 1 / 3, "technical": 2 / 3}
    assert result.agreement == pytest.approx(2 / 3)
    assert result.requested_samples == 3


def test_run_single_sample_has_null_agreement():
    e = executor()
    attempts = [e.classify_sample(ok('"technical"'))]
    result = e.run("department", {"type": "choice", "criteria": MENU}, "s", attempts)
    assert result.status is ResultStatus.ANSWERED
    assert result.agreement is None
    assert result.answer["vote_share"] == {"billing": 0, "technical": 1}


def test_tie_run_is_inability_to_answer():
    e = executor()
    attempts = [e.classify_sample(ok(t)) for t in ('"billing"', '"technical"')]
    result = e.run("department", {"type": "choice", "criteria": MENU}, "s", attempts)
    assert result.status is ResultStatus.INABILITY_TO_ANSWER
    assert result.error.code == "AGGREGATION_TIE"
    assert result.answer is None and result.agreement is None


def test_invalid_sample_gives_question_error():
    e = executor()
    attempts = [e.classify_sample(ok('"plumbing"'))]
    result = e.run("department", {"type": "choice", "criteria": MENU}, "s", attempts)
    assert result.status is ResultStatus.QUESTION_ERROR
    assert result.error.code == "INVALID_MODEL_OUTPUT"


def test_explicit_inability_gives_inability_to_answer():
    e = executor()
    attempts = [e.classify_sample(ok('{"reason": "INSUFFICIENT_EVIDENCE"}'))]
    result = e.run("department", {"type": "choice", "criteria": MENU}, "s", attempts)
    assert result.status is ResultStatus.INABILITY_TO_ANSWER
    assert result.error.code == "INSUFFICIENT_EVIDENCE"
