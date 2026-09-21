"""S3-2: Score judgments through the shared boundary (issue #15, practice:tdd).

Equations under test (docs/CONTRACT.md 'Results and Aggregation'):
vote_share[String(level)] = count(level)/n for every level;
score = sum(level * vote_share) in [0, criteria.length - 1];
agreement = max(level count)/n, null when n = 1; a tied mode stays answered.
"""

import pytest

from local_judge.executors.score import ScoreExecutor
from local_judge.ports import RawAttempt, TransportOutcome
from local_judge import ResultStatus

RUBRIC = ["Cosmetic", "Workaround", "Blocking"]


def executor():
    return ScoreExecutor(criteria=RUBRIC)


def ok(text):
    return RawAttempt(outcome=TransportOutcome.OK, output=text)


def test_valid_sample_is_the_level_index():
    record = executor().classify_sample(ok("2"))
    assert record.terminal_error is None
    assert record.parsed_value == 2


def test_invalid_samples():
    for bad in ("3", "-1", "0.5", '"1"', "x"):
        record = executor().classify_sample(ok(bad))
        assert record.terminal_error is not None, bad
        assert record.terminal_error.code == "INVALID_MODEL_OUTPUT", bad


def test_explicit_inability_is_terminal_inability():
    record = executor().classify_sample(ok('{"reason": "UNSUPPORTED_QUESTION"}'))
    assert record.terminal_error.code == "UNSUPPORTED_QUESTION"


def test_aggregate_produces_expected_position_and_full_vote_share():
    answer = executor().aggregate_from_parsed([0, 2, 2, 1])
    assert answer["score"] == pytest.approx((0 + 2 + 2 + 1) / 4)
    assert answer["vote_share"] == {"0": 0.25, "1": 0.25, "2": 0.5}
    assert answer["legend"] == {"0": "Cosmetic", "1": "Workaround", "2": "Blocking"}


def test_zero_vote_levels_are_present():
    answer = executor().aggregate_from_parsed([2])
    assert answer["vote_share"] == {"0": 0, "1": 0, "2": 1}


def test_fractional_score():
    answer = executor().aggregate_from_parsed([1, 2])
    assert answer["score"] == 1.5


def test_run_answered_two_samples():
    e = executor()
    attempts = [e.classify_sample(ok(lvl)) for lvl in ("1", "2")]
    result = e.run("severity", {"type": "score", "criteria": RUBRIC}, "s", attempts)
    assert result.status is ResultStatus.ANSWERED
    assert result.answer["score"] == 1.5
    assert result.agreement == 0.5
    assert result.requested_samples == 2


def test_run_single_sample_has_null_agreement():
    e = executor()
    attempts = [e.classify_sample(ok("1"))]
    result = e.run("severity", {"type": "score", "criteria": RUBRIC}, "s", attempts)
    assert result.status is ResultStatus.ANSWERED
    assert result.agreement is None
    assert result.answer["vote_share"] == {"0": 0, "1": 1, "2": 0}


def test_invalid_sample_gives_question_error_and_no_partial_aggregate():
    e = executor()
    attempts = [e.classify_sample(ok("9"))]
    result = e.run("severity", {"type": "score", "criteria": RUBRIC}, "s", attempts)
    assert result.status is ResultStatus.QUESTION_ERROR
    assert result.error is not None
    assert result.error.code == "INVALID_MODEL_OUTPUT"
    assert result.answer is None


def test_explicit_inability_gives_inability_to_answer():
    e = executor()
    attempts = [e.classify_sample(ok('{"reason": "INSUFFICIENT_EVIDENCE"}'))]
    result = e.run("severity", {"type": "score", "criteria": RUBRIC}, "s", attempts)
    assert result.status is ResultStatus.INABILITY_TO_ANSWER
    assert result.error.code == "INSUFFICIENT_EVIDENCE"


def test_empty_attempts_give_question_error_without_crashing():
    e = executor()
    result = e.run("severity", {"type": "score", "criteria": RUBRIC}, "s", [])
    assert result.status is ResultStatus.QUESTION_ERROR
    assert result.answer is None
    assert result.requested_samples == 0 or result.error is not None


def test_rubric_bounds_validated_at_construction():
    with pytest.raises(ValueError):
        ScoreExecutor(criteria=["only-one"])
    with pytest.raises(ValueError):
        ScoreExecutor(criteria=[f"lvl{i}" for i in range(11)])


def test_non_jsoncontent_rubric_values_rejected_at_construction():
    with pytest.raises(ValueError):
        ScoreExecutor(criteria=["ok", None])
