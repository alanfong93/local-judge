"""S3-3: Noul judgments through the shared boundary (issue #14, practice:tdd).

Equations under test (docs/CONTRACT.md 'Noul' and 'Results and Aggregation'):
aggregate = arithmetic mean of valid samples;
agreement (n > 1) = largest decile-bucket count / n with
bucket(p) = min(floor(10 * p), 9); null when n = 1.
Agreement stays agreement: no confidence field, no calibration claim.
"""

import pytest

from local_judge.executors.noul import NoulExecutor
from local_judge.ports import RawAttempt, TransportOutcome
from local_judge import ResultStatus


def executor():
    return NoulExecutor(criteria={"true": "asks for money", "false": "does not"})


def ok(text):
    return RawAttempt(outcome=TransportOutcome.OK, output=text)


def test_valid_sample_is_a_finite_number_in_range():
    record = executor().classify_sample(ok("0.9"))
    assert record.terminal_error is None
    assert record.parsed_value == 0.9


def test_invalid_samples():
    for bad in ("1.01", "-0.1", '"0.5"', "true", '{"reason": "WHIM"}'):
        record = executor().classify_sample(ok(bad))
        assert record.terminal_error is not None, bad
        assert record.terminal_error.code == "INVALID_MODEL_OUTPUT", bad


def test_mean_aggregate():
    answer = executor().aggregate_from_parsed([0.82, 0.95, 0.81])
    assert answer == {"noul": 0.86}


def test_decile_bucket_agreement():
    # 0.82 -> bucket 8, 0.95 -> bucket 9, 0.81 -> bucket 8: largest count 2 of 3
    samples = [0.82, 0.95, 0.81]
    assert executor().agreement_for(samples) == pytest.approx(2 / 3)


def test_agreement_null_for_single_sample():
    assert executor().agreement_for([0.7]) is None


def test_run_answered_multi_sample():
    e = executor()
    attempts = [e.classify_sample(ok(v)) for v in ("0.82", "0.95", "0.81")]
    result = e.run("is_refund", {"type": "noul", "instructions": "Is this a refund?"}, "s", attempts)
    assert result.status is ResultStatus.ANSWERED
    assert result.answer == {"noul": 0.86}
    assert result.agreement == pytest.approx(2 / 3)
    assert result.requested_samples == 3


def test_run_single_sample_has_null_agreement():
    e = executor()
    attempts = [e.classify_sample(ok("0.9"))]
    result = e.run("is_refund", {"type": "noul", "instructions": "i"}, "s", attempts)
    assert result.status is ResultStatus.ANSWERED
    assert result.agreement is None


def test_explicit_inability_gives_inability_to_answer():
    e = executor()
    attempts = [e.classify_sample(ok('{"reason": "AMBIGUOUS_EVIDENCE"}'))]
    result = e.run("is_refund", {"type": "noul", "instructions": "i"}, "s", attempts)
    assert result.status is ResultStatus.INABILITY_TO_ANSWER
    assert result.error.code == "AMBIGUOUS_EVIDENCE"


def test_invalid_sample_gives_question_error():
    e = executor()
    attempts = [e.classify_sample(ok(" nonsense "))]
    result = e.run("is_refund", {"type": "noul", "instructions": "i"}, "s", attempts)
    assert result.status is ResultStatus.QUESTION_ERROR
    assert result.error.code == "INVALID_MODEL_OUTPUT"
    assert result.answer is None


def test_no_confidence_field_exists_in_the_answer():
    e = executor()
    attempts = [e.classify_sample(ok("0.5"))]
    result = e.run("is_refund", {"type": "noul", "instructions": "i"}, "s", attempts)
    assert "confidence" not in result.answer
    assert "confidence" not in result.answer and set(result.answer) == {"noul"}
