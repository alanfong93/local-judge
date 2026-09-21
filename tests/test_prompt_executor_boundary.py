"""S3-0: versioned prompt compiler and the typed-executor boundary (issue #11).

Boundary pins: exactly three prompt regions; state is evidence only and never
interpolated into policy or engine configuration; question IDs are correlation
keys and never rendered; the sample union is locally revalidated; typed
aggregation/agreement stay outside this boundary.
"""

import json

import pytest

from local_judge import (
    AttemptRecord,
    ErrorObject,
    Inference,
    Policy,
    QuestionEntry,
    RequestEnvelope,
    ResultStatus,
)
from local_judge.executors.base import NativeTypeExecutor
from local_judge.ports import RawAttempt, TransportOutcome
from local_judge.prompt import VersionedPromptCompiler


def question(qtype="noul", criteria=None, instructions="Is this a refund?"):
    q = {"type": qtype, "instructions": instructions}
    if criteria is not None:
        q["criteria"] = criteria
    return q


def noul_question():
    return question("noul", {"true": "asks for money", "false": "does not"})


def choice_question():
    return question("choice", {"billing": "Payments", "technical": "Bugs"})


def score_question():
    return question("score", ["Cosmetic", "Workaround", "Blocking"])


def raw_ok(text):
    return RawAttempt(outcome=TransportOutcome.OK, output=text)


def make_executor(qtype="noul", criteria=None, aggregate=None):
    if qtype == "choice":
        q = choice_question()
    elif qtype == "score":
        q = score_question()
    else:
        q = noul_question()
    return NativeTypeExecutor(
        question_type=qtype,
        criteria=criteria if criteria is not None else q.get("criteria"),
        aggregate=aggregate or (lambda samples: None),
        versions={"prompt_template_version": "prompt-1", "output_schema_version": "schema-1"},
    )


def envelope(sample_count=1):
    return RequestEnvelope(
        state={"ticket": "evidence text"},
        model="qwen3:8b",
        policy=Policy(version="pol-1"),
        inference=Inference(sample_count=sample_count, temperature=0, seed=None, timeout_ms=1000),
        questions={},
    )


# --- prompt compiler: three regions, state separation, id non-rendering ---

def test_compiler_renders_three_distinct_regions():
    compiler = VersionedPromptCompiler(template_version="prompt-1")
    q = choice_question()
    state = {"ticket": "The printer is on fire"}
    rendered, schema = compiler.render("department", q, state)
    assert [m["role"] for m in rendered] == ["system", "user", "user"]
    system, policy, evidence = rendered
    assert "output" in system["content"].lower()
    assert q["instructions"] in policy["content"]
    assert state["ticket"] in evidence["content"]
    # state text never appears in the policy region
    assert state["ticket"] not in policy["content"]
    # policy text never appears in the engine region
    assert q["instructions"] not in system["content"]


def test_compiler_never_renders_the_question_id():
    compiler = VersionedPromptCompiler(template_version="prompt-1")
    rendered, _ = compiler.render("department", choice_question(), {"t": "x"})
    for message in rendered:
        assert "department" not in message["content"]


def test_compiler_output_schema_is_type_specific():
    compiler = VersionedPromptCompiler(template_version="prompt-1")
    _, noul_schema = compiler.render("q", noul_question(), "s")
    _, choice_schema = compiler.render("q", choice_question(), "s")
    assert noul_schema["oneOf"][0] == {"type": "number", "minimum": 0, "maximum": 1}
    assert any("enum" in json.dumps(branch) for branch in choice_schema["oneOf"])


def test_compiler_records_its_versions():
    compiler = VersionedPromptCompiler(template_version="prompt-1", output_schema_version="schema-1")
    assert compiler.template_version == "prompt-1"
    assert compiler.output_schema_version == "schema-1"


# --- sample union local revalidation ---

def test_choice_sample_union_accepts_only_menu_ids():
    executor = make_executor("choice")
    record = executor.classify_sample(RawAttempt(outcome=TransportOutcome.OK, output='"technical"'), choice_question())
    assert record.terminal_error is None
    bad = executor.classify_sample(RawAttempt(outcome=TransportOutcome.OK, output='"plumbing"'), choice_question())
    assert bad.terminal_error.code == "INVALID_MODEL_OUTPUT"


def test_score_sample_union_accepts_only_integers_in_range():
    executor = make_executor("score", aggregate=None)
    record = executor.classify_sample(RawAttempt(outcome=TransportOutcome.OK, output="2"), score_question())
    assert record.terminal_error is None
    bad = executor.classify_sample(RawAttempt(outcome=TransportOutcome.OK, output="0.5"), score_question())
    assert bad.terminal_error.code == "INVALID_MODEL_OUTPUT"
    bad2 = executor.classify_sample(RawAttempt(outcome=TransportOutcome.OK, output="3"), score_question())
    assert bad2.terminal_error.code == "INVALID_MODEL_OUTPUT"


def test_noul_sample_union_accepts_only_finite_0_to_1():
    executor = make_executor("noul")
    record = executor.classify_sample(RawAttempt(outcome=TransportOutcome.OK, output="0.9"), noul_question())
    assert record.terminal_error is None
    bad = executor.classify_sample(RawAttempt(outcome=TransportOutcome.OK, output="1.01"), noul_question())
    assert bad.terminal_error.code == "INVALID_MODEL_OUTPUT"


def test_explicit_inability_object_is_accepted_and_terminal():
    executor = make_executor("choice")
    record = executor.classify_sample(
        RawAttempt(outcome=TransportOutcome.OK, output='{"reason": "INSUFFICIENT_EVIDENCE"}'), choice_question()
    )
    assert record.terminal_error is not None
    assert record.terminal_error.code == "INSUFFICIENT_EVIDENCE"


# --- run(): aggregation hook, question_error on invalid samples ---

def test_run_passes_valid_parsed_samples_to_the_aggregate_hook():
    seen = {}

    def aggregate(parsed_samples):
        seen["samples"] = parsed_samples
        return {"noul": 0.9}

    executor = make_executor("noul", aggregate=aggregate)
    attempts = [executor.classify_sample(raw_ok("0.9"))]
    result = executor.run("q", noul_question(), "s", attempts)
    assert seen["samples"] == [0.9]
    assert result.status is ResultStatus.ANSWERED
    assert result.answer == {"noul": 0.9}


def test_invalid_sample_yields_question_error_without_calling_the_hook():
    calls = []
    executor = make_executor("noul", aggregate=lambda s: calls.append(s) or {"noul": 1})
    attempts = [executor.classify_sample(raw_ok("1.5"))]
    result = executor.run("q", noul_question(), "s", attempts)
    assert result.status is ResultStatus.QUESTION_ERROR
    assert result.error.code == "INVALID_MODEL_OUTPUT"
    assert calls == []


def test_inability_sample_yields_inability_to_answer():
    def aggregate(samples):
        raise AssertionError("aggregate must not run for inability")

    executor = make_executor("choice", aggregate=aggregate)
    attempts = [
        executor.classify_sample(RawAttempt(outcome=TransportOutcome.OK, output='{"reason": "AMBIGUOUS_EVIDENCE"}'))
    ]
    result = executor.run("q", choice_question(), "s", attempts)
    assert result.status is ResultStatus.INABILITY_TO_ANSWER
    assert result.error.code == "AMBIGUOUS_EVIDENCE"


def test_typed_aggregation_stays_outside_this_boundary():
    """S3-1/2/3 own aggregation: the base executor never computes one itself."""
    import inspect

    from local_judge.executors import base

    source = inspect.getsource(base)
    for banned in ("vote_share", "sum(", "max("):
        assert banned not in source, banned
