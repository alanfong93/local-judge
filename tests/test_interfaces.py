"""The model port and type executor interfaces exist and are satisfiable."""

from local_judge import Inference, ResultEntry, ResultStatus
from local_judge.executor import TypeExecutor
from local_judge.ports import LocalModelPort, RawAttempt, TransportOutcome


class FakePort:
    def attempt(self, model, rendered_messages, inference):
        return RawAttempt(outcome=TransportOutcome.OK, output='{"choice":"technical"}')


class FakeExecutor:
    def run(self, question_id, question, state, attempts):
        return ResultEntry(
            type_="choice",
            status=ResultStatus.ANSWERED,
            answer={"choice": "technical", "vote_share": {"technical": 1}},
            agreement=None,
            requested_samples=1,
            error=None,
            trace=None,
        )


def test_fake_port_satisfies_protocol():
    port = FakePort()
    assert isinstance(port, LocalModelPort)
    attempt = port.attempt("qwen3:8b", [], Inference())
    assert attempt.outcome is TransportOutcome.OK
    assert attempt.output == '{"choice":"technical"}'


def test_fake_executor_satisfies_protocol():
    executor = FakeExecutor()
    assert isinstance(executor, TypeExecutor)
    entry = executor.run("q", {}, "state", [])
    assert entry.status is ResultStatus.ANSWERED
