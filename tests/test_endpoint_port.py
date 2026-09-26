"""OpenAI-compatible inference endpoint adapter tests."""

import json

import pytest

from local_judge import Inference, OpenAICompatibleModelPort, OpenAICompatibleProfile
from local_judge.errors import StructuralCode, StructuralError
from local_judge.http_transport import TransportResponse, TransportTimeout, TransportUnavailable
from local_judge.ports import ModelPort, TransportOutcome


class FakeTransport:
    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = []

    def post(self, path, payload, timeout_ms, headers=None):
        self.calls.append({
            "path": path,
            "payload": payload,
            "timeout_ms": timeout_ms,
            "headers": dict(headers or {}),
        })
        if self.exc is not None:
            raise self.exc
        return self.responses.pop(0)


def response(status_code, body):
    return TransportResponse(status_code=status_code, body=body)


def success(content):
    return response(200, json.dumps({"choices": [{"message": {"content": content}}]}))


MESSAGES = [
    {"role": "system", "content": "Return one JSON value."},
    {"role": "user", "content": "classify this state"},
]
SCHEMA = {"oneOf": [{"type": "string"}, {"type": "number"}]}


def test_endpoint_model_port_sends_openwebui_chat_completions_request():
    fake = FakeTransport([success('{"choice":"billing"}')])
    profile = OpenAICompatibleProfile(
        name="qwen3:8b",
        base_url="http://host.docker.internal:3040/api",
        api_key="secret-token",
    )
    port = OpenAICompatibleModelPort({profile.name: profile}, transport=fake)

    attempt = port.attempt(
        profile.name,
        MESSAGES,
        Inference(temperature=0.2, timeout_ms=800),
        response_schema=SCHEMA,
    )

    call = fake.calls[0]
    assert isinstance(port, ModelPort)
    assert call["path"] == "http://host.docker.internal:3040/api/chat/completions"
    assert call["headers"] == {"Authorization": "Bearer secret-token"}
    assert call["payload"]["model"] == "qwen3:8b"
    assert call["payload"]["messages"] == MESSAGES
    assert call["payload"]["stream"] is False
    assert call["payload"]["temperature"] == 0.2
    assert call["payload"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "local_judge_sample",
            "strict": True,
            "schema": SCHEMA,
        },
    }
    assert call["timeout_ms"] == 800
    assert attempt.outcome is TransportOutcome.OK
    assert attempt.output == '{"choice":"billing"}'


def test_endpoint_json_object_mode_and_optional_authentication():
    fake = FakeTransport([success("{}")])
    profile = OpenAICompatibleProfile(
        name="local-model",
        base_url="http://127.0.0.1:3040/api/",
        response_format="json_object",
    )
    port = OpenAICompatibleModelPort({profile.name: profile}, transport=fake)

    port.attempt(profile.name, MESSAGES, Inference(), response_schema=SCHEMA)

    call = fake.calls[0]
    assert call["path"] == "http://127.0.0.1:3040/api/chat/completions"
    assert call["headers"] == {}
    assert call["payload"]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("base_url", "expected_path"),
    [
        ("http://host.docker.internal:3040/api", "http://host.docker.internal:3040/api/chat/completions"),
        ("https://models.example.test/v1", "https://models.example.test/v1/chat/completions"),
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080/chat/completions"),
        ("http://127.0.0.1:3040/api/", "http://127.0.0.1:3040/api/chat/completions"),
    ],
)
def test_base_url_is_joined_exactly_without_inferred_segments(base_url, expected_path):
    fake = FakeTransport([success("1")])
    profile = OpenAICompatibleProfile(name="model", base_url=base_url)
    OpenAICompatibleModelPort({profile.name: profile}, transport=fake).attempt(
        profile.name, MESSAGES, Inference()
    )
    assert fake.calls[0]["path"] == expected_path


def test_seed_is_sent_only_when_profile_declares_support():
    fake = FakeTransport([success("1")])
    profile = OpenAICompatibleProfile(
        name="seed-capable",
        base_url="https://models.example.test/v1",
        supported_inference_settings=frozenset({"sample_count", "temperature", "seed", "timeout_ms"}),
    )
    port = OpenAICompatibleModelPort({profile.name: profile}, transport=fake)

    port.attempt(profile.name, MESSAGES, Inference(seed=17))

    assert fake.calls[0]["payload"]["seed"] == 17


def test_unsupported_seed_is_rejected_before_endpoint_call():
    fake = FakeTransport()
    profile = OpenAICompatibleProfile(name="seedless", base_url="http://localhost:3040/api")
    port = OpenAICompatibleModelPort({profile.name: profile}, transport=fake)

    with pytest.raises(StructuralError) as excinfo:
        port.attempt(profile.name, MESSAGES, Inference(seed=17))

    assert excinfo.value.error.code == StructuralCode.UNSUPPORTED_INFERENCE_SETTING
    assert excinfo.value.error.path == "/inference/seed"
    assert fake.calls == []


def test_unknown_model_is_rejected_before_endpoint_call():
    fake = FakeTransport()
    port = OpenAICompatibleModelPort(
        {"known": OpenAICompatibleProfile(name="known", base_url="http://localhost:3040/api")},
        transport=fake,
    )

    with pytest.raises(StructuralError) as excinfo:
        port.attempt("unknown", MESSAGES, Inference())

    assert excinfo.value.error.code == StructuralCode.UNSUPPORTED_LOCAL_MODEL
    assert fake.calls == []


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://models.example.test/v1",
        "http://user:password@localhost:3040/api",
        "http://localhost:3040/api?token=secret",
        "http://localhost:3040/api#fragment",
        "http:///missing-host/api",
    ],
)
def test_invalid_endpoint_urls_are_rejected(base_url):
    with pytest.raises(ValueError):
        OpenAICompatibleProfile(name="model", base_url=base_url)


def test_api_key_is_not_in_profile_repr():
    profile = OpenAICompatibleProfile(
        name="model", base_url="http://localhost:3040/api", api_key="do-not-print-me"
    )
    assert "do-not-print-me" not in repr(profile)


@pytest.mark.parametrize(
    ("transport_exc", "outcome"),
    [
        (TransportTimeout("deadline"), TransportOutcome.TIMEOUT),
        (TransportUnavailable("connection refused"), TransportOutcome.UNAVAILABLE),
    ],
)
def test_transport_failures_become_explicit_outcomes(transport_exc, outcome):
    profile = OpenAICompatibleProfile(name="model", base_url="http://localhost:3040/api")
    attempt = OpenAICompatibleModelPort(
        {profile.name: profile}, transport=FakeTransport(exc=transport_exc)
    ).attempt(profile.name, MESSAGES, Inference())
    assert attempt.outcome is outcome
    assert attempt.output is None


@pytest.mark.parametrize(
    ("status", "body", "outcome"),
    [
        (504, "gateway timeout", TransportOutcome.TIMEOUT),
        (413, "request too large", TransportOutcome.CONTEXT_OVERFLOW),
        (400, "maximum context length exceeded", TransportOutcome.CONTEXT_OVERFLOW),
        (401, "unauthorized", TransportOutcome.UNAVAILABLE),
    ],
)
def test_endpoint_http_failures_map_to_transport_outcomes(status, body, outcome):
    profile = OpenAICompatibleProfile(name="model", base_url="http://localhost:3040/api")
    attempt = OpenAICompatibleModelPort(
        {profile.name: profile}, transport=FakeTransport([response(status, body)])
    ).attempt(profile.name, MESSAGES, Inference())
    assert attempt.outcome is outcome
    assert attempt.output is None


@pytest.mark.parametrize(
    "body",
    [
        "<html>not JSON</html>",
        json.dumps({"choices": []}),
        json.dumps({"choices": [{"message": {"content": None}}]}),
        json.dumps({"choices": [{"message": {"content": [{"text": "not a string"}]}}]}),
    ],
)
def test_malformed_chat_completion_responses_are_rejected(body):
    profile = OpenAICompatibleProfile(name="model", base_url="http://localhost:3040/api")
    attempt = OpenAICompatibleModelPort(
        {profile.name: profile}, transport=FakeTransport([response(200, body)])
    ).attempt(profile.name, MESSAGES, Inference())
    assert attempt.outcome is TransportOutcome.MALFORMED_RESPONSE
    assert attempt.output is None
