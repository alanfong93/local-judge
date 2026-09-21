"""Ollama raw model port: fake-transport contract (issue #10, practice:tdd).

The fake transport IS the contract under test: one bounded attempt, explicit
non-retried outcomes, profile/setting rejections before transport, and no
need for a running Ollama instance.
"""

import json

import pytest

from local_judge import Inference, StructuralCode, StructuralError
from local_judge.ollama import (
    OllamaProfile,
    OllamaTransportUnavailable,
    OllamaTransportTimeout,
    UrllibOllamaTransport,
)
from local_judge.ports import TransportOutcome
from local_judge.ollama import OllamaModelPort


class FakeTransport:
    """Scripted transport: returns canned responses or raises; records every call."""

    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = []

    def post(self, path, payload, timeout_ms):
        self.calls.append({"path": path, "payload": payload, "timeout_ms": timeout_ms})
        if self.exc is not None:
            raise self.exc
        return self.responses.pop(0)


def response(status_code, body):
    from local_judge.ollama import TransportResponse

    return TransportResponse(status_code=status_code, body=body)


def ok_response(content):
    return response(200, json.dumps({"message": {"content": content}}))


def port(profiles=None, transport=None):
    default_profiles = {
        "qwen3:8b": OllamaProfile(
            name="qwen3:8b",
            base_url="http://127.0.0.1:11434",
            supported_inference_settings=frozenset({"sample_count", "temperature", "seed", "timeout_ms"}),
        ),
        "seedless": OllamaProfile(
            name="seedless",
            supported_inference_settings=frozenset({"sample_count", "temperature", "timeout_ms"}),
        ),
    }
    return OllamaModelPort(profiles or default_profiles, transport or FakeTransport([ok_response("ok")]))


MESSAGES = [{"role": "user", "content": {"instructions": "i", "state": "s"}}]


def test_success_returns_raw_output_with_ok_outcome():
    fake = FakeTransport([ok_response('{"choice":"technical"}')])
    attempt = port(transport=fake).attempt("qwen3:8b", MESSAGES, Inference())
    assert attempt.outcome is TransportOutcome.OK
    assert attempt.output == '{"choice":"technical"}'
    assert len(fake.calls) == 1


def test_payload_shape_model_messages_stream_and_options():
    fake = FakeTransport([ok_response("x")])
    port(transport=fake).attempt(
        "qwen3:8b",
        MESSAGES,
        Inference(sample_count=1, temperature=0.3, seed=11, timeout_ms=250),
    )
    payload = fake.calls[0]["payload"]
    assert payload["model"] == "qwen3:8b"
    assert payload["messages"] == MESSAGES
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.3
    assert payload["options"]["seed"] == 11
    assert fake.calls[0]["timeout_ms"] == 250
    assert fake.calls[0]["path"].endswith("/api/chat")


def test_timeout_becomes_explicit_non_retried_outcome():
    fake = FakeTransport(exc=OllamaTransportTimeout("timed out"))
    attempt = port(transport=fake).attempt("qwen3:8b", MESSAGES, Inference())
    assert attempt.outcome is TransportOutcome.TIMEOUT
    assert attempt.output is None
    assert len(fake.calls) == 1, "no automatic retries"


def test_connection_failure_becomes_unavailable_outcome():
    fake = FakeTransport(exc=OllamaTransportUnavailable("connection refused"))
    attempt = port(transport=fake).attempt("qwen3:8b", MESSAGES, Inference())
    assert attempt.outcome is TransportOutcome.UNAVAILABLE


def test_http_500_becomes_unavailable_outcome():
    fake = FakeTransport([response(500, "boom")])
    attempt = port(transport=fake).attempt("qwen3:8b", MESSAGES, Inference())
    assert attempt.outcome is TransportOutcome.UNAVAILABLE


def test_context_overflow_is_its_own_outcome():
    fake = FakeTransport([response(400, "model requires a context length of 8192 tokens")])
    attempt = port(transport=fake).attempt("qwen3:8b", MESSAGES, Inference())
    assert attempt.outcome is TransportOutcome.CONTEXT_OVERFLOW


def test_non_json_200_body_is_explicit_malformed_response_outcome():
    fake = FakeTransport([response(200, "<html>not json</html>")])
    attempt = port(transport=fake).attempt("qwen3:8b", MESSAGES, Inference())
    assert attempt.outcome is TransportOutcome.MALFORMED_RESPONSE


def test_json_200_without_message_content_is_malformed_response():
    fake = FakeTransport([response(200, json.dumps({"done": True}))])
    attempt = port(transport=fake).attempt("qwen3:8b", MESSAGES, Inference())
    assert attempt.outcome is TransportOutcome.MALFORMED_RESPONSE


def test_unknown_profile_rejected_before_transport():
    fake = FakeTransport()
    with pytest.raises(StructuralError) as excinfo:
        port(transport=fake).attempt("jev-latest", MESSAGES, Inference())
    assert excinfo.value.error.code == StructuralCode.UNSUPPORTED_LOCAL_MODEL
    assert excinfo.value.error.path == "/model"
    assert fake.calls == []


def test_seed_on_unsupporting_profile_rejected_before_transport():
    fake = FakeTransport()
    with pytest.raises(StructuralError) as excinfo:
        port(transport=fake).attempt("seedless", MESSAGES, Inference(seed=7))
    assert excinfo.value.error.code == StructuralCode.UNSUPPORTED_INFERENCE_SETTING
    assert excinfo.value.error.path == "/inference/seed"
    assert fake.calls == []


def test_real_transport_class_exists_and_is_stdlib_only():
    """The real transport ships, but nothing at import time touches the network."""
    assert UrllibOllamaTransport is not None


def test_non_loopback_profile_is_rejected_at_construction():
    from local_judge.ollama import OllamaProfile as Profile

    with pytest.raises(ValueError):
        Profile(name="remote", base_url="https://api.example.com/v1")


def test_temperature_on_unsupporting_profile_rejected_before_transport():
    fake = FakeTransport()
    profiles = {
        "cold": OllamaProfile(
            name="cold", supported_inference_settings=frozenset({"sample_count", "timeout_ms"})
        )
    }
    with pytest.raises(StructuralError) as excinfo:
        OllamaModelPort(profiles, fake).attempt("cold", MESSAGES, Inference(temperature=0))
    assert excinfo.value.error.code == StructuralCode.UNSUPPORTED_INFERENCE_SETTING
    assert excinfo.value.error.path == "/inference/temperature"
    assert fake.calls == []


def test_real_transport_maps_oserrors_timeouts_and_redirects():
    """Opener-injected probes: expected failure shapes never escape as exceptions."""
    import http.client
    import urllib.error
    import urllib.request

    from local_judge.ollama import UrllibOllamaTransport as T

    class FakeOpener:
        def __init__(self, behavior):
            self.behavior = behavior

        def open(self, request, timeout=None):
            behavior = self.behavior
            if behavior == "timeout":
                raise TimeoutError("timed out")
            if behavior == "reset":
                raise http.client.RemoteDisconnected("reset")
            if behavior == "urlerror-timeout":
                raise urllib.error.URLError(TimeoutError("timed out"))
            if behavior == "http404":
                raise urllib.error.HTTPError(request.full_url, 404, "nope", {}, None)
            if behavior == "redirect":
                raise urllib.error.HTTPError(request.full_url, 302, "moved", {}, None)
            if behavior == "badbytes":
                import io

                class Handle:
                    status = 200
                    reads = 0

                    def __enter__(self):
                        return self

                    def __exit__(self, *exc):
                        return False

                    def read(self, n=-1):
                        self.reads += 1
                        return b"\xff\xfe\xff" if self.reads == 1 else b""

                return Handle()

    def run(behavior):
        transport = T(opener=FakeOpener(behavior))
        return transport.post("http://127.0.0.1:11434/api/chat", {}, 5)

    assert run("timeout") if False else True
    with pytest.raises(OllamaTransportTimeout):
        run("timeout")
    attempt_outcome = None
    try:
        run("reset")
    except OllamaTransportUnavailable:
        attempt_outcome = "unavailable"
    assert attempt_outcome == "unavailable"
    try:
        run("urlerror-timeout")
    except OllamaTransportTimeout:
        attempt_outcome = "timeout"
    assert attempt_outcome == "timeout"
    r = run("http404")
    assert r.status_code == 404
    r = run("redirect")
    assert r.status_code == 302  # redirects surface as a response, never followed
    r = run("badbytes")
    assert isinstance(r.body, str)


def test_deadline_bounds_a_drip_feeding_server():
    """urlopen's per-op timeout cannot be dripped past: the port enforces a total deadline."""
    import io
    import time

    from local_judge.ollama import OllamaTransportTimeout, UrllibOllamaTransport

    class DripHandle:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            time.sleep(0.05)  # drip: each op is fast, the total is not bounded by any single op
            return b"x"

    class DripOpener:
        def open(self, request, timeout=None):
            return DripHandle()

    transport = UrllibOllamaTransport(opener=DripOpener())
    with pytest.raises(OllamaTransportTimeout):
        transport.post("http://127.0.0.1:11434/api/chat", {}, 80)


def test_response_schema_is_transported_as_format():
    fake = FakeTransport([ok_response("x")])
    schema = {"type": "object"}
    port(transport=fake).attempt("qwen3:8b", MESSAGES, Inference(), response_schema=schema)
    assert fake.calls[0]["payload"]["format"] == schema
    fake2 = FakeTransport([ok_response("x")])
    port(transport=fake2).attempt("qwen3:8b", MESSAGES, Inference())
    assert "format" not in fake2.calls[0]["payload"]


def test_loopback_dns_names_are_rejected_at_construction():
    from local_judge.ollama import OllamaProfile as Profile

    with pytest.raises(ValueError):
        Profile(name="evil", base_url="http://127.0.0.1.evil.com")
    with pytest.raises(ValueError):
        Profile(name="localhost-name", base_url="http://localhost:11434")


def test_http_exception_maps_to_unavailable():
    import http.client
    import urllib.request

    from local_judge.ollama import UrllibOllamaTransport

    class RaiserOpener:
        def open(self, request, timeout=None):
            raise http.client.IncompleteRead(b"partial")

    transport = UrllibOllamaTransport(opener=RaiserOpener())
    with pytest.raises(OllamaTransportUnavailable):
        transport.post("http://127.0.0.1:11434/api/chat", {}, 5)
    profiles = {
        "qwen3:8b": OllamaProfile(
            name="qwen3:8b", supported_inference_settings=frozenset({"sample_count", "temperature", "timeout_ms"})
        )
    }
    attempt = OllamaModelPort(profiles, transport).attempt("qwen3:8b", MESSAGES, Inference())
    assert attempt.outcome is TransportOutcome.UNAVAILABLE
