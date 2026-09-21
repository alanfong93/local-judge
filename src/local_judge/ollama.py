"""Raw single-attempt model port over configured local Ollama profiles.

One bounded attempt per call, no automatic retries (docs/CONTRACT.md 'Sample
Union'). The port re-checks profile and seed support before touching
transport, and maps every transport failure shape onto an explicit
TransportOutcome for the type executor and trace recorder. It never parses
model output into typed answers.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from local_judge.errors import StructuralCode, StructuralError
from local_judge.models import Inference
from local_judge.ports import RawAttempt, TransportOutcome

_CONTEXT_MARKERS = ("context length", "context window", "num_ctx", "context size")


class TransportResponse:
    """Raw HTTP-ish response from the Ollama server."""

    __slots__ = ("status_code", "body")

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body


class OllamaTransportTimeout(Exception):
    """The attempt exceeded its timeout."""


class OllamaTransportUnavailable(Exception):
    """The Ollama server could not be reached or failed unexpectedly."""


class OllamaTransport:
    """What the port needs from a transport. Real and fake transports satisfy this."""

    def post(self, path: str, payload: Mapping[str, Any], timeout_ms: int) -> TransportResponse:
        raise NotImplementedError


@dataclass(frozen=True)
class OllamaProfile:
    """A configured local Ollama profile."""

    name: str
    base_url: str = "http://127.0.0.1:11434"
    supported_inference_settings: frozenset = frozenset(
        {"sample_count", "temperature", "timeout_ms"}
    )


class OllamaModelPort:
    """LocalModelPort over configured local Ollama profiles only."""

    def __init__(self, profiles: Mapping[str, OllamaProfile], transport) -> None:
        self._profiles = dict(profiles)
        self._transport = transport

    def attempt(
        self,
        model: str,
        rendered_messages: Sequence[Mapping[str, Any]],
        inference: Inference,
    ) -> RawAttempt:
        profile = self._profiles.get(model)
        if profile is None:
            raise StructuralError(
                StructuralCode.UNSUPPORTED_LOCAL_MODEL,
                f"requested model is not a configured local Ollama profile: {model!r}",
                "/model",
            )
        if inference.seed is not None and "seed" not in profile.supported_inference_settings:
            raise StructuralError(
                StructuralCode.UNSUPPORTED_INFERENCE_SETTING,
                "the selected profile cannot honor the requested inference setting: seed",
                "/inference/seed",
            )

        payload = {
            "model": model,
            "messages": list(rendered_messages),
            "stream": False,
            "options": self._options(inference, profile),
        }
        path = f"{profile.base_url.rstrip('/')}/api/chat"
        try:
            response = self._transport.post(path, payload, inference.timeout_ms)
        except OllamaTransportTimeout:
            return RawAttempt(outcome=TransportOutcome.TIMEOUT, output=None)
        except OllamaTransportUnavailable:
            return RawAttempt(outcome=TransportOutcome.UNAVAILABLE, output=None)

        if response.status_code != 200:
            body = response.body or ""
            if response.status_code == 400 and any(marker in body.lower() for marker in _CONTEXT_MARKERS):
                return RawAttempt(outcome=TransportOutcome.CONTEXT_OVERFLOW, output=None)
            return RawAttempt(outcome=TransportOutcome.UNAVAILABLE, output=None)

        try:
            parsed = json.loads(response.body)
        except ValueError:
            return RawAttempt(outcome=TransportOutcome.MALFORMED_RESPONSE, output=None)
        content = (
            parsed.get("message", {}).get("content")
            if isinstance(parsed, dict) and isinstance(parsed.get("message"), dict)
            else None
        )
        if not isinstance(content, str):
            return RawAttempt(outcome=TransportOutcome.MALFORMED_RESPONSE, output=None)
        return RawAttempt(outcome=TransportOutcome.OK, output=content)

    @staticmethod
    def _options(inference: Inference, profile: OllamaProfile) -> dict:
        options = {"temperature": inference.temperature}
        if inference.seed is not None and "seed" in profile.supported_inference_settings:
            options["seed"] = inference.seed
        return options


class UrllibOllamaTransport:
    """Real transport: one stdlib POST per attempt, timeout in milliseconds."""

    def post(self, path: str, payload: Mapping[str, Any], timeout_ms: int) -> TransportResponse:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(
            path, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_ms / 1000) as handle:
                return TransportResponse(status_code=handle.status, body=handle.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return TransportResponse(status_code=exc.code, body=exc.read().decode("utf-8", "replace"))
        except urllib.error.URLError as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                raise OllamaTransportTimeout(str(exc)) from exc
            raise OllamaTransportUnavailable(str(exc)) from exc
        except TimeoutError:
            raise OllamaTransportTimeout("timed out") from None
