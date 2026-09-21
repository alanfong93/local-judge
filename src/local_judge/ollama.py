"""Raw single-attempt model port over configured local Ollama profiles.

One bounded attempt per call, no automatic retries (docs/CONTRACT.md 'Sample
Union'). The port re-checks profile and seed support before touching
transport, and maps every transport failure shape onto an explicit
TransportOutcome for the type executor and trace recorder. It never parses
model output into typed answers.
"""

import http.client
import ipaddress
import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
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


def _require_loopback(base_url: str) -> None:
    """Local profiles only: http URLs on literal loopback IPs, never DNS names."""
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(
            f"Ollama profiles must use a literal loopback IP, got hostname {host!r}"
        ) from None
    if parsed.scheme != "http" or not (
        address.is_loopback and address.version == 4 or str(address) == "::1"
    ):
        raise ValueError(
            f"Ollama profiles must be local http endpoints on loopback, got {base_url!r}"
        )


@dataclass(frozen=True)
class OllamaProfile:
    """A configured local Ollama profile."""

    name: str
    base_url: str = "http://127.0.0.1:11434"
    supported_inference_settings: frozenset = frozenset(
        {"sample_count", "temperature", "timeout_ms"}
    )

    def __post_init__(self) -> None:
        _require_loopback(self.base_url)


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
        response_schema: Mapping[str, Any] | None = None,
    ) -> RawAttempt:
        profile = self._profiles.get(model)
        if profile is None:
            raise StructuralError(
                StructuralCode.UNSUPPORTED_LOCAL_MODEL,
                f"requested model is not a configured local Ollama profile: {model!r}",
                "/model",
            )
        if "temperature" not in profile.supported_inference_settings:
            raise StructuralError(
                StructuralCode.UNSUPPORTED_INFERENCE_SETTING,
                "the selected profile cannot honor the requested inference setting: temperature",
                "/inference/temperature",
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
        if response_schema is not None:
            # The backend JSON schema for the sample-output union; Stage 3's
            # versioned prompt owns the artifact, the port only transports it.
            payload["format"] = response_schema
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


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Redirects are never followed: local profiles only, one bounded attempt."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class UrllibOllamaTransport:
    """Real transport: one stdlib POST per attempt, timeout in milliseconds."""

    def __init__(self, opener: urllib.request.OpenerDirector | None = None) -> None:
        # ProxyHandler({}) ignores env proxies: local-only means the POST never leaves the box.
        self._opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect
        )

    def post(self, path: str, payload: Mapping[str, Any], timeout_ms: int) -> TransportResponse:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(
            path, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        deadline = time.monotonic() + timeout_ms / 1000

        def _remaining_seconds() -> float:
            return deadline - time.monotonic()

        def _set_remaining_timeout(handle_like, remaining: float) -> None:
            # HTTPResponse -> .fp (buffered reader) -> .raw (SocketIO) -> ._sock (socket)
            sock = getattr(getattr(getattr(handle_like, "fp", None), "raw", None), "_sock", None)
            if sock is None:
                return
            try:
                sock.settimeout(max(remaining, 0.001))
            except (AttributeError, OSError):
                pass

        try:
            handle = self._opener.open(request, timeout=timeout_ms / 1000)
            with handle:
                # Enforce the deadline DURING reads, not just between them: each
                # socket operation gets the remaining budget as its timeout.
                chunks = []
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise OllamaTransportTimeout("attempt deadline exceeded")
                    _set_remaining_timeout(handle, remaining)
                    chunk = handle.read(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                body = b"".join(chunks).decode("utf-8", "replace")
            return TransportResponse(status_code=handle.status, body=body)
        except urllib.error.HTTPError as exc:
            # The error body is inessential: one read bounded by the remaining
            # attempt budget (socket timeout); any failure yields an empty body.
            if _remaining_seconds() <= 0:
                raise OllamaTransportTimeout("attempt deadline exceeded")
            _set_remaining_timeout(exc, _remaining_seconds())
            try:
                body = exc.read(65536).decode("utf-8", "replace")
            except Exception:
                body = ""
            return TransportResponse(status_code=exc.code, body=body)
        except urllib.error.URLError as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                raise OllamaTransportTimeout(str(exc)) from exc
            raise OllamaTransportUnavailable(str(exc)) from exc
        except TimeoutError:
            raise OllamaTransportTimeout("timed out") from None
        except (OSError, http.client.HTTPException) as exc:
            # RemoteDisconnected, ConnectionResetError, IncompleteRead and
            # friends: every expected transport failure becomes an outcome.
            raise OllamaTransportUnavailable(str(exc)) from exc
