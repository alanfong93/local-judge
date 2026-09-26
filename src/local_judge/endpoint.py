"""OpenAI-compatible chat-completions model port for configured endpoints.

The endpoint URL and API key are trusted deployment configuration, never
request fields. The endpoint may be local (for example Open WebUI) or remote;
local-judge does not infer locality or billing from the protocol.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import urlparse

from local_judge.errors import StructuralCode, StructuralError
from local_judge.http_transport import (
    HttpPostTransport,
    TransportTimeout,
    TransportUnavailable,
    UrllibHttpTransport,
)
from local_judge.models import Inference
from local_judge.ports import RawAttempt, TransportOutcome

_CONTEXT_MARKERS = ("context length", "context window", "maximum context", "token limit")


def _validate_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "endpoint base_url must be an http(s) URL without credentials, query, or fragment"
        )
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("endpoint base_url has an invalid port") from None
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("endpoint base_url port must be between 1 and 65535")


@dataclass(frozen=True)
class OpenAICompatibleProfile:
    """A configured chat-completions endpoint and its model capabilities.

    `base_url` is the API prefix, not the chat-completions route. For Open WebUI,
    use a URL ending in `/api`; for a conventional OpenAI-compatible endpoint,
    use its `/v1` prefix. `name` is the model identifier sent in the request.
    """

    name: str
    base_url: str
    api_key: str | None = field(default=None, repr=False)
    response_format: Literal["json_schema", "json_object"] = "json_schema"
    supported_inference_settings: frozenset[str] = frozenset(
        {"sample_count", "temperature", "timeout_ms"}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("endpoint profile name must be a nonempty string")
        if not isinstance(self.base_url, str):
            raise ValueError("endpoint profile base_url must be a string")
        _validate_base_url(self.base_url)
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise ValueError("endpoint API key must be a nonempty string or null")
        if self.api_key is not None and self.api_key != self.api_key.strip():
            raise ValueError("endpoint API key must not have leading or trailing whitespace")
        if self.api_key is not None and (
            any(ord(ch) < 32 or ord(ch) == 127 for ch in self.api_key)
            or not self.api_key.isascii()
        ):
            # http.client would reject these at putheader time with the header
            # value embedded in the exception text — the key must never be able
            # to reach an error message.
            raise ValueError("endpoint API key must be printable ASCII without control characters")
        if self.response_format not in ("json_schema", "json_object"):
            raise ValueError("response_format must be 'json_schema' or 'json_object'")
        if not isinstance(self.supported_inference_settings, frozenset):
            raise ValueError("supported_inference_settings must be a frozenset")
        if "temperature" not in self.supported_inference_settings:
            raise ValueError("OpenAI-compatible profiles must support temperature")


class OpenAICompatibleModelPort:
    """One non-streaming, non-retried chat completion per `attempt` call.

    The default transport is the bounded stdlib POST with environment
    proxies disabled; a local Open WebUI must not leak through a configured
    proxy. Pass `use_environment_proxies=True` for remote deployments that
    reach the endpoint through a corporate proxy.
    """

    def __init__(
        self,
        profiles: Mapping[str, OpenAICompatibleProfile],
        transport: HttpPostTransport | None = None,
        *,
        use_environment_proxies: bool = False,
    ) -> None:
        self._profiles = dict(profiles)
        if transport is not None:
            self._transport = transport
        else:
            self._transport = UrllibHttpTransport(
                use_environment_proxies=use_environment_proxies
            )

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
                f"requested model is not a configured inference endpoint profile: {model!r}",
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

        payload: dict[str, Any] = {
            "model": model,
            "messages": list(rendered_messages),
            "stream": False,
            "temperature": inference.temperature,
        }
        if inference.seed is not None:
            payload["seed"] = inference.seed
        if response_schema is not None:
            if profile.response_format == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "local_judge_sample",
                        "strict": True,
                        "schema": dict(response_schema),
                    },
                }
            else:
                payload["response_format"] = {"type": "json_object"}

        headers = {"Authorization": f"Bearer {profile.api_key}"} if profile.api_key else {}
        url = f"{profile.base_url.rstrip('/')}/chat/completions"
        try:
            response = self._transport.post(url, payload, inference.timeout_ms, headers=headers)
        except TransportTimeout:
            return RawAttempt(outcome=TransportOutcome.TIMEOUT, output=None)
        except TransportUnavailable:
            return RawAttempt(outcome=TransportOutcome.UNAVAILABLE, output=None)

        if response.status_code in (408, 504):
            return RawAttempt(outcome=TransportOutcome.TIMEOUT, output=None)
        if response.status_code != 200:
            body = response.body or ""
            if response.status_code == 413 or (
                response.status_code in (400, 422)
                and any(marker in body.lower() for marker in _CONTEXT_MARKERS)
            ):
                return RawAttempt(outcome=TransportOutcome.CONTEXT_OVERFLOW, output=None)
            return RawAttempt(outcome=TransportOutcome.UNAVAILABLE, output=None)

        try:
            parsed = json.loads(response.body)
        except (TypeError, ValueError):
            return RawAttempt(outcome=TransportOutcome.MALFORMED_RESPONSE, output=None)
        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        message = (
            choices[0].get("message")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict)
            else None
        )
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            return RawAttempt(outcome=TransportOutcome.MALFORMED_RESPONSE, output=None)
        return RawAttempt(outcome=TransportOutcome.OK, output=content)
