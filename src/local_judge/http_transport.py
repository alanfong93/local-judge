"""Bounded HTTP POST transport shared by model-provider adapters."""

import http.client
import json
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Protocol


class TransportResponse:
    """Raw HTTP response consumed by a model adapter."""

    __slots__ = ("status_code", "body")

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body


class TransportTimeout(Exception):
    """The bounded HTTP request exceeded its deadline."""


class TransportUnavailable(Exception):
    """The HTTP endpoint could not be reached or failed unexpectedly."""


class HttpPostTransport(Protocol):
    def post(
        self,
        path: str,
        payload: Mapping[str, Any],
        timeout_ms: int,
        headers: Mapping[str, str] | None = None,
    ) -> TransportResponse: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward model prompts or credentials to redirected hosts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class UrllibHttpTransport:
    """One bounded JSON POST, with redirects disabled and no automatic retries."""

    def __init__(
        self,
        opener: urllib.request.OpenerDirector | None = None,
        *,
        use_environment_proxies: bool = False,
    ) -> None:
        self._opener = opener or urllib.request.build_opener(
            *(
                [urllib.request.ProxyHandler()]
                if use_environment_proxies
                else [urllib.request.ProxyHandler({})]
            ),
            _NoRedirect(),
        )

    def post(
        self,
        path: str,
        payload: Mapping[str, Any],
        timeout_ms: int,
        headers: Mapping[str, str] | None = None,
    ) -> TransportResponse:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            path, data=body, headers=request_headers, method="POST"
        )
        deadline = time.monotonic() + timeout_ms / 1000

        def remaining_seconds() -> float:
            return deadline - time.monotonic()

        def set_remaining_timeout(handle_like, remaining: float) -> None:
            # HTTPResponse -> .fp (buffered reader) -> .raw (SocketIO) -> ._sock
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
                chunks = []
                while True:
                    remaining = remaining_seconds()
                    if remaining <= 0:
                        raise TransportTimeout("request deadline exceeded")
                    set_remaining_timeout(handle, remaining)
                    chunk = handle.read(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                response_body = b"".join(chunks).decode("utf-8", "replace")
            return TransportResponse(status_code=handle.status, body=response_body)
        except urllib.error.HTTPError as exc:
            if remaining_seconds() <= 0:
                raise TransportTimeout("request deadline exceeded")
            set_remaining_timeout(exc, remaining_seconds())
            try:
                response_body = exc.read(65536).decode("utf-8", "replace")
            except Exception:
                response_body = ""
            return TransportResponse(status_code=exc.code, body=response_body)
        except urllib.error.URLError as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                raise TransportTimeout(str(exc)) from exc
            raise TransportUnavailable(str(exc)) from exc
        except TimeoutError:
            raise TransportTimeout("request timed out") from None
        except (OSError, http.client.HTTPException) as exc:
            raise TransportUnavailable(str(exc)) from exc
