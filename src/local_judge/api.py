"""Thin local FastAPI adapter (docs/API_Reference.md; docs/architecture.md).

The routes delegate every judgment decision to the library handlers and only
translate handler results into HTTP responses with the documented status
codes. Nothing here validates, renders, aggregates, or resolves replay. The
app never binds a socket at creation; serving is the deployment's concern and
defaults to the loopback interface.
"""

import json
from typing import Any, Callable, Mapping

from fastapi import FastAPI, Request

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

Handler = Callable[[bytes], "tuple[int, dict]"]

_EVALUATION_RESPONSES = {
    200: {"description": "native completed result"},
    400: {"description": "native rejected result, or REQUEST_TOO_LARGE caused by more than 64 questions"},
    413: {"description": "REQUEST_TOO_LARGE caused by the 256 KiB encoded-size limit"},
}
_REPLAY_RESPONSES = {
    200: {"description": "native completed result"},
    400: {"description": "invalid replay body"},
    409: {"description": "REPLAY_CONFIGURATION_UNAVAILABLE"},
}
_JEV_RESPONSES = {
    200: {"description": "Jev adapter result"},
    400: {"description": "adapter result whose error is a native structural code (input validation) or JEV_ADAPTER_UNMAPPABLE_RESULT (unmappable result)"},
}


def serve(app: FastAPI, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Serve the app, binding to the loopback interface by default."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def create_app(
    native_evaluator: Handler,
    replay_evaluator: Handler | None = None,
    jev_evaluator: Handler | None = None,
    stage1_defs: Mapping[str, Any] | None = None,
) -> FastAPI:
    """Build the FastAPI application around library-supplied handlers.

    native_evaluator and replay_evaluator receive the raw request body and
    return (status_code, json_body). jev_evaluator is optional at construction
    time for partial deployments; its route answers 503 until provided.

    stage1_defs, when supplied, are the Stage 1 schema $defs embedded into the
    generated OpenAPI document's components (docs/API_Reference.md).
    """
    app = FastAPI(
        title="local-judge",
        description="Local native v1 evaluation, replay, and Jev adapter routes.",
        version="v1",
    )

    def _run(handler: Handler | None, body: bytes) -> tuple[int, dict]:
        if handler is None:
            return 503, {
                "error": {"code": "ROUTE_NOT_CONFIGURED", "path": "", "message": "route not configured"}
            }
        return handler(body)

    @app.post("/v1/evaluations", responses=_EVALUATION_RESPONSES)
    async def evaluations(request: Request) -> dict:
        from fastapi.responses import JSONResponse

        body = await request.body()
        status, payload = _run(native_evaluator, body)
        return JSONResponse(status_code=status, content=payload)

    @app.post("/v1/replays", responses=_REPLAY_RESPONSES)
    async def replays(request: Request) -> dict:
        from fastapi.responses import JSONResponse

        body = await request.body()
        status, payload = _run(replay_evaluator, body)
        return JSONResponse(status_code=status, content=payload)

    @app.post("/v1/jev/evaluations", responses=_JEV_RESPONSES)
    async def jev_evaluations(request: Request) -> dict:
        from fastapi.responses import JSONResponse

        body = await request.body()
        status, payload = _run(jev_evaluator, body)
        return JSONResponse(status_code=status, content=payload)

    if stage1_defs is not None:
        original_openapi = app.openapi

        def openapi() -> dict:
            document = original_openapi()
            components = document.setdefault("components", {})
            components.setdefault("schemas", {})
            components["schemas"]["native-v1"] = {"$defs": dict(stage1_defs)}
            return document

        app.openapi = openapi

    return app
