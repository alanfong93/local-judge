"""Thin local FastAPI adapter (docs/API_Reference.md; docs/architecture.md).

The routes delegate every judgment decision to the library handlers and only
translate handler results into HTTP responses with the documented status
codes. Nothing here validates, renders, aggregates, or resolves replay. The
app never binds a socket at creation; serving is the deployment's concern and
defaults to the loopback interface.
"""

from typing import Callable

from fastapi import FastAPI, Request

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

Handler = Callable[[bytes], "tuple[int, dict]"]


def create_app(
    native_evaluator: Handler,
    replay_evaluator: Handler | None = None,
    jev_evaluator: Handler | None = None,
) -> FastAPI:
    """Build the FastAPI application around library-supplied handlers.

    native_evaluator and replay_evaluator receive the raw request body and
    return (status_code, json_body). jev_evaluator is optional at construction
    time for partial deployments; its route answers 503 until provided.
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

    @app.post("/v1/evaluations")
    async def evaluations(request: Request) -> dict:
        from fastapi.responses import JSONResponse

        body = await request.body()
        status, payload = _run(native_evaluator, body)
        return JSONResponse(status_code=status, content=payload)

    @app.post("/v1/replays")
    async def replays(request: Request) -> dict:
        from fastapi.responses import JSONResponse

        body = await request.body()
        status, payload = _run(replay_evaluator, body)
        return JSONResponse(status_code=status, content=payload)

    @app.post("/v1/jev/evaluations")
    async def jev_evaluations(request: Request) -> dict:
        from fastapi.responses import JSONResponse

        body = await request.body()
        status, payload = _run(jev_evaluator, body)
        return JSONResponse(status_code=status, content=payload)

    return app
