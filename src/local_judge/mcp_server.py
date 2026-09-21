import json

"""Thin FastMCP stdio server (docs/API_Reference.md 'MCP v1').

Tools delegate to the same library handlers as the HTTP adapter and return
the typed result objects as JSON text. Tool-level execution errors are
reserved for server startup or protocol failure.
"""

from typing import Callable

from fastmcp import FastMCP

Handler = Callable[[bytes], "tuple[int, dict]"]


def create_mcp_server(
    native_evaluator: Handler | None = None,
    replay_evaluator: Handler | None = None,
    jev_evaluator: Handler | None = None,
):
    server: FastMCP = FastMCP("local-judge")

    def _body(handler: Handler | None, raw: str) -> str:
        if handler is None:
            return json.dumps(
                {"error": {"code": "ROUTE_NOT_CONFIGURED", "path": "", "message": "tool not configured"}}
            )
        status, payload = handler(raw.encode("utf-8"))
        return json.dumps(payload)

    @server.tool
    def local_judge_evaluate(request: str) -> str:
        """Evaluate a native v1 envelope and return the native result object."""
        return _body(native_evaluator, request)

    @server.tool
    def local_judge_replay(request: str) -> str:
        """Replay one self-contained inline trace and return the native result object."""
        return _body(replay_evaluator, request)

    @server.tool
    def local_judge_evaluate_jev(request: str) -> str:
        """Evaluate a documented Jev-shaped input map and return the adapter result object."""
        return _body(jev_evaluator, request)

    return server
