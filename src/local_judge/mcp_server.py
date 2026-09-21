"""Thin FastMCP stdio server (docs/API_Reference.md 'MCP v1').

Tools take the native/Jev request as a structured object, delegate to the
same library handlers as the HTTP adapter, and return the typed result object
as JSON text. Tool-level execution errors are reserved for server startup or
protocol failure: contract validation and model outcomes arrive as typed
result objects, and a handler crash surfaces as a protocol error.
"""

import json
from typing import Callable

from local_judge.errors import ErrorObject, StructuralError

from fastmcp import FastMCP

Handler = Callable[[bytes], "tuple[int, dict]"]


def create_mcp_server(
    native_evaluator: Handler,
    replay_evaluator: Handler,
    jev_evaluator: Handler,
):
    """Build the stdio MCP server around the three library handlers (all required)."""
    server: FastMCP = FastMCP("local-judge")

    def _body(handler: Handler, request: dict) -> str:
        status, payload = handler(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        return json.dumps(payload, ensure_ascii=False)

    @server.tool
    def local_judge_evaluate(request: dict) -> str:
        """Evaluate a native v1 envelope and return the native result object."""
        return _body(native_evaluator, request)

    @server.tool
    def local_judge_replay(request: dict) -> str:
        """Replay one self-contained inline trace and return the native result object."""
        return _body(replay_evaluator, request)

    @server.tool
    def local_judge_evaluate_jev(request: dict) -> str:
        """Evaluate a documented Jev-shaped input map and return the adapter result object."""
        return _body(jev_evaluator, request)

    return server


def build_default_server(model_profiles=None):
    """Fail-closed stdio deployment: no profiles are configured by default.

    Every native evaluation therefore rejects with UNSUPPORTED_LOCAL_MODEL,
    every replay rejects with REPLAY_CONFIGURATION_UNAVAILABLE (no versioned
    artifacts are wired), and the Jev adapter validates input structurally
    before refusing on the model. Deployments with configured profiles wire
    their own handlers via create_mcp_server.
    """
    from local_judge.adapter import JevAdapter
    from local_judge.models import RejectionResponse
    from local_judge.validation import RequestValidator

    validator = RequestValidator(model_profiles or {})
    adapter = JevAdapter(validator, model_profiles or {})

    def native(raw):
        try:
            validator.parse(raw)
        except StructuralError as exc:
            rejected = RejectionResponse(contract_version=None, model=None, error=exc.error)
            return 400, rejected.to_dict()
        raise RuntimeError("no native runner is wired into this deployment")

    def replay(raw):
        try:
            json.loads(raw)
        except ValueError:
            malformed = RejectionResponse(
                contract_version=None,
                model=None,
                error=ErrorObject(code="MALFORMED_JSON", path="", message="the raw request cannot be decoded as one JSON value"),
            )
            return 400, malformed.to_dict()
        unavailable = RejectionResponse(
            contract_version="v1",
            model=None,
            error=ErrorObject(
                code="REPLAY_CONFIGURATION_UNAVAILABLE",
                path="",
                message="no versioned artifacts are wired into this deployment",
            ),
        )
        return 409, unavailable.to_dict()

    def jev(raw):
        return 400, adapter.evaluate(json.loads(raw.decode("utf-8")), None)

    return create_mcp_server(
        native_evaluator=native,
        replay_evaluator=replay,
        jev_evaluator=jev,
    )


def main(model_profiles=None) -> None:
    """Build the fail-closed default server and run it over stdio."""
    server = build_default_server(model_profiles)
    server.run()


if __name__ == "__main__":
    main()
