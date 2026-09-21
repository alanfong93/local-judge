"""S4-2: the FastMCP stdio server tools (issue #17, practice:tdd).

Tool-level execution errors are reserved for protocol failure: contract
validation and model outcomes arrive as typed result objects (JSON text).
"""

import asyncio
import json

import pytest
from fastmcp import Client

from local_judge.mcp_server import create_mcp_server

COMPLETED = {
    "contract_version": "v1",
    "model": "qwen3:8b",
    "status": "completed",
    "results": {},
    "error": None,
}


def call_tool(server, name, arguments):
    async def _run():
        async with Client(server) as client:
            result = await client.call_tool(name, arguments)
            return result.content[0].text

    return asyncio.run(_run())


def make_server(native=None, replay=None, jev=None):
    handlers = {}
    if native is not None:
        handlers["native_evaluator"] = native
    if replay is not None:
        handlers["replay_evaluator"] = replay
    if jev is not None:
        handlers["jev_evaluator"] = jev
    return create_mcp_server(**handlers)


def native_handler(payload):
    def handler(raw):
        return 200, payload

    return handler


def test_evaluate_tool_returns_the_typed_result_object():
    server = make_server(native=native_handler(COMPLETED))
    body = call_tool(server, "local_judge_evaluate", {"request": json.dumps({"contract_version": "v1"})})
    assert json.loads(body) == COMPLETED


def test_replay_tool_returns_rejected_results_as_typed_objects():
    rejected = {"contract_version": "v1", "model": "qwen3:8b", "status": "rejected",
                "results": {}, "error": {"code": "REPLAY_CONFIGURATION_UNAVAILABLE", "path": "",
                                         "message": "recorded configuration unavailable"}}

    def replay(raw):
        return 409, rejected

    server = make_server(replay=replay)
    body = call_tool(server, "local_judge_replay", {"request": json.dumps({"contract_version": "v1", "trace": {}})})
    assert json.loads(body) == rejected


def test_jev_tool_returns_the_adapter_result_object():
    adapter_result = {"answers": {}, "local_judge": {"contract_version": "v1", "traces": {},
                                                    "confidence_disclosure": "d"}, "error": None}

    def jev(raw):
        return 400, adapter_result

    server = make_server(jev=jev)
    body = call_tool(server, "local_judge_evaluate_jev", {"request": json.dumps({"state": "s", "model": "m", "questions": {"q": {}}})})
    assert json.loads(body) == adapter_result


def test_three_tools_are_exposed_with_parity():
    server = make_server(native=native_handler(COMPLETED))
    tools = asyncio.run(_list_tools(server))
    names = {t.name for t in tools}
    assert {"local_judge_evaluate", "local_judge_replay", "local_judge_evaluate_jev"} <= names


async def _list_tools(server):
    async with Client(server) as client:
        return await client.list_tools()


def test_invalid_json_request_is_a_typed_malformed_result_not_a_crash():
    from local_judge import RequestValidator, StructuralError, RejectionResponse

    def native(raw):
        try:
            RequestValidator({}).parse(raw)
        except StructuralError as exc:
            rejected = RejectionResponse(contract_version=None, model=None, error=exc.error)
            return 400, rejected.to_dict()
        raise AssertionError("expected rejection")

    server = make_server(native=native)
    body = call_tool(server, "local_judge_evaluate", {"request": "{not json"})
    parsed = json.loads(body)
    assert parsed["status"] == "rejected"
    assert parsed["error"]["code"] == "MALFORMED_JSON"
