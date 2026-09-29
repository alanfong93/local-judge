"""S4-2 fix-forward: structured tool inputs, stdio entry, fail-closed main.

Tool-level execution errors are reserved for protocol failure: contract
validation and model outcomes arrive as typed result objects (JSON text).
"""

import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
import time

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


def native_handler(payload):
    def handler(raw):
        return 200, payload

    return handler


def make_server(native, replay=None, jev=None):
    return create_mcp_server(
        native_evaluator=native,
        replay_evaluator=replay or native_handler(COMPLETED),
        jev_evaluator=jev or (lambda raw: (200, {"answers": {}, "local_judge": {}, "error": None})),
    )


def test_evaluate_tool_takes_a_structured_request_object():
    """Structured MCP calls reach the library handler (no string-arg validation wall)."""
    seen = {}

    def native(raw):
        seen["raw"] = raw
        return 200, COMPLETED

    server = make_server(native=native)
    body = call_tool(server, "local_judge_evaluate", {"request": {"contract_version": "v1", "state": "s"}})
    assert json.loads(body) == COMPLETED
    assert json.loads(seen["raw"])["state"] == "s"


def test_replay_tool_returns_rejected_results_as_typed_objects():
    rejected = {"contract_version": "v1", "model": "qwen3:8b", "status": "rejected",
                "results": {}, "error": {"code": "REPLAY_CONFIGURATION_UNAVAILABLE", "path": "",
                                         "message": "recorded configuration unavailable"}}

    def replay(raw):
        return 409, rejected

    server = make_server(native=native_handler(COMPLETED), replay=replay)
    body = call_tool(server, "local_judge_replay", {"request": {"contract_version": "v1", "trace": {}}})
    assert json.loads(body) == rejected


def test_jev_tool_returns_the_adapter_result_object():
    adapter_result = {"answers": {}, "local_judge": {"contract_version": "v1", "traces": {},
                                                    "confidence_disclosure": "d"}, "error": None}

    def jev(raw):
        return 400, adapter_result

    server = make_server(native=native_handler(COMPLETED), jev=jev)
    body = call_tool(server, "local_judge_evaluate_jev", {"request": {"state": "s", "model": "m", "questions": {"q": {}}}})
    assert json.loads(body) == adapter_result


def test_three_tools_are_exposed_with_parity():
    server = make_server(native=native_handler(COMPLETED))

    async def _list_tools(server):
        async with Client(server) as client:
            return await client.list_tools()

    tools = asyncio.run(_list_tools(server))
    names = {t.name for t in tools}
    assert {"local_judge_evaluate", "local_judge_replay", "local_judge_evaluate_jev"} <= names


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
    body = call_tool(server, "local_judge_evaluate", {"request": {"state": "s"}})
    parsed = json.loads(body)
    assert parsed["status"] == "rejected"
    assert parsed["error"]["code"] == "UNSUPPORTED_CONTRACT_VERSION"


def test_default_server_rejects_unsupported_model_as_typed_result():
    from local_judge.mcp_server import build_default_server

    server = build_default_server()
    body = call_tool(server, "local_judge_evaluate",
                     {"request": {"contract_version": "v1", "state": "s", "model": "qwen3:8b",
                                  "policy": {"version": "p"}, "inference": {},
                                  "questions": {"q": {"type": "noul", "instructions": "i"}}}})
    parsed = json.loads(body)
    assert parsed["status"] == "rejected"
    assert parsed["error"]["code"] == "UNSUPPORTED_LOCAL_MODEL"


def test_all_three_handlers_are_required():
    with pytest.raises(TypeError):
        create_mcp_server(native_evaluator=native_handler(COMPLETED))


def test_handler_crash_surfaces_as_a_protocol_error_not_a_typed_result():
    """A crashing handler is an engine/protocol failure: FastMCP raises ToolError."""
    from fastmcp.exceptions import ToolError

    def crashing(raw):
        raise RuntimeError("engine bug")

    server = make_server(native=crashing)
    with pytest.raises(ToolError):
        call_tool(server, "local_judge_evaluate", {"request": {"contract_version": "v1"}})


def test_unpaired_surrogate_is_a_typed_malformed_result_instead_of_tool_error():
    from local_judge.errors import StructuralError
    from local_judge.models import RejectionResponse
    from local_judge.validation import RequestValidator

    validator = RequestValidator({})
    received = []

    def native(raw):
        received.append(raw)
        try:
            validator.parse(raw)
        except StructuralError as exc:
            return 400, RejectionResponse(None, None, exc.error).to_dict()
        raise AssertionError("expected malformed JSON rejection")

    server = make_server(native=native)
    request = {
        "request": {
            "contract_version": "v1",
            "state": chr(0xD800),
            "model": "qwen3:8b",
            "policy": {"version": "p"},
            "inference": {},
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        }
    }
    try:
        body = call_tool(server, "local_judge_evaluate", request)
    except Exception:
        body = None

    assert body is not None, "MCP serialization failures must reach the shared typed rejection"
    payload = json.loads(body)
    assert payload["error"]["code"] == "MALFORMED_JSON"
    assert received == [b""]


def test_deep_stdio_request_returns_a_typed_malformed_result():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    env = os.environ.copy()
    env.update({
        "LOCAL_JUDGE_ENDPOINT_BASE_URL": "http://127.0.0.1:1/api",
        "LOCAL_JUDGE_MODEL_IDS": "qwen3:8b",
        "LOCAL_JUDGE_API_KEY": "",
        "LOCAL_JUDGE_RESPONSE_FORMAT": "json_schema",
        "LOCAL_JUDGE_HTTP_PORT": "8000",
    })
    source_root = os.path.join(repo_root, "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [sys.executable, "-m", "local_judge.deployment", "mcp"],
        cwd=repo_root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    messages = queue.Queue()
    seen_stdout: list[str] = []

    def read_stdout():
        for line in process.stdout:
            seen_stdout.append(line)
            messages.put(line)

    threading.Thread(target=read_stdout, daemon=True).start()

    def read_response(request_id, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = messages.get(timeout=max(0.1, deadline - time.monotonic()))
            except queue.Empty:
                return None
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") == request_id:
                return message
        return None

    try:
        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "depth-regression", "version": "1"},
            },
        }
        process.stdin.write(json.dumps(initialize) + "\n")
        process.stdin.flush()
        assert read_response(1) is not None, "MCP stdio server should initialize"
        process.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        depth = 5000
        nested_state = "[" * depth + "0" + "]" * depth
        request = (
            '{"contract_version":"v1","state":'
            + nested_state
            + ',"model":"qwen3:8b","policy":{"version":"p"},'
            + '"inference":{},"questions":{"q":{"type":"noul","instructions":"x"}}}'
        )
        call = (
            '{"jsonrpc":"2.0","id":2,"method":"tools/call",'
            + '"params":{"name":"local_judge_evaluate","arguments":{"request":'
            + json.dumps(request)
            + '}}}'
        )
        process.stdin.write(call + "\n")
        process.stdin.flush()
        response = read_response(2)
        assert response is not None, "deep MCP input should receive a JSON-RPC response"
        assert "error" not in response, "deep MCP input should return a typed tool result"
        text = response["result"]["content"][0]["text"]
        assert json.loads(text)["error"]["code"] == "MALFORMED_JSON"
        try:
            messages.get(timeout=1)
        except queue.Empty:
            pass
        for line in seen_stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                raise AssertionError(f"stdout carried a non-JSON line: {line!r}")
            assert message.get("jsonrpc") == "2.0", f"stdout carried a non-JSON-RPC message: {line!r}"
    finally:
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)


def test_main_builds_a_fail_closed_stdio_server():
    """The stdio entry exists and, with no profiles, native evaluation rejects UNSUPPORTED_LOCAL_MODEL."""
    from local_judge.mcp_server import build_default_server, main

    server = build_default_server()
    body = call_tool(server, "local_judge_evaluate",
                     {"request": {"contract_version": "v1", "state": "s", "model": "qwen3:8b",
                                  "policy": {"version": "p"}, "inference": {},
                                  "questions": {"q": {"type": "noul", "instructions": "i"}}}})
    parsed = json.loads(body)
    assert parsed["status"] == "rejected"
    assert parsed["error"]["code"] == "UNSUPPORTED_LOCAL_MODEL"
    assert callable(main)


def test_default_jev_tool_refuses_unsupported_model():
    from local_judge.mcp_server import build_default_server

    server = build_default_server()
    body = call_tool(server, "local_judge_evaluate_jev",
                     {"request": {"state": "s", "model": "qwen3:8b",
                                  "questions": {"q": {"type": "noul", "instructions": "i"}}}})
    parsed = json.loads(body)
    assert parsed["answers"] is None and parsed["local_judge"] is None
    assert parsed["error"]["code"] == "UNSUPPORTED_LOCAL_MODEL"


def test_default_replay_tool_refuses_without_artifacts():
    from local_judge.mcp_server import build_default_server

    server = build_default_server()
    body = call_tool(server, "local_judge_replay",
                     {"request": {"contract_version": "v1", "trace": {"trace_id": "x"}}})
    parsed = json.loads(body)
    assert parsed["error"]["code"] == "REPLAY_CONFIGURATION_UNAVAILABLE"
