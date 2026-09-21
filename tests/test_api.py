"""S4-1: the FastAPI adapter routes (issue #18, practice:tdd).

Route tests prove status/error preservation and delegation to the library
without a live model: handlers are fakes here, and one integration test wires
the real core pieces.
"""

import json

import pytest
from fastapi.testclient import TestClient

from local_judge.api import create_app

COMPLETED = {
    "contract_version": "v1",
    "model": "qwen3:8b",
    "status": "completed",
    "results": {},
    "error": None,
}


def make_client(native=None, replay=None, jev=None):
    return TestClient(create_app(native_evaluator=native, replay_evaluator=replay, jev_evaluator=jev))


def test_native_evaluation_delegates_and_preserves_200_body():
    seen = {}

    def native(raw: bytes):
        seen["raw"] = raw
        return 200, COMPLETED

    client = make_client(native=native)
    response = client.post("/v1/evaluations", content=b'{"contract_version": "v1"}')
    assert response.status_code == 200
    assert response.json() == COMPLETED
    assert seen["raw"] == b'{"contract_version": "v1"}'


def test_structural_rejection_preserves_400_and_body():
    rejected = {"contract_version": "v1", "model": None, "status": "rejected",
                "results": {}, "error": {"code": "MALFORMED_JSON", "path": "", "message": "m"}}

    def native(raw):
        return 400, rejected

    client = make_client(native=native)
    response = client.post("/v1/evaluations", content=b"{oops")
    assert response.status_code == 400
    assert response.json() == rejected


def test_replay_route_preserves_409_configuration_unavailable():
    rejected = {"contract_version": "v1", "model": "qwen3:8b", "status": "rejected",
                "results": {}, "error": {"code": "REPLAY_CONFIGURATION_UNAVAILABLE", "path": "",
                                         "message": "recorded configuration unavailable"}}

    def replay(raw):
        return 409, rejected

    client = make_client(replay=replay)
    response = client.post("/v1/replays", content=b'{"contract_version": "v1", "trace": {}}')
    assert response.status_code == 409
    assert response.json() == rejected


def test_jev_route_maps_success_and_refusal():
    def jev(raw):
        return 200, {"answers": {}, "local_judge": {}, "error": None}

    client = make_client(jev=jev)
    ok = client.post("/v1/jev/evaluations", content=b'{"state": "s", "model": "m", "questions": {"q": {}}}')
    assert ok.status_code == 200

    def refusing(raw):
        return 400, {"answers": None, "local_judge": None,
                     "error": {"code": "UNSUPPORTED_LOCAL_MODEL", "path": "/model", "message": "m"}}

    client = make_client(jev=refusing)
    refused = client.post("/v1/jev/evaluations", content=b'{"state": "s", "model": "m", "questions": {"q": {}}}')
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "UNSUPPORTED_LOCAL_MODEL"


def test_openapi_document_declares_the_three_routes():
    client = make_client(native=lambda raw: (200, COMPLETED))
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/evaluations" in paths
    assert "/v1/replays" in paths
    assert "/v1/jev/evaluations" in paths
    for path in paths:
        assert list(paths[path].keys()) == ["post"]


def test_committed_api_openapi_matches_the_app_exactly():
    """The committed document is byte-meaning-identical to the generated one."""
    from local_judge.api import create_app as ca

    stage1_defs = json.load(open("docs/schemas/native-v1.schema.json", encoding="utf-8"))["$defs"]
    app = ca(native_evaluator=lambda raw: (200, COMPLETED), stage1_defs=stage1_defs)
    client = TestClient(app)
    generated = client.get("/openapi.json").json()
    committed = json.load(open("docs/API_OpenAPI.json", encoding="utf-8"))
    assert generated == committed
    for path in ("/v1/evaluations", "/v1/replays", "/v1/jev/evaluations"):
        assert generated["paths"][path]["post"]["responses"]
    assert "native-v1" in generated["components"]["schemas"]
    assert generated["components"]["schemas"]["native-v1"]["$defs"]["requestEnvelope"]


def test_serve_defaults_bind_loopback(monkeypatch):
    import sys
    import types

    import local_judge.api as api_module

    captured = {}

    def fake_run(app, host, port):
        captured["host"] = host
        captured["port"] = port

    fake_uv = types.ModuleType("uvicorn")
    fake_uv.run = fake_run
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uv)
    app = api_module.create_app(native_evaluator=lambda raw: (200, {}))
    api_module.serve(app)
    assert captured == {"host": api_module.DEFAULT_HOST, "port": api_module.DEFAULT_PORT}


def test_app_does_not_bind_or_serve_at_import_or_creation():
    """create_app never starts a server; binding is the deployment's concern."""
    from local_judge.api import DEFAULT_HOST, DEFAULT_PORT

    app = create_app(native_evaluator=lambda raw: (200, COMPLETED))
    assert app is not None
    assert DEFAULT_HOST in ("127.0.0.1", "localhost")
    assert isinstance(DEFAULT_PORT, int)


def test_request_too_large_size_cause_preserves_413():
    rejected = {"contract_version": "v1", "model": None, "status": "rejected",
                "results": {}, "error": {"code": "REQUEST_TOO_LARGE", "path": "", "message": "256 KiB"}}

    def native(raw):
        return 413, rejected

    client = make_client(native=native)
    response = client.post("/v1/evaluations", content=b"x")
    assert response.status_code == 413
    assert response.json() == rejected


def test_real_core_integration_native_rejection_end_to_end():
    """Real validator + rejected-body construction through the route, no model."""
    from local_judge import RequestValidator, StructuralCode, StructuralError, RejectionResponse

    def native(raw: bytes):
        try:
            RequestValidator({}).parse(raw)
        except StructuralError as exc:
            rejected = RejectionResponse(contract_version="v1", model=None, error=exc.error)
            status = 413 if exc.error.code == "REQUEST_TOO_LARGE" and "256 KiB" in exc.error.message else 400
            return status, rejected.to_dict()
        raise AssertionError("expected rejection")

    client = make_client(native=native)
    bad = client.post("/v1/evaluations", content=b"{oops")
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "MALFORMED_JSON"
    oversize = client.post("/v1/evaluations", content=b'{"state": "' + b"x" * (256 * 1024) + b'"}')
    assert oversize.status_code == 413
    assert oversize.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_default_app_embeds_stage1_components():
    """Without an explicit stage1_defs, the default app still embeds Stage 1 defs."""
    from local_judge.api import create_app as ca

    app = ca(native_evaluator=lambda raw: (200, {}))
    client = TestClient(app)
    document = client.get("/openapi.json").json()
    assert "native-v1" in document["components"]["schemas"]
    assert document["components"]["schemas"]["native-v1"]["$defs"]["requestEnvelope"]
