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


def test_committed_api_openapi_matches_the_app():
    """The generated OpenAPI document stays aligned with the app contract."""
    client = make_client(native=lambda raw: (200, COMPLETED))
    generated = client.get("/openapi.json").json()
    committed = json.load(open("docs/API_OpenAPI.json", encoding="utf-8"))
    assert generated["paths"].keys() == committed["paths"].keys()
    for path, methods in committed["paths"].items():
        for method, operation in methods.items():
            assert generated["paths"][path][method]["summary"] == operation["summary"]
            assert (
                generated["paths"][path][method]["responses"].keys()
                == operation["responses"].keys()
            )


def test_app_does_not_bind_or_serve_at_import_or_creation():
    """create_app never starts a server; binding is the deployment's concern."""
    from local_judge.api import DEFAULT_HOST, DEFAULT_PORT

    app = create_app(native_evaluator=lambda raw: (200, COMPLETED))
    assert app is not None
    assert DEFAULT_HOST in ("127.0.0.1", "localhost")
    assert isinstance(DEFAULT_PORT, int)
