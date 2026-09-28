"""Deployment configuration tests (issue #39, practice:spec-first)."""

import json

import pytest
from hypothesis import given, strategies as st

from fastapi.testclient import TestClient

from local_judge.deployment import (
    DeploymentConfigurationError,
    DeploymentConfig,
    DeploymentRuntime,
)
from local_judge.endpoint import OpenAICompatibleProfile
from local_judge.ports import RawAttempt, TransportOutcome


BASE_ENV = {
    "LOCAL_JUDGE_ENDPOINT_BASE_URL": "http://host.docker.internal:3040/api",
    "LOCAL_JUDGE_MODEL_IDS": "qwen3.5:8b",
}


def test_load_config_parses_model_allowlist_and_optional_settings():
    config = DeploymentConfig.from_env({
        **BASE_ENV,
        "LOCAL_JUDGE_MODEL_IDS": " qwen3.5:8b , gemma4:e4b ",
        "LOCAL_JUDGE_API_KEY": "openwebui-test-key",
        "LOCAL_JUDGE_RESPONSE_FORMAT": "json_object",
        "LOCAL_JUDGE_HTTP_PORT": "8123",
    })

    assert config.endpoint_base_url == "http://host.docker.internal:3040/api"
    assert config.model_ids == ("qwen3.5:8b", "gemma4:e4b")
    assert config.api_key == "openwebui-test-key"
    assert config.response_format == "json_object"
    assert config.http_port == 8123
    assert "openwebui-test-key" not in repr(config)


@given(st.lists(
    st.from_regex(r"[a-z][a-z0-9_-]{0,8}", fullmatch=True),
    min_size=1,
    max_size=8,
    unique=True,
))
def test_model_allowlist_parsing_preserves_unique_ids(model_ids):
    config = DeploymentConfig.from_env({
        **BASE_ENV,
        "LOCAL_JUDGE_MODEL_IDS": ",".join(model_ids),
    })
    assert config.model_ids == tuple(model_ids)


@given(st.from_regex(r"[a-z][a-z0-9_-]{0,8}", fullmatch=True))
def test_duplicate_model_ids_fail_after_whitespace_normalization(model_id):
    with pytest.raises(DeploymentConfigurationError, match="LOCAL_JUDGE_MODEL_IDS"):
        DeploymentConfig.from_env({
            **BASE_ENV,
            "LOCAL_JUDGE_MODEL_IDS": f"{model_id}, {model_id} ",
        })


@pytest.mark.parametrize(
    ("changes", "expected_variable"),
    [
        ({"LOCAL_JUDGE_ENDPOINT_BASE_URL": ""}, "LOCAL_JUDGE_ENDPOINT_BASE_URL"),
        ({"LOCAL_JUDGE_MODEL_IDS": " , "}, "LOCAL_JUDGE_MODEL_IDS"),
        ({"LOCAL_JUDGE_MODEL_IDS": "one,,two"}, "LOCAL_JUDGE_MODEL_IDS"),
        ({"LOCAL_JUDGE_RESPONSE_FORMAT": "xml"}, "LOCAL_JUDGE_RESPONSE_FORMAT"),
        ({"LOCAL_JUDGE_HTTP_PORT": "0"}, "LOCAL_JUDGE_HTTP_PORT"),
        ({"LOCAL_JUDGE_HTTP_PORT": "not-a-port"}, "LOCAL_JUDGE_HTTP_PORT"),
    ],
)
def test_invalid_configuration_names_variable_without_echoing_values(changes, expected_variable):
    env = {**BASE_ENV, **changes, "LOCAL_JUDGE_API_KEY": "secret-marker"}
    with pytest.raises(DeploymentConfigurationError) as excinfo:
        DeploymentConfig.from_env(env)
    assert expected_variable in str(excinfo.value)
    assert "secret-marker" not in str(excinfo.value)


def test_missing_required_configuration_fails_with_variable_name():
    with pytest.raises(DeploymentConfigurationError) as excinfo:
        DeploymentConfig.from_env({})
    assert "LOCAL_JUDGE_ENDPOINT_BASE_URL" in str(excinfo.value)


class ScriptedPort:
    """Scripted endpoint port; records schemas and never touches a network."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def attempt(self, model, rendered_messages, inference, response_schema=None):
        self.calls.append({
            "model": model,
            "messages": rendered_messages,
            "inference": inference,
            "response_schema": response_schema,
        })
        output = self.outputs.pop(0) if self.outputs else '"billing"'
        return RawAttempt(outcome=TransportOutcome.OK, output=output)


def deployment_runtime(port):
    config = DeploymentConfig.from_env(BASE_ENV)
    return DeploymentRuntime(config, model_port=port)


def native_request(question, *, sample_count=1):
    return {
        "contract_version": "v1",
        "state": {"ticket": "Example state"},
        "model": "qwen3.5:8b",
        "policy": {"version": "test-policy"},
        "inference": {"sample_count": sample_count, "temperature": 0},
        "questions": {"q": question},
    }


CHOICE = {
    "type": "choice",
    "instructions": "Choose the best option.",
    "criteria": {"billing": "Payments", "technical": "Bugs"},
}


def test_runtime_composes_typed_question_dispatch_and_schema_forwarding():
    port = ScriptedPort(['"billing"', "0.8", "1"])
    runtime = deployment_runtime(port)
    request = native_request(CHOICE)
    request["questions"] = {
        "choice": CHOICE,
        "noul": {"type": "noul", "instructions": "Is this a refund?"},
        "score": {
            "type": "score",
            "instructions": "Rate severity.",
            "criteria": ["Low", "High"],
        },
    }

    status, result = runtime.native_evaluator(json_bytes(request))

    assert status == 200
    assert result["status"] == "completed"
    assert result["results"]["choice"]["answer"]["choice"] == "billing"
    assert result["results"]["noul"]["answer"]["noul"] == 0.8
    assert result["results"]["score"]["answer"]["score"] == 1
    assert len(port.calls) == 3
    assert all(call["response_schema"] for call in port.calls)
    expected_backend = runtime.profiles["qwen3.5:8b"].backend_identity
    assert all(entry["trace"]["backend"] == expected_backend
               for entry in result["results"].values())


def test_runtime_model_allowlist_rejects_unknown_id_before_endpoint_call():
    port = ScriptedPort([])
    runtime = deployment_runtime(port)
    request = native_request(CHOICE)
    request["model"] = "not-configured"

    status, result = runtime.native_evaluator(json_bytes(request))

    assert status == 400
    assert result["error"]["code"] == "UNSUPPORTED_LOCAL_MODEL"
    assert result["contract_version"] == "v1"
    assert result["model"] == "not-configured"
    assert port.calls == []


def test_runtime_replay_success_links_parent_trace_and_rejects_unknown_versions():
    port = ScriptedPort(['"billing"', '"billing"'])
    runtime = deployment_runtime(port)
    status, original = runtime.native_evaluator(json_bytes(native_request(CHOICE)))
    assert status == 200
    trace = original["results"]["q"]["trace"]

    replay_status, replayed = runtime.replay_evaluator(json_bytes({
        "contract_version": "v1",
        "trace": trace,
    }))
    assert replay_status == 200
    assert replayed["results"]["q"]["trace"]["parent_trace_id"] == trace["trace_id"]

    unknown = dict(trace)
    unknown["prompt_template_version"] = "not-packaged"
    previous_calls = len(port.calls)
    unavailable_status, unavailable = runtime.replay_evaluator(json_bytes({
        "contract_version": "v1",
        "trace": unknown,
    }))
    assert unavailable_status == 409
    assert unavailable["error"]["code"] == "REPLAY_CONFIGURATION_UNAVAILABLE"
    assert len(port.calls) == previous_calls, "unavailable replay config must reject before inference"

    changed_endpoint_profile = OpenAICompatibleProfile(
        name="qwen3.5:8b",
        base_url="http://different-host:3040/api",
    )
    runtime.profiles["qwen3.5:8b"] = changed_endpoint_profile
    profile_mismatch_status, profile_mismatch = runtime.replay_evaluator(json_bytes({
        "contract_version": "v1",
        "trace": trace,
    }))
    assert profile_mismatch_status == 409
    assert profile_mismatch["error"]["code"] == "REPLAY_CONFIGURATION_UNAVAILABLE"
    assert len(port.calls) == previous_calls

    inconsistent = dict(trace)
    inconsistent["accepted_request"] = dict(trace["accepted_request"], model="other-model")
    mismatch_status, mismatch = runtime.replay_evaluator(json_bytes({
        "contract_version": "v1",
        "trace": inconsistent,
    }))
    assert mismatch_status == 409
    assert mismatch["error"]["code"] == "REPLAY_CONFIGURATION_UNAVAILABLE"
    assert len(port.calls) == previous_calls


def test_runtime_jev_maps_results_with_shared_evaluator():
    port = ScriptedPort(['"billing"', '"billing"', '"billing"'])
    runtime = deployment_runtime(port)
    jev_input = {
        "state": {"ticket": "refund"},
        "model": "qwen3.5:8b",
        "questions": {"q": CHOICE},
    }

    status, result = runtime.jev_evaluator(json_bytes(jev_input))

    assert status == 200
    assert result["error"] is None
    assert result["answers"]["q"]["choice"] == "billing"
    assert result["answers"]["q"]["confidence"] == 1.0


def test_jev_handler_rejects_duplicate_json_members():
    runtime = deployment_runtime(ScriptedPort([]))
    status, result = runtime.jev_evaluator(
        b'{"state":"a","state":"b","model":"qwen3.5:8b","questions":{}}'
    )
    assert status == 400
    assert result["error"]["code"] == "MALFORMED_JSON"


def test_http_and_mcp_modes_share_the_deployment_handlers():
    port = ScriptedPort(['"billing"'] * 20)
    runtime = deployment_runtime(port)
    app = runtime.create_http_app()
    client = TestClient(app)
    response = client.post("/v1/evaluations", json=native_request(CHOICE))
    assert response.status_code == 200
    native_result = response.json()
    assert native_result["results"]["q"]["answer"]["choice"] == "billing"
    replay_response = client.post("/v1/replays", json={
        "contract_version": "v1",
        "trace": native_result["results"]["q"]["trace"],
    })
    assert replay_response.status_code == 200
    jev_response = client.post("/v1/jev/evaluations", json={
        "state": "refund",
        "model": "qwen3.5:8b",
        "questions": {"q": CHOICE},
    })
    assert jev_response.status_code == 200
    assert jev_response.json()["answers"]["q"]["choice"] == "billing"

    from fastmcp import Client
    import asyncio

    async def call_mcp():
        async with Client(runtime.create_mcp_server()) as client:
            native_tool = await client.call_tool(
                "local_judge_evaluate",
                {"request": native_request(CHOICE)},
            )
            native_payload = json.loads(native_tool.content[0].text)
            replay_tool = await client.call_tool(
                "local_judge_replay",
                {"request": {
                    "contract_version": "v1",
                    "trace": native_payload["results"]["q"]["trace"],
                }},
            )
            jev_tool = await client.call_tool(
                "local_judge_evaluate_jev",
                {"request": {
                    "state": "refund",
                    "model": "qwen3.5:8b",
                    "questions": {"q": CHOICE},
                }},
            )
            return tuple(json.loads(result.content[0].text)
                         for result in (native_tool, replay_tool, jev_tool))

    native_payload, replay_payload, jev_payload = asyncio.run(call_mcp())
    assert native_payload["results"]["q"]["answer"]["choice"] == "billing"
    assert replay_payload["results"]["q"]["trace"]["parent_trace_id"] == \
        native_payload["results"]["q"]["trace"]["trace_id"]
    assert jev_payload["answers"]["q"]["choice"] == "billing"


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False).encode("utf-8")
