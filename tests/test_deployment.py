"""Deployment tests for issues #39 and #44 (spec-first and TDD)."""

import json
from unittest.mock import patch

import pytest
from hypothesis import given, settings, strategies as st

from fastapi.testclient import TestClient
from conftest import validate_against

import local_judge.deployment as deployment_module

from local_judge.deployment import (
    DeploymentConfigurationError,
    DeploymentConfig,
    DeploymentRuntime,
)
from local_judge.endpoint import OpenAICompatibleProfile
from local_judge.ports import RawAttempt, TransportOutcome
from local_judge.validation import MAX_ENCODED_BYTES


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


@pytest.mark.parametrize("first_output", [
    '{"reason":"INSUFFICIENT_EVIDENCE"}',
    '{"reason":"AMBIGUOUS_EVIDENCE"}',
    '{"reason":"UNSUPPORTED_QUESTION"}',
    '"not-in-menu"',
])
def test_terminal_sample_stops_question_without_stopping_siblings(first_output):
    port = ScriptedPort([first_output, '"billing"', '"billing"', '"billing"'])
    runtime = deployment_runtime(port)
    request = native_request(CHOICE, sample_count=3)
    request["questions"]["sibling"] = CHOICE

    status, result = runtime.native_evaluator(json_bytes(request))

    assert status == 200
    failed = result["results"]["q"]
    assert failed["status"] != "answered"
    assert failed["answer"] is None and failed["agreement"] is None
    assert failed["requested_samples"] == 3
    assert len(failed["trace"]["attempts"]) == 1
    assert result["results"]["sibling"]["answer"]["choice"] == "billing"
    assert len(result["results"]["sibling"]["trace"]["attempts"]) == 3
    assert len(port.calls) == 4


def test_terminal_sample_preserves_prior_attempts_and_requested_count():
    port = ScriptedPort(['"billing"', '{"reason":"AMBIGUOUS_EVIDENCE"}'])
    status, result = deployment_runtime(port).native_evaluator(
        json_bytes(native_request(CHOICE, sample_count=3))
    )
    entry = result["results"]["q"]
    assert status == 200 and entry["status"] == "inability_to_answer"
    assert entry["requested_samples"] == 3
    assert entry["answer"] is None
    assert len(entry["trace"]["attempts"]) == 2
    assert entry["trace"]["attempts"][0]["parsed_value"] == "billing"
    assert len(port.calls) == 2


@pytest.mark.parametrize("outcome", [TransportOutcome.TIMEOUT, TransportOutcome.UNAVAILABLE])
def test_terminal_transport_failure_stops_remaining_samples(outcome):
    class FailingPort:
        calls = 0

        def attempt(self, *args, **kwargs):
            self.calls += 1
            return RawAttempt(outcome=outcome, output=None)

    port = FailingPort()
    status, result = deployment_runtime(port).native_evaluator(
        json_bytes(native_request(CHOICE, sample_count=3))
    )
    assert status == 200
    assert result["results"]["q"]["status"] == "question_error"
    assert result["results"]["q"]["requested_samples"] == 3
    assert len(result["results"]["q"]["trace"]["attempts"]) == 1
    assert port.calls == 1


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


def test_replay_uses_recorded_prompt_for_all_types_after_fresh_prompt_2_requests():
    questions_and_outputs = (
        (CHOICE, '"billing"'),
        ({"type": "score", "instructions": "Rate severity", "criteria": ["Low", "High"]}, "1"),
        ({"type": "noul", "instructions": "Is this a refund?"}, "0.9"),
    )
    port = ScriptedPort([
        output for _, output in questions_and_outputs for _ in range(3)
    ])
    runtime = deployment_runtime(port)
    old_versions = {
        "prompt_template_version": "prompt-1",
        "output_schema_version": "schema-1",
        "aggregation_version": "a1",
    }

    for question, _ in questions_and_outputs:
        request = native_request(question)
        envelope = runtime.validator.parse(json_bytes(request))
        old = runtime._run_envelope(envelope, versions=old_versions)["q"].to_dict()
        old_messages = port.calls[-1]["messages"]
        assert len(old_messages) == 3
        assert old["status"] == "answered"
        assert old["trace"]["prompt_template_version"] == "prompt-1"

        status, fresh = runtime.native_evaluator(json_bytes(request))
        assert status == 200
        assert fresh["results"]["q"]["status"] == "answered"
        assert fresh["results"]["q"]["trace"]["prompt_template_version"] == "prompt-2"
        assert port.calls[-1]["messages"][:3] == old_messages
        assert len(port.calls[-1]["messages"]) == 4

        replay_status, replay = runtime.replay_evaluator(json_bytes({
            "contract_version": "v1",
            "trace": old["trace"],
        }))
        assert replay_status == 200
        assert replay["results"]["q"]["status"] == "answered"
        assert port.calls[-1]["messages"] == old_messages
        assert replay["results"]["q"]["trace"]["prompt_template_version"] == "prompt-1"
        assert replay["results"]["q"]["trace"]["parent_trace_id"] == old["trace"]["trace_id"]


# DeploymentRuntime construction can have variable startup time; this is a shape check.
@settings(deadline=None)
@given(
    invalid_version=st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.text().filter(lambda value: value != "v1"),
        st.lists(st.integers()),
        st.dictionaries(st.text(), st.integers()),
    )
)
def test_replay_structural_rejections_never_echo_invalid_contract_versions(invalid_version):
    runtime = deployment_runtime(ScriptedPort([]))
    requests = (
        ({"contract_version": invalid_version, "trace": {}, "extra": 1}, "UNKNOWN_FIELD"),
        ({"contract_version": invalid_version}, "MISSING_FIELD"),
    )

    for request, expected_code in requests:
        status, payload = runtime.replay_evaluator(json_bytes(request))

        assert status == 400
        assert payload["contract_version"] is None
        assert payload["error"]["code"] == expected_code
        validate_against(payload, "#/$defs/rejectedResponse")


def test_oversized_replay_rejection_does_not_claim_a_parsed_contract_version():
    runtime = deployment_runtime(ScriptedPort([]))

    status, payload = runtime.replay_evaluator(b"x" * (MAX_ENCODED_BYTES + 1))

    assert status == 413
    assert payload["contract_version"] is None
    assert payload["error"]["code"] == "REQUEST_TOO_LARGE"
    validate_against(payload, "#/$defs/rejectedResponse")


@given(extra_bytes=st.integers(min_value=1, max_value=64))
def test_oversized_native_and_jev_bodies_are_rejected_before_json_parsing(
    extra_bytes,
):
    raw = b"x" * (MAX_ENCODED_BYTES + extra_bytes)
    port = ScriptedPort([])
    runtime = deployment_runtime(port)
    strict_parse_lengths = []
    original_strict_loads = deployment_module._strict_json_loads

    def tracked_strict_loads(body):
        strict_parse_lengths.append(len(body))
        return original_strict_loads(body)

    with patch.object(deployment_module, "_strict_json_loads", tracked_strict_loads):
        jev_status, jev_result = runtime.jev_evaluator(raw)

    assert jev_status == 400
    assert jev_result["error"]["code"] == "REQUEST_TOO_LARGE"
    assert strict_parse_lengths == []

    rejection_parse_lengths = []
    original_identifiers = runtime._rejection_identifiers

    def tracked_rejection_identifiers(body):
        rejection_parse_lengths.append(len(body))
        return original_identifiers(body)

    with patch.object(runtime, "_rejection_identifiers", tracked_rejection_identifiers):
        native_status, native_result = runtime.native_evaluator(raw)

    assert native_status == 413
    assert native_result["error"]["code"] == "REQUEST_TOO_LARGE"
    assert rejection_parse_lengths == []
    assert port.calls == []


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
    assert result["error"]["code"] == "INVALID_FIELD"
    assert result["error"]["path"] == "/state"


def test_replay_handler_rejects_duplicate_top_level_fields_with_native_code():
    runtime = deployment_runtime(ScriptedPort([]))
    status, result = runtime.replay_evaluator(
        b'{"contract_version":"v1","contract_version":"v1","trace":{}}'
    )

    assert status == 400
    assert result["contract_version"] is None
    assert result["error"]["code"] == "INVALID_FIELD"
    assert result["error"]["path"] == "/contract_version"
    validate_against(result, "#/$defs/rejectedResponse")


def test_jev_nested_state_duplicates_follow_native_last_value_semantics():
    port = ScriptedPort(['"billing"'] * 8)
    runtime = deployment_runtime(port)
    state = b'{"ticket":"first","ticket":"last"}'
    question = json.dumps(CHOICE, separators=(",", ":")).encode("utf-8")
    native = (
        b'{"contract_version":"v1","state":'
        + state
        + b',"model":"qwen3.5:8b","policy":{"version":"p"},'
        + b'"inference":{},"questions":{"q":'
        + question
        + b'}}'
    )
    jev = (
        b'{"state":'
        + state
        + b',"model":"qwen3.5:8b","questions":{"q":'
        + question
        + b'}}'
    )

    native_status, native_result = runtime.native_evaluator(native)
    jev_status, jev_result = runtime.jev_evaluator(jev)

    assert native_status == 200
    assert jev_status == 200
    assert jev_result["error"] is None
    assert jev_result["answers"]["q"]["choice"] == native_result["results"]["q"]["answer"]["choice"]


def test_jev_duplicate_question_members_match_native_last_value_behavior():
    port = ScriptedPort(['"billing"'] * 8)
    runtime = deployment_runtime(port)
    question = (
        b'{"type":"choice","instructions":"first","instructions":"last",'
        b'"criteria":{"billing":"p","technical":"b"}}'
    )
    native = (
        b'{"contract_version":"v1","state":"smoke","model":"qwen3.5:8b",'
        b'"policy":{"version":"p"},"inference":{},"questions":{"q":'
        + question
        + b'}}'
    )
    jev = (
        b'{"state":"smoke","model":"qwen3.5:8b","questions":{"q":'
        + question
        + b'}}'
    )

    native_status, native_result = runtime.native_evaluator(native)
    jev_status, jev_result = runtime.jev_evaluator(jev)

    assert native_status == 200
    assert jev_status == 200
    assert jev_result["error"] is None
    assert jev_result["answers"]["q"]["choice"] == native_result["results"]["q"]["answer"]["choice"]


def test_jev_duplicate_question_ids_use_the_native_structural_code():
    port = ScriptedPort([])
    runtime = deployment_runtime(port)
    question = json.dumps(CHOICE, separators=(",", ":")).encode("utf-8")
    questions = b'{"q":' + question + b',"q":' + question + b'}'
    native = (
        b'{"contract_version":"v1","state":"smoke","model":"qwen3.5:8b",'
        b'"policy":{"version":"p"},"inference":{},"questions":'
        + questions
        + b'}'
    )
    jev = b'{"state":"smoke","model":"qwen3.5:8b","questions":' + questions + b'}'

    native_status, native_result = runtime.native_evaluator(native)
    jev_status, jev_result = runtime.jev_evaluator(jev)

    assert native_status == 400
    assert jev_status == 400
    assert native_result["error"]["code"] == "DUPLICATE_QUESTION_ID"
    assert jev_result["error"]["code"] == native_result["error"]["code"]
    assert jev_result["error"]["path"] == native_result["error"]["path"] == ""
    assert port.calls == []
    validate_against(jev_result, "#/$defs/jevAdapterResult")


@given(depth=st.integers(min_value=1100, max_value=1600))
def test_deep_native_and_jev_states_fail_with_structured_rejections(depth):
    port = ScriptedPort([])
    runtime = deployment_runtime(port)
    nested_state = b"[" * depth + b"0" + b"]" * depth
    question = json.dumps(CHOICE, separators=(",", ":")).encode("utf-8")
    native = (
        b'{"contract_version":"v1","state":'
        + nested_state
        + b',"model":"qwen3.5:8b","policy":{"version":"p"},'
        + b'"inference":{},"questions":{"q":'
        + question
        + b'}}'
    )
    jev = (
        b'{"state":'
        + nested_state
        + b',"model":"qwen3.5:8b","questions":{"q":'
        + question
        + b'}}'
    )

    for handler, raw in (
        (runtime.native_evaluator, native),
        (runtime.jev_evaluator, jev),
    ):
        try:
            result = handler(raw)
        except RecursionError:
            result = None

        assert result is not None, "parser recursion errors must become structured rejections"
        status, payload = result
        assert status == 400
        assert payload["error"]["code"] == "MALFORMED_JSON"

    assert port.calls == []


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
