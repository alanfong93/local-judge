"""Runnable HTTP and MCP composition for the container deployment."""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping

from local_judge.adapter import JevAdapter
from local_judge.api import create_app, serve
from local_judge.endpoint import OpenAICompatibleModelPort, OpenAICompatibleProfile
from local_judge.errors import ErrorObject, StructuralCode, StructuralError
from local_judge.executors.choice import ChoiceExecutor
from local_judge.executors.noul import NoulExecutor
from local_judge.executors.score import ScoreExecutor
from local_judge.mcp_server import create_mcp_server
from local_judge.models import (
    CompletedResponse,
    RejectionResponse,
    RequestEnvelope,
    TraceRecord,
)
from local_judge.orchestrator import SamplingOrchestrator, resolve_replay
from local_judge.prompt import VersionedPromptCompiler
from local_judge.validation import MAX_ENCODED_BYTES, ModelProfile, RequestValidator

_VERSIONS = {
    "prompt_template_version": "prompt-1",
    "output_schema_version": "schema-1",
    "aggregation_version": "a1",
}


def _no_non_json_constant(value: str) -> None:
    raise ValueError(f"{value} is not a JSON value")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object member")
        result[key] = value
    return result


def _strict_json_loads(raw: bytes) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_unique_json_object,
        parse_constant=_no_non_json_constant,
    )


class DeploymentConfigurationError(ValueError):
    """Invalid or incomplete container environment configuration."""


@dataclass(frozen=True)
class DeploymentConfig:
    endpoint_base_url: str
    model_ids: tuple[str, ...]
    api_key: str | None = field(default=None, repr=False)
    response_format: str = "json_schema"
    http_port: int = 8000

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "DeploymentConfig":
        env = os.environ if environ is None else environ

        base_url = env.get("LOCAL_JUDGE_ENDPOINT_BASE_URL", "").strip()
        if not base_url:
            raise DeploymentConfigurationError(
                "LOCAL_JUDGE_ENDPOINT_BASE_URL is required"
            )

        raw_model_ids = env.get("LOCAL_JUDGE_MODEL_IDS", "")
        if not raw_model_ids.strip():
            raise DeploymentConfigurationError("LOCAL_JUDGE_MODEL_IDS is required")
        model_ids = tuple(item.strip() for item in raw_model_ids.split(","))
        if any(not model_id for model_id in model_ids):
            raise DeploymentConfigurationError(
                "LOCAL_JUDGE_MODEL_IDS must not contain empty entries"
            )
        if len(set(model_ids)) != len(model_ids):
            raise DeploymentConfigurationError(
                "LOCAL_JUDGE_MODEL_IDS must contain unique model IDs"
            )

        api_key = env.get("LOCAL_JUDGE_API_KEY") or None
        response_format = env.get(
            "LOCAL_JUDGE_RESPONSE_FORMAT", "json_schema"
        ).strip()
        if response_format not in ("json_schema", "json_object"):
            raise DeploymentConfigurationError(
                "LOCAL_JUDGE_RESPONSE_FORMAT must be json_schema or json_object"
            )

        raw_port = env.get("LOCAL_JUDGE_HTTP_PORT", "8000")
        try:
            http_port = int(raw_port)
        except (TypeError, ValueError):
            raise DeploymentConfigurationError(
                "LOCAL_JUDGE_HTTP_PORT must be an integer from 1 through 65535"
            ) from None
        if not 1 <= http_port <= 65535:
            raise DeploymentConfigurationError(
                "LOCAL_JUDGE_HTTP_PORT must be an integer from 1 through 65535"
            )

        try:
            for model_id in model_ids:
                OpenAICompatibleProfile(
                    name=model_id,
                    base_url=base_url,
                    api_key=api_key,
                    response_format=response_format,
                )
        except ValueError:
            # Never relay the rejected value: it could contain a credential.
            raise DeploymentConfigurationError(
                "LOCAL_JUDGE_ENDPOINT_BASE_URL or LOCAL_JUDGE_API_KEY is invalid"
            ) from None

        return cls(
            endpoint_base_url=base_url,
            model_ids=model_ids,
            api_key=api_key,
            response_format=response_format,
            http_port=http_port,
        )

    def endpoint_profiles(self) -> dict[str, OpenAICompatibleProfile]:
        return {
            model_id: OpenAICompatibleProfile(
                name=model_id,
                base_url=self.endpoint_base_url,
                api_key=self.api_key,
                response_format=self.response_format,
            )
            for model_id in self.model_ids
        }


class DeploymentRuntime:
    """Shared configured evaluator composition for HTTP and MCP modes."""

    def __init__(self, config: DeploymentConfig, model_port=None) -> None:
        self.config = config
        self.profiles = config.endpoint_profiles()
        self.validator = RequestValidator({
            model_id: ModelProfile(
                name=model_id,
                supported_inference_settings=profile.supported_inference_settings,
            )
            for model_id, profile in self.profiles.items()
        })
        self.adapter = JevAdapter(self.validator, self.profiles)
        self.model_port = model_port or OpenAICompatibleModelPort(self.profiles)
        compiler = VersionedPromptCompiler()
        self.artifact_registry = {
            "prompt_templates": {compiler.template_version: compiler},
            "output_schemas": {compiler.output_schema_version: compiler},
            "aggregations": {"a1": {"version": "a1"}},
            "models": self.profiles,
        }

    @staticmethod
    def _executor_for(question: Mapping[str, Any]):
        question_type = question.get("type")
        criteria = question.get("criteria")
        if question_type == "choice":
            return ChoiceExecutor(criteria)
        if question_type == "score":
            return ScoreExecutor(criteria)
        if question_type == "noul":
            return NoulExecutor(criteria)
        raise ValueError(f"unknown question type: {question_type!r}")

    def _run_envelope(
        self,
        envelope: RequestEnvelope,
        *,
        parent_trace_id: str | None = None,
        versions: Mapping[str, str] | None = None,
    ) -> dict:
        orchestrator = SamplingOrchestrator(
            port=self.model_port,
            executor=None,
            executor_factory=self._executor_for,
            versions=dict(versions or _VERSIONS),
            backend=self.profiles[envelope.model].backend_identity,
            parent_trace_id=parent_trace_id,
        )
        return orchestrator.run_questions(envelope)

    @staticmethod
    def _rejection_identifiers(raw: bytes) -> tuple[str | None, str | None]:
        try:
            document = _strict_json_loads(raw)
        except (TypeError, ValueError):
            return None, None
        if not isinstance(document, dict):
            return None, None
        version = "v1" if document.get("contract_version") == "v1" else None
        model = document.get("model")
        if not isinstance(model, str) or not model:
            model = None
        return version, model

    @staticmethod
    def _structural_status(raw: bytes, error: ErrorObject) -> int:
        if error.code == StructuralCode.REQUEST_TOO_LARGE and len(raw) > MAX_ENCODED_BYTES:
            return 413
        return 400

    def _native_rejection(self, raw: bytes, error: ErrorObject) -> tuple[int, dict]:
        contract_version, model = self._rejection_identifiers(raw)
        return self._structural_status(raw, error), RejectionResponse(
            contract_version=contract_version,
            model=model,
            error=error,
        ).to_dict()

    def native_evaluator(self, raw: bytes) -> tuple[int, dict]:
        try:
            envelope = self.validator.parse(raw)
        except StructuralError as exc:
            return self._native_rejection(raw, exc.error)
        results = self._run_envelope(envelope)
        return 200, CompletedResponse(
            contract_version="v1",
            model=envelope.model,
            results=results,
        ).to_dict()

    def jev_evaluator(self, raw: bytes) -> tuple[int, dict]:
        try:
            jev_input = _strict_json_loads(raw)
        except (TypeError, ValueError):
            error = ErrorObject(
                code=StructuralCode.MALFORMED_JSON,
                path="",
                message="the raw request cannot be decoded as one JSON value",
            )
            return 400, {"answers": None, "local_judge": None, "error": error.to_dict()}
        if not isinstance(jev_input, dict):
            error = ErrorObject(
                code=StructuralCode.INVALID_FIELD,
                path="",
                message="the Jev request must be a JSON object",
            )
            return 400, {"answers": None, "local_judge": None, "error": error.to_dict()}
        result = self.adapter.evaluate(jev_input, self._run_envelope)
        return (200 if result["error"] is None else 400), result

    def replay_evaluator(self, raw: bytes) -> tuple[int, dict]:
        if len(raw) > MAX_ENCODED_BYTES:
            error = ErrorObject(
                code=StructuralCode.REQUEST_TOO_LARGE,
                path="",
                message="the encoded request exceeds 256 KiB",
            )
            return 413, RejectionResponse("v1", None, error).to_dict()
        try:
            request = _strict_json_loads(raw)
        except (TypeError, ValueError):
            error = ErrorObject(
                code=StructuralCode.MALFORMED_JSON,
                path="",
                message="the raw request cannot be decoded as one JSON value",
            )
            return 400, RejectionResponse(None, None, error).to_dict()
        if not isinstance(request, dict):
            error = ErrorObject(
                code=StructuralCode.INVALID_FIELD,
                path="",
                message="the replay request must be a JSON object",
            )
            return 400, RejectionResponse(None, None, error).to_dict()
        unknown = set(request) - {"contract_version", "trace"}
        if unknown:
            field_name = sorted(unknown)[0]
            error = ErrorObject(
                code=StructuralCode.UNKNOWN_FIELD,
                path=f"/{field_name.replace('~', '~0').replace('/', '~1')}",
                message=f"unknown replay field: {field_name!r}",
            )
            return 400, RejectionResponse(request.get("contract_version"), None, error).to_dict()
        for required in ("contract_version", "trace"):
            if required not in request:
                error = ErrorObject(
                    code=StructuralCode.MISSING_FIELD,
                    path=f"/{required}",
                    message=f"required replay field is absent: {required!r}",
                )
                return 400, RejectionResponse(request.get("contract_version"), None, error).to_dict()
        if request["contract_version"] != "v1":
            error = ErrorObject(
                code=StructuralCode.UNSUPPORTED_CONTRACT_VERSION,
                path="/contract_version",
                message="contract_version must be exactly 'v1'",
            )
            return 400, RejectionResponse(None, None, error).to_dict()
        if not isinstance(request["trace"], dict):
            error = ErrorObject(
                code=StructuralCode.INVALID_FIELD,
                path="/trace",
                message="trace must be a closed result trace object",
            )
            return 400, RejectionResponse("v1", None, error).to_dict()
        try:
            trace = TraceRecord.from_dict(request["trace"])
        except (KeyError, TypeError, ValueError):
            error = ErrorObject(
                code=StructuralCode.INVALID_FIELD,
                path="/trace",
                message="trace is not a valid closed result trace",
            )
            return 400, RejectionResponse("v1", None, error).to_dict()

        profile = self.profiles.get(trace.model)
        if (
            profile is None
            or trace.backend != profile.backend_identity
            or trace.accepted_request.get("model") != trace.model
        ):
            unavailable = RejectionResponse(
                contract_version="v1",
                model=trace.model,
                error=ErrorObject(
                    code="REPLAY_CONFIGURATION_UNAVAILABLE",
                    path="",
                    message="the recorded model endpoint profile is not configured",
                ),
            )
            return 409, unavailable.to_dict()

        resolved = resolve_replay(trace, self.artifact_registry)
        if isinstance(resolved, RejectionResponse):
            return 409, resolved.to_dict()
        try:
            envelope = self.validator.parse(resolved.accepted_request)
        except StructuralError:
            unavailable = RejectionResponse(
                contract_version="v1",
                model=trace.model,
                error=ErrorObject(
                    code="REPLAY_CONFIGURATION_UNAVAILABLE",
                    path="",
                    message="the recorded request cannot be resolved by this deployment",
                ),
            )
            return 409, unavailable.to_dict()

        versions = {
            "prompt_template_version": trace.prompt_template_version,
            "output_schema_version": trace.output_schema_version,
            "aggregation_version": trace.aggregation_version,
        }
        results = self._run_envelope(
            envelope,
            parent_trace_id=trace.trace_id,
            versions=versions,
        )
        return 200, CompletedResponse(
            contract_version="v1",
            model=envelope.model,
            results=results,
        ).to_dict()

    def create_http_app(self):
        return create_app(
            native_evaluator=self.native_evaluator,
            replay_evaluator=self.replay_evaluator,
            jev_evaluator=self.jev_evaluator,
        )

    def create_mcp_server(self):
        return create_mcp_server(
            native_evaluator=self.native_evaluator,
            replay_evaluator=self.replay_evaluator,
            jev_evaluator=self.jev_evaluator,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="local-judge")
    parser.add_argument("mode", choices=("http", "mcp"), nargs="?", default="http")
    args = parser.parse_args(argv)

    try:
        config = DeploymentConfig.from_env()
    except DeploymentConfigurationError as exc:
        print(f"local-judge configuration error: {exc}", file=sys.stderr)
        return 2

    runtime = DeploymentRuntime(config)
    if args.mode == "http":
        serve(runtime.create_http_app(), host="0.0.0.0", port=config.http_port)
    else:
        runtime.create_mcp_server().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
