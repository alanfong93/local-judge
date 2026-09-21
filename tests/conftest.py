"""Shared helpers: Stage 1 fixtures and the published schema as contract oracle."""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "docs" / "fixtures" / "native-v1"
SCHEMA = REPO / "docs" / "schemas" / "native-v1.schema.json"


def load_fixture(name: str):
    return json.load(open(FIXTURES / name, encoding="utf-8"))


def load_schema():
    return json.load(open(SCHEMA, encoding="utf-8"))


def validate_against(instance, target: str):
    """Validate instance against a $defs target of the published schema."""
    from jsonschema import Draft202012Validator, FormatChecker

    schema = load_schema()
    validator = Draft202012Validator(
        {"$ref": target, "$defs": schema["$defs"], "$schema": "https://json-schema.org/draft/2020-12/schema"},
        format_checker=FormatChecker(),
    )
    errors = list(validator.iter_errors(instance))
    assert not errors, f"schema violations at {target}: {[e.message for e in errors[:3]]}"


TEST_PROFILE_SUPPORTED = frozenset({"sample_count", "temperature", "seed", "timeout_ms"})


@pytest.fixture()
def profile_registry():
    return {
        "qwen3:8b": frozenset({"sample_count", "temperature", "seed", "timeout_ms"}),
        "seedless": frozenset({"sample_count", "temperature", "timeout_ms"}),
    }
