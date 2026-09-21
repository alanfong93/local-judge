"""Versioned prompt compiler (issue #11; docs/CONTRACT.md 'Policy and Prompt Boundary').

Renders exactly three identified regions per sample prompt:
1. system  — engine-owned output contract and output schema,
2. user    — caller-declared policy for the typed question (instructions, criteria),
3. user    — state evidence.

State is rendered only as evidence and never enters the policy or engine
regions; question IDs are correlation keys and are never rendered.
"""

import json
from dataclasses import dataclass
from typing import Mapping

_SAMPLE_UNION_CONTRACT = (
    "Respond with exactly one JSON value chosen from the sample union for the "
    "requested question type: a type-valid answer, or an explicit inability "
    'object of the form {"reason": "<INSUFFICIENT_EVIDENCE|AMBIGUOUS_EVIDENCE|'
    'UNSUPPORTED_QUESTION>"}. No other keys, no prose.'
)


@dataclass(frozen=True)
class VersionedPromptCompiler:
    template_version: str = "prompt-1"
    output_schema_version: str = "schema-1"

    def render(self, question_id: str, question: Mapping, state) -> tuple[list, dict]:
        """Return (rendered_messages, output_schema) for one typed question.

        question_id is accepted for interface parity with the orchestrator and
        is never rendered (it is a correlation key only).
        """
        qtype = question.get("type")
        output_schema = self._output_schema(qtype, question)
        system = {
            "role": "system",
            "content": f"[engine output contract v{self.output_schema_version}] {_SAMPLE_UNION_CONTRACT}",
        }
        policy = {
            "role": "user",
            "content": json.dumps(
                {
                    "instructions": question.get("instructions"),
                    "criteria": question.get("criteria"),
                },
                ensure_ascii=False,
            ),
        }
        evidence = {
            "role": "user",
            "content": json.dumps({"state": state}, ensure_ascii=False),
        }
        return [system, policy, evidence], output_schema

    def _output_schema(self, qtype, question) -> dict:
        if qtype == "choice":
            return {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "choice": {"enum": sorted((question.get("criteria") or {}).keys())},
                            "reason": {"type": "string"},
                        },
                    },
                    {"type": "string"},
                ]
            }
        if qtype == "score":
            return {
                "oneOf": [
                    {"type": "integer", "minimum": 0},
                    {"type": "object", "properties": {"reason": {"type": "string"}}},
                    {"type": "string"},
                ]
            }
        if qtype == "noul":
            return {
                "oneOf": [
                    {"type": "number", "minimum": 0, "maximum": 1},
                    {"type": "object", "properties": {"reason": {"type": "string"}}},
                    {"type": "string"},
                ]
            }
        raise ValueError(f"unknown question type: {qtype!r}")
