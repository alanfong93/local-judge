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

VALID_INABILITY_CODES = ("INSUFFICIENT_EVIDENCE", "AMBIGUOUS_EVIDENCE", "UNSUPPORTED_QUESTION")

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
        """Exactly the sample union the base executor enforces locally."""
        inability_object = {
            "type": "object",
            "properties": {
                "reason": {"enum": list(VALID_INABILITY_CODES)}
            },
            "required": ["reason"],
            "additionalProperties": False,
        }
        if qtype == "choice":
            menu = sorted((question.get("criteria") or {}).keys())
            return {
                "oneOf": [
                    {"type": "string", "enum": menu},
                    inability_object,
                ]
            }
        if qtype == "score":
            top = max(len(question.get("criteria") or []) - 1, 0)
            return {
                "oneOf": [
                    {"type": "integer", "minimum": 0, "maximum": top},
                    inability_object,
                ]
            }
        if qtype == "noul":
            return {
                "oneOf": [
                    {"type": "number", "minimum": 0, "maximum": 1},
                    inability_object,
                ]
            }
        raise ValueError(f"unknown question type: {qtype!r}")
