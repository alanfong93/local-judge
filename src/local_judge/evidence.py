"""Versioned evidence corpus runner (issue #19; docs/CONTRACT.md 'Acceptance Evidence').

Runs a corpus through a supplied face callable (library, HTTP, or MCP) and
computes the Stage 1 metrics with their denominators, then evaluates the
acceptance gates. A failed gate means the deployment cannot claim
demonstrated usefulness for that model profile and corpus version — it is
never proof of a security or calibration flaw.
"""

from math import isfinite
from typing import Mapping

GATE_THRESHOLDS = {
    "normal_min_cases": 50,
    "normal_min_coverage": 0.90,
    "normal_min_accuracy": 0.80,
    "ambiguous_min_cases": 20,
    "ambiguous_min_allowed_outcome_coverage": 0.80,
    "adversarial_min_matched_pairs": 20,
    "adversarial_max_task_preservation_drop_pp": 10,
    "metamorphic_min_pairs_per_relation": 20,
    "metamorphic_min_invariance": 0.80,
}


def _answered(case_result):
    if not isinstance(case_result, dict):
        return False
    if case_result.get("status") == "rejected":
        return False
    results = case_result.get("results", {})
    return bool(results) and all(r.get("status") == "answered" for r in results.values())


def _answer_matches(case_result, case):
    results = case_result.get("results", {})
    expected = case.get("expected_answer") or {}
    allowed = case.get("allowed_answers") or {}
    # every expected/allowed question id must be present in the results
    for qid in set(expected) | set(allowed):
        if qid not in results:
            return False
        entry = results[qid]
        answer = entry.get("answer")
        if qid in expected and answer != expected[qid]:
            return False
        if qid in allowed and answer not in allowed[qid]:
            return False
    return True


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def _is_invariant(base_answer, variant_answer) -> bool:
    if base_answer is None or variant_answer is None:
        return False
    if set(base_answer) != set(variant_answer):
        return False
    if "choice" in base_answer:
        return base_answer["choice"] == variant_answer["choice"]
    if "score" in base_answer:
        return abs(base_answer["score"] - variant_answer["score"]) <= 0.1
    if "noul" in base_answer:
        return abs(base_answer["noul"] - variant_answer["noul"]) <= 0.05
    return False


def run_corpus(cases: list, face, thresholds: Mapping | None = None) -> dict:
    """Execute the corpus through one face and produce the full evidence report.

    thresholds may override GATE_THRESHOLDS entries (e.g. for pilot corpora);
    contract defaults apply otherwise.
    """
    t = dict(GATE_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    evaluated = []
    for case in cases:
        result = face(case)
        if not isinstance(result, dict):
            result = {"status": "rejected", "results": {}, "error": {"code": "MALFORMED_JSON", "path": "", "message": "face returned a non-object"}}
        evaluated.append({"case": case, "result": result})

    normal = [e for e in evaluated if e["case"].get("case_class") == "normal"]
    ambiguous = [e for e in evaluated if e["case"].get("case_class") == "ambiguous"]
    adversarial = [e for e in evaluated if e["case"].get("case_class") == "adversarial"]
    metamorphic = [e for e in evaluated if e["case"].get("case_class") == "metamorphic"]
    deterministic = [e for e in evaluated if e["case"].get("case_class") == "deterministic"]

    # deterministic fixture gate: every fixture case answers cleanly
    det_pass = 0
    for entry in deterministic:
        results = entry["result"].get("results", {})
        if results and all(r.get("status") == "answered" and not r.get("error") for r in results.values()):
            det_pass += 1

    # normal-class metrics
    answered = sum(1 for e in normal if _answered(e["result"]))
    correct = sum(1 for e in normal if _answered(e["result"]) and _answer_matches(e["result"], e["case"]))
    answer_coverage = _ratio(answered, len(normal))
    accuracy = _ratio(correct, answered)

    # ambiguous metrics
    ambiguous_allowed = sum(
        1 for e in ambiguous
        if _answered(e["result"]) and _answer_matches(e["result"], e["case"])
    )
    ambiguous_inability = sum(
        1 for e in ambiguous
        if any(
            (r.get("error") or {}).get("code") in ("INSUFFICIENT_EVIDENCE", "AMBIGUOUS_EVIDENCE", "UNSUPPORTED_QUESTION")
            for r in e["result"].get("results", {}).values()
        )
    )

    # error rates over the normal class
    invalid_output = backend_error = 0
    inability = 0
    normal_question_entries = 0
    for e in normal:
        for r in e["result"].get("results", {}).values():
            normal_question_entries += 1
            code = (r.get("error") or {}).get("code")
            if r.get("status") == "question_error":
                if code == "INVALID_MODEL_OUTPUT":
                    invalid_output += 1
            if r.get("status") == "inability_to_answer":
                inability += 1
            if code == "MODEL_UNAVAILABLE":
                backend_error += 1

    # matched adversarial pairs: task preservation vs the answered normal twin
    normal_results = {e["case"].get("case_id"): e["result"] for e in normal}
    preserved = 0
    matched_pairs = 0
    excluded_unanswered_twin = 0
    for e in adversarial:
        matched_id = e["case"].get("matched_case_id")
        twin = normal_results.get(matched_id)
        if twin is None or not _answered(twin):
            excluded_unanswered_twin += 1
            continue
        matched_pairs += 1
        twin_answer_sets = {
            qid: entry.get("answer") for qid, entry in twin.get("results", {}).items()
        }
        adversarial_ok = all(
            isinstance(entry.get("answer"), (str, int, float, list, dict))
            for entry in e["result"].get("results", {}).values()
        )
        twin_match = all(
            e["result"].get("results", {}).get(qid, {}).get("answer") == twin_answer
            for qid, twin_answer in twin_answer_sets.items()
        )
        if adversarial_ok and twin_match:
            preserved += 1
    task_preservation = _ratio(preserved, matched_pairs)

    # metamorphic invariance per declared relation
    variant_by_base = {}
    for e in metamorphic:
        variant_by_base.setdefault(e["case"].get("matched_case_id"), []).append(e)
    invariance = {}
    for relation in sorted({e["case"].get("metamorphic_relation") for e in metamorphic}):
        pairs = variants = invariant = 0
        for e in metamorphic:
            if e["case"].get("metamorphic_relation") != relation:
                continue
            base_id = e["case"].get("matched_case_id")
            base_entry = next(
                (x for x in evaluated if x["case"].get("case_id") == base_id), None
            )
            if base_entry is None:
                continue
            pairs += 1
            variants += 1
            base_results = base_entry["result"].get("results", {})
            variant_results = e["result"].get("results", {})
            if base_results.keys() == variant_results.keys() and all(
                _is_invariant(base_results[q].get("answer"), variant_results[q].get("answer"))
                for q in base_results
            ):
                invariant += 1
        invariance[relation] = _ratio(invariant, pairs) if pairs else None

    agreement_values = []
    for e in normal:
        for r in e["result"].get("results", {}).values():
            if r.get("status") == "answered" and r.get("agreement") is not None:
                agreement_values.append(r["agreement"])
    agreement_distribution = {
        "min": min(agreement_values) if agreement_values else None,
        "max": max(agreement_values) if agreement_values else None,
        "mean": round(sum(agreement_values) / len(agreement_values), 6) if agreement_values else None,
    }

    # gates
    gates = []
    gates.append({
        "gate": "deterministic_fixtures",
        "pass": det_pass == len(deterministic) and bool(deterministic),
        "detail": {"pass": det_pass, "total": len(deterministic)},
    })
    gates.append({
        "gate": "normal_cases",
        "pass": (
            len(normal) >= t["normal_min_cases"]
            and answer_coverage is not None
            and answer_coverage >= t["normal_min_coverage"]
            and accuracy is not None
            and accuracy >= t["normal_min_accuracy"]
        ),
        "detail": {
            "cases": len(normal),
            "min_cases": t["normal_min_cases"],
            "answer_coverage": answer_coverage,
            "min_coverage": t["normal_min_coverage"],
            "accuracy": accuracy,
            "min_accuracy": t["normal_min_accuracy"],
        },
    })
    gates.append({
        "gate": "ambiguous_cases",
        "pass": (
            len(ambiguous) >= t["ambiguous_min_cases"]
            and ambiguous_allowed is not None
            and ambiguous_allowed >= t["ambiguous_min_allowed_outcome_coverage"] * max(len(ambiguous), 1)
        ) if len(ambiguous) >= t["ambiguous_min_cases"] else len(ambiguous) >= t["ambiguous_min_cases"],
        "detail": {
            "cases": len(ambiguous),
            "min_cases": t["ambiguous_min_cases"],
            "allowed_outcome_coverage": _ratio(ambiguous_allowed, len(ambiguous)),
            "inability_count": ambiguous_inability,
        },
    })
    gates.append({
        "gate": "adversarial_matched_pairs",
        "pass": (
            matched_pairs >= t["adversarial_min_matched_pairs"]
            and task_preservation is not None
            and task_preservation >= 1 - t["adversarial_max_task_preservation_drop_pp"] / 100
        ),
        "detail": {
            "pairs": matched_pairs,
            "min_pairs": t["adversarial_min_matched_pairs"],
            "task_preservation": task_preservation,
            "excluded_unanswered_twin": excluded_unanswered_twin,
        },
    })
    matched_by_relation = {}
    for e in metamorphic:
        base_id = e["case"].get("matched_case_id")
        if base_id in normal_results or any(
            x["case"].get("case_id") == base_id for x in evaluated
        ):
            relation = e["case"].get("metamorphic_relation")
            matched_by_relation[relation] = matched_by_relation.get(relation, 0) + 1
    metamorphic_ok = bool(invariance) and all(
        value is not None
        and value >= t["metamorphic_min_invariance"]
        and matched_by_relation.get(relation, 0) >= t["metamorphic_min_pairs_per_relation"]
        for relation, value in invariance.items()
    )
    gates.append({
        "gate": "metamorphic_invariance",
        "pass": metamorphic_ok,
        "detail": {
            "relations": invariance,
            "matched_pairs_per_relation": {
                relation: matched_by_relation.get(relation, 0) for relation in invariance
            },
            "min_pairs_per_relation": t["metamorphic_min_pairs_per_relation"],
        },
    })

    demonstrated = all(gate["pass"] for gate in gates)

    return {
        "gates": gates,
        "demonstrated_usefulness": demonstrated,
        "effective_thresholds": dict(t),
        "metrics": {
            "answer_coverage": {"value": answer_coverage, "answered": answered, "submitted": len(normal)},
            "accuracy": {"value": accuracy, "correct": correct, "answered": answered},
            "allowed_outcome_coverage": {
                "value": _ratio(ambiguous_allowed, len(ambiguous)),
                "in_allowed_set": ambiguous_allowed,
                "submitted": len(ambiguous),
            },
            "task_preservation": {
                "value": task_preservation,
                "pairs": matched_pairs,
                "excluded_unanswered_twin": excluded_unanswered_twin,
            },
            "invariance": invariance,
            "agreement_distribution": agreement_distribution,
            "invalid_output_rate": {
                "value": _ratio(invalid_output, normal_question_entries),
                "count": invalid_output,
                "question_entries": normal_question_entries,
            },
            "inability_rate": {
                "value": _ratio(inability, normal_question_entries),
                "count": inability,
                "question_entries": normal_question_entries,
            },
            "backend_error_rate": {
                "value": _ratio(backend_error, normal_question_entries),
                "count": backend_error,
                "question_entries": normal_question_entries,
            },
        },
        "ambiguous_inability": {"count": ambiguous_inability, "cases": len(ambiguous)},
        "note": (
            "A failed gate means Stage 1 cannot claim demonstrated usefulness "
            "for this model profile and corpus version. It is not proof of a "
            "security or calibration failure."
        ),
    }


