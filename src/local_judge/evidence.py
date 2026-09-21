"""Versioned evidence corpus runner (issue #19; docs/CONTRACT.md 'Acceptance Evidence').

Runs a corpus through a supplied face callable (library, HTTP, or MCP) and
computes the Stage 1 metrics with their denominators. A failed gate means the
deployment cannot claim demonstrated usefulness for that model profile and
corpus version — it is never evidence of a security or calibration flaw.
"""

from statistics import mean

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
    status = case_result.get("status")
    if status == "rejected":
        return False
    results = case_result.get("results", {})
    return bool(results) and all(r.get("status") == "answered" for r in results.values())


def _answer_matches(case_result, case):
    results = case_result.get("results", {})
    expected = case.get("expected_answer")
    allowed = case.get("allowed_answers")
    for qid, entry in results.items():
        answer = entry.get("answer")
        if expected is not None and qid in expected:
            if answer != expected[qid]:
                return False
        if allowed is not None and qid in allowed:
            if answer not in allowed[qid]:
                return False
    return True


def run_corpus(cases: list, face) -> dict:
    """Execute cases through one face callable and compute the report metrics.

    face(case) receives the raw request payload and returns the native result
    dict (or, for the Jev face, the adapter result). Cases carry their scripted
    model outputs, so every face is deterministic and reproducible.
    """
    submitted = answered = correct = 0
    invalid_output = inability = backend_error = 0
    agreements = []
    ambiguous_submitted = ambiguous_allowed = 0
    gate1_pass = gate1_total = 0

    for case in cases:
        case_result = face(case)
        case_class = case.get("case_class", "deterministic")
        results = case_result.get("results", {}) if isinstance(case_result, dict) else {}
        if case_class == "deterministic":
            gate1_total += 1
            if all(
                entry.get("status") == "answered" and not entry.get("error")
                for entry in results.values()
            ):
                gate1_pass += 1
        if case_class in ("deterministic", "normal", "adversarial"):
            submitted += 1
            if _answered(case_result):
                answered += 1
                if _answer_matches(case_result, case):
                    correct += 1
        if case_class == "ambiguous":
            ambiguous_submitted += 1
            if _answered(case_result) and _answer_matches(case_result, case):
                ambiguous_allowed += 1
        for entry in results.values():
            error = entry.get("error") or {}
            code = error.get("code")
            if entry.get("status") == "question_error":
                if code == "INVALID_MODEL_OUTPUT":
                    invalid_output += 1
                elif code == "MODEL_UNAVAILABLE":
                    backend_error += 1
            elif entry.get("status") == "inability_to_answer":
                inability += 1
        for entry in results.values():
            if entry.get("status") == "answered" and entry.get("agreement") is not None:
                agreements.append(entry["agreement"])

    answer_coverage = answered / submitted if submitted else None
    accuracy = correct / answered if answered else None
    allowed_outcome_coverage = (
        ambiguous_allowed / ambiguous_submitted if ambiguous_submitted else None
    )

    # matched adversarial pairs: task preservation vs the matched normal twin
    adversarial = [c for c in cases if c.get("case_class") == "adversarial"]
    normal_by_id = {
        c.get("case_id"): c for c in cases if c.get("case_class") == "normal"
    }
    matched_pairs = []
    for case in adversarial:
        normal = normal_by_id.get(case.get("matched_case_id"))
        if normal is None:
            continue
        adversarial_result = face(case)
        normal_result = face(normal)
        preserved = _answered(adversarial_result) and _answered(normal_result) and _answer_matches(
            adversarial_result, case
        )
        matched_pairs.append(
            {"case": case, "normal_result": normal_result, "preserved": preserved}
        )

    relations = {}
    for case in cases:
        if case.get("case_class") != "metamorphic":
            continue
        relation = case.get("metamorphic_relation")
        relations.setdefault(relation, {"pairs": 0, "invariant": 0})
        relations[relation]["pairs"] += 1
    invariance = {}
    for relation, stats in relations.items():
        invariance[relation] = stats["invariant"] / stats["pairs"] if stats["pairs"] else None
    task_preservation = (
        sum(1 for pair in matched_pairs if pair["preserved"]) / len(matched_pairs)
        if matched_pairs
        else None
    )

    agreement_distribution = {
        "min": min(agreements) if agreements else None,
        "max": max(agreements) if agreements else None,
        "mean": round(mean(agreements), 6) if agreements else None,
    }

    report = {
        "metrics": {
            "answer_coverage": {
                "value": answer_coverage,
                "answered": answered,
                "submitted": submitted,
            },
            "accuracy": {
                "value": accuracy,
                "correct": correct,
                "answered": answered,
            },
            "allowed_outcome_coverage": {
                "value": allowed_outcome_coverage,
                "in_allowed_set": ambiguous_allowed,
                "submitted": ambiguous_submitted,
            },
            "task_preservation": {
                "value": task_preservation,
                "pairs": len(matched_pairs),
                "excluded_count": 0,
            },
            "invariance": invariance,
            "agreement_distribution": agreement_distribution,
            "invalid_output_rate": {
                "value": invalid_output / submitted if submitted else None,
                "count": invalid_output,
                "submitted": submitted,
            },
            "inability_rate": {
                "value": inability / submitted if submitted else None,
                "count": inability,
                "submitted": submitted,
            },
            "backend_error_rate": {
                "value": backend_error / submitted if submitted else None,
                "count": backend_error,
                "submitted": submitted,
            },
        },
        "deterministic_fixtures": {"pass": gate1_pass, "total": gate1_total},
        "demonstrated_usefulness": None,
        "note": (
            "A failed gate means Stage 1 cannot claim demonstrated usefulness "
            "for this model profile and corpus version. It is not proof of a "
            "security or calibration failure."
        ),
    }
    report["demonstrated_usefulness"] = _gates_pass(report)
    return report


def _gates_pass(report: dict) -> bool:
    metrics = report["metrics"]
    fixtures = report["deterministic_fixtures"]
    if fixtures["total"] and fixtures["pass"] != fixtures["total"]:
        return False
    coverage = metrics["answer_coverage"]
    accuracy = metrics["accuracy"]
    if coverage["submitted"] and (coverage["value"] is None or coverage["value"] < GATE_THRESHOLDS["normal_min_coverage"]):
        return False
    if accuracy["answered"] and (accuracy["value"] is None or accuracy["value"] < GATE_THRESHOLDS["normal_min_accuracy"]):
        return False
    allowed = metrics["allowed_outcome_coverage"]
    if allowed["submitted"] and (allowed["value"] is None or allowed["value"] < GATE_THRESHOLDS["ambiguous_min_allowed_outcome_coverage"]):
        return False
    task = metrics["task_preservation"]
    if task["pairs"] and (task["value"] is None or task["value"] < 1 - GATE_THRESHOLDS["adversarial_max_task_preservation_drop_pp"] / 100):
        return False
    return True
