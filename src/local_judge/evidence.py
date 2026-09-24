"""Versioned evidence corpus runner (issue #19; docs/CONTRACT.md 'Acceptance Evidence').

Runs a corpus through a supplied face callable (library, HTTP, or MCP) and
computes the Stage 1 metrics with their denominators, then evaluates the
acceptance gates. A failed gate means the deployment cannot claim
demonstrated usefulness for that model profile and corpus version — it is
never proof of a security or calibration flaw.

Issue #35: the runner enforces its acceptance contract. Deterministic
fixtures must match their declared expected outcome (answer, rejection,
question error, adapter refusal, trace fields) and are reported per contract
category. Unlabelled normal/ambiguous cases are excluded from authoritative
evidence and counted. A case counts as answered only when its results carry
exactly the submitted question IDs, each validly answered. Task preservation
is relative to the matched normal twins' labelled accuracy on the same pair
set (at most 10 percentage points lower). Metamorphic invariance counts
empty/malformed/unaligned/dangling pairs as ineligible (never invariant) and
requires every v1 relation. Caller threshold overrides are pilot diagnostics
only: they can never authorize demonstrated_usefulness.
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

DETERMINISTIC_CATEGORIES = (
    "envelope_validation",
    "typed_validation",
    "aggregate_equations",
    "trace_fields",
    "replay_configuration",
    "adapter_refusal",
)

METAMORPHIC_V1_RELATIONS = (
    "json-key-reorder",
    "question-map-reorder",
    "irrelevant-evidence-insertion",
    "id-aligned-permutation",
)

_ANSWER_TYPES = frozenset({"choice", "score", "noul"})


def _submitted_qids(case) -> set:
    return {case["question_id"]}


def _answer_type_ok(entry) -> bool:
    """A native answered entry carries a type-specific answer object: the
    answer is an object whose only type discriminator is the entry's type."""
    type_ = entry.get("type")
    answer = entry.get("answer")
    if type_ not in _ANSWER_TYPES or not isinstance(answer, dict):
        return False
    return (_ANSWER_TYPES & set(answer)) == {type_}


def _valid_answered_entry(entry) -> bool:
    return (
        isinstance(entry, dict)
        and entry.get("status") == "answered"
        and entry.get("error") is None
        and _answer_type_ok(entry)
    )


def _results_aligned(case_result, case) -> bool:
    """A result map counts only when it carries exactly the submitted IDs."""
    if not isinstance(case_result, dict):
        return False
    results = case_result.get("results")
    if not isinstance(results, dict) or not results:
        return False
    return set(results) == _submitted_qids(case)


def _answered(case_result, case) -> bool:
    if not _results_aligned(case_result, case):
        return False
    return all(_valid_answered_entry(r) for r in case_result["results"].values())


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


def _is_labelled(case) -> bool:
    return bool(case.get("expected_answer") or case.get("allowed_answers"))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def _is_invariant(base_answer, variant_answer) -> bool:
    """Invariance compares one shared type discriminator (docs/CONTRACT.md
    'Acceptance Evidence'): a Choice selects the same option, a Score changes
    by at most 0.1, a Noul changes by at most 0.05. Answers that do not carry
    exactly one matching discriminator are not type-valid and never count."""
    if not isinstance(base_answer, dict) or not isinstance(variant_answer, dict):
        return False
    base_keys = _ANSWER_TYPES & set(base_answer)
    variant_keys = _ANSWER_TYPES & set(variant_answer)
    if len(base_keys) != 1 or base_keys != variant_keys:
        return False
    if base_keys == {"choice"}:
        return base_answer["choice"] == variant_answer["choice"]
    if base_keys == {"score"}:
        return abs(base_answer["score"] - variant_answer["score"]) <= 0.1
    return abs(base_answer["noul"] - variant_answer["noul"]) <= 0.05


def _deterministic_case_passes(entry) -> bool:
    case = entry["case"]
    result = entry["result"]
    if not isinstance(result, dict):
        return False
    # a fixture that declares no expected outcome has nothing to enforce;
    # it never passes vacuously
    if not any(
        key in case
        for key in (
            "expected_answer",
            "allowed_answers",
            "expected_rejection",
            "expected_error_code",
            "expected_question_error",
        )
    ):
        return False

    if "expected_rejection" in case:
        declared = case["expected_rejection"] or {}
        error = result.get("error") or {}
        return result.get("status") == "rejected" and error.get("code") == declared.get("code")

    if "expected_error_code" in case:
        error = result.get("error") or {}
        # a completed native response carries error: null; only the adapter
        # refusal shape (no status) or a native rejection may bear an error
        return (
            result.get("status") in (None, "rejected")
            and error.get("code") == case["expected_error_code"]
        )

    expected_question_error = case.get("expected_question_error") or {}
    if expected_question_error:
        results = result.get("results") or {}
        if set(results) != _submitted_qids(case):
            return False
        for qid, code in expected_question_error.items():
            entry_ = results.get(qid)
            if not isinstance(entry_, dict) or entry_.get("status") != "question_error":
                return False
            if (entry_.get("error") or {}).get("code") != code:
                return False
        return all(
            _valid_answered_entry(results[qid])
            for qid in results
            if qid not in expected_question_error
        )

    # answered expectation: aligned, validly answered, and matching the labels
    if not _answered(result, case):
        return False
    if not _answer_matches(result, case):
        return False
    declared_trace_fields = case.get("expected_trace_fields") or []
    for qid in _submitted_qids(case):
        trace = result["results"][qid].get("trace")
        if not isinstance(trace, dict) or not all(f in trace for f in declared_trace_fields):
            return False
    return True


def run_corpus(cases: list, face, thresholds: Mapping | None = None) -> dict:
    """Execute the corpus through one face and produce the full evidence report.

    thresholds may be supplied for pilot diagnostics only. They never change
    the authoritative gates or demonstrated_usefulness, which are always
    evaluated against the fixed contract thresholds.
    """
    t = dict(GATE_THRESHOLDS)
    overrides_applied = bool(thresholds)
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

    # unlabelled normal/ambiguous cases carry no authoritative expectation
    unlabelled_ids = [
        e["case"].get("case_id")
        for e in normal + ambiguous
        if not _is_labelled(e["case"])
    ]
    unlabelled = set(unlabelled_ids)
    normal_labelled = [e for e in normal if e["case"].get("case_id") not in unlabelled]
    ambiguous_labelled = [e for e in ambiguous if e["case"].get("case_id") not in unlabelled]

    # deterministic fixtures: each must match its declared expected outcome;
    # pass rates are reported for every contract-required category
    det_results = [_deterministic_case_passes(e) for e in deterministic]
    det_pass = sum(1 for ok in det_results if ok)
    deterministic_categories = {}
    for category in DETERMINISTIC_CATEGORIES:
        members = [
            ok for e, ok in zip(deterministic, det_results)
            if e["case"].get("deterministic_category") == category
        ]
        deterministic_categories[category] = {
            "pass": sum(1 for ok in members if ok),
            "total": len(members),
            "rate": _ratio(sum(1 for ok in members if ok), len(members)),
        }

    # normal-class metrics over labelled cases
    answered = sum(1 for e in normal_labelled if _answered(e["result"], e["case"]))
    correct = sum(
        1 for e in normal_labelled
        if _answered(e["result"], e["case"]) and _answer_matches(e["result"], e["case"])
    )
    answer_coverage = _ratio(answered, len(normal_labelled))
    accuracy = _ratio(correct, answered)

    # ambiguous metrics over labelled cases
    ambiguous_allowed = sum(
        1 for e in ambiguous_labelled
        if _answered(e["result"], e["case"]) and _answer_matches(e["result"], e["case"])
    )
    ambiguous_inability = sum(
        1 for e in ambiguous
        if any(
            isinstance(r, dict) and r.get("status") == "inability_to_answer"
            for r in e["result"].get("results", {}).values()
        )
    )

    # error rates over the labelled normal class (aligned result maps only)
    invalid_output = backend_error = 0
    inability = 0
    normal_question_entries = 0
    for e in normal_labelled:
        results = e["result"].get("results")
        if not isinstance(results, dict) or set(results) != _submitted_qids(e["case"]):
            continue
        for r in results.values():
            normal_question_entries += 1
            code = (r.get("error") or {}).get("code") if isinstance(r, dict) else None
            if isinstance(r, dict) and r.get("status") == "question_error":
                if code == "INVALID_MODEL_OUTPUT":
                    invalid_output += 1
                if code == "MODEL_UNAVAILABLE":
                    backend_error += 1
            if isinstance(r, dict) and r.get("status") == "inability_to_answer":
                inability += 1

    # matched adversarial pairs: task preservation vs the labelled accuracy of
    # the answered normal twins over the same pair set (Alan's resolved
    # interpretation: preservation may be at most 10 percentage points lower)
    normal_results = {e["case"].get("case_id"): e["result"] for e in normal}
    normal_cases_by_id = {e["case"].get("case_id"): e["case"] for e in normal}
    preserved = 0
    twins_correct = 0
    matched_pairs = 0
    excluded_unanswered_twin = 0
    for e in adversarial:
        matched_id = e["case"].get("matched_case_id")
        twin = normal_results.get(matched_id)
        twin_case = normal_cases_by_id.get(matched_id)
        # labelled accuracy is undefined for an unlabelled twin: the pair is
        # excluded and reported, never scored vacuously
        if (
            twin is None
            or twin_case is None
            or not _is_labelled(twin_case)
            or not _answered(twin, twin_case)
        ):
            excluded_unanswered_twin += 1
            continue
        matched_pairs += 1
        if _answer_matches(twin, twin_case):
            twins_correct += 1
        # preserved only when the adversarial result map is aligned and every
        # submitted question is validly answered within the matched normal
        # case's labelled answer set
        if _answered(e["result"], e["case"]) and _answer_matches(e["result"], twin_case):
            preserved += 1
    task_preservation = _ratio(preserved, matched_pairs)
    matched_normal_accuracy = _ratio(twins_correct, matched_pairs)
    preservation_drop_pp = (
        round((matched_normal_accuracy - task_preservation) * 100, 6)
        if matched_normal_accuracy is not None and task_preservation is not None
        else None
    )

    # metamorphic invariance per relation; empty, unanswered, malformed,
    # partial, unaligned, and dangling pairs are ineligible and never
    # invariant — they stay in the denominator
    invariance = {}
    metamorphic_detail = {}
    case_by_id = {e["case"].get("case_id"): e for e in evaluated}
    for relation in sorted({e["case"].get("metamorphic_relation") for e in metamorphic}):
        total = eligible = ineligible = invariant = 0
        for e in metamorphic:
            if e["case"].get("metamorphic_relation") != relation:
                continue
            total += 1
            base_entry = case_by_id.get(e["case"].get("matched_case_id"))
            base_results = (
                base_entry["result"].get("results")
                if base_entry is not None and isinstance(base_entry["result"], dict)
                else None
            )
            variant_results = e["result"].get("results")
            well_formed = (
                isinstance(base_results, dict) and bool(base_results)
                and isinstance(variant_results, dict) and bool(variant_results)
                and set(base_results) == set(variant_results)
                and all(isinstance(r, dict) for r in base_results.values())
                and all(isinstance(r, dict) for r in variant_results.values())
            )
            if not well_formed:
                ineligible += 1
                continue
            eligible += 1
            if all(
                _valid_answered_entry(base_results[q])
                and _valid_answered_entry(variant_results[q])
                and _is_invariant(base_results[q].get("answer"), variant_results[q].get("answer"))
                for q in base_results
            ):
                invariant += 1
        # ineligible pairs remain in the denominator: never invariant
        invariance[relation] = _ratio(invariant, total)
        metamorphic_detail[relation] = {
            "eligible": eligible,
            "ineligible": ineligible,
            "total": total,
            "invariant": invariant,
            "rate": invariance[relation],
        }

    agreement_values = []
    for e in normal:
        for r in e["result"].get("results", {}).values():
            if isinstance(r, dict) and r.get("status") == "answered" and r.get("agreement") is not None:
                agreement_values.append(r["agreement"])
    agreement_distribution = {
        "min": min(agreement_values) if agreement_values else None,
        "max": max(agreement_values) if agreement_values else None,
        "mean": round(sum(agreement_values) / len(agreement_values), 6) if agreement_values else None,
    }

    def _build_gates(effective: Mapping, authoritative: bool) -> list:
        gates = []
        gates.append({
            "gate": "deterministic_fixtures",
            "pass": det_pass == len(deterministic) and bool(deterministic),
            "detail": {"pass": det_pass, "total": len(deterministic)},
        })
        gates.append({
            "gate": "normal_cases",
            "pass": (
                len(normal_labelled) >= effective["normal_min_cases"]
                and answer_coverage is not None
                and answer_coverage >= effective["normal_min_coverage"]
                and accuracy is not None
                and accuracy >= effective["normal_min_accuracy"]
            ),
            "detail": {
                "cases": len(normal_labelled),
                "min_cases": effective["normal_min_cases"],
                "answer_coverage": answer_coverage,
                "min_coverage": effective["normal_min_coverage"],
                "accuracy": accuracy,
                "min_accuracy": effective["normal_min_accuracy"],
            },
        })
        gates.append({
            "gate": "ambiguous_cases",
            "pass": (
                len(ambiguous_labelled) >= effective["ambiguous_min_cases"]
                and _ratio(ambiguous_allowed, len(ambiguous_labelled)) is not None
                and ambiguous_allowed >= effective["ambiguous_min_allowed_outcome_coverage"] * max(len(ambiguous_labelled), 1)
            ),
            "detail": {
                "cases": len(ambiguous_labelled),
                "min_cases": effective["ambiguous_min_cases"],
                "allowed_outcome_coverage": _ratio(ambiguous_allowed, len(ambiguous_labelled)),
                "inability_count": ambiguous_inability,
            },
        })
        gates.append({
            "gate": "adversarial_matched_pairs",
            "pass": (
                matched_pairs >= effective["adversarial_min_matched_pairs"]
                and task_preservation is not None
                and matched_normal_accuracy is not None
                and (matched_normal_accuracy - task_preservation) * 100
                <= effective["adversarial_max_task_preservation_drop_pp"]
            ),
            "detail": {
                "pairs": matched_pairs,
                "min_pairs": effective["adversarial_min_matched_pairs"],
                "task_preservation": task_preservation,
                "matched_normal_accuracy": matched_normal_accuracy,
                "drop_pp": preservation_drop_pp,
                "max_drop_pp": effective["adversarial_max_task_preservation_drop_pp"],
                "excluded_unanswered_twin": excluded_unanswered_twin,
            },
        })
        missing_relations = [
            relation for relation in METAMORPHIC_V1_RELATIONS
            if relation not in invariance
        ]
        gates.append({
            "gate": "metamorphic_invariance",
            "pass": (
                not missing_relations
                and all(
                    metamorphic_detail[relation]["total"] >= effective["metamorphic_min_pairs_per_relation"]
                    and invariance[relation] is not None
                    and invariance[relation] >= effective["metamorphic_min_invariance"]
                    for relation in METAMORPHIC_V1_RELATIONS
                )
            ),
            "detail": {
                "relations": metamorphic_detail,
                "missing_relations": missing_relations,
                "min_pairs_per_relation": effective["metamorphic_min_pairs_per_relation"],
                "min_invariance": effective["metamorphic_min_invariance"],
            },
        })
        for gate in gates:
            gate["authoritative"] = authoritative
        return gates

    gates = _build_gates(GATE_THRESHOLDS, authoritative=True)
    pilot_gates = _build_gates(t, authoritative=False) if overrides_applied else []
    demonstrated = all(gate["pass"] for gate in gates)

    return {
        "gates": gates,
        "pilot_gates": pilot_gates,
        "demonstrated_usefulness": demonstrated,
        "effective_thresholds": dict(GATE_THRESHOLDS),
        "threshold_overrides": {"applied_to_authoritative_gates": False} if overrides_applied else None,
        "metrics": {
            "answer_coverage": {"value": answer_coverage, "answered": answered, "submitted": len(normal_labelled)},
            "accuracy": {"value": accuracy, "correct": correct, "answered": answered},
            "allowed_outcome_coverage": {
                "value": _ratio(ambiguous_allowed, len(ambiguous_labelled)),
                "in_allowed_set": ambiguous_allowed,
                "submitted": len(ambiguous_labelled),
            },
            "task_preservation": {
                "value": task_preservation,
                "pairs": matched_pairs,
                "matched_normal_accuracy": matched_normal_accuracy,
                "drop_pp": preservation_drop_pp,
                "excluded_unanswered_twin": excluded_unanswered_twin,
            },
            "unlabelled_excluded": {"count": len(unlabelled_ids), "cases": unlabelled_ids},
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
        "deterministic_categories": deterministic_categories,
        "ambiguous_inability": {"count": ambiguous_inability, "cases": len(ambiguous)},
        "note": (
            "A failed gate means Stage 1 cannot claim demonstrated usefulness "
            "for this model profile and corpus version. It is not proof of a "
            "security or calibration failure."
        ),
    }
