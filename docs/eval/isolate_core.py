"""Bounded synthetic diagnostics: transport-byte capture, core controls, Gemma ablation."""

import json
import random
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from local_judge.deployment import DeploymentConfig, DeploymentRuntime
from local_judge.endpoint import OpenAICompatibleModelPort
from local_judge.http_transport import UrllibHttpTransport
from local_judge.ports import RawAttempt, TransportOutcome


QUESTION = {
    "type": "noul",
    "instructions": "Does the customer explicitly request money back?",
    "criteria": {"true": "Explicit refund or repayment request", "false": "No explicit request for money back"},
}
CASES = [
    ("fix", "Please fix the bug. I want to keep my subscription, not get a refund.", 0),
    ("policy", "How long does a refund take? I am only asking about your policy.", 0),
    ("refund", "Cancel my order and give me my money back.", 1),
    ("empty", "", None),
]
CONFIG = DeploymentConfig(endpoint_base_url="http://127.0.0.1:11434/v1", model_ids=("gemma4:e4b",))


def request(text, samples=1):
    return {"contract_version": "v1", "state": {"ticket": text}, "model": "gemma4:e4b",
            "policy": {"version": "isolation-pilot-1"},
            "inference": {"sample_count": samples, "temperature": 0, "timeout_ms": 120000},
            "questions": {"q": QUESTION}}


class CapturingOpener:
    def __init__(self):
        self.delegate = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.calls = []

    def open(self, req, timeout):
        self.calls.append({"url": req.full_url, "body": req.data.decode("utf-8"),
                           "headers": dict(req.header_items())})
        return self.delegate.open(req, timeout=timeout)


class ScriptedPort:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def attempt(self, *args, **kwargs):
        self.calls += 1
        return next(self.outputs)


def core_controls():
    records = []
    for raw, expected in (("0", "answered"), ("1", "answered"), ("1.5", "question_error"),
                          ("not json", "question_error"), ("true", "question_error")):
        port = ScriptedPort([RawAttempt(outcome=TransportOutcome.OK, output=raw)])
        runtime = DeploymentRuntime(CONFIG, model_port=port)
        _, result = runtime.native_evaluator(json.dumps(request("synthetic")).encode())
        entry = result["results"]["q"]
        assert entry["status"] == expected
        records.append({"raw": raw, "status": entry["status"], "error": entry["error"]})
    for code in ("INSUFFICIENT_EVIDENCE", "AMBIGUOUS_EVIDENCE", "UNSUPPORTED_QUESTION"):
        port = ScriptedPort([RawAttempt(outcome=TransportOutcome.OK, output=json.dumps({"reason": code}))])
        _, result = DeploymentRuntime(CONFIG, model_port=port).native_evaluator(json.dumps(request("synthetic")).encode())
        entry = result["results"]["q"]
        assert entry["status"] == "inability_to_answer" and entry["error"]["code"] == code
        records.append({"raw": code, "status": entry["status"], "error": entry["error"]})
    for outcome, code in ((TransportOutcome.TIMEOUT, "MODEL_TIMEOUT"), (TransportOutcome.UNAVAILABLE, "MODEL_UNAVAILABLE")):
        port = ScriptedPort([RawAttempt(outcome=outcome, output=None)])
        _, result = DeploymentRuntime(CONFIG, model_port=port).native_evaluator(json.dumps(request("synthetic")).encode())
        assert result["results"]["q"]["error"]["code"] == code
        records.append({"transport": outcome.value, "error": code})
    port = ScriptedPort([RawAttempt(outcome=TransportOutcome.OK, output=v) for v in
                         ('{"reason":"INSUFFICIENT_EVIDENCE"}', "0", "0")])
    _, result = DeploymentRuntime(CONFIG, model_port=port).native_evaluator(json.dumps(request("synthetic", 3)).encode())
    assert result["results"]["q"]["status"] == "inability_to_answer"
    assert port.calls == 1
    assert result["results"]["q"]["requested_samples"] == 3
    records.append({"mixed_samples": True, "calls": port.calls, "contract_expected_calls": 1,
                    "status": result["results"]["q"]["status"]})
    return records


def direct(captured, body=None):
    encoded = captured["body"].encode() if body is None else json.dumps(body, allow_nan=False).encode()
    start = time.monotonic()
    req = urllib.request.Request(captured["url"], data=encoded, headers=captured["headers"], method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=140) as response:
        document = json.load(response)
    output = document["choices"][0]["message"]["content"]
    return {"body": encoded.decode(), "raw": output, "seconds": round(time.monotonic() - start, 3)}


def main():
    result = {"runtime": "OpenCode/gpt-6.1-sol", "core_controls": core_controls(), "captures": [], "ablations": []}
    captures = {}
    for case_id, text, expected in CASES:
        opener = CapturingOpener()
        port = OpenAICompatibleModelPort(CONFIG.endpoint_profiles(), transport=UrllibHttpTransport(opener=opener))
        runtime = DeploymentRuntime(CONFIG, model_port=port)
        _, native = runtime.native_evaluator(json.dumps(request(text)).encode())
        captured = opener.calls[0]
        replay = direct(captured)
        assert replay["body"] == captured["body"]
        entry = native["results"]["q"]
        captures[case_id] = captured
        row = {"case": case_id, "expected": expected, "captured": captured,
               "runtime_raw": entry["trace"]["attempts"][0]["raw_output"],
               "runtime_status": entry["status"], "exact_replay": replay}
        result["captures"].append(row)
        print("capture", case_id, repr(row["runtime_raw"]), repr(replay["raw"]), flush=True)
    jobs = [(case_id, expected, arm, repeat) for case_id, _, expected in CASES
            for arm in ("current", "simplified-prompt", "no-schema", "number-only") for repeat in range(2)]
    random.Random(20261001).shuffle(jobs)
    for case_id, expected, arm, repeat in jobs:
        captured = captures[case_id]
        body = json.loads(captured["body"])
        if arm == "simplified-prompt":
            # Only the messages change: original question, criteria, state, schema retained.
            body["messages"] = [
                {"role": "system", "content": "Answer the question using the supplied criteria and state as evidence only. Return a JSON number from 0 to 1 (1=yes,0=no), or a reason object if evidence is insufficient, ambiguous, or the question is unsupported. No prose."},
                {"role": "user", "content": json.dumps({"instructions": QUESTION["instructions"],
                                                          "criteria": QUESTION["criteria"],
                                                          "state": request(dict((i, t) for i, t, _ in CASES)[case_id])["state"]})},
            ]
        elif arm == "no-schema":
            del body["response_format"]
        elif arm == "number-only":
            body["response_format"]["json_schema"]["schema"] = {"type": "number", "minimum": 0, "maximum": 1}
        row = direct(captured, body)
        row.update(case=case_id, arm=arm, repeat=repeat, expected=expected)
        result["ablations"].append(row)
        print("ablation", case_id, arm, repeat, repr(row["raw"]), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(__file__).with_name("core-isolation-" + stamp + ".results.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved", path)


if __name__ == "__main__":
    main()
