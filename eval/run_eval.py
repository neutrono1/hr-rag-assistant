"""
Tiny eval harness. Run after `seed.py` and with the API server up:
    python eval/run_eval.py

Checks, per question:
  - refusal questions actually get sufficient=false and no citations
  - factual questions cite the expected document and contain the
    expected substring (case-insensitive) somewhere in the answer

This is intentionally small (spec: "a tiny evaluation set"). It is meant
to be run after any prompt/chunking change to catch regressions, not to
be a statistically rigorous benchmark.
"""
import json
import sys
from pathlib import Path

import requests

API_URL = "http://localhost:8000"
EVAL_FILE = Path(__file__).parent / "eval_set.json"


def run():
    cases = json.loads(EVAL_FILE.read_text())
    passed, failed = 0, 0

    for case in cases:
        resp = requests.post(f"{API_URL}/query", json={"question": case["question"]}, timeout=30)
        data = resp.json()
        answer_lower = data["answer"].lower()

        ok = True
        reasons = []

        if case.get("must_be_refusal"):
            if data["sufficient"] or data["citations"]:
                ok = False
                reasons.append("expected a refusal but got a sufficient/cited answer")
        else:
            expected_doc = case.get("expected_doc")
            if expected_doc and not any(c["document"] == expected_doc for c in data["citations"]):
                ok = False
                reasons.append(f"expected citation from {expected_doc}, got {data['citations']}")
            must_contain = case.get("must_contain", [])
            if must_contain and not any(m.lower() in answer_lower for m in must_contain):
                ok = False
                reasons.append(f"expected one of {must_contain} in answer, got: {data['answer']!r}")

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] ({case['type']}) {case['question']}")
        if not ok:
            for r in reasons:
                print(f"        -> {r}")
        passed += ok
        failed += not ok

    print(f"\n{passed} passed, {failed} failed out of {len(cases)}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    run()
