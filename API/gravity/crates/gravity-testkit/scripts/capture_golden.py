#!/usr/bin/env python3
"""Capture golden fixtures from the live Python `gravity` library.

Regenerates `fixtures/golden.json`, which the Rust golden tests assert against.
Run from the Python GravityV2 checkout (so `gravity` is importable):

    python3 capture_golden.py /path/to/GravityV2 > ../fixtures/golden.json

Determinism: all inputs are fixed, so the output is reproducible. Re-run only
when the upstream signing/PoW algorithms change.
"""
import json
import sys

# Allow pointing at the Python checkout: `capture_golden.py <gravity_root>`.
if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])

from gravity.glm.client import _generate_signature
from gravity.chatgpt.pow import _generate_answer

GLM_CASES = [
    ("Hello world", 1700000000000, "req-123", "user-9"),
    ("", 1700000300000, "r2", "u2"),
    ("café ☕ unicode", 1699999999999, "abc-def", "id-42"),
    ("a much longer prompt that spans words", 1700000000001, "x", "y"),
    ("boundary", 1700000299999, "b1", "b2"),
]

POW_CONFIG = [
    3000, "Mon Jan 01 2024 00:00:00 GMT-0500 (Eastern Standard Time)",
    4294705152, 0, "UA/1.0", "https://chatgpt.com/s.js", "dpl-1",
    "en-US", "en-US,en", 0, "navkey-x", "location", "document",
    123.0, "uuid-fixed", "", 8, 0.0,
]
POW_CASES = [("seed-xyz", "ff"), ("abc", "ff"), ("zzz", "ff"), ("seed-2", "00ff")]


def main() -> None:
    glm = [
        {
            "signature_prompt": prompt,
            "timestamp": ts,
            "request_id": rid,
            "user_id": uid,
            "signature": _generate_signature(
                body={"signature_prompt": prompt}, timestamp=ts, request_id=rid, user_id=uid
            ),
        }
        for prompt, ts, rid, uid in GLM_CASES
    ]
    pow_cases = []
    for seed, diff in POW_CASES:
        answer, solved = _generate_answer(seed, diff, POW_CONFIG)
        pow_cases.append({"seed": seed, "diff": diff, "answer": answer, "solved": solved})

    json.dump(
        {"glm_signatures": glm, "chatgpt_pow": pow_cases, "pow_config": POW_CONFIG},
        sys.stdout,
        indent=2,
    )


if __name__ == "__main__":
    main()
