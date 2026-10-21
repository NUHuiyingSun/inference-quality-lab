"""Compare a benchmark run against a baseline and fail on regressions beyond thresholds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# metric path -> (direction, max allowed relative change). "lower" = lower is better.
DEFAULT_THRESHOLDS = {
    "ttft_ms.p95": ("lower", 0.20),
    "itl_ms.mean": ("lower", 0.20),
    "e2e_ms.p95": ("lower", 0.20),
    "output_tokens_per_s": ("higher", 0.15),
}
MAX_ERROR_RATE_INCREASE = 0.01  # absolute


def get(d: dict, path: str):
    for part in path.split("."):
        d = d.get(part) if isinstance(d, dict) else None
    return d


def compare(baseline: dict, current: dict, thresholds=DEFAULT_THRESHOLDS):
    rows, regressions = [], []
    base_by_c = {lv["concurrency"]: lv for lv in baseline["levels"]}
    for lv in current["levels"]:
        c = lv["concurrency"]
        b = base_by_c.get(c)
        if b is None:
            continue
        for metric, (direction, limit) in thresholds.items():
            bv, cv = get(b, metric), get(lv, metric)
            if not bv or cv is None:
                continue
            change = (cv - bv) / bv
            worse = change > limit if direction == "lower" else -change > limit
            rows.append((c, metric, bv, cv, change, worse))
            if worse:
                regressions.append(f"c={c} {metric}: {bv} -> {cv} ({change:+.1%}, limit {limit:.0%})")
        de = lv["error_rate"] - b["error_rate"]
        if de > MAX_ERROR_RATE_INCREASE:
            regressions.append(f"c={c} error_rate: {b['error_rate']} -> {lv['error_rate']}")
    return rows, regressions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("current", type=Path)
    args = ap.parse_args()
    rows, regressions = compare(json.loads(args.baseline.read_text()),
                                json.loads(args.current.read_text()))
    print("| conc | metric | baseline | current | change | status |")
    print("|---|---|---|---|---|---|")
    for c, m, bv, cv, ch, worse in rows:
        print(f"| {c} | {m} | {bv} | {cv} | {ch:+.1%} | {'REGRESSION' if worse else 'ok'} |")
    if regressions:
        print("\nRegressions:\n" + "\n".join(f"- {r}" for r in regressions))
        return 1
    print("\nNo regressions beyond thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
