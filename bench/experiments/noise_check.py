"""Does the regression gate false-alarm on identical runs?

Runs the SAME workload N times in two modes against the local mock server:
  single  : 1 trial, no warm-up, 64 requests        (original design)
  robust  : median of 3 trials, warm-up, >=8 req/worker (current default)
Run 1 of each mode is the baseline; runs 2..N are compared to it with compare.py's
thresholds. Any flagged regression is a false alarm, since nothing changed.

    python bench/experiments/noise_check.py --runs 6 --concurrency 1,8,32
Results depend on the machine; re-run locally and report your own numbers.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
from compare import compare  # noqa: E402

MODES = {
    "single": ["--repeats", "1", "--warmup", "0", "--min-samples-per-worker", "0", "--requests", "64"],
    "robust": ["--repeats", "3", "--warmup", "16", "--min-samples-per-worker", "8", "--requests", "64"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--concurrency", default="1,8,32")
    ap.add_argument("--server-bin", default=str(ROOT / "mock_server/build/mock_server"))
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--chunk-delay-ms", type=int, default=5)
    args = ap.parse_args()

    srv = subprocess.Popen([args.server_bin, str(args.port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.4)
    summary = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for mode, extra in MODES.items():
                results = []
                for r in range(args.runs):
                    out = Path(tmp) / f"{mode}_{r}.json"
                    subprocess.run([sys.executable, str(ROOT / "bench/bench.py"),
                                    "--base-url", f"http://127.0.0.1:{args.port}/v1",
                                    "--model", "mock-model", "--concurrency", args.concurrency,
                                    "--max-tokens", "12", "--prompt-words", "64",
                                    "--header", f"X-Mock-Fault: chunk_delay_ms={args.chunk_delay_ms}",
                                    "--out", str(out), *extra],
                                   check=True, capture_output=True)
                    results.append(json.loads(out.read_text()))
                base = results[0]
                false_alarms, worst = 0, {}
                for cur in results[1:]:
                    rows, regs = compare(base, cur)
                    false_alarms += bool(regs)
                    for c, metric, bv, cv, ch, _ in rows:
                        if metric == "ttft_ms.p95":
                            worst[c] = max(worst.get(c, 0.0), abs(ch))
                p95_c = {lv["concurrency"]: [] for lv in base["levels"]}
                for res in results:
                    for lv in res["levels"]:
                        p95_c[lv["concurrency"]].append(lv["ttft_ms"]["p95"])
                cv = {c: round(statistics.stdev(v) / statistics.mean(v), 4) for c, v in p95_c.items()}
                summary[mode] = {"comparisons": args.runs - 1, "false_alarms": false_alarms,
                                 "max_abs_ttft_p95_change_vs_baseline": {c: round(w, 4) for c, w in worst.items()},
                                 "ttft_p95_coefficient_of_variation": cv,
                                 "ttft_p95_values_ms": p95_c}
    finally:
        srv.terminate()
        srv.wait()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
