"""Streaming latency/throughput benchmark for OpenAI-compatible chat endpoints.

Measures, per concurrency level: TTFT, inter-token latency (ITL), end-to-end latency
(p50/p95/p99), output tokens/s, requests/s, and error rate. Writes one JSON file that
records the workload config and environment alongside the raw samples.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

WORDS = ("latency throughput token stream wafer engine model request batch cache "
         "schedule memory kernel compile runtime host device").split()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def make_prompt(n_words: int, seed: int) -> str:
    return " ".join(WORDS[(seed + i) % len(WORDS)] for i in range(n_words))


def one_request(session, url, headers, model, prompt_words, max_tokens, seed, timeout):
    body = {"model": model, "stream": True, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": make_prompt(prompt_words, seed)}]}
    t0 = time.perf_counter()
    token_times: list[float] = []
    usage = None
    try:
        with session.post(url, json=body, headers=headers, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                return {"ok": False, "error": f"HTTP {r.status_code}", "e2e": time.perf_counter() - t0}
            done = False
            for line in r.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices", []):
                    if ch.get("delta", {}).get("content"):
                        token_times.append(time.perf_counter())
            if not done:
                return {"ok": False, "error": "stream ended without [DONE]", "e2e": time.perf_counter() - t0}
    except (requests.RequestException, ValueError) as exc:
        return {"ok": False, "error": type(exc).__name__, "e2e": time.perf_counter() - t0}
    end = time.perf_counter()
    if not token_times:
        return {"ok": False, "error": "no content tokens", "e2e": end - t0}
    itl = [b - a for a, b in zip(token_times, token_times[1:])]
    out_tokens = (usage or {}).get("completion_tokens", len(token_times))
    return {"ok": True, "ttft": token_times[0] - t0, "itl": itl, "e2e": end - t0,
            "output_tokens": out_tokens}


def run_level_once(args, concurrency: int, n_requests: int) -> dict:
    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(args.api_key_env, "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    for h in args.header:
        k, _, v = h.partition(":")
        headers[k.strip()] = v.strip()
    local = threading.local()

    def task(i):
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return one_request(local.s, url, headers, args.model, args.prompt_words,
                           args.max_tokens, i, args.timeout)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        samples = list(pool.map(task, range(n_requests)))
    wall = time.perf_counter() - t0
    ok = [s for s in samples if s["ok"]]
    ttft = [s["ttft"] for s in ok]
    e2e = [s["e2e"] for s in ok]
    itl = [x for s in ok for x in s["itl"]]
    ms = lambda v: None if v is None else round(v * 1000, 2)  # noqa: E731
    return {
        "concurrency": concurrency,
        "requests": len(samples),
        "error_rate": round(1 - len(ok) / len(samples), 4),
        "errors": sorted({s["error"] for s in samples if not s["ok"]}),
        "ttft_ms": {f"p{p}": ms(percentile(ttft, p)) for p in (50, 95, 99)},
        "itl_ms": {"mean": ms(statistics.mean(itl)) if itl else None, "p95": ms(percentile(itl, 95))},
        "e2e_ms": {f"p{p}": ms(percentile(e2e, p)) for p in (50, 95, 99)},
        "output_tokens_per_s": round(sum(s["output_tokens"] for s in ok) / wall, 1),
        "requests_per_s": round(len(ok) / wall, 2),
    }


SCALAR_METRICS = ["error_rate", "ttft_ms.p50", "ttft_ms.p95", "ttft_ms.p99", "itl_ms.mean",
                  "itl_ms.p95", "e2e_ms.p50", "e2e_ms.p95", "e2e_ms.p99",
                  "output_tokens_per_s", "requests_per_s"]


def _get(d, path):
    for k in path.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _set(d, path, v):
    *head, last = path.split(".")
    for k in head:
        d = d.setdefault(k, {})
    d[last] = v


def run_level(args, concurrency: int) -> dict:
    """Warm up, then run `repeats` independent trials and report the median of each metric.
    Request count per trial is at least min_samples_per_worker x concurrency, so tail
    percentiles at high concurrency rest on enough samples."""
    n = max(args.requests, args.min_samples_per_worker * concurrency)
    if args.warmup:
        run_level_once(args, concurrency, max(args.warmup, concurrency))
    trials = [run_level_once(args, concurrency, n) for _ in range(args.repeats)]
    out = {"concurrency": concurrency, "requests": n, "repeats": args.repeats,
           "aggregation": "median_of_repeats",
           "errors": sorted({e for t in trials for e in t["errors"]})}
    for m in SCALAR_METRICS:
        vals = [v for v in (_get(t, m) for t in trials) if v is not None]
        _set(out, m, round(statistics.median(vals), 4 if m == "error_rate" else 2) if vals else None)
    p95s = [t["ttft_ms"]["p95"] for t in trials if t["ttft_ms"]["p95"] is not None]
    out["ttft_p95_per_repeat"] = p95s
    out["ttft_p95_spread"] = (round((max(p95s) - min(p95s)) / statistics.median(p95s), 4)
                              if len(p95s) > 1 else None)
    return out


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=2).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="IQLAB_API_KEY")
    ap.add_argument("--concurrency", default="1,4,16", help="comma-separated levels")
    ap.add_argument("--requests", type=int, default=64, help="minimum requests per trial")
    ap.add_argument("--min-samples-per-worker", type=int, default=8,
                    help="each trial sends at least this many requests per concurrent worker")
    ap.add_argument("--repeats", type=int, default=3, help="independent trials per level (median reported)")
    ap.add_argument("--warmup", type=int, default=16, help="warm-up requests per level, discarded (0 = none)")
    ap.add_argument("--prompt-words", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--header", action="append", default=[], help="extra 'Key: Value' header")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    result = {
        "label": args.label,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                   if k not in ("api_key_env",)},
        "environment": {"git_sha": git_sha(), "python": platform.python_version(),
                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        "levels": [run_level(args, c) for c in levels],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    for lv in result["levels"]:
        print(f"c={lv['concurrency']:<3} ttft_p95={lv['ttft_ms']['p95']}ms "
              f"itl_mean={lv['itl_ms']['mean']}ms e2e_p95={lv['e2e_ms']['p95']}ms "
              f"tok/s={lv['output_tokens_per_s']} err={lv['error_rate']} "
              f"(n={lv['requests']}x{lv['repeats']}, ttft_p95_spread={lv['ttft_p95_spread']})")


if __name__ == "__main__":
    main()
