"""Turn lm-eval per-sample logs into an actionable failure breakdown.

Each failed sample is put in exactly one bucket:
  api_error      - empty/missing response (request failed, timed out, or returned nothing)
  truncated      - response hit the generation limit before finishing (heuristic: very long,
                   and no answer could be extracted)
  parse_error    - model answered but the task's extraction filter found no answer ("[invalid]")
  wrong_answer   - an answer was extracted and it is incorrect
Reported separately, so a model regression is not confused with an API problem:
  request_success_rate  = valid responses / all requests          (call reliability)
  conditional_score     = task metric over valid responses only    (model quality)
  end_to_end_score      = task metric with failed calls scored 0   (what a user experiences)
Truncation is reported on its own: it usually means the generation limit or config is too
tight, not that the service failed. Each task uses its own standard metric.

Note: field names follow lm-eval's --log_samples output (doc_id, target, resps,
filtered_resps, filter, metric keys). Verify against your installed lm-eval version.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PRIMARY_METRIC = {"gsm8k": "exact_match", "ifeval": "prompt_level_strict_acc"}
PRIMARY_FILTER = {"gsm8k": "strict-match", "ifeval": "none"}
TRUNCATION_CHARS = 1500


def _first_text(x) -> str:
    while isinstance(x, list):
        if not x:
            return ""
        x = x[0]
    return "" if x is None else str(x)


def classify_sample(sample: dict, metric: str) -> str:
    raw = _first_text(sample.get("resps"))
    extracted = _first_text(sample.get("filtered_resps"))
    score = sample.get(metric)
    if isinstance(score, bool):
        score = int(score)
    if score is not None and score >= 1:
        return "correct"
    if not raw.strip():
        return "api_error"
    if extracted.strip() in ("", "[invalid]"):
        return "truncated" if len(raw) >= TRUNCATION_CHARS else "parse_error"
    return "wrong_answer"


def task_from_filename(path: Path) -> str:
    # samples_<task>_<timestamp>.jsonl
    stem = path.stem
    return stem[len("samples_"):].rsplit("_", 1)[0] if stem.startswith("samples_") else stem


def analyze(run_dir: Path) -> dict:
    out: dict = defaultdict(dict)
    for f in sorted(run_dir.rglob("samples_*.jsonl")):
        task = task_from_filename(f)
        metric = PRIMARY_METRIC.get(task, "exact_match")
        want_filter = PRIMARY_FILTER.get(task)
        model = f.relative_to(run_dir).parts[0] if f.parent != run_dir else run_dir.name
        counts: Counter = Counter()
        examples: dict[str, list] = defaultdict(list)
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            if want_filter and s.get("filter", want_filter) != want_filter:
                continue
            bucket = classify_sample(s, metric)
            counts[bucket] += 1
            if bucket != "correct" and len(examples[bucket]) < 3:
                examples[bucket].append({"doc_id": s.get("doc_id"), "target": s.get("target"),
                                         "response": _first_text(s.get("resps"))[:300]})
        n = sum(counts.values())
        valid = n - counts["api_error"]
        out[model][task] = {
            "samples": n, "metric": metric, "counts": dict(counts),
            "request_success_rate": round(valid / n, 4) if n else None,
            "conditional_score": round(counts["correct"] / valid, 4) if valid else None,
            "end_to_end_score": round(counts["correct"] / n, 4) if n else None,
            "truncation_rate": round(counts["truncated"] / valid, 4) if valid else None,
            "examples": examples,
        }
    return out


def to_markdown(res: dict) -> str:
    lines = ["| model | task | metric | n | request success | conditional score | end-to-end score "
             "| truncation | parse_error | wrong |", "|---|---|---|---|---|---|---|---|---|---|"]
    for model, tasks in res.items():
        for task, r in tasks.items():
            c = r["counts"]
            lines.append(f"| {model} | {task} | {r['metric']} | {r['samples']} | "
                         f"{r['request_success_rate']} | {r['conditional_score']} | {r['end_to_end_score']} | "
                         f"{r['truncation_rate']} | {c.get('parse_error', 0)} | {c.get('wrong_answer', 0)} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    res = analyze(args.run_dir)
    print(to_markdown(res))
    if args.json_out:
        args.json_out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
