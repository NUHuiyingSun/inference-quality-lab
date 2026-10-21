"""Run lm-evaluation-harness against an OpenAI-compatible endpoint for every model in
eval_config.json, logging per-sample outputs so failures can be classified afterwards.

    pip install "lm_eval[api]"
    IQLAB_API_KEY=... python evals/run_evals.py --config evals/eval_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def build_cmd(cfg: dict, model: str, out: Path) -> list[str]:
    ep = cfg["endpoint"]
    model_args = ",".join([
        f"model={model}", f"base_url={ep['base_url']}",
        f"num_concurrent={cfg['num_concurrent']}", "tokenized_requests=False", "max_retries=3",
    ])
    gen = ",".join(f"{k}={v}" for k, v in cfg["gen_kwargs"].items())
    return ["lm_eval", "--model", "local-chat-completions", "--model_args", model_args,
            "--tasks", ",".join(cfg["tasks"]), "--apply_chat_template", "--gen_kwargs", gen,
            "--limit", str(cfg["limit"]), "--seed", str(cfg["seed"]),
            "--log_samples", "--output_path", str(out)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("eval_config.json"))
    ap.add_argument("--results-dir", type=Path, default=Path("results/evals"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    run_dir = args.results_dir / f"{cfg['config_version']}_{time.strftime('%Y%m%d_%H%M%S')}"
    env = {**os.environ, "OPENAI_API_KEY": os.environ.get(cfg["endpoint"]["api_key_env"], "")}
    for model in cfg["models"]:
        out = run_dir / model
        cmd = build_cmd(cfg, model, out)
        print(" ".join(cmd))
        if args.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, out / "eval_config.json")
        subprocess.run(cmd, env=env, check=True)
    print(f"\nclassify with: python evals/classify.py {run_dir}")


if __name__ == "__main__":
    main()
