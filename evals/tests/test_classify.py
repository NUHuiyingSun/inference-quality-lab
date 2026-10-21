"""Unit tests for the failure classifier, using a small synthetic fixture."""
from pathlib import Path

import classify

FIXTURE = Path(__file__).parent / "fixtures" / "run"


def test_each_failure_lands_in_the_right_bucket():
    r = classify.analyze(FIXTURE)["model-a"]["gsm8k"]
    assert r["counts"] == {"correct": 1, "api_error": 1, "parse_error": 1,
                           "truncated": 1, "wrong_answer": 1}


def test_only_primary_filter_is_counted():
    assert classify.analyze(FIXTURE)["model-a"]["gsm8k"]["samples"] == 5


def test_reliability_and_quality_are_reported_separately():
    r = classify.analyze(FIXTURE)["model-a"]["gsm8k"]
    assert r["request_success_rate"] == 0.8      # 4 of 5 calls returned something
    assert r["conditional_score"] == 0.25        # 1 correct of 4 valid responses
    assert r["end_to_end_score"] == 0.2          # 1 correct of 5 requests
    assert r["truncation_rate"] == 0.25          # truncation counted apart from API errors


def test_task_name_parsing():
    assert classify.task_from_filename(Path("samples_gsm8k_2026-10-20T00-00-00.jsonl")) == "gsm8k"
    assert classify.task_from_filename(Path("samples_ifeval_x.jsonl")) == "ifeval"
