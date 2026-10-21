"""Validate an OpenAI-style chat completion stream against protocol invariants."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter"}


@dataclass
class StreamReport:
    raw_events: list[str] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    content: str = ""
    finish_reason: str | None = None            # choice 0
    contents: dict[int, str] = field(default_factory=dict)
    finish_reasons: dict[int, str] = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    done_seen: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_stream(events: list[str], *, require_usage: bool = False) -> StreamReport:
    rep = StreamReport(raw_events=list(events))
    for idx, data in enumerate(events):
        if rep.done_seen:
            rep.errors.append(f"event {idx} received after [DONE]")
            break
        if data == "[DONE]":
            rep.done_seen = True
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            rep.errors.append(f"event {idx} is not valid JSON: {exc.msg}")
            continue
        rep.chunks.append(chunk)

    if not rep.done_seen:
        rep.errors.append("stream ended without [DONE]")
    if not rep.chunks:
        rep.errors.append("stream contained no chunks")
        return rep

    ids = {c.get("id") for c in rep.chunks}
    if len(ids) != 1:
        rep.errors.append(f"chunk ids are inconsistent: {sorted(map(str, ids))}")
    for i, c in enumerate(rep.chunks):
        if c.get("object") != "chat.completion.chunk":
            rep.errors.append(f"chunk {i} has object={c.get('object')!r}")

    # Group choice deltas by choice index: each choice must start with role=assistant and
    # end with exactly one finish_reason, carried by that choice's last delta.
    per_choice: dict[int, list[dict]] = {}
    for c in rep.chunks:
        for ch in c.get("choices") or []:
            per_choice.setdefault(ch.get("index", 0), []).append(ch)
    if not per_choice:
        rep.errors.append("stream contained no choice deltas")
    for idx, deltas in sorted(per_choice.items()):
        if deltas[0].get("delta", {}).get("role") != "assistant":
            rep.errors.append(f"choice {idx}: first delta does not carry role='assistant'")
        text = "".join(d.get("delta", {}).get("content") or "" for d in deltas)
        rep.contents[idx] = text
        finishes = [(i, d["finish_reason"]) for i, d in enumerate(deltas)
                    if d.get("finish_reason") is not None]
        if len(finishes) != 1:
            rep.errors.append(f"choice {idx}: expected exactly one finish_reason, got {len(finishes)}")
            continue
        pos, reason = finishes[0]
        if reason not in FINISH_REASONS:
            rep.errors.append(f"choice {idx}: unknown finish_reason {reason!r}")
        if pos != len(deltas) - 1:
            rep.errors.append(f"choice {idx}: finish_reason appeared before its last delta")
        rep.finish_reasons[idx] = reason
    rep.content = rep.contents.get(0, "")
    rep.finish_reason = rep.finish_reasons.get(0)

    usage_chunks = [c["usage"] for c in rep.chunks if c.get("usage")]
    if usage_chunks:
        rep.usage = usage_chunks[-1]
        u = rep.usage
        if u.get("total_tokens") != u.get("prompt_tokens", 0) + u.get("completion_tokens", 0):
            rep.errors.append(f"usage totals do not add up: {u}")
    elif require_usage:
        rep.errors.append("no usage block in stream")
    return rep
