"""Unit tests for harness logic itself (no server needed)."""
import json

from iqlab.config import capability_skip_reason, enabled_capabilities
from iqlab.stream_check import validate_stream


def _chunk(idx, delta, finish=None, cid="x"):
    return json.dumps({"id": cid, "object": "chat.completion.chunk",
                       "choices": [{"index": idx, "delta": delta, "finish_reason": finish}]})


def test_capabilities_default_and_override():
    assert enabled_capabilities({}) == {"structured_outputs", "tool_calling"}
    assert enabled_capabilities({"IQLAB_CAPABILITIES": "tool_calling"}) == {"tool_calling"}
    assert capability_skip_reason(["structured_outputs"], {"tool_calling"})
    assert capability_skip_reason(["tool_calling"], {"tool_calling"}) is None


def test_two_choices_each_terminate_once_is_valid():
    ev = [_chunk(0, {"role": "assistant"}), _chunk(1, {"role": "assistant"}),
          _chunk(0, {"content": "a"}), _chunk(1, {"content": "b"}),
          _chunk(0, {}, "stop"), _chunk(1, {}, "length"), "[DONE]"]
    rep = validate_stream(ev)
    assert rep.ok, rep.errors
    assert rep.contents == {0: "a", 1: "b"}
    assert rep.finish_reasons == {0: "stop", 1: "length"}


def test_choice_missing_its_finish_is_flagged():
    ev = [_chunk(0, {"role": "assistant"}), _chunk(1, {"role": "assistant"}),
          _chunk(0, {}, "stop"), "[DONE]"]
    rep = validate_stream(ev)
    assert any("choice 1" in e for e in rep.errors), rep.errors


def test_content_after_finish_is_flagged():
    ev = [_chunk(0, {"role": "assistant"}), _chunk(0, {}, "stop"),
          _chunk(0, {"content": "late"}), "[DONE]"]
    rep = validate_stream(ev)
    assert any("before its last delta" in e for e in rep.errors), rep.errors
