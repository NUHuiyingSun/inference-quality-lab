"""Streaming (SSE) protocol: chunk shape, ordering, termination, usage."""
import pytest


def _payload(model, messages, **kw):
    # stream_options is the OpenAI-compatible way to ask for usage in the stream.
    return {"model": model, "messages": messages,
            "stream_options": {"include_usage": True}, **kw}


def test_stream_is_well_formed(client, model, user_msg):
    rep = client.stream(_payload(model, user_msg, max_tokens=32), require_usage=True)
    assert rep.ok, rep.errors
    assert rep.content


def test_stream_terminates_with_done(client, model, user_msg):
    rep = client.stream(_payload(model, user_msg, max_tokens=16))
    assert rep.done_seen
    assert rep.raw_events[-1] == "[DONE]"


def test_first_chunk_carries_assistant_role(client, model, user_msg):
    rep = client.stream(_payload(model, user_msg, max_tokens=16))
    first = next(c for c in rep.chunks if c.get("choices"))
    assert first["choices"][0]["delta"].get("role") == "assistant"


@pytest.mark.parametrize("max_tokens", [1, 3])
def test_stream_respects_max_tokens(client, model, max_tokens):
    msgs = [{"role": "user", "content": "Count slowly from one to one hundred."}]
    rep = client.stream(_payload(model, msgs, max_tokens=max_tokens), require_usage=True)
    assert rep.ok, rep.errors
    assert rep.finish_reason == "length"
    assert rep.usage["completion_tokens"] <= max_tokens


def test_stream_usage_matches_nonstream_shape(client, model, user_msg):
    rep = client.stream(_payload(model, user_msg, max_tokens=16), require_usage=True)
    assert set(rep.usage) >= {"prompt_tokens", "completion_tokens", "total_tokens"}


@pytest.mark.mock_only
def test_stream_and_nonstream_content_agree(client, model, user_msg):
    # Deterministic on the mock; on live endpoints sampling makes this flaky by design.
    rep = client.stream(_payload(model, user_msg, max_tokens=8))
    body = client.post({"model": model, "messages": user_msg, "max_tokens": 8}).json()
    assert rep.content == body["choices"][0]["message"]["content"]
