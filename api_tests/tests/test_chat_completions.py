"""Non-streaming correctness: response schema, usage accounting, max_tokens semantics."""
import pytest

FINISH = {"stop", "length", "tool_calls", "content_filter"}


def _create(client, model, messages, **kw):
    resp = client.post({"model": model, "messages": messages, **kw})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_response_schema(client, model, user_msg):
    body = _create(client, model, user_msg, max_tokens=32)
    assert isinstance(body["id"], str) and body["id"]
    assert body["object"] == "chat.completion"
    assert isinstance(body["created"], int)
    assert isinstance(body["model"], str)
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str) and choice["message"]["content"]
    assert choice["finish_reason"] in FINISH


def test_usage_accounting(client, model, user_msg):
    usage = _create(client, model, user_msg, max_tokens=32)["usage"]
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert isinstance(usage[k], int) and usage[k] >= 0, k
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.parametrize("max_tokens", [1, 2, 5])
def test_max_tokens_is_honored(client, model, max_tokens):
    msgs = [{"role": "user", "content": "Write a long story about a lighthouse keeper."}]
    body = _create(client, model, msgs, max_tokens=max_tokens)
    assert body["usage"]["completion_tokens"] <= max_tokens
    assert body["choices"][0]["finish_reason"] == "length"


def test_system_and_multiturn_messages_accepted(client, model):
    msgs = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Say hi."},
        {"role": "assistant", "content": "Hi."},
        {"role": "user", "content": "Again."},
    ]
    body = _create(client, model, msgs, max_tokens=16)
    assert body["choices"][0]["message"]["content"]


@pytest.mark.parametrize("temperature", [0, 1.0, 2.0])
def test_temperature_boundaries_accepted(client, model, user_msg, temperature):
    _create(client, model, user_msg, max_tokens=8, temperature=temperature)
