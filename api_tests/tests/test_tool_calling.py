"""Tool calling: well-formed tool calls and multi-turn validation (live endpoint only)."""
import json

import pytest

pytestmark = [pytest.mark.live_only, pytest.mark.requires_capability("tool_calling")]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}]
ASK = [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]


def _call(client, model, messages, **kw):
    return client.post({"model": model, "messages": messages, "tools": TOOLS, "max_tokens": 256, **kw})


def test_tool_call_is_well_formed(client, model):
    resp = _call(client, model, ASK, tool_choice="required")
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    calls = choice["message"]["tool_calls"]
    assert calls and choice["finish_reason"] == "tool_calls"
    ids = [c["id"] for c in calls]
    assert len(set(ids)) == len(ids)
    for c in calls:
        assert c["function"]["name"] == "get_weather"
        assert "city" in json.loads(c["function"]["arguments"])


def test_tool_result_round_trip(client, model):
    first = _call(client, model, ASK, tool_choice="required").json()["choices"][0]["message"]
    call = first["tool_calls"][0]
    follow = ASK + [first, {"role": "tool", "tool_call_id": call["id"], "content": '{"temp_c": 18}'}]
    resp = _call(client, model, follow)
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"]


def test_orphan_tool_message_is_rejected(client, model):
    msgs = ASK + [{"role": "tool", "tool_call_id": "call_does_not_exist", "content": "{}"}]
    resp = _call(client, model, msgs)
    assert 400 <= resp.status_code < 500, resp.text
