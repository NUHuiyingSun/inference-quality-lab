"""Error handling: invalid requests return 4xx with a structured error body, never 5xx."""
import pytest

from iqlab.client import ChatClient

MSG = [{"role": "user", "content": "hi"}]

INVALID_PAYLOADS = [
    ("missing_messages", {"model": "MODEL"}),
    ("empty_messages", {"model": "MODEL", "messages": []}),
    ("messages_not_array", {"model": "MODEL", "messages": "hi"}),
    ("bad_role", {"model": "MODEL", "messages": [{"role": "wizard", "content": "hi"}]}),
    ("content_wrong_type", {"model": "MODEL", "messages": [{"role": "user", "content": 42}]}),
    ("temperature_too_high", {"model": "MODEL", "messages": MSG, "temperature": 5}),
    ("temperature_negative", {"model": "MODEL", "messages": MSG, "temperature": -0.1}),
    ("max_tokens_zero", {"model": "MODEL", "messages": MSG, "max_tokens": 0}),
    ("max_tokens_negative", {"model": "MODEL", "messages": MSG, "max_tokens": -5}),
    ("stream_not_bool", {"model": "MODEL", "messages": MSG, "stream": "yes"}),
]


def _fill(payload, model):
    return {k: (model if v == "MODEL" else v) for k, v in payload.items()}


def _assert_error_body(resp):
    err = resp.json().get("error")
    assert isinstance(err, dict), resp.text
    assert isinstance(err.get("message"), str) and err["message"]


@pytest.mark.parametrize("name,payload", INVALID_PAYLOADS, ids=[n for n, _ in INVALID_PAYLOADS])
def test_invalid_request_returns_4xx(client, model, name, payload):
    resp = client.post(_fill(payload, model))
    assert 400 <= resp.status_code < 500, f"{name}: {resp.status_code} {resp.text}"
    _assert_error_body(resp)


@pytest.mark.parametrize("raw", ["{bad json", "", "[]", "null"])
def test_malformed_body_returns_4xx(client, raw):
    resp = client.post(raw, raw=True)
    assert 400 <= resp.status_code < 500, resp.text


def test_missing_api_key_returns_401(target):
    anon = ChatClient(target.base_url, api_key="")
    resp = anon.post({"model": target.model, "messages": MSG})
    assert resp.status_code == 401
    _assert_error_body(resp)


def test_wrong_api_key_returns_401(target):
    bad = ChatClient(target.base_url, api_key="definitely-wrong")
    resp = bad.post({"model": target.model, "messages": MSG})
    assert resp.status_code == 401


@pytest.mark.live_only
def test_unknown_model_returns_4xx(client):
    resp = client.post({"model": "no-such-model-xyz", "messages": MSG})
    assert 400 <= resp.status_code < 500
    _assert_error_body(resp)


@pytest.mark.mock_only
def test_unknown_route_returns_404(client):
    resp = client.post({}, path="/nope")
    assert resp.status_code == 404
