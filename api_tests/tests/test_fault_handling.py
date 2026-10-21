"""Client resilience under injected faults (mock server only)."""
import pytest
import requests

from iqlab.retry import call_with_retry

pytestmark = pytest.mark.mock_only


def _req(model, msgs, **kw):
    return {"model": model, "messages": msgs, "max_tokens": 8, **kw}


def _fault(spec):
    return {"X-Mock-Fault": spec}


def test_retry_honors_retry_after(client, model, user_msg):
    sleeps = []
    resp = call_with_retry(lambda: client.post(_req(model, user_msg), headers=_fault("status=429,retry_after=2")),
                           max_attempts=3, sleep=sleeps.append)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "2"
    assert sleeps == [2.0, 2.0]


def test_retry_recovers_after_transient_error(client, model, user_msg):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        hdr = _fault("status=503") if calls["n"] == 1 else None
        return client.post(_req(model, user_msg), headers=hdr)

    resp = call_with_retry(flaky, max_attempts=3, sleep=lambda s: None)
    assert resp.status_code == 200
    assert calls["n"] == 2


def test_retry_backs_off_exponentially_without_retry_after(client, model, user_msg):
    sleeps = []
    call_with_retry(lambda: client.post(_req(model, user_msg), headers=_fault("status=500")),
                    max_attempts=4, base_delay=0.5, sleep=sleeps.append)
    assert sleeps == [0.5, 1.0, 2.0]


def test_client_errors_are_not_retried(client, model):
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        return client.post({"model": model, "messages": []})

    resp = call_with_retry(bad, max_attempts=3, sleep=lambda s: None)
    assert resp.status_code == 400
    assert calls["n"] == 1


def test_slow_first_byte_raises_timeout(client, model, user_msg):
    with pytest.raises(requests.Timeout):
        client.post(_req(model, user_msg), headers=_fault("latency_ms=1500"), timeout=0.5)


def test_dropped_stream_is_detected(client, model, user_msg):
    rep = client.stream(_req(model, user_msg), headers=_fault("drop_after=2"))
    assert not rep.ok
    assert not rep.done_seen
    assert "stream ended without [DONE]" in rep.errors


def test_malformed_chunk_is_detected(client, model, user_msg):
    rep = client.stream(_req(model, user_msg), headers=_fault("malformed_chunk=1"))
    assert any("not valid JSON" in e for e in rep.errors), rep.errors


def test_missing_done_is_detected(client, model, user_msg):
    rep = client.stream(_req(model, user_msg), headers=_fault("no_done"))
    assert rep.errors == ["stream ended without [DONE]"]


def test_slow_chunks_still_produce_valid_stream(client, model, user_msg):
    rep = client.stream(_req(model, user_msg), headers=_fault("chunk_delay_ms=30"))
    assert rep.ok, rep.errors


def test_concurrent_streams_are_isolated(client, model):
    """32 concurrent streams, each tagged with its own marker: every stream must be complete
    and contain only its own marker (no cross-request content leakage or id collision)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        msgs = [{"role": "user", "content": f"request {i}"}]
        hdrs = {**_fault("chunk_delay_ms=10"), "X-Mock-Echo": f"req-{i}"}
        return i, client.stream(_req(model, msgs, max_tokens=4), headers=hdrs)

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(one, range(32)))
    for i, rep in results:
        assert rep.ok, (i, rep.errors)
        assert rep.content.startswith(f"[req-{i}]"), (i, rep.content)
        others = [f"[req-{j}]" for j in range(32) if j != i]
        assert not any(o in rep.content for o in others), (i, rep.content)
    ids = [rep.chunks[0]["id"] for _, rep in results]
    assert len(set(ids)) == len(ids), "stream ids collided across concurrent requests"
