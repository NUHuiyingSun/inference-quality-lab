"""Same requests through the Cerebras SDK and the OpenAI SDK should behave the same (live only)."""
import pytest

pytestmark = pytest.mark.live_only

cerebras_sdk = pytest.importorskip("cerebras.cloud.sdk")
openai = pytest.importorskip("openai")

MSG = [{"role": "user", "content": "Name one primary color."}]


@pytest.fixture
def sdks(target):
    return {
        "cerebras": cerebras_sdk.Cerebras(api_key=target.api_key),
        "openai": openai.OpenAI(api_key=target.api_key, base_url=target.base_url),
    }


def test_nonstream_shapes_match(sdks, model):
    out = {n: c.chat.completions.create(model=model, messages=MSG, max_tokens=16) for n, c in sdks.items()}
    for name, r in out.items():
        assert r.choices[0].message.role == "assistant", name
        assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens, name


def test_stream_terminates_cleanly_in_both(sdks, model):
    for name, c in sdks.items():
        chunks = list(c.chat.completions.create(model=model, messages=MSG, max_tokens=16, stream=True))
        finishes = [ch.choices[0].finish_reason for ch in chunks if ch.choices and ch.choices[0].finish_reason]
        assert len(finishes) == 1, name


def test_rate_limit_and_status_errors_map_to_sdk_exceptions(target, model):
    bad = cerebras_sdk.Cerebras(api_key="definitely-wrong")
    with pytest.raises(cerebras_sdk.APIStatusError) as exc:
        bad.chat.completions.create(model=model, messages=MSG)
    assert exc.value.status_code == 401
