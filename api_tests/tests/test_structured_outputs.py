"""Structured outputs with strict JSON schema (live endpoint only)."""
import json

import pytest

pytestmark = [pytest.mark.live_only, pytest.mark.requires_capability("structured_outputs")]

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "population_millions": {"type": "number"},
        "landmarks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["city", "population_millions", "landmarks"],
    "additionalProperties": False,
}


def _req(model, schema, strict=True):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Describe Tokyo as JSON."}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "city", "strict": strict, "schema": schema}},
        "max_tokens": 256,
    }


def test_strict_output_parses_and_matches_schema(client, model):
    resp = client.post(_req(model, SCHEMA))
    assert resp.status_code == 200, resp.text
    data = json.loads(resp.json()["choices"][0]["message"]["content"])
    assert set(data) == {"city", "population_millions", "landmarks"}
    assert isinstance(data["city"], str)
    assert isinstance(data["population_millions"], (int, float))
    assert all(isinstance(x, str) for x in data["landmarks"])


def test_strict_mode_rejects_schema_without_additional_properties_false(client, model):
    loose = {k: v for k, v in SCHEMA.items() if k != "additionalProperties"}
    resp = client.post(_req(model, loose))
    assert 400 <= resp.status_code < 500, resp.text


def test_streamed_structured_output_concatenates_to_valid_json(client, model):
    rep = client.stream(_req(model, SCHEMA))
    assert rep.ok, rep.errors
    json.loads(rep.content)
