"""Target selection from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    name: str          # "mock" or "cerebras" (any OpenAI-compatible endpoint works)
    base_url: str
    api_key: str
    model: str

    @property
    def is_mock(self) -> bool:
        return self.name == "mock"


def target_from_env(mock_base_url: str | None = None, mock_api_key: str = "") -> Target:
    name = os.environ.get("IQLAB_TARGET", "mock").lower()
    if name == "mock":
        return Target("mock", mock_base_url or "http://127.0.0.1:8080/v1", mock_api_key, "mock-model")
    return Target(
        name=name,
        base_url=os.environ.get("IQLAB_BASE_URL", "https://api.cerebras.ai/v1"),
        api_key=os.environ.get("IQLAB_API_KEY") or os.environ.get("CEREBRAS_API_KEY", ""),
        model=os.environ.get("IQLAB_MODEL", "llama3.1-8b"),
    )


DEFAULT_CAPABILITIES = "structured_outputs,tool_calling"


def enabled_capabilities(env: dict | None = None) -> set[str]:
    """Capabilities the selected model supports, from IQLAB_CAPABILITIES (comma-separated).
    Tests for unsupported capabilities are skipped, not failed: a model lacking a feature
    is not a service defect."""
    raw = (env if env is not None else os.environ).get("IQLAB_CAPABILITIES", DEFAULT_CAPABILITIES)
    return {c.strip() for c in raw.split(",") if c.strip()}


def capability_skip_reason(required: list[str], enabled: set[str]) -> str | None:
    missing = [c for c in required if c not in enabled]
    return f"model lacks capability: {', '.join(missing)}" if missing else None
