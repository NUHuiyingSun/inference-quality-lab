"""Thin HTTP client for OpenAI-compatible chat completions that records every exchange."""
from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from typing import Any

import requests

from .sse import iter_sse_data
from .stream_check import StreamReport, validate_stream


@dataclass
class Exchange:
    method: str
    url: str
    headers: dict[str, str]
    body: Any
    status: int | None = None
    response_text: str = ""
    error: str = ""

    def redacted_headers(self) -> dict[str, str]:
        return {k: ("Bearer ***" if k.lower() == "authorization" else v)
                for k, v in self.headers.items()}

    def to_curl(self) -> str:
        parts = ["curl", "-sS", "-X", self.method, shlex.quote(self.url)]
        for k, v in self.redacted_headers().items():
            if k.lower() == "authorization":
                v = "Bearer $IQLAB_API_KEY"
            parts += ["-H", shlex.quote(f"{k}: {v}")]
        if self.body is not None:
            payload = self.body if isinstance(self.body, str) else json.dumps(self.body)
            parts += ["-d", shlex.quote(payload)]
        return " ".join(parts)


@dataclass
class ChatClient:
    base_url: str
    api_key: str = ""
    timeout: float = 30.0
    history: list[Exchange] = field(default_factory=list)

    def _headers(self, extra: dict[str, str] | None) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            h.update(extra)
        return h

    def post(self, payload: Any, *, headers: dict[str, str] | None = None,
             path: str = "/chat/completions", raw: bool = False,
             timeout: float | None = None) -> requests.Response:
        """POST a JSON payload (or a raw string when raw=True) and record it."""
        url = self.base_url.rstrip("/") + path
        hdrs = self._headers(headers)
        ex = Exchange("POST", url, hdrs, payload)
        self.history.append(ex)
        data = payload if raw else json.dumps(payload)
        try:
            resp = requests.post(url, data=data, headers=hdrs, timeout=timeout or self.timeout)
        except requests.RequestException as exc:
            ex.error = f"{type(exc).__name__}: {exc}"
            raise
        ex.status, ex.response_text = resp.status_code, resp.text[:4000]
        return resp

    def stream(self, payload: dict[str, Any], *, headers: dict[str, str] | None = None,
               require_usage: bool = False, timeout: float | None = None) -> StreamReport:
        """Run a streaming request and validate the event sequence."""
        url = self.base_url.rstrip("/") + "/chat/completions"
        hdrs = self._headers(headers)
        body = {**payload, "stream": True}
        ex = Exchange("POST", url, hdrs, body)
        self.history.append(ex)
        events: list[str] = []
        transport_error = ""
        try:
            with requests.post(url, data=json.dumps(body), headers=hdrs, stream=True,
                               timeout=timeout or self.timeout) as resp:
                ex.status = resp.status_code
                if resp.status_code != 200:
                    ex.response_text = resp.text[:4000]
                    rep = StreamReport(errors=[f"HTTP {resp.status_code}: {resp.text[:200]}"])
                    return rep
                try:
                    events = list(iter_sse_data(resp.iter_lines()))
                except requests.exceptions.ChunkedEncodingError as exc:
                    transport_error = f"transport error mid-stream: {exc}"
        except requests.RequestException as exc:
            ex.error = f"{type(exc).__name__}: {exc}"
            raise
        ex.response_text = "\n".join(events)[:4000]
        rep = validate_stream(events, require_usage=require_usage)
        if transport_error:
            rep.errors.insert(0, transport_error)
        return rep
