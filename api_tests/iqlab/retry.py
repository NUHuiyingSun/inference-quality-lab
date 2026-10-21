"""Retry policy: retry 429/5xx/connection errors, honour Retry-After, never retry other 4xx."""
from __future__ import annotations

import time
from typing import Callable

import requests

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def call_with_retry(
    fn: Callable[[], requests.Response],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = fn()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt == max_attempts:
                raise
            sleep(min(max_delay, base_delay * 2 ** (attempt - 1)))
            continue
        if resp.status_code not in RETRYABLE_STATUS or attempt == max_attempts:
            return resp
        sleep(_delay_for(resp, attempt, base_delay, max_delay))
    raise RuntimeError("unreachable") from last_exc


def _delay_for(resp: requests.Response, attempt: int, base: float, cap: float) -> float:
    header = resp.headers.get("Retry-After")
    if header is not None:
        try:
            return min(cap, max(0.0, float(header)))
        except ValueError:
            pass
    return min(cap, base * 2 ** (attempt - 1))
