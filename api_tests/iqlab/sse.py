"""Server-Sent Events parsing (https://html.spec.whatwg.org/multipage/server-sent-events.html)."""
from __future__ import annotations

from typing import Iterable, Iterator


def iter_sse_data(lines: Iterable[bytes | str]) -> Iterator[str]:
    """Yield the `data` payload of each SSE event.

    Handles comment lines (":"), multi-line data fields, optional space after
    the colon, and a final event that is not followed by a blank line.
    """
    buf: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r")
        if line == "":
            if buf:
                yield "\n".join(buf)
                buf = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            buf.append(value)
    if buf:
        yield "\n".join(buf)
