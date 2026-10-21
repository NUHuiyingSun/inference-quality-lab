"""Write a self-contained failure report: what was sent, what came back, how to reproduce."""
from __future__ import annotations

import json
import platform
import re
import time
from pathlib import Path
from typing import Any


def write_failure_report(out_dir: Path, nodeid: str, failure_text: str,
                         history: list[Any], target: Any) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", nodeid)[-150:]
    last = history[-1] if history else None
    report = {
        "test": nodeid,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target": {"name": target.name, "base_url": target.base_url, "model": target.model},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "failure": failure_text[-3000:],
        "last_request": None if last is None else {
            "method": last.method, "url": last.url,
            "headers": last.redacted_headers(), "body": last.body,
        },
        "last_response": None if last is None else {
            "status": last.status, "body": last.response_text, "transport_error": last.error,
        },
        "repro": None if last is None else last.to_curl(),
        "exchanges_in_test": len(history),
    }
    path = out_dir / f"{safe}.json"
    path.write_text(json.dumps(report, indent=2))
    return path
