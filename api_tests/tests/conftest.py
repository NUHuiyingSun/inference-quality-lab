from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

from iqlab.client import ChatClient
from iqlab.config import capability_skip_reason, enabled_capabilities, target_from_env
from iqlab.report import write_failure_report

ROOT = Path(__file__).resolve().parents[2]
MOCK_BIN = Path(os.environ.get("MOCK_SERVER_BIN", ROOT / "mock_server/build/mock_server"))
MOCK_KEY = "test-key"
REPORT_DIR = Path(os.environ.get("IQLAB_REPORT_DIR", ROOT / "api_tests/reports/failures"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def target():
    if os.environ.get("IQLAB_TARGET", "mock").lower() != "mock":
        t = target_from_env()
        if not t.api_key:
            pytest.skip("live target selected but no API key set")
        yield t
        return
    if not MOCK_BIN.exists():
        pytest.exit(f"mock server binary not found at {MOCK_BIN}; build mock_server first")
    port = _free_port()
    proc = subprocess.Popen([str(MOCK_BIN), str(port)], env={**os.environ, "MOCK_API_KEY": MOCK_KEY},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            requests.get(base + "/health", timeout=0.2)
            break
        except requests.RequestException:
            time.sleep(0.05)
    yield target_from_env(mock_base_url=base + "/v1", mock_api_key=MOCK_KEY)
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture
def client(target):
    return ChatClient(target.base_url, target.api_key, timeout=30)


@pytest.fixture
def model(target):
    return target.model


@pytest.fixture
def user_msg():
    return [{"role": "user", "content": "In one sentence, why does low latency matter for chat apps?"}]


def pytest_collection_modifyitems(config, items):
    is_mock = os.environ.get("IQLAB_TARGET", "mock").lower() == "mock"
    for item in items:
        if is_mock and "live_only" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="live endpoint only"))
        if not is_mock and "mock_only" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="mock server only"))
        for m in item.iter_markers("requires_capability"):
            reason = capability_skip_reason(list(m.args), enabled_capabilities())
            if reason:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" and rep.failed and "client" in item.funcargs:
        path = write_failure_report(REPORT_DIR, item.nodeid, rep.longreprtext,
                                    item.funcargs["client"].history, item.funcargs["target"])
        rep.sections.append(("failure report", str(path)))
