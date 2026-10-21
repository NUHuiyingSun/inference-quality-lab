"""Mutation testing: inject one defect at a time into the server, rebuild, and check
that the API test suite fails. Measures how many realistic server bugs the suite catches."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "mock_server/src/main.cpp"
TEST_CMD = os.environ.get("TEST_CMD", "python -m pytest -q -x api_tests").split()

MUTATIONS = {
    "usage_total_off_by_one": (
        'std::to_string(prompt + completion)', 'std::to_string(prompt + completion + 1)'),
    "first_chunk_missing_role": (
        '"{\\"role\\":\\"assistant\\",\\"content\\":\\"\\"}"', '"{\\"content\\":\\"\\"}"'),
    "finish_reason_always_stop": (
        '(limit < natural) ? "length" : "stop"', '"stop"'),
    "max_tokens_ignored": (
        'std::min(limit, natural)', 'natural'),
    "duplicate_done_event": (
        'if (!faults.no_done) emit("[DONE]");', 'if (!faults.no_done) { emit("[DONE]"); emit("[DONE]"); }'),
    "temperature_unbounded": (
        't->num < 0.0 || t->num > 2.0', 'false'),
    "chunk_ids_not_stable": (
        'const std::string& id, long created', 'std::string id, long created'),
    "auth_check_disabled": (
        '!api_key.empty() && req.headers["authorization"]', 'false && req.headers["authorization"]'),
    "invalid_json_returns_500": (
        'send_error(fd, 400, "invalid_request_error", "Invalid JSON body: " + parse_err);',
        'send_error(fd, 500, "server_error", "Invalid JSON body: " + parse_err);'),
    "final_chunk_missing_usage": (
        'emit(chunk_json(id, created, model, "{}", finish, usage_json(prompt_tokens, n_out)))',
        'emit(chunk_json(id, created, model, "{}", finish, ""))'),
    "retry_after_header_dropped": (
        '{"Retry-After: " + std::to_string(faults.retry_after)}', '{}'),
    "empty_messages_accepted": (
        'if (messages->arr.empty()) return', 'if (false) return'),
    "stream_content_truncated": (
        'for (long i = 0; i < n_out; ++i) {\n    if (faults.drop_after', 'for (long i = 0; i + 1 < n_out; ++i) {\n    if (faults.drop_after'),
    "stream_marker_crosstalk": (
        'const std::string marker = echo_marker(req.headers);',
        'static std::string prev; const std::string marker = prev; prev = echo_marker(req.headers);'),
}

# chunk_ids_not_stable needs an extra edit so the id actually changes per chunk.
EXTRA = {"chunk_ids_not_stable": ('  std::string s = "{\\"id\\":" + json::quote(id)',
                                  '  static std::atomic<int> n{0}; id += "-" + std::to_string(++n);\n'
                                  '  std::string s = "{\\"id\\":" + json::quote(id)')}


def main() -> int:
    original = SRC.read_text()
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, (old, new) in MUTATIONS.items():
            assert original.count(old) == 1, f"{name}: anchor not unique/found"
            mutated = original.replace(old, new)
            if name in EXTRA:
                o, n = EXTRA[name]
                assert mutated.count(o) == 1, f"{name}: extra anchor"
                mutated = mutated.replace(o, n)
            src = Path(tmp) / "src"
            src.mkdir(exist_ok=True)
            (src / "main.cpp").write_text(mutated)
            (src / "json.hpp").write_text((SRC.parent / "json.hpp").read_text())
            binary = Path(tmp) / f"server_{name}"
            build = subprocess.run(["g++", "-std=c++17", "-O1", "-pthread", str(src / "main.cpp"),
                                    "-o", str(binary)], capture_output=True, text=True)
            if build.returncode != 0:
                results[name] = "BUILD_ERROR"
                print(build.stderr[:800])
                continue
            run = subprocess.run(TEST_CMD, cwd=ROOT, capture_output=True, text=True,
                                 env={**os.environ, "MOCK_SERVER_BIN": str(binary),
                                      "IQLAB_REPORT_DIR": str(Path(tmp) / "reports")})
            results[name] = "KILLED" if run.returncode != 0 else "SURVIVED"
            print(f"{name:32s} {results[name]}", flush=True)
    killed = sum(v == "KILLED" for v in results.values())
    total = sum(v in ("KILLED", "SURVIVED") for v in results.values())
    print(f"\nmutation score: {killed}/{total} defects detected")
    return 0 if killed == total else 1


if __name__ == "__main__":
    sys.exit(main())
