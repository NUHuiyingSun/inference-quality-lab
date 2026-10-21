# inference-quality-lab

Quality engineering for OpenAI-compatible LLM inference APIs (built against Cerebras
Inference Cloud). Four parts, one theme: *is this response correct, and fast enough to ship?*

| Part | Focus | Path |
|---|---|---|
| API contract test suite | correctness, streaming (SSE), errors, structured outputs, tool calling, SDK compatibility | `api_tests/` |
| C++ fault-injection server | controlled failures to test client resilience; mutation testing of the suite | `mock_server/` |
| Latency & load benchmark | TTFT, ITL, P95/P99, throughput across concurrency; regression gate | `bench/` |
| Accuracy eval harness | lm-evaluation-harness runs + failure classification | `evals/` |

## Quick start

```bash
pip install pytest requests
make build            # C++17 mock server (CMake)
make test             # full suite against the mock server
make mutation         # inject 14 server defects, check the suite catches each one

export IQLAB_API_KEY=...            # Cerebras key
make test-live                      # same suite against the real API (mock-only tests skip)
```

## Part 1: API contract tests (`api_tests/`)

57 test cases (with parametrization): 31 contract tests that run against both the mock and the live API, 12 mock-only resilience tests, 10 live-only tests, and 4 unit tests of the harness itself.

- `test_chat_completions.py`: response schema, usage arithmetic, `max_tokens` → `finish_reason=length`, multi-turn, temperature bounds
- `test_streaming.py`: SSE well-formedness, each choice starts with `role=assistant` and ends with exactly one `finish_reason` on its last delta, stable chunk ids, `[DONE]` termination, usage in stream. Usage checks verify field types and internal consistency (total = prompt + completion), not tokenizer-exact counts.
- `test_errors.py`: 10 invalid payloads + 4 malformed bodies must return 4xx with an `error` object (never 5xx); 401 on missing/wrong key
- `test_fault_handling.py` *(mock only)*: retry honors `Retry-After`, exponential backoff, no retry on 4xx, timeouts, dropped streams, malformed chunks, missing `[DONE]`, 32 concurrent streams stay isolated (each request carries an `X-Mock-Echo` marker; every stream must contain its own marker and no other)
- `test_structured_outputs.py`, `test_tool_calling.py` *(live only; skipped unless the capability is listed in `IQLAB_CAPABILITIES`, since a model lacking a feature is not a service defect)*: strict JSON schema, schema without `additionalProperties:false` rejected, tool-call shape/unique ids, tool round trip, orphan tool message rejected
- `test_sdk_compat.py` *(live only)*: Cerebras SDK vs OpenAI SDK: response structure, stream parsing, and exception mapping (not identical generated text)

Every failing test writes `reports/failures/<test>.json`: request, response, target, environment, and a copy-pasteable `curl` repro (API key redacted). CI uploads them as artifacts.

## Part 2: C++ mock server (`mock_server/`)

OpenAI-compatible `/v1/chat/completions` (streaming and non-streaming) in C++17 with POSIX sockets, one thread per connection, a small recursive-descent JSON parser, and request validation. Two different techniques use it:

- **Fault injection (runtime):** the server misbehaves on request, to test whether *clients* handle failures correctly.
- **Mutation testing (build time):** the server is rebuilt with a seeded bug, to test whether the *test suite* notices.

Faults are injected per request with the `X-Mock-Fault` header (`X-Mock-Echo: <marker>` tags a response for isolation tests):

```
status=429|500|503   retry_after=N   latency_ms=N   chunk_delay_ms=N
drop_after=N         malformed_chunk=N              no_done
```

Built with `-Wall -Wextra -Wpedantic -Werror`; CI also runs the whole suite against an AddressSanitizer + UBSan build (memory errors and undefined behavior; this does not check for data races, which would need ThreadSanitizer).

**Mutation testing** (`mock_server/mutation/run_mutations.py`) rebuilds the server with one realistic defect at a time (wrong usage totals, missing role, ignored `max_tokens`, duplicate `[DONE]`, unstable chunk ids, auth bypass, 500 on bad JSON, dropped `Retry-After`, truncated stream, cross-request content leakage, …) and checks the suite fails. Current score: **all 14 seeded defects detected**. This shows the suite catches these specific defects, not that it is complete.

## Part 3: Benchmark (`bench/`)

```bash
python bench/bench.py --base-url https://api.cerebras.ai/v1 --model <model> \
  --concurrency 1,4,16 --requests 64 --out results/current.json
python bench/compare.py results/baseline.json results/current.json   # exit 1 on regression
```

Each level runs a warm-up (discarded), then `--repeats` independent trials (default 3) of at least `--min-samples-per-worker` × concurrency requests (default 8×), and reports the **median** of each metric plus `ttft_p95_spread` across trials. Thresholds: +20% on TTFT p95 / ITL mean / E2E p95, −15% throughput, +1pt error rate. Every result file stores the workload config, git SHA, and timestamp. `locustfile.py` reports TTFT and full-stream time as separate Locust request types; `run_guidellm.sh` cross-checks with GuideLLM.

**False-alarm check** (`bench/experiments/noise_check.py`): runs the same mock workload N times in the original single-trial mode and in the repeated-trial mode, and counts how often the gate flags a regression between identical runs.

- Observation that motivated it: two single-trial runs (64 requests each, no warm-up, concurrency 32, mock server) gave p95 TTFT of 27.14 ms and 32.17 ms, a relative difference of 18.5%.
- First experiment, on a **1-CPU** sandbox, 6 runs per mode: inconclusive. The repeated-trial mode still produced 1–2 false alarms in 5 comparisons; at concurrency 8 both modes varied by roughly 50% between runs. With 32 client threads and the server sharing one core, host scheduling noise likely dominates, and more samples cannot remove that.
- **TODO:** re-run on a multi-core machine and record results here before drawing any conclusion.

## Part 4: Accuracy evals (`evals/`)

`run_evals.py` drives `lm_eval --model local-chat-completions` for each model in the versioned `eval_config.json` (GSM8K and IFEval; generative tasks, since chat-completions endpoints don't support loglikelihood tasks), with `--log_samples`. `classify.py` buckets each failed sample as `api_error`, `parse_error`, `truncated`, or `wrong_answer`, and reports call reliability separately from model quality: **request success rate** (valid responses / requests), **conditional score** (task metric on valid responses), **end-to-end score** (failed calls scored 0), and **truncation rate** on its own, since truncation usually points to a generation-limit or config issue rather than a service failure. Each task uses its own standard metric (GSM8K: `exact_match` with the strict-match filter; IFEval: `prompt_level_strict_acc`).

## Verification status

| Item | Status |
|---|---|
| C++ server builds clean (`-Werror`); suite passes on ASan/UBSan build | ✅ verified (sandbox) |
| 47 mock-target tests pass (31 contract + 12 resilience + 4 unit) | ✅ verified with a pytest stand-in; confirm with real pytest |
| Mutation testing: 14/14 seeded defects detected | ✅ verified (sandbox) |
| Failure reports + curl repro | ✅ verified (sandbox) |
| Benchmark gate flags a deliberately slowed server | ✅ verified (mock) |
| Repeated trials reduce false alarms | ❓ inconclusive on 1 CPU; re-run locally |
| Classifier logic (synthetic fixture) | ✅ verified |
| Capability-based skipping in pytest collection | ⏳ needs real pytest |
| 10 live tests (structured outputs, tool calling, SDK compat, unknown model) | ⏳ needs API key |
| Real latency numbers; Locust and GuideLLM runs | ⏳ needs API key |
| Real lm-eval runs; confirm `--log_samples` field names for installed version | ⏳ needs API key |
| Model names in `eval_config.json` | ⏳ check current Cerebras model list |

## Known limitations

The mock server's "tokens" are whitespace words, its JSON parser covers request validation only, and it uses one thread per connection (fine for tests, not for production load).
