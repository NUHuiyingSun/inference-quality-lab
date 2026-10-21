"""Locust load scenario for streaming chat completions.

Reports two custom request types so TTFT and full-stream time show up separately in the
Locust UI/CSV:  `TTFT /chat/completions` and `STREAM /chat/completions`.

    IQLAB_MODEL=llama3.1-8b IQLAB_API_KEY=... \
      locust -f bench/locustfile.py --host https://api.cerebras.ai/v1 \
             --headless -u 32 -r 4 -t 2m --csv results/locust
"""
import json
import os
import time

from locust import HttpUser, between, task

MODEL = os.environ.get("IQLAB_MODEL", "llama3.1-8b")
KEY = os.environ.get("IQLAB_API_KEY", "")


class StreamingChatUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def stream_chat(self):
        body = {"model": MODEL, "stream": True, "max_tokens": 128,
                "messages": [{"role": "user", "content": "Explain KV caching in three sentences."}]}
        headers = {"Authorization": f"Bearer {KEY}"} if KEY else {}
        start = time.perf_counter()
        first = None
        done = False
        with self.client.post("/chat/completions", json=body, headers=headers, stream=True,
                              catch_response=True, name="/chat/completions") as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            for line in resp.iter_lines():
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                if first is None and json.loads(data)["choices"][0]["delta"].get("content"):
                    first = time.perf_counter()
            if not done:
                resp.failure("stream ended without [DONE]")
                return
            resp.success()
        end = time.perf_counter()
        fire = self.environment.events.request.fire
        if first is not None:
            fire(request_type="TTFT", name="/chat/completions", response_time=(first - start) * 1000,
                 response_length=0, exception=None, context={})
        fire(request_type="STREAM", name="/chat/completions", response_time=(end - start) * 1000,
             response_length=0, exception=None, context={})
