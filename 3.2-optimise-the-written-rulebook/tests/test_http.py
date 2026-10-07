"""The real OpenAI-compatible transport against a local server: retries, fatal errors, empty replies."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("openai")

from rulebook_opt import llm as llm_mod  # noqa: E402
from rulebook_opt.config import ModelConfig  # noqa: E402
from rulebook_opt.llm import LLMClient, LLMError, LLMFatalError  # noqa: E402


class Server:
    def __init__(self):
        self.hits = 0
        self.plan: list[int] = []  # status codes to return in order; then 200
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                outer.hits += 1
                status = outer.plan.pop(0) if outer.plan else 200
                body = json.dumps(
                    {"choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}]}
                    if status == 200
                    else {"error": {"message": "nope"}}
                ).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    s = Server()
    yield s
    s.httpd.shutdown()


def client(url, retries=4):
    return LLMClient(ModelConfig(model="m", provider="custom", base_url=url), retries=retries)


MSG = [{"role": "user", "content": "ping"}]


def test_ok(server):
    assert client(server.url).complete(MSG) == "pong"


def test_retries_rate_limit_then_succeeds(server):
    server.plan = [429, 500]
    assert client(server.url).complete(MSG) == "pong" and server.hits == 3


def test_auth_error_is_fatal_and_not_retried(server):
    server.plan = [401]
    with pytest.raises(LLMFatalError):
        client(server.url).complete(MSG)
    assert server.hits == 1


def test_gives_up_after_retries(server):
    server.plan = [500] * 10
    with pytest.raises(LLMError):
        client(server.url, retries=3).complete(MSG)
    assert server.hits == 3


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        ModelConfig(model="x/y", provider="openrouter").resolve()
