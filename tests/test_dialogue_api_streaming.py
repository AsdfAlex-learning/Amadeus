"""Dialogue API client tests (ADR 0005, replaces C-7 fake-streaming path).

Hermetic: HTTP goes through the ``_open_stream`` seam with fake connections.
The one exception is the loopback cancel test, which binds 127.0.0.1 only to
verify the real socket-close unblock mechanics (Oracle review F-1).
"""

import json
import threading
import time

import pytest

import src.dialogue.model as model_module
from src.dialogue.model import DialogueModel, DialogueUnavailableError


def make_config(**overrides):
    cfg = {
        "provider": "openai",
        "api_base": "https://example.invalid/v1",
        "model": "test-model",
        "api_key_env": "AMADEUS_TEST_KEY",
        "request_timeout_seconds": 5,
        "max_new_tokens": 64,
        "temperature": 0.5,
        "top_p": 0.8,
        "repetition_penalty": 1.2,
    }
    cfg.update(overrides)
    return {"dialogue": cfg}


class FakeConn:
    def __init__(self, on_close=None):
        self.sock = self
        self.closed = False
        self._on_close = on_close

    def close(self):
        if not self.closed:
            self.closed = True
            if self._on_close:
                self._on_close()


class FakeResp:
    """Scripted bytes lines for ``readline()``; ``("block", ev)`` waits for
    ``ev`` or close; ``("sleep", s)`` delays; ``("error", exc)`` raises.
    Close of the paired conn aborts blocking reads. EOF returns ``b""``."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0
        self.lines_read = 0
        self.closed = threading.Event()

    def readline(self):
        while True:
            if self._i >= len(self._script):
                return b""
            item = self._script[self._i]
            self._i += 1
            if isinstance(item, bytes):
                self.lines_read += 1
                return item
            kind = item[0]
            if kind == "block":
                ev = item[1]
                while not ev.wait(0.01):
                    if self.closed.is_set():
                        raise OSError("connection closed by client")
                continue
            if kind == "sleep":
                time.sleep(item[1])
                continue
            if kind == "error":
                raise item[1]


def sse_line(content):
    return b"data: " + json.dumps({"choices": [{"delta": {"content": content}}]}).encode() + b"\n"


DONE = b"data: [DONE]\n"


class StreamHarness:
    def __init__(self, script, config=None):
        self.resp = FakeResp(script)
        self.conn = FakeConn(on_close=self.resp.closed.set)
        self.calls = []
        self._config = config or make_config()

    def install(self, monkeypatch):
        def fake_open_stream(url, headers, body, timeout):
            self.calls.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
            return self.resp, self.conn

        monkeypatch.setattr(model_module, "_open_stream", fake_open_stream)
        return self

    @property
    def body(self):
        return json.loads(self.calls[0]["body"])

    @property
    def url(self):
        return self.calls[0]["url"]

    @property
    def headers(self):
        return self.calls[0]["headers"]

    def model(self):
        return DialogueModel(self._config)


def collect(gen, limit=100, timeout=2.0):
    chunks = []
    deadline = time.monotonic() + timeout
    it = iter(gen)
    while len(chunks) < limit:
        if time.monotonic() > deadline:
            raise AssertionError(f"generator stalled after {chunks!r}")
        try:
            chunks.append(next(it))
        except StopIteration:
            break
    return chunks


# ---------------------------------------------------------------- unconfigured


def test_unconfigured_load_and_generate_raise():
    dm = DialogueModel({"dialogue": {}})
    assert dm.load() is False
    assert dm.is_loaded is False
    with pytest.raises(DialogueUnavailableError) as exc:
        next(dm.generate([{"role": "user", "content": "hi"}]))
    assert "未配置" in str(exc.value)


def test_unconfigured_missing_model_key_only():
    dm = DialogueModel(make_config(model=""))
    assert dm.load() is False
    with pytest.raises(DialogueUnavailableError):
        next(dm.generate([]))


def test_configured_load_succeeds():
    dm = DialogueModel(make_config())
    assert dm.load() is True
    assert dm.is_loaded is True


# ------------------------------------------------------------- openai protocol


def test_openai_happy_path_joins_deltas(monkeypatch):
    h = StreamHarness([sse_line("Hello"), sse_line(" world"), DONE, b"\n"]).install(monkeypatch)
    chunks = collect(h.model().generate([{"role": "user", "content": "hi"}]))
    assert "".join(chunks) == "Hello world"


def test_openai_crlf_and_done_crlf(monkeypatch):
    h = StreamHarness(
        [
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\r\n',
            b"\r\n",
            b"data: [DONE]\r\n",
            b"\r\n",
        ]
    ).install(monkeypatch)
    assert "".join(collect(h.model().generate([]))) == "ok"


def test_openai_multiline_data_joined(monkeypatch):
    script = [
        b'data: {"choices":[{"delta":{"content":\n',
        b'data: "hi"}}]}\n',
        b"\n",
        DONE,
    ]
    h = StreamHarness(script).install(monkeypatch)
    assert "".join(collect(h.model().generate([]))) == "hi"


def test_openai_comment_and_heartbeat_lines_ignored(monkeypatch):
    h = StreamHarness([b": keep-alive\n", b"\n", sse_line("hey"), b"\n", DONE]).install(monkeypatch)
    assert "".join(collect(h.model().generate([]))) == "hey"


def test_openai_empty_choices_and_null_content_skipped(monkeypatch):
    script = [
        b'data: {"choices": []}\n',
        b"\n",
        b'data: {"choices":[{"delta":{"content": null}}]}\n',
        b"\n",
        b'data: {"choices":[{"delta":{"reasoning_content":"x"}}]}\n',
        b"\n",
        sse_line("fine"),
        b"\n",
        DONE,
    ]
    h = StreamHarness(script).install(monkeypatch)
    assert "".join(collect(h.model().generate([]))) == "fine"


def test_openai_bad_line_tolerated(monkeypatch):
    h = StreamHarness([b"data: {not json\n", b"\n", sse_line("ok"), b"\n", DONE]).install(
        monkeypatch
    )
    assert "".join(collect(h.model().generate([]))) == "ok"


def test_openai_eof_without_done_flushes_remainder(monkeypatch):
    h = StreamHarness([sse_line("tail")]).install(monkeypatch)
    assert "".join(collect(h.model().generate([]))) == "tail"


def test_openai_instream_error_raises_after_partial(monkeypatch):
    script = [sse_line("Hi."), b"\n", b'data: {"error": {"message": "boom"}}\n', b"\n"]
    h = StreamHarness(script).install(monkeypatch)
    gen = h.model().generate([])
    assert next(gen) == "Hi."
    with pytest.raises(DialogueUnavailableError):
        next(gen)


# ------------------------------------------------------------- ollama protocol


def test_ollama_happy_path(monkeypatch):
    script = [
        b'{"message":{"role":"assistant","content":"Hi "}}\n',
        b'{"message":{"role":"assistant","content":"there"},"done":true}\n',
    ]
    h = StreamHarness(
        script, make_config(provider="ollama", api_base="http://localhost:11434")
    ).install(monkeypatch)
    assert "".join(collect(h.model().generate([]))) == "Hi there"


def test_ollama_error_object_raises(monkeypatch):
    script = [b'{"message":{"content":"partial"}}\n', b'{"error":"boom"}\n']
    h = StreamHarness(
        script, make_config(provider="ollama", api_base="http://localhost:11434")
    ).install(monkeypatch)
    gen = h.model().generate([])
    assert next(gen) == "partial"
    with pytest.raises(DialogueUnavailableError):
        next(gen)


def test_ollama_bad_line_tolerated(monkeypatch):
    script = [b"not json at all\n", b'{"message":{"content":"ok"},"done":true}\n']
    h = StreamHarness(
        script, make_config(provider="ollama", api_base="http://localhost:11434")
    ).install(monkeypatch)
    assert "".join(collect(h.model().generate([]))) == "ok"


def test_ollama_eof_without_done(monkeypatch):
    script = [b'{"message":{"content":"left"}}\n']
    h = StreamHarness(
        script, make_config(provider="ollama", api_base="http://localhost:11434")
    ).install(monkeypatch)
    assert "".join(collect(h.model().generate([]))) == "left"


# --------------------------------------------------------- smooth-mode pacing


def test_boundary_flush_up_to_last_boundary_keeps_tail(monkeypatch):
    monkeypatch.setattr(model_module, "_FLUSH_INTERVAL_S", 30.0)
    h = StreamHarness([sse_line("Hello"), sse_line(" there world"), DONE]).install(monkeypatch)
    chunks = collect(h.model().generate([]))
    assert chunks == ["Hello there ", "world"]


def test_cjk_punctuation_flushes(monkeypatch):
    monkeypatch.setattr(model_module, "_FLUSH_INTERVAL_S", 30.0)
    h = StreamHarness([sse_line("你好。"), sse_line("在的")]).install(monkeypatch)
    chunks = collect(h.model().generate([]))
    assert chunks == ["你好。", "在的"]


def test_deadline_flush_on_silent_gap(monkeypatch):
    """No-boundary text must flush within the interval even across a stream gap."""
    monkeypatch.setattr(model_module, "_FLUSH_INTERVAL_S", 0.05)
    gate = threading.Event()
    h = StreamHarness([sse_line("abc"), b"\n", ("block", gate), sse_line("def"), DONE]).install(
        monkeypatch
    )
    gen = h.model().generate([])
    t0 = time.monotonic()
    first = next(gen)
    elapsed = time.monotonic() - t0
    assert first == "abc"
    assert elapsed < 0.5, "deadline flush did not fire across the stream gap"
    gate.set()
    assert "".join([first] + collect(gen)) == "abcdef"


def test_drip_stream_not_starved(monkeypatch):
    """A steady drip of boundary-less deltas must still yield within the interval."""
    monkeypatch.setattr(model_module, "_FLUSH_INTERVAL_S", 0.05)
    script = []
    for _ in range(4):
        script += [sse_line("x"), ("sleep", 0.03)]
    script += [DONE]
    h = StreamHarness(script).install(monkeypatch)
    dm = h.model()
    t0 = time.monotonic()
    chunks = collect(dm.generate([]), timeout=3.0)
    first_at = time.monotonic() - t0
    assert "".join(chunks) == "xxxx"
    assert first_at < 1.5, f"drip stream starved: first chunk after {first_at:.2f}s"


def test_interleaving_first_chunk_before_later_deltas(monkeypatch):
    """C-7 counterexample: chunk 1 must reach the consumer before the producer
    has emitted the remaining tokens (the old code delivered everything at once)."""
    monkeypatch.setattr(model_module, "_FLUSH_INTERVAL_S", 30.0)
    gate = threading.Event()
    h = StreamHarness([sse_line("Hi."), b"\n", ("block", gate), sse_line(" more"), DONE]).install(
        monkeypatch
    )
    dm = h.model()
    got = []
    first_seen = threading.Event()

    def consume():
        for chunk in dm.generate([]):
            got.append(chunk)
            first_seen.set()

    t = threading.Thread(target=consume)
    t.start()
    assert first_seen.wait(timeout=2.0), "no chunk arrived"
    assert h.resp.lines_read <= 2, "producer already ran past the gate"
    assert got == ["Hi."]
    gate.set()
    t.join(timeout=2.0)
    assert "".join(got) == "Hi. more"


# ---------------------------------------------------------------- cancellation


def test_cancel_during_stream_silence(monkeypatch):
    gate = threading.Event()
    h = StreamHarness([sse_line("Hi."), b"\n", ("block", gate)]).install(monkeypatch)
    cancel = threading.Event()
    gen = h.model().generate([], cancel_event=cancel)
    assert next(gen) == "Hi."
    assert not h.conn.closed
    cancel.set()
    with pytest.raises(StopIteration):
        next(gen)
    assert h.conn.closed, "cancel must close the connection"
    wait_no_thread("amadeus-dialogue-stream")


def test_cancel_before_first_chunk():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    time.sleep(0.2)
            except Exception:
                pass

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cancel = threading.Event()
        dm = DialogueModel(make_config(api_base=f"http://127.0.0.1:{port}", api_key_env=""))
        gen = dm.generate([{"role": "user", "content": "你好"}], cancel_event=cancel)
        it = iter(gen)
        cancel.set()
        t0 = time.monotonic()
        with pytest.raises(StopIteration):
            next(it)
        assert time.monotonic() - t0 < 3.0, "cancel during first-token wait must exit promptly"
        wait_no_thread("amadeus-dialogue-stream")
    finally:
        server.shutdown()


def test_loopback_socket_close_unblocks_reader():
    """Oracle F-1 regression: only closing the raw socket unblocks a blocked
    read; response.close() must not be part of the teardown path."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            body = b'data: {"choices":[{"delta":{"content":"Hi."}}]}\n\n'
            try:
                self.wfile.write(b"%x\r\n" % len(body) + body + b"\r\n")
                self.wfile.flush()
                time.sleep(10)
            except Exception:
                pass

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cancel = threading.Event()
        dm = DialogueModel(make_config(api_base=f"http://127.0.0.1:{port}", api_key_env=""))
        gen = dm.generate([{"role": "user", "content": "你好"}], cancel_event=cancel)
        it = iter(gen)
        assert next(it) == "Hi."
        cancel.set()
        t0 = time.monotonic()
        with pytest.raises(StopIteration):
            next(it)
        assert time.monotonic() - t0 < 3.0, "socket-close teardown did not unblock promptly"
        wait_no_thread("amadeus-dialogue-stream")
    finally:
        server.shutdown()


def wait_no_thread(name, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [t for t in threading.enumerate() if t.name == name and t.is_alive()]
        if not alive:
            return
        time.sleep(0.02)
    raise AssertionError(f"thread {name} still alive")


# --------------------------------------------------------------------- errors


def test_connection_failure_raises(monkeypatch):
    def boom(url, headers, body, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(model_module, "_open_stream", boom)
    with pytest.raises(DialogueUnavailableError):
        next(DialogueModel(make_config()).generate([]))


def test_midstream_io_error_raises_after_partial(monkeypatch):
    h = StreamHarness([sse_line("Hi."), b"\n", ("error", OSError("reset"))]).install(monkeypatch)
    gen = h.model().generate([])
    assert next(gen) == "Hi."
    with pytest.raises(DialogueUnavailableError):
        next(gen)


# -------------------------------------------------------------- params & auth


def test_openai_body_params(monkeypatch):
    h = StreamHarness([DONE]).install(monkeypatch)
    messages = [{"role": "user", "content": "hi"}]
    collect(h.model().generate(messages))
    body = h.body
    assert body["model"] == "test-model"
    assert body["messages"] == messages
    assert body["stream"] is True
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.5
    assert body["top_p"] == 0.8
    assert "repetition_penalty" not in body


def test_ollama_body_params(monkeypatch):
    script = [b'{"done":true}\n']
    h = StreamHarness(
        script, make_config(provider="ollama", api_base="http://localhost:11434")
    ).install(monkeypatch)
    collect(h.model().generate([]))
    body = h.body
    assert body["stream"] is True
    assert body["options"]["num_predict"] == 64
    assert body["options"]["repeat_penalty"] == 1.2
    assert body["options"]["temperature"] == 0.5
    assert body["options"]["top_p"] == 0.8


def test_auth_header_from_env(monkeypatch):
    monkeypatch.setenv("AMADEUS_TEST_KEY", "sk-test-123")
    h = StreamHarness([DONE]).install(monkeypatch)
    collect(h.model().generate([]))
    assert h.headers.get("Authorization") == "Bearer sk-test-123"


def test_no_auth_header_when_env_empty(monkeypatch):
    monkeypatch.setenv("AMADEUS_TEST_KEY", "")
    h = StreamHarness([DONE]).install(monkeypatch)
    collect(h.model().generate([]))
    assert "Authorization" not in h.headers


def test_request_body_is_utf8_bytes_roundtripping_chinese():
    """http.client rejects str bodies with non-ASCII (latin-1 encode); the body
    must be pre-encoded UTF-8 bytes. The persona prompt is Chinese."""
    dm = DialogueModel(make_config())
    messages = [{"role": "system", "content": "你是 Amadeus。"}]
    _url, _headers, body = dm._build_request(messages)
    assert isinstance(body, bytes)
    assert json.loads(body.decode("utf-8"))["messages"] == messages


def test_url_join_normalizes_trailing_slash(monkeypatch):
    h = StreamHarness([DONE], make_config(api_base="https://x.invalid/v1/")).install(monkeypatch)
    collect(h.model().generate([]))
    assert h.url == "https://x.invalid/v1/chat/completions"


def test_ollama_url_join(monkeypatch):
    h = StreamHarness(
        [b'{"done":true}\n'], make_config(provider="ollama", api_base="http://localhost:11434/")
    ).install(monkeypatch)
    collect(h.model().generate([]))
    assert h.url == "http://localhost:11434/api/chat"
