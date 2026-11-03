"""Dialogue model: OpenAI-compatible / Ollama API client with smooth streaming.

Replaces the local transformers path (ADR 0005): the pinned transformers 4.32.1
cannot load the configured Qwen2.5 (no ``qwen2`` architecture, no
``apply_chat_template``), so local generation never worked. Access is API-based:
``api_base`` + ``model`` + an API key read from the environment (the key itself
is never stored in config or the repository).
"""

import json
import os
import threading
from collections.abc import Generator
from http import client as http_client
from queue import Empty, Queue
from time import monotonic
from urllib.parse import urlparse

from loguru import logger


class DialogueUnavailableError(RuntimeError):
    """The dialogue backend cannot produce a reply (unconfigured / network / protocol)."""


# Smooth-mode pacing (ADR 0005 D3): flush at the last boundary character and
# force a flush at least every _FLUSH_INTERVAL_S anchored on the previous flush
# (deadline anchoring; a per-arrival timer would starve on a steady drip).
_FLUSH_INTERVAL_S = 0.2
_POLL_S = 0.05
# The stream reader polls the socket at this granularity: on Windows, closing a
# socket does not reliably wake a blocked recv (measured 0-5 s, or only at
# socket timeout), so the reader must wake itself to notice stop/close.
_STREAM_POLL_S = 0.25
_BOUNDARY_CHARS = frozenset(" \t\n\r.,!?;:，。！？；：、…—\"'”’)）]】")

_TEXT, _DONE, _ERROR = 0, 1, 2
_UNPARSED = object()


def _open_stream(url: str, headers: dict, body: str, timeout: float):
    """POST and return ``(response, connection)``.

    Teardown notes (Oracle review F-1, re-measured locally): on Windows,
    ``response.close()`` deadlocks against a concurrent reader and
    ``closesocket`` does not reliably wake a blocked recv (0-5 s, sometimes only
    at socket timeout). The reader therefore polls at a short timeout
    (``_STREAM_POLL_S``) and the caller tears down via ``conn.sock.close()``
    as a best effort without waiting for the reader to wake.
    """
    parsed = urlparse(url)
    conn_cls = (
        http_client.HTTPSConnection if parsed.scheme == "https" else http_client.HTTPConnection
    )
    conn = conn_cls(parsed.netloc, timeout=timeout)
    conn.request("POST", parsed.path or "/", body=body, headers=headers)
    resp = conn.getresponse()
    # Streaming phase: short read timeout so the reader wakes itself to notice
    # stop/close instead of relying on closesocket to break the recv.
    conn.sock.settimeout(_STREAM_POLL_S)
    return resp, conn


def _close_quietly(conn) -> None:
    try:
        conn.sock.close()
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass


def _last_boundary(buf: str) -> int:
    for i in range(len(buf) - 1, -1, -1):
        if buf[i] in _BOUNDARY_CHARS:
            return i
    return -1


class DialogueModel:
    def __init__(self, config: dict):
        cfg = config.get("dialogue", {})
        self.provider = str(cfg.get("provider", "") or "").strip().lower()
        self.api_base = str(cfg.get("api_base", "") or "").strip().rstrip("/")
        self.model = str(cfg.get("model", "") or "").strip()
        self.api_key_env = str(cfg.get("api_key_env", "AMADEUS_API_KEY") or "")
        self.request_timeout = float(cfg.get("request_timeout_seconds", 60) or 60)
        self.max_new_tokens = int(cfg.get("max_new_tokens", 512))
        self.temperature = float(cfg.get("temperature", 0.7))
        self.top_p = float(cfg.get("top_p", 0.9))
        self.repetition_penalty = float(cfg.get("repetition_penalty", 1.1))
        self._configured = (
            self.provider in ("openai", "ollama") and bool(self.api_base) and bool(self.model)
        )

    def load(self) -> bool:
        if not self._configured:
            logger.warning(
                "dialogue: not configured (need provider/api_base/model in config); "
                "API access disabled"
            )
        return self._configured

    @property
    def is_loaded(self) -> bool:
        return self._configured

    def generate(
        self,
        messages: list[dict[str, str]],
        cancel_event: threading.Event | None = None,
    ) -> Generator[str, None, None]:
        if not self._configured:
            raise DialogueUnavailableError(
                "对话模型未配置：请在 config/default.yaml 的 dialogue 段设置 provider、api_base、"
                "model，并将 API 密钥置于 api_key_env 指定的环境变量"
            )
        if cancel_event is not None and cancel_event.is_set():
            return
        url, headers, body = self._build_request(messages)
        try:
            resp, conn = _open_stream(url, headers, body, self.request_timeout)
        except Exception as e:
            raise DialogueUnavailableError(f"对话服务连接失败：{e}") from e

        queue: Queue = Queue()
        stop = threading.Event()
        reader = threading.Thread(
            target=self._read_stream,
            args=(resp, queue, stop),
            name="amadeus-dialogue-stream",
            daemon=True,
        )
        reader.start()
        buf = ""
        last_flush = monotonic()
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return
                now = monotonic()
                if buf and now - last_flush >= _FLUSH_INTERVAL_S:
                    out, buf = buf, ""
                    last_flush = now
                    yield out
                    continue
                try:
                    kind, payload = queue.get(timeout=_POLL_S)
                except Empty:
                    continue
                if kind == _TEXT:
                    buf += payload
                    k = _last_boundary(buf)
                    if k >= 0:
                        head, tail = buf[: k + 1], buf[k + 1 :]
                        if head.strip():
                            buf = tail
                            last_flush = monotonic()
                            yield head
                elif kind == _DONE:
                    if buf:
                        yield buf
                    return
                else:  # _ERROR
                    if buf:
                        yield buf
                    raise DialogueUnavailableError(payload)
        finally:
            stop.set()
            _close_quietly(conn)
            reader.join(timeout=2.0)
            if reader.is_alive():
                logger.warning("dialogue: stream reader survived join timeout")

    # ------------------------------------------------------------------ request

    def _build_request(self, messages: list[dict[str, str]]):
        if self.provider == "openai":
            url = self.api_base + "/chat/completions"
            # repetition_penalty is intentionally absent: it is not part of the
            # OpenAI API and strict implementations reject unknown fields (D7).
            body = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
        else:
            url = self.api_base + "/api/chat"
            body = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "num_predict": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "repeat_penalty": self.repetition_penalty,
                },
            }
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return url, headers, json.dumps(body, ensure_ascii=False)

    # ------------------------------------------------------------------- reader

    def _read_stream(self, resp, queue: Queue, stop: threading.Event) -> None:
        try:
            if self.provider == "openai":
                self._read_sse(resp, queue, stop)
            else:
                self._read_ndjson(resp, queue, stop)
            queue.put((_DONE, None))
        except Exception as e:
            queue.put((_ERROR, f"对话服务读取失败：{e}"))

    def _read_sse(self, resp, queue: Queue, stop: threading.Event) -> None:
        pending: list[str] = []
        for raw in _iter_lines(resp, stop):
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                pending.clear()
                continue
            if line.startswith(":"):
                continue
            if not line.startswith("data:"):
                logger.debug(f"dialogue: ignoring SSE line: {line[:40]!r}")
                continue
            payload = line[5:].lstrip()
            if payload == "[DONE]":
                return
            obj = self._parse_sse_payload(pending, payload)
            if obj is _UNPARSED:
                continue
            if isinstance(obj, dict) and obj.get("error"):
                err = obj["error"]
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                queue.put((_ERROR, f"对话服务返回错误：{msg}"))
                return
            text = self._extract_openai_text(obj)
            if text:
                queue.put((_TEXT, text))

    def _parse_sse_payload(self, pending: list[str], payload: str):
        candidate = "\n".join(pending + [payload])
        try:
            obj = json.loads(candidate)
        except ValueError:
            pending.append(payload)
            return _UNPARSED
        pending.clear()
        return obj

    def _extract_openai_text(self, obj) -> str:
        if not isinstance(obj, dict):
            return ""
        choices = obj.get("choices") or []
        if not choices:
            return ""
        delta = (choices[0] or {}).get("delta") or {}
        content = delta.get("content")
        return content if isinstance(content, str) else ""

    def _read_ndjson(self, resp, queue: Queue, stop: threading.Event) -> None:
        for raw in _iter_lines(resp, stop):
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                logger.debug(f"dialogue: ignoring NDJSON line: {line[:40]!r}")
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("error"):
                queue.put((_ERROR, f"对话服务返回错误：{obj['error']}"))
                return
            content = (obj.get("message") or {}).get("content")
            if content:
                queue.put((_TEXT, content))
            if obj.get("done"):
                return


def _iter_lines(resp, stop: threading.Event):
    """Yield response lines, waking periodically to honour ``stop``.

    ``readline`` may raise ``TimeoutError`` between lines; that is just the
    poll tick, not a failure. EOF yields nothing further.
    """
    while not stop.is_set():
        try:
            raw = resp.readline()
        except TimeoutError:
            continue
        if not raw:
            return
        yield raw
