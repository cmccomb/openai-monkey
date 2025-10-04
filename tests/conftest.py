from __future__ import annotations

import importlib
import json
import os
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_DEFAULT_CREDENTIAL = "test-credential"


@dataclass(slots=True)
class RecordedRequest:
    """Container for requests captured by :func:`chat_server`."""

    path: str
    headers: dict[str, str]
    body: bytes


@dataclass(slots=True)
class ChatServerContext:
    """Expose information about the background HTTP test server."""

    base_url: str
    requests: list[RecordedRequest]


class _ChatRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that mimics the internal chat endpoints."""

    server_version = "ChatTestServer/1.0"
    protocol_version = "HTTP/1.1"
    _SYNC_RESPONSE: ClassVar[dict[str, Any]] = {
        "id": "chat-42",
        "model": "chat-model",
        "result": {"text": "Hello from test server"},
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 3,
            "total_tokens": 12,
        },
    }
    _STREAM_CHUNKS: ClassVar[list[bytes]] = [
        b'data: {"type": "delta", "text": "Hello"}\n\n',
        b'data: {"text": " world"}\n\n',
        b'data: {"type": "done"}\n\n',
    ]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: D401
        """Silence default logging."""

        return

    def _capture_request(self, body: bytes) -> None:
        server = self.server  # type: ignore[assignment]
        requests = getattr(server, "captured_requests", None)
        if requests is None:
            return
        headers = {key: value for key, value in self.headers.items()}
        requests.append(RecordedRequest(path=self.path, headers=headers, body=body))

    def do_POST(self) -> None:  # noqa: D401, N802
        """Handle POST requests for chat endpoints."""

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b""
        self._capture_request(body)
        if self.path == "/v1/chat":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            payload = json.dumps(self._SYNC_RESPONSE).encode("utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            self.close_connection = True
            return
        if self.path == "/v1/chat-stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in self._STREAM_CHUNKS:
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True
            return
        self.send_response(404)
        self.end_headers()


@pytest.fixture
def chat_server() -> Iterator[ChatServerContext]:
    """Start a background HTTP server for integration tests."""

    captured: list[RecordedRequest] = []

    class _Server(ThreadingHTTPServer):
        allow_reuse_address = True

    server = _Server(("127.0.0.1", 0), _ChatRequestHandler)
    setattr(server, "captured_requests", captured)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"
    try:
        yield ChatServerContext(base_url=base_url, requests=captured)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture
def configure_adapter(request: pytest.FixtureRequest) -> Callable[..., Any]:
    """Return a helper for reloading ``openai_monkey`` with a custom config."""

    original_env: dict[str, str | None] = {}

    def _setenv(name: str, value: str | None) -> None:
        if name not in original_env:
            original_env[name] = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def _configure(
        *,
        base_url: str = "https://mock.local",
        token: str | None = None,
        path_map: dict[str, str] | None = None,
        param_map: dict[str, str] | None = None,
        drop_params: list[str] | None = None,
        extra_allow: list[str] | None = None,
        default_headers: dict[str, str] | None = None,
        disable_streaming: bool = False,
        auth_type: str = "basic",
    ) -> Any:
        _setenv("OPENAI_AUTH_TYPE", auth_type)
        _setenv("OPENAI_BASE_URL", base_url)
        credential = token if token is not None else _DEFAULT_CREDENTIAL
        _setenv("OPENAI_TOKEN", credential)

        if path_map is not None:
            _setenv("OPENAI_BASIC_PATH_MAP", json.dumps(path_map))
        if param_map is not None:
            _setenv("OPENAI_BASIC_PARAM_MAP", json.dumps(param_map))
        if drop_params is not None:
            _setenv("OPENAI_BASIC_DROP_PARAMS", json.dumps(drop_params))
        if extra_allow is not None:
            _setenv("OPENAI_BASIC_EXTRA_ALLOW", json.dumps(extra_allow))
        if default_headers is not None:
            _setenv("OPENAI_BASIC_HEADERS", json.dumps(default_headers))
        _setenv("OPENAI_BASIC_DISABLE_STREAMING", "1" if disable_streaming else "0")

        import openai_monkey

        return importlib.reload(openai_monkey)

    def _cleanup() -> None:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if "openai_monkey" in sys.modules:
            import openai_monkey

            importlib.reload(openai_monkey)

    request.addfinalizer(_cleanup)
    return _configure
