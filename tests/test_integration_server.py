"""Integration tests against the background HTTP chat server fixture."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.conftest import ChatServerContext, RecordedRequest


os.environ.setdefault("OPENAI_BASE_URL", "https://mock.local")
os.environ.setdefault("OPENAI_TOKEN", "test-credential")


def _json_body(record: "RecordedRequest") -> dict[str, Any]:
    """Decode a recorded request body as JSON."""

    return json.loads(record.body.decode("utf-8"))


def _resolve_sync_response(result: Any) -> Any:
    """Return the synchronous payload from the client call."""

    if isinstance(result, dict):
        return result
    iterator = iter(result)
    with pytest.raises(StopIteration) as exc_info:
        next(iterator)
    return exc_info.value.value


def test_chat_completion_roundtrip(
    chat_server: "ChatServerContext", configure_adapter: Callable[..., Any]
) -> None:
    """The patched client should POST remapped payloads to the chat endpoint."""

    # Arrange
    module = configure_adapter(
        base_url=chat_server.base_url,
        token="integration-secret",
        path_map={
            "/chat/completions": "/v1/chat",
            "/chat/completions:stream": "/v1/chat-stream",
        },
        param_map={
            "max_tokens": "max_output_tokens",
            "temperature": "temp",
        },
        drop_params=["logprobs"],
        default_headers={"X-Test": "true"},
    )
    client = module.OpenAI()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]

    # Act
    result = _resolve_sync_response(
        client.chat.completions.create(
            model="chat-model",
            messages=messages,
            max_tokens=32,
            temperature=0.25,
            logprobs=True,
        )
    )

    # Assert
    assert len(chat_server.requests) == 1
    recorded = chat_server.requests[0]
    assert recorded.path == "/v1/chat"
    payload = _json_body(recorded)
    assert payload == {
        "model": "chat-model",
        "input": "SYSTEM: You are helpful.\nUSER: Hello\nASSISTANT:",
        "max_output_tokens": 32,
        "temp": 0.25,
    }
    assert recorded.headers["Authorization"] == "Basic integration-secret"
    assert recorded.headers["X-Test"] == "true"
    assert result == {
        "id": "chat-42",
        "model": "chat-model",
        "output_text": "Hello from test server",
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 3,
            "total_tokens": 12,
        },
    }


def test_chat_completion_streaming(
    chat_server: "ChatServerContext", configure_adapter: Callable[..., Any]
) -> None:
    """Streaming responses should surface SSE deltas from the server."""

    # Arrange
    module = configure_adapter(
        base_url=chat_server.base_url,
        path_map={
            "/chat/completions": "/v1/chat",
            "/chat/completions:stream": "/v1/chat-stream",
        },
    )
    client = module.OpenAI()

    # Act
    events = list(
        client.chat.completions.create(
            model="chat-model",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
    )

    # Assert
    assert len(chat_server.requests) == 1
    recorded = chat_server.requests[0]
    assert recorded.path == "/v1/chat-stream"
    assert events == [
        {"type": "response.delta", "delta": {"output_text": "Hello"}},
        {"type": "response.delta", "delta": {"output_text": " world"}},
        {"type": "response.completed"},
    ]
