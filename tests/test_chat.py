from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx  # type: ignore[import]


@pytest.fixture(params=["basic", "bearer"])
def chat_adapter(configure_adapter, request):
    auth_type: str = request.param
    token = "TEST_BASIC_TOKEN" if auth_type == "basic" else "TEST_BEARER_TOKEN"
    module = configure_adapter(
        path_map={
            "/chat/completions": "/v1/chat",
            "/chat/completions:stream": "/v1/chat-stream",
        },
        param_map={
            "max_tokens": "max_output_tokens",
            "temperature": "temp",
        },
        drop_params=["logprobs", "tool_choice"],
        default_headers={"X-Test": "true"},
        token=token,
        auth_type=auth_type,
    )
    expected_header = f"Basic {token}" if auth_type == "basic" else f"Bearer {token}"
    return module, expected_header


def _json_payload(request: Any) -> dict[str, Any]:
    body = request.content
    return json.loads(body.decode("utf-8"))


def _consume_sync_result(generator):
    with pytest.raises(StopIteration) as exc_info:
        next(generator)
    return exc_info.value.value


@respx.mock
def test_chat_create_remaps_and_headers(chat_adapter):
    # Arrange
    module, expected_header = chat_adapter
    client = module.OpenAI()
    route = respx.post("https://mock.local/v1/chat").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "id": "chat-123",
                "model": "chat-model",
                "result": {"text": "response"},
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )
    )
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]

    # Act
    result_iter = client.chat.completions.create(
        model="chat-model",
        messages=messages,
        max_tokens=50,
        tool_choice="none",
        temperature=0.5,
    )
    result = _consume_sync_result(result_iter)

    # Assert
    assert route.called
    request = route.calls[0].request
    payload = _json_payload(request)
    assert payload == {
        "model": "chat-model",
        "input": "SYSTEM: You are helpful.\nUSER: Hello\nASSISTANT:",
        "temp": 0.5,
        "max_output_tokens": 50,
    }
    assert request.headers["Authorization"] == expected_header
    assert request.headers["X-Test"] == "true"
    assert result == {
        "id": "chat-123",
        "model": "chat-model",
        "output_text": "response",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


@respx.mock
def test_chat_create_supports_list_text_content(chat_adapter):
    # Arrange
    module, _ = chat_adapter
    client = module.OpenAI()
    route = respx.post("https://mock.local/v1/chat").mock(
        return_value=httpx.Response(
            status_code=200,
            json={"id": "chat-234", "model": "chat-model", "result": {"text": "ok"}},
        )
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Part one"},
                {"type": "text", "text": "Part two"},
            ],
        }
    ]

    # Act
    result_iter = client.chat.completions.create(model="chat-model", messages=messages)
    _consume_sync_result(result_iter)

    # Assert
    assert route.called
    payload = _json_payload(route.calls[0].request)
    assert payload["input"] == "USER: Part one\nPart two\nASSISTANT:"


@respx.mock
def test_chat_create_streaming_handles_malformed_lines(chat_adapter):
    # Arrange
    module, expected_header = chat_adapter
    client = module.OpenAI()
    stream_body = (
        b'data: {"type": "delta", "text": "Hello"}\n\n'
        b'data: {"text": " world"}\n\n'
        b'data: {"type": "delta", "text": '
        b'"ignored"}\n\n'
        b'data: {"type": "done"}\n\n'
    )
    route = respx.post("https://mock.local/v1/chat-stream").mock(
        return_value=httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            content=stream_body,
        )
    )

    # Act
    events = list(
        client.chat.completions.create(
            model="chat-model",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
    )

    # Assert
    assert route.called
    assert events == [
        {"type": "response.delta", "delta": {"output_text": "Hello"}},
        {"type": "response.delta", "delta": {"output_text": " world"}},
        {"type": "response.delta", "delta": {"output_text": "ignored"}},
        {"type": "response.completed"},
    ]
    request = route.calls[0].request
    assert request.headers["Authorization"] == expected_header


@respx.mock
def test_chat_create_raises_for_non_200(chat_adapter):
    # Arrange
    module, _ = chat_adapter
    client = module.OpenAI()
    respx.post("https://mock.local/v1/chat").mock(
        return_value=httpx.Response(status_code=500, json={"error": "bad"})
    )

    # Act / Assert
    with pytest.raises(httpx.HTTPStatusError):
        iterator = client.chat.completions.create(model="chat-model", messages=[])
        next(iterator)


@respx.mock
def test_chat_create_rejects_unsupported_message_parts(chat_adapter):
    # Arrange
    module, _ = chat_adapter
    client = module.OpenAI()
    route = respx.post("https://mock.local/v1/chat").mock(
        return_value=httpx.Response(status_code=200, json={"id": "x"})
    )

    # Act / Assert
    with pytest.raises(TypeError, match="Unsupported chat message content part type"):
        client.chat.completions.create(
            model="chat-model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example"}},
                    ],
                }
            ],
        )

    assert not route.called


@respx.mock
def test_chat_create_normalizes_plain_text(chat_adapter):
    # Arrange
    module, _ = chat_adapter
    client = module.OpenAI()
    respx.post("https://mock.local/v1/chat").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "id": "chat-456",
                "model": "chat-model",
                "text": "direct",
                "usage": {},
            },
        )
    )

    # Act
    result_iter = client.chat.completions.create(
        model="chat-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    result = _consume_sync_result(result_iter)

    # Assert
    assert result == {
        "id": "chat-456",
        "model": "chat-model",
        "output_text": "direct",
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }
