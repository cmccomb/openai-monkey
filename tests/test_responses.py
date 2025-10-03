from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx


@pytest.fixture
def responses_adapter(configure_adapter):
    return configure_adapter(
        path_map={
            "/responses": "/v1/resp",
            "/responses:stream": "/v1/resp-stream",
        },
        param_map={
            "max_tokens": "max_output_tokens",
            "temperature": "temp",
        },
        drop_params=["logprobs", "tool_choice"],
        default_headers={"X-Test": "true"},
    )


def _json_payload(request: Any) -> dict[str, Any]:
    body = request.content
    return json.loads(body.decode("utf-8"))


@respx.mock
def test_responses_create_remaps_and_headers(responses_adapter):
    # Arrange
    client = responses_adapter.OpenAI()
    route = respx.post("https://mock.local/v1/resp").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "id": "resp-123",
                "model": "m",
                "result": {"text": "done"},
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        )
    )

    # Act
    result = client.responses.create(
        model="m",
        input="hi",
        max_tokens=5,
        logprobs=2,
        temperature=0.25,
    )

    # Assert
    assert route.called
    request = route.calls[0].request
    payload = _json_payload(request)
    assert payload == {
        "model": "m",
        "input": "hi",
        "temp": 0.25,
        "max_output_tokens": 5,
    }
    assert request.headers["Authorization"] == "Basic TEST_TOKEN"
    assert request.headers["X-Test"] == "true"
    assert result == {
        "id": "resp-123",
        "model": "m",
        "output_text": "done",
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
    }


@respx.mock
def test_responses_create_streaming_normalizes_lines(responses_adapter):
    # Arrange
    client = responses_adapter.OpenAI()
    stream_body = (
        b"data: {\"type\": \"delta\", \"text\": \"Hel\"}\n\n"
        b"data: {\"type\": \"delta\", \"text\": \"lo\"}\n\n"
        b"data: {\"type\": \"unknown\"}\n\n"
        b"not-json\n\n"
        b"\xff\n\n"
        b"data: {\"type\": \"done\"}\n\n"
    )
    route = respx.post("https://mock.local/v1/resp-stream").mock(
        return_value=httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            content=stream_body,
        )
    )

    # Act
    events = list(
        client.responses.create(
            model="stream-model",
            input="start",
            stream=True,
        )
    )

    # Assert
    assert route.called
    assert events == [
        {"type": "response.delta", "delta": {"output_text": "Hel"}},
        {"type": "response.delta", "delta": {"output_text": "lo"}},
        {"type": "response.completed"},
    ]


@respx.mock
def test_responses_create_raises_for_non_200(responses_adapter):
    # Arrange
    client = responses_adapter.OpenAI()
    respx.post("https://mock.local/v1/resp").mock(
        return_value=httpx.Response(status_code=500, json={"error": "boom"})
    )

    # Act / Assert
    with pytest.raises(httpx.HTTPStatusError):
        client.responses.create(model="m", input="hi")


@respx.mock
def test_responses_create_normalizes_missing_result(responses_adapter):
    # Arrange
    client = responses_adapter.OpenAI()
    respx.post("https://mock.local/v1/resp").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "id": "resp-456",
                "model": "m",
                "choices": [
                    {"message": {"content": "fallback"}},
                ],
                "usage": {"total_tokens": 7},
            },
        )
    )

    # Act
    result = client.responses.create(model="m", input="hi")

    # Assert
    assert result == {
        "id": "resp-456",
        "model": "m",
        "output_text": "fallback",
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": 7,
        },
    }
