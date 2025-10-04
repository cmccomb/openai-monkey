from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

import httpx
import pytest
import respx


def _ensure(condition: bool, message: str) -> None:
    """Raise ``AssertionError`` with ``message`` when ``condition`` is ``False``."""

    if not condition:
        raise AssertionError(message)


@pytest.fixture(params=["basic", "bearer"])
def responses_adapter(
    configure_adapter: Callable[..., Any], request: pytest.FixtureRequest
) -> tuple[Any, str]:
    auth_type: str = request.param
    token = "TEST_BASIC_TOKEN" if auth_type == "basic" else "TEST_BEARER_TOKEN"
    module = configure_adapter(
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
        token=token,
        auth_type=auth_type,
    )
    expected_header = f"Basic {token}" if auth_type == "basic" else f"Bearer {token}"
    return module, expected_header


def _json_payload(request: httpx.Request) -> dict[str, Any]:
    body = request.content
    return cast(dict[str, Any], json.loads(body.decode("utf-8")))


@respx.mock
def test_responses_create_remaps_and_headers(
    responses_adapter: tuple[Any, str],
) -> None:
    # Arrange
    module, expected_header = responses_adapter
    client = module.OpenAI()
    route = respx.post("https://mock.local/v1/resp").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "id": "resp-123",
                "model": "m",
                "result": {"text": "done"},
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            },
        )
    )

    # Act
    result: dict[str, Any] = client.responses.create(
        model="m",
        input="hi",
        max_tokens=5,
        logprobs=2,
        temperature=0.25,
    )

    # Assert
    _ensure(route.called, "Expected responses.create to call HTTP route")
    request = route.calls[0].request
    payload = _json_payload(request)
    expected_payload = {
        "model": "m",
        "input": "hi",
        "temp": 0.25,
        "max_output_tokens": 5,
    }
    _ensure(payload == expected_payload, f"Unexpected payload: {payload!r}")
    _ensure(
        request.headers["Authorization"] == expected_header,
        "Authorization header did not match expected auth scheme",
    )
    _ensure(
        request.headers["X-Test"] == "true",
        "Default header X-Test was not forwarded",
    )
    expected_result = {
        "id": "resp-123",
        "model": "m",
        "output_text": "done",
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
    }
    _ensure(result == expected_result, f"Unexpected normalized response: {result!r}")


@respx.mock
def test_responses_create_streaming_normalizes_lines(
    responses_adapter: tuple[Any, str],
) -> None:
    # Arrange
    module, expected_header = responses_adapter
    client = module.OpenAI()
    stream_body = (
        b'data: {"type": "delta", "text": "Hel"}\n\n'
        b'data: {"type": "delta", "text": "lo"}\n\n'
        b'data: {"type": "unknown"}\n\n'
        b"not-json\n\n"
        b"\xff\n\n"
        b'data: {"type": "done"}\n\n'
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
    _ensure(route.called, "Streaming response route was not invoked")
    expected_events = [
        {"type": "response.delta", "delta": {"output_text": "Hel"}},
        {"type": "response.delta", "delta": {"output_text": "lo"}},
        {"type": "response.completed"},
    ]
    _ensure(events == expected_events, f"Unexpected streaming events: {events!r}")
    request = route.calls[0].request
    _ensure(
        request.headers["Authorization"] == expected_header,
        "Authorization header missing from streaming request",
    )


@respx.mock
def test_responses_create_raises_for_non_200(
    responses_adapter: tuple[Any, str],
) -> None:
    # Arrange
    module, _ = responses_adapter
    client = module.OpenAI()
    respx.post("https://mock.local/v1/resp").mock(
        return_value=httpx.Response(status_code=500, json={"error": "boom"})
    )

    # Act / Assert
    with pytest.raises(httpx.HTTPStatusError):
        client.responses.create(model="m", input="hi")


@respx.mock
def test_responses_create_normalizes_missing_result(
    responses_adapter: tuple[Any, str],
) -> None:
    # Arrange
    module, _ = responses_adapter
    client = module.OpenAI()
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
    result: dict[str, Any] = client.responses.create(model="m", input="hi")

    # Assert
    expected = {
        "id": "resp-456",
        "model": "m",
        "output_text": "fallback",
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": 7,
        },
    }
    _ensure(result == expected, f"Unexpected normalized fallback response: {result!r}")


@respx.mock
def test_responses_create_honors_extra_allow(
    configure_adapter: Callable[..., Any],
) -> None:
    """Parameters listed in ``extra_allow`` should bypass the drop filter."""

    module = configure_adapter(
        path_map={"/responses": "/v1/resp"},
        param_map={},
        drop_params=["safety_profile"],
        extra_allow=["safety_profile"],
    )
    client = module.OpenAI()
    route = respx.post("https://mock.local/v1/resp").mock(
        return_value=httpx.Response(
            status_code=200,
            json={"id": "resp-789", "model": "m", "result": {"text": "ok"}},
        )
    )

    client.responses.create(
        model="m",
        input="hi",
        safety_profile={"category": "allowed"},
    )

    _ensure(route.called, "Route should have been invoked for extra_allow payload")
    payload = _json_payload(route.calls[0].request)
    _ensure(
        payload.get("safety_profile") == {"category": "allowed"},
        f"safety_profile not forwarded: {payload!r}",
    )
