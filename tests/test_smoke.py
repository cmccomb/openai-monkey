# tests/test_smoke.py (Basic-token mode)
import os
import uuid
from typing import Any, Callable

import httpx
import pytest
import respx

os.environ.setdefault("OPENAI_BASIC_BASE_URL", "https://internal.company.ai")
os.environ.setdefault("OPENAI_BASIC_TOKEN", "TEST_TOKEN")

import openai_monkey as openai


def _ensure(condition: bool, message: str) -> None:
    """Raise ``AssertionError`` with ``message`` when ``condition`` is ``False``."""

    if not condition:
        raise AssertionError(message)


@respx.mock
def test_sync_ok() -> None:
    respx.post("https://internal.company.ai/api/generate").mock(
        return_value=httpx.Response(
            200, json={"result": {"text": "ok"}, "usage": {"total_tokens": 1}}
        )
    )
    r = openai.OpenAI().responses.create(model="m", input="hi", max_tokens=5)
    _ensure(r["output_text"] == "ok", f"Unexpected response payload: {r!r}")


def test_invalid_auth_type_errors(configure_adapter: Callable[..., Any]) -> None:
    with pytest.raises(ValueError, match="Unsupported OPENAI_AUTH_TYPE"):
        configure_adapter(auth_type="oauth", token=uuid.uuid4().hex)
