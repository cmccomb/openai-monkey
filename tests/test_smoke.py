# tests/test_smoke.py (Basic-token mode)
import os

import httpx
import pytest
import respx  # type: ignore[import]

os.environ.setdefault("OPENAI_BASIC_BASE_URL", "https://internal.company.ai")
os.environ.setdefault("OPENAI_BASIC_TOKEN", "TEST_TOKEN")

import openai_monkey as openai


@respx.mock
def test_sync_ok():
    respx.post("https://internal.company.ai/api/generate").mock(
        return_value=httpx.Response(
            200, json={"result": {"text": "ok"}, "usage": {"total_tokens": 1}}
        )
    )
    r = openai.OpenAI().responses.create(model="m", input="hi", max_tokens=5)
    assert r["output_text"] == "ok"


def test_invalid_auth_type_errors(configure_adapter):
    with pytest.raises(ValueError, match="Unsupported OPENAI_AUTH_TYPE"):
        configure_adapter(auth_type="oauth", token="fake")
