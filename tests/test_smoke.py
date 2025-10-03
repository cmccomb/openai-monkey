# tests/test_smoke.py (Basic-token mode)
import os, httpx, respx
os.environ.setdefault("OPENAI_BASIC_BASE_URL", "https://internal.company.ai")
os.environ.setdefault("OPENAI_BASIC_TOKEN", "TEST_TOKEN")

import openai_monkey as openai

@respx.mock
def test_sync_ok():
    respx.post("https://internal.company.ai/api/generate").mock(
        return_value=httpx.Response(200, json={"result":{"text":"ok"}, "usage":{"total_tokens":1}})
    )
    r = openai.OpenAI().responses.create(model="m", input="hi", max_tokens=5)
    assert r["output_text"] == "ok"
