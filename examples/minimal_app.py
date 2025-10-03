# examples/minimal_app.py (Basic-token mode)
import os
os.environ.setdefault("OPENAI_BASIC_BASE_URL", "https://internal.company.ai")
os.environ.setdefault("OPENAI_BASIC_TOKEN", "abc.def.ghi")  # this is sent as: Authorization: Basic abc.def.ghi

import openai_basic as openai

client = openai.OpenAI()

r = client.responses.create(model="gpt-4o-mini", input="hello")
print("SYNC:", r["output_text"])

for ev in client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":"stream a tiny poem no punctuation"}],
    stream=True,
):
    if ev.get("type") == "response.delta":
        print(ev["delta"]["output_text"], end="", flush=True)
    elif ev.get("type") == "response.completed":
        print("\n[done]")
