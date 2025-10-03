# openai-basic-compat (Basic-token mode)

This variant treats your **API key** as a **Basic token**. No username/password are stored or used.

## Two ways to use

### 1) Explicit (recommended)
```python
import openai_monkey as openai
client = openai.OpenAI()
print(client.responses.create(model="gpt-4o-mini", input="ping")["output_text"])
```

### 2) Stealth mode (alias to `openai`)
```bash
export OPENAI_BASIC_ALIAS_OPENAI=1
python your_app.py
```

## Configuration (env)

Minimal (Basic auth, default):
```bash
export OPENAI_AUTH_TYPE="basic"
export OPENAI_BASE_URL="https://internal.company.ai"
export OPENAI_TOKEN="$YOUR_BASIC_TOKEN"   # sends: Authorization: Basic $YOUR_BASIC_TOKEN
```

Bearer support:
```bash
export OPENAI_AUTH_TYPE="bearer"
export OPENAI_BASE_URL="https://internal.company.ai"
export OPENAI_TOKEN="$YOUR_BEARER_TOKEN"   # sends: Authorization: Bearer $YOUR_BEARER_TOKEN
```

For backwards compatibility the adapter still honours `OPENAI_BASIC_BASE_URL` and
`OPENAI_BASIC_TOKEN` when the relaxed names are not set. Token fallbacks also check
`OPENAI_BEARER_TOKEN`, `OPENAI_API_KEY`, and `OPENAI_KEY`.

Optional JSON knobs:
```bash
export OPENAI_BASIC_PATH_MAP='{
  "/responses": "/api/generate",
  "/responses:stream": "/api/stream",
  "/chat/completions": "/api/generate",
  "/chat/completions:stream": "/api/stream"
}'
export OPENAI_BASIC_PARAM_MAP='{"max_tokens":"max_output_tokens","top_p":"nucleus"}'
export OPENAI_BASIC_DROP_PARAMS='["logprobs","tool_choice"]'
export OPENAI_BASIC_EXTRA_ALLOW='["safety_profile"]'
export OPENAI_BASIC_MODEL_ROUTES='{"llama3.*":{"path":"/api/generate_llama"}}'
export OPENAI_BASIC_HEADERS='{"X-Org":"design-research"}'
export OPENAI_BASIC_DISABLE_STREAMING=0
```

## Examples
- `examples/minimal_app.py` demonstrates the default Basic flow.
- `examples/bearer_app.py` shows how to call the adapter with Bearer tokens.
