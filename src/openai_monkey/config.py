from __future__ import annotations
import json, os

def _load_json_env(name: str, default):
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default

def load_config():
    base_url = os.getenv("OPENAI_BASIC_BASE_URL", "https://internal.company.ai")
    # Basic-token mode: prefer OPENAI_BASIC_TOKEN; fall back to OPENAI_API_KEY, OPENAI_KEY
    basic_token = (
        os.getenv("OPENAI_BASIC_TOKEN")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_KEY")
        or "REPLACE_ME"
    )

    path_map = _load_json_env("OPENAI_BASIC_PATH_MAP", {
        "/responses": "/api/generate",
        "/responses:stream": "/api/stream",
        "/chat/completions": "/api/generate",
        "/chat/completions:stream": "/api/stream",
    })
    param_map = _load_json_env("OPENAI_BASIC_PARAM_MAP", {
        "max_tokens": "max_output_tokens",
        "top_p": "nucleus",
    })
    drop_params = set(_load_json_env("OPENAI_BASIC_DROP_PARAMS", ["logprobs", "tool_choice"]))
    extra_allow = set(_load_json_env("OPENAI_BASIC_EXTRA_ALLOW", ["safety_profile"]))
    model_routes = _load_json_env("OPENAI_BASIC_MODEL_ROUTES", {})
    disable_streaming = os.getenv("OPENAI_BASIC_DISABLE_STREAMING", "0") not in ("", "0", "false", "False")
    default_headers = _load_json_env("OPENAI_BASIC_HEADERS", {})

    return dict(
        base_url=base_url,
        basic_token=basic_token,
        path_map=path_map,
        param_map=param_map,
        drop_params=drop_params,
        extra_allow=extra_allow,
        model_routes=model_routes,
        disable_streaming=disable_streaming,
        default_headers=default_headers,
    )
