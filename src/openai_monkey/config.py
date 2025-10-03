from __future__ import annotations

import json
import os
from typing import Any, TypeVar


_T = TypeVar("_T")


def _load_json_env(name: str, default: _T) -> _T:
    """Load an environment variable as JSON, falling back to ``default``."""

    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _pick_env(*names: str, default: str) -> str:
    """Return the first set environment variable among ``names`` or ``default``."""

    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def load_config() -> dict[str, Any]:
    """Load adapter configuration from environment variables."""

    auth_type = os.getenv("OPENAI_AUTH_TYPE", "basic").strip().lower() or "basic"

    base_url = _pick_env(
        "OPENAI_BASE_URL",
        "OPENAI_BASIC_BASE_URL",
        default="https://internal.company.ai",
    )

    token = _pick_env(
        "OPENAI_TOKEN",
        "OPENAI_BEARER_TOKEN",
        "OPENAI_BASIC_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_KEY",
        default="REPLACE_ME",
    )

    path_map = _load_json_env(
        "OPENAI_BASIC_PATH_MAP",
        {
            "/responses": "/api/generate",
            "/responses:stream": "/api/stream",
            "/chat/completions": "/api/generate",
            "/chat/completions:stream": "/api/stream",
        },
    )
    param_map = _load_json_env(
        "OPENAI_BASIC_PARAM_MAP",
        {
            "max_tokens": "max_output_tokens",
            "top_p": "nucleus",
        },
    )
    drop_params = set(
        _load_json_env("OPENAI_BASIC_DROP_PARAMS", ["logprobs", "tool_choice"])
    )
    extra_allow = set(_load_json_env("OPENAI_BASIC_EXTRA_ALLOW", ["safety_profile"]))
    model_routes: dict[str, dict[str, Any]] = _load_json_env(
        "OPENAI_BASIC_MODEL_ROUTES", {}
    )
    disable_streaming = os.getenv("OPENAI_BASIC_DISABLE_STREAMING", "0") not in (
        "",
        "0",
        "false",
        "False",
    )
    default_headers: dict[str, str] = _load_json_env("OPENAI_BASIC_HEADERS", {})

    return {
        "auth_type": auth_type,
        "base_url": base_url,
        "token": token,
        "path_map": path_map,
        "param_map": param_map,
        "drop_params": drop_params,
        "extra_allow": extra_allow,
        "model_routes": model_routes,
        "disable_streaming": disable_streaming,
        "default_headers": default_headers,
    }
