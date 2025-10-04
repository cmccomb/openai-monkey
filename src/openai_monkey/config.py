"""Configuration helpers for the OpenAI compatibility adapter."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


def _load_json_env(name: str, *, default: Any) -> Any:
    """Return the parsed JSON value for ``name``.

    Args:
        name: Environment variable that potentially stores JSON encoded data.
        default: Value returned when the variable is not defined.

    Raises:
        ValueError: If the variable is defined but does not contain valid JSON.
    """

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive, re-raised
        raise ValueError(
            f"Environment variable {name} must contain valid JSON: {exc}"
        ) from exc


def _pick_env(*names: str, default: str) -> str:
    """Return the first defined environment variable among ``names``.

    Leading and trailing whitespace is stripped from the returned value.  When
    no variables are defined ``default`` is returned.
    """

    for name in names:
        value = os.getenv(name)
        if value is not None:
            stripped = value.strip()
            if stripped:
                return stripped
    return default


def _ensure_non_empty(value: str, *, name: str) -> str:
    """Validate that ``value`` is a non-empty string."""

    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _normalize_bool_env(name: str, *, default: bool) -> bool:
    """Convert an environment variable into a strict boolean."""

    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean flag (accepted values: 1/0/true/false/yes/no/on/off)"
    )


def _ensure_str_mapping(name: str, value: Any) -> dict[str, str]:
    """Ensure ``value`` is a mapping of strings to strings."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must decode to an object mapping strings to strings")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{name} keys and values must be strings")
        result[key] = item
    return result


def _ensure_str_set(name: str, value: Any, *, default: Iterable[str]) -> set[str]:
    """Ensure ``value`` is a collection of strings."""

    if value is None:
        return set(default)
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must decode to an array of strings")
    items: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{name} entries must be strings")
        items.add(item)
    return items


def _ensure_model_routes(name: str, value: Any) -> dict[str, dict[str, Any]]:
    """Ensure ``value`` is a mapping of route patterns to configuration dictionaries."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must decode to an object mapping strings to objects")
    normalized: dict[str, dict[str, Any]] = {}
    for pattern, cfg in value.items():
        if not isinstance(pattern, str):
            raise ValueError(f"{name} keys must be strings")
        if not isinstance(cfg, Mapping):
            raise ValueError(f"{name} entries must be objects")
        normalized[pattern] = dict(cfg)
    return normalized


@dataclass(frozen=True)
class AdapterConfig:
    """Runtime configuration for the OpenAI adapter."""

    auth_type: str
    base_url: str
    token: str
    path_map: dict[str, str]
    param_map: dict[str, str]
    drop_params: set[str]
    extra_allow: set[str]
    model_routes: dict[str, dict[str, Any]]
    disable_streaming: bool
    default_headers: dict[str, str]


def load_config() -> AdapterConfig:
    """Load adapter configuration from environment variables."""

    auth_type = os.getenv("OPENAI_AUTH_TYPE", "basic").strip().lower() or "basic"

    base_url = _ensure_non_empty(
        _pick_env(
            "OPENAI_BASE_URL",
            "OPENAI_BASIC_BASE_URL",
            default="",
        ),
        name="OPENAI_BASE_URL",
    )

    token = _ensure_non_empty(
        _pick_env(
            "OPENAI_TOKEN",
            "OPENAI_BEARER_TOKEN",
            "OPENAI_BASIC_TOKEN",
            "OPENAI_API_KEY",
            "OPENAI_KEY",
            default="",
        ),
        name="OPENAI_TOKEN",
    )

    if token == "REPLACE_ME":
        raise ValueError("OPENAI_TOKEN must be configured with a real credential")

    raw_path_map = _load_json_env(
        "OPENAI_BASIC_PATH_MAP",
        default={
            "/responses": "/api/generate",
            "/responses:stream": "/api/stream",
            "/chat/completions": "/api/generate",
            "/chat/completions:stream": "/api/stream",
        },
    )
    path_map = _ensure_str_mapping("OPENAI_BASIC_PATH_MAP", raw_path_map)

    raw_param_map = _load_json_env(
        "OPENAI_BASIC_PARAM_MAP",
        default={
            "max_tokens": "max_output_tokens",
            "top_p": "nucleus",
        },
    )
    param_map = _ensure_str_mapping("OPENAI_BASIC_PARAM_MAP", raw_param_map)

    drop_params = _ensure_str_set(
        "OPENAI_BASIC_DROP_PARAMS",
        _load_json_env("OPENAI_BASIC_DROP_PARAMS", default=["logprobs", "tool_choice"]),
        default=["logprobs", "tool_choice"],
    )

    extra_allow = _ensure_str_set(
        "OPENAI_BASIC_EXTRA_ALLOW",
        _load_json_env("OPENAI_BASIC_EXTRA_ALLOW", default=["safety_profile"]),
        default=["safety_profile"],
    )

    model_routes = _ensure_model_routes(
        "OPENAI_BASIC_MODEL_ROUTES",
        _load_json_env("OPENAI_BASIC_MODEL_ROUTES", default={}),
    )

    disable_streaming = _normalize_bool_env(
        "OPENAI_BASIC_DISABLE_STREAMING", default=False
    )

    default_headers = _ensure_str_mapping(
        "OPENAI_BASIC_HEADERS",
        _load_json_env("OPENAI_BASIC_HEADERS", default={}),
    )

    return AdapterConfig(
        auth_type=auth_type,
        base_url=base_url,
        token=token,
        path_map=path_map,
        param_map=param_map,
        drop_params=drop_params,
        extra_allow=extra_allow,
        model_routes=model_routes,
        disable_streaming=disable_streaming,
        default_headers=default_headers,
    )
