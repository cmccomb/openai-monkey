from __future__ import annotations

import importlib
import os
from typing import Any

import pytest
from openai_monkey import config as config_module

_BASELINE_CREDENTIAL = "super-secret"


def _ensure(condition: bool, message: str) -> None:
    """Raise ``AssertionError`` with ``message`` when ``condition`` is ``False``."""

    if not condition:
        raise AssertionError(message)


def _reload_config_env(**env: Any) -> Any:
    """Reload the configuration module with the provided environment values."""

    previous: dict[str, str | None] = {}
    for key, value in env.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        importlib.reload(config_module)
        return config_module.load_config()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(config_module)


def _baseline_env() -> dict[str, str]:
    """Return a minimal valid configuration for testing."""

    return {
        "OPENAI_BASE_URL": "https://secure.local",
        "OPENAI_TOKEN": _BASELINE_CREDENTIAL,
    }


def test_load_config_rejects_invalid_disable_streaming_flag() -> None:
    """Non-boolean values for the streaming flag should raise an error."""

    env = _baseline_env()
    env["OPENAI_BASIC_DISABLE_STREAMING"] = "maybe"

    with pytest.raises(ValueError, match="boolean flag"):
        _reload_config_env(**env)


def test_load_config_rejects_malformed_json_payload() -> None:
    """Invalid JSON configuration payloads should raise informative errors."""

    env = _baseline_env()
    env["OPENAI_BASIC_HEADERS"] = "not-json"

    with pytest.raises(ValueError, match="valid JSON"):
        _reload_config_env(**env)


def test_load_config_requires_base_url() -> None:
    """The base URL must be configured explicitly to avoid silent fallbacks."""

    env = _baseline_env()
    env["OPENAI_BASE_URL"] = ""
    env["OPENAI_BASIC_BASE_URL"] = ""

    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        _reload_config_env(**env)


def test_load_config_rejects_non_string_headers() -> None:
    """Mappings that are not string to string should be rejected."""

    env = _baseline_env()
    env["OPENAI_BASIC_HEADERS"] = '{"X-Test": 1}'

    with pytest.raises(ValueError, match="strings"):
        _reload_config_env(**env)


def test_load_config_rejects_placeholder_token() -> None:
    """Placeholder tokens should not be accepted."""

    env = _baseline_env()
    env["OPENAI_TOKEN"] = config_module._PLACEHOLDER_SENTINEL

    with pytest.raises(ValueError, match="real credential"):
        _reload_config_env(**env)


def test_load_config_accepts_valid_payload() -> None:
    """A complete, valid configuration should be returned as a dataclass."""

    env = _baseline_env()
    env.update(
        {
            "OPENAI_AUTH_TYPE": "Bearer",
            "OPENAI_BASIC_HEADERS": '{"X-Test": "true"}',
            "OPENAI_BASIC_DROP_PARAMS": '["a", "b"]',
            "OPENAI_BASIC_EXTRA_ALLOW": '["safety"]',
            "OPENAI_BASIC_DISABLE_STREAMING": "true",
        }
    )

    config = _reload_config_env(**env)

    _ensure(config.auth_type == "bearer", f"Unexpected auth_type: {config.auth_type!r}")
    _ensure(
        config.base_url == "https://secure.local",
        f"Unexpected base_url: {config.base_url!r}",
    )
    _ensure(config.token == _BASELINE_CREDENTIAL, "Token should match baseline env")
    _ensure(
        config.default_headers == {"X-Test": "true"},
        f"Unexpected default headers: {config.default_headers!r}",
    )
    _ensure(
        config.drop_params == {"a", "b"},
        f"Unexpected drop params: {config.drop_params!r}",
    )
    _ensure(
        config.extra_allow == {"safety"},
        f"Unexpected extra_allow: {config.extra_allow!r}",
    )
    _ensure(
        config.disable_streaming is True,
        "disable_streaming flag should normalize truthy values",
    )
