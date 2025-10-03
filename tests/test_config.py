from __future__ import annotations

import importlib
import os
from typing import Any

import pytest

from openai_monkey import config as config_module


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
        "OPENAI_TOKEN": "super-secret",
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


def test_load_config_rejects_non_string_headers() -> None:
    """Mappings that are not string to string should be rejected."""

    env = _baseline_env()
    env["OPENAI_BASIC_HEADERS"] = '{"X-Test": 1}'

    with pytest.raises(ValueError, match="strings"):
        _reload_config_env(**env)


def test_load_config_rejects_placeholder_token() -> None:
    """Placeholder tokens should not be accepted."""

    env = _baseline_env()
    env["OPENAI_TOKEN"] = "REPLACE_ME"

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

    assert config.auth_type == "bearer"
    assert config.base_url == "https://secure.local"
    assert config.token == "super-secret"
    assert config.default_headers == {"X-Test": "true"}
    assert config.drop_params == {"a", "b"}
    assert config.extra_allow == {"safety"}
    assert config.disable_streaming is True
