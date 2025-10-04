from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_DEFAULT_CREDENTIAL = "test-credential"


@pytest.fixture
def configure_adapter(request: pytest.FixtureRequest) -> Callable[..., Any]:
    """Return a helper for reloading ``openai_monkey`` with a custom config."""

    original_env: dict[str, str | None] = {}

    def _setenv(name: str, value: str | None) -> None:
        if name not in original_env:
            original_env[name] = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def _configure(
        *,
        base_url: str = "https://mock.local",
        token: str | None = None,
        path_map: dict[str, str] | None = None,
        param_map: dict[str, str] | None = None,
        drop_params: list[str] | None = None,
        extra_allow: list[str] | None = None,
        default_headers: dict[str, str] | None = None,
        disable_streaming: bool = False,
        auth_type: str = "basic",
    ) -> Any:
        _setenv("OPENAI_AUTH_TYPE", auth_type)
        _setenv("OPENAI_BASE_URL", base_url)
        credential = token if token is not None else _DEFAULT_CREDENTIAL
        _setenv("OPENAI_TOKEN", credential)

        if path_map is not None:
            _setenv("OPENAI_BASIC_PATH_MAP", json.dumps(path_map))
        if param_map is not None:
            _setenv("OPENAI_BASIC_PARAM_MAP", json.dumps(param_map))
        if drop_params is not None:
            _setenv("OPENAI_BASIC_DROP_PARAMS", json.dumps(drop_params))
        if extra_allow is not None:
            _setenv("OPENAI_BASIC_EXTRA_ALLOW", json.dumps(extra_allow))
        if default_headers is not None:
            _setenv("OPENAI_BASIC_HEADERS", json.dumps(default_headers))
        _setenv("OPENAI_BASIC_DISABLE_STREAMING", "1" if disable_streaming else "0")

        import openai_monkey

        return importlib.reload(openai_monkey)

    def _cleanup() -> None:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if "openai_monkey" in sys.modules:
            import openai_monkey

            importlib.reload(openai_monkey)

    request.addfinalizer(_cleanup)
    return _configure
