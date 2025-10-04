from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, Optional

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def configure_adapter(request: pytest.FixtureRequest) -> Callable[..., Any]:
    """Return a helper for reloading ``openai_monkey`` with a custom config."""

    original_env: dict[str, Optional[str]] = {}

    def _setenv(name: str, value: Optional[str]) -> None:
        if name not in original_env:
            original_env[name] = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def _configure(
        *,
        base_url: str = "https://mock.local",
        token: str = "TEST_TOKEN",
        path_map: Optional[Dict[str, str]] = None,
        param_map: Optional[Dict[str, str]] = None,
        drop_params: Optional[list[str]] = None,
        extra_allow: Optional[list[str]] = None,
        default_headers: Optional[Dict[str, str]] = None,
        disable_streaming: bool = False,
        auth_type: str = "basic",
    ) -> Any:
        _setenv("OPENAI_AUTH_TYPE", auth_type)
        _setenv("OPENAI_BASE_URL", base_url)
        _setenv("OPENAI_TOKEN", token)

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
