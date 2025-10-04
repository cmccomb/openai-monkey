"""Regression tests ensuring public modules carry descriptive docstrings."""

from collections.abc import Callable
from importlib import import_module
from typing import Any


def _ensure(condition: bool, message: str) -> None:
    """Raise ``AssertionError`` with ``message`` when ``condition`` is ``False``."""

    if not condition:
        raise AssertionError(message)


def test_adapter_docstrings_present(configure_adapter: Callable[..., Any]) -> None:
    """The adapter exposes descriptive docstrings for modules and functions."""

    module = configure_adapter()
    _ensure(
        module.__doc__ is not None, "openai_monkey module docstring should be defined"
    )

    adapter = import_module("openai_monkey.adapter")
    _ensure(
        adapter.__doc__ is not None,
        "adapter module docstring should describe the patch",
    )
    _ensure(
        adapter.apply_adapter_patch.__doc__ is not None,
        "apply_adapter_patch must explain the monkeypatching process",
    )


def test_config_module_docstring_present() -> None:
    """Configuration helpers are documented for discoverability."""

    config = import_module("openai_monkey.config")
    _ensure(
        config.__doc__ is not None,
        "config module docstring should describe its purpose",
    )
