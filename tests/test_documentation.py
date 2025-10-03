"""Regression tests ensuring public modules carry descriptive docstrings."""

from importlib import import_module


def test_adapter_docstrings_present(configure_adapter):
    """The adapter exposes descriptive docstrings for modules and functions."""

    module = configure_adapter()
    assert module.__doc__, "openai_monkey module docstring should be defined"

    adapter = import_module("openai_monkey.adapter")
    assert adapter.__doc__, "adapter module docstring should describe the patch"
    assert (
        adapter.apply_adapter_patch.__doc__
    ), "apply_adapter_patch must explain the monkeypatching process"


def test_config_module_docstring_present():
    """Configuration helpers are documented for discoverability."""

    config = import_module("openai_monkey.config")
    assert config.__doc__, "config module docstring should describe its purpose"
