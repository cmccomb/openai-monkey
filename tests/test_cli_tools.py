"""Tests for the command-line utilities shipped with openai-monkey."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_BASE_URL", "https://internal.company.ai")
os.environ.setdefault("OPENAI_TOKEN", "TEST_TOKEN")

from openai_monkey import cli


def _ensure(condition: bool, message: str) -> None:
    """Raise ``AssertionError`` with ``message`` when ``condition`` is ``False``."""

    if not condition:
        raise AssertionError(message)


def _write(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")


def test_monkeyify_repository_rewrites_common_imports(tmp_path: Path) -> None:
    """The repository rewriter should cover a variety of import styles."""

    source = tmp_path / "sample.py"
    _write(
        source,
        "\n".join(
            [
                "import os",
                "import openai",
                "import openai as openai_client",
                "import openai, sys",
                "from openai import OpenAI, AsyncOpenAI",
                "from openai.types import ChatCompletion",
                "",
                "print(OpenAI, AsyncOpenAI, openai_client, os, sys)",
            ]
        )
        + "\n",
    )

    changed = cli.monkeyify_repository(tmp_path)

    _ensure(changed == [source], f"Expected changed files to equal {[source]!r}")
    rewritten = source.read_text(encoding="utf-8")
    _ensure(
        "import openai_monkey as openai" in rewritten,
        "Expected import alias for openai",
    )
    _ensure(
        "import openai_monkey as openai_client" in rewritten,
        "Expected renamed alias for openai_client",
    )
    _ensure(
        "import openai_monkey as openai, sys" in rewritten,
        "Expected multi-import rewrite",
    )
    _ensure(
        "from openai_monkey import OpenAI, AsyncOpenAI" in rewritten,
        "Expected direct import rewrite",
    )
    _ensure(
        "from openai_monkey.types import ChatCompletion" in rewritten,
        "Expected nested import rewrite",
    )


def test_monkeyify_repository_supports_dry_run(tmp_path: Path) -> None:
    """When dry_run is enabled, files are not modified."""

    source = tmp_path / "module.py"
    original = "import openai\n"
    _write(source, original)

    changed = cli.monkeyify_repository(tmp_path, dry_run=True)

    _ensure(changed == [source], f"Expected dry-run to report {[source]!r}")
    _ensure(
        source.read_text(encoding="utf-8") == original,
        "Dry-run should not modify source file",
    )


def test_install_alias_creates_pth_file(tmp_path: Path) -> None:
    """The alias installer should produce an executable .pth file."""

    alias_path = cli.install_alias(site_packages=tmp_path)

    _ensure(alias_path.exists(), "Alias path should exist after installation")
    contents = alias_path.read_text(encoding="utf-8")
    expected_contents = (
        "import importlib, sys; "
        "sys.modules.setdefault('openai', importlib.import_module('openai_monkey'))\n"
    )
    _ensure(
        contents == expected_contents,
        f"Alias file contents unexpected: {contents!r}",
    )

    previous = sys.modules.get("openai")
    try:
        sys.modules.pop("openai", None)
        module = importlib.import_module("openai_monkey")
        result = sys.modules.setdefault("openai", module)
        _ensure(result is module, "Alias execution should install openai module")
    finally:
        if previous is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = previous
