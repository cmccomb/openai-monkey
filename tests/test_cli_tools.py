"""Tests for the command-line utilities shipped with openai-monkey."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_BASE_URL", "https://internal.company.ai")
os.environ.setdefault("OPENAI_TOKEN", "TEST_TOKEN")

from openai_monkey import cli


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

    assert changed == [source]
    rewritten = source.read_text(encoding="utf-8")
    assert "import openai_monkey as openai" in rewritten
    assert "import openai_monkey as openai_client" in rewritten
    assert "import openai_monkey as openai, sys" in rewritten
    assert "from openai_monkey import OpenAI, AsyncOpenAI" in rewritten
    assert "from openai_monkey.types import ChatCompletion" in rewritten


def test_monkeyify_repository_supports_dry_run(tmp_path: Path) -> None:
    """When dry_run is enabled, files are not modified."""

    source = tmp_path / "module.py"
    original = "import openai\n"
    _write(source, original)

    changed = cli.monkeyify_repository(tmp_path, dry_run=True)

    assert changed == [source]
    assert source.read_text(encoding="utf-8") == original


def test_install_alias_creates_pth_file(tmp_path: Path) -> None:
    """The alias installer should produce an executable .pth file."""

    alias_path = cli.install_alias(site_packages=tmp_path)

    assert alias_path.exists()
    contents = alias_path.read_text(encoding="utf-8")
    assert "import importlib, sys" in contents

    previous = sys.modules.get("openai")
    try:
        sys.modules.pop("openai", None)
        exec(contents, {"sys": sys, "importlib": importlib})
        assert sys.modules["openai"] is importlib.import_module("openai_monkey")
    finally:
        if previous is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = previous
