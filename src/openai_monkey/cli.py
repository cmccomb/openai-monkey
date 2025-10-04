"""Command line utilities for the openai-monkey package."""

from __future__ import annotations

import argparse
import ast
import logging
import sys
import sysconfig
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Replacement:
    start: int
    end: int
    text: str


def _line_offsets(source: str) -> list[int]:
    """Return the starting offset for each line in *source*."""

    offsets: list[int] = []
    position = 0
    for line in source.splitlines(keepends=True):
        offsets.append(position)
        position += len(line)
    if not offsets:
        offsets.append(0)
    return offsets


def _absolute_offset(offsets: Sequence[int], lineno: int, col: int) -> int:
    """Translate a line/column pair into an absolute character offset."""

    return offsets[lineno - 1] + col


def _transform_import(node: ast.Import) -> tuple[ast.Import, bool]:
    """Return a transformed import node and whether it changed."""

    new_aliases: list[ast.alias] = []
    changed = False
    for alias in node.names:
        if alias.name == "openai":
            new_aliases.append(
                ast.alias(name="openai_monkey", asname=alias.asname or "openai")
            )
            changed = True
        else:
            new_aliases.append(alias)
    return ast.Import(names=new_aliases), changed


def _transform_import_from(node: ast.ImportFrom) -> tuple[ast.ImportFrom, bool]:
    """Return a transformed import-from node and whether it changed."""

    if not node.module or node.level != 0:
        return node, False

    if node.module == "openai":
        return ast.ImportFrom(module="openai_monkey", names=node.names, level=0), True

    if node.module.startswith("openai."):
        new_module = "openai_monkey" + node.module[len("openai") :]
        return ast.ImportFrom(module=new_module, names=node.names, level=0), True

    return node, False


def _rewrite_source(source: str) -> tuple[str, bool]:
    """Rewrite import statements in *source* to reference ``openai_monkey``."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover - defensive guard
        LOGGER.warning("Skipping file with syntax error: %s", exc)
        return source, False

    replacements: list[_Replacement] = []
    offsets = _line_offsets(source)

    for node in ast.walk(tree):
        replacement_node: ast.AST
        if isinstance(node, ast.Import):
            replacement_node, changed = _transform_import(node)
            if not changed:
                continue
        elif isinstance(node, ast.ImportFrom):
            replacement_node, changed = _transform_import_from(node)
            if not changed:
                continue
        else:
            continue

        if node.end_lineno is None or node.end_col_offset is None:
            continue

        start = _absolute_offset(offsets, node.lineno, node.col_offset)
        end = _absolute_offset(offsets, node.end_lineno, node.end_col_offset)
        replacements.append(
            _Replacement(start=start, end=end, text=ast.unparse(replacement_node))
        )

    if not replacements:
        return source, False

    new_source = source
    for replacement in sorted(replacements, key=lambda repl: repl.start, reverse=True):
        new_source = (
            new_source[: replacement.start]
            + replacement.text
            + new_source[replacement.end :]
        )

    return new_source, True


def monkeyify_repository(path: Path, *, dry_run: bool = False) -> list[Path]:
    """Rewrite ``openai`` imports under *path* to use ``openai_monkey``.

    Args:
        path: Base directory that will be searched recursively.
        dry_run: When ``True`` no files are modified.

    Returns:
        A sorted list of files that would be (or were) rewritten.
    """

    if not path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Repository path must be a directory: {path}")

    changed: list[Path] = []
    for python_file in sorted(path.rglob("*.py")):
        source = python_file.read_text(encoding="utf-8")
        rewritten, modified = _rewrite_source(source)
        if not modified:
            continue
        changed.append(python_file)
        if dry_run:
            continue
        python_file.write_text(rewritten, encoding="utf-8")

    return changed


def _create_alias_file(site_packages: Path) -> Path:
    """Create a ``.pth`` alias that exposes ``openai_monkey`` as ``openai``."""

    site_packages.mkdir(parents=True, exist_ok=True)
    alias_path = site_packages / "openai_monkey_as_openai.pth"
    alias_contents = (
        "import importlib, sys; "
        "sys.modules.setdefault('openai', importlib.import_module('openai_monkey'))\n"
    )
    alias_path.write_text(alias_contents, encoding="utf-8")
    return alias_path


def install_alias(*, site_packages: Path | None = None) -> Path:
    """Install the ``openai`` alias for ``openai_monkey`` in ``site-packages``."""

    try:
        __import__("openai_monkey")
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("openai_monkey must be installed before aliasing") from exc

    if site_packages is None:
        site_packages = Path(sysconfig.get_path("purelib"))

    return _create_alias_file(site_packages)


def monkeyify_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point that rewrites ``openai`` imports in a repository."""

    parser = argparse.ArgumentParser(
        prog="openai-monkey-ify",
        description="Rewrite openai imports in a repository to use openai_monkey.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        type=Path,
        help="Repository root to rewrite (defaults to current working directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would change without modifying them.",
    )

    args = parser.parse_args(argv)

    try:
        changed = monkeyify_repository(args.path.resolve(), dry_run=args.dry_run)
    except (FileNotFoundError, NotADirectoryError) as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - unreachable because parser.error exits

    if changed:
        for file_path in changed:
            print(file_path)
        message = "Updated" if not args.dry_run else "Would update"
        print(f"{message} {len(changed)} file(s).")
    else:
        message = (
            "No files needed changes." if not args.dry_run else "No files would change."
        )
        print(message)

    return 0


def install_alias_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point that installs the ``openai`` alias for ``openai_monkey``."""

    parser = argparse.ArgumentParser(
        prog="openai-monkey-install-openai",
        description="Install a site-packages alias so 'import openai' loads openai_monkey.",
    )
    parser.add_argument(
        "--site-packages",
        type=Path,
        help="Override the target site-packages directory (primarily for testing).",
    )

    args = parser.parse_args(argv)

    try:
        alias_path = install_alias(site_packages=args.site_packages)
    except RuntimeError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - parser.error exits

    print(f"Alias installed at {alias_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution guard
    sys.exit(monkeyify_main())
