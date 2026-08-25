#!/usr/bin/env python3
"""Validate math markup in Python docstrings, RST files, and MyST Markdown."""

from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

MATH_DIRECTIVE = re.compile(r"^(?P<indent>[ \t]*)\.\. math::(?P<argument>[ \t]+.*)?$")
STRING_LITERAL = re.compile(r"^(?P<prefix>[rubf]*)['\"]", re.IGNORECASE)
MARKDOWN_FENCE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,}|:{3,})(?P<info>.*)$")
PLAIN_TEXT_FORMULA = re.compile(r"^[ \t]*(?:[χλρΦΣ]|[A-Za-z](?:_[A-Za-z0-9{}+\-]+)?\([^)]*\)|\\[A-Za-z]+)[^=]*=")
DOCSTRING_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _indent_width(line: str) -> int:
    """Return the visual width of a line's indentation."""
    leading = line[: len(line) - len(line.lstrip(" \t"))]
    return len(leading.expandtabs())


def _validate_lines(lines: list[str], origin: str, first_line: int) -> tuple[int, list[str]]:
    """Validate math directives in a sequence of reStructuredText lines."""
    directive_count = 0
    errors: list[str] = []

    for index, line in enumerate(lines):
        match = MATH_DIRECTIVE.match(line)
        if match is None:
            continue

        directive_count += 1
        line_number = first_line + index
        location = f"{origin}:{line_number}"

        if index > 0 and lines[index - 1].strip():
            errors.append(f"{location}: math directive must be preceded by a blank line")

        if match.group("argument") and match.group("argument").strip():
            continue

        if index + 1 >= len(lines) or lines[index + 1].strip():
            errors.append(f"{location}: math directive must be followed by a blank line")
            continue

        body_index = index + 1
        while body_index < len(lines) and not lines[body_index].strip():
            body_index += 1

        directive_indent = _indent_width(line)
        if body_index >= len(lines) or _indent_width(lines[body_index]) <= directive_indent:
            errors.append(f"{location}: math directive body must be non-empty and indented")
            continue

        while body_index < len(lines):
            body_line = lines[body_index]
            if body_line.strip() and _indent_width(body_line) <= directive_indent:
                break
            if "\t" in body_line:
                errors.append(f"{origin}:{first_line + body_index}: tab character in math directive body")
            body_index += 1

    return directive_count, errors


def _validate_plain_text_formulae(lines: list[str], origin: str, first_line: int) -> list[str]:
    """Reject standalone formula-like text outside RST math directives."""
    errors: list[str] = []
    math_indent: int | None = None

    for index, line in enumerate(lines):
        directive = MATH_DIRECTIVE.match(line)
        if directive is not None:
            math_indent = _indent_width(line)
            continue

        if math_indent is not None:
            if not line.strip() or _indent_width(line) > math_indent:
                continue
            math_indent = None

        if PLAIN_TEXT_FORMULA.match(line):
            errors.append(f"{origin}:{first_line + index}: formula-like text must use a math directive")

    return errors


def _validate_python(path: Path) -> tuple[int, list[str]]:
    """Validate math directives in every docstring in a Python file."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        return 0, [f"{path}:{error.lineno}: could not parse Python source: {error.msg}"]

    directive_count = 0
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, DOCSTRING_NODES) or not node.body:
            continue
        expression = node.body[0]
        if not (
            isinstance(expression, ast.Expr)
            and isinstance(expression.value, ast.Constant)
            and isinstance(expression.value.value, str)
        ):
            continue

        docstring = inspect.cleandoc(expression.value.value)
        lines = docstring.splitlines()
        count, docstring_errors = _validate_lines(lines, str(path), expression.lineno)
        directive_count += count
        errors.extend(docstring_errors)
        errors.extend(_validate_plain_text_formulae(lines, str(path), expression.lineno))
        if count:
            literal = ast.get_source_segment(source, expression.value) or ""
            literal_match = STRING_LITERAL.match(literal.lstrip())
            if literal_match is None or "r" not in literal_match.group("prefix").lower():
                errors.append(f"{path}:{expression.lineno}: docstring containing math directives must be a raw string")

    return directive_count, errors


def _validate_rst(path: Path) -> tuple[int, list[str]]:
    """Validate math directives in a reStructuredText file."""
    return _validate_lines(path.read_text(encoding="utf-8").splitlines(), str(path), 1)


def _is_closing_fence(line: str, fence: str) -> bool:
    """Return whether a Markdown line closes the active fenced block."""
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {fence[0]} and len(stripped) >= len(fence)


def _validate_markdown(path: Path) -> tuple[int, list[str]]:
    """Validate MyST math fences and reject math markup MyST leaves literal."""
    lines = path.read_text(encoding="utf-8").splitlines()
    math_count = 0
    errors: list[str] = []
    active_fence: str | None = None
    math_fence_line = 0
    math_has_body = False

    for index, line in enumerate(lines, start=1):
        if active_fence is not None:
            if _is_closing_fence(line, active_fence):
                if math_fence_line and not math_has_body:
                    errors.append(f"{path}:{math_fence_line}: MyST math fence must have a non-empty body")
                active_fence = None
                math_fence_line = 0
                math_has_body = False
            elif math_fence_line:
                math_has_body = math_has_body or bool(line.strip())
                if "\t" in line:
                    errors.append(f"{path}:{index}: tab character in MyST math fence")
            continue

        fence_match = MARKDOWN_FENCE.match(line)
        if fence_match is not None:
            active_fence = fence_match.group("fence")
            if fence_match.group("info").strip() == "{math}":
                math_count += 1
                math_fence_line = index
            continue

        if ".. math::" in line:
            errors.append(f"{path}:{index}: use a fenced {{math}} directive in MyST Markdown")
        if ":math:`" in line:
            errors.append(f"{path}:{index}: use the MyST {{math}} role instead of the RST math role")
        if "$$" in line:
            errors.append(f"{path}:{index}: use a fenced {{math}} directive instead of $$ delimiters")

    if active_fence is not None and math_fence_line:
        errors.append(f"{path}:{math_fence_line}: unterminated MyST math fence")

    return math_count, errors


def _iter_source_files(paths: list[Path]) -> list[Path]:
    """Expand input paths into Python, RST, and Markdown source files."""
    files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix in {".py", ".rst", ".md"}:
            files.add(path)
        elif path.is_dir():
            for suffix in ("*.py", "*.rst", "*.md"):
                files.update(
                    candidate
                    for candidate in path.rglob(suffix)
                    if not any(part.startswith(".") for part in candidate.parts)
                )
        else:
            raise ValueError(f"source path does not exist or is unsupported: {path}")
    return sorted(files)


def main(arguments: list[str]) -> int:
    """Run the directive validation command."""
    if not arguments:
        print("usage: check_rst_math.py PATH [PATH ...]", file=sys.stderr)
        return 2

    try:
        files = _iter_source_files([Path(argument) for argument in arguments])
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    directive_count = 0
    errors: list[str] = []
    for path in files:
        if path.suffix == ".py":
            count, file_errors = _validate_python(path)
        elif path.suffix == ".rst":
            count, file_errors = _validate_rst(path)
        else:
            count, file_errors = _validate_markdown(path)
        directive_count += count
        errors.extend(file_errors)

    if errors:
        print("Invalid documentation mathematics:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Validated {directive_count} math directives across {len(files)} source files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
