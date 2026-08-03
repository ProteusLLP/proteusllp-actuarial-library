#!/usr/bin/env python3
"""Validate reStructuredText math directives in Python docstrings and RST files."""

from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

MATH_DIRECTIVE = re.compile(r"^(?P<indent>[ \t]*)\.\. math::(?P<argument>[ \t]+.*)?$")
STRING_LITERAL = re.compile(r"^(?P<prefix>[rubf]*)['\"]", re.IGNORECASE)
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
        count, docstring_errors = _validate_lines(docstring.splitlines(), str(path), expression.lineno)
        directive_count += count
        errors.extend(docstring_errors)
        if count:
            literal = ast.get_source_segment(source, expression.value) or ""
            literal_match = STRING_LITERAL.match(literal.lstrip())
            if literal_match is None or "r" not in literal_match.group("prefix").lower():
                errors.append(f"{path}:{expression.lineno}: docstring containing math directives must be a raw string")

    return directive_count, errors


def _validate_rst(path: Path) -> tuple[int, list[str]]:
    """Validate math directives in a reStructuredText file."""
    return _validate_lines(path.read_text(encoding="utf-8").splitlines(), str(path), 1)


def _iter_source_files(paths: list[Path]) -> list[Path]:
    """Expand input paths into Python and RST source files."""
    files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix in {".py", ".rst"}:
            files.add(path)
        elif path.is_dir():
            files.update(
                candidate
                for candidate in path.rglob("*.py")
                if not any(part.startswith(".") for part in candidate.parts)
            )
            files.update(
                candidate
                for candidate in path.rglob("*.rst")
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
        else:
            count, file_errors = _validate_rst(path)
        directive_count += count
        errors.extend(file_errors)

    if errors:
        print("Invalid reStructuredText math directives:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Validated {directive_count} math directives across {len(files)} source files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
