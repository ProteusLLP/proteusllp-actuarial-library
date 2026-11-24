"""Check complete Google-style argument and attribute documentation in PAL.

This uses the syntax tree rather than importing modules, so it also runs
without optional dependencies, on GPU-independent developer environments.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "src/pal"
SECTION = re.compile(r"^([A-Za-z][A-Za-z ]*):\s*$")
ENTRY = re.compile(r"^\s+(\*{0,2}[A-Za-z_]\w*)(?:\s*\([^)]*\))?\s*:\s*(.*)$")


def documented(doc: str, sections: set[str]) -> set[str]:
    """Collect nonempty entries in the requested Google-style sections.

    Args:
        doc: Cleaned docstring.
        sections: Section headings to inspect, without trailing colons.

    Returns:
        Documented names, with variadic stars stripped.
    """
    result = set()
    active = False
    current = None
    indent = None
    for line in doc.splitlines():
        heading = SECTION.fullmatch(line)
        if heading:
            active = heading.group(1) in sections
            current = None
            indent = None
            continue
        if not active or not line.strip():
            continue
        match = ENTRY.match(line)
        depth = len(line) - len(line.lstrip())
        if match and (indent is None or depth == indent):
            current = match.group(1).lstrip("*")
            indent = depth
            if match.group(2).strip():
                result.add(current)
        elif current and indent is not None and depth > indent:
            result.add(current)
    return result


def decorator_name(node: ast.expr) -> str:
    """Resolve the final name in a decorator expression.

    Args:
        node: Decorator syntax, including qualified names or calls.
    """
    if isinstance(node, ast.Call):
        return decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def argument_names(node: ast.FunctionDef | ast.AsyncFunctionDef, *, is_method: bool = False) -> set[str]:
    """Collect all callable arguments except the implicit receiver.

    Args:
        node: Function or method definition to inspect.
        is_method: Whether the callable has an implicit instance or class receiver.
    """
    args = node.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    if is_method and names and names[0] in {"self", "cls"}:
        names.pop(0)
    return set(names)


def public_attributes(node: ast.ClassDef) -> set[str]:
    """Collect attributes declared as fields or assigned in the constructor.

    Args:
        node: Class definition. Inherited and private fields are excluded.

    Notes:
        Properties are documented by their own method docstrings. Enum
        member assignments are included in the class attribute contract.
    """
    names = set()
    is_enum = any(decorator_name(base).endswith("Enum") for base in node.bases)
    for member in node.body:
        if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
            names.add(member.target.id)
        elif is_enum and isinstance(member, ast.Assign):
            names.update(target.id for target in member.targets if isinstance(target, ast.Name))
        elif isinstance(member, ast.FunctionDef) and member.name == "__init__":
            for sub in ast.walk(member):
                targets = sub.targets if isinstance(sub, ast.Assign) else (
                    [sub.target] if isinstance(sub, ast.AnnAssign) else []
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        names.add(target.attr)
    return {name for name in names if not name.startswith("_")}


class Audit(ast.NodeVisitor):
    """Audit definitions without importing the package.

    Attributes:
        path: Repository-relative source path.
        scope: Stack of enclosing definition names.
        issues: Stable identifiers for documentation defects.
        class_scopes: Qualified scopes of class definitions encountered during traversal.
    """

    def __init__(self, path: str) -> None:
        """Initialize an audit for one source file.

        Args:
            path: Repository-relative file path used in diagnostic identifiers.
        """
        self.path = path
        self.scope: list[str] = []
        self.issues: set[str] = set()
        self.class_scopes: set[tuple[str, ...]] = set()

    def report(self, message: str) -> None:
        """Record a defect against the current qualified name.

        Args:
            message: Missing documentation or undocumented name.
        """
        self.issues.add(f"{self.path}:{'.'.join(self.scope)}: {message}")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Check a class docstring and its locally declared public attributes.

        Args:
            node: Class syntax tree.
        """
        self.scope.append(node.name)
        self.class_scopes.add(tuple(self.scope))
        doc = ast.get_docstring(node) or ""
        if not doc:
            self.report("missing docstring")
        attrs = documented(doc, {"Attributes"})
        for name in sorted(public_attributes(node) - attrs):
            self.report(f"missing attribute {name}")
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Check a callable, excluding overload declarations.

        Args:
            node: Function or method syntax tree.

        Notes:
            Overloads describe typing, not separate implementations. The
            implementation itself must document its callable contract.
        """
        decorators = {decorator_name(d) for d in node.decorator_list}
        is_method = tuple(self.scope) in self.class_scopes and "staticmethod" not in decorators
        self.scope.append(node.name)
        if "overload" not in decorators:
            doc = ast.get_docstring(node) or ""
            if not doc:
                self.report("missing docstring")
            expected = argument_names(node, is_method=is_method)
            present = documented(doc, {"Args", "Arguments", "Parameters"})
            for name in sorted(expected - present):
                self.report(f"missing argument {name}")
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Apply the same documentation rules to asynchronous callables.

        Args:
            node: Asynchronous function definition to inspect.
        """
        self.visit_FunctionDef(node)


def scan(path: str, content: str) -> set[str]:
    """Return the documentation defects in a Python source file.

    Args:
        path: Repository-relative source path.
        content: Python source text.
    """
    audit = Audit(path)
    audit.visit(ast.parse(content, filename=path))
    return audit.issues


def audit_tree() -> set[str]:
    """Inspect every shipped library module except runnable examples."""
    issues = set()
    for path in sorted((ROOT / SOURCE).rglob("*.py")):
        relative = path.relative_to(ROOT)
        if "examples" in relative.parts or path.name == "_version.py":
            continue
        issues.update(scan(relative.as_posix(), path.read_text(encoding="utf-8")))
    return issues


def main() -> int:
    """Report and reject every missing docstring, argument or attribute description."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", action="store_true", help="Print the complete machine-readable defect inventory")
    args = parser.parse_args()
    issues = audit_tree()
    if args.inventory:
        print(json.dumps(sorted(issues), indent=2))
        return 0
    for issue in sorted(issues):
        print(issue)
    print(f"Docstrings: {len(issues)} outstanding defects.")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
