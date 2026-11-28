"""Check complete and well-formed Google-style docstrings in PAL.

Every callable must document each of its arguments, and nothing else, and
every class must document its locally declared public attributes. Entries
must use the Google form ``name: description``. NumPy-style ``name : type``
entries, bulleted lists and unindented section bodies are reported because
Sphinx Napoleon renders them incorrectly.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "src/pal"
SECTION = re.compile(r"^([A-Za-z][A-Za-z ]*):\s*$")
# The optional parenthesis is the Google form "name (type): description".
ENTRY = re.compile(r"^(\*{0,2}[A-Za-z_]\w*)(?: \([^)]*\))?:(?:\s+(.*))?$")
ARGUMENT_SECTIONS = {"Args", "Arguments", "Parameters"}
ATTRIBUTE_SECTIONS = {"Attributes"}
# Napoleon sections whose bodies must be indented beneath the heading.
GOOGLE_SECTIONS = {
    *ARGUMENT_SECTIONS,
    *ATTRIBUTE_SECTIONS,
    "Example",
    "Examples",
    "Keyword Args",
    "Keyword Arguments",
    "Methods",
    "Note",
    "Notes",
    "Other Parameters",
    "Raises",
    "References",
    "Return",
    "Returns",
    "See Also",
    "Todo",
    "Warning",
    "Warnings",
    "Warns",
    "Yield",
    "Yields",
}


@dataclass
class Entries:
    """Names and formatting defects found in docstring sections.

    Attributes:
        described: Names with a nonempty description, with variadic stars stripped.
        names: Every entry name in order of appearance, including repeats.
        malformed: Descriptions of formatting defects.
    """

    described: set[str] = field(default_factory=set)
    names: list[str] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)


def section_entries(doc: str, sections: set[str]) -> Entries:
    """Parse the entries of the requested Google-style sections.

    Within a section, each line at the indentation of its first entry must be
    a ``name: description`` entry; deeper lines continue the current entry.
    Any known section whose first line is not indented is also reported.

    Args:
        doc: Cleaned docstring.
        sections: Section headings to inspect, without trailing colons.
    """
    result = Entries()
    active = False
    heading = None
    awaiting_body = False
    current = None
    indent = None
    for line in doc.splitlines():
        match = SECTION.fullmatch(line)
        if match:
            heading = match.group(1)
            active = heading in sections
            awaiting_body = heading in GOOGLE_SECTIONS
            current = None
            indent = None
            continue
        if not line.strip():
            continue
        depth = len(line) - len(line.lstrip())
        if awaiting_body and depth == 0:
            result.malformed.append(f"unindented {heading} section")
        awaiting_body = False
        if not active:
            continue
        if depth == 0:
            # Unindented text after an entry section ends that section.
            active = False
            continue
        if indent is None:
            indent = depth
        if depth > indent:
            if current:
                result.described.add(current)
            continue
        entry = ENTRY.match(line.strip())
        if depth < indent or not entry:
            result.malformed.append(f"malformed {heading} entry {line.strip()!r}")
            current = None
            continue
        current = entry.group(1).lstrip("*")
        result.names.append(current)
        if entry.group(2):
            result.described.add(current)
    return result


def duplicates(names: list[str]) -> list[str]:
    """Return names that occur more than once, in sorted order.

    Args:
        names: Entry names in order of appearance.
    """
    return sorted({name for name in names if names.count(name) > 1})


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
                targets = (
                    sub.targets
                    if isinstance(sub, ast.Assign)
                    else ([sub.target] if isinstance(sub, ast.AnnAssign) else [])
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
        entries = section_entries(doc, ATTRIBUTE_SECTIONS)
        for name in sorted(public_attributes(node) - entries.described):
            self.report(f"missing attribute {name}")
        for name in duplicates(entries.names):
            self.report(f"duplicate attribute {name}")
        for message in entries.malformed:
            self.report(message)
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
            entries = section_entries(doc, ARGUMENT_SECTIONS)
            for name in sorted(expected - entries.described):
                self.report(f"missing argument {name}")
            for name in sorted(set(entries.names) - expected):
                self.report(f"unknown argument {name}")
            for name in duplicates(entries.names):
                self.report(f"duplicate argument {name}")
            for message in entries.malformed:
                self.report(message)
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
