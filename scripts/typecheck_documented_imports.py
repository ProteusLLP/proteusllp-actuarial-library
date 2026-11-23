"""Type-check PAL imports from user-facing Markdown examples.

Executable documentation snippets are exercised by pytest-codeblocks. This
additional check catches imports which work at runtime but are not advertised
as public exports by the package's typing information.
"""

from __future__ import annotations

import ast
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_FENCE = re.compile(r"^\x60\x60\x60(?:python|py)\s*\n(.*?)^\x60\x60\x60\s*$", re.MULTILINE | re.DOTALL)


def documented_pal_imports() -> list[str]:
    """Collect PAL imports from runnable Python fences in public documentation."""
    documents = [
        ROOT / "README.md",
        *(ROOT / "docs" / "tutorials").rglob("*.md"),
        *(ROOT / "docs" / "source").rglob("*.md"),
    ]
    imports: set[str] = set()
    for path in documents:
        for match in PYTHON_FENCE.finditer(path.read_text(encoding="utf-8")):
            code = match.group(1)
            try:
                tree = ast.parse(code, filename=str(path))
            except SyntaxError:
                # Some illustrative fragments are not standalone Python.
                # Executable snippets are validated separately by pytest-codeblocks.
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    is_pal = node.module == "pal" or (node.module or "").startswith("pal.")
                elif isinstance(node, ast.Import):
                    is_pal = any(alias.name == "pal" or alias.name.startswith("pal.") for alias in node.names)
                else:
                    continue
                if is_pal:
                    statement = ast.get_source_segment(code, node)
                    if statement is not None:
                        imports.add(statement)
    return sorted(imports)


def main() -> None:
    """Run the usual project type check with documented imports included."""
    statements = documented_pal_imports()
    if not statements:
        raise RuntimeError("No PAL imports found in Markdown; check the documentation scanner")

    # Each statement has its own scope so unrelated tutorials can reuse names.
    checks = ["# pyright: reportUnusedImport=false"]
    for index, statement in enumerate(statements):
        checks.append(f"def _documented_import_{index}() -> None:")
        checks.extend(f"    {line}" for line in statement.splitlines())
        checks.append("")

    # examples/ is already in pyrightconfig.json's includes, so the ordinary
    # pyright invocation checks both installed PAL imports and this temporary file.
    with tempfile.TemporaryDirectory(prefix="typecheck_docs_", dir=ROOT / "examples") as tmp:
        file = Path(tmp) / "documented_imports.py"
        file.write_text("\n".join(checks), encoding="utf-8")
        print(f"Type-checking {len(statements)} documented PAL import statements", flush=True)
        subprocess.run(["pyright"], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
