"""Tests for the documented PAL import style."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]

_ALLOWED_TOP_LEVEL_IMPORTS = {
    "api",
    "config",
    "contracts",
    "copulas",
    "couplings",
    "distributions",
    "frequency_severity",
    "maths",
    "multivariate_distributions",
    "risk_measures",
    "set_default_n_sims",
    "set_random_seed",
    "stats",
    "variables",
}

_TOP_LEVEL_IMPORT = re.compile(r"from\s+pal\s+import\s+(\([^)]*\)|[^\n]+)", re.MULTILINE | re.DOTALL)
_SUBMODULE_IMPORT = re.compile(
    r"from\s+pal\.(?:contracts|copulas|couplings|distributions|frequency_severity|maths|"
    r"multivariate_distributions|risk_measures|stats|stochastic_scalar)\s+import\s+"
)


def _user_facing_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "mirror-package" / "README.md"]
    files.extend(sorted((ROOT / "examples").glob("*.py")))
    files.extend(sorted((ROOT / "examples").glob("*.ipynb")))
    files.extend(sorted((ROOT / "docs" / "tutorials").rglob("*.md")))
    files.extend(path for path in sorted((ROOT / "docs" / "source").glob("*.md")) if path.name != "development.md")
    files.extend([ROOT / "docs" / "usage.md", ROOT / "docs" / "source" / "usage.md"])
    return [path for path in files if path.exists()]


def _source_text(path: Path) -> str:
    if path.suffix != ".ipynb":
        return path.read_text(encoding="utf-8")

    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"
    )


def _normalise_import_names(import_text: str) -> set[str]:
    text = import_text.strip().removeprefix("(").removesuffix(")")
    names: set[str] = set()
    for item in text.split(","):
        name = item.strip().split(" as ", 1)[0].strip()
        name = name.split("#", 1)[0].strip()
        if name:
            names.add(name)
    return names


def test_user_examples_use_documented_pal_imports() -> None:
    violations: list[str] = []

    for path in _user_facing_files():
        text = _source_text(path)
        relative = path.relative_to(ROOT)

        if _SUBMODULE_IMPORT.search(text):
            violations.append(
                f"{relative}: import PAL domain modules from `pal`; "
                "StochasticScalar and ProteusVariable belong in `pal.variables`"
            )

        for match in _TOP_LEVEL_IMPORT.finditer(text):
            imported = _normalise_import_names(match.group(1))
            forbidden = sorted(imported - _ALLOWED_TOP_LEVEL_IMPORTS)
            if forbidden:
                violations.append(f"{relative}: top-level PAL imports are not allowed for {', '.join(forbidden)}")

        if "stochastic_scalar.StochasticScalar" in text or "from pal import stochastic_scalar" in text:
            violations.append(f"{relative}: import StochasticScalar with `from pal.variables import StochasticScalar`")

    assert not violations, "\n" + "\n".join(violations)
