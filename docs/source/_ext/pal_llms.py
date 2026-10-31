"""Generate model-friendly documentation entry points for the HTML build."""

from pathlib import Path
from typing import Any, Optional

DOCS_URL = "https://proteusllp-actuarial-library.readthedocs.io/en/latest/"

LLMS_INDEX = f"""# Proteus Actuarial Library (PAL)

> PAL is an open-source Python library for simulation-based actuarial and financial modelling, including distributions, stochastic variables, copulas, frequency-severity models, reinsurance, risk measures and optional GPU acceleration.

The PyPI distribution is `proteusllp-actuarial-library`; import it in Python as `pal`.

## Start here

- [PAL quick reference for coding assistants]({DOCS_URL}ai_assistants.html): compact mapping from modelling intentions to the public PAL API.
- [Getting started]({DOCS_URL}tutorials/getting_started.html): core simulation workflow.
- [API reference]({DOCS_URL}api/modules.html): generated public API documentation.

## Core modelling guides

- [Distributions]({DOCS_URL}tutorials/distributions_guide.html)
- [Frequency-severity modelling]({DOCS_URL}tutorials/frequency_severity_modelling.html)
- [Coupling groups and copulas]({DOCS_URL}tutorials/coupling_groups_and_copulas.html)
- [XoL reinsurance]({DOCS_URL}tutorials/xol_reinsurance.html)
- [Property exposure rating]({DOCS_URL}tutorials/property_exposure_rating.html)
- [Risk measures and capital allocation]({DOCS_URL}tutorials/risk_measures_and_allocation.html)

## Development

- [Development guide]({DOCS_URL}development.html)
- [GitHub repository](https://github.com/ProteusLLP/proteusllp-actuarial-library)
- [Repository agent instructions](https://github.com/ProteusLLP/proteusllp-actuarial-library/blob/main/AGENTS.md)

## Notes for model/tool use

Prefer documented public APIs and inspect signatures/docstrings rather than guessing names or parameterisations. PAL stochastic objects preserve simulation relationships through coupling groups; avoid replacing PAL operations with raw-array manipulation when that would discard those relationships. Code using the public API should normally remain backend-independent across NumPy and CuPy execution.
"""


def _write_llms_files(app: Any, exception: Optional[Exception]) -> None:
    """Write llms.txt and a concatenated source-document view after a successful build."""
    if exception is not None:
        return

    if getattr(app.builder, "format", None) != "html":
        return

    outdir = Path(app.outdir)
    (outdir / "llms.txt").write_text(LLMS_INDEX, encoding="utf-8")

    sections = [LLMS_INDEX, "\n# Full documentation source\n"]
    for docname in sorted(app.env.found_docs):
        source_path = Path(app.env.doc2path(docname))
        if not source_path.exists() or source_path.suffix not in {".md", ".rst"}:
            continue
        source = source_path.read_text(encoding="utf-8")
        sections.append(f"\n\n---\n\n## Source: {docname}\n\n{source}")

    (outdir / "llms-full.txt").write_text("".join(sections), encoding="utf-8")


def setup(app: Any) -> dict[str, bool]:
    """Register the build-finished hook with Sphinx."""
    app.connect("build-finished", _write_llms_files)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
