"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import re
import sys

# Add the source package and local Sphinx extensions to the import path.
sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("_ext"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Proteus Actuarial Library"
copyright = "2025, ProteusLLP"
author = "James Norman"

# The version info for the project you're documenting
# This is set dynamically from the package
from importlib.metadata import version, PackageNotFoundError  # noqa

try:
    release = version("proteusllp-actuarial-library")
except (PackageNotFoundError, ImportError):
    release = "0.0.1"

version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "pal_llms",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
]

# MyST parser settings for markdown support
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "deflist",
]

# Keep long display equations readable on narrow screens. Sphinx 9 uses
# MathJax 4, whose line-breaking support is configured through this block.
mathjax4_config = {
    "output": {
        "displayOverflow": "linebreak",
        "linebreaks": {
            "inline": True,
            "width": "100%",
        },
    },
}

templates_path = ["_templates"]
exclude_patterns = [
    "api/pal*.rst",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["css/proteus.css", "css/sidebar-search-fix.css"]
html_logo = "_static/logo/proteus-white-horizontal.svg"
html_favicon = "_static/logo/pal-favicon.svg"
html_title = "PAL Documentation"
html_short_title = "PAL"
html_last_updated_fmt = "%d %B %Y"
html_show_sourcelink = False

# Theme options
html_theme_options = {
    "announcement": "PAL is in active development — feedback and contributions are welcome.",
    "navigation_with_keys": True,
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/ProteusLLP/proteusllp-actuarial-library/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#1d4ed8",
        "color-brand-content": "#1d4ed8",
        "color-api-name": "#001a64",
        "color-api-pre-name": "#475569",
        "color-api-background": "#f8fafc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8fb8ff",
        "color-brand-content": "#8fb8ff",
        "color-api-name": "#b8d2ff",
        "color-api-pre-name": "#94a3b8",
        "color-api-background": "#10182d",
    },
}

# Avoid copying interactive prompts when readers copy example code.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Extension configuration -------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "__weakref__,__dict__,__module__",
    "inherited-members": True,
}

# Additional autodoc settings
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_class_signature = "mixed"

# Mock imports for dependencies that aren't installed during docs build
autodoc_mock_imports = [
    "cupy",
    "cupyx",
    "cupyx.scipy",
    "cupyx.scipy.stats",
    "cupyx.scipy.special",
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx mapping
# In many environments (CI, offline devcontainers) we cannot fetch inventories.
# Keep it opt-in to avoid noisy warnings when building with -W.
if os.environ.get("PAL_DOCS_INTERSPHINX") == "1":
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    }
else:
    intersphinx_mapping = {}

# Autosummary settings
autosummary_generate = True  # Enable autosummary generation
autosummary_imported_members = True

# Master document (for older Sphinx/RTD compatibility)
master_doc = "index"


_INLINE_DOLLAR_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")


def _convert_dollar_math_in_docstring(lines: list[str]) -> list[str]:
    """Convert $...$ / $$...$$ to Sphinx math in autodoc docstrings.

    Sphinx renders math via docutils nodes. MyST's ``dollarmath`` only applies to
    Markdown sources, not to Python docstrings (which are parsed as reST).

    This hook lets us write dollar-math consistently in docstrings and tutorials.
    It is intentionally conservative: it avoids transforming inside fenced or
    literal code blocks.
    """
    out: list[str] = []
    in_fenced_code = False
    literal_pending = False
    in_literal_block = False
    in_display_math = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            in_fenced_code = not in_fenced_code
            out.append(line)
            continue

        if in_fenced_code:
            out.append(line)
            continue

        if in_display_math:
            if stripped == "$$":
                in_display_math = False
                out.append("")
            else:
                out.append("   " + line)
            continue

        if stripped == "$$":
            in_display_math = True
            out.append(".. math::")
            out.append("")
            continue

        if in_literal_block:
            if stripped and not line[:1].isspace():
                in_literal_block = False
            else:
                out.append(line)
                continue

        if literal_pending:
            if stripped == "":
                out.append(line)
                continue
            if line[:1].isspace():
                in_literal_block = True
                out.append(line)
                continue
            literal_pending = False

        if stripped.startswith(".. code-block::") or stripped.startswith(".. literalinclude::"):
            literal_pending = True
            out.append(line)
            continue

        if stripped.endswith("::"):
            literal_pending = True
            out.append(line)
            continue

        def repl(match: re.Match[str]) -> str:
            expr = match.group(1).strip()
            return f":math:`{expr}`"

        out.append(_INLINE_DOLLAR_MATH_RE.sub(repl, line))

    return out


def setup(app):
    """Setup function for the Sphinx extension."""

    def _autodoc_process_docstring(app, what, name, obj, options, lines):  # noqa: ANN001
        converted = _convert_dollar_math_in_docstring(list(lines))
        lines[:] = converted

    app.connect("autodoc-process-docstring", _autodoc_process_docstring)
