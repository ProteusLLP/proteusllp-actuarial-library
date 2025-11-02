# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath("../.."))

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
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

# MyST parser settings for markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

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
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Autosummary settings
autosummary_generate = True  # Enable autosummary generation
autosummary_imported_members = True

# Master document (for older Sphinx/RTD compatibility)
master_doc = "index"
