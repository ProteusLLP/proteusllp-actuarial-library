# Documentation

This directory contains the documentation for the Proteus Actuarial Library.

## Hosted Documentation

The documentation is hosted on Read the Docs:
https://proteusllp-actuarial-library-private.readthedocs.io/

## Building Documentation Locally

### Prerequisites

Install the documentation dependencies:

```bash
pdm install -G docs
```

Or with pip:

```bash
pip install -e .[docs]
```

### Build HTML Documentation

From the `docs` directory:

```bash
cd docs
make html
```

The built documentation will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser to view.

### Other Build Formats

```bash
make pdf      # Build PDF documentation
make epub     # Build ePub documentation
make clean    # Clean build directory
make help     # Show all available targets
```

## Documentation Structure

- `source/` - Sphinx source files
  - `conf.py` - Sphinx configuration
  - `index.rst` - Main documentation index
  - `usage.md` - Usage guide (Markdown)
  - `development.md` - Development guide (Markdown)
  - `api/` - API reference documentation
  - `_static/` - Static files (CSS, images, etc.)
  - `_templates/` - Custom Sphinx templates
- `usage.md` - Original usage guide (copied to source/)
- `development.md` - Original development guide (copied to source/)

## Read the Docs Configuration

The `.readthedocs.yaml` file in the repository root configures how Read the Docs builds the documentation:

- Python version: 3.13
- Formats: HTML, PDF, ePub
- Dependencies: Installed from `[docs]` optional dependencies

## Updating Documentation

1. Edit the relevant `.rst` or `.md` files in `docs/source/`
2. Build locally to preview changes: `make html`
3. Commit and push changes
4. Read the Docs will automatically rebuild the documentation

## Adding New Pages

1. Create a new `.rst` or `.md` file in `docs/source/`
2. Add it to the appropriate `toctree` directive in `index.rst` or other parent pages
3. Build and test locally

## API Documentation

API documentation is automatically generated from docstrings using Sphinx autodoc. To add documentation for a new module:

1. Add an `.rst` file in `docs/source/api/`
2. Use autodoc directives to include the module
3. Add it to `docs/source/api/modules.rst`
