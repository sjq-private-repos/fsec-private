# Repository Guidelines

## Project Structure & Module Organization

`fsec/` contains the installable Python package. Singularity-subtraction methods
for exact exchange, MP2, and CCD are under `fsec/singularity_subtraction/`,
with reusable model functions and structure-factor code in their nested
packages. Periodic restricted CCD code is in `fsec/staggered_mesh/cc/`.
Tests live beside the relevant implementation, primarily in `*/tests/`.
`examples/` contains runnable PySCF workflows; Markdown files document methods
and implementation decisions. `pyproject.toml` defines packaging, dependencies,
and pytest configuration.

## Build, Test, and Development Commands

Use Python 3.9+ and install dependencies with:

```bash
python -m pip install -e .
```

Run the complete test suite with `python -m pytest`. Run a focused module test,
for example `python -m pytest fsec/singularity_subtraction/tests/test_mp2ss.py`.
The full periodic SCF/CCD reference tests are expensive; select them with
`python -m pytest -m slow`. The archived k-CCD reference comparison additionally
requires `FSEC_RUN_ARCHIVED_REFERENCE=1`.

Furthermore, if there exists an `fsec-312` environment or something named similarly, use it.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, readable line lengths,
`snake_case` for functions, methods, and variables, and `PascalCase` for
classes. Keep numerical array shapes, units, and physical sign conventions
explicit in code and docstrings. No formatter or linter is configured, so
preserve surrounding formatting and use standard-library-compatible imports.

As much as possible, follow PySCF code-style

## Testing Guidelines

Tests use both `unittest.TestCase` and pytest assertions/fixtures; follow the
style of the target test file. Name files `test_<feature>.py` and test methods
with descriptive `test_` names. Add regression coverage for numerical changes,
including tolerances appropriate to the calculation, and run focused tests
before the full suite. Avoid enabling slow tests by default in routine edits.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, sentence-case summaries (for example,
`Add ability to release constraint 1` and `Bug fix`). Keep commits focused and
explain scientific or numerical behavior changes in the body when needed.
Pull requests should describe the motivation, affected methods, validation
commands and results, and any performance or precision tradeoffs. Include
updated documentation or examples when public behavior changes.

## Security & Configuration Tips

Do not commit generated caches, local data, or secrets. PySCF calculations can
be compute- and memory-intensive; use small cells/meshes for development and
document any required environment variables or external data in the PR.
