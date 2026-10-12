# PAL Documentation Agent Guide

These rules apply to changes under `docs/` in addition to the root `AGENTS.md`.

## Audience and Language

Write for actuaries, quantitative analysts and Python users. Explain what a public function or model means to the user before discussing how it is implemented.

Avoid exposing internal implementation language in user-facing descriptions when it does not help the user. In particular, do not describe ordinary operations in terms of "backend ndarrays", internal wrapper machinery or coupling metadata unless that mechanism is itself the topic of the documentation.

When modifying existing documentation, first identify the purpose of the surrounding sentence, paragraph and section from the reader's perspective. Preserve useful existing explanations unless the task requires them to change. A change to an API, import path or implementation detail should not cause unrelated conceptual prose to be rewritten around that change.

Review edited prose as part of the whole document, not only as a description of the current code change. Remove task-centric wording such as discussion of what is "exposed", "re-exported", "moved" or "now available" when a user only needs to know what an object represents and how to use it. Prefer the smallest wording change that keeps the document accurate and natural.

## Mathematical Documentation

- Keep formulas consistent with the implemented parameterisation.
- Check symbols, parameter domains and limiting cases against code and a reliable reference.
- Use display mathematics for substantive formulas rather than ASCII approximations.
- Treat rendered mathematics as part of correctness: a formula that is valid in source but broken in Sphinx is not complete.

## Examples

Prefer small canonical examples that demonstrate one concept clearly. Documentation code should use the public API a normal PAL user should copy.

Introduce classes and modelling concepts by explaining what they represent and why a user would use them. Import-path guidance is secondary and should not replace the conceptual introduction.

Use top-level imports for PAL's four core modelling abstractions: `from pal import ProteusVariable, StochasticScalar, FreqSevSims, FrequencySeverityModel`. Continue to import domain-specific classes such as distributions, copulas and contracts directly from their documented submodules rather than flattening them into `pal`. When an example uses top-level configuration helpers such as `config` or `set_random_seed`, make sure those names are imported in the executable block or in the continuation chain on which it relies.

Executable code blocks should remain compatible with `pytest-codeblocks`. Each fenced block is executed independently by default. If a block deliberately relies on imports, variables or other state established by an earlier block, place `<!--pytest-codeblocks:cont-->` immediately before it so the sequence is tested in one shared namespace.

Use `<!--pytest-codeblocks:skip-->` only when a block is intentionally non-executable in the test environment, such as illustrative development shell commands, external-resource examples, interactive display or unavailable hardware. Do not skip a user-facing example merely to make CI green; make executable examples self-contained or use `cont` as appropriate. Be especially careful with shell blocks in repository guidance because `pytest-codeblocks` can execute them and accidentally make the test suite slow or recursive.

When a tutorial calculates a result and visualisation is useful, show the chart in the built documentation rather than only showing plotting code.

## Charts

Use Plotly for documentation illustrations and PAL's Proteus Plotly styling/template where available. Do not introduce Matplotlib for user-facing documentation charts unless a task explicitly requires it.

Keep Proteus branding subtle and consistent with the existing documentation design.

## API Documentation

Signatures and type annotations are the authoritative source for types. Do not repeat type names in prose merely to duplicate the signature.

Public docstrings should cover:

- the purpose and statistical/actuarial interpretation;
- non-obvious parameter meaning;
- return semantics;
- important exceptions or constraints;
- references where the implementation depends on a specific algorithm or paper.

## Validation

Build documentation with warnings treated seriously. Run the documentation code-block tests after changing examples or public API documentation. The normal CI test matrix runs `pytest` with `--codeblocks`, so executable documentation examples are covered there.

The AI quick reference at `docs/source/ai_assistants.md` is intentionally concise and should remain a high-density map from common user intentions to canonical PAL API usage.
