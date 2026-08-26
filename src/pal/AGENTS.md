# PAL Source Agent Guide

These rules apply to changes under `src/pal/` in addition to the root `AGENTS.md`.

## Public API

- Treat documented public classes, functions, methods and import paths as compatibility-sensitive.
- PAL's top-level namespace is intentionally small. Domain functionality belongs under documented namespaces: use `from pal import distributions` followed by `distributions.Gamma(...)`, `from pal import copulas` followed by `copulas.GaussianCopula(...)`, and the equivalent pattern for contracts, risk measures, frequency-severity models and other broad domains.
- Core variable types are imported directly from `pal.variables`. Use `from pal.variables import StochasticScalar, ProteusVariable`; do not teach `stochastic_scalar.StochasticScalar` or `from pal.stochastic_scalar import StochasticScalar` as public API.
- Do not re-export domain classes or functions from `pal.__init__`. In particular, examples should not teach `from pal import Gamma`, `XoL`, `StochasticScalar`, `GaussianCopula`, or similar shortcuts.
- The `config` singleton and its small configuration helpers are intentional top-level conveniences; do not generalise that exception to modelling classes.
- User-facing docstrings should explain actuarial/statistical behaviour, not internal backend mechanics.
- Prefer terminology a PAL user would recognise. For example, `StochasticScalar.mean()` takes the mean across simulations; users do not need implementation language such as "backend ndarray".
- Keep signatures, type annotations and docstrings mutually consistent.
- If a symbol is intended to be public, make its exposure deliberate rather than relying on accidental wildcard-import behaviour.

## Stochastic Semantics and Coupling

PAL tracks relationships between simulated variables through coupling groups. Preserve those relationships when transforming or wrapping results.

When a function accepts stochastic parameters or returns a stochastic object:

- verify that derived results remain in the correct coupling groups;
- do not silently detach distribution parameters from their existing dependencies;
- preserve simulation alignment through arithmetic, transformations and copula reordering;
- add tests that would fail if coupling metadata were lost.

## CPU and GPU Behaviour

Backend support is a behavioural contract, not just an implementation detail.

- CPU and GPU paths should implement the same statistical model and public semantics.
- Avoid unnecessary host/device transfers. Do not convert CuPy arrays to NumPy merely to call a convenient CPU implementation when a practical GPU implementation exists.
- Keep random-number generation reproducible according to PAL's established backend conventions.
- Backend-specific optimisations must not introduce undocumented special cases tied to isolated parameter values.
- Validate numerical accuracy as well as speed when replacing mathematical kernels.

## Numerical and Distribution Code

For distributions, copulas, special functions and risk calculations:

- derive behaviour from the mathematical definition rather than fitting tests to the implementation;
- test against theoretical moments, known values, trusted reference implementations or literature examples where practical;
- include boundary and limiting cases that matter mathematically;
- pay particular attention to tail probabilities, inverse CDFs and numerically extreme parameters;
- keep mathematical documentation consistent with the implemented parameterisation.

Self-consistency checks such as `cdf(ppf(p)) == p` are useful but are not sufficient evidence of correctness on their own.

## Type-System Architecture

Read `docs/structure.md` before changing protocols or core types.

Critical invariants include:

- `types.py` defines abstractions and must not import concrete PAL implementations;
- protocols are for structural typing, not runtime inheritance;
- higher architectural layers may depend on lower layers, not the reverse;
- avoid solving type errors by introducing circular imports or broad `Any`/`type: ignore` escapes.

Use a narrow, documented type ignore only when the type checker or a third-party stub cannot express correct behaviour.

## Imports and Initialisation

PAL has some established import-order constraints. Before adding package-level imports or changing `__init__.py`, inspect the existing dependency chain and ensure import-time cycles are not introduced.
