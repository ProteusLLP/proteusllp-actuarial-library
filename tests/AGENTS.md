# PAL Test Agent Guide

These rules apply to changes under `tests/` in addition to the root `AGENTS.md`.

## Testing Philosophy

PAL is a numerical actuarial library. Tests should establish that an implementation represents the intended mathematics and stochastic semantics, not merely that the code is internally self-consistent.

## Numerical Tests

For distributions, copulas, special functions and risk calculations, prefer one or more independent checks:

- closed-form moments or probabilities;
- published numerical examples;
- SciPy, Boost or another trusted implementation where parameterisations agree;
- direct numerical integration or high-precision calculation;
- simulation checks when no exact benchmark is practical.

Self-consistency tests such as `cdf(ppf(p)) == p` are useful secondary checks, but should not be the only numerical validation.

Include important boundary, limiting and tail cases. Choose tolerances from expected numerical error rather than simply loosening them until a test passes.

## Stochastic and Coupling Tests

When code accepts stochastic parameters or returns PAL stochastic objects, test more than the output values. Verify that:

- coupling groups are preserved or combined as intended;
- dependent inputs remain aligned after transformations;
- copula reordering propagates to derived variables correctly;
- wrapper/helper functions do not detach parameters or results from existing coupling relationships.

## CPU/GPU Tests

For backend-sensitive changes:

- test the CPU implementation normally;
- exercise the GPU implementation in GPU CI where available;
- compare CPU and GPU statistical/numerical results within appropriate tolerances;
- add a regression test for unnecessary host/device conversion when the change is specifically about backend efficiency and it can be tested robustly.

Do not make ordinary CPU tests require CuPy or CUDA.

## Reproducibility

Use PAL's established seed/configuration helpers rather than global ad-hoc random state. Keep tests deterministic unless the test is explicitly statistical; statistical tests should use stable sample sizes and tolerances chosen to avoid flaky CI.

## Scope

Prefer focused tests close to the changed behaviour. Add broad integration tests only when they protect an important cross-module contract that unit tests cannot capture.
