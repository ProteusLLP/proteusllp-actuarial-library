# PAL release checklist

Use this checklist before publishing a GitHub Release. **Publishing a release triggers automatic PyPI publication** of both `proteusllp-actuarial-library` and the compatibility mirror `proteus-actuarial-library`. Do not publish a GitHub Release as a dry run.

## Release candidate validation

- [ ] Confirm the release branch is merged into `main` and the intended release commit is identified.
- [ ] Confirm CPU CI passes on the supported Python versions, including static analysis, pytest, package builds and rendered documentation checks.
- [ ] Confirm the dedicated GPU test workflow passes on the release candidate, with GPU-resident simulation and stochastic coupling checked. Record any known CPU-backed operations.
- [ ] Verify the new probability fitting and optional parameter uncertainty on representative severity models, including bounds, censoring and truncation.
- [ ] Verify model-averaged simulations retain one candidate family and joint parameter set per scenario, including frequency-severity claim expansion.
- [ ] Verify correlation repair, completion and copula integration, including an invalid input matrix and the reported adjusted matrix.
- [ ] Verify ordinary Monte Carlo and randomised quasi-Monte Carlo sampling, including copulas and GPU where supported.
- [ ] Verify CSV/Parquet and PAL snapshot round-trips for nested dimensions and frequency-severity data; smoke-test optional cloud backends.
- [ ] Review newly added distributions, stochastic types and documented CPU/GPU limitations against the intended feature claims.
- [ ] Check that source distribution and both wheels build cleanly, that each version matches the release tag, and that a fresh environment can import the installed package.
- [ ] Review the release notes and public README/tutorial links. Confirm open PRs and roadmap issues do not represent an unaddressed release blocker.
- [ ] Check any downstream integrations that depend on a `main` commit or on a prior PyPI version; plan their versioned-dependency update.

## Publish v0.4.0

1. Create a GitHub Release from the validated `main` commit with the tag `v0.4.0` and notes from [v0.4.0](releases/v0.4.0.md). This starts the automated PyPI pipeline.
2. Confirm the release workflow's test, static analysis, docs, build and PyPI publication jobs complete successfully.
3. Confirm both PyPI projects publish version 0.4.0, then install the primary wheel in a clean environment and import `pal`.
4. Confirm the online documentation corresponds to the released commit and publish the GIRO announcement.

If any validation fails, correct it on a branch and repeat the release-candidate checks before publishing.
