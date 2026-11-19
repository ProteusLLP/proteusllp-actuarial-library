# Actuarial recipe cookbook

These short examples are **executable Python modules shipped with PAL**. Each
recipe starts from a modelling objective and uses the documented public API.
They run in CI. Copy a recipe into your analysis or ask an AI coding assistant
to adapt it to your own data and contract terms.

The PyPI distribution is `proteusllp-actuarial-library`; import it as `pal`.
Recipes use synthetic data and small simulation counts so they run quickly.

## Search from an installed PAL package

```python
from pal import api

matches = api.recipes("price excess of loss layer")
print(matches[0]["title"])
print(api.recipe(matches[0]["id"])["code"])
```

`api.recipes()` lists all recipes. The search returns compact, JSON-serialisable
metadata including modelling caveats; `api.recipe(id)` returns the complete,
copyable Python source. Reading a recipe does not execute it and needs no network
connection. The examples below are sourced from the **same packaged files** that
the runtime API returns.

## Fit a Gamma claim-severity distribution

Maximum-likelihood estimation of shape and scale; downstream calculations
condition on the estimates and do **not** include parameter uncertainty.

```{literalinclude} ../../src/pal/examples/recipe_gamma_fit.py
:language: python
```

## Fit thresholded, policy-limited large losses

Use left truncation for a reporting threshold and right censoring when a loss
reaches the policy limit. The fitted model estimates underlying pre-limit
severity above the reporting threshold.

```{literalinclude} ../../src/pal/examples/recipe_censored_severity.py
:language: python
```

## Model frequency and severity

Generate individual claims, then derive aggregate and occurrence statistics.
`FreqSevSims` tracks **claims** as well as their simulation membership.

```{literalinclude} ../../src/pal/examples/recipe_frequency_severity.py
:language: python
```

## Price an occurrence excess-of-loss layer

Apply a layer to each event and aggregate recoveries by simulation. The expected
recovery is a technical loss estimate, not a quoted reinsurance premium.

```{literalinclude} ../../src/pal/examples/recipe_xol_layer.py
:language: python
```

## Combine correlated risks with a copula

A copula reorders marginal simulations. PAL's coupling groups also reorder
derived values to preserve arithmetic relationships.

```{literalinclude} ../../src/pal/examples/recipe_copula_portfolio.py
:language: python
```

## Apply stochastic inflation to claims

A single inflation draw per simulation is applied consistently to every claim
in that simulation.

```{literalinclude} ../../src/pal/examples/recipe_stochastic_inflation.py
:language: python
```

## Use randomized quasi-Monte Carlo

Switch between Monte Carlo and scrambled Sobol sampling. Use independent
randomisations to estimate RQMC simulation error.

```{literalinclude} ../../src/pal/examples/recipe_rqmc_sampling.py
:language: python
```

## Extending a recipe

Inspect public signatures with `api.describe("Gamma")` or search for a
different model with `api.search("tail dependence")`. Use
`StochasticScalar` for simulation-level values, `FreqSevSims` for claim
events, and PAL arithmetic for derived values so the coupling group is not
lost. Test numerical results and stochastic ancestry whenever extending
an example. See [the AI assistant quick reference](ai_assistants.md) and the
[distribution fitting tutorial](tutorials/distribution_fitting.md).
