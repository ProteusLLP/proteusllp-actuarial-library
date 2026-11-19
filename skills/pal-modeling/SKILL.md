---
name: pal-modeling
description: Use the Proteus Actuarial Library (PAL) to build, extend, validate, or explain simulation-based actuarial Python models, including severity fitting, frequency-severity, copulas, reinsurance and risk measures.
---

# PAL modelling skill

Use this skill when the user requests Python actuarial modelling with PAL.
The PyPI package is `proteusllp-actuarial-library`; import it as `pal`.

## Discover, then write

1. Check whether PAL is installed and inspect the local version rather than
   assuming the newest documentation matches the environment.
2. When a task is familiar, search the **packaged, executable recipes** first:
   ```python
   from pal import api

   matches = api.recipes("price excess of loss layer")
   source = api.recipe(matches[0]["id"])["code"]
   ```
   Run and adapt a relevant recipe; do not invent a method or import path.
3. For an unfamiliar class, use `api.search("keyword")` and
   `api.describe("pal.distributions.Gamma")`. Verify constructor parameter
   names, meaning, units and the supported return type from the installed
   version. Consult the [PAL cookbook](https://proteusllp-actuarial-library.readthedocs.io/en/latest/cookbook.html)
   and the [API reference](https://proteusllp-actuarial-library.readthedocs.io/en/latest/api/modules.html).
4. Use `from pal import StochasticScalar, ProteusVariable, FreqSevSims,
   FrequencySeverityModel` for core classes and `from pal.distributions
   import Gamma`, `from pal.contracts import XoL`, etc. for domain classes.
5. Produce a runnable program with a deterministic seed, a manageable initial
   simulation count, and tests or assertions that express the mathematical
   result and coupling relationships.

## Essential actuarial semantics

- Gamma's `theta` is a **scale**, not a rate; LogNormal's `mu` and
  `sigma` parameterise the **logarithm** of loss.
- A `StochasticScalar` has **one value per simulation**. A
  `FreqSevSims` contains **claim-level values** and an index mapping them to
  simulations. Use `.aggregate()` or `.occurrence()` for annual summaries.
- Operate on PAL stochastic variables, rather than extracting, reordering and
  reconstructing raw NumPy arrays. PAL's coupling groups preserve upstream
  dependencies, including after applying copulas.
- A copula applied to generated marginals alters their **joint dependence**
  without altering their individual values.
- Model a policy-limited large-loss observation as **right-censored** if the
  loss hits the limit. Reporting thresholds lead to **left truncation**.
  Fitting by MLE and plugging in estimates does **not** account for parameter
  uncertainty.
- Check whether the model is an occurrence/claim layer or aggregate contract;
  technical expected recovery alone is not a market premium.
- PAL can run with CuPy when `PAL_USE_GPU=1` is configured before import;
  avoid unnecessary host copies. RQMC changes the sampler, not the model,
  and uncertainty estimates require independent scrambles.

## Verify before presenting code

- Confirm that imports and signatures exist in the installed PAL version.
- Run the smallest example before increasing the number of simulations.
- Check parameter domains, units, expected magnitude, tail behaviour and
  theoretical identities where possible.
- If a result depends on stochastic parameters or copulas, verify its coupling
  group with upstream variables and check scenario alignment.
- If PAL cannot perform the requested operation, state the limitation and
  suggest a documented workaround rather than fabricating APIs.

## Portability

This skill is designed for coding agents working in a **consumer project**.
Copy this folder into the agent's supported skills location, for example
`.agents/skills/pal-modeling/SKILL.md` for agents that discover skills there.
It does not require or start an MCP service. For agents editing PAL's own
repository, also follow that repository's `AGENTS.md`.
