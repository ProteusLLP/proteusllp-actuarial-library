# Coupling Groups, Copulas and Variable Reordering

This guide explains how PAL manages dependencies between stochastic
variables. It covers three interconnected concepts:

1. **Coupling Groups** — automatic tracking of related variables
2. **Copulas** — creating dependency structures between variables
3. **Variable Reordering** — how copulas work under the hood

A runnable version of all examples below is available at
`examples/example_couplings_and_copulas.py`.

## 1. Coupling Groups

### What Is a Coupling Group?

Every `StochasticScalar` in PAL belongs to a **coupling group** — a
set of variables that are linked by computation. When you create an
independent variable, it gets its own coupling group. When you combine
variables in arithmetic, their coupling groups merge.

```python
from pal import config, copulas, distributions, frequency_severity, set_random_seed, variables

config.n_sims = 10_000
set_random_seed(42)

motor = distributions.LogNormal(mu=14, sigma=0.5).generate()
prop = distributions.LogNormal(mu=15, sigma=0.8).generate()

# Each starts in its own group
motor.coupled_variable_group is prop.coupled_variable_group
# => False
```

When variables are combined, they become coupled:

<!--pytest-codeblocks:cont-->

```python
total = motor + prop

# Now all three share the same coupling group
motor.coupled_variable_group is prop.coupled_variable_group
# => True

len(motor.coupled_variable_group)
# => 3
```

Derived variables also join automatically:

<!--pytest-codeblocks:cont-->

```python
motor_with_expenses = motor * 1.1

motor_with_expenses.coupled_variable_group is motor.coupled_variable_group
# => True

len(motor.coupled_variable_group)
# => 4
```

### Why Coupling Groups Matter

Coupling groups solve a critical problem: **when a copula reorders one
variable's simulations, all related variables must be reordered in
exactly the same way** to preserve their mathematical relationships.

Consider this scenario:

<!--pytest-codeblocks:cont-->

```python
set_random_seed(42)
loss_a = distributions.LogNormal(mu=14, sigma=0.5).generate()
loss_b = distributions.LogNormal(mu=15, sigma=0.8).generate()

# Derive a variable from loss_a
loss_a_expenses = loss_a * 1.15  # 15% expense loading
```

Before any copula, `loss_a` and `loss_a_expenses` have a perfect 1.15×
relationship because simulation `i` of both variables corresponds to the
same underlying scenario.

Now apply a copula to correlate `loss_a` with `loss_b`:

<!--pytest-codeblocks:cont-->

```python
copulas.GaussianCopula([[1.0, 0.7], [0.7, 1.0]]).apply([loss_a, loss_b])
```

After the copula reorders `loss_a`, the 1.15× ratio is preserved because
PAL's coupling group system automatically reorders `loss_a_expenses` with
the same permutation as `loss_a`.

## 2. Copulas

A copula defines a dependency structure (correlation pattern) between
random variables. PAL provides several copula families, each producing
different patterns of dependence, especially in the tails.

### Available Copula Types

| Family | Class | Key Parameters | Tail Dependence |
|--------|-------|----------------|-----------------|
| **Elliptical** | | | |
| Gaussian | `GaussianCopula` | Correlation matrix | None |
| Student's T | `StudentsTCopula` | Correlation matrix, `dof` | Symmetric |
| **Archimedean** | | | |
| Gumbel | `GumbelCopula` | `theta`, `n` | Upper |
| Clayton | `ClaytonCopula` | `theta`, `n` | Lower |
| Frank | `FrankCopula` | `theta`, `n` | None |
| Joe | `JoeCopula` | `theta`, `n` | Upper |
| **Extreme value** | | | |
| MM1 | `MM1Copula` | `delta_matrix`, `theta` | Upper |
| Galambos | `GalambosCopula` | `theta`, `d` | Upper |
| Hüsler-Reiss | `HuslerReissCopula` | `lambda_matrix` | Upper |
| Extremal T | `ExtremalTCopula` | `correlation_matrix`, `nu` | Upper |
| **Other** | | | |
| Plackett | `PlackettCopula` | `delta` | None |

### Comparing Copula Types

<!--pytest-codeblocks:cont-->

```python
import numpy as np

config.n_sims = 10_000
set_random_seed(42)
x = distributions.LogNormal(mu=10, sigma=1.0).generate()
y = distributions.LogNormal(mu=10, sigma=1.0).generate()
copulas.GaussianCopula([[1.0, 0.8], [0.8, 1.0]]).apply([x, y])
np.corrcoef(x.ranks, y.ranks)[0, 1]
```

PAL can also generate all pairwise scatter plots directly from a
`ProteusVariable`:

<!--pytest.mark.skip-->

```python
dependency = variables.ProteusVariable("variable", {"X": x, "Y": y})
dependency.rank_scatter_plot(title="Dependency in rank space").show()
dependency.value_scatter_plot(title="Dependency in value space").show()
```

## 3. Variable Reordering

When you call `copula.apply([var_x, var_y])`, PAL generates copula samples,
computes their ranks, reorders existing simulations to match those ranks,
preserves each marginal set of values exactly, and merges the corresponding
coupling groups.

<!--pytest-codeblocks:cont-->

```python
config.n_sims = 10_000
set_random_seed(42)
var_x = distributions.Normal(0, 1).generate()
var_y = distributions.Normal(0, 1).generate()

sorted_x = np.sort(var_x.values)
sorted_y = np.sort(var_y.values)

copulas.GaussianCopula([[1.0, 0.9], [0.9, 1.0]]).apply([var_x, var_y])

np.allclose(np.sort(var_x.values), sorted_x)  # => True
np.allclose(np.sort(var_y.values), sorted_y)  # => True
```

### Reordering Across a Chain of Derived Variables

Coupling groups enable transitive reordering across chains of derived
variables:

<!--pytest-codeblocks:cont-->

```python
set_random_seed(42)
base_loss = distributions.LogNormal(mu=14, sigma=0.5).generate()
gross_loss = base_loss * 1.0
expense_loaded = gross_loss * 1.10
tax = expense_loaded * 0.21
net_loss = expense_loaded - tax

cat_loss = distributions.LogNormal(mu=16, sigma=1.2).generate()
copulas.GumbelCopula(theta=1.5).apply([base_loss, cat_loss])
```

All variables derived from `base_loss` are reordered together, so their
algebraic relationships remain exact.

## 4. Multivariate Copulas

Elliptical copulas naturally extend to any number of dimensions:

<!--pytest-codeblocks:cont-->

```python
set_random_seed(42)

lobs = {
    "Motor": distributions.LogNormal(mu=14, sigma=0.4).generate(),
    "Property": distributions.LogNormal(mu=15, sigma=0.6).generate(),
    "Liability": distributions.LogNormal(mu=13, sigma=0.5).generate(),
    "Marine": distributions.LogNormal(mu=12, sigma=0.7).generate(),
}

corr_matrix = [
    [1.0, 0.6, 0.3, 0.2],
    [0.6, 1.0, 0.4, 0.3],
    [0.3, 0.4, 1.0, 0.5],
    [0.2, 0.3, 0.5, 1.0],
]

copulas.GaussianCopula(corr_matrix).apply(list(lobs.values()))
```

For a Gaussian copula, Spearman's rank correlation is related to the
underlying correlation parameter by

```{math}
\rho_S = \frac{6}{\pi}\arcsin\left(\frac{r}{2}\right).
```

## 5. `generate()` vs `apply()`

`generate()` creates correlated uniform samples. `apply()` reorders existing
PAL stochastic variables, and is the usual workflow when marginal distributions
have already been generated.

<!--pytest-codeblocks:cont-->

```python
samples = copulas.GaussianCopula([[1.0, 0.8], [0.8, 1.0]]).generate()

v1 = distributions.Gamma(alpha=5, theta=1000).generate()
v2 = distributions.Pareto(shape=2, scale=10000).generate()
copulas.GaussianCopula([[1.0, 0.8], [0.8, 1.0]]).apply([v1, v2])
```

## 6. Frequency-Severity Models with Copulas

After generating event-level simulations, aggregate them to simulation-level
totals and use a copula to impose dependencies between lines of business.

<!--pytest-codeblocks:cont-->

```python
config.n_sims = 10_000
set_random_seed(42)

motor_model = frequency_severity.FrequencySeverityModel(
    freq_dist=distributions.Poisson(mean=50),
    sev_dist=distributions.LogNormal(mu=10, sigma=1.5),
)
motor_events = motor_model.generate()

property_model = frequency_severity.FrequencySeverityModel(
    freq_dist=distributions.Poisson(mean=20),
    sev_dist=distributions.Pareto(shape=2.5, scale=50_000),
)
property_events = property_model.generate()

motor_agg = motor_events.aggregate()
property_agg = property_events.aggregate()
copulas.GaussianCopula([[1, 0.6], [0.6, 1]]).apply([motor_agg, property_agg])
```

Because `aggregate()` and `occurrence()` share a coupling group with the
underlying event simulations, reordering one derived quantity also reorders
other coupled quantities consistently.

## Important Constraints

1. Variables must be independent before `apply()`; PAL raises a `ValueError`
   when input variables are already coupled.
2. Once variables are coupled, they remain in the same coupling group for the
   rest of the simulation.
3. The order passed to `apply()` must match the copula's dimension order.

## References

- Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer.
- McNeil, A. J., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management:
  Concepts, Techniques and Tools*, 2nd ed. Princeton University Press.
- Iman, R. L. & Conover, W. J. (1982). "A distribution-free approach to inducing
  rank correlation among input variables." *Communications in Statistics —
  Simulation and Computation*, 11(3), 311–334.
- Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.
