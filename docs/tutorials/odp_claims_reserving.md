# Over-Dispersed Poisson Claims Reserving

This tutorial builds the exact Bayesian predictive distribution of outstanding claims under the over-dispersed Poisson (ODP) reserving model.

The construction follows Norman (2025), which shows that the posterior predictive distribution can be simulated exactly using only beta, gamma and Poisson random variables. It avoids the usual bootstrap approximation while retaining a familiar actuarial property: with the non-informative priors used below, the predictive mean is exactly the chain-ladder reserve estimate.

A later extension, Norman (2026), places informative priors on the origin-period parameters. That gives a continuous credibility bridge between chain ladder and Cape Cod. The final section explains how that extension fits into the same framework.

## 1. Model

For origin period $i$ and development period $j$, write the incremental claim amount as

```{math}
X_{ij} = \phi N_{ij},
\qquad
N_{ij}\mid\mu_i,\beta_j,\phi
\sim \operatorname{Poisson}\!\left(\frac{\mu_i\beta_j}{\phi}\right),
```

with

```{math}
\sum_{j=1}^n \beta_j = 1.
```

The parameters have direct reserving interpretations:

- $\mu_i$ is the expected ultimate claim amount for origin period $i$;
- $\beta_j$ is the expected proportion of ultimate claims paid in development period $j$;
- $\phi$ is the ODP dispersion parameter.

This parameterisation is useful because the outstanding reserve for origin period $i$ is simply the sum of the future incremental payments.

## 2. Example triangle

We use the 10 × 10 incremental triangle shipped with PAL's reserve-risk example.

```python
import numpy as np

from pal import config, set_random_seed
from pal.distributions import Beta, Gamma, Poisson
from pal.variables import ProteusVariable, StochasticScalar

config.n_sims = 100_000
set_random_seed(42)

triangle = np.array(
    [
        [357848, 766940, 610542, 482940, 527326, 574398, 146342, 139950, 227229, 67948],
        [352118, 884021, 933894, 1183289, 445745, 320996, 527804, 266172, 425046, np.nan],
        [290507, 1001799, 926219, 1016654, 750816, 146923, 495992, 280405, np.nan, np.nan],
        [310608, 1108250, 776189, 1562400, 272482, 352053, 206286, np.nan, np.nan, np.nan],
        [443160, 693190, 991983, 769488, 504851, 470639, np.nan, np.nan, np.nan, np.nan],
        [396132, 937085, 847498, 805037, 705960, np.nan, np.nan, np.nan, np.nan, np.nan],
        [440832, 847631, 1131398, 1063269, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [359480, 1061648, 1443370, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [376686, 986608, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [344014, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ],
    dtype=float,
)

n = triangle.shape[0]
observed = ~np.isnan(triangle)
cumulative = np.cumsum(np.nan_to_num(triangle), axis=1)
cumulative[~observed] = np.nan
```

## 3. Estimate the dispersion parameter

The point estimates of the ODP mean structure reproduce chain ladder. We therefore obtain the fitted development pattern from the ordinary chain-ladder link ratios and estimate $\phi$ from the Pearson residuals.

<!--pytest-codeblocks:cont-->

```python
link = np.ones(n)
for j in range(1, n):
    link[j] = (
        np.sum(cumulative[: n - j, j])
        / np.sum(cumulative[: n - j, j - 1])
    )

remaining_factor = np.ones(n)
for j in range(n - 2, -1, -1):
    remaining_factor[j] = remaining_factor[j + 1] * link[j + 1]

cumulative_beta_hat = 1 / remaining_factor
beta_hat = np.diff(cumulative_beta_hat, prepend=0)

mu_hat = np.array(
    [
        cumulative[i, n - i - 1] / cumulative_beta_hat[n - i - 1]
        for i in range(n)
    ]
)

fitted = np.outer(mu_hat, beta_hat)
degrees_of_freedom = n * (n + 1) / 2 - 2 * n + 1
phi = np.sum((((triangle - fitted) ** 2) / fitted)[observed]) / degrees_of_freedom

print(f"Estimated dispersion: {phi:,.0f}")
```

For this triangle the estimate is approximately $52,600$.

## 4. Exact posterior distribution of the development pattern

The key result in Norman (2025) is a reparameterisation of the development pattern that makes the posterior independent and beta distributed.

Define a sequence $\psi_j$. Under the non-informative prior used for the chain-ladder case,

```{math}
\psi_1 = 1,
```

and, for $j>1$,

```{math}
\psi_j \mid X_n
\sim
\operatorname{Beta}\!\left(
    c_j,
    1 + \sum_{i=1}^{n-j+1} d_{i,j-1}
\right),
```

where $c_j$ is the observed development-column total divided by $\phi$, and $d_{ij}$ is the cumulative claim amount through development period $j$ divided by $\phi$.

The original payment proportions $\beta_j$ are then reconstructed recursively from the $\psi_j$ draws.

<!--pytest-codeblocks:cont-->

```python
d_i = np.nansum(triangle, axis=1) / phi
c_j = np.nansum(triangle, axis=0) / phi
d_ij = cumulative / phi

sum_dij = [
    np.sum(d_ij[: n - j, j - 1])
    for j in range(1, n)
]

psi_values = [StochasticScalar([1])]
for j in range(1, n):
    psi_values.append(
        Beta(
            a=float(c_j[j]),
            b=1.0 + float(sum_dij[j - 1]),
        ).generate()
    )

psi = ProteusVariable(
    dim_name="development_period",
    values={str(j + 1): psi_values[j] for j in range(n)},
)

betas = [StochasticScalar([])] * n
betas[-1] = psi[str(n)]
future_beta = betas[-1]

for j in range(n - 2, -1, -1):
    betas[j] = psi[str(j + 1)] * (1 - future_beta)
    future_beta = future_beta + betas[j]

beta = ProteusVariable(
    dim_name="development_period",
    values={str(j + 1): betas[j] for j in range(n)},
)
```

This is parameter uncertainty in the development pattern itself: every simulation contains one internally consistent set of development proportions that sums to one.

## 5. Exact posterior distribution of origin-period ultimates

Conditional on the simulated development pattern, each origin-period ultimate has a gamma posterior. If $k_i=n-i+1$ is the latest observed development period for origin period $i$, then

```{math}
\mu_i\mid\boldsymbol{\beta},X_n
\sim
\operatorname{Gamma}\!\left(
    d_i,
    \frac{\phi}{B_{k_i}}
\right),
```

where

```{math}
B_k = \sum_{j=1}^{k}\beta_j.
```

<!--pytest-codeblocks:cont-->

```python
cumulative_beta = []
running_beta = 0
for j in range(n):
    running_beta = running_beta + beta[str(j + 1)]
    cumulative_beta.append(running_beta)

mu = ProteusVariable(
    dim_name="origin_period",
    values={
        str(i + 1): (
            phi
            / cumulative_beta[n - i - 1]
            * Gamma(a=float(d_i[i]), scale=1).generate()
        )
        for i in range(n)
    },
)
```

PAL's stochastic variables keep the simulated $\mu_i$ and $\beta_j$ values aligned scenario by scenario, so the subsequent predictive simulation uses the same posterior parameter draw throughout each scenario.

## 6. Simulate future claim payments

The last step is process uncertainty. For every unobserved cell we draw a Poisson payment conditional on that scenario's posterior parameters.

<!--pytest-codeblocks:cont-->

```python
reserve_by_origin = ProteusVariable(
    dim_name="origin_period",
    values={str(i + 1): StochasticScalar([0]) for i in range(n)},
)

for i in range(n):
    origin = str(i + 1)
    for j in range(n - i, n):
        development = str(j + 1)
        future_cell = phi * Poisson(
            mu[origin] * beta[development] / phi
        ).generate()
        reserve_by_origin[origin] = reserve_by_origin[origin] + future_cell

reserve = reserve_by_origin.sum()
```

`reserve` is now a Monte Carlo sample from the exact posterior predictive distribution, rather than from a fitted bootstrap approximation.

## 7. Reserve estimate and risk measures

<!--pytest-codeblocks:cont-->

```python
mean_reserve = reserve.mean()
sd_reserve = reserve.std()

print(f"Mean reserve: {mean_reserve:,.0f}")
print(f"Reserve SD:   {sd_reserve:,.0f}")
print(f"95th:         {reserve.percentile(95):,.0f}")
print(f"99th:         {reserve.percentile(99):,.0f}")
```

With 100,000 simulations the results should be close to:

| Statistic | Approximate value |
|-----------|------------------:|
| Mean | 18.7m |
| Standard deviation | 3.0m |
| 95th percentile | 23.9m |
| 99th percentile | 26.7m |

Monte Carlo quantiles will vary slightly with the simulation sample.

### Chain-ladder mean

The posterior predictive mean under these non-informative priors is exactly the chain-ladder reserve. For this triangle the deterministic chain-ladder reserve is about 18.68m, which provides a useful implementation check: the simulated mean above should converge to that value as the number of simulations increases.

The important distinction is that chain ladder supplies only the centre of the result. The Bayesian ODP construction supplies the complete predictive distribution around it, including parameter and process uncertainty.

## 8. Inspect reserve risk by origin period

Because each origin-period reserve remains a separate `StochasticScalar`, the same simulations can be used to examine where reserve uncertainty comes from.

<!--pytest-codeblocks:cont-->

```python
for origin, origin_reserve in reserve_by_origin.items():
    if origin != "1":
        print(
            origin,
            f"mean={origin_reserve.mean():,.0f}",
            f"sd={origin_reserve.std():,.0f}",
        )
```

The most recent origin periods naturally carry the largest parameter uncertainty because only a small fraction of their eventual development has been observed.

## 9. Origin-period priors: the Chain Ladder–Cape Cod bridge

Norman (2026) extends the same exact construction by placing informative gamma priors on the origin-period parameters $\mu_i$. These priors can encode exposure volumes, expected loss ratios, rate changes or other external information about the relative level of different origin periods.

The useful actuarial interpretation is a credibility continuum:

- with a weak origin-period prior, the posterior reserve tends to the chain-ladder result;
- with a strong exposure-based prior, it tends to Cape Cod;
- intermediate prior strength blends the two according to the information available for each origin period.

Older origin periods, where much of the development is already observed, receive relatively little influence from the prior. Recent origin periods receive more. Unlike applying an ad-hoc blend after fitting two separate methods, the weighting falls out of one coherent predictive model, and the uncertainty in future payments is propagated consistently.

The non-informative example above is therefore the chain-ladder endpoint of the broader model. It is a useful starting point because it isolates the exact predictive-distribution machinery before adding prior information.

## References

- Norman, J. P. (2025). *The Predictive Distribution of the Over-dispersed Poisson Claims Reserving Model*. SSRN 5163359.
- Norman, J. P. (2026). *Exact Bayesian Predictive Distributions for the Over-dispersed Poisson Reserving Model with Origin-Period Priors: An Exact Analytical Bridge between Chain Ladder and Cape Cod*. SSRN 6700780.
- Renshaw, A. E. and Verrall, R. J. (1998). "A stochastic model underlying the chain-ladder technique." *British Actuarial Journal*, 4(4), 903–923.
- England, P. D. and Verrall, R. J. (2002). "Stochastic claims reserving in general insurance." *British Actuarial Journal*, 8(3), 443–518.

## See also

The executable script [`examples/example_odp_reserve_risk.py`](../../examples/example_odp_reserve_risk.py) contains the same non-informative exact predictive model in a compact reusable class.