# Operational Risk LDA — Böcker & Klüppelberg (2005)

This tutorial implements the loss distribution approach (LDA) from the classic
operational-risk paper:

> Böcker, K. and Klüppelberg, C. (2005). "Operational VaR: a closed-form
> approximation." *Risk*, 18(12), 90–93.

The paper shows that, for a compound frequency-severity model with a
heavy-tailed severity distribution, very high aggregate-loss quantiles can be
approximated by a single severity quantile. This gives a useful benchmark for a
Monte Carlo operational-risk model.

The example below follows the paper's Pareto-Poisson construction. The frequency
parameter used here is a transparent worked-example choice rather than an attempt
to reproduce every numerical point in the paper's Figure 1.

## 1. The loss distribution approach

Let annual aggregate operational loss be

```{math}
S(t) = \sum_{i=1}^{N(t)} X_i,
```

where `N(t)` is the number of loss events up to time `t` and the `X_i` are
independent severities. For a subexponential severity distribution with CDF
`F`, Böcker and Klüppelberg obtain the first-order high-quantile approximation

```{math}
\operatorname{VaR}_t(\kappa)
\sim
F^{-1}\!\left(1 - \frac{1-\kappa}{\operatorname{E}[N(t)]}\right),
\qquad \kappa \to 1.
```

The intuition is that an extreme annual loss is increasingly dominated by one
exceptionally large event rather than by an unusually large sum of ordinary
events. This is often called a **single-loss approximation**.

For a Poisson process with annual intensity `lambda`,
`E[N(t)] = lambda * t`.

## 2. Pareto severity model

The paper uses the Pareto Type II (Lomax) severity distribution

```{math}
F(x) = 1 - \left(1 + \frac{x}{\theta}\right)^{-\alpha},
\qquad x > 0,
```

with shape `alpha > 0` and scale `theta > 0`. Substituting its inverse CDF into
the single-loss approximation gives

```{math}
\operatorname{VaR}_t(\kappa)
\approx
\theta\left[
\left(\frac{\lambda t}{1-\kappa}\right)^{1/\alpha} - 1
\right].
```

PAL's `Pareto` class is Pareto Type I, with support starting at `scale`. A
Pareto Type II random variable with the paper's parameterisation is therefore
obtained by subtracting the scale:

```{math}
X = Y - \theta,
\qquad Y \sim \operatorname{ParetoI}(\alpha, \theta).
```

## 3. Build the model in PAL

We use an annual Poisson frequency of 10 and the paper's `theta = 1`. The
heavier-tailed `alpha = 1.1` case is one of the shape parameters illustrated in
the paper.

```python
import pandas as pd

from pal.config import set_random_seed
from pal.distributions import Pareto, Poisson
from pal.frequency_severity import FrequencySeverityModel

set_random_seed(42)

MEAN_FREQUENCY = 10.0
ALPHA = 1.1
THETA = 1.0
N_SIMS = 250_000
```

<!--pytest-codeblocks:cont-->

```python
model = FrequencySeverityModel(
    freq_dist=Poisson(MEAN_FREQUENCY),
    sev_dist=Pareto(shape=ALPHA, scale=THETA),
)

# Shifting Pareto Type I by theta gives the paper's Pareto Type II severity.
events = model.generate(n_sims=N_SIMS) - THETA
annual_loss = events.aggregate()
```

`events` contains the individual operational losses and `annual_loss` contains
one total loss for each simulated year.

## 4. The closed-form approximation

The paper's Pareto-Poisson approximation can be written directly:

<!--pytest-codeblocks:cont-->

```python
def bocker_kluppelberg_var(
    alpha: float,
    theta: float,
    mean_frequency: float,
    confidence_level: float,
    time_horizon: float = 1.0,
) -> float:
    """Return the Pareto-Poisson first-order operational VaR approximation."""
    expected_events = mean_frequency * time_horizon
    return theta * ((expected_events / (1.0 - confidence_level)) ** (1.0 / alpha) - 1.0)
```

At the 99.9% confidence level:

<!--pytest-codeblocks:cont-->

```python
approx_var_999 = bocker_kluppelberg_var(
    alpha=ALPHA,
    theta=THETA,
    mean_frequency=MEAN_FREQUENCY,
    confidence_level=0.999,
)
print(f"99.9% single-loss approximation: {approx_var_999:,.2f}")
```

```text
99.9% single-loss approximation: 4,327.76
```

The result is deterministic: no simulation is needed to calculate the
approximation once the frequency and severity parameters are known.

## 5. Compare simulation with the paper's approximation

Böcker and Klüppelberg compare their approximation with Monte Carlo values at
very high confidence levels. We can make the same comparison using PAL. Note
that `StochasticScalar.percentile()` uses percentile levels from 0 to 100,
whereas the paper writes the confidence level as a probability between 0 and 1.

<!--pytest-codeblocks:cont-->

```python
confidence_levels = [0.990, 0.995, 0.998, 0.999]
simulated_var = annual_loss.percentile([100 * level for level in confidence_levels])

rows = []
for confidence_level, simulated in zip(confidence_levels, simulated_var):
    approximation = bocker_kluppelberg_var(
        alpha=ALPHA,
        theta=THETA,
        mean_frequency=MEAN_FREQUENCY,
        confidence_level=confidence_level,
    )
    rows.append(
        {
            "confidence": confidence_level,
            "simulated VaR": simulated,
            "single-loss approximation": approximation,
            "relative difference": (approximation / simulated) - 1.0,
        }
    )

comparison = pd.DataFrame(rows).set_index("confidence")
print(comparison)
```

The two columns should become close in the far tail, but they are not expected
to be identical. The analytical result is a first-order asymptotic approximation
as `kappa -> 1`, while the simulated quantile also has Monte Carlo error. For a
production capital calculation, the simulation count should be chosen to give
adequate precision at the target percentile.

## 6. Why the tail index matters

The approximation makes the impact of the Pareto tail index particularly clear.
Keeping frequency and scale fixed, compare the two shape parameters used in the
paper's Figure 1:

<!--pytest-codeblocks:cont-->

```python
shape_comparison = pd.DataFrame(
    {
        "alpha": [1.5, 1.1],
        "99.9% approximate VaR": [
            bocker_kluppelberg_var(1.5, THETA, MEAN_FREQUENCY, 0.999),
            bocker_kluppelberg_var(1.1, THETA, MEAN_FREQUENCY, 0.999),
        ],
    }
).set_index("alpha")
print(shape_comparison)
```

| alpha | 99.9% approximate VaR |
|------:|-----------------------:|
| 1.5 | 463.16 |
| 1.1 | 4,327.76 |

A relatively small change in the tail index has an enormous effect on extreme
capital. This is one reason severity-tail selection and parameter uncertainty
matter so much in operational-risk models.

## 7. Time scaling

For the Pareto model, the leading term of the approximation implies the
`alpha`-root-of-time rule derived in the paper:

```{math}
\frac{\operatorname{VaR}_t(\kappa)}
     {\operatorname{VaR}_1(\kappa)}
\sim t^{1/\alpha},
\qquad \kappa \to 1.
```

This differs from the square-root-of-time rule associated with Gaussian risks.
We can see the scaling directly from the closed form:

<!--pytest-codeblocks:cont-->

```python
one_year = bocker_kluppelberg_var(ALPHA, THETA, MEAN_FREQUENCY, 0.999)
scaling_rows = []
for years in [1, 2, 5]:
    scaled_var = bocker_kluppelberg_var(
        ALPHA,
        THETA,
        MEAN_FREQUENCY,
        0.999,
        time_horizon=float(years),
    )
    scaling_rows.append(
        {
            "years": years,
            "closed-form VaR ratio": scaled_var / one_year,
            "alpha-root rule": years ** (1.0 / ALPHA),
        }
    )

scaling = pd.DataFrame(scaling_rows).set_index("years")
print(scaling)
```

The ratios are not exactly equal at a finite confidence level because the
Pareto-II quantile contains the `-1` location term. They converge to the
`alpha`-root rule deeper in the tail.

## 8. What this example does — and does not — show

This example is deliberately stylised. It demonstrates three useful ideas:

- operational risk fits naturally into PAL's frequency-severity framework;
- heavy-tailed aggregate capital can be dominated by the severity tail; and
- the Böcker-Klüppelberg approximation provides an independent analytical
  benchmark for a Monte Carlo result.

A practical operational-risk model may also need multiple risk cells,
reporting thresholds or truncation, body-tail splicing, non-Poisson frequency,
parameter uncertainty, scenario information and dependence between cells. The
single-loss approximation remains valuable as a diagnostic even when the full
model is more complicated.

## References

- Böcker, K. and Klüppelberg, C. (2005). "Operational VaR: a closed-form
  approximation." *Risk*, 18(12), 90–93.
- Embrechts, P., Klüppelberg, C. and Mikosch, T. (1997). *Modelling Extremal
  Events for Insurance and Finance*. Springer.
