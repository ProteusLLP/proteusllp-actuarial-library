r"""Risk measures and capital allocation.

This module provides risk measures that produce both a scalar risk
measure value and per-simulation weights for Euler-style capital
allocation. Each function returns a ``RiskMeasureResult`` with:

- ``.value`` -- the risk measure applied to the input profile
- ``.weights`` -- per-simulation weights
- ``.allocate(component)`` -- weighted expectation for allocation

Spectral risk measures are a subclass where the weights derive from
a distortion of the loss distribution's percentiles:

.. math::

    \\rho(X) = E[\\phi(F(X)) \\cdot X]

Non-spectral measures (e.g. standard deviation principle) derive
weights from gradient-based (Euler) allocation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, overload

import numpy as np

from pal import distributions
from pal._maths import xp
from pal.stochastic_scalar import StochasticScalar

RiskProfileType = Literal["loss", "profit"]


class RiskMeasureResult:
    """Result of a risk measure computation.

    Attributes:
        value: The scalar risk measure of the input profile.
        weights: Per-simulation weights.
    """

    def __init__(
        self,
        value: float,
        weights: StochasticScalar,
    ) -> None:
        """Initialise a RiskMeasureResult."""
        self._value = value
        self._weights = weights

    @property
    def value(self) -> float:
        """The risk measure applied to the input profile."""
        return self._value

    @property
    def weights(self) -> StochasticScalar:
        """Per-simulation weights for allocation."""
        return self._weights

    @overload
    def allocate(self, variable: StochasticScalar) -> float: ...

    @overload
    def allocate(self, variable: ProteusVariable[StochasticScalar]) -> ProteusVariable[float]: ...

    def allocate(self, variable: Any) -> Any:
        """Compute the weighted expectation of a variable.

        For Euler allocation, pass individual components to
        obtain their contribution to the total risk measure.
        Works with StochasticScalar and ProteusVariable.

        Args:
            variable: Component to allocate. When all
                components are passed individually, their
                allocated values sum to ``self.value``.
        """
        return np.mean(variable * self._weights)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"RiskMeasureResult(value={self._value:.6f})"


def _spectral_risk_measure(
    risk_profile: StochasticScalar,
    weight_fn: Callable[[StochasticScalar], StochasticScalar],
    risk_profile_type: RiskProfileType = "loss",
) -> RiskMeasureResult:
    """Apply a spectral weight function to a risk profile.

    Computes unnormalized weights from ``weight_fn``, normalizes them
    to average 1, maps them to simulations via rank ordering, and
    computes the risk measure value.

    Args:
        risk_profile: Stochastic risk profile to weight.
        weight_fn: Maps uniform percentiles to unnormalized weights.
        risk_profile_type: Whether the profile represents losses
            or profits.
    """
    ranks = risk_profile.ranks
    n_sims = risk_profile.n_sims
    u = StochasticScalar((xp.arange(n_sims) + 0.5) / n_sims)
    raw_weights = weight_fn(u)
    normalized = raw_weights / raw_weights.mean()
    if risk_profile_type == "loss":
        weights = normalized[ranks]
    else:
        # Complement ranks so smallest values get highest weights
        weights = normalized[n_sims - 1 - ranks]
    weights_ss = StochasticScalar(weights)
    value = float(xp.mean(risk_profile.values * weights_ss.values))
    return RiskMeasureResult(value, weights_ss)


def proportional_hazards_transform(
    risk_profile: StochasticScalar,
    alpha: float,
    risk_profile_type: RiskProfileType = "loss",
) -> RiskMeasureResult:
    r"""Proportional hazard transform.

    The proportional hazard (PH) transform is a distortion risk
    measure that applies a power transformation to the survival
    function of the loss distribution. For a loss variable
    :math:`X \geq 0` with survival function :math:`S(x)`, the
    PH premium principle is:

    .. math::

        H(X) = \int_0^\infty [S(x)]^\alpha \, dx

    This corresponds to the distortion function
    :math:`g(s) = s^\alpha` applied to survival probabilities.
    The measure is coherent (and hence sub-additive) when
    :math:`0 < \alpha \leq 1`, since :math:`g` is then concave.

    Differentiating the distortion gives the risk spectrum
    (weight function) expressed in terms of the CDF
    :math:`F = 1 - S`:

    .. math::

        w(u) = g'(1-u) = \alpha\,(1-u)^{\alpha - 1}

    For :math:`\alpha < 1` the weights are increasing in
    :math:`u`, placing more emphasis on the tail of the loss
    distribution. When :math:`\alpha = 1` the weights are
    uniform and the measure reduces to the expected value.

    The representation extends to losses with support on the
    whole real line via
    :math:`H(X) = E[w(F(X)) \cdot X]`.

    References:
        Wang, S.S. (1995). "Insurance pricing and increased
        limits ratemaking by proportional hazards transforms."
        *Insurance: Mathematics and Economics*, 17(1), 43-54.

        Wirch, J.L. and Hardy, M.R. (1999). "A synthesis of
        risk measures for capital adequacy." *Insurance:
        Mathematics and Economics*, 25(3), 337-347.

    Args:
        risk_profile: Stochastic risk profile to weight.
        alpha: PH parameter (0 < alpha <= 1 for coherence).
            Lower values give more weight to tail losses.
        risk_profile_type: Whether the profile represents losses
            or profits.
    """
    return _spectral_risk_measure(
        risk_profile,
        lambda u: alpha * (1 - u) ** (alpha - 1),
        risk_profile_type,
    )


def wang_transform(
    risk_profile: StochasticScalar,
    alpha: float,
    risk_profile_type: RiskProfileType = "loss",
) -> RiskMeasureResult:
    r"""Wang transform.

    The Wang transform applies a normal-distribution-based
    distortion to the survival function. For a loss variable
    :math:`X` with survival function :math:`S(x)`, the
    Wang premium principle is:

    .. math::

        W(X) = \int_0^\infty
            \Phi\bigl(\Phi^{-1}(S(x)) + \alpha\bigr) \, dx

    where :math:`\Phi` is the standard normal CDF and
    :math:`\Phi^{-1}` its inverse. The distortion function
    is :math:`g(s) = \Phi(\Phi^{-1}(s) + \alpha)`, which is
    concave for :math:`\alpha > 0`, making the measure
    coherent.

    Differentiating :math:`g` and substituting
    :math:`u = F(x) = 1 - S(x)` yields the risk spectrum:

    .. math::

        w(u) = g'(1-u)
             = \frac{\phi(\Phi^{-1}(1-u) + \alpha)}
                    {\phi(\Phi^{-1}(1-u))}
             = \exp\!\left(
                 -\alpha\,\Phi^{-1}(1-u)
                 - \tfrac{\alpha^2}{2}
               \right)

    where :math:`\phi` is the standard normal PDF. The second
    equality follows from the ratio of normal densities. For
    :math:`\alpha > 0`, :math:`w(u)` is increasing in
    :math:`u`, giving more weight to tail losses. When
    :math:`\alpha = 0` the weights are uniform and the measure
    reduces to the expected value.

    The Wang transform has the special property of recovering
    the CAPM when applied to normal distributions: if :math:`X`
    is normally distributed, :math:`W(X) = E[X] + \alpha\,
    \text{Std}(X)`.

    References:
        Wang, S.S. (2000). "A class of distortion operators for
        pricing financial and insurance risks." *Journal of
        Risk and Insurance*, 67(1), 15-36.

        Wang, S.S. (2002). "A universal framework for pricing
        financial and insurance risks." *ASTIN Bulletin*,
        32(2), 213-234.

    Args:
        risk_profile: Stochastic risk profile to weight.
        alpha: Wang parameter. Higher values give more weight
            to tail losses. alpha = 0 gives the expected value.
        risk_profile_type: Whether the profile represents losses
            or profits.
    """
    standard_normal = distributions.Normal(0, 1)

    return _spectral_risk_measure(
        risk_profile,
        lambda u: np.exp(-alpha * standard_normal.invcdf(1 - u) - alpha**2 / 2),  # type:ignore
        risk_profile_type,
    )


def tvar(
    risk_profile: StochasticScalar,
    percentile: float,
    risk_profile_type: RiskProfileType = "loss",
) -> RiskMeasureResult:
    r"""Tail Value at Risk (Expected Shortfall).

    TVaR (also called Conditional Tail Expectation or Expected
    Shortfall) is the average of all losses above the
    :math:`\alpha`-quantile. It is the simplest coherent
    alternative to Value at Risk and is widely used in
    insurance regulation (e.g. Solvency II, Swiss Solvency
    Test) and banking (Basel III).

    For a loss variable :math:`X` with CDF :math:`F`:

    .. math::

        \text{TVaR}_\alpha(X)
            = \frac{1}{1-\alpha}
              \int_\alpha^1 F^{-1}(u)\,du

    This can be expressed as a spectral risk measure with the
    step-function risk spectrum:

    .. math::

        w(u) = \begin{cases}
            \frac{1}{1-\alpha} & \text{if } u > \alpha \\
            0 & \text{otherwise}
        \end{cases}

    When :math:`\alpha = 0`, TVaR reduces to the expected
    value. As :math:`\alpha \to 1`, it converges to the
    maximum loss.

    Note that TVaR is the only spectral risk measure whose
    risk spectrum is not continuous, which can lead to
    instability in capital allocation for small simulation
    counts.

    References:
        Acerbi, C. and Tasche, D. (2002). "On the coherence of
        expected shortfall." *Journal of Banking & Finance*,
        26(7), 1487-1503.

        Artzner, P., Delbaen, F., Eber, J.-M. and Heath, D.
        (1999). "Coherent measures of risk." *Mathematical
        Finance*, 9(3), 203-228.

    Args:
        risk_profile: Stochastic risk profile to weight.
        percentile: Percentile level (0-100), e.g. 99 for
            99% TVaR.
        risk_profile_type: Whether the profile represents losses
            or profits.
    """
    if not (0 <= percentile <= 100):
        raise ValueError("percentile must be between 0 and 100")

    n = risk_profile.n_sims
    alpha = percentile / 100
    start = int(np.ceil(alpha * n))
    if start >= n:
        # Degenerate tail: TVaR(100) = max (loss) / min (profit).
        return svar(risk_profile, 100, 100, risk_profile_type)

    # Choose a cutoff on the helper's u-grid so exactly the worst (n-start)
    # simulations receive non-zero weight.
    cutoff = (start - 0.5) / n
    return _spectral_risk_measure(
        risk_profile,
        lambda u: np.where(u > cutoff, 1.0, 0.0),  # type: ignore
        risk_profile_type,
    )


def var(
    risk_profile: StochasticScalar,
    percentile: float,
) -> RiskMeasureResult:
    r"""Value at Risk with capital allocation.

    VaR is the loss threshold at a given percentile of the
    distribution:

    .. math::

        \text{VaR}_p(X) = F^{-1}(p/100)

    VaR is not a coherent risk measure — it can violate
    sub-additivity — but it is widely used in practice
    (e.g. Solvency II) for its simplicity and
    interpretability.

    VaR is the degenerate case of SVaR where the window
    collapses to a single point. Capital is allocated
    entirely to the simulation at the VaR threshold.

    Args:
        risk_profile: Stochastic risk profile.
        percentile: Percentile level (0-100), e.g. 99 for
            99% VaR.
    """
    return svar(risk_profile, percentile, percentile)


def dual_power_transform(
    risk_profile: StochasticScalar,
    beta: float,
    risk_profile_type: RiskProfileType = "loss",
) -> RiskMeasureResult:
    r"""Dual-power distortion transform.

    The dual-power transform applies a power distortion to the
    CDF rather than the survival function, making it the
    natural complement to the proportional hazard transform.
    For a loss variable :math:`X` with CDF :math:`F(x)`:

    .. math::

        D(X) = \int_0^\infty
            \bigl[1 - F(x)^\beta\bigr] \, dx

    The distortion function is
    :math:`g(s) = 1 - (1-s)^\beta`, which is concave for
    :math:`\beta \geq 1`, making the measure coherent.

    Differentiating gives the risk spectrum in terms of the
    CDF :math:`u = F(x)`:

    .. math::

        w(u) = g'(1-u) = \beta\,u^{\beta - 1}

    For :math:`\beta > 1` the weights are increasing in
    :math:`u`, giving more emphasis to tail losses. When
    :math:`\beta = 1` the weights are uniform and the measure
    reduces to the expected value.

    While the PH transform distorts survival probabilities
    (:math:`S \mapsto S^\alpha`), the dual-power transform
    distorts exceedance probabilities from the CDF side
    (:math:`F \mapsto F^\beta`), providing a complementary
    perspective on tail risk.

    References:
        Wirch, J.L. and Hardy, M.R. (1999). "A synthesis of
        risk measures for capital adequacy." *Insurance:
        Mathematics and Economics*, 25(3), 337-347.

        Wang, S.S. (1996). "Premium calculation by transforming
        the layer premium density." *ASTIN Bulletin*, 26(1),
        71-92.

    Args:
        risk_profile: Stochastic risk profile to weight.
        beta: Dual-power parameter (beta >= 1 for coherence).
            Higher values give more weight to tail losses.
        risk_profile_type: Whether the profile represents losses
            or profits.
    """
    return _spectral_risk_measure(
        risk_profile,
        lambda u: beta * u ** (beta - 1),
        risk_profile_type,
    )


def exponential_transform(
    risk_profile: StochasticScalar,
    gamma: float,
    risk_profile_type: RiskProfileType = "loss",
) -> RiskMeasureResult:
    r"""Exponential spectral risk measure.

    The exponential risk spectrum is a smooth, strictly
    increasing weight function that provides continuous
    emphasis across the entire loss distribution, unlike
    TVaR which applies a hard cutoff.

    The (unnormalized) risk spectrum is:

    .. math::

        \phi(u) = \gamma\,e^{\gamma u}

    After normalization to integrate to 1 over :math:`[0, 1]`:

    .. math::

        \phi(u) = \frac{\gamma\,e^{\gamma u}}
                       {e^\gamma - 1}

    For :math:`\gamma > 0` the spectrum is non-decreasing,
    satisfying Acerbi's conditions for a coherent spectral
    risk measure. As :math:`\gamma \to 0` the weights become
    uniform, recovering the expected value. As :math:`\gamma
    \to \infty` the measure concentrates on the maximum loss.

    The exponential form is analytically convenient and arises
    naturally from exponential utility theory: an agent with
    constant absolute risk aversion :math:`\gamma` evaluates
    risk using this spectrum.

    References:
        Acerbi, C. (2002). "Spectral measures of risk: a
        coherent representation of subjective risk aversion."
        *Journal of Banking & Finance*, 26(7), 1505-1518.

        Cotter, J. and Dowd, K. (2006). "Extreme spectral risk
        measures: an application to futures clearinghouse
        margin requirements." *Journal of Banking & Finance*,
        30(12), 3469-3485.

    Args:
        risk_profile: Stochastic risk profile to weight.
        gamma: Risk aversion parameter. Higher values give
            more weight to tail losses. gamma = 0 gives the
            expected value.
        risk_profile_type: Whether the profile represents losses
            or profits.
    """
    return _spectral_risk_measure(
        risk_profile,
        lambda u: np.exp(gamma * u),  # type: ignore
        risk_profile_type,
    )


def svar(
    risk_profile: StochasticScalar,
    lower: float,
    upper: float,
    risk_profile_type: RiskProfileType = "loss",
) -> RiskMeasureResult:
    r"""Spread VaR risk measure.

    The spread VaR (SVaR), also known as 'Window VaR' is the
    conditional expectation of a random variable between two
    percentiles.

    For a loss variable :math:`X` with CDF :math:`F`:

    .. math::

        \text{SVaR}_{\alpha,\beta}(X)
            = \frac{1}{\beta-\alpha}
              \\int_\alpha^\beta F^{-1}(u)\\,du

    This can be expressed as a spectral risk measure with the
    step-function risk spectrum over the half-open interval
    :math:`(\alpha, \beta]`:

    .. math::

        w(u) = \begin{cases}
            \frac{1}{\beta-\alpha}
                & \text{if } \alpha < u \leq \beta \\
            0 & \text{otherwise}
        \\end{cases}

    When ``lower == upper`` the interval collapses to a point
    and the measure reduces to VaR (a single order statistic).

    Args:
        risk_profile: StochasticScalar risk profile to weight.
        lower: Lower percentile (0-100).
        upper: Upper percentile (0-100).
        risk_profile_type: Whether the profile represents losses
            or profits.
    """
    if not (0 <= lower <= upper <= 100):
        raise ValueError("Invalid percentiles: require 0 <= lower <= upper <= 100")

    n = risk_profile.n_sims
    ranks = risk_profile.ranks
    lower_q = lower / 100
    upper_q = upper / 100
    if lower == upper:
        # Degenerate case: VaR at a single percentile
        target_rank = min(int(lower_q * n), n - 1)
        if risk_profile_type == "loss":
            mask = ranks.values == target_rank
        else:
            mask = ranks.values == (n - 1 - target_rank)
        weights_arr = xp.where(mask, float(n), 0.0)
        weights = StochasticScalar(weights_arr)
        value = float(xp.mean(risk_profile.values * weights_arr))
        return RiskMeasureResult(value, weights)

    start = int(np.ceil(lower_q * n))
    end = int(np.ceil(upper_q * n))
    if end <= start:
        # Can occur for small n with a narrow window; fall back to VaR at upper.
        target_rank = min(int(upper_q * n), n - 1)
        if risk_profile_type == "loss":
            mask = ranks.values == target_rank
        else:
            mask = ranks.values == (n - 1 - target_rank)
        weights_arr = xp.where(mask, float(n), 0.0)
        weights = StochasticScalar(weights_arr)
        value = float(xp.mean(risk_profile.values * weights_arr))
        return RiskMeasureResult(value, weights)

    # Choose cutoffs on the helper's u-grid so we select exactly the order
    # statistics in ranks [start, end). This matches the docstring interval
    # (lower, upper] on the u-grid.
    lower_cutoff = (start - 0.5) / n
    upper_cutoff = (end - 0.5) / n
    return _spectral_risk_measure(
        risk_profile,
        lambda u: np.where((u > lower_cutoff) & (u <= upper_cutoff), 1.0, 0.0),  # type: ignore
        risk_profile_type,
    )


# =============================================
# Non-spectral risk measures
# =============================================


def standard_deviation_principle(
    risk_profile: StochasticScalar,
    k: float,
) -> RiskMeasureResult:
    r"""Standard deviation premium principle.

    The standard deviation principle prices risk as the mean
    plus a multiple of the standard deviation:

    .. math::

        \rho(X) = E[X] + k \cdot \sigma(X)

    Capital allocation follows Euler's theorem. The gradient
    of :math:`\rho` with respect to the contribution of
    component :math:`X_i` gives the allocation weight:

    .. math::

        w_j = 1 + k \cdot \frac{X_j - E[X_j]}{\sigma(X)}

    where :math:`X = \sum_i X_i` is the total and :math:`X_j`
    is a single simulation. This ensures that
    :math:`\sum_i E[w_j \cdot X_{i,j}] = \rho(X)`.

    The standard deviation principle is **not** coherent —
    it is not monotone (a uniformly larger loss can receive
    a smaller risk measure if it has lower variance). However,
    it is widely used in practice for its simplicity and its
    connection to the CAPM.

    References:
        Dhaene, J., Tsanakas, A., Valdez, E.A. and Vanduffel, S.
        (2012). "Optimal capital allocation principles."
        *Journal of Risk and Insurance*, 79(1), 1-28.

        McNeil, A.J., Frey, R. and Embrechts, P. (2015).
        *Quantitative Risk Management*, 2nd ed. Princeton
        University Press. Section 8.4.

    Args:
        risk_profile: Stochastic risk profile.
        k: Loading factor (multiplier on standard deviation).
    """
    mean = float(np.mean(risk_profile.values))
    std = float(np.std(risk_profile.values))
    value = mean + k * std

    # Euler weights: w_j = 1 + k * (X_j - mean) / std
    if std > 0:
        weights_arr = 1.0 + k * (risk_profile.values - mean) / std
    else:
        weights_arr = xp.ones(risk_profile.n_sims)
    weights = StochasticScalar(weights_arr)
    return RiskMeasureResult(value, weights)


def percentile_layer(
    risk_profile: StochasticScalar,
    capital: float,
) -> RiskMeasureResult:
    r"""Capital Allocation by Percentile Layer (CAPL).

    CAPL allocates a given capital amount across simulations
    by splitting the aggregate loss distribution into
    percentile layers. For sorted non-negative losses
    :math:`X_{(1)} \leq \cdots \leq X_{(N)}`, define the
    capped loss:

    .. math::

        Y_{(i)} = \min(X_{(i)},\, C)

    with :math:`Y_{(0)} = 0`. Layer :math:`j` has thickness
    :math:`\Delta_j = Y_{(j)} - Y_{(j-1)}` and is penetrated
    by :math:`N - j + 1` simulations (those with losses
    :math:`\geq X_{(j)}`). Each simulation receives an equal
    share from every layer it reaches:

    .. math::

        C_{(i)} = \sum_{j=1}^{i}
            \frac{\Delta_j}{N - j + 1}

    The per-simulation weight is:

    .. math::

        w_{(i)} = N \cdot \frac{C_{(i)}}{X_{(i)}}

    with :math:`w_{(i)} = 0` when :math:`X_{(i)} = 0`.

    **Telescoping property.** Swapping the order of summation
    in :math:`\sum_i C_{(i)}` shows that each layer's capital
    :math:`\Delta_j` is counted exactly :math:`N - j + 1`
    times and then divided by the same factor, so:

    .. math::

        \sum_i C_{(i)} = \sum_j \Delta_j
                       = Y_{(N)} = \min(X_{(N)}, C)

    The value equals :math:`C` whenever the maximum loss
    exceeds :math:`C`.

    Unlike spectral risk measures, CAPL weights depend on the
    actual simulated values (not just percentiles) and do not
    in general average to 1.

    For a multivariate loss :math:`X = \sum_k X_k`, passing
    each component through ``.allocate()`` gives its capital
    share :math:`C_k`, and :math:`\sum_k C_k = C`.

    References:
        Bodoff, N.M. (2009). "Capital allocation by percentile
        layer." *Variance*, 3(1), 13-26.

    Args:
        risk_profile: Stochastic aggregate loss profile
            (non-negative values).
        capital: Total capital to allocate (typically VaR at
            a chosen confidence level).
    """
    n = risk_profile.n_sims
    values = risk_profile.values

    order = xp.argsort(values)
    sorted_vals = values[order]

    y = xp.minimum(sorted_vals, capital)
    y_shifted = xp.zeros_like(y)
    y_shifted[1:] = y[:-1]
    delta_y = y - y_shifted

    # Layer j is penetrated by (n - j) sims (0-indexed)
    counts = xp.arange(n, 0, -1, dtype=float)
    per_sim_layer = delta_y / counts
    alloc = xp.cumsum(per_sim_layer)

    sorted_weights = xp.where(
        sorted_vals > 0,
        n * alloc / sorted_vals,
        0.0,
    )

    weights_arr = xp.empty(n)
    weights_arr[order] = sorted_weights

    weights = StochasticScalar(weights_arr)
    value = float(xp.mean(values * weights_arr))
    return RiskMeasureResult(value, weights)


if __name__ == "__main__":
    from pal import ProteusVariable, copulas, distributions

    x = ProteusVariable(
        dim_name="lob",
        values={
            "lob1": distributions.Gamma(alpha=2, theta=0.5).generate(),
            "lob2": distributions.Gamma(alpha=3, theta=0.5).generate(),
        },
    )

    copulas.GalambosCopula(2).apply(x)

    total = x.sum()

    # Spectral: value + allocation
    rm = proportional_hazards_transform(total, 0.5, "loss")
    print(f"PH risk measure: {rm.value:.4f}")
    print(f"PH allocation:   {rm.allocate(x)}")

    # TVaR
    rm_tvar = tvar(total, 0.99)
    print(f"TVaR 99%:        {rm_tvar.value:.4f}")
    print(f"TVaR allocation: {rm_tvar.allocate(x)}")

    # Standard deviation principle
    rm_sd = standard_deviation_principle(total, 2.0)
    print(f"Std dev (k=2):   {rm_sd.value:.4f}")
    print(f"Std dev alloc:   {rm_sd.allocate(x)}")

    # Percentile layer at TVaR 99% capital level
    rm_pl = percentile_layer(total, rm_tvar.value)
    print(f"Pctl layer:      {rm_pl.value:.4f}")
    print(f"Pctl allocation: {rm_pl.allocate(x)}")
