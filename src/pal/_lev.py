"""Limited expected values of numeric univariate distributions."""

from __future__ import annotations

import datetime as dt
import typing as t

import numpy as np
from scipy import special, stats

from . import distributions as dist
from ._gig import _gig_log_moments
from ._lev_integrals import _adaptive_intervals, _positive_survival_integrals
from ._mean_excess import (
    _gamma_logsf_coordinate,
    _gev_small_tail,
    _lognormal_coordinate,
    _normal_tail_difference,
    _positive_log_survival_array,
)
from ._severity_probability import (
    _inverse_gaussian_log_tails,
    _log_beta_probability,
    _log_gamma_lower_probability,
    _log_survival_from_log_hazard,
)
from .fitting import _frozen_distribution, _other_families, _supported_families
from .stochastic_scalar import StochasticScalar

if t.TYPE_CHECKING:
    from .distributions import DistributionBase


def _log_expm1_positive(x: np.ndarray) -> np.ndarray:
    """Return log(expm1(x)) without overflowing its intermediate value."""
    return np.where(x > 40, x + np.log1p(-np.exp(-x)), np.log(np.expm1(x)))


def _scaled_power_integral(log_scale: float, exponent: float, t_: np.ndarray) -> np.ndarray:
    """Evaluate scale times the exponential integral in stable log units."""
    if exponent == 0:
        return np.exp(log_scale + np.log(t_))
    v = exponent * t_
    log_value = _log_expm1_positive(v) if exponent > 0 else np.log(-np.expm1(v))
    # Very small exponent times t may underflow even when the ratio is finite.
    small = np.abs(v) < 1e-8
    value = np.exp(log_scale + log_value - np.log(abs(exponent)))
    value[small] = np.exp(log_scale + np.log(t_[small])) * (1 + v[small] / 2)
    return value


def _pareto_lev(d: np.ndarray, shape: float, scale: float) -> np.ndarray:
    x = np.maximum(d, scale)
    log_ratio = np.log(x) - np.log(scale)
    near = x <= 2 * scale
    log_ratio[near] = np.log1p((x[near] - scale) / scale)
    return np.minimum(d, scale) + _scaled_power_integral(np.log(scale), 1 - shape, log_ratio)


def _gpd_lev(d: np.ndarray, shape: float, scale: float) -> np.ndarray:
    x = np.maximum(d, 0)
    if shape < 0:
        x = np.minimum(x, -scale / shape)
    if shape == 0:
        ratio = x / scale
        return np.where(ratio < 1e-8, x * (1 - ratio / 2), scale * (-np.expm1(-ratio)))
    log_ratio = np.log(x) - np.log(scale)
    if shape > 0:
        log_base = np.logaddexp(0, np.log(shape) + log_ratio)
    else:
        log_base = np.log1p(np.maximum(-1, shape * (x / scale)))
    t_ = log_base / shape
    small = np.log(abs(shape)) + log_ratio < -36
    t_[small] = np.exp(log_ratio[small])
    # At a finite endpoint the integral is scale/(1-shape).
    value = _scaled_power_integral(np.log(scale), shape - 1, t_)
    return np.where(t_ < 1e-8, x * (1 - t_ / 2), value)


def _capped_count(d: np.ndarray, moment: np.ndarray, survival: np.ndarray) -> np.ndarray:
    """Combine a truncated count moment and capped tail.

    Args:
        d: Policy limits in count units, which need not be integers.
        moment: First moments truncated at the integer parts of the limits.
        survival: Probabilities of exceeding the integer parts of the limits.
    """
    return np.where(d <= 0, d, moment + d * survival)


def _poisson_lev(d: np.ndarray, mean: float) -> np.ndarray:
    """Evaluate the Poisson capped expectation using shifted count probabilities.

    Args:
        d: Policy limits in count units.
        mean: Nonnegative expected event count.
    """
    k = np.floor(d)
    rv = stats.poisson(mean)
    return _capped_count(d, mean * rv.cdf(k - 1), rv.sf(k))


def _binomial_lev(d: np.ndarray, n: float, p: float) -> np.ndarray:
    """Evaluate the binomial capped expectation using shifted count probabilities.

    Args:
        d: Policy limits in count units.
        n: Number of Bernoulli trials, represented as an integer-valued scalar.
        p: Probability of success on each trial.
    """
    trials = int(n)
    k = np.floor(d)
    rv = stats.binom(trials, p)
    moment = trials * p * stats.binom.cdf(k - 1, trials - 1, p) if trials > 0 else np.zeros_like(d)
    return _capped_count(d, moment, rv.sf(k))


def _negative_binomial_lev(d: np.ndarray, n: float, p: float) -> np.ndarray:
    """Evaluate the negative binomial capped expectation.

    Args:
        d: Policy limits in count units.
        n: Positive negative binomial size parameter.
        p: Probability of success under the failures-before-successes convention.
    """
    k = np.floor(d)
    rv = stats.nbinom(n, p)
    moment = n * (1 - p) / p * stats.nbinom.cdf(k - 1, n + 1, p)
    return _capped_count(d, moment, rv.sf(k))


def _hypergeometric_lev(d: np.ndarray, ngood: float, nbad: float, n: float) -> np.ndarray:
    """Evaluate the hypergeometric capped expectation using a reduced population.

    Args:
        d: Policy limits in count units.
        ngood: Number of successful items in the population.
        nbad: Number of unsuccessful items in the population.
        n: Number of items drawn without replacement.
    """
    good, bad, draws = int(ngood), int(nbad), int(n)
    population = good + bad
    k = np.floor(d)
    rv = stats.hypergeom(population, good, draws)
    moment = (
        draws * good / population * stats.hypergeom.cdf(k - 1, population - 1, good - 1, draws - 1)
        if good > 0 and draws > 0
        else np.zeros_like(d)
    )
    return _capped_count(d, moment, rv.sf(k))


def _mbbefd_lev(d: np.ndarray, g: float, b: float) -> np.ndarray:
    """Integrate MBBEFD survival without overflowing the exposure-curve factors.

    Args:
        d: Policy limits as fractions of the insured value.
        g: Reciprocal of the probability of a total loss.
        b: MBBEFD exposure-curve shape parameter.
    """
    x = np.clip(d, 0, 1)
    if g == 1 or b == 0:
        return np.minimum(d, 1)
    if b == 1:
        value = np.log1p((g - 1) * x) / (g - 1)
    else:
        log_b = np.log(b)
        log_gb = np.log(g) + log_b
        log_abs_b = float(_log_expm1_positive(np.array([log_b]))[0]) if b > 1 else np.log(-np.expm1(log_b))
        v = x * log_b
        log_abs_increment = _log_expm1_positive(v) if b > 1 else np.log(-np.expm1(v))
        if log_gb == 0:
            value = np.exp(log_abs_increment - np.log(abs(log_b)))
        else:
            log_abs_gb = float(_log_expm1_positive(np.array([log_gb]))[0]) if log_gb > 0 else np.log(-np.expm1(log_gb))
            log_coefficient = log_abs_gb - log_abs_b
            log_q = log_coefficient + log_abs_increment
            if log_gb > 0:
                log_numerator = np.log(np.logaddexp(0.0, log_q))
                small = log_q < -18
            else:
                # For q near -1, use the sum of two positive terms in 1+q.
                q = np.exp(np.minimum(log_q, 0.0))
                log_ratio = np.logaddexp(v, np.log(g - 1) + log_b - log_abs_b + log_abs_increment)
                log_numerator = np.log(-np.where(q < 0.5, np.log1p(-q), log_ratio))
                small = log_q < -18
            value = np.exp(log_numerator - log_coefficient - np.log(abs(log_b)))
            sign = 1 if log_gb > 0 else -1
            value[small] = np.exp(log_abs_increment[small] - np.log(abs(log_b))) * (1 - sign * np.exp(log_q[small]) / 2)
    return np.where(d <= 0, d, value)


def _uniform_lev(d: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate the uniform capped expectation, including a point mass.

    Args:
        d: Policy limits in original loss units.
        a: Lower endpoint of the uniform support.
        b: Upper endpoint of the uniform support.
    """
    if a == b:
        return np.minimum(d, a)
    x = np.clip(d, a, b) - a
    return np.where(d <= a, d, a + x * (1 - 0.5 * x / (b - a)))


def _beta_lev(d: np.ndarray, alpha: float, beta: float, scale: float, loc: float) -> np.ndarray:
    """Evaluate the scaled and shifted beta capped expectation.

    Args:
        d: Policy limits in original loss units.
        alpha: First positive beta shape parameter.
        beta: Second positive beta shape parameter.
        scale: Width of the bounded support.
        loc: Lower endpoint of the support.
    """
    x = np.clip(d - loc, 0, scale)
    u = x / scale
    moment = scale * alpha / (alpha + beta) * special.betainc(alpha + 1, beta, u)
    return np.where(d <= loc, d, loc + moment + x * special.betaincc(alpha, beta, u))


def _normal_lev(d: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Evaluate the normal capped expectation using normal tail probabilities.

    Args:
        d: Policy limits in original loss units.
        mu: Normal mean.
        sigma: Positive normal standard deviation.
    """
    z = (d - mu) / sigma
    return mu * special.ndtr(z) - sigma * np.exp(stats.norm.logpdf(z)) + d * special.ndtr(-z)


def _logistic_lev(d: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Evaluate the logistic capped expectation with stable softplus expressions.

    Args:
        d: Policy limits in original loss units.
        mu: Logistic location.
        sigma: Positive logistic scale.
    """
    z = (d - mu) / sigma
    return np.where(z > 0, mu - sigma * np.logaddexp(0, -z), d - sigma * np.logaddexp(0, z))


def _students_t_lev(d: np.ndarray, nu: float, mu: float, sigma: float) -> np.ndarray:
    """Evaluate the Student's t capped expectation, including divergent left moments.

    Args:
        d: Policy limits in original loss units.
        nu: Positive degrees of freedom.
        mu: Student's t location.
        sigma: Positive Student's t scale.
    """
    if nu <= 1:
        return np.full_like(d, -np.inf)
    z = (d - mu) / sigma
    density_moment = np.exp(stats.t.logpdf(z, nu) + np.logaddexp(np.log(nu), 2 * np.log(np.abs(z))))
    return mu * stats.t.cdf(z, nu) - sigma * density_moment / (nu - 1) + d * stats.t.sf(z, nu)


def _noncentral_chi_squared_lev(d: np.ndarray, df: float, nonc: float) -> np.ndarray:
    """Evaluate the noncentral chi-squared capped expectation using shifted degrees of freedom.

    Args:
        d: Policy limits in original loss units.
        df: Positive degrees of freedom.
        nonc: Nonnegative noncentrality parameter.
    """
    moment = df * stats.ncx2.cdf(d, df + 2, nonc) + nonc * stats.ncx2.cdf(d, df + 4, nonc)
    return np.where(d <= 0, d, moment + d * stats.ncx2.sf(d, df, nonc))


_REMAINING_LEV_FORMULAS: dict[type[DistributionBase], t.Callable[..., np.ndarray]] = {
    dist.Poisson: _poisson_lev,
    dist.Binomial: _binomial_lev,
    dist.Bernoulli: _binomial_lev,
    dist.NegBinomial: _negative_binomial_lev,
    dist.HyperGeometric: _hypergeometric_lev,
    dist.Uniform: _uniform_lev,
    dist.Beta: _beta_lev,
    dist.Normal: _normal_lev,
    dist.Logistic: _logistic_lev,
    dist.StudentsT: _students_t_lev,
    dist.NonCentralChiSquared: _noncentral_chi_squared_lev,
    dist.MBBEFD: _mbbefd_lev,
}


def _remaining_lev(family: type[DistributionBase], p: dict[str, float], d: np.ndarray) -> np.ndarray | None:
    """Dispatch the capped moments for the remaining numeric families.

    Args:
        family: PAL distribution class.
        p: Natural distribution parameter values.
        d: Policy limits in original loss units.

    Returns:
        Vectorised capped expectations, or None for families handled elsewhere.
    """
    formula = _REMAINING_LEV_FORMULAS.get(family)
    return None if formula is None else formula(d, **p)


def _gb2_lev(x: np.ndarray, a: float, b: float, p: float, q: float) -> np.ndarray:
    """Use incomplete-beta lower moments for GB2 and its special cases.

    Args:
        x: Nonnegative limits relative to the location.
        a: Positive power of the scaled loss.
        b: Positive loss scale.
        p: Lower-tail beta shape.
        q: Upper-tail beta shape.

    Returns:
        Capped expectations, with NaN marking unresolved analytical values.
    """
    alpha, beta = p, q
    shift = 1 / a
    log_x = np.log(x)
    power = a * (log_x - np.log(b))
    log_u, log_w = -np.logaddexp(0, -power), -np.logaddexp(0, power)
    log_sf = _log_beta_probability(beta, alpha, log_w, log_u)
    if beta > shift:
        log_mean = np.log(b) + special.betaln(alpha + shift, beta - shift) - special.betaln(alpha, beta)
        log_moment = log_mean + _log_beta_probability(alpha + shift, beta - shift, log_u, log_w)
        value = np.exp(log_moment) + np.exp(log_x + log_sf)
        # Gamma-ratio cancellation for very large shape arguments can
        # contaminate an otherwise finite answer. Use survival integration.
        if max(alpha, beta, shift) > 1e7:
            value[:] = np.nan
    else:
        # The lower incomplete beta integral exists for finite limits even
        # when the unconditional mean does not. Its hypergeometric form is
        # well conditioned away from the upper endpoint.
        value = np.full_like(x, np.nan)
        low = log_u < -np.log(2)
        h = special.hyp2f1(alpha + shift, 1 - beta + shift, alpha + shift + 1, np.exp(log_u[low]))
        log_moment = (
            np.log(b) + (alpha + shift) * log_u[low] - np.log(alpha + shift) - special.betaln(alpha, beta) + np.log(h)
        )
        value[low] = np.exp(log_moment) + np.exp(log_x[low] + log_sf[low])
    return np.where(x == 0, 0.0, value)


def _burr_lev(x: np.ndarray, power: float, shape: float, scale: float) -> np.ndarray:
    """Evaluate Burr XII as a special case of GB2.

    Args:
        x: Nonnegative limits relative to the location.
        power: Positive exponent of the scaled loss.
        shape: Positive Burr shape.
        scale: Positive loss scale.
    """
    return _gb2_lev(x, power, scale, 1.0, shape)


def _inverse_burr_lev(x: np.ndarray, power: float, shape: float, scale: float) -> np.ndarray:
    """Evaluate Burr III as a special case of GB2.

    Args:
        x: Nonnegative limits relative to the location.
        power: Positive exponent of the scaled loss.
        shape: Positive inverse Burr shape.
        scale: Positive loss scale.
    """
    return _gb2_lev(x, power, scale, shape, 1.0)


def _loglogistic_lev(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """Evaluate the log-logistic capped expectation.

    Args:
        x: Nonnegative limits relative to the location.
        shape: Positive log-logistic tail index.
        scale: Positive loss scale.
    """
    return _gb2_lev(x, shape, scale, 1.0, 1.0)


def _paralogistic_lev(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """Evaluate the paralogistic capped expectation.

    Args:
        x: Nonnegative limits relative to the location.
        shape: Positive common Burr power and shape.
        scale: Positive loss scale.
    """
    return _burr_lev(x, shape, shape, scale)


def _inverse_paralogistic_lev(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """Evaluate the inverse paralogistic capped expectation.

    Args:
        x: Nonnegative limits relative to the location.
        shape: Positive common inverse Burr power and shape.
        scale: Positive loss scale.
    """
    return _inverse_burr_lev(x, shape, shape, scale)


def _exponential_lev(x: np.ndarray, scale: float) -> np.ndarray:
    """Evaluate the exponential capped expectation as a zero-shape GPD.

    Args:
        x: Nonnegative limits relative to the location.
        scale: Positive exponential mean.
    """
    return _gpd_lev(x, 0.0, scale)


def _gamma_lev(x: np.ndarray, alpha: float, theta: float) -> np.ndarray:
    """Evaluate the gamma capped expectation using incomplete gamma probabilities.

    Args:
        x: Nonnegative limits relative to the location.
        alpha: Positive gamma shape.
        theta: Positive gamma scale.
    """
    log_z = np.log(x) - np.log(theta)
    log_sf = _gamma_logsf_coordinate(alpha, log_z)
    value = np.exp(np.log(alpha) + np.log(theta) + _log_gamma_lower_probability(alpha + 1, log_z))
    return value + np.exp(np.log(x) + log_sf)


def _lognormal_lev(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Evaluate the lognormal capped expectation without an overflowing mean.

    Args:
        x: Nonnegative limits.
        mu: Mean of log loss.
        sigma: Positive standard deviation of log loss.
    """
    z = _lognormal_coordinate(x, mu, sigma)
    log_x = np.log(x)
    shifted = z - sigma
    log_moment = np.empty_like(x)
    left = shifted <= 0
    # The normal exponential tilt cancels analytically, including when
    # the unconditional mean is outside the floating-point range.
    log_moment[left] = log_x[left] - z[left] ** 2 / 2 + np.log(0.5 * special.erfcx(-shifted[left] / np.sqrt(2)))
    log_moment[~left] = mu + sigma**2 / 2 + special.log_ndtr(shifted[~left])
    return np.exp(log_moment) + np.exp(log_x + special.log_ndtr(-z))


def _weibull_lev(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """Evaluate the Weibull survival integral using the incomplete gamma function.

    Args:
        x: Nonnegative limits relative to the location.
        shape: Positive Weibull shape.
        scale: Positive loss scale.
    """
    a = 1 / shape
    log_z = shape * (np.log(x) - np.log(scale))
    value = np.exp(np.log(scale) + special.gammaln(1 + a) + _log_gamma_lower_probability(a, log_z))
    if a > 1e7:
        value[:] = np.nan
    return value


def _gig_lev(x: np.ndarray, p: float, chi: float, psi: float) -> np.ndarray:
    """Evaluate GIG capped moments using the shared log-coordinate moment kernel.

    Args:
        x: Nonnegative limits relative to the location.
        p: GIG index controlling the power of loss in the density.
        chi: Positive inverse-loss coefficient.
        psi: Positive loss coefficient.
    """
    _, logsf, log_moment, _ = _gig_log_moments(x, p, chi, psi)
    return np.exp(log_moment) + np.exp(np.log(x) + logsf)


def _inverse_generalized_gamma_lev(
    x: np.ndarray,
    alpha: float,
    scale: float,
    power: float,
    boundary_logsf: t.Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Evaluate capped inverse-generalized-gamma moments.

    Args:
        x: Nonnegative limits relative to the location.
        alpha: Positive shape of the underlying unit-scale gamma variable.
        scale: Positive loss scale.
        power: Positive power of the inverse transformation.
        boundary_logsf: Log-survival kernel at the infinite-mean boundary,
            accepting the log of the underlying gamma coordinate.

    Returns:
        Capped expectations, with NaN marking cases requiring survival integration.
    """
    k = 1 / power
    log_z = power * (np.log(scale) - np.log(x))
    z = np.exp(log_z)
    if alpha > k:
        logmean = np.log(scale) + special.gammaln(alpha - k) - special.gammaln(alpha)
        value = np.exp(logmean + _gamma_logsf_coordinate(alpha - k, log_z))
        return value + np.exp(np.log(x) + _log_gamma_lower_probability(alpha, log_z))
    if alpha == k:
        e1 = np.where(log_z < -36, -np.euler_gamma - log_z, special.exp1(z))
        log_sf = boundary_logsf(log_z)
        return np.exp(np.log(scale) - special.gammaln(alpha) + np.log(e1)) + np.exp(np.log(x) + log_sf)
    return np.full_like(x, np.nan)


def _inverse_gamma_lev(x: np.ndarray, alpha: float, theta: float) -> np.ndarray:
    """Evaluate the inverse gamma capped expectation.

    Args:
        x: Nonnegative limits relative to the location.
        alpha: Positive inverse gamma shape.
        theta: Positive inverse gamma scale.
    """
    return _inverse_generalized_gamma_lev(
        x, alpha, theta, 1.0, lambda log_z: _log_gamma_lower_probability(alpha, log_z)
    )


def _inverse_weibull_lev(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """Evaluate the inverse Weibull capped expectation.

    Args:
        x: Nonnegative limits relative to the location.
        shape: Positive inverse Weibull tail index.
        scale: Positive loss scale.
    """
    return _inverse_generalized_gamma_lev(x, 1.0, scale, shape, lambda log_z: _log_survival_from_log_hazard(log_z, np))


def _inverse_exponential_lev(x: np.ndarray, scale: float) -> np.ndarray:
    """Evaluate the inverse exponential capped expectation.

    Args:
        x: Nonnegative limits relative to the location.
        scale: Positive loss scale.
    """
    return _inverse_weibull_lev(x, 1.0, scale)


def _loggamma_lev(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Evaluate the log-gamma capped expectation, including infinite means.

    Args:
        x: Nonnegative limits; the distribution has lower support one.
        alpha: Positive shape of the gamma-distributed log loss.
        beta: Positive scale of the gamma-distributed log loss.
    """
    z = np.log(np.maximum(x, 1)) / beta
    if beta < 1:
        moment = np.exp(-alpha * np.log1p(-beta) + np.log(special.gammainc(alpha, (1 - beta) * z)))
    elif beta == 1:
        moment = np.exp(alpha * np.log(z) - special.gammaln(alpha + 1))
    else:
        moment = np.exp(
            alpha * np.log(z) - special.gammaln(alpha + 1) + np.log(special.hyp1f1(alpha, alpha + 1, (beta - 1) * z))
        )
    return np.where(x <= 1, x, moment + x * special.gammaincc(alpha, z))


def _inverse_gaussian_lev(x: np.ndarray, mu: float, lambda_: float) -> np.ndarray:
    """Evaluate inverse Gaussian capped moments using stable normal-tail differences.

    Args:
        x: Nonnegative limits.
        mu: Positive inverse Gaussian mean.
        lambda_: Positive inverse Gaussian shape.
    """
    value = np.zeros_like(x)
    pos = x > 0
    xp = x[pos]
    a, b, root, log_a, log_b = _inverse_gaussian_log_tails(xp, mu, lambda_)
    log_lower_a = special.log_ndtr(a)
    gap = 2 * np.sqrt(lambda_) * np.sqrt(xp) / mu
    log_moment = np.log(mu) + _normal_tail_difference(-a, b, gap, log_lower_a, log_b)
    log_sf = _normal_tail_difference(a, b, 2 * root, log_a, log_b)
    value[pos] = np.exp(log_moment) + np.exp(np.log(xp) + log_sf)
    return value


_SEVERITY_LEV_FORMULAS: dict[type[DistributionBase], t.Callable[..., np.ndarray]] = {
    dist.Pareto: _pareto_lev,
    dist.GPD: _gpd_lev,
    dist.Exponential: _exponential_lev,
    dist.Gamma: _gamma_lev,
    dist.LogNormal: _lognormal_lev,
    dist.Weibull: _weibull_lev,
    dist.GeneralizedInverseGaussian: _gig_lev,
    dist.GeneralizedBetaPrime: _gb2_lev,
    dist.Burr: _burr_lev,
    dist.LogLogistic: _loglogistic_lev,
    dist.Paralogistic: _paralogistic_lev,
    dist.InverseBurr: _inverse_burr_lev,
    dist.InverseParalogistic: _inverse_paralogistic_lev,
    dist.InverseGamma: _inverse_gamma_lev,
    dist.InverseWeibull: _inverse_weibull_lev,
    dist.InverseExponential: _inverse_exponential_lev,
    dist.LogGamma: _loggamma_lev,
    dist.InverseGaussian: _inverse_gaussian_lev,
}


def _severity_lev(family: type[DistributionBase], p: dict[str, float], d: np.ndarray) -> np.ndarray:
    """Dispatch vectorised capped severity moments and apply the location shift.

    Args:
        family: PAL severity distribution class.
        p: Natural distribution parameters, including location where applicable.
        d: Policy limits in original loss units.
    """
    formula = _SEVERITY_LEV_FORMULAS.get(family)
    if formula is None:
        return np.full_like(d, np.nan)
    loc = p.get("loc", 0.0)
    x = np.maximum(d - loc, 0)
    value = formula(x, **{name: value for name, value in p.items() if name != "loc"})
    return np.where(d <= loc, d, loc + value)


def _positive_fallback(family: type[DistributionBase], p: dict[str, float], d: np.ndarray) -> np.ndarray:
    """Recover unresolved positive severities by a sorted forward sweep."""
    loc = p.get("loc", 0.0)
    x = np.maximum(d - loc, 0)
    out = d.copy()
    positive = x > 0
    rv = None if family in (dist.LogGamma, dist.GeneralizedBetaPrime) else _frozen_distribution(family, p)

    def log_survival(t_: np.ndarray) -> np.ndarray:
        value = _positive_log_survival_array(family, p, t_)
        if value is not None:
            return value
        if rv is None:
            raise RuntimeError(f"Missing survival kernel for {family.__name__}")
        return np.asarray(rv.logsf(loc + np.exp(t_)))

    scale = p.get("scale", p.get("b", p.get("theta", p.get("mu", 1.0))))
    log_scale = p["mu"] if family is dist.LogNormal else np.log(scale)
    if family in (dist.Burr, dist.LogLogistic, dist.Paralogistic, dist.InverseBurr, dist.InverseParalogistic):
        power = p.get("power", p["shape"])
        shape = 1.0 if family is dist.LogLogistic else p["shape"]
        log_h = float(_log_expm1_positive(np.array([np.log(2) / shape]))[0])
        direction = -1 if family in (dist.InverseBurr, dist.InverseParalogistic) else 1
        log_scale += direction * log_h / power
    elif family is dist.GeneralizedBetaPrime:
        u = special.betaincinv(p["p"], p["q"], 0.5)
        w = special.betaincinv(p["q"], p["p"], 0.5)
        if u > 0 and w > 0:
            log_scale += (np.log(u) - np.log(w)) / p["a"]
    out[positive] = loc + _positive_survival_integrals(log_survival, np.log(x[positive]), float(log_scale))
    return out


def _gev_mean(shape: float) -> float:
    """Standardized GEV mean, with a stable Gumbel limit."""
    if abs(shape) < 1e-5:
        gamma = np.euler_gamma
        second = (gamma**2 + np.pi**2 / 6) / 2
        third = (gamma**3 + gamma * np.pi**2 / 2 + 2 * special.zeta(3)) / 6
        return float(gamma + shape * (second + shape * third))
    return float(np.expm1(special.gammaln(1 - shape)) / shape)


def _gev_lev(p: dict[str, float], d: np.ndarray) -> np.ndarray:
    """Vectorized GEV LEV, using a forward integral only for difficult shapes."""
    xi, loc, scale = p["shape"], p["loc"], p["scale"]
    z = (d - loc) / scale
    if xi == 0:
        # d-E1(exp(-z))*scale loses the limiting mean by cancellation.
        return np.where(z > 35, loc + scale * np.euler_gamma, d - scale * special.exp1(np.exp(-z)))
    if xi >= 1:
        lower = loc - scale / xi
        x = np.maximum(d - lower, 0)
        out = d.copy()

        def log_survival(t_: np.ndarray) -> np.ndarray:
            log_y = -(t_ + np.log(xi) - np.log(scale)) / xi
            return _log_survival_from_log_hazard(log_y, np)

        pos = x > 0
        out[pos] = lower + _positive_survival_integrals(log_survival, np.log(x[pos]), np.log(scale) - np.log(xi))
        return out

    # The finite-mean case can use the same tail integral as mean excess.
    log_gamma = special.gammaln(1 - xi)
    if xi < -1e-5 and log_gamma > 40:
        mean = loc - np.exp(np.log(scale) + log_gamma - np.log(-xi))
    else:
        mean = loc + scale * _gev_mean(xi)
    lower = loc - scale / xi if xi > 0 else -np.inf
    upper = loc - scale / xi if xi < 0 else np.inf
    inside = (d > lower) & (d < upper)
    out = np.where(d <= lower, d, mean)
    zi = z[inside]
    base = 1 + xi * zi
    if xi < 0:
        near = base < 0.5
        base[near] = (-xi / scale) * (upper - d[inside][near])
    log_base = np.where(base < 0.5, np.log(base), np.log1p(xi * zi))
    log_y = -log_base / xi
    y = np.exp(log_y)
    values = d[inside].copy()
    tail = (y < 1e-4) & (xi < 0.9)
    log_area = np.log(scale) + (1 - xi) * log_y[tail] + np.log(_gev_small_tail(xi, y[tail]))
    # The series in _gev_small_tail is conditional; restore S/y.
    sf_factor = np.where(y[tail] > 0, -np.expm1(-y[tail]) / y[tail], 1.0)
    values[tail] = mean - np.exp(log_area) * sf_factor
    central = (~tail) & (y < 40)
    if xi <= -1e-5:
        # The lower-tail moment can be substantial even for very small CDF
        # when the left tail is heavy (large negative shape).
        rest = ~tail
        values[rest] = d[inside][rest] - np.exp(
            np.log(scale) + special.gammaln(-xi) + _gamma_logsf_coordinate(-xi, log_y[rest])
        )
    elif xi >= 1e-5:
        yc = y[central]
        log_f = -yc
        log_moment_ratio = special.gammaln(1 - xi) + _gamma_logsf_coordinate(1 - xi, log_y[central]) - log_f
        log_s = _log_survival_from_log_hazard(log_y[central], np)
        capped = np.sign(zi[central]) * np.exp(np.log(np.abs(zi[central])) + np.log(scale) + log_s)
        lower_moment = scale * np.exp(log_f) * np.expm1(log_moment_ratio) / xi
        values[central] = loc + capped + lower_moment
    else:
        # At shapes extremely close to zero, the incomplete-gamma difference
        # is ill-conditioned. Integrate CDF from a point with negligible mass.
        c = zi[central]
        if c.size:
            anchor = np.expm1(-xi * np.log(40)) / xi

            def log_cdf(t_: np.ndarray) -> np.ndarray:
                return -np.exp(-np.log1p(xi * t_) / xi)

            areas = _adaptive_intervals(log_cdf, np.r_[anchor, c[:-1]], c)
            values[central] = d[inside][central] - scale * np.cumsum(areas)
    out[inside] = values
    return out


def limited_expected_value(distribution: DistributionBase, limit: float | np.ndarray) -> float | np.ndarray:
    """Evaluate the expected capped loss, preserving limit shape and order."""
    family = type(distribution)
    p: dict[str, float] = {}
    for name, value in distribution._params.items():
        if isinstance(value, StochasticScalar):
            raise ValueError("lev requires scalar distribution parameters")
        if isinstance(value, (dt.date, np.datetime64)) or family is dist.Categorical:
            raise TypeError("lev requires numeric losses")
        p[name] = float(value)
    inputs = np.asarray(limit, dtype=float)
    result = np.full(inputs.shape, np.nan)
    valid = np.isfinite(inputs)
    d, inverse = np.unique(inputs[valid], return_inverse=True)
    with np.errstate(all="ignore"):
        out = _remaining_lev(family, p, d)
        if out is None:
            if family is dist.GEV:
                out = _gev_lev(p, d)
            elif family in (*_supported_families(), *_other_families()):
                out = _severity_lev(family, p, d)
                loc = p.get("loc", 0.0)
                bad = (~np.isfinite(out)) | (out > d + 1e-12 * np.maximum(1, np.abs(d))) | ((d > loc) & (out <= loc))
                if np.any(bad):
                    out[bad] = _positive_fallback(family, p, d[bad])
            else:
                raise ValueError(f"No numeric limited expected value for {family.__name__}")
        result[valid] = out[inverse]
    return float(result) if result.ndim == 0 else result
