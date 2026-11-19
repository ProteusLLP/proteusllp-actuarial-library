"""Reference tests for fitting PAL count, real-valued and labelled distributions."""

import datetime as dt

import numpy as np
import pytest
from scipy import integrate, stats

from pal._maths import asnumpy
from pal.distributions import (
    MBBEFD,
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    HyperGeometric,
    Logistic,
    NegBinomial,
    NonCentralChiSquared,
    Normal,
    Poisson,
    StudentsT,
    Uniform,
)
from pal.fitting_remaining import fit_remaining


def test_weighted_poisson_analytic_mle_and_conditional_excess() -> None:
    x = np.array([0, 1, 2, 0, 4])
    w = np.array([1.0, 2.0, 1.0, 3.0, 1.0])
    fitted = Poisson.fit(x, weights=w)
    mu = np.average(x, weights=w)
    assert fitted.parameters["mean"] == pytest.approx(mu)
    assert fitted.log_likelihood == pytest.approx(np.dot(w, stats.poisson.logpmf(x, mu)))
    assert fitted.converged
    d = 1.5
    expected = mu * stats.poisson.sf(0, mu) / stats.poisson.sf(1, mu) - d
    assert fitted.distribution.mean_excess(d) == pytest.approx(expected)
    assert np.isfinite(Poisson.fit([0, 0, 0]).log_likelihood)


def test_binomial_and_bernoulli_require_identifiable_experiment_sizes() -> None:
    counts = np.array([1, 2, 4, 2, 3, 5, 0])
    fitted = Binomial.fit(counts, n=7)
    p = np.mean(counts) / 7
    assert fitted.parameters == {"n": 7, "p": pytest.approx(p)}
    assert fitted.log_likelihood == pytest.approx(np.sum(stats.binom.logpmf(counts, 7, p)))
    assert fitted.distribution.mean_excess(2.0) == pytest.approx(
        7 * p * stats.binom.sf(1, 6, p) / stats.binom.sf(2, 7, p) - 2.0
    )
    fitted_bernoulli = Bernoulli.fit([0, 1, 1, 0, 1])
    assert fitted_bernoulli.parameters["p"] == pytest.approx(0.6)
    assert fitted_bernoulli.distribution.mean_excess(0.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="trial count"):
        Binomial.fit(counts)
    with pytest.raises(ValueError, match="exceed"):
        Binomial.fit(counts, n=4)
    with pytest.raises(ValueError, match="nonnegative integers"):
        Poisson.fit([0, 2.5])


def test_negative_binomial_profile_mle() -> None:
    rng = np.random.default_rng(98)
    sample = stats.nbinom.rvs(3.0, 0.45, size=650, random_state=rng)
    fitted = NegBinomial.fit(sample)
    assert fitted.parameters["n"] == pytest.approx(3.0, abs=1.0)
    assert fitted.parameters["p"] == pytest.approx(0.45, abs=0.07)
    n, p = fitted.parameters["n"], fitted.parameters["p"]
    assert fitted.log_likelihood == pytest.approx(np.sum(stats.nbinom.logpmf(sample, n, p)))
    assert fitted.distribution.mean_excess(2.0) == pytest.approx(
        n * (1 - p) / p * stats.nbinom.sf(1, n + 1, p) / stats.nbinom.sf(2, n, p) - 2.0
    )
    with pytest.raises(ValueError, match="overdispersed"):
        NegBinomial.fit([1, 1, 2, 2, 3, 3])


def test_hypergeometric_integer_mle_matches_exhaustive_reference() -> None:
    rng = np.random.default_rng(117)
    sample = stats.hypergeom.rvs(80, 27, 12, size=250, random_state=rng)
    fitted = HyperGeometric.fit(sample, population_size=80, n_draws=12)
    reference = {
        k: float(np.sum(stats.hypergeom.logpmf(sample, 80, k, 12)))
        for k in range(int(sample.max()), 80 - 12 + int(sample.min()) + 1)
    }
    best = max(reference, key=reference.get)
    assert fitted.parameters["ngood"] == best
    assert fitted.parameters["nbad"] == 80 - best
    assert fitted.log_likelihood == pytest.approx(reference[best])
    model = fitted.distribution
    d = 2.5
    actual = sum((k - d) * stats.hypergeom.pmf(k, 80, best, 12) for k in range(3, 13))
    assert model.mean_excess(d) == pytest.approx(actual / stats.hypergeom.sf(2, 80, best, 12))
    with pytest.raises(ValueError, match="population_size"):
        HyperGeometric.fit(sample)


def test_beta_fits_fixed_support_and_censored_truncation() -> None:
    rng = np.random.default_rng(345)
    raw = stats.beta.rvs(2.5, 4.0, size=500, random_state=rng)
    result = Beta.fit(raw)
    assert result.parameters["alpha"] == pytest.approx(2.5, abs=0.7)
    assert result.parameters["beta"] == pytest.approx(4.0, abs=1.1)
    truncated = raw[raw > 0.12]
    censor = truncated > 0.75
    observed = np.minimum(truncated, 0.75)
    fit = Beta.fit(observed, threshold=0.12, censored=censor)
    alpha, beta = fit.parameters["alpha"], fit.parameters["beta"]
    expected = (
        np.sum(stats.beta.logpdf(observed[~censor], alpha, beta))
        + np.sum(stats.beta.logsf(observed[censor], alpha, beta))
        - len(observed) * stats.beta.logsf(0.12, alpha, beta)
    )
    assert fit.log_likelihood == pytest.approx(expected)
    d = 0.4
    area, _ = integrate.quad(stats.beta(2.5, 4.0).sf, d, 1.0)
    assert Beta(alpha=2.5, beta=4.0).mean_excess(d) == pytest.approx(area / stats.beta.sf(d, 2.5, 4.0))
    with pytest.raises(ValueError, match="open support"):
        Beta.fit([0.0, 0.4, 0.8])


def test_normal_analytic_mle_and_tail_mean_excess() -> None:
    x = np.array([-4.0, -1.0, 0.0, 2.0, 6.0])
    w = np.array([2.0, 1.0, 1.0, 1.0, 2.0])
    fit = Normal.fit(x, weights=w)
    mu, sigma = fit.parameters["mu"], fit.parameters["sigma"]
    assert fit.method == "analytic_mle"
    assert mu == pytest.approx(np.average(x, weights=w))
    assert sigma == pytest.approx(np.sqrt(np.average((x - mu) ** 2, weights=w)))
    assert fit.log_likelihood == pytest.approx(np.dot(w, stats.norm.logpdf(x, mu, sigma)))
    d = 1.5
    z = (d - mu) / sigma
    assert fit.distribution.mean_excess(d) == pytest.approx(mu - d + sigma * stats.norm.pdf(z) / stats.norm.sf(z))
    with pytest.raises(ValueError, match="constant"):
        Normal.fit([2.0, 2.0, 2.0])


def test_logistic_mle_and_conditional_mean() -> None:
    rng = np.random.default_rng(992)
    sample = stats.logistic.rvs(loc=-1.2, scale=2.2, size=400, random_state=rng)
    result = Logistic.fit(sample)
    mu, sigma = result.parameters["mu"], result.parameters["sigma"]
    assert mu == pytest.approx(-1.2, abs=0.4)
    assert sigma == pytest.approx(2.2, abs=0.4)
    assert result.log_likelihood == pytest.approx(np.sum(stats.logistic.logpdf(sample, mu, sigma)))
    d = 0.3
    area, _ = integrate.quad(stats.logistic(loc=mu, scale=sigma).sf, d, np.inf)
    assert result.distribution.mean_excess(d) == pytest.approx(area / stats.logistic.sf(d, mu, sigma))


def test_students_t_mle_and_infinite_mean_boundary() -> None:
    rng = np.random.default_rng(121)
    sample = stats.t.rvs(df=5, loc=-0.5, scale=1.5, size=330, random_state=rng)
    fit = StudentsT.fit(sample)
    nu, mu, sigma = (fit.parameters[k] for k in ("nu", "mu", "sigma"))
    assert nu > 1
    assert mu == pytest.approx(-0.5, abs=0.35)
    assert sigma == pytest.approx(1.5, abs=0.5)
    assert fit.log_likelihood == pytest.approx(np.sum(stats.t.logpdf(sample, nu, mu, sigma)))
    model = StudentsT(nu=3.5, mu=1.0, sigma=2.0)
    d = 3.0
    rv = stats.t(df=3.5, loc=1, scale=2)
    integral, _ = integrate.quad(rv.sf, d, np.inf)
    assert model.mean_excess(d) == pytest.approx(integral / rv.sf(d), rel=1e-6)
    assert np.isinf(StudentsT(nu=1.0).mean_excess(5.0))


def test_noncentral_chi_squared_mle_and_excess() -> None:
    rng = np.random.default_rng(1211)
    sample = stats.ncx2.rvs(4.0, 3.0, size=160, random_state=rng)
    fitted = NonCentralChiSquared.fit(sample)
    df, nc = fitted.parameters["df"], fitted.parameters["nonc"]
    assert df > 0 and nc >= 0
    assert np.isfinite(fitted.log_likelihood)
    assert fitted.log_likelihood == pytest.approx(np.sum(stats.ncx2.logpdf(sample, df, nc)))
    reference = stats.ncx2(4.0, 3.0)
    area, _ = integrate.quad(reference.sf, 4.0, np.inf)
    assert NonCentralChiSquared(df=4.0, nonc=3.0).mean_excess(4.0) == pytest.approx(area / reference.sf(4.0), rel=1e-6)


def test_uniform_numeric_and_calendar_dates() -> None:
    fit = Uniform.fit([-4.0, -2.5, 3.5, 1.2])
    assert fit.parameters == {"a": -4.0, "b": 3.5}
    assert fit.distribution.mean_excess(1.0) == pytest.approx(1.25)
    assert fit.distribution.mean_excess(-5.0) == pytest.approx(4.75)
    assert np.isnan(fit.distribution.mean_excess(3.5))
    dates = [dt.date(2026, 1, 2), dt.date(2026, 1, 4), dt.date(2026, 1, 7)]
    temporal = Uniform.fit(dates)
    assert temporal.distribution._date_unit == "D"
    assert temporal.distribution.invcdf(0.0) == np.datetime64("2026-01-02")
    assert temporal.distribution.invcdf(1.0) == np.datetime64("2026-01-07")
    assert temporal.log_likelihood == pytest.approx(-3 * np.log(6))
    with pytest.raises(TypeError, match="date"):
        temporal.distribution.mean_excess(0.5)


def test_categorical_mle_preserves_labels_and_unobserved_categories() -> None:
    data = ["low", "high", "low", "mid", "low"]
    weights = [2.0, 1.0, 3.0, 1.0, 4.0]
    result = Categorical.fit(data, values=["low", "mid", "high", "other"], weights=weights)
    assert result.parameters["probabilities"] == pytest.approx([9 / 11, 1 / 11, 1 / 11, 0])
    assert result.distribution.cdf("low") == pytest.approx(9 / 11)
    assert result.log_likelihood == pytest.approx(9 * np.log(9 / 11) + 2 * np.log(1 / 11))
    with pytest.raises(ValueError, match="supplied labels"):
        Categorical.fit(data, values=["low", "mid"])
    date = Categorical.fit([dt.date(2026, 1, 1), dt.date(2026, 1, 2)])
    assert len(date.parameters["probabilities"]) == 2


def test_mbbefd_mixed_atom_likelihood_and_mean_excess() -> None:
    rng = np.random.default_rng(86)
    baseline = MBBEFD(g=5.0, b=0.5)
    samples = asnumpy(baseline.invcdf(rng.random(350)))
    fitted = MBBEFD.fit(samples)
    assert fitted.converged
    assert np.isfinite(fitted.log_likelihood)
    assert fitted.parameters["g"] > 1
    assert fitted.parameters["b"] > 0
    d = 0.4
    area, _ = integrate.quad(lambda v: 1.0 - baseline.cdf(v), d, 1.0)
    assert baseline.mean_excess(d) == pytest.approx(area / (1.0 - baseline.cdf(d)), rel=1e-6)
    assert np.isnan(baseline.mean_excess(1.0))
    with pytest.raises(ValueError, match="interior and total-loss"):
        MBBEFD.fit([0.2, 0.4, 0.6])


def test_invalid_weight_and_label_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="weights"):
        Poisson.fit([1, 2, 3], weights=[1, -1, 1])
    with pytest.raises(ValueError, match="nonempty"):
        Categorical.fit([])
    with pytest.raises(TypeError, match="censored"):
        Poisson.fit([1, 2], censored=[False, False])


def test_remaining_dispatch_requires_actual_pal_class() -> None:
    """Coincidentally matching class names must not select a fitter."""
    fake_normal = type("Normal", (), {})
    with pytest.raises(ValueError, match="Unsupported PAL fitting family"):
        fit_remaining(fake_normal, [1.0, 2.0, 4.0])
