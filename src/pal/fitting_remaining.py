"""Maximum-likelihood estimation for non-severity PAL distribution families.

Fitting operates on CPU observation arrays and never consumes PAL random state.
Specialized estimators retain the parameter identifiability requirements of
the underlying probability families.
"""

from __future__ import annotations

import datetime as dt
import typing as t

import numpy as np
from scipy import integrate, optimize, special, stats

from .fitting import UncertaintyMethod, _FitMethod, _with_uncertainty

if t.TYPE_CHECKING:
    from .distributions import DistributionBase
    from .fitting import FitResult

_T = t.TypeVar("_T", bound="DistributionBase")


def _continuous_families() -> frozenset[type[DistributionBase]]:
    """Resolve PAL distribution classes when fitting is first invoked."""
    from . import distributions as _distributions

    return frozenset(
        {
            _distributions.Beta,
            _distributions.Normal,
            _distributions.Logistic,
            _distributions.NonCentralChiSquared,
            _distributions.StudentsT,
        }
    )


def _discrete_families() -> frozenset[type[DistributionBase]]:
    """Resolve PAL distribution classes when fitting is first invoked."""
    from . import distributions as _distributions

    return frozenset(
        {
            _distributions.Poisson,
            _distributions.NegBinomial,
            _distributions.Binomial,
            _distributions.HyperGeometric,
            _distributions.Bernoulli,
        }
    )


def _input(
    observations: t.Sequence[t.Any] | np.ndarray,
    weights: t.Sequence[float] | np.ndarray | None,
    *,
    dtype: t.Any = float,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned one-dimensional samples and non-negative weights."""
    values = np.asarray(observations, dtype=dtype)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("observations must be a nonempty one-dimensional sequence")
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != values.shape or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite, non-negative and match observations")
    active = w > 0
    if not np.any(active):
        raise ValueError("at least one positive observation weight is required")
    return values[active], w[active]


def _result(
    distribution: _T,
    params: dict[str, t.Any],
    ll: float,
    w: np.ndarray,
    *,
    k: int,
    n_censored: int = 0,
    threshold: float = 0.0,
    method: str = _FitMethod.MLE.value,
    converged: bool = True,
) -> FitResult[_T]:
    """Construct consistent log-likelihood and information criterion diagnostics."""
    from .fitting import FitResult

    aic = 2.0 * k - 2.0 * ll
    n = float(np.sum(w))
    if np.all(np.floor(w) == w) and n > k + 1:
        aicc = aic + 2.0 * k * (k + 1) / (n - k - 1)
    else:
        aicc = np.nan
    return FitResult(
        distribution=distribution,
        parameters=params,
        log_likelihood=float(ll),
        aic=float(aic),
        aicc=float(aicc),
        converged=converged,
        n_observations=len(w),
        n_censored=n_censored,
        threshold=threshold,
        method=method,
    )


def fit_remaining(
    distribution: type[_T],
    observations: t.Sequence[t.Any] | np.ndarray,
    *,
    weights: t.Sequence[float] | np.ndarray | None = None,
    censored: t.Sequence[bool] | np.ndarray | None = None,
    threshold: float | None = None,
    uncertainty_method: UncertaintyMethod = UncertaintyMethod.NONE,
    **options: t.Any,
) -> FitResult[_T]:
    """Fit a PAL distribution not represented by the severity fitting engine.

    Normal, logistic, Student's t, beta and noncentral chi-squared permit
    right censoring and left truncation. Other families use complete samples.
    Binomial and hypergeometric require known experiment sizes.
    """
    if UncertaintyMethod(uncertainty_method) is not UncertaintyMethod.NONE:
        fitted = fit_remaining(
            distribution,
            observations,
            weights=weights,
            censored=censored,
            threshold=threshold,
            **options,
        )
        return _with_uncertainty(
            fitted,
            observations,
            uncertainty_method=uncertainty_method,
            censored=censored,
            weights=weights,
            threshold_specified=threshold is not None,
        )

    from . import distributions as _distributions

    family = distribution
    if family in _continuous_families():
        return _fit_continuous(distribution, observations, weights, censored, threshold, options)
    if censored is not None or threshold is not None:
        raise ValueError(f"{family.__name__} fitting does not support truncation or censoring")
    if family in _discrete_families():
        return _fit_discrete(distribution, observations, weights, options)
    if family is _distributions.MBBEFD:
        return _fit_mbbefd(distribution, observations, weights, options)
    if family is _distributions.Uniform:
        return _fit_uniform(distribution, observations, weights, options)
    if family is _distributions.Categorical:
        return _fit_categorical(distribution, observations, weights, options)
    raise ValueError(f"Unsupported PAL fitting family: {family.__name__}")


def _fit_continuous(
    cls: type[_T],
    observations: t.Sequence[t.Any] | np.ndarray,
    weights: t.Sequence[float] | np.ndarray | None,
    censored: t.Sequence[bool] | np.ndarray | None,
    threshold: float | None,
    options: dict[str, t.Any],
) -> FitResult[_T]:
    """Estimate a continuous family's parameters with a full likelihood."""
    from . import distributions as _distributions

    family = cls
    x, w = _input(observations, weights)
    if not np.all(np.isfinite(x)):
        raise ValueError("continuous observations must be finite")
    c = np.zeros(len(x), dtype=bool) if censored is None else np.asarray(censored, dtype=bool)
    if censored is not None:
        # Remove zero-weighted rows from censoring indicators as well.
        raw_weights = np.ones(len(observations)) if weights is None else np.asarray(weights, dtype=float)
        if c.shape != raw_weights.shape:
            raise ValueError("censored indicators must match observations")
        c = c[raw_weights > 0]
    if np.all(c):
        raise ValueError("at least one uncensored observation is needed")
    if threshold is not None:
        if not np.isfinite(threshold) or np.any(x < threshold) or np.any((x == threshold) & ~c):
            raise ValueError("observations must exceed the truncation threshold")

    loc = float(options.pop("loc", 0.0)) if family is _distributions.Beta else 0.0
    scale = float(options.pop("scale", 1.0)) if family is _distributions.Beta else 1.0
    if options:
        raise TypeError(f"Unsupported fitting options for {family.__name__}: {sorted(options)}")
    if family is _distributions.Beta:
        if not np.isfinite(loc) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Beta support requires finite loc and positive scale")
        y = (x - loc) / scale
        if np.any(y[~c] <= 0) or np.any(y[~c] >= 1):
            raise ValueError("uncensored Beta observations must lie within its open support")
        if np.any(y[c] < 0) or np.any(y[c] >= 1):
            raise ValueError("Beta censoring points must lie below the upper support")
    if family is _distributions.NonCentralChiSquared and np.any(x < 0):
        raise ValueError("noncentral chi-squared observations must be nonnegative")

    total = float(np.sum(w))
    mu = float(np.sum(w * x) / total)
    variance = float(np.sum(w * (x - mu) ** 2) / total)
    std = max(np.sqrt(variance), 1e-6)
    if variance <= 0:
        raise ValueError(f"{family.__name__} parameters cannot be fitted to a constant sample")

    if family is _distributions.Normal and not np.any(c) and threshold is None:
        estimates = {"mu": mu, "sigma": std}
        ll = float(np.sum(w * stats.norm.logpdf(x, loc=mu, scale=std)))
        return _result(cls(**estimates), estimates, ll, w, k=2, method=_FitMethod.ANALYTIC.value)

    if family is _distributions.Beta:
        y = (x - loc) / scale
        center = float(np.average(y, weights=w))
        v = float(np.average((y - center) ** 2, weights=w))
        common = max(center * (1.0 - center) / max(v, 1e-6) - 1.0, 0.25)
        starts = [[np.log(max(center * common, 0.2)), np.log(max((1 - center) * common, 0.2))], [0.0, 0.0]]
        k = 2

        def params(raw: np.ndarray) -> dict[str, float]:
            return {"alpha": float(np.exp(raw[0])), "beta": float(np.exp(raw[1])), "scale": scale, "loc": loc}

        def frozen(p: dict[str, float]) -> t.Any:
            return stats.beta(a=p["alpha"], b=p["beta"], loc=loc, scale=scale)

    elif family in (_distributions.Normal, _distributions.Logistic):
        starts = [[mu, np.log(std)], [float(np.median(x)), np.log(std)]]
        k = 2

        def params(raw: np.ndarray) -> dict[str, float]:
            return {"mu": float(raw[0]), "sigma": float(np.exp(raw[1]))}

        def frozen(p: dict[str, float]) -> t.Any:
            rv = stats.norm if family is _distributions.Normal else stats.logistic
            return rv(loc=p["mu"], scale=p["sigma"])

    elif family is _distributions.StudentsT:
        starts = [[np.log(v), mu, np.log(std)] for v in (2.0, 5.0, 12.0)]
        k = 3

        def params(raw: np.ndarray) -> dict[str, float]:
            return {"nu": float(np.exp(raw[0])), "mu": float(raw[1]), "sigma": float(np.exp(raw[2]))}

        def frozen(p: dict[str, float]) -> t.Any:
            return stats.t(df=p["nu"], loc=p["mu"], scale=p["sigma"])

    else:
        if family is not _distributions.NonCentralChiSquared:
            raise ValueError(f"Unsupported continuous fitting family: {family.__name__}")
        starts = [
            [np.log(max(mu / 2, 0.5)), np.sqrt(max(mu / 2, 0.1))],
            [np.log(max(mu, 0.5)), 0.0],
            [np.log(max(mu / 3, 0.5)), np.sqrt(max(2 * mu / 3, 0.1))],
        ]
        k = 2

        def params(raw: np.ndarray) -> dict[str, float]:
            return {"df": float(np.exp(raw[0])), "nonc": float(raw[1] ** 2)}

        def frozen(p: dict[str, float]) -> t.Any:
            return stats.ncx2(df=p["df"], nc=p["nonc"])

    def log_likelihood(p: dict[str, float]) -> float:
        rv = frozen(p)
        value = np.zeros(len(x), dtype=float)
        if np.any(~c):
            value[~c] = rv.logpdf(x[~c])
        if np.any(c):
            value[c] = rv.logsf(x[c])
        if threshold is not None:
            value -= float(rv.logsf(threshold))
        ll = float(np.dot(w, value))
        return ll if np.isfinite(ll) else -np.inf

    def objective(raw: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            ll = log_likelihood(params(raw))
        return -ll if np.isfinite(ll) else 1e100

    fits = [
        optimize.minimize(
            objective,
            np.asarray(start, dtype=float),
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8},
        )
        for start in starts
    ]
    best = min(fits, key=lambda v: float(v.fun))
    if best.fun >= 1e100:
        raise RuntimeError(f"No finite {family.__name__} likelihood could be obtained")
    estimated = params(best.x)
    return _result(
        cls(**estimated),
        estimated,
        log_likelihood(estimated),
        w,
        k=k,
        n_censored=int(np.count_nonzero(c)),
        threshold=0.0 if threshold is None else threshold,
        converged=bool(best.success),
    )


def _counts(
    observations: t.Sequence[t.Any] | np.ndarray,
    weights: t.Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Require nonnegative integer observations including possible zero counts."""
    x, w = _input(observations, weights)
    if not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x != np.floor(x)):
        raise ValueError("count observations must be nonnegative integers")
    return x.astype(np.int64), w


def _fit_discrete(
    cls: type[_T],
    observations: t.Sequence[t.Any] | np.ndarray,
    weights: t.Sequence[float] | np.ndarray | None,
    options: dict[str, t.Any],
) -> FitResult[_T]:
    """Fit counting distributions with required fixed experiment parameters."""
    from . import distributions as _distributions

    family = cls
    x, w = _counts(observations, weights)
    mu = float(np.average(x, weights=w))
    if family is _distributions.Poisson:
        pars: dict[str, t.Any] = {"mean": mu}
        rv = stats.poisson(mu)
        k = 1
        method = _FitMethod.ANALYTIC.value
    elif family in (_distributions.Binomial, _distributions.Bernoulli):
        n = 1 if family is _distributions.Bernoulli else options.pop("n", None)
        if n is None or int(n) != n or n < 1:
            raise ValueError("Binomial fitting requires a positive integer trial count n")
        if np.any(x > n):
            raise ValueError("observed counts exceed the binomial trial count")
        p = mu / n
        pars = {"p": p} if family is _distributions.Bernoulli else {"n": int(n), "p": p}
        rv = stats.binom(n, p)
        k = 1
        method = _FitMethod.ANALYTIC.value
    elif family is _distributions.NegBinomial:
        if mu == 0:
            raise ValueError("negative binomial shape is unidentified for all-zero counts")
        variance = float(np.average((x - mu) ** 2, weights=w))
        if variance <= mu:
            raise ValueError("negative binomial finite-size MLE requires overdispersed counts")

        def profile(log_n: float) -> float:
            n_value = float(np.exp(log_n))
            p_value = n_value / (n_value + mu)
            return -float(np.dot(w, stats.nbinom.logpmf(x, n_value, p_value)))

        result = optimize.minimize_scalar(profile, bounds=(-12.0, 20.0), method="bounded")
        if not result.success:
            raise RuntimeError("negative binomial size optimisation failed")
        n_value = float(np.exp(result.x))
        pars = {"n": n_value, "p": n_value / (n_value + mu)}
        rv = stats.nbinom(pars["n"], pars["p"])
        k = 2
        method = _FitMethod.PROFILE.value
    else:
        population = options.pop("population_size", None)
        draws = options.pop("n_draws", None)
        if (
            population is None
            or draws is None
            or int(population) != population
            or int(draws) != draws
            or population < 1
            or draws < 1
            or draws > population
        ):
            raise ValueError("HyperGeometric.fit requires integer population_size and n_draws")
        population, draws = int(population), int(draws)
        if np.any(x > draws):
            raise ValueError("observed counts exceed n_draws")
        low = int(np.max(x))
        high = int(population - draws + np.min(x))
        if low > high:
            raise ValueError("No feasible population success count for observations")

        def negll(successes: int) -> float:
            return -float(np.dot(w, stats.hypergeom.logpmf(x, population, successes, draws)))

        # For fixed population and sample size the log-likelihood is
        # concave in the integer number of good items: it is a sum of
        # log binomial coefficients. Search its discrete forward difference
        # for the exact global MLE in O(log(population_size)) evaluations.
        left, right = low, high
        while left < right:
            mid = (left + right) // 2
            if negll(mid + 1) < negll(mid):
                left = mid + 1
            else:
                right = mid
        good = left
        pars = {"ngood": good, "nbad": population - good, "n_draws": draws}
        rv = stats.hypergeom(population, good, draws)
        k = 1
        method = _FitMethod.INTEGER.value

    if options:
        raise TypeError(f"Unsupported {family.__name__} fitting options: {sorted(options)}")
    ll = float(np.dot(w, rv.logpmf(x)))
    return _result(cls(**pars), pars, ll, w, k=k, method=method)


def _fit_mbbefd(
    cls: type[_T],
    observations: t.Sequence[t.Any] | np.ndarray,
    weights: t.Sequence[float] | np.ndarray | None,
    options: dict[str, t.Any],
) -> FitResult[_T]:
    """Fit MBBEFD including the point mass at the policy limit."""
    if options:
        raise TypeError(f"Unexpected MBBEFD fit options: {sorted(options)}")
    x, w = _input(observations, weights)
    if not np.all(np.isfinite(x)) or np.any(x <= 0) or np.any(x > 1):
        raise ValueError("MBBEFD observations must be in (0, 1]")
    atom = x == 1.0
    if not np.any(atom) or not np.any(~atom):
        raise ValueError("MBBEFD requires interior and total-loss observations for two-parameter fitting")
    interior = x[~atom]
    w_inner, w_atom = w[~atom], float(np.sum(w[atom]))

    def params(raw: np.ndarray) -> tuple[float, float]:
        return 1.0 + float(np.exp(raw[0])), float(np.exp(raw[1]))

    def likelihood(g: float, b: float) -> float:
        logb = float(np.log(b))
        if abs(logb) < 1e-7:
            density = (g - 1) / (1 + (g - 1) * interior) ** 2
        elif abs(g * b - 1) < 1e-7:
            density = -logb * np.exp(logb * interior)
        else:
            bx = np.exp(logb * interior)
            denominator = (g - 1) * b + (1 - g * b) * bx
            density = -(1 - b) * (g - 1) * logb * b * bx / (denominator**2)
        if np.any(density <= 0) or not np.all(np.isfinite(density)):
            return -np.inf
        return float(np.dot(w_inner, np.log(density)) - w_atom * np.log(g))

    def objective(raw: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            value = likelihood(*params(raw))
        return -value if np.isfinite(value) else 1e100

    atom_share = float(np.sum(w[atom]) / np.sum(w))
    initial_g = max(1.0 / atom_share, 1.01)
    starts = [[np.log(initial_g - 1), logb] for logb in (-2.0, -0.3, 0.3, 2.0)]
    fits = [
        optimize.minimize(
            objective,
            np.asarray(start, dtype=float),
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8},
        )
        for start in starts
    ]
    best = min(fits, key=lambda r: float(r.fun))
    if best.fun >= 1e100:
        raise RuntimeError("No finite MBBEFD likelihood found")
    g, b = params(best.x)
    pars = {"g": g, "b": b}
    return _result(cls(**pars), pars, likelihood(g, b), w, k=2, converged=bool(best.success))


def _fit_uniform(
    cls: type[_T],
    observations: t.Sequence[t.Any] | np.ndarray,
    weights: t.Sequence[float] | np.ndarray | None,
    options: dict[str, t.Any],
) -> FitResult[_T]:
    """Fit unknown endpoints for numeric or calendar/date uniform samples."""
    if options:
        raise TypeError(f"Unexpected Uniform fit options: {sorted(options)}")
    raw = np.asarray(observations)
    is_dates = raw.dtype.kind == "M" or (
        raw.dtype.kind == "O" and all(isinstance(v, (dt.date, np.datetime64)) for v in raw)
    )
    if is_dates:
        date_only = all(
            isinstance(v, dt.date)
            and not isinstance(v, dt.datetime)
            or isinstance(v, np.datetime64)
            and np.datetime_data(v.dtype)[0] in ("Y", "M", "W", "D")
            for v in raw
        )
        unit = "D" if date_only else "us"
        values, w = _input(observations, weights, dtype=f"datetime64[{unit}]")
        if np.any(np.isnat(values)):
            raise ValueError("date observations cannot contain NaT")
        a, b = np.min(values), np.max(values)
        days = int((b - a) / np.timedelta64(1, unit)) + (1 if unit == "D" else 0)
        if days <= 0:
            raise ValueError("datetime observations must span a positive interval")
        ll = -float(np.sum(w)) * np.log(days) if unit == "D" else -float(np.sum(w)) * np.log(days)
        pars: dict[str, t.Any] = {"a": a, "b": b}
    else:
        values, w = _input(observations, weights)
        if not np.all(np.isfinite(values)):
            raise ValueError("uniform observations must be finite")
        a, b = float(np.min(values)), float(np.max(values))
        if not a < b:
            raise ValueError("two distinct observations are required to estimate numeric Uniform endpoints")
        ll = -float(np.sum(w)) * np.log(b - a)
        pars = {"a": a, "b": b}
    return _result(cls(**pars), pars, ll, w, k=2, method=_FitMethod.ENDPOINT.value)


def _fit_categorical(
    cls: type[_T],
    observations: t.Sequence[t.Any] | np.ndarray,
    weights: t.Sequence[float] | np.ndarray | None,
    options: dict[str, t.Any],
) -> FitResult[_T]:
    """Estimate multinomial category probabilities and preserve label order."""
    observed, w = _input(observations, weights, dtype=object)
    given = options.pop("values", None)
    if options:
        raise TypeError(f"Unexpected Categorical fit options: {sorted(options)}")
    observed_labels = observed.tolist()
    labels = list(dict.fromkeys(observed_labels)) if given is None else list(given)
    if len(labels) == 0 or len(set(labels)) != len(labels):
        raise ValueError("category labels must be nonempty and unique")
    positions = {label: i for i, label in enumerate(labels)}
    if any(label not in positions for label in observed_labels):
        raise ValueError("observed category is not among the supplied labels")
    counts = np.zeros(len(labels), dtype=float)
    for label, weight in zip(observed_labels, w, strict=False):
        counts[positions[label]] += weight
    probabilities = counts / counts.sum()
    ll = float(np.dot(counts[counts > 0], np.log(probabilities[counts > 0])))
    params: dict[str, t.Any] = {
        "values": labels,
        "probabilities": probabilities.tolist(),
    }
    return _result(
        t.cast(_T, t.cast(t.Any, cls)(values=labels, probabilities=probabilities)),
        params,
        ll,
        w,
        k=len(labels) - 1,
        method=_FitMethod.MULTINOMIAL.value,
    )


def mean_excess_remaining(
    distribution: DistributionBase,
    threshold: float | np.ndarray,
) -> float | np.ndarray:
    """Compute conditional mean residual life for supported numeric families.

    Returns NaN when an attachment is at or above a finite upper endpoint.
    Numeric support is required; dates and categories do not have a loss
    magnitude interpretation.
    """
    from . import distributions as _distributions

    family = type(distribution)
    allowed = _continuous_families() | _discrete_families() | {_distributions.MBBEFD, _distributions.Uniform}
    if family not in allowed:
        raise ValueError(f"No numeric mean excess for {family.__name__}")
    from .stochastic_scalar import StochasticScalar

    pars: dict[str, float] = {}
    for name, value in distribution._params.items():
        if isinstance(value, StochasticScalar):
            raise ValueError("mean excess requires scalar parameters")
        if isinstance(value, (dt.date, np.datetime64)):
            raise TypeError("mean excess is not defined for date-valued distributions")
        pars[name] = float(value)
    ds = np.asarray(threshold, dtype=float)
    result = np.full(ds.shape, np.nan, dtype=float)

    valid = np.isfinite(ds)
    d = ds[valid]
    out = np.full(d.shape, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        if family is _distributions.MBBEFD:
            # Preserve the existing mixed-distribution endpoint semantics.
            model = t.cast(t.Any, distribution)
            expected = float(model.mean())
            for i, attachment in enumerate(d):
                if attachment < 0:
                    out[i] = expected - attachment
                elif attachment < 1:
                    survival = 1 - float(model.cdf(attachment))
                    if survival > 0:
                        out[i] = expected * (1 - float(model.exposure_curve(attachment))) / survival
        elif family is _distributions.Uniform:
            a, b = pars["a"], pars["b"]
            out = np.where(d < a, 0.5 * (a - d) + 0.5 * (b - d),
                           np.where(d < b, 0.5 * (b - d), np.nan))
        elif family is _distributions.Beta:
            a, b, scale, loc = pars["alpha"], pars["beta"], pars["scale"], pars["loc"]
            u = (d - loc) / scale
            w = (loc + scale - d) / scale
            pos = (u >= 0) & (w > 0)
            # Use the complementary variable Y=1-X. E[X-u|X>u]
            # equals w minus the conditional lower-tail mean of Y.
            sf = special.betainc(b, a, w[pos])
            ratio = b / (a + b) * special.betainc(b + 1, a, w[pos]) / sf / w[pos]
            excess = w[pos] * (1 - ratio)
            # B_w(b,a) = w**b/b * 2F1(b,1-a;b+1;w).
            # The ratio cancels the tiny w**b and normalizing factors.
            bad = (~np.isfinite(excess)) | (excess <= 1e-7 * w[pos])
            wb = w[pos][bad]
            h0 = special.hyp2f1(b, 1 - a, b + 1, wb)
            h1 = special.hyp2f1(b + 1, 1 - a, b + 2, wb)
            excess[bad] = wb * (1 - b / (b + 1) * h1 / h0)
            out[pos] = scale * excess
            out[u < 0] = loc - d[u < 0] + scale * a / (a + b)
        elif family is _distributions.Normal:
            mu, sigma = pars["mu"], pars["sigma"]
            z = (d - mu) / sigma
            low = z < 0
            out[low] = sigma * (np.exp(stats.norm.logpdf(z[low]) - special.log_ndtr(-z[low])) - z[low])
            central = (z >= 0) & (z < 30)
            out[central] = sigma * (np.sqrt(2 / np.pi) / special.erfcx(z[central] / np.sqrt(2)) - z[central])
            # Inverse Mills ratio minus z: asymptotic expansion in 1/z.
            far = z >= 30
            inv = 1 / z[far]
            q = inv**2
            out[far] = sigma * inv * (1 + q * (-2 + q * (10 + q * (-74 + q * (706 - 8162 * q)))))
        elif family is _distributions.Logistic:
            mu, sigma = pars["mu"], pars["sigma"]
            z = (d - mu) / sigma
            out = np.where(z > 35, sigma, sigma * np.logaddexp(0, -z) / special.expit(-z))
        elif family is _distributions.StudentsT:
            nu, mu, sigma = pars["nu"], pars["mu"], pars["sigma"]
            if nu <= 1:
                out.fill(np.inf)
            else:
                z = (d - mu) / sigma
                residual = np.exp(stats.t.logpdf(z, nu) - stats.t.logsf(z, nu))
                out = sigma * ((nu + z * z) / (nu - 1) * residual - z)
        elif family is _distributions.NonCentralChiSquared:
            df, nonc = pars["df"], pars["nonc"]
            rv = stats.ncx2(df, nonc)
            # Poisson mixture: x f_df,nc = df f_df+2,nc + nc f_df+4,nc.
            sf = rv.sf(d)
            moment = df * stats.ncx2.sf(d, df + 2, nonc) + nonc * stats.ncx2.sf(d, df + 4, nonc)
            out = moment / sf - d
            out[d < 0] = df + nonc - d[d < 0]
            bad = (d >= 0) & (sf > 0) & ((out <= 1e-7 * np.maximum(d, 1)) | ~np.isfinite(out))
            for i in np.flatnonzero(bad):
                area, _ = integrate.quad(rv.sf, d[i], np.inf, epsrel=1e-8)
                out[i] = area / sf[i]
        else:
            # Counts: truncated first moment divided by strict survival.
            k = np.floor(d)
            if family is _distributions.Poisson:
                lam = pars["mean"]
                sf = stats.poisson.sf(k, lam)
                moment = lam * stats.poisson.sf(k - 1, lam)
            elif family in (_distributions.Binomial, _distributions.Bernoulli):
                n = 1 if family is _distributions.Bernoulli else int(pars["n"])
                p = pars["p"]
                sf = stats.binom.sf(k, n, p)
                moment = n * p * stats.binom.sf(k - 1, n - 1, p) if n > 0 else np.zeros_like(d)
            elif family is _distributions.NegBinomial:
                n, p = pars["n"], pars["p"]
                sf = stats.nbinom.sf(k, n, p)
                moment = n * (1 - p) / p * stats.nbinom.sf(k - 1, n + 1, p)
            else:
                good, bad, draws = int(pars["ngood"]), int(pars["nbad"]), int(pars["n"])
                population = good + bad
                sf = stats.hypergeom.sf(k, population, good, draws)
                moment = (
                    draws * good / population * stats.hypergeom.sf(k - 1, population - 1, good - 1, draws - 1)
                    if good > 0 and draws > 0
                    else np.zeros_like(d)
                )
            out = np.where(sf > 0, moment / sf - d, np.nan)
    result[valid] = out
    return float(result) if result.ndim == 0 else result
