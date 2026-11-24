"""Stable severity probability kernels shared by fitting and simulation APIs."""

from __future__ import annotations

import typing as t

import numpy as np


def _log_softplus(x: t.Any, module: t.Any) -> t.Any:
    """Return the log of softplus without losing its exponentially small tail."""
    return module.where(x < -36, x, module.log(module.logaddexp(0.0, x)))


def _log_survival_from_log_hazard(log_hazard: t.Any, module: t.Any) -> t.Any:
    """Evaluate log survival given the logarithm of minus log CDF."""
    return module.where(
        log_hazard < -36,
        log_hazard,
        module.log(-module.expm1(-module.exp(log_hazard))),
    )


def _burr_terms(
    x: t.Any,
    power: t.Any,
    shape: t.Any,
    scale: t.Any,
    loc: t.Any,
    *,
    inverse: bool = False,
    module: t.Any = np,
) -> tuple[t.Any, t.Any, t.Any]:
    """Return log density, log survival and log CDF for Burr XII or III."""
    x, power, shape, scale, loc = (module.asarray(v, dtype=float) for v in (x, power, shape, scale, loc))
    y = x - loc
    with np.errstate(all="ignore"):
        log_y = module.log(y)
        log_ratio = log_y - module.log(scale)
        z = power * log_ratio
        if inverse:
            log_hazard = module.log(shape) + _log_softplus(-z, module)
            log_cdf = -module.exp(log_hazard)
            log_sf = _log_survival_from_log_hazard(log_hazard, module)
            log_pdf = module.log(power) + module.log(shape) - log_y + log_cdf - module.logaddexp(0, z)
        else:
            log_hazard = module.log(shape) + _log_softplus(z, module)
            log_sf = -module.exp(log_hazard)
            log_cdf = _log_survival_from_log_hazard(log_hazard, module)
            log_pdf = module.log(power) + module.log(shape) - log_y + log_sf - module.logaddexp(0, -z)
        # The right-hand density limit at the lower support endpoint.
        exponent = power * shape if inverse else power
        endpoint = module.where(
            exponent < 1,
            module.inf,
            module.where(exponent == 1, module.log(power) + module.log(shape) - module.log(scale), -module.inf),
        )
        log_pdf = module.where(y == 0, endpoint, module.where(y < 0, -module.inf, log_pdf))
        log_pdf = module.where(module.isposinf(y), -module.inf, log_pdf)
        log_sf = module.where(y <= 0, 0.0, log_sf)
        log_cdf = module.where(y <= 0, -module.inf, log_cdf)
        valid = (power > 0) & (shape > 0) & (scale > 0)
        return (
            module.where(valid, log_pdf, module.nan),
            module.where(valid, log_sf, module.nan),
            module.where(valid, log_cdf, module.nan),
        )


def _inverse_weibull_terms(
    x: t.Any,
    shape: t.Any,
    scale: t.Any,
    loc: t.Any,
    *,
    module: t.Any = np,
) -> tuple[t.Any, t.Any, t.Any]:
    """Return log density, log survival and log CDF for inverse Weibull."""
    x, shape, scale, loc = (module.asarray(v, dtype=float) for v in (x, shape, scale, loc))
    y = x - loc
    with np.errstate(all="ignore"):
        log_y = module.log(y)
        log_hazard = shape * (module.log(scale) - log_y)
        log_cdf = -module.exp(log_hazard)
        log_sf = _log_survival_from_log_hazard(log_hazard, module)
        log_pdf = module.log(shape) - log_y + log_hazard + log_cdf
        log_pdf = module.where((y <= 0) | module.isposinf(y) | module.isneginf(log_cdf), -module.inf, log_pdf)
        log_sf = module.where(y <= 0, 0.0, log_sf)
        log_cdf = module.where(y <= 0, -module.inf, log_cdf)
        valid = (shape > 0) & (scale > 0)
        return (
            module.where(valid, log_pdf, module.nan),
            module.where(valid, log_sf, module.nan),
            module.where(valid, log_cdf, module.nan),
        )


class _StableSeverityReference:
    """Delegate non-probability operations to SciPy, not its unstable tails."""

    def __init__(self, frozen: t.Any, terms: t.Callable[[t.Any], tuple[t.Any, t.Any, t.Any]]) -> None:
        self._frozen = frozen
        self._terms = terms

    def __getattr__(self, name: str) -> t.Any:
        return getattr(self._frozen, name)

    def logpdf(self, x: t.Any) -> t.Any:
        """Return the stable log density."""
        return self._terms(x)[0]

    def logsf(self, x: t.Any) -> t.Any:
        """Return the stable log survival."""
        return self._terms(x)[1]

    def logcdf(self, x: t.Any) -> t.Any:
        """Return the stable log CDF."""
        return self._terms(x)[2]

    def pdf(self, x: t.Any) -> t.Any:
        """Return the density."""
        return np.exp(self.logpdf(x))

    def sf(self, x: t.Any) -> t.Any:
        """Return the survival probability."""
        return np.exp(self.logsf(x))

    def cdf(self, x: t.Any) -> t.Any:
        """Return the cumulative probability."""
        return np.exp(self.logcdf(x))
