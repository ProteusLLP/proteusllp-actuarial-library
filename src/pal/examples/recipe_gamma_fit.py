"""Fit an uncensored claim-severity Gamma model by maximum likelihood.

Gamma(alpha=shape, theta=scale). Fitted parameters are treated as fixed,
so downstream probabilities do not include parameter uncertainty.
"""

import numpy as np

from pal.distributions import Gamma


def main() -> None:
    """Fit observed claim sizes and inspect a fitted tail probability."""
    rng = np.random.default_rng(42)
    claim_sizes = rng.gamma(shape=2.8, scale=35_000.0, size=500)

    fit = Gamma.fit(claim_sizes)
    fitted_severity = fit.distribution

    assert fit.converged
    assert fit.parameters["alpha"] > 0
    assert fit.parameters["theta"] > 0
    assert 0 < fitted_severity.cdf(100_000) < 1

    print("Gamma MLE:", fit.parameters)
    print("AIC:", fit.aic)
    print("Probability of loss above 100,000:", 1 - fitted_severity.cdf(100_000))


if __name__ == "__main__":
    main()
