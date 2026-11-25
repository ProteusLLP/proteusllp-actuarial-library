"""Fit an uncensored claim-severity Gamma model by maximum likelihood.

Gamma(alpha=shape, theta=scale). Fitted parameters are treated as fixed,
so downstream probabilities do not include parameter uncertainty.
"""

from pal import set_random_seed
from pal.distributions import Gamma
from pal.maths import asnumpy


def main() -> None:
    """Fit observed claim sizes and inspect a fitted tail probability."""
    set_random_seed(42)
    claim_sizes = asnumpy(Gamma(alpha=2.8, theta=35_000.0).generate(n_sims=500).values)

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
