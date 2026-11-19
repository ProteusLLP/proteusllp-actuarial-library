"""Fit a pre-limit loss-severity model to a thresholded, capped listing.

Left truncation omits losses below the reporting threshold. A policy
limit right-censors larger claims; capped payments are not fully observed
ground-up severities. MLE plug-in estimates exclude parameter uncertainty.
"""

import numpy as np

from pal.distributions import LogNormal


def main() -> None:
    """Fit underlying loss severity using reporting and policy-limit information."""
    rng = np.random.default_rng(103)
    ultimate_losses = rng.lognormal(mean=np.log(80_000), sigma=0.95, size=1_500)
    reporting_threshold = 75_000.0
    policy_limit = 300_000.0

    reported = ultimate_losses[ultimate_losses > reporting_threshold]
    censored = reported >= policy_limit
    observed = np.minimum(reported, policy_limit)

    fit = LogNormal.fit(observed, threshold=reporting_threshold, censored=censored)
    assert fit.converged
    assert fit.n_censored == int(censored.sum())
    assert 0 < fit.distribution.cdf(200_000) < 1

    print("Pre-limit LogNormal MLE:", fit.parameters)
    print("Censored observations:", fit.n_censored)
    print("Estimated P(ultimate loss > 200,000):", 1 - fit.distribution.cdf(200_000))


if __name__ == "__main__":
    main()
