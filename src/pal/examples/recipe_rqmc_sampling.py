"""Compare Monte Carlo and randomized quasi-Monte Carlo draws.

RQMC uses scrambled Sobol sampling. A power-of-two draw count is
convenient; use independent scrambles to assess sampling uncertainty.
"""

from pal import set_random_seed
from pal.config import SamplingMethod, set_sampling_method
from pal.distributions import Normal


def main() -> None:
    """Switch sampling method without changing the distribution API."""
    distribution = Normal(mu=0, sigma=1)
    try:
        set_sampling_method(SamplingMethod.MC)
        set_random_seed(42)
        mc_samples = distribution.generate(n_sims=512)

        set_sampling_method(SamplingMethod.RQMC)
        set_random_seed(42)
        rqmc_samples = distribution.generate(n_sims=512)
    finally:
        set_sampling_method(SamplingMethod.MC)

    assert mc_samples.n_sims == rqmc_samples.n_sims == 512
    print("MC sample mean:", mc_samples.mean())
    print("RQMC sample mean:", rqmc_samples.mean())


if __name__ == "__main__":
    main()
