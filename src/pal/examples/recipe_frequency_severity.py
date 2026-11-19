"""Simulate claim-level losses and annual aggregate/occurrence results.

FreqSevSims is indexed by individual claim, while aggregate() and
occurrence() return one value per simulation.
"""

from pal import FrequencySeverityModel, set_random_seed
from pal.distributions import LogNormal, Poisson


def main() -> None:
    """Generate compound losses and summarize aggregate and occurrence risk."""
    set_random_seed(42)
    events = FrequencySeverityModel(
        freq_dist=Poisson(mean=3),
        sev_dist=LogNormal(mu=12, sigma=0.9),
    ).generate(n_sims=512)

    annual_losses = events.aggregate()
    largest_claim = events.occurrence()

    assert annual_losses.n_sims == 512
    assert largest_claim.n_sims == 512
    assert annual_losses.coupled_variable_group is events.coupled_variable_group
    assert largest_claim.coupled_variable_group is events.coupled_variable_group

    print("Annual aggregate expected loss:", annual_losses.mean())
    print("Annual aggregate 99.5th percentile:", annual_losses.percentile(99.5))
    print("Largest-claim expected loss:", largest_claim.mean())


if __name__ == "__main__":
    main()
