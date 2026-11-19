"""Apply one stochastic inflation rate per simulation to every claim.

PAL broadcasts a StochasticScalar along the FreqSevSims claim index
without treating claim-level values as independent inflation draws.
"""

from pal import FrequencySeverityModel, set_random_seed
from pal.distributions import LogNormal, Normal, Poisson


def main() -> None:
    """Inflate event losses by scenario and retain the simulation coupling."""
    set_random_seed(42)
    events = FrequencySeverityModel(
        freq_dist=Poisson(mean=3),
        sev_dist=LogNormal(mu=12, sigma=0.9),
    ).generate(n_sims=512)
    inflation = Normal(mu=0.05, sigma=0.02).generate(n_sims=512)

    inflated_events = events * (1 + inflation)
    annual_inflated_loss = inflated_events.aggregate()

    assert annual_inflated_loss.n_sims == inflation.n_sims
    assert events.coupled_variable_group is inflation.coupled_variable_group
    assert annual_inflated_loss.coupled_variable_group is inflation.coupled_variable_group

    print("Uninflated expected annual loss:", events.aggregate().mean())
    print("Inflated expected annual loss:", annual_inflated_loss.mean())


if __name__ == "__main__":
    main()
