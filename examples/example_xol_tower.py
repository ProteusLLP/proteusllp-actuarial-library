"""Example of generating losses through an XoL Tower."""

from pal import XoLTower, config, distributions, np
from pal.frequency_severity import FreqSevSims, FrequencySeverityModel

config.n_sims = 100_000

sev_dist = distributions.GPD(shape=0.33, scale=100_000, loc=1_000_000)
freq_dist = distributions.Poisson(mean=2)

losses_pre_cap = FrequencySeverityModel(freq_dist, sev_dist).generate()
policy_limit = 5_000_000
# you can apply standard numpy ufuncs to the losses
losses_post_cap: FreqSevSims = np.minimum(losses_pre_cap, policy_limit)  # type: ignore[misc]

# you can apply standard numerical operations to the losses
losses_with_LAE = losses_post_cap * 1.05
stochastic_inflation = distributions.Normal(0.05, 0.02).generate()

# you can multiply frequency severity losses with other standard simulations
gross_losses = losses_with_LAE * (1 + stochastic_inflation)

prog = XoLTower(
    limit=[1_000_000, 1_000_000, 1_000_000, 1_000_000, 10_000_000],
    excess=[1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000],
    aggregate_limit=[3_000_000, 2_000_000, 1_000_000, 1_000_000, 10_000_000],
    premium=[5_000, 4_000, 3_000, 2_000, 1_000],
    reinstatement_cost=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
)

prog_results = prog.apply(gross_losses)

prog.print_summary()

prog_results.recoveries.aggregate().show_cdf("Recoveries")
