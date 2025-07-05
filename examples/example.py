import numpy as np
from pal import XoLTower, config, distributions
from pal.frequency_severity import FrequencySeverityModel

config.n_sims = 100000

sev_dist = distributions.GPD(shape=0.33, scale=100000, loc=1000000)
freq_dist = distributions.Poisson(mean=2)

losses_pre_cap = FrequencySeverityModel(freq_dist, sev_dist).generate()
policy_limit = 5000000
# you can apply standard numpy ufuncs to the losses
losses_post_cap = np.minimum(losses_pre_cap, policy_limit)

# you can apply standard numerical operations to the losses
losses_with_lae = losses_post_cap * 1.05
stochastic_inflation = distributions.Normal(0.05, 0.02).generate()

# you can multiply frequency severity losses with other standard simulations
gross_losses = losses_with_lae * (1 + stochastic_inflation)

prog = XoLTower(
    limit=[1000000, 1000000, 1000000, 1000000, 10000000],
    excess=[1000000, 2000000, 3000000, 4000000, 5000000],
    aggregate_limit=[3000000, 2000000, 1000000, 1000000, 10000000],
    premium=[5000, 4000, 3000, 2000, 1000],
    reinstatement_cost=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
)

prog_results = prog.apply(gross_losses)

prog.print_summary()

prog_results.recoveries.aggregate().show_cdf("Recoveries")
