"An example underwriting risk model with multiple lines of business and reinsurance."

import time

from pal import config, copulas, distributions
from pal.contracts import XoLTower
from pal.frequency_severity import FrequencySeverityModel
from pal.stochastic_scalar import StochasticScalar
from pal.variables import ProteusVariable

config.n_sims = 100000


start = time.time()
lobs = [f"lob{i}" for i in range(100)]
individual_large_losses_by_lob = ProteusVariable(
    dim_name="class",
    values={
        name: FrequencySeverityModel(
            distributions.Poisson(mean=5),
            distributions.GPD(shape=0.33, scale=100000, loc=1000000),
        ).generate()
        for name in lobs
    },
)
# Generate the attritional losses by class
attritional_losses_by_lob = ProteusVariable(
    "class",
    values={
        lob: distributions.Gamma(alpha=i + 1, theta=1000000).generate()
        for i, lob in enumerate(lobs)
    },
)

losses_with_lae = individual_large_losses_by_lob * 1.05

# create the aggregate losses by class
aggregate_large_losses_by_class = ProteusVariable(
    "class", {name: losses_with_lae[name].aggregate() for name in lobs}
)
# correlate the attritional and large losses. Use a pairwise copula to do this
for lob in lobs:
    copulas.GumbelCopula(theta=1.2, n=2).apply(
        [aggregate_large_losses_by_class[lob], attritional_losses_by_lob[lob]]
    )
# calculate the total losses
total_losses_by_lob = aggregate_large_losses_by_class + attritional_losses_by_lob
# apply a copula to the total losses by lob
copulas.GumbelCopula(1.5, len(lobs)).apply(total_losses_by_lob)
# apply stochastic inflation
stochastic_inflation = distributions.Normal(0.05, 0.02).generate()
inflated_total_losses_by_lob = total_losses_by_lob * (1 + stochastic_inflation)
inflated_large_losses = individual_large_losses_by_lob * (1 + stochastic_inflation)

# reinsurance
net_aggregate_large_losses_dict: dict[str, StochasticScalar] = {}
for lob in lobs:
    prog = XoLTower(
        limit=[1000000, 1000000, 1000000, 1000000, 10000000],
        excess=[1000000, 2000000, 3000000, 4000000, 5000000],
        premium=[100000, 50000, 30000, 20000, 10000],
        aggregate_limit=[3000000, 2000000, 1000000, 1000000, 10000000],
    )
    result = prog.apply(inflated_large_losses[lob])
    aggregate_recoveries = result.recoveries.aggregate()
    net_aggregate_large_losses_dict[lob] = (
        aggregate_large_losses_by_class[lob] - aggregate_recoveries
    )


net_aggregate_large_losses = ProteusVariable("class", net_aggregate_large_losses_dict)

total_net_losses_by_lob = net_aggregate_large_losses + attritional_losses_by_lob


# create the total losses
total_net_losses = total_net_losses_by_lob.sum()

print("TVAR: ", total_net_losses.tvar(99))
end = time.time()

print("Elapsed time: ", end - start, " seconds")
