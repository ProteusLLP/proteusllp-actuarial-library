"""This example demonstrates capital allocation approaches in pal.

The example is taken from the paper 'Capital Allocation by Percentile Layer', by Neil Bodoff, Variance, 28,
 Casualty Acturaial Society Volume 3 issue 1
"""

from pal import ProteusVariable, risk_measures, set_default_n_sims, set_random_seed
from pal.distributions import Bernoulli, Exponential

set_random_seed(42)
set_default_n_sims(100_000)
losses = ProteusVariable(
    "lob",
    {
        "Fire": Bernoulli(0.25).generate() * Exponential(4e6).generate(),
        "Wind": Bernoulli(0.05).generate() * Exponential(20e6).generate(),
        "EQ": Bernoulli(0.01).generate() * Exponential(100e6).generate(),
    },
)

total_losses = losses.sum()
total_capital_var = total_losses.percentile(99.0)
print(f"Total capital required at 99% percentile: {total_capital_var}")
allocated_capital = risk_measures.percentile_layer(total_losses, total_capital_var).allocate(losses)
print("Allocated capital to each line of business:")
print(allocated_capital)
allocated_proportions = allocated_capital / total_capital_var
print("Proportion of total capital allocated to each line of business:")
print(allocated_proportions)
total_captial_tvar = total_losses.tvar(90)

print(f"Total capital required at 99% TVaR: {total_captial_tvar}")
allocated_capital_tvar = risk_measures.tvar(total_losses, 0.9).allocate(losses)
print("Allocated capital to each line of business using TVaR:")
print(allocated_capital_tvar)
allocated_proportions_tvar = allocated_capital_tvar / total_captial_tvar
print("Proportion of total capital allocated to each line of business using TVaR:")
print(allocated_proportions_tvar)
