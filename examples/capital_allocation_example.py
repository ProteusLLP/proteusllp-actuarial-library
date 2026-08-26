"""This example demonstrates capital allocation approaches in pal.

The example is taken from the paper 'Capital Allocation by Percentile Layer', by Neil Bodoff, Variance, 28,
 Casualty Acturaial Society Volume 3 issue 1
"""

from pal import distributions, risk_measures, set_default_n_sims, set_random_seed, variables

set_random_seed(42)
set_default_n_sims(100_000)
losses = variables.ProteusVariable(
    "lob",
    {
        "Fire": distributions.Bernoulli(0.25).generate() * distributions.Exponential(4e6).generate(),
        "Wind": distributions.Bernoulli(0.05).generate() * distributions.Exponential(20e6).generate(),
        "EQ": distributions.Bernoulli(0.01).generate() * distributions.Exponential(100e6).generate(),
    },
)

total_losses = losses.sum()
total_capital_var = risk_measures.var(total_losses, 99.0).value
print(f"Total capital required at 99% VaR: {total_capital_var}")
allocated_capital = risk_measures.percentile_layer(total_losses, total_capital_var).allocate(losses)
print("Allocated capital to each line of business:")
print(allocated_capital)
allocated_proportions = allocated_capital / total_capital_var
print("Proportion of total capital allocated to each line of business:")
print(allocated_proportions)
total_captial_tvar = total_losses.tvar(90)

print(f"Total capital required at 90% TVaR: {total_captial_tvar}")
allocated_capital_tvar = risk_measures.tvar(total_losses, 90).allocate(losses)
print("Allocated capital to each line of business using TVaR:")
print(allocated_capital_tvar)
allocated_proportions_tvar = allocated_capital_tvar / total_captial_tvar
print("Proportion of total capital allocated to each line of business using TVaR:")
print(allocated_proportions_tvar)
