"""An example of loading an inflation series from the Proteus Scenario Generator."""

from pal import config, variables

n_sims = 100000
config.n_sims = n_sims

inflation_index = variables.ProteusVariable.from_csv(
    "data/proteus_scenario_generator/Economics_USD_Inflation_Inflation Index.csv",
    "Time",
    "Index",
)
# upsample the inflation index to the correct number of simulations
upsampled_inflation_index = inflation_index.upsample(n_sims)

upsampled_inflation_index.cdf_plot().show()
