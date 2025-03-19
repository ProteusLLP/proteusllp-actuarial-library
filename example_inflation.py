from pcm import config
from pcm.config import xp as np
from pcm.variables import ProteusVariable, StochasticScalar

n_sims = 100000
config.n_sims = n_sims

inflation_index = ProteusVariable.from_csv(
    "data/proteus_scenario_generator/Economics_USD_Inflation_Inflation Index.csv",
    "Time",
    "Index",
)
# upsample the inflation index to the correct number of simulations
upsampler = StochasticScalar(np.arange(0, n_sims) % inflation_index.n_sims)
upsampled_inflation_index = inflation_index.get_value_at_sim(upsampler)
inflation_rate = np.diff(np.log(upsampled_inflation_index), prepend=np.nan)

inflation_rate.show_cdf()
