import math

import pandas as pd  # type: ignore
from pal import config, distributions, np
from pal.variables import FreqSevSims, ProteusVariable

n_sims = 100_000
config.n_sims = n_sims
lobs = ["Motor", "Property", "Liability", "Marine", "Aviation"]
# load the cat ylts
df = pd.read_csv("data/catastrophes/cat_ylt.csv", index_col=0)  # type: ignore[misc]
# upsample the cat ylts to the correct number of simulations
ylt_sims = 10000
up_sample_factor = math.ceil(n_sims / ylt_sims)
sim_index = np.array(df["sim"].values).repeat(up_sample_factor)
cat_losses = ProteusVariable(
    "lob",
    {
        lob: FreqSevSims(
            sim_index,
            df[lob].values.repeat(up_sample_factor),  # type: ignore[misc]
            n_sims=config.n_sims,
        )
        for lob in lobs
    },
)

inflation_rate = distributions.Normal(0.05, 0.02).generate()

scaled_cat_losses_by_lob = cat_losses * (1 + inflation_rate)

scaled_cat_losses = sum(scaled_cat_losses_by_lob)

recoveries: FreqSevSims = np.minimum(
    np.maximum(scaled_cat_losses - 10000000, 0), 10000000
)  # type: ignore[misc]

recoveries.aggregate().show_cdf()
