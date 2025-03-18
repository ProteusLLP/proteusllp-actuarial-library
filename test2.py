from rippy import distributions
import numpy as np

# create a simulated year event table

rng = np.random.default_rng(12346)
n_sims = 100000
sim_event_id = np.unique(rng.integers(100000, 1000000, n_sims * 10))
sim_no = rng.integers(0, n_sims, len(sim_event_id))
sim_no.sort()
print(sim_event_id)
print(sim_no)

import pandas as pd

df = pd.DataFrame({"SimNo": sim_no, "SimEventId": sim_event_id})
df.to_csv("data/master_ylt_full.csv", index=False)

# generate some simulated losses

loss1 = distributions.GPD(shape=0.33, scale=100000, loc=0).generate(
    len(sim_event_id)
) * distributions.Binomial(n=1, p=0.5).generate(len(sim_event_id))
loss2 = distributions.GPD(shape=0.33, scale=100000, loc=0).generate(
    len(sim_event_id)
) * distributions.Binomial(n=1, p=0.5).generate(len(sim_event_id))

loss1values = loss1[loss1 > 0].values
loss2values = loss2[loss2 > 0].values
loss1eventids = sim_event_id[loss1.values > 0]
loss2eventids = sim_event_id[loss2.values > 0]

df1 = pd.DataFrame({"SimEventId": loss1eventids, "Loss": loss1values})
df2 = pd.DataFrame({"SimEventId": loss2eventids, "Loss": loss2values})
df1.to_csv("data/cat_ylt_full.csv", index=False)
df2.to_csv("data/cat_ylt2_full.csv", index=False)
