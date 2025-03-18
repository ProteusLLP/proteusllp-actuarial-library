from rippy.variables import ProteusVariable
from rippy.copulas import (
    GaussianCopula,
    ClaytonCopula,
    GumbelCopula,
    FrankCopula,
    JoeCopula,
    StudentsTCopula,
)
from rippy.frequency_severity import FrequencySeverityModel, FreqSevSims
from rippy.distributions import GPD, Poisson, Normal
import numpy as np
from rippy import config

config.n_sims = 10000
config.seed = 1234
x = StudentsTCopula([[1, 0.5], [0.5, 1]], 5).generate()

import plotly.graph_objects as go

fig = go.Figure(go.Scatter(x=x[0].values, y=x[1].values))
fig.show()

# generate a set of frequency severity losses
sev_dist = GPD(shape=0.33, scale=100000, loc=0)
freq_dist = Poisson(mean=10)
class_names = ["Motor", "Property", "Liability", "Marine", "Aviation"]
losses = ProteusVariable(
    dim_name="class",
    values={
        name: FrequencySeverityModel(freq_dist, sev_dist).generate()
        for name in class_names
    },
)
re_ordered_class_names = class_names[::-1]

scale_by_class = ProteusVariable(
    dim_name="class",
    values={name: 1 + i * 0.1 for i, name in enumerate(re_ordered_class_names)},
)

scaled_losses = losses * scale_by_class
print(scaled_losses["Property"])
# aggregate the losses
agg_losses = [
    loss.aggregate() if type(loss) is FreqSevSims else 0 for loss in scaled_losses
]
exit()
# apply a copula to the losses
for i in range(9):
    JoeCopula(2, 2).apply([agg_losses[i], agg_losses[i + 1]])

# test the losses have the correct correlation
xa = np.array([xx.values for xx in agg_losses])
print(np.corrcoef(xa))

# look at the individual losses
re_calculated_scaled_losses = (losses[1] * 2).aggregate()
re_calculated_aggregated_losses = (scaled_losses[1]).aggregate()
print(re_calculated_scaled_losses)
print(re_calculated_aggregated_losses)
print((agg_losses[1]))

print((losses[1]))

inflation_factor = 1 + Normal(0.05, 0.02).generate()
# apply inflation to the losses
inflated_losses = losses * inflation_factor

inflated_losses.show_histogram()
