from pal import config, distributions
from pal.frequency_severity import FrequencySeverityModel
from pal import copulas
from pal.variables import ProteusVariable
import plotly.graph_objects as go  # type: ignore

config.n_sims = 100000

lobs = ["Motor", "Property", "Liability", "Marine", "Aviation"]
# Generate the individual large losses by class
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

large_losses_with_LAE = individual_large_losses_by_lob * 1.05

# create the aggregate losses by class
aggregate_large_losses_by_class = ProteusVariable(
    "class", {name: large_losses_with_LAE[name].aggregate() for name in lobs}
)
# correlate the attritional and large losses. Use a pairwise copula to do this
for lob in lobs:
    copulas.GumbelCopula(theta=1.2, n=2).apply(
        [aggregate_large_losses_by_class[lob], attritional_losses_by_lob[lob]]
    )
# calculate the total losses
total_losses_by_lob = aggregate_large_losses_by_class + attritional_losses_by_lob

# apply a copula to the total losses by lob
correlation_matrix = [
    [1.0, 0.5, 0.3, 0.2, 0.1],
    [0.5, 1.0, 0.4, 0.3, 0.2],
    [0.3, 0.4, 1.0, 0.5, 0.4],
    [0.2, 0.3, 0.5, 1.0, 0.6],
    [0.1, 0.2, 0.4, 0.6, 1.0],
]
copulas.StudentsTCopula(correlation_matrix, 5, "linear").apply(total_losses_by_lob)
# apply stochastic inflation
stochastic_inflation = distributions.Normal(0.05, 0.02).generate()
inflated_total_losses_by_lob: ProteusVariable = total_losses_by_lob * (
    1 + stochastic_inflation
)

# create the total losses
total_inflated_losses = inflated_total_losses_by_lob.sum()

print(total_inflated_losses.tvar(99))

total_inflated_losses.show_cdf()

fig = go.Figure(
    go.Scattergl(
        x=inflated_total_losses_by_lob["Motor"].ranks.values,
        y=inflated_total_losses_by_lob["Property"].ranks.values,
        mode="markers",
    ),
    layout=dict(
        xaxis=dict(title="Motor - Rank"),
        yaxis=dict(title="Property - Rank"),
        title="Scatter plot of Motor and Property losses",
    ),
)
fig.show()
