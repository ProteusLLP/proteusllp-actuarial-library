import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pal import ProteusVariable, StochasticScalar


def rank_scatter_plot(variables: ProteusVariable[StochasticScalar]) -> go.Figure:
    """Create a rank scatter plot for the given variables."""
    variable_names = list(variables.values.keys())
    n = len(variable_names) - 1
    fig = make_subplots(rows=n, cols=n)

    for item1 in range(len(variables)):
        for item2 in range(len(variables)):
            if item1 < item2:
                fig.add_trace(
                    go.Scatter(
                        y=variables[variable_names[item2]].ranks,
                        x=variables[variable_names[item1]].ranks,
                        mode="markers",
                        marker={"size": 3, "opacity": 0.5},
                        name=f"{variable_names[item1]} vs {variable_names[item2]}",
                    ),
                    row=item1 + 1,
                    col=item2,
                )
                fig.update_yaxes(
                    title_text=f"Ranks - {variable_names[item1]}",
                    row=item1 + 1,
                    col=item2,
                )
                fig.update_xaxes(
                    title_text=f"Ranks - {variable_names[item2]}",
                    row=item1 + 1,
                    col=item2,
                )

    return fig
