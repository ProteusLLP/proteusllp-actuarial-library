"""Tests for Plotly figure helpers on stochastic variables."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go  # type: ignore
import pytest

from pal.stochastic_scalar import StochasticScalar
from pal.variables import ProteusVariable


@pytest.fixture
def variables() -> ProteusVariable[StochasticScalar]:
    """Return three stochastic variables with distinct values and ranks."""
    return ProteusVariable(
        "factor",
        {
            "A": StochasticScalar([1.0, 2.0, 3.0, 4.0]),
            "B": StochasticScalar([4.0, 2.0, 3.0, 1.0]),
            "C": StochasticScalar([1.0, 3.0, 2.0, 4.0]),
        },
    )


def test_rank_scatter_plot_returns_all_pairs(variables: ProteusVariable[StochasticScalar]) -> None:
    fig = variables.rank_scatter_plot(title="Ranks")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Ranks"
    assert len(fig.data) == 3
    assert [trace.name for trace in fig.data] == ["A vs B", "A vs C", "B vs C"]
    assert all(trace.type == "scattergl" for trace in fig.data)
    np.testing.assert_array_equal(np.asarray(fig.data[0].x), [0, 1, 2, 3])
    np.testing.assert_array_equal(np.asarray(fig.data[0].y), [3, 1, 2, 0])
    fig.to_json()


def test_value_scatter_plot_returns_all_pairs(variables: ProteusVariable[StochasticScalar]) -> None:
    fig = variables.value_scatter_plot()

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    np.testing.assert_array_equal(np.asarray(fig.data[0].x), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(np.asarray(fig.data[0].y), [4.0, 2.0, 3.0, 1.0])
    fig.to_json()


@pytest.mark.parametrize("method_name", ["rank_scatter_plot", "value_scatter_plot"])
def test_pair_scatter_can_use_frames(variables: ProteusVariable[StochasticScalar], method_name: str) -> None:
    fig = getattr(variables, method_name)(frames=True)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert [frame.name for frame in fig.frames] == ["A vs B", "A vs C", "B vs C"]
    assert len(fig.layout.sliders) == 1
    assert len(fig.layout.sliders[0].steps) == 3
    fig.to_json()


def test_pair_scatter_requires_two_variables() -> None:
    variable = ProteusVariable("factor", {"A": StochasticScalar([1.0, 2.0])})

    with pytest.raises(ValueError, match="at least two variables"):
        variable.rank_scatter_plot()


def test_stochastic_scalar_histogram_and_cdf_plots_return_figures() -> None:
    values = StochasticScalar([4.0, 5.0, 2.0, 1.0, 3.0])

    histogram = values.histogram_plot(title="Histogram")
    cdf = values.cdf_plot(title="CDF")

    assert isinstance(histogram, go.Figure)
    assert isinstance(cdf, go.Figure)
    assert histogram.layout.title.text == "Histogram"
    assert cdf.layout.title.text == "CDF"
    np.testing.assert_array_equal(np.asarray(histogram.data[0].x), [4.0, 5.0, 2.0, 1.0, 3.0])
    np.testing.assert_array_equal(np.asarray(cdf.data[0].x), [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(np.asarray(cdf.data[0].y), [0.0, 0.2, 0.4, 0.6, 0.8])


def test_show_helpers_return_figures_without_showing(
    monkeypatch: pytest.MonkeyPatch,
    variables: ProteusVariable[StochasticScalar],
) -> None:
    monkeypatch.setenv("PAL_SUPPRESS_PLOTS", "true")
    values = StochasticScalar([1.0, 2.0, 3.0])

    assert isinstance(values.show_histogram(), go.Figure)
    assert isinstance(values.show_cdf(), go.Figure)
    assert isinstance(variables.show_histogram(), go.Figure)
    assert isinstance(variables.show_cdf(), go.Figure)
    assert isinstance(variables.show_box_plot(), go.Figure)


def test_proteus_variable_histogram_and_cdf_plots_return_figures(
    variables: ProteusVariable[StochasticScalar],
) -> None:
    histogram = variables.histogram_plot()
    cdf = variables.cdf_plot()

    assert isinstance(histogram, go.Figure)
    assert isinstance(cdf, go.Figure)
    assert len(histogram.data) == 3
    assert len(cdf.data) == 3
    histogram.to_json()
    cdf.to_json()


@pytest.mark.parametrize("constant", [3, 3.5])
def test_proteus_variable_constant_histogram_and_cdf_plots_return_figures(constant: int | float) -> None:
    variable = ProteusVariable("factor", {"constant": constant})

    histogram = variable.histogram_plot()
    cdf = variable.cdf_plot()

    assert isinstance(histogram, go.Figure)
    assert isinstance(cdf, go.Figure)
    np.testing.assert_array_equal(np.asarray(histogram.data[0].x), [constant])
    np.testing.assert_array_equal(np.asarray(cdf.data[0].x), [constant])
    np.testing.assert_array_equal(np.asarray(cdf.data[0].y), [0.0])
    histogram.to_json()
    cdf.to_json()


def test_proteus_variable_box_plot_returns_one_box_per_variable(
    variables: ProteusVariable[StochasticScalar],
) -> None:
    figure = variables.box_plot(title="Box Plot")

    assert isinstance(figure, go.Figure)
    assert figure.layout.title.text == "Box Plot"
    assert len(figure.data) == 3
    assert all(trace.type == "box" for trace in figure.data)
    assert [trace.name for trace in figure.data] == ["A", "B", "C"]
    assert all(trace.marker.color == "#2F6B7C" for trace in figure.data)
    assert all(trace.line.color == "#2F6B7C" for trace in figure.data)
    assert all(trace.boxpoints is False for trace in figure.data)
    assert all(trace.orientation == "v" for trace in figure.data)
    assert figure.layout.xaxis.title.text == "factor"
    assert figure.layout.yaxis.title.text == "Value"
    assert all(trace.showlegend is False for trace in figure.data)
    assert all(list(trace.x) == [trace.name] for trace in figure.data)
    assert figure.data[0].q1 == (1.75,)
    assert figure.data[0].median == (2.5,)
    assert figure.data[0].q3 == (3.25,)
    figure.to_json()


def test_proteus_variable_box_plot_accepts_numeric_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAL_SUPPRESS_PLOTS", "true")
    variable = ProteusVariable("factor", {"integer": 3, "float": 3.5})

    figure = variable.show_box_plot(title="Constants")

    assert isinstance(figure, go.Figure)
    assert [trace.name for trace in figure.data] == ["integer", "float"]
    assert figure.data[0].x == ("integer",)
    assert figure.data[1].x == ("float",)
    figure.to_json()


def test_proteus_variable_box_plot_uses_configured_whisker_percentiles() -> None:
    variable = ProteusVariable("factor", {"loss": StochasticScalar([0.0, 1.0, 2.0, 100.0])})

    figure = variable.box_plot(lower_percentile=25.0, upper_percentile=75.0)
    trace = figure.data[0]

    assert trace.lowerfence == (0.75,)
    assert trace.upperfence == (26.5,)
    assert trace.mean == (25.75,)


@pytest.mark.parametrize(
    ("lower_percentile", "upper_percentile"),
    [(-1.0, 99.5), (0.5, 101.0), (75.0, 25.0)],
)
def test_proteus_variable_box_plot_rejects_invalid_percentiles(
    lower_percentile: float,
    upper_percentile: float,
) -> None:
    variable = ProteusVariable("factor", {"loss": StochasticScalar([1.0, 2.0])})

    with pytest.raises(ValueError, match="Whisker percentiles"):
        variable.box_plot(
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )


def test_proteus_variable_percentile_fan_plot_returns_bands_and_summary_lines(
    variables: ProteusVariable[StochasticScalar],
) -> None:
    figure = variables.percentile_fan_plot(title="Fan")

    assert isinstance(figure, go.Figure)
    assert figure.layout.title.text == "Fan"
    assert figure.layout.xaxis.title.text == "factor"
    assert figure.layout.yaxis.title.text == "Value"
    assert len(figure.data) == 30
    assert [trace.name for trace in figure.data[-2:]] == ["Median", "Mean"]
    assert figure.data[1].fill == "tonexty"
    assert figure.data[1].fillcolor == "rgba(0, 188, 254, 0.56)"
    assert figure.data[13].fillcolor == "rgba(28, 45, 145, 0.56)"
    assert figure.data[27].fillcolor == "rgba(0, 188, 254, 0.56)"
    assert figure.data[1].name == "0.1-0.5th percentile"
    assert figure.data[27].name == "99-99.5th percentile"
    assert all(trace.line.width == 0 for trace in figure.data[:-2])
    assert figure.data[-1].line.color == "#E76F2F"
    assert list(figure.data[-2].x) == ["A", "B", "C"]
    np.testing.assert_allclose(np.asarray(figure.data[-2].y), [2.5, 2.5, 2.5])
    figure.to_json()


def test_proteus_variable_percentile_fan_plot_accepts_constants_and_custom_percentiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PAL_SUPPRESS_PLOTS", "true")
    variable = ProteusVariable("period", {"first": 2, "second": 4.0})

    figure = variable.show_percentile_fan_plot(percentiles=(10.0, 50.0, 90.0))

    assert isinstance(figure, go.Figure)
    assert list(figure.data[0].x) == ["first", "second"]
    assert list(figure.data[-1].y) == [2.0, 4.0]


def test_proteus_variable_percentile_fan_plot_generates_dynamic_band_colors() -> None:
    variable = ProteusVariable("factor", {"loss": StochasticScalar([1.0, 2.0, 3.0, 4.0])})
    percentiles = tuple(float(value) for value in (*range(1, 50, 2), 50, *range(51, 100, 2)))

    figure = variable.percentile_fan_plot(percentiles=percentiles)
    band_colors = [trace.fillcolor for trace in figure.data[1:-2:2]]

    assert len(band_colors) == 50
    assert band_colors[0] == "rgba(0, 188, 254, 0.56)"
    assert band_colors[len(band_colors) // 2] == "rgba(28, 45, 145, 0.56)"
    assert band_colors[-1] == "rgba(0, 188, 254, 0.56)"


@pytest.mark.parametrize(
    "percentiles",
    [(25.0, 75.0), (0.0, 25.0, 40.0, 75.0, 100.0), (10.0, 40.0, 90.0)],
)
def test_proteus_variable_percentile_fan_plot_rejects_invalid_percentiles(
    percentiles: tuple[float, ...],
) -> None:
    variable = ProteusVariable("factor", {"loss": StochasticScalar([1.0, 2.0])})

    with pytest.raises(ValueError, match="percentiles must be sorted"):
        variable.percentile_fan_plot(percentiles=percentiles)


def test_returned_figure_can_be_saved(tmp_path: Path, variables: ProteusVariable[StochasticScalar]) -> None:
    fig = variables.rank_scatter_plot()
    output = tmp_path / "rank-scatter.html"

    fig.write_html(output)

    assert output.exists()


def test_ambiguous_plot_method_names_are_not_exposed(
    variables: ProteusVariable[StochasticScalar],
) -> None:
    values = StochasticScalar([1.0, 2.0, 3.0])

    for obj in (values, variables):
        assert not hasattr(obj, "histogram")
        assert not hasattr(obj, "cdf")

    assert not hasattr(variables, "rank_scatter")
    assert not hasattr(variables, "value_scatter")
