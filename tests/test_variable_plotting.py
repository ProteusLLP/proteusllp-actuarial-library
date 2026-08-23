"""Tests for Plotly figure helpers on stochastic variables."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go  # type: ignore
import pytest

from pal import ProteusVariable, StochasticScalar


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


def test_rank_scatter_returns_all_pairs(variables: ProteusVariable[StochasticScalar]) -> None:
    fig = variables.rank_scatter(title="Ranks")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Ranks"
    assert len(fig.data) == 3
    assert [trace.name for trace in fig.data] == ["A vs B", "A vs C", "B vs C"]
    assert all(trace.type == "scattergl" for trace in fig.data)
    np.testing.assert_array_equal(np.asarray(fig.data[0].x), [0, 1, 2, 3])
    np.testing.assert_array_equal(np.asarray(fig.data[0].y), [3, 1, 2, 0])
    fig.to_json()


def test_value_scatter_returns_all_pairs(variables: ProteusVariable[StochasticScalar]) -> None:
    fig = variables.value_scatter()

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    np.testing.assert_array_equal(np.asarray(fig.data[0].x), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(np.asarray(fig.data[0].y), [4.0, 2.0, 3.0, 1.0])
    fig.to_json()


@pytest.mark.parametrize("method_name", ["rank_scatter", "value_scatter"])
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
        variable.rank_scatter()


def test_stochastic_scalar_histogram_and_cdf_return_figures() -> None:
    values = StochasticScalar([4.0, 5.0, 2.0, 1.0, 3.0])

    histogram = values.histogram(title="Histogram")
    cdf = values.cdf(title="CDF")

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


def test_proteus_variable_histogram_and_cdf_return_figures(
    variables: ProteusVariable[StochasticScalar],
) -> None:
    histogram = variables.histogram()
    cdf = variables.cdf()

    assert isinstance(histogram, go.Figure)
    assert isinstance(cdf, go.Figure)
    assert len(histogram.data) == 3
    assert len(cdf.data) == 3
    histogram.to_json()
    cdf.to_json()


def test_returned_figure_can_be_saved(tmp_path: Path, variables: ProteusVariable[StochasticScalar]) -> None:
    fig = variables.rank_scatter()
    output = tmp_path / "rank-scatter.html"

    fig.write_html(output)

    assert output.exists()
