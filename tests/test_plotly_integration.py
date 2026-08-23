"""Regression tests for Plotly integration with PAL stochastic variables."""

import json
import typing as t

import numpy as np
import plotly.graph_objects as go  # type: ignore
import pytest

from pal.variables import StochasticScalar


def _capture_show(monkeypatch: pytest.MonkeyPatch) -> list[t.Any]:
    """Capture figures passed to ``show`` after exercising JSON serialization."""
    figures: list[t.Any] = []

    def capture_show(figure: t.Any, *args: t.Any, **kwargs: t.Any) -> None:
        del args, kwargs
        figure.to_json()
        figures.append(figure)

    monkeypatch.delenv("PAL_SUPPRESS_PLOTS", raising=False)
    monkeypatch.setattr(go.Figure, "show", capture_show)
    return figures


def _assert_typed_array(value: t.Any) -> None:
    """Assert that Plotly serialized an array through its typed-array path."""
    assert isinstance(value, dict)
    assert "dtype" in value
    assert "bdata" in value


def test_plotly_scatter_serializes_stochastic_scalar() -> None:
    """Plotly should accept StochasticScalar directly, including GPU-backed values."""
    values = StochasticScalar([4, 5, 2, 1, 3])

    fig = go.Figure(go.Scattergl(x=values.ranks, y=values.ranks))
    serialized = json.loads(fig.to_json())

    _assert_typed_array(serialized["data"][0]["x"])
    _assert_typed_array(serialized["data"][0]["y"])
    np.testing.assert_array_equal(np.asarray(fig.data[0].x), [3, 4, 1, 0, 2])
    np.testing.assert_array_equal(np.asarray(fig.data[0].y), [3, 4, 1, 0, 2])


def test_show_histogram_serializes_stochastic_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    """The histogram helper should pass the PAL object through Plotly serialization."""
    figures = _capture_show(monkeypatch)
    values = StochasticScalar([4, 5, 2, 1, 3])

    values.show_histogram(title="Histogram")

    assert len(figures) == 1
    fig = figures[0]
    serialized = json.loads(fig.to_json())
    assert fig.data[0].type == "histogram"
    assert fig.layout.title.text == "Histogram"
    _assert_typed_array(serialized["data"][0]["x"])
    np.testing.assert_array_equal(np.asarray(fig.data[0].x), [4, 5, 2, 1, 3])


def test_show_cdf_serializes_stochastic_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CDF helper should serialize backend-neutral PAL objects for both axes."""
    figures = _capture_show(monkeypatch)
    values = StochasticScalar([4, 5, 2, 1, 3])

    values.show_cdf(title="CDF")

    assert len(figures) == 1
    fig = figures[0]
    serialized = json.loads(fig.to_json())
    assert fig.data[0].type == "scatter"
    assert fig.layout.title.text == "CDF"
    _assert_typed_array(serialized["data"][0]["x"])
    _assert_typed_array(serialized["data"][0]["y"])
    np.testing.assert_array_equal(np.asarray(fig.data[0].x), [1, 2, 3, 4, 5])
    np.testing.assert_allclose(np.asarray(fig.data[0].y), [0.0, 0.2, 0.4, 0.6, 0.8])
