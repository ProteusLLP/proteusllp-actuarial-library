"""Tests for general array-like StochasticScalar inputs."""

import pandas as pd

from pal import StochasticScalar
from tests._assertions import array_equal


def test_stochastic_scalar_accepts_pandas_series() -> None:
    """A pandas Series can be used directly as a StochasticScalar input."""
    values = pd.Series([4.0, 5.0, 2.0, 1.0, 3.0])

    result = StochasticScalar(values)

    assert result.n_sims == 5
    assert array_equal(result.values, [4.0, 5.0, 2.0, 1.0, 3.0])
