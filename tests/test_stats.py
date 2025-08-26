"""Tests for statistical utilities in the stats module."""

import numpy as np
import pytest
from pal.stats import tvar
from pal.variables import StochasticScalar


@pytest.mark.parametrize(
    "values,percentile,expected",
    [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 80, 9.5),
        ([1, 2, 3, 4, 5], 60, 4.5),
        ([42], 50, 42),
        ([5, 5, 5, 5, 5], 80, 5.0),
    ],
)
def test_tvar_single_percentile(
    values: list[float], percentile: float, expected: float
) -> None:
    """Test TVAR calculation with single percentiles."""
    result = tvar(np.array(values), percentile)
    assert result == expected


def test_tvar_stochastic_scalar() -> None:
    """Test TVAR works with StochasticScalar via __array__ conversion."""
    ss = StochasticScalar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = tvar(ss, 80)
    assert result == 9.5


def test_tvar_multiple_percentiles() -> None:
    """Test TVAR calculation with multiple percentiles."""
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = tvar(values, [50, 80])
    assert result == [8.0, 9.5]


def test_tvar_empty_array() -> None:
    """Test TVAR with empty array raises appropriate error."""
    values = np.array([])
    with pytest.raises(ValueError, match="Cannot compute TVAR for empty array"):
        tvar(values, 50)
