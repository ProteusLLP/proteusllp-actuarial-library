"""Check the public host conversion used by plotting integrations."""

import numpy as np

from pal._maths import xp
from pal.maths import asnumpy


def test_asnumpy_converts_backend_values_to_host_numpy():
    values = xp.asarray([0.1, 0.25, 0.4])

    result = asnumpy(values)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [0.1, 0.25, 0.4])


def test_asnumpy_accepts_numpy_values_without_copying():
    values = np.asarray([0.2, 0.6, 0.8])

    result = asnumpy(values)

    assert result is values
