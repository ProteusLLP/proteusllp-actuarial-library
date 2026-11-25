"""Independent tail checks for the FRA1 auxiliary transform."""

import numpy as np
import pytest

from pal import _fra1
from pal._maths import asnumpy, xp


@pytest.mark.parametrize("p", [0.0, 0.5])
def test_auxiliary_transform_large_log_arguments(p):
    log_s = np.array([400.0, 500.0, 700.0, 1000.0])
    # F_0(s) = acosh(1+s)^2 / 2; F_1/2(s) = 4(sqrt(1+s/2)-1).
    # At these arguments the exponentially small correction to acosh(1+s)
    # is below double precision, while s itself need not be representable.
    if p == 0.0:
        a = log_s + np.log(2.0)
        expected = (2 * np.log(a) - np.log(2.0), np.log(a) - log_s, np.log(a - 1) - 2 * log_s)
    else:
        log_root = 0.5 * np.logaddexp(0.0, log_s - np.log(2.0))
        expected = (
            np.log(4.0) + log_root + np.log1p(-np.exp(-log_root)),
            -log_root,
            -np.log(4.0) - 3 * log_root,
        )
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = _fra1._log_f(xp.asarray(log_s), p)
    for values, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(asnumpy(values), reference, rtol=0, atol=5e-13)
