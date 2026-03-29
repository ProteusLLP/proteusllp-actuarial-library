import os

import numpy as np
import pytest


def _skip_unless_gpu_mode() -> None:
    if os.environ.get("PAL_USE_GPU") != "1":
        pytest.skip("PAL_USE_GPU is not enabled")

    try:
        import cupy  # noqa: F401
    except ImportError:
        pytest.skip("cupy is not installed")


def test_gpu_backend_selected() -> None:
    _skip_unless_gpu_mode()

    from pal._maths import xp

    assert xp.__name__ == "cupy"


def test_numpy_ufunc_dispatch_on_stochastic_scalar() -> None:
    _skip_unless_gpu_mode()

    import cupy as cp

    from pal import StochasticScalar

    x = StochasticScalar([1.0, 2.0, 3.0])
    y = np.exp(x)

    assert isinstance(y, StochasticScalar)
    assert isinstance(y.values, cp.ndarray)


def test_distribution_generate_returns_cupy_array() -> None:
    _skip_unless_gpu_mode()

    import cupy as cp

    from pal import distributions

    sims = distributions.Normal(0.0, 1.0).generate(n_sims=128)

    assert isinstance(sims.values, cp.ndarray)
    assert sims.n_sims == 128
