"""GPU-only checks for backend residency and NumPy dispatch."""

import numpy as np
import pytest

from pal import InverseGaussian, NegBinomial, Normal, set_random_seed
from pal._maths import xp
from pal.config import config

pytestmark = pytest.mark.skipif(xp.__name__ != "cupy", reason="requires the CuPy backend")


def test_generation_and_numpy_dispatch_do_not_transfer_to_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep generated arrays and ordinary NumPy-dispatched maths on the GPU."""
    old_rng = config.rng
    original_asarray = xp.asarray
    host_to_device_transfers: list[type[object]] = []

    def track_asarray(value: object, *args: object, **kwargs: object):
        if isinstance(value, np.ndarray):
            host_to_device_transfers.append(type(value))
        return original_asarray(value, *args, **kwargs)

    def reject_asnumpy(*args: object, **kwargs: object):
        raise AssertionError("unexpected device-to-host array transfer")

    monkeypatch.setattr(xp, "asarray", track_asarray)
    monkeypatch.setattr(xp, "asnumpy", reject_asnumpy)

    try:
        set_random_seed(123456789)
        simulations = Normal(100, 15).generate(4096)
        inverse_gaussian = InverseGaussian(2, 3).generate(256)
        negative_binomial = NegBinomial(4, 0.5).generate(256)
        transformed = np.where(simulations > 100, np.exp(simulations / 100), np.square(simulations / 100))

        assert type(config.rng).__module__.startswith("cupy")
        assert isinstance(simulations.values, xp.ndarray)
        assert isinstance(inverse_gaussian.values, xp.ndarray)
        assert isinstance(negative_binomial.values, xp.ndarray)
        assert isinstance(transformed.values, xp.ndarray)
        assert not host_to_device_transfers
    finally:
        config.rng = old_rng
