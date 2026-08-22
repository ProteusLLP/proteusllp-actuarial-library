"""GPU-only checks for backend residency and NumPy dispatch."""

import numpy as np
import pytest

from pal import Beta, InverseGaussian, NegBinomial, Normal, StudentsT, set_random_seed
from pal._maths import xp
from pal.config import config
from pal.copulas import StudentsTCopula

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
        students_t = StudentsT(5, 100, 15).generate(4096)
        students_t_copula = StudentsTCopula([[1, 0.5], [0.5, 1]], 5).generate(4096)
        inverse_gaussian = InverseGaussian(2, 3).generate(256)
        negative_binomial = NegBinomial(4, 0.5).generate(256)
        beta_probability = Beta(2, 5).cdf(simulations / 200)
        beta_quantile = Beta(2, 5).invcdf(beta_probability)
        transformed = np.where(simulations > 100, np.exp(simulations / 100), np.square(simulations / 100))

        assert type(config.rng).__module__.startswith("cupy")
        assert isinstance(simulations.values, xp.ndarray)
        assert isinstance(students_t.values, xp.ndarray)
        assert all(isinstance(margin.values, xp.ndarray) for margin in students_t_copula)
        assert isinstance(inverse_gaussian.values, xp.ndarray)
        assert isinstance(negative_binomial.values, xp.ndarray)
        assert isinstance(beta_probability.values, xp.ndarray)
        assert isinstance(beta_quantile.values, xp.ndarray)
        assert isinstance(transformed.values, xp.ndarray)
        assert not host_to_device_transfers
    finally:
        config.rng = old_rng
