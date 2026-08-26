"""GPU-only checks for backend residency and NumPy dispatch."""

import numpy as np
import pytest

from pal import distributions, set_random_seed
from pal._maths import xp
from pal.config import config
from pal.copulas import StudentsTCopula
from pal.stochastic_scalar import StochasticScalar

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
        simulations = distributions.Normal(100, 15).generate(4096)
        students_t = distributions.StudentsT(5, 100, 15).generate(4096)
        students_t_copula = StudentsTCopula([[1, 0.5], [0.5, 1]], 5).generate(4096)
        inverse_gaussian = distributions.InverseGaussian(2, 3).generate(256)
        negative_binomial = distributions.NegBinomial(4, 0.5).generate(256)
        multivariate_normal_distribution = distributions.MultivariateNormal([0, 0], [[1, 0.5], [0.5, 1]])
        multivariate_students_t_distribution = distributions.MultivariateStudentsT(5, [0, 0], [[1, 0.5], [0.5, 1]])
        dirichlet_distribution = distributions.Dirichlet([2, 3, 4])
        inverted_dirichlet_distribution = distributions.InvertedDirichlet([2, 3, 8])
        generalized_dirichlet_distribution = distributions.GeneralizedDirichlet([2, 3], [4, 5])
        inverted_generalized_dirichlet_distribution = distributions.InvertedGeneralizedDirichlet([2, 3], [6, 8])
        multinomial_distribution = distributions.Multinomial(20, [0.2, 0.3, 0.5])
        wishart_distribution = distributions.Wishart(8, [[1, 0.2], [0.2, 0.7]])
        inverse_wishart_distribution = distributions.InverseWishart(10, [[1, 0.2], [0.2, 0.7]])
        multivariate_normal = multivariate_normal_distribution.generate(256)
        multivariate_students_t = multivariate_students_t_distribution.generate(256)
        dirichlet = dirichlet_distribution.generate(256)
        inverted_dirichlet = inverted_dirichlet_distribution.generate(256)
        generalized_dirichlet = generalized_dirichlet_distribution.generate(256)
        inverted_generalized_dirichlet = inverted_generalized_dirichlet_distribution.generate(256)
        multinomial = multinomial_distribution.generate(256)
        wishart = wishart_distribution.generate(256)
        inverse_wishart = inverse_wishart_distribution.generate(256)
        transformed = np.where(simulations > 100, np.exp(simulations / 100), np.square(simulations / 100))

        assert type(config.rng).__module__.startswith("cupy")
        assert isinstance(simulations.values, xp.ndarray)
        assert isinstance(students_t.values, xp.ndarray)
        assert all(isinstance(margin.values, xp.ndarray) for margin in students_t_copula)
        assert isinstance(inverse_gaussian.values, xp.ndarray)
        assert isinstance(negative_binomial.values, xp.ndarray)
        multivariate_samples = (
            multivariate_normal,
            multivariate_students_t,
            dirichlet,
            inverted_dirichlet,
            generalized_dirichlet,
            inverted_generalized_dirichlet,
            multinomial,
        )
        multivariate_densities = (
            multivariate_normal_distribution.logpdf(multivariate_normal),
            multivariate_students_t_distribution.logpdf(multivariate_students_t),
            dirichlet_distribution.logpdf(dirichlet),
            inverted_dirichlet_distribution.logpdf(inverted_dirichlet),
            generalized_dirichlet_distribution.logpdf(generalized_dirichlet),
            inverted_generalized_dirichlet_distribution.logpdf(inverted_generalized_dirichlet),
            multinomial_distribution.logpmf(multinomial),
            wishart_distribution.logpdf(wishart),
            inverse_wishart_distribution.logpdf(inverse_wishart),
        )
        assert all(isinstance(component.values, xp.ndarray) for sample in multivariate_samples for component in sample)
        assert all(
            isinstance(entry.values, xp.ndarray)
            for sample in (wishart, inverse_wishart)
            for row in sample
            for entry in row
        )
        assert all(
            isinstance(density, StochasticScalar) and isinstance(density.values, xp.ndarray)
            for density in multivariate_densities
        )
        assert isinstance(transformed.values, xp.ndarray)
        assert not host_to_device_transfers
    finally:
        config.rng = old_rng
