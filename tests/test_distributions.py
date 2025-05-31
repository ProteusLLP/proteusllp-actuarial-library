from pal import distributions
from pal.config import set_random_seed, xp as np
import pytest
import math
import scipy.special
from scipy.special import gamma


def test_Poisson():
    set_random_seed(12345678910)
    lamda = 3.5
    dist = distributions.Poisson(lamda)
    assert dist.cdf(0) == np.exp(-lamda)
    assert dist.invcdf(0) == 0
    assert np.allclose(
        dist.invcdf(dist.cdf(np.array([0, 2, 5, 10]))),
        np.array([0, 2, 5, 10]),
        1e-8,
    )
    sims = dist.generate(100000)
    assert np.isclose(sims.mean(), lamda, 1e-3)
    assert np.isclose(sims.std() ** 2, lamda, 1e-2)


def test_Poisson_gamma():
    """Tests the Poisson distribution with a gamma distributed lambda."""
    set_random_seed(12345678910)
    alpha = 0.5
    beta = 3
    lamda = distributions.Gamma(alpha, beta).generate(100000)
    sims = distributions.Poisson(lamda).generate(100000)
    sims_mean = sims.mean()
    sims_std = sims.std()
    assert np.isclose(sims_mean, alpha * beta, 1e-2)
    assert np.isclose(sims_std**2, alpha * beta + alpha * beta**2, 1e-2)
    assert sims.coupled_variable_group == lamda.coupled_variable_group


def test_gamma_exp():
    """Tests the Exponential distribution with a gamma distributed lambda."""
    set_random_seed(12345678910)
    alpha = 1.5
    beta = 3
    lamda = distributions.Gamma(alpha, beta).generate(1000000)
    sims = distributions.Exponential(lamda).generate(1000000)
    sims_mean = sims.mean()
    sims_std = sims.std()
    assert np.isclose(sims_mean, alpha * beta, 1e-2)
    assert np.isclose(sims_std**2, (2 * alpha + alpha**2) * beta**2, 1e-2)
    assert sims.coupled_variable_group == lamda.coupled_variable_group


def test_Beta():
    set_random_seed(12345678910)
    alpha = 2
    beta = 3
    scale = 10000000
    loc = 1000000
    dist = distributions.Beta(alpha, beta, scale, loc)
    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert np.allclose(
        dist.invcdf(dist.cdf(np.array([1234560.1, 2345670, 3456780]))),
        np.array([1234560.1, 2345670, 3456780]),
        1e-8,
    )

    sims = dist.generate(1000000)
    assert np.allclose(sims.mean(), alpha / (alpha + beta) * scale + loc, 1e-3)
    assert np.allclose(
        sims.std(),
        math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))) * scale,
        1e-3,
    )


def test_GPD():
    set_random_seed(12345678910)
    shape = 0.25
    scale = 100000
    threshold = 1000000
    dist = distributions.GPD(shape, scale, threshold)
    assert dist.cdf(1000000) == 0.0
    assert dist.cdf(1500000) == pytest.approx(0.960981557689, 1e-4)
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(0.960981557689) == pytest.approx(1500000, 1e-4)

    sims = dist.generate(100000000)
    assert sims.mean() == pytest.approx(scale / (1 - shape) + threshold, 1e-3)
    assert sims.std() == pytest.approx(
        scale / (1 - shape) / math.sqrt(1 - 2 * shape), 1e-3
    )


def test_Burr():
    set_random_seed(12345678910)
    power = 2
    shape = 3
    scale = 100000
    loc = 1000000
    dist = distributions.Burr(power, shape, scale, loc)
    assert dist.cdf(1000000) == 0.0
    assert dist.cdf(1500000) == pytest.approx(0.9999431042330451, 1e-8)
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(0.9999431042330451) == pytest.approx(1500000, 1e-8)

    sims = dist.generate(10000000)
    assert sims.mean() == pytest.approx(
        shape * scipy.special.beta(shape - 1 / power, 1 + 1 / power) * scale + loc, 1e-3
    )
    assert sims.std() == pytest.approx(
        math.sqrt(
            shape * scipy.special.beta(shape - 2 / power, 1 + 2 / power)
            - shape**2 * scipy.special.beta(shape - 1 / power, 1 + 1 / power) ** 2
        )
        * scale,
        1e-3,
    )


def test_InverseBurr():
    set_random_seed(12345678910)
    power = 4
    shape = 5
    scale = 100000
    loc = 1000000
    dist = distributions.InverseBurr(power, shape, scale, loc)
    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000

    assert dist.invcdf(0.5) == scale * (1 / (2 ** (1 / shape) - 1)) ** (1 / power) + loc

    sims = dist.generate(10000000)

    assert sims.mean() == pytest.approx(
        gamma(1 - 1 / power) * gamma(shape + 1 / power) / gamma(shape) * scale + loc,
        1e-3,
    )
    assert sims.std() == pytest.approx(
        math.sqrt(
            gamma(1 - 2 / power) * gamma(shape + 2 / power) / gamma(shape)
            - (gamma(1 - 1 / power) * gamma(shape + 1 / power) / gamma(shape)) ** 2
        )
        * scale,
        1e-3,
    )


def test_Logistic():
    set_random_seed(12345678910)
    mu = 2.5
    sigma = 2
    dist = distributions.Logistic(mu, sigma)
    assert dist.cdf(2.5) == 0.5
    assert dist.invcdf(0.5) == 2.5
    assert np.allclose(
        dist.invcdf(dist.cdf(np.array([1.1, 2, 3]))),
        np.array([1.1, 2, 3]),
    )

    sims = dist.generate(10000000)
    sims_mean = sims.mean()
    sims_std = sims.std()
    assert np.isclose(sims_mean, mu, 1e-3)
    assert np.isclose(sims_std, np.pi * sigma / np.sqrt(3), 1e-3)


def test_LogLogistic():
    set_random_seed(12345678910)
    shape = 4
    scale = 100000
    loc = 1000000
    dist = distributions.LogLogistic(shape, scale, loc)
    assert dist.cdf(1000000) == 0.0
    assert dist.cdf(1500000) == pytest.approx(0.9984025559105432, 1e-8)
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(0.5) == scale + loc
    assert dist.invcdf(0.9984025559105432) == pytest.approx(1500000, 1e-8)

    sims = dist.generate(100000000)
    assert sims.mean() == pytest.approx(
        scipy.special.beta(1 - 1 / shape, 1 + 1 / shape) * scale + loc, 1e-3
    )
    assert sims.std() == pytest.approx(
        math.sqrt(
            scipy.special.beta(1 - 2 / shape, 1 + 2 / shape)
            - scipy.special.beta(1 - 1 / shape, 1 + 1 / shape) ** 2
        )
        * scale,
        1e-3,
    )


def test_ParaLogistic():
    set_random_seed(12345678910)
    shape = 2.5
    scale = 100000
    loc = 1000000
    dist = distributions.Paralogistic(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(
        dist.cdf(np.array([1234560.1, 2345670, 3456780]))
    ) == pytest.approx(np.array([1234560.1, 2345670, 3456780]), 1e-8)

    sims = dist.generate(100000000)

    assert sims.mean() == pytest.approx(
        scale * gamma(1 + 1 / shape) * gamma(shape - 1 / shape) / gamma(shape) + loc,
        1e-5,
    )


def test_InverseParaLogistic():
    set_random_seed(12345678910)
    shape = 5
    scale = 100000
    loc = 1000000
    dist = distributions.InverseParalogistic(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(
        dist.cdf(np.array([1234560.1, 2345670, 3456780]))
    ) == pytest.approx(np.array([1234560.1, 2345670, 3456780]), 1e-8)

    sims = dist.generate(100000000)

    assert sims.mean() == pytest.approx(
        scale * gamma(shape + 1 / shape) * gamma(1 - 1 / shape) / gamma(shape) + loc,
        1e-3,
    )
    assert sims.std() == pytest.approx(
        scale
        * np.sqrt(
            (gamma(shape + 2 / shape) * gamma(1 - 2 / shape) / gamma(shape))
            - (gamma(shape + 1 / shape) * gamma(1 - 1 / shape) / gamma(shape)) ** 2
        ),
        1e-3,
    )


def test_Weibull():
    set_random_seed(12345678910)
    shape = 2
    scale = 1000000
    loc = 1000000
    dist = distributions.Weibull(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(
        dist.cdf(np.array([1234560.1, 2345670, 3456780]))
    ) == pytest.approx(np.array([1234560.1, 2345670, 3456780]), 1e-8)

    sims = dist.generate(100000000)

    assert sims.mean() == pytest.approx(scale * gamma(1 + 1 / shape) + loc, 1e-3)
    assert sims.std() == pytest.approx(
        scale * np.sqrt(gamma(1 + 2 / shape) - (gamma(1 + 1 / shape)) ** 2), 1e-3
    )


def test_InverseWeibull():
    set_random_seed(12345678910)
    shape = 4
    scale = 1000000
    loc = 1000000
    dist = distributions.InverseWeibull(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(
        dist.cdf(np.array([1234560.1, 2345670, 3456780]))
    ) == pytest.approx(np.array([1234560.1, 2345670, 3456780]), 1e-8)

    sims = dist.generate(100000000)

    assert sims.mean() == pytest.approx(scale * gamma(1 - 1 / shape) + loc, 1e-3)
    assert sims.std() == pytest.approx(
        scale * np.sqrt(gamma(1 - 2 / shape) - (gamma(1 - 1 / shape)) ** 2), 1e-3
    )


def test_Exponential():
    set_random_seed(12345678910)
    scale = 1000000
    loc = 1000000
    dist = distributions.Exponential(scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(
        dist.cdf(np.array([1234560.1, 2345670, 3456780]))
    ) == pytest.approx(np.array([1234560.1, 2345670, 3456780]), 1e-8)

    sims = dist.generate(100000000)

    assert sims.mean() == pytest.approx(scale + loc, 1e-3)
    assert sims.std() == pytest.approx(scale, 1e-3)


def test_InverseExponential():
    set_random_seed(12345678910)
    scale = 1000000
    loc = 1000000
    dist = distributions.InverseExponential(scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(
        dist.cdf(np.array([1234560.1, 2345670, 3456780]))
    ) == pytest.approx(np.array([1234560.1, 2345670, 3456780]), 1e-8)


def test_Gamma():
    set_random_seed(12345678910)
    scale = 1000000
    shape = 4.5
    loc = 1000000
    dist = distributions.Gamma(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert np.allclose(
        dist.invcdf(dist.cdf(np.array([1234560.1, 2345670, 3456780]))),
        np.array([1234560.1, 2345670, 3456780]),
        1e-8,
    )

    sims = dist.generate(10000000)

    assert np.allclose(sims.mean(), scale * shape + loc, 1e-3)
    assert np.allclose(sims.std(), scale * np.sqrt(shape), 1e-3)


def test_LogNormal():
    set_random_seed(12345678910)
    mu = 8
    sigma = 1.25
    dist = distributions.LogNormal(mu, sigma)

    assert dist.cdf(0) == 0.0
    assert dist.invcdf(0) == 0
    assert np.allclose(
        dist.invcdf(dist.cdf(np.array([1234560.1, 2345670, 3456780]))),
        np.array([1234560.1, 2345670, 3456780]),
        1e-8,
    )

    sims = dist.generate(100000000)

    mean = np.exp(mu + 0.5 * sigma**2)
    sd = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))

    assert np.allclose(sims.mean(), mean, 1e-3)
    assert np.allclose(sims.std(), sd, 1e-3)


def test_InverseGamma():
    set_random_seed(12345678910)
    scale = 1000000
    shape = 3.5
    loc = 1000000
    dist = distributions.InverseGamma(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert np.allclose(
        dist.invcdf(dist.cdf(np.array([1234560.1, 2345670, 3456780]))),
        np.array([1234560.1, 2345670, 3456780]),
        1e-8,
    )

    sims = dist.generate(10000000)

    assert np.allclose(sims.mean(), scale * gamma(shape - 1) / gamma(shape) + loc, 1e-3)
    assert np.allclose(
        sims.std(),
        scale
        * np.sqrt(
            gamma(shape - 2) / gamma(shape) - (gamma(shape - 1) / gamma(shape)) ** 2
        ),
        1e-3,
    )
