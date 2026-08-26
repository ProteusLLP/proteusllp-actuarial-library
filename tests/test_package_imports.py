"""Import smoke tests for PAL's public package namespace."""

import pal


def test_pal_top_level_namespace_is_module_oriented() -> None:
    from pal import config, distributions, set_default_n_sims, set_random_seed
    from pal import contracts, copulas, couplings, frequency_severity
    from pal import maths, multivariate_distributions, risk_measures, stats
    from pal import stochastic_scalar, variables

    assert config is not None
    assert distributions is not None
    assert contracts is not None
    assert copulas is not None
    assert couplings is not None
    assert frequency_severity is not None
    assert maths is not None
    assert multivariate_distributions is not None
    assert risk_measures is not None
    assert stats is not None
    assert stochastic_scalar is not None
    assert variables is not None
    assert set_default_n_sims is not None
    assert set_random_seed is not None


def test_domain_objects_are_not_reexported_at_top_level() -> None:
    for name in (
        "Gamma",
        "GaussianCopula",
        "FreqSevSims",
        "ProteusVariable",
        "StochasticScalar",
        "XoL",
        "XoLTower",
    ):
        assert not hasattr(pal, name), f"pal.{name} should be accessed through its module namespace"
