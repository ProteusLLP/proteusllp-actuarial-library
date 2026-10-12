"""Import smoke tests for PAL's public package namespace."""

import pal


def test_pal_top_level_namespace_exposes_core_types_and_modules() -> None:
    from pal import (
        FreqSevSims,
        FrequencySeverityModel,
        ProteusVariable,
        StochasticScalar,
        config,
        contracts,
        copulas,
        couplings,
        distributions,
        frequency_severity,
        maths,
        multivariate_distributions,
        risk_measures,
        set_default_n_sims,
        set_random_seed,
        stats,
        variables,
    )

    assert ProteusVariable is pal.variables.ProteusVariable
    assert StochasticScalar is pal.variables.StochasticScalar
    assert FreqSevSims is pal.frequency_severity.FreqSevSims
    assert FrequencySeverityModel is pal.frequency_severity.FrequencySeverityModel
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
    assert variables is not None
    assert set_default_n_sims is not None
    assert set_random_seed is not None
    assert "stochastic_scalar" not in pal.__all__


def test_core_types_remain_available_from_domain_modules() -> None:
    from pal.frequency_severity import FreqSevSims, FrequencySeverityModel
    from pal.variables import ProteusVariable, StochasticScalar

    assert ProteusVariable is pal.ProteusVariable
    assert StochasticScalar is pal.StochasticScalar
    assert FreqSevSims is pal.FreqSevSims
    assert FrequencySeverityModel is pal.FrequencySeverityModel


def test_other_domain_objects_are_not_reexported_at_top_level() -> None:
    for name in (
        "Gamma",
        "GaussianCopula",
        "XoL",
        "XoLTower",
    ):
        assert not hasattr(pal, name), f"pal.{name} should be accessed through its documented namespace"
