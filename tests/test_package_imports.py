"""Import smoke tests.

These tests are intentionally simple: they ensure that common public imports work
at runtime. This catches circular-import regressions that static type checkers
won't reliably flag.
"""


def test_pal_top_level_imports() -> None:
    from pal import (
        FreqSevSims,
        ProteusVariable,
        StochasticScalar,
        XoL,
        XoLTower,
        config,
        copulas,
        distributions,
        maths,
        risk_measures,
        set_default_n_sims,
        set_random_seed,
        stats,
    )

    assert FreqSevSims is not None
    assert ProteusVariable is not None
    assert StochasticScalar is not None
    assert XoL is not None
    assert XoLTower is not None
    assert config is not None
    assert copulas is not None
    assert distributions is not None
    assert maths is not None
    assert risk_measures is not None
    assert set_default_n_sims is not None
    assert set_random_seed is not None
    assert stats is not None
