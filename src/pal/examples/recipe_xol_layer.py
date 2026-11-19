"""Estimate recoveries from a per-occurrence excess-of-loss layer.

The attachment (excess) and limit apply to each claim, while the
resulting recoveries can be aggregated by simulation.
"""

from pal import FrequencySeverityModel, set_random_seed
from pal.contracts import XoL
from pal.distributions import LogNormal, Poisson


def main() -> None:
    """Price the expected recovery of a 1m excess of 1m layer."""
    set_random_seed(42)
    gross_events = FrequencySeverityModel(
        freq_dist=Poisson(mean=2),
        sev_dist=LogNormal(mu=13.5, sigma=1.0),
    ).generate(n_sims=512)

    layer = XoL(name="1m xs 1m", limit=1_000_000, excess=1_000_000, premium=80_000)
    recoveries = layer.apply(gross_events).recoveries.aggregate()
    gross = gross_events.aggregate()

    assert recoveries.n_sims == gross.n_sims
    assert recoveries.coupled_variable_group is gross_events.coupled_variable_group
    assert 0 < recoveries.mean() <= gross.mean()

    print("Expected annual gross loss:", gross.mean())
    print("Expected annual layer recovery:", recoveries.mean())
    print("99.5th percentile layer recovery:", recoveries.percentile(99.5))


if __name__ == "__main__":
    main()
