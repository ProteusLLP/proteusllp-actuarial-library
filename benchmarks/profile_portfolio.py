"""End-to-end portfolio workload for scaling and peak-memory investigation.

Generate several independent frequency-severity lines, aggregate their event
streams, and run a non-overlapping excess-of-loss tower on the combined losses.
This is a synthetic performance workload, not a calibrated insurance model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from pal import set_random_seed
from pal._maths import xp
from pal.contracts import XoL
from pal.distributions import LogNormal, Poisson
from pal.frequency_severity import FreqSevSims, FrequencySeverityModel

from benchmarks.benchmark_core import environment, synchronise


def run_portfolio(n_sims: int, n_lobs: int = 10, n_layers: int = 10) -> Any:
    """Simulate multiple lines and apply non-overlapping layers to all losses."""
    if min(n_sims, n_lobs, n_layers) < 1:
        raise ValueError("Simulation count, lines of business and layers must be positive")

    # Each line produces a separate event stream; concatenate once to avoid
    # repeated O(number_of_lines x total_events) copying.
    indices = []
    values = []
    for line in range(n_lobs):
        frequency = Poisson(mean=1.5 + (line % 3) * 0.5)
        severity = LogNormal(mu=8 + (line % 4) * 0.05, sigma=0.8)
        losses = FrequencySeverityModel(frequency, severity).generate(n_sims)
        indices.append(losses.sim_index)
        values.append(losses.values)
    portfolio = FreqSevSims(xp.concatenate(indices), xp.concatenate(values), n_sims)
    # Concatenation owns new arrays. Release per-line event arrays before the
    # reinsurance loop, otherwise the benchmark exaggerates peak RAM/VRAM.
    del indices, values, losses
    net = portfolio.aggregate().values.copy()

    # These layers are adjacent and disjoint, so their total recoveries cannot
    # exceed the portfolio's gross event losses.
    for i in range(n_layers):
        layer = XoL(name=f"Layer {i}", excess=1000.0 + 2500.0 * i, limit=2500.0, premium=1000.0)
        net -= layer.apply(portfolio).recoveries.aggregate().values
    return xp.percentile(net, 99.5)


def main(argv: list[str] | None = None) -> int:
    """Run a configurable portfolio workload for performance investigations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--lobs", type=int, default=10)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("pal-portfolio.json"))
    args = parser.parse_args(argv)
    set_random_seed(20261108)
    synchronise()
    start = time.perf_counter()
    value = run_portfolio(args.simulations, args.lobs, args.layers)
    synchronise()
    elapsed = time.perf_counter() - start
    result = {
        "schema_version": 1,
        "environment": environment(),
        "simulations": args.simulations,
        "lobs": args.lobs,
        "layers": args.layers,
        "elapsed_seconds": elapsed,
        "net_percentile_99_5": float(value),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Portfolio: {args.lobs} lines, {args.layers} layers, {args.simulations:,} simulations")
    print(f"Wall time: {elapsed:.3f} seconds; net 99.5th percentile: {float(value):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
