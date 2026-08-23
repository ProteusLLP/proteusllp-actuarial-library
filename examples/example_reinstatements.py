"""Reproducing Mata (2000) — XoL reinstatement pricing.

Compares pure premiums and PH transform premiums for excess-of-loss
layers with free reinstatements against the analytical results in:

  Mata, A.J. (2000). "Pricing Excess of Loss Reinsurance with
  Reinstatements." ASTIN Bulletin, 30(2), 349-368.

Model:
  - Frequency: Poisson(lambda=10)
  - Severity: Pareto Type II (alpha=3, beta=10)
  - Layers: 10 xs 10, 10 xs 20, combined 20 xs 10
  - One free reinstatement (K=1, c=0)
"""

import pandas as pd

from pal.config import set_default_n_sims, set_random_seed
from pal.contracts import XoLTower
from pal.distributions import Pareto, Poisson
from pal.frequency_severity import FrequencySeverityModel
from pal.risk_measures import proportional_hazards_transform

# ── Setup ───────────────────────────────────────────────────────────

set_random_seed(42)
set_default_n_sims(100_000)

model = FrequencySeverityModel(Poisson(10.0), Pareto(shape=3, scale=10))
claims = model.generate() - 10.0

# Two separate layers + the combined layer, all in one tower
tower_one_reinst = XoLTower(
    name=["10 xs 10", "10 xs 20", "20 xs 10"],
    limit=[10, 10, 20],
    excess=[10, 20, 10],
    premium=[1, 1, 1],
    reinstatement_cost=[[0.0], [0.0], [0.0]],
    aggregate_limit=[20, 20, 40],
)

# ── Table 2: Pure premiums (lambda=10, free reinstatements) ─────────

print("Table 2 — Pure premiums (lambda=10)")
print("=" * 55)

tower_unlimited_reinst = XoLTower(
    name=["10 xs 10", "10 xs 20", "20 xs 10"],
    limit=[10, 10, 20],
    excess=[10, 20, 10],
    premium=[1, 1, 1],
)

for tower in [tower_one_reinst, tower_unlimited_reinst]:
    tower.apply(claims)

# Create a DataFrame to compare the pure premiums for each layer and the sum of the two separate layers against the
# #combined layer.
df2 = pd.DataFrame(
    [
        {
            "Layer": "10 xs 10",
            "K=1 (c=0)": tower_one_reinst.layers[0].summary["mean"],
            "K=inf": tower_unlimited_reinst.layers[0].summary["mean"],
        },
        {
            "Layer": "10 xs 20",
            "K=1 (c=0)": tower_one_reinst.layers[1].summary["mean"],
            "K=inf": tower_unlimited_reinst.layers[1].summary["mean"],
        },
        {
            "Layer": "SUM",
            "K=1 (c=0)": tower_one_reinst.layers[0].summary["mean"] + tower_one_reinst.layers[1].summary["mean"],
            "K=inf": tower_unlimited_reinst.layers[0].summary["mean"]
            + tower_unlimited_reinst.layers[1].summary["mean"],
        },
        {
            "Layer": "20 xs 10",
            "K=1 (c=0)": tower_one_reinst.layers[2].summary["mean"],
            "K=inf": tower_unlimited_reinst.layers[2].summary["mean"],
        },
    ]
).set_index("Layer")
print(df2)
print()

# ── Table 5: PH transform premiums (One free reinstatement) ────────

print("Table 5 — PH transform premiums (lambda=10, K=1 free)")
print("=" * 70)
agg1 = tower_one_reinst.layers[0].apply(claims).recoveries.aggregate()
agg2 = tower_one_reinst.layers[1].apply(claims).recoveries.aggregate()
aggc = tower_one_reinst.layers[2].apply(claims).recoveries.aggregate()

rows = []
for p in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
    alpha = 1.0 / p
    ph1 = proportional_hazards_transform(agg1, alpha=alpha)
    ph2 = proportional_hazards_transform(agg2, alpha=alpha)
    ph_sum = proportional_hazards_transform(agg1 + agg2, alpha=alpha)
    phc = proportional_hazards_transform(aggc, alpha=alpha)
    rows.append(
        {
            "p": p,
            "pi_p(S1*+S2*)": ph_sum.value,
            "pi_p(S1*)": ph1.value,
            "pi_p(S2*)": ph2.value,
            "pi_p(S1*)+pi_p(S2*)": ph1.value + ph2.value,
            "pi_p(Sc*)": phc.value,
        }
    )

df5 = pd.DataFrame(rows).set_index("p")
print(df5)
