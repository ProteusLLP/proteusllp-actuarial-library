"""Coupling Groups, Copulas and Variable Reordering - Worked Examples.

This script demonstrates how PAL manages dependencies between stochastic
variables using coupling groups, copulas, and automatic reordering.

These concepts are central to building correct multi-variable simulation
models where correlations between risks need to be preserved.
"""

import os

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from pal import (  # isort: skip
    config,
    copulas,
    distributions,
    set_random_seed,
    StochasticScalar,
)

SUPPRESS_PLOTS = os.getenv("PAL_SUPPRESS_PLOTS", "").lower() == "true"

# ============================================================================
# Setup
# ============================================================================

config.n_sims = 10_000
set_random_seed(42)

# ============================================================================
# Part 1: Coupling Groups - Automatic Dependency Tracking
# ============================================================================

print("=" * 70)
print("PART 1: COUPLING GROUPS")
print("=" * 70)

# When you create a stochastic variable, it gets its own coupling group.
motor_losses = distributions.LogNormal(mu=14, sigma=0.5).generate()
property_losses = distributions.LogNormal(mu=15, sigma=0.8).generate()

# Each variable starts in its own independent coupling group.
print("\n--- Independent variables ---")
motor_group = id(motor_losses.coupled_variable_group)
prop_group = id(property_losses.coupled_variable_group)
print(f"Motor    coupling group id: {motor_group}")
print(f"Property coupling group id: {prop_group}")
print(
    "Same group?",
    motor_losses.coupled_variable_group is property_losses.coupled_variable_group,
)

# When you combine variables in a formula, they become COUPLED.
# PAL tracks that they share a dependency relationship.
total_losses = motor_losses + property_losses

print("\n--- After computing total = motor + property ---")
motor_group = id(motor_losses.coupled_variable_group)
prop_group = id(property_losses.coupled_variable_group)
total_group = id(total_losses.coupled_variable_group)
print(f"Motor    coupling group id: {motor_group}")
print(f"Property coupling group id: {prop_group}")
print(f"Total    coupling group id: {total_group}")
print(
    "All in same group?",
    (
        motor_losses.coupled_variable_group
        is property_losses.coupled_variable_group
        is total_losses.coupled_variable_group
    ),
)

# Coupling groups track how many variables are linked together.
group_size = len(motor_losses.coupled_variable_group)
print(f"\nVariables in coupling group: {group_size}")

# Derived variables join the same group automatically.
motor_with_expenses = motor_losses * 1.1
print(
    "\nMotor with expenses in same group?",
    motor_with_expenses.coupled_variable_group is motor_losses.coupled_variable_group,
)
group_size = len(motor_losses.coupled_variable_group)
print(f"Variables in coupling group: {group_size}")

# ============================================================================
# Part 2: Why Coupling Groups Matter - The Reordering Problem
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: WHY COUPLING GROUPS MATTER")
print("=" * 70)

# Create two INDEPENDENT loss variables.
set_random_seed(42)
loss_a = distributions.LogNormal(mu=14, sigma=0.5).generate()
loss_b = distributions.LogNormal(mu=15, sigma=0.8).generate()

# Compute a derived variable from loss_a.
loss_a_expenses = loss_a * 1.15  # 15% expense loading

# Before any copula, simulation indices are in their original order.
# loss_a[i] and loss_a_expenses[i] correspond to the same scenario.
print("\n--- Before copula ---")
print(f"loss_a first 5:          {loss_a.values[:5]}")
print(f"loss_a_expenses first 5: {loss_a_expenses.values[:5]}")
ratio = loss_a_expenses.values[:5] / loss_a.values[:5]
print(f"Ratio (should be 1.15):  {ratio}")

# Now apply a Gaussian copula to correlate loss_a and loss_b.
# Because loss_a and loss_a_expenses are COUPLED, when loss_a
# is reordered, loss_a_expenses is automatically reordered too.
copulas.GaussianCopula([[1.0, 0.7], [0.7, 1.0]]).apply([loss_a, loss_b])

print("\n--- After copula reordering ---")
print(f"loss_a first 5:          {loss_a.values[:5]}")
print(f"loss_a_expenses first 5: {loss_a_expenses.values[:5]}")
ratio = loss_a_expenses.values[:5] / loss_a.values[:5]
print(f"Ratio (still 1.15!):     {ratio}")

print(
    "\nThe 1.15x relationship is preserved because PAL"
    " automatically\nreordered loss_a_expenses when it"
    " reordered loss_a."
)

# ============================================================================
# Part 3: Copulas - Creating Dependency Structures
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: COPULAS - CREATING DEPENDENCY STRUCTURES")
print("=" * 70)


def rank_corr(a: StochasticScalar, b: StochasticScalar) -> float:  # noqa: ANN001, ANN201
    """Compute rank correlation between two variables."""
    return np.corrcoef(a.ranks.values, b.ranks.values)[0, 1]


# --- 3a: Independent variables (no copula) ---
set_random_seed(42)
x_indep = distributions.LogNormal(mu=10, sigma=1.0).generate()
y_indep = distributions.LogNormal(mu=10, sigma=1.0).generate()

print("\n--- Independent variables (no copula) ---")
print(f"Rank correlation: {rank_corr(x_indep, y_indep):.4f}")

# --- 3b: Gaussian copula with positive correlation ---
set_random_seed(42)
x_gauss = distributions.LogNormal(mu=10, sigma=1.0).generate()
y_gauss = distributions.LogNormal(mu=10, sigma=1.0).generate()
copulas.GaussianCopula([[1.0, 0.8], [0.8, 1.0]]).apply([x_gauss, y_gauss])

print("\n--- Gaussian copula (rho=0.8) ---")
print(f"Rank correlation: {rank_corr(x_gauss, y_gauss):.4f}")

# --- 3c: Gumbel copula (upper tail dependence) ---
set_random_seed(42)
x_gumbel = distributions.LogNormal(mu=10, sigma=1.0).generate()
y_gumbel = distributions.LogNormal(mu=10, sigma=1.0).generate()
copulas.GumbelCopula(theta=3.0, n=2).apply([x_gumbel, y_gumbel])

print("\n--- Gumbel copula (theta=3.0) ---")
print(f"Rank correlation: {rank_corr(x_gumbel, y_gumbel):.4f}")
print("Gumbel copulas have stronger UPPER tail dependence.")

# --- 3d: Clayton copula (lower tail dependence) ---
set_random_seed(42)
x_clay = distributions.LogNormal(mu=10, sigma=1.0).generate()
y_clay = distributions.LogNormal(mu=10, sigma=1.0).generate()
copulas.ClaytonCopula(theta=4.0, n=2).apply([x_clay, y_clay])

print("\n--- Clayton copula (theta=4.0) ---")
print(f"Rank correlation: {rank_corr(x_clay, y_clay):.4f}")
print("Clayton copulas have stronger LOWER tail dependence.")

# --- 3e: Student's T copula (symmetric tail dependence) ---
set_random_seed(42)
x_t = distributions.LogNormal(mu=10, sigma=1.0).generate()
y_t = distributions.LogNormal(mu=10, sigma=1.0).generate()
copulas.StudentsTCopula([[1.0, 0.7], [0.7, 1.0]], dof=3).apply([x_t, y_t])

print("\n--- Student's T copula (rho=0.7, dof=3) ---")
print(f"Rank correlation: {rank_corr(x_t, y_t):.4f}")
print("Student's T copula has symmetric tail dependence, stronger with fewer dof.")

# ============================================================================
# Part 4: Scatter Plots - Visualising Dependency Structures
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: SCATTER PLOTS")
print("=" * 70)

fig = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=[
        "Independent (no copula)",
        "Gaussian (\u03c1=0.8)",
        "Gumbel (\u03b8=3.0)",
        "Clayton (\u03b8=4.0)",
        "Student's T (\u03c1=0.7, \u03bd=3)",
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.12,
)

scatter_kwargs = {
    "mode": "markers",
    "marker": {"size": 2, "opacity": 0.3},
}

pairs = [
    (x_indep, y_indep, "Independent", 1, 1),
    (x_gauss, y_gauss, "Gaussian", 1, 2),
    (x_gumbel, y_gumbel, "Gumbel", 1, 3),
    (x_clay, y_clay, "Clayton", 2, 1),
    (x_t, y_t, "Student's T", 2, 2),
]

for x_var, y_var, name, row, col in pairs:
    fig.add_trace(  # type: ignore[misc]
        go.Scattergl(
            x=x_var.ranks.values.tolist(),
            y=y_var.ranks.values.tolist(),
            name=name,
            **scatter_kwargs,
        ),
        row=row,
        col=col,
    )

fig.update_layout(  # type: ignore[misc]
    title_text="Copula Dependency Structures (Rank Space)",
    height=700,
    width=1000,
    showlegend=False,
)

for _, (_, _, _, row, col) in enumerate(pairs):
    fig.update_xaxes(title_text="X Rank", row=row, col=col)  # type: ignore[misc]
    fig.update_yaxes(title_text="Y Rank", row=row, col=col)  # type: ignore[misc]

if not SUPPRESS_PLOTS:
    fig.show()  # type: ignore[misc]
print("Scatter plots generated showing rank-space dependencies.")

# ============================================================================
# Part 5: Variable Reordering in Detail
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: VARIABLE REORDERING IN DETAIL")
print("=" * 70)

# The copula.apply() method works by REORDERING the simulations of one
# variable to match the rank structure defined by the copula.
#
# Step by step:
# 1. Two variables start with independent random orderings.
# 2. The copula generates correlated UNIFORM samples.
# 3. The copula's rank structure reorders one variable's sims
#    so that the ranks match the copula's dependency.
# 4. Marginal distributions are PRESERVED (same values, new order).

set_random_seed(42)
var_x = distributions.Normal(0, 1).generate()
var_y = distributions.Normal(0, 1).generate()

print("\n--- Before copula ---")
sorted_x = np.sort(var_x.values)
sorted_y = np.sort(var_y.values)
print(f"X mean: {var_x.mean():.4f}, std: {var_x.std():.4f}")
print(f"Y mean: {var_y.mean():.4f}, std: {var_y.std():.4f}")

copulas.GaussianCopula([[1.0, 0.9], [0.9, 1.0]]).apply([var_x, var_y])

print("\n--- After copula ---")
print(f"X mean: {var_x.mean():.4f}, std: {var_x.std():.4f}")
print(f"Y mean: {var_y.mean():.4f}, std: {var_y.std():.4f}")

# The marginal values are EXACTLY the same, just in different order.
x_unchanged = np.allclose(np.sort(var_x.values), sorted_x)
y_unchanged = np.allclose(np.sort(var_y.values), sorted_y)
rc = rank_corr(var_x, var_y)
print(f"\nX values unchanged? {x_unchanged}")
print(f"Y values unchanged? {y_unchanged}")
print(f"Rank correlation:   {rc:.4f}")

# ============================================================================
# Part 6: Coupled Variable Reordering Across a Chain
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: COUPLED VARIABLE REORDERING ACROSS A CHAIN")
print("=" * 70)

set_random_seed(42)

# Create a chain of derived variables.
base_loss = distributions.LogNormal(mu=14, sigma=0.5).generate()
gross_loss = base_loss * 1.0  # Copy
expense_loaded = gross_loss * 1.10  # 10% expenses
tax = expense_loaded * 0.21  # 21% tax
net_loss = expense_loaded - tax

# All of these are in the same coupling group.
chain = [gross_loss, expense_loaded, tax, net_loss]
all_coupled = all(v.coupled_variable_group is base_loss.coupled_variable_group for v in chain)
group_size = len(base_loss.coupled_variable_group)
print(f"\nAll coupled? {all_coupled}")
print(f"Coupling group size: {group_size}")

# Create an independent variable and correlate it with base_loss.
cat_loss = distributions.LogNormal(mu=16, sigma=1.2).generate()
cat_independent = cat_loss.coupled_variable_group is not base_loss.coupled_variable_group
print(f"\nCat loss independent? {cat_independent}")

# Apply copula between base_loss and cat_loss.
copulas.GumbelCopula(theta=1.5, n=2).apply([base_loss, cat_loss])

# After the copula, ALL variables derived from base_loss were
# reordered together, preserving their relationships.
print("\n--- After copula between base_loss and cat_loss ---")
r1 = gross_loss.values[:3] / base_loss.values[:3]
r2 = expense_loaded.values[:3] / gross_loss.values[:3]
r3 = tax.values[:3] / expense_loaded.values[:3]
print(f"gross/base ratio (should be 1.0):    {r1}")
print(f"expense/gross ratio (should be 1.1): {r2}")
print(f"tax/expense ratio (should be 0.21):  {r3}")

# Everyone is now in the same coupling group.
full_chain = [cat_loss, *chain]
all_now_coupled = all(v.coupled_variable_group is base_loss.coupled_variable_group for v in full_chain)
print(f"\nAll now coupled? {all_now_coupled}")

# ============================================================================
# Part 7: Multivariate Copulas
# ============================================================================

print("\n" + "=" * 70)
print("PART 7: MULTIVARIATE COPULAS")
print("=" * 70)

set_random_seed(42)

# Create variables for 4 lines of business.
lobs = {
    "Motor": distributions.LogNormal(mu=14, sigma=0.4).generate(),
    "Property": distributions.LogNormal(mu=15, sigma=0.6).generate(),
    "Liability": distributions.LogNormal(mu=13, sigma=0.5).generate(),
    "Marine": distributions.LogNormal(mu=12, sigma=0.7).generate(),
}

# Define a correlation structure.
corr_matrix = [
    [1.0, 0.6, 0.3, 0.2],
    [0.6, 1.0, 0.4, 0.3],
    [0.3, 0.4, 1.0, 0.5],
    [0.2, 0.3, 0.5, 1.0],
]

# Apply Gaussian copula to all 4 variables at once.
lob_list = list(lobs.values())
copulas.GaussianCopula(corr_matrix).apply(lob_list)

# Check the resulting rank correlations.
print("\n--- Rank correlation matrix after Gaussian copula ---")
names = list(lobs.keys())
ranks = np.array([lobs[name].ranks.values for name in names])
rank_corr_matrix = np.corrcoef(ranks)
print(f"{'':>12}", end="")
for name in names:
    print(f"{name:>12}", end="")
print()
for i, name in enumerate(names):
    print(f"{name:>12}", end="")
    for j in range(len(names)):
        print(f"{rank_corr_matrix[i, j]:>12.3f}", end="")
    print()

print("\nInput correlation matrix for comparison:")
for row in corr_matrix:
    print("  ", [f"{x:.1f}" for x in row])

# ============================================================================
# Part 8: Generate vs Apply
# ============================================================================

print("\n" + "=" * 70)
print("PART 8: GENERATE vs APPLY")
print("=" * 70)

set_random_seed(42)

# generate() creates NEW correlated uniform samples.
gauss_cop = copulas.GaussianCopula([[1.0, 0.8], [0.8, 1.0]])
gauss_samples = gauss_cop.generate()
print("\n--- generate() returns correlated uniform samples ---")
print(f"Type: {type(gauss_samples)}")
print(f"Number of variables: {len(gauss_samples)}")
m0 = gauss_samples[0].mean()
m1 = gauss_samples[1].mean()
print(f"Variable 0 mean (should be ~0.5): {m0:.4f}")
print(f"Variable 1 mean (should be ~0.5): {m1:.4f}")
rc = rank_corr(gauss_samples[0], gauss_samples[1])
print(f"Rank correlation: {rc:.4f}")

# apply() reorders EXISTING variables to match copula structure.
set_random_seed(42)
v1 = distributions.Gamma(alpha=5, theta=1000).generate()
v2 = distributions.Pareto(shape=2, scale=10000).generate()

print("\n--- apply() reorders existing variables ---")
print(f"Before copula - V1 mean: {v1.mean():.0f}, V2 mean: {v2.mean():.0f}")
copulas.GaussianCopula([[1.0, 0.8], [0.8, 1.0]]).apply([v1, v2])
print(f"After copula  - V1 mean: {v1.mean():.0f}, V2 mean: {v2.mean():.0f}")
print("Means unchanged - only the ordering changed!")
print(f"Rank correlation: {rank_corr(v1, v2):.4f}")

print("\n" + "=" * 70)
print("EXAMPLES COMPLETE")
print("=" * 70)
