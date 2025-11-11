# type: ignore
"""
Over-Dispersed Poisson (ODP) Bayesian Posterior Predictive Model
================================================================
Implements the model from:
   Norman, J.P. (2025). "The Predictive Distribution of the Over-dispersed Poisson
   Claims Reserving Model"

This uses an exact analytical solution of the Bayesian predictive distribution
for claims reserves under the ODP model, implemented using the
Proteus Actuarial Library.

"""

import numpy as np
from pal import ProteusVariable, StochasticScalar, config, distributions
from pal import maths as pnp

config.n_sims = 100_000


class ODPModel:
    """Bayesian ODP claims reserving model."""

    phi: float
    """Dispersion or scale parameter."""
    n: int
    """Number of development periods."""
    triangle: np.ndarray
    """Incremental claims triangle."""
    mu: ProteusVariable[StochasticScalar]
    """Stochastic variables for origin period means."""
    betas: ProteusVariable[StochasticScalar]
    """Stochastic variables for development period payment pattern."""

    def __init__(self, incremental_triangle: np.ndarray):
        self.triangle = np.array(incremental_triangle, dtype=float)
        self.n = self.triangle.shape[0]
        self.obs_mask: np.ndarray = observed_mask(self.triangle)
        self.triangle[~self.obs_mask] = np.nan
        self.cumtri = cumulative_triangle(self.triangle)
        self.origin_periods = [str(i) for i in range(1, self.n + 1)]
        self.dev_periods = [str(j) for j in range(1, self.n + 1)]
        self.future_dev_periods = {
            str(op): [str(j) for j in range(self.n - op + 2, self.n + 1)]
            for op in range(1, self.n + 1)
        }

    # ---------------------------------------------------------
    # Step 1: estimate dispersion φ̂
    # ---------------------------------------------------------

    def estimate_phi(self):
        """Estimate the dispersion parameter φ from the observed triangle.

        Returns:
            float: Estimated dispersion φ̂
        """
        n = self.n
        cumtri = self.cumtri

        # Link ratios
        link = np.zeros(n)
        for j in range(1, n):
            num = np.sum(cumtri[: n - j, j])
            den = np.sum(cumtri[: n - j, j - 1])
            link[j] = num / den

        # Tail factors and beta pattern
        tail = np.ones(n)
        for j in range(n - 2, -1, -1):
            tail[j] = tail[j + 1] * link[j + 1]
        cumulative_beta_hat = 1 / tail

        beta_hat = np.diff(cumulative_beta_hat, prepend=0)
        mu_hat = np.array(
            [cumtri[i, n - i - 1] / (cumulative_beta_hat[n - i - 1]) for i in range(n)]
        )
        m_hat = np.outer(mu_hat, beta_hat)

        num = np.sum(((m_hat - self.triangle) ** 2 / m_hat)[self.obs_mask])
        denom = n * (n + 1) / 2 - 2 * n + 1

        self.phi = num / denom
        return self.phi

    # ---------------------------------------------------------
    # Step 2: build posteriors ψ_j, β_j, μ_i
    # ---------------------------------------------------------
    def build_posterior(self):
        n, phi = self.n, self.phi
        cumtri = self.cumtri
        d_i = np.nansum(self.triangle, axis=1) / phi  # the scaled origin period totals
        c_j = (
            np.nansum(self.triangle, axis=0) / phi
        )  # the scaled development period totals
        d_ij = cumtri / phi
        # column sums of d_ij not including the diagonal
        sum_dij = [np.sum(d_ij[: n - j, j - 1]) for j in range(1, n)]

        # ψ_j ~ Beta(a_j + C_j, b_j + Σ_i D_{i,j-1})
        # psi is the incremental amount of the remaining development to be paid
        #
        psi_vars = [StochasticScalar([1])]  # ψ₁ = 1
        for j in range(1, n):
            a_j, b_j = 0.0, 1.0
            psi_vars.append(
                distributions.Beta(
                    a_j + float(c_j[j]), b_j + float(sum_dij[j - 1])
                ).generate()
            )
        psi = ProteusVariable("dp", {str(dp + 1): psi_vars[dp] for dp in range(n)})

        # β_j recursively from ψ
        betas = [StochasticScalar([])] * n
        betas[-1] = psi[-1]
        future_sum_beta = psi[-1]
        for j in range(n - 2, -1, -1):
            betas[j] = psi[j] * (1 - future_sum_beta)
            future_sum_beta = future_sum_beta + betas[j]
        cumulative_payment_pattern = pnp.cumsum(betas)

        # μ_i ~ φGamma(D_i, 1/(Σ β_j))
        # These are the origin period means
        self.mu = ProteusVariable(
            dim_name="op",
            values={
                str(i + 1): phi
                * np.maximum(1 / (cumulative_payment_pattern[n - i - 1]), 0)
                * distributions.Gamma(
                    float(d_i[i]),
                    1,
                ).generate()
                for i in range(n)
            },
        )
        self.betas = ProteusVariable("dp", {str(dp + 1): betas[dp] for dp in range(n)})

    # ---------------------------------------------------------
    # Step 3: simulate predictive distribution
    # ---------------------------------------------------------
    def simulate_reserves(self) -> StochasticScalar:
        """Simulate the predictive distribution of future claims.

        Returns:
            StochasticScalar: Total future claims payments
        """
        self.estimate_phi()
        self.build_posterior()
        phi = self.phi
        total_by_origin: ProteusVariable[StochasticScalar] = ProteusVariable("op", {})

        for op in self.origin_periods:
            total_by_origin[op] = 0.0
            for dp in self.future_dev_periods[op]:
                x_ij = distributions.Poisson(
                    self.mu[op] * self.betas[dp] / phi
                ).generate()  # could also use a Gamma as the forecasting distribution
                total_by_origin[op] = total_by_origin[op] + phi * x_ij

        total = total_by_origin.sum()
        self.total_future_claims = total
        self.total_future_claims_by_origin = total_by_origin
        return total

    # ---------------------------------------------------------
    # Step 4: summary reporting
    # ---------------------------------------------------------

    def describe(
        self,
        percentiles: list[float] | None = None,
    ):
        if percentiles is None:
            percentiles = [0.5, 1, 2.5, 5, 10, 50, 90, 95, 99, 99.5]
        x = self.total_future_claims
        mean, sd = x.mean(), x.std()
        cv = sd / mean
        print("\nBayesian ODP predictive reserve distribution:")
        print(f"Dispersion φ̂ = {self.phi:,.2f}")
        print(f"Mean  = {mean:,.0f}")
        print(f"SD    = {sd:,.0f}")
        print(f"CV    = {100 * cv:.1f}%")
        claims_percentiles: list[float] = x.percentile(percentiles)  # type: ignore
        for p, v in zip(percentiles, claims_percentiles, strict=False):
            print(f"{p:>5.1f}th = {v:,.0f}")
        return {"mean": mean, "sd": sd, "cv": cv}


# ---------------------------------------------------------------------
# Helper: standard cumulative and mask logic
# ---------------------------------------------------------------------


def cumulative_triangle(tri: np.ndarray) -> np.ndarray:
    """Row-wise cumulative sums (NaNs treated as zero)."""
    return np.cumsum(np.nan_to_num(tri), axis=1)


def observed_mask(tri: np.ndarray) -> np.ndarray:
    """Upper-left triangle mask for observed cells."""
    n = tri.shape[0]
    mask = np.ones_like(tri, dtype=bool)
    mask[np.tril_indices(n, -1)] = False
    mask[:, -1::-1] = mask
    return mask


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    import plotly.graph_objects as go

    triangle = pd.read_csv(
        "data/reserve_risk/claims_triangle.csv", index_col=0
    ).to_numpy()

    model = ODPModel(triangle)
    model.simulate_reserves()
    model.describe()
    total_future_claims_by_origin = model.total_future_claims_by_origin

    fig = go.Figure()
    for i in range(2, model.n + 1):
        fig.add_trace(
            go.Scatter(
                x=np.sort(total_future_claims_by_origin[str(i)].tolist()),  # type: ignore
                y=np.linspace(0, 1, config.n_sims),
                name=f"Origin Period {i}",
            )
        )
        fig.update_layout(
            title="Predictive CDFs of Future Claims by Origin Period",
            xaxis_title="Future Claims Payments",
            yaxis_title="Cumulative Probability",
        )
    print("Displaying predictive CDF plot...")
    fig.show()
