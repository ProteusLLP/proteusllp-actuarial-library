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

import typing as t

import numpy as np
from pal import ProteusVariable, StochasticScalar, config, distributions
from pal import maths as pnp

config.n_sims = 100_000

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
# Core model class
# ---------------------------------------------------------------------


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
        self.obs_mask = observed_mask(self.triangle)
        self.triangle[~self.obs_mask] = np.nan
        self.cumtri = cumulative_triangle(self.triangle)

    # ---------------------------------------------------------
    # Step 1: estimate dispersion φ̂
    # ---------------------------------------------------------

    def estimate_phi(self):
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
        d_i = (
            np.nansum(self.triangle, axis=1) / phi
        ).tolist()  # the scaled origin period totals
        c_j = (
            np.nansum(self.triangle, axis=0) / phi
        ).tolist()  # the scaled development period totals
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
        psi = ProteusVariable("dp", {str(dp): psi_vars[dp] for dp in range(n)})

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
                str(i): phi
                * np.maximum(1 / (cumulative_payment_pattern[n - i - 1]), 0)
                * distributions.Gamma(
                    d_i[i],
                    1,
                ).generate()
                for i in range(n)
            },
        )
        self.betas = ProteusVariable("dp", {str(dp): betas[dp] for dp in range(n)})

    # ---------------------------------------------------------
    # Step 3: simulate predictive distribution
    # ---------------------------------------------------------
    def simulate_reserves(self):
        self.estimate_phi()
        self.build_posterior()
        n, phi = self.n, self.phi
        total = StochasticScalar(np.zeros_like(self.mu[0].values))
        total_by_origin: ProteusVariable[StochasticScalar] = ProteusVariable("op", {})

        for i in range(n):
            total_by_origin.values[str(i)] = 0.0  # type: ignore[assignment]
            for j in range(n):
                if self.obs_mask[i, j]:
                    continue
                mean_ij = self.mu[i] * self.betas[j]
                lam = mean_ij / phi
                x_ij = distributions.Gamma(
                    lam, 1
                ).generate()  # use a Gamma distribution as the forecasting distribution
                total_by_origin.values[str(i)] = total_by_origin[str(i)] + phi * x_ij  # type: ignore[assignment]

        total = total_by_origin.sum()
        self.total_future_claims = total
        self.total_future_claims_by_origin = total_by_origin
        return total

    # ---------------------------------------------------------
    # Step 4: summary reporting
    # ---------------------------------------------------------

    def describe(
        self,
        percentiles: t.Sequence[float] = (0.5, 1, 2.5, 5, 10, 50, 90, 95, 99, 99.5),
    ):
        x = self.total_future_claims
        mean, sd = pnp.mean(x), np.sqrt(pnp.var(x))
        cv = sd / mean
        print("\nBayesian ODP predictive reserve distribution:")
        print(f"Dispersion φ̂ = {self.phi:,.2f}")
        print(f"Mean  = {mean:,.0f}")
        print(f"SD    = {sd:,.0f}")
        print(f"CV    = {100 * cv:.1f}%")
        claims_percentiles = pnp.percentile(x, percentiles)
        for p, v in zip(percentiles, claims_percentiles, strict=False):
            print(f"{p:>5.1f}th = {v:,.0f}")
        return {"mean": mean, "sd": sd, "cv": cv}


# ---------------------------------------------------------------------
# Example usage (with global PAL context)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    taylor_ashe_triangle = np.array(
        [
            [
                357848,
                766940,
                610542,
                482940,
                527326,
                574398,
                146342,
                139950,
                227229,
                67948,
            ],
            [
                352118,
                884021,
                933894,
                1183289,
                445745,
                320996,
                527804,
                266172,
                425046,
                0,
            ],
            [290507, 1001799, 926219, 1016654, 750816, 146923, 495992, 280405, 0, 0],
            [310608, 1108250, 776189, 1562400, 272482, 352053, 206286, 0, 0, 0],
            [443160, 693190, 991983, 769488, 504851, 470639, 0, 0, 0, 0],
            [396132, 937085, 847498, 805037, 705960, 0, 0, 0, 0, 0],
            [440832, 847631, 1131398, 1063269, 0, 0, 0, 0, 0, 0],
            [359480, 1061648, 1443370, 0, 0, 0, 0, 0, 0, 0],
            [376686, 986608, 0, 0, 0, 0, 0, 0, 0, 0],
            [344014, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    model = ODPModel(taylor_ashe_triangle)
    model.simulate_reserves()
    model.describe()
