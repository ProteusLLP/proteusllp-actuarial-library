"""Apply a Gaussian copula while preserving derived simulation values.

Copula.apply reorders simulations but does not change marginal values.
PAL coupling groups reorder derived variables together with their inputs.
"""

import numpy as np

from pal import set_random_seed
from pal.copulas import GaussianCopula
from pal.distributions import LogNormal
from pal.maths import asnumpy


def main() -> None:
    """Combine dependent lines of business without breaking their ancestry."""
    set_random_seed(42)
    motor = LogNormal(mu=12, sigma=0.7).generate(n_sims=512)
    property_loss = LogNormal(mu=13, sigma=0.9).generate(n_sims=512)
    motor_with_expenses = 1.1 * motor
    before = np.sort(asnumpy(motor.values))

    GaussianCopula([[1.0, 0.6], [0.6, 1.0]]).apply([motor, property_loss])
    total = motor_with_expenses + property_loss

    np.testing.assert_array_equal(np.sort(asnumpy(motor.values)), before)
    np.testing.assert_allclose(asnumpy(motor_with_expenses.values), 1.1 * asnumpy(motor.values))
    assert total.coupled_variable_group is motor.coupled_variable_group
    assert motor_with_expenses.coupled_variable_group is property_loss.coupled_variable_group

    print("Dependent portfolio expected loss:", total.mean())
    print("Dependent portfolio 99.5th percentile:", total.percentile(99.5))


if __name__ == "__main__":
    main()
