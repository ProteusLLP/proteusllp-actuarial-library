Reinsurance Contracts
=====================

The contracts module provides reinsurance contract modeling capabilities.

.. automodule:: pal.contracts
   :members:
   :undoc-members:
   :show-inheritance:

Available Contract Types
-------------------------

The following reinsurance contract types are supported:

- **XoL**: Excess of loss reinsurance layer with attachment point and limit
- **XoLTower**: A tower of stacked XoL layers

Usage Example
-------------

.. code-block:: python

   from pal import distributions
   from pal.contracts import XoL
   from pal.frequency_severity import FrequencySeverityModel

   # Generate claims
   claims = FrequencySeverityModel(
       freq_dist=distributions.Poisson(mean=5),
       sev_dist=distributions.LogNormal(mu=10, sigma=1),
   ).generate()

   # Apply excess of loss reinsurance
   result = XoL(
       name="5m xs 1m",
       limit=5_000_000,
       excess=1_000_000,
       premium=500_000,
   ).apply(claims)
