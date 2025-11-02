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

- **QuotaShare**: Proportional reinsurance where the reinsurer pays a fixed percentage
- **ExcessOfLoss**: Non-proportional reinsurance with attachment point and limit

Usage Example
-------------

.. code-block:: python

   from pal import distributions, contracts

   # Create loss variable
   losses = distributions.Gamma(alpha=2.5, beta=2).generate()

   # Apply quota share reinsurance (50% cession)
   ceded, retained = contracts.QuotaShare(cession_rate=0.5).apply(losses)

   # Apply excess of loss reinsurance
   ceded, retained = contracts.ExcessOfLoss(
       attachment_point=100,
       limit=1000
   ).apply(losses)
