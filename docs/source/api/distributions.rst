Distributions
=============

The distributions module provides statistical distributions for generating stochastic variables. Each concrete distribution has its own reference page.

Discrete distributions
----------------------

.. autosummary::
   :toctree: distributions
   :nosignatures:

   pal.distributions.Poisson
   pal.distributions.NegBinomial
   pal.distributions.Binomial
   pal.distributions.HyperGeometric
   pal.distributions.Bernoulli

Continuous distributions
------------------------

.. autosummary::
   :toctree: distributions
   :nosignatures:

   pal.distributions.GPD
   pal.distributions.Burr
   pal.distributions.Beta
   pal.distributions.MBBEFD
   pal.distributions.Normal
   pal.distributions.Logistic
   pal.distributions.LogNormal
   pal.distributions.Gamma
   pal.distributions.NonCentralChiSquared
   pal.distributions.InverseGamma
   pal.distributions.Pareto
   pal.distributions.Paralogistic
   pal.distributions.InverseBurr
   pal.distributions.InverseParalogistic
   pal.distributions.Weibull
   pal.distributions.InverseWeibull
   pal.distributions.GEV
   pal.distributions.StudentsT
   pal.distributions.InverseGaussian
   pal.distributions.GeneralizedInverseGaussian
   pal.distributions.Exponential
   pal.distributions.Uniform
   pal.distributions.InverseExponential

Supporting API
--------------

.. autosummary::
   :nosignatures:

   pal.distributions.DistributionBase
   pal.distributions.DiscreteDistributionBase
   pal.distributions.DistributionGeneratorBase
   pal.distributions.DiscreteDistributionGenerator
   pal.distributions.ContinuousDistributionGenerator

All distributions follow the same general API pattern and support the active PAL backend where implemented.
