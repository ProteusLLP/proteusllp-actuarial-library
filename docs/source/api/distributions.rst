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
   pal.distributions.Empirical

   pal.distributions.Categorical

Continuous distributions
------------------------

.. autosummary::
   :toctree: distributions
   :nosignatures:

   pal.distributions.GPD
   pal.distributions.Burr
   pal.distributions.GeneralizedBetaPrime
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
   pal.distributions.HyperExponential
   pal.distributions.Uniform
   pal.distributions.InverseExponential

Multivariate distributions
--------------------------

Multivariate samples are returned as a :class:`pal.variables.ProteusVariable`
with one named :class:`pal.variables.StochasticScalar` per component.

Multivariate normal and Student's t models automatically repair invalid
correlations using Higham's algorithm, raising a ``UserWarning`` when repair
is necessary. The marginal variances or scale-matrix diagonal are preserved.
The ``adjusted_correlation_matrix`` property gives the correlations used;
``adjusted_covariance_matrix`` (normal) and ``adjusted_scale_matrix``
(Student's t) give the full adjusted matrices. Valid matrices are unchanged.
Pass ``repair="none"`` to require a valid input, or use Higham's positive
vector ``weights`` and fixed-entry ``pattern`` parameters.

.. autosummary::
   :toctree: distributions
   :nosignatures:

   pal.distributions.MultivariateNormal
   pal.distributions.MultivariateStudentsT
   pal.distributions.Multinomial
   pal.distributions.Dirichlet
   pal.distributions.InvertedDirichlet
   pal.distributions.GeneralizedDirichlet
   pal.distributions.InvertedGeneralizedDirichlet
   pal.distributions.Wishart
   pal.distributions.InverseWishart

Supporting API
--------------

.. autosummary::
   :nosignatures:

   pal.distributions.DistributionBase
   pal.distributions.DiscreteDistributionBase
   pal.distributions.MultivariateDistributionBase
   pal.distributions.MatrixDistributionBase
   pal.distributions.DistributionGeneratorBase
   pal.distributions.DiscreteDistributionGenerator
   pal.distributions.ContinuousDistributionGenerator

All distributions follow the same general API pattern and support the active PAL backend where implemented.
