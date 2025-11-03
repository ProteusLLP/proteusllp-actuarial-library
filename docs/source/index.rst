Proteus Actuarial Library Documentation
========================================

Welcome to the Proteus Actuarial Library (PAL) documentation!

PAL is a fast, lightweight framework for building simulation-based actuarial and financial models. It handles complex statistical dependencies using copulas while providing simple, intuitive syntax.

**Key Features:**

- Built on NumPy/SciPy for performance
- Optional GPU acceleration with CuPy
- Automatic dependency tracking between variables
- Comprehensive statistical distributions
- Clean, Pythonic API

Quick Start
-----------

.. code-block:: python

   from pal import distributions, copulas

   # Create stochastic variables
   losses = distributions.Gamma(alpha=2.5, beta=2).generate()
   expenses = distributions.LogNormal(mu=1, sigma=0.5).generate()

   # Apply statistical dependencies
   copulas.GumbelCopula(alpha=1.2, n=2).apply([losses, expenses])

   # Variables are now correlated
   total = losses + expenses

Installation
------------

.. code-block:: bash

   # Basic installation
   pip install proteus-actuarial-library

   # With GPU support
   pip install proteus-actuarial-library[gpu]

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage
   
.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   contributing
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
