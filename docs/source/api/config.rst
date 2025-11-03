Configuration
=============

The config module provides global configuration settings for PAL.

.. automodule:: pal.config
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Options
---------------------

The following configuration options are available:

n_sims
^^^^^^

The number of simulations to run. Default is 100,000.

.. code-block:: python

   from pal import config
   config.n_sims = 1000000

rng
^^^

The random number generator instance. Uses numpy's default_rng.

.. code-block:: python

   from pal import config
   config.set_random_seed(123456)

GPU Mode
--------

PAL supports GPU acceleration with CuPy. Enable it via environment variable:

.. code-block:: bash

   export PAL_USE_GPU=1

Or in Python:

.. code-block:: python

   import os
   os.environ['PAL_USE_GPU'] = '1'
