Proteus Actuarial Library documentation
=======================================

.. raw:: html

   <section class="pal-hero" aria-labelledby="pal-hero-title">
     <div class="pal-eyebrow"><span>PAL</span> Open-source actuarial modelling in Python</div>
     <p class="pal-hero-title" id="pal-hero-title" role="heading" aria-level="2">Build models for the unexpected.</p>
     <p class="pal-hero-lead">The Proteus Actuarial Library is a fast, lightweight framework for simulation-based actuarial and financial models—with dependency tracking, copulas and optional GPU acceleration built in.</p>
     <div class="pal-actions">
       <a class="pal-button pal-button-primary" href="tutorials/getting_started.html">Get started <span aria-hidden="true">→</span></a>
       <a class="pal-button pal-button-secondary" href="api/modules.html">Explore the API</a>
     </div>
     <div class="pal-install" aria-label="Install PAL"><code>pip install proteusllp-actuarial-library</code></div>
   </section>

Start with PAL
--------------

.. grid:: 1 2 2 4
   :gutter: 3
   :class-container: pal-card-grid

   .. grid-item-card:: Learn the essentials
      :link: tutorials/getting_started
      :link-type: doc
      :class-card: pal-card

      Create stochastic variables, combine risks and run your first model.

      +++
      **Getting started** →

   .. grid-item-card:: Model dependence
      :link: tutorials/coupling_groups_and_copulas
      :link-type: doc
      :class-card: pal-card

      Apply copulas and understand PAL's automatic coupling groups.

      +++
      **Copulas guide** →

   .. grid-item-card:: Price reinsurance
      :link: tutorials/xol_reinsurance
      :link-type: doc
      :class-card: pal-card

      Build XoL layers, towers, reinstatements and aggregate covers.

      +++
      **Reinsurance tutorial** →

   .. grid-item-card:: Find a class or method
      :link: api/modules
      :link-type: doc
      :class-card: pal-card

      Browse the complete, searchable reference generated from PAL itself.

      +++
      **API reference** →

Why PAL
-------

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: pal-feature-grid

   .. grid-item::

      .. rubric:: Fast by default

      NumPy and SciPy at the core, with optional CuPy acceleration for large simulation workloads.

   .. grid-item::

      .. rubric:: Dependence made explicit

      Automatic variable coupling and a broad choice of copulas make complex models easier to reason about.

   .. grid-item::

      .. rubric:: Built for actuarial work

      Frequency–severity models, reinsurance contracts, risk measures and capital allocation are first-class concepts.

A small model, end to end
-------------------------

.. code-block:: python
   :caption: Combine two risks and apply a dependency structure

   from pal import copulas, distributions

   losses = distributions.Gamma(alpha=2.5, theta=2).generate()
   expenses = distributions.LogNormal(mu=1, sigma=0.5).generate()

   copulas.GumbelCopula(theta=1.2).apply([losses, expenses])
   total = losses + expenses

.. raw:: html

   <aside class="pal-community">
     <div>
       <span class="pal-community-kicker">Open source · MIT licensed</span>
       <h2>Use PAL, inspect it, improve it.</h2>
       <p>PAL is developed in the open by Proteus Consulting. Questions, ideas and focused pull requests are welcome.</p>
     </div>
     <a class="pal-button pal-button-secondary" href="https://github.com/ProteusLLP/proteusllp-actuarial-library">View on GitHub <span aria-hidden="true">↗</span></a>
   </aside>

.. toctree::
   :maxdepth: 2
   :caption: User guide
   :hidden:

   usage

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/README
   tutorials/getting_started
   tutorials/distributions_guide
   tutorials/frequency_severity_modelling
   tutorials/coupling_groups_and_copulas
   tutorials/xol_reinsurance
   tutorials/property_exposure_rating
   tutorials/reinstatement_pricing
   tutorials/risk_measures_and_allocation

.. toctree::
   :maxdepth: 3
   :caption: API reference
   :hidden:

   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Project
   :hidden:

   development
   contributing
   development
   license
