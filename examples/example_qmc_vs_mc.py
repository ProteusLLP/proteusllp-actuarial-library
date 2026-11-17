"""Interactive MC and RQMC copula comparison, runnable from a checkout.

The implementation also ships with PAL so installed users can run
`python -m pal.examples.example_qmc_vs_mc`.
"""

from pal.examples.example_qmc_vs_mc import copula_density_chart, copula_samples, convergence_chart, main

if __name__ == "__main__":
    main()
