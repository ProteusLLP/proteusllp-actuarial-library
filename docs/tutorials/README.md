# Tutorials

These tutorials walk through the main features of the Proteus Actuarial
Library (PAL) with worked examples and outputs.

## Getting Started

| Tutorial | Description |
|----------|-------------|
| [Getting Started](getting_started.md) | First steps — distributions, stochastic variables, basic arithmetic and configuration |
| [Distributions Guide](distributions_guide.md) | Choosing and parameterising severity and frequency distributions |
| [Frequency-Severity Modelling](frequency_severity_modelling.md) | Compound models, aggregate losses, occurrence maxima and derived statistics |
| [Coupling Groups, Copulas and Variable Reordering](coupling_groups_and_copulas.md) | Dependency structures, copula families, rank reordering and coupling groups |
| [Pricing an Excess-of-Loss Reinsurance Program](xol_reinsurance.md) | XoL layers, towers, reinstatements, aggregate limits and net loss calculation |

## Prerequisites

The tutorials assume PAL is installed and importable:

```python
from pal import distributions, copulas, config
```

All examples use `set_random_seed(42)` for reproducibility.

## Interactive Notebooks

Jupyter notebook versions of several tutorials are available in the
`examples/` directory for interactive exploration with inline plots.
