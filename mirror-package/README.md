# proteus-actuarial-library (Legacy Mirror Package)

**Note:** This package has been renamed to `proteusllp-actuarial-library`. This mirror package is maintained for backward compatibility only.

## Migration Notice

This is a legacy package name. For new projects, please install the renamed package:

```bash
pip install proteusllp-actuarial-library
```

Or upgrade existing projects:

```bash
pip uninstall proteus-actuarial-library
pip install proteusllp-actuarial-library
```

## About This Package

This mirror package automatically installs the current `proteusllp-actuarial-library` package and exposes the same `pal` API:

```python
from pal import stochastic_scalar

variable = stochastic_scalar.StochasticScalar([1, 2, 3])
```

## Documentation

All documentation is available at the main package repository:
- Homepage: https://github.com/ProteusLLP/proteusllp-actuarial-library
- Documentation: https://proteusllp-actuarial-library.readthedocs.io/

## Support

For issues or questions, please use the main repository's issue tracker:
https://github.com/ProteusLLP/proteusllp-actuarial-library/issues
