  ProteusLike (Protocol)
  ├── inherits from NumericLike (arithmetic operations)
  ├── inherits from collections.abc.Sequence (sequence operations)
  ├── defines: n_sims, values, __getitem__, sum(), mean(), upsample()
  │
  ├── ProteusVariable (concrete class)
  │   └── implements the ProteusLike protocol
  │
  └── ProteusStochasticVariable (abstract base class)
      ├── inherits from NDArrayOperatorsMixin (NumPy integration)
      └── concrete implementations:
          ├── StochasticScalar
          └── FreqSevSims