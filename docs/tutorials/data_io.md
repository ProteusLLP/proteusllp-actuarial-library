# Saving and sharing PAL data

Actuarial models often need to exchange simulated losses with other systems,
store large portfolios in Parquet, or save a model for a later session.
PAL supports two different operations:

| Task | API | What is retained? |
| --- | --- | --- |
| Exchange data with pandas, Excel or another system | `to_dataframe`, `to_csv`, `to_parquet` and corresponding `from_*` methods | Dimensions, values, simulation IDs and claim event indices; imported joint scenarios form one coupling group by default |
| Save a PAL object for reuse in PAL | `pal.serialization.save` and `load` | The above, plus leaf types, numeric dtypes and internal coupling groups |

Use **data interchange** when another system should understand the file.
Use **serialization** when a subsequent PAL calculation needs to recover the
same variable structure and the original coupling-group boundaries. An ordinary
table does not record those boundaries, although PAL can couple the jointly
imported scenarios.

Parquet is generally preferable for large numerical datasets because it
stores data in a typed, compressed columnar format. CSV is convenient for
inspection and sharing with applications that do not support Parquet.

## 1. Create a small portfolio

Suppose we have simulated annual losses for Motor and Property business in
two markets. Each row represents the same simulation across the portfolio.

```python
import tempfile
from pathlib import Path

import pandas as pd

from pal import FreqSevSims, ProteusVariable, StochasticScalar
from pal.serialization import load, save

directory = tempfile.TemporaryDirectory()
folder = Path(directory.name)

portfolio = ProteusVariable(
    "Market",
    {
        "UK": ProteusVariable(
            "LoB",
            {
                "Motor": StochasticScalar([100.0, 125.0, 140.0, 115.0]),
                "Property": StochasticScalar([75.0, 80.0, 150.0, 90.0]),
            },
        ),
        "US": ProteusVariable(
            "LoB",
            {
                "Motor": StochasticScalar([120.0, 160.0, 130.0, 145.0]),
                "Property": StochasticScalar([90.0, 110.0, 170.0, 105.0]),
            },
        ),
    },
)

frame = portfolio.to_dataframe()
assert isinstance(frame.columns, pd.MultiIndex)
print(frame[("UK", "Motor")].tolist())  # [100.0, 125.0, 140.0, 115.0]
```

The DataFrame's columns have two named levels, `Market` and `LoB`,
while the rows are indexed by `Simulation`. The values can be passed to
other pandas-based tools without introducing any PAL-specific file format.

## 2. Export a table and import it again

The default layout for a regular nested portfolio is a pandas MultiIndex.
PAL preserves those column levels when writing either CSV or Parquet.

<!--pytest-codeblocks:cont-->

```python
csv_path = folder / "portfolio.csv"
portfolio.to_csv(csv_path)
from_csv = ProteusVariable.from_csv(csv_path)

assert from_csv.dim_name == "Market"
assert from_csv["UK"].dim_name == "LoB"
assert from_csv["UK"]["Motor"].mean() == portfolio["UK"]["Motor"].mean()
```

PAL recognises the extra CSV header rows it writes. For externally produced
MultiIndex CSV files, `header_rows` can be supplied when automatic detection
is not suitable. Parquet retains the pandas column-level metadata:

<!--pytest-codeblocks:skip-->

```python
parquet_path = folder / "portfolio.parquet"
portfolio.to_parquet(parquet_path)
from_parquet = ProteusVariable.from_parquet(parquet_path)
```

Install `pip install "proteusllp-actuarial-library[parquet]"` to enable
Parquet. Exported CSV and Parquet files can be imported again; the special
PAL snapshot format in section 6 is a different kind of file.

## 3. Choose the dimension, simulation and value columns

Many downstream systems expect one row per observation. Long-form data
supports configurable dimension columns, simulation IDs and value fields.

<!--pytest-codeblocks:cont-->

```python
long_path = folder / "portfolio_long.csv"
portfolio.to_csv(
    long_path,
    layout="long",
    dimension_columns=["Market", "Class"],
    simulation_column="Scenario",
    values_column="AnnualLoss",
)

reimported = ProteusVariable.from_csv(
    long_path,
    dimension_columns=["Market", "Class"],
    simulation_column="Scenario",
    values_column="AnnualLoss",
)
assert reimported["US"]["Property"].mean() == portfolio["US"]["Property"].mean()
```

The names and order of `dimension_columns` determine the hierarchy
constructed on import. For an external table, specify the columns that
identify the dimensions and observations in the same way. `event_column`
is also configurable when the observations represent individual claims.

## 4. Exchange individual claims, including zero-claim years

A `FreqSevSims` value represents a collection of individual claims, each
associated with a simulation. A simulation may contain several claims or
none. Unlike a regular annual-loss portfolio, this data should use the
long layout.

<!--pytest-codeblocks:cont-->

```python
claims = ProteusVariable(
    "LoB",
    {
        "Property": FreqSevSims(
            sim_index=[0, 2, 2],
            values=[5_000.0, 10_000.0, 20_000.0],
            n_sims=5,
        )
    },
)

claims_path = folder / "claims.csv"
claims.to_csv(claims_path, layout="long")
imported_claims = ProteusVariable.from_csv(claims_path)["Property"]

assert imported_claims.n_sims == 5
assert len(imported_claims.values) == 3
assert imported_claims.aggregate().mean() == 7_000.0
```

The fifth simulation remains in the model even though it has no claims.
PAL's own long-form exports include the information needed to reconstruct
the full number of simulations, including an entirely zero-claim portfolio.
For an *external* claims listing that does not record this information,
pass `n_sims=` to the importer.

## 5. Coupling groups when importing joint scenarios

A simulation ID identifies the same scenario across multiple risk variables.
For example, UK Motor and UK Property results in simulation 2 may reflect
the same economic and catastrophe conditions. If PAL later reorders scenarios
for one variable, its corresponding values in other variables from that
scenario should move together.

By default, PAL puts all stochastic leaves **imported in one call** into
one coupling group, because they share the scenario axis. This preserves
their joint simulated outcomes during subsequent reordering.

<!--pytest-codeblocks:cont-->

```python
assert (
    from_csv["UK"]["Motor"].coupled_variable_group
    is from_csv["UK"]["Property"].coupled_variable_group
)
```

A **coupling group is not a copula or correlation model**. Coupling does
not change any imported values, generate a correlation, or infer a dependency
structure. It only tells PAL to apply the same future permutation to the
variables. The dependence already present in the data remains as imported.

If the columns instead contain independently sampled **marginal**
distributions that you plan to connect using a copula, opt out:

<!--pytest-codeblocks:cont-->

```python
marginals = ProteusVariable.from_dataframe(frame, couple=False)
assert (
    marginals["UK"]["Motor"].coupled_variable_group
    is not marginals["UK"]["Property"].coupled_variable_group
)
```

The `couple` option is supported by `from_dataframe`, `from_csv` and
`from_parquet`. Separate imports never automatically merge their groups.
If an ordinary table contains several independently sampled groups, import
those subsets separately or use `couple=False` and establish the intended
groups as part of the model.

When you require the **original group boundaries**, rather than one
joint group for an imported table, save a PAL snapshot.

## 6. Save a PAL snapshot when coupling must survive

Tabular interchange represents results, not every relationship between
variables. If two stochastic variables are coupled, such as gross losses
and a derived ceded amount, a PAL snapshot is the appropriate format.

<!--pytest-codeblocks:cont-->

```python
gross = StochasticScalar([100.0, 150.0, 200.0, 125.0])
ceded = gross * 0.4

model_results = ProteusVariable(
    "Measure", {"Gross": gross, "Ceded": ceded}
)

snapshot_path = folder / "model_snapshot.csv"
save(model_results, snapshot_path)
restored = load(snapshot_path)

assert (
    restored["Gross"].coupled_variable_group
    is restored["Ceded"].coupled_variable_group
)
assert restored["Ceded"].mean() == ceded.mean()
```

For large models, prefer `save(model_results, "model_snapshot.parquet")`
and `load("model_snapshot.parquet")`. The Parquet snapshot uses compact
typed columns rather than serialising each simulation value as JSON.

A PAL snapshot is **not** a conventional data interchange file:
read it with `load`, not `ProteusVariable.from_parquet` or pandas.
Conversely, load ordinary table files through `ProteusVariable.from_csv`
or `ProteusVariable.from_parquet`, not `load`.

## 7. Write the same data to cloud storage

The file paths above can also be cloud URLs. PAL uses
[fsspec](https://filesystem-spec.readthedocs.io/) for S3, Azure Blob
Storage and other compatible filesystems.

For Amazon S3, install `pip install "proteusllp-actuarial-library[parquet,s3]"`.
For Azure, use `[parquet,azure]`. Authentication normally comes from
AWS IAM roles, Azure managed identity or the relevant SDK's default
credentials. Filesystem-specific options can also be supplied.

<!--pytest-codeblocks:skip-->

```python
# S3: role-based AWS credentials
portfolio.to_parquet("s3://risk-model-data/scenarios/portfolio.parquet")
from_s3 = ProteusVariable.from_parquet(
    "s3://risk-model-data/scenarios/portfolio.parquet"
)

# Azure Blob Storage: account/container selected by the location
azure_options = {"account_name": "my-storage-account"}
save(
    model_results,
    "az://models/snapshots/portfolio.parquet",
    storage_options=azure_options,
)
from_azure = load(
    "az://models/snapshots/portfolio.parquet",
    storage_options=azure_options,
)
```

The `abfs://` scheme is also supported via the Azure filesystem driver.
Never embed account keys or long-lived access tokens in example code or
source control.

You can exercise the same storage interface without a cloud account using
`fsspec`'s in-memory filesystem:

<!--pytest-codeblocks:cont-->

```python
memory_path = "memory://pal-data-io-tutorial/portfolio.csv"
portfolio.to_csv(memory_path)
from_memory = ProteusVariable.from_csv(memory_path)

assert from_memory["UK"]["Property"].mean() == 98.75
```

## 8. Read and write SQL databases through pandas

PAL does not need a database-specific storage API. Pandas reads and writes
database tables, while PAL's DataFrame methods convert the resulting
records into actuarial variables.

For a small, runnable example, use an in-memory SQLite database. Long-form
data maps naturally onto relational columns, including the simulation ID
that identifies the corresponding outcomes across lines of business.

<!--pytest-codeblocks:cont-->

```python
import sqlite3

connection = sqlite3.connect(":memory:")
try:
    portfolio.to_dataframe(layout="long").to_sql(
        "annual_losses", connection, if_exists="replace", index=False
    )
    # Parameterised queries avoid interpolating user input into SQL.
    records = pd.read_sql_query(
        "SELECT * FROM annual_losses WHERE Simulation >= ?",
        connection,
        params=(0,),
    )
finally:
    connection.close()

from_database = ProteusVariable.from_dataframe(
    records,
    dimension_columns=["Market", "LoB"],
    simulation_column="Simulation",
    values_column="value",
)
assert from_database["UK"]["Motor"].mean() == 120.0
assert (
    from_database["UK"]["Motor"].coupled_variable_group
    is from_database["US"]["Property"].coupled_variable_group
)
```

For SQL Server, PostgreSQL or another production database, install
[SQLAlchemy](https://www.sqlalchemy.org/) and the relevant database driver,
then pass the resulting engine to pandas. For example:

<!--pytest-codeblocks:skip-->

```python
from sqlalchemy import create_engine

# Configure the URL and credentials for your own environment.
engine = create_engine("postgresql+psycopg://user:password@host/database")
portfolio.to_dataframe(layout="long").to_sql(
    "annual_losses", engine, if_exists="append", index=False, chunksize=10_000
)
with engine.connect() as connection:
    records = pd.read_sql_query(
        "SELECT * FROM annual_losses", connection
    )
restored = ProteusVariable.from_dataframe(
    records, dimension_columns=["Market", "LoB"]
)
```

Prefer connection configuration from environment variables, managed
identities or secrets managers to credentials embedded in source code.
Filter and aggregate in SQL before importing large tables into PAL,
and use SQLAlchemy for transactional writes and database-specific upserts.
If an SQL query returns only part of a jointly simulated portfolio, its
variables will be coupled within that import, not automatically to
variables loaded by separate queries.

## Practical considerations

- **GPU:** pandas and Arrow operate on host memory, so exporting moves
  numeric values from a configured CuPy backend to the CPU. Imported
  numeric stochastic values use PAL's active backend again.
- **External data:** column types and some distinctions between types may
  have to be inferred from CSV. Use Parquet or PAL snapshots when types
  matter. Ordinary interchange does not preserve the original coupling-group
  boundaries; by default it creates one joint group for stochastic leaves
  in the same imported table. Pass `couple=False` for independent marginals.
- **Large datasets:** Parquet is usually faster and more compact than CSV.
  Current high-level imports load an individual file into memory; selective
  column reads, row-group filtering and partitioned datasets are not yet
  provided by the variable API.
- **Formats:** files written with a PAL-specific snapshot schema should
  be loaded with `pal.serialization.load`, not treated as ordinary tables.

For the full set of options, see the [usage guide](../usage.md), and see
[Frequency-Severity Modelling](frequency_severity_modelling.md) for working
with event-level losses after import.
