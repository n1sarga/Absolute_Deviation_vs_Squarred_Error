# Data

The analysis reads only from `processed/`. Files under `raw/` preserve available
source downloads and should not be edited in place.

## Processed datasets

| File | Purpose |
|---|---|
| `boston_housing.csv` | Analysis-ready Boston Housing data |
| `concrete_strength.csv` | Analysis-ready Concrete Strength data |
| `hbk.csv` | Analysis-ready Hawkins-Bradu-Kass data |
| `synthetic_ols_lad_outliers.csv` | Supplied synthetic regression data |

## Raw sources

- `raw/concrete_strength/` contains the original spreadsheet, archive, and
  dataset notes.
- `raw/concrete_strength/original_copy/` preserves the additional identical
  spreadsheet copy that existed before repository organization.
- `raw/hbk/hbk.csv` preserves the downloaded HBK source file.

Identifiers are not used as measurement variables. See
[`docs/outlier_method.md`](../docs/outlier_method.md) for the exact columns used.
