# Data

The raw data are not part of this repository and are not redistributed. They come from a public dataset:

- Title: Two-Tank Water System Dataset for RLS-Based Identification and Adaptive PID Control Under Diverse Operating Conditions
- DOI: [10.5281/zenodo.17688567](https://doi.org/10.5281/zenodo.17688567) (Zenodo record, version v1)
- License: CC BY 4.0, as stated on the Zenodo record
- File: `Dataset.zip`

## How to obtain the data

1. Download `Dataset.zip` from the DOI above.
2. Extract it so that the three subfolders sit directly under `data/raw/`. If the archive extracts into a top-level folder, move the subfolders out of it.

## Expected layout

```
data/raw/
├── Baseline/          Baseline_10pct.csv ... Baseline_100pct.csv        (10 files)
├── PassiveOutflow/    PassiveOutflow_0pct.csv ... PassiveOutflow_100pct.csv (11 files)
└── Disturbance/       Disturbance_{High,Low}Outflow_<N>pct.csv           (4 files)
```

The main experiments use the `Baseline` subfolder. `data/raw/` is ignored by git.
