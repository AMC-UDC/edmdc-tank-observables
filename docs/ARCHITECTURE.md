# Architecture

How the code is organized, for readers who want to modify or extend it. How to install it and run it is in the `README.md`. Every module and script starts with a docstring that explains its choices in detail; this page only shows how the pieces fit together.

## Layout

```
src/edmdc_tank_observables/
├── paths.py                  repository root and default raw-data folder (data/raw/)
├── provenance.py             git state and metadata.json of every run
├── reporting.py              common figure style; figures saved as PDF and PNG
├── data/
│   ├── loader.py             reads the Baseline CSV files; rebuilds time; detects sensor faults
│   └── preprocess.py         decimation, gap check, normalization, temporal split
├── observables/
│   └── dictionaries.py       the eight dictionaries (D0 ... D3c)
├── models/
│   └── edmdc.py              EDMDc fit (least squares) and simulation
└── evaluation/
    ├── metrics.py            windows anchored at the setpoint steps; multi-step error
    └── protocol.py           fit-and-evaluate loop and the tables shared by the experiments

experiments/                  one script per analysis, run in order by run_all.py
tests/                        pytest tests; tests/reference_data/E2/ holds the regression reference
results/                      outputs of the scripts (one folder per script)
```

The package holds everything that more than one script needs. The scripts hold the settings of each analysis as constants at their top; there is no configuration file and the scripts take no arguments.

## Processing of one run

The same chain is used by every script that fits models (`E1_hyperparameters.py`, `E2_dictionaries.py`):

1. **Load** (`data.loader.load_baseline`). One CSV per run, sampled at 1 Hz. Time is rebuilt from the row index because the timestamp column of the Baseline files is not reliable. Samples with a sensor fault (level 0 with a positive setpoint) are dropped by default. The electrovalve opening is read from the file name.
2. **Prepare** (`data.preprocess.prepare`). Decimation to the sampling period of the model by keeping the samples whose time is a multiple of it, without averaging. If a gap remains after decimation, preprocessing stops with an error: gaps are not filled. Level and control are scaled to [0, 1] with their physical range (0–100 %), not with statistics of the data.
3. **Split** (`data.preprocess.temporal_split`). A temporal cut at an absolute instant, a fraction of the common acquisition time, so the cut falls at the same instant in every run. E1 cuts the fitting part again in the same way to obtain its validation part; it never uses the evaluation part.
4. **Lift** (`observables.dictionaries.make_dictionary`). Each dictionary maps the level to a vector of observables. The level is always the first observable and the square root, when present, the last one. With delays, `Dictionary.align_input` trims the control so that observables and input refer to the same instants.
5. **Fit** (`models.edmdc.fit_edmdc`). Ordinary least squares, without regularization, of `Psi(h_{k+1}) ≈ A Psi(h_k) + B u_k + c`. The predicted level is the first component of the lifted state, so no output matrix is fitted. The model keeps the condition number of the regression matrix and its in-sample one-step error.
6. **Evaluate** (`evaluation.metrics`, `evaluation.protocol.evaluate`). One window per setpoint step of the evaluation part. Each window starts at the last sample before the step; the model is initialized from the measured level up to that sample and then simulated in open loop with the recorded control over the horizon. Errors are in % of level and are pooled into one RMSE per run, dictionary and horizon.

`evaluation.protocol.sweep` runs steps 4–6 for a set of dictionaries and runs and returns the tables `models`, `errors`, `windows` and, optionally, `coefficients`.

## Dependencies between scripts

The scripts exchange information only through `results/`:

```
data_summary.py ─────────────────────────────── raw data only
E1_hyperparameters.py ── results/E1_hyperparameters/metadata.json (selected hyperparameters)
        │
        ▼
E2_dictionaries.py ───── results/E2_dictionaries/ (errors, windows, coefficients, models)
        │
        ├──► E2_comparisons.py   (reads the errors table; fits nothing)
        └──► E2_one_step.py      (rebuilds the models from the coefficients table; fits nothing)
plant_static_curve.py ────────────────────────── raw data only; no model
paper_assets.py ──────── reads E2_dictionaries, E2_comparisons and E2_one_step results and one raw run
```

`data_summary.py` and `E2_one_step.py` import helper functions of other scripts, but not their results. A script that reads earlier results stops if their `metadata.json` does not come from a clean commit, and E2 stops if a hyperparameter selected by E1 lies at the upper end of its grid.

## Invariants checked in the code

- Every dictionary of a run and horizon is scored on the same windows, which depend only on the setpoint, the horizon and the offset of the window start, and are saved in the `windows` table (`evaluation.protocol.check_same_windows`).
- The runs are checked to end at the same instant before the temporal cut is made.
- `E2_one_step.py` checks that the in-sample one-step error recomputed from the saved coefficients matches the one saved by E2, within a relative tolerance.
- `paper_assets.py` checks that every dictionary, horizon and pair it formats appears exactly once in the tables it reads.

## Statistics of the comparisons

`E2_comparisons.py` works on the run-by-run differences of RMSE between two dictionaries. For each pair it gives the counts of runs won by each side, the exact sign test and the exact Wilcoxon signed-rank test, and percentile bootstrap intervals of the median and the mean of the differences. Holm's adjustment is applied within each family of comparisons, separately to the sign-test and to the Wilcoxon p-values; the single comparisons are not adjusted. The bootstrap uses a fixed seed, and the resampled indices are applied to the runs in alphabetical order of their labels, so the intervals depend on that order.

## Provenance

`provenance.write_metadata` writes a `metadata.json` next to the outputs of each script, with the script, the UTC time, the git commit and branch, whether the working tree was clean (changes inside `results/` are not counted, because the scripts write there while they run), the versions of Python and the main libraries, the configuration and a summary of the main figures.

## Vocabulary

- **Run**: one recorded experiment of the dataset (one CSV file). The word *test* is kept for pytest tests and for the evaluation part of a split.
- **Opening**: the number `NNpct` in the file names is the electrovalve opening in percent. The dataset column `Valve_Closure_pct` is the closure (100 − opening); it is only used in a test of that identity.
