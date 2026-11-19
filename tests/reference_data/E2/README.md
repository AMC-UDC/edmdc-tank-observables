# Regression reference: dictionary comparison (E2)

These tables are the reference output of `experiments/E2_dictionaries.py`.
They are copies of the tables of the same name in
`results/E2_dictionaries/tables/`, as produced by the current code.

`tests/test_e2_regression.py` runs E2 on the raw data and checks that it
reproduces these tables: same columns, column types and number of rows, text
and integer columns exactly, numeric columns within a relative tolerance of
1e-9 and an absolute tolerance of 1e-12. E2 reads the hyperparameters selected
by E1 from `results/E1_hyperparameters/metadata.json`, so the test is skipped
if any run of the raw data (see `data/README.md`) or that file is missing.

## Files

- `models.csv`: one row per run and dictionary.
- `errors.csv`: one row per run, series and horizon.
- `error_summary.csv`: median, minimum and maximum RMSE across runs per series
  and horizon.
- `improvements.csv`: runs in which each dictionary beats D0, per horizon.

## Updating

Update these files only on purpose, when a hyperparameter or the evaluation
protocol changes: regenerate the results from a clean commit, copy the four
tables here and commit the change on its own.
