# edmdc-tank-observables

Code and results of a comparison of observable dictionaries for the identification of a laboratory-scale tank level system with extended dynamic mode decomposition with control (EDMDc).

Eight nested dictionaries are compared: the level alone (`D0`), its generic extensions with polynomial terms (`D1a`), Gaussian radial basis functions (`D1b`) or delayed levels (`D1c`), the physics-derived square root of the level alone (`D2`) and the square root added to each generic extension (`D3a`, `D3b`, `D3c`). One model per dictionary is fitted on each run of real closed-loop data and evaluated out of sample with the multi-step error on windows anchored at the setpoint steps.

The repository reproduces every result, table and figure of the paper listed in [Citation](#citation), including the static curve of the plant, which the paper only summarizes.

## Requirements

- Python 3.11 (the results were produced with 3.11.9).
- The libraries in `pyproject.toml` (NumPy, pandas, SciPy, Matplotlib; pytest for the tests). Only minimum versions are fixed; the exact versions of each run are recorded in its `metadata.json`.
- git, if the provenance of new runs is to be recorded (see [Provenance](#provenance)).

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

## Data

The raw data are not included. They are the `Baseline` subset (ten runs) of a public dataset archived on Zenodo under CC BY 4.0. `data/README.md` gives the DOI, the version used and the folder layout expected under `data/raw/`.

## Reproducing the results

```bash
python experiments/run_all.py
```

`run_all.py` runs the scripts below in this order, each in its own process, and stops at the first one that fails. Every script writes to its own folder in `results/`, together with a `metadata.json`.

| Order | Script | What it does | Output folder |
| --- | --- | --- | --- |
| 1 | `data_summary.py` | Runs, sensor faults, quantization of the level and bounds on the sampling period, from the raw data | `results/data_summary/` |
| 2 | `E1_hyperparameters.py` | Selects the hyperparameter of each dictionary family (polynomial order, number of RBF centres, number of delays) on a validation part of the fitting data | `results/E1_hyperparameters/` |
| 3 | `E2_dictionaries.py` | Fits the eight dictionaries on every run with the hyperparameters of E1 and computes the multi-step errors | `results/E2_dictionaries/` |
| 4 | `E2_comparisons.py` | Paired comparisons between dictionaries (counts, sign and Wilcoxon tests, Holm adjustment, bootstrap intervals) and condition numbers, from the errors of E2 | `results/E2_comparisons/` |
| 5 | `E2_one_step.py` | One-step-ahead errors of the E2 models, in and out of sample | `results/E2_one_step/` |
| 6 | `plant_static_curve.py` | Static curve of the plant and times to 95 % of the setpoint steps, measured from the raw data without fitting any model | `results/plant_static_curve/` |
| 7 | `paper_assets.py` | LaTeX tables and the figure of the paper, from the saved results | `results/paper_assets/` |

The scripts take no arguments; their settings are the constants at the top of each script. Scripts 3–5 and 7 read the results of earlier scripts and stop if those were not produced from a clean commit.

The committed `results/` were produced by a single run of `run_all.py` from a clean commit of this repository. A new run overwrites them, so `git diff` shows any difference; the figures can differ in embedded metadata such as their creation date even when their content is the same.

## Where each element of the paper comes from

| Paper | Script | Results |
| --- | --- | --- |
| Table 1 (dictionaries) | `src/edmdc_tank_observables/observables/dictionaries.py` | — |
| Fig. 1 (fitting segment and evaluation windows) | `paper_assets.py` | `paper_assets/figures/figure_run_windows.pdf` |
| Table 2 (prediction errors) | `E2_dictionaries.py` → `paper_assets.py` | `E2_dictionaries/tables/fit_indicators.csv`, `error_summary.csv` → `paper_assets/tables/table_errors.tex` |
| Table 3 (paired comparisons at 120 s) | `E2_comparisons.py` → `paper_assets.py` | `E2_comparisons/paired_tests.csv` → `paper_assets/tables/table_paired_120s.tex` |
| Table 4 (one-step and multi-step comparisons) | `E2_one_step.py` → `paper_assets.py` | `E2_one_step/one_step_pair_counts.csv` → `paper_assets/tables/table_one_step.tex` |
| Data description and sampling period | `data_summary.py` | `data_summary/runs.csv`, `quantization.csv`, `sampling_bounds.csv` |
| Hyperparameter selection | `E1_hyperparameters.py` | `E1_hyperparameters/tables/selection.csv`, `error_summary.csv` |
| Static curve of the plant and rise times | `plant_static_curve.py` | `plant_static_curve/tables/curve_fits.csv`, `curve_pair_counts.csv`, `plateaus.csv`, `steps.csv`; `figures/static_curve_by_run.pdf` |
| Condition numbers | `E2_dictionaries.py`, `E2_comparisons.py` | `E2_dictionaries/tables/models.csv`, `E2_comparisons/conditioning_sqrt_h.csv` |

All result paths are relative to `results/`.

## Tests

```bash
pytest
```

Tests that need the raw data or the committed results are skipped, with the reason, when these are not available. The regression test of E2 compares a fresh fit with the reference tables in `tests/reference_data/E2/`.

## Provenance

Every `metadata.json` records the script, the UTC time, the git commit and whether the working tree was clean (changes inside `results/` are not counted), the versions of Python and the main libraries, the configuration used and a summary of the main figures.

## Repository structure

```
src/edmdc_tank_observables/   package: data loading and preprocessing, dictionaries, EDMDc, evaluation, provenance
experiments/                  one script per analysis, and run_all.py
results/                      outputs of the scripts, with their metadata.json
tests/                        pytest tests and reference tables
data/                         data/README.md; raw data go in data/raw/ (not versioned)
docs/ARCHITECTURE.md          how the code is organized
```

## License

- Code: MIT License (see `LICENSE`).
- Results in `results/`: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- The raw data are not redistributed and keep the license of their own record.

## Citation

If you use this code or its results, please cite the paper (also in `CITATION.cff`):

A. Martí-Coll, N. Freire-Mahía, P.-P. Gómez-González, Á. Michelena, E. Jove, J. L. Calvo-Rolle. *What Does a Physics-Derived Observable Add to Generic EDMDc Dictionaries? A Nested Comparison on a Laboratory Tank*. In: Advances in Technical Sciences and Architecture: Selected Contributions of the XXII Scientific Convention of Engineering and Architecture, CCIA 2026. Springer, 2026 (to appear).
