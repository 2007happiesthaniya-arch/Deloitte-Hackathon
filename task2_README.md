# Task 2 Starter: Insurance Premium Regression

This starter adapts the Task 1 workflow into a no-leakage regression pipeline for Task 2.

## What It Does

- loads the insurance dataset
- aggregates raw rows into one row per `zip × year`
- engineers lagged features from prior years only
- predicts 2021 earned premium using historical data through 2020
- compares:
  - `ridge_pca`
  - `ridge_hybrid` using the 4-qubit quantum feature map
  - `random_forest_raw`
  - `gradient_boosting_raw`

## Main Script

- [task2_insurance_quantum_regression.py](C:\Users\namfa\OneDrive\Documents\New%20project\task2_insurance_quantum_regression.py)

## Inputs

- insurance dataset:
  `C:\Users\namfa\Downloads\abfa2rbci2UF6CTj_cal_insurance_fire_census_weather (2).csv`
- optional external wildfire-risk feature file:
  a CSV with columns `zip`, `target_year`, `risk_score`

## Run

```powershell
$env:MPLCONFIGDIR='C:\Users\namfa\OneDrive\Documents\New project\.mplconfig'
.\.venv\Scripts\python.exe task2_insurance_quantum_regression.py
```

With an optional external risk file:

```powershell
.\.venv\Scripts\python.exe task2_insurance_quantum_regression.py --risk-features "C:\path\to\risk_scores.csv"
```

## Outputs

- `results_task2/task2_zip_year_aggregates.csv`
- `results_task2/task2_supervised_dataset.csv`
- `results_task2/task2_predictions_2021.csv`
- `results_task2/task2_results_summary.json`
- `results_task2/top_predicted_premiums_2021.png`

## Notes

- The target is `Earned Premium`, modeled on the `log1p` scale.
- The quantum component is used as a feature map, not as the sole regressor.
- On the current run, the classical raw-feature tree models were stronger than the hybrid ridge model, which is useful benchmarking information rather than a failure.
