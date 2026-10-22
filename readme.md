# Spotter Freight Rate Machine Learning Challenge

This repository contains a production-grade Machine Learning pipeline engineered to forecast logistical freight rates using historical lane data.

## Architectural Overview
- **Core Model:** LightGBM Regressor (Gradient Boosting Decision Trees).
- **Validation Scheme:** 5-Fold Cross-Validation to eliminate data leakage and ensure generalized inference.
- **Feature Engineering:** Geospatial preservation and cyclical temporal decomposition (Year, Month, Day, Day-of-Week).
- **Environment Management:** Powered by `uv` for lightning-fast, reproducible execution.

## Getting Started & Execution

### 1. Prerequisites
Ensure you have the robust Astral `uv` toolchain installed on your machine.

### 2. Dependency Sychronization
Sync the precise environment dependencies specified in the project configuration:
```bash
uv sync
```

### 3. Running the Forecasting Pipeline
Execute the main script to automatically ingest the assets, train the cross-validated models, and export the submission matrices:
```bash
uv run baseline.py
```
This generates `validation_predictions.csv` and populates the placeholder columns.

### 4. Running the Evaluator
Run the official assessment validator to produce the target chart asset:
```bash
uv run python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```
The final visualization will be accessible under `scorer_results/candidate_december.png`.
