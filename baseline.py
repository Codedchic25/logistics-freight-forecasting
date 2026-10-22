"""Spotter Freight Challenge - Production-Grade ML Ingestion & Inference.

This module automates strict file matching, executes cyclical feature engineering,
implements 5-Fold Cross-Validation using LightGBM, and generates structured
predictions satisfying scoring constraints.
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


def locate_exact_file(filename: str) -> Path:
    """Locates target CSV files by searching prioritized local paths.

    Args:
        filename: Standard expected filename string.

    Returns:
        Path: Resolved absolute path to the file.
    """
    file_mapping = {
        "train_test": ["train test.csv", "train-test.csv", "train_test.csv"],
        "validation": ["validation.csv"],
        "december": ["december-chart-inputs.csv", "december_chart_inputs.csv"],
    }

    candidates = file_mapping.get(filename, [filename])

    # Target workspace execution points
    for name in candidates:
        for path in [Path(name), Path(".vscode") / name, Path("data") / name]:
            if path.is_file():
                print(f"[RESOLVED] Located file map: '{filename}' -> '{path}'")
                return path

    print(f"CRITICAL ERROR: Matrix source '{filename}' could not be resolved.")
    print("Please ensure your CSV files are in the main folder or .vscode directory.")
    sys.exit(1)


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts date-time vectors to capture cyclical logistical trends.

    Args:
        df: Input DataFrame with raw date dimensions.

    Returns:
        pd.DataFrame: Augmented matrix with explicit sub-temporal fields.
    """
    processed_df = df.copy()
    processed_df["date"] = pd.to_datetime(processed_df["date"])
    processed_df["year"] = processed_df["date"].dt.year
    processed_df["month"] = processed_df["date"].dt.month
    processed_df["day"] = processed_df["date"].dt.day
    processed_df["dayofweek"] = processed_df["date"].dt.dayofweek
    return processed_df


def execute_freight_pipeline() -> None:
    """Coordinates cross-validated model fitting and outputs prediction arrays."""
    print("==================================================================")
    print("Initializing Spotter Freight ML Production Framework...")
    print("==================================================================\n")

    # 1. Exact Ingestion Sequence
    train_path = locate_exact_file("train_test")
    val_path = locate_exact_file("validation.csv")
    dec_path = locate_exact_file("december")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_dec = pd.read_csv(dec_path)

    # 2. Strict Target Variable Isolation
    target_variable = "posted_rate"
    if target_variable not in df_train.columns:
        potential_targets = list(set(df_train.columns) - set(df_val.columns))
        target_variable = potential_targets[0] if potential_targets else "posted_rate"

    print(f"[INFO] Target field isolated successfully as: '{target_variable}'")

    X_train_raw = df_train.drop(columns=[target_variable], errors="ignore")
    y_train = df_train[target_variable].values

    # 3. Structural Feature Transformation & In-Sample Augmentation
    print("Engineering transactional feature attributes...")
    X_train_eng = engineer_temporal_features(X_train_raw)
    X_val_eng = engineer_temporal_features(df_val)
    X_dec_eng = engineer_temporal_features(df_dec)

    # Reconstruct continuous base columns map
    base_numerical_features = [
        "distance",
        "weight",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "year",
        "month",
        "day",
        "dayofweek",
    ]
    ignored = ["load_id", "pickup", "delivery", "equipment", "date"]
    extra_features = [
        c
        for c in X_train_eng.columns
        if c not in base_numerical_features and c not in ignored
    ]
    active_features = base_numerical_features + extra_features
    print(f"[SCHEMA] Engaged numerical metrics array: {active_features}")

    # Fallback Imputation for the isolated December matrix constraints
    # Since December lacks coordinates and indicators, we populate them with local mean references
    for col in active_features:
        if col not in X_dec_eng.columns:
            # Match baseline historical means from train to populate missing features stably
            X_dec_eng[col] = X_train_eng[col].mean()

    # 4. Dense Categorical Encoding
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_train_cat = encoder.fit_transform(X_train_eng[["equipment"]])
    encoded_val_cat = encoder.transform(X_val_eng[["equipment"]])
    encoded_dec_cat = encoder.transform(X_dec_eng[["equipment"]])

    X_train_final = np.hstack(
        [X_train_eng[active_features].fillna(0).values, encoded_train_cat]
    )
    X_val_final = np.hstack(
        [X_val_eng[active_features].fillna(0).values, encoded_val_cat]
    )
    X_dec_final = np.hstack(
        [X_dec_eng[active_features].fillna(0).values, encoded_dec_cat]
    )

    # 5. 5-Fold Cross-Validation Framework with LightGBM
    print(
        f"\n[VALIDATION] Initiating 5-Fold Cross-Validation on {X_train_final.shape[0]} samples..."
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    val_preds_oof = np.zeros(X_val_final.shape[0])
    dec_preds_oof = np.zeros(X_dec_final.shape[0])

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 800,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 31,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_train_final, y_train)):
        X_tr, y_train_fold = X_train_final[train_idx], y_train[train_idx]
        X_te, y_test_fold = X_train_final[test_idx], y_train[test_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr,
            y_train_fold,
            eval_set=[(X_te, y_test_fold)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        # Accumulate blended predictions across folds
        val_preds_oof += model.predict(X_val_final) / kf.n_splits
        dec_preds_oof += model.predict(X_dec_final) / kf.n_splits
        print(f" -> Fold {fold + 1} processing complete.")

    # 6. Format Submission Templates complying with validation criteria
    df_sub1 = pd.DataFrame(
        {"load_id": df_val["load_id"], "predicted_rate": val_preds_oof}
    )
    df_sub1.to_csv("validation_predictions.csv", index=False)
    print("\n[EXPORT SUCCESS] Generated 'validation_predictions.csv'")

    df_sub2 = df_dec.copy()
    df_sub2["predicted_rate"] = dec_preds_oof
    df_sub2.to_csv(dec_path, index=False)
    print(f"[EXPORT SUCCESS] Populated asset target file: '{dec_path}'")

    print("\n==================================================================")
    print("Ensemble Cross-Validation Pipeline Executed Successfully.")
    print("==================================================================")


if __name__ == "__main__":
    execute_freight_pipeline()
