"""Spotter Freight Challenge - Exploratory Data Analysis & Schema Verification.

This module provides structural data verification tools to audit dataset features,
verify data types, and validate shape integrity across local development spaces.
"""

import sys
from pathlib import Path

import pandas as pd


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

    for name in candidates:
        for path in [Path(name), Path(".vscode") / name, Path("data") / name]:
            if path.is_file():
                return path

    print(f"CRITICAL ERROR: Matrix source '{filename}' could not be resolved.")
    sys.exit(1)


def inspect_development_schemas() -> None:
    """Loads operational CSV matrices and audits internal features and shapes."""
    print("==================================================================")
    print("Executing Structural Dataset Verification Protocol...")
    print("==================================================================\n")

    train_path = locate_exact_file("train_test")
    val_path = locate_exact_file("validation.csv")

    df_train = pd.read_csv(train_path, nrows=5)
    df_val = pd.read_csv(val_path, nrows=5)

    print("--- Training Data Frame Dimensions ---")
    print(f"Detected Attributes: {list(df_train.columns)}")
    print(f"Total Structural Feature Count: {len(df_train.columns)}\n")

    print("--- Inference Validation Data Frame Dimensions ---")
    print(f"Detected Attributes: {list(df_val.columns)}")
    print(f"Total Structural Feature Count: {len(df_val.columns)}\n")

    target_fields = list(set(df_train.columns) - set(df_val.columns))
    print("--- Isolated Dependent Target Framework ---")
    print(f"Target Column isolated: {target_fields}")
    print("==================================================================")


if __name__ == "__main__":
    inspect_development_schemas()
