"""
Preprocess the Kaggle breast cancer dataset.

Outputs:
    data/processed/Cancer_Data_processed.csv
    data/processed/Cancer_Data_feature_groups.csv
    data/processed/Cancer_Data_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_feature_groups(feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if col.endswith("_mean"):
            group = "mean"
            base_measure = col[:-5]
        elif col.endswith("_se"):
            group = "se"
            base_measure = col[:-3]
        elif col.endswith("_worst"):
            group = "worst"
            base_measure = col[:-6]
        else:
            group = "other"
            base_measure = col

        rows.append({
            "feature": col,
            "group": group,
            "base_measure": base_measure,
        })
    return pd.DataFrame(rows)


def preprocess(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    if "diagnosis" not in df.columns:
        raise ValueError("Expected a 'diagnosis' column with B/M labels.")

    df["diagnosis_label"] = df["diagnosis"].map({"B": 0, "M": 1})
    if df["diagnosis_label"].isna().any():
        bad = df.loc[df["diagnosis_label"].isna(), "diagnosis"].unique()
        raise ValueError(f"Unexpected diagnosis labels: {bad}")

    feature_cols = [
        c for c in df.columns
        if c not in {"diagnosis", "diagnosis_label"} and pd.api.types.is_numeric_dtype(df[c])
    ]

    cleaned = df[["diagnosis", "diagnosis_label"] + feature_cols].copy()
    feature_groups = build_feature_groups(feature_cols)

    summary = pd.DataFrame({
        "metric": [
            "n_samples",
            "n_features",
            "benign_count",
            "malignant_count",
            "malignant_rate",
            "missing_values_in_X",
        ],
        "value": [
            len(cleaned),
            len(feature_cols),
            int((cleaned["diagnosis_label"] == 0).sum()),
            int((cleaned["diagnosis_label"] == 1).sum()),
            float((cleaned["diagnosis_label"] == 1).mean()),
            int(cleaned[feature_cols].isna().sum().sum()),
        ],
    })

    cleaned.to_csv(output_dir / "Cancer_Data_processed.csv", index=False)
    feature_groups.to_csv(output_dir / "Cancer_Data_feature_groups.csv", index=False)
    summary.to_csv(output_dir / "Cancer_Data_summary.csv", index=False)

    print("Wrote:")
    print(f"  {output_dir / 'Cancer_Data_processed.csv'}")
    print(f"  {output_dir / 'Cancer_Data_feature_groups.csv'}")
    print(f"  {output_dir / 'Cancer_Data_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess breast cancer CSV")
    parser.add_argument("--input", required=True, help="Path to raw Kaggle CSV")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    preprocess(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
