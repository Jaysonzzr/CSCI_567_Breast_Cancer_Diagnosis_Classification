"""
This file creates out-of-fold predictions using the same model definitions as
the main pipeline, then summarizes false positives and false negatives.

Outputs:
    results/tables/error_analysis_summary.csv
    results/tables/error_cases_<model>.csv
    results/tables/error_profile_<model>.csv
    results/figures/error_counts_by_model.png
    results/figures/error_profile_<model>_fn_vs_tp.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oof_utils import get_oof_predictions, summarize_predictions
from train_breast_cancer_models import load_and_clean_data


def label_error_cases(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.copy()
    pred_df["error_type"] = np.select(
        [
            (pred_df["y_true"] == 1) & (pred_df["y_pred"] == 1),
            (pred_df["y_true"] == 0) & (pred_df["y_pred"] == 0),
            (pred_df["y_true"] == 0) & (pred_df["y_pred"] == 1),
            (pred_df["y_true"] == 1) & (pred_df["y_pred"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="UNKNOWN",
    )
    return pred_df


def export_error_profile(
    X: pd.DataFrame,
    pred_df: pd.DataFrame,
    model_name: str,
    tables_dir: Path,
    figures_dir: Path,
    top_k: int = 12,
) -> None:
    merged = pred_df[["row_index", "error_type", "y_true", "y_score"]].merge(
        X.reset_index().rename(columns={"index": "row_index"}),
        on="row_index",
        how="left",
    )

    malignant = merged[merged["y_true"] == 1].copy()
    if "FN" not in set(malignant["error_type"]) or "TP" not in set(malignant["error_type"]):
        return

    feature_cols = list(X.columns)
    scaler = StandardScaler()
    z = pd.DataFrame(
        scaler.fit_transform(merged[feature_cols]),
        columns=feature_cols,
        index=merged.index,
    )
    z["error_type"] = merged["error_type"].values
    z["y_true"] = merged["y_true"].values

    tp_mean = z[(z["y_true"] == 1) & (z["error_type"] == "TP")][feature_cols].mean()
    fn_mean = z[(z["y_true"] == 1) & (z["error_type"] == "FN")][feature_cols].mean()

    profile = pd.DataFrame({
        "feature": feature_cols,
        "fn_mean_z": fn_mean.values,
        "tp_mean_z": tp_mean.values,
        "fn_minus_tp_z": (fn_mean - tp_mean).values,
        "abs_difference": np.abs((fn_mean - tp_mean).values),
    }).sort_values("abs_difference", ascending=False)

    profile.to_csv(tables_dir / f"error_profile_{model_name}.csv", index=False)

    top = profile.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3.5, top_k * 0.28)))
    colors = np.where(top["fn_minus_tp_z"] >= 0, "#c0392b", "#2980b9")
    ax.barh(top["feature"], top["fn_minus_tp_z"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized mean difference: false negatives - true positives")
    ax.set_title(f"Error Profile for {model_name}: False Negatives vs True Positives")
    fig.tight_layout()
    fig.savefig(figures_dir / f"error_profile_{model_name}_fn_vs_tp.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export error analysis")
    parser.add_argument("--input", required=True, help="Path to processed CSV")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg_l2", "svm_rbf", "mlp_small", "random_forest"],
        help="Model names from train_breast_cancer_models.py",
    )
    parser.add_argument("--profile_model", default="logreg_l2", help="Model for detailed FN-vs-TP profile")
    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_and_clean_data(args.input)
    outputs = get_oof_predictions(X, y, args.models)

    summary_rows = []
    count_rows = []

    for model_name, pred_df in outputs.items():
        pred_df = label_error_cases(pred_df)

        case_df = pred_df.merge(
            X.reset_index().rename(columns={"index": "row_index"}),
            on="row_index",
            how="left",
        )
        case_df.to_csv(tables_dir / f"error_cases_{model_name}.csv", index=False)

        summary = summarize_predictions(pred_df, threshold=0.5)
        summary["model_name"] = model_name
        summary_rows.append(summary)

        counts = pred_df["error_type"].value_counts().to_dict()
        for error_type in ["TP", "TN", "FP", "FN"]:
            count_rows.append({
                "model_name": model_name,
                "error_type": error_type,
                "count": int(counts.get(error_type, 0)),
            })

        if model_name == args.profile_model:
            export_error_profile(X, pred_df, model_name, tables_dir, figures_dir)

    summary_df = pd.DataFrame(summary_rows)
    ordered_cols = [
        "model_name", "accuracy", "precision", "recall", "f1", "roc_auc",
        "specificity", "tp", "tn", "fp", "fn", "threshold",
    ]
    summary_df = summary_df[ordered_cols]
    summary_df.to_csv(tables_dir / "error_analysis_summary.csv", index=False)

    counts_df = pd.DataFrame(count_rows)
    counts_df.to_csv(tables_dir / "error_counts_by_model.csv", index=False)

    pivot = counts_df.pivot(index="model_name", columns="error_type", values="count").fillna(0)
    pivot[["FP", "FN"]].plot(kind="bar", figsize=(8, 4.8))
    plt.ylabel("Count")
    plt.title("False Positives and False Negatives by Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_counts_by_model.png", dpi=200)
    plt.close()

    print("Wrote error analysis outputs to:")
    print(f"  {tables_dir}")
    print(f"  {figures_dir}")


if __name__ == "__main__":
    main()
