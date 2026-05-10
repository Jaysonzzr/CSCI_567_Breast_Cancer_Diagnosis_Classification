"""
This file uses out-of-fold probability scores and evaluates multiple classification thresholds.

Outputs:
    results/tables/threshold_analysis.csv
    results/figures/threshold_precision_recall_<model>.png
    results/figures/threshold_error_counts_<model>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oof_utils import get_oof_predictions, summarize_predictions
from train_breast_cancer_models import load_and_clean_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Export threshold analysis")
    parser.add_argument("--input", required=True, help="Path to processed CSV")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--models", nargs="+", default=["logreg_l2", "svm_rbf"])
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_and_clean_data(args.input)
    outputs = get_oof_predictions(X, y, args.models)

    rows = []
    for model_name, pred_df in outputs.items():
        for threshold in args.thresholds:
            summary = summarize_predictions(pred_df, threshold=threshold)
            summary["model_name"] = model_name
            rows.append(summary)

    threshold_df = pd.DataFrame(rows)
    threshold_df = threshold_df[
        [
            "model_name", "threshold", "accuracy", "precision", "recall", "f1",
            "roc_auc", "specificity", "tp", "tn", "fp", "fn",
        ]
    ]
    threshold_df.to_csv(tables_dir / "threshold_analysis.csv", index=False)

    for model_name in args.models:
        sub = threshold_df[threshold_df["model_name"] == model_name].copy()

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(sub["threshold"], sub["precision"], marker="o", label="precision")
        ax.plot(sub["threshold"], sub["recall"], marker="o", label="recall")
        ax.plot(sub["threshold"], sub["f1"], marker="o", label="F1")
        ax.set_xlabel("Decision threshold for malignant class")
        ax.set_ylabel("Metric value")
        ax.set_title(f"Threshold Tradeoff: {model_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"threshold_precision_recall_{model_name}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(sub["threshold"], sub["fn"], marker="o", label="false negatives")
        ax.plot(sub["threshold"], sub["fp"], marker="o", label="false positives")
        ax.set_xlabel("Decision threshold for malignant class")
        ax.set_ylabel("Count")
        ax.set_title(f"Threshold Error Counts: {model_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"threshold_error_counts_{model_name}.png", dpi=200)
        plt.close(fig)

    print("Wrote threshold analysis outputs to:")
    print(f"  {tables_dir / 'threshold_analysis.csv'}")
    print(f"  {figures_dir}")


if __name__ == "__main__":
    main()
