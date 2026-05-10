"""
Generate compact summary figures from experiment CSVs.

Outputs under results/figures:
    model_recall_comparison.png
    model_rocauc_comparison.png
    ablation_recall.png
    learning_curve_recall.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(df[x], df[y])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export summary figures")
    parser.add_argument("--results_dir", default="results", help="Root results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    main_results = pd.read_csv(tables_dir / "main_results.csv")
    plot_df = main_results[main_results["model_name"] != "dummy_most_frequent"].copy()

    save_bar(
        plot_df,
        "model_name",
        "recall_mean",
        "Malignant Recall by Model",
        "Recall",
        figures_dir / "model_recall_comparison.png",
    )
    save_bar(
        plot_df,
        "model_name",
        "roc_auc_mean",
        "ROC-AUC by Model",
        "ROC-AUC",
        figures_dir / "model_rocauc_comparison.png",
    )

    ablation_path = tables_dir / "ablation_results.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for model_name in ["logreg_l2", "svm_rbf", "mlp_small"]:
            sub = ablation[ablation["model_name"] == model_name]
            if len(sub) > 0:
                ax.plot(sub["feature_group"], sub["recall_mean"], marker="o", label=model_name)
        ax.set_ylabel("Recall")
        ax.set_title("Feature-Group Ablation: Recall by Feature Group")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "ablation_recall.png", dpi=200)
        plt.close(fig)

    learning_path = tables_dir / "learning_curve_results.csv"
    if learning_path.exists():
        learning = pd.read_csv(learning_path)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for model_name in ["logreg_l2", "svm_rbf", "mlp_small"]:
            sub = learning[learning["model_name"] == model_name]
            if len(sub) > 0:
                ax.plot(sub["train_fraction"], sub["recall_mean"], marker="o", label=model_name)
        ax.set_xlabel("Training Fraction")
        ax.set_ylabel("Recall")
        ax.set_title("Learning Curve: Recall vs Training Fraction")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "learning_curve_recall.png", dpi=200)
        plt.close(fig)

    print(f"Saved summary figures to {figures_dir}")


if __name__ == "__main__":
    main()
