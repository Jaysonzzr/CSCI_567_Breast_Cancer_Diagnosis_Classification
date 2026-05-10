"""
Export a small PCA analysis for the final report.
1. A 2D PCA visualization of benign vs malignant cases.
2. A PCA components sweep for Logistic Regression and RBF SVM.

Outputs:
    results/tables/pca_component_results.csv
    results/figures/pca_2d_visualization.png
    results/figures/pca_component_sweep_recall.png
    results/figures/pca_component_sweep_roc_auc.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from train_breast_cancer_models import RANDOM_SEED, load_and_clean_data


def plot_pca_2d(X, y, figures_dir: Path) -> None:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2, random_state=RANDOM_SEED)),
    ])
    X_pca = pipe.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for label, name in [(0, "Benign"), (1, "Malignant")]:
        mask = y.to_numpy() == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.75, label=name, s=28)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("2D PCA Visualization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "pca_2d_visualization.png", dpi=200)
    plt.close(fig)


def evaluate_pca_components(X, y, tables_dir: Path, figures_dir: Path) -> None:
    component_grid = [2, 5, 10, 15, 20, 30]
    models = {
        "pca_logreg_l2": LogisticRegression(
            penalty="l2",
            solver="liblinear",
            C=0.1,
            max_iter=5000,
            random_state=RANDOM_SEED,
        ),
        "pca_svm_rbf": SVC(
            kernel="rbf",
            C=10,
            gamma=0.01,
            probability=True,
            random_state=RANDOM_SEED,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rows = []

    for n_components in component_grid:
        for model_name, estimator in models.items():
            fold_rows = []
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=n_components, random_state=RANDOM_SEED)),
                    ("model", estimator),
                ])
                pipe.fit(X_train, y_train)

                y_pred = pipe.predict(X_test)
                y_score = pipe.predict_proba(X_test)[:, 1]

                fold_rows.append({
                    "fold": fold,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                    "roc_auc": roc_auc_score(y_test, y_score),
                })

            fold_df = pd.DataFrame(fold_rows)
            rows.append({
                "model_name": model_name,
                "n_components": n_components,
                "accuracy_mean": fold_df["accuracy"].mean(),
                "accuracy_std": fold_df["accuracy"].std(ddof=1),
                "precision_mean": fold_df["precision"].mean(),
                "precision_std": fold_df["precision"].std(ddof=1),
                "recall_mean": fold_df["recall"].mean(),
                "recall_std": fold_df["recall"].std(ddof=1),
                "f1_mean": fold_df["f1"].mean(),
                "f1_std": fold_df["f1"].std(ddof=1),
                "roc_auc_mean": fold_df["roc_auc"].mean(),
                "roc_auc_std": fold_df["roc_auc"].std(ddof=1),
            })

    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "pca_component_results.csv", index=False)

    for metric, ylabel, fname in [
        ("recall_mean", "Recall", "pca_component_sweep_recall.png"),
        ("roc_auc_mean", "ROC-AUC", "pca_component_sweep_roc_auc.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model_name in out["model_name"].unique():
            sub = out[out["model_name"] == model_name]
            ax.plot(sub["n_components"], sub[metric], marker="o", label=model_name)
        ax.set_xlabel("Number of PCA components")
        ax.set_ylabel(ylabel)
        ax.set_title(f"PCA Component Sweep: {ylabel}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / fname, dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PCA analysis")
    parser.add_argument("--input", required=True, help="Path to processed CSV")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_and_clean_data(args.input)

    plot_pca_2d(X, y, figures_dir)
    evaluate_pca_components(X, y, tables_dir, figures_dir)

    print("Wrote PCA outputs to:")
    print(f"  {tables_dir / 'pca_component_results.csv'}")
    print(f"  {figures_dir}")


if __name__ == "__main__":
    main()
