"""
Export interpretability artifacts for the final report.

Outputs:
    results/tables/logreg_l2_coefficients.csv
    results/tables/random_forest_importance_gini.csv
    results/tables/random_forest_importance_gini_permutation.csv
    results/figures/logreg_l2_coefficients_top.png
    results/figures/random_forest_importance_top.png
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
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from train_breast_cancer_models import RANDOM_SEED, build_models, load_and_clean_data  # noqa: E402


def export_logreg_coefficients(
    pipeline: Pipeline,
    feature_cols: list[str],
    out_csv: Path,
    out_fig: Path,
    top_k: int,
) -> None:
    model = pipeline.named_steps["model"]
    coef = np.ravel(model.coef_)

    df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
        "odds_ratio_per_1sd": np.exp(coef),
    }).sort_values("abs_coefficient", ascending=False)
    df.to_csv(out_csv, index=False)

    top = df.head(top_k).iloc[::-1]
    colors = np.where(top["coefficient"].to_numpy() >= 0, "#c0392b", "#2980b9")

    fig, ax = plt.subplots(figsize=(7, max(3.5, top_k * 0.28)))
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient on standardized features")
    ax.set_title(f"Logistic Regression (L2): Top {top_k} Coefficients")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)


def export_random_forest_importance(
    pipeline: Pipeline,
    feature_cols: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    out_csv: Path,
    out_fig: Path,
    top_k: int,
) -> None:
    forest = pipeline.named_steps["model"]

    df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gini": forest.feature_importances_,
    }).sort_values("importance_gini", ascending=False)
    df.to_csv(out_csv, index=False)

    top = df.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3.5, top_k * 0.28)))
    ax.barh(top["feature"], top["importance_gini"], color="#27ae60")
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Random Forest: Top {top_k} Feature Importances")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    X_transformed = pipeline.named_steps["prep"].transform(X_train)
    perm = permutation_importance(
        forest,
        X_transformed,
        y_train,
        n_repeats=20,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "feature": feature_cols,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)
    perm_df.to_csv(out_csv.with_name(out_csv.stem + "_permutation.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model interpretability artifacts")
    parser.add_argument("--input", required=True, help="Path to processed CSV")
    parser.add_argument("--output_dir", default="results", help="Results root directory")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=15)
    args = parser.parse_args()

    tables_dir = Path(args.output_dir) / "tables"
    figures_dir = Path(args.output_dir) / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_and_clean_data(args.input)
    feature_cols = list(X.columns)

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    models = build_models()
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    logreg_spec = models["logreg_l2"]
    logreg_search = GridSearchCV(
        clone(logreg_spec["estimator"]),
        logreg_spec["param_grid"],
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )
    logreg_search.fit(X_train, y_train)
    pd.Series(logreg_search.best_params_).to_csv(tables_dir / "interpretability_logreg_l2_best_params.csv")
    logreg_best = logreg_search.best_estimator_

    rf_spec = models["random_forest"]
    rf_search = GridSearchCV(
        clone(rf_spec["estimator"]),
        rf_spec["param_grid"],
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )
    rf_search.fit(X_train, y_train)
    pd.Series(rf_search.best_params_).to_csv(tables_dir / "interpretability_random_forest_best_params.csv")
    rf_best = rf_search.best_estimator_

    export_logreg_coefficients(
        logreg_best,
        feature_cols,
        tables_dir / "logreg_l2_coefficients.csv",
        figures_dir / "logreg_l2_coefficients_top.png",
        args.top_k,
    )
    export_random_forest_importance(
        rf_best,
        feature_cols,
        X_train,
        y_train,
        tables_dir / "random_forest_importance_gini.csv",
        figures_dir / "random_forest_importance_top.png",
        args.top_k,
    )

    print("Wrote interpretability outputs to:")
    print(f"  {tables_dir}")
    print(f"  {figures_dir}")


if __name__ == "__main__":
    main()
