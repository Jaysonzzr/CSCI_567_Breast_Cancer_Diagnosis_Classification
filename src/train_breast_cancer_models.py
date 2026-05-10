import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_SEED = 42


def load_and_clean_data(path: str):
    df = pd.read_csv(path)

    if "id" in df.columns:
        df = df.drop(columns=["id"])
    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])

    if "diagnosis_label" not in df.columns:
        if "diagnosis" not in df.columns:
            raise ValueError("Input file must contain either 'diagnosis' or 'diagnosis_label'.")
        df["diagnosis_label"] = df["diagnosis"].map({"B": 0, "M": 1})

    if df["diagnosis_label"].isna().any():
        raise ValueError("diagnosis_label contains missing values after encoding.")

    y = df["diagnosis_label"].astype(int)
    feature_cols = [
        c for c in df.columns
        if c not in {"diagnosis", "diagnosis_label"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols].copy()

    feature_groups = []
    for col in feature_cols:
        if col.endswith("_mean"):
            group = "mean"
        elif col.endswith("_se"):
            group = "se"
        elif col.endswith("_worst"):
            group = "worst"
        else:
            group = "other"
        feature_groups.append({"feature": col, "group": group})

    return X, y, pd.DataFrame(feature_groups)


def build_models():
    scaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    unscaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    scaled_prep = ColumnTransformer(
        [("num", scaled, slice(0, None))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    unscaled_prep = ColumnTransformer(
        [("num", unscaled, slice(0, None))],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return {
        "dummy_most_frequent": {
            "estimator": Pipeline([
                ("prep", unscaled_prep),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]),
            "param_grid": {},
        },
        "logreg_l1": {
            "estimator": Pipeline([
                ("prep", scaled_prep),
                ("model", LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                )),
            ]),
            "param_grid": {"model__C": [0.01, 0.1, 1, 10, 100]},
        },
        "logreg_l2": {
            "estimator": Pipeline([
                ("prep", scaled_prep),
                ("model", LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                )),
            ]),
            "param_grid": {"model__C": [0.01, 0.1, 1, 10, 100]},
        },
        "svm_linear": {
            "estimator": Pipeline([
                ("prep", scaled_prep),
                ("model", SVC(kernel="linear", probability=True, random_state=RANDOM_SEED)),
            ]),
            "param_grid": {"model__C": [0.01, 0.1, 1, 10, 100]},
        },
        "svm_rbf": {
            "estimator": Pipeline([
                ("prep", scaled_prep),
                ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
            ]),
            "param_grid": {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", 0.01, 0.1, 1.0],
            },
        },
        "mlp_small": {
            "estimator": Pipeline([
                ("prep", scaled_prep),
                ("model", MLPClassifier(
                    max_iter=2000,
                    early_stopping=True,
                    random_state=RANDOM_SEED,
                )),
            ]),
            "param_grid": {
                "model__hidden_layer_sizes": [(16,), (32,), (32, 16)],
                "model__alpha": [1e-4, 1e-3, 1e-2],
                "model__learning_rate_init": [1e-3, 1e-2],
            },
        },
        "random_forest": {
            "estimator": Pipeline([
                ("prep", unscaled_prep),
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                )),
            ]),
            "param_grid": {
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        },
    }


def nested_cv_evaluate(X, y, models, outer_splits=5, inner_splits=3, scoring="roc_auc"):
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_SEED)

    summary_rows = []
    param_rows = []
    artifacts = {}

    for model_name, spec in models.items():
        fold_metrics = []
        all_true, all_pred, all_score = [], [], []
        fold_best_params = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            estimator = clone(spec["estimator"])
            param_grid = spec["param_grid"]

            if param_grid:
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=inner_cv,
                    n_jobs=-1,
                    refit=True,
                )
                search.fit(X_train, y_train)
                best_estimator = search.best_estimator_
                best_params = search.best_params_
            else:
                best_estimator = estimator.fit(X_train, y_train)
                best_params = {}

            y_pred = best_estimator.predict(X_test)
            if hasattr(best_estimator, "predict_proba"):
                y_score = best_estimator.predict_proba(X_test)[:, 1]
            elif hasattr(best_estimator, "decision_function"):
                y_score = best_estimator.decision_function(X_test)
            else:
                y_score = y_pred.astype(float)

            fold_metrics.append({
                "fold": fold_idx,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_score),
            })
            all_true.extend(y_test.tolist())
            all_pred.extend(y_pred.tolist())
            all_score.extend(y_score.tolist())
            fold_best_params.append({"model_name": model_name, "fold": fold_idx, **best_params})

        fold_df = pd.DataFrame(fold_metrics)
        summary_rows.append({
            "model_name": model_name,
            "cv_type": f"Stratified{outer_splits}Fold",
            "random_seed": RANDOM_SEED,
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
        param_rows.extend(fold_best_params)
        artifacts[model_name] = {
            "y_true": np.array(all_true),
            "y_pred": np.array(all_pred),
            "y_score": np.array(all_score),
        }

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["roc_auc_mean", "recall_mean", "f1_mean"],
        ascending=False,
    ).reset_index(drop=True)
    params_df = pd.DataFrame(param_rows)
    return summary_df, params_df, artifacts


def plot_confusion(y_true, y_pred, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign (0)", "Malignant (1)"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc(y_true, y_score, out_path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_feature_group_ablation(X, y, models, feature_group_df, out_dir: Path):
    groups = {
        "mean": feature_group_df.loc[feature_group_df["group"] == "mean", "feature"].tolist(),
        "se": feature_group_df.loc[feature_group_df["group"] == "se", "feature"].tolist(),
        "worst": feature_group_df.loc[feature_group_df["group"] == "worst", "feature"].tolist(),
        "all": X.columns.tolist(),
    }
    rows = []
    subset_names = ["logreg_l2", "svm_rbf", "mlp_small"]
    for group_name, cols in groups.items():
        subset_models = {k: models[k] for k in subset_names}
        summary_df, _, _ = nested_cv_evaluate(X[cols], y, subset_models)
        summary_df.insert(0, "feature_group", group_name)
        summary_df.insert(1, "n_features", len(cols))
        rows.append(summary_df)
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "ablation_results.csv", index=False)


def run_learning_curve(X, y, models, out_dir: Path):
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    rng = np.random.RandomState(RANDOM_SEED)
    subset_names = ["logreg_l2", "svm_rbf", "mlp_small"]
    idx = np.arange(len(X))
    y_array = y.to_numpy()

    for frac in fractions:
        selected = []
        for cls in np.unique(y_array):
            cls_idx = idx[y_array == cls]
            k = max(1, int(round(len(cls_idx) * frac)))
            selected.extend(rng.choice(cls_idx, size=k, replace=False).tolist())
        selected = np.array(sorted(set(selected)))
        X_sub, y_sub = X.iloc[selected], y.iloc[selected]
        subset_models = {k: models[k] for k in subset_names}
        summary_df, _, _ = nested_cv_evaluate(X_sub, y_sub, subset_models)
        summary_df.insert(0, "train_fraction", frac)
        summary_df.insert(1, "n_samples", len(X_sub))
        rows.append(summary_df)

    pd.concat(rows, ignore_index=True).to_csv(out_dir / "learning_curve_results.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="CSCI 567 breast cancer baseline pipeline")
    parser.add_argument("--input", required=True, help="Path to raw or processed CSV")
    parser.add_argument("--output_dir", default="results", help="Directory to save outputs")
    parser.add_argument("--run_ablation", action="store_true")
    parser.add_argument("--run_learning_curve", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_group_df = load_and_clean_data(args.input)

    pd.DataFrame({
        "metric": [
            "n_samples",
            "n_features",
            "benign_count",
            "malignant_count",
            "malignant_rate",
            "missing_values_in_X",
        ],
        "value": [
            len(X),
            X.shape[1],
            int((y == 0).sum()),
            int((y == 1).sum()),
            float((y == 1).mean()),
            int(X.isna().sum().sum()),
        ],
    }).to_csv(tables_dir / "dataset_summary.csv", index=False)
    feature_group_df.to_csv(tables_dir / "feature_groups.csv", index=False)

    models = build_models()
    summary_df, params_df, artifacts = nested_cv_evaluate(X, y, models)
    summary_df.to_csv(tables_dir / "main_results.csv", index=False)
    params_df.to_csv(tables_dir / "best_params_by_fold.csv", index=False)

    for model_name, artifact in artifacts.items():
        plot_confusion(
            artifact["y_true"],
            artifact["y_pred"],
            figures_dir / f"{model_name}_confusion_matrix.png",
            f"{model_name} Confusion Matrix",
        )
        plot_roc(
            artifact["y_true"],
            artifact["y_score"],
            figures_dir / f"{model_name}_roc_curve.png",
            f"{model_name} ROC Curve",
        )

    if args.run_ablation:
        run_feature_group_ablation(X, y, models, feature_group_df, tables_dir)
    if args.run_learning_curve:
        run_learning_curve(X, y, models, tables_dir)

    with open(tables_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "input_file": args.input,
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "positive_class": "1 = malignant",
            "best_model_by_roc_auc": summary_df.iloc[0]["model_name"],
        }, f, indent=2)

    print("Finished.")
    print(f"Saved main results to: {tables_dir / 'main_results.csv'}")
    print(f"Saved best params to: {tables_dir / 'best_params_by_fold.csv'}")
    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
