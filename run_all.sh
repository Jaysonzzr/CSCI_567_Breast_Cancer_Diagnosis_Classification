#!/usr/bin/env bash
set -e

python src/preprocess_dataset.py \
  --input data/raw/Cancer_Data.csv \
  --output_dir data/processed

python src/train_breast_cancer_models.py \
  --input data/processed/Cancer_Data_processed.csv \
  --output_dir results \
  --run_ablation \
  --run_learning_curve

python src/export_summary_figures.py \
  --results_dir results

python src/export_interpretability.py \
  --input data/processed/Cancer_Data_processed.csv \
  --output_dir results

python src/export_error_analysis.py \
  --input data/processed/Cancer_Data_processed.csv \
  --output_dir results \
  --models logreg_l2 svm_rbf mlp_small random_forest \
  --profile_model logreg_l2

python src/export_threshold_analysis.py \
  --input data/processed/Cancer_Data_processed.csv \
  --output_dir results \
  --models logreg_l2 svm_rbf

python src/export_pca_analysis.py \
  --input data/processed/Cancer_Data_processed.csv \
  --output_dir results
