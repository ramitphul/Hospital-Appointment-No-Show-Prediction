"""
Evaluation and verification script for trained hospital no-show models.
Loads serialized pipeline and evaluates metrics across multiple thresholds.

Usage:
    python scripts/evaluate_model.py
"""
import json
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.models.evaluate import compute_classification_metrics
from src.models.predict import NoShowPredictor
from src.utils.config import get_resolved_path, load_config
from src.utils.logger import get_logger

logger = get_logger("evaluate_model_script")


def main():
    config = load_config()
    test_data_path = get_resolved_path("data/processed/test_set.csv")

    if not test_data_path.exists():
        print(f"Test set not found at {test_data_path}. Please execute scripts/train_pipeline.py first.")
        sys.exit(1)

    print("\nLoading trained predictor and test dataset...")
    predictor = NoShowPredictor()
    test_df = pd.read_csv(test_data_path)

    target_col = config["features"]["target_col"]
    y_true = test_df[target_col].to_numpy()
    X_raw = test_df.drop(columns=[target_col])

    # Run batch prediction
    results_df = predictor.predict_batch(X_raw)
    y_prob = results_df["no_show_probability"].to_numpy()

    # Evaluate at default 0.50 vs optimal threshold
    m_default = compute_classification_metrics(y_true, y_prob, threshold=0.50, model_name="Default (0.50)")
    m_optimal = compute_classification_metrics(y_true, y_prob, threshold=predictor.threshold, model_name=f"Optimal ({predictor.threshold:.2f})")

    print("\n" + "=" * 65)
    print("         MODEL PERFORMANCE COMPARISON ON HELD-OUT TEST SET       ")
    print("=" * 65)
    print(f"{'Metric':<25} | {'Default (0.50)':<15} | {'Optimal (' + str(round(predictor.threshold, 2)) + ')':<15}")
    print("-" * 65)
    print(f"{'ROC-AUC Score':<25} | {m_default.roc_auc:<15.4f} | {m_optimal.roc_auc:<15.4f}")
    print(f"{'PR-AUC Score':<25} | {m_default.pr_auc:<15.4f} | {m_optimal.pr_auc:<15.4f}")
    print(f"{'Recall (Sensitivity)':<25} | {m_default.recall:<15.4f} | {m_optimal.recall:<15.4f}")
    print(f"{'Precision':<25} | {m_default.precision:<15.4f} | {m_optimal.precision:<15.4f}")
    print(f"{'Specificity':<25} | {m_default.specificity:<15.4f} | {m_optimal.specificity:<15.4f}")
    print(f"{'F1-Score':<25} | {m_default.f1_score:<15.4f} | {m_optimal.f1_score:<15.4f}")
    print(f"{'Accuracy':<25} | {m_default.accuracy:<15.4f} | {m_optimal.accuracy:<15.4f}")
    print(f"{'Brier Score':<25} | {m_default.brier_score:<15.4f} | {m_optimal.brier_score:<15.4f}")
    print("-" * 65)
    print(f"True Positives:           {m_optimal.true_positives:,} no-shows detected")
    print(f"False Negatives:          {m_optimal.false_negatives:,} no-shows missed")
    print(f"False Positives:          {m_optimal.false_positives:,} false alarms")
    print(f"True Negatives:           {m_optimal.true_negatives:,} correct attendances")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
