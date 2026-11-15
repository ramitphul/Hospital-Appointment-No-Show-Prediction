"""
Model evaluation module for hospital no-show prediction.
Computes comprehensive clinical and operational metrics, confusion matrices,
ROC curves, PR curves, and comparison leaderboards.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger("model_evaluator")


@dataclass
class EvaluationMetrics:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    threshold: float
    confusion_matrix: List[List[int]]
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    model_name: str = "Model",
) -> EvaluationMetrics:
    """
    Calculate full suite of clinical and statistical classification metrics.
    Calculates Specificity (True Negative Rate) alongside Precision, Recall, and AUC.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix elements: tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    brier = float(brier_score_loss(y_true, y_prob))

    metrics = EvaluationMetrics(
        model_name=model_name,
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        specificity=round(spec, 4),
        f1_score=round(f1, 4),
        roc_auc=round(roc_auc, 4),
        pr_auc=round(pr_auc, 4),
        brier_score=round(brier, 4),
        threshold=round(threshold, 3),
        confusion_matrix=cm.tolist(),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
    )
    return metrics


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.50,
    model_name: str = "Classifier",
) -> Tuple[EvaluationMetrics, np.ndarray]:
    """Evaluate a trained model and return structured metrics and predicted probabilities."""
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_vals = model.decision_function(X_test)
        # Sigmoid calibration of decision function
        y_prob = 1.0 / (1.0 + np.exp(-df_vals))
    else:
        y_pred = model.predict(X_test)
        y_prob = y_pred.astype(float)

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_prob=y_prob,
        threshold=threshold,
        model_name=model_name,
    )
    return metrics, y_prob


def plot_confusion_matrix(cm: List[List[int]], model_name: str, output_path: Path):
    """Plot and save publication-quality confusion matrix."""
    import seaborn as sns
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    cm_array = np.array(cm)
    sns.heatmap(
        cm_array,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Attended (0)", "No-Show (1)"],
        yticklabels=["Attended (0)", "No-Show (1)"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Outcome")
    plt.ylabel("Actual Outcome")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_comparative_roc_curves(
    model_probs: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: Path,
):
    """Plot overlaid ROC curves for all evaluated models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))

    for name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_val = roc_auc_score(y_true, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", alpha=0.7)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_comparative_pr_curves(
    model_probs: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: Path,
):
    """Plot overlaid Precision-Recall curves for all evaluated models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))

    baseline = float(np.mean(y_true))
    plt.axhline(baseline, color="gray", linestyle="--", label=f"No-Skill Baseline ({baseline:.3f})")

    for name, probs in model_probs.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        pr_auc_val = average_precision_score(y_true, probs)
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc_val:.3f})", linewidth=2)

    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.title("Precision-Recall (PR) Curve Comparison")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compare_models(
    results: List[EvaluationMetrics],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Generate model comparison leaderboard table and save as CSV and JSON.
    Ranks models by ROC-AUC and F1-score.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in results:
        rows.append({
            "Model": m.model_name,
            "Accuracy": m.accuracy,
            "Precision": m.precision,
            "Recall": m.recall,
            "Specificity": m.specificity,
            "F1-Score": m.f1_score,
            "ROC-AUC": m.roc_auc,
            "PR-AUC": m.pr_auc,
            "Brier Score": m.brier_score,
            "Threshold": m.threshold,
        })

    df = pd.DataFrame(rows).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)

    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)

    json_path = output_dir / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([m.to_dict() for m in results], f, indent=2)

    logger.info(f"Model comparison saved to {csv_path} and {json_path}")
    return df
