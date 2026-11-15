"""
Threshold optimization and probability calibration module.
Ensures decision thresholds are aligned with healthcare operational tradeoffs
and probabilities reflect true observed frequencies via CalibratedClassifierCV.
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score

from src.utils.logger import get_logger

logger = get_logger("threshold_optimizer")


@dataclass
class ThresholdOptimizationResult:
    best_threshold: float
    best_f1: float
    best_precision: float
    best_recall: float
    best_specificity: float
    default_f1: float
    default_recall: float
    default_precision: float
    rationale: str
    threshold_sweep_data: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Evaluate precision, recall, specificity, and F1 across a range of thresholds.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 46)

    records = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        records.append({
            "threshold": round(float(t), 3),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        })

    return pd.DataFrame(records)


def optimize_decision_threshold(
    y_val: np.ndarray,
    y_val_prob: np.ndarray,
    target_metric: str = "f1",
    target_recall: float = 0.60,
    output_dir: Optional[Path] = None,
) -> ThresholdOptimizationResult:
    """
    Select optimal classification threshold on validation set.
    Balances identifying potential no-shows against generating false operational alarms.
    """
    sweep_df = sweep_thresholds(y_val, y_val_prob)

    # 1. Performance at default 0.5 threshold
    default_row = sweep_df.iloc[(sweep_df["threshold"] - 0.50).abs().argmin()]
    def_f1 = float(default_row["f1_score"])
    def_rec = float(default_row["recall"])
    def_prec = float(default_row["precision"])

    # 2. Select optimal threshold
    if target_metric == "f1":
        best_idx = sweep_df["f1_score"].idxmax()
        best_row = sweep_df.loc[best_idx]
        rationale = (
            f"Threshold {best_row['threshold']:.2f} maximizes F1-score ({best_row['f1_score']:.3f}), "
            f"providing the optimal mathematical harmonic balance between recall ({best_row['recall']:.1%}) "
            f"and precision ({best_row['precision']:.1%})."
        )
    elif target_metric == "recall_constrained":
        # Find threshold that achieves target recall while maximizing precision
        valid_candidates = sweep_df[sweep_df["recall"] >= target_recall]
        if not valid_candidates.empty:
            best_idx = valid_candidates["precision"].idxmax()
            best_row = valid_candidates.loc[best_idx]
            rationale = (
                f"Selected threshold {best_row['threshold']:.2f} satisfies the operational requirement of "
                f"capturing >= {target_recall:.0%} of no-shows (achieved {best_row['recall']:.1%}) with maximum "
                f"precision ({best_row['precision']:.1%})."
            )
        else:
            best_idx = sweep_df["f1_score"].idxmax()
            best_row = sweep_df.loc[best_idx]
            rationale = f"Threshold {best_row['threshold']:.2f} chosen by maximizing F1 score."
    else:
        best_idx = sweep_df["f1_score"].idxmax()
        best_row = sweep_df.loc[best_idx]
        rationale = f"Threshold {best_row['threshold']:.2f} selected by default F1 optimization."

    result = ThresholdOptimizationResult(
        best_threshold=float(best_row["threshold"]),
        best_f1=float(best_row["f1_score"]),
        best_precision=float(best_row["precision"]),
        best_recall=float(best_row["recall"]),
        best_specificity=float(best_row["specificity"]),
        default_f1=def_f1,
        default_recall=def_rec,
        default_precision=def_prec,
        rationale=rationale,
        threshold_sweep_data=sweep_df.to_dict(orient="records"),
    )

    # Plot threshold curve
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", color="#3498db", linewidth=2)
        plt.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", color="#e74c3c", linewidth=2)
        plt.plot(sweep_df["threshold"], sweep_df["f1_score"], label="F1-Score", color="#2ecc71", linewidth=2.5)
        plt.axvline(result.best_threshold, color="purple", linestyle="--", label=f"Optimal Threshold ({result.best_threshold:.2f})")
        plt.axvline(0.50, color="gray", linestyle=":", label="Default 0.50 Threshold")
        plt.title("Threshold Optimization: Precision, Recall & F1 Tradeoffs")
        plt.xlabel("Decision Threshold")
        plt.ylabel("Metric Score")
        plt.ylim(0, 1.02)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / "threshold_optimization_curve.png", dpi=200)
        plt.close()

        # Save result JSON
        with open(output_dir / "optimal_threshold.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

    logger.info(
        f"Optimal threshold: {result.best_threshold:.2f} | "
        f"F1: {result.best_f1:.3f} (vs default 0.5: {def_f1:.3f}) | "
        f"Recall: {result.best_recall:.1%} | Precision: {result.best_precision:.1%}"
    )
    return result


def calibrate_model_probabilities(
    base_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "sigmoid",
    output_dir: Optional[Path] = None,
) -> Tuple[Any, float, float]:
    """
    Calibrate model probabilities using CalibratedClassifierCV.
    Evaluates reliability before and after calibration using Brier score and calibration curves.
    """
    # Evaluate raw uncalibrated probabilities
    raw_probs = base_model.predict_proba(X_val)[:, 1]
    raw_brier = float(brier_score_loss(y_val, raw_probs))

    # Fit calibrated model on training/validation data
    # cv='prefit' uses already trained base_model and calibrates using X_val, y_val
    calibrator = CalibratedClassifierCV(estimator=base_model, method=method, cv="prefit")
    calibrator.fit(X_val, y_val)

    cal_probs = calibrator.predict_proba(X_val)[:, 1]
    cal_brier = float(brier_score_loss(y_val, cal_probs))

    logger.info(f"Probability Calibration ({method}): Brier score {raw_brier:.4f} -> {cal_brier:.4f}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Generate calibration curve
        prob_true_raw, prob_pred_raw = calibration_curve(y_val, raw_probs, n_bins=10)
        prob_true_cal, prob_pred_cal = calibration_curve(y_val, cal_probs, n_bins=10)

        plt.figure(figsize=(7, 6))
        plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        plt.plot(prob_pred_raw, prob_true_raw, "s-", label=f"Uncalibrated (Brier: {raw_brier:.4f})", color="#e74c3c")
        plt.plot(prob_pred_cal, prob_true_cal, "o-", label=f"Calibrated ({method}) (Brier: {cal_brier:.4f})", color="#2ecc71")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of True No-Shows")
        plt.title("Reliability Diagram (Calibration Curve)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / "calibration_curve.png", dpi=200)
        plt.close()

    # If calibrated brier score improved or remained consistent, return calibrator
    final_model = calibrator if cal_brier <= raw_brier else base_model
    return final_model, raw_brier, cal_brier
