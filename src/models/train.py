"""
Master model training, comparison, hyperparameter optimization,
probability calibration, and serialization pipeline.
"""
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from src.explainability.shap_explainer import generate_global_shap_explanations
from src.models.evaluate import (
    compare_models,
    evaluate_classifier,
    plot_comparative_pr_curves,
    plot_comparative_roc_curves,
    plot_confusion_matrix,
)
from src.models.optimize import tune_hyperparameters
from src.models.threshold import calibrate_model_probabilities, optimize_decision_threshold
from src.utils.config import get_resolved_path, load_config
from src.utils.logger import get_logger

logger = get_logger("model_trainer")


def initialize_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiate the 6 core classification algorithms configured for tabular clinical prediction.
    Balanced class weighting is applied to address the ~80:20 attendance imbalance.
    """
    models = {
        "Logistic_Regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=2,
        ),
        "K_Nearest_Neighbors": KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            n_jobs=2,
        ),
        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_state,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=3.95,  # Ratio of attended (79.8%) to no-shows (20.2%)
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=2,
        ),
    }
    return models


def train_single_model(model_name: str, model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Train an individual classifier."""
    logger.info(f"Training {model_name} on {X_train.shape[0]:,} samples ({X_train.shape[1]} features)...")
    if model_name == "K_Nearest_Neighbors" and len(X_train) > 20000:
        # Fit KNN on representative stratified subsample for performance efficiency
        idx = np.random.choice(len(X_train), size=20000, replace=False)
        model.fit(X_train[idx], y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx])
    else:
        model.fit(X_train, y_train)
    return model


def train_and_compare_models(
    X_train_transformed: np.ndarray,
    y_train: np.ndarray,
    X_val_transformed: np.ndarray,
    y_val: np.ndarray,
    X_test_transformed: np.ndarray,
    y_test: np.ndarray,
    metrics_dir: Path,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Train and evaluate all 6 classification models.
    Generates comparison tables, confusion matrices, and ROC/PR curve plots.
    Returns:
        (comparison_df, trained_models_dict, best_model_name)
    """
    candidate_models = initialize_candidate_models(random_state=random_state)
    trained_models = {}
    eval_results = []
    model_val_probs = {}

    for name, model in candidate_models.items():
        trained_model = train_single_model(name, model, X_train_transformed, y_train)
        trained_models[name] = trained_model

        # Evaluate on validation set
        metrics, probs = evaluate_classifier(
            trained_model,
            X_val_transformed,
            y_val,
            threshold=0.50,
            model_name=name,
        )
        eval_results.append(metrics)
        model_val_probs[name] = probs

        # Save individual confusion matrix
        plot_confusion_matrix(metrics.confusion_matrix, name, metrics_dir / f"cm_{name.lower()}.png")

    # Generate comparative plots
    plot_comparative_roc_curves(model_val_probs, y_val, metrics_dir / "roc_curves_comparison.png")
    plot_comparative_pr_curves(model_val_probs, y_val, metrics_dir / "pr_curves_comparison.png")

    comparison_df = compare_models(eval_results, metrics_dir)
    best_model_name = comparison_df.iloc[0]["Model"]
    logger.info(f"Model comparison complete. Top ranked model by ROC-AUC: {best_model_name}")

    return comparison_df, trained_models, best_model_name


def run_full_training_pipeline(
    df: pd.DataFrame,
    scenario: str = "booking",
    tune_top_model: bool = True,
    n_tune_iter: int = 10,
) -> Dict[str, Any]:
    """
    Execute complete end-to-end ML pipeline:
    1. Preprocessing & Split
    2. Multi-model Training & Comparison
    3. Hyperparameter Tuning
    4. Probability Calibration
    5. Threshold Optimization
    6. SHAP Explainability
    7. Artifact Serialization
    """
    config = load_config()
    models_dir = get_resolved_path(config["paths"]["models_dir"])
    metrics_dir = get_resolved_path(config["paths"]["metrics_dir"])
    shap_dir = get_resolved_path(config["paths"]["shap_dir"])

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    shap_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare data splits
    from src.data.preprocess import prepare_train_test_data
    from sklearn.model_selection import train_test_split

    X_train_full, X_test, y_train_full, y_test, preprocessor, feature_names = prepare_train_test_data(
        df,
        scenario=scenario,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    # Further split training data into Train and Validation for threshold tuning and calibration
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=config["split"]["val_size"],
        stratify=y_train_full,
        random_state=config["split"]["random_state"],
    )

    # Transform features with preprocessor (fitted ONLY on X_train_full)
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # Save processed datasets for reproducible experiments
    X_test.assign(**{config["features"]["target_col"]: y_test}).to_csv(
        get_resolved_path("data/processed/test_set.csv"), index=False
    )

    # 2. Train and compare 6 candidate models
    comparison_df, trained_models, best_model_name = train_and_compare_models(
        X_train_transformed=X_train_trans,
        y_train=y_train,
        X_val_transformed=X_val_trans,
        y_val=y_val,
        X_test_transformed=X_test_trans,
        y_test=y_test,
        metrics_dir=metrics_dir,
        random_state=config["split"]["random_state"],
    )

    best_base_model = trained_models[best_model_name]

    # 3. Hyperparameter Tuning for Top Model
    if tune_top_model and best_model_name in ["Random_Forest", "XGBoost", "Logistic_Regression"]:
        tune_type_map = {
            "Random_Forest": "random_forest",
            "XGBoost": "xgboost",
            "Logistic_Regression": "logistic_regression",
        }
        tuned_model, tune_result = tune_hyperparameters(
            model_type=tune_type_map[best_model_name],
            X_train=X_train_trans,
            y_train=y_train,
            n_iter=n_tune_iter,
            cv_folds=3,
            scoring="roc_auc",
            random_state=config["split"]["random_state"],
            output_dir=metrics_dir,
        )
        # Check if tuned model outperforms base model on validation set
        val_m_base, _ = evaluate_classifier(best_base_model, X_val_trans, y_val, model_name="Base")
        val_m_tuned, _ = evaluate_classifier(tuned_model, X_val_trans, y_val, model_name="Tuned")
        logger.info(f"Validation AUC: Base={val_m_base.roc_auc:.4f} vs Tuned={val_m_tuned.roc_auc:.4f}")
        if val_m_tuned.roc_auc >= val_m_base.roc_auc:
            selected_model = tuned_model
            logger.info("Retained tuned model as optimal estimator.")
        else:
            selected_model = best_base_model
            logger.info("Base model retained (validation AUC was higher/equal).")
    else:
        selected_model = best_base_model

    # 4. Probability Calibration
    calibrated_model, raw_brier, cal_brier = calibrate_model_probabilities(
        selected_model,
        X_train=X_train_trans,
        y_train=y_train,
        X_val=X_val_trans,
        y_val=y_val,
        method="sigmoid",
        output_dir=metrics_dir,
    )

    # 5. Threshold Optimization on Validation Set
    val_probs = calibrated_model.predict_proba(X_val_trans)[:, 1]
    threshold_result = optimize_decision_threshold(
        y_val=y_val.to_numpy(),
        y_val_prob=val_probs,
        target_metric="f1",
        output_dir=metrics_dir,
    )
    optimal_threshold = threshold_result.best_threshold

    # 6. Final Evaluation on Held-Out Test Set (Unseen during threshold & training)
    final_test_metrics, test_probs = evaluate_classifier(
        calibrated_model,
        X_test_trans,
        y_test.to_numpy(),
        threshold=optimal_threshold,
        model_name=f"{best_model_name}_Optimized",
    )
    with open(metrics_dir / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_test_metrics.to_dict(), f, indent=2)

    plot_confusion_matrix(
        final_test_metrics.confusion_matrix,
        f"{best_model_name} (Threshold {optimal_threshold:.2f})",
        metrics_dir / "final_test_confusion_matrix.png",
    )

    logger.info(
        f"FINAL HELD-OUT TEST PERFORMANCE | Accuracy: {final_test_metrics.accuracy:.4f} | "
        f"ROC-AUC: {final_test_metrics.roc_auc:.4f} | PR-AUC: {final_test_metrics.pr_auc:.4f} | "
        f"Recall: {final_test_metrics.recall:.4f} | Precision: {final_test_metrics.precision:.4f} | "
        f"F1-Score: {final_test_metrics.f1_score:.4f}"
    )

    # 7. SHAP Explainability
    logger.info("Computing global SHAP explainability on test sample...")
    explainer = generate_global_shap_explanations(
        calibrated_model,
        X_test_trans,
        feature_names,
        output_dir=shap_dir,
        sample_size=300,
    )

    # 8. Serialization of Model Artifacts
    model_artifact_path = models_dir / "final_model.joblib"
    preprocessor_artifact_path = models_dir / "preprocessor.joblib"

    joblib.dump(calibrated_model, model_artifact_path)
    joblib.dump(preprocessor, preprocessor_artifact_path)

    metadata = {
        "model_version": config["project"]["version"],
        "project_name": config["project"]["name"],
        "trained_timestamp": datetime.utcnow().isoformat() + "Z",
        "best_algorithm": best_model_name,
        "scenario": scenario,
        "optimal_threshold": optimal_threshold,
        "default_threshold": 0.50,
        "raw_brier_score": raw_brier,
        "calibrated_brier_score": cal_brier,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "categorical_cols": list(config["features"]["categorical_cols"]),
        "numeric_cols": list(config["features"]["numeric_cols"]),
        "binary_cols": list(config["features"]["binary_cols"]),
        "test_metrics": final_test_metrics.to_dict(),
        "risk_thresholds": config["risk_thresholds"],
    }

    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Artifacts successfully saved in {models_dir}")
    return metadata
