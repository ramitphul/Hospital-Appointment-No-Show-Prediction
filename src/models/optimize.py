"""
Hyperparameter optimization module for hospital no-show models.
Implements RandomizedSearchCV and GridSearchCV with StratifiedKFold
for Random Forest, XGBoost, and Logistic Regression.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import xgboost as xgb

from src.utils.logger import get_logger

logger = get_logger("hyperparameter_tuner")


@dataclass
class TuningResult:
    model_name: str
    best_params: Dict[str, Any]
    best_cv_score: float
    scoring_metric: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_param_distributions(model_type: str) -> Dict[str, Any]:
    """Define hyperparameter search spaces tailored to tabular clinical data."""
    if model_type == "random_forest":
        return {
            "n_estimators": [100, 150, 200],
            "max_depth": [8, 12, 16, None],
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [2, 4, 8],
            "max_features": ["sqrt", "log2", 0.5],
            "class_weight": ["balanced", "balanced_subsample"],
        }
    elif model_type == "xgboost":
        return {
            "n_estimators": [100, 150, 200],
            "learning_rate": [0.03, 0.05, 0.1, 0.15],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "scale_pos_weight": [1.0, 2.5, 3.95],  # 80:20 class imbalance ratio ~ 3.95
        }
    elif model_type == "logistic_regression":
        return {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
            "class_weight": ["balanced", None],
        }
    else:
        raise ValueError(f"Unsupported model type for hyperparameter tuning: {model_type}")


def tune_hyperparameters(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 12,
    cv_folds: int = 4,
    scoring: str = "roc_auc",
    random_state: int = 42,
    output_dir: Optional[Path] = None,
) -> Tuple[Any, TuningResult]:
    """
    Execute RandomizedSearchCV with StratifiedKFold on the training feature matrix.
    Saves tuning results and best parameters.
    """
    logger.info(f"Starting hyperparameter tuning for: {model_type} (n_iter={n_iter}, cv={cv_folds})")

    param_dist = get_param_distributions(model_type)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    if model_type == "random_forest":
        base_estimator = RandomForestClassifier(random_state=random_state, n_jobs=2)
    elif model_type == "xgboost":
        base_estimator = xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=2,
        )
    elif model_type == "logistic_regression":
        base_estimator = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best_estimator = search.best_estimator_
    best_params = search.best_params_
    best_score = float(search.best_score_)

    logger.info(f"Tuning finished for {model_type} | Best {scoring}: {best_score:.4f} | Best Params: {best_params}")

    result = TuningResult(
        model_name=model_type,
        best_params=best_params,
        best_cv_score=round(best_score, 4),
        scoring_metric=scoring,
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"tuning_{model_type}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            # Convert non-serializable values (like numpy/types) to string/python types
            clean_params = {k: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v) for k, v in best_params.items()}
            json.dump({
                "model_name": model_type,
                "best_params": clean_params,
                "best_cv_score": result.best_cv_score,
                "scoring_metric": scoring,
            }, f, indent=2)

    return best_estimator, result
