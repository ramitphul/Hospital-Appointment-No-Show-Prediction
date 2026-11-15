"""
Prediction and inference engine for hospital appointment no-show risk.
Loads serialized pipeline artifacts, applies leak-free feature engineering,
and generates calibrated probabilities, risk categories, and explainability tags.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.explainability.shap_explainer import ModelExplainer
from src.features.build_features import engineer_appointment_features
from src.utils.config import get_resolved_path, load_config
from src.utils.logger import get_logger

logger = get_logger("predictor")


@dataclass
class SinglePredictionResult:
    prediction: int
    prediction_label: str
    no_show_probability: float
    attendance_probability: float
    risk_category: str
    threshold_used: float
    model_version: str
    top_contributing_factors: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NoShowPredictor:
    """
    Production inference engine.
    Ensures identical feature engineering and transformation between training and inference.
    """
    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        config = load_config()
        self.models_dir = (
            Path(models_dir) if models_dir else get_resolved_path(config["paths"]["models_dir"])
        )

        self.model_path = self.models_dir / "final_model.joblib"
        self.preprocessor_path = self.models_dir / "preprocessor.joblib"
        self.metadata_path = self.models_dir / "model_metadata.json"

        self.model = None
        self.preprocessor = None
        self.metadata = None
        self.explainer = None
        self.feature_names = []
        self.threshold = 0.50
        self.risk_thresholds = config.get("risk_thresholds", {"low_risk_max": 0.30, "medium_risk_max": 0.60})

        self._load_artifacts()

    def _load_artifacts(self):
        """Load serialized model, preprocessor, metadata, and explainer."""
        if not self.model_path.exists() or not self.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in {self.models_dir}. "
                "Please run scripts/train_pipeline.py first to generate trained models."
            )

        logger.info(f"Loading model artifacts from {self.models_dir}...")
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.threshold = float(self.metadata.get("optimal_threshold", 0.50))
            self.feature_names = self.metadata.get("feature_names", [])
            self.risk_thresholds = self.metadata.get("risk_thresholds", self.risk_thresholds)
        else:
            self.metadata = {"model_version": "1.0.0", "best_algorithm": "Model"}
            self.feature_names = list(self.preprocessor.get_feature_names_out())

        # Initialize explainer for on-demand prediction attribution
        try:
            self.explainer = ModelExplainer(self.model, self.feature_names)
        except Exception as e:
            logger.warning(f"Could not initialize explainer for predictor: {e}")
            self.explainer = None

        logger.info(
            f"Artifacts loaded | Version: {self.metadata.get('model_version')} | "
            f"Algorithm: {self.metadata.get('best_algorithm')} | Decision Threshold: {self.threshold:.3f}"
        )

    def determine_risk_category(self, prob: float) -> str:
        """
        Classify continuous probability into 3 actionable operational tiers:
        - Low Risk (< 30%)
        - Medium Risk (30% to 59.9%)
        - High Risk (>= 60%)
        """
        low_max = self.risk_thresholds.get("low_risk_max", 0.30)
        med_max = self.risk_thresholds.get("medium_risk_max", 0.60)

        if prob < low_max:
            return "Low"
        elif prob < med_max:
            return "Medium"
        else:
            return "High"

    def predict_single(
        self,
        record: Dict[str, Any],
        explain: bool = True,
    ) -> SinglePredictionResult:
        """
        Predict no-show probability and risk classification for a single appointment.
        """
        df_single = pd.DataFrame([record])
        df_processed, X_trans = self._preprocess_input(df_single)

        prob = float(self.model.predict_proba(X_trans)[0, 1])
        prediction = 1 if prob >= self.threshold else 0
        pred_label = "No-show risk" if prediction == 1 else "Likely to attend"
        risk_category = self.determine_risk_category(prob)

        explanation = None
        if explain and self.explainer is not None:
            explanation = self.explainer.explain_instance(X_trans[0])

        result = SinglePredictionResult(
            prediction=prediction,
            prediction_label=pred_label,
            no_show_probability=round(prob, 4),
            attendance_probability=round(1.0 - prob, 4),
            risk_category=risk_category,
            threshold_used=round(self.threshold, 3),
            model_version=str(self.metadata.get("model_version", "1.0.0")),
            top_contributing_factors=explanation,
        )
        return result

    def predict_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Batch prediction for multiple appointment records.
        Returns original dataframe enriched with predictions, probabilities, and risk tiers.
        """
        df_processed, X_trans = self._preprocess_input(df_input)

        probs = self.model.predict_proba(X_trans)[:, 1]
        preds = (probs >= self.threshold).astype(int)
        labels = np.where(preds == 1, "No-show risk", "Likely to attend")
        risk_cats = [self.determine_risk_category(p) for p in probs]

        results_df = df_input.copy()
        results_df["predicted_no_show"] = preds
        results_df["prediction_label"] = labels
        results_df["no_show_probability"] = np.round(probs, 4)
        results_df["attendance_probability"] = np.round(1.0 - probs, 4)
        results_df["risk_category"] = risk_cats

        return results_df

    def _preprocess_input(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply identical feature engineering and column transformations.
        """
        data = df.copy()

        # Fill default patient history if absent in incoming raw payload
        if "prior_appointments" not in data.columns:
            data["prior_appointments"] = 0
        if "prior_no_shows" not in data.columns:
            data["prior_no_shows"] = 0
        if "prior_no_show_rate" not in data.columns:
            denom = data["prior_appointments"].replace(0, np.nan)
            data["prior_no_show_rate"] = (data["prior_no_shows"] / denom).fillna(0.0)

        # Standardize health conditions
        for c in ["Scholarship", "Hipertension", "Diabetes", "Alcoholism"]:
            if c not in data.columns:
                data[c] = 0
            else:
                data[c] = (data[c] > 0).astype(int)

        if "Handcap" not in data.columns:
            data["Handcap"] = 0
        else:
            data["Handcap"] = (data["Handcap"] > 0).astype(int)

        if "Gender" not in data.columns:
            data["Gender"] = "F"
        if "Neighbourhood" not in data.columns:
            data["Neighbourhood"] = "CENTRO"

        # Apply feature engineering if raw datetime columns are present
        scenario = self.metadata.get("scenario", "booking") if self.metadata else "booking"
        if "ScheduledDay" in data.columns and "AppointmentDay" in data.columns:
            feat_df = engineer_appointment_features(data, scenario=scenario, target_col=None)
        else:
            feat_df = data.copy()

        # Transform using fitted preprocessor
        X_trans = self.preprocessor.transform(feat_df)
        return feat_df, X_trans
