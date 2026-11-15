"""
Prediction and explainability endpoints for FastAPI backend.
"""
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
import pandas as pd

from api.schemas import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplainResponse,
    PatientAppointmentInput,
    PredictionResponse,
)
from src.models.predict import NoShowPredictor
from src.utils.logger import get_logger

logger = get_logger("prediction_routes")
router = APIRouter(tags=["Predictions & Inference"])

# Global predictor instance (loaded during lifespan/first request)
_PREDICTOR: Any = None


def get_predictor() -> NoShowPredictor:
    """Retrieve or initialize singleton predictor."""
    global _PREDICTOR
    if _PREDICTOR is None:
        try:
            _PREDICTOR = NoShowPredictor()
        except FileNotFoundError as e:
            logger.error(f"Cannot initialize predictor: {e}")
            raise HTTPException(
                status_code=503,
                detail="ML model artifacts not loaded. Please execute scripts/train_pipeline.py first."
            )
    return _PREDICTOR


@router.post("/predict", response_model=PredictionResponse, summary="Predict Single Appointment No-Show Risk")
def predict_appointment(payload: PatientAppointmentInput):
    """
    Accepts patient appointment details and outputs calibrated no-show probability,
    operational risk category (Low, Medium, High), and top contributing factors.
    """
    predictor = get_predictor()
    record = payload.model_dump()

    try:
        result = predictor.predict_single(record, explain=True)
        return PredictionResponse(**result.to_dict())
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse, summary="Batch Appointment Predictions")
def predict_batch_appointments(payload: BatchPredictionRequest):
    """
    Predict attendance outcome for a batch of scheduled appointments.
    Computes overall summary statistics and per-patient risk classifications.
    """
    predictor = get_predictor()
    if not payload.appointments:
        raise HTTPException(status_code=400, detail="Appointments list cannot be empty.")

    records = [a.model_dump() for a in payload.appointments]
    df_input = pd.DataFrame(records)

    try:
        results_df = predictor.predict_batch(df_input)

        total = len(results_df)
        no_shows = int(results_df["predicted_no_show"].sum())
        rate = round(float(no_shows / total), 4) if total > 0 else 0.0

        risk_counts = results_df["risk_category"].value_counts()
        high_risk = int(risk_counts.get("High", 0))
        med_risk = int(risk_counts.get("Medium", 0))
        low_risk = int(risk_counts.get("Low", 0))

        items = []
        for idx, row in results_df.iterrows():
            items.append(BatchPredictionItem(
                index=int(idx),
                PatientId=float(row["PatientId"]) if pd.notnull(row.get("PatientId")) else None,
                prediction=int(row["predicted_no_show"]),
                prediction_label=str(row["prediction_label"]),
                no_show_probability=float(row["no_show_probability"]),
                risk_category=str(row["risk_category"]),
            ))

        return BatchPredictionResponse(
            total_records=total,
            predicted_no_shows=no_shows,
            predicted_no_show_rate=rate,
            high_risk_count=high_risk,
            medium_risk_count=med_risk,
            low_risk_count=low_risk,
            threshold_used=float(predictor.threshold),
            predictions=items,
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")


@router.post("/explain", response_model=ExplainResponse, summary="Explain Prediction Factors")
def explain_appointment(payload: PatientAppointmentInput):
    """
    Returns feature attributions for a specific patient appointment using SHAP.
    Separates positive risk-increasing and negative risk-decreasing contributors.
    """
    predictor = get_predictor()
    record = payload.model_dump()

    try:
        result = predictor.predict_single(record, explain=True)
        if not result.top_contributing_factors:
            raise HTTPException(status_code=500, detail="Explainability module unavailable for this model.")

        factors = result.top_contributing_factors
        return ExplainResponse(
            no_show_probability=result.no_show_probability,
            risk_category=result.risk_category,
            explanation_method=factors.get("explanation_method", "shap"),
            top_risk_increasing_factors=factors.get("top_risk_increasing_factors", []),
            top_risk_decreasing_factors=factors.get("top_risk_decreasing_factors", []),
            all_contributions=factors.get("all_contributions", []),
        )
    except Exception as e:
        logger.error(f"Explainability request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Explainability computation error: {str(e)}")
