"""
Model metadata and evaluation metrics endpoints.
"""
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
import pandas as pd

from api.schemas import ModelInfoResponse
from src.utils.config import get_resolved_path, load_config
from src.utils.logger import get_logger

logger = get_logger("metrics_route")
router = APIRouter(tags=["Model & Metrics"])


@router.get("/model-info", response_model=ModelInfoResponse, summary="Get Model Metadata")
def get_model_info():
    """Retrieve metadata, feature count, version, and threshold of active model."""
    config = load_config()
    metadata_path = get_resolved_path(config["paths"]["models_dir"]) / "model_metadata.json"

    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model metadata artifact not found. Please run the training pipeline first."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return ModelInfoResponse(
        model_version=meta.get("model_version", "1.0.0"),
        project_name=meta.get("project_name", "hospital-no-show-prediction"),
        best_algorithm=meta.get("best_algorithm", "Unknown"),
        scenario=meta.get("scenario", "booking"),
        optimal_threshold=float(meta.get("optimal_threshold", 0.50)),
        feature_count=int(meta.get("feature_count", 0)),
        calibrated_brier_score=float(meta.get("calibrated_brier_score", 0.0)),
        test_metrics=meta.get("test_metrics", {}),
        risk_thresholds=meta.get("risk_thresholds", {}),
    )


@router.get("/metrics", summary="Get Full Evaluation Metrics & Comparison")
def get_metrics():
    """Retrieve multi-algorithm comparison leaderboard and test set evaluation results."""
    config = load_config()
    metrics_dir = get_resolved_path(config["paths"]["metrics_dir"])

    comparison_path = metrics_dir / "model_comparison.json"
    final_metrics_path = metrics_dir / "final_test_metrics.json"

    if not comparison_path.exists() or not final_metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model evaluation artifacts not found. Please run the training pipeline first."
        )

    with open(comparison_path, "r", encoding="utf-8") as f:
        comparison_data = json.load(f)

    with open(final_metrics_path, "r", encoding="utf-8") as f:
        final_test_metrics = json.load(f)

    return {
        "final_model_test_metrics": final_test_metrics,
        "algorithm_comparison_leaderboard": comparison_data,
    }
