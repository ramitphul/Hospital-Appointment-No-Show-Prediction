"""
Automated unit and integration tests for FastAPI REST API endpoints and schemas.
Verifies response schemas, clinical validation constraints, error handling,
and end-to-end inference logic.
"""
from pathlib import Path
import sys
import pytest
from pydantic import ValidationError

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from api.routes.health import health_endpoint, root_endpoint
from api.routes.metrics import get_metrics, get_model_info
from api.routes.prediction import (
    explain_appointment,
    predict_appointment,
    predict_batch_appointments,
)
from api.schemas import (
    BatchPredictionRequest,
    PatientAppointmentInput,
)


def test_root_endpoint():
    """Verify root status endpoint returns online status and metadata."""
    res = root_endpoint()
    assert res["status"] == "online"
    assert "documentation" in res
    assert res["documentation"] == "/docs"
    assert "service" in res


def test_health_endpoint():
    """Verify health endpoint confirms model availability."""
    res = health_endpoint()
    assert res.model_loaded is True
    assert "healthy" in res.status


def test_model_info_endpoint():
    """Verify model-info endpoint returns algorithm, threshold, and feature dimensions."""
    res = get_model_info()
    assert res.best_algorithm == "XGBoost"
    assert 0.0 < res.optimal_threshold < 1.0
    assert res.feature_count == 106
    assert res.model_version == "1.0.0"
    assert res.test_metrics["roc_auc"] > 0.70


def test_metrics_endpoint():
    """Verify metrics endpoint returns comparison leaderboard and final test set metrics."""
    res = get_metrics()
    assert "final_model_test_metrics" in res
    assert "algorithm_comparison_leaderboard" in res
    assert len(res["algorithm_comparison_leaderboard"]) == 6


def test_predict_single_valid():
    """Verify prediction endpoint returns valid schema for a proper appointment request."""
    payload = PatientAppointmentInput(
        PatientId=551234567.0,
        AppointmentID=5600002,
        Gender="F",
        ScheduledDay="2016-05-01T08:30:00Z",
        AppointmentDay="2016-05-10T00:00:00Z",
        Age=28,
        Neighbourhood="CENTRO",
        Scholarship=1,
        Hipertension=0,
        Diabetes=0,
        Alcoholism=0,
        Handcap=0,
        prior_appointments=1,
        prior_no_shows=0,
    )
    res = predict_appointment(payload)

    assert res.prediction in [0, 1]
    assert res.prediction_label in ["Likely to attend", "No-show risk"]
    assert 0.0 <= res.no_show_probability <= 1.0
    assert 0.0 <= res.attendance_probability <= 1.0
    assert res.risk_category in ["Low", "Medium", "High"]
    assert res.threshold_used == 0.23


def test_predict_single_invalid_age():
    """Verify Pydantic input validation rejects negative or impossible age with ValidationError."""
    with pytest.raises(ValidationError):
        PatientAppointmentInput(
            Gender="F",
            ScheduledDay="2016-05-01T08:30:00Z",
            AppointmentDay="2016-05-10T00:00:00Z",
            Age=-5,  # Invalid age
            Neighbourhood="CENTRO",
        )

    with pytest.raises(ValidationError):
        PatientAppointmentInput(
            Gender="F",
            ScheduledDay="2016-05-01T08:30:00Z",
            AppointmentDay="2016-05-10T00:00:00Z",
            Age=130,  # Impossible age
            Neighbourhood="CENTRO",
        )


def test_predict_single_invalid_gender():
    """Verify Pydantic validation rejects invalid gender code."""
    with pytest.raises(ValidationError):
        PatientAppointmentInput(
            Gender="INVALID_GENDER",
            ScheduledDay="2016-05-01T08:30:00Z",
            AppointmentDay="2016-05-10T00:00:00Z",
            Age=30,
            Neighbourhood="CENTRO",
        )


def test_batch_prediction_endpoint():
    """Verify batch prediction processes multiple appointments and returns summary KPIs."""
    item = PatientAppointmentInput(
        Gender="M",
        ScheduledDay="2016-05-01T08:30:00Z",
        AppointmentDay="2016-05-05T00:00:00Z",
        Age=55,
        Neighbourhood="JARDIM CAMBURI",
        Scholarship=0,
        Hipertension=1,
        Diabetes=1,
        Alcoholism=0,
        Handcap=0,
        prior_appointments=4,
        prior_no_shows=0,
    )
    payload = BatchPredictionRequest(appointments=[item, item, item])
    res = predict_batch_appointments(payload)

    assert res.total_records == 3
    assert len(res.predictions) == 3
    assert res.predicted_no_shows in [0, 1, 2, 3]
    assert 0.0 <= res.predicted_no_show_rate <= 1.0
    assert (res.high_risk_count + res.medium_risk_count + res.low_risk_count) == 3


def test_explain_endpoint():
    """Verify explainability endpoint returns positive and negative SHAP factor attributions."""
    payload = PatientAppointmentInput(
        Gender="F",
        ScheduledDay="2016-05-01T08:30:00Z",
        AppointmentDay="2016-05-20T00:00:00Z",
        Age=22,
        Neighbourhood="ITARARÉ",
        Scholarship=1,
        Hipertension=0,
        Diabetes=0,
        Alcoholism=0,
        Handcap=0,
        prior_appointments=1,
        prior_no_shows=1,
    )
    res = explain_appointment(payload)

    assert 0.0 <= res.no_show_probability <= 1.0
    assert res.risk_category in ["Low", "Medium", "High"]
    assert res.explanation_method in ["tree", "linear", "kernel_or_generic", "fallback_permutation"]
    assert len(res.top_risk_increasing_factors) > 0 or len(res.top_risk_decreasing_factors) > 0
    assert len(res.all_contributions) > 0
