"""
Automated unit tests for trained machine learning models, inference engine,
probability calibration, thresholding, and risk classification.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.models.evaluate import compute_classification_metrics
from src.models.predict import NoShowPredictor
from src.models.threshold import sweep_thresholds


@pytest.fixture
def predictor():
    """Load serialized trained predictor."""
    return NoShowPredictor()


@pytest.fixture
def sample_patient_record():
    """Single valid raw patient appointment dictionary."""
    return {
        "PatientId": 123456789.0,
        "AppointmentID": 5600001,
        "Gender": "F",
        "ScheduledDay": "2016-05-01T09:15:00Z",
        "AppointmentDay": "2016-05-15T00:00:00Z",
        "Age": 42,
        "Neighbourhood": "JARDIM DA PENHA",
        "Scholarship": 0,
        "Hipertension": 1,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handcap": 0,
        "prior_appointments": 2,
        "prior_no_shows": 1,
    }


def test_predictor_artifacts_loaded(predictor):
    """Verify that predictor properly deserializes trained artifacts and metadata."""
    assert predictor.model is not None
    assert predictor.preprocessor is not None
    assert predictor.metadata is not None
    assert 0.0 < predictor.threshold < 1.0
    assert len(predictor.feature_names) > 0


def test_single_prediction_output(predictor, sample_patient_record):
    """Verify individual prediction result schema, types, and mathematical bounds."""
    res = predictor.predict_single(sample_patient_record, explain=True)

    # 1. Output types & values
    assert res.prediction in [0, 1]
    assert res.prediction_label in ["Likely to attend", "No-show risk"]
    assert 0.0 <= res.no_show_probability <= 1.0
    assert 0.0 <= res.attendance_probability <= 1.0
    assert np.isclose(res.no_show_probability + res.attendance_probability, 1.0, atol=1e-4)

    # 2. Risk categorization
    assert res.risk_category in ["Low", "Medium", "High"]

    # 3. Explainability attributions
    if res.top_contributing_factors is not None:
        assert "top_risk_increasing_factors" in res.top_contributing_factors
        assert "top_risk_decreasing_factors" in res.top_contributing_factors


def test_risk_category_boundaries(predictor):
    """Verify boundary conditions for Low, Medium, and High risk classifications."""
    assert predictor.determine_risk_category(0.15) == "Low"
    assert predictor.determine_risk_category(0.299) == "Low"
    assert predictor.determine_risk_category(0.30) == "Medium"
    assert predictor.determine_risk_category(0.45) == "Medium"
    assert predictor.determine_risk_category(0.599) == "Medium"
    assert predictor.determine_risk_category(0.60) == "High"
    assert predictor.determine_risk_category(0.85) == "High"


def test_threshold_sweep_logic():
    """Verify threshold evaluation generates valid monotonic tradeoffs."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    sweep_df = sweep_thresholds(y_true, y_prob, thresholds=np.array([0.25, 0.50, 0.75]))
    assert len(sweep_df) == 3
    # As threshold increases, recall must be non-increasing
    recalls = sweep_df["recall"].tolist()
    assert recalls[0] >= recalls[1] >= recalls[2]


def test_batch_prediction(predictor, sample_patient_record):
    """Verify batch inference produces enriched DataFrame matching row count."""
    batch_df = pd.DataFrame([sample_patient_record, sample_patient_record])
    results_df = predictor.predict_batch(batch_df)

    assert len(results_df) == 2
    assert "no_show_probability" in results_df.columns
    assert "predicted_no_show" in results_df.columns
    assert "risk_category" in results_df.columns
    assert (results_df["no_show_probability"] >= 0.0).all()
    assert (results_df["no_show_probability"] <= 1.0).all()


def test_classification_metrics_computation():
    """Verify evaluation metrics calculation accuracy."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])
    m = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    assert m.accuracy == 1.0
    assert m.recall == 1.0
    assert m.precision == 1.0
    assert m.f1_score == 1.0
    assert m.specificity == 1.0
    assert m.roc_auc == 1.0
    assert m.true_positives == 2
    assert m.false_positives == 0
