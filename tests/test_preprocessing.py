"""
Automated unit tests for data ingestion, cleaning, validation, and feature engineering.
Verifies temporal integrity and leakage prevention.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.data.load_data import generate_synthetic_data, load_dataset
from src.data.preprocess import clean_raw_data, create_preprocessing_pipeline, get_feature_lists
from src.data.validate_data import validate_dataset
from src.features.build_features import (
    compute_patient_history_features,
    engineer_appointment_features,
    extract_date_features,
)


@pytest.fixture
def raw_test_df():
    """Create a controlled sample dataset with intentional anomalies for testing."""
    return pd.DataFrame({
        "PatientId": [101.0, 101.0, 102.0, 103.0, 104.0],
        "AppointmentID": [1, 2, 3, 4, 5],
        "Gender": ["F", "F", "M", "F", "M"],
        "ScheduledDay": [
            "2016-04-29T10:00:00Z",
            "2016-05-05T11:00:00Z",
            "2016-04-28T09:00:00Z",
            "2016-04-29T08:00:00Z",
            "2016-05-10T14:00:00Z",
        ],
        "AppointmentDay": [
            "2016-05-02T00:00:00Z",  # Wait = 3
            "2016-05-10T00:00:00Z",  # Wait = 5
            "2016-04-28T00:00:00Z",  # Wait = 0 (same day)
            "2016-04-20T00:00:00Z",  # INVALID: Appt precedes Sched!
            "2016-05-15T00:00:00Z",  # Wait = 5
        ],
        "Age": [25, 25, 45, -1, 130],  # -1 and 130 are invalid ages
        "Neighbourhood": ["CENTRO", "CENTRO", "JARDIM DA PENHA", "CENTRO", "OTHER"],
        "Scholarship": [0, 0, 1, 0, 0],
        "Hipertension": [0, 0, 1, 0, 1],
        "Diabetes": [0, 0, 0, 0, 1],
        "Alcoholism": [0, 0, 0, 0, 0],
        "Handcap": [0, 0, 0, 0, 1],
        "SMS_received": [0, 1, 0, 0, 1],
        "No-show": ["No", "Yes", "No", "Yes", "No"],
    })


def test_synthetic_data_generation():
    """Verify synthetic dataset generator produces valid schema and labels."""
    df = generate_synthetic_data(num_samples=100, random_seed=42)
    assert len(df) == 100
    assert "No-show" in df.columns
    assert set(df["No-show"].unique()).issubset({"Yes", "No"})
    assert (df["Age"] >= 0).all()
    assert (df["Age"] <= 115).all()


def test_data_validation(raw_test_df):
    """Verify that validation correctly detects impossible ages and invalid appointment dates."""
    report = validate_dataset(raw_test_df, min_age=0, max_age=115)
    assert report.total_records == 5
    assert report.invalid_age_count == 2  # -1 and 130
    assert report.invalid_date_count == 1  # 2016-04-20 < 2016-04-29
    assert report.validation_status == "PASSED WITH WARNINGS"


def test_data_cleaning_removes_anomalies(raw_test_df):
    """Verify cleaning drops invalid ages and invalid dates."""
    clean_df = clean_raw_data(raw_test_df, min_age=0, max_age=115, drop_negative_waiting=True)
    # Records with age -1, age 130, and appt < sched must be removed
    assert len(clean_df) == 3
    assert (clean_df["Age"] >= 0).all()
    assert (clean_df["Age"] <= 115).all()
    assert set(clean_df["No-show"].unique()).issubset({0, 1})


def test_date_feature_extraction():
    """Verify feature engineering properly calculates lead time and calendar fields."""
    sample = pd.DataFrame({
        "ScheduledDay": ["2016-05-01T08:30:00Z"],
        "AppointmentDay": ["2016-05-06T00:00:00Z"],
        "Age": [30],
        "Scholarship": [0],
        "Hipertension": [1],
        "Diabetes": [0],
        "Alcoholism": [0],
        "Handcap": [0],
    })
    res = extract_date_features(sample)
    assert res["Waiting_Days"].iloc[0] == 5
    assert res["Is_Same_Day"].iloc[0] == 0
    assert res["Scheduled_Hour"].iloc[0] == 8
    assert res["Appointment_Day_Of_Week"].iloc[0] == "Friday"
    assert res["Num_Conditions"].iloc[0] == 1


def test_patient_history_prevents_target_leakage():
    """
    CRITICAL TEST: Verify cumulative prior no-shows strictly uses past records
    and NEVER leaks the current or future appointment outcomes.
    """
    patient_df = pd.DataFrame({
        "PatientId": [999, 999, 999],
        "ScheduledDay": [
            "2016-05-01T10:00:00Z",
            "2016-05-10T10:00:00Z",
            "2016-05-20T10:00:00Z",
        ],
        "AppointmentDay": [
            "2016-05-02T00:00:00Z",
            "2016-05-12T00:00:00Z",
            "2016-05-22T00:00:00Z",
        ],
        "No-show": [1, 0, 1],  # First was a no-show, second was attended, third was a no-show
    })

    hist_df = compute_patient_history_features(patient_df, target_col="No-show")

    # At appointment 1 (index 0): 0 prior appointments, 0 prior no-shows
    assert hist_df.iloc[0]["prior_appointments"] == 0
    assert hist_df.iloc[0]["prior_no_shows"] == 0
    assert hist_df.iloc[0]["prior_no_show_rate"] == 0.0

    # At appointment 2 (index 1): 1 prior appointment, 1 prior no-show
    assert hist_df.iloc[1]["prior_appointments"] == 1
    assert hist_df.iloc[1]["prior_no_shows"] == 1
    assert hist_df.iloc[1]["prior_no_show_rate"] == 1.0

    # At appointment 3 (index 2): 2 prior appointments, still 1 prior no-show (since appt 2 was attended)
    assert hist_df.iloc[2]["prior_appointments"] == 2
    assert hist_df.iloc[2]["prior_no_shows"] == 1
    assert hist_df.iloc[2]["prior_no_show_rate"] == 0.5


def test_column_transformer_pipeline():
    """Verify ColumnTransformer transforms features without crashing."""
    cat_cols, num_cols, bin_cols = get_feature_lists(scenario="booking")
    pipe = create_preprocessing_pipeline(cat_cols, num_cols, bin_cols)

    sample_engineered = pd.DataFrame({
        "Gender": ["F", "M"],
        "Neighbourhood": ["CENTRO", "OTHER"],
        "Age_Group": ["Adult_40-59", "Child_0-12"],
        "Waiting_Time_Category": ["Short_1-3_Days", "Same_Day"],
        "Appointment_Day_Of_Week": ["Monday", "Friday"],
        "Age": [45, 10],
        "Waiting_Days": [2, 0],
        "Scheduled_Hour": [10, 14],
        "prior_appointments": [1, 0],
        "prior_no_shows": [0, 0],
        "prior_no_show_rate": [0.0, 0.0],
        "Num_Conditions": [1, 0],
        "Scholarship": [0, 0],
        "Hipertension": [1, 0],
        "Diabetes": [0, 0],
        "Alcoholism": [0, 0],
        "Handcap": [0, 0],
        "Is_Weekend": [0, 0],
        "Is_Same_Day": [0, 1],
    })

    pipe.fit(sample_engineered)
    X_trans = pipe.transform(sample_engineered)
    assert X_trans.shape[0] == 2
    assert X_trans.shape[1] > 10
    assert not np.isnan(X_trans).any()
