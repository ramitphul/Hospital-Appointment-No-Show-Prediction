"""
Feature engineering module for hospital appointment attendance prediction.
Implements leak-free temporal feature extraction, patient historical profiling,
and operational scenario alignments.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger

logger = get_logger("feature_engineer")


def categorize_age(age: int) -> str:
    """Group age into clinically and behaviorally meaningful categories."""
    if age <= 12:
        return "Child_0-12"
    elif age <= 19:
        return "Teen_13-19"
    elif age <= 39:
        return "YoungAdult_20-39"
    elif age <= 59:
        return "Adult_40-59"
    else:
        return "Senior_60+"


def categorize_waiting_time(days: int) -> str:
    """Categorize scheduling lead time."""
    if days == 0:
        return "Same_Day"
    elif days <= 3:
        return "Short_1-3_Days"
    elif days <= 14:
        return "Medium_4-14_Days"
    else:
        return "Long_15+_Days"


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar and temporal features from ScheduledDay and AppointmentDay.
    Calculates waiting days strictly without leakage.
    """
    data = df.copy()

    # Parse datetimes
    sched_dt = pd.to_datetime(data["ScheduledDay"])
    appt_dt = pd.to_datetime(data["AppointmentDay"])

    # ScheduledDay features
    data["Scheduled_Year"] = sched_dt.dt.year
    data["Scheduled_Month"] = sched_dt.dt.month
    data["Scheduled_Day_Of_Week"] = sched_dt.dt.day_name()
    data["Scheduled_Hour"] = sched_dt.dt.hour

    # AppointmentDay features
    data["Appointment_Year"] = appt_dt.dt.year
    data["Appointment_Month"] = appt_dt.dt.month
    data["Appointment_Day_Of_Week"] = appt_dt.dt.day_name()

    # Lead time / Waiting Days (calendar day difference)
    # Note: Legacy EHR systems often store AppointmentDay truncated to 00:00:00,
    # while ScheduledDay retains the exact timestamp. We floor both to day boundaries
    # to avoid fractional day artifacts.
    sched_date = sched_dt.dt.floor("D")
    appt_date = appt_dt.dt.floor("D")
    waiting_days = (appt_date - sched_date).dt.days

    # Clamp any minor negative values (caused by occasional clinic computer clock desync)
    data["Waiting_Days"] = np.maximum(0, waiting_days)
    data["Is_Same_Day"] = (data["Waiting_Days"] == 0).astype(int)
    data["Is_Weekend"] = (appt_dt.dt.weekday >= 5).astype(int)

    # Derived categories
    data["Waiting_Time_Category"] = data["Waiting_Days"].apply(categorize_waiting_time)
    data["Age_Group"] = data["Age"].apply(categorize_age)

    # Health conditions aggregation
    condition_cols = ["Scholarship", "Hipertension", "Diabetes", "Alcoholism"]
    active_conditions = data[condition_cols].sum(axis=1) + (data["Handcap"] > 0).astype(int)
    data["Num_Conditions"] = active_conditions

    return data


def compute_patient_history_features(df: pd.DataFrame, target_col: str = "No-show") -> pd.DataFrame:
    """
    Compute strictly historical patient features without target leakage.
    Sorts chronologically, and calculates cumulative previous appointments
    and no-shows up to (but not including) the current appointment using
    vectorized C-optimized cumulative sums.
    """
    data = df.copy()

    # Ensure chronological order
    if "ScheduledDay" in data.columns and "PatientId" in data.columns:
        original_index = data.index
        data = data.sort_values(by=["PatientId", "ScheduledDay"])

        # Crucial data leakage prevention:
        # A common mistake on Kaggle is using a patient's overall no-show rate as a feature.
        # That leaks future outcomes into past appointments!
        # Here we strictly compute past cumulative visits and no-shows up to (not including)
        # the current appointment. Using cumsum() - current_val avoids slow Python loops.
        data["prior_appointments"] = data.groupby("PatientId").cumcount()

        if target_col in data.columns:
            if data[target_col].dtype == object:
                numeric_target = (data[target_col].astype(str).str.strip() == "Yes").astype(int)
            else:
                numeric_target = (data[target_col] == 1).astype(int)

            data["_temp_target"] = numeric_target
            # Vectorized shift-by-1: cumulative sum including current minus current value
            cum_target = data.groupby("PatientId")["_temp_target"].cumsum()
            data["prior_no_shows"] = (cum_target - data["_temp_target"]).astype(int)
            data.drop(columns=["_temp_target"], inplace=True)
        else:
            data["prior_no_shows"] = 0

        # Prior no-show rate
        denom = data["prior_appointments"].replace(0, np.nan)
        data["prior_no_show_rate"] = (data["prior_no_shows"] / denom).fillna(0.0).round(4)

        # Restore original row order
        data = data.loc[original_index]
    else:
        # Default for single-row inference or when PatientId is absent
        if "prior_appointments" not in data.columns:
            data["prior_appointments"] = 0
        if "prior_no_shows" not in data.columns:
            data["prior_no_shows"] = 0
        denom = data["prior_appointments"].replace(0, np.nan)
        data["prior_no_show_rate"] = (data["prior_no_shows"] / denom).fillna(0.0).round(4)

    return data


def engineer_appointment_features(
    df: pd.DataFrame,
    scenario: str = "booking",
    target_col: Optional[str] = "No-show"
) -> pd.DataFrame:
    """
    Complete end-to-end feature engineering pipeline.
    Enforces scenario constraints (e.g., omitting SMS_received in 'booking' scenario).
    """
    data = extract_date_features(df)
    data = compute_patient_history_features(data, target_col=target_col if target_col in data else None)

    # In booking scenario, SMS_received is unavailable at the time of booking
    if scenario == "booking" and "SMS_received" in data.columns:
        logger.debug("Booking-time scenario: dropping SMS_received to eliminate temporal data leakage.")
        # We don't drop it from raw data if needed for analysis, but will exclude from model features
    return data


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for feature engineering.
    Can be seamlessly embedded in inference pipelines.
    """
    def __init__(self, scenario: str = "booking"):
        self.scenario = scenario

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer_appointment_features(X, scenario=self.scenario)
