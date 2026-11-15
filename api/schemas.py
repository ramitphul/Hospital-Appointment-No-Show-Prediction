"""
Pydantic data validation schemas for FastAPI backend.
Validates clinical inputs, date formats, ranges, and structures JSON responses.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class PatientAppointmentInput(BaseModel):
    """Schema for individual appointment prediction request."""
    PatientId: Optional[float] = Field(default=None, description="Unique patient identifier")
    AppointmentID: Optional[int] = Field(default=None, description="Unique appointment identifier")
    Gender: str = Field(default="F", description="Gender of the patient ('F' or 'M')")
    ScheduledDay: str = Field(
        ...,
        description="Booking timestamp in ISO 8601 (e.g., '2016-04-29T18:38:08Z' or '2026-05-01 10:00:00')",
        examples=["2016-04-29T18:38:08Z"]
    )
    AppointmentDay: str = Field(
        ...,
        description="Appointment date in ISO 8601 (e.g., '2016-05-05T00:00:00Z' or '2026-05-05')",
        examples=["2016-05-05T00:00:00Z"]
    )
    Age: int = Field(..., ge=0, le=115, description="Patient age in years (0 to 115)")
    Neighbourhood: str = Field(default="CENTRO", description="Clinic/Neighbourhood location")
    Scholarship: int = Field(default=0, ge=0, le=1, description="Enrolled in Bolsa Família welfare (1=Yes, 0=No)")
    Hipertension: int = Field(default=0, ge=0, le=1, description="Hypertension diagnosis (1=Yes, 0=No)")
    Diabetes: int = Field(default=0, ge=0, le=1, description="Diabetes diagnosis (1=Yes, 0=No)")
    Alcoholism: int = Field(default=0, ge=0, le=1, description="Alcoholism diagnosis (1=Yes, 0=No)")
    Handcap: int = Field(default=0, ge=0, le=4, description="Handicap status (0=None, 1+=Handicapped)")
    SMS_received: Optional[int] = Field(default=0, ge=0, le=1, description="SMS reminder sent (1=Yes, 0=No)")
    prior_appointments: Optional[int] = Field(default=0, ge=0, description="Number of past appointments for this patient")
    prior_no_shows: Optional[int] = Field(default=0, ge=0, description="Number of past missed appointments for this patient")

    @field_validator("Gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        clean = v.strip().upper()
        if clean not in ["F", "M"]:
            raise ValueError("Gender must be 'F' or 'M'")
        return clean

    @field_validator("Neighbourhood")
    @classmethod
    def validate_neighbourhood(cls, v: str) -> str:
        clean = v.strip().upper()
        return clean if clean else "OTHER"


class ContributingFactor(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    direction: str


class PredictionResponse(BaseModel):
    """Schema for individual prediction output."""
    prediction: int = Field(..., description="Binary prediction: 1 = No-show risk, 0 = Likely to attend")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    no_show_probability: float = Field(..., description="Calibrated probability of missing appointment (0.0 to 1.0)")
    attendance_probability: float = Field(..., description="Probability of attending appointment (0.0 to 1.0)")
    risk_category: str = Field(..., description="Operational risk tier ('Low', 'Medium', 'High')")
    threshold_used: float = Field(..., description="Classification threshold applied")
    model_version: str = Field(..., description="Model version string")
    top_contributing_factors: Optional[Dict[str, Any]] = Field(default=None, description="SHAP feature attributions")


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction upload."""
    appointments: List[PatientAppointmentInput]


class BatchPredictionItem(BaseModel):
    index: int
    PatientId: Optional[float] = None
    prediction: int
    prediction_label: str
    no_show_probability: float
    risk_category: str


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction summary."""
    total_records: int
    predicted_no_shows: int
    predicted_no_show_rate: float
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    threshold_used: float
    predictions: List[BatchPredictionItem]


class ExplainResponse(BaseModel):
    """Schema for explainability inspection endpoint."""
    no_show_probability: float
    risk_category: str
    explanation_method: str
    top_risk_increasing_factors: List[ContributingFactor]
    top_risk_decreasing_factors: List[ContributingFactor]
    all_contributions: List[ContributingFactor]


class ModelInfoResponse(BaseModel):
    """Schema for model metadata endpoint."""
    model_version: str
    project_name: str
    best_algorithm: str
    scenario: str
    optimal_threshold: float
    feature_count: int
    calibrated_brier_score: float
    test_metrics: Dict[str, Any]
    risk_thresholds: Dict[str, float]


class HealthResponse(BaseModel):
    """Schema for health check endpoint."""
    status: str
    service: str
    version: str
    model_loaded: bool
    timestamp: str
