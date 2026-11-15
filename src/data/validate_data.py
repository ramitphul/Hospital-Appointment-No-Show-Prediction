"""
Data validation and quality assurance module.
Performs integrity checks, type assertions, anomaly detection,
and generates structured Data Quality Reports.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.data.load_data import REQUIRED_COLUMNS
from src.utils.logger import get_logger

logger = get_logger("data_validator")


@dataclass
class DataValidationReport:
    total_records: int
    missing_values: Dict[str, int]
    duplicate_records: int
    duplicate_appointments: int
    invalid_age_count: int
    invalid_date_count: int
    gender_distribution: Dict[str, int]
    target_distribution: Dict[str, int]
    target_no_show_rate: float
    validation_status: str
    anomalies: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_dataset(df: pd.DataFrame, min_age: int = 0, max_age: int = 115) -> DataValidationReport:
    """
    Validate dataset integrity and detect anomalies.
    Checks required schema, data types, impossible ages, and chronological date logic.
    """
    anomalies = []

    # 1. Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        msg = f"Missing required columns in dataset: {missing_cols}"
        anomalies.append(msg)
        raise ValueError(msg)

    # 2. Check null values
    null_counts = df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        anomalies.append(f"Detected {total_nulls} null values across columns.")

    # 3. Check duplicates
    dup_rows = int(df.duplicated().sum())
    dup_appts = int(df.duplicated(subset=["AppointmentID"]).sum())
    if dup_rows > 0:
        anomalies.append(f"Detected {dup_rows} identical duplicate rows.")
    if dup_appts > 0:
        anomalies.append(f"Detected {dup_appts} duplicate AppointmentID entries.")

    # 4. Check impossible ages
    invalid_ages = int(((df["Age"] < min_age) | (df["Age"] > max_age)).sum())
    if invalid_ages > 0:
        anomalies.append(f"Detected {invalid_ages} impossible age values (< {min_age} or > {max_age}).")

    # 5. Check chronological date logic (ScheduledDay vs AppointmentDay)
    try:
        sched_dt = pd.to_datetime(df["ScheduledDay"]).dt.floor("D")
        appt_dt = pd.to_datetime(df["AppointmentDay"]).dt.floor("D")
        invalid_dates = int((appt_dt < sched_dt).sum())
        if invalid_dates > 0:
            anomalies.append(f"Detected {invalid_dates} records where AppointmentDay precedes ScheduledDay.")
    except Exception as e:
        invalid_dates = -1
        anomalies.append(f"Date parsing error during validation: {e}")

    # 6. Target and gender distribution
    gender_dist = df["Gender"].value_counts().to_dict() if "Gender" in df else {}

    target_dist = {}
    no_show_rate = 0.0
    if "No-show" in df:
        target_counts = df["No-show"].value_counts()
        target_dist = target_counts.to_dict()
        yes_count = int(target_counts["Yes"]) if "Yes" in target_counts else int(target_counts.get(1, 0))
        no_show_rate = round(yes_count / len(df), 4) if len(df) > 0 else 0.0

    status = "PASSED WITH WARNINGS" if anomalies else "PASSED"
    if missing_cols:
        status = "FAILED"

    report = DataValidationReport(
        total_records=int(len(df)),
        missing_values=null_counts,
        duplicate_records=dup_rows,
        duplicate_appointments=dup_appts,
        invalid_age_count=invalid_ages,
        invalid_date_count=invalid_dates,
        gender_distribution=gender_dist,
        target_distribution=target_dist,
        target_no_show_rate=no_show_rate,
        validation_status=status,
        anomalies=anomalies,
    )

    logger.info(
        f"Validation Complete | Status: {status} | Total Records: {len(df):,} | "
        f"No-Show Rate: {no_show_rate:.2%} | Anomalies: {len(anomalies)}"
    )
    return report


def save_validation_report(report: DataValidationReport, output_path: Path) -> Path:
    """Save validation report as JSON and Markdown for transparency."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    md_path = output_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Data Quality and Validation Report\n\n")
        f.write(f"- **Validation Status:** `{report.validation_status}`\n")
        f.write(f"- **Total Records Analyzed:** {report.total_records:,}\n")
        f.write(f"- **Target No-Show Rate:** {report.target_no_show_rate:.2%}\n")
        f.write(f"- **Duplicate Records:** {report.duplicate_records}\n")
        f.write(f"- **Impossible Age Count (<0 or >115):** {report.invalid_age_count}\n")
        f.write(f"- **Invalid Appointment Date Count (Appt < Sched):** {report.invalid_date_count}\n\n")
        f.write("## Target Distribution\n")
        for k, v in report.target_distribution.items():
            f.write(f"- `{k}`: {v:,}\n")
        f.write("\n## Identified Anomalies\n")
        if report.anomalies:
            for a in report.anomalies:
                f.write(f"- [WARNING] {a}\n")
        else:
            f.write("- No critical anomalies detected.\n")

    logger.info(f"Saved validation report to {json_path} and {md_path}")
    return json_path
