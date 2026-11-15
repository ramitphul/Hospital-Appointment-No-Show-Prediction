"""
Data loading and dataset acquisition module.
Supports loading the Brazilian Medical Appointment No-Shows dataset,
checking existence, downloading from verified source, and generating
labeled synthetic demo data when needed.
"""
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import requests

from src.utils.config import get_project_root, get_resolved_path, load_config
from src.utils.logger import get_logger

logger = get_logger("data_loader")

KAGGLE_DATASET_URL = (
    "https://raw.githubusercontent.com/AhmadYasser1/Udacity-Investigate-A-Dataset/master/"
    "noshowappointments-kagglev2-may-2016.csv"
)

REQUIRED_COLUMNS = [
    "PatientId",
    "AppointmentID",
    "Gender",
    "ScheduledDay",
    "AppointmentDay",
    "Age",
    "Neighbourhood",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "No-show",
]


def check_dataset_exists(filepath: Path) -> bool:
    """Check if the dataset file exists and is non-empty."""
    return filepath.exists() and filepath.stat().st_size > 1024


def download_dataset(destination_path: Path, url: str = KAGGLE_DATASET_URL) -> bool:
    """
    Download the dataset from the verified open mirror.
    Returns True if successful, False otherwise.
    """
    try:
        logger.info(f"Downloading dataset from mirror: {url}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(destination_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)

        logger.info(f"Successfully downloaded dataset to: {destination_path} ({destination_path.stat().st_size} bytes)")
        return True
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False


def generate_synthetic_data(num_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate clearly labeled synthetic demonstration dataset.
    Follows Brazilian dataset schema with realistic statistical distributions.
    """
    logger.warning("Generating SYNTHETIC DEMONSTRATION DATASET (clearly distinguished from real data).")
    np.random.seed(random_seed)

    neighbourhoods = [
        "JARDIM DA PENHA", "MATA DA PRAIA", "BENTO FERREIRA", "SANTA MARTHA",
        "CENTRO", "ITOUXIS", "TABUAZEIRO", "MARUÍPE", "SANTO ANTÔNIO", "PRAIA DO CANTO"
    ]

    base_dates = pd.date_range(start="2016-04-29", end="2016-06-08")
    appointment_dates = np.random.choice(base_dates, size=num_samples)

    # Waiting days: positive integers with median around 4-10
    waiting_days = np.random.exponential(scale=10, size=num_samples).astype(int)
    waiting_days = np.clip(waiting_days, 0, 90)

    # Patient IDs (repeats for some patients)
    patient_ids = np.random.randint(100000000, 999999999, size=max(1, int(num_samples * 0.65)))
    patient_pool = np.random.choice(patient_ids, size=num_samples)

    appointment_ids = np.arange(5000000, 5000000 + num_samples)
    genders = np.random.choice(["F", "M"], size=num_samples, p=[0.65, 0.35])
    ages = np.random.gamma(shape=3.0, scale=12.0, size=num_samples).astype(int)
    ages = np.clip(ages, 0, 102)

    scholarships = np.random.binomial(1, 0.098, size=num_samples)
    hipertension = (ages > 40).astype(int) * np.random.binomial(1, 0.35, size=num_samples)
    diabetes = (ages > 45).astype(int) * np.random.binomial(1, 0.15, size=num_samples)
    alcoholism = np.random.binomial(1, 0.03, size=num_samples)
    handcap = np.random.choice([0, 1, 2], size=num_samples, p=[0.98, 0.018, 0.002])
    sms_received = (waiting_days > 2).astype(int) * np.random.binomial(1, 0.40, size=num_samples)

    scheduled_dates = []
    for app_date, wait in zip(appointment_dates, waiting_days):
        sched_date = pd.to_datetime(app_date) - pd.Timedelta(days=int(wait))
        hour = np.random.randint(7, 19)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        sched_str = f"{sched_date.strftime('%Y-%m-%d')}T{hour:02d}:{minute:02d}:{second:02d}Z"
        scheduled_dates.append(sched_str)

    app_strs = [f"{pd.to_datetime(d).strftime('%Y-%m-%d')}T00:00:00Z" for d in appointment_dates]

    # No-show probability function (higher for long waiting days, younger adults, alcoholism)
    logit = (
        -1.8
        + 0.04 * np.sqrt(waiting_days)
        + 0.15 * scholarships
        - 0.10 * (ages > 55).astype(int)
        + 0.25 * alcoholism
        - 0.50 * (waiting_days == 0).astype(int)
    )
    prob_no_show = 1.0 / (1.0 + np.exp(-logit))
    no_show_binary = (np.random.rand(num_samples) < prob_no_show).astype(int)
    no_show_labels = np.where(no_show_binary == 1, "Yes", "No")

    df = pd.DataFrame({
        "PatientId": patient_pool.astype(float),
        "AppointmentID": appointment_ids,
        "Gender": genders,
        "ScheduledDay": scheduled_dates,
        "AppointmentDay": app_strs,
        "Age": ages,
        "Neighbourhood": np.random.choice(neighbourhoods, size=num_samples),
        "Scholarship": scholarships,
        "Hipertension": hipertension,
        "Diabetes": diabetes,
        "Alcoholism": alcoholism,
        "Handcap": handcap,
        "SMS_received": sms_received,
        "No-show": no_show_labels,
    })
    logger.info(f"Synthetic dataset generated with {len(df)} records.")
    return df


def load_dataset(
    filepath: Optional[str] = None,
    allow_download: bool = True,
    fallback_synthetic: bool = True
) -> Tuple[pd.DataFrame, str]:
    """
    Load the dataset from the given path or default config path.
    Returns:
        (DataFrame, data_source_label)
    """
    config = load_config()
    default_path = get_resolved_path(config["paths"]["raw_data"])
    target_path = Path(filepath) if filepath else default_path

    data_source = "Real Kaggle Dataset"

    if not check_dataset_exists(target_path):
        logger.warning(f"Dataset not found at: {target_path}")
        if allow_download:
            download_success = download_dataset(target_path)
            if not download_success and fallback_synthetic:
                df = generate_synthetic_data()
                data_source = "Synthetic Demonstration Dataset"
                return df, data_source
        elif fallback_synthetic:
            df = generate_synthetic_data()
            data_source = "Synthetic Demonstration Dataset"
            return df, data_source
        else:
            raise FileNotFoundError(
                f"Dataset not found at {target_path}. Please download it from "
                "https://www.kaggle.com/datasets/joniarroba/noshowappointments and place it in data/raw/"
            )

    logger.info(f"Loading dataset from: {target_path}")
    df = pd.read_csv(target_path)
    logger.info(f"Loaded {len(df):,} records with columns: {list(df.columns)}")
    return df, data_source


def create_sample_dataset(source_df: pd.DataFrame, sample_size: int = 200, output_path: Optional[Path] = None) -> Path:
    """Create a sample slice for quick integration tests and UI demonstration."""
    if output_path is None:
        config = load_config()
        output_path = get_resolved_path(config["paths"]["sample_data"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df = source_df.sample(n=min(sample_size, len(source_df)), random_state=42)
    sample_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(sample_df)} sample records to {output_path}")
    return output_path
