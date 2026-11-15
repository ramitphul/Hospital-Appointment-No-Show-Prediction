"""
Data preprocessing, cleaning, transformation, and train-test splitting module.
Implements leak-free Scikit-learn ColumnTransformer pipelines.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import engineer_appointment_features
from src.utils.config import get_resolved_path, load_config
from src.utils.logger import get_logger

logger = get_logger("preprocessor")


def clean_raw_data(
    df: pd.DataFrame,
    min_age: int = 0,
    max_age: int = 115,
    drop_negative_waiting: bool = True,
    rare_neighbourhood_threshold: int = 100,
) -> pd.DataFrame:
    """
    Clean raw Brazilian appointment dataset.
    - Standardizes target: 'No' -> 0 (Attended), 'Yes' -> 1 (No-show).
    - Removes impossible ages (< min_age or > max_age).
    - Removes impossible appointment dates (where appointment date is prior to scheduled date).
    - Removes exact duplicate records.
    - Groups ultra-rare neighbourhoods into 'OTHER'.
    """
    initial_len = len(df)
    data = df.copy()

    # Drop full duplicates
    data = data.drop_duplicates()

    # Clean target variable if present
    if "No-show" in data.columns:
        if data["No-show"].dtype == object:
            data["No-show"] = data["No-show"].str.strip()
            data["No-show"] = data["No-show"].map({"No": 0, "Yes": 1})
            # Drop rows with null target
            data = data.dropna(subset=["No-show"])
            data["No-show"] = data["No-show"].astype(int)

    # Filter out impossible ages
    data = data[(data["Age"] >= min_age) & (data["Age"] <= max_age)]

    # Validate and filter impossible dates
    if drop_negative_waiting and "ScheduledDay" in data.columns and "AppointmentDay" in data.columns:
        sched_date = pd.to_datetime(data["ScheduledDay"]).dt.floor("D")
        appt_date = pd.to_datetime(data["AppointmentDay"]).dt.floor("D")
        valid_date_mask = appt_date >= sched_date
        data = data[valid_date_mask]

    # Handle rare neighbourhoods
    if "Neighbourhood" in data.columns and rare_neighbourhood_threshold > 0:
        counts = data["Neighbourhood"].value_counts()
        rare = counts[counts < rare_neighbourhood_threshold].index
        if len(rare) > 0:
            data["Neighbourhood"] = data["Neighbourhood"].replace(rare, "OTHER")

    # Ensure binary fields are clean integers (0 or 1)
    binary_cols = ["Scholarship", "Hipertension", "Diabetes", "Alcoholism"]
    for col in binary_cols:
        if col in data.columns:
            data[col] = (data[col] > 0).astype(int)

    if "Handcap" in data.columns:
        data["Handcap"] = (data["Handcap"] > 0).astype(int)

    removed = initial_len - len(data)
    logger.info(f"Cleaned dataset: {len(data):,} rows retained ({removed:,} anomalous/duplicate rows removed).")
    return data


def create_preprocessing_pipeline(
    categorical_cols: List[str],
    numeric_cols: List[str],
    binary_cols: List[str],
) -> ColumnTransformer:
    """
    Construct Scikit-learn ColumnTransformer for modular, leak-free preprocessing.
    - Numeric features: Median imputation + StandardScaler
    - Categorical features: Constant imputation ('Missing') + OneHotEncoder
    - Binary features: Most frequent imputation
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    bin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
            ("bin", bin_pipeline, binary_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_lists(scenario: str = "booking", config: Optional[Dict] = None) -> Tuple[List[str], List[str], List[str]]:
    """Return lists of categorical, numerical, and binary columns based on scenario."""
    if config is None:
        config = load_config()

    cat_cols = list(config["features"]["categorical_cols"])
    num_cols = list(config["features"]["numeric_cols"])
    bin_cols = list(config["features"]["binary_cols"])

    if scenario == "pre_appointment":
        if "SMS_received" not in bin_cols:
            bin_cols.append("SMS_received")
    else:
        # In booking scenario, explicitly exclude SMS_received
        bin_cols = [c for c in bin_cols if c != "SMS_received"]

    return cat_cols, num_cols, bin_cols


def prepare_train_test_data(
    df: pd.DataFrame,
    scenario: str = "booking",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, List[str]]:
    """
    End-to-end data preparation:
    1. Cleans raw data
    2. Engineers appointment and temporal features
    3. Splits data into train and test sets using stratification
    4. Fits ColumnTransformer ONLY on train data
    5. Returns train/test feature matrices, labels, fitted transformer, and feature names.
    """
    config = load_config()
    target_col = config["features"]["target_col"]

    # 1. Cleaning
    clean_df = clean_raw_data(
        df,
        min_age=config["data_cleaning"]["min_age"],
        max_age=config["data_cleaning"]["max_age"],
        drop_negative_waiting=config["data_cleaning"]["drop_negative_waiting_days"],
    )

    # 2. Feature engineering
    feat_df = engineer_appointment_features(clean_df, scenario=scenario, target_col=target_col)

    cat_cols, num_cols, bin_cols = get_feature_lists(scenario=scenario, config=config)
    selected_feature_cols = cat_cols + num_cols + bin_cols

    # Ensure all required features are present
    missing_feats = [c for c in selected_feature_cols if c not in feat_df.columns]
    if missing_feats:
        raise KeyError(f"Missing expected engineered features: {missing_feats}")

    X = feat_df[selected_feature_cols]
    y = feat_df[target_col]

    # 3. Stratified Train-Test Split (leak-free)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    logger.info(
        f"Train-Test split completed: "
        f"Train={len(X_train):,} ({y_train.mean():.2%} no-shows) | "
        f"Test={len(X_test):,} ({y_test.mean():.2%} no-shows)"
    )

    # 4. Construct and fit preprocessor ONLY on X_train
    preprocessor = create_preprocessing_pipeline(cat_cols, num_cols, bin_cols)
    preprocessor.fit(X_train)

    feature_names = list(preprocessor.get_feature_names_out())
    logger.info(f"Preprocessor fitted. Transformed feature dimensions: {len(feature_names)} features.")

    return X_train, X_test, y_train, y_test, preprocessor, feature_names
