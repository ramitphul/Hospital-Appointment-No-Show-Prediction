"""
Script to create the 3 project Jupyter notebooks.
"""
import json
from pathlib import Path

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

root = Path(__file__).resolve().parents[1]
nb_dir = root / "notebooks"
nb_dir.mkdir(parents=True, exist_ok=True)

# 01_data_exploration.ipynb
c1 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Notebook 01: Exploratory Data Analysis\n",
            "## Hospital Appointment No-Show Prediction\n",
            "Investigates the Brazilian Outpatient dataset (110k records), verifying distributions, lead time, and confounding variables."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys\n",
            "sys.path.append('..')\n",
            "import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns\n",
            "from src.data.load_data import load_dataset\n",
            "from src.data.validate_data import validate_dataset\n",
            "\n",
            "df, source = load_dataset()\n",
            "print('Loaded records:', len(df), 'from', source)\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "report = validate_dataset(df)\n",
            "print('Validation status:', report.validation_status)\n",
            "print('Baseline No-Show Rate:', f'{report.target_no_show_rate:.2%}')\n",
            "print('Detected anomalies:', report.anomalies)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot target distribution\n",
            "target_counts = df['No-show'].value_counts()\n",
            "plt.bar(['Attended (No)', 'No-Show (Yes)'], target_counts.values, color=['#2ecc71', '#e74c3c'])\n",
            "plt.title('Appointment Outcome Distribution')\n",
            "plt.ylabel('Appointments')\n",
            "plt.show()"
        ]
    }
]
create_notebook(c1, nb_dir / "01_data_exploration.ipynb")

# 02_feature_engineering.ipynb
c2 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Notebook 02: Leak-Free Feature Engineering\n",
            "## Temporal & Patient History Transformation\n",
            "Demonstrates chronological feature extraction and ColumnTransformer without data leakage."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.append('..')\n",
            "from src.data.load_data import load_dataset\n",
            "from src.data.preprocess import clean_raw_data, prepare_train_test_data\n",
            "\n",
            "df, _ = load_dataset()\n",
            "clean_df = clean_raw_data(df)\n",
            "X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_train_test_data(clean_df)\n",
            "print('Train size:', len(X_train), 'Test size:', len(X_test))\n",
            "print('Transformed feature dimensions:', len(feature_names))\n",
            "print('Sample feature names:', feature_names[:15])"
        ]
    }
]
create_notebook(c2, nb_dir / "02_feature_engineering.ipynb")

# 03_model_training.ipynb
c3 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Notebook 03: Machine Learning Model Benchmarks & SHAP Explainability\n",
            "Evaluates 6 classification algorithms, threshold optimization, probability calibration, and SHAP interpretability."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys, json\n",
            "sys.path.append('..')\n",
            "import pandas as pd\n",
            "from src.models.predict import NoShowPredictor\n",
            "\n",
            "predictor = NoShowPredictor()\n",
            "print('Best Model Algorithm:', predictor.metadata.get('best_algorithm'))\n",
            "print('Optimal Classification Threshold:', predictor.threshold)\n",
            "\n",
            "res = predictor.predict_single({\n",
            "    'ScheduledDay': '2016-05-01T09:00:00Z',\n",
            "    'AppointmentDay': '2016-05-15T00:00:00Z',\n",
            "    'Age': 28,\n",
            "    'Gender': 'F',\n",
            "    'Neighbourhood': 'CENTRO'\n",
            "}, explain=True)\n",
            "print('Prediction outcome:', res.prediction_label)\n",
            "print('Calibrated probability:', res.no_show_probability)\n",
            "print('Risk tier:', res.risk_category)"
        ]
    }
]
create_notebook(c3, nb_dir / "03_model_training.ipynb")

print("All 3 notebooks successfully written to:", nb_dir)
