# Hospital Appointment No-Show Prediction

An end-to-end machine learning system that predicts whether a patient will miss a scheduled hospital appointment. Designed as an operational decision-support tool for clinic receptionists and outpatient scheduling coordinators.

Built with Python, Scikit-learn, XGBoost, SHAP, FastAPI, and Streamlit.

---

## The Problem & Operational Context

Outpatient clinics and hospitals face an average 20% no-show rate. When patients miss appointments without canceling in advance:
- Doctors and exam rooms sit idle, wasting expensive clinical staff hours.
- Other patients wait weeks for open appointment slots that could have been filled.
- The hospital loses outpatient revenue while fixed overhead remains identical.

Most hospitals try to address this with blanket SMS reminders or aggressive overbooking. Both have drawbacks:
1. **Blanket reminders waste outreach resources** on patients who were always going to attend anyway.
2. **Naive overbooking causes chaotic waiting rooms** and doctor burnout on days when everyone actually shows up.

### The Objective
Instead of guessing, this project builds a classification pipeline that estimates patient-specific no-show probabilities at the time of booking. Staff can then apply targeted, tiered interventions:
- **Low Risk (<30%):** Standard confirmation text. No staff time needed.
- **Medium Risk (30%–60%):** Two automated reminders (72h and 24h prior) with a 1-click confirmation button.
- **High Risk (≥60%):** A clinic coordinator calls the patient 48h prior to check on transportation/work conflicts, and the scheduler can strategically double-book that specific slot.

---

## Key Engineering Decisions & What Surprised Me in the Data

### 1. The SMS Reminder Paradox (Simpson's Paradox in Healthcare)
When first inspecting the raw data, patients who received an SMS had a **higher no-show rate (27.6%)** than patients who received no SMS (**16.7%**).

If an analyst blindly fed this into a model without domain context, the model might conclude that sending SMS messages causes patients to miss appointments.

Looking deeper into the hospital's scheduling protocol revealed why:
- Same-day appointments almost never miss (only 4.6% no-show rate), and the clinic never sends SMS reminders for appointments booked on the same day.
- SMS reminders were only dispatched when appointments were booked at least 3 days in advance.
- Longer waiting times are the real driver of missed visits. Once you control for lead time (`Waiting_Days`), the paradox disappears.

### 2. Strict Temporal Discipline (No Data Leakage)
A common flaw in Kaggle-based projects is leaking future data into training features:
- **`SMS_received` at booking:** In the booking scenario, the SMS hasn't been sent yet. Therefore, `SMS_received` is excluded from the booking-time feature set.
- **Historical Patient Features:** If a patient has multiple appointments in the dataset, calculating their historical no-show rate using their entire record leaks future visits into past predictions. We sorted chronologically and used an expanding cumulative sum shifted by 1:

$$\text{prior\_no\_shows}_t = \sum_{k=1}^{t-1} y_k$$

For a patient's first appointment, prior counts are strictly 0.

### 3. Why Accuracy is a Misleading Metric Here
Because 79.8% of patients attend their appointments, a dummy classifier that always predicts "Will Attend" achieves **79.8% accuracy**. But it catches **0% of no-shows**, making it completely useless in a clinic.

We evaluated models using:
- **PR-AUC (Precision-Recall AUC):** Focuses specifically on the minority positive class.
- **ROC-AUC:** Measures discrimination across all decision thresholds.
- **Recall (Sensitivity):** Percentage of actual no-shows correctly flagged.
- **Brier Score:** Evaluates whether predicted probabilities match observed frequencies.

### 4. Why We Lowered the Decision Threshold from 0.50 to 0.23
The default threshold of 0.50 assumes symmetric error costs. In a hospital, the costs are heavily asymmetric:
- **False Negative (missed no-show):** Costs the hospital an empty doctor slot (~$150–$300 loss).
- **False Positive (false alarm):** Costs the hospital an automated reminder or a 60-second phone call (~$1–$2).

By sweeping thresholds on validation folds to maximize F1, we selected **0.230**:

| Metric | Default Threshold (0.50) | Optimized Threshold (0.23) | Operational Impact |
| :--- | :---: | :---: | :--- |
| **Recall (No-Shows Caught)** | **7.5%** | **71.95%** | Catches 3,211 no-shows instead of just 333 |
| **Precision** | 61.1% | 33.1% | 1 out of 3 flagged patients actually misses |
| **F1-Score** | 0.133 | 0.454 | More than 3x improvement in harmonic balance |
| **Accuracy** | 80.4% | 65.0% | Sacrificed headline accuracy to gain actionable recall |

---

## Model Benchmark Results

Trained on 88,416 appointments, evaluated on 22,105 held-out test appointments.

| Model | ROC-AUC | PR-AUC | F1-Score | Recall | Precision | Specificity | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Calibrated)** | **0.756** | **0.408** | **0.460** | **82.2%** | **32.0%** | **55.8%** | **61.1%** |
| **Gradient Boosting (sklearn)** | 0.751 | 0.400 | 0.104 | 5.7% | 59.8% | 99.0% | 80.2% |
| **Random Forest** | 0.751 | 0.398 | 0.457 | 85.6% | 31.1% | 52.1% | 58.8% |
| **Logistic Regression** | 0.743 | 0.383 | 0.451 | 82.8% | 31.0% | 53.3% | 59.3% |
| **Decision Tree** | 0.736 | 0.371 | 0.451 | 83.0% | 30.9% | 53.1% | 59.1% |
| **K-Nearest Neighbors** | 0.711 | 0.345 | 0.194 | 12.5% | 43.3% | 95.9% | 79.0% |

### Why XGBoost Won
XGBoost handled the non-linear relationship between waiting time and attendance better than linear models, and generalized better than Random Forest on rare neighbourhood categories. After Sigmoid probability calibration (`CalibratedClassifierCV`), the Brier score improved to **0.1407**.

---

## Explainable AI: What the Model Actually Learned

Using SHAP (`shap.TreeExplainer`), we computed global feature importance and patient-level explanations:

1. **Lead Time (`Waiting_Days`):** By far the most influential variable. Same-day appointments rarely miss; appointments booked >2 weeks out have a >33% no-show rate.
2. **Patient History (`prior_appointments`, `prior_no_shows`):** Prior unexcused absences strongly indicate future no-show risk.
3. **Age & Life Stage:** Young adults (20–39) miss appointments most frequently (work and childcare conflicts). Seniors (60+) have significantly higher attendance rates (regular chronic disease checkups).
4. **Chronic Diseases (`Hipertension`, `Diabetes`):** Diagnosed chronic patients attend more reliably, recognizing the clinical necessity of their prescriptions.
5. **Welfare Scholarship (`Scholarship` / Bolsa Família):** Patients receiving welfare support face slightly higher no-show risk, primarily driven by transit and hourly wage constraints.

---

## Project Structure

```
hospital-no-show-prediction/
├── api/                      # FastAPI REST service
│   ├── routes/               # Modular endpoints: health, prediction, metrics
│   ├── main.py               # App factory & CORS setup
│   └── schemas.py            # Pydantic v2 validation models
├── artifacts/                # Generated figures and serialized models
│   ├── eda/                  # 8 EDA plots & summary report
│   ├── metrics/              # Leaderboard CSV, ROC curves, confusion matrices
│   ├── models/               # final_model.joblib, preprocessor.joblib, metadata
│   └── shap/                 # Beeswarm summary plot & bar chart
├── config/
│   └── config.yaml           # Central YAML configuration
├── data/
│   ├── raw/                  # Full Brazilian dataset (110k records)
│   ├── processed/            # Train/test parquet files
│   └── sample/               # 300-patient test CSV for batch demo
├── frontend/                 # Streamlit UI
│   ├── app.py                # Main navigation & landing dashboard
│   └── pages/                # 5 functional pages (EDA, Single, Batch, Eval, SHAP)
├── notebooks/                # 3 documented exploratory notebooks
├── scripts/                  # Standalone execution scripts
│   ├── download_data.py      # Data fetcher & validator
│   ├── evaluate_model.py     # Evaluation CLI on test set
│   └── train_pipeline.py     # Master training script
├── src/                      # Reusable core library
│   ├── data/                 # Loading, validation, cleaning, ColumnTransformer
│   ├── features/             # Date extraction & patient history engineering
│   ├── models/               # Training, tuning, threshold sweep, inference engine
│   ├── explainability/       # SHAP TreeExplainer wrapper
│   └── utils/                # Config & logging utilities
├── tests/                    # 21 Pytest unit & integration tests
├── Dockerfile                # Multi-stage production container
├── docker-compose.yml        # Orchestration for API + Frontend
├── requirements.txt          # Pinned Python packages
└── start_services.bat        # One-click Windows runner
```

---

## Quickstart Guide

### 1. Setup Environment
```bash
# Clone and enter directory
cd hospital-no-show-prediction

# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline
```bash
python scripts/train_pipeline.py
```
This loads and validates the 110k records, generates EDA figures, trains the 6 algorithms, runs hyperparameter tuning, calibrates probabilities, optimizes the threshold, computes SHAP plots, and serializes the final model.

### 3. Run Automated Tests
```bash
python -m pytest -v
```
All 21 tests cover data cleaning, leakage prevention, probability calibration, risk tier boundaries, and API routes.

### 4. Start the Application

You can launch both services with one command using the included batch file:
```bash
# Windows:
.\start_services.bat
```

Or run them individually in separate terminals:

**Terminal 1 (Backend API):**
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
- API root: `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

**Terminal 2 (Frontend Web App):**
```bash
python -m streamlit run frontend/app.py --server.port 8501
```
- Streamlit UI: `http://localhost:8501`

---

## API Usage Examples

### Single Patient Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 32,
    "Gender": "F",
    "Neighbourhood": "JARDIM DA PENHA",
    "ScheduledDay": "2016-05-01T09:00:00Z",
    "AppointmentDay": "2016-05-18T00:00:00Z",
    "Scholarship": 0,
    "Hipertension": 0,
    "Diabetes": 0,
    "Alcoholism": 0,
    "Handcap": 0,
    "prior_appointments": 2,
    "prior_no_shows": 1
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "No-show risk",
  "no_show_probability": 0.342,
  "attendance_probability": 0.658,
  "risk_category": "Medium",
  "threshold_used": 0.23,
  "model_version": "1.0.0",
  "top_contributing_factors": {
    "top_risk_increasing_factors": [
      {"feature": "Waiting_Days", "shap_value": 0.451, "feature_value": 17.0, "direction": "Increases Risk"},
      {"feature": "Age", "shap_value": 0.112, "feature_value": 32.0, "direction": "Increases Risk"}
    ],
    "top_risk_decreasing_factors": [
      {"feature": "Gender_F", "shap_value": -0.019, "feature_value": 1.0, "direction": "Decreases Risk"}
    ]
  }
}
```

---

## Practical Limitations & Future Improvements

### Real-World Data Constraints
1. **No Granular Socioeconomic or Transit Data:** The Kaggle dataset only has `Scholarship` (welfare status). In a real hospital EHR, having distance to clinic, public transit access, and car ownership would significantly improve prediction accuracy for high-risk patients.
2. **Missing Provider Specialties:** A dermatology follow-up has vastly different no-show dynamics than an oncology or cardiology appointment. Adding the clinic department as a feature would be a high-impact improvement.
3. **Weather Data:** Heavy rain or severe weather days reliably spike no-show rates. Joining local historical weather data (via open APIs) would add useful seasonal context.

### Next Steps for Production
- **Concept Drift Monitoring:** Track weekly distribution shifts in lead times and no-show rates using Evidently AI.
- **Dynamic Overbooking Solver:** Combine predicted no-show probabilities with a simple linear programming model to compute the optimal number of slots to overbook per doctor per day, maximizing revenue while keeping patient wait time under 15 minutes.
