"""
Streamlit Outpatient Appointment No-Show Predictor.
Interactive scheduling assistant for hospital receptionists and clinic coordinators.
"""
from pathlib import Path
import json
import requests
import streamlit as st

st.set_page_config(
    page_title="Clinic Outpatient No-Show Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean, professional styling without marketing fluff
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .stat-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.2rem;
    }
    .stat-desc {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🏥 Clinic Scheduling Assistant")
st.sidebar.info(
    "This tool helps scheduling coordinators evaluate appointment no-show risks "
    "at the time of booking, prioritize reminder calls, and optimize clinic capacity."
)

api_status = "Offline (Local Engine Active)"
api_color = "orange"
try:
    r = requests.get("http://127.0.0.1:8000/health", timeout=1.0)
    if r.status_code == 200 and r.json().get("model_loaded"):
        api_status = "Connected to FastAPI Backend"
        api_color = "green"
except Exception:
    pass

st.sidebar.markdown(f"**Backend Status:** :{api_color}[● {api_status}]")
st.sidebar.markdown("**Active Model:** Calibrated XGBoost")
st.sidebar.markdown("**Decision Threshold:** `0.230`")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Navigation")
st.sidebar.markdown("""
- **1. Overview Dashboard** - Cohort trends & lead time patterns
- **2. Patient Prediction** - Single appointment risk check
- **3. Batch Prediction** - Upload daily schedule CSV
- **4. Model Performance** - Algorithm benchmarks & curves
- **5. Explainable AI** - SHAP feature importance breakdown
""")

# Main Content
st.markdown('<div class="main-title">Hospital Outpatient No-Show Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'A decision-support tool for clinic receptionists and outpatient scheduling coordinators '
    'to anticipate missed appointments, prioritize outreach, and reduce idle doctor hours.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-label">Training Appointments</div>
        <div class="stat-num">110,521</div>
        <div class="stat-desc">Brazilian Public Clinic Cohort</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-label">Historical No-Show Rate</div>
        <div class="stat-num" style="color: #b91c1c;">20.19%</div>
        <div class="stat-desc">Approximately 1 in 5 missed</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-label">Selected Model (ROC-AUC)</div>
        <div class="stat-num" style="color: #047857;">0.7443</div>
        <div class="stat-desc">Evaluated on 22,105 test visits</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-label">Operational Recall</div>
        <div class="stat-num" style="color: #4338ca;">71.95%</div>
        <div class="stat-desc">Catches 3,211 of 4,463 test no-shows</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("How Staff Use This Tool")
    st.markdown("""
    When an appointment is scheduled, the model calculates the probability of a no-show based on booking lead time, age group, historical attendance, and clinic location.

    Instead of sending the same reminder text to everyone, staff can use a tiered approach:

    - **Low Risk (<30% probability):** Standard automated confirmation text. No staff follow-up needed.
    - **Medium Risk (30% to 59% probability):** Two automated text reminders (72h and 24h prior) asking the patient to confirm attendance with a single tap.
    - **High Risk (≥60% probability):** A clinic receptionist calls the patient 48h prior to check if they need transportation help or a rescheduled time. Schedulers can also strategically overbook this slot knowing there is a high likelihood of vacancy.
    """)

with col_right:
    st.subheader("Quick System Summary")
    st.markdown("""
    - **Dataset:** 110,527 appointments from public clinics in Vitória, Brazil.
    - **Target:** 0 = Attended (79.8%), 1 = No-Show (20.2%).
    - **Models Tested:** Logistic Regression, Decision Tree, Random Forest, KNN, Gradient Boosting, XGBoost.
    - **Top Performer:** XGBoost calibrated with Sigmoid probability scaling.
    - **Threshold Selection:** Lowered from 0.50 to 0.23 to raise recall from 7.5% to 72.0%.
    - **Data Leakage Check:** Strict chronological train-test split; post-booking SMS signals excluded from booking predictions.
    """)

st.markdown("---")
st.info("Select a page from the left sidebar to try a patient prediction, upload a schedule CSV, or review the benchmark results.")
