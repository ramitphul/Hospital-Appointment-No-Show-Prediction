"""
Page 2: Real-time Patient Appointment No-Show Risk Predictor.
Interactive clinical decision form connecting to FastAPI backend
with live SHAP feature attribution and operational guidance.
"""
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
import plotly.graph_objects as go
import requests
import streamlit as st

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))

from src.models.predict import NoShowPredictor
from src.utils.config import load_config

st.set_page_config(page_title="Patient Risk Prediction | Hospital AI", page_icon="🩺", layout="wide")

st.markdown("# 🩺 Patient Appointment No-Show Risk Calculator")
st.markdown("Clinical decision-support tool to evaluate missed appointment probability at the moment of booking.")

# Top Neighbourhood options from real dataset
NEIGHBOURHOODS = [
    "JARDIM CAMBURI", "MARIA ORTIZ", "RESISTÊNCIA", "JARDIM DA PENHA",
    "ITARARÉ", "CENTRO", "TABUAZEIRO", "SANTA MARTHA", "JESUS DE NAZARETH",
    "BONFIM", "SANTO ANTÔNIO", "CARATOÍRA", "SÃO PEDRO", "ILHA DO PRÍNCIPE",
    "PRAIA DO CANTO", "ANDORINHAS", "DA PENHA", "GURIGICA", "SÃO JOSÉ", "BELA VISTA",
    "OTHER"
]

# Form
with st.form(key="prediction_form"):
    st.subheader("1. Patient Demographics & Identification")
    col1, col2, col3 = st.columns(3)
    with col1:
        patient_id = st.text_input("Patient ID (Optional)", value="99284712039")
    with col2:
        age = st.number_input("Patient Age", min_value=0, max_value=115, value=35, step=1)
    with col3:
        gender = st.selectbox("Gender", options=["F", "M"], index=0, format_func=lambda x: "Female (F)" if x == "F" else "Male (M)")

    st.subheader("2. Appointment Scheduling Details")
    col4, col5, col6 = st.columns(3)
    today = date.today()
    with col4:
        sched_date = st.date_input("Scheduled Booking Date", value=today)
        sched_time = st.time_input("Booking Time", value=datetime.strptime("10:30", "%H:%M").time())
    with col5:
        appt_date = st.date_input("Appointment Date", value=today + timedelta(days=7))
    with col6:
        neighbourhood = st.selectbox("Clinic / Neighbourhood", options=NEIGHBOURHOODS, index=5)

    st.subheader("3. Clinical History & Medical Conditions")
    col7, col8, col9, col10 = st.columns(4)
    with col7:
        scholarship = st.checkbox("Welfare Scholarship (Bolsa Família)", value=False)
        hipertension = st.checkbox("Diagnosed Hypertension", value=False)
    with col8:
        diabetes = st.checkbox("Diagnosed Diabetes", value=False)
        alcoholism = st.checkbox("Alcohol Dependency", value=False)
    with col9:
        handicap = st.checkbox("Physical / Motor Handicap", value=False)
    with col10:
        prior_appts = st.number_input("Prior Hospital Appointments", min_value=0, max_value=50, value=2, step=1)
        prior_noshows = st.number_input("Prior Missed Appointments", min_value=0, max_value=50, value=1, step=1)

    submit_button = st.form_submit_button(label="🔮 Predict No-Show Risk", use_container_width=True)

if submit_button:
    # 1. Validation
    if appt_date < sched_date:
        st.error("❌ Validation Error: Appointment date cannot precede scheduled booking date.")
        st.stop()

    lead_days = (appt_date - sched_date).days
    sched_dt_str = f"{sched_date.isoformat()}T{sched_time.strftime('%H:%M:%S')}Z"
    appt_dt_str = f"{appt_date.isoformat()}T00:00:00Z"

    payload = {
        "PatientId": float(patient_id) if patient_id and patient_id.replace(".", "").isdigit() else None,
        "AppointmentID": 5600000,
        "Gender": gender,
        "ScheduledDay": sched_dt_str,
        "AppointmentDay": app_dt_str if 'app_dt_str' in locals() else appt_dt_str,
        "Age": int(age),
        "Neighbourhood": neighbourhood,
        "Scholarship": 1 if scholarship else 0,
        "Hipertension": 1 if hipertension else 0,
        "Diabetes": 1 if diabetes else 0,
        "Alcoholism": 1 if alcoholism else 0,
        "Handcap": 1 if handicap else 0,
        "prior_appointments": int(prior_appts),
        "prior_no_shows": int(prior_noshows),
    }

    # 2. Call FastAPI or fallback
    prediction_result = None
    with st.spinner("Analyzing appointment risk through ML pipeline..."):
        try:
            res = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=3.0)
            if res.status_code == 200:
                prediction_result = res.json()
        except Exception:
            pass

        # Fallback to local inference engine if FastAPI is not yet running
        if prediction_result is None:
            try:
                predictor = NoShowPredictor()
                res_obj = predictor.predict_single(payload, explain=True)
                prediction_result = res_obj.to_dict()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

    # 3. Display Results
    st.markdown("---")
    st.subheader("📋 Clinical Assessment Results")

    prob = prediction_result["no_show_probability"]
    category = prediction_result["risk_category"]
    pred_label = prediction_result["prediction_label"]
    threshold = prediction_result["threshold_used"]

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        # Gauge Meter for Probability
        gauge_color = "#10b981" if prob < 0.30 else ("#f59e0b" if prob < 0.60 else "#ef4444")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Predicted No-Show Probability (%)", 'font': {'size': 18}},
            number={'suffix': "%", 'font': {'size': 32}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 30], 'color': "#d1fae5"},
                    {'range': [30, 60], 'color': "#fef3c7"},
                    {'range': [60, 100], 'color': "#fee2e2"},
                ],
                'threshold': {
                    'line': {'color': "purple", 'width': 3},
                    'thickness': 0.75,
                    'value': threshold * 100,
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        badge_class = f"risk-{category.lower()}"
        st.markdown(f"### Overall Risk Tier: <span class='{badge_class}'>{category} Risk</span>", unsafe_allow_html=True)
        st.markdown(f"**Predicted Attendance Outcome:** `{pred_label}`")
        st.markdown(f"**Calculated Lead Time:** `{lead_days}` days between booking and appointment")
        st.markdown(f"**Decision Threshold Applied:** `{threshold:.2f}` (Optimal F1 calibration)")

        # Recommended Operational Action
        if category == "High":
            st.error(
                "**Recommended Staff Action (High Risk):**\n"
                "- Flag for a personal reminder call from reception 48 hours prior.\n"
                "- Ask if the patient needs help with public transit or a different time.\n"
                "- Safe to consider a selective backup overbook if the clinic has a waitlist."
            )
        elif category == "Medium":
            st.warning(
                "**Recommended Staff Action (Medium Risk):**\n"
                "- Send automated text reminders 72 hours and 24 hours prior.\n"
                "- Include a 1-tap confirmation / reschedule link in the reminder message."
            )
        else:
            st.success(
                "**Recommended Staff Action (Low Risk):**\n"
                "- Patient profile indicates reliable attendance.\n"
                "- Standard automated booking confirmation is sufficient. No staff phone calls needed."
            )

    # 4. SHAP Feature Contributions Breakdown
    factors = prediction_result.get("top_contributing_factors")
    if factors:
        st.markdown("---")
        st.subheader("Why Did the Model Assign This Score?")
        st.caption("SHAP feature contributions showing the main patient and scheduling variables that pushed this prediction higher or lower.")

        pos_factors = factors.get("top_risk_increasing_factors", [])
        neg_factors = factors.get("top_risk_decreasing_factors", [])

        sh_col1, sh_col2 = st.columns(2)
        with sh_col1:
            st.markdown("#### Factors Increasing No-Show Risk")
            if pos_factors:
                for f in pos_factors:
                    st.markdown(f"- **{f['feature']}**: `+{f['shap_value']:.3f}` log-odds impact")
            else:
                st.info("No significant risk-increasing factors identified.")

        with sh_col2:
            st.markdown("#### Factors Supporting Attendance")
            if neg_factors:
                for f in neg_factors:
                    st.markdown(f"- **{f['feature']}**: `{f['shap_value']:.3f}` log-odds impact")
            else:
                st.info("No significant protective factors identified.")
