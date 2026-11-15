"""
Page 5: Explainable AI (XAI) using SHAP (SHapley Additive exPlanations).
Displays global feature importance beeswarm and bar plots, clinical feature
interpretability rankings, and interactive patient scenario attribution.
"""
from pathlib import Path
import json
import pandas as pd
from PIL import Image
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Explainable AI | Hospital AI", page_icon="🔍", layout="wide")

st.markdown("# 🔍 Explainable AI (SHAP Interpretability)")
st.markdown("Auditable machine learning explanations ensuring clinical transparency and operational trust.")

root_dir = Path(__file__).resolve().parents[2]
shap_dir = root_dir / "artifacts" / "shap"

st.subheader("1. Global Feature Importance (Test Cohort Sample)")
st.caption("SHAP values quantify each feature's average marginal contribution to the model's log-odds output.")

col_sum, col_bar = st.columns(2)

with col_sum:
    st.markdown("#### SHAP Beeswarm Summary Plot")
    sum_img = shap_dir / "shap_summary.png"
    if sum_img.exists():
        st.image(str(sum_img), use_container_width=True)
    else:
        st.info("SHAP Summary plot will be displayed once training pipeline finishes.")
    st.caption("Red points indicate high feature values; blue points indicate low feature values. Horizontal displacement reflects increase or decrease in no-show log-odds.")

with col_bar:
    st.markdown("#### SHAP Mean Absolute Importance Bar Plot")
    bar_img = shap_dir / "shap_bar.png"
    if bar_img.exists():
        st.image(str(bar_img), use_container_width=True)
    else:
        st.info("SHAP Bar plot will be displayed once training pipeline finishes.")
    st.caption("Average absolute impact on model output magnitude across all test cohort patients.")

st.markdown("---")

st.subheader("2. Clinical Interpretation of Top Predictive Drivers")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🕒 1. Scheduling Lead Time (`Waiting_Days` & `Is_Same_Day`)
    - **Observed Effect:** By far the strongest predictive factor in the model.
    - **Mechanism:** As days between booking and appointment increase, patient forgetfulness, schedule conflicts, symptom resolution, or alternate healthcare seeking rise dramatically.
    - **Hospital Action:** Introduce open-access same-day scheduling pods and automated cancellation-refill waitlist protocols.

    ### 📜 2. Historical Patient Profile (`prior_appointments` & `prior_no_shows`)
    - **Observed Effect:** A patient's past attendance frequency is strongly predictive of future behavior.
    - **Mechanism:** Chronic no-show behavior reflects persistent structural barriers (transportation, work inflexibility, dependent care).
    - **Hospital Action:** Flag repeat no-show patients for personalized social work or transportation assistance rather than punitive policies.
    """)

with col2:
    st.markdown("""
    ### 🎂 3. Age & Life Stage (`Age` & `Age_Group`)
    - **Observed Effect:** Young adults (20-39) show the highest risk of missing appointments, while seniors (60+) show the highest attendance compliance.
    - **Mechanism:** Older adults frequently have scheduled ongoing maintenance treatments and less volatile work hours.
    - **Hospital Action:** Tailor digital and text reminders to younger demographics while maintaining traditional phone reminders for older patients.

    ### 🩺 4. Chronic Illness Burden (`Num_Conditions`, `Hipertension`, `Diabetes`)
    - **Observed Effect:** Diagnosed chronic diseases correlate with *reduced* no-show probability (higher attendance).
    - **Mechanism:** Patients with severe conditions recognize the direct clinical necessity of regular medical checkups and medication refills.
    """)

st.markdown("---")

st.subheader("3. Ethical & Clinical Safety Guidelines")
st.warning(
    "⚠️ **Important Clinical Decision Support Notice:**\n\n"
    "1. **Association vs Causation:** SHAP values represent statistical feature attributions within the machine learning model, not verified medical causes.\n"
    "2. **Equitable Care:** Algorithmic risk predictions must never be used to deny care or discriminate against vulnerable patient groups.\n"
    "3. **Human-in-the-Loop:** Risk scores should guide affirmative supportive outreach (e.g., offering reminder calls, transit vouchers, or flexible scheduling), not punitive restrictions."
)
