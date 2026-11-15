"""
Page 3: Batch Appointment Prediction & Capacity Planning.
Upload hospital scheduling CSVs, validate schema, generate bulk predictions,
and export prioritized high-risk patient intervention lists.
"""
from pathlib import Path
import sys
import io
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))

from src.models.predict import NoShowPredictor
from src.utils.config import get_resolved_path, load_config

st.set_page_config(page_title="Batch Predictions | Hospital AI", page_icon="📑", layout="wide")

st.markdown("# 📑 Batch Schedule Prediction & Capacity Planning")
st.markdown("Evaluate entire daily or weekly hospital appointment rosters to optimize clinic utilization.")

config = load_config()
sample_csv_path = get_resolved_path(config["paths"]["sample_data"])

col_up, col_btn = st.columns([3, 1])

with col_up:
    uploaded_file = st.file_uploader("Upload Appointment Schedule CSV", type=["csv"])

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    load_demo_btn = st.button("📂 Load Demo Batch (300 Patients)", use_container_width=True)

df_to_predict = None

if uploaded_file is not None:
    try:
        df_to_predict = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded uploaded CSV: {len(df_to_predict):,} records.")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

elif load_demo_btn:
    if sample_csv_path.exists():
        df_to_predict = pd.read_csv(sample_csv_path)
        st.info(f"Loaded demo dataset from {sample_csv_path.name} ({len(df_to_predict):,} appointments).")
    else:
        st.warning("Sample dataset not found. Please upload a CSV.")

if df_to_predict is not None:
    st.markdown("### Raw Appointments Preview")
    st.dataframe(df_to_predict.head(5), use_container_width=True)

    # Validate minimal columns
    required = ["Age", "Gender", "ScheduledDay", "AppointmentDay"]
    missing = [c for c in required if c not in df_to_predict.columns]
    if missing:
        st.error(f"❌ Missing required columns in uploaded CSV: {missing}. Required: {required}")
        st.stop()

    if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
        with st.spinner("Processing batch through calibrated ML model..."):
            try:
                predictor = NoShowPredictor()
                results_df = predictor.predict_batch(df_to_predict)
            except Exception as e:
                st.error(f"Batch prediction error: {e}")
                st.stop()

        # Summary KPIs
        total = len(results_df)
        pred_noshows = int(results_df["predicted_no_show"].sum())
        noshow_pct = (pred_noshows / total) * 100
        risk_counts = results_df["risk_category"].value_counts()
        high_risk_count = int(risk_counts.get("High", 0))
        med_risk_count = int(risk_counts.get("Medium", 0))
        low_risk_count = int(risk_counts.get("Low", 0))

        st.markdown("---")
        st.subheader("📊 Batch Summary & Risk Stratification")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Appointments", f"{total:,}")
        with kpi2:
            st.metric("Predicted No-Shows", f"{pred_noshows:,}", delta=f"{noshow_pct:.1f}% Projected Missed", delta_color="inverse")
        with kpi3:
            st.metric("High-Risk Patients (≥60%)", f"{high_risk_count:,}", delta="Urgent Intervention Required", delta_color="inverse")
        with kpi4:
            st.metric("Low-Risk Patients (<30%)", f"{low_risk_count:,}", delta="Likely to Attend")

        # Visual breakdown of risk tiers
        pie_df = pd.DataFrame({
            "Risk Tier": ["Low Risk (<30%)", "Medium Risk (30-60%)", "High Risk (≥60%)"],
            "Count": [low_risk_count, med_risk_count, high_risk_count]
        })
        fig_pie = px.pie(
            pie_df,
            names="Risk Tier",
            values="Count",
            color="Risk Tier",
            color_discrete_map={
                "Low Risk (<30%)": "#10b981",
                "Medium Risk (30-60%)": "#f59e0b",
                "High Risk (≥60%)": "#ef4444"
            },
            hole=0.45,
            title="Distribution of Patient Risk Tiers in Batch"
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Filterable results table
        st.subheader("📋 Enriched Predictions Table")
        filter_tier = st.multiselect("Filter by Risk Tier", options=["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        filtered_df = results_df[results_df["risk_category"].isin(filter_tier)]

        # Highlight high risk records
        display_cols = [
            "PatientId", "Age", "Gender", "Neighbourhood", "ScheduledDay", "AppointmentDay",
            "no_show_probability", "predicted_no_show", "prediction_label", "risk_category"
        ]
        available_display_cols = [c for c in display_cols if c in filtered_df.columns]

        st.dataframe(
            filtered_df[available_display_cols].sort_values(by="no_show_probability", ascending=False),
            use_container_width=True
        )

        # Download button
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Full Predictions CSV",
            data=csv_buffer.getvalue(),
            file_name="hospital_noshow_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
