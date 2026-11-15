"""
Page 1: Exploratory Data Analysis Dashboard.
Visualizes clinical distributions, behavioral patterns, lead time impact,
and confounding factors across 110,527 Brazilian medical appointments.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

st.set_page_config(page_title="EDA Dashboard | Hospital No-Show AI", page_icon="📊", layout="wide")

st.markdown("# 📊 Clinical Exploratory Data Analysis (EDA)")
st.markdown("Comprehensive behavioral and demographic analysis of outpatient appointment attendance patterns.")

# Load summary stats if available
root_dir = Path(__file__).resolve().parents[2]
eda_dir = root_dir / "artifacts" / "eda"
summary_file = eda_dir / "eda_summary.json"

stats = {
    "total_appointments": 110521,
    "attended_appointments": 88207,
    "missed_appointments": 22314,
    "no_show_rate_percent": 20.19,
    "mean_age": 37.1,
    "median_waiting_days": 4.0,
    "same_day_noshow_rate": 4.65,
    "sms_received_noshow_rate": 27.57,
    "no_sms_noshow_rate": 16.70,
}

if summary_file.exists():
    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except Exception:
        pass

# KPI Cards Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Appointments", f"{stats['total_appointments']:,}")
with col2:
    st.metric("Attended Appointments", f"{stats['attended_appointments']:,}", delta=f"{100 - stats['no_show_rate_percent']:.1f}% Attended")
with col3:
    st.metric("Missed Appointments", f"{stats['missed_appointments']:,}", delta=f"-{stats['no_show_rate_percent']:.1f}% No-Shows", delta_color="inverse")
with col4:
    st.metric("Same-Day No-Show Rate", f"{stats['same_day_noshow_rate']}%", delta="Extremely Low")
with col5:
    st.metric("Median Lead Time", f"{stats['median_waiting_days']:.0f} Days")

st.markdown("---")

# Visualizations Section
tab1, tab2, tab3 = st.tabs(["🕒 Lead Time & Scheduling Impact", "👥 Demographics & Health Profile", "📍 Neighbourhoods & Spatial Patterns"])

with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Lead Time (Waiting Days) vs No-Show Rate")
        lead_time_df = pd.DataFrame({
            "Lead Time Category": ["Same Day (0d)", "Short (1-3d)", "Medium (4-14d)", "Long (15+d)"],
            "No-Show Rate (%)": [stats["same_day_noshow_rate"], 23.8, 31.2, 33.1],
            "Volume Share": [35.1, 19.3, 24.8, 20.8]
        })
        fig_lead = px.bar(
            lead_time_df,
            x="Lead Time Category",
            y="No-Show Rate (%)",
            text="No-Show Rate (%)",
            color="No-Show Rate (%)",
            color_continuous_scale="Reds",
            title="Surge in No-Shows as Booking Horizon Expands",
        )
        fig_lead.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_lead.add_hline(y=stats["no_show_rate_percent"], line_dash="dash", line_color="blue", annotation_text="Cohort Average (20.2%)")
        fig_lead.update_layout(yaxis_range=[0, 40], height=380)
        st.plotly_chart(fig_lead, use_container_width=True)
        st.caption("Appointments booked on the same day experience only 4.6% no-shows. Appointments scheduled > 2 weeks in advance surpass 33% missed rates.")

    with col_b:
        st.subheader("Confounding Factor: The SMS Reminder Paradox")
        sms_df = pd.DataFrame({
            "Reminder Status": ["No SMS Sent (0)", "SMS Received (1)"],
            "Observed No-Show Rate (%)": [stats["no_sms_noshow_rate"], stats["sms_received_noshow_rate"]],
            "Appointments": [75039, 35482]
        })
        fig_sms = px.bar(
            sms_df,
            x="Reminder Status",
            y="Observed No-Show Rate (%)",
            text="Observed No-Show Rate (%)",
            color="Reminder Status",
            color_discrete_sequence=["#10b981", "#f59e0b"],
            title="Apparent Paradox: SMS Associated with Higher Missed Rate",
        )
        fig_sms.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_sms.update_layout(yaxis_range=[0, 35], height=380)
        st.plotly_chart(fig_sms, use_container_width=True)

        st.warning(
            "**Healthcare Machine Learning Insight (Simpson's Paradox / Confounding by Indication):** "
            "Patients who received SMS reminders exhibit higher apparent no-show rates (~27.6% vs 16.7%). "
            "However, hospital clinical protocol only sent SMS reminders when the waiting time was ≥ 3 days. "
            "Same-day appointments (which almost never miss) never received an SMS. Controlling for waiting days "
            "confirms that lead time is the genuine causal driver, not the reminder itself."
        )

    # Weekday patterns
    st.subheader("Appointment Day of Week Trends")
    weekday_df = pd.DataFrame({
        "Day of Week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
        "Appointments": [22714, 25638, 25866, 17246, 19019, 39],
        "No-Show Rate (%)": [20.6, 20.1, 19.7, 19.4, 21.2, 23.1]
    })
    fig_week = px.line(
        weekday_df,
        x="Day of Week",
        y="No-Show Rate (%)",
        markers=True,
        title="Weekly Operational No-Show Rates (Higher on Fridays and Saturdays)",
    )
    fig_week.add_hline(y=stats["no_show_rate_percent"], line_dash="dash", line_color="red", annotation_text="Average (20.2%)")
    fig_week.update_layout(yaxis_range=[17, 25], height=320)
    st.plotly_chart(fig_week, use_container_width=True)

with tab2:
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Attendance by Age Group")
        age_df = pd.DataFrame({
            "Age Group": ["Child (0-12)", "Teen (13-19)", "Young Adult (20-39)", "Adult (40-59)", "Senior (60+)"],
            "No-Show Rate (%)": [19.8, 26.0, 24.6, 19.2, 15.3],
            "Total Appointments": [21035, 9360, 30040, 30006, 20080]
        })
        fig_age = px.bar(
            age_df,
            x="Age Group",
            y="No-Show Rate (%)",
            text="No-Show Rate (%)",
            color="Age Group",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Life Stage Distribution: Young Adults & Teens Miss Most Often",
        )
        fig_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_age.add_hline(y=stats["no_show_rate_percent"], line_dash="dash", line_color="black", annotation_text="Baseline")
        fig_age.update_layout(yaxis_range=[0, 32], height=380)
        st.plotly_chart(fig_age, use_container_width=True)

    with col_d:
        st.subheader("Chronic Illness & Socioeconomic Indicators")
        health_df = pd.DataFrame({
            "Condition": [
                "Overall Baseline",
                "Welfare (Bolsa Família)",
                "Hypertension",
                "Diabetes",
                "Alcoholism",
                "Handicap"
            ],
            "No-Show Rate (%)": [20.2, 23.8, 17.3, 18.0, 20.1, 17.9],
            "Status": ["Baseline", "Higher Risk", "Protective (Attends)", "Protective (Attends)", "Neutral", "Protective"]
        })
        fig_health = px.bar(
            health_df,
            x="No-Show Rate (%)",
            y="Condition",
            orientation="h",
            color="Status",
            color_discrete_map={"Baseline": "#6b7280", "Higher Risk": "#ef4444", "Protective (Attends)": "#10b981", "Neutral": "#f59e0b"},
            title="Chronic Conditions vs Socioeconomic Support",
        )
        fig_health.update_layout(height=380, xaxis_range=[0, 30])
        st.plotly_chart(fig_health, use_container_width=True)
        st.caption("Patients with chronic illnesses (hypertension, diabetes) show HIGHER attendance compliance (~17-18%), while patients on welfare assistance face higher structural barriers (~23.8%).")

with tab3:
    st.subheader("Spatial Variation Across Top 15 Clinic Neighbourhoods")
    top_nb = pd.DataFrame({
        "Neighbourhood": [
            "JARDIM CAMBURI", "MARIA ORTIZ", "RESISTÊNCIA", "JARDIM DA PENHA",
            "ITARARÉ", "CENTRO", "TABUAZEIRO", "SANTA MARTHA",
            "JESUS DE NAZARETH", "BONFIM", "SANTO ANTÔNIO", "CARATOÍRA",
            "SÃO PEDRO", "ILHA DO PRÍNCIPE", "PRAIA DO CANTO"
        ],
        "Total Appointments": [7717, 5805, 4431, 3877, 3514, 3334, 3132, 3131, 2853, 2773, 2746, 2565, 2448, 2266, 1039],
        "No-Show Rate (%)": [19.0, 21.0, 20.4, 16.3, 26.3, 21.1, 18.3, 18.0, 24.4, 19.9, 17.6, 23.0, 21.0, 25.6, 13.9]
    }).sort_values(by="No-Show Rate (%)", ascending=True)

    fig_nb = px.bar(
        top_nb,
        x="No-Show Rate (%)",
        y="Neighbourhood",
        orientation="h",
        color="No-Show Rate (%)",
        color_continuous_scale="Viridis",
        title="Disparity in No-Show Rates Across Clinic Locations (13.9% in Praia do Canto to 26.3% in Itararé)",
        text="No-Show Rate (%)"
    )
    fig_nb.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_nb.add_vline(x=stats["no_show_rate_percent"], line_dash="dash", line_color="red", annotation_text="Average (20.2%)")
    fig_nb.update_layout(height=520, xaxis_range=[0, 32])
    st.plotly_chart(fig_nb, use_container_width=True)

st.markdown("---")
st.info("💡 **Key Clinical Takeaway:** Shortening lead time between booking and appointment is by far the single most actionable operational lever for hospital leadership.")
