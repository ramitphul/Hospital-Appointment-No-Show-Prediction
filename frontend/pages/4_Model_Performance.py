"""
Page 4: Machine Learning Model Benchmarks & Operational Performance.
Evaluates the 6 trained classification algorithms, ROC/PR comparative curves,
threshold optimization curves, and probability calibration reliability.
"""
from pathlib import Path
import json
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Model Performance | Hospital AI", page_icon="📈", layout="wide")

st.markdown("# 📈 Machine Learning Benchmarks & Model Evaluation")
st.markdown("Comprehensive performance metrics and clinical evaluation across 6 classification algorithms.")

root_dir = Path(__file__).resolve().parents[2]
metrics_dir = root_dir / "artifacts" / "metrics"

# Load comparison table
comparison_file = metrics_dir / "model_comparison.csv"
final_test_file = metrics_dir / "final_test_metrics.json"

if comparison_file.exists():
    comp_df = pd.read_csv(comparison_file)
else:
    # Fallback to defaults
    comp_df = pd.DataFrame([
        {"Model": "XGBoost", "ROC-AUC": 0.7559, "PR-AUC": 0.4080, "F1-Score": 0.4604, "Recall": 0.8215, "Precision": 0.3198, "Accuracy": 0.6112, "Specificity": 0.5580},
        {"Model": "Gradient_Boosting", "ROC-AUC": 0.7514, "PR-AUC": 0.3998, "F1-Score": 0.1043, "Recall": 0.0571, "Precision": 0.5977, "Accuracy": 0.8019, "Specificity": 0.9903},
        {"Model": "Random_Forest", "ROC-AUC": 0.7511, "PR-AUC": 0.3976, "F1-Score": 0.4565, "Recall": 0.8562, "Precision": 0.3112, "Accuracy": 0.5883, "Specificity": 0.5205},
        {"Model": "Logistic_Regression", "ROC-AUC": 0.7427, "PR-AUC": 0.3831, "F1-Score": 0.4507, "Recall": 0.8279, "Precision": 0.3096, "Accuracy": 0.5925, "Specificity": 0.5329},
        {"Model": "Decision_Tree", "ROC-AUC": 0.7357, "PR-AUC": 0.3710, "F1-Score": 0.4505, "Recall": 0.8301, "Precision": 0.3091, "Accuracy": 0.5911, "Specificity": 0.5307},
        {"Model": "K_Nearest_Neighbors", "ROC-AUC": 0.7105, "PR-AUC": 0.3451, "F1-Score": 0.1941, "Recall": 0.1251, "Precision": 0.4328, "Accuracy": 0.7902, "Specificity": 0.9585},
    ])

# 1. Model Comparison Leaderboard Table
st.subheader("🏆 6-Algorithm Comparison Leaderboard (Validation Set)")
st.dataframe(
    comp_df.style.highlight_max(subset=["ROC-AUC", "PR-AUC", "F1-Score", "Recall"], color="#dcfce7")
                 .format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}", "F1-Score": "{:.4f}", "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "Specificity": "{:.4f}"}),
    use_container_width=True
)

st.markdown("---")

# 2. Final Held-Out Test Set Metrics
st.subheader("🎯 Selected Model: Calibrated XGBoost (Held-Out Test Set)")
if final_test_file.exists():
    with open(final_test_file, "r", encoding="utf-8") as f:
        m = json.load(f)
else:
    m = {
        "accuracy": 0.6501,
        "roc_auc": 0.7443,
        "pr_auc": 0.4045,
        "recall": 0.7195,
        "precision": 0.3312,
        "specificity": 0.6325,
        "f1_score": 0.4536,
        "threshold": 0.230,
        "true_positives": 3211,
        "false_negatives": 1252,
        "false_positives": 6485,
        "true_negatives": 11157,
    }

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("ROC-AUC Score", f"{m['roc_auc']:.4f}")
with m2:
    st.metric("PR-AUC Score", f"{m['pr_auc']:.4f}", delta="Baseline: 0.202", delta_color="normal")
with m3:
    st.metric("Recall (Sensitivity)", f"{m['recall']:.1%}", delta=f"{m['true_positives']:,} No-Shows Caught", delta_color="normal")
with m4:
    st.metric("Precision (PPV)", f"{m['precision']:.1%}")
with m5:
    st.metric("F1-Score", f"{m['f1_score']:.4f}", delta="Threshold: 0.230")

st.markdown("---")

# 3. Curve Visualizations Row
col_roc, col_pr = st.columns(2)

with col_roc:
    st.subheader("Comparative ROC Curves (All Models)")
    roc_img_path = metrics_dir / "roc_curves_comparison.png"
    if roc_img_path.exists():
        st.image(str(roc_img_path), use_container_width=True)
    else:
        st.info("ROC Curve generated during training pipeline.")

with col_pr:
    st.subheader("Comparative Precision-Recall Curves")
    pr_img_path = metrics_dir / "pr_curves_comparison.png"
    if pr_img_path.exists():
        st.image(str(pr_img_path), use_container_width=True)
    else:
        st.info("PR Curve generated during training pipeline.")

st.markdown("---")

# 4. Threshold & Calibration Row
col_thresh, col_cm = st.columns(2)

with col_thresh:
    st.subheader("Threshold Optimization Curve")
    thresh_img_path = metrics_dir / "threshold_optimization_curve.png"
    if thresh_img_path.exists():
        st.image(str(thresh_img_path), use_container_width=True)
    st.markdown("""
    **Why Decision Threshold 0.230 Was Selected:**
    - At the arbitrary **0.50 threshold**, recall collapses to only **7.5%**, missing 92.5% of all no-shows!
    - Setting the threshold to **0.230** strikes the optimal harmonic F1 balance, boosting recall to **71.95%** while retaining acceptable clinic intervention precision.
    """)

with col_cm:
    st.subheader("Confusion Matrix on Held-Out Test Cohort")
    cm_img_path = metrics_dir / "final_test_confusion_matrix.png"
    if cm_img_path.exists():
        st.image(str(cm_img_path), use_container_width=True)
    st.markdown(f"""
    **Test Cohort Breakdown (22,105 unseen appointments):**
    - **True Positives (No-Shows Correctly Caught):** `{m.get('true_positives', 3211):,}`
    - **False Negatives (Missed No-Shows):** `{m.get('false_negatives', 1252):,}`
    - **False Alarms (Attended but Flagged):** `{m.get('false_positives', 6485):,}`
    - **True Negatives (Attended & Unflagged):** `{m.get('true_negatives', 11157):,}`
    """)
