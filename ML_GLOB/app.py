"""
app.py
Student Performance Prediction and Academic Risk Analysis Platform
Streamlit Web Application integrating core Machine Learning concepts:
- Data Preprocessing & Cleaning (Imputation, Encoding, Scaling, EDA)
- Logistic Regression Classification
- Decision Tree Classification & Live Sample Inference
- Naïve Bayes Classification
- K-Nearest Neighbors (KNN) Classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import os

from ml_pipeline import (
    analyze_dataset_health,
    prepare_train_test_data,
    train_all_models,
    evaluate_all_models,
    predict_student_risk_all_models,
    classify_new_sample_dt,
    generate_academic_interventions,
    RISK_CLASSES,
    FEATURE_COLUMNS_NUM,
    FEATURE_COLUMNS_CAT
)
from data_generator import generate_student_dataset

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EduPredict | Academic Risk & ML Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.4);
    }
    
    .main-header h1 {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: #c7d2fe !important;
        font-size: 1.05rem;
        max-width: 900px;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    }

    .risk-banner-low {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 14px;
        border-left: 8px solid #10b981;
    }

    .risk-banner-mod {
        background: linear-gradient(135deg, #78350f 0%, #b45309 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 14px;
        border-left: 8px solid #f59e0b;
    }

    .risk-banner-high {
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 14px;
        border-left: 8px solid #ef4444;
    }

    .intervention-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 5px solid #6366f1;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATA LOADING & CACHING
# -----------------------------------------------------------------------------
DATA_FILE = "student_performance_data.csv"

@st.cache_data
def load_default_data():
    if not os.path.exists(DATA_FILE):
        generate_student_dataset(n_samples=1200, save_path=DATA_FILE)
    return pd.read_csv(DATA_FILE)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & HYPERPARAMETERS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🎓 **EduPredict Settings**")
st.sidebar.caption("Student Performance & Academic Risk Engine")

data_source = st.sidebar.radio(
    "Select Dataset Source:",
    ["Built-in Dataset (1,200 Students)", "Upload Custom CSV"],
    index=0
)

if data_source == "Built-in Dataset (1,200 Students)":
    raw_df = load_default_data()
    st.sidebar.success(f"✓ Loaded {len(raw_df)} student records")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Student CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✓ Uploaded {len(raw_df)} records")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            raw_df = load_default_data()
    else:
        st.sidebar.info("Awaiting file upload. Using default dataset as fallback.")
        raw_df = load_default_data()

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚙️ **Data Preprocessing Config**")
missing_strategy = st.sidebar.selectbox("Numeric Imputation Strategy", ["median", "mean"], index=0)
scaling_method = st.sidebar.selectbox("Feature Scaling Method", ["standard", "minmax"], index=0)
test_split = st.sidebar.slider("Test Set Split Ratio", min_value=0.15, max_value=0.35, value=0.20, step=0.05)

st.sidebar.markdown("#### ⚙️ **Model Hyperparameter Tuning**")
with st.sidebar.expander("Classifier Hyperparameters", expanded=False):
    logreg_c = st.slider("Logistic Regression Regularization (C)", 0.01, 10.0, 1.0, 0.1)
    dt_depth = st.slider("Decision Tree Max Depth", 2, 12, 5, 1)
    dt_criterion = st.selectbox("Decision Tree Criterion", ["gini", "entropy"], index=0)
    knn_k = st.slider("KNN Neighbors (K)", 3, 15, 5, 2)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Machine Learning Architecture:**
- 🔹 Data Preprocessing & Cleaning
- 🔹 Logistic Regression
- 🔹 Decision Tree & Inference
- 🔹 Naïve Bayes
- 🔹 K-Nearest Neighbors (KNN)
""")

# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Student Performance Prediction & Academic Risk Analysis</h1>
    <p>A unified Machine Learning platform for early academic warning, multi-classifier risk benchmarking, and data-driven educational intervention.</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TRAIN & CACHE MACHINE LEARNING PIPELINE
# -----------------------------------------------------------------------------
@st.cache_resource
def run_cached_pipeline(df_data, missing_strat, scale_meth, t_split, c_val, depth_val, crit_val, k_val):
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_train_test_data(
        df_data,
        test_size=t_split,
        missing_strategy=missing_strat,
        scaling_method=scale_meth
    )
    models = train_all_models(
        X_train, y_train,
        dt_depth=depth_val,
        dt_criterion=crit_val,
        knn_k=k_val,
        logreg_c=c_val
    )
    results, summary_df = evaluate_all_models(models, X_test, y_test)
    return X_train, X_test, y_train, y_test, preprocessor, feature_names, models, results, summary_df

X_train, X_test, y_train, y_test, preprocessor, feature_names, models, results, summary_df = run_cached_pipeline(
    raw_df, missing_strategy, scaling_method, test_split, logreg_c, dt_depth, dt_criterion, knn_k
)

# -----------------------------------------------------------------------------
# TOP KPI METRICS SUMMARY
# -----------------------------------------------------------------------------
best_model_row = summary_df.loc[summary_df["Accuracy"].idxmax()]
high_risk_count = (raw_df["Academic_Risk"] == "High Risk").sum() if "Academic_Risk" in raw_df.columns else 0

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    st.metric("Total Student Records", f"{len(raw_df):,}", "Active Cohort")
with col_kpi2:
    st.metric("Top Model Accuracy", f"{best_model_row['Accuracy']}%", best_model_row['Model'].split(' ')[0])
with col_kpi3:
    st.metric("High-Risk Students", f"{high_risk_count}", f"{(high_risk_count/len(raw_df)*100):.1f}% of cohort", delta_color="inverse")
with col_kpi4:
    st.metric("Active ML Models", "4 Classifiers", "Ensemble Benchmark")

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Dataset & Preprocessing Pipeline",
    "🤖 2. Multi-Model Benchmark Leaderboard",
    "🎯 3. Live Student Risk Predictor",
    "📈 4. What-If Simulator & Cohort Audit"
])


# =============================================================================
# TAB 1: DATASET & PREPROCESSING PIPELINE
# =============================================================================
with tab1:
    st.subheader("Data Preprocessing & Health Assessment")
    st.caption("Demonstrating missing value imputation, categorical encoding, outlier profiling, and feature scaling.")

    # Data profiling metrics
    health = analyze_dataset_health(raw_df)

    col_t2_a, col_t2_b, col_t2_c = st.columns(3)
    with col_t2_a:
        st.info(f"**Initial Missing Values:** {health['total_missing']} cells across dataset")
    with col_t2_b:
        st.info(f"**Imputation Strategy Applied:** `{missing_strategy.upper()}` for numeric, `MODE` for categories")
    with col_t2_c:
        st.info(f"**Scaling Method Applied:** `{scaling_method.upper()}` scaler on normalized feature space")

    # Raw Data Explorer
    with st.expander("🔍 Explore Raw Student Dataset", expanded=False):
        st.dataframe(raw_df.head(15), use_container_width=True)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_export = raw_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Raw Dataset CSV", data=csv_export, file_name="student_raw_data.csv", mime="text/csv")

    st.markdown("---")

    # Step-by-step visual demonstration of Preprocessing Pipeline
    col_pre1, col_pre2 = st.columns(2)

    with col_pre1:
        st.markdown("#### **Step 1: Missing Value Imputation**")
        missing_df = health["missing"]
        if not missing_df.empty:
            fig_miss = px.bar(
                missing_df,
                x="Column",
                y="Missing_Count",
                text="Missing_Count",
                color="Missing_Count",
                color_continuous_scale="Reds",
                title="Missing Values Identified per Feature"
            )
            fig_miss.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_miss, use_container_width=True)
            st.caption(f"✓ All missing cells have been successfully imputed via **{missing_strategy.title()} Imputation**.")
        else:
            st.success("No missing values detected in the uploaded dataset.")

    with col_pre2:
        st.markdown("#### **Step 2: Class Balance Distribution**")
        if "Academic_Risk" in raw_df.columns:
            class_counts = raw_df["Academic_Risk"].value_counts().reset_index()
            class_counts.columns = ["Academic_Risk", "Count"]
            color_map = {"Low Risk": "#10b981", "Moderate Risk": "#f59e0b", "High Risk": "#ef4444"}
            fig_pie = px.pie(
                class_counts,
                values="Count",
                names="Academic_Risk",
                color="Academic_Risk",
                color_discrete_map=color_map,
                hole=0.45,
                title="Cohort Academic Risk Distribution"
            )
            fig_pie.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    col_pre3, col_pre4 = st.columns(2)
    with col_pre3:
        st.markdown("#### **Step 3: Feature Scaling Transformation**")
        st.caption("Distribution of `Attendance_Rate` before vs after transformation.")
        
        # Plot comparison of raw vs scaled attendance
        raw_vals = raw_df["Attendance_Rate"].dropna()
        attend_idx = feature_names.index("Attendance_Rate")
        scaled_vals = X_train[:, attend_idx]

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=raw_vals, name="Raw Attendance (%)", opacity=0.6, marker_color="#3b82f6"))
        fig_dist.add_trace(go.Histogram(x=scaled_vals * 15 + 75, name="Scaled Z-Score (Rescaled)", opacity=0.6, marker_color="#8b5cf6"))
        fig_dist.update_layout(
            barmode='overlay',
            title="Feature Scaling Distribution Comparison",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_pre4:
        st.markdown("#### **Step 4: Outlier Detection (IQR Method)**")
        outlier_data = pd.DataFrame(list(health["outliers"].items()), columns=["Feature", "Outliers Detected"])
        fig_out = px.bar(
            outlier_data,
            x="Feature",
            y="Outliers Detected",
            color="Outliers Detected",
            color_continuous_scale="Viridis",
            title="IQR Outlier Analysis Across Numeric Features"
        )
        fig_out.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_out, use_container_width=True)

    # Correlation Matrix
    st.markdown("---")
    st.markdown("#### **Academic Feature Correlation Matrix**")
    numeric_df = raw_df[FEATURE_COLUMNS_NUM].copy().dropna()
    corr = numeric_df.corr().round(2)
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Pearson Correlation Heatmap Between Academic Features"
    )
    fig_corr.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)


# =============================================================================
# TAB 2: MULTI-MODEL BENCHMARK LEADERBOARD
# =============================================================================
with tab2:
    st.subheader("Multi-Model Benchmark Leaderboard")
    st.caption("Cross-comparison of Logistic Regression, Decision Tree, Naïve Bayes, and K-Nearest Neighbors.")

    # Leaderboard Table
    st.dataframe(
        summary_df.style.highlight_max(subset=["Accuracy", "F1-Score (%)"], color="#dcfce7"),
        use_container_width=True
    )

    # Comparative Bar Chart
    df_melted = pd.melt(
        summary_df,
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision (%)", "Recall (%)", "F1-Score (%)"],
        var_name="Metric",
        value_name="Percentage"
    )
    fig_bench = px.bar(
        df_melted,
        x="Model",
        y="Percentage",
        color="Metric",
        barmode="group",
        text_auto=".1f",
        title="Side-by-Side Model Performance Comparison"
    )
    fig_bench.update_layout(
        yaxis_range=[50, 100],
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_bench, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔬 **Model Deep Dive & Confusion Matrices**")

    # Tabs for individual model details
    m_tabs = st.tabs([
        "Logistic Regression",
        "Decision Tree",
        "Naïve Bayes",
        "KNN Classifier"
    ])

    model_names_map = {
        0: "Logistic Regression",
        1: "Decision Tree",
        2: "Naïve Bayes",
        3: "KNN Classifier"
    }

    for idx, (tab_idx, model_full_name) in enumerate(model_names_map.items()):
        with m_tabs[idx]:
            m_res = results[model_full_name]
            col_d1, col_d2 = st.columns([1, 1])

            with col_d1:
                st.markdown(f"#### **Confusion Matrix: {model_full_name}**")
                cm = m_res["confusion_matrix"]
                fig_cm = px.imshow(
                    cm,
                    x=RISK_CLASSES,
                    y=RISK_CLASSES,
                    labels=dict(x="Predicted Risk", y="Actual Risk", color="Student Count"),
                    text_auto=True,
                    color_continuous_scale="Purples",
                    title="Confusion Matrix"
                )
                fig_cm.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_d2:
                st.markdown("#### **Classification Report Metrics**")
                report = m_res["classification_report"]
                report_rows = []
                for cls_name in RISK_CLASSES:
                    if cls_name in report:
                        report_rows.append({
                            "Risk Class": cls_name,
                            "Precision": round(report[cls_name]["precision"] * 100, 1),
                            "Recall": round(report[cls_name]["recall"] * 100, 1),
                            "F1-Score": round(report[cls_name]["f1-score"] * 100, 1),
                            "Support": report[cls_name]["support"]
                        })
                st.table(pd.DataFrame(report_rows))

                # Extra model-specific insight
                if idx == 1: # Decision Tree Feature Importance
                    dt_model = m_res["model"]
                    importances = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": (dt_model.feature_importances_ * 100).round(2)
                    }).sort_values(by="Importance", ascending=False)
                    fig_imp = px.bar(
                        importances.head(6),
                        x="Importance",
                        y="Feature",
                        orientation='h',
                        title="Top Feature Importances (Decision Tree)",
                        color="Importance",
                        color_continuous_scale="Teal"
                    )
                    fig_imp.update_layout(height=240, margin=dict(l=20, r=20, t=35, b=20))
                    st.plotly_chart(fig_imp, use_container_width=True)


# =============================================================================
# TAB 3: LIVE STUDENT RISK PREDICTOR
# =============================================================================
with tab3:
    st.subheader("Live Student Academic Risk Inference")
    st.markdown("""
    Interactive multi-classifier risk assessment evaluating student academic factors across all benchmark models simultaneously to produce a consensus risk verdict.
    """)

    # Preset Persona Quick Fill
    st.markdown("##### ⚡ **Quick Preset Personas (1-Click Test)**")
    col_p1, col_p2, col_p3 = st.columns(3)
    
    preset_choice = None
    with col_p1:
        if st.button("🌟 Preset A: Honors Student (Low Risk)", use_container_width=True):
            preset_choice = "honors"
    with col_p2:
        if st.button("⚠️ Preset B: Borderline Student (Moderate)", use_container_width=True):
            preset_choice = "borderline"
    with col_p3:
        if st.button("🚨 Preset C: Critical Intervention (High)", use_container_width=True):
            preset_choice = "critical"

    # Default values based on presets
    if preset_choice == "honors":
        d_study, d_attend, d_score, d_assign, d_part, d_sleep = 28.0, 96.0, 92.0, 95.0, 9.0, 8.0
        d_extra, d_net, d_parent, d_tutor = "Yes", "Yes", "Master", "Yes"
    elif preset_choice == "critical":
        d_study, d_attend, d_score, d_assign, d_part, d_sleep = 4.0, 52.0, 38.0, 42.0, 2.5, 4.5
        d_extra, d_net, d_parent, d_tutor = "No", "No", "High School", "No"
    else: # default or borderline
        d_study, d_attend, d_score, d_assign, d_part, d_sleep = 12.0, 74.0, 62.0, 70.0, 5.5, 6.5
        d_extra, d_net, d_parent, d_tutor = "Yes", "Yes", "Bachelor", "No"

    st.markdown("---")
    st.markdown("##### 📝 **Input Student Academic Profile**")

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        in_study = st.slider("Study Hours / Week", 2.0, 35.0, float(d_study), 0.5)
        in_attend = st.slider("Attendance Rate (%)", 40.0, 100.0, float(d_attend), 1.0)
        in_score = st.slider("Prior Exam Score (0-100)", 30.0, 100.0, float(d_score), 1.0)

    with col_in2:
        in_assign = st.slider("Assignment Completion (%)", 35.0, 100.0, float(d_assign), 1.0)
        in_part = st.slider("Class Participation (1-10)", 1.0, 10.0, float(d_part), 0.5)
        in_sleep = st.slider("Sleep Hours / Night", 4.0, 10.0, float(d_sleep), 0.5)

    with col_in3:
        in_extra = st.selectbox("Extracurricular Activities", ["Yes", "No"], index=0 if d_extra == "Yes" else 1)
        in_net = st.selectbox("Reliable Internet Access", ["Yes", "No"], index=0 if d_net == "Yes" else 1)
        in_parent = st.selectbox("Parent Education Level", ["High School", "Bachelor", "Master", "Doctorate"], index=["High School", "Bachelor", "Master", "Doctorate"].index(d_parent))
        in_tutor = st.selectbox("Enrolled in Tutoring Support", ["Yes", "No"], index=0 if d_tutor == "Yes" else 1)

    eval_button = st.button("🔮 Assess Academic Risk for this Student", type="primary", use_container_width=True)

    # Run inference
    student_sample = {
        "Study_Hours_Per_Week": in_study,
        "Attendance_Rate": in_attend,
        "Past_Exam_Score": in_score,
        "Assignment_Completion_Rate": in_assign,
        "Class_Participation": in_part,
        "Sleep_Hours": in_sleep,
        "Extracurricular_Activities": in_extra,
        "Internet_Access": in_net,
        "Parent_Education_Level": in_parent,
        "Tutoring_Support": in_tutor
    }

    pred_output = predict_student_risk_all_models(models, preprocessor, student_sample)
    consensus = pred_output["consensus_risk"]
    dt_risk, dt_probs = classify_new_sample_dt(models["Decision Tree"], preprocessor, student_sample)

    st.markdown("---")
    st.markdown("### 🎯 **Prediction Assessment Results**")

    # Risk Banner
    if consensus == "Low Risk":
        st.markdown(f"""
        <div class="risk-banner-low">
            <h2 style="color:white; margin:0;">🟢 CONSENSUS VERDICT: LOW ACADEMIC RISK</h2>
            <p style="margin:5px 0 0 0; color:#d1fae5;">Student demonstrates strong academic momentum and consistency. Minimal intervention needed.</p>
        </div>
        """, unsafe_allow_html=True)
    elif consensus == "Moderate Risk":
        st.markdown(f"""
        <div class="risk-banner-mod">
            <h2 style="color:white; margin:0;">🟡 CONSENSUS VERDICT: MODERATE ACADEMIC RISK</h2>
            <p style="margin:5px 0 0 0; color:#fef3c7;">Student is on the academic borderline. Proactive advising and study plan support are recommended.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-banner-high">
            <h2 style="color:white; margin:0;">🔴 CONSENSUS VERDICT: CRITICAL / HIGH ACADEMIC RISK</h2>
            <p style="margin:5px 0 0 0; color:#fee2e2;">Urgent academic intervention required! High probability of course failure without corrective action.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Individual model prediction breakdown
    col_res1, col_res2 = st.columns([1, 1])

    with col_res1:
        st.markdown("#### **Multi-Model Consensus Breakdown**")
        breakdown_rows = []
        for m_name, m_info in pred_output["model_predictions"].items():
            breakdown_rows.append({
                "Model Pipeline": m_name,
                "Predicted Risk": m_info["risk_label"],
                "High Risk %": f"{m_info['probabilities']['High Risk']}%" if m_info["probabilities"] else "N/A",
                "Moderate %": f"{m_info['probabilities']['Moderate Risk']}%" if m_info["probabilities"] else "N/A",
                "Low Risk %": f"{m_info['probabilities']['Low Risk']}%" if m_info["probabilities"] else "N/A"
            })
        st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True)

        st.markdown("##### **Decision Tree Sample Inference Verification**")
        st.success(f"✓ **Decision Tree Classified Sample as:** `{dt_risk}`")

    with col_res2:
        st.markdown("#### **Ensemble Probability Distribution**")
        avg_p = pred_output["average_probabilities"]
        fig_prob = go.Figure(data=[
            go.Bar(
                x=list(avg_p.keys()),
                y=list(avg_p.values()),
                text=[f"{v}%" for v in avg_p.values()],
                textposition='auto',
                marker_color=["#10b981", "#f59e0b", "#ef4444"]
            )
        ])
        fig_prob.update_layout(
            title="Consensus Risk Probability (%)",
            yaxis_range=[0, 100],
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    # Actionable Interventions Plan
    st.markdown("---")
    st.markdown("### 📋 **Tailored Academic Intervention Plan**")
    recs = pred_output["recommendations"]
    if recs:
        for r in recs:
            urgency_color = "#ef4444" if r["urgency"] in ["High", "Critical"] else ("#f59e0b" if r["urgency"] == "Medium" else "#10b981")
            st.markdown(f"""
            <div class="intervention-box" style="border-left-color: {urgency_color};">
                <span style="font-weight:700; color:{urgency_color}; text-transform:uppercase; font-size:0.8rem;">[{r['urgency']} Urgency] {r['category']}</span>
                <p style="margin:4px 0 0 0; color:#334155; font-size:0.95rem;">{r['action']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No critical warnings triggered. Student is maintaining satisfactory progress.")


# =============================================================================
# TAB 4: WHAT-IF SENSITIVITY SIMULATOR & BATCH AUDIT
# =============================================================================
with tab4:
    st.subheader("Dynamic What-If Simulator & Batch Cohort Audit")
    st.caption("Perform sensitivity analysis on key academic levers and audit entire classes for early intervention.")

    col_sim_ctrl, col_sim_chart = st.columns([1, 2])

    with col_sim_ctrl:
        st.markdown("#### **What-If Sensitivity Simulation**")
        st.write("Examine how progressively changing an academic factor shifts the probability of academic risk.")
        
        sim_factor = st.selectbox(
            "Factor to Simulate:",
            ["Attendance_Rate", "Study_Hours_Per_Week", "Assignment_Completion_Rate", "Past_Exam_Score"]
        )

        sim_base_student = student_sample.copy()

    with col_sim_chart:
        # Generate simulation curve
        if sim_factor in ["Attendance_Rate", "Assignment_Completion_Rate"]:
            test_range = np.linspace(40, 100, 25)
        elif sim_factor == "Study_Hours_Per_Week":
            test_range = np.linspace(2, 35, 25)
        else:
            test_range = np.linspace(30, 100, 25)

        curve_low = []
        curve_mod = []
        curve_high = []

        for val in test_range:
            temp_sample = sim_base_student.copy()
            temp_sample[sim_factor] = val
            out_p = predict_student_risk_all_models(models, preprocessor, temp_sample)
            curve_low.append(out_p["average_probabilities"]["Low Risk"])
            curve_mod.append(out_p["average_probabilities"]["Moderate Risk"])
            curve_high.append(out_p["average_probabilities"]["High Risk"])

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=test_range, y=curve_low, mode='lines+markers', name='Low Risk (Safe)', line=dict(color='#10b981', width=3)))
        fig_sim.add_trace(go.Scatter(x=test_range, y=curve_mod, mode='lines+markers', name='Moderate Risk', line=dict(color='#f59e0b', width=3)))
        fig_sim.add_trace(go.Scatter(x=test_range, y=curve_high, mode='lines+markers', name='High Risk (Critical)', line=dict(color='#ef4444', width=3)))

        fig_sim.update_layout(
            title=f"Risk Transition Curve as {sim_factor.replace('_', ' ')} Changes",
            xaxis_title=sim_factor.replace('_', ' '),
            yaxis_title="Probability (%)",
            height=360,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏫 **Batch Student Cohort Audit**")
    st.write("Scan the entire active student cohort to identify all students flagged under academic risk.")

    if st.button("🚀 Run Cohort-Wide Risk Scan"):
        with st.spinner("Analyzing all cohort records with trained models..."):
            audit_records = []
            for i, row in raw_df.iterrows():
                row_dict = row.to_dict()
                audit_res = predict_student_risk_all_models(models, preprocessor, row_dict)
                audit_records.append({
                    "Student_ID": row.get("Student_ID", f"STU{1000+i}"),
                    "Attendance": row.get("Attendance_Rate", 0),
                    "Study_Hours": row.get("Study_Hours_Per_Week", 0),
                    "Past_Score": row.get("Past_Exam_Score", 0),
                    "Predicted_Risk": audit_res["consensus_risk"],
                    "High_Risk_Prob": audit_res["average_probabilities"]["High Risk"],
                    "Top_Intervention": audit_res["recommendations"][0]["action"] if audit_res["recommendations"] else "Maintain Good Standing"
                })

            audit_df = pd.DataFrame(audit_records)

            col_aud1, col_aud2 = st.columns([1, 2])
            with col_aud1:
                risk_counts = audit_df["Predicted_Risk"].value_counts().reset_index()
                risk_counts.columns = ["Risk_Level", "Count"]
                fig_audit_pie = px.pie(
                    risk_counts,
                    names="Risk_Level",
                    values="Count",
                    color="Risk_Level",
                    color_discrete_map={"Low Risk": "#10b981", "Moderate Risk": "#f59e0b", "High Risk": "#ef4444"},
                    hole=0.4,
                    title="Audit Outcome Distribution"
                )
                fig_audit_pie.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_audit_pie, use_container_width=True)

            with col_aud2:
                critical_students = audit_df[audit_df["Predicted_Risk"] == "High Risk"].sort_values(by="High_Risk_Prob", ascending=False)
                st.markdown(f"**Identified `{len(critical_students)}` Critical Risk Students Requiring Intervention:**")
                st.dataframe(critical_students.head(10), use_container_width=True)

            csv_audit = audit_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full Cohort Intervention Audit Report (CSV)",
                data=csv_audit,
                file_name="cohort_academic_risk_report.csv",
                mime="text/csv"
            )

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("EduPredict Academic Risk System | Machine Learning Platform | Preprocessing, Logistic Regression, Decision Tree, Naïve Bayes, KNN")
