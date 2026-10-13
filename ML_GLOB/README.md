# 🎓 Student Performance Prediction and Academic Risk Analysis Platform

A unified, modern Machine Learning web application designed to predict student academic outcomes, identify at-risk learners early, and deliver actionable educational interventions.

Built with **Python**, **Scikit-Learn**, **Plotly**, and **Streamlit**.

---

## 📌 Machine Learning Architecture & Modules

The platform implements five core Machine Learning pillars:

| Module | Technical Concept | Implementation Description |
| :--- | :--- | :--- |
| **Data Preprocessing** | **Data Cleaning & Preprocessing Pipeline** | Comprehensive data cleaning, missing value imputation (median/mean & mode), IQR outlier profiling, categorical label encoding, and feature scaling (`StandardScaler` / `MinMaxScaler`). |
| **Logistic Regression** | **Linear Probabilistic Classification** | Multinomial Logistic Regression classifier with L2 regularization (`C` hyperparameter tuning), odds ratios, and probability estimations across 3 risk tiers. |
| **Decision Tree** | **Rule-Based Tree Classification** | Decision Tree Classifier (`gini` / `entropy`, configurable max depth), feature importance extraction, and **interactive live classification of new student samples**. |
| **Naïve Bayes** | **Bayesian Conditional Probability** | Gaussian Naïve Bayes model utilizing prior class probabilities and conditional likelihoods for fast, probabilistic risk screening. |
| **K-Nearest Neighbors** | **Instance-Based Classification** | K-Nearest Neighbors classifier with Euclidean distance metrics and configurable neighbor count ($k$). |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure Python 3.10+ is installed with the required dependencies:
```bash
pip install streamlit scikit-learn pandas numpy plotly matplotlib seaborn
```

### 2. Run the Application
Launch the web application with a single command:
```bash
streamlit run app.py
```
*Or via Python module:*
```bash
python -m streamlit run app.py
```

The application will start automatically in your default browser at:
👉 **`http://localhost:8501`**

---

## 🏛️ Project Structure

```
d:/ML_GLOB/
│
├── app.py                         # Main Streamlit web application & UI dashboard
├── ml_pipeline.py                 # Core ML engine (Preprocessing, Classifiers, Evaluation, Inference)
├── data_generator.py              # Realistic academic dataset generator
├── student_performance_data.csv   # Pre-generated 1,200 student dataset
└── README.md                      # Complete project documentation
```

---

## 🖥️ Application Features & Walkthrough

### 📊 Tab 1: Dataset & Preprocessing Pipeline
- **Raw Data Explorer**: View raw student records with instant CSV export.
- **Missing Value Handling**: Visual comparison of detected missing values and post-imputation integrity.
- **Feature Scaling**: Distribution overlay comparing raw vs. normalized/scaled features.
- **Outlier Detection**: IQR-based outlier detection across academic metrics.
- **Correlation Heatmap**: Pearson correlation matrix illustrating relationships between attendance, study hours, past scores, and risk.

### 🤖 Tab 2: Multi-Model Benchmark Leaderboard
- **Leaderboard Table**: Compares **Accuracy**, **Precision**, **Recall**, and **F1-Score** side-by-side.
- **Comparative Bar Charts**: Interactive visual metric comparison across all 4 algorithms.
- **Model Deep Dives**:
  - Interactive **Confusion Matrices** for each classifier.
  - Full **Classification Reports** with class-level support.
  - **Feature Importance** rankings from the Decision Tree model.

### 🎯 Tab 3: Live Student Risk Predictor
- **1-Click Preset Personas**:
  - 🌟 **Honors Student**: High attendance, regular study hours $\rightarrow$ Low Risk.
  - ⚠️ **Borderline Student**: Moderate attendance, average grades $\rightarrow$ Moderate Risk.
  - 🚨 **Critical Intervention**: Low attendance, poor past exam score $\rightarrow$ High Risk.
- **Custom Input Sliders**: Adjust Study Hours, Attendance %, Past Exam Score, Assignment Completion %, Sleep, and Support.
- **Multi-Model Consensus Card**: Glowing color-coded risk badge with model-by-model voting.
- **Tailored Academic Intervention Plan**: Provides prioritized recommendations (e.g., automated attendance warnings, 1-on-1 peer tutoring, study schedules).

### 📈 Tab 4: What-If Sensitivity Simulator & Cohort Audit
- **Sensitivity Curve**: Dynamically slide any factor (e.g., Attendance Rate from 40% to 100%) and watch the probability curves shift between High, Moderate, and Low risk in real time.
- **Batch Cohort Audit**: Run inference across the entire student roster, view cohort risk breakdown donut chart, filter high-risk students, and download an Academic Intervention Report CSV.

---

## 🎯 Target Risk Definitions

- 🟢 **Low Risk (Safe / Good Performance)**: High attendance ($>85\%$), consistent study habits ($>15$ hrs/week), strong prior performance.
- 🟡 **Moderate Risk (Academic Watchlist)**: Borderline metrics ($70-84\%$ attendance, $10-15$ study hours). Needs proactive monitoring.
- 🔴 **High Risk (Critical Intervention Needed)**: Low attendance ($<70\%$), deficient prior scores ($<50$), low coursework completion. High danger of course failure or dropping out.
