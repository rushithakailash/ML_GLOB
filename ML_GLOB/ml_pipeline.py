"""
ml_pipeline.py
Machine Learning Pipeline for Student Performance Prediction and Academic Risk Analysis.
Integrates core ML concepts:
- Data Preprocessing (Imputation, Encoding, Scaling, Train-Test Split)
- Logistic Regression Model
- Decision Tree Classification Model & Sample Inference
- Naïve Bayes Classification
- K-Nearest Neighbors (KNN) Classification Model
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

TARGET_COLUMN = "Academic_Risk"
FEATURE_COLUMNS_NUM = [
    "Study_Hours_Per_Week",
    "Attendance_Rate",
    "Past_Exam_Score",
    "Assignment_Completion_Rate",
    "Class_Participation",
    "Sleep_Hours"
]
FEATURE_COLUMNS_CAT = [
    "Extracurricular_Activities",
    "Internet_Access",
    "Parent_Education_Level",
    "Tutoring_Support"
]

RISK_CLASSES = ["Low Risk", "Moderate Risk", "High Risk"]
RISK_MAP = {"Low Risk": 0, "Moderate Risk": 1, "High Risk": 2}
INV_RISK_MAP = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

# -------------------------------------------------------------
# DATA PREPROCESSING FUNCTIONS
# -------------------------------------------------------------

def analyze_dataset_health(df):
    """
    Performs comprehensive data profiling:
    - Missing value counts and percentages
    - Column data types
    - IQR-based outlier detection for numeric columns
    """
    profile = {}
    
    # Missing values
    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": missing_counts.values,
        "Missing_Percentage (%)": missing_pct.round(2).values
    })
    profile["missing"] = missing_df[missing_df["Missing_Count"] > 0]
    profile["total_missing"] = int(missing_counts.sum())

    # Outlier detection using IQR
    outlier_counts = {}
    for col in FEATURE_COLUMNS_NUM:
        if col in df.columns:
            valid_vals = df[col].dropna()
            q1 = valid_vals.quantile(0.25)
            q3 = valid_vals.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = valid_vals[(valid_vals < lower_bound) | (valid_vals > upper_bound)]
            outlier_counts[col] = len(outliers)
    profile["outliers"] = outlier_counts

    return profile


class StudentDataPreprocessor:
    """
    End-to-end Preprocessing Pipeline:
    - Handling Missing Values (Numeric: Median/Mean, Categorical: Mode)
    - Encoding Categorical Features (One-Hot or Ordinal)
    - Feature Scaling (StandardScaler or MinMaxScaler)
    - Target Label Encoding
    """
    def __init__(self, missing_strategy="median", scaling_method="standard"):
        self.missing_strategy = missing_strategy
        self.scaling_method = scaling_method
        self.num_imputer = SimpleImputer(strategy=missing_strategy)
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()
        self.cat_encoders = {}
        self.classes = RISK_CLASSES
        self.is_fitted = False
        self.feature_names_out = []
        self.preprocessing_summary = {}

    def fit_transform(self, df_raw):
        df = df_raw.copy()
        initial_missing = int(df.isnull().sum().sum())

        # 1. Numerical Imputation
        X_num = df[FEATURE_COLUMNS_NUM].copy()
        X_num_imputed = self.num_imputer.fit_transform(X_num)
        df_num_imputed = pd.DataFrame(X_num_imputed, columns=FEATURE_COLUMNS_NUM, index=df.index)

        # 2. Categorical Imputation & Ordinal/Binary Encoding
        X_cat = df[FEATURE_COLUMNS_CAT].copy()
        X_cat_imputed = self.cat_imputer.fit_transform(X_cat)
        df_cat_imputed = pd.DataFrame(X_cat_imputed, columns=FEATURE_COLUMNS_CAT, index=df.index)

        df_cat_encoded = pd.DataFrame(index=df.index)
        for col in FEATURE_COLUMNS_CAT:
            le = LabelEncoder()
            df_cat_encoded[col] = le.fit_transform(df_cat_imputed[col])
            self.cat_encoders[col] = le

        # 3. Combine Features
        X_combined = pd.concat([df_num_imputed, df_cat_encoded], axis=1)
        self.feature_names_out = list(X_combined.columns)

        # 4. Feature Scaling Transformation
        X_scaled = self.scaler.fit_transform(X_combined)
        
        # 5. Target Encoding (Explicit deterministic order)
        y_encoded = df[TARGET_COLUMN].map(RISK_MAP).values

        self.is_fitted = True
        post_missing = 0

        self.preprocessing_summary = {
            "initial_missing_cells": initial_missing,
            "post_missing_cells": post_missing,
            "numeric_imputation_strategy": self.missing_strategy,
            "scaling_method": self.scaling_method,
            "features_processed": len(self.feature_names_out),
            "classes_encoded": self.classes
        }

        return X_scaled, y_encoded, self.feature_names_out

    def transform_single_sample(self, sample_dict):
        """
        Preprocesses a single student sample for live inference.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet!")

        df_single = pd.DataFrame([sample_dict])

        # Impute numeric
        X_num = self.num_imputer.transform(df_single[FEATURE_COLUMNS_NUM])
        df_num = pd.DataFrame(X_num, columns=FEATURE_COLUMNS_NUM)

        # Impute & encode categorical
        X_cat = self.cat_imputer.transform(df_single[FEATURE_COLUMNS_CAT])
        df_cat = pd.DataFrame(X_cat, columns=FEATURE_COLUMNS_CAT)

        df_cat_encoded = pd.DataFrame()
        for col in FEATURE_COLUMNS_CAT:
            le = self.cat_encoders[col]
            # Handle unknown categories safely
            val = df_cat[col].iloc[0]
            if val in le.classes_:
                encoded_val = le.transform([val])[0]
            else:
                encoded_val = 0
            df_cat_encoded[col] = [encoded_val]

        X_combined = pd.concat([df_num, df_cat_encoded], axis=1)
        X_scaled = self.scaler.transform(X_combined)
        return X_scaled


def prepare_train_test_data(df, test_size=0.2, random_state=42, missing_strategy="median", scaling_method="standard"):
    """
    Full pipeline to preprocess and split data into train and test sets.
    """
    preprocessor = StudentDataPreprocessor(missing_strategy=missing_strategy, scaling_method=scaling_method)
    X, y, feature_names = preprocessor.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor, feature_names


# -------------------------------------------------------------
# LOGISTIC REGRESSION MODEL
# -------------------------------------------------------------

def train_logistic_regression(X_train, y_train, C=1.0, max_iter=300):
    """
    Multinomial Logistic Regression model for 3-tier risk classification.
    """
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# -------------------------------------------------------------
# DECISION TREE CLASSIFICATION & SAMPLE INFERENCE
# -------------------------------------------------------------

def train_decision_tree(X_train, y_train, max_depth=5, criterion="gini"):
    """
    Decision Tree Classification model.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def classify_new_sample_dt(dt_model, preprocessor, sample_dict):
    """
    Classify a new student sample using the Decision Tree model.
    """
    X_sample = preprocessor.transform_single_sample(sample_dict)
    pred_class_idx = dt_model.predict(X_sample)[0]
    pred_prob = dt_model.predict_proba(X_sample)[0]
    predicted_risk = INV_RISK_MAP[pred_class_idx]
    return predicted_risk, pred_prob


# -------------------------------------------------------------
# NAÏVE BAYES CLASSIFICATION
# -------------------------------------------------------------

def train_naive_bayes(X_train, y_train):
    """
    Naïve Bayes Classification (GaussianNB).
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


# -------------------------------------------------------------
# KNN CLASSIFICATION
# -------------------------------------------------------------

def train_knn(X_train, y_train, n_neighbors=5, metric="minkowski"):
    """
    K-Nearest Neighbors Classification model.
    """
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric
    )
    model.fit(X_train, y_train)
    return model


# -------------------------------------------------------------
# UNIFIED EVALUATION & PREDICTION ENGINE
# -------------------------------------------------------------

def train_all_models(X_train, y_train, dt_depth=5, dt_criterion="gini", knn_k=5, logreg_c=1.0):
    """
    Trains all 4 benchmark models.
    """
    models = {
        "Logistic Regression": train_logistic_regression(X_train, y_train, C=logreg_c),
        "Decision Tree": train_decision_tree(X_train, y_train, max_depth=dt_depth, criterion=dt_criterion),
        "Naïve Bayes": train_naive_bayes(X_train, y_train),
        "KNN Classifier": train_knn(X_train, y_train, n_neighbors=knn_k)
    }
    return models


def evaluate_all_models(models, X_test, y_test, class_names=RISK_CLASSES):
    """
    Evaluates all 4 models and returns metrics table and confusion matrices.
    """
    results = {}
    metrics_list = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "classification_report": report,
            "y_pred": y_pred
        }

        metrics_list.append({
            "Model": name,
            "Accuracy": round(acc * 100, 2),
            "Precision (%)": round(prec * 100, 2),
            "Recall (%)": round(rec * 100, 2),
            "F1-Score (%)": round(f1 * 100, 2)
        })

    summary_df = pd.DataFrame(metrics_list)
    return results, summary_df


def predict_student_risk_all_models(models, preprocessor, sample_dict):
    """
    Predicts academic risk for a new student across all 4 models simultaneously.
    Computes consensus risk level, class probability distributions, and intervention recommendations.
    """
    X_sample = preprocessor.transform_single_sample(sample_dict)
    classes = RISK_CLASSES
    
    predictions = {}
    vote_counts = {c: 0 for c in classes}
    avg_probs = np.zeros(len(classes))

    for name, model in models.items():
        pred_idx = model.predict(X_sample)[0]
        prob = model.predict_proba(X_sample)[0] if hasattr(model, "predict_proba") else None
        risk_label = INV_RISK_MAP[pred_idx]
        
        predictions[name] = {
            "risk_label": str(risk_label),
            "probabilities": {cls: round(float(prob[i] * 100), 1) for i, cls in enumerate(classes)} if prob is not None else None
        }
        vote_counts[risk_label] += 1
        if prob is not None:
            avg_probs += prob

    avg_probs = (avg_probs / len(models)) * 100
    consensus_risk = max(vote_counts, key=vote_counts.get)

    # Generate tailored academic recommendations
    recommendations = generate_academic_interventions(sample_dict, consensus_risk)

    return {
        "consensus_risk": str(consensus_risk),
        "vote_counts": vote_counts,
        "average_probabilities": {cls: round(float(avg_probs[i]), 1) for i, cls in enumerate(classes)},
        "model_predictions": predictions,
        "recommendations": recommendations
    }


def generate_academic_interventions(student, risk_level):
    """
    Generates intelligent, actionable educational interventions based on student risk and feature indicators.
    """
    advice = []

    # Attendance check
    attend = student.get("Attendance_Rate", 100)
    if attend < 75:
        advice.append({
            "category": "Attendance Warning",
            "urgency": "High",
            "action": f"Current attendance is {attend}%. Enroll student in automated attendance monitoring and schedule a mandatory meeting with their academic advisor."
        })
    elif attend < 85:
        advice.append({
            "category": "Attendance Caution",
            "urgency": "Medium",
            "action": f"Attendance is {attend}%. Send weekly reminder notifications to prevent further decline."
        })

    # Study Hours check
    study = student.get("Study_Hours_Per_Week", 15)
    if study < 10:
        advice.append({
            "category": "Study Habits",
            "urgency": "High",
            "action": f"Weekly study time ({study} hrs/week) is significantly below the 15-20 hr target. Provide time management counseling and assign structured study schedules."
        })

    # Past Exam Score
    score = student.get("Past_Exam_Score", 70)
    if score < 50:
        advice.append({
            "category": "Remedial Tutoring",
            "urgency": "Critical",
            "action": f"Prior exam score ({score}/100) indicates foundational knowledge gaps. Recommend enrollment in 1-on-1 peer tutoring immediately."
        })

    # Assignment Completion
    assign = student.get("Assignment_Completion_Rate", 80)
    if assign < 70:
        advice.append({
            "category": "Coursework Follow-up",
            "urgency": "Medium",
            "action": f"Assignment completion is at {assign}%. Set up milestone checkpoint submissions to avoid last-minute backlog."
        })

    # Sleep hours
    sleep = student.get("Sleep_Hours", 7)
    if sleep < 6:
        advice.append({
            "category": "Wellness Alert",
            "urgency": "Low",
            "action": f"Student averages {sleep} hrs of sleep per night. Encourage participation in campus student wellness and stress management workshops."
        })

    # General encouragement if Low Risk
    if risk_level == "Low Risk" and len(advice) == 0:
        advice.append({
            "category": "Honors & Enrichment",
            "urgency": "Low",
            "action": "Student demonstrates robust academic standing. Recommend consideration for teaching assistantships, honors tracks, or undergraduate research."
        })

    return advice
