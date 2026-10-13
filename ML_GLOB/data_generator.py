"""
data_generator.py
Generates a realistic student academic performance dataset with genuine educational correlations,
including slight missing values to demonstrate Data Preprocessing and imputation.
"""

import numpy as np
import pandas as pd
import os

def generate_student_dataset(n_samples=1200, random_state=42, save_path="student_performance_data.csv"):
    np.random.seed(random_state)

    # 1. Generate core academic & lifestyle features
    student_ids = [f"STU{1000 + i}" for i in range(n_samples)]
    
    # Study hours per week (2 to 35 hours)
    study_hours = np.clip(np.random.normal(loc=15, scale=6, size=n_samples), 2, 35).round(1)
    
    # Attendance rate percentage (40% to 100%)
    attendance_rate = np.clip(np.random.normal(loc=78, scale=14, size=n_samples), 40, 100).round(1)
    
    # Past exam score (30 to 100)
    past_exam_score = np.clip(np.random.normal(loc=68, scale=16, size=n_samples), 30, 100).round(1)
    
    # Assignment completion rate percentage (35% to 100%)
    assignment_rate = np.clip(np.random.normal(loc=76, scale=15, size=n_samples), 35, 100).round(1)
    
    # Class participation score (1 to 10)
    participation_score = np.clip(np.random.normal(loc=6.5, scale=2.0, size=n_samples), 1, 10).round(1)
    
    # Sleep hours per night (4 to 10 hours)
    sleep_hours = np.clip(np.random.normal(loc=6.8, scale=1.2, size=n_samples), 4, 10).round(1)
    
    # Categorical features
    extracurricular = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    internet_access = np.random.choice(["Yes", "No"], size=n_samples, p=[0.88, 0.12])
    parent_education = np.random.choice(
        ["High School", "Bachelor", "Master", "Doctorate"], 
        size=n_samples, 
        p=[0.35, 0.45, 0.15, 0.05]
    )
    tutoring_support = np.random.choice(["Yes", "No"], size=n_samples, p=[0.32, 0.68])

    # 2. Derive realistic ground truth Academic Risk score
    # Normalize features between 0 and 1 for weighted calculation
    norm_study = (study_hours - 2) / 33
    norm_attend = (attendance_rate - 40) / 60
    norm_past = (past_exam_score - 30) / 70
    norm_assign = (assignment_rate - 35) / 65
    norm_part = (participation_score - 1) / 9
    norm_sleep = 1.0 - (np.abs(sleep_hours - 7.5) / 3.5) # optimum around 7.5 hrs
    extra_boost = np.where(extracurricular == "Yes", 0.05, 0.0)
    internet_boost = np.where(internet_access == "Yes", 0.05, 0.0)
    tutoring_boost = np.where(tutoring_support == "Yes", 0.08, 0.0)

    # Composite academic capability score (0 to 100)
    capability_score = (
        0.28 * norm_past +
        0.26 * norm_attend +
        0.18 * norm_study +
        0.15 * norm_assign +
        0.08 * norm_part +
        0.05 * norm_sleep +
        extra_boost +
        internet_boost +
        tutoring_boost +
        np.random.normal(0, 0.04, size=n_samples) # slight noise
    ) * 100

    capability_score = np.clip(capability_score, 10, 100)

    # 3. Categorize into 3 Academic Risk Levels
    # High Risk: capability_score < 48 (Needs urgent intervention)
    # Moderate Risk: 48 <= capability_score < 72 (Needs academic monitoring)
    # Low Risk: capability_score >= 72 (Good performance / safe)
    risk_levels = []
    for score in capability_score:
        if score < 48:
            risk_levels.append("High Risk")
        elif score < 72:
            risk_levels.append("Moderate Risk")
        else:
            risk_levels.append("Low Risk")

    # 4. Construct DataFrame
    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Study_Hours_Per_Week": study_hours,
        "Attendance_Rate": attendance_rate,
        "Past_Exam_Score": past_exam_score,
        "Assignment_Completion_Rate": assignment_rate,
        "Class_Participation": participation_score,
        "Sleep_Hours": sleep_hours,
        "Extracurricular_Activities": extracurricular,
        "Internet_Access": internet_access,
        "Parent_Education_Level": parent_education,
        "Tutoring_Support": tutoring_support,
        "Academic_Risk": risk_levels
    })

    # 5. Inject realistic missing values (~3%) to demonstrate Data Preprocessing and Imputation
    missing_indices_exam = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    missing_indices_sleep = np.random.choice(n_samples, size=int(n_samples * 0.025), replace=False)
    missing_indices_parent = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)

    df.loc[missing_indices_exam, "Past_Exam_Score"] = np.nan
    df.loc[missing_indices_sleep, "Sleep_Hours"] = np.nan
    df.loc[missing_indices_parent, "Parent_Education_Level"] = np.nan

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Dataset successfully created with {n_samples} records at: {save_path}")

    return df

if __name__ == "__main__":
    generate_student_dataset()
