import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from scipy import sparse

# -----------------------------
# DEFINE MODEL CLASS
# -----------------------------
class DepressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

# -----------------------------
# LOAD MODEL + PREPROCESSOR
# -----------------------------
preprocessor = joblib.load("preprocessor.pkl")
INPUT_DIM = 39
model = DepressionModel(input_dim=INPUT_DIM)
state_dict = torch.load("depression_model.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# -----------------------------
# TITLE
# -----------------------------
st.title("🧠 Mental Health Depression Prediction App")
st.write("Enter user details to predict depression risk")

# -----------------------------
# RAW INPUT FIELDS
# -----------------------------

age = st.number_input("Age", min_value=10, max_value=100, value=25)

gender = st.selectbox("Gender", ["Male", "Female"])

city = st.selectbox("City", ["Kalyan", "Patna", "Vasai-Virar", "Kolkata", "Ahmedabad", "Other"])

profession = st.selectbox(
    "Profession",
    ["Student", "Teacher", "Unknown", "Content Writer", "Architect", "Other"]
)

sleep_duration = st.selectbox(
    "Sleep Duration",
    ["Less than 5 hours", "7-8 hours", "More than 8 hours", "5-6 hours", "Other"]
)

dietary_habits = st.selectbox(
    "Dietary Habits",
    ["Moderate", "Unhealthy", "Healthy", "Other"]
)

degree = st.selectbox(
    "Degree",
    ["Class 12", "B.Ed", "B.Arch", "B.Com", "B.Pharm", "Other"]
)

work_pressure = st.slider("Work Pressure", 0, 10, 5)
financial_stress = st.slider("Financial Stress", 0, 10, 5)
study_satisfaction = st.slider("Study Satisfaction", 0, 10, 5)
academic_pressure = st.slider("Academic Pressure", 0, 10, 5)
cgpa = st.slider("CGPA", 0.0, 4.0, 3.0)
job_satisfaction = st.slider("Job Satisfaction", 0, 10, 5)
work_study_hours = st.slider("Work/Study Hours", 0, 24, 8)
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts ?", ["No", "Yes"])
family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

# -----------------------------
# ENCODING FUNCTION (SAME AS TRAINING)
# -----------------------------
def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])

    # Apply preprocessor (ColumnTransformer)
    df_processed = preprocessor.transform(df)

    # Convert sparse to dense if needed
    if sparse.issparse(df_processed):
        df_processed = df_processed.toarray()

    return df_processed

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Depression Risk"):

    input_data = {
        "Age": age,
        "Gender": gender,
        "City": city,
        "Profession": profession,
        "Sleep Duration": sleep_duration,
        "Dietary Habits": dietary_habits,
        "Degree": degree,
        "Work Pressure": work_pressure,
        "Financial Stress": financial_stress,
        "Study Satisfaction": study_satisfaction,
        "Academic Pressure": academic_pressure,
        "CGPA": cgpa,
        "Job Satisfaction": job_satisfaction,
        "Work/Study Hours": work_study_hours,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
        "Family History of Mental Illness": family_history
    }

    processed = preprocess_input(input_data)

    tensor_input = torch.tensor(processed, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor_input)
        prob = torch.sigmoid(output).item()

    threshold = 0.35  # tuned for recall

    prediction = 1 if prob > threshold else 0

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠ High Risk of Depression")
    else:
        st.success(f"✅ Low Risk of Depression")

    st.write(f"Probability Score: {prob:.4f}")
    st.write(f"Threshold Used: {threshold}")
