
# Designing an Employee performance streamlit app

# importing the necessary dependencies
import streamlit as st
import pandas as pd
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------
# loading the model and encoders
#-------------------------------------------
@st.cache_resource
def load_model():
    model = pkl.load(open("m_random_classifier.pkl", "rb"))
    return model

def load_encoders():
    Over_encoder = pkl.load(open("Over_encoder.pkl", "rb"))
    Att_encoder = pkl.load(open("Att_encoder.pkl", "rb"))
    return Over_encoder, Att_encoder

# calling the functions
model = load_model()
Over_encoder, Att_encoder = load_encoders()

# -----------------------------------------------------
# MANUAL ENCODING DICTIONARIES
# -----------------------------------------------------
gender_map = {"Male": 1, "Female": 0}

marital_status_map = {'Married': 2, 'Single': 1, 'Divorced': 0}

business_travel_map = {'Travel_Rarely': 2, 'Travel_Frequently': 1, 'Non-Travel': 0}

education_background_map = {
    'Life Sciences': 5, 'Medical': 4, 'Marketing': 3,
    'Technical Degree': 2, 'Other': 1, 'Human Resources': 0
}

emp_department_map = {
    'Sales': 5, 'Development': 4, 'Research & Development': 3,
    'Human Resources': 2, 'Finance': 1, 'Data Science': 0
}

emp_jobrole_map = {
    'Sales Executive': 18, 'Developer': 17, 'Manager R&D': 16, 'Research Scientist': 15,
    'Sales Representative': 14, 'Laboratory Technician': 13, 'Senior Developer': 12,
    'Manager': 11, 'Finance Manager': 10, 'Human Resources': 9, 'Technical Lead': 8,
    'Manufacturing Director': 7, 'Healthcare Representative': 6, 'Data Scientist': 5,
    'Research Director': 4, 'Business Analyst': 3, 'Senior Manager R&D': 2,
    'Delivery Manager': 1, 'Technical Architect': 0
}

# -----------------------------------------------------
# PAGE TITLE
# -----------------------------------------------------
st.title("📊 Employee Performance Prediction System")
st.write(
    """
    This is a **RandomForest-based prediction system** that estimates employee performance rating 
    using employee demographics, work evironment and personal factors,expereience and career progression and job-related features.
    """
)

st.divider()    # displays horizontal rule separating content sections

# =====================================================
# INPUT FORM (CLEAN + SINGLE PAGE)
# =====================================================
st.header("📝 Enter Employee Details")

with st.expander("🍭 Employee Demographics"):      # hiding secondary content and displaying when needed
    Age = st.number_input("Age", 18, 60, 30)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    EducationBackground = st.selectbox("EmpEducation Level", list(education_background_map.keys()))
    EmpEducationLevel = st.selectbox("Education Level (1-5)", [1, 2, 3, 4, 5])

with st.expander("🏢 Job & Workplace"):        # hiding secondary content and displaying when needed
    EmpDepartment = st.selectbox("Department", list(emp_department_map.keys()))
    EmpJobRole = st.selectbox("Job Role", list(emp_jobrole_map.keys()))
    BusinessTravelFrequency = st.selectbox("Business Travel Frequency", list(business_travel_map.keys()))
    EmpJobLevel = st.selectbox("Job Level (1-5)", [1, 2, 3, 4, 5])
    EmpJobInvolvement = st.selectbox("Job Involvement (1-4)", [1, 2, 3, 4])
    EmpHourlyRate = st.number_input("Hourly Rate (30-100)", 30, 100)
    OverTime = st.selectbox("OverTime", ["Yes", "No"])
    Attrition = st.selectbox("Attrition", ["Yes", "No"])

with st.expander("💻 Work Environment & Personal Factors"):       # hiding secondary content and displaying when needed
    EmpJobSatisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 2)
    EmpEnvironmentSatisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 2)
    EmpRelationshipSatisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 2)
    EmpWorkLifeBalance = st.slider("Work-Life Balance (2-4)", 2, 4, 3)
    EmpLastSalaryHikePercent = st.number_input("Percent Salary Hike", 0, 50, 10)
    TrainingTimesLastYear = st.number_input("Training Times Last Year", 0, 10, 2)
    DistanceFromHome = st.number_input("Distance From Home (1-29)", 1, 29)

with st.expander("💰 Experience & Career Progression"):     # hiding secondary content and displaying when needed
    ExperienceYearsAtThisCompany = st.number_input("Years at Company", 0, 40, 5)
    ExperienceYearsInCurrentRole = st.number_input("Years in Current Role", 0, 20, 3)
    YearsWithCurrManager = st.number_input("Years with Current Manager (1-17)", 1, 17)
    NumCompaniesWorked = st.slider("Number of Companies Worked (0-9)", 0, 9, 2)
    TotalWorkExperienceInYears = st.number_input("Total Work Experience in Years (1-36)", 1, 36)
    YearsSinceLastPromotion = st.number_input("Years Since Last Promotion (1-15)", 1, 15)

st.divider()      # displays horizontal rule separating content sections

# =============================================================
# PROCESS INPUTS
# =============================================================
if st.button("🔮 Predict Performance Rating"):
    try:
        # ------------------------------
        # APPLY MANUAL ENCODING
        # ------------------------------
        gender_val = gender_map[Gender]
        marital_val = marital_status_map[MaritalStatus]
        travel_val = business_travel_map[BusinessTravelFrequency]

        education_val = education_background_map[EducationBackground]
        department_val = emp_department_map[EmpDepartment]
        jobrole_val = emp_jobrole_map[EmpJobRole]

        # ------------------------------
        # APPLY LABEL ENCODERS
        # ------------------------------
        overtime_val = Over_encoder.transform([OverTime])[0]
        attrition_val = Att_encoder.transform([Attrition])[0]

        # ------------------------------
        # BUILD INPUT DATA
        # ------------------------------
        input_df = pd.DataFrame({
        "Age": [Age],
        "Gender": [gender_val],
        "EducationBackground": [education_val],
        "MaritalStatus": [marital_val],
        "EmpDepartment": [department_val],
        "EmpJobRole": [jobrole_val],
        "BusinessTravelFrequency": [travel_val],
        "DistanceFromHome": [DistanceFromHome],
        "EmpEducationLevel": [EmpEducationLevel],  # FIXED SPELLING
        "EmpEnvironmentSatisfaction": [EmpEnvironmentSatisfaction],
        "EmpHourlyRate": [EmpHourlyRate],
        "EmpJobInvolvement": [EmpJobInvolvement],
        "EmpJobLevel": [EmpJobLevel],  # ADDED — REQUIRED FEATURE
        "EmpJobSatisfaction": [EmpJobSatisfaction],
        "NumCompaniesWorked": [NumCompaniesWorked],
        "OverTime": [overtime_val],
        "EmpLastSalaryHikePercent": [EmpLastSalaryHikePercent],  # FIXED NAME
        "EmpRelationshipSatisfaction": [EmpRelationshipSatisfaction],
        "TotalWorkExperienceInYears": [TotalWorkExperienceInYears],
        "TrainingTimesLastYear": [TrainingTimesLastYear],
        "EmpWorkLifeBalance": [EmpWorkLifeBalance],
        "ExperienceYearsAtThisCompany": [ExperienceYearsAtThisCompany],
        "ExperienceYearsInCurrentRole": [ExperienceYearsInCurrentRole],
        "YearsSinceLastPromotion": [YearsSinceLastPromotion],
        "YearsWithCurrManager": [YearsWithCurrManager],
        "Attrition": [attrition_val]
    })
        
        # ensuring the input data is the correct order as X_train_columns
        #input_df = input_df[X_train.columns]
        # ------------------------------
        # PREDICT
        # ------------------------------
        prediction = model.predict(input_df)[0]
        st.success(f"🌟 Predicted Performance Rating: **{prediction}**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.divider()       # displays horizontal rule separating content sections


# -----------------------------------------------------
# MODEL DETAILS
# -----------------------------------------------------
with st.expander("Model Information"):    # hiding secondary content and displaying when needed
    st.write("""
    **Model:** RandomForestClassifier  
    **Encoding:** Manual + LabelEncoding + Frequency Encoding  
    **Features:** 25  
    **Purpose:** Exam Project (Employee Performance Prediction)
    """)
