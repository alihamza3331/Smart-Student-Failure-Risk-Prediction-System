import io
import os
import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Student Risk Prediction System.",
    page_icon="🎓",
    layout="wide"
)

# Base API URL pointing to internal FastAPI running on port 8001
API_URL = "http://127.0.0.1:8001"

st.title("🎓 Smart Student Failure Risk Prediction System")
st.write("Predict student performance using both Classification and Regression models.")

# --- SECTION 1: INDIVIDUAL ---
st.header("Single Student Prediction")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Input Features")
    study_hours = st.number_input(
        "Study Hours (Daily)",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )
    attendance_pct = st.number_input(
        "Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )
    previous_marks = st.number_input(
        "Previous Marks Percentage",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )
    assignment_pct = st.number_input(
        "Assignment Percentage",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

with col2:
    st.subheader("2. Model Selection")
    
    cls_model_map = {
        "Decision Tree Classifier": "decision_tree_classifier",
        "Random Forest Classifier": "random_forest_classifier",
        "Logistic Regression": "logistic_regression"
    }
    
    reg_model_map = {
        "Decision Tree Regressor": "decision_tree_regressor",
        "Random Forest Regressor": "random_forest_regressor",
        "Linear Regression": "linear_regression"
    }
    
    selected_cls_label = st.selectbox("Classification Model", list(cls_model_map.keys()))
    selected_reg_label = st.selectbox("Regression Model", list(reg_model_map.keys()))
    
    predict_btn = st.button("Predict Both Results", type="primary", use_container_width=True)

# Process Individual Prediction
if predict_btn:
    payload = {
        "study_hours": study_hours,
        "attendance_pct": attendance_pct,
        "previous_marks": previous_marks,
        "assignment_pct": assignment_pct
    }
    
    st.divider()
    st.subheader("Results")
    res_col1, res_col2 = st.columns(2)
    
    # Call Classification API
    try:
        cls_param = cls_model_map[selected_cls_label]
        cls_response = requests.post(
            f"{API_URL}/predict/classification",
            params={"model_name": cls_param},
            json=payload
        )
        
        if cls_response.status_code == 200:
            cls_data = cls_response.json()
            res_col1.success(f"**Classification Model:** {selected_cls_label}")
            res_col1.metric(label="Predicted Result Class", value=str(cls_data["prediction"]))
        else:
            res_col1.error(f"Classification API Error: {cls_response.json().get('detail', 'Unknown error')}")
            
    except requests.exceptions.RequestException as e:
        res_col1.error(f"Failed to connect to backend server: {str(e)}")

    # Call Regression API
    try:
        reg_param = reg_model_map[selected_reg_label]
        reg_response = requests.post(
            f"{API_URL}/predict/regression",
            params={"model_name": reg_param},
            json=payload
        )
        
        if reg_response.status_code == 200:
            reg_data = reg_response.json()
            res_col2.info(f"**Regression Model:** {selected_reg_label}")
            res_col2.metric(label="Predicted Score / Value", value=f"{reg_data['prediction']:.2f}")
        else:
            res_col2.error(f"Regression API Error: {reg_response.json().get('detail', 'Unknown error')}")
            
    except requests.exceptions.RequestException as e:
        res_col2.error(f"Failed to connect to backend server: {str(e)}")

st.divider()

# --- SECTION 2: CSV  ---
st.header("Batch CSV Prediction")

all_models_map = {
    "Decision Tree Classifier": "decision_tree_classifier",
    "Decision Tree Regressor": "decision_tree_regressor",
    "Random Forest Classifier": "random_forest_classifier",
    "Random Forest Regressor": "random_forest_regressor",
    "Logistic Regression": "logistic_regression",
    "Linear Regression": "linear_regression"
}

batch_model_label = st.selectbox("Select Model for Batch Prediction", list(all_models_map.keys()))
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    if st.button("Process Batch Predictions"):
        batch_model_param = all_models_map[batch_model_label]
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        
        with st.spinner("Processing CSV file..."):
            try:
                batch_response = requests.post(
                    f"{API_URL}/predict-file",
                    params={"model_name": batch_model_param},
                    files=files
                )
                
                if batch_response.status_code == 200:
                    st.success("Batch prediction completed successfully!")
                    
                    result_df = pd.read_csv(io.BytesIO(batch_response.content))
                    st.dataframe(result_df, use_container_width=True)
                    
                    st.download_button(
                        label="Download Predicted CSV",
                        data=batch_response.content,
                        file_name="student_risk_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Batch Processing Error: {batch_response.json().get('detail', 'Unknown error')}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend server: {str(e)}")
