# 📦 Streamlit App Starter Kit 
```
⬆️ (Replace above with your app's name)
```

Description of the app ...

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://app-starter-kit.streamlit.app/)

🎓 Smart Student Failure Risk Prediction System
A full-stack machine learning web application to evaluate and forecast student academic performance and risk. The system features a FastAPI backend serving trained Machine Learning models and a Streamlit user interface for individual interactive predictions and batch CSV file processing.
📌 Project Overview
This project provides real-time risk assessment using both Classification and Regression machine learning models.
Classification Models: Predict discrete failure/pass risk outcomes.
Regression Models: Predict continuous performance scores.
Batch Processing: Process bulk CSV datasets and output downloadable predictions in real time.
🛠️ Built With
Frontend: Streamlit
Backend Framework: FastAPI
Data Processing & ML Tools: Pandas, Joblib
Server: Uvicorn
📊 Dataset Features
Both individual and batch inputs require the following core feature variables:
Feature Column
Type
Range
Description
 
study_hours
float
0.0 - 24.0
Average daily study hours
attendance_pct
float
0.0 - 100.0
Overall attendance percentage
previous_marks
float
0.0 - 100.0
Previous academic percentage marks
assignment_pct
float
0.0 - 100.0
Average assignment score percentage

⚙️ Available Models
Classification Models
Decision Tree Classifier (decision_tree_classifier)
Random Forest Classifier (random_forest_classifier)
Logistic Regression (logistic_regression)
Regression Models
Decision Tree Regressor (decision_tree_regressor)
Random Forest Regressor (random_forest_regressor)
Linear Regression (linear_regression)
🚀 Getting Started
1. Installation & Environment Setup
Clone the repository and install the required dependencies:
# Clone repository
git clone https://github.com/your-username/student-risk-prediction.git
cd student-risk-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install fastapi uvicorn streamlit pandas joblib requests pydantic
2. Model Artifacts
Ensure the trained model .joblib files are available in the project root directory:
features.joblib
decision_tree_classifier.joblib
decision_tree_regressor.joblib
random_forest_classifier.joblib
random_forest_regressor.joblib
LogisticRegression.joblib
LinearRegression.joblib
🏃 Running the Application
Step 1: Start the FastAPI Backend Server
Run the Uvicorn server on port 8001:
uvicorn api:app --host 127.0.0.1 --port 8001 --reload
Swagger API Documentation: http://127.0.0.1:8001/docs
Step 2: Start the Streamlit Frontend App
Open a new terminal session and launch Streamlit:
streamlit run streamlit_app.py
Open local application view: http://localhost:8501
🔌 API Endpoints Summary
HTTP Method
Endpoint
Description
 
POST
/predict/classification
Predicts risk using classification models.
POST
/predict/regression
Predicts continuous score using regression models.
POST
/predict-file
Accepts a .csv file upload and returns predicted CSV results.


