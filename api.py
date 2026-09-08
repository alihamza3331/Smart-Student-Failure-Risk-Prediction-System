import io
from enum import Enum
import joblib
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI(
    title="Smart Student Failure Risk Prediction System",
    description="API for predicting Risk of Student Failure",
    version="1.0"
)

# Load features and models
features = joblib.load("features.joblib")

models = {
    "decision_tree_classifier": joblib.load("decision_tree_classifier.joblib"),
    "decision_tree_regressor": joblib.load("decision_tree_regressor.joblib"),
    "random_forest_classifier": joblib.load("random_forest_classifier.joblib"),
    "random_forest_regressor": joblib.load("random_forest_regressor.joblib"),
    "logistic_regression": joblib.load("LogisticRegression.joblib"),
    "linear_regression": joblib.load("LinearRegression.joblib"),
}

# Unified Enum for batch prediction model selection
class AllModelNames(str, Enum):
    dt_classifier = "decision_tree_classifier"
    dt_regressor = "decision_tree_regressor"
    rf_classifier = "random_forest_classifier"
    rf_regressor = "random_forest_regressor"
    logistic = "logistic_regression"
    linear = "linear_regression"

# Enums for individual single-item prediction endpoints
class ClassificationModel(str, Enum):
    dt_classifier = "decision_tree_classifier"
    rf_classifier = "random_forest_classifier"
    logistic = "logistic_regression"

class RegressionModel(str, Enum):
    dt_regressor = "decision_tree_regressor"
    rf_regressor = "random_forest_regressor"
    linear = "linear_regression"


# Pydantic schema
class RiskFeatures(BaseModel):
    study_hours: float = Field(ge=0, le=24, description="Daily study hours of the student")
    attendance_pct: float = Field(ge=0, le=100, description="Attendance percentage of the student")
    previous_marks: float = Field(ge=0, le=100, description="Marks percentage obtained in previous exams")
    assignment_pct: float = Field(ge=0, le=100, description="Assignment score percentage")


def _run_prediction(risk: RiskFeatures, model_key: str):
    try:
        input_data = pd.DataFrame([
            {
                "study_hours": risk.study_hours,
                "attendance_pct": risk.attendance_pct,
                "previous_marks": risk.previous_marks,
                "assignment_pct": risk.assignment_pct
            }
        ])

        selected_model = models[model_key]
        raw_prediction = selected_model.predict(input_data)[0]
        return raw_prediction

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# Endpoint 1: Classification Models
@app.post("/predict/classification")
def predict_classification(
    risk: RiskFeatures,
    model_name: ClassificationModel = Query(..., description="Select classification model")
):
    prediction = _run_prediction(risk, model_name.value)
    return {
        "status": "success",
        "task": "classification",
        "selected_model": model_name.value,
        "prediction": str(prediction)
    }


# Endpoint 2: Regression Models
@app.post("/predict/regression")
def predict_regression(
    risk: RiskFeatures,
    model_name: RegressionModel = Query(..., description="Select regression model")
):
    prediction = _run_prediction(risk, model_name.value)
    return {
        "status": "success",
        "task": "regression",
        "selected_model": model_name.value,
        "prediction": float(prediction)
    }


# Endpoint 3: Batch CSV Prediction Endpoint
@app.post("/predict-file")
async def predict_file(
    file: UploadFile = File(...),
    model_name: AllModelNames = Query(..., description="Select model for batch prediction")
):
    # Validate file extension
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="please upload a CSV File only"
        )

    # Read binary contents of the uploaded file into memory
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Validate presence of required dataset columns
    required_columns = [
        "study_hours", "attendance_pct", "previous_marks", "assignment_pct"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"These columns are missing from your file: {missing_columns}"
        )

    # Check that file contains rows to process
    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file has no data rows"
        )

    try:
        selected_model = models[model_name.value]

        # Run batch model inferences on specified feature columns
        predictions = selected_model.predict(df[required_columns])

        # Append predictions and selected model information to DataFrame
        df["predicted_result"] = predictions
        df["selected_model"] = model_name.value

        # Convert output DataFrame back to CSV string format
        output = df.to_csv(index=False)

        # Stream CSV back as a downloadable file response
        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=student_risk_predictions.csv"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
