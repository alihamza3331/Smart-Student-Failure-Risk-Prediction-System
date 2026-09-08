# 🎓 Smart Student Failure Risk Prediction System

## Live App

## 🌐 Live Demo

🚀 **Try the application online:**

[👉 Smart Student Failure Risk Prediction System](https://smart-student-failure-risk-prediction.onrender.com)

or 

my website link: https://smart-student-failure-risk-prediction.onrender.com

The App features:

- **FastAPI** backend serving trained Machine Learning models
- **Streamlit** user interface for individual interactive predictions
- **Batch CSV processing** for bulk predictions for teachers

---

## 📌 Project Overview

This project provides real-time risk assessment using both **Classification** and **Regression** machine learning models.

### Classification Models

Predict discrete **failure/pass risk outcomes**.

### Regression Models

Predict continuous **academic performance scores**.

### Batch Processing

Process bulk CSV datasets and generate **downloadable prediction results** in real time.

---

## 🛠️ Built With

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Backend Framework | FastAPI |
| Data Processing & ML Tools | Pandas, Joblib |
| Server | Uvicorn |

---

## 📊 Dataset Features

Both individual and batch inputs require the following core feature variables:

| Feature Column | Type | Range | Description |
|---|---|---|---|
| `study_hours` | float | 0.0 - 24.0 | Average daily study hours |
| `attendance_pct` | float | 0.0 - 100.0 | Overall attendance percentage |
| `previous_marks` | float | 0.0 - 100.0 | Previous academic percentage marks |
| `assignment_pct` | float | 0.0 - 100.0 | Average assignment score percentage |

---

## ⚙️ Available Models

### Classification Models

- **Decision Tree Classifier** — `decision_tree_classifier`
- **Random Forest Classifier** — `random_forest_classifier`
- **Logistic Regression** — `logistic_regression`

### Regression Models

- **Decision Tree Regressor** — `decision_tree_regressor`
- **Random Forest Regressor** — `random_forest_regressor`
- **Linear Regression** — `linear_regression`

---

## 🚀 Getting Started

### 1. Installation & Environment Setup

Clone the repository and install the required dependencies:

```bash
# Clone repository
git clone https://github.com/your-username/student-risk-prediction.git

# Navigate to project directory
cd student-risk-prediction

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# On Windows use:
# venv\Scripts\activate

# Install required packages
pip install fastapi uvicorn streamlit pandas joblib requests pydantic
