# Diabetes Prediction System

A complete machine learning system for predicting diabetes risk using Naïve Bayes and Logistic Regression models with a React frontend and Flask backend.

## Features

- **Data Analysis**: Complete EDA with correlation heatmaps and feature distributions
- **Model Comparison**: Naïve Bayes vs Logistic Regression performance evaluation
- **Web Interface**: React form for easy patient data input
- **Real-time Predictions**: Flask API serving ML models
- **Visual Results**: Clear risk assessment with confidence scores

## Setup Instructions

### 1. Download Dataset
Download the PIMA Indians Diabetes Dataset from Kaggle and save as `diabetes.csv` in the backend folder.

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python diabetes_model.py  # Train models first
python app.py            # Start Flask API
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm start
```

## Model Performance

The system compares two models:
- **Naïve Bayes**: Fast, probabilistic classifier
- **Logistic Regression**: Linear model with better interpretability

Performance metrics include:
- Accuracy scores
- Confusion matrices
- ROC curves and AUC
- Precision, Recall, F1-score

## API Endpoints

- `POST /predict`: Submit patient data for prediction
- `GET /health`: Check API status

## Input Features

1. Pregnancies (0-20)
2. Glucose Level (mg/dL)
3. Blood Pressure (mmHg)
4. Skin Thickness (mm)
5. Insulin (μU/mL)
6. BMI
7. Diabetes Pedigree Function
8. Age

## Output

- ✅ **Non-Diabetic**: Low risk
- ⚠️ **Diabetic - Moderate/High Risk**: Based on confidence level
- Model comparison showing both predictions
- Confidence percentage

## Technology Stack

- **Backend**: Python, Flask, scikit-learn, pandas
- **Frontend**: React, Axios
- **ML Models**: Naïve Bayes, Logistic Regression
- **Visualization**: matplotlib, seaborn