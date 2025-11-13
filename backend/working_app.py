from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables for models
scaler = None
nb_model = None
lr_model = None

def train_models():
    global scaler, nb_model, lr_model
    
    try:
        # Create sample data if diabetes.csv doesn't exist
        try:
            df = pd.read_csv('diabetes.csv')
        except:
            print("Creating sample dataset...")
            np.random.seed(42)
            n_samples = 768
            
            data = {
                'Pregnancies': np.random.randint(0, 17, n_samples),
                'Glucose': np.random.normal(120, 30, n_samples).clip(0, 200),
                'BloodPressure': np.random.normal(70, 15, n_samples).clip(0, 122),
                'SkinThickness': np.random.normal(20, 10, n_samples).clip(0, 99),
                'Insulin': np.random.normal(80, 100, n_samples).clip(0, 846),
                'BMI': np.random.normal(32, 7, n_samples).clip(0, 67),
                'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
                'Age': np.random.randint(21, 81, n_samples)
            }
            
            df = pd.DataFrame(data)
            outcome_prob = (
                (df['Glucose'] > 140) * 0.3 +
                (df['BMI'] > 30) * 0.2 +
                (df['Age'] > 50) * 0.2 +
                np.random.uniform(0, 0.1, n_samples)
            )
            df['Outcome'] = (outcome_prob > 0.5).astype(int)
            df.to_csv('diabetes.csv', index=False)
        
        # Clean data
        zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_columns:
            if col in df.columns:
                df[col] = df[col].replace(0, df[col].mean())
        
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        nb_model = GaussianNB()
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        
        nb_model.fit(X_train_scaled, y_train)
        lr_model.fit(X_train_scaled, y_train)
        
        print("Models trained successfully!")
        return True
        
    except Exception as e:
        print(f"Error training models: {e}")
        return False

@app.route('/predict', methods=['POST'])
def predict():
    global scaler, nb_model, lr_model
    
    if scaler is None or lr_model is None:
        if not train_models():
            return jsonify({'error': 'Failed to train models'}), 500
    
    try:
        data = request.json
        features = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['bloodPressure']),
            float(data['skinThickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetesPedigreeFunction']),
            float(data['age'])
        ]
        
        features_scaled = scaler.transform([features])
        
        lr_prediction = lr_model.predict(features_scaled)[0]
        lr_probability = lr_model.predict_proba(features_scaled)[0][1]
        
        nb_prediction = nb_model.predict(features_scaled)[0]
        nb_probability = nb_model.predict_proba(features_scaled)[0][1]
        
        if lr_prediction == 0:
            result = "Non-Diabetic"
            risk_level = "low"
            icon = "✅"
        else:
            if lr_probability > 0.7:
                result = "Diabetic - High Risk"
                risk_level = "high"
            else:
                result = "Diabetic - Moderate Risk"
                risk_level = "moderate"
            icon = "⚠️"
        
        return jsonify({
            'prediction': result,
            'probability': round(lr_probability * 100, 2),
            'risk_level': risk_level,
            'icon': icon,
            'models': {
                'logistic_regression': {
                    'prediction': int(lr_prediction),
                    'probability': round(lr_probability * 100, 2)
                },
                'naive_bayes': {
                    'prediction': int(nb_prediction),
                    'probability': round(nb_probability * 100, 2)
                }
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': scaler is not None and lr_model is not None
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Diabetes Prediction API is running!'})

if __name__ == '__main__':
    print("Starting Diabetes Prediction API...")
    train_models()
    app.run(debug=True, port=5000, host='0.0.0.0')