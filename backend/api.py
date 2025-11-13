from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Train models on import
def get_trained_models():
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
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_scaled, y)
    
    return scaler, lr_model

scaler, lr_model = get_trained_models()

@app.route('/api/predict', methods=['POST'])
def predict():
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
        prediction = lr_model.predict(features_scaled)[0]
        probability = lr_model.predict_proba(features_scaled)[0][1]
        
        if prediction == 0:
            result = "Non-Diabetic"
            icon = "✅"
        else:
            result = "Diabetic - High Risk"
            icon = "⚠️"
        
        return jsonify({
            'prediction': result,
            'probability': round(probability * 100, 2),
            'icon': icon
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run()