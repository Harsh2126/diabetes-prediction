import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    pregnancies: '',
    glucose: '',
    bloodPressure: '',
    skinThickness: '',
    insulin: '',
    bmi: '',
    diabetesPedigreeFunction: '',
    age: ''
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await axios.post('http://localhost:5000/predict', formData);
      setPrediction(response.data);
    } catch (error) {
      console.error('Error:', error);
      alert('Error making prediction. Please check if the backend is running.');
    }
    
    setLoading(false);
  };

  const resetForm = () => {
    setFormData({
      pregnancies: '',
      glucose: '',
      bloodPressure: '',
      skinThickness: '',
      insulin: '',
      bmi: '',
      diabetesPedigreeFunction: '',
      age: ''
    });
    setPrediction(null);
  };

  return (
    <div className="App">
      <div className="container">
        <h1>🩺 Diabetes Prediction System</h1>
        <p>Enter patient medical data to predict diabetes risk</p>
        
        <form onSubmit={handleSubmit} className="prediction-form">
          <div className="form-grid">
            <div className="form-group">
              <label>Pregnancies:</label>
              <input
                type="number"
                name="pregnancies"
                value={formData.pregnancies}
                onChange={handleChange}
                min="0"
                max="20"
                required
              />
            </div>
            
            <div className="form-group">
              <label>Glucose Level (mg/dL):</label>
              <input
                type="number"
                name="glucose"
                value={formData.glucose}
                onChange={handleChange}
                min="0"
                max="300"
                required
              />
            </div>
            
            <div className="form-group">
              <label>Blood Pressure (mmHg):</label>
              <input
                type="number"
                name="bloodPressure"
                value={formData.bloodPressure}
                onChange={handleChange}
                min="0"
                max="200"
                required
              />
            </div>
            
            <div className="form-group">
              <label>Skin Thickness (mm):</label>
              <input
                type="number"
                name="skinThickness"
                value={formData.skinThickness}
                onChange={handleChange}
                min="0"
                max="100"
                required
              />
            </div>
            
            <div className="form-group">
              <label>Insulin (μU/mL):</label>
              <input
                type="number"
                name="insulin"
                value={formData.insulin}
                onChange={handleChange}
                min="0"
                max="1000"
                required
              />
            </div>
            
            <div className="form-group">
              <label>BMI:</label>
              <input
                type="number"
                step="0.1"
                name="bmi"
                value={formData.bmi}
                onChange={handleChange}
                min="0"
                max="70"
                required
              />
            </div>
            
            <div className="form-group">
              <label>Diabetes Pedigree Function:</label>
              <input
                type="number"
                step="0.001"
                name="diabetesPedigreeFunction"
                value={formData.diabetesPedigreeFunction}
                onChange={handleChange}
                min="0"
                max="3"
                required
              />
            </div>
            
            <div className="form-group">
              <label>Age:</label>
              <input
                type="number"
                name="age"
                value={formData.age}
                onChange={handleChange}
                min="1"
                max="120"
                required
              />
            </div>
          </div>
          
          <div className="button-group">
            <button type="submit" disabled={loading} className="predict-btn">
              {loading ? 'Predicting...' : 'Predict Diabetes Risk'}
            </button>
            <button type="button" onClick={resetForm} className="reset-btn">
              Reset Form
            </button>
          </div>
        </form>

        {prediction && (
          <div className={`prediction-result ${prediction.risk_level}`}>
            <div className="result-header">
              <span className="result-icon">{prediction.icon}</span>
              <h2>{prediction.prediction}</h2>
            </div>
            
            <div className="result-details">
              <p><strong>Confidence:</strong> {prediction.probability}%</p>
              
              <div className="model-comparison">
                <h3>Model Predictions:</h3>
                <div className="models">
                  <div className="model">
                    <strong>Logistic Regression:</strong> {prediction.models.logistic_regression.probability}%
                  </div>
                  <div className="model">
                    <strong>Naïve Bayes:</strong> {prediction.models.naive_bayes.probability}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;