from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    glucose = float(data['glucose'])
    bmi = float(data['bmi'])
    age = float(data['age'])
    
    risk_score = (glucose - 100) * 0.01 + (bmi - 25) * 0.02 + (age - 30) * 0.005
    probability = min(max(risk_score * 10, 0), 100)
    
    if glucose > 140 or bmi > 30 or age > 50:
        result = "Diabetic - High Risk"
        icon = "⚠️"
    else:
        result = "Non-Diabetic"
        icon = "✅"
    
    return jsonify({
        'prediction': result,
        'probability': round(probability, 2),
        'icon': icon
    })

if __name__ == '__main__':
    app.run()