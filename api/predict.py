from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
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
        
        response = {
            'prediction': result,
            'probability': round(probability, 2),
            'icon': icon
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()