from flask import Flask, request
from flask_cors import CORS
import json
import numpy as np
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)

@app.route('/')
def root():
    return "/"

@app.route('/api/potholes')
def get_data():
    with open("./data/potholes.json") as file:
        data = json.load(file)

    return data

@app.route('/api/upload', methods=['POST'])
def upload():
    print(request)
    return {
            'id': f"PH{str(uuid.uuid4())[:6].upper()}",
            "latitude": 40.7128,
            "longitude": -74.006,
            "depth": 8.5,
            "volume": 245.7,
            'dimensions': {
                'length': round(np.random.uniform(20, 60), 1),
                'width': round(np.random.uniform(15, 40), 1)
            },
            'severity': "severity",
            'confidence': round(np.random.uniform(0.75, 0.98), 2),
            'description': f"pothole detected by CV pipeline",
            'detectedAt': datetime.utcnow().isoformat() + 'Z',
            'imageUrl': None
        }

if __name__ == "__main__":
    app.run(debug=True)