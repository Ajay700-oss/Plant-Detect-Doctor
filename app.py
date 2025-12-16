import os
import sys
from flask import Flask, render_template, request, jsonify

# ===== FORCE PYTHON TO SEE THE MODEL FILE =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "SAMPLE_PLANT_DETECT")

if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from predict_service import (
    process_image,
    predict_diagnosis,
    load_data
)

app = Flask(__name__)

# Load model once
load_data()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image_tensor = process_image(file.read())
    diagnosis, confidence, recommendation = predict_diagnosis(image_tensor)

    return jsonify({
        "success": True,
        "diagnosis": diagnosis,
        "confidence": confidence,
        "recommendation": recommendation
    })

if __name__ == "__main__":
    app.run(debug=False)
