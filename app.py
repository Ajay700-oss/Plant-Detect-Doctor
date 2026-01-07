import os
import sys
import subprocess
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== MODEL PATH =====
MODEL_DIR = os.path.join(BASE_DIR, "SAMPLE_PLANT_DETECT")
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from predict_service import (
    process_image,
    predict_diagnosis,
    load_data
)

app = Flask(__name__)

load_data()

# ================= VIRTUAL MOUSE PROCESS =================
mouse_process = None

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

# ================= START / STOP VIRTUAL MOUSE =================
@app.route("/toggle_mouse", methods=["POST"])
def toggle_mouse():
    global mouse_process

    if mouse_process is None:
        mouse_process = subprocess.Popen(
            [sys.executable, "hand_control.py"]
        )
        return jsonify({"status": "started"})
    else:
        mouse_process.terminate()
        mouse_process = None
        return jsonify({"status": "stopped"})

if __name__ == "__main__":
    app.run(debug=False)
