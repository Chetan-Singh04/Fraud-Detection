"""
Flask web + API for fraud detection
-----------------------------------
• GET  /              → HTML form
• POST /predict-form  → returns HTML result
• POST /predict       → JSON API
• GET  /api           → health-check string
"""

import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from src.predict import predict_transaction  # <-- keep the src. prefix

import csv
from datetime import datetime

LOG_FILE = "logs/predictions.csv"

def log_prediction(data: dict, label: str):
    # Ensure file exists with headers
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + list(data.keys()) + ["prediction"])
    
    # Write one row per prediction
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat()] + list(data.values()) + [label])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
    "V28", "Amount"
]

# Tell Flask where to find templates/ if you didn’t create one at project root
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api")
def api_health():
    """Simple health-check."""
    return "Fraud Detection API is running!"

# ---------- Dashboard ----------

@app.route("/")
def index():
    return render_template("index.html", features=FEATURES)

@app.route("/predict-form", methods=["POST"])
def predict_form():
    try:
        data = {f: float(request.form[f]) for f in FEATURES}
        label = predict_transaction(data)
        log_prediction(data, label)
        return render_template("result.html", prediction=label)
    except Exception as exc:
        return f"Error: {exc}", 400

# ---------- JSON API ----------

@app.route("/predict", methods=["POST"])
def predict_api():
    if not request.is_json:
        return jsonify(error="JSON expected"), 415
    data = request.get_json()
    try:
        label = predict_transaction(data)
        log_prediction(data, label)
        return jsonify(prediction=label)
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    

@app.route("/history")
def history():
    try:
        with open(LOG_FILE, mode="r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            headers = rows[0]
            data = rows[1:]
        return render_template("history.html", headers=headers, rows=data[::-1])  # reverse = latest first
    except Exception as exc:
        return f"No logs yet or error reading log file: {exc}", 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure project root is on PYTHONPATH when you run from shell
    os.environ.setdefault("PYTHONPATH", ".")
    app.run(debug=os.getenv("FLASK_ENV") == "development")
