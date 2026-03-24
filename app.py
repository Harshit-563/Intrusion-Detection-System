from flask import Flask, request, jsonify
import torch
import pickle
import numpy as np
from flask import render_template

from nsl_kdd import Net

# ==============================
# App Init
# ==============================
app = Flask(__name__)

# ==============================
# Load Model + Preprocessing
# ==============================
MODEL_PATH = "models/global_model_final.pth"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"



model = Net()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

INPUT_DIM = 41


# ==============================
# Routes
# ==============================

# Health check
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ui")
def ui():
    return render_template("index.html")

# Predict
import torch.nn.functional as F

@app.route("/predict", methods=["POST"])
def predict():
    label_map = {
    "dos": "DoS Attack",
    "probe": "Probe Attack",
    "r2l": "R2L Attack",
    "u2r": "U2R Attack",
    "normal": "Normal Traffic"
}
    try:
        data = request.json.get("features")

        # ---- Validation ----
        if not data:
            return jsonify({"error": "No input provided"}), 400

        if len(data) != INPUT_DIM:
            return jsonify({"error": f"Expected {INPUT_DIM} features"}), 400

        # ---- Preprocess ----
        data = np.array(data).reshape(1, -1)
        data_scaled = scaler.transform(data)
        tensor = torch.tensor(data_scaled, dtype=torch.float32)

        # ---- Prediction + Confidence ----
        with torch.no_grad():
            output = model(tensor)

            probs = F.softmax(output, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs.max().item()

        raw_label = label_encoder.inverse_transform([pred])[0]
        label = label_map.get(raw_label, raw_label)

        # ---- Response ----
        return jsonify({
            "label": label,
            "status": "Attack Detected" if raw_label != "normal" else "Normal Traffic",
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# Run Server
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
