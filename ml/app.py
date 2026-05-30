from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

from model import InhibitorCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT   = Path(__file__).parent / "checkpoints"

def _load(filename):
    m = InhibitorCNN(num_classes=3).to(DEVICE)
    m.load_state_dict(torch.load(CKPT / filename, map_location=DEVICE))
    m.eval()
    return m

MODELS = {
    "standard":  _load("best_model.pt"),
    "detection": _load("best_model_detection.pt"),
}
print(f"Models loaded on {DEVICE}: {list(MODELS)}")

LABELS = ["🟢 Inhibidor selectivo", "🔴 Inhibidor tóxico", "⚫ Inactivo"]
COLORS = ["#c8e6c9", "#ffccbc", "#cfd8dc"]

ROOT = Path(__file__).parent.parent
app = Flask(__name__, static_folder=str(ROOT))


@app.route("/")
def index():
    return send_from_directory(str(ROOT), "app_webcam_v2.html")


@app.route("/imgs/<path:filename>")
def imgs(filename):
    return send_from_directory(str(ROOT / "imgs"), filename)


@app.route("/examples/<path:filename>")
def examples_file(filename):
    return send_from_directory(str(ROOT / "data" / "examples"), filename)


@app.route("/examples")
def examples_list():
    folder = ROOT / "data" / "examples"
    files = sorted(f.name for f in folder.iterdir() if f.is_file())
    return jsonify(files)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body   = request.json
        pixels = np.array(body["pixels"], dtype=np.float32)
        model_key = body.get("model", "standard")
        if model_key not in MODELS:
            return jsonify({"error": f"Unknown model '{model_key}'"}), 400

        tensor = torch.from_numpy(pixels).reshape(1, 1, 28, 28).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(MODELS[model_key](tensor), dim=1)[0].cpu().numpy()
        cls = int(probs.argmax())
        return jsonify({
            "label":      LABELS[cls],
            "color":      COLORS[cls],
            "confidence": round(float(probs[cls]) * 100),
        })
    except Exception as e:
        print(f"[predict error] {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
