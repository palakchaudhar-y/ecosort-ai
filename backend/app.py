from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "waste_model.h5")

# Load model safely for compatibility
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

IMG_SIZE = 224

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "message": "EcoSort AI backend is running!",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        image = Image.open(file).convert("RGB")

        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)

        prob = float(prediction[0][0])

        if prob > 0.5:
            label = "non-biodegradable"
            confidence = prob
        else:
            label = "biodegradable"
            confidence = 1 - prob

        if confidence < 0.70:
            label = "uncertain"

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 4),
            "raw_score": round(prob, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)