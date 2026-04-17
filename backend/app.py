from flask import Flask, request, jsonify
from flask_cors import CORS  # Added for frontend-backend communication
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # This allows your frontend to talk to this API

# ✅ Using os.path.abspath to ensure the path is correct regardless of where it's run
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")

# Load model with a fallback for potential Keras 3 / Keras 2 mismatches
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

IMG_SIZE = 224

def preprocess_image(image):
    """Resizes and normalizes the image for the model."""
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
        
        # Preprocess
        processed_image = preprocess_image(image)

        # Predict
        prediction = model.predict(processed_image)
        
        # Logic for binary classification
        prob = float(prediction[0][0])
        
        # If your model outputs a single sigmoid value (0 to 1):
        # 0.5 is usually the threshold
        if prob > 0.5:
            label = "non-biodegradable"
            confidence = prob
        else:
            label = "biodegradable"
            confidence = 1 - prob

        # Apply your uncertainty threshold
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
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
