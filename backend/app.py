from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Keras 2 to 3 Compatibility Patch ---
# This tells Keras to ignore the keywords it doesn't recognize
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import InputLayer

class CompatibleInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        # Remove keywords that cause the crash in newer Keras versions
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        super().__init__(*args, **kwargs)

get_custom_objects()['InputLayer'] = CompatibleInputLayer
# ---------------------------------------

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")

model = None
try:
    # We load using the custom object we just defined
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ SUCCESS: Model loaded successfully with compatibility patch!")
except Exception as e:
    print(f"❌ ERROR: Could not load model. Reason: {e}")

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
        "model_loaded": model is not None,
        "backend": "EcoSort AI"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        image = Image.open(file).convert("RGB")
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)
        prob = float(prediction[0][0])
        
        # Binary Classification Logic
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
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
