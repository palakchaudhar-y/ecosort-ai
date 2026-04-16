from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# load model
model = tf.keras.models.load_model("waste_model.h5")

IMG_SIZE = 224

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/")
def home():
    return "EcoSort AI backend is running!"


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")

    image = preprocess_image(image)

    prediction = model.predict(image)

    prob = float(prediction[0][0])

    confidence = max(prob, 1 - prob)

    print("Raw probability:", prob)
    print("Confidence:", confidence)

    if confidence >= 0.70:

        if prob > 0.5:
            label = "non-biodegradable"
        else:
            label = "biodegradable"

    else:
        label = "uncertain"

    print("Prediction:", label)

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)