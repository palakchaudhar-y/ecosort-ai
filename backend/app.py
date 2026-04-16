from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("waste_model.h5")

IMG_SIZE = 224

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")

    image = preprocess_image(image)

    prediction = model.predict(image)

    prob = float(prediction[0][0])

    # convert probability to confidence
    confidence = max(prob, 1 - prob)

    print("Raw probability:", prob)
    print("Confidence:", confidence)

    # 🔥 confidence check
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
    app.run(debug=True)