# EcoSort AI 🌱

EcoSort AI is an AI-powered web application that helps users classify waste into **Biodegradable** and **Non-Biodegradable** categories.

Users can upload an image of waste (such as food waste, plastic, glass, etc.), and the AI model predicts the correct category to help improve waste segregation.

## 🚀 Features

* Upload an image of waste
* AI model analyzes the image
* Classifies waste into:

  * Biodegradable
  * Non-Biodegradable
* If the model is not confident, it asks the user for input
* Simple and clean user interface

## 🧠 Technologies Used

* **Python**
* **Flask**
* **TensorFlow / Keras**
* **HTML**
* **CSS**
* **JavaScript**

## 📂 Project Structure

EcoSortAI
│
├── backend
│   ├── app.py
│   ├── waste_model.h5
│   └── requirements.txt

└── frontend
└── index.html

## ⚙️ How It Works

1. User uploads an image.
2. The image is sent to the Flask backend.
3. The TensorFlow model analyzes the image.
4. The model returns a prediction:

   * Biodegradable
   * Non-Biodegradable
5. The result is displayed on the website.

## 🌍 Purpose

The goal of this project is to promote **smart waste segregation** and support **environmental sustainability** using Artificial Intelligence.

## 👩‍💻 Author

Palak Chaudhary
