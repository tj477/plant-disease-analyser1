import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model
MODEL_PATH = "model/plant_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Dataset path
DATASET_PATH = "dataset/PlantVillage"
CLASS_NAMES = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print("Classes:")
print(CLASS_NAMES)


def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    print("\nPrediction probabilities:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls}: {predictions[0][i]:.4f}")

    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx]) * 100

    return CLASS_NAMES[predicted_class_idx], confidence


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded.")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", error="Please select an image.")

    upload_folder = os.path.join("static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    disease, confidence = predict_disease(file_path)

    return render_template(
        "index.html",
        disease=disease,
        confidence=f"{confidence:.2f}%",
        user_image=file_path
    )


if __name__ == "__main__":
    app.run(debug=True)