import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = "model/plant_disease_model.h5"
DATASET_PATH = "dataset/PlantVillage"

model = tf.keras.models.load_model(MODEL_PATH)

class_names = sorted(tf.io.gfile.listdir(DATASET_PATH))
print(class_names)

img = image.load_img(
    "dataset/PlantVillage/Tomato___healthy/your_image.jpg",
    target_size=(224, 224)
)

img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

pred = model.predict(img)

print(pred)
print("Prediction:", class_names[np.argmax(pred)])