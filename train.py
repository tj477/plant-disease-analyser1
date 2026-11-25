import os
import tensorflow as tf

# Silence non-critical TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = "dataset/PlantVillage"

print("--- Loading Datasets ---")

# Load Training Set
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"  # Critical for categorical_crossentropy loss
)

# Load Validation Set
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# Get total number of classes
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"\nDetected {num_classes} classes: {class_names}\n")

# Performance optimization (prefetching data into memory)
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# Build CNN Model
print("--- Building CNN Model ---")
model = tf.keras.Sequential([
    # Rescaling pixels from [0, 255] to [0, 1]
    tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train Model
print("--- Starting Training ---")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Save Model
os.makedirs("model", exist_ok=True)
save_path = "model/plant_disease_model.h5"
model.save(save_path)

print(f"\n✅ Model trained successfully and saved to: {save_path}")