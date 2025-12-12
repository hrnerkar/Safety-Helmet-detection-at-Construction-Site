import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load trained model
model = tf.keras.models.load_model("helmet_classifier_model.h5")
print("✅ Model loaded successfully.")

# Prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        print(f"🟢 Prediction: HELMET ({prediction:.2f})")
    else:
        print(f"🔴 Prediction: NO HELMET ({prediction:.2f})")

# Replace this with your image path
test_image_path = "violation_20250624-165331.jpg"
predict_image(test_image_path)
