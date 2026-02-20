import numpy as np
from PIL import Image
import tensorflow as tf
from src.config import settings

model = tf.keras.models.load_model(settings.MODEL_PATH)

def preprocess_image(file):
    image = Image.open(file).convert("RGB")
    image = image.resize(settings.IMAGE_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(file):
    img_array = preprocess_image(file)
    prediction = model.predict(img_array)
    prob = float(prediction[0][0])
    probabilities = [round(1 - prob, 4), round(prob, 4)]

    # If model isn't confident enough, return unknown
    confidence = max(prob, 1 - prob)
    if confidence < settings.THRESHOLD:
        return "unknown", probabilities

    predicted_class = settings.CLASS_NAMES[1] if prob >= 0.5 else settings.CLASS_NAMES[0]
    return predicted_class, probabilities