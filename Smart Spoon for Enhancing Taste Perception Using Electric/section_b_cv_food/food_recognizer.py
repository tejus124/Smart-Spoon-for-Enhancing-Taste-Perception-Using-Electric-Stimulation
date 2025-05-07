from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def predict_food(img_path):
    # Load the trained model
    model = load_model('section_b_cv_food/food_model.h5')

    # Preprocess the image
    img = image.load_img(img_path, target_size=(100, 100))
    x   = image.img_to_array(img) / 255.0
    x   = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)               # e.g. [[0.1, 0.8]]
    class_idx = np.argmax(preds, axis=1)[0]

    # Map index back to label (from your training generator)
    labels = list(model.class_indices.keys()) if hasattr(model, 'class_indices') else ['pizza','burger']
    print("Food detected:", labels[class_idx])

