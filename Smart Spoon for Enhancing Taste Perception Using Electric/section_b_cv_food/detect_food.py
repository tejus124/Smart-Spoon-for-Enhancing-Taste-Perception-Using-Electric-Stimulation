import cv2
import tensorflow as tf
import numpy as np

def recognize_food(image_path):
    model = tf.keras.models.load_model("section_b_cv_food/food_model.h5")
    class_names = ['Pizza', 'Burger', 'Salad', 'Sushi']

    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 3)

    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]
    print("Food detected:", predicted_class)