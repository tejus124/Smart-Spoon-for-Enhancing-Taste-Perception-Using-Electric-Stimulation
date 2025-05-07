from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Ensure the target directory exists
os.makedirs('section_b_cv_food', exist_ok=True)

# Build a minimal CNN matching your predictor’s expectations
model = Sequential([
    Conv2D(8, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(2, activation='softmax')   # e.g. two classes: pizza vs burger
])

# Save it as HDF5
model.save('section_b_cv_food/food_model.h5')
print("✅ Dummy food_model.h5 generated.")
