import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Dummy training data
X = np.array([
    [0.8, 0.2, 0.7, 0.1, 0.1],
    [0.1, 0.9, 0.2, 0.1, 0.5],
    [0.5, 0.4, 0.5, 0.3, 0.2],
])
y = ['Sweet Lover', 'Salty Lover', 'Balanced']

model = LogisticRegression()
model.fit(X, y)

with open("models/taste_profile_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved.")

