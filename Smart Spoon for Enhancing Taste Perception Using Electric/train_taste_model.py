import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load data
df = pd.read_csv('taste_data.csv')

# Features and labels
X = df[['sweet', 'salty', 'spicy', 'bitter', 'umami']]
y = df['label']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
os.makedirs('models', exist_ok=True)
with open('models/taste_profile_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as models/taste_profile_model.pkl")


