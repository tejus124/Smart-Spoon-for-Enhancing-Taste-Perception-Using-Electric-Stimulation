# import json
# import pickle
# import numpy as np
#
# def predict_taste(user_preferences_file):
#     with open(user_preferences_file, 'r') as f:
#         prefs = json.load(f)
#
#     features = np.array([
#         prefs.get("sweet", 0),
#         prefs.get("salty", 0),
#         prefs.get("umami", 0),
#         prefs.get("sour", 0),
#         prefs.get("bitter", 0)
#     ]).reshape(1, -1)
#
#     with open("models/taste_profile_model.pkl", "rb") as f:
#         model = pickle.load(f)
#
#     prediction = model.predict(features)
#
#     # with open("models/taste_profile_model.pkl", "rb") as model_file:
#     #     model = pickle.load(model_file)
#
#     # prediction = model.predict(features)
#     print("Predicted taste profile:", prediction[0])
# import json
# import pickle
#
# # Load user preferences (you must convert to proper features here)
# with open("data/user_preferences.json", "r") as f:
#     user_prefs = json.load(f)
#
# # Example conversion: this must match your training features
# features = [[int(user_prefs.get("likes_sweet", 0)), int(user_prefs.get("likes_spicy", 0))]]
#
# # Load trained model
# with open("models/taste_profile_model.pkl", "rb") as f:
#     model = pickle.load(f)
#
# # Predict using the actual model
# prediction = model.predict(features)
# print("Predicted class:", prediction)

import json
import pickle

def predict_taste(user_preferences_file):
    # Load user preferences
    with open(user_preferences_file, 'r') as f:
        prefs = json.load(f)

    # Convert JSON to features in the same order as model training
    # features = [[
    #     int(prefs.get("likes_sweet", False)),
    #     int(prefs.get("likes_spicy", False)),
    #     int(prefs.get("likes_sour", False))
    # ]]
    # ... after loading prefs = json.load(f)
    features = [[
        int(prefs.get("likes_sweet", False)),
        int(prefs.get("likes_salty", False)),
        int(prefs.get("likes_spicy", False)),
        int(prefs.get("likes_bitter", False)),
        int(prefs.get("likes_umami", False)),
    ]]

    # Load trained model
    with open("models/taste_profile_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predict
    prediction = model.predict(features)
    print("Predicted taste profile:", prediction[0])

