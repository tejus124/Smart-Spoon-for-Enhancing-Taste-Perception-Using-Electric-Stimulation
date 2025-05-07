def predict_behavior_change(sentiment):
    if sentiment == "Positive":
        return "User will likely continue similar food choices."
    else:
        return "User might explore new flavors."