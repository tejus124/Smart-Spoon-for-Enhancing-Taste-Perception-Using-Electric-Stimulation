from section_a_ml_taste1.taste_predictor import predict_taste
from section_b_cv_food.detect_food import recognize_food
from section_c_data_analysis.analyze_data import run_analysis
from section_d_sentiment_user.sentiment_model import analyze_sentiment
from section_b_cv_food.food_recognizer import predict_food


def main():
    print("Running Section A: Taste Profile Prediction")
    try:
        predict_taste("data/user_preferences.json")
    except Exception as e:
        print(f"Error occurred: {e}")

    print("\nRunning Section B: Food Recognition")
    try:
        recognize_food("data/sample_image.jpg")
    except Exception as e:
        print(f"Error occurred: {e}")

    # predict_taste("data/user_preferences.json")

    print("\nRunning Section C: Data Analysis")
    try:
        run_analysis("data/sample_data.csv")
    except Exception as e:
        print(f"Error occurred: {e}")

    print("\nRunning Section D: Sentiment Analysis")
    try:
        analyze_sentiment("I enjoyed the spicy curry today!")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()

