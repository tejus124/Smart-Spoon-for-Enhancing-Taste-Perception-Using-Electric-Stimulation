# section_d_sentiment_user/sentiment_model.py
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    sentiment = "Positive" if score['compound'] > 0 else "Negative"
    print("Detected sentiment:", sentiment)

    if 'spicy' in text.lower():
        print("Predicted user preference shift: Towards spicy dishes")
    else:
        print("No major shift detected")
