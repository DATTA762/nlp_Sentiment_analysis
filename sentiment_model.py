import re
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib

regions = ["Delhi", "Uttar Pradesh", "Maharashtra", "Bihar"]

class ElectionSentimentAnalyzer:
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def clean_text(self, text):
        text = re.sub(r"[^\w\s]", "", str(text))
        return text.lower()
    
    def get_sentiment_label(self, compound_score):
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    def extract_region(self, text):
        for region in regions:
            if region.lower() in text.lower():
                return region
        return "Other"
    
    def classify_theme(self, sentence):
        sentence = sentence.lower()
        
        if any(word in sentence for word in ["jobs", "employment", "economy", "inflation", "tax"]):
            return "Economy"
        elif any(word in sentence for word in ["health", "hospital", "scheme", "welfare", "education"]):
            return "Welfare"
        elif any(word in sentence for word in ["corrupt", "scandal", "protest", "violence", "mismanage", "failure"]):
            return "Controversy"
        elif any(word in sentence for word in ["announce", "visit", "official", "statement", "schedule"]):
            return "Announcement"
        else:
            return "Other"

    def analyze(self, text):
        text = self.clean_text(text)
        score = self.sia.polarity_scores(text)["compound"]
        
        return {
            "sentiment": self.get_sentiment_label(score),
            "score": score,
            "region": self.extract_region(text),
            "theme": self.classify_theme(text)
        }

# Save the model
if __name__ == "__main__":
    model = ElectionSentimentAnalyzer()
    joblib.dump(model, "sentiment_model.pkl")
    print("Model saved as sentiment_model.pkl")
