import streamlit as st
import requests
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import pandas as pd

# Download required resources
nltk.download('vader_lexicon')
nltk.download('punkt')

sia = SentimentIntensityAnalyzer()

st.title("🗳️ Election News Sentiment Analyzer (Live API)")

# 🔐 Use secrets in production
API_KEY = "9e9021d1753aa3694f5dde03f8104bb0"

# Text cleaning
def clean_text(text):
    text = re.sub(r"[^\w\s]", "", str(text))
    return text.lower()

# Word-level sentiment extraction
def extract_word_sentiment(text):
    words = word_tokenize(text.lower())
    
    positive_words = []
    negative_words = []
    neutral_words = []

    for word in words:
        score = sia.lexicon.get(word, 0)
        
        if score > 0:
            positive_words.append(word)
        elif score < 0:
            negative_words.append(word)
        else:
            neutral_words.append(word)
    
    return positive_words, negative_words, neutral_words


if st.button("Fetch Latest Election News"):

    url = f"https://gnews.io/api/v4/search?q=election%20India&lang=en&country=in&max=5&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    articles = data.get("articles", [])

    for article in articles:
        
        st.subheader(article["title"])
        content = article.get("content", "")
        
        cleaned = clean_text(content)

        # Overall sentiment
        overall_score = sia.polarity_scores(cleaned)["compound"]

        if overall_score >= 0.05:
            overall_sentiment = "Positive"
        elif overall_score <= -0.2263:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"

        st.write("### 📊 Overall Sentiment:", overall_sentiment)
        st.write("Score:", overall_score)

        # Word-level sentiment
        pos_words, neg_words, neu_words = extract_word_sentiment(cleaned)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### ✅ Positive Words")
            st.write(list(set(pos_words)))

        with col2:
            st.markdown("#### ❌ Negative Words")
            st.write(list(set(neg_words)))

        with col3:
            st.markdown("#### ⚖ Neutral Words")
            st.write(list(set(neu_words[:20])))

        # Chart
        sentiment_counts = {
            "Positive": len(pos_words),
            "Negative": len(neg_words),
            "Neutral": len(neu_words)
        }

        df = pd.DataFrame(
            list(sentiment_counts.items()),
            columns=["Sentiment", "Count"]
        )

        st.bar_chart(df.set_index("Sentiment"))

        st.markdown("---")
