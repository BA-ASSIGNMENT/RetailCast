import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def nlp_analysis():
    """
    Simulate NLP: create dummy reviews, preprocess, train sentiment model.
    """
    # Dummy data since no reviews in dataset
    reviews = [
        "Great product, love it!",
        "Bad quality, disappointed.",
        "Excellent value for money.",
        "Terrible, do not buy.",
        "Amazing, highly recommend.",
        "Poor service, awful.",
        "Fantastic item.",
        "Worst purchase ever.",
        "Good, but could be better.",
        "Neutral, nothing special."
    ] * 100  # Multiply for more data

    reviews = [r.lower() for r in reviews]  # Make lowercase for matching
    sentiments = [1 if 'great' in r or 'excellent' in r or 'amazing' in r or 'fantastic' in r or 'good' in r else 0 for r in reviews]

    # Preprocess: tokenize, clean (simple)
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(reviews)
    y = sentiments

    # Train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Plot sentiment distribution
    sentiment_counts = pd.Series(sentiments).value_counts()
    sentiment_counts.plot(kind='bar')
    plt.title('Sentiment Distribution (Dummy)')
    plt.xticks([0,1], ['Negative', 'Positive'])
    plt.savefig('images/sentiment_distribution.png')
    plt.close()

    return acc
