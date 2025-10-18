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
    print("Starting NLP analysis...")
    print("Creating dummy reviews data...")

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
        "Neutral, nothing special.",
        "Absolutely wonderful, exceeded expectations!",
        "Horrible experience, never again.",
        "Superb quality and fast shipping.",
        "Awful, waste of money.",
        "Incredible design, very satisfied.",
        "Disappointing, not as described.",
        "Outstanding customer support.",
        "Regret buying this, poor build.",
        "Highly impressed with the features.",
        "Subpar performance, not recommended.",
        "Brilliant purchase, will buy more.",
        "Frustrating, full of defects.",
        "Exceptional value, great price.",
        "Dreadful, avoid at all costs.",
        "Marvelous, better than expected.",
        "Unsatisfactory, low quality.",
        "Perfect for my needs, happy!",
        "Terrible packaging, arrived damaged.",
        "Top-notch product, five stars.",
        "Unacceptable, customer service ignored me."
    ] * 50  # Multiply for more data

    print(f"Base reviews: {reviews[:10]}")
    print(f"Total reviews after multiplication: {len(reviews)}")

    reviews = [r.lower() for r in reviews]  # Make lowercase for matching
    sentiments = [1 if 'great' in r or 'excellent' in r or 'amazing' in r or 'fantastic' in r or 'good' in r else 0 for r in reviews]

    print(f"Sample processed reviews: {reviews[:5]}")
    print(f"Corresponding sentiments: {sentiments[:5]}")
    print(f"Sentiment distribution: Positive: {sum(sentiments)}, Negative: {len(sentiments) - sum(sentiments)}")

    # Preprocess: tokenize, clean (simple)
    print("Preprocessing: vectorizing reviews...")
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(reviews)
    y = sentiments
    print(f"Feature matrix shape: {X.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Sample features: {list(vectorizer.vocabulary_.keys())[:10]}")

    # Train/test
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Model
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # Predict
    print("Making predictions on test set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Predictions sample: {y_pred[:10]}")
    print(f"Actual labels sample: {y_test[:10]}")
    print(f"Accuracy: {acc}")

    # Plot sentiment distribution
    print("Creating sentiment distribution plot...")
    sentiment_counts = pd.Series(sentiments).value_counts()
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar')
    plt.title('Sentiment Distribution (Dummy)')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.xticks([0,1], ['Negative', 'Positive'])
    plt.tight_layout()
    plt.savefig('images/sentiment_distribution.png')
    plt.close()
    print("Plot saved as images/sentiment_distribution.png")

    print("NLP analysis completed.")
    return acc

if __name__ == "__main__":
    nlp_analysis()
