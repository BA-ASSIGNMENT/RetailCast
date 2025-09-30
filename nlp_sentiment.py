import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from config import *

def generate_synthetic_reviews(df):
    """Generate realistic synthetic customer feedback"""
    print("\nNote: Original dataset has no reviews.")
    print("Generating realistic synthetic customer feedback for demonstration...")

    np.random.seed(42)

    # Get actual products from dataset
    products_with_names = df[df['Description'].notna()].groupby('StockCode').agg({
        'Description': 'first',
        'Quantity': 'sum'
    }).sort_values('Quantity', ascending=False).head(50)  # Reduced from 100

    # Generate reviews
    reviews_data = []
    for stock_code, row in products_with_names.iterrows():
        product_desc = row['Description'].lower()
        n_reviews = np.random.randint(3, 8)  # Reduced number of reviews
        
        for _ in range(n_reviews):
            sentiment_choice = np.random.choice(['positive', 'negative', 'neutral'], 
                                                p=[0.6, 0.25, 0.15])
            
            if sentiment_choice == 'positive':
                review = np.random.choice(POSITIVE_TEMPLATES).format(product=product_desc)
                sentiment = 1
            elif sentiment_choice == 'negative':
                review = np.random.choice(NEGATIVE_TEMPLATES).format(product=product_desc)
                sentiment = 0
            else:
                review = np.random.choice(NEUTRAL_TEMPLATES).format(product=product_desc)
                sentiment = 1
            
            reviews_data.append({
                'StockCode': stock_code,
                'Product': product_desc,
                'Review': review,
                'Sentiment': sentiment
            })

    reviews_df = pd.DataFrame(reviews_data)
    print(f"\nGenerated {len(reviews_df)} product reviews")
    print(f"Positive reviews: {(reviews_df['Sentiment'] == 1).sum()}")
    print(f"Negative reviews: {(reviews_df['Sentiment'] == 0).sum()}")
    
    return reviews_df, products_with_names

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'of', 'with', 'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had',
                 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    tokens = [word for word in tokens if word not in stopwords and len(word) > 2]
    return ' '.join(tokens)

def train_sentiment_models(reviews_df):
    """Train and compare sentiment analysis models"""
    print("\n" + "-" * 60)
    print("TEXT PREPROCESSING")
    print("-" * 60)

    reviews_df['Processed_Review'] = reviews_df['Review'].apply(preprocess_text)

    print("Sample preprocessed reviews:")
    for i in range(2):  # Show fewer examples
        print(f"\nOriginal: {reviews_df.iloc[i]['Review']}")
        print(f"Processed: {reviews_df.iloc[i]['Processed_Review']}")

    # Split data
    X = reviews_df['Processed_Review']
    y = reviews_df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining set: {len(X_train)} reviews")
    print(f"Test set: {len(X_test)} reviews")

    # Feature extraction with TF-IDF
    vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))  # Reduced features
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"\nTF-IDF features: {X_train_tfidf.shape[1]}")

    # Method 1: Naive Bayes
    print("\n" + "-" * 60)
    print("MODEL 1: NAIVE BAYES CLASSIFIER")
    print("-" * 60)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_pred = nb_model.predict(X_test_tfidf)
    nb_accuracy = (nb_pred == y_test).mean()

    print(f"Accuracy: {nb_accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, nb_pred, target_names=['Negative', 'Positive']))

    # Method 2: Logistic Regression
    print("-" * 60)
    print("MODEL 2: LOGISTIC REGRESSION")
    print("-" * 60)

    lr_model = LogisticRegression(max_iter=500, random_state=42)  # Reduced iterations
    lr_model.fit(X_train_tfidf, y_train)
    lr_pred = lr_model.predict(X_test_tfidf)
    lr_accuracy = (lr_pred == y_test).mean()

    print(f"Accuracy: {lr_accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, lr_pred, target_names=['Negative', 'Positive']))
    
    return lr_model, vectorizer, nb_accuracy, lr_accuracy, X_test, y_test, lr_pred, nb_pred

def analyze_product_sentiment(reviews_df, products_with_names, lr_model, vectorizer):
    """Analyze sentiment by product"""
    print("\n" + "-" * 60)
    print("SENTIMENT INSIGHTS BY PRODUCT")
    print("-" * 60)

    reviews_df['Predicted_Sentiment'] = lr_model.predict(vectorizer.transform(reviews_df['Processed_Review']))
    product_sentiment = reviews_df.groupby('StockCode').agg({
        'Predicted_Sentiment': ['mean', 'count']
    }).round(3)

    product_sentiment.columns = ['Positive_Rate', 'Review_Count']
    product_sentiment = product_sentiment.sort_values('Positive_Rate')

    print("\nTop 5 Products with LOWEST sentiment:")
    worst_products = product_sentiment.head(5)  # Reduced from 10
    for i, (stock_code, row) in enumerate(worst_products.iterrows(), 1):
        product_name = products_with_names.loc[stock_code, 'Description']
        print(f"{i}. {stock_code} - {product_name[:35]}")
        print(f"   Positive Rate: {row['Positive_Rate']:.1%} ({int(row['Review_Count'])} reviews)")

    print("\nTop 5 Products with HIGHEST sentiment:")
    best_products = product_sentiment.tail(5)  # Reduced from 10
    for i, (stock_code, row) in enumerate(best_products.iterrows(), 1):
        product_name = products_with_names.loc[stock_code, 'Description']
        print(f"{i}. {stock_code} - {product_name[:35]}")
        print(f"   Positive Rate: {row['Positive_Rate']:.1%} ({int(row['Review_Count'])} reviews)")
    
    return product_sentiment

def integrate_sentiment_with_forecast(df, product_sentiment, products_with_names):
    """Integrate sentiment analysis with demand forecasting"""
    print("\n" + "-" * 60)
    print("SENTIMENT-ADJUSTED DEMAND FORECASTING")
    print("-" * 60)

    # Merge sentiment with product sales
    product_sales_sentiment = df.groupby('StockCode').agg({
        'Quantity': 'sum',
        'TotalPrice': 'sum'
    }).join(product_sentiment)

    product_sales_sentiment = product_sales_sentiment.dropna()

    # Adjust forecast based on sentiment
    sentiment_adjustment_factor = product_sales_sentiment['Positive_Rate']
    product_sales_sentiment['Adjusted_Forecast_Multiplier'] = sentiment_adjustment_factor

    print("\nSample Sentiment-Based Forecast Adjustments:")
    sample_products = product_sales_sentiment.head(5)  # Reduced from 10
    for stock_code, row in sample_products.iterrows():
        product_name = products_with_names.loc[stock_code, 'Description'] if stock_code in products_with_names.index else 'Unknown'
        print(f"{stock_code} - {product_name[:35]}")
        print(f"  Sentiment: {row['Positive_Rate']:.1%} → Multiplier: {row['Adjusted_Forecast_Multiplier']:.2f}x")
    
    return product_sales_sentiment

def plot_sentiment_analysis(y_test, lr_pred, reviews_df, product_sentiment, nb_accuracy, lr_accuracy):
    """Visualize sentiment analysis results - Compact version"""
    # Single column layout instead of 2x2 grid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    # Confusion matrix - Logistic Regression
    cm = confusion_matrix(y_test, lr_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax1.set_title('Confusion Matrix', fontweight='bold', fontsize=10)
    ax1.set_ylabel('True Label', fontsize=9)
    ax1.set_xlabel('Predicted Label', fontsize=9)

    # Model comparison
    models = ['Naive Bayes', 'Logistic Regression']
    accuracies = [nb_accuracy, lr_accuracy]
    bars = ax2.bar(models, accuracies, color=['coral', 'skyblue'], edgecolor='black')
    ax2.set_ylabel('Accuracy', fontsize=9)
    ax2.set_title('Model Performance', fontweight='bold', fontsize=10)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('nlp_sentiment_analysis.png', dpi=DPI, bbox_inches='tight')
    plt.show()

def get_nlp_summary(reviews_df, lr_accuracy):
    """Get summary statistics for NLP analysis"""
    return {
        'total_reviews_analyzed': len(reviews_df),
        'sentiment_accuracy': lr_accuracy,
        'positive_reviews': (reviews_df['Sentiment'] == 1).sum(),
        'negative_reviews': (reviews_df['Sentiment'] == 0).sum()
    }