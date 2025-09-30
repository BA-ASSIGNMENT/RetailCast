import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import *

def build_customer_product_matrix(df):
    """Build Customer-Product matrix"""
    print("\nBuilding Customer-Product matrix...")
    customer_product = df.groupby(['CustomerID', 'StockCode'])['Quantity'].sum().reset_index()
    customer_product_matrix = customer_product.pivot(index='CustomerID', 
                                                   columns='StockCode', 
                                                   values='Quantity').fillna(0)

    print(f"Matrix shape: {customer_product_matrix.shape}")
    print(f"Customers: {customer_product_matrix.shape[0]}, Products: {customer_product_matrix.shape[1]}")
    print(f"Sparsity: {(customer_product_matrix == 0).sum().sum() / customer_product_matrix.size * 100:.2f}%")

    # Limit to top products for computational efficiency
    product_sales = df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False)
    top_products = product_sales.head(TOP_N_PRODUCTS).index
    customer_product_matrix = customer_product_matrix[top_products]

    print(f"\nUsing top {TOP_N_PRODUCTS} products for analysis")
    print(f"Reduced matrix shape: {customer_product_matrix.shape}")
    
    return customer_product_matrix

def collaborative_filtering_analysis(customer_product_matrix, df):
    """Perform collaborative filtering with SVD"""
    print("\n" + "-" * 60)
    print("METHOD 1: COLLABORATIVE FILTERING (SVD)")
    print("-" * 60)

    svd = TruncatedSVD(n_components=N_COMPONENTS_SVD, random_state=42)
    customer_features = svd.fit_transform(customer_product_matrix)
    product_features = svd.components_.T

    print(f"Reduced dimensions: {N_COMPONENTS_SVD}")
    print(f"Explained variance: {svd.explained_variance_ratio_.sum():.2%}")

    # Reconstruct matrix for recommendations
    reconstructed_matrix = np.dot(customer_features, svd.components_)
    reconstructed_df = pd.DataFrame(reconstructed_matrix, 
                                     index=customer_product_matrix.index,
                                     columns=customer_product_matrix.columns)

    # Product similarity (products frequently bought together)
    product_similarity = cosine_similarity(product_features)
    product_similarity_df = pd.DataFrame(product_similarity,
                                          index=customer_product_matrix.columns,
                                          columns=customer_product_matrix.columns)

    print("\nTop 3 Product Associations (Frequently Bought Together):")
    print("-" * 60)

    # Get product names
    product_names = df.groupby('StockCode')['Description'].first()

    for i, product_code in enumerate(customer_product_matrix.columns[:3]):
        similar_products = product_similarity_df[product_code].sort_values(ascending=False)[1:4]
        product_name = product_names.get(product_code, 'Unknown')
        print(f"\n{i+1}. {product_code} - {product_name}")
        for j, (sim_code, sim_score) in enumerate(similar_products.items(), 1):
            sim_name = product_names.get(sim_code, 'Unknown')
            print(f"   {j}. {sim_code} ({sim_name[:40]}) - Similarity: {sim_score:.3f}")
    
    return reconstructed_df, product_similarity_df, svd, product_names

def build_autoencoder(customer_product_matrix):
    """Build and train autoencoder for purchase patterns"""
    print("\n" + "-" * 60)
    print("METHOD 2: AUTOENCODER NEURAL NETWORK")
    print("-" * 60)

    # Normalize data
    scaler = MinMaxScaler()
    matrix_normalized = scaler.fit_transform(customer_product_matrix.values)

    # Build Autoencoder
    input_dim = matrix_normalized.shape[1]

    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation='relu')(input_layer)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(ENCODING_DIM, activation='relu', name='bottleneck')(encoded)

    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dropout(0.2)(decoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = keras.Model(input_layer, decoded)
    encoder = keras.Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("\nAutoencoder Architecture:")
    autoencoder.summary()

    print("\nTraining Autoencoder (this may take a moment)...")
    # Reduce epochs for faster training
    history = autoencoder.fit(matrix_normalized, matrix_normalized,
                              epochs=30,  # Reduced from 50
                              batch_size=32,
                              shuffle=True,
                              validation_split=0.2,
                              verbose=0)

    print(f"Training complete!")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

    # Extract learned features
    customer_embeddings = encoder.predict(matrix_normalized, verbose=0)

    # Find anomalies (unusual purchase patterns)
    reconstructed = autoencoder.predict(matrix_normalized, verbose=0)
    mse = np.mean(np.power(matrix_normalized - reconstructed, 2), axis=1)
    threshold = np.percentile(mse, 95)

    anomalous_customers = customer_product_matrix.index[mse > threshold]
    print(f"\nIdentified {len(anomalous_customers)} customers with unusual purchase patterns")
    
    return autoencoder, encoder, history, matrix_normalized, mse, threshold, anomalous_customers

def get_personalized_recommendations(reconstructed_df, customer_product_matrix, product_names, n_customers=2):
    """Generate personalized recommendations for sample customers"""
    print("\n" + "-" * 60)
    print("PERSONALIZED RECOMMENDATIONS (Sample)")
    print("-" * 60)

    def get_recommendations(customer_id, n_recommendations=3):
        if customer_id not in reconstructed_df.index:
            return None
        
        customer_purchases = customer_product_matrix.loc[customer_id]
        customer_predictions = reconstructed_df.loc[customer_id]
        
        # Recommend products not yet purchased but with high predicted scores
        not_purchased = customer_purchases[customer_purchases == 0].index
        recommendations = customer_predictions[not_purchased].sort_values(ascending=False).head(n_recommendations)
        
        return recommendations

    # Show recommendations for sample customers
    sample_customers = customer_product_matrix.index[:n_customers]
    for i, customer_id in enumerate(sample_customers, 1):
        print(f"\n{i}. Customer {int(customer_id)}:")
        recommendations = get_recommendations(customer_id)
        if recommendations is not None:
            for j, (product_code, score) in enumerate(recommendations.items(), 1):
                product_name = product_names.get(product_code, 'Unknown')
                print(f"   {j}. {product_code} - {product_name[:40]} (Score: {score:.3f})")

def plot_deep_learning_results(history, mse, threshold):
    """Visualize deep learning training and results - Compact version"""
    # Single plot instead of subplots
    plt.figure(figsize=FIGURE_SIZE)
    
    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss', linewidth=1.5)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=1.5)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss (MSE)', fontsize=10)
    plt.title('Autoencoder Training History', fontsize=11, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('deep_learning_analysis.png', dpi=DPI, bbox_inches='tight')
    plt.show()

def get_deep_learning_summary(customer_product_matrix, anomalous_customers):
    """Get summary statistics for deep learning analysis"""
    return {
        'total_customers_analyzed': customer_product_matrix.shape[0],
        'total_products_analyzed': customer_product_matrix.shape[1],
        'anomalous_customers': len(anomalous_customers)
    }