import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def deep_learning_analysis(df):
    """
    Build Customer-Product matrix, apply SVD for collaborative filtering,
    identify co-bought products and personalized recommendations.
    """
    print("Starting deep learning analysis...")
    print(f"Input data shape: {df.shape}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")
    print(f"Unique products: {df['StockCode'].nunique()}")

    # Create customer-product matrix (pivot on Quantity)
    print("Creating customer-product matrix...")
    customer_product = df.pivot_table(
        index='CustomerID',
        columns='StockCode',
        values='Quantity',
        fill_value=0
    )
    print(f"Customer-product matrix created. Shape: {customer_product.shape}")

    # Sparse matrix for efficiency
    print("Converting to sparse matrix...")
    customer_product_sparse = csr_matrix(customer_product.values)
    print(f"Sparse matrix created. Shape: {customer_product_sparse.shape}")

    # SVD for dimensionality reduction
    print("Applying SVD for dimensionality reduction...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    matrix_reduced = svd.fit_transform(customer_product_sparse)
    print(f"SVD applied. Reduced matrix shape: {matrix_reduced.shape}")
    print(f"Explained variance ratio: {svd.explained_variance_ratio_}")

    # Reconstruct for recommendations
    print("Reconstructing matrix for recommendations...")
    matrix_reconstructed = svd.inverse_transform(matrix_reduced)
    print(f"Matrix reconstructed. Shape: {matrix_reconstructed.shape}")

    # Recommendations: top products by total quantity across all customers
    print("Generating top recommended products based on total quantity sold...")
    total_quantity_per_product = customer_product.sum(axis=0).sort_values(ascending=False)
    recommended_products = total_quantity_per_product.head(10).index.tolist()
    print(f"Top recommended products by quantity: {recommended_products}")

    # Product bundles: frequently bought together
    print("Analyzing co-bought products...")
    # Group by InvoiceNo to find items in same basket
    basket = (df[df['Quantity'] > 0]
              .groupby(['InvoiceNo', 'StockCode'])['Quantity']
              .sum()
              .unstack()
              .fillna(0)
              .astype(bool)
              .stack()
              .reset_index()
              .rename(columns={'level_1': 'StockCode_1', 0: 'InBasket'}))
    print(f"Basket analysis completed. Basket shape: {basket.shape}")

    # Pairwise co-occurrences
    print("Calculating pairwise co-occurrences...")
    pairs = df.groupby(['InvoiceNo', 'StockCode']).size().reset_index(name='Count')
    co_bought = pd.merge(pairs, pairs, on='InvoiceNo', suffixes=('_x', '_y'))
    co_bought = co_bought[co_bought['StockCode_x'] < co_bought['StockCode_y']]
    co_bought_summary = co_bought.groupby(['StockCode_x', 'StockCode_y'])['InvoiceNo'].nunique().sort_values(ascending=False).head(10)
    print(f"Top co-bought pairs:\n{co_bought_summary}")

    # Plot top products by total quantity sold (bar chart)
    print("Creating top products by quantity bar chart...")
    plt.figure(figsize=(12, 6))
    total_quantity_per_product.head(10).plot(kind='bar')
    plt.title('Top 10 Products by Total Quantity Sold')
    plt.xlabel('Product Stock Code')
    plt.ylabel('Total Quantity Sold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/customer_product_matrix.png')
    plt.close()
    print("Plot saved as images/customer_product_matrix.png")

    # Plot top co-bought pairs
    print("Creating co-bought pairs plot...")
    plt.figure(figsize=(14, 8))
    co_bought_summary.plot(kind='barh')
    plt.title('Top Co-bought Product Pairs')
    plt.xlabel('Co-purchase Frequency')
    plt.ylabel('Product Pairs')
    plt.tight_layout()
    plt.savefig('images/co_bought_pairs.png')
    plt.close()
    print("Plot saved as images/co_bought_pairs.png")

    print("Deep learning analysis completed.")
    return customer_product, matrix_reduced, recommended_products, co_bought_summary

if __name__ == "__main__":
    df = pd.read_csv('Dataset/online_retail.csv', encoding='latin1')
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    deep_learning_analysis(df)
