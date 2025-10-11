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
    # Create customer-product matrix (pivot on Quantity)
    customer_product = df.pivot_table(
        index='CustomerID', 
        columns='StockCode', 
        values='Quantity', 
        fill_value=0
    )

    # Sparse matrix for efficiency
    customer_product_sparse = csr_matrix(customer_product.values)

    # SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=50, random_state=42)
    matrix_reduced = svd.fit_transform(customer_product_sparse)

    # Reconstruct for recommendations
    matrix_reconstructed = svd.inverse_transform(matrix_reduced)

    # Personalized recommendations: for a sample customer (first one)
    sample_customer = 0
    customer_reconstructed = matrix_reconstructed[sample_customer]
    recommendations = np.argsort(customer_reconstructed)[-10:]  # Top 10 recommended products
    recommended_products = customer_product.columns[recommendations]

    # Product bundles: frequently bought together
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

    # Pairwise co-occurrences
    pairs = df.groupby(['InvoiceNo', 'StockCode']).size().reset_index(name='Count')
    co_bought = pd.merge(pairs, pairs, on='InvoiceNo', suffixes=('_x', '_y'))
    co_bought = co_bought[co_bought['StockCode_x'] < co_bought['StockCode_y']]
    co_bought_summary = co_bought.groupby(['StockCode_x', 'StockCode_y'])['InvoiceNo'].nunique().sort_values(ascending=False).head(10)

    # Plot customer-product matrix heatmap (sample)
    plt.figure(figsize=(10, 6))
    sample_matrix = customer_product.iloc[:10, :10]  # Small sample for plot
    plt.imshow(sample_matrix, cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.title('Sample Customer-Product Matrix')
    plt.xlabel('Products')
    plt.ylabel('Customers')
    plt.savefig('images/customer_product_matrix.png')
    plt.close()

    # Plot top co-bought pairs
    co_bought_summary.plot(kind='bar')
    plt.title('Top Co-bought Product Pairs')
    plt.savefig('images/co_bought_pairs.png')
    plt.close()

    return customer_product, matrix_reduced, recommended_products, co_bought_summary
