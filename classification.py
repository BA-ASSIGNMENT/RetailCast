import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def customer_segmentation(df):
    """
    Perform RFM analysis, normalize, cluster with K-Means.
    """
    # RFM
    max_date = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency
        'TotalPrice': 'sum'  # Monetary
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

    # Normalize
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Assign groups (simple mapping)
    cluster_names = {0: 'High-value', 1: 'Loyal', 2: 'At-risk', 3: 'New'}
    rfm['Segment'] = rfm['Cluster'].map(cluster_names)

    # Plot
    plt.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'])
    plt.xlabel('Recency')
    plt.ylabel('Monetary')
    plt.title('Customer Segments')
    plt.savefig('images/customer_segments.png')
    plt.close()

    return rfm