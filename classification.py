import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def customer_segmentation(df):
    """
    Perform RFM analysis, normalize, cluster with K-Means.
    """
    print("Starting customer segmentation...")
    print(f"Input data shape: {df.shape}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")

    # RFM
    print("Calculating RFM metrics...")
    max_date = df['InvoiceDate'].max()
    print(f"Max date for recency: {max_date}")
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency
        'TotalPrice': 'sum'  # Monetary
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})
    print(f"RFM calculated. Shape: {rfm.shape}")
    print(f"RFM sample:\n{rfm.head()}")

    # Normalize
    print("Normalizing RFM data...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    print(f"Data normalized. Scaled shape: {rfm_scaled.shape}")

    # K-Means
    print("Performing K-Means clustering with 4 clusters...")
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    print(f"Clustering completed. Cluster centers:\n{kmeans.cluster_centers_}")
    print(f"Cluster labels: {rfm['Cluster'].value_counts().to_dict()}")

    # Assign groups (simple mapping)
    print("Assigning segment names...")
    cluster_names = {0: 'High-value', 1: 'Loyal', 2: 'At-risk', 3: 'New'}
    rfm['Segment'] = rfm['Cluster'].map(cluster_names)
    print(f"Segments assigned: {rfm['Segment'].value_counts().to_dict()}")

    # Plot
    print("Creating customer segments plot...")
    segment_counts = rfm['Segment'].value_counts()
    segment_counts.plot(kind='bar')
    plt.title('Customer Segments')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/customer_segments.png')
    plt.close()
    print("Plot saved as images/customer_segments.png")

    print("Customer segmentation completed.")
    return rfm

if __name__ == "__main__":
    df = pd.read_csv('Dataset/online_retail.csv', encoding='latin1')
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    customer_segmentation(df)
