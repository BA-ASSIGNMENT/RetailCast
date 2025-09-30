import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import timedelta
from config import *

def perform_rfm_analysis(df):
    """Perform RFM analysis and customer segmentation"""
    print("\n" + "=" * 60)
    print("STEP 3: CUSTOMER SEGMENTATION - RFM ANALYSIS")
    print("=" * 60)

    # Calculate RFM metrics
    analysis_date = df['InvoiceDate'].max() + timedelta(days=1)
    print(f"\nAnalysis date (reference): {analysis_date.date()}")

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency
        'TotalPrice': 'sum'  # Monetary
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    print(f"\nRFM metrics calculated for {len(rfm)} customers")
    print("\nRFM Statistics:")
    print(rfm[['Recency', 'Frequency', 'Monetary']].describe())
    
    return rfm

def perform_clustering(rfm):
    """Perform K-means clustering on RFM data"""
    # Normalize RFM features
    scaler = StandardScaler()
    rfm_normalized = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # Find optimal number of clusters using Silhouette Score
    print("\nFinding optimal number of clusters...")
    silhouette_scores = []

    for k in CLUSTER_RANGE:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = kmeans_temp.fit_predict(rfm_normalized)
        score = silhouette_score(rfm_normalized, labels_temp)
        silhouette_scores.append(score)
        print(f"K={k}: Silhouette Score = {score:.3f}")

    optimal_k = CLUSTER_RANGE[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters: {optimal_k}")

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_normalized)
    
    return rfm, optimal_k

def analyze_segments(rfm):
    """Analyze and label customer segments"""
    print("\n" + "-" * 60)
    print("CLUSTER ANALYSIS")
    print("-" * 60)

    cluster_summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).round(2)

    cluster_summary.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Customer_Count']
    cluster_summary = cluster_summary.sort_values('Avg_Monetary', ascending=False)

    # Assign meaningful labels based on RFM characteristics
    def assign_segment_label(row):
        if row['Avg_Monetary'] > rfm['Monetary'].median() and row['Avg_Recency'] < rfm['Recency'].median():
            return "High-Value Loyal"
        elif row['Avg_Monetary'] > rfm['Monetary'].median() and row['Avg_Recency'] > rfm['Recency'].median():
            return "High-Value At-Risk"
        elif row['Avg_Frequency'] > rfm['Frequency'].median() and row['Avg_Recency'] < rfm['Recency'].median():
            return "Frequent Shoppers"
        elif row['Avg_Recency'] > rfm['Recency'].quantile(0.75):
            return "Lost Customers"
        else:
            return "Potential Loyals"

    cluster_summary['Segment'] = cluster_summary.apply(assign_segment_label, axis=1)

    print("\nCustomer Segments:")
    print(cluster_summary.to_string())
    
    return cluster_summary

def plot_segmentation(rfm):
    """Visualize customer segmentation results - Compact version"""
    # Single plot instead of subplots to reduce size
    plt.figure(figsize=COMPACT_FIGURE_SIZE)
    
    # Create a single informative scatter plot
    scatter = plt.scatter(rfm['Recency'], rfm['Monetary'], 
                         c=rfm['Cluster'], cmap='viridis', alpha=0.6, s=30)
    plt.xlabel('Recency (days)', fontsize=9)
    plt.ylabel('Monetary Value ($)', fontsize=9)
    plt.title('Customer Segmentation: Recency vs Monetary', fontsize=10, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('customer_segmentation.png', dpi=DPI, bbox_inches='tight')
    plt.show()

def get_segmentation_summary(rfm, optimal_k):
    """Get summary statistics for segmentation"""
    print(f"\n✓ Customer segments identified: {optimal_k} clusters")
    print(f"✓ Total customers analyzed: {len(rfm)}")
    print(f"✓ Average customer value: ${rfm['Monetary'].mean():,.2f}")
    
    return {
        'total_customers': len(rfm),
        'optimal_clusters': optimal_k,
        'avg_customer_value': rfm['Monetary'].mean()
    }