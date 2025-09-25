import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



# Load the dataset
df = pd.read_csv('dataset/online_retail.csv', encoding='ISO-8859-1')

# Clean data
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Aggregate daily sales
daily_sales = df.groupby(df['InvoiceDate'].dt.date).agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum'
}).reset_index()

daily_sales.columns = ['Date', 'TotalQuantity', 'TotalSales']
ts = daily_sales.set_index('Date')['TotalSales']

#------------------- Time Series Forecasting ------------------

daily_sales.columns = ['Date', 'TotalQuantity', 'TotalSales']
ts = daily_sales.set_index('Date')['TotalSales']

# Fit SARIMA model (p,d,q) x (P,D,Q,s)
# s=30 (monthly seasonality in daily data)
model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,30))
model_fit = model.fit(disp=False)

# Forecast for next 90 days
forecast_steps = 90
forecast = model_fit.forecast(steps=forecast_steps)

# Plot forecast
plt.figure(figsize=(10,5))
plt.plot(ts, label="Historical Sales")
plt.plot(pd.date_range(ts.index[-1], periods=forecast_steps+1, freq="D")[1:], 
         forecast, label="Forecast", color="red")
plt.title("SARIMA Sales Forecast (Seasonality & Trend)")
plt.legend()
plt.show()     


#------------------- Classification (Customer Segmentation) ------------------

# Reference date for recency (last purchase date in dataset)
reference_date = df['InvoiceDate'].max()

# Compute RFM values
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                   # Frequency
    'TotalPrice': 'sum'                                       # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Normalize RFM features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Assign labels to clusters (simple interpretation)
cluster_labels = {
    0: 'Loyal Customers',
    1: 'At-Risk Customers',
    2: 'High-Value Customers',
    3: 'Others'
}
rfm['Segment'] = rfm['Cluster'].map(cluster_labels)

# Display a few results
print(rfm.head(10))

# Plot clusters (Recency vs Monetary for visualization)
plt.figure(figsize=(8,6))
plt.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel("Recency (days since last purchase)")
plt.ylabel("Monetary (total spent)")
plt.title("Customer Segmentation (K-Means Clusters)")
plt.show()

