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
