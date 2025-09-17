
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#Load the dataset
df=pd.read_csv('Dataset/online_retail.csv',encoding='ISO-8859-1')

#remove rows with missing CustomerID
df=df.dropna(subset='CustomerID')
#Remove negative quantities
df=df[df['Quantity']>0]
#Create a new column for total price
df['TotalPrice']=df['Quantity']*df['UnitPrice']

#convert Invoicedate to datetime format
df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'])
#Extract time,year,day and time
df['Year']=df['InvoiceDate'].dt.year
df['Month']=df['InvoiceDate'].dt.month
df['Day']=df['InvoiceDate'].dt.day
df['Hour']=df['InvoiceDate'].dt.hour

#------------------ Aggregate sales by Date ------------------

daily_sales = df.groupby(df['InvoiceDate'].dt.date).agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum'
}).reset_index()

daily_sales.columns = ['Date', 'TotalQuantity', 'TotalSales']


#print("Aggregated Daily Sales Data:")
#print(daily_sales.head())

#------------------- Time Series Forecasting ------------------

daily_sales.columns = ['Date', 'TotalQuantity', 'TotalSales']
ts = daily_sales.set_index('Date')['TotalSales']

# Fit SARIMA model (p,d,q) x (P,D,Q,s)
# s=30 (monthly seasonality in daily data)
model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,30))
model_fit = model.fit(disp=False)

print(model_fit.summary())

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


