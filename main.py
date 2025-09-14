import pandas as pd   
import numpy as np  
import matplotlib.pyplot as plt
from statsmodel.tsa.arima.model import arima
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

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

#-------Train forecast model using ARIMA------

#use daily sales
ts=daily_sales.set_index('Date')['TotalSales']
#plot time series
ts.plot(figsize=(10,5),title="Daily Sales")
plt.show()

#Make data stationary
ts_diff=ts.diff().dropna()

plot_acf(ts_diff,lags=30)
plot_pacf(ts_diff,lags=30)
plt.show()

#Train arima model
model=arima(ts, order=(1,1,1))
model_fit=model.fit()


print(model_fit.summary())

#Forecast
forecast_steps=90  #forecast 90 days
forecast=model_fit.forecast(steps=forecast_steps)


#plot forecast 
plt.figure(figsize=(10,5))
plt.plot(ts,label="Historical sales")
plt.plot(pd.date_range(ts.index[-1], periods=forecast_steps+1, freq="D")[1:], 
         forecast, label="Forecast", color="red")
plt.title("ARIMA Sales Forecast")
plt.legend()
plt.show()



