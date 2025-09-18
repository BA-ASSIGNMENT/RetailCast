import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


#Load the dataset
df=pd.read_csv('Dataset/online_retail.csv',encoding='ISO-8859-1')

#remove rows with missing CustomerID
df=df.dropna(subset=['CustomerID'])
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

print("Aggregated Daily Sales Data:")
print(daily_sales.head())

#-------Train forecast model using ARIMA------

#use daily sales
ts=daily_sales.set_index('Date')['TotalSales']

# Convert the index to datetime objects
ts.index = pd.to_datetime(ts.index)


#Make data stationary 
ts_diff=ts.diff().dropna()


#Train arima model
model=ARIMA(ts, order=(1,1,1))
model_fit=model.fit()

# Print the model summary for detailed information
print(model_fit.summary())

#----------predict demands for future months---------#
#Forecast
forecast_steps=90  #forecast 90 days
forecast=model_fit.forecast(steps=forecast_steps)


print("\nForecasted Sales (next 3 months):")
display(forecast)

# 
