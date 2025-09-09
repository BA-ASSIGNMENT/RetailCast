import pandas as pd   
import numpy as np  

#Load the dataset
df=pd.read_csv('Dataset/online_retail.csv',encoding='ISO-8859-1')
print(df.head())
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

print(df[['InvoiceDate','Year','Month','Day','Hour']].head())
