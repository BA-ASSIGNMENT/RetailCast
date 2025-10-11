import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Dataset/online_retail.csv', encoding='latin1')

# Basic info
print("Dataset Shape:", df.shape)
print("Columns:", list(df.columns))
print("Data Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# Data cleaning
# Remove rows with missing CustomerID if needed, but for now keep
# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Calculate Total
df['Total'] = df['Quantity'] * df['UnitPrice']

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Top countries by sales
top_countries = df.groupby('Country')['Total'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Countries by Total Sales:")
print(top_countries)

# Top products by quantity sold
top_products_qty = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Products by Quantity Sold:")
print(top_products_qty)

# Top products by total sales
top_products_sales = df.groupby('Description')['Total'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Products by Total Sales:")
print(top_products_sales)

# Sales over time (monthly)
df['Month'] = df['InvoiceDate'].dt.to_period('M')
monthly_sales = df.groupby('Month')['Total'].sum()
print("\nMonthly Sales:")
print(monthly_sales)

# Save plots (optional, but since no display, just print)
# plt.figure(figsize=(10,5))
# monthly_sales.plot()
# plt.title('Monthly Sales')
# plt.savefig('monthly_sales.png')
# print("Monthly sales plot saved as monthly_sales.png")
