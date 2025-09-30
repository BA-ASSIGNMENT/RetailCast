import pandas as pd
import numpy as np
from config import DATA_PATH

def load_and_clean_data():
    """Load and clean the retail dataset"""
    print("=" * 60)
    print("STEP 1: LOADING AND CLEANING DATA")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Display initial data info
    print("\nInitial missing values:")
    print(df.isnull().sum())

    # Clean data
    print("\nCleaning data...")
    df_original_size = len(df)

    # Remove rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    print(f"Rows removed (missing CustomerID): {df_original_size - len(df)}")

    # Remove transactions with Quantity <= 0 (returns/cancellations)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    print(f"Final dataset shape after cleaning: {df.shape}")

    # Create new feature: TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Extract time features
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['Week'] = df['InvoiceDate'].dt.isocalendar().week
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Date'] = df['InvoiceDate'].dt.date

    print("\nData cleaning completed!")
    print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    print(f"Total revenue: ${df['TotalPrice'].sum():,.2f}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")
    
    return df