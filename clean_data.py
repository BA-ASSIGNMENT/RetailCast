import pandas as pd

def clean_data(df):
    """
    Clean the data as per the steps:
    - Remove rows with missing CustomerID
    - Remove transactions with Quantity <= 0
    - Create TotalPrice = Quantity * UnitPrice
    - Convert InvoiceDate to datetime
    - Extract Year, Month, Day, Week, Hour
    """
    # Remove missing CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Remove Quantity <= 0
    df = df[df['Quantity'] > 0]

    # Create TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Convert InvoiceDate
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Extract features
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['Week'] = df['InvoiceDate'].dt.isocalendar().week
    df['Hour'] = df['InvoiceDate'].dt.hour

    return df
