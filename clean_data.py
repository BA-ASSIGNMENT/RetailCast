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
    print(f"Starting data cleaning. Initial shape: {df.shape}")
    print(f"Initial missing values:\n{df.isnull().sum()}")

    # Remove missing CustomerID
    print("Removing rows with missing CustomerID...")
    initial_shape = df.shape
    df = df.dropna(subset=['CustomerID'])
    print(f"Removed {initial_shape[0] - df.shape[0]} rows with missing CustomerID. New shape: {df.shape}")

    # Remove Quantity <= 0
    print("Removing transactions with Quantity <= 0...")
    initial_shape = df.shape
    df = df[df['Quantity'] > 0]
    print(f"Removed {initial_shape[0] - df.shape[0]} rows with Quantity <= 0. New shape: {df.shape}")

    # Create TotalPrice
    print("Creating TotalPrice column as Quantity * UnitPrice...")
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    print(f"TotalPrice column created. Sample values: {df['TotalPrice'].head().tolist()}")

    # Convert InvoiceDate
    print("Converting InvoiceDate to datetime...")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"InvoiceDate converted. Sample dates: {df['InvoiceDate'].head().tolist()}")

    # Extract features
    print("Extracting date features: Year, Month, Day, Week, Hour...")
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['Week'] = df['InvoiceDate'].dt.isocalendar().week
    df['Hour'] = df['InvoiceDate'].dt.hour
    print(f"Date features extracted. New columns: {['Year', 'Month', 'Day', 'Week', 'Hour']}")
    print(f"Final shape after cleaning: {df.shape}")
    print(f"Final missing values:\n{df.isnull().sum()}")

    return df

if __name__ == "__main__":
    df = pd.read_csv('Dataset/online_retail.csv', encoding='latin1')
    clean_data(df)
