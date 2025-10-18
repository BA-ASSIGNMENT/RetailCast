import pandas as pd

def load_data():
    """
    Load the online_retail.csv dataset.
    """
    print("Loading data from Dataset/online_retail.csv...")
    df = pd.read_csv('Dataset/online_retail.csv', encoding='latin1')
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"First 5 rows:\n{df.head()}")
    return df

if __name__ == "__main__":
    load_data()
