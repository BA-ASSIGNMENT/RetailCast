import pandas as pd

def load_data():
    """
    Load the online_retail.csv dataset.
    """
    df = pd.read_csv('Dataset/online_retail.csv', encoding='latin1')
    return df
