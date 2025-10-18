import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def time_series_analysis(df):
    """
    Aggregate sales by date, train Prophet model, predict future demand.
    """
    print("Starting time series analysis...")
    print(f"Input data shape: {df.shape}")
    print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

    # Aggregate by date (daily sum of TotalPrice)
    print("Aggregating sales by date...")
    df['Date'] = df['InvoiceDate'].dt.date
    daily_sales = df.groupby('Date')['TotalPrice'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']  # Prophet format
    print(f"Daily sales aggregated. Shape: {daily_sales.shape}")
    print(f"Sample daily sales:\n{daily_sales.head()}")

    # Train Prophet
    print("Training Prophet model...")
    model = Prophet()
    model.fit(daily_sales)
    print("Prophet model trained successfully.")

    # Predict future (e.g., next 30 days)
    print("Making future predictions for 30 days...")
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    print(f"Forecast generated. Shape: {forecast.shape}")
    print(f"Forecast sample:\n{forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)}")

    # Plot
    print("Creating demand forecasting plot...")
    fig = model.plot(forecast)
    plt.title('Demand Forecasting')
    plt.xlabel('Date')
    plt.ylabel('Predicted Sales')
    plt.savefig('images/demand_forecast.png')
    plt.close()
    print("Plot saved as images/demand_forecast.png")

    print("Time series analysis completed.")
    return forecast

if __name__ == "__main__":
    df = pd.read_csv('Dataset/online_retail.csv', encoding='latin1')
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    time_series_analysis(df)
