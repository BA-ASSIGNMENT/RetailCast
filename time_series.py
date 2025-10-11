import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def time_series_analysis(df):
    """
    Aggregate sales by date, train Prophet model, predict future demand.
    """
    # Aggregate by date (daily sum of TotalPrice)
    df['Date'] = df['InvoiceDate'].dt.date
    daily_sales = df.groupby('Date')['TotalPrice'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']  # Prophet format

    # Train Prophet
    model = Prophet()
    model.fit(daily_sales)

    # Predict future (e.g., next 30 days)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Plot
    fig = model.plot(forecast)
    plt.title('Demand Forecasting')
    plt.savefig('images/demand_forecast.png')
    plt.close()

    return forecast
