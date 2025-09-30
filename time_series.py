import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
from config import *

def prepare_daily_sales(df):
    """Prepare daily sales data for time series analysis"""
    daily_sales = df.groupby('Date').agg({
        'Quantity': 'sum',
        'TotalPrice': 'sum',
        'InvoiceNo': 'nunique'
    }).reset_index()

    daily_sales.columns = ['Date', 'TotalQuantity', 'TotalSales', 'NumTransactions']
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    daily_sales = daily_sales.sort_values('Date')
    
    return daily_sales

def forecast_sales(df):
    """Perform time series forecasting using SARIMA"""
    print("\n" + "=" * 60)
    print("STEP 2: TIME SERIES ANALYSIS - DEMAND FORECASTING")
    print("=" * 60)
    
    daily_sales = prepare_daily_sales(df)
    print(f"\nDaily sales data prepared: {len(daily_sales)} days")
    print(f"Average daily sales: ${daily_sales['TotalSales'].mean():,.2f}")

    # Create time series
    ts = daily_sales.set_index('Date')['TotalSales']

    # Fill missing dates with 0 (if any gaps exist)
    date_range = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq='D')
    ts = ts.reindex(date_range, fill_value=0)

    print(f"Time series prepared with {len(ts)} data points")

    # Fit SARIMA model
    print("\nFitting SARIMA model (this may take a moment)...")
    try:
        # Using weekly seasonality (s=7) as it's more stable for daily data
        model = SARIMAX(ts, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER)
        model_fit = model.fit(disp=False, maxiter=200)
        
        print("SARIMA model fitted successfully!")
        print(f"AIC: {model_fit.aic:.2f}")
        
        # Forecast for next 90 days
        forecast = model_fit.forecast(steps=FORECAST_STEPS)
        
        # Calculate forecast statistics
        forecast_mean = forecast.mean()
        forecast_total = forecast.sum()
        
        print(f"\nForecast Summary (next {FORECAST_STEPS} days):")
        print(f"Average daily sales: ${forecast_mean:,.2f}")
        print(f"Total forecasted sales: ${forecast_total:,.2f}")
        
        # Plot forecast
        plot_forecast(ts, forecast, FORECAST_STEPS)
        
        return forecast, model_fit
        
    except Exception as e:
        print(f"Error in SARIMA modeling: {e}")
        print("Using simple moving average forecast as fallback...")
        return None, None

def plot_forecast(ts, forecast, forecast_steps):
    """Plot historical data and forecast"""
    plt.figure(figsize=FIGURE_SIZE)
    
    # Show only last 120 days for cleaner visualization
    recent_days = 120
    plt.plot(ts.index[-recent_days:], ts[-recent_days:], 
             label=f"Historical Sales (Last {recent_days} days)", linewidth=1.5)
    
    forecast_dates = pd.date_range(ts.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
    plt.plot(forecast_dates, forecast, label=f"Forecast ({forecast_steps} days)", color="red", linewidth=1.5)
    
    plt.title("Sales Forecast - Demand Forecasting", fontsize=12, fontweight='bold')
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Daily Sales ($)", fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('forecast_plot.png', dpi=DPI, bbox_inches='tight')
    plt.show()