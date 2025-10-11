import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data
from clean_data import clean_data
from time_series import time_series_analysis
from classification import customer_segmentation
from deep_learning import deep_learning_analysis
from nlp import nlp_analysis

def create_dashboard():
    """
    Create dashboard with plots for:
    - Predicted demand
    - Customer segments
    - Product bundles & recommendations
    - Suggested stock levels
    - Insights for staff scheduling and product placement
    """
    # Load and clean data
    df = load_data()
    df = clean_data(df)

    # Run analyses
    forecast = time_series_analysis(df)
    rfm = customer_segmentation(df)
    customer_product, matrix_reduced, recommended_products, co_bought_summary = deep_learning_analysis(df)
    acc = nlp_analysis()

    # Calculate insights
    future_forecast = forecast[['ds', 'yhat']].tail(30)
    segment_counts = rfm['Segment'].value_counts()
    top_products = df.groupby('StockCode')['TotalPrice'].sum().sort_values(ascending=False).head(5).index
    stock_levels = {}
    for product in top_products:
        product_sales = df[df['StockCode'] == product]['TotalPrice'].sum()
        avg_daily = product_sales / len(df['Date'].unique())
        suggested = avg_daily * 30 + np.std(df[df['StockCode'] == product]['Quantity'])  # Simple heuristic
        stock_levels[product] = suggested
    hourly_sales = df.groupby('Hour')['TotalPrice'].sum()
    top_prod_sales = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(5)

    # Create figures
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.subplots_adjust(bottom=0.2)

    # 1. Predicted demand (last 30 days forecast)
    axs[0, 0].plot(future_forecast['ds'], future_forecast['yhat'])
    axs[0, 0].set_title('Predicted Demand (Next 30 Days)')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Total Sales')
    axs[0, 0].tick_params(axis='x', rotation=90)

    # 2. Customer segments
    axs[0, 1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
    axs[0, 1].set_title('Customer Segments')

    # 3. Top co-bought pairs (product bundles)
    co_bought_summary.plot(kind='bar', ax=axs[0, 2])
    axs[0, 2].set_title('Top Product Bundles (Co-bought)')
    axs[0, 2].tick_params(axis='x', rotation=45)

    # 4. Suggested stock levels (example: avg + forecast for top products)
    products = list(stock_levels.keys())
    levels = list(stock_levels.values())
    axs[1, 0].bar(products, levels)
    axs[1, 0].set_title('Suggested Stock Levels (Top Products)')
    axs[1, 0].tick_params(axis='x', rotation=45)

    # 5. Staff scheduling insights (hourly sales)
    axs[1, 1].plot(hourly_sales.index, hourly_sales.values)
    axs[1, 1].set_title('Hourly Sales for Staff Scheduling')
    axs[1, 1].set_xlabel('Hour')
    axs[1, 1].set_ylabel('Total Sales')

    # 6. Product placement (top products by sales)
    axs[1, 2].bar(range(len(top_prod_sales)), top_prod_sales.values)
    axs[1, 2].set_title('Top Products for Placement')
    axs[1, 2].set_xlabel('Products')
    axs[1, 2].set_ylabel('Total Sales')
    axs[1, 2].set_xticks(range(len(top_prod_sales)))
    axs[1, 2].set_xticklabels(top_prod_sales.index, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('images/dashboard.png')
    plt.close()

    # Print insights
    print("Dashboard Insights:")
    print(f"1. Predicted Demand: Average forecast for next 30 days: {future_forecast['yhat'].mean():.2f}")
    print(f"2. Customer Segments: {segment_counts.to_dict()}")
    print(f"3. Product Bundles: {co_bought_summary.head().to_dict()}")
    print(f"4. Recommendations: {list(recommended_products)}")
    print(f"5. Suggested Stock Levels: {stock_levels}")
    print(f"6. Peak Hours for Staff: {hourly_sales.idxmax()}:00 - {hourly_sales.idxmax()+2}:00")
    print(f"7. Top Products for Placement: {top_prod_sales.head().index.tolist()}")
    print(f"8. NLP Sentiment Accuracy: {acc:.2f}")

if _name_ == "_main_":
    create_dashboard()