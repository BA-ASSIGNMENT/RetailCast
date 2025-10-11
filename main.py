from load_data import load_data
from clean_data import clean_data
from time_series import time_series_analysis
from classification import customer_segmentation
from deep_learning import deep_learning_analysis
from nlp import nlp_analysis
from dashboard import create_dashboard

def main():
    """
    Run all steps of the analysis.
    """
    print("Starting data loading...")
    df = load_data()
    print("Data loaded.")

    print("Cleaning data...")
    df = clean_data(df)
    print("Data cleaned.")

    print("Performing time series analysis...")
    forecast = time_series_analysis(df)
    print("Time series analysis completed.")
    print("Forecast (next 5 days):", forecast[['ds', 'yhat']].tail(5).to_dict('records'))

    print("Performing customer segmentation...")
    rfm = customer_segmentation(df)
    print("Customer segmentation completed.")
    print("Customer segments:", rfm['Segment'].value_counts().to_dict())

    print("Performing deep learning analysis...")
    customer_product, matrix_reduced, recommended_products, co_bought_summary = deep_learning_analysis(df)
    print("Deep learning analysis completed.")
    print("Recommended products:", list(recommended_products))
    print("Top co-bought pairs:", co_bought_summary.head().to_dict())

    print("Performing NLP analysis...")
    acc = nlp_analysis()
    print("NLP analysis completed.")
    print("NLP sentiment accuracy:", acc)

    print("Creating dashboard...")
    create_dashboard()
    print("Dashboard created.")

    print("All steps completed successfully!")

if __name__ == "__main__":
    main()
