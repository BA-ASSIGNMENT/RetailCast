from load_data import load_data
from clean_data import clean_data
from time_series import time_series_analysis
from classification import customer_segmentation
from deep_learning import deep_learning_analysis
from nlp import nlp_analysis
from dashboard import create_dashboard

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_subsection(title):
    """Print a formatted subsection."""
    print(f"\n--- {title} ---")

def main():
    """
    Run all steps of the analysis with enhanced output formatting.
    """
    
    print("  COMPREHENSIVE DATA ANALYSIS PIPELINE")
    
    # Step 1: Load Data
    print_section_header("DATA LOADING")
    print("Loading data from source...")
    df = load_data()
    print(f"✓ Successfully loaded {len(df):,} records")
    print(f"  Columns: {', '.join(df.columns.tolist())}")

    # Step 2: Clean Data
    print_section_header("DATA CLEANING")
    print("Applying data cleaning procedures...")
    original_count = len(df)
    df = clean_data(df)
    print(f"✓ Data cleaning completed")
    print(f"  Records retained: {len(df):,} / {original_count:,}")
    if len(df) < original_count:
        print(f"  Removed: {original_count - len(df):,} records ({100*(original_count-len(df))/original_count:.1f}%)")

    # Step 3: Time Series Analysis
    print_section_header("TIME SERIES FORECASTING")
    print("Analyzing temporal patterns and generating forecast...")
    forecast = time_series_analysis(df)
    print("✓ Time series analysis completed")
    
    print_subsection("Next 5-Day Forecast")
    forecast_display = forecast[['ds', 'yhat']].tail(5)
    for idx, row in forecast_display.iterrows():
        print(f"  {row['ds']}: ${row['yhat']:,.2f}")

    # Step 4: Customer Segmentation
    print_section_header("CUSTOMER SEGMENTATION")
    print("Performing RFM analysis and clustering customers...")
    rfm = customer_segmentation(df)
    print("✓ Customer segmentation completed")
    
    print_subsection("Customer Segment Distribution")
    segment_counts = rfm['Segment'].value_counts()
    total_customers = len(rfm)
    for segment, count in segment_counts.items():
        percentage = 100 * count / total_customers
        bar = "█" * int(percentage / 2)
        print(f"  {segment:20s}: {count:5d} ({percentage:5.1f}%) {bar}")

    # Step 5: Deep Learning Analysis
    print_section_header("PRODUCT RECOMMENDATION ENGINE")
    print("Training neural network for product recommendations...")
    customer_product, matrix_reduced, recommended_products, co_bought_summary = deep_learning_analysis(df)
    print("✓ Deep learning analysis completed")
    
    print_subsection("Top Product Recommendations")
    for i, product in enumerate(list(recommended_products)[:10], 1):
        print(f"  {i:2d}. {product}")
    
    print_subsection("Frequently Co-Purchased Products")
    if not co_bought_summary.empty:
        for (product1, product2), count in co_bought_summary.head(5).items():
            print(f"  • {product1} + {product2}")
            print(f"    Co-purchase count: {count}")

    # Step 6: NLP Sentiment Analysis
    print_section_header("SENTIMENT ANALYSIS")
    print("Analyzing customer reviews with NLP models...")
    acc = nlp_analysis()
    print("✓ NLP analysis completed")
    print(f"\n  Model Performance:")
    print(f"  Sentiment Classification Accuracy: {acc:.2%}")
    
    quality_rating = "Excellent" if acc > 0.85 else "Good" if acc > 0.75 else "Fair"
    print(f"  Quality Rating: {quality_rating}")

    # Step 7: Dashboard Creation
    print_section_header("DASHBOARD GENERATION")
    print("Compiling visualizations and creating interactive dashboard...")
    create_dashboard()
    print("✓ Dashboard created successfully")
    print("  Access your dashboard to explore interactive visualizations")

    # Completion Summary
    print("\n" + "✓ " * 35)
    print("  ALL ANALYSIS STEPS COMPLETED SUCCESSFULLY!")
    print("✓ " * 35 + "\n")
    
    print("Summary of Outputs:")
    print(f"  • {len(df):,} records processed")
    print(f"  • {len(segment_counts)} customer segments identified")
    print(f"  • {len(recommended_products)} product recommendations generated")
    print(f"  • Sentiment model accuracy: {acc:.2%}")
    print("\nNext Steps:")
    print("  1. Review the generated dashboard for insights")
    print("  2. Examine customer segments for targeted marketing")
    print("  3. Implement product recommendation system")
    print("  4. Monitor forecast accuracy over time\n")

if __name__ == "__main__":
    main()