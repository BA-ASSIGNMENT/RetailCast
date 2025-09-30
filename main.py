from data_loader import load_and_clean_data
from time_series import forecast_sales
from customer_segmentation import (perform_rfm_analysis, perform_clustering, 
                                 analyze_segments, plot_segmentation, get_segmentation_summary)
from deep_learning import (build_customer_product_matrix, collaborative_filtering_analysis,
                         build_autoencoder, get_personalized_recommendations,
                         plot_deep_learning_results, get_deep_learning_summary)
from nlp_sentiment import (generate_synthetic_reviews, train_sentiment_models,
                          analyze_product_sentiment, integrate_sentiment_with_forecast,
                          plot_sentiment_analysis, get_nlp_summary)
from chatbot_speech import run_chatbot_interface, demonstrate_voice_capabilities
from utils import setup_plotting, print_section_header, print_progress
from config import *

def main():
    """Main orchestration function"""
    setup_plotting()
    
    # Step 1: Load and clean data
    print_progress(1, 6, "Loading and cleaning data")
    df = load_and_clean_data()
    
    # Step 2: Time series forecasting
    print_progress(2, 6, "Time series analysis and forecasting")
    forecast, model_fit = forecast_sales(df)
    
    # Step 3: Customer segmentation
    print_progress(3, 6, "Customer segmentation and RFM analysis")
    rfm = perform_rfm_analysis(df)
    rfm, optimal_k = perform_clustering(rfm)
    cluster_summary = analyze_segments(rfm)
    plot_segmentation(rfm)
    segmentation_stats = get_segmentation_summary(rfm, optimal_k)
    
    # Step 4: Deep learning analysis
    print_progress(4, 6, "Deep learning purchase pattern analysis")
    customer_product_matrix = build_customer_product_matrix(df)
    reconstructed_df, product_similarity_df, svd, product_names = collaborative_filtering_analysis(customer_product_matrix, df)
    autoencoder, encoder, history, matrix_normalized, mse, threshold, anomalous_customers = build_autoencoder(customer_product_matrix)
    get_personalized_recommendations(reconstructed_df, customer_product_matrix, product_names)
    plot_deep_learning_results(history, mse, threshold)
    dl_stats = get_deep_learning_summary(customer_product_matrix, anomalous_customers)
    
    # Step 5: NLP sentiment analysis
    print_progress(5, 6, "NLP sentiment analysis")
    reviews_df, products_with_names = generate_synthetic_reviews(df)
    lr_model, vectorizer, nb_accuracy, lr_accuracy, X_test, y_test, lr_pred, nb_pred = train_sentiment_models(reviews_df)
    product_sentiment = analyze_product_sentiment(reviews_df, products_with_names, lr_model, vectorizer)
    product_sales_sentiment = integrate_sentiment_with_forecast(df, product_sentiment, products_with_names)
    plot_sentiment_analysis(y_test, lr_pred, reviews_df, product_sentiment, nb_accuracy, lr_accuracy)
    nlp_stats = get_nlp_summary(reviews_df, lr_accuracy)
    
    # Step 6: Chatbot & Speech Recognition
    print_progress(6, 6, "AI Chatbot & Speech Recognition")
    
    # Test speech capabilities
    demonstrate_voice_capabilities()
    
    # Run interactive chatbot with ALL analyzed real data
    print("\n🚀 Launching Chatbot with Real Data...")
    run_chatbot_interface(df, rfm, forecast)
    
    # Final comprehensive summary
    print_final_summary(df, forecast, segmentation_stats, dl_stats, nlp_stats)

def print_final_summary(df, forecast, segmentation_stats, dl_stats, nlp_stats):
    """Print comprehensive analysis summary"""
    print_section_header("COMPREHENSIVE ANALYSIS COMPLETE")
    
    print("\n📊 ANALYSIS SUMMARY:")
    print("-" * 60)
    print(f"✓ Step 1: Data Cleaning - {len(df):,} transactions processed")
    print(f"✓ Step 2: Time Series Forecasting - {FORECAST_STEPS} days forecasted")
    print(f"✓ Step 3: Customer Segmentation - {segmentation_stats['optimal_clusters']} segments identified")
    print(f"✓ Step 4: Deep Learning - {dl_stats['total_customers_analyzed']} customers, {dl_stats['total_products_analyzed']} products analyzed")
    print(f"✓ Step 5: NLP Sentiment Analysis - {nlp_stats['total_reviews_analyzed']} reviews analyzed")
    print(f"✓ Step 6: AI Chatbot - Interactive analytics assistant with speech recognition")

    print("\n🔑 KEY INSIGHTS:")
    print("-" * 60)
    if forecast is not None:
        forecast_total = forecast.sum()
        print(f"1. Revenue Forecast: ${forecast_total:,.2f} expected in next {FORECAST_STEPS} days")
    print(f"2. Customer Segments: {segmentation_stats['total_customers']} customers across {segmentation_stats['optimal_clusters']} behavioral groups")
    print(f"3. Product Patterns: {dl_stats['anomalous_customers']} customers with unusual buying behavior")
    print(f"4. Sentiment Analysis: {nlp_stats['sentiment_accuracy']:.1%} accuracy in predicting customer satisfaction")
    print(f"5. AI Assistant: Chatbot with voice commands for interactive analytics")

    print("\n💾 OUTPUT FILES:")
    print("-" * 60)
    print("✓ forecast_plot.png - Sales predictions")
    print("✓ customer_segmentation.png - Customer clusters")
    print("✓ deep_learning_analysis.png - Purchase patterns")
    print("✓ nlp_sentiment_analysis.png - Sentiment results")
    print("✓ chatbot_conversation_log.csv - Chat history")

    print("\n🎯 NEXT STEPS:")
    print("-" * 60)
    print("• Use the chatbot for ongoing analytics queries")
    print("• Enable voice commands for hands-free operation")
    print("• Integrate with real-time data streams")
    print("• Deploy as web application with voice interface")

    print("\n" + "=" * 60)
    print("All analyses completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()