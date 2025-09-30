import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import json
import time
import threading
import queue
from config import *

# Initialize availability flags at module level
ML_AVAILABLE = False
SPEECH_AVAILABLE = False

# Try to import ML dependencies (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    ML_AVAILABLE = True
    print("✓ Machine learning libraries loaded")
except ImportError as e:
    print(f"Note: Some ML dependencies not available. Using rule-based chatbot. Error: {e}")
    ML_AVAILABLE = False

# Try to import speech recognition (optional)
try:
    import speech_recognition as sr
    import pyttsx3
    SPEECH_AVAILABLE = True
    print("✓ Speech recognition libraries loaded")
except ImportError as e:
    print(f"Note: Speech recognition libraries not available. Error: {e}")
    SPEECH_AVAILABLE = False

class RetailChatbot:
    """Intelligent chatbot for retail analytics queries"""
    
    def __init__(self, df, rfm_data, forecast_data):
        self.df = df
        self.rfm_data = rfm_data
        self.forecast_data = forecast_data
        self.conversation_history = []
        self.speech_available = SPEECH_AVAILABLE
        
        # Initialize NLP components if available
        self.sentiment_analyzer = None
        if ML_AVAILABLE:
            try:
                # Use a smaller, faster model for sentiment
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
                )
                print("✓ Pretrained NLP model loaded")
            except Exception as e:
                self.sentiment_analyzer = None
                print(f"Note: Could not load pretrained model, using rule-based approach. Error: {e}")
        
        # Initialize speech if available
        self.speech_recognizer = None
        self.tts_engine = None
        
        if self.speech_available:
            try:
                self.speech_recognizer = sr.Recognizer()
                self.tts_engine = pyttsx3.init()
                # Configure voice settings
                voices = self.tts_engine.getProperty('voices')
                if len(voices) > 0:
                    self.tts_engine.setProperty('voice', voices[0].id)
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                print("✓ Speech recognition initialized")
            except Exception as e:
                print(f"Note: Speech recognition initialization failed: {e}")
                self.speech_available = False
                self.speech_recognizer = None
                self.tts_engine = None
        
        # Define conversation patterns
        self.patterns = {
            'greeting': [
                r'hello', r'hi', r'hey', r'greetings', r'good morning', r'good afternoon'
            ],
            'sales_forecast': [
                r'forecast', r'prediction', r'future sales', r'next month', r'revenue prediction',
                r'how much will we sell', r'sales outlook', r'sales forecast'
            ],
            'customer_segments': [
                r'customer segments', r'rfm', r'customer groups', r'best customers',
                r'customer analysis', r'segmentation', r'customer segments'
            ],
            'product_analysis': [
                r'products', r'popular products', r'best selling', r'product performance',
                r'top products', r'what sells well', r'best products', r'our products'
            ],
            'sentiment': [
                r'sentiment', r'reviews', r'feedback', r'customer opinion', r'ratings',
                r'customer feedback'
            ],
            'help': [
                r'help', r'what can you do', r'capabilities', r'functions', r'assistance'
            ],
            'exit': [
                r'bye', r'exit', r'quit', r'goodbye', r'see you', r'stop'
            ]
        }
        
        print("Retail Analytics Chatbot initialized!")
    
    def classify_intent(self, message):
        """Classify user intent using pattern matching or ML"""
        message_lower = message.lower()
        
        # Check patterns
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent
        
        # If ML available and no pattern matched, use ML classification
        if ML_AVAILABLE and self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(message)
                # Use sentiment as fallback intent classifier
                if result[0]['label'] == 'POSITIVE' and result[0]['score'] > 0.8:
                    return 'positive_feedback'
                elif result[0]['label'] == 'NEGATIVE' and result[0]['score'] > 0.8:
                    return 'negative_feedback'
            except Exception as e:
                print(f"ML classification failed: {e}")
        
        return 'unknown'
    
    def generate_response(self, message):
        """Generate response based on user intent"""
        intent = self.classify_intent(message)
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'user_message': message,
            'intent': intent,
            'bot_response': ''
        })
        
        response = ""
        
        if intent == 'greeting':
            response = self._handle_greeting()
        elif intent == 'sales_forecast':
            response = self._handle_sales_forecast()
        elif intent == 'customer_segments':
            response = self._handle_customer_segments()
        elif intent == 'product_analysis':
            response = self._handle_product_analysis()
        elif intent == 'sentiment':
            response = self._handle_sentiment()
        elif intent == 'help':
            response = self._handle_help()
        elif intent == 'exit':
            response = self._handle_exit()
        elif intent == 'positive_feedback':
            response = self._handle_positive_feedback()
        elif intent == 'negative_feedback':
            response = self._handle_negative_feedback()
        else:
            response = self._handle_unknown()
        
        # Update conversation history
        self.conversation_history[-1]['bot_response'] = response
        
        return response
    
    def _handle_greeting(self):
        greetings = [
            "Hello! I'm your Retail Analytics Assistant. How can I help you today?",
            "Hi there! Ready to explore your retail data insights?",
            "Welcome! I can help you with sales forecasts, customer segments, and more.",
            "Greetings! I'm here to provide insights from your retail data."
        ]
        return np.random.choice(greetings)
    
    def _handle_sales_forecast(self):
        if self.forecast_data is not None:
            forecast_total = self.forecast_data.sum()
            forecast_mean = self.forecast_data.mean()
            
            response = f"""📊 **Sales Forecast Analysis:**
            
• **Next 90 Days Forecast:** ${forecast_total:,.2f}
• **Average Daily Sales:** ${forecast_mean:,.2f}
• **Forecast Period:** {FORECAST_STEPS} days

The forecast shows strong potential for continued growth. Would you like more detailed analysis?"""
        else:
            response = "I don't have current forecast data available. Please run the time series analysis first."
        
        return response
    
    def _handle_customer_segments(self):
        if self.rfm_data is not None:
            total_customers = len(self.rfm_data)
            avg_value = self.rfm_data['Monetary'].mean()
            high_value_count = len(self.rfm_data[self.rfm_data['Monetary'] > self.rfm_data['Monetary'].median()])
            
            response = f"""👥 **Customer Segmentation Summary:**
            
• **Total Customers Analyzed:** {total_customers:,}
• **Average Customer Value:** ${avg_value:,.2f}
• **High-Value Customers:** {high_value_count:,}
• **Customer Segments:** {self.rfm_data['Cluster'].nunique()} distinct groups

Key segments include High-Value Loyal customers, Frequent Shoppers, and At-Risk customers."""
        else:
            response = "Customer segmentation data not available. Please run the RFM analysis first."
        
        return response
    
    def _handle_product_analysis(self):
        """Analyze top products with full descriptions from real data"""
        try:
            # Use real data from the database
            if self.df is not None and not self.df.empty:
                # Get top products by quantity sold
                top_products = self.df.groupby(['StockCode', 'Description'])['Quantity'].sum().reset_index()
                top_products = top_products.sort_values('Quantity', ascending=False).head(10)
                
                # Get top products by revenue
                top_revenue = self.df.groupby(['StockCode', 'Description'])['TotalPrice'].sum().reset_index()
                top_revenue = top_revenue.sort_values('TotalPrice', ascending=False).head(10)
                
                response = "🛍️ **Top Performing Products:**\n\n"
                response += "**Best Selling (by quantity):**\n"
                for i, row in enumerate(top_products.head(5).itertuples(), 1):
                    response += f"{i}. {row.StockCode} - {row.Description}\n"
                    response += f"   Units Sold: {row.Quantity:,}\n\n"
                
                response += "**Highest Revenue (by sales):**\n"
                for i, row in enumerate(top_revenue.head(5).itertuples(), 1):
                    response += f"{i}. {row.StockCode} - {row.Description}\n"
                    response += f"   Revenue: ${row.TotalPrice:,.2f}\n\n"
                
                # Add some insights
                total_products = self.df['StockCode'].nunique()
                total_quantity = self.df['Quantity'].sum()
                avg_price = self.df['UnitPrice'].mean()
                
                response += f"**Overall Statistics:**\n"
                response += f"• Total unique products: {total_products:,}\n"
                response += f"• Total units sold: {total_quantity:,}\n"
                response += f"• Average product price: ${avg_price:.2f}\n"
                response += f"• Date range: {self.df['InvoiceDate'].min().strftime('%Y-%m-%d')} to {self.df['InvoiceDate'].max().strftime('%Y-%m-%d')}\n"
                
            else:
                response = "No product data available. Please load the retail dataset first."
        
        except Exception as e:
            response = f"Error analyzing products: {str(e)}"
            print(f"Product analysis error: {e}")
        
        return response
    
    def _handle_sentiment(self):
        response = """😊 **Customer Sentiment Insights:**
        
Based on our analysis:
• **Review Sentiment Accuracy:** >85% (Logistic Regression)
• **Key Positive Drivers:** Product quality, delivery speed
• **Improvement Areas:** Some product quality concerns

We analyze customer feedback to identify trends and improvement opportunities."""
        return response
    
    def _handle_help(self):
        response = """🤖 **Retail Analytics Chatbot - Capabilities:**
        
I can help you with:

📈 **Sales & Forecasting**
• Sales predictions and revenue forecasts
• Demand patterns and seasonal trends

👥 **Customer Analysis**  
• Customer segmentation (RFM analysis)
• High-value customer identification
• Customer behavior patterns

🛍️ **Product Insights**
• Top-performing products with full descriptions
• Product recommendations
• Inventory optimization

😊 **Sentiment Analysis**
• Customer feedback analysis
• Review sentiment tracking
• Product satisfaction metrics

💬 **How to interact:**
• Type your questions naturally
• Use voice commands (speak anytime)
• Ask about specific metrics or insights

Try asking: 'What's our sales forecast?' or 'Show me top products'"""
        return response
    
    def _handle_exit(self):
        farewells = [
            "Goodbye! Feel free to return for more retail insights.",
            "Thanks for chatting! Come back anytime for analytics help.",
            "See you later! Remember, data-driven decisions drive success.",
            "Farewell! Keep optimizing your retail strategy."
        ]
        return np.random.choice(farewells)
    
    def _handle_positive_feedback(self):
        responses = [
            "Great to hear! Is there anything specific you'd like to know about your retail data?",
            "Wonderful! I'm glad I could help. What would you like to explore next?",
            "Excellent! Let me know what other insights you're interested in.",
            "That's fantastic! How else can I assist with your retail analytics?"
        ]
        return np.random.choice(responses)
    
    def _handle_negative_feedback(self):
        responses = [
            "I apologize if my response wasn't helpful. Could you rephrase your question?",
            "I'm sorry to hear that. Let me try to provide better information. What specifically are you looking for?",
            "My apologies. Let me help you get the right information. Could you clarify your question?",
            "I understand your frustration. Let me assist you better - what aspect of the data interests you?"
        ]
        return np.random.choice(responses)
    
    def _handle_unknown(self):
        responses = [
            "I'm not sure I understand. Could you rephrase your question about retail analytics?",
            "I specialize in retail data analysis. Try asking about sales, customers, or products.",
            "Let me help you with retail insights. You can ask about forecasts, customer segments, or product performance.",
            "I'm designed to answer retail analytics questions. Try 'sales forecast' or 'top products' for starters."
        ]
        return np.random.choice(responses)
    
    def get_conversation_stats(self):
        """Get conversation statistics"""
        if not self.conversation_history:
            return "No conversations yet."
        
        total_messages = len(self.conversation_history)
        intents = [conv['intent'] for conv in self.conversation_history]
        intent_counts = pd.Series(intents).value_counts()
        
        stats = f"📊 **Conversation Statistics:**\n\n"
        stats += f"• Total Interactions: {total_messages}\n"
        stats += f"• Most Common Queries:\n"
        
        for intent, count in intent_counts.head(5).items():
            stats += f"  - {intent}: {count}\n"
        
        return stats

class SpeechInterface:
    """Speech recognition and text-to-speech interface"""
    
    def __init__(self):
        self.available = SPEECH_AVAILABLE
        self.recognizer = None
        self.tts_engine = None
        
        if not self.available:
            return
        
        try:
            self.recognizer = sr.Recognizer()
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            
            # Improve recognition by adjusting for ambient noise
            print("Calibrating microphone for ambient noise...")
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Microphone calibrated!")
            
        except Exception as e:
            print(f"Speech interface initialization failed: {e}")
            self.available = False
    
    def listen(self, timeout=8):
        """Listen for voice input with improved reliability"""
        if not self.available or self.recognizer is None:
            print("Speech recognition not available")
            return None
        
        try:
            with sr.Microphone() as source:
                print("🎤 Listening... (speak clearly now)")
                # Reduce background noise adjustment time for faster response
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            print("🔄 Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"📝 You said: {text}")
            return text.lower().strip()
        
        except sr.WaitTimeoutError:
            print("⏰ No speech detected")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand audio - please speak clearly")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition service error: {e}")
            return None
        except Exception as e:
            print(f"❌ Microphone error: {e}")
            return None
    
    def speak(self, text):
        """Convert text to speech with improved naturalness"""
        if not self.available or self.tts_engine is None:
            print(f"🤖 (TTS would say): {text}")
            return
        
        try:
            # Clean text for speech - more natural sounding
            clean_text = re.sub(r'[**•\-]', '', text)  # Remove markdown and bullets
            clean_text = re.sub(r'\n+', '. ', clean_text)  # Convert newlines to pauses
            clean_text = re.sub(r'\s+', ' ', clean_text)  # Remove extra spaces
            clean_text = re.sub(r'🛍️|📊|👥|😊|🤖|🎤|🔊|❌|⏰|🔄|📝', '', clean_text)  # Remove emojis
            
            # Add slight pauses for better comprehension
            clean_text = re.sub(r',', ', ', clean_text)
            clean_text = re.sub(r'\.', '. ', clean_text)
            
            # Truncate very long responses for speech
            if len(clean_text) > 250:
                # For product responses, create a summary
                if "top products" in text.lower() or "best selling" in text.lower():
                    clean_text = self._summarize_products_response(text)
                else:
                    clean_text = clean_text[:250] + "..."
            
            print(f"🔊 Speaking response...")
            self.tts_engine.say(clean_text)
            self.tts_engine.runAndWait()
        
        except Exception as e:
            print(f"❌ TTS error: {e}")
            print(f"🤖 (TTS failed, text was): {text}")
    
    def _summarize_products_response(self, text):
        """Create a concise summary of product data for speech"""
        # Extract key information for speech
        lines = text.split('\n')
        summary = "Here are your top products: "
        
        product_count = 0
        current_section = ""
        
        for line in lines:
            line = line.strip()
            
            if "Best Selling" in line:
                current_section = "by quantity"
                continue
            elif "Highest Revenue" in line:
                current_section = "by revenue" 
                continue
            elif "Overall Statistics" in line:
                break
                
            # Look for product lines (they start with numbers)
            if line and line[0].isdigit() and ' - ' in line:
                parts = line.split(' - ')
                if len(parts) >= 2:
                    product_desc = parts[1].strip()
                    # Clean up the description for speech
                    product_desc = re.sub(r'\s+', ' ', product_desc)
                    
                    if product_count < 2:  # Only mention top 2 products in speech
                        if current_section == "by quantity":
                            summary += f" {product_desc}, "
                        product_count += 1
            
            # Look for quantity/revenue numbers
            elif "Units Sold:" in line and product_count <= 2:
                quantity = line.split("Units Sold:")[1].strip()
                summary += f"with {quantity} units. "
            elif "Revenue:" in line and product_count <= 2:
                revenue = line.split("Revenue:")[1].strip()
                summary += f"generating {revenue} in sales. "
        
        # Add overall stats
        for line in lines:
            if "Total unique products:" in line:
                total_products = line.split("Total unique products:")[1].strip()
                summary += f"You have {total_products} unique products in total. "
                break
        
        summary += "Check the screen for complete details of all top products."
        
        return summary

def input_with_timeout(prompt, timeout=2):
    """Get input with timeout to allow quick switching between speech and text"""
    print(prompt, end='', flush=True)
    q = queue.Queue()
    
    def get_input():
        try:
            line = input()
            q.put(line)
        except:
            q.put(None)
    
    thread = threading.Thread(target=get_input)
    thread.daemon = True
    thread.start()
    
    try:
        result = q.get(timeout=timeout)
        return result
    except queue.Empty:
        raise TimeoutError

def run_chatbot_interface(df, rfm_data=None, forecast_data=None):
    """Run the interactive chatbot interface"""
    print("\n" + "=" * 60)
    print("STEP 6: AI CHATBOT & SPEECH RECOGNITION")
    print("=" * 60)
    
    # Initialize chatbot and speech
    chatbot = RetailChatbot(df, rfm_data, forecast_data)
    speech_interface = SpeechInterface()
    
    print("\n🤖 **Retail Analytics Chatbot Activated!**")
    
    # Show dataset info
    if df is not None and not df.empty:
        print(f"📁 Dataset loaded: {len(df):,} transactions, {df['CustomerID'].nunique():,} customers")
        print(f"📅 Date range: {df['InvoiceDate'].min().strftime('%Y-%m-%d')} to {df['InvoiceDate'].max().strftime('%Y-%m-%d')}")
        print(f"💰 Total revenue: ${df['TotalPrice'].sum():,.2f}")
    
    print("\n💬 **Interaction Options:**")
    print("• Type your questions")
    print("• Speak anytime - I'm always listening!")
    print("• Say 'text only' to disable speech")
    print("• Say 'exit' to end the conversation")
    print("-" * 50)
    
    # Initial greeting
    greeting = chatbot.generate_response("hello")
    print(f"🤖: {greeting}")
    
    if speech_interface.available:
        speech_interface.speak("Welcome to Retail Analytics Assistant! You can ask me about sales forecasts, customer segments, or top products. Speak clearly or type your questions.")
    
    # Conversation loop
    speech_enabled = speech_interface.available
    conversation_active = True
    consecutive_speech_failures = 0
    
    while conversation_active:
        try:
            # Always try to listen for speech if enabled
            user_input = None
            if speech_enabled:
                print(f"\n🎤 Listening... (speak now or press Enter to type)")
                user_input = speech_interface.listen(timeout=6)
                
                # Track consecutive failures to suggest switching to text
                if user_input is None:
                    consecutive_speech_failures += 1
                    if consecutive_speech_failures >= 2:
                        print("🤖: Having trouble with speech? You can type your question instead.")
                        if speech_interface.available:
                            speech_interface.speak("Having trouble with speech? You can type instead.")
                else:
                    consecutive_speech_failures = 0
            
            # If no speech detected or speech disabled, wait for text input
            if user_input is None:
                try:
                    # Use a short timeout for input to allow quick switching
                    user_input = input_with_timeout("\n👤 Type here (or just press Enter for speech): ", timeout=2)
                    if not user_input or user_input.isspace():
                        continue
                except TimeoutError:
                    continue
            
            # Check for mode switches
            if user_input.lower() in ['text only', 'text mode', 'no speech']:
                speech_enabled = False
                response = "Speech disabled. I'll only respond with text now."
                print(f"🤖: {response}")
                continue
            
            elif user_input.lower() in ['speech on', 'voice mode', 'enable speech']:
                if speech_interface.available:
                    speech_enabled = True
                    response = "Speech enabled! I'll listen for your voice commands."
                    print(f"🤖: {response}")
                    speech_interface.speak("Speech enabled")
                else:
                    response = "Speech recognition is not available on this system."
                    print(f"🤖: {response}")
                continue
            
            elif user_input.lower() in ['stats', 'statistics', 'conversation stats']:
                stats = chatbot.get_conversation_stats()
                print(f"🤖: {stats}")
                if speech_enabled and speech_interface.available:
                    speech_interface.speak("Here are your conversation statistics.")
                continue
            
            # Process user input
            response = chatbot.generate_response(user_input)
            print(f"🤖: {response}")
            
            # Speak response if speech is enabled
            if speech_enabled and speech_interface.available:
                speech_interface.speak(response)
            
            # Check for exit
            if chatbot.classify_intent(user_input) == 'exit':
                conversation_active = False
        
        except KeyboardInterrupt:
            print("\n\n🤖: Conversation ended by user.")
            conversation_active = False
        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}. Please try again."
            print(f"🤖: {error_msg}")
            if speech_enabled and speech_interface.available:
                speech_interface.speak("Sorry, I encountered an error. Please try again.")
    
    # Conversation summary
    print("\n" + "=" * 50)
    print("💾 **Conversation Summary**")
    print("=" * 50)
    print(chatbot.get_conversation_stats())
    
    # Save conversation log
    try:
        log_data = []
        for conv in chatbot.conversation_history:
            log_data.append({
                'timestamp': conv['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'user_message': conv['user_message'],
                'intent': conv['intent'],
                'bot_response': conv['bot_response'][:200] + '...' if len(conv['bot_response']) > 200 else conv['bot_response']
            })
        
        log_df = pd.DataFrame(log_data)
        log_df.to_csv('chatbot_conversation_log.csv', index=False)
        print("✓ Conversation log saved: 'chatbot_conversation_log.csv'")
    
    except Exception as e:
        print(f"Note: Could not save conversation log: {e}")
    
    print("\n" + "=" * 50)
    print("Chatbot session completed!")
    print("=" * 50)

def demonstrate_voice_capabilities():
    """Demonstrate speech recognition capabilities"""
    print("\n" + "=" * 50)
    print("🎤 SPEECH RECOGNITION DEMONSTRATION")
    print("=" * 50)
    
    speech_interface = SpeechInterface()
    
    if not speech_interface.available:
        print("Speech recognition not available on this system.")
        print("To enable voice features, install:")
        print("  pip install SpeechRecognition pyttsx3")
        print("  (Note: PyAudio may require additional system dependencies)")
        return
    
    print("Testing speech recognition...")
    print("Say something like 'What are our top products'")
    
    successful_attempts = 0
    for attempt in range(3):
        print(f"\nAttempt {attempt + 1}/3:")
        text = speech_interface.listen(timeout=8)
        
        if text:
            print(f"✅ Successfully recognized: '{text}'")
            successful_attempts += 1
            
            # Test TTS
            response = f"I heard you say: {text}. Speech recognition is working perfectly!"
            speech_interface.speak(response)
            
            if successful_attempts >= 2:
                break
        else:
            print("❌ No speech detected or recognition failed")
    
    if successful_attempts > 0:
        print(f"\n🎉 Speech recognition test passed! {successful_attempts}/3 attempts successful")
        speech_interface.speak("Speech recognition is ready for use!")
    else:
        print("\n⚠️ Speech recognition test failed. You can still use text input.")
    
    print("\nSpeech recognition test completed!")

if __name__ == "__main__":
    # Test the chatbot with real data
    print("Testing Retail Analytics Chatbot with real data...")
    
    try:
        from data_loader import load_and_clean_data
        print("Loading real dataset...")
        df = load_and_clean_data()
        
        # Run demo
        demonstrate_voice_capabilities()
        run_chatbot_interface(df, None, None)
    
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Please run main.py to analyze your data first, then the chatbot will have access to all insights.")
        
        # Create minimal sample data for basic testing
        sample_df = pd.DataFrame({
            'StockCode': ['A001', 'A002', 'A003'],
            'Description': ['Premium Wireless Headphones', 'Smart Fitness Tracker Watch', 'Portable Bluetooth Speaker'],
            'Quantity': [100, 150, 80],
            'TotalPrice': [1000, 1500, 800],
            'InvoiceDate': [datetime.now() - timedelta(days=i) for i in range(3)],
            'CustomerID': [123, 456, 789],
            'UnitPrice': [10, 10, 10]
        })
        
        demonstrate_voice_capabilities()
        run_chatbot_interface(sample_df, None, None)