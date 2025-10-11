import pandas as pd
import random
from load_data import load_data
from clean_data import clean_data
import speech_recognition as sr
import pyttsx3

class SimpleChatbot:
    def __init__(self):
        # Load and clean data
        df = load_data()
        self.df = clean_data(df)

        # Precompute insights
        self.top_products = self.df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(5).index.tolist()
        self.loyal_customers = self.df.groupby('CustomerID')['InvoiceNo'].nunique().sort_values(ascending=False).head(5).index.tolist()
        uk_df = self.df[self.df['Country'] == 'United Kingdom']
        self.uk_top_products = uk_df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(5).index.tolist()
        self.total_sales = self.df['TotalPrice'].sum()
        self.num_customers = self.df['CustomerID'].nunique()
        self.avg_order_value = self.df.groupby('InvoiceNo')['TotalPrice'].sum().mean()
        self.peak_hour = self.df.groupby('Hour')['TotalPrice'].sum().idxmax()
        self.top_co_bought = self.df.groupby(['InvoiceNo', 'Description']).size().reset_index(name='Count').groupby('Description')['Count'].sum().sort_values(ascending=False).head(5).index.tolist()
        self.num_transactions = self.df['InvoiceNo'].nunique()

        self.responses = {
            "hello": ["Hi there!", "Hello!", "Greetings!"],
            "how_are_you": ["I'm doing well, thank you!", "I'm fine, how about you?"],
            "thank_you": ["You're welcome!", "Glad to help!"],
            "what_can_you_do": ["I can tell you about top products, loyal customers, forecasts, customer segments, and more. Just ask!"],
            "what_is_your_name": ["I am RetailCast, your retail analytics assistant."],
            "tell_me_about_project": ["RetailCast is an AI-powered retail analytics project using machine learning for insights like demand forecasting and customer segmentation."],
            "product": ["We have many products. What are you looking for?", "Tell me about a product."],
            "recommend": ["Based on your history, I recommend...", "Popular items include..."],
            "bye": ["Goodbye!", "See you later!"],
            "default": ["I'm sorry, I didn't understand.", "Can you rephrase?"],
            "top_products": [f"Top products by sales: {', '.join(self.top_products)}"],
            "loyal_customers": [f"Loyal customers (by frequency): {', '.join(map(str, self.loyal_customers))}"],
            "uk_frequent": [f"Frequently bought in the United Kingdom: {', '.join(self.uk_top_products)}"],
            "forecast": ["Demand forecast: Check the time series analysis for predictions."],
            "segments": ["Customer segments: High-value, Loyal, At-risk, New. See classification results."],
            "recommendations": ["Personalized recommendations: Based on deep learning, top suggestions include..."],
            "total_sales": [f"Total sales: {self.total_sales:.2f}"],
            "num_customers": [f"Number of customers: {self.num_customers}"],
            "avg_order": [f"Average order value: {self.avg_order_value:.2f}"],
            "peak_hour": [f"Peak hour for sales: {self.peak_hour}"],
            "top_co_bought": [f"Top co-bought products: {', '.join(self.top_co_bought)}"],
            "num_transactions": [f"Number of transactions: {self.num_transactions}"]
        }

    def get_customer_buys(self, customer_id):
        try:
            customer_id = float(customer_id)
            customer_data = self.df[self.df['CustomerID'] == customer_id]
            if customer_data.empty:
                return "Customer not found."
            top_buys = customer_data['Description'].value_counts().head(5).index.tolist()
            return f"Customer {customer_id} usually buys: {', '.join(top_buys)}"
        except ValueError:
            return "Invalid customer ID."

    def get_response(self, user_input):
        user_input = user_input.lower()
        if "hello" in user_input or "hi" in user_input:
            return random.choice(self.responses["hello"])
        elif "how are you" in user_input:
            return random.choice(self.responses["how_are_you"])
        elif "thank you" in user_input or "thanks" in user_input:
            return random.choice(self.responses["thank_you"])
        elif "what can you do" in user_input or "help" in user_input:
            return self.responses["what_can_you_do"][0]
        elif "what is your name" in user_input or "who are you" in user_input:
            return self.responses["what_is_your_name"][0]
        elif "tell me about" in user_input and "project" in user_input:
            return self.responses["tell_me_about_project"][0]
        elif "total" in user_input and "sales" in user_input:
            return self.responses["total_sales"][0]
        elif ("how many" in user_input or "number" in user_input) and "customer" in user_input:
            return self.responses["num_customers"][0]
        elif "average" in user_input and "order" in user_input:
            return self.responses["avg_order"][0]
        elif "peak" in user_input and "hour" in user_input:
            return self.responses["peak_hour"][0]
        elif "co" in user_input and "bought" in user_input:
            return self.responses["top_co_bought"][0]
        elif ("how many" in user_input or "number" in user_input) and "transaction" in user_input:
            return self.responses["num_transactions"][0]
        elif "product" in user_input and not ("top" in user_input or "frequent" in user_input):
            return random.choice(self.responses["product"])
        elif "recommend" in user_input and not ("personal" in user_input):
            return random.choice(self.responses["recommend"])
        elif "bye" in user_input:
            return random.choice(self.responses["bye"])
        elif "top" in user_input and "product" in user_input:
            return self.responses["top_products"][0]
        elif ("loyal" in user_input or "loyl" in user_input) and "customer" in user_input:
            return self.responses["loyal_customers"][0]
        elif ("frequent" in user_input or "frequestly" in user_input) and ("united kingdom" in user_input or "uk" in user_input or "inn" in user_input):
            return self.responses["uk_frequent"][0]
        elif "customer" in user_input and ("buy" in user_input or "usully" in user_input or "usually" in user_input):
            # Extract customer ID, assume format like "customer 12345"
            words = user_input.split()
            for i, word in enumerate(words):
                if word == "customer" and i+1 < len(words):
                    customer_id = words[i+1].strip('?[]')
                    return self.get_customer_buys(customer_id)
            return "Please specify a customer ID, e.g., 'What does customer 12345 usually buy?'"
        elif "forecast" in user_input or "demand" in user_input:
            return self.responses["forecast"][0]
        elif "segment" in user_input:
            return self.responses["segments"][0]
        elif "recommendation" in user_input or "personal" in user_input:
            return self.responses["recommendations"][0]
        else:
            return random.choice(self.responses["default"])

def run_chatbot():
    """
    Run a voice-based chatbot with speech recognition and synthesis.
    """
    bot = SimpleChatbot()
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Chatbot: Hello! How can I help you today?")
    engine.say("Hello! How can I help you today?")
    engine.runAndWait()

    while True:
        with microphone as source:
            print("Listening...")
            audio = recognizer.listen(source)

        try:
            user_input = recognizer.recognize_google(audio).lower()
            print(f"You said: {user_input}")
        except sr.UnknownValueError:
            user_input = ""
            print("No speech detected, listening again...")
            continue
        except sr.RequestError:
            user_input = ""
            print("Sorry, my speech service is down.")
            engine.say("Sorry, my speech service is down.")
            engine.runAndWait()
            continue

        if "quit" in user_input:
            print("Chatbot: Goodbye!")
            engine.say("Goodbye!")
            engine.runAndWait()
            break

        response = bot.get_response(user_input)
        print(f"Chatbot: {response}")
        engine.say(response)
        engine.runAndWait()

def run_text_chatbot():
    """
    Run a text-based chatbot.
    """
    bot = SimpleChatbot()
    print("Chatbot: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Chatbot: Goodbye!")
            break
        response = bot.get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    mode = input("Choose mode: 1 for voice, 2 for text: ")
    if mode == '1':
        run_chatbot()
    elif mode == '2':
        run_text_chatbot()
    else:
        print("Invalid choice")
