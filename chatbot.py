import random

class SimpleChatbot:
    def __init__(self):
        self.responses = {
            "hello": ["Hi there!", "Hello!", "Greetings!"],
            "product": ["We have many products. What are you looking for?", "Tell me about a product."],
            "recommend": ["Based on your history, I recommend...", "Popular items include..."],
            "bye": ["Goodbye!", "See you later!"],
            "default": ["I'm sorry, I didn't understand.", "Can you rephrase?"]
        }

    def get_response(self, user_input):
        user_input = user_input.lower()
        if "hello" in user_input or "hi" in user_input:
            return random.choice(self.responses["hello"])
        elif "product" in user_input:
            return random.choice(self.responses["product"])
        elif "recommend" in user_input:
            return random.choice(self.responses["recommend"])
        elif "bye" in user_input:
            return random.choice(self.responses["bye"])
        else:
            return random.choice(self.responses["default"])

def run_chatbot():
    """
    Run a simple text-based chatbot.
    Note: Speech recognition not implemented, as it requires external tools.
    """
    bot = SimpleChatbot()
    print("Chatbot: Hello! Ask me about products or recommendations. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break
        response = bot.get_response(user_input)
        print(f"Chatbot: {response}")

# For non-interactive, just print a demo
def chatbot_demo():
    bot = SimpleChatbot()
    demo_inputs = ["hello", "what products do you have?", "recommend something", "bye"]
    for inp in demo_inputs:
        resp = bot.get_response(inp)
        print(f"Input: {inp} -> Response: {resp}")

if __name__ == "__main__":
    chatbot_demo()
