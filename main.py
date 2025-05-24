### mental_health_chatbot.py

# Required Libraries
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from better_profanity import profanity

# Initialize Flask app
app = Flask(__name__)

# Load pretrained conversational model
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up logging
logging.basicConfig(filename="chat_logs.txt", level=logging.INFO, format='%(asctime)s - %(message)s')

# Empathetic response override (optional)
empathetic_keywords = {
    "stress": "I'm sorry to hear you're feeling stressed. Want to talk about what’s causing it?",
    "lonely": "Loneliness can be tough. I'm here for you. Would you like to talk about it?",
    "anxious": "Feeling anxious is completely valid. Can you share what's making you feel this way?",
    "depressed": "I'm really sorry you're feeling this way. You're not alone. I'm here to listen."
}

# Route for chatbot
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")

    # Profanity filter
    if profanity.contains_profanity(user_input):
        return jsonify({"response": "Let's keep this conversation respectful. I'm here to help you."})

    # Empathetic keyword-based response
    for keyword, response in empathetic_keywords.items():
        if keyword in user_input.lower():
            log_interaction(user_input, response)
            return jsonify({"response": response})

    # Tokenize and generate response from model
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    response_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    log_interaction(user_input, response_text)
    return jsonify({"response": response_text})

# Log interactions to file
def log_interaction(user_input, bot_response):
    logging.info(f"User: {user_input}\nBot: {bot_response}\n")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
