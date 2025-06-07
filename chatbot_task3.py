import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, render_template
import joblib
import json
import random
import numpy as np
from tensorflow.keras.models import load_model
from scripts.preprocess import clean_text

# Load model, vectorizer, and encoder
model = load_model('model/intent_classifier.h5')
vectorizer = joblib.load('model/vectorizer.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

# Load intents and responses
with open('intents.json', 'r', encoding="utf-8") as f:
    intents_data = json.load(f)

# Create response mapping
response_map = {intent['tag']: intent['responses'] for intent in intents_data['intents']}

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('chat_interface.html')

@app.route('/query', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message', '')
    
    if not user_msg:
        return jsonify({'error': 'Empty message'}), 400
    
    cleaned = clean_text(user_msg)
    if not cleaned:
        return jsonify({'error': 'Invalid input after preprocessing'}), 400
    
    print("User input:", user_msg)
    print("Cleaned message:", cleaned)
    
    vectorized = vectorizer.transform([cleaned]).toarray()
    predicted_intent = label_encoder.inverse_transform(np.argmax(model.predict(vectorized, verbose=0), axis=1))[0]
    
    print("Predicted intent:", predicted_intent)
    
    response = random.choice(response_map.get(predicted_intent, ["I'm not sure how to help with that."]))
    
    # Log intent for insights
    with open('intent_logs.txt', 'a') as f:
        f.write(f"{predicted_intent}\n")
    
    return jsonify({
        'intent': predicted_intent,
        'response': response
    })

if __name__ == '__main__':
    app.run(debug=True)