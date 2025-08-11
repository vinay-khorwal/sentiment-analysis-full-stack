from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import string
import numpy as np
import pickle
import os

# Correct Keras / TensorFlow imports
from keras.models import load_model
from keras.utils import pad_sequences

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
# (You only need to run these downloads once, you can comment them out later)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# --- Your Flask App Code Starts Below ---
app = Flask(__name__)
CORS(app) # This will allow the frontend to make requests to the backend

# --- Load Model and Preprocessing Objects ---
# Get the absolute path to the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model_9.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 200  # Make sure this matches your training max length

# --- Preprocessing Functions (Your code) ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def to_lowercase(text):
    return text.lower() if isinstance(text, str) else text

def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text) if isinstance(text, str) else text

def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text) if isinstance(text, str) else text

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation)) if isinstance(text, str) else text

def remove_stopwords(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        filtered = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered)
    return text

def tokenize_text(text):
    return word_tokenize(text) if isinstance(text, str) else text

def lemmatize_text(tokens):
    if isinstance(tokens, list):
        return [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def preprocess_text(text):
    text = to_lowercase(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    tokens = tokenize_text(text)
    tokens = lemmatize_text(tokens)
    return ' '.join(tokens)


# --- Prediction Function ---
def predict_sentiment(text):
    clean_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    probabilities = model.predict(padded, verbose=0)
    prediction_index = np.argmax(probabilities, axis=1)[0]
    sentiment = label_encoder.inverse_transform([prediction_index])[0]
    return sentiment


# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text') # Use .get() for safety
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        sentiment = predict_sentiment(text)
        return jsonify({'prediction': sentiment})
    except Exception as e:
        # THIS IS THE CRITICAL CHANGE: PRINT THE ERROR TO THE CONSOLE
        print(f"--- AN ERROR OCCURRED IN THE BACKEND ---")
        import traceback
        traceback.print_exc() # This will print the full error stack
        # ----------------------------------------------------
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For local development
    app.run(debug=True, port=5000)