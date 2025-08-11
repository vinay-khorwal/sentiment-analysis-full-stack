# ==============================================================================
# train_model.py
#
# This script trains a new sentiment analysis model from scratch.
# It performs the following steps:
#   1. Loads the dataset.
#   2. Preprocesses the text data using the exact same steps as the API.
#   3. Creates and fits a Tokenizer and a LabelEncoder.
#   4. Builds, compiles, and trains a Keras LSTM model.
#   5. Saves the trained model, tokenizer, and label encoder for use in app.py.
#
# ==============================================================================

import pandas as pd
import numpy as np
import pickle
import re
import string
import os

# TensorFlow and Keras for model building
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Scikit-learn for data handling and label encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# NLTK for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- Ensure NLTK data is available ---
# This checks for and downloads necessary NLTK packages.
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading NLTK resource: {resource}...")
            nltk.download(resource, quiet=True)

download_nltk_resources()


# ==============================================================================
# --- CONFIGURATION ---
# All major parameters are defined here for easy modification.
# ==============================================================================

# File paths and column names
DATASET_PATH = r'D:\sentiment analysis\IMDB-Dataset.csv' # Use raw string for Windows paths
TEXT_COLUMN = 'review'
LABEL_COLUMN = 'sentiment'
OUTPUT_MODEL_NAME = "sentiment_model_new.h5"
OUTPUT_TOKENIZER_NAME = "tokenizer_new.pkl"
OUTPUT_LABEL_ENCODER_NAME = "label_encoder_new.pkl"


# Model and training parameters
VOCAB_SIZE = 15000         # Max number of words to keep in the vocabulary
MAX_LEN = 250              # Max length of sequences (must match in app.py!)
EMBEDDING_DIM = 128        # Dimension of the word embeddings
EPOCHS = 3                 # Number of training epochs (3-5 is a good start)
BATCH_SIZE = 64            # Number of samples per training batch

# ==============================================================================
# --- STEP 1: LOAD AND PREPROCESS DATA ---
# ==============================================================================

print("1. Loading and preprocessing dataset...")

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"ERROR: Dataset not found at '{DATASET_PATH}'. Please check the path.")
    exit()

# This preprocessing function MUST be identical to the one in `app.py`
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Ensure text is a string
    if not isinstance(text, str):
        return ""
    text = text.lower()                                                  # Lowercase
    text = re.sub(r'<.*?>', '', text)                                    # Remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)                     # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))     # Remove punctuation
    tokens = word_tokenize(text)                                         # Tokenize
    filtered = [word for word in tokens if word not in stop_words]       # Remove stopwords
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered]       # Lemmatize
    return ' '.join(lemmatized)

# Apply the preprocessing to the text column
df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_text)

print("   ...Preprocessing complete.")


# ==============================================================================
# --- STEP 2: TOKENIZE TEXT AND ENCODE LABELS ---
# ==============================================================================

print("2. Tokenizing text and encoding labels...")

# Create and fit the tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
tokenizer.fit_on_texts(df[TEXT_COLUMN])
sequences = tokenizer.texts_to_sequences(df[TEXT_COLUMN])
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# Create and fit the label encoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[LABEL_COLUMN])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("   ...Tokenization and encoding complete.")


# ==============================================================================
# --- STEP 3: BUILD AND COMPILE THE MODEL ---
# ==============================================================================

print("3. Building and compiling the Keras model...")

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    SpatialDropout1D(0.3),  # Helps prevent overfitting
    LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True),
    LSTM(64, dropout=0.3, recurrent_dropout=0.3),
    Dense(len(label_encoder.classes_), activation='softmax') # Softmax for multi-class classification
])

# Using 'softmax' requires one-hot encoded labels and 'categorical_crossentropy'
from keras.utils import to_categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("   ...Model built successfully.")


# ==============================================================================
# --- STEP 4: TRAIN THE MODEL ---
# ==============================================================================

print("\n4. Starting model training...")

history = model.fit(
    X_train, y_train_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test_cat),
    verbose=1
)

print("\n   ...Training complete.")


# ==============================================================================
# --- STEP 5: SAVE THE MODEL AND PREPROCESSING OBJECTS ---
# ==============================================================================

print("\n5. Saving model and artifacts...")

# Save the Keras model
model.save(OUTPUT_MODEL_NAME)
print(f"   -> Model saved to '{OUTPUT_MODEL_NAME}'")

# Save the tokenizer
with open(OUTPUT_TOKENIZER_NAME, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"   -> Tokenizer saved to '{OUTPUT_TOKENIZER_NAME}'")

# Save the label encoder
with open(OUTPUT_LABEL_ENCODER_NAME, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"   -> Label Encoder saved to '{OUTPUT_LABEL_ENCODER_NAME}'")


print("\n=========================================================")
print("=== All tasks completed successfully!                 ===")
print("=== Your new model is ready to be used by app.py.     ===")
print("=========================================================")