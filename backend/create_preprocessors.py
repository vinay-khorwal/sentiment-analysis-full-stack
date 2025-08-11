# create_preprocessors.py

import pandas as pd
import pickle
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

# --- IMPORTANT ---
# 1. Place your original training data CSV in the `backend` folder.
# 2. Change the filename and column names below to match your data.
DATASET_PATH = r'D:\sentiment analysis\cleaned_test_data (3) (1).csv'  # <--- CHANGE THIS
TEXT_COLUMN = 'text'                    # <--- CHANGE THIS
LABEL_COLUMN = 'sentiment'              # <--- CHANGE THIS

print("Loading original dataset...")
# Make sure to handle potential errors with a try-except block
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: Dataset not found at '{DATASET_PATH}'")
    print("Please place your training data CSV in the backend folder and update the filename in this script.")
    exit()

# Validate required columns exist
missing_columns = [col for col in [TEXT_COLUMN, LABEL_COLUMN] if col not in df.columns]
if missing_columns:
    print(f"Error: Missing expected column(s): {missing_columns}. Available columns: {list(df.columns)}")
    exit()

# Coerce text and label columns to strings and handle missing values
text_series = df[TEXT_COLUMN].fillna("").astype(str)
label_series = df[LABEL_COLUMN].fillna("").astype(str)

# 1. Create and save the Tokenizer
print("Creating and fitting the Tokenizer...")
# Use the same num_words you used during training, or a reasonable limit
tokenizer = Tokenizer(num_words=10000, oov_token="<unk>")
tokenizer.fit_on_texts(text_series.tolist())

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("-> tokenizer.pkl saved successfully.")


# 2. Create and save the Label Encoder
print("\nCreating and fitting the LabelEncoder...")
label_encoder = LabelEncoder()
label_encoder.fit(label_series.tolist())

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("-> label_encoder.pkl saved successfully.")

print("\nAll preprocessing files have been re-created with your current library versions.")