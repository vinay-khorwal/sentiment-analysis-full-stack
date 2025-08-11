# Full-Stack Sentiment Analysis Web App

![Frontend](https://img.shields.io/badge/Frontend-React-61DAFB?style=for-the-badge&logo=react)
![Backend](https://img.shields.io/badge/Backend-Flask-000000?style=for-the-badge&logo=flask)
![ML Model](https://img.shields.io/badge/ML%20Model-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow)

---

## 📌 Overview

This is a full-stack web application that performs real-time sentiment analysis on user-provided text. The frontend is built with React, and the backend is a Flask API serving a TensorFlow/Keras model.

Users can enter any text, and the backend will classify its sentiment as Positive or Negative. The UI updates with corresponding colors and icons.

![Screenshot of the Sentiment Analysis App](./public/screenshot.png)
Note: Save a screenshot of your running app as `frontend/public/screenshot.png` to display it here.

---

## ✨ Features

- Real-time sentiment prediction
- Dynamic UI with loading states
- Clean, modern styling
- Decoupled frontend and backend
- Includes training and preprocessing scripts

---

## 🛠️ Technology Stack

### Frontend
- React.js
- CSS3
- Fetch API

### Backend
- Flask
- TensorFlow / Keras
- Scikit-learn
- NLTK

---

## 📂 Project Structure

```plaintext
sentiment_analysis_web_stack/
├── backend/
│   ├── app.py                    # Flask API
│   ├── train_model.py            # Train a new model
│   ├── create_preprocessors.py   # Recreate tokenizer/label encoder
│   ├── sentiment_model_9.h5      # Pre-trained Keras model (in repo)
│   ├── tokenizer.pkl             # Saved Keras Tokenizer (in repo)
│   ├── label_encoder.pkl         # Saved LabelEncoder (in repo)
│   └── requirements.txt          # Python dependencies
│
└── frontend/
    ├── public/
    │   ├── index.html
    │   └── screenshot.png        # Optional screenshot for README
    ├── src/
    │   ├── App.js
    │   ├── App.css
    │   └── index.js
    ├── package.json
    └── README.md
```

---

## ⚙️ Setup (Local Development)

### 1) Backend

```bash
cd backend
python -m venv .venv

# Activate
# Windows PowerShell
. .venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

# Run the API
python app.py
```

The API runs at `http://127.0.0.1:5000`.

### 2) Frontend

```bash
cd frontend
npm install

# Create environment file
echo REACT_APP_API_URL=http://127.0.0.1:5000 > .env.local

# Start the React dev server
npm start
```

Open `http://localhost:3000` in your browser.

Note (Windows PowerShell): If the echo command does not create the file, create `frontend/.env.local` manually with this line:

```text
REACT_APP_API_URL=http://127.0.0.1:5000
```

---

## 🧠 Model Training

You can train your own model with `backend/train_model.py`.

1. Place your dataset CSV somewhere accessible and update these variables near the top of `backend/train_model.py`:

```python
DATASET_PATH = r'path_to_your_dataset.csv'
TEXT_COLUMN = 'your_text_column'
LABEL_COLUMN = 'your_label_column'
```

2. Run the training script:

```bash
cd backend
python train_model.py
```

By default, the script saves: `sentiment_model_new.h5`, `tokenizer_new.pkl`, and `label_encoder_new.pkl`.

The API currently expects these filenames in `backend/app.py`:

```python
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model_9.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
```

After training, either rename the new files to the expected names, or update the constants above to point to your new files.

---

## 🚀 Deployment (Brief)

- Deploy the backend (Flask) to a Python-friendly host (e.g., Render, Railway, or similar).
- Deploy the frontend (React) to a static host (e.g., Vercel or Netlify).
- Set `REACT_APP_API_URL` in your frontend environment to the deployed backend URL.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 👨‍💻 Author

- Name: Vinay Khorwal
- GitHub: `https://github.com/your-username`