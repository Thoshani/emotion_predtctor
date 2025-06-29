# app.py

import os
import pandas as pd
import numpy as np
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

import gradio as gr

# 1️⃣ Load CSVs
df_train = pd.read_csv("train.csv", sep=";", names=["text", "emotion"])
df_val = pd.read_csv("val.csv", sep=";", names=["text", "emotion"])
df_test = pd.read_csv("test.csv", sep=";", names=["text", "emotion"])

df = pd.concat([df_train, df_val, df_test], ignore_index=True)
df.dropna(subset=['text', 'emotion'], inplace=True)

# 2️⃣ Clean text
stop_words = set(stopwords.words('english'))
custom_stopwords = {
    'feel', 'feeling', 'really', 'very', 'just', 'always',
    'today', 'now', 'im', 'ive', 'ill', 'cant', 'dont',
    'get', 'got', 'much', 'lot', 'one', 'thing', 'know', 'like', 'people', 'time'
}
all_stopwords = stop_words.union(custom_stopwords)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in all_stopwords]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# 3️⃣ Train/Load model
model_path = "lr_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    print("Loading saved Logistic Regression model and TF-IDF vectorizer...")
    lr_model = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)
else:
    print("Training Logistic Regression model and saving...")
    X = df['clean_text'].values
    y = df['emotion'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    lr_model = LogisticRegression(max_iter=200)
    lr_model.fit(X_train_tfidf, y_train)

    joblib.dump(lr_model, model_path)
    joblib.dump(tfidf, vectorizer_path)
    print("Model saved successfully.")

# 4️⃣ Gradio function
def predict_emotion(text):
    text = clean_text(text)
    vec = tfidf.transform([text])
    pred = lr_model.predict(vec)
    return f"Predicted Emotion: {pred[0]}"

iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=2, placeholder="Type your text here..."),
    outputs="text",
    title="Emotion Predictor Chatbot (Logistic Regression)",
    description="Enter any text to predict the emotion category using Logistic Regression."
)

if __name__ == "__main__":
    iface.launch(inbrowser=True)

