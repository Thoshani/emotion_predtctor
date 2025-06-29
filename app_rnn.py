# app_rnn.py

import pandas as pd
import numpy as np
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import gradio as gr
import os

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

# 3️⃣ Split and encode
X = df['clean_text'].values
y = df['emotion'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

y_train_cat = to_categorical(y_train_enc)
y_test_cat = to_categorical(y_test_enc)

# 4️⃣ Load GloVe
embedding_index = {}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

word_index = tokenizer.word_index
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    vector = embedding_index.get(word)
    if vector is not None:
        embedding_matrix[i] = vector

# 5️⃣ Build or Load Model
model_path = "emotion_rnn_model.h5"

if os.path.exists(model_path):
    print("Loading saved model...")
    model = load_model(model_path)
else:
    print("Training new model...")
    model = Sequential()
    model.add(Embedding(input_dim=num_words,
                    output_dim=EMBEDDING_DIM,
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=True))

    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(X_train_pad, y_train_cat,
              validation_split=0.2,
              epochs=5,
              batch_size=64,
              callbacks=[early_stop])

    model.save(model_path)

# 6️⃣ Gradio interface
def predict_emotion_rnn(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    pred = model.predict(pad)
    label = label_encoder.inverse_transform([np.argmax(pred)])
    return f"Predicted Emotion: {label[0]}"

iface = gr.Interface(
    fn=predict_emotion_rnn,
    inputs=gr.Textbox(lines=2, placeholder="Type your text here..."),
    outputs="text",
    title="Emotion Predictor Chatbot (RNN)",
    description="Enter any text to predict the emotion using a Bi-LSTM model."
)

if __name__ == "__main__":
    iface.launch(inbrowser=True)

