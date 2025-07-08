
# 📌 Emotion-Aware Feedback Analyzer

This project is an **Emotion-Aware Feedback Analyzer** for mentoring, mental health, or educational chat systems. It automatically detects the **emotional tone** of user messages (e.g., *sad*, *happy*, *angry*, *annoyed*, *dull*) to help mentors understand feedback sentiment in real time.

---

## 📂 Project Structure

| File | Description |
|------|--------------|
| `app.py` | Main app script for inference or API serving |
| `app_rnn.py` | RNN-based emotion classifier app |
| `emotion_predictor.ipynb` | Jupyter notebook for training, testing & experiments |
| `emotion_rnn_model.h5` | Pre-trained RNN (LSTM) model weights |
| `glove.6B.100d.txt` | GloVe word embeddings for the RNN |
| `train.txt` | Training data |
| `val.txt` | Validation data |
| `test.txt` | Test data |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## 🔍 **How It Works**

- **Dataset**: Text files (`train.txt`, `val.txt`, `test.txt`) containing messages labeled with emotions.
- **Preprocessing**: Messages are tokenized, cleaned, and vectorized using **TF-IDF** (for Logistic Regression) or **Word Embeddings** (GloVe for RNN).
- **Models**:
  - `Logistic Regression` → Baseline sentiment classifier.
  - `RNN/LSTM` → Sequence model for capturing context.
- **Outputs**: Emotion label for each input message.

---

## 🚀 **How to Run**

**1️⃣ Install Dependencies**


pip install -r requirements.txt
2️⃣ Run the Logistic Regression or TF-IDF baseline


python app.py
3️⃣ Run the RNN Emotion Predictor

python app_rnn.py
4️⃣ Train or test interactively

Use emotion_predictor.ipynb in Google Colab or locally to:

Preprocess data

Train your model

Evaluate metrics

⚙️ Key Files
glove.6B.100d.txt: Pretrained embeddings for richer text representation in the RNN.

emotion_rnn_model.h5: Saved RNN model.

app_rnn.py: Loads the RNN model, tokenizes input, and predicts emotion.

✅ Expected Outcome
The final app predicts emotions like:
😞 Sad
😀 Happy
😡 Angry
😐 Dull
😠 Annoyed

This helps mentors detect tone in user feedback and adapt support accordingly.
OUTPUT
![Screenshot 2025-06-29 181322](https://github.com/user-attachments/assets/abefa2ca-738e-4872-ba91-7065ac7dcd91)


