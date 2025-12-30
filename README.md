# 🎬 Movie Review Sentiment Analysis

A beginner-friendly **Sentiment Analysis** project using **RNN, LSTM, and GRU** with **TensorFlow/Keras**.  
The project classifies movie reviews from the **IMDb dataset** as **Positive** or **Negative**, and includes a **Streamlit web app** for real-time prediction.

---

## 🧠 Features

- Vanilla RNN, LSTM, and GRU models for sequence-to-one sentiment classification
- IMDb dataset with **preprocessing, tokenization, and padding**
- Embedding layers for word representation
- Real-time **Streamlit app** for predicting sentiment from custom reviews
- Accuracy comparison between RNN, LSTM, and GRU
- Easy-to-upgrade for **Attention mechanisms** or **Transformer-based models**

---

## 📊 Accuracy Comparison (Test Set)

| Model       | Test Accuracy (Approx) |
| ----------- | ---------------------- |
| Vanilla RNN | 75–80%                 |
| LSTM        | 85–88%                 |
| GRU         | 84–87%                 |

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```

2. Install dependencies:

```bash
pip install tensorflow streamlit
```

---

## 📂 Project Structure

```
sentiment-analysis/
│
├─ app.py                # Streamlit web app
├─ sentiment_model.h5    # Saved trained LSTM model
├─ train_models.py       # Training code for RNN, LSTM, GRU
├─ README.md             # Project documentation
└─ requirements.txt      # Python dependencies
```

---

## 🏗 Training the Models

Example for **LSTM**:

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load IMDb dataset
vocab_size = 10000
max_length = 200
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Build LSTM model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Save model for Streamlit
model.save("sentiment_model.h5")
```

Replace `LSTM(64)` with `SimpleRNN(64)` or `GRU(64)` to train other models.

---

## 🌐 Running the Streamlit App

```bash
streamlit run app.py
```

- Enter a movie review in the text area
- Click **Analyze** to see if it is **Positive** or **Negative**

---

## 📚 References

- [IMDb Dataset – TensorFlow/Keras](https://keras.io/api/datasets/imdb/)
- [TensorFlow LSTM Tutorial](https://www.tensorflow.org/tutorials/text/text_classification_rnn)
- [Streamlit Documentation](https://docs.streamlit.io/)
