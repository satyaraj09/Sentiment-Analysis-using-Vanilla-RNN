import streamlit as st
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

model = load_model("model/sentiment_analysis.keras")

vocab_size = 10000
max_length = 200
word_index = imdb.get_word_index()


def encode_text(text):
    tokens = []
    for word in text.lower().split():
        if word in word_index and word_index[word] < vocab_size:
            tokens.append(word_index[word])
    return pad_sequences([tokens], maxlen=max_length)


st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a review to predict sentiment")

user_input = st.text_area("Your Review")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        encoded = encode_text(user_input)
        prediction = model.predict(encoded)[0][0]

        if prediction > 0.5:
            st.success(f"😊 Positive Review ({prediction:.2f})")
        else:
            st.error(f"😞 Negative Review ({prediction:.2f})")
