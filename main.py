import streamlit as st
import json
import random
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# KONFIGURASI HALAMAN WEB
st.set_page_config(page_title="Chatbot KORDER", page_icon="🤖")
st.title("🤖 Chatbot Layanan Pelanggan KORDER")
st.markdown("Selamat datang! Silakan tanya seputar harga dan pemesanan kuota.")

# 1. LOAD DATASET [cite: 13]
with open('intents.json') as file:
    data = json.load(file)

X_train, y_train, responses = [], [], {}
stemmer = StemmerFactory().create_stemmer()

# 2. PREPROCESSING & TRAINING [cite: 13]
for intent in data['intents']:
    for pattern in intent['patterns']:
        X_train.append(stemmer.stem(pattern.lower()))
        y_train.append(intent['tag'])
    responses[intent['tag']] = intent['responses']

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# 3. INTERFACE CHAT DI WEB
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input pengguna
if prompt := st.chat_input("Ketik pesan Anda di sini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prediksi menggunakan Model ML [cite: 13]
    processed_input = stemmer.stem(prompt.lower())
    tag = model.predict([processed_input])[0]
    response = random.choice(responses[tag])

    # Tampilkan balasan chatbot
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})