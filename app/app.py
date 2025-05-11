import streamlit as st
import joblib

# Load trained model
model = joblib.load('models/sentiment_model.pkl')

# Streamlit UI
st.title("📝 Analisis Sentimen Ulasan")

text = st.text_area("Masukkan Ulasan:")

if st.button("Analisis"):
    if text.strip() != "":
        prediction = model.predict([text])[0]
        st.success(f"Hasil Rating: **{prediction.upper()}**")
    else:
        st.warning("Masukkan teks terlebih dahulu.")
