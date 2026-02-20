import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Amazon Sentiment Analyzer")

st.title("🛒 Amazon Product Review Sentiment Classifier")
st.write("Enter a product review below to predict its sentiment (Negative / Neutral / Positive).")

review = st.text_area("✍ Enter Amazon Product Review")

if st.button("🔍 Predict Sentiment"):
    vec = vectorizer.transform([review])
    prediction = model.predict(vec)
    st.success(f"Predicted Sentiment: {prediction[0]}")
