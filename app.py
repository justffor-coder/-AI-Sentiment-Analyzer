import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not API_KEY:
    st.error(
        "Hugging Face API key not found. Please set HUGGINGFACE_API_KEY in .env file.")
    st.stop()

headers = {"Authorization": f"Bearer {API_KEY}"}


def query_sentiment(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    if response.status_code == 200:
        return response.json()[0]
    else:
        return {"error": response.status_code, "message": response.text}


# Streamlit UI
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="😊")
st.title("😊 AI Sentiment Analyzer")
st.write("Enter a sentence to analyze its sentiment using a pre-trained AI model.")

user_input = st.text_area("Your text:", placeholder="I love this weather!")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        with st.spinner("Analyzing..."):
            result = query_sentiment(user_input)

        if "error" in result:
            st.error(f"API Error: {result['message']}")
        else:
            # Extract label and confidence
            label_map = {
                "LABEL_0": "Negative 😞",
                "LABEL_1": "Neutral 😐",
                "LABEL_2": "Positive 😊"
            }
            label = result[0]['label']
            score = result[0]['score']

            st.success(f"**Sentiment**: {label_map.get(label, label)}")
            st.progress(float(score))
            st.write(f"Confidence: {score:.2%}")
