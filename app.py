# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import time
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from urllib.parse import urlparse
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Ghana Fake News Detector",
    page_icon="🇬🇭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------- Load Model -------------
MODEL_DIR = Path("models")
DATA_DIR  = Path("data")

vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
clf        = joblib.load(MODEL_DIR / "logreg_fake_news.pkl")

try:
    dataset = pd.read_csv(DATA_DIR / "full_dataset.csv")
except:
    dataset = pd.DataFrame()

LOW, HIGH = 0.3, 0.7

# ------------- Custom CSS -------------
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #0c7d4f, #01411c);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.fake-news {
    background: linear-gradient(135deg, #ff6b6b, #ee5a52);
    color: white; padding: 1rem; border-radius: 10px; text-align: center; font-weight: bold;
}
.real-news {
    background: linear-gradient(135deg, #51cf66, #40c057);
    color: white; padding: 1rem; border-radius: 10px; text-align: center; font-weight: bold;
}
.uncertain-news {
    background: linear-gradient(135deg, #ffd43b, #fab005);
    color: white; padding: 1rem; border-radius: 10px; text-align: center; font-weight: bold;
}
.analysis-section {
    background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
}
.stButton > button {
    background: linear-gradient(135deg, #0c7d4f 0%, #01411c 100%);
    color: white; border: none; border-radius: 25px; padding: 0.5rem 2rem;
    font-weight: bold; transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# ------------- Prediction Logic -------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def classify_with_model(text):
    vec = vectorizer.transform([clean_text(text)])
    prob = clf.predict_proba(vec)[0][1]
    if prob >= HIGH:
        label = "FAKE"
    elif prob <= LOW:
        label = "REAL"
    else:
        label = "UNCERTAIN"
    return label, prob, vec

def suggest_sources(vec, k=3):
    if dataset.empty:
        return []
    real_articles = dataset[dataset['label'].str.upper() == 'REAL']
    vecs = vectorizer.transform(real_articles["clean_text"])
    sims = cosine_similarity(vec, vecs).flatten()
    idx = sims.argsort()[-k:][::-1]
    return real_articles.iloc[idx][["title", "source"]].to_dict("records")

def extract_features(text):
    return {
        "Characters": len(text),
        "Words": len(text.split()),
        "Sentences": len(text.split(".")),
        "Exclamations": text.count("!"),
        "Questions": text.count("?"),
        "Capital Letters": sum(1 for c in text if c.isupper()),
        "Numbers": len(re.findall(r"\d+", text))
    }

# ------------- UI Logic -------------
if "count" not in st.session_state:
    st.session_state.count = 0

# --- Header
st.markdown("""
<div class="main-header">
    <h1>🇬🇭 Ghana Fake News Detector</h1>
    <p>Detect fake news using AI trained on local news articles</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7, 0.05)
    show_features = st.checkbox("Show Feature Analysis", value=True)
    show_confidence = st.checkbox("Show Confidence Score", value=True)
    st.metric("Total Analyses", st.session_state.count)

# --- Input Section
st.subheader("📝 Enter News Text")

news_input = st.text_area("Paste your news text here:", height=200)

if st.button("🔍 Analyze"):
    if len(news_input.strip()) < 10:
        st.error("Please provide a valid news article.")
    else:
        label, prob, vec = classify_with_model(news_input)
        st.session_state.count += 1

        # Result Display
        if label == "FAKE":
            st.markdown('<div class="fake-news">🚨 FAKE NEWS DETECTED</div>', unsafe_allow_html=True)
        elif label == "REAL":
            st.markdown('<div class="real-news">✅ THIS NEWS SEEMS REAL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="uncertain-news">⚠️ CANNOT VERIFY THIS NEWS</div>', unsafe_allow_html=True)

        # Confidence Score
        if show_confidence:
            st.metric("Confidence (fake probability)", f"{prob:.2f}")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Confidence Level"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 60], "color": "lightgray"},
                        {"range": [60, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "red"},
                    ],
                    "threshold": {
                        "line": {"color": "blue", "width": 4},
                        "value": threshold * 100,
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        # Feature Section
        if show_features:
            st.markdown('<div class="analysis-section"><h4>🔍 Feature Analysis</h4></div>', unsafe_allow_html=True)
            feats = extract_features(news_input)
            col1, col2 = st.columns(2)
            for i, (k, v) in enumerate(feats.items()):
                (col1 if i % 2 == 0 else col2).metric(k, v)

            chart = px.bar(x=list(feats.keys()), y=list(feats.values()), title="Text Features")
            chart.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(chart, use_container_width=True)

        # Suggest real source if REAL
        if label == "REAL":
            suggestions = suggest_sources(vec)
            if suggestions:
                st.markdown("#### 📰 Possible Real Sources")
                for s in suggestions:
                    st.markdown(f"- [{s['title']}]({s['source']})")

        # Explanation
        st.markdown(f"""
        <div class="analysis-section">
            <h4>💡 Explanation</h4>
            <p>This classification was based on the TF-IDF features of your text, using a Logistic Regression model trained on real Ghanaian fake/real news samples. 
            Probability closer to 1.0 = more likely fake; closer to 0 = more likely real.</p>
            <hr style="margin-top:25px;">
            <div style="text-align:center;color:gray;font-size:0.9em;">
                 Built with ❤️ by Prince Ofosu Fiebor · Powered by Streamlit & scikit‑learn
            </div>
        </div>
        """, unsafe_allow_html=True)


