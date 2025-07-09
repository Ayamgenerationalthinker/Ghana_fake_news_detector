# 🔍 Ghana Fake News Detector

A machine learning-powered Streamlit web app that detects fake news articles, built specifically for Ghana's media space.



## 🚀 Features
- Classifies news as `FAKE`, `REAL`, or `UNCERTAIN`
- Interactive Streamlit interface with modern UI
- Confidence score visualization
- Batch analysis for multiple headlines
- URL-based article analysis
- Extracts key features like word count, capital letters, numbers, etc.
- Built with `scikit-learn`, `Streamlit`, `Plotly`, and `pandas`

## 📦 Installation

```bash
git clone https://github.com/Ayamgenerationalthinker/Ghana_fake_news_detector.git
cd Ghana_fake_news_detector
python -m venv venv
venv\Scripts\activate  # For Windows
pip install -r requirements.txt
▶️ Run the App
bash
Copy
Edit
streamlit run app.py
The app will open in your browser at http://localhost:8501.

🧠 Model Details
Model: Logistic Regression

Text vectorization: TF-IDF (max 5000 features)

Trained on labeled Ghanaian news (fake & real)

Supports custom training and fine-tuning

📚 Sources & Datasets
GhanaWeb, JoyNews, CitiNews, Dubawa Ghana fact-checks

Fake news headlines from social media and verified misinformation posts

🔧 File Structure
graphql
Copy
Edit
Ghana_fake_news_detector/
│
├── app.py                 # Main Streamlit app
├── models/                # Saved TF-IDF and classifier model
├── data/                  # Optional: training dataset
├── utils.py               # Helper functions
├── requirements.txt       # All dependencies
└── README.md              # This file
👨‍💻 Author
Prince Ofosu Fiebor (AyamGenerationalThinker)
Digital Skills Advocate | Tech Educator | Data & AI Enthusiast

🌐 Live Deployment (optional)
You can deploy this app on:

Streamlit Cloud

Render

Hugging Face Spaces

🙌 Contributions
Pull requests are welcome! For major changes, please open an issue first.

