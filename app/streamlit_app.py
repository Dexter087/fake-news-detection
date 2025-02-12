import sys
from pathlib import Path

import joblib
import streamlit as st


# ---------------------------------------------------------
# Fix import path so Streamlit can import files from src/
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import MODEL_FILE, VECTORIZER_FILE
from src.preprocess_text import clean_text


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
)


# ---------------------------------------------------------
# Load model and vectorizer
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    if not MODEL_FILE.exists():
        st.error(
            "Model file not found. Please train the model first using:\n\n"
            "`python -m src.train_tfidf_models`"
        )
        st.stop()

    if not VECTORIZER_FILE.exists():
        st.error(
            "Vectorizer file not found. Please train the model first using:\n\n"
            "`python -m src.train_tfidf_models`"
        )
        st.stop()

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    return model, vectorizer


def predict_news(title, article_text, model, vectorizer):
    combined_text = str(title) + " " + str(article_text)
    cleaned_text = clean_text(combined_text)

    if cleaned_text.strip() == "":
        return None, None

    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]

    if prediction == 1:
        label = "Real News"
    else:
        label = "Fake News"

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_tfidf)[0]
        confidence = float(max(probabilities))

    return label, confidence


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------
model, vectorizer = load_model_and_vectorizer()

st.title("📰 Fake News Detection")
st.write(
    "This app predicts whether a news article is more likely to be "
    "**Real News** or **Fake News** based on patterns learned from a labelled dataset."
)

st.info(
    "This is a machine learning classification system, not a real-world truth verification tool."
)

title = st.text_input("News Title")

article_text = st.text_area(
    "News Article Text",
    height=250,
    placeholder="Paste the article text here...",
)

predict_button = st.button("Predict")

if predict_button:
    if title.strip() == "" and article_text.strip() == "":
        st.warning("Please enter a news title or article text before predicting.")
    else:
        prediction, confidence = predict_news(
            title,
            article_text,
            model,
            vectorizer,
        )

        if prediction is None:
            st.warning("The input became empty after cleaning. Please enter more meaningful text.")
        else:
            st.subheader("Prediction Result")

            if prediction == "Real News":
                st.success(f"Predicted Label: {prediction}")
            else:
                st.error(f"Predicted Label: {prediction}")

            if confidence is not None:
                st.write(f"Confidence Score: **{confidence:.4f}**")
            else:
                st.write("Confidence Score: Not available for the selected model.")

            st.caption(
                "Note: This prediction is based on patterns learned from the training dataset. "
                "It should not be treated as absolute proof that an article is true or false."
            )


st.divider()

st.subheader("How to Interpret the Output")

st.write(
    """
The model uses TF-IDF features and a trained machine learning classifier.  
A **Real News** prediction means the article is more similar to real news examples in the training dataset.  
A **Fake News** prediction means the article is more similar to fake news examples in the training dataset.

The prediction should be used only as an educational machine learning result, not as a final truth judgment.
"""
)

st.subheader("Project Limitations")

st.write(
    """
- The model learns from labelled dataset patterns.
- It may learn writing style or topic patterns instead of factual truth.
- It does not verify claims using live sources.
- It may not generalize well to news from completely different sources.
- External validation on another dataset is needed before making stronger real-world claims.
"""
)