import joblib

from src.config import MODEL_FILE, VECTORIZER_FILE
from src.preprocess_text import clean_text


def load_model_and_vectorizer():
    """
    Loads the trained model and TF-IDF vectorizer.
    """

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_FILE}\n"
            "Run this first: python -m src.train_tfidf_models"
        )

    if not VECTORIZER_FILE.exists():
        raise FileNotFoundError(
            f"Vectorizer file not found at: {VECTORIZER_FILE}\n"
            "Run this first: python -m src.train_tfidf_models"
        )

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    return model, vectorizer


def predict_news(title, text):
    """
    Predicts whether a news article is real or fake.

    Label meaning:
        0 = Fake News
        1 = Real News
    """

    model, vectorizer = load_model_and_vectorizer()

    combined_text = str(title) + " " + str(text)
    cleaned_text = clean_text(combined_text)

    if cleaned_text.strip() == "":
        raise ValueError("Input text is empty after cleaning.")

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


def main():
    print("=" * 60)
    print("Fake News Detection System")
    print("=" * 60)

    title = input("\nEnter news title: ")
    text = input("\nEnter news article text: ")

    prediction, confidence = predict_news(title, text)

    print("\nPrediction Result")
    print("-" * 60)
    print(f"Predicted Label: {prediction}")

    if confidence is not None:
        print(f"Confidence Score: {confidence:.4f}")
    else:
        print("Confidence Score: Not available for this model")

    print("-" * 60)

    print(
        "\nNote: This prediction is based on patterns learned from the training dataset. "
        "It should not be treated as an absolute truth verification system."
    )


if __name__ == "__main__":
    main()