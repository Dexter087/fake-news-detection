import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.config import (
    PROCESSED_FILE,
    MODEL_FILE,
    VECTORIZER_FILE,
    REPORTS_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

from src.preprocess_text import clean_dataframe


def load_artifacts():
    """
    Loads the saved model and TF-IDF vectorizer.
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


def load_test_data():
    """
    Loads the processed dataset and recreates the same test split
    used during training.

    The training script first separates the final test set using TEST_SIZE.
    This function repeats that same first split so the saved model is
    evaluated on the same untouched test set.
    """

    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Processed file not found at: {PROCESSED_FILE}\n"
            "Run this first: python -m src.data_loader"
        )

    df = pd.read_csv(PROCESSED_FILE)

    if "combined_text" not in df.columns or "label" not in df.columns:
        raise ValueError("Processed dataset must contain combined_text and label columns.")

    df = clean_dataframe(df, text_column="combined_text")

    X = df["clean_text"]
    y = df["label"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_test, y_test


def evaluate_saved_model(model, vectorizer, X_test, y_test):
    """
    Evaluates the saved model on the test set.
    """

    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Fake News", "Real News"],
        ),
    }

    return metrics, y_pred


def save_evaluation_report(metrics):
    """
    Saves evaluation metrics to reports/final_evaluation.json.
    """

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / "final_evaluation.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation report saved to: {report_path}")


def save_confusion_matrix(y_test, y_pred):
    """
    Saves confusion matrix image to reports/figures/confusion_matrix.png.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake News", "Real News"],
    )

    display.plot(values_format="d")
    plt.title("Confusion Matrix - Fake News Detection")
    plt.tight_layout()

    figure_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"Confusion matrix saved to: {figure_path}")


def save_text_summary(metrics):
    """
    Saves a human-readable final evaluation summary.
    """

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = REPORTS_DIR / "final_summary.txt"

    cm = metrics["confusion_matrix"]

    summary = f"""
Fake News Detection - Final Evaluation Summary
================================================

Best Saved Model Evaluation

Accuracy : {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall   : {metrics['recall']:.4f}
F1-score : {metrics['f1_score']:.4f}

Confusion Matrix
----------------
Rows represent actual labels.
Columns represent predicted labels.

                Predicted Fake    Predicted Real
Actual Fake     {cm[0][0]}              {cm[0][1]}
Actual Real     {cm[1][0]}              {cm[1][1]}

Interpretation
--------------
The model performs strongly on the held-out test split. However, the result should be interpreted as dataset-based classification performance, not real-world truth verification.

Important Limitation
--------------------
The model learns patterns from the labelled dataset. It may learn writing style, topic patterns, or source-specific signals instead of directly identifying whether a news article is factually true or false. External validation on another dataset is recommended before making stronger generalization claims.
"""

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary.strip())

    print(f"Final text summary saved to: {summary_path}")


def main():
    print("=" * 70)
    print("Evaluating Saved Fake News Detection Model")
    print("=" * 70)

    model, vectorizer = load_artifacts()

    X_test, y_test = load_test_data()

    metrics, y_pred = evaluate_saved_model(
        model,
        vectorizer,
        X_test,
        y_test,
    )

    print("\nEvaluation Metrics")
    print("-" * 70)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")

    print("\nClassification Report")
    print("-" * 70)
    print(metrics["classification_report"])

    save_evaluation_report(metrics)
    save_confusion_matrix(y_test, y_pred)
    save_text_summary(metrics)


if __name__ == "__main__":
    main()