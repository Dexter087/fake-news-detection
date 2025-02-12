import json
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    PassiveAggressiveClassifier,
    SGDClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import (
    PROCESSED_FILE,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    MODEL_FILE,
    VECTORIZER_FILE,
    MODEL_COMPARISON_CSV,
    MODEL_COMPARISON_JSON,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
)

from src.preprocess_text import clean_dataframe


def load_processed_dataset():
    """
    Loads the processed dataset created by data_loader.py.
    """

    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {PROCESSED_FILE}\n"
            "Run this first: python -m src.data_loader"
        )

    print("Loading processed dataset...")
    df = pd.read_csv(PROCESSED_FILE)

    required_columns = ["combined_text", "label"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column missing from processed dataset: {col}")

    print("Dataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    return df


def split_train_validation_test(df):
    """
    Splits the dataset into train, validation, and test sets.

    Train set:
        Used to train the models.

    Validation set:
        Used to compare models and select the best one.

    Test set:
        Used only once at the end for final evaluation.
    """

    X = df["clean_text"]
    y = df["label"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    validation_ratio_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=validation_ratio_adjusted,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("\nDataset split:")
    print("Train size:", len(X_train))
    print("Validation size:", len(X_val))
    print("Test size:", len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


def vectorize_text(X_train, X_val, X_test):
    """
    Converts cleaned text into TF-IDF feature vectors.
    """

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        stop_words="english",
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print("TF-IDF train shape:", X_train_tfidf.shape)
    print("TF-IDF validation shape:", X_val_tfidf.shape)
    print("TF-IDF test shape:", X_test_tfidf.shape)

    return vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf


def make_calibrated_ridge():
    """
    Creates a calibrated Ridge Classifier.

    Newer scikit-learn versions use 'estimator'.
    Older versions use 'base_estimator'.

    This try/except keeps the code compatible with both.
    """

    try:
        return CalibratedClassifierCV(
            estimator=RidgeClassifier(
                random_state=RANDOM_STATE,
            ),
            method="sigmoid",
            cv=3,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=RidgeClassifier(
                random_state=RANDOM_STATE,
            ),
            method="sigmoid",
            cv=3,
        )


def get_models():
    """
    Returns all models to be compared.
    """

    models = {
        "multinomial_naive_bayes": MultinomialNB(),

        "complement_naive_bayes": ComplementNB(),

        "logistic_regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE,
        ),

        "linear_svm": LinearSVC(
            C=1.0,
            random_state=RANDOM_STATE,
        ),

        "ridge_classifier": RidgeClassifier(
            random_state=RANDOM_STATE,
        ),

        "calibrated_ridge_classifier": make_calibrated_ridge(),

        "passive_aggressive": PassiveAggressiveClassifier(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),

        "sgd_log_loss": SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),

        "sgd_hinge_svm": SGDClassifier(
            loss="hinge",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "extra_trees": ExtraTreesClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


def evaluate_predictions(y_true, y_pred):
    """
    Calculates evaluation metrics.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    return metrics


def train_and_compare_models(X_train_tfidf, X_val_tfidf, y_train, y_val):
    """
    Trains all models and compares them on the validation set.
    """

    models = get_models()

    trained_models = {}
    comparison_rows = []
    detailed_results = {}

    print("\nTraining and comparing models...")

    for model_name, model in models.items():
        print("\n" + "=" * 70)
        print(f"Training model: {model_name}")
        print("=" * 70)

        start_time = time.time()

        model.fit(X_train_tfidf, y_train)

        training_time = time.time() - start_time

        y_val_pred = model.predict(X_val_tfidf)

        metrics = evaluate_predictions(y_val, y_val_pred)

        supports_probability = hasattr(model, "predict_proba")

        trained_models[model_name] = model

        row = {
            "model_name": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "training_time_seconds": training_time,
            "supports_probability": supports_probability,
        }

        comparison_rows.append(row)

        detailed_results[model_name] = {
            "validation_metrics": metrics,
            "training_time_seconds": training_time,
            "supports_probability": supports_probability,
            "confusion_matrix": confusion_matrix(y_val, y_val_pred).tolist(),
            "classification_report": classification_report(
                y_val,
                y_val_pred,
                target_names=["Fake News", "Real News"],
            ),
        }

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1_score']:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Supports probability/confidence: {supports_probability}")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values(by="f1_score", ascending=False)

    return trained_models, comparison_df, detailed_results


def choose_best_model(comparison_df):
    """
    Selects the best model using validation F1-score.

    By default, this chooses the highest validation F1-score.
    However, if the calibrated Ridge Classifier is extremely close
    to the best model, it is selected because it supports predict_proba(),
    which allows the Streamlit app to show a confidence score.

    Rule:
        If calibrated Ridge is within 0.001 F1-score of the best model,
        choose calibrated Ridge.
    """

    best_row = comparison_df.iloc[0]
    best_model_name = best_row["model_name"]
    best_f1 = best_row["f1_score"]

    calibrated_name = "calibrated_ridge_classifier"

    calibrated_rows = comparison_df[comparison_df["model_name"] == calibrated_name]

    if not calibrated_rows.empty:
        calibrated_f1 = calibrated_rows.iloc[0]["f1_score"]

        if best_model_name != calibrated_name and (best_f1 - calibrated_f1) <= 0.001:
            print("\nCalibrated Ridge Classifier selected.")
            print("Reason: Its F1-score is within 0.001 of the best model and it supports confidence scores.")
            return calibrated_name

    return best_model_name


def evaluate_best_on_test(best_model, X_test_tfidf, y_test):
    """
    Evaluates the selected best model on the untouched test set.
    """

    y_test_pred = best_model.predict(X_test_tfidf)

    test_metrics = evaluate_predictions(y_test, y_test_pred)

    supports_probability = hasattr(best_model, "predict_proba")

    test_results = {
        "test_metrics": test_metrics,
        "supports_probability": supports_probability,
        "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_test_pred,
            target_names=["Fake News", "Real News"],
        ),
    }

    return test_results


def save_comparison_results(comparison_df, detailed_results, best_model_name, test_results):
    """
    Saves model comparison results as CSV and JSON.
    """

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(MODEL_COMPARISON_CSV, index=False)

    final_results = {
        "best_model_selected": best_model_name,
        "selection_method": (
            "Highest validation F1-score. If calibrated Ridge is within 0.001 "
            "of the best F1-score, calibrated Ridge is preferred because it supports confidence scores."
        ),
        "validation_results": detailed_results,
        "final_test_results_for_best_model": test_results,
    }

    with open(MODEL_COMPARISON_JSON, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    print(f"\nModel comparison CSV saved to: {MODEL_COMPARISON_CSV}")
    print(f"Detailed model comparison JSON saved to: {MODEL_COMPARISON_JSON}")


def save_model_comparison_charts(comparison_df):
    """
    Saves model comparison charts to reports/figures.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sorted_df = comparison_df.sort_values(by="f1_score", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_df["model_name"], sorted_df["f1_score"])
    plt.xlabel("Validation F1-score")
    plt.ylabel("Model")
    plt.title("Model Comparison by Validation F1-score")
    plt.tight_layout()

    f1_chart_path = FIGURES_DIR / "model_f1_scores.png"
    plt.savefig(f1_chart_path, dpi=300)
    plt.close()

    print(f"Model F1-score chart saved to: {f1_chart_path}")

    time_sorted_df = comparison_df.sort_values(
        by="training_time_seconds",
        ascending=True,
    )

    plt.figure(figsize=(10, 6))
    plt.barh(time_sorted_df["model_name"], time_sorted_df["training_time_seconds"])
    plt.xlabel("Training Time in Seconds")
    plt.ylabel("Model")
    plt.title("Model Comparison by Training Time")
    plt.tight_layout()

    time_chart_path = FIGURES_DIR / "model_training_times.png"
    plt.savefig(time_chart_path, dpi=300)
    plt.close()

    print(f"Model training-time chart saved to: {time_chart_path}")


def save_best_model(best_model, vectorizer, best_model_name):
    """
    Saves the selected best model and TF-IDF vectorizer.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print("\nBest model saved successfully.")
    print(f"Best model name: {best_model_name}")
    print(f"Supports probability/confidence: {hasattr(best_model, 'predict_proba')}")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Vectorizer saved to: {VECTORIZER_FILE}")


def main():
    df = load_processed_dataset()

    df = clean_dataframe(df, text_column="combined_text")

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(df)

    vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf = vectorize_text(
        X_train,
        X_val,
        X_test,
    )

    trained_models, comparison_df, detailed_results = train_and_compare_models(
        X_train_tfidf,
        X_val_tfidf,
        y_train,
        y_val,
    )

    print("\n" + "=" * 70)
    print("Model Comparison Based on Validation F1-score")
    print("=" * 70)
    print(comparison_df)

    best_model_name = choose_best_model(comparison_df)
    best_model = trained_models[best_model_name]

    test_results = evaluate_best_on_test(best_model, X_test_tfidf, y_test)

    print("\n" + "=" * 70)
    print("Final Test Results for Selected Best Model")
    print("=" * 70)
    print(f"Best model: {best_model_name}")
    print(f"Supports probability/confidence: {test_results['supports_probability']}")
    print(f"Test Accuracy : {test_results['test_metrics']['accuracy']:.4f}")
    print(f"Test Precision: {test_results['test_metrics']['precision']:.4f}")
    print(f"Test Recall   : {test_results['test_metrics']['recall']:.4f}")
    print(f"Test F1-score : {test_results['test_metrics']['f1_score']:.4f}")

    save_comparison_results(
        comparison_df,
        detailed_results,
        best_model_name,
        test_results,
    )

    save_model_comparison_charts(comparison_df)

    save_best_model(best_model, vectorizer, best_model_name)


if __name__ == "__main__":
    main()