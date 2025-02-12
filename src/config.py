from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRUE_FILE = RAW_DATA_DIR / "True.csv"
FAKE_FILE = RAW_DATA_DIR / "Fake.csv"

PROCESSED_FILE = PROCESSED_DATA_DIR / "cleaned_news.csv"

MODEL_FILE = MODELS_DIR / "best_model.joblib"
VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.joblib"

MODEL_COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
MODEL_COMPARISON_JSON = REPORTS_DIR / "model_comparison.json"

RANDOM_STATE = 42

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15