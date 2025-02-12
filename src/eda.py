import pandas as pd
import matplotlib.pyplot as plt

from src.config import PROCESSED_FILE, REPORTS_DIR, FIGURES_DIR
from src.preprocess_text import clean_text


def load_data():
    """
    Loads the processed dataset created by data_loader.py.
    """

    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {PROCESSED_FILE}\n"
            "Run this first: python -m src.data_loader"
        )

    df = pd.read_csv(PROCESSED_FILE)

    required_columns = ["title", "text", "combined_text", "label"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    return df


def add_length_features(df):
    """
    Adds simple text length features for EDA.
    """

    df = df.copy()

    df["clean_text"] = df["combined_text"].apply(clean_text)

    df["title_length_words"] = df["title"].fillna("").astype(str).apply(
        lambda x: len(x.split())
    )

    df["article_length_words"] = df["text"].fillna("").astype(str).apply(
        lambda x: len(x.split())
    )

    df["combined_length_words"] = df["combined_text"].fillna("").astype(str).apply(
        lambda x: len(x.split())
    )

    df["clean_length_words"] = df["clean_text"].apply(
        lambda x: len(x.split())
    )

    return df


def save_label_distribution(df):
    """
    Saves a bar chart showing real vs fake label distribution.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    label_counts = df["label"].value_counts().sort_index()

    label_names = ["Fake News", "Real News"]

    plt.figure(figsize=(7, 5))
    plt.bar(label_names, label_counts.values)
    plt.xlabel("Label")
    plt.ylabel("Number of Articles")
    plt.title("Label Distribution")
    plt.tight_layout()

    output_path = FIGURES_DIR / "label_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved label distribution chart to: {output_path}")


def save_article_length_distribution(df):
    """
    Saves a histogram showing article length distribution.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.hist(df["article_length_words"], bins=50)
    plt.xlabel("Article Length in Words")
    plt.ylabel("Number of Articles")
    plt.title("Article Length Distribution")
    plt.tight_layout()

    output_path = FIGURES_DIR / "article_length_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved article length distribution chart to: {output_path}")


def save_average_article_length_by_label(df):
    """
    Saves a bar chart comparing average article length for fake and real news.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    avg_lengths = df.groupby("label")["article_length_words"].mean().sort_index()

    label_names = ["Fake News", "Real News"]

    plt.figure(figsize=(7, 5))
    plt.bar(label_names, avg_lengths.values)
    plt.xlabel("Label")
    plt.ylabel("Average Article Length in Words")
    plt.title("Average Article Length by Label")
    plt.tight_layout()

    output_path = FIGURES_DIR / "average_article_length_by_label.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved average article length chart to: {output_path}")


def save_eda_summary(df):
    """
    Saves a text summary of important dataset statistics.
    """

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    label_counts = df["label"].value_counts().sort_index()

    fake_count = label_counts.get(0, 0)
    real_count = label_counts.get(1, 0)

    summary_path = REPORTS_DIR / "eda_summary.txt"

    summary = f"""
Fake News Detection - EDA Summary
=================================

Dataset Shape
-------------
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Label Distribution
------------------
Fake News: {fake_count}
Real News: {real_count}

Article Length Statistics
-------------------------
Minimum article length: {df['article_length_words'].min():.0f} words
Maximum article length: {df['article_length_words'].max():.0f} words
Average article length: {df['article_length_words'].mean():.2f} words
Median article length: {df['article_length_words'].median():.2f} words

Cleaned Text Length Statistics
------------------------------
Minimum cleaned length: {df['clean_length_words'].min():.0f} words
Maximum cleaned length: {df['clean_length_words'].max():.0f} words
Average cleaned length: {df['clean_length_words'].mean():.2f} words
Median cleaned length: {df['clean_length_words'].median():.2f} words

Average Article Length by Label
-------------------------------
Fake News: {df[df['label'] == 0]['article_length_words'].mean():.2f} words
Real News: {df[df['label'] == 1]['article_length_words'].mean():.2f} words

Notes
-----
This EDA is used to understand class balance and text length patterns before model training.
High model performance should still be interpreted carefully because labelled fake news datasets may contain source-specific or topic-specific patterns.
"""

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary.strip())

    print(f"Saved EDA summary to: {summary_path}")


def main():
    print("=" * 70)
    print("Running Exploratory Data Analysis")
    print("=" * 70)

    df = load_data()

    print("Loaded dataset shape:", df.shape)

    df = add_length_features(df)

    save_label_distribution(df)
    save_article_length_distribution(df)
    save_average_article_length_by_label(df)
    save_eda_summary(df)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()