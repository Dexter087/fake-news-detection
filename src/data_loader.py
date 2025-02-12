import pandas as pd

from src.config import (
    TRUE_FILE,
    FAKE_FILE,
    PROCESSED_DATA_DIR,
    PROCESSED_FILE,
)


def load_raw_data():
    """
    Loads True.csv and Fake.csv from data/raw.
    Adds labels:
        Real news = 1
        Fake news = 0
    """

    print("Loading raw dataset...")

    if not TRUE_FILE.exists():
        raise FileNotFoundError(f"True.csv not found at: {TRUE_FILE}")

    if not FAKE_FILE.exists():
        raise FileNotFoundError(f"Fake.csv not found at: {FAKE_FILE}")

    true_df = pd.read_csv(TRUE_FILE)
    fake_df = pd.read_csv(FAKE_FILE)

    true_df["label"] = 1
    fake_df["label"] = 0

    print(f"True news shape: {true_df.shape}")
    print(f"Fake news shape: {fake_df.shape}")

    df = pd.concat([true_df, fake_df], axis=0, ignore_index=True)

    return df


def prepare_dataset(df):
    """
    Cleans the dataset at a basic structural level:
    - Checks required columns
    - Handles missing title/text
    - Creates combined_text
    - Removes empty rows
    - Removes duplicates
    """

    print("Preparing dataset...")

    required_columns = ["title", "text", "label"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    df["combined_text"] = df["title"].astype(str) + " " + df["text"].astype(str)

    df["combined_text"] = df["combined_text"].str.strip()

    before_empty = len(df)
    df = df[df["combined_text"] != ""]
    after_empty = len(df)

    print(f"Removed empty rows: {before_empty - after_empty}")

    before_duplicates = len(df)
    df = df.drop_duplicates(subset=["combined_text"])
    after_duplicates = len(df)

    print(f"Removed duplicate rows: {before_duplicates - after_duplicates}")

    df = df[["title", "text", "combined_text", "label"]]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Final dataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    return df


def save_processed_data(df):
    """
    Saves the cleaned dataset to data/processed/cleaned_news.csv
    """

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_FILE, index=False)

    print(f"\nProcessed dataset saved to: {PROCESSED_FILE}")


def main():
    raw_df = load_raw_data()
    processed_df = prepare_dataset(raw_df)
    save_processed_data(processed_df)


if __name__ == "__main__":
    main()