import re
import html
import string


def clean_text(text):
    """
    Cleans raw news text for traditional machine learning models.

    This cleaning is designed for TF-IDF based models, not BERT.
    For BERT/DistilBERT, we will later use minimal cleaning only.
    """

    if text is None:
        return ""

    text = str(text)

    # Convert HTML entities such as &amp; to normal text
    text = html.unescape(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_dataframe(df, text_column="combined_text"):
    """
    Applies clean_text() to a dataframe column.
    Creates a new column called clean_text.
    """

    if text_column not in df.columns:
        raise ValueError(f"Column not found: {text_column}")

    print("Cleaning text data...")

    df["clean_text"] = df[text_column].apply(clean_text)

    before_empty = len(df)
    df = df[df["clean_text"] != ""]
    after_empty = len(df)

    print(f"Removed rows with empty cleaned text: {before_empty - after_empty}")

    return df