def main():
    print("=" * 70)
    print("Fake News Detection using NLP and Machine Learning")
    print("=" * 70)

    print("\nAvailable commands:")

    print("\n1. Prepare dataset:")
    print("   python -m src.data_loader")

    print("\n2. Run exploratory data analysis:")
    print("   python -m src.eda")

    print("\n3. Train and compare TF-IDF machine learning models:")
    print("   python -m src.train_tfidf_models")

    print("\n4. Evaluate saved model:")
    print("   python -m src.evaluate")

    print("\n5. Predict using saved model:")
    print("   python -m src.predict")

    print("\n6. Run Streamlit app:")
    print("   streamlit run app/streamlit_app.py")

    print("\nRecommended order:")
    print("   python -m src.data_loader")
    print("   python -m src.eda")
    print("   python -m src.train_tfidf_models")
    print("   python -m src.evaluate")
    print("   streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()