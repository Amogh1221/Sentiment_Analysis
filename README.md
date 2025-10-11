# Mental Health Status Classifier

A machine learning project that classifies textual statements into mental health categories using NLP and supervised learning models.

---

## Overview

This project uses natural language processing (NLP) techniques to preprocess text, TF-IDF vectorization for feature extraction, and an **ExtraTreesClassifier** to predict mental health status. It is deployed as a **Streamlit app** for real-time inference.

---

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/mental-health-classifier.git
    cd mental-health-classifier
    ```

2.  Create and activate a virtual environment:

    ```bash
    # Windows
    python -m venv sa
    sa\Scripts\activate

    # Linux/Mac
    python -m venv sa
    source sa/bin/activate
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2.  Enter a statement in the text box.

3.  Click `Classify` to see the predicted mental health status.

## Dataset

*   Source file: `Combined_Data.csv`
*   Cleaned and balanced subset of 6,000 statements used for training.
*   Preprocessing includes:
    *   Lowercasing
    *   Removing non-alphabetic characters
    *   Tokenization & stopword removal

## Model

*   Feature extraction: TF-IDF vectorization
*   Model: ExtraTreesClassifier
*   Label encoding used for output classes

## Results

*   Test Accuracy: 92.42%
*   Training Accuracy: 99.96%

## Files Included

*   `model.pkl` – trained ExtraTreesClassifier model
*   `tfidf.pkl` – TF-IDF vectorizer
*   `encoder.pkl` – Label encoder for output classes
*   `app.py` – Streamlit application
*   `cleaned_data_SA.csv` – preprocessed dataset

## Future Improvements

*   Use transformer-based embeddings (BERT, RoBERTa) for better contextual understanding
*   Expand dataset for more robust predictions
*   Add multi-class classification for specific mental health conditions
*   Integrate with web or mobile platforms for real-time mental health monitoring
