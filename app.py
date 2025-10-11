import streamlit as st
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)
with open('label_encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = nltk.corpus.stopwords.words('english')

def cleaning(text):
    preprocessed = text.lower()
    preprocessed = re.sub(r"[^a-zA-Z\s]", "", preprocessed)
    words = nltk.word_tokenize(preprocessed)
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

st.title("Mental Health Status")

user_input = st.text_area("Enter your statement here:")

if st.button("Classify"):
    if user_input:
        cleaned_input = cleaning(user_input)
        input_vector = tfidf.transform([cleaned_input])

        prediction_encoded = model.predict(input_vector)
        prediction_label = encoder.inverse_transform(prediction_encoded)

        st.write("**Predicted Status:**", prediction_label[0])
    else:
        st.warning("Please enter a statement to classify.")
