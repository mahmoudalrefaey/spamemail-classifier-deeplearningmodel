# Main program for SpamSentry app
# This program uses Streamlit as the web interface and TensorFlow as the
# machine learning backend.

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import re
import pandas as pd
from collections import Counter
from tensorflow.keras.models import load_model
import joblib


def scale_email(vector):
    """
    Scale the email vector using the standard scaler from the training data.
    """
    scaler = joblib.load('main\standard_scaler.pkl')
    email_vector = vector.reshape(1, -1)  # Reshape for scaler
    email_vector_scaled = scaler.transform(email_vector)
    return email_vector_scaled
    
def text_to_vector(text):
    """
    Convert the text to a vector of word frequencies.
    """
    data = pd.read_csv('dataset\emails.csv')
    data = data.drop(['Prediction'], axis=1)
    
    word_list = list(data.columns[:-1])
    # Clean text: Remove punctuation, lowercase, and split into words
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    words = text.lower().split()
    
    # Count word occurrences
    word_counts = Counter(words)
    
    # Create a word frequency vector
    vector = np.zeros(len(word_list), dtype=np.int32)
    for i, word in enumerate(word_list):
        if word in word_counts:
            vector[i] = word_counts[word]
    return vector

def add_textbox():
    """
    Add a text box to the web page and a submit button.
    """
    st.markdown("<h3 style='text-align: center; color: white;'>Enter the email text</h3>", unsafe_allow_html=True)
    text = st.text_area(label="Email Text", height=150)
    submit_button = st.button("Submit")
    if submit_button:
        st.session_state['text'] = text
        prediction_page()
    
def page_home():
    """
    Home page of the app: Welcome message and a text box to enter the email text.
    """
    st.set_page_config(
        page_title="SpamSentry",
        page_icon=":email:", 
        layout="centered")
    st.markdown("<h3 style='text-align: center; color: white;'>Welcome to SpamSentry - Spam Email Classifier</h3>", unsafe_allow_html=True)
    add_textbox()
    return

def prediction_page():
    """
    Prediction page: Display the prediction results.
    """
    st.write(" ")
    st.write(" ")
    st.markdown("<h4 style='text-align: center; color: white;'>Prediction Results</h4>", unsafe_allow_html=True)
    scaled_emails = scale_email(text_to_vector(st.session_state['text']))
    model = load_model('main\spam_classifier_model.h5')
    prediction = model.predict(scaled_emails)
    if prediction[0] >= 0.5:
        st.write("<span class='not-spam'> Not Spammy Email </span>", unsafe_allow_html=True)
    elif prediction[0] <= 0.5:
        st.write("<span class='spam'>Spammy Email </span>", unsafe_allow_html=True)
        
    st.write("**Confidence of being ham email:**","{:.2f}".format(prediction[0][0] * 100),"%")

def main():
    """
    Main function of the program.
    """
    page_home()
    with open("style\style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
if __name__ == '__main__':
    main()

