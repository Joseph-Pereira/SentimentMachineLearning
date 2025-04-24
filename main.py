import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
import string
import re  
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
from num2words import num2words 
import os

# --- 1. Load the Model and Data/Tokenizer ---
model = tf.keras.models.load_model('artifacts/sentiment_model.h5')  
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

try:
    df = pd.read_csv('Notebooks/reviews_cleaned.csv')
except FileNotFoundError:
    print("Error: 'reviews_cleaned.csv' not found.")
    exit()

if 'Sentiment_Label' not in df.columns:
    print("Error: 'Sentiment_Label' column not found in the DataFrame.")
    exit()

label_binarizer = LabelBinarizer()
label_binarizer.fit(df['Sentiment_Label'])

MAX_LEN = 200

# --- 2. Define the Dash App ---
app = dash.Dash(__name__, external_stylesheets=["/assets/style.css"])

# --- 3. App Layout ---
app.layout = html.Div(
    className="body-container",
    children=[
        html.Div(
            className="container",
            children=[
                html.H1("🌐Sentiment Analyzer"),
                html.Label("Enter your review:"),
                dcc.Textarea(
                    id="review-input",
                    placeholder="Write your review here...",
                    className="text-area"
                ),
                html.Button("Analyze Sentiment", id="analyze-button", n_clicks=0),
                html.Div(
                    [
                        html.Label("Predicted Sentiment:"),
                        html.Div(id="sentiment-output", className="sentiment-display")
                    ]
                ),
                html.Div(
                    [
                        html.Label("Full Output:"),
                        html.Div(id="full-output", className="full-display")
                    ]
                ),

                
            ]
        )
    ]
)


# --- 4. Callback Function ---
@app.callback(
    Output("sentiment-output", "children"),
    Output("full-output", "children"),
    Input("analyze-button", "n_clicks"),
    State("review-input", "value"),
)
def analyze_sentiment(n_clicks, review_text):
    if n_clicks > 0 and review_text:
        cleaned_review_text = clean_text(review_text)
        review_sequence = tokenizer.texts_to_sequences([cleaned_review_text])
        review_padded = pad_sequences(review_sequence, maxlen=MAX_LEN, truncating='post')

        if len(review_padded) == 0:
            return "No valid tokens found in review.", ""

        prediction = model.predict(review_padded)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_label = label_binarizer.classes_[predicted_class_index]

        sentiment_labels = label_binarizer.classes_
        probabilities = prediction[0]

        full_output_text = "Probabilities:\n"
        for i, label in enumerate(sentiment_labels):
            full_output_text += f"{label}: {probabilities[i]:.4f}\n"
        full_output_text += f"\nPredicted Class: {predicted_class_label}"

        return predicted_class_label, full_output_text
    else:
        return "Enter a review to analyze.", ""

# --- 5. Text Cleaning Helper Function ---
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    if text is None:
        return ""
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = str(text).lower()
    text = re.sub(r'\d+', lambda match: num2words(int(match.group())), text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

nltk.download('stopwords', download_dir='/tmp')
nltk.data.path.append('/tmp')

# --- 6. Run the App ---
if __name__ == "__main__":
    app.run(debug=True)
