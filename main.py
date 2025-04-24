import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import pickle
import string
import re  
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from num2words import num2words 

# --- Load Vectorizer and Sklearn Model ---
with open("artifacts/vectorizer.pickle", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("artifacts/sentiment_clf.pickle", "rb") as clf_file:
    sentiment_clf = pickle.load(clf_file)

# --- Load Data for Label Reference ---
try:
    df = pd.read_csv('Notebooks/reviews_cleaned.csv')
except FileNotFoundError:
    print("Error: 'reviews_cleaned.csv' not found.")
    exit()

if 'Sentiment_Label' not in df.columns:
    print("Error: 'Sentiment_Label' column not found in the DataFrame.")
    exit()

# --- Define the Dash App ---
app = dash.Dash(__name__, external_stylesheets=["/assets/style.css"])

# --- App Layout ---
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

# --- Text Cleaning Helper Function ---
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

# --- Callback Function ---
@app.callback(
    Output("sentiment-output", "children"),
    Output("full-output", "children"),
    Input("analyze-button", "n_clicks"),
    State("review-input", "value"),
)
def analyze_sentiment(n_clicks, review_text):
    if n_clicks > 0 and review_text:
        cleaned_review_text = clean_text(review_text)
        X_vec = vectorizer.transform([cleaned_review_text])
        prediction = sentiment_clf.predict(X_vec)
        probabilities = sentiment_clf.predict_proba(X_vec)[0]
        sentiment_labels = sentiment_clf.classes_

        full_output_text = "Probabilities:\n"
        for i, label in enumerate(sentiment_labels):
            full_output_text += f"{label}: {probabilities[i]:.4f}\n"
        full_output_text += f"\nPredicted Class: {prediction[0]}"

        return prediction[0], full_output_text
    else:
        return "Enter a review to analyze.", ""

# --- Run the App ---
if __name__ == "__main__":
    app.run(debug=True)
