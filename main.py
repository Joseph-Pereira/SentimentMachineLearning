import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import joblib
import string
import re  
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
from num2words import num2words 
import os

# --- Load Vectorizer and Model ---
vectorizer = joblib.load('artifacts/vectorizer.pickle')
sentiment_clf = joblib.load('artifacts/sentiment_clf.pickle')

# For storing last prediction
last_prediction = {"review": "", "sentiment": ""}

# --- Define the Dash App ---
app = dash.Dash(__name__, external_stylesheets=["/assets/style.css"])

# --- App Layout ---
app.layout = html.Div(
    className="body-container",
    children=[
        html.Div(
            className="container",
            children=[
                html.H1("\U0001F310Sentiment Analyzer"),
                html.Label("Enter your review:"),
                dcc.Textarea(
                    id="review-input",
                    placeholder="Write your review here...",
                    className="text-area",
                    style={"border": "2px solid transparent"}
                ),
                html.Button("Analyze Sentiment", id="analyze-button", n_clicks=0),
                html.Div([
                    html.Label("Predicted Sentiment:"),
                    html.Div(id="sentiment-output", className="sentiment-display")
                ]),
                html.Div([
                    html.Label("Full Output:"),
                    html.Div(id="full-output", className="full-display")
                ]),
                html.Button("Save Results", id="save-button", n_clicks=0, className="save-button"),
                html.Div(id="save-confirmation", className="save-confirmation")
            ]
        )
    ]
)

# --- Callbacks ---
@app.callback(
    Output("sentiment-output", "children"),
    Output("full-output", "children"),
    Output("review-input", "style"),
    Input("analyze-button", "n_clicks"),
    State("review-input", "value"),
)
def analyze_sentiment(n_clicks, review_text):
    global last_prediction
    if n_clicks > 0:
        if not review_text or review_text.strip() == "":
            return "\u26a0\ufe0f Please enter a review before analyzing.", "", {"border": "2px solid red"}

        cleaned_review_text = clean_text(review_text)
        X_vec = vectorizer.transform([cleaned_review_text])
        prediction = sentiment_clf.predict(X_vec)
        probabilities = sentiment_clf.predict_proba(X_vec)[0]
        sentiment_labels = sentiment_clf.classes_

        last_prediction["review"] = review_text
        last_prediction["sentiment"] = prediction[0]

        full_output_text = "Probabilities:\n"
        for i, label in enumerate(sentiment_labels):
            full_output_text += f"{label}: {probabilities[i]:.4f}\n"
        full_output_text += f"\nPredicted Class: {prediction[0]}"

        return prediction[0], full_output_text, {"border": "2px solid green"}
    return "", "", {"border": "2px solid transparent"}


@app.callback(
    Output("save-confirmation", "children"),
    Input("save-button", "n_clicks"),
    State("review-input", "value")
)
def save_results(n_clicks, review_text):
    if n_clicks > 0:
        if not review_text or review_text.strip() == "":
            return "\u26a0\ufe0f Cannot save an empty review."

        sentiment = last_prediction.get("sentiment", "").lower()
        review = last_prediction.get("review", "")

        if sentiment == "":
            return "\u26a0\ufe0f Please analyze the review before saving."

        filename = "positive_reviews.txt" if sentiment in ["positive", "neutral"] else "negative_reviews.txt"

        if os.path.exists(filename):
            with open(filename, "a", encoding="utf-8") as file:
                file.write(f"{review}\n\n")
            return f"\u2705 Review saved to {filename}."
        else:
            return f"\u26a0\ufe0f File '{filename}' not found. Make sure it exists."
    return ""

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

# --- Run the App ---
if __name__ == "__main__":
    app.run(debug=True)
