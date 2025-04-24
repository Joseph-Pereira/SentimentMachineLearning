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
from nltk.stem import PorterStemmer
from num2words import num2words 

# --- 1. Load the Model and Data/Tokenizer ---
# Load the model
model = tf.keras.models.load_model('artifacts/sentiment_model.h5')  

# Load the tokenizer and label binarizer. Important to be consistent with training.
# Assuming these were saved, or are available from your training environment
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

# Load your training data.  This is crucial for fitting the LabelBinarizer correctly.
# **Replace 'your_training_data.csv' with the actual path to your training data.**
try:
    df = pd.read_csv('Notebooks/reviews_cleaned.csv')  # Or wherever your data is
except FileNotFoundError:
    print("Error: 'reviews_cleaned.csv' not found.  Please make sure the file is in the correct location.")
    #  You might want to exit the program here or provide a default DataFrame if the program can continue.
    exit()

# Ensure 'Sentiment_Label' is in the DataFrame
if 'Sentiment_Label' not in df.columns:
    print("Error: 'Sentiment_Label' column not found in the DataFrame.")
    exit()

label_binarizer = LabelBinarizer()
label_binarizer.fit(df['Sentiment_Label'])  # Fit on the original training data

MAX_LEN = 200  # Consistent with your training

# --- 2. Define the Dash App ---
app = dash.Dash(__name__)

# --- 3. App Layout ---
app.layout = html.Div(
    [
        html.H1("Sentiment Analysis Web App", style={'textAlign': 'center', 'margin-bottom': '20px'}),
        html.Div(
            [
                html.Label("Enter your review:", style={'font-weight': 'bold'}),
                dcc.Textarea(
                    id="review-input",
                    placeholder="Write your review here...",
                    style={'width': '100%', 'height': '150px', 'margin-bottom': '10px'},
                ),
                html.Button("Analyze Sentiment", id="analyze-button", n_clicks=0, style={'background-color': '#4CAF50', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'cursor': 'pointer'}),
            ],
            style={'margin-bottom': '20px', 'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px'}
        ),
        html.Div(
            [
                html.Label("Predicted Sentiment:", style={'font-weight': 'bold'}),
                html.Div(id="sentiment-output", style={'font-size': '20px', 'margin-top': '10px', 'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px', 'background-color': '#f0f0f0'}),
            ],
            style={'margin-bottom': '20px'}
        ),
        html.Div(
            [
                html.Label("Full Output:", style={'font-weight': 'bold'}),
                html.Div(id="full-output", style={'font-size': '12px', 'margin-top': '10px', 'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px', 'background-color': '#ffffff', 'overflow-wrap': 'break-word'}),
            ],
            style={'margin-bottom': '20px'}
        ),
    ],
    style={'padding': '20px', 'font-family': 'Arial'}
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
        # Applying the same method to clean the text
        cleaned_review_text = clean_text(review_text)  # Clean the input review
        review_sequence = tokenizer.texts_to_sequences([cleaned_review_text])  # Tokenize the cleaned review

        # Preprocess the input review

        review_padded = pad_sequences(review_sequence, maxlen=MAX_LEN, truncating='post')

        # Check if the padded sequence is empty
        if len(review_padded) == 0:
            return "No valid tokens found in review.", ""

        # Make prediction
        prediction = model.predict(review_padded)
        predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the index of the max probability
        predicted_class_label = label_binarizer.classes_[predicted_class_index]

        # Get detailed output
        sentiment_labels = label_binarizer.classes_
        probabilities = prediction[0]

        full_output_text = "Probabilities:\n"
        for i, label in enumerate(sentiment_labels):
            full_output_text += f"{label}: {probabilities[i]:.4f}\n"
        full_output_text += f"\nPredicted Class: {predicted_class_label}"

        return predicted_class_label, full_output_text
    else:
        return "Enter a review to analyze.", ""  # Default message
    
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    if text is None:
        return ""
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = str(text).lower()
    text = re.sub(r'\d+', lambda match: num2words(int(match.group())), text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join([word for word in text.split() if word not in stop_words])
    # stemmer = PorterStemmer()
    return text

# --- 5. Run the App ---
if __name__ == "__main__":
    app.run(debug=True)
