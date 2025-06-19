from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

app = Flask(__name__)
CORS(app)

# Load the lyrics_analyze DataFrame if not already loaded
model_pipeline_path = './Model/logistic_regression_pipeline.joblib'
model_label_path = './Model/label_to_genre.joblib'
if not os.path.exists(model_pipeline_path) or not os.path.exists(model_label_path):
    raise FileNotFoundError("Model files not found. Please run the save_model.py script first.")

print(os.path)

try:
    model_pipeline = joblib.load('./Model/logistic_regression_pipeline.joblib')
    label_to_genre = joblib.load('./Model/label_to_genre.joblib')
    print("Model and label mapping loaded successfully.")
except FileNotFoundError:
    print("Model files not found! Please run the save_model.py script first.")
    model_pipeline = None
    label_to_genre = None

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

profanity_list = set([
    'arse', 'arsehole', 'ass', 'asshat', 'asshole', 'bastard', 'bitch', 'bloody', 'bollocks', 'bugger',
    'bullshit', 'cock', 'crap', 'cunt', 'damn', 'dick', 'dickhead', 'fag', 'faggot', 'fuck', 'fucked',
    'fucker', 'fucking', 'hell', 'horseshit', 'motherfucker', 'nigga', 'piss', 'prick', 'pussy', 'shit',
    'slut', 'twat', 'wanker', 'whore'
])

def get_sentiment_label(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# --- Define API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model_pipeline or not label_to_genre:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    data = request.get_json()
    if not data or 'lyrics' not in data:
        return jsonify({'error': 'Invalid input: "lyrics" key not found.'}), 400

    lyrics = data['lyrics']
    
    predicted_label_index = model_pipeline.predict([lyrics])[0]
    predicted_genre = label_to_genre[predicted_label_index]
    
    probabilities = model_pipeline.predict_proba([lyrics])[0]
    genre_probabilities = {label_to_genre[i]: prob for i, prob in enumerate(probabilities)}
    
    sentiment_scores = vader.polarity_scores(lyrics)
    compound_score = sentiment_scores['compound']
    sentiment_label = get_sentiment_label(compound_score)
    
    words = set(re.findall(r'\b\w+\b', lyrics.lower()))
    explicit_words_found = [word for word in words if word in profanity_list]
    explicit_word_count = len(explicit_words_found)
    explicitness_label = 'Explicit' if explicit_word_count > 0 else 'Not Explicit'

    response = {
        'predictedGenre': predicted_genre,
        'genreProbabilities': genre_probabilities,
        'sentimentLabel': sentiment_label,
        'sentimentScore': compound_score,
        'explicitnessLabel': explicitness_label,
        'explicitWordCount': explicit_word_count,
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
