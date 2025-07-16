from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import os
from src.model import GenreClassifier
from src.preprocessing import get_sentiment_label

app = Flask(__name__)
CORS(app)

# Initialize sentiment analyzer
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

# Initialize genre classifier
genre_classifier = GenreClassifier()

# Load models on startup
print("Loading models...")
genre_classifier.load_models()

if not genre_classifier.is_loaded:
    print("Warning: Models failed to load. Check model files.")

# Profanity word list for explicit content detection
profanity_list = set([
    'arse', 'arsehole', 'ass', 'asshat', 'asshole', 'bastard', 'bitch', 'bloody', 'bollocks', 'bugger',
    'bullshit', 'cock', 'crap', 'cunt', 'damn', 'dick', 'dickhead', 'fag', 'faggot', 'fuck', 'fucked',
    'fucker', 'fucking', 'hell', 'horseshit', 'motherfucker', 'nigga', 'piss', 'prick', 'pussy', 'shit',
    'slut', 'twat', 'wanker', 'whore'
])


@app.route('/')
def home():
    """
    Serve the main application page
    """
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint for genre classification
    """
    if not genre_classifier.is_loaded:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    data = request.get_json()
    if not data or 'lyrics' not in data:
        return jsonify({'error': 'Invalid input: "lyrics" key not found.'}), 400

    lyrics = data['lyrics']
    
    if not lyrics or len(lyrics.strip()) == 0:
        return jsonify({'error': 'Empty lyrics provided.'}), 400
    
    try:
        # Genre prediction using new model
        genre_result = genre_classifier.predict(lyrics)
        
        # Sentiment analysis
        sentiment_scores = vader.polarity_scores(lyrics)
        compound_score = sentiment_scores['compound']
        sentiment_label = get_sentiment_label(compound_score)
        
        # Explicit content detection
        words = set(re.findall(r'\b\w+\b', lyrics.lower()))
        explicit_words_found = [word for word in words if word in profanity_list]
        explicit_word_count = len(explicit_words_found)
        explicitness_label = 'Explicit' if explicit_word_count > 0 else 'Not Explicit'

        response = {
            'predictedGenre': genre_result['predicted_genre'],
            'confidence': genre_result['confidence'],
            'genreProbabilities': genre_result['genre_probabilities'],
            'sentimentLabel': sentiment_label,
            'sentimentScore': compound_score,
            'explicitnessLabel': explicitness_label,
            'explicitWordCount': explicit_word_count,
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': genre_classifier.is_loaded
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
