from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import os
from src.model import GenreClassifier
from src.legacy_model import LegacyClassifier
from src.preprocessing import get_sentiment_label
from src.audio_processor import AudioProcessor

app = Flask(__name__)
CORS(app)

# Initialize sentiment analyzer
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

# Initialize both classifiers and audio processor
new_classifier = GenreClassifier()
legacy_classifier = LegacyClassifier()
audio_processor = AudioProcessor()

# Load models on startup
print("Loading new hybrid model...")
new_classifier.load_models()

print("Loading legacy model...")
legacy_classifier.load_models()

if not new_classifier.is_loaded:
    print("Warning: New model failed to load. Check model files.")

if not legacy_classifier.is_loaded:
    print("Warning: Legacy model failed to load. Check model files.")

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
    Main prediction endpoint for genre classification using both models
    """
    if not new_classifier.is_loaded and not legacy_classifier.is_loaded:
        return jsonify({'error': 'No models loaded. Please check server logs.'}), 500

    data = request.get_json()
    if not data or 'lyrics' not in data:
        return jsonify({'error': 'Invalid input: "lyrics" key not found.'}), 400

    lyrics = data['lyrics']
    
    if not lyrics or len(lyrics.strip()) == 0:
        return jsonify({'error': 'Empty lyrics provided.'}), 400
    
    try:
        results = {}
        
        # Get predictions from both models if available
        if new_classifier.is_loaded:
            new_result = new_classifier.predict(lyrics)
            results['new_model'] = new_result
        
        if legacy_classifier.is_loaded:
            legacy_result = legacy_classifier.predict(lyrics)
            results['legacy_model'] = legacy_result
        
        # Sentiment analysis
        sentiment_scores = vader.polarity_scores(lyrics)
        compound_score = sentiment_scores['compound']
        sentiment_label = get_sentiment_label(compound_score)
        
        # Explicit content detection
        words = set(re.findall(r'\b\w+\b', lyrics.lower()))
        explicit_words_found = [word for word in words if word in profanity_list]
        explicit_word_count = len(explicit_words_found)
        explicitness_label = 'Explicit' if explicit_word_count > 0 else 'Not Explicit'

        # Prepare response with both model results
        response = {
            'models': results,
            'sentimentLabel': sentiment_label,
            'sentimentScore': compound_score,
            'explicitnessLabel': explicitness_label,
            'explicitWordCount': explicit_word_count,
        }
        
        # For backward compatibility, use new model as primary if available
        if new_classifier.is_loaded:
            response['predictedGenre'] = results['new_model']['predicted_genre']
            response['confidence'] = results['new_model']['confidence']
            response['genreProbabilities'] = results['new_model']['genre_probabilities']
        elif legacy_classifier.is_loaded:
            response['predictedGenre'] = results['legacy_model']['predicted_genre']
            response['confidence'] = results['legacy_model']['confidence']
            response['genreProbabilities'] = results['legacy_model']['genre_probabilities']
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500
        return jsonify({'error': 'No models loaded. Please check server logs.'}), 500

    data = request.get_json()
    if not data or 'lyrics' not in data:
        return jsonify({'error': 'Invalid input: "lyrics" key not found.'}), 400

    lyrics = data['lyrics']
    
    if not lyrics or len(lyrics.strip()) == 0:
        return jsonify({'error': 'Empty lyrics provided.'}), 400
    
    try:
        # Predictions from both models
        new_model_result = None
        legacy_model_result = None
        
        if new_classifier.is_loaded:
            new_model_result = new_classifier.predict(lyrics)
        
        if legacy_classifier.is_loaded:
            legacy_model_result = legacy_classifier.predict(lyrics)
        
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
            'sentimentLabel': sentiment_label,
            'sentimentScore': compound_score,
            'explicitnessLabel': explicitness_label,
            'explicitWordCount': explicit_word_count,
        }
        
        # Add new model results if available
        if new_model_result:
            response['newModel'] = {
                'predictedGenre': new_model_result['predicted_genre'],
                'confidence': new_model_result['confidence'],
                'genreProbabilities': new_model_result['genre_probabilities'],
                'modelType': 'Hybrid (Sentence Transformers + TF-IDF)',
                'accuracy': '73.9%'
            }
        
        # Add legacy model results if available
        if legacy_model_result:
            response['legacyModel'] = {
                'predictedGenre': legacy_model_result['predicted_genre'],
                'confidence': legacy_model_result['confidence'],
                'genreProbabilities': legacy_model_result['genre_probabilities'],
                'modelType': 'Logistic Regression + TF-IDF',
                'accuracy': '~65%'
            }
        
        # Set primary prediction (prefer new model)
        if new_model_result:
            response['predictedGenre'] = new_model_result['predicted_genre']
            response['confidence'] = new_model_result['confidence']
            response['genreProbabilities'] = new_model_result['genre_probabilities']
        elif legacy_model_result:
            response['predictedGenre'] = legacy_model_result['predicted_genre']
            response['confidence'] = legacy_model_result['confidence']
            response['genreProbabilities'] = legacy_model_result['genre_probabilities']
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500


@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    """
    Audio-to-text prediction endpoint using both models
    """
    if not new_classifier.is_loaded and not legacy_classifier.is_loaded:
        return jsonify({'error': 'No models loaded. Please check server logs.'}), 500
    
    # Check if audio file is provided
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided.'}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected.'}), 400
    
    # Check file extension
    allowed_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}
    file_ext = os.path.splitext(audio_file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({'error': f'Unsupported file format. Supported: {", ".join(allowed_extensions)}'}), 400
    
    try:
        # Convert audio to text
        print(f"Processing audio file: {audio_file.filename}")
        transcribed_text = audio_processor.process_audio_file(audio_file, audio_file.filename)
        
        if not transcribed_text or len(transcribed_text.strip()) == 0:
            return jsonify({'error': 'No speech detected in audio file.'}), 400
        
        # Use both models for prediction
        results = {}
        
        if new_classifier.is_loaded:
            new_result = new_classifier.predict(transcribed_text)
            results['new_model'] = new_result
        
        if legacy_classifier.is_loaded:
            legacy_result = legacy_classifier.predict(transcribed_text)
            results['legacy_model'] = legacy_result
        
        # Sentiment analysis
        sentiment_scores = vader.polarity_scores(transcribed_text)
        compound_score = sentiment_scores['compound']
        sentiment_label = get_sentiment_label(compound_score)
        
        # Explicit content detection
        words = set(re.findall(r'\b\w+\b', transcribed_text.lower()))
        explicit_words_found = [word for word in words if word in profanity_list]
        explicit_word_count = len(explicit_words_found)
        explicitness_label = 'Explicit' if explicit_word_count > 0 else 'Not Explicit'

        response = {
            'transcribedText': transcribed_text,
            'models': results,
            'sentimentLabel': sentiment_label,
            'sentimentScore': compound_score,
            'explicitnessLabel': explicitness_label,
            'explicitWordCount': explicit_word_count,
        }
        
        # For backward compatibility, use new model as primary if available
        if new_classifier.is_loaded:
            response['predictedGenre'] = results['new_model']['predicted_genre']
            response['confidence'] = results['new_model']['confidence']
            response['genreProbabilities'] = results['new_model']['genre_probabilities']
        elif legacy_classifier.is_loaded:
            response['predictedGenre'] = results['legacy_model']['predicted_genre']
            response['confidence'] = results['legacy_model']['confidence']
            response['genreProbabilities'] = results['legacy_model']['genre_probabilities']
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        return jsonify({'error': f'Audio processing failed: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint showing status of both models
    """
    return jsonify({
        'status': 'healthy',
        'newModel': {
            'loaded': new_classifier.is_loaded,
            'type': 'Hybrid (Sentence Transformers + TF-IDF)',
            'accuracy': '73.9%'
        },
        'legacyModel': {
            'loaded': legacy_classifier.is_loaded,
            'type': 'Logistic Regression + TF-IDF',
            'accuracy': '~65%'
        },
        'audioProcessor': True
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
