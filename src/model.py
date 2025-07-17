import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from src.preprocessing import preprocess_lyrics


class GenreClassifier:
    """
    Genre classifier using hybrid approach with sentence transformers and TF-IDF
    """
    
    def __init__(self, model_dir='models/final'):
        self.model_dir = model_dir
        self.sentence_model = None
        self.tfidf_vectorizer = None
        self.svd_transformer = None
        self.classifier = None
        self.label_encoder = None
        self.is_loaded = False
        self.model_info = {
            'name': 'Hybrid Ensemble Model',
            'type': 'Sentence Transformers + TF-IDF + Ensemble',
            'accuracy': '73.9%',
            'features': '684-dimensional (384 embeddings + 300 TF-IDF)'
        }
    
    def load_models(self):
        """
        Load all model components
        """
        try:
            # Load sentence transformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load other components
            self.tfidf_vectorizer = joblib.load(f'{self.model_dir}/final_tfidf_vectorizer.joblib')
            self.svd_transformer = joblib.load(f'{self.model_dir}/final_svd_transformer.joblib')
            self.classifier = joblib.load(f'{self.model_dir}/final_ensemble_model.joblib')
            self.label_encoder = joblib.load(f'{self.model_dir}/final_label_encoder.joblib')
            
            self.is_loaded = True
            print("Models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_loaded = False
    
    def predict(self, lyrics_text):
        """
        Predict genre for given lyrics
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Preprocess lyrics
        processed_lyrics = preprocess_lyrics(lyrics_text)
        
        # Get sentence embeddings
        embeddings = self.sentence_model.encode([processed_lyrics])
        
        # Get TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([processed_lyrics])
        tfidf_reduced = self.svd_transformer.transform(tfidf_features)
        
        # Combine features
        combined_features = np.hstack([embeddings, tfidf_reduced])
        
        # Make prediction
        prediction = self.classifier.predict(combined_features)[0]
        probabilities = self.classifier.predict_proba(combined_features)[0]
        
        # Convert to genre name
        predicted_genre = self.label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        # Get all genre probabilities
        genre_probabilities = {
            self.label_encoder.classes_[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return {
            'predicted_genre': predicted_genre,
            'confidence': float(confidence),
            'genre_probabilities': genre_probabilities,
            'model_info': self.model_info
        }
