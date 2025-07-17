import joblib
import os


class LegacyClassifier:
    """
    Legacy classifier using the original logistic regression pipeline
    """
    
    def __init__(self, model_dir='Model'):
        self.model_dir = model_dir
        self.pipeline = None
        self.label_mapping = None
        self.is_loaded = False
        self.model_info = {
            'name': 'Original Logistic Regression',
            'type': 'TF-IDF + Logistic Regression Pipeline',
            'accuracy': '~65-70%',
            'features': 'TF-IDF Vectorization'
        }
    
    def load_models(self):
        """
        Load the legacy model components
        """
        try:
            pipeline_path = os.path.join(self.model_dir, 'logistic_regression_pipeline.joblib')
            label_path = os.path.join(self.model_dir, 'label_to_genre.joblib')
            
            if not os.path.exists(pipeline_path) or not os.path.exists(label_path):
                print(f"Legacy model files not found in {self.model_dir}")
                return False
            
            self.pipeline = joblib.load(pipeline_path)
            self.label_mapping = joblib.load(label_path)
            
            self.is_loaded = True
            print("Legacy models loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading legacy models: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, lyrics_text):
        """
        Predict genre using legacy model
        """
        if not self.is_loaded:
            raise ValueError("Legacy models not loaded. Call load_models() first.")
        
        try:
            # Use original pipeline prediction
            predicted_label_index = self.pipeline.predict([lyrics_text])[0]
            predicted_genre = self.label_mapping[predicted_label_index]
            
            # Get probabilities
            probabilities = self.pipeline.predict_proba([lyrics_text])[0]
            confidence = max(probabilities)
            
            # Create genre probabilities mapping
            genre_probabilities = {
                self.label_mapping[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
            
            return {
                'predicted_genre': predicted_genre,
                'confidence': float(confidence),
                'genre_probabilities': genre_probabilities,
                'model_info': self.model_info
            }
            
        except Exception as e:
            print(f"Legacy prediction error: {e}")
            raise e
